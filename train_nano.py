"""
train_nano.py — Harold-Nano scaling law check
===============================================
Script autonomo per verificare che l'architettura Harold converga
prima di lanciare il full run da 3.2B.

Avvio:
    python train_nano.py
    python train_nano.py --max_iters 5000   # test rapido ~$5
    python train_nano.py --no_compile        # se compile crasha

Cosa aspettarsi:
    iter    0: val_loss ~10.4  (CE random su vocab 32k = log(32k))
    iter  500: val_loss ~8-9   (il modello impara la distribuzione dei token)
    iter 5000: val_loss ~6-7   (struttura linguistica emergente)
    iter 20k:  val_loss ~4-5   (buona convergenza per 300M)

Se la loss è piatta dopo 2000 iter → bug architetturale, NON lanciare il 3.2B.
Se la loss scende con pendenza log-lineare → architettura sana, via libera.

Output:
    /workspace/checkpoints/nano/harold_nano_final.pt
    /workspace/checkpoints/nano/loss_curve.json
    /workspace/checkpoints/nano/loss_curve.png  (se matplotlib disponibile)
"""

import argparse
import json
import math
import os
import time
import warnings
from collections import deque

import torch
from tqdm.auto import tqdm

from config_nano import NanoTrainConfig, get_nano_model_config
from core.model import Harold, build_model
from core.dataset import build_loaders
from training.optimizer import build_optimizer
from training.trainer import DiffusionTrainer, run_grad_accum
from training.lr_schedule import get_lr
from utils.checkpoint import save_checkpoint
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Harold-Nano — scaling law check",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--max_iters",  type=int,   default=None)
    parser.add_argument("--batch_size", type=int,   default=None)
    parser.add_argument("--lr",         type=float, default=None)
    parser.add_argument("--no_compile",    action="store_true")
    parser.add_argument("--device",        type=str,   default=None)
    parser.add_argument("--use_fp8",       action="store_true",
                        help="Abilita FP8 per i linear layers degli expert")
    parser.add_argument("--use_hash_moe",  action="store_true",
                        help="Abilita Hash MoE deterministico invece del routing learnable")
    return parser.parse_args()


def estimate_params(model: Harold) -> str:
    n = sum(p.numel() for p in model.parameters()) / 1e6
    return f"{n/1000:.2f}B" if n >= 1000 else f"{n:.0f}M"


def plot_loss_curve(loss_log: list, out_path: str) -> None:
    """Salva la loss curve come PNG. Skip silenzioso se matplotlib non disponibile."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        iters  = [x["iter"]      for x in loss_log]
        train  = [x["train"]     for x in loss_log if "train" in x]
        val    = [(x["iter"], x["val"]) for x in loss_log if "val" in x]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(iters[:len(train)], train, alpha=0.4, color="#888", label="train")
        if val:
            vi, vv = zip(*val)
            ax.plot(vi, vv, "o-", color="#185FA5", label="val", linewidth=2)

        ax.set_xlabel("iter")
        ax.set_ylabel("loss")
        ax.set_title("Harold-Nano — loss curve")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        print(f"  Loss curve -> {out_path}")
    except ImportError:
        pass


def run_validation(
    model: Harold,
    val_loader,
    train_cfg: NanoTrainConfig,
    iter_num: int,
    device: str,
) -> float:
    """Val loss su t=[0.1, 0.3, 0.5, 0.7, 0.9]."""
    model.eval()
    t_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    losses   = {t: [] for t in t_values}

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= train_cfg.eval_iters:
                break
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            mask      = (input_ids != 0)
            if mask.sum() < 10:
                continue
            B = input_ids.shape[0]
            for t_val in t_values:
                t_tensor = torch.full((B,), t_val, device=device)
                with train_cfg.ctx:
                    _, loss_dict = model.compute_loss(
                        x0=input_ids, mask=mask,
                        ce_weight=train_cfg.ce_loss_weight,
                        fixed_t=t_tensor, self_cond_prob=0.0,
                    )
                losses[t_val].append(loss_dict["total"])

    model.train()

    per_t = {t: (sum(v) / len(v) if v else float("inf"))
             for t, v in losses.items()}

    print(f"  val: " + "  ".join(f"t={t}:{v:.4f}" for t, v in per_t.items()))
    return sum(per_t.values()) / len(per_t)


def run_nano_training(args: argparse.Namespace) -> dict:
    train_cfg  = NanoTrainConfig()

    # Override da CLI — FP8 e Hash MoE prima di get_nano_model_config
    # perché vengono letti per costruire ModelConfig
    if args.max_iters  is not None: train_cfg.max_iters   = args.max_iters
    if args.batch_size is not None: train_cfg.batch_size  = args.batch_size
    if args.lr         is not None: train_cfg.lr          = args.lr
    if args.no_compile:             train_cfg.use_compile = False
    if args.device     is not None: train_cfg.device      = args.device
    if args.use_fp8:                train_cfg.use_fp8      = True
    if args.use_hash_moe:           train_cfg.use_hash_moe = True

    model_cfg  = get_nano_model_config(train_cfg)

    device = train_cfg.device
    os.makedirs(train_cfg.checkpoint_dir, exist_ok=True)

    fp8_str  = "+FP8"  if train_cfg.use_fp8      else ""
    hash_str = "+Hash" if train_cfg.use_hash_moe else ""
    print("=" * 60)
    print(f"  Harold-Nano{fp8_str}{hash_str} — Scaling Law Check")
    print("=" * 60)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(train_cfg.tokenizer_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_token_id = tokenizer.pad_token_id
    model_cfg.vocab_size    = tokenizer.vocab_size
    model_cfg.mask_token_id = int(
        getattr(tokenizer, "mask_token_id", None) or tokenizer.vocab_size
    )

    # Modello
    # raw_model: Harold originale — usato per build_optimizer, DiffusionTrainer,
    #            update_router_biases, save_checkpoint (non compilato)
    # active_model: eventualmente compilato — usato nel training loop
    raw_model: Harold = build_model(model_cfg).to(device)
    print(f"  Parametri: {estimate_params(raw_model)}")
    print(f"  Device:    {device} / {train_cfg.dtype}")
    print(f"  Iters:     {train_cfg.max_iters:,}")
    eff = train_cfg.batch_size * train_cfg.grad_accum
    print(f"  Batch eff: {eff} (seq_len={train_cfg.seq_len})")

    if device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    if (train_cfg.use_compile
            and hasattr(torch, "compile")
            and device.startswith("cuda")):
        print(f"  torch.compile ({train_cfg.compile_mode})...")
        active_model = torch.compile(raw_model, mode=train_cfg.compile_mode)
    else:
        active_model = raw_model

    # Dataset e optimizer — usano raw_model (Harold, non compilato)
    train_loader, val_loader = build_loaders(train_cfg, tokenizer)
    optimizer = build_optimizer(raw_model, train_cfg)
    scaler    = torch.GradScaler("cuda", enabled=train_cfg.use_scaler)
    trainer   = DiffusionTrainer(raw_model, model_cfg, train_cfg,
                                 pad_token_id=pad_token_id)

    # Loop
    loss_log:   list   = []
    train_losses = deque(maxlen=train_cfg.loss_history_size)
    val_losses:  list  = []
    best_val    = float("inf")
    train_iter  = iter(train_loader)
    start_time  = time.time()

    print()
    pbar = tqdm(range(train_cfg.max_iters), desc="Nano")

    for iter_num in pbar:
        lr = get_lr(iter_num, train_cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)

        loss_sum, score_sum, ce_sum, valid_count, train_iter, extra = run_grad_accum(
            trainer, train_iter, train_loader, train_cfg, scaler, device,
        )
        if valid_count == 0:
            continue

        if train_cfg.use_scaler:
            scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            raw_model.parameters(), train_cfg.max_grad_norm,
        )
        scaler.step(optimizer)
        scaler.update()

        if iter_num % 10 == 0:
            raw_model.update_router_biases()
            raw_model.update_fp8_weights()  # [v0.7-FP8] no-op se use_fp8=False

        avg_loss = loss_sum / valid_count
        train_losses.append(avg_loss)
        loss_log.append({"iter": iter_num, "train": round(avg_loss, 6)})

        pbar.set_postfix({
            "loss": f"{avg_loss:.4f}",
            "lr":   f"{lr:.1e}",
            "grad": f"{float(grad_norm):.2f}",
        })

        # Validation
        if iter_num % train_cfg.eval_interval == 0 or iter_num == train_cfg.max_iters - 1:
            val_loss = run_validation(raw_model, val_loader, train_cfg, iter_num, device)
            val_losses.append(val_loss)
            elapsed = (time.time() - start_time) / 60
            print(f"\n[{iter_num:6d}] train={avg_loss:.4f}  val={val_loss:.4f}  "
                  f"lr={lr:.1e}  {elapsed:.1f}min")
            loss_log.append({"iter": iter_num, "val": round(val_loss, 6)})

            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(
                    train_cfg.best_ckpt_path(), raw_model,
                    optimizer, scaler, iter_num, best_val,
                    model_cfg, train_cfg, train_losses, val_losses,
                )
                print(f"  Best -> {best_val:.4f}")

        # Checkpoint periodico
        if iter_num > 0 and iter_num % train_cfg.save_every == 0:
            p = train_cfg.ckpt_path(iter_num)
            save_checkpoint(p, raw_model, optimizer, scaler, iter_num, best_val,
                            model_cfg, train_cfg, train_losses, val_losses)
            train_cfg.write_latest(iter_num, p)

    # Finale
    elapsed = (time.time() - start_time) / 60
    final_path = train_cfg.final_ckpt_path()
    save_checkpoint(final_path, raw_model, optimizer, scaler,
                    train_cfg.max_iters, best_val,
                    model_cfg, train_cfg, train_losses, val_losses)
    train_cfg.write_latest(train_cfg.max_iters, final_path)

    # Salva loss log JSON
    log_path = os.path.join(train_cfg.checkpoint_dir, "loss_curve.json")
    with open(log_path, "w") as f:
        json.dump(loss_log, f, indent=2)

    # Loss curve PNG
    png_path = os.path.join(train_cfg.checkpoint_dir, "loss_curve.png")
    plot_loss_curve(loss_log, png_path)

    print(f"\nNano completato in {elapsed:.1f} min")
    print(f"  Best val loss: {best_val:.4f}")
    print(f"  Checkpoint:    {final_path}")
    print(f"  Loss log:      {log_path}")

    return {
        "best_val_loss":   best_val,
        "val_losses":      val_losses,
        "train_losses":    list(train_losses),
        "elapsed_minutes": elapsed,
        "checkpoint_path": final_path,
        "loss_log_path":   log_path,
    }


def check_convergence(results: dict) -> None:
    """
    Analisi semplice della convergenza — dà un giudizio go/no-go per il 3.2B.
    """
    val = results["val_losses"]
    if len(val) < 5:
        print("\n[CONVERGENZA] Troppo pochi punti di validation per analisi.")
        return

    first5_avg = sum(val[:5])  / 5
    last5_avg  = sum(val[-5:]) / 5
    drop       = first5_avg - last5_avg
    pct_drop   = drop / first5_avg * 100

    print("\n" + "=" * 60)
    print("  Analisi convergenza")
    print("=" * 60)
    print(f"  Val loss iniziale (media primi 5):  {first5_avg:.4f}")
    print(f"  Val loss finale   (media ultimi 5): {last5_avg:.4f}")
    print(f"  Drop totale: {drop:.4f} ({pct_drop:.1f}%)")
    print()

    if last5_avg > 8.0:
        print("  [NO-GO] Loss finale > 8.0 — il modello non sta convergendo.")
        print("          Controlla: learning rate, architettura MoE, dataset.")
    elif pct_drop < 10:
        print("  [ATTENZIONE] Drop < 10% — convergenza lenta o plateau precoce.")
        print("               Considera: aumentare lr, controllare grad norm.")
    elif last5_avg < 6.0 and pct_drop > 30:
        print("  [GO] Convergenza sana. Via libera per il run da 3.2B.")
    else:
        print("  [PROBABILE GO] Convergenza nella norma per 20k iter a 300M.")
        print("                 Atteso: val_loss < 5.0 a convergenza piena.")

    print("=" * 60)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    args    = parse_args()
    results = run_nano_training(args)
    check_convergence(results)