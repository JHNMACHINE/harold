"""
Harold v0.6 — train.py  (unified single-GPU + DDP)
====================================================
Avvio single-GPU:
    torchrun --nproc_per_node=1 train.py

Avvio multi-GPU (es. 4 GPU):
    torchrun --nproc_per_node=4 train.py

Comportamento automatico:
  - world_size=1  → single-GPU, torch.compile abilitato, nessun overhead DDP
  - world_size>1  → DDP attivo, torch.compile disabilitato,
                    dataset partizionato per rank, val loss sincronizzata via all_reduce

Moduli ausiliari:
  trainer.py     — DiffusionTrainer, run_grad_accum
  validation.py  — ValidationScheduler, estimate_loss, estimate_loss_single_t
  lr_schedule.py — get_lr
  optimizer.py   — MuonAdamW, build_optimizer
"""

import os
import time
import warnings
from collections import deque
from typing import Union, cast

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from config import ModelConfig, TrainConfig, get_model_config, get_train_config
from model import Harold, build_model
from optimizer import build_optimizer
from trainer import DiffusionTrainer, run_grad_accum
from validation import ValidationScheduler, estimate_loss, estimate_loss_single_t
from lr_schedule import get_lr
from dataset import build_loaders, build_loaders_ddp
from logger import AsyncLogger
from checkpoint import save_checkpoint, load_checkpoint
from ddp import DDPContext, is_ddp, is_main, all_reduce_mean, broadcast_model


def run_training(model_cfg: ModelConfig, train_cfg: TrainConfig) -> dict:

    # ── Setup ─────────────────────────────────────────────────────────────
    use_ddp    = is_ddp()
    rank       = 0
    local_rank = 0
    world_size = 1

    if use_ddp:
        ctx = DDPContext().setup()
        rank, local_rank, world_size = ctx.rank, ctx.local_rank, ctx.world_size
        device = f"cuda:{local_rank}"
    else:
        device = train_cfg.device

    main = is_main()

    if main:
        print("Harold v0.6 — Jamba (Mamba2 + Attention + MoE) + Flow Matching")
        print(f"Modalità:       {'DDP (' + str(world_size) + ' GPU)' if use_ddp else 'Single-GPU'}")
        print(f"Device:         {device}")
        print(f"Dtype:          {train_cfg.dtype}  (scaler={'ON' if train_cfg.use_scaler else 'OFF'})")
        eff = train_cfg.batch_size * train_cfg.grad_accum * world_size
        print(f"Batch effettivo:{eff}  ({train_cfg.batch_size} × {train_cfg.grad_accum} × {world_size} GPU)")

    if device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    # ── Tokenizer ─────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        train_cfg.tokenizer_model,
        token=os.environ.get("HF_TOKEN"),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_token_id = tokenizer.pad_token_id
    model_cfg.vocab_size    = tokenizer.vocab_size
    model_cfg.mask_token_id = int(getattr(tokenizer, "mask_token_id", None) or tokenizer.vocab_size)

    # ── Modello ───────────────────────────────────────────────────────────
    model = build_model(model_cfg).to(device)

    if not use_ddp:
        use_compile = (
            getattr(train_cfg, "use_compile", True)
            and hasattr(torch, "compile")
            and device.startswith("cuda")
            and torch.cuda.get_device_capability()[0] >= 7
        )
        if use_compile:
            compile_mode = getattr(train_cfg, "compile_mode", "reduce-overhead")
            if main:
                print(f"torch.compile() abilitato (mode='{compile_mode}')")
            model = cast(Harold, torch.compile(model, mode=compile_mode))
        elif main:
            print("torch.compile() disabilitato")
    elif main:
        print("torch.compile() disabilitato (DDP)")

    if main:
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"Harold v0.6 — {n_params:.1f}M parametri totali")

    active_model: Union[Harold, DDP] = (
        DDP(model, device_ids=[local_rank], output_device=local_rank)
        if use_ddp else model
    )
    raw_model: Harold = cast(
        Harold,
        active_model.module if isinstance(active_model, DDP) else active_model,
    )

    # ── Optimizer + scaler ────────────────────────────────────────────────
    optimizer = build_optimizer(active_model, train_cfg)
    scaler    = torch.GradScaler("cuda", enabled=train_cfg.use_scaler)

    # ── Dataset ───────────────────────────────────────────────────────────
    if use_ddp:
        train_loader, val_loader = build_loaders_ddp(train_cfg, tokenizer, rank)
    else:
        train_loader, val_loader = build_loaders(train_cfg, tokenizer)

    trainer = DiffusionTrainer(raw_model, model_cfg, train_cfg, pad_token_id=pad_token_id)

    val_scheduler = ValidationScheduler(
        base_interval       = train_cfg.eval_interval,
        min_interval        = max(100, train_cfg.eval_interval // 5),
        max_interval        = train_cfg.eval_interval * 4,
        stability_threshold = 0.03,
        patience            = 3,
    )
    window_losses: deque = deque(maxlen=train_cfg.eval_interval)

    # ── Checkpoint resume ─────────────────────────────────────────────────
    if main:
        os.makedirs(train_cfg.checkpoint_dir, exist_ok=True)

    initial_iter  = 0
    best_val_loss = float("inf")
    train_losses: deque = deque(maxlen=getattr(train_cfg, "loss_history_size", 100_000))
    val_losses:   list  = []

    if train_cfg.preload:
        ckpt_path = (
            (train_cfg.read_latest() or (None, None))[1]
            if train_cfg.preload == "latest"
            else train_cfg.preload
        )
        if ckpt_path:
            initial_iter, best_val_loss, _tl, val_losses = load_checkpoint(
                ckpt_path, raw_model, optimizer, scaler, device
            )
            train_losses.extend(_tl)

    if use_ddp:
        broadcast_model(raw_model)

    # ── Logger ────────────────────────────────────────────────────────────
    logger = None
    if main:
        log_path = os.path.join(train_cfg.checkpoint_dir, "training.log")
        logger   = AsyncLogger(log_path, flush_every=10)
        print(f"Log -> {log_path}\nAvvio training -> {train_cfg.max_iters} optimizer steps\n")

    # ── Loop ──────────────────────────────────────────────────────────────
    active_model.train()
    start_time = time.time()
    train_iter = iter(train_loader)
    accum_loss = 0.0

    pbar = tqdm(
        range(initial_iter, train_cfg.max_iters),
        desc="Harold v0.6" + (" DDP" if use_ddp else ""),
        disable=not main,
    )

    for iter_num in pbar:
        lr = get_lr(iter_num, train_cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)

        loss_sum, score_sum, ce_sum, valid_count, train_iter = run_grad_accum(
            trainer, train_iter, train_loader, train_cfg, scaler, device,
        )
        if valid_count == 0:
            continue

        if train_cfg.use_scaler:
            scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(active_model.parameters(), train_cfg.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        raw_model.update_router_biases()

        avg_loss  = loss_sum  / valid_count
        avg_score = score_sum / valid_count
        avg_ce    = ce_sum    / valid_count
        accum_loss += avg_loss
        window_losses.append(avg_loss)
        train_losses.append(avg_loss)

        if main:
            pbar.set_postfix({"loss": f"{avg_loss:.4f}", "score": f"{avg_score:.4f}",
                              "lr": f"{lr:.2e}", "grad": f"{grad_norm:.2f}"})
            if logger:
                logger.log({"type": "train", "iter": iter_num,
                            "loss": round(avg_loss, 6), "score": round(avg_score, 6),
                            "ce": round(avg_ce, 6), "lr": lr,
                            "grad_norm": round(float(grad_norm), 6),
                            "elapsed_min": round((time.time() - start_time) / 60, 2)})

        # ── Validation adattiva ───────────────────────────────────────────
        current_train_loss = sum(window_losses) / len(window_losses)
        force_val = (iter_num == train_cfg.max_iters - 1)
        should_val, reason = val_scheduler.should_validate(
            iter_num, current_train_loss, force=force_val,
        )

        if should_val and iter_num > 0:
            # Ogni 3 validation: completa; altrimenti rapida
            if val_scheduler.total_val_calls % 3 == 0 or force_val or "unstable" in reason:
                local_val = estimate_loss(
                    raw_model, train_cfg, val_loader, pad_token_id,
                    iter_num=iter_num, logger=logger if main else None,
                )
                val_type = "full"
            else:
                local_val = estimate_loss_single_t(
                    raw_model, train_cfg, val_loader, pad_token_id, t=0.5,
                )
                val_type = "quick"

            if use_ddp:
                val_tensor = torch.tensor(local_val, device=device)
                val_loss   = float(all_reduce_mean(val_tensor, world_size).item())
            else:
                val_loss = local_val

            val_scheduler.record(iter_num, val_loss)

            if main:
                elapsed   = (time.time() - start_time) / 60
                avg_train = accum_loss / max(iter_num - val_scheduler.last_val_iter, 1)

                if val_type == "full":
                    val_losses.append(val_loss)
                    accum_loss = 0.0
                    print(f"\n[iter {iter_num:7d}] train={avg_train:.4f}  val={val_loss:.4f}  "
                          f"lr={lr:.2e}  elapsed={elapsed:.1f}min  [{val_type}|{reason}]")
                    if logger:
                        logger.log({"type": "val", "iter": iter_num,
                                    "train_loss": round(avg_train, 6),
                                    "val_loss": round(val_loss, 6),
                                    "val_type": val_type, "reason": reason,
                                    "lr": lr, "elapsed_min": round(elapsed, 2)})

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_path = os.path.join(
                            train_cfg.checkpoint_dir,
                            f"{train_cfg.checkpoint_prefix}_best.pt",
                        )
                        save_checkpoint(best_path, raw_model, optimizer, scaler,
                                        iter_num, val_loss, model_cfg, train_cfg,
                                        train_losses, val_losses, push_hf=True)
                        print(f"  ★ Best val loss: {val_loss:.4f} → {best_path}")
                        train_cfg.write_latest(iter_num, best_path)
                        if logger:
                            logger.log({"type": "best_checkpoint", "iter": iter_num,
                                        "val_loss": round(val_loss, 6), "path": best_path})
                else:
                    print(f"  [quick val] iter={iter_num} loss={val_loss:.4f} ({reason})")
                    if logger:
                        logger.log({"type": "quick_val", "iter": iter_num,
                                    "val_loss": round(val_loss, 6), "reason": reason})

            model.train()

        # ── Checkpoint periodico ──────────────────────────────────────────
        if iter_num > 0 and iter_num % train_cfg.save_every == 0 and main:
            p = train_cfg.ckpt_path(iter_num)
            save_checkpoint(p, raw_model, optimizer, scaler,
                            iter_num, best_val_loss, model_cfg, train_cfg,
                            train_losses, val_losses, full=False)
            train_cfg.write_latest(iter_num, p)
            print(f"  Checkpoint periodico → {p}")
            if logger:
                logger.log({"type": "periodic_checkpoint", "iter": iter_num, "path": p})

    # ── Finale ────────────────────────────────────────────────────────────
    elapsed    = (time.time() - start_time) / 60
    final_path = os.path.join(train_cfg.checkpoint_dir,
                              f"{train_cfg.checkpoint_prefix}_final.pt")

    if main:
        save_checkpoint(final_path, raw_model, optimizer, scaler,
                        train_cfg.max_iters, best_val_loss, model_cfg, train_cfg,
                        train_losses, val_losses, push_hf=True, wait_hf=True)
        train_cfg.write_latest(train_cfg.max_iters, final_path)
        if logger:
            logger.log({"type": "finished", "total_iters": train_cfg.max_iters,
                        "best_val_loss": round(best_val_loss, 6),
                        "elapsed_min": round(elapsed, 2),
                        "final_checkpoint": final_path})
            logger.close()
        print(f"\nTraining completato in {elapsed:.1f} min → {final_path}")

    if use_ddp:
        ctx.teardown()

    return {"train_losses": list(train_losses), "val_losses": val_losses,
            "best_val_loss": best_val_loss, "train_time_minutes": elapsed,
            "checkpoint_path": final_path}


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    model_cfg = get_model_config()
    train_cfg = get_train_config()
    results   = run_training(model_cfg, train_cfg)
    if is_main():
        print(f"Best val loss: {results['best_val_loss']:.4f}")