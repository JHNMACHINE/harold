"""
Harold v3 — train.py
=====================
Training su singola GPU (A6000 48GB) per test architettura v3 ~200M.
Per scalare a multi-GPU con FSDP cambia le sezioni marcate con # FSDP.

Avvio:
  # Precomputa token weights (una volta sola, ~10 min)
  python -c "
  from transformers import AutoTokenizer
  from dataset import compute_token_weights
  tok = AutoTokenizer.from_pretrained('bert-base-uncased')
  compute_token_weights(tok, save_path='token_weights.pt')
  "

  # Training
  python train.py
"""

import math
import os
import time
import warnings
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from config import ModelConfig, TrainConfig, get_model_config, get_train_config
from model import Harold, MaskDiffusionSchedule, build_model
from dataset import build_loaders, compute_token_weights


# ─────────────────────────────────────────────────────────────────────────────
# Trainer v3 — continuous diffusion + self-conditioning
# ─────────────────────────────────────────────────────────────────────────────

class DiffusionTrainer:
    """
    Trainer per Harold v3:
      - Continuous diffusion: input embedding, output embedding, loss MSE+CE
      - Self-conditioning: 50% dei batch usano x0_prev come hint
      - Token-weighted q_sample: già nel schedule, trasparente qui
    """

    def __init__(
        self,
        model:     Harold,
        config:    ModelConfig,
        train_cfg: TrainConfig,
        schedule:  MaskDiffusionSchedule,
    ):
        self.model     = model
        self.config    = config
        self.train_cfg = train_cfg
        self.schedule  = schedule

    def train_step(
        self,
        batch: torch.Tensor,   # (B, L) token IDs
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        B      = batch.shape[0]
        device = batch.device

        t            = torch.randint(1, self.config.diffusion_T + 1, (B,), device=device)
        xt_ids, mask = self.schedule.q_sample(batch, t)

        with torch.no_grad():
            xt_emb = self.model.token_emb(xt_ids)

        # Self-conditioning: 50% delle volte usa predizione precedente
        self_cond = None
        if torch.rand(1).item() < self.train_cfg.self_cond_prob:
            with torch.no_grad():
                x0_prev, _, _ = self.model(xt_emb, t, self_cond=None)
                self_cond = x0_prev.mean(dim=1).detach()  # (B, d_model)

        x0_pred, ce_logits, _ = self.model(xt_emb, t, self_cond=self_cond)

        loss, loss_dict = self.model.compute_loss(
            x0_pred, ce_logits, batch, mask,
            ce_weight=self.train_cfg.ce_loss_weight,
        )

        return loss, loss_dict


# ─────────────────────────────────────────────────────────────────────────────
# LR schedule
# ─────────────────────────────────────────────────────────────────────────────

def get_lr(it: int, cfg: TrainConfig) -> float:
    if it < cfg.warmup_iters:
        return cfg.lr * max(it, 1) / cfg.warmup_iters
    if it >= cfg.max_iters:
        return cfg.min_lr
    ratio = (it - cfg.warmup_iters) / max(cfg.max_iters - cfg.warmup_iters, 1)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return cfg.min_lr + coeff * (cfg.lr - cfg.min_lr)


# ─────────────────────────────────────────────────────────────────────────────
# Valutazione
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def estimate_loss(
    model:      Harold,
    train_cfg:  TrainConfig,
    schedule:   MaskDiffusionSchedule,
    val_loader: DataLoader,
) -> float:
    device   = next(model.parameters()).device
    model.eval()

    t_values  = [8, 16, 32, 48, 56]
    all_total = {t: [] for t in t_values}
    all_mse   = {t: [] for t in t_values}
    all_ce    = {t: [] for t in t_values}
    iterator  = iter(val_loader)

    for _ in range(train_cfg.eval_iters):
        try:
            batch = next(iterator)
        except StopIteration:
            break

        input_ids = batch["input_ids"].to(device)
        B         = input_ids.shape[0]

        for t_val in t_values:
            t            = torch.full((B,), t_val, dtype=torch.long, device=device)
            xt_ids, mask = schedule.q_sample(input_ids, t)

            with train_cfg.ctx:
                xt_emb                = model.token_emb(xt_ids)
                x0_pred, ce_logits, _ = model(xt_emb, t, self_cond=None)

            if mask.sum() > 0:
                _, loss_dict = model.compute_loss(
                    x0_pred, ce_logits, input_ids, mask,
                    ce_weight=train_cfg.ce_loss_weight,
                )
                all_total[t_val].append(loss_dict["total"])
                all_mse[t_val].append(loss_dict["mse"])
                all_ce[t_val].append(loss_dict["ce"])

    model.train()

    per_t_total = {t: float(torch.tensor(v).mean()) if v else float("inf") for t, v in all_total.items()}
    per_t_mse   = {t: float(torch.tensor(v).mean()) if v else float("inf") for t, v in all_mse.items()}
    per_t_ce    = {t: float(torch.tensor(v).mean()) if v else float("inf") for t, v in all_ce.items()}

    print("  val total: " + "  ".join(f"t={t}:{v:.2f}" for t, v in per_t_total.items()))
    print("  val MSE:   " + "  ".join(f"t={t}:{v:.2f}" for t, v in per_t_mse.items()))
    print("  val CE:    " + "  ".join(f"t={t}:{v:.2f}" for t, v in per_t_ce.items()))

    return sum(per_t_total.values()) / len(per_t_total)


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    path:         str,
    model:        Harold,
    optimizer:    torch.optim.Optimizer,
    iter_num:     int,
    val_loss:     float,
    model_cfg:    ModelConfig,
    train_cfg:    TrainConfig,
    train_losses: list,
    val_losses:   list,
) -> None:
    torch.save(
        {
            "iter_num":        iter_num,
            "model_state":     model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_loss":        val_loss,
            "model_cfg":       model_cfg,
            "train_cfg":       train_cfg,
            "train_losses":    train_losses,
            "val_losses":      val_losses,
        },
        path,
    )


def load_checkpoint(
    path:      str,
    model:     Harold,
    optimizer: torch.optim.Optimizer,
    device:    str,
    train_cfg: TrainConfig,
) -> Tuple[int, float, list, list]:
    print(f"Carico checkpoint: {path}")
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    optimizer.load_state_dict(state["optimizer_state"])
    iter_num     = state.get("iter_num", 0) + 1
    best_val     = state.get("val_loss", float("inf"))
    train_losses = state.get("train_losses", [])
    val_losses   = state.get("val_losses", [])
    del state
    return iter_num, best_val, train_losses, val_losses


# ─────────────────────────────────────────────────────────────────────────────
# Run training
# ─────────────────────────────────────────────────────────────────────────────

def run_training(model_cfg: ModelConfig, train_cfg: TrainConfig) -> dict:
    device = train_cfg.device

    print(f"Harold v3 — singola GPU")
    print(f"Device:         {device}")
    print(f"Dtype:          {train_cfg.dtype}  (scaler={'ON' if train_cfg.use_scaler else 'OFF — BF16'})")
    print(f"Batch virtuale: {train_cfg.batch_size} × {train_cfg.grad_accum} = {train_cfg.effective_batch_size}")

    # ── Tokenizer ─────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(train_cfg.tokenizer_model)
    model_cfg.vocab_size    = tokenizer.vocab_size
    model_cfg.mask_token_id = int(getattr(tokenizer, "mask_token_id", None) or tokenizer.vocab_size)

    # ── Token weights per noise schedule ──────────────────────────────────
    token_weights = None
    if train_cfg.token_weights_path and os.path.exists(train_cfg.token_weights_path):
        token_weights = torch.load(
            train_cfg.token_weights_path, map_location="cpu", weights_only=True
        )
        print(f"Token weights caricati: {train_cfg.token_weights_path}")

    # ── Modello ───────────────────────────────────────────────────────────
    model    = build_model(model_cfg, token_weights).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Harold v3 — {n_params:.1f}M parametri totali")

    # ── Schedule ──────────────────────────────────────────────────────────
    schedule = MaskDiffusionSchedule(model_cfg, token_weights).to(device)

    # ── Optimizer ─────────────────────────────────────────────────────────
    muon_names  = {"q_proj", "k_up", "v_up", "o_proj", "w1", "w2", "w3", "kv_down"}
    muon_params = [p for n, p in model.named_parameters()
                   if any(k in n for k in muon_names) and p.requires_grad]
    adam_params = [p for n, p in model.named_parameters()
                   if not any(k in n for k in muon_names) and p.requires_grad]

    optimizer = torch.optim.AdamW(
        [
            {"params": muon_params, "lr": train_cfg.lr},
            {"params": adam_params, "lr": train_cfg.lr},
        ],
        betas=(0.9, 0.95),
        weight_decay=0.1,
        fused=True,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=train_cfg.use_scaler)  # type: ignore

    # ── Dataset ───────────────────────────────────────────────────────────
    train_loader, val_loader = build_loaders(train_cfg, tokenizer)

    # ── Trainer ───────────────────────────────────────────────────────────
    trainer = DiffusionTrainer(model, model_cfg, train_cfg, schedule)

    # ── Checkpoint resume ─────────────────────────────────────────────────
    os.makedirs(train_cfg.checkpoint_dir, exist_ok=True)
    initial_iter  = 0
    best_val_loss = float("inf")
    train_losses: list = []
    val_losses:   list = []

    if train_cfg.preload:
        if train_cfg.preload == "latest":
            result    = train_cfg.read_latest()
            ckpt_path = result[1] if result else None
        else:
            ckpt_path = train_cfg.preload

        if ckpt_path and os.path.isfile(ckpt_path):
            initial_iter, best_val_loss, train_losses, val_losses = load_checkpoint(
                ckpt_path, model, optimizer, device, train_cfg
            )
        else:
            print("Nessun checkpoint trovato, parto da zero.")

    # ── Loop principale ────────────────────────────────────────────────────
    print(f"\nAvvio training — {train_cfg.max_iters} optimizer steps\n")

    model.train()
    start_time = time.time()
    train_iter = iter(train_loader)
    accum_loss = 0.0

    pbar = tqdm(range(initial_iter, train_cfg.max_iters), desc="Harold v3")

    for iter_num in pbar:
        lr = get_lr(iter_num, train_cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # ── Gradient accumulation ─────────────────────────────────────────
        optimizer.zero_grad(set_to_none=True)
        step_loss_sum = 0.0
        step_mse_sum  = 0.0
        step_ce_sum   = 0.0

        for _ in range(train_cfg.grad_accum):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch      = next(train_iter)

            input_ids = batch["input_ids"].to(device, non_blocking=True)

            with train_cfg.ctx:
                loss, loss_dict = trainer.train_step(input_ids)

            scaler.scale(loss / train_cfg.grad_accum).backward()
            step_loss_sum += loss.item()
            step_mse_sum  += loss_dict.get("mse", 0.0)
            step_ce_sum   += loss_dict.get("ce",  0.0)

        # ── Optimizer step ────────────────────────────────────────────────
        if train_cfg.use_scaler:
            scaler.unscale_(optimizer)

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), train_cfg.max_grad_norm
        )
        scaler.step(optimizer)
        scaler.update()
        model.update_router_biases()

        # ── Logging ───────────────────────────────────────────────────────
        avg_loss    = step_loss_sum / train_cfg.grad_accum
        avg_mse     = step_mse_sum  / train_cfg.grad_accum
        avg_ce      = step_ce_sum   / train_cfg.grad_accum
        accum_loss += avg_loss
        train_losses.append(avg_loss)

        pbar.set_postfix({
            "loss": f"{avg_loss:.4f}",
            "mse":  f"{avg_mse:.4f}",
            "ce":   f"{avg_ce:.4f}",
            "lr":   f"{lr:.2e}",
            "grad": f"{grad_norm:.2f}",
        })

        # ── Valutazione ───────────────────────────────────────────────────
        if iter_num % train_cfg.eval_interval == 0 or iter_num == train_cfg.max_iters - 1:
            if iter_num == 0:
                continue

            val_loss  = estimate_loss(model, train_cfg, schedule, val_loader)
            val_losses.append(val_loss)
            avg_train  = accum_loss / train_cfg.eval_interval
            accum_loss = 0.0
            elapsed    = (time.time() - start_time) / 60

            print(
                f"\n[iter {iter_num:7d}] "
                f"train={avg_train:.4f}  val={val_loss:.4f}  "
                f"lr={lr:.2e}  elapsed={elapsed:.1f}min"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path     = os.path.join(
                    train_cfg.checkpoint_dir,
                    f"{train_cfg.checkpoint_prefix}_best.pt",
                )
                save_checkpoint(
                    best_path, model, optimizer, iter_num, val_loss,
                    model_cfg, train_cfg, train_losses, val_losses,
                )
                print(f"  ★ Best val loss: {val_loss:.4f} → {best_path}")
                train_cfg.write_latest(iter_num, best_path)

            model.train()

        # ── Checkpoint periodico ──────────────────────────────────────────
        if iter_num > 0 and iter_num % train_cfg.save_every == 0:
            p = train_cfg.ckpt_path(iter_num)
            save_checkpoint(
                p, model, optimizer, iter_num, best_val_loss,
                model_cfg, train_cfg, train_losses, val_losses,
            )
            train_cfg.write_latest(iter_num, p)
            print(f"  Checkpoint periodico → {p}")

    # ── Checkpoint finale ─────────────────────────────────────────────────
    elapsed    = (time.time() - start_time) / 60
    final_path = os.path.join(
        train_cfg.checkpoint_dir,
        f"{train_cfg.checkpoint_prefix}_final.pt",
    )
    save_checkpoint(
        final_path, model, optimizer, train_cfg.max_iters, best_val_loss,
        model_cfg, train_cfg, train_losses, val_losses,
    )
    train_cfg.write_latest(train_cfg.max_iters, final_path)
    print(f"\nTraining completato in {elapsed:.1f} min")
    print(f"Modello finale → {final_path}")

    return {
        "train_losses":       train_losses,
        "val_losses":         val_losses,
        "best_val_loss":      best_val_loss,
        "train_time_minutes": elapsed,
        "checkpoint_path":    final_path,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    model_cfg = get_model_config()
    train_cfg = get_train_config()

    results = run_training(model_cfg, train_cfg)
    print(f"Best val loss: {results['best_val_loss']:.4f}")