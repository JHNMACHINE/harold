"""
Harold — train_fase1.py
========================
Pretraining fase 1: FineWeb-Edu 40% + Wikipedia EN 60%
Modello 200M parametri su A6000 48GB.

Differenze rispetto al train.py del POC:
  - ModelConfig 200M (d_model=768, n_layers=12)
  - MixedStreamingDataset invece di StreamingTextDataset
  - LR 2e-4 con warmup 800 iter
  - seq_len=256 per convergenza stabile
  - 40000 iter totali
  - Niente MTF: q_sample diretto più pulito e stabile
"""

import math
import os
import time
import warnings
from typing import Dict, Tuple, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from config import ModelConfig, TrainConfig, get_model_config, get_train_config
from model import Harold, MaskDiffusionSchedule, build_model
from dataset import build_loaders


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class DiffusionTrainer:
    """
    Trainer semplificato per fase 1:
      - Solo q_sample (masking coseno), niente MTF/T2T
      - Un forward per micro-step
      - Loss solo sulle posizioni mascherate
    """

    def __init__(self, model: Harold, config: ModelConfig, schedule: MaskDiffusionSchedule):
        self.model    = model
        self.config   = config
        self.schedule = schedule

    def train_step(self, batch: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        B      = batch.shape[0]
        device = batch.device
        t      = torch.randint(1, self.config.diffusion_T + 1, (B,), device=device)
        xt, mask  = self.schedule.q_sample(batch, t)
        logits, _ = self.model(xt, t)
        loss      = self._compute_loss(logits, batch, mask)
        return loss, {"total_loss": loss.item()}

    def _compute_loss(self, logits, targets, mask):
        if mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        return F.cross_entropy(logits[mask], targets[mask], reduction="mean")


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
    device = next(model.parameters()).device
    model.eval()

    t_values   = [8, 16, 32, 48, 56]
    all_losses = {t: [] for t in t_values}
    iterator   = iter(val_loader)

    for _ in range(train_cfg.eval_iters):
        try:
            batch = next(iterator)
        except StopIteration:
            break

        input_ids = batch["input_ids"].to(device)
        B         = input_ids.shape[0]

        for t_val in t_values:
            t        = torch.full((B,), t_val, dtype=torch.long, device=device)
            xt, mask = schedule.q_sample(input_ids, t)
            with train_cfg.ctx:
                logits, _ = model(xt, t)
            if mask.sum() > 0:
                loss = F.cross_entropy(logits[mask], input_ids[mask]).item()
                all_losses[t_val].append(loss)

    model.train()

    per_t = {t: float(torch.tensor(v).mean()) if v else float("inf")
             for t, v in all_losses.items()}
    print("  val per t: " + "  ".join(f"t={t}:{v:.2f}" for t, v in per_t.items()))
    return sum(per_t.values()) / len(per_t)


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(path, model, optimizer, scaler, iter_num, val_loss,
                    model_cfg, train_cfg, train_losses, val_losses):
    torch.save({
        "iter_num":        iter_num,
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state":    scaler.state_dict() if train_cfg.use_scaler else {},
        "val_loss":        val_loss,
        "model_cfg":       model_cfg,
        "train_cfg":       train_cfg,
        "train_losses":    train_losses,
        "val_losses":      val_losses,
    }, path)


def load_checkpoint(path, model, optimizer, scaler, device, train_cfg):
    print(f"Carico checkpoint: {path}")
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    optimizer.load_state_dict(state["optimizer_state"])
    if train_cfg.use_scaler and state.get("scaler_state"):
        scaler.load_state_dict(state["scaler_state"])
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
    print(f"Device:         {device}")
    print(f"Dtype:          {train_cfg.dtype}  (scaler={'ON' if train_cfg.use_scaler else 'OFF — BF16'})")
    print(f"Batch virtuale: {train_cfg.batch_size} × {train_cfg.grad_accum} = {train_cfg.effective_batch_size}")

    # ── Tokenizer ─────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(train_cfg.tokenizer_model)
    model_cfg.vocab_size    = tokenizer.vocab_size
    model_cfg.mask_token_id = getattr(tokenizer, "mask_token_id", None) or tokenizer.vocab_size

    # ── Modello ───────────────────────────────────────────────────────────
    model    = build_model(model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    n_active = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"Harold 200M — {n_params:.1f}M parametri totali, {n_active:.1f}M trainabili")

    # ── Optimizer + scaler ────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.lr,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        fused=True,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=train_cfg.use_scaler)  # type: ignore

    # ── Dataset ───────────────────────────────────────────────────────────
    train_loader, val_loader = build_loaders(train_cfg, tokenizer)

    # ── Schedule e trainer ────────────────────────────────────────────────
    schedule = MaskDiffusionSchedule(model_cfg).to(device)
    trainer  = DiffusionTrainer(model, model_cfg, schedule)

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
                ckpt_path, model, optimizer, scaler, device, train_cfg
            )
        else:
            print("Nessun checkpoint trovato, parto da zero.")

    # ── Loop principale ────────────────────────────────────────────────────
    print(f"\nAvvio training — {train_cfg.max_iters} optimizer steps "
          f"({train_cfg.max_iters * train_cfg.grad_accum} micro-steps)\n")

    model.train()
    start_time = time.time()
    train_iter = iter(train_loader)
    accum_loss = 0.0

    pbar = tqdm(range(initial_iter, train_cfg.max_iters), desc="Harold fase 1")

    for iter_num in pbar:
        lr = get_lr(iter_num, train_cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # ── Gradient accumulation ─────────────────────────────────────────
        optimizer.zero_grad(set_to_none=True)
        step_loss_sum = 0.0

        for _ in range(train_cfg.grad_accum):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch      = next(train_iter)

            input_ids = batch["input_ids"].to(device, non_blocking=True)

            with train_cfg.ctx:
                loss, _ = trainer.train_step(input_ids)

            (loss / train_cfg.grad_accum).backward()
            step_loss_sum += loss.item()

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
        accum_loss += avg_loss
        train_losses.append(avg_loss)

        pbar.set_postfix({
            "loss": f"{avg_loss:.4f}",
            "lr":   f"{lr:.2e}",
            "grad": f"{grad_norm:.2f}",
        })

        # ── Valutazione ───────────────────────────────────────────────────
        if iter_num % train_cfg.eval_interval == 0 or iter_num == train_cfg.max_iters - 1:
            if iter_num == 0:
                continue

            val_loss  = estimate_loss(model, train_cfg, schedule, val_loader)
            val_losses.append(val_loss)
            avg_train = accum_loss / train_cfg.eval_interval
            accum_loss = 0.0
            elapsed   = (time.time() - start_time) / 60

            print(f"\n[iter {iter_num:7d}] train={avg_train:.4f}  val={val_loss:.4f}  "
                  f"lr={lr:.2e}  elapsed={elapsed:.1f}min")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path     = os.path.join(train_cfg.checkpoint_dir,
                                             f"{train_cfg.checkpoint_prefix}_best.pt")
                save_checkpoint(best_path, model, optimizer, scaler,
                                iter_num, val_loss, model_cfg, train_cfg,
                                train_losses, val_losses)
                print(f"  ★ Best val loss: {val_loss:.4f} → {best_path}")
                train_cfg.write_latest(iter_num, best_path)

            model.train()

        # ── Checkpoint periodico ──────────────────────────────────────────
        if iter_num > 0 and iter_num % train_cfg.save_every == 0:
            p = train_cfg.ckpt_path(iter_num)
            save_checkpoint(p, model, optimizer, scaler,
                            iter_num, best_val_loss, model_cfg, train_cfg,
                            train_losses, val_losses)
            train_cfg.write_latest(iter_num, p)
            print(f"  Checkpoint periodico → {p}")

    # ── Checkpoint finale ─────────────────────────────────────────────────
    elapsed    = (time.time() - start_time) / 60
    final_path = os.path.join(train_cfg.checkpoint_dir,
                              f"{train_cfg.checkpoint_prefix}_final.pt")
    save_checkpoint(final_path, model, optimizer, scaler,
                    train_cfg.max_iters, best_val_loss, model_cfg, train_cfg,
                    train_losses, val_losses)
    train_cfg.write_latest(train_cfg.max_iters, final_path)
    print(f"\nTraining completato in {elapsed:.1f} min")
    print(f"Modello finale → {final_path}")

    return {
        "model":              model.cpu(),
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
    import os
    from dotenv import load_dotenv
    load_dotenv()
    warnings.filterwarnings("ignore")
    os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

    model_cfg = ModelConfig(
        d_model=768,
        n_layers=12,
        n_heads=12,
        n_kv_heads=4,
        d_ff=3072,
        moe_n_routed_experts=4,
        moe_top_k=2,
        ds_moe_n_shared_experts=1,
        max_seq_len=512,
        diffusion_T=64,
    )

    train_cfg = TrainConfig(
        batch_size=8,
        grad_accum=16,
        max_iters=40000,
        seq_len=256,
        lr=2e-4,
        warmup_iters=800,
        min_lr=2e-5,
        eval_interval=1000,
        eval_iters=20,
        save_every=2000,
        fineweb_weight=0.4,
        wikipedia_weight=0.6,
        preload="latest",
    )

    results = run_training(model_cfg, train_cfg)
    print(f"Best val loss: {results['best_val_loss']:.4f}")