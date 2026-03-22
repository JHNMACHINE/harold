"""
Harold v0.4 — train.py
=====================
Trainer per VP-SDE continuous diffusion + self-conditioning.

Fix rispetto alla versione precedente:
  [FIX #1] train_step delega compute_loss al modello (score matching)
  [FIX #2] Rimosso MaskDiffusionSchedule, usa VPSDESchedule interno al modello
  [FIX #4] self_cond gestito internamente da compute_loss, sempre detached
  [FIX #6] self_cond_prob = 0.5 in config (coerente col sampler)
  [FIX #7] estimate_loss passa fixed_t per valutazione per-timestep
  [FIX #8] estimate_loss: tutto dentro train_cfg.ctx
  [FIX #9] gradient accumulation: salta micro-batch con mask vuota
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
from model import Harold, build_model
from dataset import build_loaders
from logger import AsyncLogger

# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class DiffusionTrainer:
    """
    Trainer per Harold v0.4 VP-SDE.
    Tutta la logica di diffusion (noise, self-cond, loss) vive in model.compute_loss.
    Il trainer si occupa solo di: iterare i batch, chiamare compute_loss, backward.
    """

    def __init__(self, model: Harold, config: ModelConfig, train_cfg: TrainConfig,
                 pad_token_id: int = 0):
        self.model        = model
        self.config       = config
        self.train_cfg    = train_cfg
        self.pad_token_id = pad_token_id   # GPT-2: eos_token_id (50256)

    def train_step(
        self,
        batch: torch.Tensor,   # (B, L) token IDs
    ) -> Optional[Tuple[torch.Tensor, Dict[str, float]]]:
        """
        Ritorna None se tutti i token sono padding (batch invalido).
        Il caller deve saltare il micro-batch in questo caso.
        """
        # Maschera token validi (non-padding)
        # GPT-2 usa eos_token come pad (id=50256) — usiamo pad_token_id dal trainer
        mask = (batch != self.pad_token_id)   # (B, L)

        # [FIX #9] Salta batch completamente vuoti
        if mask.sum() == 0:
            return None

        # [FIX #1] Delega tutto a compute_loss — gestisce noise, self-cond, loss
        loss, loss_dict = self.model.compute_loss(
            x0=batch,
            mask=mask,
            ce_weight=self.train_cfg.ce_loss_weight,
            self_cond_prob=self.train_cfg.self_cond_prob,
        )

        return loss, loss_dict


# ─────────────────────────────────────────────────────────────────────────────
# LR schedule — cosine decay con warmup
# ─────────────────────────────────────────────────────────────────────────────

def get_lr(it: int, cfg: TrainConfig) -> float:
    if it < cfg.warmup_iters:
        return cfg.lr * max(it, 1) / cfg.warmup_iters
    if it >= cfg.max_iters:
        return cfg.min_lr
    ratio = (it - cfg.warmup_iters) / max(cfg.max_iters - cfg.warmup_iters, 1)
    return cfg.min_lr + 0.5 * (1.0 + math.cos(math.pi * ratio)) * (cfg.lr - cfg.min_lr)


# ─────────────────────────────────────────────────────────────────────────────
# Valutazione per-timestep
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def estimate_loss(
    model:        Harold,
    train_cfg:    TrainConfig,
    val_loader:   DataLoader,
    pad_token_id: int = 50256,
    iter_num:     int = 0,
    logger:       Optional["AsyncLogger"] = None,
) -> float:
    """
    Valuta la loss su diversi timestep continui in [0,1].

    [FIX #7] Usa fixed_t per valutare esattamente a t specifici
             (invece di campionare t random come in training).
    [FIX #8] Tutto dentro train_cfg.ctx per consistenza col training.
    """
    device = next(model.parameters()).device
    model.eval()

    t_values  = [0.1, 0.3, 0.5, 0.7, 0.9]
    all_total = {t: [] for t in t_values}
    all_score = {t: [] for t in t_values}
    all_ce    = {t: [] for t in t_values}
    iterator  = iter(val_loader)

    for _ in range(train_cfg.eval_iters):
        try:
            batch = next(iterator)
        except StopIteration:
            break

        input_ids = batch["input_ids"].to(device)
        mask      = (input_ids != pad_token_id)

        if mask.sum() == 0:
            continue

        B = input_ids.shape[0]

        for t_val in t_values:
            fixed_t = torch.full((B,), t_val, dtype=torch.float32, device=device)

            with train_cfg.ctx:
                _, loss_dict = model.compute_loss(
                    x0=input_ids,
                    mask=mask,
                    ce_weight=train_cfg.ce_loss_weight,
                    fixed_t=fixed_t,
                    self_cond_prob=0.0,
                )

            all_total[t_val].append(loss_dict["total"])
            all_score[t_val].append(loss_dict["score"])
            all_ce[t_val].append(loss_dict["ce"])

    model.train()

    per_t_total = {
        t: float(torch.tensor(v).mean()) if v else float("inf")
        for t, v in all_total.items()
    }
    per_t_score = {
        t: float(torch.tensor(v).mean()) if v else float("inf")
        for t, v in all_score.items()
    }
    per_t_ce = {
        t: float(torch.tensor(v).mean()) if v else float("inf")
        for t, v in all_ce.items()
    }

    print("  val total: " + "  ".join(f"t={t}:{v:.4f}" for t, v in per_t_total.items()))
    print("  val score: " + "  ".join(f"t={t}:{v:.4f}" for t, v in per_t_score.items()))
    print("  val CE:    " + "  ".join(f"t={t}:{v:.4f}" for t, v in per_t_ce.items()))

    # Log asincrono dei dettagli per-timestep
    if logger is not None:
        logger.log({
            "type":        "val_detail",
            "iter":        iter_num,
            "total_per_t": {str(t): round(v, 6) for t, v in per_t_total.items()},
            "score_per_t": {str(t): round(v, 6) for t, v in per_t_score.items()},
            "ce_per_t":    {str(t): round(v, 6) for t, v in per_t_ce.items()},
        })

    valid = [v for v in per_t_total.values() if v != float("inf")]
    return sum(valid) / len(valid) if valid else float("inf")


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    path:         str,
    model:        Harold,
    optimizer:    torch.optim.Optimizer,
    scaler:       torch.GradScaler,  # type: ignore
    iter_num:     int,
    val_loss:     float,
    model_cfg:    ModelConfig,
    train_cfg:    TrainConfig,
    train_losses: list,
    val_losses:   list,
) -> None:
    torch.save({
        "iter_num":        iter_num,
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state":    scaler.state_dict(),
        "val_loss":        val_loss,
        "model_cfg":       model_cfg,
        "train_cfg":       train_cfg,
        "train_losses":    train_losses,
        "val_losses":      val_losses,
    }, path)


def load_checkpoint(
    path:      str,
    model:     Harold,
    optimizer: torch.optim.Optimizer,
    scaler:    torch.GradScaler,  # type: ignore
    device:    str,
) -> Tuple[int, float, list, list]:
    print(f"Carico checkpoint: {path}")
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    optimizer.load_state_dict(state["optimizer_state"])
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

    print("Harold v0.4 — VP-SDE Continuous Diffusion")
    print(f"Device:         {device}")
    print(f"Dtype:          {train_cfg.dtype}  (scaler={'ON' if train_cfg.use_scaler else 'OFF'})")
    print(f"Batch virtuale: {train_cfg.batch_size} × {train_cfg.grad_accum} = {train_cfg.effective_batch_size}")
    print(f"Self-cond prob: {train_cfg.self_cond_prob}")
    print(f"Beta:           [{model_cfg.diffusion_beta_min}, {model_cfg.diffusion_beta_max}]")

    # ── Tokenizer ─────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(train_cfg.tokenizer_model)

    # GPT-2 non ha pad token — usa eos come pad (standard per generazione)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_token_id = tokenizer.pad_token_id

    model_cfg.vocab_size    = tokenizer.vocab_size
    model_cfg.mask_token_id = int(getattr(tokenizer, "mask_token_id", None) or tokenizer.vocab_size)

    # ── Modello ───────────────────────────────────────────────────────────
    model    = build_model(model_cfg).to(device)
    
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Harold v0.4 — {n_params:.1f}M parametri totali")

    # ── Optimizer ─────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.lr,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        fused=True,
    )
    scaler = torch.GradScaler("cuda", enabled=train_cfg.use_scaler)

    # ── Dataset ───────────────────────────────────────────────────────────
    train_loader, val_loader = build_loaders(train_cfg, tokenizer)

    # ── Trainer ───────────────────────────────────────────────────────────
    trainer = DiffusionTrainer(model, model_cfg, train_cfg, pad_token_id=pad_token_id)

    # ── Checkpoint resume ─────────────────────────────────────────────────
    os.makedirs(train_cfg.checkpoint_dir, exist_ok=True)
    initial_iter  = 0
    best_val_loss = float("inf")
    train_losses: list = []
    val_losses:   list = []

    if train_cfg.preload:
        ckpt_path = (
            (train_cfg.read_latest() or (None, None))[1]
            if train_cfg.preload == "latest"
            else train_cfg.preload
        )
        if ckpt_path and os.path.isfile(ckpt_path):
            initial_iter, best_val_loss, train_losses, val_losses = load_checkpoint(
                ckpt_path, model, optimizer, scaler, device
            )
        else:
            print("Nessun checkpoint trovato, parto da zero.")

    # ── AsyncLogger ───────────────────────────────────────────────────────
    log_path = os.path.join(train_cfg.checkpoint_dir, "training.log")
    logger   = AsyncLogger(log_path, flush_every=10)
    print(f"Log asincrono → {log_path}")

    # ── Loop principale ────────────────────────────────────────────────────
    print(f"\nAvvio training — {train_cfg.max_iters} optimizer steps\n")

    model.train()
    start_time = time.time()
    train_iter = iter(train_loader)
    accum_loss = 0.0

    pbar = tqdm(range(initial_iter, train_cfg.max_iters), desc="Harold v0.4")

    for iter_num in pbar:
        lr = get_lr(iter_num, train_cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # ── Gradient accumulation ─────────────────────────────────────────
        optimizer.zero_grad(set_to_none=True)
        step_loss_sum  = 0.0
        step_score_sum = 0.0
        step_ce_sum    = 0.0
        valid_count    = 0
        mb_idx         = 0

        while valid_count < train_cfg.grad_accum:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch      = next(train_iter)

            input_ids = batch["input_ids"].to(device, non_blocking=True)

            with train_cfg.ctx:
                result = trainer.train_step(input_ids)

            # [FIX #9] Salta micro-batch invalidi
            if result is None:
                mb_idx += 1
                if mb_idx > train_cfg.grad_accum * 10:
                    break
                continue

            loss, loss_dict = result
            scaler.scale(loss / train_cfg.grad_accum).backward()

            step_loss_sum  += loss.item()
            step_score_sum += loss_dict.get("score", 0.0)
            step_ce_sum    += loss_dict.get("ce",    0.0)
            valid_count    += 1
            mb_idx         += 1

        if valid_count == 0:
            continue

        # ── Optimizer step ────────────────────────────────────────────────
        if train_cfg.use_scaler:
            scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        model.update_router_biases()

        # ── Logging ───────────────────────────────────────────────────────
        avg_loss  = step_loss_sum  / valid_count
        avg_score = step_score_sum / valid_count
        avg_ce    = step_ce_sum    / valid_count

        accum_loss += avg_loss
        train_losses.append(avg_loss)

        pbar.set_postfix({
            "loss":  f"{avg_loss:.4f}",
            "score": f"{avg_score:.4f}",
            "lr":    f"{lr:.2e}",
            "grad":  f"{grad_norm:.2f}",
        })

        # Log asincrono ogni step — zero impatto sul training
        logger.log({
            "type":       "train",
            "iter":       iter_num,
            "loss":       round(avg_loss,  6),
            "score":      round(avg_score, 6),
            "ce":         round(avg_ce,    6),
            "lr":         lr,
            "grad_norm":  round(float(grad_norm), 6),
            "elapsed_min": round((time.time() - start_time) / 60, 2),
        })

        # ── Valutazione ───────────────────────────────────────────────────
        if iter_num % train_cfg.eval_interval == 0 or iter_num == train_cfg.max_iters - 1:
            if iter_num == 0:
                continue

            val_loss   = estimate_loss(model, train_cfg, val_loader, pad_token_id, iter_num, logger)
            val_losses.append(val_loss)
            avg_train  = accum_loss / max(train_cfg.eval_interval, 1)
            accum_loss = 0.0
            elapsed    = (time.time() - start_time) / 60

            print(
                f"\n[iter {iter_num:7d}] "
                f"train={avg_train:.4f}  val={val_loss:.4f}  "
                f"lr={lr:.2e}  elapsed={elapsed:.1f}min"
            )

            # Log val asincrono
            logger.log({
                "type":        "val",
                "iter":        iter_num,
                "train_loss":  round(avg_train, 6),
                "val_loss":    round(val_loss,  6),
                "lr":          lr,
                "elapsed_min": round(elapsed, 2),
            })

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path     = os.path.join(
                    train_cfg.checkpoint_dir,
                    f"{train_cfg.checkpoint_prefix}_best.pt",
                )
                save_checkpoint(
                    best_path, model, optimizer, scaler,
                    iter_num, val_loss, model_cfg, train_cfg,
                    train_losses, val_losses,
                )
                print(f"  ★ Best val loss: {val_loss:.4f} → {best_path}")
                train_cfg.write_latest(iter_num, best_path)
                logger.log({
                    "type":      "best_checkpoint",
                    "iter":      iter_num,
                    "val_loss":  round(val_loss, 6),
                    "path":      best_path,
                })

            model.train()

        # ── Checkpoint periodico ──────────────────────────────────────────
        if iter_num > 0 and iter_num % train_cfg.save_every == 0:
            p = train_cfg.ckpt_path(iter_num)
            save_checkpoint(
                p, model, optimizer, scaler,
                iter_num, best_val_loss, model_cfg, train_cfg,
                train_losses, val_losses,
            )
            train_cfg.write_latest(iter_num, p)
            print(f"  Checkpoint periodico → {p}")
            logger.log({
                "type": "periodic_checkpoint",
                "iter": iter_num,
                "path": p,
            })

    # ── Finale ────────────────────────────────────────────────────────────
    elapsed    = (time.time() - start_time) / 60
    final_path = os.path.join(
        train_cfg.checkpoint_dir,
        f"{train_cfg.checkpoint_prefix}_final.pt",
    )
    save_checkpoint(
        final_path, model, optimizer, scaler,
        train_cfg.max_iters, best_val_loss, model_cfg, train_cfg,
        train_losses, val_losses,
    )
    train_cfg.write_latest(train_cfg.max_iters, final_path)
    logger.log({
        "type":             "finished",
        "total_iters":      train_cfg.max_iters,
        "best_val_loss":    round(best_val_loss, 6),
        "elapsed_min":      round(elapsed, 2),
        "final_checkpoint": final_path,
    })
    logger.close()   # flush finale garantito prima di uscire
    print(f"\nTraining completato in {elapsed:.1f} min → {final_path}")

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
    results   = run_training(model_cfg, train_cfg)
    print(f"Best val loss: {results['best_val_loss']:.4f}")