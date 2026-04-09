"""
Harold v0.6 — train.py  (unified single-GPU + DDP)
====================================================
Avvio single-GPU:
    torchrun --nproc_per_node=1 train.py

Avvio multi-GPU (es. 4 GPU):
    torchrun --nproc_per_node=4 train.py

Moduli ausiliari:
  setup.py       — build_training_context
  trainer.py     — DiffusionTrainer, run_grad_accum
  validation.py  — ValidationScheduler, run_validation_step, estimate_loss
  lr_schedule.py — get_lr
  optimizer.py   — MuonAdamW, build_optimizer
"""

import os
import time
import warnings
from collections import deque

import torch
from tqdm.auto import tqdm

from config import ModelConfig, TrainConfig, get_model_config, get_train_config
from setup import build_training_context
from validation import ValidationScheduler, run_validation_step
from lr_schedule import get_lr
from trainer import run_grad_accum
from checkpoint import save_checkpoint
from ddp import is_main


def run_training(model_cfg: ModelConfig, train_cfg: TrainConfig) -> dict:

    ctx = build_training_context(model_cfg, train_cfg)

    val_scheduler = ValidationScheduler(
        base_interval       = train_cfg.eval_interval,
        min_interval        = max(100, train_cfg.eval_interval // 5),
        max_interval        = train_cfg.eval_interval * 4,
        stability_threshold = 0.03,
        patience            = 3,
    )
    window_losses: deque = deque(maxlen=train_cfg.eval_interval)

    # ── Loop ──────────────────────────────────────────────────────────────
    ctx.active_model.train()
    start_time = time.time()
    train_iter = iter(ctx.train_loader)
    accum_loss = 0.0

    pbar = tqdm(
        range(ctx.initial_iter, train_cfg.max_iters),
        desc="Harold v0.6" + (" DDP" if ctx.use_ddp else ""),
        disable=not ctx.main,
    )

    for iter_num in pbar:
        lr = get_lr(iter_num, train_cfg)
        for pg in ctx.optimizer.param_groups:
            pg["lr"] = lr

        ctx.optimizer.zero_grad(set_to_none=True)

        loss_sum, score_sum, ce_sum, valid_count, train_iter = run_grad_accum(
            ctx.trainer, train_iter, ctx.train_loader, train_cfg, ctx.scaler, ctx.device,
        )
        if valid_count == 0:
            continue

        if train_cfg.use_scaler:
            ctx.scaler.unscale_(ctx.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            ctx.active_model.parameters(), train_cfg.max_grad_norm,
        )
        ctx.scaler.step(ctx.optimizer)
        ctx.scaler.update()
        ctx.model.update_router_biases()

        avg_loss  = loss_sum  / valid_count
        avg_score = score_sum / valid_count
        avg_ce    = ce_sum    / valid_count
        accum_loss += avg_loss
        window_losses.append(avg_loss)
        ctx.train_losses.append(avg_loss)

        if ctx.main:
            pbar.set_postfix({"loss": f"{avg_loss:.4f}", "score": f"{avg_score:.4f}",
                              "lr": f"{lr:.2e}", "grad": f"{grad_norm:.2f}"})
            if ctx.logger:
                ctx.logger.log({"type": "train", "iter": iter_num,
                                "loss": round(avg_loss, 6), "score": round(avg_score, 6),
                                "ce": round(avg_ce, 6), "lr": lr,
                                "grad_norm": round(float(grad_norm), 6),
                                "elapsed_min": round((time.time() - start_time) / 60, 2)})

        # ── Validation ────────────────────────────────────────────────────
        val_result = run_validation_step(
            ctx           = ctx,
            model_cfg     = model_cfg,
            train_cfg     = train_cfg,
            val_scheduler = val_scheduler,
            iter_num      = iter_num,
            accum_loss    = accum_loss,
            start_time    = start_time,
            lr            = lr,
            force_val     = (iter_num == train_cfg.max_iters - 1),
        )
        if val_result is not None:
            accum_loss = val_result.accum_loss
            if val_result.is_best:
                ctx.best_val_loss = val_result.val_loss

        # ── Checkpoint periodico ──────────────────────────────────────────
        if iter_num > 0 and iter_num % train_cfg.save_every == 0 and ctx.main:
            p = train_cfg.ckpt_path(iter_num)
            save_checkpoint(p, ctx.model, ctx.optimizer, ctx.scaler,
                            iter_num, ctx.best_val_loss, model_cfg, train_cfg,
                            ctx.train_losses, ctx.val_losses, full=False)
            train_cfg.write_latest(iter_num, p)
            print(f"  Checkpoint periodico → {p}")
            if ctx.logger:
                ctx.logger.log({"type": "periodic_checkpoint", "iter": iter_num, "path": p})

    # ── Finale ────────────────────────────────────────────────────────────
    elapsed    = (time.time() - start_time) / 60
    final_path = os.path.join(train_cfg.checkpoint_dir,
                              f"{train_cfg.checkpoint_prefix}_final.pt")

    if ctx.main:
        save_checkpoint(final_path, ctx.model, ctx.optimizer, ctx.scaler,
                        train_cfg.max_iters, ctx.best_val_loss, model_cfg, train_cfg,
                        ctx.train_losses, ctx.val_losses, push_hf=True, wait_hf=True)
        train_cfg.write_latest(train_cfg.max_iters, final_path)
        if ctx.logger:
            ctx.logger.log({"type": "finished", "total_iters": train_cfg.max_iters,
                            "best_val_loss": round(ctx.best_val_loss, 6),
                            "elapsed_min": round(elapsed, 2),
                            "final_checkpoint": final_path})
            ctx.logger.close()
        print(f"\nTraining completato in {elapsed:.1f} min → {final_path}")

    if ctx.use_ddp and ctx.ddp_ctx is not None:
        ctx.ddp_ctx.teardown()

    return {"train_losses": list(ctx.train_losses), "val_losses": ctx.val_losses,
            "best_val_loss": ctx.best_val_loss, "train_time_minutes": elapsed,
            "checkpoint_path": final_path}


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    model_cfg = get_model_config()
    train_cfg = get_train_config()
    results   = run_training(model_cfg, train_cfg)
    if is_main():
        print(f"Best val loss: {results['best_val_loss']:.4f}")