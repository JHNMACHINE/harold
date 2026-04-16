"""
Harold v0.7 — train.py  (unified single-GPU + DDP)
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

Cambiamenti rispetto a v0.6:
  [v0.7-T1] val forzata a iter_num==0: baseline loss pre-training.
             Con un'architettura nuova (Mamba3 + x0-prediction) è utile
             avere il punto di partenza prima di qualsiasi aggiornamento.
  [v0.7-T2] Stringa versione aggiornata a Harold v0.7.
  [v0.7-T3] update_router_biases — annotato: con 16 expert routed il
             bilanciamento del carico è più critico che in v0.6 (8 expert).
             Nessuna modifica al codice, la logica è invariata in model.py.
"""

import os
import time
import warnings
from collections import deque

import torch
from tqdm.auto import tqdm

from core.config import ModelConfig, TrainConfig, get_model_config, get_train_config
from training.setup import build_training_context
from training.validation import ValidationScheduler, run_validation_step
from training.lr_schedule import get_lr
from training.trainer import run_grad_accum
from utils.checkpoint import save_checkpoint
from utils.ddp import is_main


def run_training(model_cfg: ModelConfig, train_cfg: TrainConfig) -> dict:

    ctx = build_training_context(model_cfg, train_cfg)

    val_scheduler = ValidationScheduler(
        base_interval       = train_cfg.eval_interval,
        min_interval        = max(100, train_cfg.eval_interval // 5),
        max_interval        = train_cfg.eval_interval * 4,
        stability_threshold = 0.03,
        patience            = 3,
    )
    if ctx.initial_iter > 0:
        val_scheduler._last_val_iter = ctx.initial_iter
        val_scheduler._total_val    = 1
    window_losses: deque = deque(maxlen=train_cfg.eval_interval)

    # ── Loop ──────────────────────────────────────────────────────────────
    ctx.active_model.train()
    start_time = time.time()
    train_iter = iter(ctx.train_loader)
    accum_loss      = 0.0
    steps_since_val = 0

    pbar = tqdm(
        range(ctx.initial_iter, train_cfg.max_iters),
        desc="Harold v0.7" + (" DDP" if ctx.use_ddp else ""),
        disable=not ctx.main,
    )

    for iter_num in pbar:
        lr = get_lr(iter_num, train_cfg)
        for pg in ctx.optimizer.param_groups:
            pg["lr"] = lr

        ctx.optimizer.zero_grad(set_to_none=True)

        loss_sum, score_sum, ce_sum, valid_count, train_iter, extra_metrics = run_grad_accum(
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

        # [v0.7-T3] Con 16 expert routed il router bias update è più critico
        # che in v0.6 (8 expert). Eseguito ogni 10 iter invece di ogni step:
        # - router_indices si accumula su finestra 10 step -> conteggio
        #   più stabile e rappresentativo per il bincount
        # - riduce overhead: 40 layer x 16 expert non è gratis ad ogni step
        if iter_num % 10 == 0:
            ctx.model.update_router_biases()

        avg_loss  = loss_sum  / valid_count
        avg_score = score_sum / valid_count
        avg_ce    = ce_sum    / valid_count
        accum_loss      += avg_loss
        steps_since_val += 1
        window_losses.append(avg_loss)
        ctx.train_losses.append(avg_loss)

        # [v0.7-M2c] Grad norm separato Mamba vs Attention — primi 10k step
        # Se norm_mamba > 10x norm_attn il modello usa solo memoria a breve termine
        mamba_norm = attn_norm = 0.0
        if iter_num < 10_000 and iter_num % 50 == 0:
            mamba_params = [p for n, p in ctx.active_model.named_parameters()
                            if 'mamba' in n and p.grad is not None]
            attn_params  = [p for n, p in ctx.active_model.named_parameters()
                            if any(x in n for x in ('q_proj', 'k_up', 'v_up', 'o_proj'))
                            and p.grad is not None]
            if mamba_params:
                mamba_norm = float(torch.nn.utils.get_total_norm(mamba_params, 2.0))
            if attn_params:
                attn_norm  = float(torch.nn.utils.get_total_norm(attn_params, 2.0))
            if mamba_norm > 0 and attn_norm > 0:
                ratio = mamba_norm / attn_norm
                if ratio > 10.0 and ctx.main:
                    print(f"  WARNING: grad Mamba/Attn ratio={ratio:.1f}x > 10 "
                          f"(mamba={mamba_norm:.3f}, attn={attn_norm:.3f})")

        if ctx.main:
            pbar.set_postfix({"loss": f"{avg_loss:.4f}", "score": f"{avg_score:.4f}",
                              "lr": f"{lr:.2e}", "grad": f"{grad_norm:.2f}"})
            if ctx.logger:
                log_entry = {"type": "train", "iter": iter_num,
                             "loss": round(avg_loss, 6), "score": round(avg_score, 6),
                             "ce": round(avg_ce, 6), "lr": lr,
                             "grad_norm": round(float(grad_norm), 6),
                             "elapsed_min": round((time.time() - start_time) / 60, 2)}
                if mamba_norm > 0 and attn_norm > 0:
                    log_entry["grad_norm_mamba"] = round(mamba_norm, 4)
                    log_entry["grad_norm_attn"]  = round(attn_norm, 4)
                    log_entry["grad_ratio_mamba_attn"] = round(mamba_norm / attn_norm, 3)
                # [v0.7-M4] x0_pred norm — rilevamento collapse (ogni 50 step)
                if iter_num % 50 == 0 and extra_metrics:
                    for k in ("x0_norm_mean", "x0_norm_std", "x0_var_tokens"):
                        if k in extra_metrics:
                            log_entry[k] = extra_metrics[k]
                ctx.logger.log(log_entry)

        # ── Validation ────────────────────────────────────────────────────
        # [v0.7-T1] force_val a iter_num==0: baseline loss pre-training.
        # Utile per un'architettura nuova (Mamba3 + x0-prediction) per
        # verificare che il punto di partenza sia ragionevole prima di
        # qualsiasi aggiornamento dei pesi.
        val_result = run_validation_step(
            ctx             = ctx,
            model_cfg       = model_cfg,
            train_cfg       = train_cfg,
            val_scheduler   = val_scheduler,
            iter_num        = iter_num,
            accum_loss      = accum_loss,
            steps_since_val = steps_since_val,
            start_time      = start_time,
            lr              = lr,
            force_val       = (iter_num == 0 or iter_num == train_cfg.max_iters - 1),
        )
        if val_result is not None:
            accum_loss      = val_result.accum_loss
            steps_since_val = 0
            if val_result.is_best:
                ctx.best_val_loss = val_result.val_loss
                if ctx.main:
                    best_path = train_cfg.best_ckpt_path()
                    save_checkpoint(best_path, ctx.model, ctx.optimizer, ctx.scaler,
                                    iter_num, ctx.best_val_loss, model_cfg, train_cfg,
                                    ctx.train_losses, ctx.val_losses, full=True)
                    print(f"  Best checkpoint -> {best_path}")
                    if ctx.logger:
                        ctx.logger.log({"type": "best_checkpoint", "iter": iter_num,
                                        "val_loss": round(ctx.best_val_loss, 6),
                                        "path": best_path})

        # ── Checkpoint periodico ──────────────────────────────────────────
        # [v0.7-P3] Ogni 10k iter: salva checkpoint + eval qualitativa inline
        if iter_num > 0 and iter_num % train_cfg.save_every == 0 and ctx.main:
            p = train_cfg.ckpt_path(iter_num)
            save_checkpoint(p, ctx.model, ctx.optimizer, ctx.scaler,
                            iter_num, ctx.best_val_loss, model_cfg, train_cfg,
                            ctx.train_losses, ctx.val_losses, full=False)
            train_cfg.write_latest(iter_num, p)
            print(f"  Checkpoint periodico -> {p}")
            if ctx.logger:
                ctx.logger.log({"type": "periodic_checkpoint", "iter": iter_num, "path": p})

            # [v0.7-P3] Eval qualitativa inline — usa il modello già in VRAM.
            # run_eval_on_model bypassa load_model(), evitando il doppio carico
            # che causerebbe OOM. Il modello viene messo in eval() e poi
            # riportato in train() automaticamente dalla funzione.
            try:
                from eval.eval_generation import run_eval_on_model
                out_dir = os.path.join(train_cfg.checkpoint_dir, "eval_results")
                run_eval_on_model(
                    model        = ctx.model,
                    tokenizer    = ctx.tokenizer,
                    pad_token_id = ctx.pad_token_id,
                    step         = iter_num,
                    out_dir      = out_dir,
                    max_steps    = 16,
                    cfg_scale    = 3.0,
                    device       = ctx.device,
                )
            except Exception as e:
                print(f"  WARNING: eval qualitativa fallita: {e}")

    # ── Finale ────────────────────────────────────────────────────────────
    elapsed    = (time.time() - start_time) / 60
    final_path = train_cfg.final_ckpt_path()

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
        print(f"\nTraining completato in {elapsed:.1f} min -> {final_path}")

    if ctx.use_ddp and ctx.ddp_ctx is not None:
        ctx.ddp_ctx.teardown()

    # [v0.7-P3] Eval qualitativa finale — sul checkpoint final, modello già in memoria
    if ctx.main:
        try:
            from eval.eval_generation import run_eval_on_model
            out_dir = os.path.join(train_cfg.checkpoint_dir, "eval_results")
            print(f"\nEval qualitativa finale -> {out_dir}")
            run_eval_on_model(
                model        = ctx.model,
                tokenizer    = ctx.tokenizer,
                pad_token_id = ctx.pad_token_id,
                step         = train_cfg.max_iters,
                out_dir      = out_dir,
                max_steps    = 16,
                cfg_scale    = 3.0,
                device       = ctx.device,
            )
        except Exception as e:
            print(f"  WARNING: eval qualitativa finale fallita: {e}")

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