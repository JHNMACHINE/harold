"""
Harold v3 — train_sft.py
=========================
Supervised Fine-Tuning con Classifier-Free Guidance (CFG).

Pipeline:
  1. Carica checkpoint del pretraining (harold_200m_v3_final.pt)
  2. Strato 1: UltraChat 200k (conversazioni multi-turn)
  3. Strato 2: OpenOrca (Q&A con chain-of-thought)

CFG:
  - ctx_emb = mean pooling degli embedding del prompt (context)
  - Con p_uncond=0.1: ctx_emb viene azzerato → training unconditional
  - In inferenza: ε_guided = ε_uncond + cfg_scale*(ε_cond - ε_uncond)

Differenze rispetto al pretraining:
  - lr molto più basso (2e-5 invece di 2e-4) — adattamento, non riscrittura
  - Loss solo sulla risposta (mask = response_mask, non padding mask)
  - ctx_emb passato a compute_loss per il conditioning
  - cfg_proj trainabile, tutti gli altri parametri anche (full fine-tuning)
"""

import math
import os
import time
import warnings
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from config import ModelConfig, SFTConfig
from logger import AsyncLogger
from model import Harold, build_model
from dataset import SFTDataset, build_sft_loaders


# ─────────────────────────────────────────────────────────────────────────────
# Context encoder — mean pooling degli embedding del prompt
# ─────────────────────────────────────────────────────────────────────────────

def encode_context(
    model:      Harold,
    prompt_ids: torch.Tensor,   # (B, L_ctx)
) -> torch.Tensor:
    """
    Encoda il contesto (prompt) come mean pooling degli embedding.

    Usiamo direttamente token_emb — non un encoder separato.
    Questo è semplice, efficiente, e coerente con lo spazio embedding
    in cui opera già il modello.

    Returns:
        ctx_emb: (B, D) — context embedding normalizzato
    """
    with torch.no_grad():
        # Maschera il padding (padding_idx=0)
        pad_mask  = (prompt_ids != 0).float()               # (B, L_ctx)
        emb       = model.token_emb(prompt_ids)              # (B, L_ctx, D)
        # Mean pooling sui token non-padding
        n_tokens  = pad_mask.sum(dim=1, keepdim=True).clamp(min=1)  # (B, 1)
        ctx_emb   = (emb * pad_mask.unsqueeze(-1)).sum(dim=1) / n_tokens  # (B, D)
    return ctx_emb


# ─────────────────────────────────────────────────────────────────────────────
# LR schedule
# ─────────────────────────────────────────────────────────────────────────────

def get_lr(it: int, cfg: SFTConfig, max_iters: int, base_lr: float, min_lr: float) -> float:
    if it < cfg.warmup_iters:
        return base_lr * max(it, 1) / cfg.warmup_iters
    if it >= max_iters:
        return min_lr
    ratio = (it - cfg.warmup_iters) / max(max_iters - cfg.warmup_iters, 1)
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * ratio)) * (base_lr - min_lr)


# ─────────────────────────────────────────────────────────────────────────────
# Valutazione
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def estimate_sft_loss(
    model:      Harold,
    sft_cfg:    SFTConfig,
    val_loader: DataLoader,
    iter_num:   int = 0,
    logger:     Optional[AsyncLogger] = None,
) -> float:
    device = next(model.parameters()).device
    model.eval()

    t_values  = [0.1, 0.3, 0.5, 0.7, 0.9]
    all_total = {t: [] for t in t_values}
    all_score = {t: [] for t in t_values}
    all_ce    = {t: [] for t in t_values}
    iterator  = iter(val_loader)

    for _ in range(sft_cfg.eval_iters):
        try:
            batch = next(iterator)
        except StopIteration:
            break

        prompt_ids   = batch["prompt_ids"].to(device)    # (B, L_ctx)
        response_ids = batch["response_ids"].to(device)  # (B, L_resp)
        resp_mask    = batch["response_mask"].to(device)  # (B, L_resp)

        if resp_mask.sum() == 0:
            continue

        # Encoda il contesto — sempre condizionato in val (no CFG dropout)
        ctx_emb = encode_context(model, prompt_ids)

        B = response_ids.shape[0]
        for t_val in t_values:
            fixed_t = torch.full((B,), t_val, dtype=torch.float32, device=device)

            with sft_cfg.ctx:
                _, loss_dict = model.compute_loss(
                    x0=response_ids,
                    mask=resp_mask,
                    ce_weight=sft_cfg.ce_loss_weight,
                    fixed_t=fixed_t,
                    self_cond_prob=0.0,
                    ctx_emb=ctx_emb,
                    p_uncond=0.0,   # no dropout in val
                )

            all_total[t_val].append(loss_dict["total"])
            all_score[t_val].append(loss_dict["score"])
            all_ce[t_val].append(loss_dict["ce"])

    model.train()

    per_t_total = {t: float(torch.tensor(v).mean()) if v else float("inf") for t, v in all_total.items()}
    per_t_score = {t: float(torch.tensor(v).mean()) if v else float("inf") for t, v in all_score.items()}
    per_t_ce    = {t: float(torch.tensor(v).mean()) if v else float("inf") for t, v in all_ce.items()}

    print("  val total: " + "  ".join(f"t={t}:{v:.4f}" for t, v in per_t_total.items()))
    print("  val score: " + "  ".join(f"t={t}:{v:.4f}" for t, v in per_t_score.items()))
    print("  val CE:    " + "  ".join(f"t={t}:{v:.4f}" for t, v in per_t_ce.items()))

    if logger:
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

def save_sft_checkpoint(
    path:         str,
    model:        Harold,
    optimizer:    torch.optim.Optimizer,
    scaler:       torch.amp.GradScaler,  # type: ignore
    stage:        int,
    iter_num:     int,
    val_loss:     float,
    model_cfg:    ModelConfig,
    sft_cfg:      SFTConfig,
    train_losses: list,
    val_losses:   list,
) -> None:
    torch.save({
        "stage":           stage,
        "iter_num":        iter_num,
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state":    scaler.state_dict(),
        "val_loss":        val_loss,
        "model_cfg":       model_cfg,
        "sft_cfg":         sft_cfg,
        "train_losses":    train_losses,
        "val_losses":      val_losses,
    }, path)


def load_sft_checkpoint(
    path:      str,
    model:     Harold,
    optimizer: torch.optim.Optimizer,
    scaler:    torch.amp.GradScaler,  # type: ignore
    device:    str,
) -> Tuple[int, int, float, list, list]:
    print(f"Carico SFT checkpoint: {path}")
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    optimizer.load_state_dict(state["optimizer_state"])
    if "scaler_state" in state:
        scaler.load_state_dict(state["scaler_state"])
    stage        = state.get("stage", 1)
    iter_num     = state.get("iter_num", 0) + 1
    best_val     = state.get("val_loss", float("inf"))
    train_losses = state.get("train_losses", [])
    val_losses   = state.get("val_losses", [])
    del state
    return stage, iter_num, best_val, train_losses, val_losses


# ─────────────────────────────────────────────────────────────────────────────
# Training stage — loop generico usato per strato 1 e 2
# ─────────────────────────────────────────────────────────────────────────────

def run_stage(
    stage:        int,
    model:        Harold,
    model_cfg:    ModelConfig,
    sft_cfg:      SFTConfig,
    optimizer:    torch.optim.Optimizer,
    scaler:       torch.amp.GradScaler,  # type: ignore
    train_loader: DataLoader,
    val_loader:   DataLoader,
    max_iters:    int,
    base_lr:      float,
    initial_iter: int,
    best_val:     float,
    train_losses: list,
    val_losses:   list,
    logger:       AsyncLogger,
) -> Tuple[float, list, list]:
    """
    Loop di training generico per un singolo strato SFT.
    Ritorna (best_val_loss, train_losses, val_losses).
    """
    device     = sft_cfg.device
    start_time = time.time()
    train_iter = iter(train_loader)
    accum_loss = 0.0

    pbar = tqdm(range(initial_iter, max_iters), desc=f"Harold SFT s{stage}")

    for iter_num in pbar:
        lr = get_lr(iter_num, sft_cfg, max_iters, base_lr, sft_cfg.min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        step_loss_sum  = 0.0
        step_score_sum = 0.0
        step_ce_sum    = 0.0
        valid_count    = 0
        mb_idx         = 0

        while valid_count < sft_cfg.grad_accum:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch      = next(train_iter)

            prompt_ids   = batch["prompt_ids"].to(device, non_blocking=True)
            response_ids = batch["response_ids"].to(device, non_blocking=True)
            resp_mask    = batch["response_mask"].to(device, non_blocking=True)

            if resp_mask.sum() == 0:
                mb_idx += 1
                if mb_idx > sft_cfg.grad_accum * 10:
                    break
                continue

            # Encoda il contesto per CFG
            ctx_emb = encode_context(model, prompt_ids)   # (B, D)

            with sft_cfg.ctx:
                loss, loss_dict = model.compute_loss(
                    x0=response_ids,
                    mask=resp_mask,
                    ce_weight=sft_cfg.ce_loss_weight,
                    self_cond_prob=sft_cfg.self_cond_prob,
                    ctx_emb=ctx_emb,
                    p_uncond=sft_cfg.p_uncond,
                )

            scaler.scale(loss / sft_cfg.grad_accum).backward()
            step_loss_sum  += loss.item()
            step_score_sum += loss_dict.get("score", 0.0)
            step_ce_sum    += loss_dict.get("ce",    0.0)
            valid_count    += 1
            mb_idx         += 1

        if valid_count == 0:
            continue

        if sft_cfg.use_scaler:
            scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), sft_cfg.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        model.update_router_biases()

        avg_loss  = step_loss_sum  / valid_count
        avg_score = step_score_sum / valid_count
        avg_ce    = step_ce_sum    / valid_count
        accum_loss += avg_loss
        train_losses.append(avg_loss)

        pbar.set_postfix({
            "loss":  f"{avg_loss:.4f}",
            "score": f"{avg_score:.4f}",
            "ce":    f"{avg_ce:.4f}",
            "lr":    f"{lr:.2e}",
            "grad":  f"{grad_norm:.2f}",
        })

        logger.log({
            "type":        "train",
            "stage":       stage,
            "iter":        iter_num,
            "loss":        round(avg_loss,  6),
            "score":       round(avg_score, 6),
            "ce":          round(avg_ce,    6),
            "lr":          lr,
            "grad_norm":   round(float(grad_norm), 6),
            "elapsed_min": round((time.time() - start_time) / 60, 2),
        })

        # Valutazione
        if iter_num % sft_cfg.eval_interval == 0 or iter_num == max_iters - 1:
            if iter_num == 0:
                continue

            val_loss  = estimate_sft_loss(model, sft_cfg, val_loader, iter_num, logger)
            val_losses.append(val_loss)
            avg_train  = accum_loss / max(sft_cfg.eval_interval, 1)
            accum_loss = 0.0
            elapsed    = (time.time() - start_time) / 60

            print(
                f"\n[s{stage} iter {iter_num:7d}] "
                f"train={avg_train:.4f}  val={val_loss:.4f}  "
                f"lr={lr:.2e}  elapsed={elapsed:.1f}min"
            )
            logger.log({
                "type":        "val",
                "stage":       stage,
                "iter":        iter_num,
                "train_loss":  round(avg_train, 6),
                "val_loss":    round(val_loss,  6),
                "lr":          lr,
                "elapsed_min": round(elapsed, 2),
            })

            if val_loss < best_val:
                best_val  = val_loss
                best_path = os.path.join(
                    sft_cfg.checkpoint_dir,
                    f"{sft_cfg.checkpoint_prefix}_s{stage}_best.pt",
                )
                save_sft_checkpoint(
                    best_path, model, optimizer, scaler,
                    stage, iter_num, val_loss, model_cfg, sft_cfg,
                    train_losses, val_losses,
                )
                print(f"  ★ Best val loss s{stage}: {val_loss:.4f} → {best_path}")
                sft_cfg.write_latest(stage, iter_num, best_path)
                logger.log({"type": "best_checkpoint", "stage": stage, "iter": iter_num,
                            "val_loss": round(val_loss, 6), "path": best_path})

            model.train()

        # Checkpoint periodico
        if iter_num > 0 and iter_num % sft_cfg.save_every == 0:
            p = sft_cfg.ckpt_path(stage, iter_num)
            save_sft_checkpoint(
                p, model, optimizer, scaler,
                stage, iter_num, best_val, model_cfg, sft_cfg,
                train_losses, val_losses,
            )
            sft_cfg.write_latest(stage, iter_num, p)
            print(f"  Checkpoint periodico → {p}")
            logger.log({"type": "periodic_checkpoint", "stage": stage, "iter": iter_num, "path": p})

    return best_val, train_losses, val_losses


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_sft(sft_cfg: SFTConfig) -> dict:
    device = sft_cfg.device

    print("Harold v3 — SFT con CFG")
    print(f"Device:         {device}")
    print(f"Dtype:          {sft_cfg.dtype}")
    print(f"Batch virtuale: {sft_cfg.batch_size} × {sft_cfg.grad_accum} = {sft_cfg.effective_batch_size}")
    print(f"p_uncond:       {sft_cfg.p_uncond}  (CFG dropout)")
    print(f"ctx_len:        {sft_cfg.max_ctx_len}  resp_len: {sft_cfg.max_resp_len}")

    # ── Tokenizer ─────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(sft_cfg.tokenizer_model)

    # ── Modello: carica dal checkpoint di pretraining ─────────────────────
    print(f"\nCarico pretraining checkpoint: {sft_cfg.pretrain_ckpt}")
    state     = torch.load(sft_cfg.pretrain_ckpt, map_location="cpu", weights_only=False)
    model_cfg = state["model_cfg"]

    # cfg_proj non è nel checkpoint del pretraining — verrà inizializzato
    # a zero da build_model (come definito in Harold.__init__)
    model = build_model(model_cfg).to(device)

    # Carica i pesi del pretraining — strict=False permette a cfg_proj
    # di essere inizializzato a zero anche se non è nel checkpoint
    missing, unexpected = model.load_state_dict(state["model_state"], strict=False)
    if missing:
        print(f"Pesi mancanti (nuovi layer): {missing}")
    if unexpected:
        print(f"Pesi inattesi (ignorati): {unexpected}")
    del state

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Harold v3 SFT — {n_params:.1f}M parametri")

    # ── Optimizer ─────────────────────────────────────────────────────────
    # cfg_proj ha lr più alto degli altri — deve imparare da zero
    cfg_params   = list(model.cfg_proj.parameters())
    other_params = [p for n, p in model.named_parameters()
                    if "cfg_proj" not in n and p.requires_grad]

    optimizer = torch.optim.AdamW(
        [
            {"params": cfg_params,   "lr": sft_cfg.lr * 10},  # cfg_proj: 10x lr
            {"params": other_params, "lr": sft_cfg.lr},
        ],
        betas=(0.9, 0.95),
        weight_decay=0.1,
        fused=True,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=sft_cfg.use_scaler)  # type: ignore

    # ── AsyncLogger ────────────────────────────────────────────────────────
    os.makedirs(sft_cfg.checkpoint_dir, exist_ok=True)
    log_path = os.path.join(sft_cfg.checkpoint_dir, "training_sft.log")
    logger   = AsyncLogger(log_path)
    print(f"Log SFT → {log_path}")

    # ── Resume dal checkpoint SFT se esiste ───────────────────────────────
    initial_stage = 1
    initial_iter  = 0
    best_val      = float("inf")
    train_losses: list = []
    val_losses:   list = []

    if sft_cfg.preload:
        ckpt_path = None
        if sft_cfg.preload == "latest":
            result = sft_cfg.read_latest()
            if result:
                initial_stage, initial_iter, ckpt_path = result
        else:
            ckpt_path = sft_cfg.preload

        if ckpt_path and os.path.isfile(ckpt_path):
            initial_stage, initial_iter, best_val, train_losses, val_losses = load_sft_checkpoint(
                ckpt_path, model, optimizer, scaler, device
            )
        else:
            print("Nessun SFT checkpoint trovato, parto dal pretraining.")

    # ── Strato 1: UltraChat ────────────────────────────────────────────────
    if initial_stage <= 1:
        print(f"\n{'='*60}")
        print(f"Strato 1: UltraChat 200k — {sft_cfg.max_iters} step")
        print(f"{'='*60}\n")

        train_loader, val_loader = build_sft_loaders(
            sft_cfg, tokenizer,
            max_ctx_len=sft_cfg.max_ctx_len,
            max_resp_len=sft_cfg.max_resp_len,
            max_ctx_turns=sft_cfg.max_ctx_turns,
        )

        model.train()
        best_val, train_losses, val_losses = run_stage(
            stage=1,
            model=model, model_cfg=model_cfg, sft_cfg=sft_cfg,
            optimizer=optimizer, scaler=scaler,
            train_loader=train_loader, val_loader=val_loader,
            max_iters=sft_cfg.max_iters,
            base_lr=sft_cfg.lr,
            initial_iter=initial_iter if initial_stage == 1 else 0,
            best_val=best_val,
            train_losses=train_losses, val_losses=val_losses,
            logger=logger,
        )

        # Checkpoint fine strato 1
        s1_final = os.path.join(sft_cfg.checkpoint_dir, f"{sft_cfg.checkpoint_prefix}_s1_final.pt")
        save_sft_checkpoint(
            s1_final, model, optimizer, scaler,
            1, sft_cfg.max_iters, best_val, model_cfg, sft_cfg,
            train_losses, val_losses,
        )
        print(f"\nStrato 1 completato → {s1_final}")

    # ── Strato 2: OpenOrca ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Strato 2: OpenOrca — {sft_cfg.stage2_max_iters} step")
    print(f"{'='*60}\n")

    # Dataset solo OpenOrca per strato 2
    train_ds2 = SFTDataset(
        tokenizer=tokenizer, split="train",
        max_ctx_len=sft_cfg.max_ctx_len,
        max_resp_len=sft_cfg.max_resp_len,
        max_ctx_turns=sft_cfg.max_ctx_turns,
        val_every=sft_cfg.val_every, seed=43,
        weights={"openorca": 1.0},
    )
    val_ds2 = SFTDataset(
        tokenizer=tokenizer, split="val",
        max_ctx_len=sft_cfg.max_ctx_len,
        max_resp_len=sft_cfg.max_resp_len,
        max_ctx_turns=sft_cfg.max_ctx_turns,
        val_every=sft_cfg.val_every, seed=43,
        weights={"openorca": 1.0},
    )
    from torch.utils.data import DataLoader
    train_loader2 = DataLoader(train_ds2, batch_size=sft_cfg.batch_size, num_workers=0, pin_memory=True)
    val_loader2   = DataLoader(val_ds2,   batch_size=sft_cfg.batch_size, num_workers=0, pin_memory=True)

    # LR più basso per strato 2
    for pg in optimizer.param_groups:
        pg["lr"] = sft_cfg.stage2_lr

    model.train()
    best_val2, train_losses, val_losses = run_stage(
        stage=2,
        model=model, model_cfg=model_cfg, sft_cfg=sft_cfg,
        optimizer=optimizer, scaler=scaler,
        train_loader=train_loader2, val_loader=val_loader2,
        max_iters=sft_cfg.stage2_max_iters,
        base_lr=sft_cfg.stage2_lr,
        initial_iter=initial_iter if initial_stage == 2 else 0,
        best_val=best_val,
        train_losses=train_losses, val_losses=val_losses,
        logger=logger,
    )

    # Checkpoint finale
    final_path = os.path.join(sft_cfg.checkpoint_dir, f"{sft_cfg.checkpoint_prefix}_final.pt")
    save_sft_checkpoint(
        final_path, model, optimizer, scaler,
        2, sft_cfg.stage2_max_iters, best_val2, model_cfg, sft_cfg,
        train_losses, val_losses,
    )
    sft_cfg.write_latest(2, sft_cfg.stage2_max_iters, final_path)
    logger.log({"type": "finished", "best_val_s1": round(best_val, 6),
                "best_val_s2": round(best_val2, 6), "path": final_path})
    logger.close()
    print(f"\nSFT completato → {final_path}")

    return {
        "best_val_s1":     best_val,
        "best_val_s2":     best_val2,
        "train_losses":    train_losses,
        "val_losses":      val_losses,
        "checkpoint_path": final_path,
    }


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    sft_cfg = SFTConfig()
    results = run_sft(sft_cfg)
    print(f"Best val s1: {results['best_val_s1']:.4f}")
    print(f"Best val s2: {results['best_val_s2']:.4f}")