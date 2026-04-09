"""
Harold v0.5 — train_sft.py
===========================
Supervised Fine-Tuning con Classifier-Free Guidance (CFG).

Pipeline:
  1. Carica checkpoint del pretraining (harold_v05_final.pt)
  2. Strato 1: oasst2 60% + OpenOrca 40% (mix YAML)
  3. Strato 2: OpenOrca 100% (Q&A chain-of-thought)

CFG (Flow Matching):
  - ctx_emb = mean pooling degli embedding del prompt
  - p_uncond=0.1 -> ctx_emb azzerato durante training (unconditional dropout)
  - In inferenza: vel_guided = vel_uncond + cfg_scale*(vel_cond - vel_uncond)

Avvio:
  torchrun --nproc_per_node=1 train_sft.py
  python train_sft.py
"""

import math
import os
import time
import warnings
from typing import Optional, Tuple, Union, cast

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.ddp import DDPContext, is_ddp, is_main, all_reduce_mean, broadcast_model
from core.config import ModelConfig, SFTConfig
from core.dataset import SFTDataset, build_sft_loaders, load_dataset_config
from utils.logger import AsyncLogger
from core.model import Harold, build_model
from torch.nn.parallel import DistributedDataParallel as DDP


def encode_context(
    model:        Harold,
    prompt_ids:   torch.Tensor,
    pad_token_id: int,
) -> torch.Tensor:
    """
    Mean pooling degli embedding del prompt -> ctx_emb (B, D).
    Coerente con _encode_context() in sampler.py.
    """
    with torch.no_grad():
        pad_mask = (prompt_ids != pad_token_id).float()
        emb      = model.token_emb(prompt_ids)
        n_tokens = pad_mask.sum(dim=1, keepdim=True).clamp(min=1)
    return (emb * pad_mask.unsqueeze(-1)).sum(dim=1) / n_tokens


def get_lr(it: int, cfg: SFTConfig, max_iters: int, base_lr: float) -> float:
    if it < cfg.warmup_iters:
        return base_lr * max(it, 1) / cfg.warmup_iters
    if it >= max_iters:
        return cfg.min_lr
    ratio = (it - cfg.warmup_iters) / max(max_iters - cfg.warmup_iters, 1)
    return cfg.min_lr + 0.5 * (1.0 + math.cos(math.pi * ratio)) * (base_lr - cfg.min_lr)


@torch.no_grad()
def estimate_sft_loss(
    model:        Harold,
    sft_cfg:      SFTConfig,
    val_loader:   DataLoader,
    pad_token_id: int,
    iter_num:     int = 0,
    logger:       Optional[AsyncLogger] = None,
) -> float:
    device    = next(model.parameters()).device
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

        prompt_ids   = batch["prompt_ids"].to(device)
        response_ids = batch["response_ids"].to(device)
        resp_mask    = batch["response_mask"].to(device)
        if resp_mask.sum() == 0:
            continue

        ctx_emb = encode_context(model, prompt_ids, pad_token_id)
        B = response_ids.shape[0]

        for t_val in t_values:
            fixed_t = torch.full((B,), t_val, dtype=torch.float32, device=device)
            with sft_cfg.ctx:
                _, loss_dict = model.compute_loss(
                    x0=response_ids, mask=resp_mask,
                    ce_weight=sft_cfg.ce_loss_weight,
                    fixed_t=fixed_t, self_cond_prob=0.0,
                    ctx_emb=ctx_emb, p_uncond=0.0,
                )
            all_total[t_val].append(loss_dict["total"])
            all_score[t_val].append(loss_dict["score"])
            all_ce[t_val].append(loss_dict["ce"])

    model.train()

    per_t_total = {t: float(torch.tensor(v).mean()) if v else float("inf") for t, v in all_total.items()}
    per_t_score = {t: float(torch.tensor(v).mean()) if v else float("inf") for t, v in all_score.items()}
    per_t_ce    = {t: float(torch.tensor(v).mean()) if v else float("inf") for t, v in all_ce.items()}

    if is_main():
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


def run_stage(
    stage:        int,
    model:        Harold,
    model_cfg:    ModelConfig,
    sft_cfg:      SFTConfig,
    optimizer:    torch.optim.Optimizer,
    scaler:       torch.GradScaler,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    pad_token_id: int,
    max_iters:    int,
    base_lr:      float,
    initial_iter: int,
    best_val:     float,
    train_losses: list,
    val_losses:   list,
    logger:       Optional[AsyncLogger],
) -> Tuple[float, list, list]:
    _logger = cast(AsyncLogger, logger)  # sempre non-None su rank 0
    device     = sft_cfg.device
    world_size = sft_cfg.world_size if hasattr(sft_cfg, 'world_size') else 1
    start_time = time.time()
    train_iter = iter(train_loader)
    accum_loss = 0.0

    pbar = tqdm(
        range(initial_iter, max_iters),
        desc=f"Harold v0.5 SFT s{stage}",
        disable=not is_main(),
    )

    for iter_num in pbar:
        lr = get_lr(iter_num, sft_cfg, max_iters, base_lr)
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

            ctx_emb = encode_context(model, prompt_ids, pad_token_id)

            with sft_cfg.ctx:
                loss, loss_dict = model.compute_loss(
                    x0=response_ids, mask=resp_mask,
                    ce_weight=sft_cfg.ce_loss_weight,
                    self_cond_prob=sft_cfg.self_cond_prob,
                    ctx_emb=ctx_emb, p_uncond=sft_cfg.p_uncond,
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
        
        # Assicurati che model abbia il metodo update_router_biases
        if hasattr(model, 'update_router_biases'):
            model.update_router_biases()

        avg_loss  = step_loss_sum  / valid_count
        avg_score = step_score_sum / valid_count
        avg_ce    = step_ce_sum    / valid_count
        accum_loss += avg_loss
        train_losses.append(avg_loss)

        if is_main():
            pbar.set_postfix({
                "loss":  f"{avg_loss:.4f}",
                "score": f"{avg_score:.4f}",
                "ce":    f"{avg_ce:.4f}",
                "lr":    f"{lr:.2e}",
                "grad":  f"{grad_norm:.2f}",
            })
            _logger.log({
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

            local_val = estimate_sft_loss(model, sft_cfg, val_loader, pad_token_id, iter_num, logger)
            if is_ddp():
                val_loss = float(all_reduce_mean(
                    torch.tensor(local_val, device=device), world_size
                ).item())
            else:
                val_loss = local_val
            val_losses.append(val_loss)
            avg_train  = accum_loss / max(sft_cfg.eval_interval, 1)
            accum_loss = 0.0
            elapsed    = (time.time() - start_time) / 60

            if is_main():
                print(
                    f"\n[s{stage} iter {iter_num:7d}] "
                    f"train={avg_train:.4f}  val={val_loss:.4f}  "
                    f"lr={lr:.2e}  elapsed={elapsed:.1f}min"
                )
                _logger.log({
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
                    save_checkpoint(
                        best_path, model, optimizer, scaler,
                        iter_num, val_loss, model_cfg, sft_cfg,
                        train_losses, val_losses, stage=stage,
                    )
                    print(f"  Best val loss s{stage}: {val_loss:.4f} -> {best_path}")
                    sft_cfg.write_latest(stage, iter_num, best_path)
                    _logger.log({"type": "best_checkpoint", "stage": stage,
                                "iter": iter_num, "val_loss": round(val_loss, 6),
                                "path": best_path})

            model.train()

        # Checkpoint periodico
        if iter_num > 0 and iter_num % sft_cfg.save_every == 0 and is_main():
            p = sft_cfg.ckpt_path(stage, iter_num)
            save_checkpoint(
                p, model, optimizer, scaler,
                iter_num, best_val, model_cfg, sft_cfg,
                train_losses, val_losses, full=False, stage=stage,
            )
            sft_cfg.write_latest(stage, iter_num, p)
            print(f"  Checkpoint periodico -> {p}")
            _logger.log({"type": "periodic_checkpoint", "stage": stage,
                        "iter": iter_num, "path": p})

    return best_val, train_losses, val_losses


def run_sft(sft_cfg: SFTConfig) -> dict:
    ctx = DDPContext(default_device=sft_cfg.device).setup()
    device     = ctx.device
    world_size = ctx.world_size
    use_ddp    = ctx.active
    
    # Aggiungi world_size a sft_cfg per uso in run_stage
    sft_cfg.world_size = world_size

    if is_main():
        print("Harold v0.5 — SFT con CFG (Flow Matching)")
        print(f"Modalità:       {'DDP (' + str(world_size) + ' GPU)' if use_ddp else 'Single-GPU'}")
        print(f"Device:         {device}")
        print(f"Dtype:          {sft_cfg.dtype}")
        print(f"Batch virtuale: {sft_cfg.batch_size} x {sft_cfg.grad_accum} = {sft_cfg.effective_batch_size}")
        print(f"p_uncond:       {sft_cfg.p_uncond}  (CFG dropout)")
        print(f"ctx_len:        {sft_cfg.max_ctx_len}  resp_len: {sft_cfg.max_resp_len}")

    tokenizer = AutoTokenizer.from_pretrained(sft_cfg.tokenizer_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_token_id: int = int(tokenizer.pad_token_id)

    if is_main():
        print(f"\nCarico pretraining checkpoint: {sft_cfg.pretrain_ckpt}")
    state     = torch.load(sft_cfg.pretrain_ckpt, map_location="cpu", weights_only=False)
    model_cfg = state["model_cfg"]
    model     = build_model(model_cfg).to(device)

    model: Union[Harold, DDP] = (
        DDP(model, device_ids=[ctx.local_rank], find_unused_parameters=True)
        if use_ddp else model
    )

    raw_model: Harold = cast(Harold, model.module if isinstance(model, DDP) else model)

    # Gestisci il caricamento dei pesi con DDP
    if use_ddp:
        missing, unexpected = model.load_state_dict(state["model_state"], strict=False)
    else:
        missing, unexpected = model.load_state_dict(state["model_state"], strict=False)
    
    if is_main():
        if missing:
            print(f"Pesi mancanti (nuovi layer): {missing}")
        if unexpected:
            print(f"Pesi inattesi (ignorati): {unexpected}")
    del state

    if is_main():
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        if n_params >= 1000:
            print(f"Harold v0.5 SFT — {n_params/1000:.2f}B parametri")
        else:
            print(f"Harold v0.5 SFT — {n_params:.1f}M parametri")

    cfg_params   = list(raw_model.cfg_proj.parameters())
    other_params = [p for n, p in raw_model.named_parameters()
                    if "cfg_proj" not in n and p.requires_grad]

    optimizer = torch.optim.AdamW(
        [
            {"params": cfg_params,   "lr": sft_cfg.lr * 10},
            {"params": other_params, "lr": sft_cfg.lr},
        ],
        betas=(0.9, 0.95),
        weight_decay=0.1,
        fused=True,
    )
    scaler = torch.GradScaler("cuda", enabled=sft_cfg.use_scaler)

    logger: Optional[AsyncLogger] = None
    if is_main():
        os.makedirs(sft_cfg.checkpoint_dir, exist_ok=True)
        log_path = os.path.join(sft_cfg.checkpoint_dir, "training_sft.log")
        logger   = AsyncLogger(log_path)
        print(f"Log SFT -> {log_path}\n")

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
            initial_stage, initial_iter, best_val, train_losses, val_losses = load_checkpoint(
                ckpt_path, model, optimizer, scaler, device, load_stage=True,
            )
        elif is_main():
            print("Nessun SFT checkpoint trovato, parto dal pretraining.")

    # ── Strato 1 ──────────────────────────────────────────────────────────────
    if initial_stage <= 1:
        if is_main():
            print(f"\nStrato 1: oasst2 + OpenOrca — {sft_cfg.max_iters} step\n")

        train_loader, val_loader = build_sft_loaders(
            sft_cfg, tokenizer,
            max_ctx_len=sft_cfg.max_ctx_len,
            max_resp_len=sft_cfg.max_resp_len,
            max_ctx_turns=sft_cfg.max_ctx_turns,
            yaml_path="datasets_config.yaml",
        )

        model.train()
        best_val, train_losses, val_losses = run_stage(
            stage=1,
            model=raw_model, model_cfg=model_cfg, sft_cfg=sft_cfg,
            optimizer=optimizer, scaler=scaler,
            train_loader=train_loader, val_loader=val_loader,
            pad_token_id=pad_token_id,
            max_iters=sft_cfg.max_iters,
            base_lr=sft_cfg.lr,
            initial_iter=initial_iter if initial_stage == 1 else 0,
            best_val=best_val,
            train_losses=train_losses, val_losses=val_losses,
            logger=logger,
        )

        if is_main():
            s1_final = os.path.join(sft_cfg.checkpoint_dir,
                                    f"{sft_cfg.checkpoint_prefix}_s1_final.pt")
            save_checkpoint(
                s1_final, model, optimizer, scaler,
                sft_cfg.max_iters, best_val, model_cfg, sft_cfg,
                train_losses, val_losses, stage=1,
            )
            print(f"\nStrato 1 completato -> {s1_final}")

    # ── Strato 2: OpenOrca 100% ───────────────────────────────────────────────
    if is_main():
        print(f"\nStrato 2: OpenOrca — {sft_cfg.stage2_max_iters} step\n")

    full_cfg  = load_dataset_config("datasets_config.yaml")
    s2_ds_cfg = [d for d in full_cfg["sft"] if d["name"] == "openorca"]
    for d in s2_ds_cfg:
        d["weight"] = 1.0

    train_ds2 = SFTDataset(
        tokenizer=tokenizer, dataset_cfg=s2_ds_cfg, split="train",
        max_ctx_len=sft_cfg.max_ctx_len, max_resp_len=sft_cfg.max_resp_len,
        max_ctx_turns=sft_cfg.max_ctx_turns, val_every=sft_cfg.val_every, seed=43,
    )
    val_ds2 = SFTDataset(
        tokenizer=tokenizer, dataset_cfg=s2_ds_cfg, split="val",
        max_ctx_len=sft_cfg.max_ctx_len, max_resp_len=sft_cfg.max_resp_len,
        max_ctx_turns=sft_cfg.max_ctx_turns, val_every=sft_cfg.val_every, seed=43,
    )
    train_loader2 = DataLoader(train_ds2, batch_size=sft_cfg.batch_size,
                               num_workers=0, pin_memory=False)
    val_loader2   = DataLoader(val_ds2,   batch_size=sft_cfg.batch_size,
                               num_workers=0, pin_memory=False)

    for pg in optimizer.param_groups:
        pg["lr"] = sft_cfg.stage2_lr

    model.train()
    best_val2, train_losses, val_losses = run_stage(
        stage=2,
        model=raw_model, model_cfg=model_cfg, sft_cfg=sft_cfg,
        optimizer=optimizer, scaler=scaler,
        train_loader=train_loader2, val_loader=val_loader2,
        pad_token_id=pad_token_id,
        max_iters=sft_cfg.stage2_max_iters,
        base_lr=sft_cfg.stage2_lr,
        initial_iter=initial_iter if initial_stage == 2 else 0,
        best_val=best_val,
        train_losses=train_losses, val_losses=val_losses,
        logger=logger,
    )

    if is_main():
        final_path = os.path.join(sft_cfg.checkpoint_dir,
                                  f"{sft_cfg.checkpoint_prefix}_final.pt")
        save_checkpoint(
            final_path, model, optimizer, scaler,
            sft_cfg.stage2_max_iters, best_val2, model_cfg, sft_cfg,
            train_losses, val_losses, stage=2,
        )
        sft_cfg.write_latest(2, sft_cfg.stage2_max_iters, final_path)
        assert logger is not None
        logger.log({"type": "finished", "best_val_s1": round(best_val, 6),
                    "best_val_s2": round(best_val2, 6), "path": final_path})
        logger.close()
        print(f"\nSFT completato -> {final_path}")

    ctx.teardown()

    return {
        "best_val_s1":     best_val,
        "best_val_s2":     best_val2,
        "train_losses":    train_losses,
        "val_losses":      val_losses,
    }


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    sft_cfg = SFTConfig()
    results = run_sft(sft_cfg)
    if is_main():
        print(f"Best val s1: {results['best_val_s1']:.4f}")
        print(f"Best val s2: {results['best_val_s2']:.4f}")