"""
Harold v0.6 - train_sft.py
===========================
Supervised Fine-Tuning con Classifier-Free Guidance (CFG).

Pipeline:
  1. Carica checkpoint del pretraining
  2. Strato 1: tulu3 60% + OpenOrca 40% (mix YAML)
  3. Strato 2: OpenOrca 100% (Q&A chain-of-thought)

CFG (Flow Matching):
  - ctx_emb = mean pooling degli embedding del prompt
  - p_uncond=0.1 -> ctx_emb azzerato durante training (unconditional dropout)
  - In inferenza: vel_guided = vel_uncond + cfg_scale*(vel_cond - vel_uncond)

Ottimizzazioni rispetto alla versione precedente:
  [OPT-S1] torch.compile - stesso compile mode del pretraining
  [OPT-S2] ValidationScheduler adattivo - frequenza variabile in base alla stabilità
  [OPT-S3] Quick validation - alterna full (5 t) e quick (t=0.5) per risparmiare tempo
  [OPT-S4] Fix allocazioni CPU - sum/mean puri Python invece di torch.tensor(v).mean()
  [OPT-S5] non_blocking=True in estimate_sft_loss
  [OPT-S6] _optimal_num_workers per stage 2

Avvio:
  torchrun --nproc_per_node=1 main.py --mode sft
"""

import math
import os
import time
import warnings
from collections import deque
from typing import Optional, Tuple, Union, cast

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.ddp import DDPContext, is_ddp, is_main, all_reduce_mean, broadcast_model
from core.config import ModelConfig, SFTConfig
from core.dataset import SFTDataset, build_sft_loaders, load_dataset_config, _optimal_num_workers
from utils.logger import AsyncLogger
from core.model import Harold, build_model
from training.validation import ValidationScheduler
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
    quick:        bool = False,
) -> float:
    """
    Validation SFT.

    quick=True  -> solo t=0.5, metà dei batch - rapida tra le full
    quick=False -> t = [0.1, 0.3, 0.5, 0.7, 0.9], completa
    """
    device   = next(model.parameters()).device
    model.eval()

    # [OPT-S3] Quick validation usa solo t=0.5
    t_values  = [0.5] if quick else [0.1, 0.3, 0.5, 0.7, 0.9]
    eval_iters = sft_cfg.eval_iters // 2 if quick else sft_cfg.eval_iters

    # [OPT-S4] Accumulatori puri Python invece di liste + torch.tensor().mean()
    sum_total = {t: 0.0 for t in t_values}
    sum_score = {t: 0.0 for t in t_values}
    sum_ce    = {t: 0.0 for t in t_values}
    count     = {t: 0   for t in t_values}

    t_cache: dict = {}
    iterator = iter(val_loader)

    for _ in range(eval_iters):
        try:
            batch = next(iterator)
        except StopIteration:
            break

        # [OPT-S5] non_blocking=True
        prompt_ids   = batch["prompt_ids"].to(device, non_blocking=True)
        response_ids = batch["response_ids"].to(device, non_blocking=True)
        resp_mask    = batch["response_mask"].to(device, non_blocking=True)
        if resp_mask.sum() == 0:
            continue

        ctx_emb = encode_context(model, prompt_ids, pad_token_id)
        B = response_ids.shape[0]

        for t_val in t_values:
            key = (t_val, B)
            if key not in t_cache:
                t_cache[key] = torch.full((B,), t_val, dtype=torch.float32, device=device)
            with sft_cfg.ctx:
                _, loss_dict = model.compute_loss(
                    x0=response_ids, mask=resp_mask,
                    ce_weight=sft_cfg.ce_loss_weight,
                    fixed_t=t_cache[key], self_cond_prob=0.0,
                    ctx_emb=ctx_emb, p_uncond=0.0,
                )
            sum_total[t_val] += loss_dict["total"]
            sum_score[t_val] += loss_dict["score"]
            sum_ce[t_val]    += loss_dict["ce"]
            count[t_val]     += 1

    model.train()

    per_t_total = {t: sum_total[t] / max(count[t], 1) for t in t_values}
    per_t_score = {t: sum_score[t] / max(count[t], 1) for t in t_values}
    per_t_ce    = {t: sum_ce[t]    / max(count[t], 1) for t in t_values}

    if is_main():
        if not quick:
            print("  val total: " + "  ".join(f"t={t}:{v:.4f}" for t, v in per_t_total.items()))
            print("  val score: " + "  ".join(f"t={t}:{v:.4f}" for t, v in per_t_score.items()))
            print("  val CE:    " + "  ".join(f"t={t}:{v:.4f}" for t, v in per_t_ce.items()))
        if logger:
            logger.log({
                "type":        "val_detail",
                "iter":        iter_num,
                "quick":       quick,
                "total_per_t": {str(t): round(v, 6) for t, v in per_t_total.items()},
                "score_per_t": {str(t): round(v, 6) for t, v in per_t_score.items()},
                "ce_per_t":    {str(t): round(v, 6) for t, v in per_t_ce.items()},
            })

    valid = [v for v in per_t_total.values() if v < float("inf")]
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
    _logger    = cast(AsyncLogger, logger)
    device     = sft_cfg.device
    world_size = sft_cfg.world_size if hasattr(sft_cfg, "world_size") else 1
    start_time = time.time()
    train_iter = iter(train_loader)
    accum_loss      = 0.0
    steps_since_val = 0

    # [OPT-S2] ValidationScheduler adattivo
    val_scheduler = ValidationScheduler(
        base_interval       = sft_cfg.eval_interval,
        min_interval        = max(100, sft_cfg.eval_interval // 5),
        max_interval        = sft_cfg.eval_interval * 4,
        stability_threshold = 0.03,
        patience            = 3,
    )

    pbar = tqdm(
        range(initial_iter, max_iters),
        desc=f"Harold v0.6 SFT s{stage}",
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

        if hasattr(model, "update_router_biases"):
            model.update_router_biases()

        avg_loss  = step_loss_sum  / valid_count
        avg_score = step_score_sum / valid_count
        avg_ce    = step_ce_sum    / valid_count
        accum_loss      += avg_loss
        steps_since_val += 1
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

        # ── Validation adattiva ───────────────────────────────────────────
        current_loss = accum_loss / max(steps_since_val, 1)
        should_val, reason = val_scheduler.should_validate(
            iter_num, current_loss, force=(iter_num == max_iters - 1),
        )

        if should_val and iter_num > 0:
            # [OPT-S3] alterna full e quick validation
            is_quick = (val_scheduler.total_val_calls % 3 != 0
                        and not "unstable" in reason
                        and iter_num != max_iters - 1)

            local_val = estimate_sft_loss(
                model, sft_cfg, val_loader, pad_token_id,
                iter_num=iter_num, logger=logger if is_main() else None,
                quick=is_quick,
            )

            if is_ddp():
                val_tensor = torch.tensor(local_val, device=device)
                val_loss   = float(all_reduce_mean(val_tensor, world_size).item())
            else:
                val_loss = local_val

            val_scheduler.record(iter_num, val_loss)
            avg_train       = accum_loss / max(steps_since_val, 1)
            accum_loss      = 0.0
            steps_since_val = 0
            elapsed         = (time.time() - start_time) / 60

            if is_main():
                val_type = "quick" if is_quick else "full"
                if not is_quick:
                    val_losses.append(val_loss)
                    print(
                        f"\n[s{stage} iter {iter_num:7d}] "
                        f"train={avg_train:.4f}  val={val_loss:.4f}  "
                        f"lr={lr:.2e}  elapsed={elapsed:.1f}min  [{val_type}|{reason}]"
                    )
                else:
                    print(f"  [quick val] iter={iter_num} loss={val_loss:.4f} ({reason})")

                _logger.log({
                    "type":        "val",
                    "stage":       stage,
                    "iter":        iter_num,
                    "train_loss":  round(avg_train, 6),
                    "val_loss":    round(val_loss,  6),
                    "val_type":    val_type,
                    "reason":      reason,
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
                    print(f"  ★ Best val loss s{stage}: {val_loss:.4f} → {best_path}")
                    sft_cfg.write_latest(stage, iter_num, best_path)
                    _logger.log({"type": "best_checkpoint", "stage": stage,
                                 "iter": iter_num, "val_loss": round(val_loss, 6),
                                 "path": best_path})

            model.train()

        # ── Checkpoint periodico ──────────────────────────────────────────
        if iter_num > 0 and iter_num % sft_cfg.save_every == 0 and is_main():
            p = sft_cfg.ckpt_path(stage, iter_num)
            save_checkpoint(
                p, model, optimizer, scaler,
                iter_num, best_val, model_cfg, sft_cfg,
                train_losses, val_losses, full=False, stage=stage,
            )
            sft_cfg.write_latest(stage, iter_num, p)
            print(f"  Checkpoint periodico → {p}")
            _logger.log({"type": "periodic_checkpoint", "stage": stage,
                         "iter": iter_num, "path": p})

    return best_val, train_losses, val_losses


def run_sft(sft_cfg: SFTConfig) -> dict:
    ctx        = DDPContext(default_device=sft_cfg.device).setup()
    device     = ctx.device
    world_size = ctx.world_size
    use_ddp    = ctx.active
    sft_cfg.world_size = world_size

    if is_main():
        print("Harold v0.6 - SFT con CFG (Flow Matching)")
        print(f"Modalità:       {'DDP (' + str(world_size) + ' GPU)' if use_ddp else 'Single-GPU'}")
        print(f"Device:         {device}")
        print(f"Dtype:          {sft_cfg.dtype}")
        print(f"Batch virtuale: {sft_cfg.batch_size} x {sft_cfg.grad_accum} = {sft_cfg.effective_batch_size}")
        print(f"p_uncond:       {sft_cfg.p_uncond}  (CFG dropout)")
        print(f"ctx_len:        {sft_cfg.max_ctx_len}  resp_len: {sft_cfg.max_resp_len}")

    if device.startswith("cuda"):
        torch.backends.cudnn.benchmark     = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    tokenizer = AutoTokenizer.from_pretrained(
        sft_cfg.tokenizer_model,
        token=os.environ.get("HF_TOKEN"),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_token_id: int = int(tokenizer.pad_token_id)

    if is_main():
        print(f"\nCarico pretraining checkpoint: {sft_cfg.pretrain_ckpt}")
    state     = torch.load(sft_cfg.pretrain_ckpt, map_location="cpu", weights_only=False)
    model_cfg = state["model_cfg"]
    model     = build_model(model_cfg).to(device)

    # [OPT-S1] torch.compile - solo single-GPU (DDP non supportato)
    if not use_ddp:
        use_compile = (
            getattr(sft_cfg, "use_compile", True)
            and hasattr(torch, "compile")
            and device.startswith("cuda")
            and torch.cuda.get_device_capability()[0] >= 7
        )
        if use_compile:
            compile_mode = getattr(sft_cfg, "compile_mode", "reduce-overhead")
            if is_main():
                print(f"torch.compile() abilitato (mode='{compile_mode}')")
            model = cast(Harold, torch.compile(model, mode=compile_mode))
        elif is_main():
            print("torch.compile() disabilitato")
    elif is_main():
        print("torch.compile() disabilitato (DDP)")

    active_model: Union[Harold, DDP] = (
        DDP(model, device_ids=[ctx.local_rank], find_unused_parameters=True)
        if use_ddp else model
    )
    raw_model: Harold = cast(Harold, active_model.module if isinstance(active_model, DDP) else active_model)

    raw_model.load_state_dict(state["model_state"], strict=False)
    del state

    if is_main():
        n_params = sum(p.numel() for p in raw_model.parameters()) / 1e6
        print(f"Harold v0.6 SFT - {n_params/1000:.2f}B parametri" if n_params >= 1000
              else f"Harold v0.6 SFT - {n_params:.1f}M parametri")

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
        print(f"Log SFT → {log_path}\n")

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
                ckpt_path, raw_model, optimizer, scaler, device, load_stage=True,
            )
        elif is_main():
            print("Nessun SFT checkpoint trovato, parto dal pretraining.")

    if use_ddp:
        broadcast_model(raw_model)

    # ── Strato 1 ──────────────────────────────────────────────────────────────
    if initial_stage <= 1:
        if is_main():
            print(f"\nStrato 1: tulu3 + OpenOrca - {sft_cfg.max_iters} step\n")

        train_loader, val_loader = build_sft_loaders(
            sft_cfg, tokenizer,
            max_ctx_len=sft_cfg.max_ctx_len,
            max_resp_len=sft_cfg.max_resp_len,
            max_ctx_turns=sft_cfg.max_ctx_turns,
            yaml_path="datasets_config.yaml",
        )

        active_model.train()
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
                s1_final, raw_model, optimizer, scaler,
                sft_cfg.max_iters, best_val, model_cfg, sft_cfg,
                train_losses, val_losses, stage=1,
            )
            print(f"\nStrato 1 completato → {s1_final}")

    # ── Strato 2: OpenOrca 100% ───────────────────────────────────────────────
    if is_main():
        print(f"\nStrato 2: OpenOrca - {sft_cfg.stage2_max_iters} step\n")

    full_cfg  = load_dataset_config("datasets_config.yaml")
    s2_ds_cfg = [d for d in full_cfg["sft"] if d["name"] == "openorca"]
    for d in s2_ds_cfg:
        d["weight"] = 1.0

    # [OPT-S6] _optimal_num_workers invece di hardcoded 0
    num_workers = _optimal_num_workers(max_workers=2)
    prefetch    = 2 if num_workers > 0 else None

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
    train_loader2 = DataLoader(
        train_ds2, batch_size=sft_cfg.batch_size,
        num_workers=num_workers, pin_memory=True,
        prefetch_factor=prefetch, persistent_workers=num_workers > 0,
    )
    val_loader2 = DataLoader(
        val_ds2, batch_size=sft_cfg.batch_size,
        num_workers=num_workers, pin_memory=True,
        prefetch_factor=prefetch, persistent_workers=num_workers > 0,
    )

    for pg in optimizer.param_groups:
        pg["lr"] = sft_cfg.stage2_lr

    active_model.train()
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

    final_path = os.path.join(sft_cfg.checkpoint_dir, f"{sft_cfg.checkpoint_prefix}_final.pt")
    if is_main():
        save_checkpoint(
            final_path, raw_model, optimizer, scaler,
            sft_cfg.stage2_max_iters, best_val2, model_cfg, sft_cfg,
            train_losses, val_losses, stage=2,
        )
        sft_cfg.write_latest(2, sft_cfg.stage2_max_iters, final_path)
        assert logger is not None
        logger.log({"type": "finished", "best_val_s1": round(best_val, 6),
                    "best_val_s2": round(best_val2, 6), "path": final_path})
        logger.close()
        print(f"\nSFT completato → {final_path}")

    ctx.teardown()

    return {
        "best_val_s1":     best_val,
        "best_val_s2":     best_val2,
        "train_losses":    train_losses,
        "val_losses":      val_losses,
        "best_val_loss":   min(best_val, best_val2),
    }


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    sft_cfg = SFTConfig()
    results = run_sft(sft_cfg)
    if is_main():
        print(f"Best val s1: {results['best_val_s1']:.4f}")
        print(f"Best val s2: {results['best_val_s2']:.4f}")