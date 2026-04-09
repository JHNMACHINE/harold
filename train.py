"""
Harold v0.6 — train.py  (unified single-GPU + DDP)
====================================================
Entry point unico per training single-GPU e multi-GPU DDP.

Avvio single-GPU:
    torchrun --nproc_per_node=1 train.py

Avvio multi-GPU (es. 4 GPU):
    torchrun --nproc_per_node=4 train.py

Comportamento automatico:
  - world_size=1  → single-GPU, torch.compile abilitato, nessun overhead DDP
  - world_size>1  → DDP attivo, torch.compile disabilitato (instabile con DDP),
                    dataset partizionato per rank tramite seed offset,
                    checkpoint/logging/HF push solo su rank 0,
                    val loss sincronizzata via all_reduce

Compatibilità checkpoint:
  - I checkpoint sono sempre salvati unwrapped (model.module se DDP)
  - Intercambiabili tra run single-GPU e multi-GPU

Changelog v0.6:
  - Aggiornato per Harold v0.6 (architettura Jamba: Mamba2 + Attention + MoE)
  - Rimosso gradient checkpointing (non utilizzato)
"""

import math
import os
import time
import warnings
from collections import deque
from typing import Any, Dict, Optional, Tuple, Union, cast, Protocol, runtime_checkable

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from config import MAX_SKIP_RATIO, ModelConfig, TrainConfig, get_model_config, get_train_config
from optimizer import build_optimizer
from model import Harold, build_model
from dataset import build_loaders, build_loaders_ddp
from logger import AsyncLogger
from checkpoint import save_checkpoint, load_checkpoint
from ddp import DDPContext, is_ddp, is_main, all_reduce_mean, broadcast_model


@runtime_checkable
class TrainableModel(Protocol):
    def compute_loss(
        self,
        x0:             torch.Tensor,
        mask:           torch.Tensor,
        ce_weight:      float = 0.1,
        fixed_t:        Optional[torch.Tensor] = None,
        self_cond_prob: float = 0.0,
        ctx_emb:        Optional[torch.Tensor] = None,
        p_uncond:       float = 0.0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]: ...

    def parameters(self) -> Any: ...
    def update_router_biases(self) -> None: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...


class DiffusionTrainer:
    def __init__(self, model: TrainableModel, config: ModelConfig, train_cfg: TrainConfig,
                 pad_token_id: int = 0):
        self.model        = model
        self.config       = config
        self.train_cfg    = train_cfg
        self.pad_token_id = pad_token_id

    def train_step(self, batch: torch.Tensor) -> Optional[Tuple[torch.Tensor, Dict[str, float]]]:
        mask = (batch != self.pad_token_id)
        if mask.sum() == 0:
            return None
        loss, loss_dict = self.model.compute_loss(
            x0=batch, mask=mask,
            ce_weight=self.train_cfg.ce_loss_weight,
            self_cond_prob=self.train_cfg.self_cond_prob,
        )
        return loss, loss_dict


def get_lr(it: int, cfg: TrainConfig) -> float:
    if it < cfg.warmup_iters:
        return cfg.lr * max(it, 1) / cfg.warmup_iters
    if it >= cfg.max_iters:
        return cfg.min_lr
    ratio = (it - cfg.warmup_iters) / max(cfg.max_iters - cfg.warmup_iters, 1)
    return cfg.min_lr + 0.5 * (1.0 + math.cos(math.pi * ratio)) * (cfg.lr - cfg.min_lr)


@torch.no_grad()
def estimate_loss(
    model:        Harold,
    train_cfg:    TrainConfig,
    val_loader:   DataLoader,
    pad_token_id: int = 50256,
    iter_num:     int = 0,
    logger:       Optional["AsyncLogger"] = None,
) -> float:
    device   = next(model.parameters()).device
    model.eval()

    t_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    all_total: Dict[float, list] = {t: [] for t in t_values}
    all_score: Dict[float, list] = {t: [] for t in t_values}
    all_ce:    Dict[float, list] = {t: [] for t in t_values}

    iterator = iter(val_loader)
    for _ in range(train_cfg.eval_iters):
        try:
            batch = next(iterator)
        except StopIteration:
            break
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        mask      = (input_ids != pad_token_id)
        if mask.sum() == 0:
            continue
        B = input_ids.shape[0]
        for t_val in t_values:
            fixed_t = torch.full((B,), t_val, dtype=torch.float32, device=device)
            with train_cfg.ctx:
                _, loss_dict = model.compute_loss(
                    x0=input_ids, mask=mask,
                    ce_weight=train_cfg.ce_loss_weight,
                    fixed_t=fixed_t, self_cond_prob=0.0,
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


def _run_grad_accum(
    trainer: DiffusionTrainer, train_iter, train_loader: DataLoader,
    train_cfg: TrainConfig, scaler: torch.GradScaler, device: str,
) -> Tuple[float, float, float, int, object]:
    step_loss_sum = step_score_sum = step_ce_sum = 0.0
    valid_count = mb_idx = 0
    max_skip = train_cfg.grad_accum * MAX_SKIP_RATIO

    while valid_count < train_cfg.grad_accum:
        if mb_idx > max_skip:
            break
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch      = next(train_iter)

        input_ids = batch["input_ids"].to(device, non_blocking=True)
        with train_cfg.ctx:
            result = trainer.train_step(input_ids)

        if result is None:
            mb_idx += 1
            continue

        loss, loss_dict = result
        scaler.scale(loss / train_cfg.grad_accum).backward()
        step_loss_sum  += loss.item()
        step_score_sum += loss_dict.get("score", 0.0)
        step_ce_sum    += loss_dict.get("ce",    0.0)
        valid_count += 1
        mb_idx      += 1

    return step_loss_sum, step_score_sum, step_ce_sum, valid_count, train_iter


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
        print("Harold")
        print(f"Modalità:       {'DDP (' + str(world_size) + ' GPU)' if use_ddp else 'Single-GPU'}")
        print(f"Device:         {device}")
        print(f"Dtype:          {train_cfg.dtype}  (scaler={'ON' if train_cfg.use_scaler else 'OFF'})")
        eff = train_cfg.batch_size * train_cfg.grad_accum * world_size
        print(f"Batch effettivo: {eff}  ({train_cfg.batch_size} × {train_cfg.grad_accum} × {world_size} GPU)")

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
        print(f"Harold — {n_params/1000:.2f}B parametri")

    active_model: Union[Harold, DDP] = (
        DDP(model, device_ids=[local_rank], output_device=local_rank)
        if use_ddp else model
    )
    raw_model: Harold = cast(Harold, active_model.module if isinstance(active_model, DDP) else active_model)

    # ── Optimizer ─────────────────────────────────────────────────────────
    optimizer = build_optimizer(active_model, train_cfg)
    scaler = torch.GradScaler("cuda", enabled=train_cfg.use_scaler)

    # ── Dataset ───────────────────────────────────────────────────────────
    if use_ddp:
        train_loader, val_loader = build_loaders_ddp(train_cfg, tokenizer, rank)
    else:
        train_loader, val_loader = build_loaders(train_cfg, tokenizer)

    # Trainer
    trainer = DiffusionTrainer(raw_model, model_cfg, train_cfg, pad_token_id=pad_token_id)

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
        desc="Harold" + (" DDP" if use_ddp else ""),
        disable=not main,
    )

    for iter_num in pbar:
        lr = get_lr(iter_num, train_cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)

        step_loss_sum, step_score_sum, step_ce_sum, valid_count, train_iter = (
            _run_grad_accum(trainer, train_iter, train_loader, train_cfg, scaler, device)
        )
        if valid_count == 0:
            continue

        if train_cfg.use_scaler:
            scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(active_model.parameters(), train_cfg.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        raw_model.update_router_biases()

        avg_loss  = step_loss_sum  / valid_count
        avg_score = step_score_sum / valid_count
        avg_ce    = step_ce_sum    / valid_count
        accum_loss += avg_loss
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

        # ── Val ───────────────────────────────────────────────────────────
        if iter_num % train_cfg.eval_interval == 0 or iter_num == train_cfg.max_iters - 1:
            if iter_num == 0:
                continue

            local_val = estimate_loss(
                raw_model, train_cfg, val_loader, pad_token_id,
                iter_num=iter_num, logger=logger if main else None,
            )

            if use_ddp:
                val_tensor = torch.tensor(local_val, device=device)
                val_loss   = float(all_reduce_mean(val_tensor, world_size).item())
            else:
                val_loss = local_val

            val_losses.append(val_loss)
            avg_train  = accum_loss / max(train_cfg.eval_interval, 1)
            accum_loss = 0.0
            elapsed    = (time.time() - start_time) / 60

            if main:
                print(f"\n[iter {iter_num:7d}] train={avg_train:.4f}  val={val_loss:.4f}  "
                      f"lr={lr:.2e}  elapsed={elapsed:.1f}min")
                if logger:
                    logger.log({"type": "val", "iter": iter_num,
                                "train_loss": round(avg_train, 6), "val_loss": round(val_loss, 6),
                                "lr": lr, "elapsed_min": round(elapsed, 2)})

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_path = os.path.join(train_cfg.checkpoint_dir,
                                             f"{train_cfg.checkpoint_prefix}_best.pt")
                    save_checkpoint(best_path, raw_model, optimizer, scaler,
                                    iter_num, val_loss, model_cfg, train_cfg,
                                    train_losses, val_losses, push_hf=True)
                    print(f"  ★ Best val loss: {val_loss:.4f} → {best_path}")
                    train_cfg.write_latest(iter_num, best_path)
                    if logger:
                        logger.log({"type": "best_checkpoint", "iter": iter_num,
                                    "val_loss": round(val_loss, 6), "path": best_path})

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
                        "elapsed_min": round(elapsed, 2), "final_checkpoint": final_path})
            logger.close()
        print(f"\nTraining completato in {elapsed:.1f} min → {final_path}")

    if use_ddp:
        ctx.teardown() if use_ddp else None

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