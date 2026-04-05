"""
Harold v0.4 — train.py  (unified single-GPU + DDP)
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

Changelog:
  - Rimosso _patch_loader_for_perf (gestito in dataset.py)
  - use_compile letto da train_cfg
  - estimate_loss: loop per t_value (evita OOM su GPU < 80GB)
  - Deque con maxlen da train_cfg.loss_history_size
  - save_checkpoint: full=True (best/final) vs full=False (periodici)
  - DDP: partizionamento dataset per seed, all_reduce val loss
"""

import math
import os
import time
import warnings
from collections import deque
from typing import Dict, Optional, Tuple

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from config import ModelConfig, TrainConfig, get_model_config, get_train_config
from model import Harold, build_model
from dataset import build_loaders, build_loaders_ddp
from logger import AsyncLogger

MAX_SKIP_RATIO = 10


def _setup_ddp() -> Tuple[int, int, int]:
    dist.init_process_group(backend="nccl")
    rank       = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def _cleanup_ddp() -> None:
    dist.destroy_process_group()


def _is_main(rank: int) -> bool:
    return rank == 0


def _all_reduce_mean(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor / world_size


def _is_ddp_available() -> bool:
    return (
        "RANK" in os.environ
        and "WORLD_SIZE" in os.environ
        and int(os.environ.get("WORLD_SIZE", 1)) > 1
    )


class DiffusionTrainer:
    def __init__(self, model: Harold, config: ModelConfig, train_cfg: TrainConfig,
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


def save_checkpoint(
    path:         str,
    model:        Harold,
    optimizer:    torch.optim.Optimizer,
    scaler:       torch.GradScaler,
    iter_num:     int,
    val_loss:     float,
    model_cfg:    ModelConfig,
    train_cfg:    TrainConfig,
    train_losses: deque,
    val_losses:   list,
    push_hf:      bool = False,
    hf_repo_id:   str  = "JHN-MACHINE/harold-v0.4",
    full:         bool = True,
) -> None:
    ckpt = {
        "iter_num":     iter_num,
        "model_state":  model.state_dict(),
        "val_loss":     val_loss,
        "model_cfg":    model_cfg,
        "train_cfg":    train_cfg,
        "train_losses": list(train_losses),
        "val_losses":   val_losses,
    }
    if full:
        ckpt["optimizer_state"] = optimizer.state_dict()
        ckpt["scaler_state"]    = scaler.state_dict()
    torch.save(ckpt, path)

    if push_hf:
        import threading
        def _push():
            try:
                from huggingface_hub import HfApi
                hf_token = os.environ.get("HF_TOKEN")
                if not hf_token:
                    print("HF_TOKEN non trovato — skip push HuggingFace.")
                    return
                api = HfApi()
                api.create_repo(repo_id=hf_repo_id, repo_type="model", exist_ok=True, token=hf_token)
                api.upload_file(
                    path_or_fileobj=path,
                    path_in_repo="harold-v0.4-700M.pt",
                    repo_id=hf_repo_id, repo_type="model", token=hf_token,
                )
                print(f"HuggingFace → {hf_repo_id}/harold-v0.4-700M.pt")
            except Exception as e:
                print(f"Errore push HuggingFace: {e}")
        threading.Thread(target=_push, daemon=True).start()


def load_checkpoint(
    path:      str,
    model:     Harold,
    optimizer: torch.optim.Optimizer,
    scaler:    torch.GradScaler,
    device:    str,
) -> Tuple[int, float, list, list]:
    print(f"Carico checkpoint: {path}")
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    if "optimizer_state" in state:
        optimizer.load_state_dict(state["optimizer_state"])
        print("  optimizer state caricato")
    else:
        print("  optimizer state assente (checkpoint periodico) — riparte da zero")
    if "scaler_state" in state:
        scaler.load_state_dict(state["scaler_state"])
    iter_num     = state.get("iter_num", 0) + 1
    best_val     = state.get("val_loss", float("inf"))
    train_losses = state.get("train_losses", [])
    val_losses   = state.get("val_losses", [])
    del state
    return iter_num, best_val, train_losses, val_losses

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

    # Setup
    use_ddp    = _is_ddp_available()
    rank       = 0
    local_rank = 0
    world_size = 1

    if use_ddp:
        rank, local_rank, world_size = _setup_ddp()
        device = f"cuda:{local_rank}"
    else:
        device = train_cfg.device

    main = _is_main(rank)

    if main:
        print("Harold v0.4 — VP-SDE Continuous Diffusion")
        print(f"Modalità:       {'DDP (' + str(world_size) + ' GPU)' if use_ddp else 'Single-GPU'}")
        print(f"Device:         {device}")
        print(f"Dtype:          {train_cfg.dtype}  (scaler={'ON' if train_cfg.use_scaler else 'OFF'})")
        eff = train_cfg.batch_size * train_cfg.grad_accum * world_size
        print(f"Batch effettivo:{eff}  ({train_cfg.batch_size} x {train_cfg.grad_accum} x {world_size} GPU)")
        print(f"Beta:           [{model_cfg.diffusion_beta_min}, {model_cfg.diffusion_beta_max}]")

    if device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(train_cfg.tokenizer_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_token_id = tokenizer.pad_token_id
    model_cfg.vocab_size    = tokenizer.vocab_size
    model_cfg.mask_token_id = int(getattr(tokenizer, "mask_token_id", None) or tokenizer.vocab_size)

    # Modello
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
            model = torch.compile(model, mode=compile_mode)
        elif main:
            print("torch.compile() disabilitato")
    elif main:
        print("torch.compile() disabilitato (DDP)")

    if main:
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"Harold v0.4 — {n_params:.1f}M parametri totali")

    if use_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)  # type: ignore

    raw_model: Harold = model.module if isinstance(model, DDP) else model  # type: ignore

    # Optimizer 
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg.lr,
        betas=(0.9, 0.95), weight_decay=0.1, fused=True,
    )
    scaler = torch.GradScaler("cuda", enabled=train_cfg.use_scaler)

    # Dataset 
    if use_ddp:
        train_loader, val_loader = build_loaders_ddp(train_cfg, tokenizer, rank)
    else:
        train_loader, val_loader = build_loaders(train_cfg, tokenizer)

    # Trainer 
    trainer = DiffusionTrainer(model, model_cfg, train_cfg, pad_token_id=pad_token_id)  # type: ignore

    # Checkpoint resume
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
        if ckpt_path and os.path.isfile(ckpt_path):
            initial_iter, best_val_loss, _tl, val_losses = load_checkpoint(
                ckpt_path, raw_model, optimizer, scaler, device  # type: ignore
            )
            train_losses.extend(_tl)
        elif main:
            print("Nessun checkpoint trovato, parto da zero.")

    if use_ddp:
        for param in model.parameters():
            dist.broadcast(param.data, src=0)

    # Logger 
    logger = None
    if main:
        log_path = os.path.join(train_cfg.checkpoint_dir, "training.log")
        logger   = AsyncLogger(log_path, flush_every=10)
        print(f"Log -> {log_path}\nAvvio training -> {train_cfg.max_iters} optimizer steps\n")

    # Loop
    model.train()
    start_time = time.time()
    train_iter = iter(train_loader)
    accum_loss = 0.0

    pbar = tqdm(
        range(initial_iter, train_cfg.max_iters + 1),
        desc="Harold v0.4" + (" DDP" if use_ddp else ""),
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
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.max_grad_norm)
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

        # Val 
        if iter_num % train_cfg.eval_interval == 0 or iter_num == train_cfg.max_iters - 1:
            if iter_num == 0:
                continue

            local_val = estimate_loss(
                raw_model, train_cfg, val_loader, pad_token_id,
                iter_num=iter_num, logger=logger if main else None,
            )

            if use_ddp:
                val_tensor = torch.tensor(local_val, device=device)
                val_loss   = float(_all_reduce_mean(val_tensor, world_size).item())
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
                    save_checkpoint(best_path, raw_model, optimizer, scaler,  # type: ignore
                                    iter_num, val_loss, model_cfg, train_cfg,
                                    train_losses, val_losses, push_hf=True)
                    print(f"  ★ Best val loss: {val_loss:.4f} → {best_path}")
                    train_cfg.write_latest(iter_num, best_path)
                    if logger:
                        logger.log({"type": "best_checkpoint", "iter": iter_num,
                                    "val_loss": round(val_loss, 6), "path": best_path})

            model.train()

        # Checkpoint periodico
        if iter_num > 0 and iter_num % train_cfg.save_every == 0 and main:
            p = train_cfg.ckpt_path(iter_num)
            save_checkpoint(p, raw_model, optimizer, scaler,  # type: ignore
                            iter_num, best_val_loss, model_cfg, train_cfg,
                            train_losses, val_losses, full=False)
            train_cfg.write_latest(iter_num, p)
            print(f"  Checkpoint periodico → {p}")
            if logger:
                logger.log({"type": "periodic_checkpoint", "iter": iter_num, "path": p})

    elapsed    = (time.time() - start_time) / 60
    final_path = os.path.join(train_cfg.checkpoint_dir,
                              f"{train_cfg.checkpoint_prefix}_final.pt")

    if main:
        save_checkpoint(final_path, raw_model, optimizer, scaler,  # type: ignore
                        train_cfg.max_iters, best_val_loss, model_cfg, train_cfg,
                        train_losses, val_losses, push_hf=True)
        train_cfg.write_latest(train_cfg.max_iters, final_path)
        if logger:
            logger.log({"type": "finished", "total_iters": train_cfg.max_iters,
                        "best_val_loss": round(best_val_loss, 6),
                        "elapsed_min": round(elapsed, 2), "final_checkpoint": final_path})
            logger.close()
        print(f"\nTraining completato in {elapsed:.1f} min → {final_path}")

    if use_ddp:
        _cleanup_ddp()

    return {"train_losses": list(train_losses), "val_losses": val_losses,
            "best_val_loss": best_val_loss, "train_time_minutes": elapsed,
            "checkpoint_path": final_path}


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    model_cfg = get_model_config()
    train_cfg = get_train_config()
    results   = run_training(model_cfg, train_cfg)
    if _is_main(int(os.environ.get("RANK", 0))):
        print(f"Best val loss: {results['best_val_loss']:.4f}")