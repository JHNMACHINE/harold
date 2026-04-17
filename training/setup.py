"""
Harold v0.7 — setup.py
========================
build_training_context: inizializza tutto il necessario per il training loop.
Ritorna un dataclass con model, optimizer, loaders, logger, ecc.

Cambiamenti rispetto a v0.6:
  [v0.7-S1] Stringa versione aggiornata a Harold v0.7.
  [v0.7-S2] os.makedirs per checkpoint_dir al setup.
  [v0.7-S3] Supporto FSDP via TrainConfig.use_fsdp (default False).
             Quando use_fsdp=True e world_size > 1:
             - Usa FSDPContext invece di DDPContext
             - Wrappa il modello con wrap_model_fsdp (JambaBlock policy)
             - torch.compile abilitato con use_orig_params=True
             - active_model e il modello FSDP, raw_model e lo stesso oggetto
               (FSDP non ha .module come DDP)
             Retrocompatibile: use_fsdp=False usa DDP come prima.
"""

import os
from collections import deque
from typing import Optional, Union, cast

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer

from core.config import ModelConfig, TrainConfig, HF_FILENAME
from core.model import Harold, build_model
from training.optimizer import build_optimizer
from training.trainer import DiffusionTrainer
from core.dataset import build_loaders, build_loaders_ddp
from utils.logger import AsyncLogger
from utils.checkpoint import load_checkpoint
from utils.ddp import DDPContext, is_ddp, is_main, broadcast_model
from core.context import TrainingContext


def build_training_context(
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
) -> TrainingContext:
    """
    Inizializza e ritorna tutto il necessario per il training loop.

    Gestisce: DDP/FSDP, tokenizer, modello, torch.compile, optimizer,
    dataset, checkpoint resume, logger.
    """
    # ── Parallelismo / device ─────────────────────────────────────────────
    use_fsdp   = getattr(train_cfg, "use_fsdp", False)
    use_ddp    = is_ddp() and not use_fsdp
    rank       = 0
    local_rank = 0
    world_size = 1
    ddp_ctx:  Optional[DDPContext] = None
    fsdp_ctx                       = None

    if use_fsdp and is_ddp():
        from utils.fsdp import FSDPContext
        fsdp_ctx   = FSDPContext().setup()
        rank, local_rank, world_size = fsdp_ctx.rank, fsdp_ctx.local_rank, fsdp_ctx.world_size
        device     = f"cuda:{local_rank}"
    elif use_ddp:
        ddp_ctx    = DDPContext().setup()
        rank, local_rank, world_size = ddp_ctx.rank, ddp_ctx.local_rank, ddp_ctx.world_size
        device     = f"cuda:{local_rank}"
    else:
        device     = train_cfg.device

    main = is_main()

    if main:
        mode_str = (
            f"FSDP ({world_size} GPU)" if use_fsdp and world_size > 1
            else f"DDP ({world_size} GPU)" if use_ddp
            else "Single-GPU"
        )
        print("Harold v0.7 — Jamba (Mamba3 + Attention + MoE) + Flow Matching + x0-prediction")
        print(f"Modalita:       {mode_str}")
        print(f"Device:         {device}")
        print(f"Dtype:          {train_cfg.dtype}  (scaler={'ON' if train_cfg.use_scaler else 'OFF'})")
        eff = train_cfg.batch_size * train_cfg.grad_accum * world_size
        print(f"Batch effettivo:{eff}  ({train_cfg.batch_size} x {train_cfg.grad_accum} x {world_size} GPU)")

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
    model_cfg.mask_token_id = int(
        getattr(tokenizer, "mask_token_id", None) or tokenizer.vocab_size
    )

    # ── Modello ───────────────────────────────────────────────────────────
    if use_fsdp and is_ddp():
        model = build_model(model_cfg)           # CPU — FSDP sposta su GPU
    else:
        model = build_model(model_cfg).to(device)

    # ── Wrapping: FSDP / DDP / single-GPU ────────────────────────────────
    if use_fsdp and is_ddp():
        from utils.fsdp import wrap_model_fsdp
        active_model = wrap_model_fsdp(model, local_rank, mixed_precision=True)
        raw_model    = cast(Harold, active_model)

        use_compile = (
            getattr(train_cfg, "use_compile", True)
            and hasattr(torch, "compile")
            and torch.cuda.get_device_capability()[0] >= 7
        )
        if use_compile:
            compile_mode = getattr(train_cfg, "compile_mode", "reduce-overhead")
            if main:
                print(f"torch.compile() abilitato (mode='{compile_mode}', FSDP)")
            active_model = cast(Harold, torch.compile(active_model, mode=compile_mode))
        elif main:
            print("torch.compile() disabilitato")

    elif use_ddp:
        active_model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        raw_model    = cast(Harold, active_model.module)
        if main:
            print("torch.compile() disabilitato (DDP)")

    else:
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
        active_model = model
        raw_model    = cast(Harold, model)

    if main:
        n_params = sum(p.numel() for p in raw_model.parameters()) / 1e6
        label    = f"{n_params/1000:.2f}B" if n_params >= 1000 else f"{n_params:.1f}M"
        gc_str   = " + GradCkpt" if getattr(model_cfg, "use_gradient_checkpointing", False) else ""
        print(f"Harold v0.7 — {label} parametri totali{gc_str}")

    # [v0.7-OPT9] Pre-campiona buffer timestep dopo build — zero allocazioni nel loop
    raw_model.schedule.warmup_buffer(size=4096, device=device)

    # ── Optimizer + scaler ────────────────────────────────────────────────
    optimizer = build_optimizer(active_model, train_cfg)
    scaler    = torch.GradScaler("cuda", enabled=train_cfg.use_scaler)

    # ── Dataset ───────────────────────────────────────────────────────────
    if use_ddp or (use_fsdp and is_ddp()):
        train_loader, val_loader = build_loaders_ddp(train_cfg, tokenizer, rank)
    else:
        train_loader, val_loader = build_loaders(train_cfg, tokenizer)

    trainer = DiffusionTrainer(raw_model, model_cfg, train_cfg, pad_token_id=pad_token_id)

    # ── Checkpoint resume ─────────────────────────────────────────────────
    if main:
        os.makedirs(train_cfg.checkpoint_dir, exist_ok=True)

    initial_iter  = 0
    best_val_loss = float("inf")
    train_losses: deque = deque(maxlen=getattr(train_cfg, "loss_history_size", 100_000))
    val_losses:   list  = []

    if train_cfg.preload:
        if train_cfg.preload == "latest":
            ckpt_path = (train_cfg.read_latest() or (None, None))[1]
        else:
            ckpt_path = train_cfg.preload

        if not ckpt_path:
            ckpt_path = os.path.join(train_cfg.checkpoint_dir, HF_FILENAME)

        if use_fsdp and is_ddp():
            from utils.fsdp import load_fsdp_checkpoint
            initial_iter, best_val_loss, _tl, val_losses = load_fsdp_checkpoint(
                ckpt_path, active_model, optimizer, device
            )
        else:
            initial_iter, best_val_loss, _tl, val_losses = load_checkpoint(
                ckpt_path, raw_model, optimizer, scaler, device
            )
        train_losses.extend(_tl)

    if use_ddp:
        broadcast_model(raw_model)

    # ── Logger ────────────────────────────────────────────────────────────
    logger: Optional[AsyncLogger] = None
    if main:
        log_path = os.path.join(train_cfg.checkpoint_dir, "training.log")
        logger   = AsyncLogger(log_path, flush_every=10)
        print(f"Log -> {log_path}\nAvvio training -> {train_cfg.max_iters} optimizer steps\n")

    return TrainingContext(
        model        = raw_model,
        active_model = active_model,
        optimizer    = optimizer,
        scaler       = scaler,
        trainer      = trainer,
        tokenizer    = tokenizer,
        train_loader = train_loader,
        val_loader   = val_loader,
        pad_token_id = pad_token_id,
        initial_iter = initial_iter,
        best_val_loss= best_val_loss,
        train_losses = train_losses,
        val_losses   = val_losses,
        device       = device,
        use_ddp      = use_ddp or (use_fsdp and is_ddp()),
        world_size   = world_size,
        main         = main,
        logger       = logger,
        ddp_ctx      = ddp_ctx or fsdp_ctx,
    )