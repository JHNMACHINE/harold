"""
Harold v0.6 — setup.py
========================
build_training_context: inizializza tutto il necessario per il training loop.
Ritorna un dataclass con model, optimizer, loaders, logger, ecc.
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

    Gestisce: DDP, tokenizer, modello, torch.compile, optimizer,
    dataset, checkpoint resume, logger.
    """
    # ── DDP / device ──────────────────────────────────────────────────────
    use_ddp    = is_ddp()
    rank       = 0
    local_rank = 0
    world_size = 1
    ddp_ctx: Optional[DDPContext] = None

    if use_ddp:
        ddp_ctx = DDPContext().setup()
        rank, local_rank, world_size = ddp_ctx.rank, ddp_ctx.local_rank, ddp_ctx.world_size
        device = f"cuda:{local_rank}"
    else:
        device = train_cfg.device

    main = is_main()

    if main:
        print("Harold v0.6 — Jamba (Mamba2 + Attention + MoE) + Flow Matching")
        print(f"Modalità:       {'DDP (' + str(world_size) + ' GPU)' if use_ddp else 'Single-GPU'}")
        print(f"Device:         {device}")
        print(f"Dtype:          {train_cfg.dtype}  (scaler={'ON' if train_cfg.use_scaler else 'OFF'})")
        eff = train_cfg.batch_size * train_cfg.grad_accum * world_size
        print(f"Batch effettivo:{eff}  ({train_cfg.batch_size} × {train_cfg.grad_accum} × {world_size} GPU)")

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
        print(f"Harold v0.6 — {n_params:.1f}M parametri totali")

    active_model: Union[Harold, DDP] = (
        DDP(model, device_ids=[local_rank], output_device=local_rank)
        if use_ddp else model
    )
    raw_model: Harold = cast(
        Harold,
        active_model.module if isinstance(active_model, DDP) else active_model,
    )

    # ── Optimizer + scaler ────────────────────────────────────────────────
    optimizer = build_optimizer(active_model, train_cfg)
    scaler    = torch.GradScaler("cuda", enabled=train_cfg.use_scaler)

    # ── Dataset ───────────────────────────────────────────────────────────
    if use_ddp:
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
        # Cerca il checkpoint nell'ordine: latest.json → path esplicito → HuggingFace
        if train_cfg.preload == "latest":
            ckpt_path = (train_cfg.read_latest() or (None, None))[1]
        else:
            ckpt_path = train_cfg.preload

        # Se non trovato localmente, usa il path HF come fallback
        # load_checkpoint scaricherà da HF se il file non esiste
        if not ckpt_path:
            ckpt_path = os.path.join(train_cfg.checkpoint_dir, HF_FILENAME)

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
        use_ddp      = use_ddp,
        world_size   = world_size,
        main         = main,
        logger       = logger,
        ddp_ctx      = ddp_ctx,
    )