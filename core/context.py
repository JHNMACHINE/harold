"""
Harold v0.7 — context.py
=========================
TrainingContext: stato completo del training, passato tra train.py, setup.py e validation.py.
Definito in un modulo separato per evitare dipendenze circolari.

Cambiamenti rispetto a v0.6:
  [v0.7-S3] active_model accetta anche nn.Module — torch.compile ritorna
             nn.Module invece di Harold o DDP, e FSDP wrappa in nn.Module.
             ddp_ctx accetta anche FSDPContext per compatibilita drop-in.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Protocol, Union

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from core.model import Harold
from training.trainer import DiffusionTrainer
from utils.logger import AsyncLogger


class _ParallelCtx(Protocol):
    """Protocol minimo condiviso da DDPContext e FSDPContext."""
    def teardown(self) -> None: ...


@dataclass
class TrainingContext:
    # Core
    model:        Harold
    active_model: Union[Harold, DDP, nn.Module]  # nn.Module copre torch.compile + FSDP
    optimizer:    torch.optim.Optimizer
    scaler:       torch.GradScaler
    trainer:      DiffusionTrainer
    tokenizer:    PreTrainedTokenizerBase

    # Data
    train_loader: DataLoader
    val_loader:   DataLoader
    pad_token_id: int

    # State (mutabile durante il training)
    initial_iter:  int
    best_val_loss: float
    train_losses:  deque
    val_losses:    list

    # Runtime
    device:     str
    use_ddp:    bool
    world_size: int
    main:       bool
    logger:     Optional[AsyncLogger]

    # DDP/FSDP context (per teardown finale)
    # Soddisfatto da DDPContext e FSDPContext via _ParallelCtx Protocol
    ddp_ctx: Optional[_ParallelCtx] = field(default=None)