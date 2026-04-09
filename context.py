"""
Harold v0.6 — context.py
=========================
TrainingContext: stato completo del training, passato tra train.py, setup.py e validation.py.
Definito in un modulo separato per evitare dipendenze circolari.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Union

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from model import Harold
from trainer import DiffusionTrainer
from logger import AsyncLogger
from ddp import DDPContext


@dataclass
class TrainingContext:
    # Core
    model:        Harold
    active_model: Union[Harold, DDP]
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

    # DDP context (per teardown finale)
    ddp_ctx: Optional[DDPContext] = field(default=None)