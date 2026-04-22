"""
Harold v0.5 — ddp.py
=====================
Utilities DDP condivise tra train.py e train_sft.py.

Uso:
  from ddp import setup, cleanup, is_main, is_ddp, all_reduce_mean, DDPContext

  with DDPContext() as ctx:
      if ctx.main:
          print(f"rank 0 di {ctx.world_size}")
"""

import os
import torch
import torch.distributed as dist
from typing import Optional


def is_ddp() -> bool:
    """True se torchrun ha lanciato con world_size > 1."""
    return (
        "RANK"       in os.environ
        and "WORLD_SIZE" in os.environ
        and int(os.environ.get("WORLD_SIZE", 1)) > 1
    )


def is_main() -> bool:
    """True se questo processo è rank 0."""
    return int(os.environ.get("RANK", 0)) == 0


def setup() -> tuple[int, int, int]:
    """
    Inizializza il process group NCCL.
    Ritorna (rank, local_rank, world_size).
    """
    # 1. Ottieni il local_rank (fornito da torchrun / ambiente distribuito)
    local_rank = int(os.environ["LOCAL_RANK"])
    
    # 2. Imposta il dispositivo CUDA per questo processo
    torch.cuda.set_device(local_rank)
    
    # 3. Inizializza il process group (NCCL userà il dispositivo corrente)
    dist.init_process_group(backend="nccl")
    
    # 4. Ora possiamo ottenere rank e world_size
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    return rank, local_rank, world_size


def cleanup() -> None:
    """Distrugge il process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def all_reduce_mean(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    """Reduce della media tra tutti i rank."""
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor / world_size


def broadcast_model(model: torch.nn.Module, src: int = 0) -> None:
    """Broadcast dei parametri dal rank src a tutti gli altri."""
    for param in model.parameters():
        dist.broadcast(param.data, src=src)


class DDPContext:
    """
    Context manager per setup/cleanup DDP.

    Uso:
      ctx = DDPContext()
      ctx.setup()   # chiama setup() se is_ddp()
      ...
      ctx.teardown()

    Attributi pubblici:
      ctx.rank        int
      ctx.local_rank  int
      ctx.world_size  int
      ctx.main        bool  — True se rank 0
      ctx.active      bool  — True se DDP abilitato
      ctx.device      str   — "cuda:N" o train_cfg.device
    """

    def __init__(self, default_device: str = "cuda") -> None:
        self.rank        = 0
        self.local_rank  = 0
        self.world_size  = 1
        self.main        = True
        self.active      = False
        self.device      = default_device

    def setup(self) -> "DDPContext":
        if is_ddp():
            self.rank, self.local_rank, self.world_size = setup()
            self.main   = self.rank == 0
            self.active = True
            self.device = f"cuda:{self.local_rank}"
        else:
            self.rank        = 0
            self.local_rank  = 0
            self.world_size  = 1
            self.main        = True
            self.active      = False
        return self

    def teardown(self) -> None:
        if self.active:
            cleanup()

    def __enter__(self) -> "DDPContext":
        return self.setup()

    def __exit__(self, *_) -> None:
        self.teardown()