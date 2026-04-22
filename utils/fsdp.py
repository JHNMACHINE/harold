"""
Harold v0.7 — fsdp.py
======================
Utilities FSDP (Fully Sharded Data Parallel) per il full run multi-GPU.

FSDP vs DDP per Harold 3B:
  - DDP replica i parametri su ogni GPU → 3.2B * N GPU in VRAM totale
  - FSDP sharda parametri, gradienti e stato ottimizzatore tra le GPU
    → ~1/N della VRAM per GPU, permette batch size molto più grandi

Strategia di sharding per Harold:
  - AUTO_WRAP: wrappa automaticamente i moduli > min_num_params
  - Layer wrappati separatamente: JambaBlock (40x), token_emb, norm_out
  - Mamba3Block NON viene wrappato separatamente — i kernel Triton richiedono
    che il modulo sia intero su una GPU durante il forward
  - MixedPrecision: parametri bf16, gradienti bf16, buffer bf16

Cambiamenti rispetto alla versione precedente:
  [v0.7-F1] nn.Embedding aggiunto alla wrap policy.
            Senza wrapping esplicito l'embedding finisce nel FSDP root,
            viene flattenato in 1D, e torch.embedding crasha con:
            RuntimeError: 'weight' must be 2-D

Avvio:
  torchrun --nproc_per_node=8 main.py --mode pretrain --use_fsdp

Interfaccia identica a DDPContext per compatibilità con setup.py:
  from utils.fsdp import FSDPContext, is_fsdp_available, wrap_model_fsdp

Uso in setup.py:
  if use_fsdp:
      from utils.fsdp import FSDPContext
      fsdp_ctx = FSDPContext().setup()
  else:
      from utils.ddp import DDPContext
      ddp_ctx = DDPContext().setup()
"""

import os
from typing import Optional

import torch
import torch.nn as nn

from utils.ddp import setup, cleanup, is_ddp, is_main


def is_fsdp_available() -> bool:
    """True se PyTorch ha FSDP disponibile (>= 1.12)."""
    return hasattr(torch.distributed, "fsdp")


def _get_Harold_wrap_policy():
    """
    Ritorna una wrap policy per Harold che wrappa:
    - JambaBlock: modulo principale, uno per layer (40 total)
    - nn.Embedding: token_emb — DEVE essere wrappato separatamente
      perché torch.embedding richiede weight 2D. Se finisce nel
      FSDP root viene flattenato in 1D e crasha.

    Mamba3Block NON viene wrappato separatamente perché i kernel CUDA
    Triton/TileLang richiedono che tutti i parametri SSM siano co-locati
    sulla stessa GPU durante il forward pass.
    """
    from torch.distributed.fsdp.wrap import ModuleWrapPolicy
    try:
        from core.model.blocks import JambaBlock
        return ModuleWrapPolicy({JambaBlock, nn.Embedding})
    except ImportError:
        # Fallback: size-based policy se JambaBlock non importabile
        from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
        import functools
        return functools.partial(size_based_auto_wrap_policy, min_num_params=1_000_000)


def wrap_model_fsdp(
    model:           nn.Module,
    device_id:       int,
    mixed_precision: bool = True,
    cpu_offload:     bool = False,
) -> nn.Module:
    r"""wrap_model_fsdp(model, device_id, ...) -> nn.Module

    Wrappa Harold con FSDP per training multi-GPU shardato.

    Sharda parametri, gradienti e stato ottimizzatore tra tutte le GPU.
    Con 8x H200, ogni GPU mantiene ~1/8 dei parametri in VRAM.

    Args:
        model (nn.Module): istanza di :class:`~core.model.Harold` non wrappata
        device_id (int): indice GPU locale (``local_rank``)
        mixed_precision (bool, optional): usa bf16 per parametri e gradienti.
            Default: ``True``
        cpu_offload (bool, optional): offload dei parametri non attivi su CPU.
            Utile per GPU con VRAM limitata, ma molto più lento. Default: ``False``

    Returns:
        nn.Module: modello wrappato con FSDP, pronto per il training

    .. note::
        Il modello deve essere su CPU prima di chiamare questa funzione.
        FSDP gestisce il movimento su GPU internamente.

    .. warning::
        ``cpu_offload=True`` è incompatibile con ``torch.compile``.
        Disabilita ``use_compile`` in ``TrainConfig`` se lo usi.
    """
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        ShardingStrategy,
        CPUOffload,
        MixedPrecision,
        StateDictType,
    )
    from torch.distributed.fsdp.wrap import ModuleWrapPolicy

    mp_policy = None
    if mixed_precision:
        mp_policy = MixedPrecision(
            param_dtype  = torch.bfloat16,
            reduce_dtype = torch.bfloat16,
            buffer_dtype = torch.bfloat16,
        )

    cpu_offload_policy = CPUOffload(offload_params=cpu_offload)
    wrap_policy        = _get_Harold_wrap_policy()
    model = FSDP(
        model,
        auto_wrap_policy     = wrap_policy,
        mixed_precision      = mp_policy,
        sharding_strategy    = ShardingStrategy.FULL_SHARD,
        cpu_offload          = cpu_offload_policy,
        device_id            = torch.device(f"cuda:{device_id}"),
        sync_module_states   = True,   # broadcast pesi rank0 → tutti al wrap
        use_orig_params      = True,   # necessario per torch.compile + FSDP
    )

    return model


def save_fsdp_checkpoint(
    path:      str,
    model:     nn.Module,
    optimizer: torch.optim.Optimizer,
    iter_num:  int,
    val_loss:  float,
    model_cfg,
    train_cfg,
    train_losses,
    val_losses,
) -> None:
    r"""save_fsdp_checkpoint(path, model, optimizer, ...) -> None

    Salva un checkpoint FSDP consolidato (solo rank 0).

    Usa ``FULL_STATE_DICT`` per raccogliere i pesi shardati da tutte le GPU
    e salvarli in un unico file .pt leggibile anche senza FSDP.

    Args:
        path (str): path di destinazione del checkpoint
        model (nn.Module): modello FSDP wrappato
        optimizer: optimizer (può essere wrappato da FSDP)
        iter_num (int): iterazione corrente
        val_loss (float): miglior val loss
        model_cfg: istanza di :class:`~core.config.ModelConfig`
        train_cfg: istanza di :class:`~core.config.TrainConfig`
        train_losses: storico loss training
        val_losses: storico loss validation

    .. note::
        Deve essere chiamato da tutti i rank — la sincronizzazione avviene
        internamente. Solo rank 0 scrive su disco.
    """
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import StateDictType, FullStateDictConfig

    cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
        model_state = model.state_dict()
        opt_state   = FSDP.optim_state_dict(model, optimizer)

    if is_main():
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        torch.save({
            "iter_num":       iter_num,
            "model_state":    model_state,
            "optimizer_state": opt_state,
            "val_loss":       val_loss,
            "model_cfg":      model_cfg,
            "train_cfg":      train_cfg,
            "train_losses":   list(train_losses),
            "val_losses":     val_losses,
            "fsdp":           True,
        }, tmp)
        os.replace(tmp, path)
        print(f"  FSDP checkpoint -> {path}")


def load_fsdp_checkpoint(
    path:      str,
    model:     nn.Module,
    optimizer: torch.optim.Optimizer,
    device:    str,
) -> tuple:
    r"""load_fsdp_checkpoint(path, model, optimizer, device) -> tuple

    Carica un checkpoint FSDP su tutti i rank.

    Supporta sia checkpoint FSDP (``fsdp=True``) che checkpoint DDP standard
    — in quest'ultimo caso carica i pesi direttamente senza FSDP state dict.

    Args:
        path (str): path del checkpoint
        model (nn.Module): modello FSDP wrappato
        optimizer: optimizer
        device (str): device string

    Returns:
        tuple: ``(iter_num, best_val_loss, train_losses, val_losses)``
    """
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import StateDictType, FullStateDictConfig

    if not os.path.isfile(path):
        if is_main():
            print("  Nessun checkpoint trovato — parto da zero.")
        return 0, float("inf"), [], []

    print(f"  Carico checkpoint FSDP: {path}")
    state = torch.load(path, map_location="cpu", weights_only=False)

    if state.get("fsdp", False):
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
            model.load_state_dict(state["model_state"])
        if "optimizer_state" in state:
            opt_state = FSDP.optim_state_dict_to_load(
                model, optimizer, state["optimizer_state"]
            )
            optimizer.load_state_dict(opt_state)
    else:
        # Checkpoint DDP standard — carica direttamente
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
            model.load_state_dict(state["model_state"], strict=False)
        if "optimizer_state" in state:
            optimizer.load_state_dict(state["optimizer_state"])

    return (
        state.get("iter_num",     0),
        state.get("val_loss",     float("inf")),
        state.get("train_losses", []),
        state.get("val_losses",   []),
    )


class FSDPContext:
    r"""Context manager per setup/teardown FSDP multi-GPU.

    Interfaccia identica a :class:`~utils.ddp.DDPContext` per
    compatibilità drop-in con ``setup.py``.

    Uso::

        ctx = FSDPContext().setup()
        model = wrap_model_fsdp(model, ctx.local_rank)
        ...
        ctx.teardown()

    Attributi pubblici (identici a DDPContext):
        rank (int): rank globale del processo
        local_rank (int): rank locale (indice GPU)
        world_size (int): numero totale di processi
        main (bool): True se rank 0
        active (bool): True se FSDP attivo
        device (str): ``"cuda:N"``
    """

    def __init__(self, default_device: str = "cuda") -> None:
        self.rank        = 0
        self.local_rank  = 0
        self.world_size  = 1
        self.main        = True
        self.active      = False
        self.device      = default_device

    def setup(self) -> "FSDPContext":
        r"""Inizializza il process group NCCL per FSDP.

        Richiede che il processo sia stato lanciato con ``torchrun``.
        No-op se ``WORLD_SIZE=1``.

        Returns:
            :class:`FSDPContext`: self, per chaining
        """
        if is_ddp():
            self.rank, self.local_rank, self.world_size = setup()
            self.main   = self.rank == 0
            self.active = True
            self.device = f"cuda:{self.local_rank}"
        return self

    def teardown(self) -> None:
        """Distrugge il process group."""
        if self.active:
            cleanup()

    def __enter__(self) -> "FSDPContext":
        return self.setup()

    def __exit__(self, *_) -> None:
        self.teardown()