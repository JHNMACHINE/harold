"""
Harold v0.6 — lr_schedule.py
==============================
Cosine decay con linear warmup.
"""

import math
from core.config import TrainConfig


def get_lr(it: int, cfg: TrainConfig) -> float:
    """
    Learning rate schedule: linear warmup + cosine decay.

    - ``[0, warmup_iters)``:   warmup lineare da 0 a ``cfg.lr``
    - ``[warmup_iters, max_iters]``: cosine decay da ``cfg.lr`` a ``cfg.min_lr``
    - ``>= max_iters``:        ``cfg.min_lr`` fisso

    Args:
        it:  iterazione corrente (0-indexed)
        cfg: TrainConfig

    Returns:
        learning rate per questa iterazione
    """
    if it >= cfg.max_iters:
        return cfg.min_lr
    if it < cfg.warmup_iters:
        return cfg.lr * max(it, 1) / cfg.warmup_iters
    ratio = (it - cfg.warmup_iters) / max(cfg.max_iters - cfg.warmup_iters, 1)
    return cfg.min_lr + 0.5 * (1.0 + math.cos(math.pi * ratio)) * (cfg.lr - cfg.min_lr)