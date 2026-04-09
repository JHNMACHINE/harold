"""
Harold v0.6 — trainer.py
=========================
DiffusionTrainer: wrappa Harold per il training step.
_run_grad_accum: esegue gradient accumulation su N mini-batch.
"""

from typing import Any, Dict, Optional, Tuple, Protocol, runtime_checkable

import torch
from torch.utils.data import DataLoader

from config import MAX_SKIP_RATIO, ModelConfig, TrainConfig


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
    def __init__(
        self,
        model:        TrainableModel,
        config:       ModelConfig,
        train_cfg:    TrainConfig,
        pad_token_id: int = 0,
    ):
        self.model        = model
        self.config       = config
        self.train_cfg    = train_cfg
        self.pad_token_id = pad_token_id

    def train_step(
        self,
        batch: torch.Tensor,
    ) -> Optional[Tuple[torch.Tensor, Dict[str, float]]]:
        mask = (batch != self.pad_token_id)
        if mask.sum() == 0:
            return None
        loss, loss_dict = self.model.compute_loss(
            x0=batch, mask=mask,
            ce_weight=self.train_cfg.ce_loss_weight,
            self_cond_prob=self.train_cfg.self_cond_prob,
        )
        return loss, loss_dict


def run_grad_accum(
    trainer:      DiffusionTrainer,
    train_iter:   Any,
    train_loader: DataLoader,
    train_cfg:    TrainConfig,
    scaler:       torch.GradScaler,
    device:       str,
) -> Tuple[float, float, float, int, Any]:
    """
    Esegue gradient accumulation per ``train_cfg.grad_accum`` mini-batch validi.

    Gestisce automaticamente StopIteration ricreando l'iteratore, e skippa
    batch vuoti fino a ``MAX_SKIP_RATIO * grad_accum`` tentativi.

    Returns:
        (loss_sum, score_sum, ce_sum, valid_count, train_iter)
    """
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