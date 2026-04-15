"""
Harold v0.7 — trainer.py
=========================
DiffusionTrainer: wrappa Harold per il training step.
run_grad_accum: esegue gradient accumulation su N mini-batch.

Cambiamenti rispetto a v0.6:
  [v0.7-T4] DiffusionTrainer.train_step salva l'ultimo loss_dict in
             self._last_metrics — accessibile da train.py per il logging
             delle metriche x0_norm senza forward aggiuntivi.
  [v0.7-T5] run_grad_accum ritorna anche x0_norm_mean e x0_var_tokens
             accumulati — train.py li logga ogni 50 step senza dipendere
             da attributi interni del trainer.
"""

from typing import Any, Dict, Optional, Tuple, Protocol, runtime_checkable

import torch
from torch.utils.data import DataLoader

from core.config import MAX_SKIP_RATIO, ModelConfig, TrainConfig


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
        self.model         = model
        self.config        = config
        self.train_cfg     = train_cfg
        self.pad_token_id  = pad_token_id
        # [v0.7-T4] Ultimo loss_dict — accessibile da train.py per logging metriche
        self._last_metrics: Dict[str, float] = {}

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
        # [v0.7-T4] Salva metriche per accesso esterno da train.py
        self._last_metrics = loss_dict
        return loss, loss_dict


def run_grad_accum(
    trainer:      DiffusionTrainer,
    train_iter:   Any,
    train_loader: DataLoader,
    train_cfg:    TrainConfig,
    scaler:       torch.GradScaler,
    device:       str,
) -> Tuple[float, float, float, int, Any, Dict[str, float]]:
    r"""run_grad_accum(...) -> (loss_sum, score_sum, ce_sum, valid_count, train_iter, extra_metrics)

    Esegue gradient accumulation per ``train_cfg.grad_accum`` mini-batch validi.

    Gestisce automaticamente StopIteration ricreando l'iteratore, e skippa
    batch vuoti fino a ``MAX_SKIP_RATIO * grad_accum`` tentativi.

    .. rubric:: [v0.7-T5] Extra metrics

    Ritorna un dict con metriche aggiuntive accumulate durante il grad accum:

    - ``x0_norm_mean``: norma media di ``x0_pred`` sui token validi
    - ``x0_norm_std``: std della norma — rileva collapse verso zero
    - ``x0_var_tokens``: varianza media tra token — rileva mode collapse

    Args:
        trainer:      istanza di :class:`DiffusionTrainer`
        train_iter:   iteratore corrente sul DataLoader
        train_loader: DataLoader per ricreare l'iteratore a StopIteration
        train_cfg:    configurazione training
        scaler:       GradScaler per mixed precision
        device:       device string (es. ``"cuda"``)

    Returns:
        tuple:
            - **loss_sum** (*float*) — somma loss sui grad_accum step
            - **score_sum** (*float*) — somma score loss
            - **ce_sum** (*float*) — somma CE loss
            - **valid_count** (*int*) — numero di batch validi processati
            - **train_iter** — iteratore aggiornato
            - **extra_metrics** (*dict*) — metriche x0_norm accumulate
    """
    step_loss_sum = step_score_sum = step_ce_sum = 0.0
    valid_count = mb_idx = 0
    max_skip = train_cfg.grad_accum * MAX_SKIP_RATIO

    # [v0.7-T5] Accumulatori per metriche x0_norm
    x0_norm_mean_sum = 0.0
    x0_norm_std_sum  = 0.0
    x0_var_sum       = 0.0
    x0_metric_count  = 0

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

        # [v0.7-T5] Accumula metriche x0_norm se disponibili
        if "x0_norm_mean" in loss_dict:
            x0_norm_mean_sum += loss_dict["x0_norm_mean"]
            x0_norm_std_sum  += loss_dict.get("x0_norm_std",  0.0)
            x0_var_sum       += loss_dict.get("x0_var_tokens", 0.0)
            x0_metric_count  += 1

        valid_count += 1
        mb_idx      += 1

    # Calcola medie delle metriche extra
    extra_metrics: Dict[str, float] = {}
    if x0_metric_count > 0:
        extra_metrics["x0_norm_mean"]   = round(x0_norm_mean_sum / x0_metric_count, 4)
        extra_metrics["x0_norm_std"]    = round(x0_norm_std_sum  / x0_metric_count, 4)
        extra_metrics["x0_var_tokens"]  = round(x0_var_sum       / x0_metric_count, 4)

    return step_loss_sum, step_score_sum, step_ce_sum, valid_count, train_iter, extra_metrics