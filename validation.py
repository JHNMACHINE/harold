"""
Harold v0.6 — validation.py
============================
ValidationScheduler:  frequenza adattiva basata su stabilità della training loss.
estimate_loss:        validation completa su t = [0.3, 0.5, 0.7].
estimate_loss_single_t: validation rapida a singolo timestep.
run_validation_step:  orchestrazione completa di un singolo step di validation.
"""

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import time
import torch
from torch.utils.data import DataLoader

from config import ModelConfig, TrainConfig
from model import Harold
from logger import AsyncLogger
from checkpoint import save_checkpoint
from ddp import all_reduce_mean

if TYPE_CHECKING:
    from context import TrainingContext


class ValidationScheduler:
    """
    Scheduler adattivo per la validation.

    - Loss stabile (bassa varianza, pendenza piatta) → aumenta intervallo
      fino a ``max_interval``.
    - Loss instabile o in risalita → riduce intervallo fino a ``min_interval``,
      forza validation immediata se molto instabile (cv > 0.2).
    """

    def __init__(
        self,
        base_interval:       int   = 500,
        min_interval:        int   = 100,
        max_interval:        int   = 2000,
        stability_threshold: float = 0.03,
        patience:            int   = 3,
    ):
        self.base_interval       = base_interval
        self.min_interval        = min_interval
        self.max_interval        = max_interval
        self.current_interval    = base_interval
        self.stability_threshold = stability_threshold
        self.patience            = patience

        self._train_losses:  deque = deque(maxlen=100)
        self._val_losses:    deque = deque(maxlen=20)
        self._stable_count   = 0
        self._unstable_count = 0
        self._last_val_iter  = 0
        self._skip_count     = 0
        self._total_val      = 0

    def should_validate(
        self,
        iter_num:   int,
        train_loss: float,
        force:      bool = False,
    ) -> Tuple[bool, str]:
        """
        Decide se eseguire validation a questo step.

        Returns:
            (should_validate, reason)
        """
        if force:
            return True, "forced"

        if self._total_val == 0:
            if iter_num >= self.base_interval:
                return True, "first_validation"
            return False, "waiting_warmup"

        self._train_losses.append(train_loss)

        if len(self._train_losses) >= 10:
            recent = list(self._train_losses)[-10:]
            mean   = sum(recent) / len(recent)
            std    = float(torch.tensor(recent).std())
            cv     = std / mean if mean > 0 else 0.0

            n    = len(recent)
            sx   = n * (n - 1) / 2
            sxx  = sum(i * i for i in range(n))
            sxy  = sum(i * v for i, v in enumerate(recent))
            sy   = sum(recent)
            den  = n * sxx - sx * sx
            slope = (n * sxy - sx * sy) / den if den != 0 else 0.0

            time_since = iter_num - self._last_val_iter

            if cv < self.stability_threshold and abs(slope) < 0.001:
                self._stable_count   += 1
                self._unstable_count  = 0
                if self._stable_count >= self.patience:
                    self.current_interval = min(
                        self.max_interval, int(self.current_interval * 1.5))
                    self._stable_count = 0
                if time_since < self.current_interval:
                    self._skip_count += 1
                    return False, f"stable(cv={cv:.4f})"

            elif cv > self.stability_threshold * 2 or slope > 0.01:
                self._unstable_count += 1
                self._stable_count    = 0
                if self._unstable_count >= 2:
                    self.current_interval = max(
                        self.min_interval, int(self.current_interval * 0.7))
                    self._unstable_count = 0
                if cv > 0.2 and time_since > self.min_interval:
                    return True, f"unstable(cv={cv:.4f})"
            else:
                self._stable_count   = 0
                self._unstable_count = 0

        if iter_num - self._last_val_iter >= self.current_interval:
            return True, f"interval({self.current_interval})"

        self._skip_count += 1
        return False, f"skip({self.current_interval - (iter_num - self._last_val_iter)} left)"

    def record(self, iter_num: int, val_loss: float) -> None:
        """Registra una validation completata."""
        self._last_val_iter = iter_num
        self._total_val    += 1
        self._val_losses.append(val_loss)

    @property
    def total_val_calls(self) -> int:
        return self._total_val

    @property
    def last_val_iter(self) -> int:
        return self._last_val_iter


@torch.no_grad()
def estimate_loss_single_t(
    model:        Harold,
    train_cfg:    TrainConfig,
    val_loader:   DataLoader,
    pad_token_id: int,
    t:            float = 0.5,
) -> float:
    """
    Validation rapida a singolo timestep — usa metà dei batch normali.

    Args:
        t: timestep fisso per la validation (default: 0.5, punto medio della traiettoria)
    """
    device  = next(model.parameters()).device
    model.eval()
    losses: list = []
    t_cache: Dict[int, torch.Tensor] = {}

    for i, batch in enumerate(val_loader):
        if i >= train_cfg.eval_iters // 2:
            break
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        mask      = (input_ids != pad_token_id)
        if mask.sum() < 10:
            continue
        B = input_ids.shape[0]
        if B not in t_cache:
            t_cache[B] = torch.full((B,), t, dtype=torch.float32, device=device)
        with train_cfg.ctx:
            _, loss_dict = model.compute_loss(
                x0=input_ids, mask=mask,
                ce_weight=train_cfg.ce_loss_weight,
                fixed_t=t_cache[B], self_cond_prob=0.0,
            )
        losses.append(loss_dict["total"])

    model.train()
    return sum(losses) / len(losses) if losses else float("inf")


@torch.no_grad()
def estimate_loss(
    model:        Harold,
    train_cfg:    TrainConfig,
    val_loader:   DataLoader,
    pad_token_id: int = 50256,
    iter_num:     int = 0,
    logger:       Optional[AsyncLogger] = None,
) -> float:
    """
    Validation completa su t = [0.3, 0.5, 0.7].

    Rispetto alla versione originale:
    - t_values ridotti da 5 a 3 (0.1 e 0.9 poco informativi per Flow Matching)
    - Cache ``fixed_t`` per evitare allocazioni GPU ripetute
    - Accumulatori su GPU invece di liste CPU
    """
    device  = next(model.parameters()).device
    model.eval()

    t_values = [0.3, 0.5, 0.7]
    n_t      = len(t_values)

    sum_total = torch.zeros(n_t, device=device)
    sum_score = torch.zeros(n_t, device=device)
    sum_ce    = torch.zeros(n_t, device=device)
    count     = torch.zeros(n_t, device=device)

    t_cache: Dict[Tuple[float, int], torch.Tensor] = {}

    iterator = iter(val_loader)
    for _ in range(train_cfg.eval_iters):
        try:
            batch = next(iterator)
        except StopIteration:
            break
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        mask      = (input_ids != pad_token_id)
        if mask.sum() < 10:
            continue
        B = input_ids.shape[0]
        for idx, t_val in enumerate(t_values):
            key = (t_val, B)
            if key not in t_cache:
                t_cache[key] = torch.full((B,), t_val, dtype=torch.float32, device=device)
            with train_cfg.ctx:
                _, loss_dict = model.compute_loss(
                    x0=input_ids, mask=mask,
                    ce_weight=train_cfg.ce_loss_weight,
                    fixed_t=t_cache[key], self_cond_prob=0.0,
                )
            sum_total[idx] += loss_dict["total"]
            sum_score[idx] += loss_dict["score"]
            sum_ce[idx]    += loss_dict["ce"]
            count[idx]     += 1

    model.train()

    c         = count.clamp(min=1)
    avg_total = (sum_total / c).cpu().tolist()
    avg_score = (sum_score / c).cpu().tolist()
    avg_ce    = (sum_ce    / c).cpu().tolist()

    per_t_total = {t: avg_total[i] for i, t in enumerate(t_values)}
    per_t_score = {t: avg_score[i] for i, t in enumerate(t_values)}
    per_t_ce    = {t: avg_ce[i]    for i, t in enumerate(t_values)}

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


# ---------------------------------------------------------------------------
# run_validation_step — orchestrazione completa
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    val_loss:     float
    val_type:     str       # "full" | "quick"
    reason:       str
    is_best:      bool
    accum_loss:   float     # accum_loss aggiornato (azzerato se full)


def run_validation_step(
    ctx:           "TrainingContext",
    model_cfg:     ModelConfig,
    train_cfg:     TrainConfig,
    val_scheduler: "ValidationScheduler",
    iter_num:      int,
    accum_loss:    float,
    start_time:    float,
    lr:            float,
    force_val:     bool,
) -> Optional[ValidationResult]:
    """
    Orchestrazione completa di un singolo step di validation.

    Decide se validare, esegue la validation (full o quick),
    sincronizza in DDP, logga, salva il checkpoint best se migliorato.

    Args:
        ctx:          TrainingContext con model, loaders, optimizer, ecc.
        model_cfg:    configurazione modello (per save_checkpoint)
        train_cfg:    configurazione training
        val_scheduler: scheduler adattivo
        iter_num:     iterazione corrente
        accum_loss:   loss accumulata dall'ultimo full val (per avg_train)
        start_time:   timestamp avvio training (per elapsed)
        lr:           learning rate corrente (per logging)
        force_val:    forza validation indipendentemente dallo scheduler

    Returns:
        ``ValidationResult`` se la validation è stata eseguita, ``None`` altrimenti.
    """
    current_train_loss = (
        sum(ctx.train_losses) / len(ctx.train_losses)
        if ctx.train_losses else accum_loss
    )
    should_val, reason = val_scheduler.should_validate(
        iter_num, current_train_loss, force=force_val,
    )

    if not should_val or iter_num == 0:
        return None

    # Scegli tipo di validation
    if val_scheduler.total_val_calls % 3 == 0 or force_val or "unstable" in reason:
        local_val = estimate_loss(
            ctx.model, train_cfg, ctx.val_loader, ctx.pad_token_id,
            iter_num=iter_num, logger=ctx.logger if ctx.main else None,
        )
        val_type = "full"
    else:
        local_val = estimate_loss_single_t(
            ctx.model, train_cfg, ctx.val_loader, ctx.pad_token_id, t=0.5,
        )
        val_type = "quick"

    # Sincronizza DDP
    if ctx.use_ddp:
        val_tensor = torch.tensor(local_val, device=ctx.device)
        val_loss   = float(all_reduce_mean(val_tensor, ctx.world_size).item())
    else:
        val_loss = local_val

    val_scheduler.record(iter_num, val_loss)
    is_best   = False
    new_accum = accum_loss

    if ctx.main:
        elapsed   = (time.time() - start_time) / 60
        avg_train = accum_loss / max(iter_num - val_scheduler.last_val_iter, 1)

        if val_type == "full":
            ctx.val_losses.append(val_loss)
            new_accum = 0.0
            print(f"\n[iter {iter_num:7d}] train={avg_train:.4f}  val={val_loss:.4f}  "
                  f"lr={lr:.2e}  elapsed={elapsed:.1f}min  [{val_type}|{reason}]")
            if ctx.logger:
                ctx.logger.log({"type": "val", "iter": iter_num,
                                "train_loss": round(avg_train, 6),
                                "val_loss": round(val_loss, 6),
                                "val_type": val_type, "reason": reason,
                                "lr": lr, "elapsed_min": round(elapsed, 2)})

            if val_loss < ctx.best_val_loss:
                is_best   = True
                best_path = f"{train_cfg.checkpoint_dir}/{train_cfg.checkpoint_prefix}_best.pt"
                save_checkpoint(best_path, ctx.model, ctx.optimizer, ctx.scaler,
                                iter_num, val_loss, model_cfg, train_cfg,
                                ctx.train_losses, ctx.val_losses, push_hf=True)
                print(f"  ★ Best val loss: {val_loss:.4f} → {best_path}")
                train_cfg.write_latest(iter_num, best_path)
                if ctx.logger:
                    ctx.logger.log({"type": "best_checkpoint", "iter": iter_num,
                                    "val_loss": round(val_loss, 6), "path": best_path})
        else:
            print(f"  [quick val] iter={iter_num} loss={val_loss:.4f} ({reason})")
            if ctx.logger:
                ctx.logger.log({"type": "quick_val", "iter": iter_num,
                                "val_loss": round(val_loss, 6), "reason": reason})

    ctx.active_model.train()

    return ValidationResult(
        val_loss   = val_loss,
        val_type   = val_type,
        reason     = reason,
        is_best    = is_best,
        accum_loss = new_accum,
    )