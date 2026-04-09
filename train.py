"""
Harold v0.6 — train.py  (unified single-GPU + DDP)
====================================================
Entry point unico per training single-GPU e multi-GPU DDP.

Avvio single-GPU:
    torchrun --nproc_per_node=1 train.py

Avvio multi-GPU (es. 4 GPU):
    torchrun --nproc_per_node=4 train.py

Comportamento automatico:
  - world_size=1  → single-GPU, torch.compile abilitato, nessun overhead DDP
  - world_size>1  → DDP attivo, torch.compile disabilitato (instabile con DDP),
                    dataset partizionato per rank tramite seed offset,
                    checkpoint/logging/HF push solo su rank 0,
                    val loss sincronizzata via all_reduce

Compatibilità checkpoint:
  - I checkpoint sono sempre salvati unwrapped (model.module se DDP)
  - Intercambiabili tra run single-GPU e multi-GPU

Changelog v0.6:
  - Aggiornato per Harold v0.6 (architettura Jamba: Mamba2 + Attention + MoE)
  - Rimosso gradient checkpointing (non utilizzato)
"""

import math
import os
import time
import warnings
from collections import deque
from typing import Any, Dict, Optional, Tuple, Union, cast, Protocol, runtime_checkable

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from config import MAX_SKIP_RATIO, ModelConfig, TrainConfig, get_model_config, get_train_config
from optimizer import build_optimizer
from model import Harold, build_model
from dataset import build_loaders, build_loaders_ddp
from logger import AsyncLogger
from checkpoint import save_checkpoint, load_checkpoint
from ddp import DDPContext, is_ddp, is_main, all_reduce_mean, broadcast_model
from itertools import cycle

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
    def __init__(self, model: TrainableModel, config: ModelConfig, train_cfg: TrainConfig,
                 pad_token_id: int = 0):
        self.model        = model
        self.config       = config
        self.train_cfg    = train_cfg
        self.pad_token_id = pad_token_id

    def train_step(self, batch: torch.Tensor) -> Optional[Tuple[torch.Tensor, Dict[str, float]]]:
        mask = (batch != self.pad_token_id)
        if mask.sum() == 0:
            return None
        loss, loss_dict = self.model.compute_loss(
            x0=batch, mask=mask,
            ce_weight=self.train_cfg.ce_loss_weight,
            self_cond_prob=self.train_cfg.self_cond_prob,
        )
        return loss, loss_dict

class ValidationScheduler:
    """
    Scheduler intelligente per validation:
    - Riduce frequenza quando training è stabile
    - Aumenta frequenza quando loss peggiora
    - Skip adattivo basato su trend e varianza
    """
    
    def __init__(
        self,
        base_interval: int = 500,
        min_interval: int = 100,
        max_interval: int = 2000,
        stability_threshold: float = 0.05,  # 5% variazione
        patience: int = 3,  # Epoche di stabilità prima di aumentare intervallo
    ):
        self.base_interval = base_interval
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.current_interval = base_interval
        self.stability_threshold = stability_threshold
        self.patience = patience
        
        # Tracking metriche
        self.train_losses = deque(maxlen=100)
        self.val_losses = deque(maxlen=20)
        self.stable_counter = 0
        self.unstable_counter = 0
        self.last_val_iter = 0
        self.skip_count = 0
        self.total_val_calls = 0
        
    def should_validate(
        self, 
        iter_num: int, 
        train_loss: float,
        force: bool = False
    ) -> Tuple[bool, str]:
        """
        Decide se eseguire validation.
        
        Returns:
            (should_validate, reason)
        """
        if force:
            return True, "forced"
        
        # Prima validation sempre dopo warmup
        if self.total_val_calls == 0:
            if iter_num >= self.base_interval:
                return True, "first_validation"
            return False, "waiting_warmup"
        
        # Aggiorna statistiche
        self.train_losses.append(train_loss)
        
        # Calcola metriche di stabilità
        if len(self.train_losses) >= 10:
            recent_losses = list(self.train_losses)[-10:]
            mean_loss = sum(recent_losses) / len(recent_losses)
            std_loss = torch.tensor(recent_losses).std().item()
            cv = std_loss / mean_loss if mean_loss > 0 else 0  # Coefficiente di variazione
            
            # Trend recente (pendenza)
            if len(recent_losses) >= 5:
                x = list(range(len(recent_losses)))
                y = recent_losses
                n = len(x)
                slope = (n * sum(xi*yi for xi, yi in zip(x, y)) - sum(x) * sum(y)) / \
                        (n * sum(xi*xi for xi in x) - sum(x)**2)
            else:
                slope = 0
            
            # Decisione adattiva
            time_since_last = iter_num - self.last_val_iter
            
            # Caso 1: Loss molto stabile → aumenta intervallo
            if cv < self.stability_threshold and abs(slope) < 0.001:
                self.stable_counter += 1
                self.unstable_counter = 0
                
                if self.stable_counter >= self.patience:
                    self.current_interval = min(
                        self.max_interval,
                        int(self.current_interval * 1.5)
                    )
                    self.stable_counter = 0
                    
                if time_since_last < self.current_interval:
                    self.skip_count += 1
                    return False, f"stable_loss(cv={cv:.4f})"
            
            # Caso 2: Loss instabile o in aumento → riduci intervallo
            elif cv > self.stability_threshold * 2 or slope > 0.01:
                self.unstable_counter += 1
                self.stable_counter = 0
                
                if self.unstable_counter >= 2:
                    self.current_interval = max(
                        self.min_interval,
                        int(self.current_interval * 0.7)
                    )
                    self.unstable_counter = 0
                    
                # Forza validation se molto instabile
                if cv > 0.2 and time_since_last > self.min_interval:
                    return True, f"unstable_loss(cv={cv:.4f})"
            
            # Caso 3: Normale - segui intervallo corrente
            else:
                self.stable_counter = 0
                self.unstable_counter = 0
        
        # Decisione finale basata su intervallo
        if iter_num - self.last_val_iter >= self.current_interval:
            return True, f"interval_reached({self.current_interval})"
        
        self.skip_count += 1
        return False, f"skip({self.current_interval - (iter_num - self.last_val_iter)} left)"
    
    def record_validation(self, iter_num: int, val_loss: float):
        """Registra validation eseguita."""
        self.last_val_iter = iter_num
        self.total_val_calls += 1
        self.val_losses.append(val_loss)
    
    def get_stats(self) -> dict:
        """Statistiche per logging."""
        return {
            "current_interval": self.current_interval,
            "total_val_calls": self.total_val_calls,
            "skip_count": self.skip_count,
            "skip_rate": self.skip_count / max(1, self.total_val_calls + self.skip_count),
            "stable_counter": self.stable_counter,
            "recent_val_losses": list(self.val_losses)[-5:] if self.val_losses else [],
        }



def get_lr(it: int, cfg: TrainConfig) -> float:
    # 1. Gestione limite finale
    if it >= cfg.max_iters:
        return cfg.min_lr
        
    if it < cfg.warmup_iters:
        if cfg.warmup_iters <= 1:
            return cfg.lr
        warmup_ratio = it / (cfg.warmup_iters - 1)
        return cfg.min_lr + warmup_ratio * (cfg.lr - cfg.min_lr)
        
    # 3. Cosine Decay
    # Assicuriamoci che il primo step di decay sia ESATTAMENTE cfg.lr
    decay_iters = cfg.max_iters - cfg.warmup_iters
    if decay_iters <= 0:
        return cfg.min_lr
        
    ratio = (it - cfg.warmup_iters) / decay_iters
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return cfg.min_lr + coeff * (cfg.lr - cfg.min_lr)


@torch.no_grad()
def estimate_loss_single_t(model, train_cfg, val_loader, pad_token_id, t=0.5):
    """Validation ultra-rapida per monitoring frequente."""
    device = next(model.parameters()).device
    model.eval()
    
    losses = []
    for i, batch in enumerate(val_loader):
        if i >= train_cfg.eval_iters // 2:  # Metà batch
            break
        
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        mask = (input_ids != pad_token_id)
        if mask.sum() < 10:
            continue
            
        fixed_t = torch.full((input_ids.shape[0],), t, device=device)
        with train_cfg.ctx:
            _, loss_dict = model.compute_loss(
                x0=input_ids, mask=mask,
                ce_weight=train_cfg.ce_loss_weight,
                fixed_t=fixed_t, self_cond_prob=0.0
            )
        losses.append(loss_dict["total"])
    
    model.train()
    return sum(losses) / len(losses) if losses else float("inf")

@torch.no_grad()
def estimate_loss(
    model: Harold,
    train_cfg: TrainConfig,
    val_loader: DataLoader,
    pad_token_id: int = 50256,
    iter_num: int = 0,
    logger: Optional["AsyncLogger"] = None,
) -> float:
    """
    Versione ottimizzata specifica per Harold:
    - Cache intelligente per fixed_t (evita allocazioni GPU ripetute)
    - Early stopping su batch vuoti
    - Riduzione t_values da 5 a 3 (0.3, 0.5, 0.7 sono i più informativi per diffusion)
    - Accumulo diretto su GPU invece di CPU
    """
    device = next(model.parameters()).device
    model.eval()
    
    # Per Harold, t=0.3, 0.5, 0.7 sono sufficienti (evita estremi poco informativi)
    t_values = [0.3, 0.5, 0.7]
    n_t = len(t_values)
    
    # Accumulatori su GPU (più veloce di CPU)
    sum_total = torch.zeros(n_t, device=device)
    sum_score = torch.zeros(n_t, device=device)
    sum_ce = torch.zeros(n_t, device=device)
    count = torch.zeros(n_t, device=device)
    
    # Cache per tensori fixed_t (evita allocazioni ripetute)
    t_cache = {}
    
    valid_batches = 0
    skipped_batches = 0
    batch_iterator = cycle(val_loader)
    
    while valid_batches < train_cfg.eval_iters:
        batch = next(batch_iterator)
        
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        mask = (input_ids != pad_token_id)
        
        # Early skip per batch problematici
        valid_tokens = mask.sum().item()
        if valid_tokens < 10:  # Troppo pochi token validi
            skipped_batches += 1
            if skipped_batches > 5:  # Massimo 5 skip consecutivi
                break
            continue
        
        B = input_ids.shape[0]
        valid_batches += 1
        skipped_batches = 0
        
        # Processa ogni t_value
        for idx, t_val in enumerate(t_values):
            # Cache lookup/creation
            cache_key = (t_val, B)
            if cache_key not in t_cache:
                t_cache[cache_key] = torch.full((B,), t_val, dtype=torch.float32, device=device)
            fixed_t = t_cache[cache_key]
            
            # Forward pass
            with train_cfg.ctx:
                _, loss_dict = model.compute_loss(
                    x0=input_ids, 
                    mask=mask,
                    ce_weight=train_cfg.ce_loss_weight,
                    fixed_t=fixed_t, 
                    self_cond_prob=0.0,  # Validation sempre senza self-conditioning
                )
            
            # Accumula su GPU
            sum_total[idx] += loss_dict["total"]
            sum_score[idx] += loss_dict["score"]
            sum_ce[idx] += loss_dict["ce"]
            count[idx] += 1
    
    model.train()
    
    # Calcola medie su GPU
    count_clamped = count.clamp(min=1)
    avg_total = (sum_total / count_clamped).cpu().tolist()
    avg_score = (sum_score / count_clamped).cpu().tolist()
    avg_ce = (sum_ce / count_clamped).cpu().tolist()
    
    per_t_total = {t: avg_total[i] for i, t in enumerate(t_values)}
    per_t_score = {t: avg_score[i] for i, t in enumerate(t_values)}
    per_t_ce = {t: avg_ce[i] for i, t in enumerate(t_values)}
    
    # Logging (solo main process)
    if not is_ddp() or dist.get_rank() == 0:
        print(f"  val (optimized, {valid_batches} batches):")
        print("    total: " + "  ".join(f"t={t}:{v:.4f}" for t, v in per_t_total.items()))
        print("    score: " + "  ".join(f"t={t}:{v:.4f}" for t, v in per_t_score.items()))
        print("    CE:    " + "  ".join(f"t={t}:{v:.4f}" for t, v in per_t_ce.items()))
    
    if logger is not None:
        logger.log({
            "type": "val_detail",
            "iter": iter_num,
            "batches_used": valid_batches,
            "total_per_t": {str(t): round(v, 6) for t, v in per_t_total.items()},
            "score_per_t": {str(t): round(v, 6) for t, v in per_t_score.items()},
            "ce_per_t": {str(t): round(v, 6) for t, v in per_t_ce.items()},
        })
    
    # Media pesata sui t_values (tutti stesso peso)
    valid = [v for v in per_t_total.values() if v != float("inf")]
    return sum(valid) / len(valid) if valid else float("inf")


def _run_grad_accum(
    trainer: DiffusionTrainer, 
    train_loader: DataLoader,
    train_cfg: TrainConfig, 
    scaler: torch.GradScaler, 
    device: str,
) -> Tuple[float, float, float, int]:
    """
    Versione ottimizzata di gradient accumulation:
    - Usa cycle iterator (no StopIteration overhead)
    - Gestione intelligente batch vuoti
    - Accumulo loss in fp32 per stabilità
    - Skip rate tracking con soglia adattiva
    """
    step_loss_sum = 0.0
    step_score_sum = 0.0
    step_ce_sum = 0.0
    valid_count = 0
    skipped_count = 0
    consecutive_skips = 0
    
    # Soglia adattiva per skip consecutivi
    max_consecutive_skips = max(10, train_cfg.grad_accum * MAX_SKIP_RATIO)
    
    # Usa cycle iterator per training (mai StopIteration)
    # NOTA: train_iter dovrebbe essere creato una volta all'inizio del training
    # e passato come ciclo infinito
    if not hasattr(_run_grad_accum, "train_cycle"):
        _run_grad_accum.train_cycle = cycle(train_loader)
    
    train_cycle = _run_grad_accum.train_cycle
    
    # Pre-alloca lista per loss scaling (evita divisioni ripetute)
    grad_accum_inv = 1.0 / train_cfg.grad_accum
    
    while valid_count < train_cfg.grad_accum:
        batch = next(train_cycle)
        
        # Verifica rapida se batch è valido (senza spostare su GPU)
        if "input_ids" not in batch:
            skipped_count += 1
            continue
            
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        
        with train_cfg.ctx:
            result = trainer.train_step(input_ids)
        
        if result is None:
            skipped_count += 1
            consecutive_skips += 1
            
            # Warning solo se troppi skip consecutivi (evita spam)
            if consecutive_skips == max_consecutive_skips:
                if dist.get_rank() == 0:
                    print(f"WARNING: {consecutive_skips} batch vuoti consecutivi. "
                          f"Verifica dataset o tokenizzazione.")
            # Se troppi skip, fai una pausa per evitare loop infinito
            elif consecutive_skips > max_consecutive_skips * 2:
                if dist.get_rank() == 0:
                    print(f"ERROR: Troppi batch vuoti ({consecutive_skips}). "
                          f"Interruzione gradient accumulation.")
                break
            continue
        
        # Reset skip counter su batch valido
        consecutive_skips = 0
        
        loss, loss_dict = result
        
        # Gradient accumulation con scaling
        scaled_loss = loss * grad_accum_inv
        scaler.scale(scaled_loss).backward()
        
        # Accumula in fp32 per precisione
        step_loss_sum += loss.item()
        step_score_sum += loss_dict.get("score", 0.0)
        step_ce_sum += loss_dict.get("ce", 0.0)
        valid_count += 1
    
    # Logging statistiche skip (solo main process)
    if skipped_count > 0 and dist.get_rank() == 0:
        total_batches = valid_count + skipped_count
        skip_rate = skipped_count / total_batches
        
        if skip_rate > 0.2:  # >20% batch saltati
            print(f"  WARNING: {skip_rate*100:.1f}% batch saltati "
                  f"({skipped_count}/{total_batches}) - Dataset potrebbe avere problemi")
        elif skip_rate > 0.05:  # 5-20%
            # Logga solo ogni tanto per non spamare
            if valid_count % (train_cfg.grad_accum * 10) == 0:
                print(f"  Info: {skip_rate*100:.1f}% batch con pochi token validi")
    
    # Se valid_count == 0, qualcosa è andato storto
    if valid_count == 0:
        raise RuntimeError(f"Gradient accumulation fallita: 0/{train_cfg.grad_accum} batch validi. "
                          f"Verifica dataset e tokenizzazione.")
    
    return step_loss_sum, step_score_sum, step_ce_sum, valid_count

def run_training(model_cfg: ModelConfig, train_cfg: TrainConfig) -> dict:

    # ── Setup ─────────────────────────────────────────────────────────────
    use_ddp    = is_ddp()
    rank       = 0
    local_rank = 0
    world_size = 1

    if use_ddp:
        ctx = DDPContext().setup()
        rank, local_rank, world_size = ctx.rank, ctx.local_rank, ctx.world_size
        device = f"cuda:{local_rank}"
    else:
        device = train_cfg.device

    main = is_main()

    if main:
        print("Harold")
        print(f"Modalità:       {'DDP (' + str(world_size) + ' GPU)' if use_ddp else 'Single-GPU'}")
        print(f"Device:         {device}")
        print(f"Dtype:          {train_cfg.dtype}  (scaler={'ON' if train_cfg.use_scaler else 'OFF'})")
        eff = train_cfg.batch_size * train_cfg.grad_accum * world_size
        print(f"Batch effettivo: {eff}  ({train_cfg.batch_size} × {train_cfg.grad_accum} × {world_size} GPU)")

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
    model_cfg.mask_token_id = int(getattr(tokenizer, "mask_token_id", None) or tokenizer.vocab_size)

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
        print(f"Harold — {n_params/1000:.2f}B parametri")

    active_model: Union[Harold, DDP] = (
        DDP(model, device_ids=[local_rank], output_device=local_rank)
        if use_ddp else model
    )
    raw_model: Harold = cast(Harold, active_model.module if isinstance(active_model, DDP) else active_model)

    # ── Optimizer ─────────────────────────────────────────────────────────
    optimizer = build_optimizer(active_model, train_cfg)
    scaler = torch.GradScaler("cuda", enabled=train_cfg.use_scaler)

    # ── Dataset ───────────────────────────────────────────────────────────
    if use_ddp:
        train_loader, val_loader = build_loaders_ddp(train_cfg, tokenizer, rank)
    else:
        train_loader, val_loader = build_loaders(train_cfg, tokenizer)

    # Trainer
    trainer = DiffusionTrainer(raw_model, model_cfg, train_cfg, pad_token_id=pad_token_id)

    val_scheduler = ValidationScheduler(
        base_interval=train_cfg.eval_interval,
        min_interval=max(100, train_cfg.eval_interval // 5),
        max_interval=train_cfg.eval_interval * 4,
        stability_threshold=0.03,  # 3% variazione
        patience=3
    )

    window_losses = deque(maxlen=train_cfg.eval_interval)

    # ── Checkpoint resume ─────────────────────────────────────────────────
    if main:
        os.makedirs(train_cfg.checkpoint_dir, exist_ok=True)

    initial_iter  = 0
    best_val_loss = float("inf")
    train_losses: deque = deque(maxlen=getattr(train_cfg, "loss_history_size", 100_000))
    val_losses:   list  = []

    if train_cfg.preload:
        ckpt_path = (
            (train_cfg.read_latest() or (None, None))[1]
            if train_cfg.preload == "latest"
            else train_cfg.preload
        )
        if ckpt_path:
            initial_iter, best_val_loss, _tl, val_losses = load_checkpoint(
                ckpt_path, raw_model, optimizer, scaler, device
            )
            train_losses.extend(_tl)

    if use_ddp:
        broadcast_model(raw_model)

    # ── Logger ────────────────────────────────────────────────────────────
    logger = None
    if main:
        log_path = os.path.join(train_cfg.checkpoint_dir, "training.log")
        logger   = AsyncLogger(log_path, flush_every=10)
        print(f"Log -> {log_path}\nAvvio training -> {train_cfg.max_iters} optimizer steps\n")

    # ── Loop ──────────────────────────────────────────────────────────────
    active_model.train()
    start_time = time.time()
    accum_loss = 0.0

    pbar = tqdm(
        range(initial_iter, train_cfg.max_iters),
        desc="Harold" + (" DDP" if use_ddp else ""),
        disable=not main,
    )

    for iter_num in pbar:
        # Memory monitoring (ogni 500 iter)
        if main and iter_num % 500 == 0:
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1e9
                reserved = torch.cuda.memory_reserved() / 1e9
                print(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

        lr = get_lr(iter_num, train_cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)

        step_loss_sum, step_score_sum, step_ce_sum, valid_count = (
            _run_grad_accum(trainer, train_loader, train_cfg, scaler, device)
        )
        if valid_count == 0:
            continue

        if train_cfg.use_scaler:
            scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(active_model.parameters(), train_cfg.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        
        if iter_num % 10 == 0:
            raw_model.update_router_biases()

        avg_loss  = step_loss_sum  / valid_count
        avg_score = step_score_sum / valid_count
        avg_ce    = step_ce_sum    / valid_count
        
        # Aggiorna finestra per media mobile
        window_losses.append(avg_loss)
        train_losses.append(avg_loss)

        if main:
            pbar.set_postfix({"loss": f"{avg_loss:.4f}", "score": f"{avg_score:.4f}",
                              "lr": f"{lr:.2e}", "grad": f"{grad_norm:.2f}"})
            if logger:
                logger.log({"type": "train", "iter": iter_num,
                            "loss": round(avg_loss, 6), "score": round(avg_score, 6),
                            "ce": round(avg_ce, 6), "lr": lr,
                            "grad_norm": round(float(grad_norm), 6),
                            "elapsed_min": round((time.time() - start_time) / 60, 2)})

        # Decisione validation
        current_train_loss = sum(window_losses) / len(window_losses)
        force_val = (iter_num == train_cfg.max_iters - 1)
        should_val, reason = val_scheduler.should_validate(
            iter_num, current_train_loss, force=force_val
        )
 
        # ── Val ───────────────────────────────────────────────────────────        
        if should_val:
            if iter_num == 0:
                continue
            
            # Memory cleanup
            if torch.cuda.is_available() and iter_num % 500 == 0:
                torch.cuda.empty_cache()
            
            # Scegli tipo di validation
            stats = val_scheduler.get_stats()
            
            # Alterna: 1 full validation ogni 3, altrimenti single_t
            if stats["total_val_calls"] % 3 == 0 or force_val or "unstable" in reason:
                # Validation completa con tutti i t_values
                local_val = estimate_loss(
                    raw_model, train_cfg, val_loader, pad_token_id,
                    iter_num=iter_num, logger=logger if main else None,
                )
                val_type = "full"
                
            else:
                # Validation rapida con singolo t_value
                local_val = estimate_loss_single_t(
                    raw_model, train_cfg, val_loader, pad_token_id, t=0.5
                )
                val_type = "quick"
            
            # Sincronizza DDP
            if use_ddp:
                val_tensor = torch.tensor(local_val, device=device)
                val_loss = float(all_reduce_mean(val_tensor, world_size).item())
            else:
                val_loss = local_val
            
            # Registra validation
            val_scheduler.record_validation(iter_num, val_loss)
            
            # Logging e checkpoint (solo main process)
            if main:
                elapsed = (time.time() - start_time) / 60
                
                if val_type == "full":
                    # Usa media della finestra per training loss
                    avg_train = sum(window_losses) / len(window_losses)
                    window_losses.clear()
                    
                    val_losses.append(val_loss)
                    print(f"\n[iter {iter_num:7d}] train={avg_train:.4f}  val={val_loss:.4f}  "
                          f"lr={lr:.2e}  elapsed={elapsed:.1f}min  [{val_type}]")
                    
                    if logger:
                        logger.log({
                            "type": "val",
                            "iter": iter_num,
                            "train_loss": round(avg_train, 6),
                            "val_loss": round(val_loss, 6),
                            "val_type": val_type,
                            "reason": reason,
                            "lr": lr,
                            "elapsed_min": round(elapsed, 2)
                        })
                    
                    # Checkpoint best model
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_path = os.path.join(
                            train_cfg.checkpoint_dir,
                            f"{train_cfg.checkpoint_prefix}_best.pt"
                        )
                        save_checkpoint(best_path, raw_model, optimizer, scaler,
                                        iter_num, val_loss, model_cfg, train_cfg,
                                        train_losses, val_losses, push_hf=True)
                        print(f"  ★ Best val loss: {val_loss:.4f} → {best_path}")
                        train_cfg.write_latest(iter_num, best_path)
                        
                        if logger:
                            logger.log({
                                "type": "best_checkpoint",
                                "iter": iter_num,
                                "val_loss": round(val_loss, 6),
                                "path": best_path
                            })
                else:
                    # Quick val: logging minimale
                    print(f"  [quick val] iter={iter_num} loss={val_loss:.4f} ({reason})")
                    
                    if logger:
                        logger.log({
                            "type": "quick_val",
                            "iter": iter_num,
                            "val_loss": round(val_loss, 6),
                            "reason": reason,
                            "lr": lr
                        })
            
            model.train()
        else:
            # Log skip occasionale
            if main and iter_num % 200 == 0:
                stats = val_scheduler.get_stats()
                skip_rate = stats["skip_rate"]
                if skip_rate > 0.3:
                    print(f"  [val skip] iter={iter_num} reason={reason} "
                          f"skip_rate={skip_rate:.1%} interval={stats['current_interval']}")

        # ── Checkpoint periodico ──────────────────────────────────────────
        if iter_num > 0 and iter_num % train_cfg.save_every == 0 and main:
            p = train_cfg.ckpt_path(iter_num)
            save_checkpoint(p, raw_model, optimizer, scaler,
                            iter_num, best_val_loss, model_cfg, train_cfg,
                            train_losses, val_losses, full=False)
            train_cfg.write_latest(iter_num, p)
            print(f"  Checkpoint periodico → {p}")
            if logger:
                logger.log({"type": "periodic_checkpoint", "iter": iter_num, "path": p})

    # ── Finale ────────────────────────────────────────────────────────────
    elapsed    = (time.time() - start_time) / 60
    final_path = os.path.join(train_cfg.checkpoint_dir,
                              f"{train_cfg.checkpoint_prefix}_final.pt")

    if main:
        save_checkpoint(final_path, raw_model, optimizer, scaler,
                        train_cfg.max_iters, best_val_loss, model_cfg, train_cfg,
                        train_losses, val_losses, push_hf=True, wait_hf=True)
        train_cfg.write_latest(train_cfg.max_iters, final_path)
        if logger:
            logger.log({"type": "finished", "total_iters": train_cfg.max_iters,
                        "best_val_loss": round(best_val_loss, 6),
                        "elapsed_min": round(elapsed, 2), "final_checkpoint": final_path})
            logger.close()
        print(f"\nTraining completato in {elapsed:.1f} min → {final_path}")

    if use_ddp:
        ctx.teardown() if use_ddp else None

    return {"train_losses": list(train_losses), "val_losses": val_losses,
            "best_val_loss": best_val_loss, "train_time_minutes": elapsed,
            "checkpoint_path": final_path}


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    model_cfg = get_model_config()
    train_cfg = get_train_config()
    results   = run_training(model_cfg, train_cfg)
    if is_main():
        print(f"Best val loss: {results['best_val_loss']:.4f}")