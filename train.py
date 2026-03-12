"""
DiffusionMoE v2 — train.py
===========================
Loop di training aggiornato per DiffusionMoE.

Differenze rispetto al train.py v1:
  - Importa DiffusionMoE da modelv2 (non DiffusionMoE da model)
  - MaskDiffusionSchedule importata da modelv2
  - forward() ritorna (logits, _): unpack ovunque
  - update_router_biases() sostituisce il loop manuale su block.moe.update_bias()
  - DiffusionMoETrainer v2 supporta M2T + T2T + MTF augmentation
  - estimate_loss aggiornato per forward v2
  - TrainConfig.use_mtf: bool per abilitare il training misto (default True)
"""

import math
import os
import time
import warnings

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from config import ModelConfig, TrainConfig, get_weights_file_path, get_latest_weights_file_path
from model import DiffusionMoE, MaskDiffusionSchedule
from dataset import MaskedDataset
from typing import Dict, List, Tuple, Optional

class DiffusionMoETrainer:
    """
    Training con obiettivi misti:
      M2T (Mask-to-Token):   maschera casuale → ricostruisce token originali
      T2T (Token-to-Token):  token rumorosi   → ricostruisce token originali
      MTF augmentation:      moltiplica i campioni per turn, mixing random M2T/T2T

    Il timestep t è campionato uniformemente per ogni batch, così il modello
    impara a denoising a tutti i livelli di rumore (fondamentale per diffusion).
    """
    def __init__(
        self,
        model:       DiffusionMoE,
        config:      ModelConfig,
        optimizer:   Optional[torch.optim.Optimizer] = None,
    ):
        self.model      = model
        self.config     = config
        self.optimizer  = optimizer
        
        # DEBUG: contatori
        self.debug_stats = {
            'total_steps': 0,
            'm2t_zero_loss': 0,
            't2t_zero_loss': 0,
            'min_mask_ratio': 1.0,
            'max_mask_ratio': 0.0,
        }

    # ── Corruption ──────────────────────────────────────────────────────────

    def apply_random_masking(
        self,
        tokens:      torch.Tensor,  # (B, L)
        noise_ratio: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """M2T: sostituisce token con [MASK]."""
        masked = tokens.clone()
        mask   = torch.rand_like(tokens.float()) < noise_ratio
        masked[mask] = self.config.mask_token_id
        
        # DEBUG: log ratio di mascheramento
        mask_ratio = mask.float().mean().item()
        self.debug_stats['min_mask_ratio'] = min(self.debug_stats['min_mask_ratio'], mask_ratio)
        self.debug_stats['max_mask_ratio'] = max(self.debug_stats['max_mask_ratio'], mask_ratio)
        
        return masked, mask

    def apply_random_noise(
        self,
        tokens:      torch.Tensor,
        noise_ratio: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """T2T: sostituisce token con token casuali (non [MASK] e non padding)."""
        noisy        = tokens.clone()
        mask         = torch.rand_like(tokens.float()) < noise_ratio
        # Genera token in [1, vocab_size-1]: esclude padding(0) e mask_token_id
        safe_upper   = max(2, self.config.vocab_size)   # almeno 2 token validi
        random_toks  = torch.randint(
            1, safe_upper,
            tokens.shape, device=tokens.device,
        )
        noisy[mask] = random_toks[mask]
        return noisy, mask

    # ── Loss ────────────────────────────────────────────────────────────────

    def compute_loss(
        self,
        logits:  torch.Tensor,   # (B, L, V)
        targets: torch.Tensor,   # (B, L)
        mask:    torch.Tensor,   # (B, L) bool — solo posizioni corrotte
    ) -> torch.Tensor:
        """Cross-entropy solo sulle posizioni corrotte."""
        if mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
            
        return F.cross_entropy(
            logits[mask],    # (N, V)
            targets[mask],   # (N,)
            reduction="mean",
        )

    # ── MTF augmentation ────────────────────────────────────────────────────

    def multi_turn_forward(
        self,
        tokens:      torch.Tensor,
        num_turns:   int   = 3,
        noise_ratio: float = 0.3,
    ) -> List[Tuple[torch.Tensor, torch.Tensor, str]]:
        """Genera versioni corrotte del batch."""
        augmentations = []
        for _ in range(num_turns):
            if torch.rand(1).item() < 0.5:
                corrupted, mask = self.apply_random_masking(tokens, noise_ratio)
                aug_type = "m2t"
            else:
                corrupted, mask = self.apply_random_noise(tokens, noise_ratio)
                aug_type = "t2t"
            augmentations.append((corrupted, mask, aug_type))
        return augmentations

    # ── Training step ────────────────────────────────────────────────────────

    def train_step(
        self,
        batch:       torch.Tensor,   # (B, L) token IDs
        use_mtf:     bool  = True,
        num_turns:   int   = 3,
        noise_ratio: Optional[float] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Singolo step di training con obiettivi misti.

        Il timestep t viene campionato uniformemente in [1, T] per ogni batch,
        il che garantisce che il modello impari a denoising a tutti i livelli.
        """
        if noise_ratio is None:
            noise_ratio = self.config.noise_ratio

        B, L   = batch.shape
        device = batch.device

        # Campiona un timestep per sample
        t = torch.randint(1, self.config.diffusion_T + 1, (B,), device=device)
        self.last_t = t

        total_loss = torch.tensor(0.0, device=device)
        losses: Dict[str, float] = {}
        n_streams = 0

        if use_mtf:
            augmentations = self.multi_turn_forward(batch, num_turns, noise_ratio)
            
            for i, (corrupted, mask, aug_type) in enumerate(augmentations):
                logits, _ = self.model(corrupted, t)
                
                loss = self.compute_loss(logits, batch, mask)
                w    = self.config.m2t_weight if aug_type == "m2t" else self.config.t2t_weight
                total_loss = total_loss + loss * w
                losses[f"{aug_type}_loss"] = loss.item()
                n_streams += 1
        else:
            # M2T stream
            m_tokens, m_mask = self.apply_random_masking(batch, noise_ratio)
            
            logits, _ = self.model(m_tokens, t)
            m2t_loss = self.compute_loss(logits, batch, m_mask)
            total_loss = total_loss + m2t_loss * self.config.m2t_weight
            losses["m2t_loss"] = m2t_loss.item()

            # T2T stream
            n_tokens, n_mask = self.apply_random_noise(batch, noise_ratio)
            logits, _ = self.model(n_tokens, t)
            t2t_loss = self.compute_loss(logits, batch, n_mask)
            total_loss = total_loss + t2t_loss * self.config.t2t_weight
            losses["t2t_loss"] = t2t_loss.item()
            n_streams = 2

        if n_streams > 1:
            total_loss = total_loss / n_streams

        losses["total_loss"] = total_loss.item()
        
        self.debug_stats['total_steps'] += 1
        
        return total_loss, losses

    def optimizer_step(self, loss: torch.Tensor, max_grad_norm: float = 1.0):
        """Backward + clip + step + aggiorna router biases."""
        assert self.optimizer is not None, "Optimizer non impostato"
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
        self.optimizer.step()
        self.model.update_router_biases()


# ─────────────────────────────────────────────────────────────────────────────
# LR schedule
# ─────────────────────────────────────────────────────────────────────────────

def get_lr(it: int, cfg: TrainConfig) -> float:
    """Cosine decay con warmup lineare."""
    if it < cfg.warmup_iters:
        return cfg.lr * it / cfg.warmup_iters
    if it > cfg.max_iters:
        return cfg.min_lr
    decay_ratio = (it - cfg.warmup_iters) / (cfg.max_iters - cfg.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return cfg.min_lr + coeff * (cfg.lr - cfg.min_lr)


# ─────────────────────────────────────────────────────────────────────────────
# Valutazione
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def estimate_loss(
    model:     DiffusionMoE,
    train_cfg: TrainConfig,
    schedule:  MaskDiffusionSchedule,
    loaders:   dict[str, DataLoader],
) -> dict[str, float]:
    """
    Stima la loss su train e val senza aggiornare i pesi.
    Usa solo M2T (masking puro) per una metrica comparabile tra run.
    """
    device = next(model.parameters()).device
    model.eval()
    out = {}

    for split, loader in loaders.items():
        losses   = torch.zeros(train_cfg.eval_iters)
        iterator = iter(loader)

        for k in range(train_cfg.eval_iters):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch    = next(iterator)

            input_ids = batch["input_ids"].to(device)
            B         = input_ids.shape[0]

            t        = torch.randint(1, schedule.T + 1, (B,), device=device)
            xt, mask = schedule.q_sample(input_ids, t)

            with train_cfg.ctx:
                # v2: forward ritorna (logits, present_kvs)
                logits, _ = model(xt, t)

            masked_logits  = logits[mask]
            masked_targets = input_ids[mask]

            losses[k] = (
                F.cross_entropy(masked_logits, masked_targets).item()
                if masked_targets.numel() > 0
                else 0.0
            )

        out[split] = losses.mean().item()

    model.train()
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Run training
# ─────────────────────────────────────────────────────────────────────────────

def run_training(model_cfg: ModelConfig, train_cfg: TrainConfig):
    device = train_cfg.device

    tokenizer = AutoTokenizer.from_pretrained(train_cfg.tokenizer_model)
    model_cfg.vocab_size    = tokenizer.vocab_size
    model_cfg.mask_token_id = tokenizer.mask_token_id

    model    = DiffusionMoE(model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"DiffusionMoE — parametri: {n_params:.1f}M  (device={device})")

    train_loader = DataLoader(
        MaskedDataset("train", train_cfg),
        batch_size  = train_cfg.batch_size,
        shuffle     = True,
        num_workers = 4,
        pin_memory  = True,
    )
    val_loader = DataLoader(
        MaskedDataset("val", train_cfg),
        batch_size  = train_cfg.batch_size,
        shuffle     = False,
        num_workers = 2,
        pin_memory  = True,
    )
    eval_loaders = {"train": train_loader, "val": val_loader}
    schedule = MaskDiffusionSchedule(model_cfg)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = train_cfg.lr,
        betas        = (0.9, 0.95),
        weight_decay = 0.1,
    )

    trainer = DiffusionMoETrainer(model, model_cfg, optimizer)

    initial_iter  = 0
    best_val_loss = float("inf")
    os.makedirs(train_cfg.checkpoint_dir, exist_ok=True)

    if train_cfg.preload:
        ckpt_path = (
            get_latest_weights_file_path(train_cfg)
            if train_cfg.preload == "latest"
            else train_cfg.preload
        )
        if ckpt_path and os.path.isfile(ckpt_path):
            print(f"Carico checkpoint: {ckpt_path}")
            state = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state["model_state"])
            optimizer.load_state_dict(state["optimizer_state"])
            initial_iter  = state.get("iter_num", 0) + 1
            best_val_loss = state.get("val_loss", float("inf"))
            del state
        else:
            print(f"Nessun checkpoint trovato ({train_cfg.preload!r}), parto da zero.")

    use_mtf    = getattr(train_cfg, "use_mtf", True)
    num_turns  = getattr(train_cfg, "mtf_turns", 3)
    #print(f"Training M2T+T2T  MTF={'ON' if use_mtf else 'OFF'}  turns={num_turns if use_mtf else '-'}")
    print(f"Avvio training — {train_cfg.max_iters} iterazioni …\n")

    train_losses, val_losses = [], []
    start_time = time.time()
    train_iter = iter(train_loader)

    for iter_num in tqdm(range(initial_iter, train_cfg.max_iters), desc="Training DiffusionMoE"):
        lr = get_lr(iter_num, train_cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        if iter_num % train_cfg.eval_interval == 0 or iter_num == train_cfg.max_iters - 1:
            losses   = estimate_loss(model, train_cfg, schedule, eval_loaders)
            val_loss = losses["val"]
            val_losses.append(val_loss)
            train_losses.append(losses["train"])

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ckpt_path     = get_weights_file_path(train_cfg, iter_num)
                torch.save(
                    {
                        "iter_num":        iter_num,
                        "model_state":     model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "val_loss":        val_loss,
                        "model_cfg":       model_cfg,
                        "train_cfg":       train_cfg,
                    },
                    ckpt_path,
                )

        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch      = next(train_iter)

        input_ids = batch["input_ids"].to(device)

        with train_cfg.ctx:
            loss, _ = trainer.train_step(
                input_ids,
                use_mtf   = use_mtf,
                num_turns = num_turns,
            )

        train_cfg.scaler.scale(loss).backward()
        train_cfg.scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        train_cfg.scaler.step(optimizer)
        train_cfg.scaler.update()
        optimizer.zero_grad(set_to_none=True)

        model.update_router_biases()

    elapsed = (time.time() - start_time) / 60

    final_path = os.path.join(
        train_cfg.checkpoint_dir,
        f"{train_cfg.checkpoint_prefix}_final.pt",
    )
    torch.save(
        {
            "model_state":     model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "model_cfg":       model_cfg,
            "train_cfg":       train_cfg,
            "val_losses":      val_losses,
            "train_losses":    train_losses,
            "best_val_loss":   best_val_loss,
            "total_iters":     train_cfg.max_iters,
        },
        final_path,
    )
    print(f"Modello finale salvato → {final_path}")

    return {
        "model":              model.cpu(),
        "val_losses":         val_losses,
        "train_losses":       train_losses,
        "train_time_minutes": elapsed,
        "best_val_loss":      best_val_loss,
        "checkpoint_path":    final_path,
        "elapsed":            elapsed
    }


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    model_cfg = ModelConfig()
    train_cfg = TrainConfig()
    results   = run_training(model_cfg, train_cfg)
    print(f"\nTraining completato...")
    print(f"\nResults: {results}")