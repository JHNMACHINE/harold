"""
Harold — train.py
====================================
Miglioramenti rispetto alla versione precedente:

  BF16 nativo
    - GradScaler disabilitato in BF16 (inutile e controproducente)
    - autocast con torch.amp.autocast("cuda", dtype=torch.bfloat16)

  Gradient accumulation
    - batch virtuale = batch_size × grad_accum (default 16×8 = 128)
    - loss scalata per 1/grad_accum prima del backward
    - optimizer.step() solo ogni grad_accum micro-step
    - zero_grad solo dopo l'optimizer step

  Checkpoint robusto
    - salva sempre ogni save_every iters (indipendente dal val loss)
    - salva il best checkpoint separatamente
    - latest.json scritto atomicamente → resume sicuro anche dopo crash
    - ripristina scaler_state (utile se si torna a FP16)

  Fix doppio backward
    - DiffusionMoETrainer.train_step() ritorna la loss RAW (non fa backward)
    - tutto il backward/clip/step è in run_training(), una sola volta
    - optimizer_step() rimosso dal trainer (era la fonte del bug)

  FineWeb-Edu streaming
    - usa build_loaders() da dataset.py
    - nessun download su disco, buffer shuffle on-the-fly
"""

import json
import math
import os
import time
import warnings
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from config import ModelConfig, TrainConfig, get_train_config, get_model_config
from model import Harold, MaskDiffusionSchedule
from dataset import build_loaders


# ─────────────────────────────────────────────────────────────────────────────
# Trainer  (solo forward + loss, niente backward)
# ─────────────────────────────────────────────────────────────────────────────

class DiffusionMoETrainer:
    """
    Calcola la loss con obiettivi misti M2T + T2T (+ MTF augmentation).

    NON esegue backward né optimizer step: quella responsabilità sta
    interamente in run_training() per gestire correttamente la
    gradient accumulation.
    """

    def __init__(self, model, config, schedule):
        self.model    = model
        self.config   = config
        self.schedule = schedule

    # ── Corruption ──────────────────────────────────────────────────────────

    def apply_random_masking(
        self, tokens: torch.Tensor, noise_ratio: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        masked = tokens.clone()
        mask   = torch.rand_like(tokens.float()) < noise_ratio
        masked[mask] = self.config.mask_token_id
        return masked, mask

    def apply_random_noise(
        self, tokens: torch.Tensor, noise_ratio: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        noisy  = tokens.clone()
        mask   = torch.rand_like(tokens.float()) < noise_ratio
        rnd    = torch.randint(1, max(2, self.config.vocab_size), tokens.shape, device=tokens.device)
        noisy[mask] = rnd[mask]
        return noisy, mask

    # ── Loss ────────────────────────────────────────────────────────────────

    def compute_loss(
        self,
        logits:  torch.Tensor,
        targets: torch.Tensor,
        mask:    torch.Tensor,
    ) -> torch.Tensor:
        if mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        return F.cross_entropy(logits[mask], targets[mask], reduction="mean")

    # ── MTF ─────────────────────────────────────────────────────────────────

    def multi_turn_forward(
        self,
        tokens:      torch.Tensor,
        num_turns:   int,
        noise_ratio: float,
    ) -> List[Tuple[torch.Tensor, torch.Tensor, str]]:
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

    # ── Train step (solo forward + loss, NO backward) ────────────────────────

    def train_step(self, batch, use_mtf=True, num_turns=2, noise_ratio=None):
        B      = batch.shape[0]
        device = batch.device
        t      = torch.randint(1, self.config.diffusion_T + 1, (B,), device=device)

        # Un solo q_sample per step — pulito, segnale pieno
        xt, mask  = self.schedule.q_sample(batch, t)
        logits, _ = self.model(xt, t)
        loss      = self.compute_loss(logits, batch, mask)

        return loss, {"total_loss": loss.item()}


# ─────────────────────────────────────────────────────────────────────────────
# LR schedule
# ─────────────────────────────────────────────────────────────────────────────

def get_lr(it: int, cfg: TrainConfig) -> float:
    """Cosine decay con warmup lineare. it = optimizer step (non micro-step)."""
    if it < cfg.warmup_iters:
        return cfg.lr * max(it, 1) / cfg.warmup_iters
    if it >= cfg.max_iters:
        return cfg.min_lr
    ratio = (it - cfg.warmup_iters) / max(cfg.max_iters - cfg.warmup_iters, 1)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return cfg.min_lr + coeff * (cfg.lr - cfg.min_lr)


# ─────────────────────────────────────────────────────────────────────────────
# Valutazione
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def estimate_loss(model, train_cfg, schedule, val_loader):
    device = next(model.parameters()).device
    model.eval()
    
    # Valuta a t fissi rappresentativi invece di t casuale
    # → metrica stabile e comparabile tra run
    t_values  = [8, 16, 32, 48, 56]   # dal facile al difficile
    all_losses = {t: [] for t in t_values}
    
    iterator = iter(val_loader)
    for k in range(train_cfg.eval_iters):
        try:
            batch = next(iterator)
        except StopIteration:
            break
        
        input_ids = batch["input_ids"].to(device)
        B         = input_ids.shape[0]
        
        for t_val in t_values:
            t    = torch.full((B,), t_val, dtype=torch.long, device=device)
            xt, mask = schedule.q_sample(input_ids, t)
            
            with train_cfg.ctx:
                logits, _ = model(xt, t)
            
            if mask.sum() > 0:
                loss = F.cross_entropy(logits[mask], input_ids[mask]).item()
                all_losses[t_val].append(loss)
    
    model.train()
    
    # Stampa breakdown per t e ritorna la media
    per_t = {}
    for t_val, losses in all_losses.items():
        per_t[t_val] = float(torch.tensor(losses).mean()) if losses else float("inf")
    
    print(f"  val per t: " + "  ".join(f"t={t}:{v:.2f}" for t, v in per_t.items()))
    return sum(per_t.values()) / len(per_t)


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    path:      str,
    model:     Harold,
    optimizer: torch.optim.Optimizer,
    scaler:    torch.amp.GradScaler,  # type: ignore
    iter_num:  int,
    val_loss:  float,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    train_losses: list,      # Queste sono le virtual losses
    val_losses:   list,
) -> None:
    state = {
        "iter_num":        iter_num,
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state":    scaler.state_dict() if train_cfg.use_scaler else {},
        "val_loss":        val_loss,
        "model_cfg":       model_cfg,
        "train_cfg":       train_cfg,
        "train_losses":    train_losses,
        "val_losses":      val_losses,
    }
    
    torch.save(state, path)

def load_checkpoint(
    path:      str,
    model:     Harold,
    optimizer: torch.optim.Optimizer,
    scaler:    torch.amp.GradScaler,  # type: ignore
    device:    str,
    train_cfg: TrainConfig,
) -> Tuple[int, float, list, list]:
    """Carica checkpoint. Ritorna (iter_num, best_val_loss, train_losses, val_losses)."""
    print(f"Carico checkpoint: {path}")
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    optimizer.load_state_dict(state["optimizer_state"])
    if train_cfg.use_scaler and state.get("scaler_state"):
        scaler.load_state_dict(state["scaler_state"])
    iter_num     = state.get("iter_num", 0) + 1
    best_val     = state.get("val_loss", float("inf"))
    train_losses = state.get("train_losses", [])
    val_losses   = state.get("val_losses", [])
    del state
    return iter_num, best_val, train_losses, val_losses


# ─────────────────────────────────────────────────────────────────────────────
# Run training
# ─────────────────────────────────────────────────────────────────────────────

def run_training(model_cfg: ModelConfig, train_cfg: TrainConfig) -> dict:
    device = train_cfg.device
    print(f"Device:  {device}")
    print(f"Dtype:   {train_cfg.dtype}  (scaler={'ON' if train_cfg.use_scaler else 'OFF — BF16 nativo'})")
    print(f"Batch virtuale: {train_cfg.batch_size} × {train_cfg.grad_accum} = {train_cfg.effective_batch_size}")

    # ── Tokenizer e modello ────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(train_cfg.tokenizer_model)
    model_cfg.vocab_size    = tokenizer.vocab_size
    model_cfg.mask_token_id = getattr(tokenizer, "mask_token_id", None) or tokenizer.vocab_size

    model    = Harold(model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Harold — {n_params:.1f}M parametri totali")

    # ── Optimizer + scaler ────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.lr,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        fused=True,   # fused AdamW: più veloce su CUDA, disponibile da PyTorch 2.0
    )
    scaler = torch.amp.GradScaler("cuda", enabled=train_cfg.use_scaler)  # type: ignore

    # ── Dataset e loader ──────────────────────────────────────────────────
    train_loader, val_loader = build_loaders(train_cfg, tokenizer)

    # ── Schedule e trainer ────────────────────────────────────────────────
    schedule = MaskDiffusionSchedule(model_cfg).to(device)
    trainer = DiffusionMoETrainer(model, model_cfg, schedule)

    # ── Checkpoint resume ─────────────────────────────────────────────────
    os.makedirs(train_cfg.checkpoint_dir, exist_ok=True)

    initial_iter  = 0
    best_val_loss = float("inf")
    train_losses: list = []
    val_losses:   list = []

    if train_cfg.preload:
        if train_cfg.preload == "latest":
            result = train_cfg.read_latest()
            ckpt_path = result[1] if result else None
        else:
            ckpt_path = train_cfg.preload

        if ckpt_path and os.path.isfile(ckpt_path):
            initial_iter, best_val_loss, train_losses, val_losses = load_checkpoint(
                ckpt_path, model, optimizer, scaler, device, train_cfg
            )
        else:
            print("Nessun checkpoint trovato, parto da zero.")

    # ── Loop principale ────────────────────────────────────────────────────
    print(f"\nAvvio training — {train_cfg.max_iters} optimizer steps "
          f"({train_cfg.max_iters * train_cfg.grad_accum} micro-steps)\n")

    model.train()
    start_time  = time.time()
    train_iter  = iter(train_loader)
    micro_step  = 0   # contatore micro-step totali
    accum_loss  = 0.0 # loss accumulata per logging

    # tqdm su optimizer steps (non micro-steps)
    pbar = tqdm(range(initial_iter, train_cfg.max_iters), desc="Harold training")
    for iter_num in pbar:
        # ── LR schedule ───────────────────────────────────────────────────
        lr = get_lr(iter_num, train_cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # ── Gradient accumulation loop ────────────────────────────────────
        # Nota: loss_raw è la loss per il singolo micro-step (batch_size)
        # loss_scaled = loss_raw / grad_accum è la loss scalata per il backward
        # La somma delle loss_scaled su grad_accum step dà la loss equivalente
        # per l'intero batch virtuale (effective_batch_size)
        optimizer.zero_grad(set_to_none=True)
        step_loss_raw_sum = 0.0  # Rinominata per chiarezza
        step_loss_scaled_sum = 0.0  # Nuova variabile per tracciare loss scalata

        for accum_idx in range(train_cfg.grad_accum):
            # Fetch batch
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch      = next(train_iter)

            input_ids = batch["input_ids"].to(device, non_blocking=True)

            # Forward + loss (dentro autocast)
            with train_cfg.ctx:
                loss_raw, loss_dict = trainer.train_step(
                    input_ids,
                    use_mtf=train_cfg.use_mtf,
                    num_turns=train_cfg.mtf_turns,
                )

            # Scala la loss per grad_accum: i gradienti si sommano su N micro-step
            # quindi ogni contributo deve essere 1/N del totale
            loss_scaled = loss_raw / train_cfg.grad_accum

            # Backward (con scaler per FP16, no-op in BF16)
            scaler.scale(loss_scaled).backward()

            # Accumula entrambe per logging
            step_loss_raw_sum += loss_raw.item()
            step_loss_scaled_sum += loss_scaled.item()  # Accumula loss scalata
            micro_step    += 1

        # ── Optimizer step (dopo tutti i micro-step) ──────────────────────
        if train_cfg.use_scaler:
            scaler.unscale_(optimizer)

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), train_cfg.max_grad_norm
        )

        scaler.step(optimizer)
        scaler.update()

        # Aggiorna router biases MoE
        model.update_router_biases()

        # ── Logging ───────────────────────────────────────────────────────
        avg_loss = step_loss_raw_sum / train_cfg.grad_accum

        train_losses.append(avg_loss)
        accum_loss += avg_loss

        pbar.set_postfix({
            "loss": f"{avg_loss:.4f}",
            "lr": f"{lr:.2e}",
            "grad": f"{grad_norm:.2f}",
        })

        # ── Valutazione ───────────────────────────────────────────────────
        if iter_num % train_cfg.eval_interval == 0 or iter_num == train_cfg.max_iters - 1:
            if iter_num == 0:
                continue
            val_loss = estimate_loss(model, train_cfg, schedule, val_loader)
            val_losses.append(val_loss)
            avg_train = accum_loss / max(train_cfg.eval_interval, 1)
            accum_loss = 0.0

            elapsed = (time.time() - start_time) / 60
            print(
                f"\n[iter {iter_num:7d}] "
                f"train={avg_train:.4f}  val={val_loss:.4f}  "
                f"lr={lr:.2e}  elapsed={elapsed:.1f}min"
            )

            # Best checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = os.path.join(
                    train_cfg.checkpoint_dir,
                    f"{train_cfg.checkpoint_prefix}_best.pt",
                )
                save_checkpoint(
                    best_path, model, optimizer, scaler,
                    iter_num, val_loss, model_cfg, train_cfg,
                    train_losses, val_losses
                )
                print(f"  ★ Nuovo best val loss: {val_loss:.4f} → {best_path}")
                train_cfg.write_latest(iter_num, best_path)

            model.train()

        # ── Checkpoint periodico ──────────────────────────────────────────
        if iter_num > 0 and iter_num % train_cfg.save_every == 0:
            periodic_path = train_cfg.ckpt_path(iter_num)
            save_checkpoint(
                periodic_path, model, optimizer, scaler,
                iter_num, best_val_loss, model_cfg, train_cfg,
                train_losses, val_losses
            )
            train_cfg.write_latest(iter_num, periodic_path)
            print(f"  Checkpoint periodico → {periodic_path}")

    # ── Checkpoint finale ─────────────────────────────────────────────────
    elapsed = (time.time() - start_time) / 60
    final_path = os.path.join(
        train_cfg.checkpoint_dir,
        f"{train_cfg.checkpoint_prefix}_final.pt",
    )
    save_checkpoint(
        final_path, model, optimizer, scaler,
        train_cfg.max_iters, best_val_loss, model_cfg, train_cfg,
        train_losses, val_losses
    )
    train_cfg.write_latest(train_cfg.max_iters, final_path)
    print(f"\nTraining completato in {elapsed:.1f} min")
    print(f"Modello finale → {final_path}")

    return {
        "model":              model.cpu(),
        "train_losses":       train_losses,
        "val_losses":         val_losses,
        "best_val_loss":      best_val_loss,
        "train_time_minutes": elapsed,
        "checkpoint_path":    final_path,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    model_cfg = ModelConfig(
        # modifica qui per scalare
        d_model=512,
        n_layers=8,
        n_heads=8,
        n_kv_heads=2,
        d_ff=2048,
        moe_n_routed_experts=4,
        moe_top_k=2,
        ds_moe_n_shared_experts=1,
        max_seq_len=512,
        diffusion_T=64,
    )

    train_cfg = TrainConfig(
        batch_size=16,
        grad_accum=8,        # batch virtuale = 128
        max_iters=20000,
        seq_len=512,
        lr=3e-4,
        warmup_iters=400,
        min_lr=3e-5,
        eval_interval=500,
        eval_iters=20,
        save_every=1000,
        dataset_name="HuggingFaceFW/fineweb-edu",
        dataset_split_name="CC-MAIN-2024-10",
        tokenizer_model="bert-base-uncased",
        preload="",
    )

    results = run_training(model_cfg, train_cfg)
    print(f"Best val loss: {results['best_val_loss']:.4f}")