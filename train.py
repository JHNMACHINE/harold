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


def get_lr(it: int, cfg: TrainConfig) -> float:
    """Cosine decay con warmup lineare."""
    if it < cfg.warmup_iters:
        return cfg.lr * it / cfg.warmup_iters
    if it > cfg.max_iters:
        return cfg.min_lr
    decay_ratio = (it - cfg.warmup_iters) / (cfg.max_iters - cfg.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return cfg.min_lr + coeff * (cfg.lr - cfg.min_lr)


@torch.no_grad()
def estimate_loss(
    model: DiffusionMoE,
    train_cfg: TrainConfig,
    schedule: MaskDiffusionSchedule,
    loaders: dict[str, DataLoader],
) -> dict[str, float]:
    """Stima la loss su train e val senza aggiornare i pesi."""
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
                logits = model(xt, t)

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


class DiffusionMoETrainer:
    def __init__(self, model, schedule, device):
        self.model    = model   # già su device, non chiamare .to() di nuovo
        self.schedule = schedule
        self.device   = device

    def compute_loss(self, batch):
        """
        1. Campiona un timestep t ~ U[1, T] per ogni sample.
        2. Corrompe i token con q_sample  →  xt, mask.
        3. Il modello predice i logits su xt.
        4. Cross-entropy SOLO sui token mascherati.
        """
        input_ids = batch["input_ids"].to(self.device)
        B         = input_ids.shape[0]

        t        = torch.randint(1, self.schedule.T + 1, (B,), device=self.device)
        xt, mask = self.schedule.q_sample(input_ids, t)

        logits = self.model(xt, t)

        masked_logits  = logits[mask]
        masked_targets = input_ids[mask]

        if masked_targets.numel() == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        return F.cross_entropy(masked_logits, masked_targets)


def run_training(model_cfg: ModelConfig, train_cfg: TrainConfig):
    device = train_cfg.device

    # ── Tokenizer ────────────────────────────────────────────────────────
    # Caricato PRIMA di costruire il modello così model_cfg.vocab_size
    # è già aggiornato quando nn.Embedding lo legge
    tokenizer = AutoTokenizer.from_pretrained(train_cfg.tokenizer_model)
    model_cfg.vocab_size    = tokenizer.vocab_size
    model_cfg.mask_token_id = tokenizer.mask_token_id

    # ── Modello ──────────────────────────────────────────────────────────
    model = DiffusionMoE(model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parametri totali: {n_params:.1f}M  (device={device})")

    # ── DataLoaders ──────────────────────────────────────────────────────
    train_loader = DataLoader(
        MaskedDataset("train", train_cfg),
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        MaskedDataset("val", train_cfg),
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    eval_loaders = {"train": train_loader, "val": val_loader}

    # ── Schedule + Trainer ───────────────────────────────────────────────
    schedule = MaskDiffusionSchedule(model_cfg)
    trainer  = DiffusionMoETrainer(model, schedule, device)

    # ── Optimizer ────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.lr,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )

    # ── Checkpoint preload ───────────────────────────────────────────────
    initial_iter  = 0
    best_val_loss = float("inf")
    os.makedirs(train_cfg.checkpoint_dir, exist_ok=True)

    if train_cfg.preload:
        if train_cfg.preload == "latest":
            ckpt_path = get_latest_weights_file_path(train_cfg)
        else:
            ckpt_path = train_cfg.preload

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

    # ── Loop principale ──────────────────────────────────────────────────
    print(f"Avvio training — {train_cfg.max_iters} iterazioni …\n")
    train_losses, val_losses = [], []
    start_time = time.time()

    train_iter = iter(train_loader)

    for iter_num in tqdm(range(initial_iter, train_cfg.max_iters), desc="Training DiffusionMoE"):

        # lr schedule
        lr = get_lr(iter_num, train_cfg)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # ── Valutazione periodica ─────────────────────────────────────────
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
                print(f"best val_loss={val_loss:.4f} → {ckpt_path}")

        # ── Forward / Backward ───────────────────────────────────────────
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch      = next(train_iter)

        with train_cfg.ctx:
            loss = trainer.compute_loss(batch)

        train_cfg.scaler.scale(loss).backward()
        train_cfg.scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        train_cfg.scaler.step(optimizer)
        train_cfg.scaler.update()
        optimizer.zero_grad(set_to_none=True)

        # ── Router bias update DeepSeek ───────────────────────────────────
        for block in model.blocks:
            block.moe.update_bias()  # type: ignore

    # ── Riepilogo finale ─────────────────────────────────────────────────
    elapsed = (time.time() - start_time) / 60
    print(f"\nTraining completato in {elapsed:.2f} minuti.")
    print(f"Best val loss: {best_val_loss:.4f}")
    print("=" * 60)

    final_path = os.path.join(train_cfg.checkpoint_dir, f"{train_cfg.checkpoint_prefix}_final.pt")
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
    }


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    model_cfg = ModelConfig()
    train_cfg = TrainConfig()

    results = run_training(model_cfg, train_cfg)
    print(f"Training completato. Best val loss: {results['best_val_loss']:.4f}")