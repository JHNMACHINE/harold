import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.distributed import init_process_group, destroy_process_group

import warnings
from tqdm import tqdm
import os
from pathlib import Path
import argparse

from datasets import load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from model import DiffusionMoE, MaskDiffusionSchedule, build_model
from dataset import DistributedDataset
from config import (
    DistributedConfig, ModelConfig,
    get_distributed_config, get_model_config,
    get_distributed_weights_file_path, get_distributed_latest_weights_file_path,
)


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_validation(
    model:       nn.Module,
    val_loader:  DataLoader,
    schedule:    MaskDiffusionSchedule,
    device:      torch.device,
    global_step: int,
) -> float:
    model.eval()

    total_loss = 0.0
    n_batches  = 0

    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        B         = input_ids.shape[0]

        t        = torch.randint(1, schedule.T + 1, (B,), device=device)
        xt, mask = schedule.q_sample(input_ids, t)

        logits         = model(xt, t)
        masked_logits  = logits[mask]
        masked_targets = input_ids[mask]

        if masked_targets.numel() > 0:
            total_loss += F.cross_entropy(masked_logits, masked_targets).item()
            n_batches  += 1

    val_loss = total_loss / n_batches if n_batches > 0 else 0.0
    print(f"  val_loss={val_loss:.4f}  (step {global_step})")

    model.train()
    return val_loss


# ─────────────────────────────────────────────────────────────────────────────
# Tokenizer
# ─────────────────────────────────────────────────────────────────────────────

def get_or_build_tokenizer(config: DistributedConfig) -> PreTrainedTokenizerBase:
    tokenizer_path = Path(config.tokenizer_file)
    if not tokenizer_path.exists():
        print(f"GPU {config.local_rank} - Downloading tokenizer {config.tokenizer_model}...")
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_model)
        tokenizer.save_pretrained(str(tokenizer_path))
        print(f"GPU {config.local_rank} - Tokenizer salvato → {tokenizer_path}")
    else:
        print(f"GPU {config.local_rank} - Tokenizer trovato → {tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
    return tokenizer  # type: ignore


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

def get_ds(config: DistributedConfig):
    print(f"GPU {config.local_rank} - Loading dataset...")

    # Scarica il dataset se i .bin non esistono ancora
    if not (os.path.exists(config.train_bin_file) and os.path.exists(config.val_bin_file)):
        load_dataset(config.dataset_name, split='train')

    tokenizer = get_or_build_tokenizer(config)

    train_ds = DistributedDataset("train", config)
    val_ds   = DistributedDataset("val",   config)

    train_dataloader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=False,
        sampler=DistributedSampler(train_ds, shuffle=True),
        num_workers=4,
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return train_dataloader, val_dataloader, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

def get_model(model_cfg: ModelConfig, tokenizer: PreTrainedTokenizerBase) -> DiffusionMoE:
    model_cfg.vocab_size    = tokenizer.vocab_size      # type: ignore
    model_cfg.mask_token_id = tokenizer.mask_token_id   # type: ignore
    return build_model(model_cfg)


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_model(dist_cfg: DistributedConfig, model_cfg: ModelConfig):
    assert torch.cuda.is_available(), "Training on CPU is not supported"
    device = torch.device(f"cuda:{dist_cfg.local_rank}")
    print(f"GPU {dist_cfg.local_rank} - Using device: {device}")

    Path(dist_cfg.model_folder).mkdir(parents=True, exist_ok=True)

    # ── Dataset ──────────────────────────────────────────────────────────────
    train_dataloader, val_dataloader, tokenizer = get_ds(dist_cfg)

    # ── Model ────────────────────────────────────────────────────────────────
    model = get_model(model_cfg, tokenizer).to(device)

    if dist_cfg.local_rank == 0:
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"GPU {dist_cfg.local_rank} - Parametri totali: {n_params:.1f}M")

    # ── Optimizer ────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=dist_cfg.lr, eps=1e-9)

    # ── Checkpoint preload (PRIMA del wrap DDP) ───────────────────────────────
    initial_epoch = 0
    global_step   = 0

    if dist_cfg.preload != "":
        if dist_cfg.preload == "latest":
            model_filename = get_distributed_latest_weights_file_path(dist_cfg)
        else:
            model_filename = dist_cfg.preload  # è già un path completo
        if model_filename is not None:
            print(f"GPU {dist_cfg.local_rank} - Preloading model {model_filename}")
            state = torch.load(model_filename, map_location=device)
            model.load_state_dict(state["model_state_dict"])
            initial_epoch = state["epoch"] + 1
            optimizer.load_state_dict(state["optimizer_state_dict"])
            global_step   = state["global_step"]
            del state
        else:
            print(f"GPU {dist_cfg.local_rank} - Could not find model to preload: {dist_cfg.preload}. Starting from scratch")

    # ── DDP wrap (DOPO preload, PRIMA degli step) ─────────────────────────────
    model = DistributedDataParallel(model, device_ids=[dist_cfg.local_rank])

    # ── Schedule ─────────────────────────────────────────────────────────────
    schedule = MaskDiffusionSchedule(model_cfg)

    # ── Epoch loop ───────────────────────────────────────────────────────────
    for epoch in range(initial_epoch, dist_cfg.num_epochs):
        torch.cuda.empty_cache()
        model.train()

        batch_iterator = tqdm(
            train_dataloader,
            desc=f"Processing Epoch {epoch:02d} on rank {dist_cfg.global_rank}",
            disable=dist_cfg.local_rank != 0,
        )

        for batch in batch_iterator:
            input_ids = batch["input_ids"].to(device)
            B         = input_ids.shape[0]

            t        = torch.randint(1, schedule.T + 1, (B,), device=device)
            xt, mask = schedule.q_sample(input_ids, t)

            logits         = model(xt, t)
            masked_logits  = logits[mask]
            masked_targets = input_ids[mask]

            if masked_targets.numel() == 0:
                loss = torch.tensor(0.0, device=device, requires_grad=True)
            else:
                loss = F.cross_entropy(masked_logits, masked_targets)

            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}", "global_step": global_step})

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            # Router bias update DeepSeek
            for block in model.module.blocks:  # type: ignore
                block.moe.update_bias()

            global_step += 1

        # ── Validation + checkpoint (solo rank 0) ────────────────────────────
        if dist_cfg.global_rank == 0:
            val_loss = run_validation(
                model.module,  # type: ignore
                val_dataloader,
                schedule,
                device,
                global_step,
            )

            model_filename = get_distributed_weights_file_path(dist_cfg, epoch)
            torch.save(
                {
                    "epoch":                epoch,
                    "model_state_dict":     model.module.state_dict(),  # type: ignore
                    "optimizer_state_dict": optimizer.state_dict(),
                    "global_step":          global_step,
                    "val_loss":             val_loss,
                },
                model_filename,
            )
            print(f"GPU {dist_cfg.local_rank} - Checkpoint salvato → {model_filename}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    dist_cfg  = get_distributed_config()
    model_cfg = get_model_config()

    # Argomenti da CLI (opzionali, sovrascrivono dist_cfg)
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size",    type=int,   default=dist_cfg.batch_size)
    parser.add_argument("--num_epochs",    type=int,   default=dist_cfg.num_epochs)
    parser.add_argument("--lr",            type=float, default=dist_cfg.lr)
    parser.add_argument("--seq_len",       type=int,   default=dist_cfg.seq_len)
    parser.add_argument("--model_folder",  type=str,   default=dist_cfg.model_folder)
    parser.add_argument("--model_basename",type=str,   default=dist_cfg.model_basename)
    parser.add_argument("--preload",       type=str,   default=dist_cfg.preload)
    parser.add_argument("--tokenizer_file",type=str,   default=dist_cfg.tokenizer_file)
    args = parser.parse_args()
    dist_cfg.__dict__.update(vars(args))

    # Rank iniettati da torchrun
    dist_cfg.local_rank  = int(os.environ["LOCAL_RANK"])
    dist_cfg.global_rank = int(os.environ["RANK"])

    assert dist_cfg.local_rank  != -1, "LOCAL_RANK environment variable not set"
    assert dist_cfg.global_rank != -1, "RANK environment variable not set"

    # Print configuration (solo rank 0)
    if dist_cfg.local_rank == 0:
        print("=== DistributedConfig ===")
        for key, value in dist_cfg.__dict__.items():
            print(f"{key:>25}: {value}")
        print("\n=== ModelConfig ===")
        for key, value in vars(model_cfg).items():
            print(f"{key:>25}: {value}")

    # Init DDP
    init_process_group(backend="nccl")
    torch.cuda.set_device(dist_cfg.local_rank)

    # Train
    train_model(dist_cfg, model_cfg)

    # Cleanup
    destroy_process_group()