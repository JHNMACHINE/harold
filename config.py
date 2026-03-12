from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass, field
from contextlib import nullcontext
import torch


@dataclass
class DistributedConfig:
    batch_size:     int
    num_epochs:     int
    lr:             float
    seq_len:        int
    d_model:        int
    model_folder:   str
    model_basename: str
    preload:        str
    tokenizer_file: str
    tokenizer_model: str
    dataset_name:   str
    local_rank:     int = -1
    global_rank:    int = -1

    @property
    def train_bin_file(self) -> str:
        # es. "weights/tinystories/train.bin"
        name = self.dataset_name.split("/")[-1].lower()
        return str(Path(self.model_folder) / name / "train.bin")

    @property
    def val_bin_file(self) -> str:
        name = self.dataset_name.split("/")[-1].lower()
        return str(Path(self.model_folder) / name / "val.bin")
    

def get_distributed_config() -> DistributedConfig:

    return DistributedConfig(
        batch_size=4,
        num_epochs=30,
        lr=10**-4,
        seq_len=350,
        d_model=512,
        model_folder="weights",
        model_basename="tmodel_{0:02d}.pt",
        preload="latest",
        tokenizer_file="tokenizer_{0}.json",
        tokenizer_model="bert-base-uncased",
        dataset_name= "roneneldan/TinyStories"
    )

def get_distributed_weights_file_path(config: DistributedConfig, epoch: int) -> str:
    model_folder   = config.model_folder
    model_basename = config.model_basename
    model_filename = model_basename.format(epoch)
    return str(Path('.') / model_folder / model_filename)

def get_distributed_latest_weights_file_path(config: DistributedConfig) -> str | None:
    model_folder = config.model_folder
    model_basename = config.model_basename
    # Check all files in the model folder
    model_files = Path(model_folder).glob(f"*.pt")
    # Sort by epoch number (ascending order)
    model_files = sorted(model_files, key=lambda x: int(x.stem.split('_')[-1]))
    if len(model_files) == 0:
        return None
    # Get the last one
    model_filename = model_files[-1]
    return str(model_filename)


# ─────────────────────────────────────────────────────────────────────────────
# Model config  — solo architettura, nessun iperparametro di training
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    # ── Vocabolario ───────────────────────────────────────────────────────
    vocab_size:     int   = 128256
    mask_token_id:  int   = 128256   # [MASK] token (fuori dal vocabolario)

    # ── Architettura ──────────────────────────────────────────────────────
    d_model:        int   = 4096
    n_layers:       int   = 32
    n_heads:        int   = 32
    n_kv_heads:     int   = 8       # GQA: key/value heads (< n_heads)
    d_ff:           int   = 14336    # FFN hidden dim base

    # ── MoE ───────────────────────────────────────────────────────────────
    moe_n_routed_experts:   int = 8
    moe_top_k:              int = 2
    ds_moe_n_shared_experts: int = 1

    # ── Sequenza & Positional ─────────────────────────────────────────────
    max_seq_len:    int   = 4096
    block_size:     int   = 512     # Block-causal: dimensione del blocco

    # ── Diffusion ─────────────────────────────────────────────────────────
    diffusion_T:    int   = 128     # Timestep massimo

    # ── Decoding ──────────────────────────────────────────────────────────
    mask_threshold: float = 0.7     # τ_mask default (Q mode)
    edit_threshold: float = 0.9     # τ_edit default (Q mode)

    # ── Training ──────────────────────────────────────────────────────────
    dropout:        float = 0.0
    bias:           bool  = False
    m2t_weight:     float = 0.5
    t2t_weight:     float = 0.5
    noise_ratio:    float = 0.3

    # ── RoPE ──────────────────────────────────────────────────────────────
    rope_theta:     float = 500000.0 


def get_model_config() -> ModelConfig:
    return ModelConfig()


# ─────────────────────────────────────────────────────────────────────────────
# Train config  — iperparametri, dataset, checkpoint, distributed
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    # ── Training ──────────────────────────────
    batch_size:    int   = 8
    max_iters:     int   = 10
    lr:            float = 3e-4
    seq_len:       int   = 128
    warmup_iters:  int   = 20
    min_lr:        float = 3e-5
    eval_interval: int   = 50
    eval_iters:    int   = 10

    # ── Dataset ───────────────────────────────
    dataset_name:    str = "roneneldan/TinyStories"
    tokenizer_model: str = "bert-base-uncased"
    train_bin_file:  str = "train.bin"
    val_bin_file:    str = "val.bin"

    # ── Checkpoint ────────────────────────────
    checkpoint_dir:    str = "checkpoints"
    checkpoint_prefix: str = "diffusion_moe"
    preload:           str = ""     # "" | "latest" | "<path esplicito>"

    # ── Campi runtime (non editare direttamente) ───────────────────────
    device:  str                      = field(init=False)
    dtype:   str                      = field(init=False)
    _scaler: torch.amp.GradScaler     = field(init=False, repr=False)  # type: ignore

    def __post_init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype  = (
            "bfloat16"
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else "float16"
        )
        self._scaler = torch.amp.GradScaler(  # type: ignore
            "cuda", enabled=(self.dtype == "float16")
        )

    def finalize_distributed(self, local_rank: int, global_rank: int) -> None:
        """Chiama dopo setup_distributed() in train_distributed.py."""
        self.local_rank  = local_rank
        self.global_rank = global_rank
        self.device      = f"cuda:{local_rank}"
        self._scaler     = torch.amp.GradScaler(  # type: ignore
            "cuda", enabled=(self.dtype == "float16")
        )

    # ── Proprietà derivate ────────────────────

    @property
    def ptdtype(self) -> torch.dtype:
        return {
            "float16":  torch.float16,
            "bfloat16": torch.bfloat16,
            "float32":  torch.float32,
        }[self.dtype]

    @property
    def ctx(self):
        if self.device == "cpu":
            return nullcontext()
        return torch.cuda.amp.autocast(dtype=self.ptdtype)

    @property
    def scaler(self) -> torch.amp.GradScaler:  # type: ignore
        return self._scaler


def get_train_config() -> TrainConfig:
    return TrainConfig()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def get_weights_file_path(config: TrainConfig, iter_num: int) -> str:
    return str(Path(config.checkpoint_dir) / f"{config.checkpoint_prefix}_{iter_num}.pt")


def get_latest_weights_file_path(config: TrainConfig) -> str | None:
    files = Path(config.checkpoint_dir).glob(f"{config.checkpoint_prefix}_*.pt")
    # Esclude _final.pt e ordina per numero iterazione
    numeric = []
    for f in files:
        stem_suffix = f.stem.split("_")[-1]
        if stem_suffix.isdigit():
            numeric.append((int(stem_suffix), f))
    if not numeric:
        return None
    return str(sorted(numeric)[-1][1])