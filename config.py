from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass, field
from contextlib import nullcontext
import json
import torch


@dataclass
class DistributedConfig:
    batch_size:      int
    num_epochs:      int
    lr:              float
    seq_len:         int
    d_model:         int
    model_folder:    str
    model_basename:  str
    preload:         str
    tokenizer_file:  str
    tokenizer_model: str
    dataset_name:    str
    local_rank:      int = -1
    global_rank:     int = -1

    @property
    def train_bin_file(self) -> str:
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
        dataset_name="roneneldan/TinyStories",
    )


def get_distributed_weights_file_path(config: DistributedConfig, epoch: int) -> str:
    return str(Path(".") / config.model_folder / config.model_basename.format(epoch))


def get_distributed_latest_weights_file_path(config: DistributedConfig) -> str | None:
    model_files = sorted(
        Path(config.model_folder).glob("*.pt"),
        key=lambda x: int(x.stem.split("_")[-1]),
    )
    return str(model_files[-1]) if model_files else None


# ─────────────────────────────────────────────────────────────────────────────
# ModelConfig — solo architettura
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    # ── Vocabolario ───────────────────────────────────────────────────────
    vocab_size:     int   = 128256
    mask_token_id:  int   = 128256

    # ── Architettura ──────────────────────────────────────────────────────
    d_model:        int   = 4096
    n_layers:       int   = 32
    n_heads:        int   = 32
    n_kv_heads:     int   = 8
    d_ff:           int   = 14336

    # ── MoE ───────────────────────────────────────────────────────────────
    moe_n_routed_experts:    int = 8
    moe_top_k:               int = 2
    ds_moe_n_shared_experts: int = 1

    # ── Sequenza & Positional ─────────────────────────────────────────────
    max_seq_len: int   = 4096
    block_size:  int   = 512

    # ── Diffusion ─────────────────────────────────────────────────────────
    diffusion_T: int   = 128

    # ── Decoding ──────────────────────────────────────────────────────────
    mask_threshold: float = 0.7
    edit_threshold: float = 0.9

    # ── Training ──────────────────────────────────────────────────────────
    dropout:     float = 0.0
    bias:        bool  = False
    m2t_weight: float = 1.0
    t2t_weight: float = 1.0
    noise_ratio: float = 0.3

    # ── RoPE ──────────────────────────────────────────────────────────────
    rope_theta: float = 500000.0


def get_model_config() -> ModelConfig:
    return ModelConfig()


# ─────────────────────────────────────────────────────────────────────────────
# TrainConfig — iperparametri ottimizzati per A6000 48GB + FineWeb-Edu
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    # ── Training ──────────────────────────────────────────────────────────
    batch_size:    int   = 16      # micro-batch per step (fit in 48GB con POC)
    grad_accum:    int   = 8       # batch virtuale = batch_size * grad_accum = 128
    max_iters:     int   = 20000
    lr:            float = 3e-4
    seq_len:       int   = 512
    warmup_iters:  int   = 400
    min_lr:        float = 3e-5
    eval_interval: int   = 500
    eval_iters:    int   = 20
    max_grad_norm: float = 1.0

    # ── MTF augmentation ──────────────────────────────────────────────────
    use_mtf:   bool = True
    mtf_turns: int  = 2       # ridotto da 3: meno forward per step, più iters

    # ── Dataset ───────────────────────────────────────────────────────────
    dataset_name:       str = "HuggingFaceFW/fineweb-edu"
    dataset_split_name: str = "CC-MAIN-2024-10"   # subset ~430GB, gestibile
    tokenizer_model:    str = "bert-base-uncased"
    stream_buffer_size: int = 10000   # token nel buffer shuffle per streaming
    val_samples:        int = 500     # campioni da tenere per validation

    # ── Checkpoint ────────────────────────────────────────────────────────
    checkpoint_dir:    str = "checkpoints"
    checkpoint_prefix: str = "harold"
    preload:           str = "latest"   # "" | "latest" | "<path esplicito>"
    save_every:        int = 1000       # salva checkpoint ogni N iters
                                        # (indipendente dal best val loss)

    # ── Campi runtime ─────────────────────────────────────────────────────
    device: str = field(init=False)
    dtype:  str = field(init=False)

    def __post_init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            # A6000 è Ampere: BF16 nativo, nessun loss scaling necessario
            self.dtype = "bfloat16"
        elif torch.cuda.is_available():
            self.dtype = "float16"
        else:
            self.dtype = "float32"

    def finalize_distributed(self, local_rank: int, global_rank: int) -> None:
        self.local_rank  = local_rank
        self.global_rank = global_rank
        self.device      = f"cuda:{local_rank}"

    @property
    def ptdtype(self) -> torch.dtype:
        return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[self.dtype]

    @property
    def ctx(self):
        if self.device == "cpu":
            return nullcontext()
        return torch.amp.autocast("cuda", dtype=self.ptdtype)  # type: ignore

    @property
    def use_scaler(self) -> bool:
        """Loss scaling serve solo con FP16, non con BF16."""
        return self.dtype == "float16"

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.grad_accum

    # ── Paths checkpoint ──────────────────────────────────────────────────

    def ckpt_path(self, iter_num: int) -> str:
        return str(Path(self.checkpoint_dir) / f"{self.checkpoint_prefix}_{iter_num:07d}.pt")

    def latest_json_path(self) -> str:
        return str(Path(self.checkpoint_dir) / "latest.json")

    def write_latest(self, iter_num: int, path: str) -> None:
        """Scrive latest.json con iter e path — atomico tramite rename."""
        tmp = self.latest_json_path() + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"iter_num": iter_num, "path": path}, f)
        Path(tmp).replace(self.latest_json_path())

    def read_latest(self) -> tuple[int, str] | None:
        """Legge latest.json. Ritorna (iter_num, path) o None se non esiste."""
        p = Path(self.latest_json_path())
        if not p.exists():
            return None
        with open(p) as f:
            d = json.load(f)
        return d["iter_num"], d["path"]


def get_train_config() -> TrainConfig:
    return TrainConfig()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers checkpoint (mantenuti per compatibilità)
# ─────────────────────────────────────────────────────────────────────────────

def get_weights_file_path(config: TrainConfig, iter_num: int) -> str:
    return config.ckpt_path(iter_num)


def get_latest_weights_file_path(config: TrainConfig) -> str | None:
    result = config.read_latest()
    return result[1] if result else None