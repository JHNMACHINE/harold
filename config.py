from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass, field
from contextlib import nullcontext
import json
import torch


# ─────────────────────────────────────────────────────────────────────────────
# ModelConfig — Harold v3
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    # ── Vocabolario ───────────────────────────────────────────────────────
    vocab_size:     int   = 30522   # bert-base-uncased
    mask_token_id:  int   = 103     # [MASK] in bert

    # ── Architettura ──────────────────────────────────────────────────────
    d_model:    int = 768
    n_layers:   int = 12
    n_heads:    int = 12
    n_kv_heads: int = 4
    d_ff:       int = 2048

    # ── MoE ───────────────────────────────────────────────────────────────
    moe_n_routed_experts:    int = 4
    moe_top_k:               int = 2
    ds_moe_n_shared_experts: int = 2

    # ── MLA ───────────────────────────────────────────────────────────────
    mla_latent_dim: int = 96

    # ── DSA ───────────────────────────────────────────────────────────────
    dsa_window_size:  int = 256
    dsa_global_every: int = 64

    # ── Sequenza ──────────────────────────────────────────────────────────
    max_seq_len: int = 512
    block_size:  int = 512

    # ── Diffusion VP-SDE ──────────────────────────────────────────────────
    diffusion_beta_min: float = 0.1
    diffusion_beta_max: float = 20.0

    # diffusion_T mantenuto per compatibilità con vecchi checkpoint
    diffusion_T: int = 64

    # ── Training ──────────────────────────────────────────────────────────
    dropout: float = 0.0

    # ── RoPE ──────────────────────────────────────────────────────────────
    rope_theta: float = 500000.0


def get_model_config() -> ModelConfig:
    return ModelConfig()


# ─────────────────────────────────────────────────────────────────────────────
# TrainConfig — Harold v3
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    # ── Training ──────────────────────────────────────────────────────────
    batch_size:    int   = 16
    grad_accum:    int   = 8
    max_iters:     int   = 20000
    lr:            float = 2e-4
    seq_len:       int   = 256
    warmup_iters:  int   = 400
    min_lr:        float = 2e-5
    eval_interval: int   = 500
    eval_iters:    int   = 20
    max_grad_norm: float = 1.0

    # ── Self-conditioning ─────────────────────────────────────────────────
    self_cond_prob: float = 0.5

    # ── Loss ──────────────────────────────────────────────────────────────
    ce_loss_weight: float = 0.1

    # ── Dataset ───────────────────────────────────────────────────────────
    dataset_name:       str   = "v3"
    tokenizer_model:    str   = "bert-base-uncased"
    stream_buffer_size: int   = 1000
    val_every:          int   = 200
    fineweb_weight:     float = 0.30
    wikipedia_weight:   float = 0.20
    books_weight:       float = 0.20
    c4_weight:          float = 0.15
    owt_weight:         float = 0.15

    # ── Token weights (opzionale) ──────────────────────────────────────────
    token_weights_path: str = "token_weights/token_weights.pt"

    # ── Checkpoint ────────────────────────────────────────────────────────
    checkpoint_dir:    str = "checkpoints_v3"
    checkpoint_prefix: str = "harold_200m_v3"
    preload:           str = "latest"
    save_every:        int = 1000

    # ── Runtime ───────────────────────────────────────────────────────────
    device: str = field(init=False)
    dtype:  str = field(init=False)

    def __post_init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype  = (
            "bfloat16"
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else "float16" if torch.cuda.is_available()
            else "float32"
        )

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
        return self.dtype == "float16"

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.grad_accum

    def ckpt_path(self, iter_num: int) -> str:
        return str(Path(self.checkpoint_dir) / f"{self.checkpoint_prefix}_{iter_num:07d}.pt")

    def latest_json_path(self) -> str:
        return str(Path(self.checkpoint_dir) / "latest.json")

    def write_latest(self, iter_num: int, path: str) -> None:
        tmp = self.latest_json_path() + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"iter_num": iter_num, "path": path}, f)
        Path(tmp).replace(self.latest_json_path())

    def read_latest(self) -> tuple[int, str] | None:
        p = Path(self.latest_json_path())
        if not p.exists():
            return None
        with open(p) as f:
            d = json.load(f)
        return d["iter_num"], d["path"]

    @property
    def is_main_process(self) -> bool:
        return True

    @property
    def world_size(self) -> int:
        return 1


def get_train_config() -> TrainConfig:
    return TrainConfig()


def get_weights_file_path(config: TrainConfig, iter_num: int) -> str:
    return config.ckpt_path(iter_num)


def get_latest_weights_file_path(config: TrainConfig) -> str | None:
    result = config.read_latest()
    return result[1] if result else None