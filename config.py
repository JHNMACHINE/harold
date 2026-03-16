from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass, field
from contextlib import nullcontext
import json
import torch


# ─────────────────────────────────────────────────────────────────────────────
# ModelConfig — architettura 200M
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    # ── Vocabolario ───────────────────────────────────────────────────────
    vocab_size:     int   = 30522   # bert-base-uncased
    mask_token_id:  int   = 103     # [MASK] in bert

    # ── Architettura 200M ─────────────────────────────────────────────────
    d_model:        int   = 768     # era 512
    n_layers:       int   = 12      # era 8
    n_heads:        int   = 12      # era 8  — head_dim = 768/12 = 64
    n_kv_heads:     int   = 4       # era 2  — GQA: 3x compressione KV
    d_ff:           int   = 3072    # era 2048 — regola: 4 × d_model

    # ── MoE ───────────────────────────────────────────────────────────────
    moe_n_routed_experts:    int = 4
    moe_top_k:               int = 2
    ds_moe_n_shared_experts: int = 1

    # ── Sequenza & Positional ─────────────────────────────────────────────
    max_seq_len: int   = 512
    block_size:  int   = 512

    # ── Diffusion ─────────────────────────────────────────────────────────
    diffusion_T: int   = 64

    # ── Decoding ──────────────────────────────────────────────────────────
    mask_threshold: float = 0.7
    edit_threshold: float = 0.9

    # ── Training ──────────────────────────────────────────────────────────
    dropout:     float = 0.0
    bias:        bool  = False
    m2t_weight:  float = 1.0
    t2t_weight:  float = 1.0
    noise_ratio: float = 0.3

    # ── RoPE ──────────────────────────────────────────────────────────────
    rope_theta: float = 500000.0


def get_model_config() -> ModelConfig:
    return ModelConfig()


# ─────────────────────────────────────────────────────────────────────────────
# TrainConfig — fase 1: pretraining FineWeb-Edu + Wikipedia
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    # ── Training ──────────────────────────────────────────────────────────
    batch_size:    int   = 8       # modello più grande → meno fit in VRAM
    grad_accum:    int   = 16      # batch virtuale = 8×16 = 128
    max_iters:     int   = 40000   # doppio rispetto al 61M
    lr:            float = 2e-4    # più bassa per modello più grande
    seq_len:       int   = 256     # sequenze brevi per convergenza stabile
    warmup_iters:  int   = 800     # proporzionale ai parametri
    min_lr:        float = 2e-5
    eval_interval: int   = 1000
    eval_iters:    int   = 20
    max_grad_norm: float = 1.0
    use_mtf:       bool  = False   # disabilitato: q_sample diretto è più pulito
    mtf_turns:     int   = 2

    # ── Dataset fase 1 ────────────────────────────────────────────────────
    # dataset_name="fase1" è un flag di routing in build_loaders
    # i dataset reali sono definiti in MixedStreamingDataset
    dataset_name:         str   = "fase1"
    dataset_split_name:   str   = ""        # non usato in fase 1
    tokenizer_model:      str   = "bert-base-uncased"
    stream_buffer_size:   int   = 1000      # buffer shuffle HF (in documenti)
    val_every:            int   = 200       # 1 doc ogni N va in val
    fineweb_weight:       float = 0.4       # proporzione FineWeb-Edu
    wikipedia_weight:     float = 0.6       # proporzione Wikipedia EN

    # ── Checkpoint ────────────────────────────────────────────────────────
    checkpoint_dir:    str = "checkpoints_fase1"
    checkpoint_prefix: str = "harold_200m"
    preload:           str = "latest"   # "" | "latest" | "<path>"
    save_every:        int = 2000

    # ── Campi runtime ─────────────────────────────────────────────────────
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

    # ── Paths checkpoint ──────────────────────────────────────────────────

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


def get_train_config() -> TrainConfig:
    return TrainConfig()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers checkpoint (compatibilità con train.py)
# ─────────────────────────────────────────────────────────────────────────────

def get_weights_file_path(config: TrainConfig, iter_num: int) -> str:
    return config.ckpt_path(iter_num)


def get_latest_weights_file_path(config: TrainConfig) -> str | None:
    result = config.read_latest()
    return result[1] if result else None