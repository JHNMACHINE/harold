from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass, field
from contextlib import nullcontext
import json
import torch
from typing import Optional


@dataclass
class ModelConfig:
    vocab_size:     int   = 50257
    mask_token_id:  int   = 50256
    d_model:    int = 768
    n_layers:   int = 12
    n_heads:    int = 12
    n_kv_heads: int = 4
    d_ff:       int = 2048
    moe_n_routed_experts:    int = 4
    moe_top_k:               int = 2
    ds_moe_n_shared_experts: int = 2
    mla_latent_dim: int = 96
    dsa_window_size:  int = 256
    dsa_global_every: int = 64
    max_seq_len: int = 512
    block_size:  int = 512
    diffusion_beta_min: float = 0.1
    diffusion_beta_max: float = 20.0
    diffusion_T:        int   = 64
    dropout: float = 0.0
    rope_theta: float = 500000.0

def get_model_config() -> ModelConfig:
    return ModelConfig()


@dataclass
class TrainConfig:
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
    self_cond_prob: float = 0.5
    ce_loss_weight: float = 0.1
    dataset_name:       str   = "v4"
    tokenizer_model:    str   = "gpt2"
    stream_buffer_size: int   = 1000
    val_every:          int   = 200
    fineweb_weight:     float = 0.30
    wikipedia_weight:   float = 0.20
    books_weight:       float = 0.20
    c4_weight:          float = 0.15
    owt_weight:         float = 0.15
    token_weights_path: str = "token_weights/token_weights.pt"
    checkpoint_dir:    str = "checkpoints_v4"
    checkpoint_prefix: str = "harold_v04"
    preload:           str = "latest"
    save_every:        int = 1000
    device: str = field(init=False)
    dtype:  str = field(init=False)

    def __post_init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype  = (
            "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
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
        return torch.amp.autocast("cuda", dtype=self.ptdtype) # type: ignore

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


@dataclass
class SFTConfig:
    pretrain_ckpt: str = "checkpoints_v4/harold_v04_final.pt"
    batch_size:    int   = 8
    grad_accum:    int   = 16
    max_iters:     int   = 10000
    lr:            float = 2e-5
    warmup_iters:  int   = 200
    min_lr:        float = 2e-6
    max_grad_norm: float = 1.0
    eval_interval: int   = 500
    eval_iters:    int   = 20
    max_ctx_len:   int = 128
    max_resp_len:  int = 128
    max_ctx_turns: int = 3
    p_uncond:      float = 0.1
    cfg_scale:     float = 3.0
    ce_loss_weight:  float = 0.1
    self_cond_prob:  float = 0.5
    tokenizer_model: str   = "gpt2"
    val_every:       int   = 200
    stage2_max_iters: int   = 5000
    stage2_lr:        float = 1e-5
    checkpoint_dir:    str = "checkpoints_sft_v4"
    checkpoint_prefix: str = "harold_v04_sft"
    save_every:        int = 1000
    preload:           str = "latest"
    device: str = field(init=False)
    dtype:  str = field(init=False)

    def __post_init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype  = (
            "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
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
        return torch.amp.autocast("cuda", dtype=self.ptdtype) # type: ignore

    @property
    def use_scaler(self) -> bool:
        return self.dtype == "float16"

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.grad_accum

    def ckpt_path(self, stage: int, iter_num: int) -> str:
        return str(Path(self.checkpoint_dir) / f"{self.checkpoint_prefix}_s{stage}_{iter_num:07d}.pt")

    def latest_json_path(self) -> str:
        return str(Path(self.checkpoint_dir) / "latest_sft.json")

    def write_latest(self, stage: int, iter_num: int, path: str) -> None:
        tmp = self.latest_json_path() + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"stage": stage, "iter_num": iter_num, "path": path}, f)
        Path(tmp).replace(self.latest_json_path())

    def read_latest(self) -> Optional[tuple]:
        p = Path(self.latest_json_path())
        if not p.exists():
            return None
        with open(p) as f:
            d = json.load(f)
        return d["stage"], d["iter_num"], d["path"]