from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass, field
from contextlib import nullcontext
import json
import torch
from typing import Optional

HF_FILENAME:  str = "Harold-v0.7-3B-base.pt"
HF_REPO_ID:   str = "JHN-MACHINE/harold-v0.7-training"
MAX_SKIP_RATIO = 10

@dataclass
class ModelConfig:
    # Vocabolario
    vocab_size:     int   = 32000
    mask_token_id:  int   = 0

    # Architettura — ~3.1B
    # d_model=1792: head_dim = 1792 // 28 = 64 (consistente con v0.7)
    # n_layers=40:  con jamba_attn_every=4 → 10 attention + 30 Mamba3 (ratio 3:1 invariato)
    d_model:    int = 1792
    n_layers:   int = 40
    n_heads:    int = 28
    n_kv_heads: int = 7
    d_ff:       int = 4864

    # MoE — 2 shared + 16 routed top-2
    # Compute attivo per token: 2+2=4 expert fwd
    # Specializzazione: top-2 su 16 = 12.5% pool attivo
    moe_n_routed_experts:    int = 16
    moe_top_k:               int = 2
    ds_moe_n_shared_experts: int = 2
    # [v0.7] Hidden dim espliciti — calibrati per ~3.2B totali
    # d_ff=4864: routed=4864//8=608, shared=4864//4=1216
    moe_routed_hidden:       int = 608
    moe_shared_hidden:       int = 1216

    # MLA — latent_dim scala proporzionalmente a d_model (ratio ~0.125 invariato)
    mla_latent_dim: int = 224

    # DSA
    dsa_window_size:  int = 256
    dsa_global_every: int = 64

    # Sequenza
    max_seq_len: int = 4096
    block_size:  int = 4096

    # Flow Matching
    flow_sigma_min:      float = 1e-4
    # [v0.7-T1] Timestep sampling: 'logit_normal' (default), 'cosine', 'uniform'
    # logit_normal con std=0.5 concentra i campioni intorno a t=0.5,
    # prevenendo velocity collapse e migliorando la qualita del training.
    t_sampling:          str   = "logit_normal"
    t_logit_normal_std:  float = 0.5

    # Training
    dropout: float = 0.0

    # RoPE
    rope_theta:                float = 500000.0
    rope_original_max_seq_len: int   = 1024
    rope_scale_factor:         float = 4.0

    # Flash Attention 2
    use_flash_attention: bool = True

    # [v0.6-J1] Jamba: pattern ibrido SSM + Attention
    # Un layer attention ogni `jamba_attn_every` layer.
    # Con n_layers=40 e jamba_attn_every=4: layer 3,7,11,...,39 → 10 attention, 30 Mamba3.
    jamba_attn_every: int = 4

    # [v0.7-M3] Mamba3 iperparametri
    # mamba_d_conv e mamba_expand rimossi: Mamba3 non usa causal conv1d esplicita.
    mamba_d_state:    int = 128  # dimensione dello stato SSM
    mamba_mimo_rank:  int = 4    # rank MIMO; chunk_size = 64 // mimo_rank (ottimale per bf16)

    # [v0.7-OPT-GC] Gradient checkpointing — riduce attivazioni ~50%, permette batch più grandi.
    # Costo: ~33% compute extra per il recompute nel backward.
    # Default False; True per full run su B200 con seq_len=4096.
    use_gradient_checkpointing: bool = True

    # [v0.7-FP8] FP8 per linear layers — riduce memoria del 50% e aumenta throughput.
    # Usa _scaled_mm con scale dinamica. Compatibile con PyTorch 2.1+ su Hopper/Blackwell.
    # Default False — abilitare dopo validazione sul nano run.
    use_fp8: bool = False

    # [v0.7-HASH] Hash MoE — routing deterministico via hash invece di topk learnable.
    # Elimina router, topk, searchsorted. Assignment perfettamente bilanciato per costruzione.
    # Incompatibile con use_fp8=True sul router (il router non esiste più).
    # Default False — abilitare per confronto con routing learnable sul nano run.
    use_hash_moe: bool = True


def get_model_config() -> ModelConfig:
    return ModelConfig()


@dataclass
class TrainConfig:
    batch_size:    int   = 4
    grad_accum:    int   = 16
    max_iters:     int   = 10000
    seq_len:       int   = 4096


    lr:            float = 8e-5
    min_lr:        float = 8e-6   

    # warmup: 10% del run
    warmup_iters:  int   = 100

    eval_interval: int   = 200
    eval_iters:    int   = 20
    max_grad_norm: float = 1.0

    self_cond_prob:     float = 0.5
    ce_loss_weight:     float = 0.1
    tokenizer_model:    str   = "JHN-MACHINE/harold"
    stream_buffer_size: int   = 5000
    val_every:          int   = 200

    checkpoint_dir:    str = "/workspace/checkpoints/v0.7"
    checkpoint_prefix: str = "harold_v07"
    preload:           str = "latest"
    save_every:        int = 1000

    # torch.compile
    use_compile:  bool = True
    compile_mode: str  = "max-autotune"

    # [v0.7-S3] FSDP per multi-GPU (default False — usa DDP o single-GPU)
    # Attivare con: use_fsdp=True + torchrun --nproc_per_node=N
    use_fsdp: bool = False

    # Optimizer — Muon invariato, parametri calibrati per 3B
    use_muon:      bool  = True
    muon_momentum: float = 0.95
    muon_beta2:    float = 0.95
    muon_ns_steps: int   = 5
    muon_wd:       float = 0.01
    adamw_betas:   tuple = (0.9, 0.95)
    adamw_eps:     float = 1e-8
    adamw_wd:      float = 0.1

    loss_history_size: int = 50_000

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
        return torch.autocast("cuda", dtype=self.ptdtype)

    @property
    def use_scaler(self) -> bool:
        return self.dtype == "float16"

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.grad_accum

    def ckpt_path(self, iter_num: int) -> str:
        return str(Path(self.checkpoint_dir) / f"{self.checkpoint_prefix}_{iter_num:07d}.pt")

    def best_ckpt_path(self) -> str:
        return str(Path(self.checkpoint_dir) / f"{self.checkpoint_prefix}_best.pt")

    def final_ckpt_path(self) -> str:
        return str(Path(self.checkpoint_dir) / f"{self.checkpoint_prefix}_final.pt")

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
    pretrain_ckpt: str = "checkpoints_v7/harold_v07_final.pt"
    batch_size:    int   = 8
    grad_accum:    int   = 16
    max_iters:     int   = 10000
    lr:            float = 1e-5
    warmup_iters:  int   = 200
    min_lr:        float = 2e-6
    max_grad_norm: float = 1.0
    eval_interval: int   = 500
    eval_iters:    int   = 20
    max_ctx_len:   int   = 512
    max_resp_len:  int   = 512
    max_ctx_turns: int   = 3
    p_uncond:      float = 0.1
    cfg_scale:     float = 3.0
    ce_loss_weight:  float = 0.1
    self_cond_prob:  float = 0.5
    tokenizer_model:    str   = "JHN-MACHINE/harold"
    val_every:       int   = 200
    stage2_max_iters: int   = 5000
    stage2_lr:        float = 2e-6
    checkpoint_dir:    str = "checkpoints_sft_v7"
    checkpoint_prefix: str = "harold_v07_sft"
    save_every:        int = 1000
    preload:           str = "latest"
    use_compile:  bool = True
    compile_mode: str  = "max-autotune"
    world_size: int = 0
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
        return torch.autocast("cuda", dtype=self.ptdtype)

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