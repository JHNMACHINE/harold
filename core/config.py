from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass, field
from contextlib import nullcontext
import json
import torch
from typing import Optional

HF_FILENAME:  str = "harold-3B.pt"
HF_REPO_ID:   str = "JHN-MACHINE/harold"
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
    # Compute attivo per token: 2+2=4 expert fwd (vs 1+2=3 in v0.7, +33%)
    # Specializzazione: top-2 su 16 = 12.5% pool attivo (vs 25% in v0.7)
    moe_n_routed_experts:    int = 16
    moe_top_k:               int = 2
    ds_moe_n_shared_experts: int = 2
    moe_routed_hidden:       int = 608   # d_ff // 8 — calibrato per ~3.2B totali
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
    flow_sigma_min: float = 1e-4

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


def get_model_config() -> ModelConfig:
    return ModelConfig()


@dataclass
class TrainConfig:
    # ── Run di test: 10k iter su A100 SXM4 (80GB) ────────────────────────────
    # Modello 3B bf16: ~24GB overhead fisso (pesi+opt+grad) -> ~56GB liberi
    # batch_size=4 + grad_accum=16 -> 64 seq/step effettivo (4096 tok/seq)
    # = ~262k token/step. Sicuro; se no OOM: prova batch_size=6.
    # Per full run 100k iter: aumentare grad_accum=32 (128 seq/step).
    batch_size:    int   = 4
    grad_accum:    int   = 16
    max_iters:     int   = 10_000
    seq_len:       int   = 4096

    # lr scalato con sqrt(d_model/d_model_ref): 1e-4 * sqrt(1792/1280) ~ 1.18e-4
    # Conservativo per run di test: 8e-5. Full run: rivalutare a 1e-4.
    lr:            float = 8e-5
    min_lr:        float = 8e-6   # ratio min_lr/lr invariato (0.1x)

    # warmup: 10% del run (era 10% anche in v0.6: 2000/20000)
    warmup_iters:  int   = 1000

    eval_interval: int   = 200
    eval_iters:    int   = 20
    max_grad_norm: float = 1.0

    self_cond_prob:     float = 0.5
    ce_loss_weight:     float = 0.1
    tokenizer_model:    str   = "NousResearch/Llama-2-7b-hf"
    stream_buffer_size: int   = 5000
    val_every:          int   = 200

    # Checkpoint periodici sul disco locale dell'istanza (overlay, 100GB, temporaneo)
    # Spariscono quando l'istanza viene distrutta — vanno bene per i periodici.
    checkpoint_dir: str = "/workspace/checkpoints/v0.7"
    checkpoint_prefix: str = "harold_v07"
    preload:           str = "latest"
    save_every:        int = 2500   # 4 checkpoint totali nel run di test

    # Best e final sul volume persistente (/workspace, 50GB, sopravvive tra istanze)
    best_ckpt_dir:     str = "/workspace/checkpoints/v0.7" 

    # torch.compile
    use_compile:  bool = True
    compile_mode: str  = "max-autotune"

    # Optimizer — Muon invariato, parametri calibrati per 3B
    # Muon gestisce matrici 2D (attention, MoE, Mamba3 projections)
    # AdamW gestisce embedding, LayerNorm, bias, scalari SSM
    use_muon:      bool  = True
    muon_momentum: float = 0.95
    muon_beta2:    float = 0.95
    muon_ns_steps: int   = 5
    muon_wd:       float = 0.01
    adamw_betas:   tuple = (0.9, 0.95)
    adamw_eps:     float = 1e-8
    adamw_wd:      float = 0.1

    # Storico loss — ridotto proporzionalmente al run piu corto
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
    tokenizer_model: str   = "NousResearch/Llama-2-7b-hf"
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