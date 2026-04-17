"""
Harold-Nano — config_nano.py
=============================
Configurazione ridotta per scaling law check pre-training.

Obiettivo: verificare che la loss scenda con la pendenza giusta
prima di spendere $1500 sul run da 3.2B.

Architettura Nano (~300M parametri):
  Stessa struttura del 3.2B — [Mamba3, Mamba3, Mamba3, Attn] × N
  Solo d_model e n_layers ridotti. Tutto il resto invariato:
  stesso MoE, stesso Flow Matching, stesso x0-prediction.

Se la loss di Nano non converge, il 3.2B fallirà per lo stesso motivo.
Se la loss di Nano converge con la pendenza attesa dalle scaling laws,
il 3.2B è pronto per il full run.

Costo stimato: ~$30-50 su singola GPU (A10G o RTX 4090 via Vast.ai).
"""

from __future__ import annotations
from dataclasses import dataclass
from core.config import ModelConfig, TrainConfig


def get_nano_model_config(train_cfg: 'NanoTrainConfig | None' = None) -> ModelConfig:
    """
    Harold-Nano ~300M parametri.

    Invariati rispetto al 3.2B:
    - Architettura Jamba [Mamba3×3, Attn] × N
    - MoE: 1 shared + 4 routed top-2
    - Flow Matching x0-prediction
    - Logit-Normal timestep sampling
    - YaRN RoPE

    Ridotti:
    - d_model: 1792 → 512
    - n_layers: 40  → 16  (4 attention, 12 Mamba3)
    - n_heads:  28  → 8   (head_dim=64 invariato)
    - n_kv_heads: 7 → 2
    - d_ff:  4864   → 1024
    - MoE experts: 2+16 → 1+4 (ratio invariato)
    - moe_routed_hidden: 608 → 128
    - moe_shared_hidden: 1216 → 256
    - mla_latent_dim: 224 → 64
    """
    cfg = ModelConfig(
        # Architettura ridotta
        d_model    = 512,
        n_layers   = 16,   # 12 Mamba3 + 4 Attention
        n_heads    = 8,    # head_dim = 512 // 8 = 64 (invariato)
        n_kv_heads = 2,    # GQA ratio 4:1 invariato
        d_ff       = 1024,

        # MoE ridotto — ratio invariato rispetto al 3.2B
        moe_n_routed_experts    = 4,
        moe_top_k               = 2,
        ds_moe_n_shared_experts = 1,
        moe_routed_hidden       = 128,   # d_ff // 8
        moe_shared_hidden       = 256,   # d_ff // 4

        # MLA
        mla_latent_dim = 64,

        # Invariati
        dsa_window_size  = 256,
        dsa_global_every = 64,
        max_seq_len      = 4096,   # ridotto per velocità
        block_size       = 1024,
        flow_sigma_min   = 1e-4,
        t_sampling       = "logit_normal",
        t_logit_normal_std = 0.5,
        dropout          = 0.0,
        rope_theta       = 500000.0,
        rope_original_max_seq_len = 1024,
        rope_scale_factor = 1.0,   # no YaRN a questa seq_len
        use_flash_attention = True,
        jamba_attn_every = 4,
        mamba_d_state    = 64,     # ridotto da 128
        mamba_mimo_rank  = 4,
        # [v0.7-FP8] e [v0.7-HASH] — letti da NanoTrainConfig se fornito
        use_fp8      = getattr(train_cfg, "use_fp8",      False) if train_cfg else False,
        use_hash_moe = getattr(train_cfg, "use_hash_moe", False) if train_cfg else False,
    )
    return cfg


@dataclass
class NanoTrainConfig(TrainConfig):
    """
    Config training per Harold-Nano.
    Eredita da TrainConfig — compatibile con build_optimizer, DiffusionTrainer, ecc.
    Ottimizzato per singola GPU consumer (RTX 4090, A10G, ecc.).
    """
    # Override rispetto a TrainConfig
    batch_size:    int   = 8
    grad_accum:    int   = 8       # 64 seq/step effettivo — uguale al 3.2B
    max_iters:     int   = 20_000
    seq_len:       int   = 1024    # ridotto per fitting su GPU consumer

    # LR scalato con sqrt(d_model/d_model_ref)
    # 1e-4 * sqrt(512/1280) ~ 6.3e-5
    lr:            float = 6e-5
    min_lr:        float = 6e-6
    warmup_iters:  int   = 500

    stream_buffer_size: int = 1000

    checkpoint_dir:    str = "/workspace/checkpoints/nano"
    checkpoint_prefix: str = "harold_nano"
    preload:           str = ""      # no resume per default
    save_every:        int = 5_000

    compile_mode: str = "reduce-overhead"  # più veloce da compilare di max-autotune

    # Override rispetto a TrainConfig — calibrati per 300M
    self_cond_prob: float = 0.0    # disabilitato: su modello non trained produce
                                   # segnali casuali che appiattiscono i gradienti
    warmup_iters:   int   = 200    # warmup più corto — 300M converge più veloce
    lr:             float = 3e-4   # lr più alto: scaling ~sqrt(3B/300M) ~ 3x rispetto al 3.2B

    loss_history_size: int = 20_000

    # [v0.7-FP8] e [v0.7-HASH] — off per default, attivare per test comparativi:
    # Test 1 baseline:       use_fp8=False, use_hash_moe=False  (default)
    # Test 2 FP8 only:       use_fp8=True,  use_hash_moe=False
    # Test 3 Hash MoE only:  use_fp8=False, use_hash_moe=True
    # Test 4 FP8 + Hash MoE: use_fp8=True,  use_hash_moe=True
    use_fp8:      bool = False
    use_hash_moe: bool = False