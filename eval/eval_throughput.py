"""
eval_throughput.py — Harold v0.6
=================================
Misura il throughput (token/sec) a diverse lunghezze di sequenza.

Confronta:
  - Harold v0.6 (Jamba: Mamba2 + Attention ibrido)
  - Baseline Transformer puro (solo attention, nessun Mamba2)

Il vantaggio di Mamba2 dovrebbe emergere a seq_len >= 2048,
dove la complessità lineare supera quella quadratica dell'attention.

Avvio:
  python eval/eval_throughput.py --checkpoint checkpoints_v6/harold_v06_final.pt
  python eval/eval_throughput.py --checkpoint harold-v0.6-1B-sft.safetensors
"""

import argparse
import os
import sys
import time
from dataclasses import dataclass

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import ModelConfig, get_model_config
from core.model import Harold, build_model


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

SEQ_LENS    = [256, 512, 1024, 2048, 4096]
BATCH_SIZE  = 1
N_WARMUP    = 3
N_MEASURE   = 5


# ─────────────────────────────────────────────────────────────────────────────
# Caricamento modello
# ─────────────────────────────────────────────────────────────────────────────

def load_harold(checkpoint_path: str, device: str) -> tuple[Harold, ModelConfig]:
    if checkpoint_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        model_cfg = get_model_config()
        model     = build_model(model_cfg).to(device)
        weights   = load_file(checkpoint_path, device="cpu")
        model.load_state_dict(weights, strict=False)
    else:
        import torch
        state     = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model_cfg = state["model_cfg"]
        model     = build_model(model_cfg).to(device)
        model.load_state_dict(state["model_state"], strict=False)

    model.eval()
    return model, model_cfg


def build_transformer_baseline(model_cfg: ModelConfig, device: str) -> Harold:
    """
    Harold con tutti i layer Attention (nessun Mamba2) — baseline Transformer puro.
    Ottenuto settando jamba_attn_every=1 (ogni layer è attention).
    """
    cfg_copy = ModelConfig(
        vocab_size          = model_cfg.vocab_size,
        d_model             = model_cfg.d_model,
        n_layers            = model_cfg.n_layers,
        n_heads             = model_cfg.n_heads,
        n_kv_heads          = model_cfg.n_kv_heads,
        d_ff                = model_cfg.d_ff,
        moe_n_routed_experts    = model_cfg.moe_n_routed_experts,
        moe_top_k               = model_cfg.moe_top_k,
        ds_moe_n_shared_experts = model_cfg.ds_moe_n_shared_experts,
        mla_latent_dim      = model_cfg.mla_latent_dim,
        dsa_window_size     = model_cfg.dsa_window_size,
        dsa_global_every    = model_cfg.dsa_global_every,
        max_seq_len         = model_cfg.max_seq_len,
        block_size          = model_cfg.block_size,
        rope_scale_factor   = model_cfg.rope_scale_factor,
        jamba_attn_every    = 1,   # ← tutti i layer sono attention
    )
    model = build_model(cfg_copy).to(device)
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BenchResult:
    seq_len:       int
    tokens_per_sec: float
    ms_per_step:   float
    memory_gb:     float


def benchmark_model(
    model:    Harold,
    seq_len:  int,
    device:   str,
    dtype:    torch.dtype,
    label:    str = "",
) -> BenchResult:
    """Misura throughput forward pass a una data seq_len."""
    B, L, D = BATCH_SIZE, seq_len, model.d_model

    # Input casuali nello spazio degli embedding (come in training)
    x_t = torch.randn(B, L, D, device=device, dtype=dtype)
    t   = torch.rand(B, device=device)

    # Reset memoria GPU
    if device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # Warmup
    for _ in range(N_WARMUP):
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype):
            model.forward(x_t, t)
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    # Misura
    times = []
    for _ in range(N_MEASURE):
        t0 = time.perf_counter()
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype):
            model.forward(x_t, t)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    ms_per_step    = sum(times) / len(times) * 1000
    tokens_per_sec = (B * L) / (ms_per_step / 1000)
    memory_gb      = (torch.cuda.max_memory_allocated() / 1e9
                      if device.startswith("cuda") else 0.0)

    return BenchResult(
        seq_len        = seq_len,
        tokens_per_sec = tokens_per_sec,
        ms_per_step    = ms_per_step,
        memory_gb      = memory_gb,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    device = args.device
    dtype  = torch.bfloat16 if device.startswith("cuda") else torch.float32

    print("=" * 70)
    print("Harold v0.6 — Throughput Benchmark")
    print("=" * 70)

    # Carica Harold v0.6 (Jamba)
    print(f"\nCarico Harold v0.6 (Jamba): {args.checkpoint}")
    harold, model_cfg = load_harold(args.checkpoint, device)
    n_params = sum(p.numel() for p in harold.parameters()) / 1e6
    print(f"Parametri: {n_params:.0f}M  |  Device: {device}  |  dtype: {dtype}")

    # Costruisce baseline Transformer puro
    print("\nCostruisco baseline Transformer puro (jamba_attn_every=1)...")
    transformer = build_transformer_baseline(model_cfg, device)
    n_params_tf = sum(p.numel() for p in transformer.parameters()) / 1e6
    print(f"Parametri baseline: {n_params_tf:.0f}M")

    results_jamba = []
    results_tf    = []

    print(f"\n{'seq_len':>8}  {'Jamba tok/s':>12}  {'Transf tok/s':>13}  "
          f"{'Speedup':>8}  {'Jamba ms':>9}  {'Jamba GB':>9}")
    print("-" * 70)

    for seq_len in SEQ_LENS:
        # Salta seq_len > max_seq_len
        if seq_len > model_cfg.max_seq_len:
            print(f"{seq_len:>8}  {'(skip: > max_seq_len)':>40}")
            continue

        try:
            r_j = benchmark_model(harold,      seq_len, device, dtype, "Jamba")
            r_t = benchmark_model(transformer, seq_len, device, dtype, "Transformer")

            speedup = r_j.tokens_per_sec / r_t.tokens_per_sec

            results_jamba.append(r_j)
            results_tf.append(r_t)

            print(f"{seq_len:>8}  {r_j.tokens_per_sec:>12,.0f}  {r_t.tokens_per_sec:>13,.0f}  "
                  f"{speedup:>8.2f}x  {r_j.ms_per_step:>9.1f}  {r_j.memory_gb:>9.2f}")

        except torch.cuda.OutOfMemoryError:
            print(f"{seq_len:>8}  {'OOM':>12}  {'OOM':>13}")
            break

    # Tabella LaTeX per il paper
    print("\n\n--- LaTeX Table ---")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\begin{tabular}{rrrrc}")
    print("\\toprule")
    print("Seq Len & Jamba tok/s & Transformer tok/s & Speedup & Jamba mem (GB) \\\\")
    print("\\midrule")
    for r_j, r_t in zip(results_jamba, results_tf):
        speedup = r_j.tokens_per_sec / r_t.tokens_per_sec
        print(f"{r_j.seq_len} & {r_j.tokens_per_sec:,.0f} & {r_t.tokens_per_sec:,.0f} "
              f"& {speedup:.2f}x & {r_j.memory_gb:.2f} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\caption{Harold v0.6 Jamba vs Transformer baseline throughput.}")
    print("\\end{table}")

    # Salva risultati JSON
    import json
    out = {
        "model": "Harold v0.6",
        "checkpoint": args.checkpoint,
        "device": device,
        "dtype": str(dtype),
        "jamba": [{"seq_len": r.seq_len, "tokens_per_sec": r.tokens_per_sec,
                   "ms_per_step": r.ms_per_step, "memory_gb": r.memory_gb}
                  for r in results_jamba],
        "transformer": [{"seq_len": r.seq_len, "tokens_per_sec": r.tokens_per_sec,
                         "ms_per_step": r.ms_per_step, "memory_gb": r.memory_gb}
                        for r in results_tf],
    }
    out_path = "eval/results_throughput.json"
    os.makedirs("eval", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nRisultati salvati in {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Harold v0.6 — Throughput Benchmark")
    parser.add_argument("--checkpoint", type=str,
                        default="harold-v0.6-1B-sft.safetensors")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())