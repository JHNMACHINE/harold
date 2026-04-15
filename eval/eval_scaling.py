"""
eval_scaling.py — Harold v0.6
===============================
Verifica empiricamente che il costo computazionale di Harold v0.6 (Jamba)
scala in modo sub-quadratico con seq_len grazie a Mamba2.

Confronta la pendenza del log(tempo) vs log(seq_len):
  - Transformer puro: pendenza ≈ 2.0 (quadratico)
  - Mamba2 puro:      pendenza ≈ 1.0 (lineare)
  - Harold v0.6:      pendenza attesa ≈ 1.2-1.5 (ibrido)

Avvio:
  python eval/eval_scaling.py --checkpoint harold-v0.6-1B-sft.safetensors
"""

import argparse
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import ModelConfig, get_model_config
from core.model import Harold, build_model


SEQ_LENS   = [128, 256, 512, 1024, 2048, 4096]
BATCH_SIZE = 1
N_WARMUP   = 2
N_MEASURE  = 5


def load_model(checkpoint_path: str, device: str, jamba_attn_every: int = 4) -> tuple[Harold, ModelConfig]:
    if checkpoint_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        model_cfg = get_model_config()
        model_cfg.jamba_attn_every = jamba_attn_every
        model     = build_model(model_cfg).to(device)
        weights   = load_file(checkpoint_path, device="cpu")
        model.load_state_dict(weights, strict=False)
    else:
        state     = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model_cfg = state["model_cfg"]
        model_cfg.jamba_attn_every = jamba_attn_every
        model     = build_model(model_cfg).to(device)
        model.load_state_dict(state["model_state"], strict=False)
    model.eval()
    return model, model_cfg


def measure_time(model: Harold, seq_len: int, device: str, dtype: torch.dtype) -> float:
    """Ritorna il tempo medio per forward pass in ms."""
    B, L, D = BATCH_SIZE, seq_len, model.d_model
    x_t = torch.randn(B, L, D, device=device, dtype=dtype)
    t   = torch.rand(B, device=device)

    for _ in range(N_WARMUP):
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype):
            model.forward(x_t, t)
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    times = []
    for _ in range(N_MEASURE):
        t0 = time.perf_counter()
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype):
            model.forward(x_t, t)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    return sum(times) / len(times) * 1000


def fit_scaling_exponent(seq_lens: list[int], times_ms: list[float]) -> float:
    """
    Stima la pendenza log(tempo) ~ alpha * log(seq_len).
    Ritorna alpha (1.0 = lineare, 2.0 = quadratico).
    """
    log_lens  = [math.log(s) for s in seq_lens]
    log_times = [math.log(t) for t in times_ms]
    n  = len(log_lens)
    sx = sum(log_lens)
    sy = sum(log_times)
    sxx = sum(x * x for x in log_lens)
    sxy = sum(x * y for x, y in zip(log_lens, log_times))
    alpha = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    return alpha


def run(args: argparse.Namespace) -> None:
    device = args.device
    dtype  = torch.bfloat16 if device.startswith("cuda") else torch.float32

    print("=" * 65)
    print("Harold v0.6 — Scaling Exponent Benchmark")
    print("=" * 65)

    configs = [
        ("Harold v0.6 (Jamba)",       4),   # 3 Mamba + 1 Attn ogni 4
        ("Transformer puro (attn=1)", 1),   # tutti attention
    ]

    all_results = {}

    for label, attn_every in configs:
        print(f"\n{label} (jamba_attn_every={attn_every})")
        print(f"  {'seq_len':>8}  {'ms/step':>10}  {'tok/s':>10}")
        print("  " + "-" * 32)

        times  = []
        lenses = []

        for seq_len in SEQ_LENS:
            if seq_len > 4096:
                continue
            try:
                model, model_cfg = load_model(args.checkpoint, device, attn_every)
                ms = measure_time(model, seq_len, device, dtype)
                tps = (BATCH_SIZE * seq_len) / (ms / 1000)
                times.append(ms)
                lenses.append(seq_len)
                print(f"  {seq_len:>8}  {ms:>10.1f}  {tps:>10,.0f}")
                del model
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
            except torch.cuda.OutOfMemoryError:
                print(f"  {seq_len:>8}  {'OOM':>10}")
                break

        alpha = fit_scaling_exponent(lenses, times)
        print(f"  → Scaling exponent α = {alpha:.3f}  "
              f"({'≈linear' if alpha < 1.3 else '≈subquadratic' if alpha < 1.7 else '≈quadratic'})")

        all_results[label] = {
            "attn_every": attn_every,
            "seq_lens":   lenses,
            "times_ms":   times,
            "alpha":      alpha,
        }

    # Tabella LaTeX
    print("\n\n--- LaTeX Table ---")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\begin{tabular}{lc}")
    print("\\toprule")
    print("Model & Scaling exponent $\\alpha$ \\\\")
    print("\\midrule")
    for label, data in all_results.items():
        print(f"{label} & {data['alpha']:.3f} \\\\")
    print("\\midrule")
    print("Theoretical Transformer & 2.000 \\\\")
    print("Theoretical Mamba & 1.000 \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\caption{Empirical scaling exponents: $\\log(T) \\sim \\alpha \\cdot \\log(L)$.}")
    print("\\end{table}")

    # Salva JSON
    out_path = "eval/results_scaling.json"
    os.makedirs("eval", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nRisultati salvati in {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Harold v0.6 — Scaling Benchmark")
    parser.add_argument("--checkpoint", type=str,
                        default="harold-v0.6-1B-sft.safetensors")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())