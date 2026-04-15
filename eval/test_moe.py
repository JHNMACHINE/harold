"""
test_moe.py
===========
Confronta throughput con 4 vs 8 routed experts.

Lancia con:
    python3 test_moe.py
"""

import time
import torch
import sys
sys.path.insert(0, "/workspace/harold")

from core.config import get_model_config, get_train_config
from core.model import build_model

train_cfg = get_train_config()
B, L = train_cfg.batch_size, train_cfg.seq_len

N_WARMUP = 3
N_STEPS  = 10

def benchmark(n_routed, n_shared, label):
    model_cfg = get_model_config()
    model_cfg.moe_n_routed_experts    = n_routed
    model_cfg.ds_moe_n_shared_experts = n_shared

    model = build_model(model_cfg).cuda()
    model.train()

    x    = torch.randint(0, model_cfg.vocab_size, (B, L)).cuda()
    mask = torch.ones(B, L, dtype=torch.bool).cuda()
    opt  = torch.optim.AdamW(model.parameters(), lr=1e-4)

    times = []
    for i in range(N_WARMUP + N_STEPS):
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        with train_cfg.ctx:
            loss, _ = model.compute_loss(x, mask, ce_weight=0.1)
        loss.backward()
        opt.step()
        torch.cuda.synchronize()
        if i >= N_WARMUP:
            times.append(time.perf_counter() - t0)

    avg = sum(times) / len(times)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"{label}: {avg:.3f}s/step  ({params:.0f}M params)")
    return avg

print(f"Batch: {B}x{L}\n")

t1 = benchmark(4, 2, "MoE 2 shared + 4 routed (originale)")
t2 = benchmark(8, 1, "MoE 1 shared + 8 routed (attuale)  ")

diff = (t2 - t1) / t1 * 100
print(f"\nDelta: {diff:+.1f}%  ({t2-t1:+.3f}s/step)")
print(f"Per 20k step: {diff/100 * t1 * 20000 / 60:+.1f} min")