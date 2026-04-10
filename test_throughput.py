"""
test_throughput.py
==================
Confronta il throughput con e senza ValidationScheduler overhead.

Lancia con:
    python3 test_throughput.py
"""

import time
import torch
import sys
sys.path.insert(0, "/workspace/harold")

from core.config import get_model_config, get_train_config
from core.model import build_model
from training.validation import ValidationScheduler

# ── Setup ──────────────────────────────────────────────────────────────────
model_cfg = get_model_config()
train_cfg = get_train_config()
model     = build_model(model_cfg).cuda()
model.train()

B, L = train_cfg.batch_size, train_cfg.seq_len
x    = torch.randint(0, model_cfg.vocab_size, (B, L)).cuda()
mask = torch.ones(B, L, dtype=torch.bool).cuda()

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

N_WARMUP = 5
N_STEPS  = 20

def run_steps(n, with_scheduler=False, label=""):
    scheduler = ValidationScheduler(base_interval=500) if with_scheduler else None
    times = []

    for i in range(n):
        t0 = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)
        with train_cfg.ctx:
            loss, _ = model.compute_loss(x, mask, ce_weight=0.1)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()

        if scheduler is not None:
            # Simula overhead validation
            avg_loss = loss.item()
            scheduler.should_validate(i, avg_loss, force=False)

        elapsed = time.perf_counter() - t0
        if i >= N_WARMUP:
            times.append(elapsed)

    valid = times[:]
    avg = sum(valid) / len(valid)
    print(f"{label}: {avg:.3f}s/step  ({1/avg:.2f} step/s)  [n={len(valid)}]")
    return avg

print(f"Batch: {B}x{L}  |  Model: {sum(p.numel() for p in model.parameters())/1e6:.0f}M params")
print(f"Warmup: {N_WARMUP} step  |  Misura: {N_STEPS} step\n")

# Test 1: solo compute
t1 = run_steps(N_WARMUP + N_STEPS, with_scheduler=False, label="Senza scheduler    ")

# Test 2: con ValidationScheduler
t2 = run_steps(N_WARMUP + N_STEPS, with_scheduler=True,  label="Con scheduler      ")

overhead = (t2 - t1) / t1 * 100
print(f"\nOverhead scheduler: {overhead:+.1f}%")
print(f"Per 20k step: +{overhead/100 * t1 * 20000 / 60:.1f} min")