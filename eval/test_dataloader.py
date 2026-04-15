"""
test_dataloader.py
==================
Misura il tempo di fetch dei batch dal DataLoader
con diversi num_workers.

Lancia con:
    python3 test_dataloader.py
"""

import time
import sys
sys.path.insert(0, "/workspace/harold")

from core.config import get_train_config
from core.dataset import build_loaders, _optimal_num_workers
from transformers import AutoTokenizer

train_cfg = get_train_config()
tokenizer = AutoTokenizer.from_pretrained(train_cfg.tokenizer_model)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

N_BATCHES = 30

def benchmark_loader(num_workers_override=None, label=""):
    import core.dataset as ds_module
    orig = ds_module._optimal_num_workers

    if num_workers_override is not None:
        ds_module._optimal_num_workers = lambda max_workers=4: num_workers_override

    train_loader, _ = build_loaders(train_cfg, tokenizer)
    it = iter(train_loader)

    # Primo batch (include init dataset)
    t0 = time.perf_counter()
    next(it)
    t_first = time.perf_counter() - t0

    # Batch successivi
    times = []
    for _ in range(N_BATCHES - 1):
        t0 = time.perf_counter()
        try:
            next(it)
        except StopIteration:
            it = iter(train_loader)
            next(it)
        times.append(time.perf_counter() - t0)

    avg = sum(times) / len(times)
    print(f"{label}: primo={t_first:.2f}s  media={avg:.3f}s  max={max(times):.3f}s")

    ds_module._optimal_num_workers = orig
    return avg

print(f"Batch size: {train_cfg.batch_size}  seq_len: {train_cfg.seq_len}\n")
print(f"num_workers attuale: {_optimal_num_workers()}\n")

benchmark_loader(0,    "num_workers=0")
benchmark_loader(1,    "num_workers=1")
benchmark_loader(2,    "num_workers=2")
benchmark_loader(None, f"num_workers={_optimal_num_workers()} (auto)")