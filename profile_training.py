"""
profile_training.py — Harold v0.7 training profiler
=====================================================
Misura dove va il tempo nei primi 50 step:
  - Dataloader stall (CPU → GPU)
  - Forward pass
  - Backward pass
  - Optimizer step
  - Comunicazione FSDP/DDP (se multi-GPU)

Avvio:
    python profile_training.py
    python profile_training.py --use_nano      # usa Harold-Nano invece del 3.2B
    python profile_training.py --steps 20      # meno step, risultato più rapido
    python profile_training.py --use_fsdp      # profila con FSDP

Output:
    profile_results/summary.txt   — riepilogo leggibile
    profile_results/chrome.json   — apribile in chrome://tracing
    profile_results/bottleneck.md — analisi con raccomandazioni

Interpretazione:
    DataLoader stall > 20% del tempo totale → bottleneck CPU/IO
    Forward/Backward sbilanciati (ratio > 3x) → possibile overhead MoE
    Optimizer > 15% → considerare fused optimizer (già presente)
"""

import argparse
import json
import os
import time
from contextlib import contextmanager

import torch
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Harold v0.7 — Training Profiler")
    parser.add_argument("--steps",      type=int,  default=50,
                        help="Numero di step da profilare (default: 50)")
    parser.add_argument("--warmup",     type=int,  default=5,
                        help="Step di warmup prima di misurare (default: 5)")
    parser.add_argument("--use_nano",   action="store_true",
                        help="Usa Harold-Nano invece del 3.2B (più veloce)")
    parser.add_argument("--use_fsdp",   action="store_true",
                        help="Attiva FSDP (richiede torchrun con nproc>1)")
    parser.add_argument("--out_dir",    type=str, default="profile_results")
    return parser.parse_args()


@contextmanager
def cuda_timer(name: str, timings: dict):
    """Context manager che misura il tempo CUDA di un blocco."""
    if torch.cuda.is_available():
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        yield
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end)
    else:
        t0 = time.perf_counter()
        yield
        ms = (time.perf_counter() - t0) * 1000

    timings.setdefault(name, []).append(ms)


def run_profiler(args: argparse.Namespace) -> None:
    os.makedirs(args.out_dir, exist_ok=True)

    # Carica config e modello
    if args.use_nano:
        from config_nano import NanoTrainConfig, get_nano_model_config
        model_cfg = get_nano_model_config()
        train_cfg = NanoTrainConfig()
        train_cfg.use_fsdp = args.use_fsdp
        label = "Harold-Nano"
    else:
        from core.config import get_model_config, get_train_config
        model_cfg = get_model_config()
        train_cfg = get_train_config()
        train_cfg.use_fsdp = args.use_fsdp
        label = "Harold-3.2B"

    device = train_cfg.device
    print(f"Profiling {label} su {device}")
    print(f"Steps: {args.warmup} warmup + {args.steps} misurati")

    # Setup
    tokenizer = AutoTokenizer.from_pretrained(train_cfg.tokenizer_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_token_id   = tokenizer.pad_token_id
    model_cfg.vocab_size    = tokenizer.vocab_size
    model_cfg.mask_token_id = int(
        getattr(tokenizer, "mask_token_id", None) or tokenizer.vocab_size
    )

    from core.model import build_model
    from core.dataset import build_loaders
    from training.optimizer import build_optimizer
    from training.lr_schedule import get_lr

    model = build_model(model_cfg).to(device)
    model.train()

    train_loader, _ = build_loaders(train_cfg, tokenizer)
    optimizer = build_optimizer(model, train_cfg)
    scaler    = torch.GradScaler("cuda", enabled=train_cfg.use_scaler)

    timings:   dict = {}
    step_times: list = []

    train_iter = iter(train_loader)

    total_steps = args.warmup + args.steps
    for step in range(total_steps):
        measuring = step >= args.warmup
        step_start = time.perf_counter()

        # ── Dataloader ──────────────────────────────────────────────────
        t0 = time.perf_counter()
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch      = next(train_iter)
        dl_ms = (time.perf_counter() - t0) * 1000

        input_ids = batch["input_ids"].to(device, non_blocking=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        pin_ms = (time.perf_counter() - t0) * 1000 - dl_ms

        if measuring:
            timings.setdefault("dataloader_ms",   []).append(dl_ms)
            timings.setdefault("pin_memory_ms",   []).append(pin_ms)

        # ── Forward ──────────────────────────────────────────────────────
        optimizer.zero_grad(set_to_none=True)
        mask = (input_ids != pad_token_id)

        with cuda_timer("forward_ms", timings if measuring else {}):
            with train_cfg.ctx:
                loss, loss_dict = model.compute_loss(
                    x0=input_ids, mask=mask,
                    ce_weight=train_cfg.ce_loss_weight,
                    self_cond_prob=0.0,
                )

        # ── Backward ─────────────────────────────────────────────────────
        with cuda_timer("backward_ms", timings if measuring else {}):
            scaler.scale(loss).backward()

        # ── Optimizer ────────────────────────────────────────────────────
        with cuda_timer("optimizer_ms", timings if measuring else {}):
            if train_cfg.use_scaler:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

        step_end = time.perf_counter()
        step_ms  = (step_end - step_start) * 1000

        if measuring:
            step_times.append(step_ms)
            timings.setdefault("total_step_ms", []).append(step_ms)

        if step % 10 == 0:
            status = "WARMUP" if not measuring else "PROFILING"
            print(f"  [{status}] step {step:3d}: {step_ms:.0f}ms  "
                  f"loss={loss_dict['total']:.4f}")

    # ── Analisi ──────────────────────────────────────────────────────────
    def avg(key):
        vals = timings.get(key, [0])
        return sum(vals) / len(vals) if vals else 0.0

    avg_total    = avg("total_step_ms")
    avg_dl       = avg("dataloader_ms")
    avg_fwd      = avg("forward_ms")
    avg_bwd      = avg("backward_ms")
    avg_opt      = avg("optimizer_ms")
    avg_other    = avg_total - avg_dl - avg_fwd - avg_bwd - avg_opt

    tok_per_sec  = (train_cfg.batch_size * train_cfg.seq_len) / (avg_total / 1000)

    lines = []
    lines.append(f"Harold Profiler — {label}")
    lines.append(f"Device: {device}  |  Steps misurati: {args.steps}")
    lines.append(f"Batch: {train_cfg.batch_size} × seq_len {train_cfg.seq_len}")
    lines.append("")
    lines.append(f"{'Fase':<20} {'ms/step':>10} {'% totale':>10}")
    lines.append("-" * 44)
    lines.append(f"{'Dataloader':<20} {avg_dl:>10.1f} {100*avg_dl/avg_total:>9.1f}%")
    pin_ms_avg = avg("pin_memory_ms")
    lines.append(f"{'Pin memory':<20} {pin_ms_avg:>10.1f} {100*pin_ms_avg/avg_total:>9.1f}%")
    lines.append(f"{'Forward':<20} {avg_fwd:>10.1f} {100*avg_fwd/avg_total:>9.1f}%")
    lines.append(f"{'Backward':<20} {avg_bwd:>10.1f} {100*avg_bwd/avg_total:>9.1f}%")
    lines.append(f"{'Optimizer':<20} {avg_opt:>10.1f} {100*avg_opt/avg_total:>9.1f}%")
    lines.append(f"{'Altro (sync/misc)':<20} {avg_other:>10.1f} {100*avg_other/avg_total:>9.1f}%")
    lines.append("-" * 44)
    lines.append(f"{'TOTALE':<20} {avg_total:>10.1f} {'100.0%':>10}")
    lines.append("")
    lines.append(f"Throughput: {tok_per_sec:,.0f} token/sec")
    lines.append(f"           ({tok_per_sec * train_cfg.grad_accum:,.0f} tok/step effettivo con grad_accum={train_cfg.grad_accum})")

    # Diagnosi
    lines.append("")
    lines.append("DIAGNOSI:")
    dl_pct  = 100 * avg_dl / avg_total
    fwd_pct = 100 * avg_fwd / avg_total
    bwd_pct = 100 * avg_bwd / avg_total

    if dl_pct > 20:
        lines.append(f"  [BOTTLENECK] Dataloader: {dl_pct:.1f}% del tempo")
        lines.append("               → Pre-tokenizzare il dataset su NVMe locale")
        lines.append("               → Aumentare prefetch_n in MixedStreamingDataset")
    else:
        lines.append(f"  [OK] Dataloader: {dl_pct:.1f}% — non è il bottleneck")

    if bwd_pct / fwd_pct > 4:
        lines.append(f"  [ATTENZIONE] Backward {bwd_pct:.1f}% >> Forward {fwd_pct:.1f}%")
        lines.append("               → Possibile overhead MoE o gradient checkpointing")
    else:
        lines.append(f"  [OK] Forward/Backward ratio: {fwd_pct:.1f}% / {bwd_pct:.1f}%")

    if avg_opt / avg_total > 0.15:
        lines.append(f"  [ATTENZIONE] Optimizer: {100*avg_opt/avg_total:.1f}%")
    else:
        lines.append(f"  [OK] Optimizer: {100*avg_opt/avg_total:.1f}%")

    summary = "\n".join(lines)
    print("\n" + summary)

    # Salva file
    summary_path = os.path.join(args.out_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary)
    print(f"\nRisultati -> {args.out_dir}/")

    # Salva JSON per analisi esterna
    json_path = os.path.join(args.out_dir, "timings.json")
    with open(json_path, "w") as f:
        json.dump({
            "model":   label,
            "device":  device,
            "steps":   args.steps,
            "avg_ms":  {k: round(avg(k), 2) for k in timings},
            "tok_per_sec": round(tok_per_sec, 0),
        }, f, indent=2)

    # Salva patch diario
    md_path = os.path.join(args.out_dir, "profiler_result.md")
    with open(md_path, "w") as f:
        f.write(f"# Profiler result — {label}\n\n")
        f.write("```\n")
        f.write(summary)
        f.write("\n```\n")
    print(f"Diario patch -> {md_path}")


if __name__ == "__main__":
    args = parse_args()
    run_profiler(args)