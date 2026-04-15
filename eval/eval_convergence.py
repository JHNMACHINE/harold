"""
eval_convergence.py — Harold v0.6
===================================
Confronta le curve di convergenza di Harold v0.5 vs v0.6
usando i dati del paper v0.5 come baseline e il training.log di v0.6.

Non richiede GPU — legge i log e produce tabelle/plot per il paper.

Avvio:
  python eval/eval_convergence.py --log checkpoints_v6/training.log
"""

import argparse
import json
import os
import sys

# ─────────────────────────────────────────────────────────────────────────────
# Dati baseline Harold v0.5 (dal paper)
# ─────────────────────────────────────────────────────────────────────────────

V05_DATA = [
    # (step, val_loss, score_t01, score_t09, ce_t01, ce_t05, ce_t09)
    (500,   1.574, 0.914, 0.957, 3.422, 7.240, 7.564),
    (1000,  1.119, 0.500, 0.514, 3.554, 7.315, 7.620),
    (2000,  0.653, 0.085, 0.029, 2.497, 7.174, 7.699),
    (4000,  0.602, 0.067, 0.020, 0.843, 7.037, 7.773),
    (8000,  0.584, 0.059, 0.016, 0.381, 6.976, 7.742),
    (12000, 0.582, 0.051, 0.015, 0.337, 6.960, 7.733),
    (16500, 0.576, 0.049, 0.014, 0.336, 6.897, 7.665),
]


# ─────────────────────────────────────────────────────────────────────────────
# Parsing del training.log di v0.6
# ─────────────────────────────────────────────────────────────────────────────

def parse_training_log(log_path: str) -> list[dict]:
    """
    Legge il training.log (formato JSONL) e ritorna i record di validation.
    """
    records = []
    if not os.path.isfile(log_path):
        print(f"  WARNING: log non trovato: {log_path}")
        return records

    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if obj.get("type") == "val" and obj.get("val_type") == "full":
                    records.append(obj)
            except json.JSONDecodeError:
                continue

    return records


def parse_val_detail_log(log_path: str) -> list[dict]:
    """Legge i record val_detail (per-t breakdown)."""
    records = []
    if not os.path.isfile(log_path):
        return records
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if obj.get("type") == "val_detail":
                    records.append(obj)
            except json.JSONDecodeError:
                continue
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Confronto e output
# ─────────────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    print("=" * 70)
    print("Harold v0.5 vs v0.6 — Convergence Comparison")
    print("=" * 70)

    # Carica dati v0.6
    val_records    = parse_training_log(args.log)
    detail_records = parse_val_detail_log(args.log)

    # Costruisci lookup per-timestep
    detail_by_iter: dict[int, dict] = {}
    for r in detail_records:
        detail_by_iter[r["iter"]] = r

    # ── Tabella val loss ──────────────────────────────────────────────────────
    print("\n--- Val Loss Comparison: Harold v0.5 vs v0.6 ---\n")
    print(f"{'Step':>7}  {'v0.5 val':>10}  {'v0.6 val':>10}  {'Δ':>8}")
    print("-" * 42)

    v06_by_iter = {r["iter"]: r for r in val_records}
    v05_steps   = [row[0] for row in V05_DATA]

    for row in V05_DATA:
        step, v05_val = row[0], row[1]
        # Trova il record v0.6 più vicino
        closest = min(v06_by_iter.keys(), key=lambda x: abs(x - step)) if v06_by_iter else None
        if closest is not None and abs(closest - step) <= step * 0.3:
            v06_val = v06_by_iter[closest]["val_loss"]
            delta   = v06_val - v05_val
            print(f"{step:>7}  {v05_val:>10.4f}  {v06_val:>10.4f}  {delta:>+8.4f}")
        else:
            print(f"{step:>7}  {v05_val:>10.4f}  {'—':>10}  {'—':>8}")

    # ── Tabella per-timestep v0.6 ─────────────────────────────────────────────
    if detail_records:
        print("\n--- Harold v0.6: Per-timestep Score/CE at selected steps ---\n")
        print(f"{'Step':>7}  {'Sc t=0.3':>9}  {'Sc t=0.5':>9}  {'Sc t=0.7':>9}  "
              f"{'CE t=0.3':>9}  {'CE t=0.5':>9}  {'CE t=0.7':>9}")
        print("-" * 70)

        shown = set()
        for r in sorted(detail_records, key=lambda x: x["iter"]):
            it = r["iter"]
            if it in shown:
                continue
            shown.add(it)
            sp = r.get("score_per_t", {})
            cp = r.get("ce_per_t", {})
            print(f"{it:>7}  "
                  f"{sp.get('0.3', float('nan')):>9.4f}  "
                  f"{sp.get('0.5', float('nan')):>9.4f}  "
                  f"{sp.get('0.7', float('nan')):>9.4f}  "
                  f"{cp.get('0.3', float('nan')):>9.4f}  "
                  f"{cp.get('0.5', float('nan')):>9.4f}  "
                  f"{cp.get('0.7', float('nan')):>9.4f}")

    # ── LaTeX Table ───────────────────────────────────────────────────────────
    print("\n\n--- LaTeX Table: Val Loss Comparison ---")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\begin{tabular}{rrr}")
    print("\\toprule")
    print("Step & Harold v0.5 val loss & Harold v0.6 val loss \\\\")
    print("\\midrule")
    for row in V05_DATA:
        step, v05_val = row[0], row[1]
        closest = min(v06_by_iter.keys(), key=lambda x: abs(x - step)) if v06_by_iter else None
        if closest is not None and abs(closest - step) <= step * 0.3:
            v06_val = v06_by_iter[closest]["val_loss"]
            print(f"{step} & {v05_val:.4f} & {v06_val:.4f} \\\\")
        else:
            print(f"{step} & {v05_val:.4f} & — \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\caption{Validation loss: Harold v0.5 (Transformer + Flow Matching) "
          "vs Harold v0.6 (Jamba + Flow Matching).}")
    print("\\end{table}")

    # Salva JSON
    out = {
        "v05": [{"step": r[0], "val_loss": r[1], "score_t01": r[2],
                 "score_t09": r[3], "ce_t01": r[4], "ce_t05": r[5], "ce_t09": r[6]}
                for r in V05_DATA],
        "v06_val":    val_records,
        "v06_detail": detail_records,
    }
    out_path = "eval/results_convergence.json"
    os.makedirs("eval", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nRisultati salvati in {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Harold v0.6 — Convergence Comparison")
    parser.add_argument("--log", type=str,
                        default="checkpoints_v6/training.log",
                        help="Path del training.log di v0.6")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())