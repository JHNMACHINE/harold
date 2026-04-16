"""
eval_generation.py — Harold v0.7
==================================
Valutazione qualitativa su 20 prompt fissi. Produce JSON + Markdown
per il diario di bordo e confronto visivo tra checkpoint.

Avvio singolo checkpoint:
  python eval/eval_generation.py --checkpoint /workspace/checkpoints/v0.7/harold_v07_best.pt

Avvio su tutti i checkpoint in una directory (modalità automatica):
  python eval/eval_generation.py --checkpoint_dir /workspace/checkpoints/v0.7 --auto

Cambiamenti rispetto a v0.6:
  [v0.7-E1] 20 prompt fissi (era 10) — copertura più ampia per confronto visivo
  [v0.7-E2] Modalità --auto: gira su tutti i checkpoint ordinati per step,
             produce results/step_XXXXXX.md e un confronto comparativo
  [v0.7-E3] Output Markdown strutturato per diario di bordo
  [v0.7-E4] Usa SamplerConfig v0.7 (iterative decoding, x0-prediction)
  [v0.7-E5] Metriche aggiuntive: avg_words, pct_coherent, pct_on_topic
"""

import argparse
import glob
import json
import os
import sys
import time
from datetime import datetime

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.sampler import load_model, sample, SamplerConfig
from transformers import AutoTokenizer


# ─────────────────────────────────────────────────────────────────────────────
# [v0.7-E1] 20 prompt fissi — invariati tra run per confronto visivo
# ─────────────────────────────────────────────────────────────────────────────
#
# Categorie scelte per rilevare capacità diverse che emergono in momenti diversi:
# - Completamento semplice: prime ad emergere (~10-20k iter)
# - Grammatica/sintassi: emerge ~20-40k iter
# - Ragionamento: emerge ~40-70k iter
# - Coding e long-form: emerge ~50-80k iter

EVAL_PROMPTS = [
    # ── Completamento semplice ────────────────────────────────────────────
    {"id": "simple_1",    "category": "Simple",      "prompt": "The capital of France is"},
    {"id": "simple_2",    "category": "Simple",      "prompt": "The sun rises in the"},
    {"id": "simple_3",    "category": "Simple",      "prompt": "Water boils at"},

    # ── Factual ───────────────────────────────────────────────────────────
    {"id": "factual_1",   "category": "Factual",     "prompt": "Who wrote Romeo and Juliet?"},
    {"id": "factual_2",   "category": "Factual",     "prompt": "What is the speed of light?"},
    {"id": "factual_3",   "category": "Factual",     "prompt": "How many planets are in the solar system?"},

    # ── Ragionamento ──────────────────────────────────────────────────────
    {"id": "reason_1",    "category": "Reasoning",   "prompt": "If a train travels at 60 mph for 2 hours, how far does it travel?"},
    {"id": "reason_2",    "category": "Reasoning",   "prompt": "What comes next in the sequence: 2, 4, 8, 16?"},
    {"id": "reason_3",    "category": "Reasoning",   "prompt": "If I have 3 apples and eat 1, how many are left?"},

    # ── Istruzione ────────────────────────────────────────────────────────
    {"id": "instruct_1",  "category": "Instruction", "prompt": "List three benefits of regular exercise."},
    {"id": "instruct_2",  "category": "Instruction", "prompt": "Explain machine learning in simple terms."},
    {"id": "instruct_3",  "category": "Instruction", "prompt": "What are the steps to make a cup of tea?"},

    # ── Creativo ──────────────────────────────────────────────────────────
    {"id": "creative_1",  "category": "Creative",    "prompt": "Write a short poem about the ocean."},
    {"id": "creative_2",  "category": "Creative",    "prompt": "Once upon a time in a forest far away,"},
    {"id": "creative_3",  "category": "Creative",    "prompt": "Describe a sunset using vivid imagery."},

    # ── Coding ────────────────────────────────────────────────────────────
    {"id": "code_1",      "category": "Coding",      "prompt": "Write a Python function that returns the factorial of a number."},
    {"id": "code_2",      "category": "Coding",      "prompt": "What is a dictionary in Python?"},
    {"id": "code_3",      "category": "Coding",      "prompt": "How do you reverse a string in Python?"},

    # ── Long-form ─────────────────────────────────────────────────────────
    {"id": "longform_1",  "category": "Long-form",   "prompt": "Explain how the internet works."},
    {"id": "longform_2",  "category": "Long-form",   "prompt": "What are the main causes of climate change?"},
]

assert len(EVAL_PROMPTS) == 20, f"Devono essere esattamente 20 prompt, trovati {len(EVAL_PROMPTS)}"


# ─────────────────────────────────────────────────────────────────────────────
# Valutazione euristica
# ─────────────────────────────────────────────────────────────────────────────

def score_response(prompt: str, response: str) -> dict:
    """
    Valutazione euristica della risposta.

    - coherent: la risposta contiene parole reali (non solo rumore)
    - on_topic: overlap con le parole chiave del prompt
    - length:   lunghezza in parole e caratteri
    """
    words    = [w for w in response.split() if len(w) > 1 and w.isalpha()]
    n_words  = len(words)
    coherent = n_words > 5 and len(set(words)) > 3

    prompt_words = set(prompt.lower().split())
    resp_words   = set(response.lower().split())
    overlap      = len(prompt_words & resp_words)
    on_topic     = overlap >= 1 or n_words > 10

    return {
        "coherent": coherent,
        "on_topic": on_topic,
        "n_words":  n_words,
        "n_chars":  len(response),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Output Markdown
# ─────────────────────────────────────────────────────────────────────────────

def results_to_markdown(
    results:    list,
    checkpoint: str,
    step:       int,
    summary:    dict,
    cfg_scale:  float,
    n_steps:    int,
) -> str:
    """Genera un file Markdown per il diario di bordo."""
    lines = []
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines.append(f"# Harold v0.7 — Generation eval @ step {step:,}")
    lines.append(f"")
    lines.append(f"**Data**: {ts}  ")
    lines.append(f"**Checkpoint**: `{os.path.basename(checkpoint)}`  ")
    lines.append(f"**CFG scale**: {cfg_scale}  |  **Steps**: {n_steps}  ")
    lines.append(f"")
    lines.append(f"## Sommario")
    lines.append(f"")
    lines.append(f"| Metrica | Valore |")
    lines.append(f"|---|---|")
    lines.append(f"| Coerenza | {summary['n_coherent']}/{summary['n_total']} ({100*summary['n_coherent']//summary['n_total']}%) |")
    lines.append(f"| On-topic | {summary['n_on_topic']}/{summary['n_total']} ({100*summary['n_on_topic']//summary['n_total']}%) |")
    lines.append(f"| Parole medie | {summary['avg_words']:.1f} |")
    lines.append(f"")

    # Per categoria
    categories = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(r)

    for cat, items in categories.items():
        n_coh = sum(1 for r in items if r["scores"]["coherent"])
        lines.append(f"### {cat} ({n_coh}/{len(items)} coerenti)")
        lines.append(f"")
        for r in items:
            coh = "✓" if r["scores"]["coherent"] else "✗"
            ont = "✓" if r["scores"]["on_topic"] else "✗"
            resp_short = r["response"][:300].replace("\n", " ")
            if len(r["response"]) > 300:
                resp_short += "..."
            lines.append(f"**`{r['id']}`** [{coh} coh | {ont} topic | {r['scores']['n_words']} words]  ")
            lines.append(f"> **Prompt**: {r['prompt']}  ")
            lines.append(f"> **Response**: {resp_short}")
            lines.append(f"")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Funzione principale di valutazione
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_checkpoint(
    checkpoint:   str,
    tokenizer,
    pad_token_id: int,
    args:         argparse.Namespace,
    step:         int = 0,
    model=None,
) -> dict:
    """
    Genera risposte ai 20 prompt fissi e ritorna i risultati.

    Args:
        model: se fornito, usa il modello già in memoria invece di
               ricaricare da checkpoint. Utile per eval durante il training
               senza doppio load in VRAM.
    """
    device = args.device
    dtype  = torch.bfloat16 if device.startswith("cuda") else torch.float32

    print(f"\n{'='*70}")
    print(f"Harold v0.7 — Generation Evaluation @ step {step:,}")
    print(f"Checkpoint: {checkpoint}")
    print(f"{'='*70}")

    _model_provided = model is not None
    if model is None:
        model = load_model(checkpoint, device=device, dtype=dtype)

    cfg = SamplerConfig(
        min_steps        = args.min_steps,
        max_steps        = args.max_steps,
        cfg_scale        = args.cfg_scale,
        iterative        = True,
        freeze_threshold = 0.85,  # leggermente più permissivo per eval
        self_cond        = True,
    )

    results    = []
    n_coherent = 0
    n_on_topic = 0
    total_words = 0

    print(f"\n{'ID':<14} {'Category':<12} {'Words':>6}  {'Coh':>4}  {'OnT':>4}  {'Time':>6}")
    print("-" * 56)

    for item in EVAL_PROMPTS:
        t0 = time.time()
        try:
            response = sample(
                model        = model,
                tokenizer    = tokenizer,
                prompt       = item["prompt"],
                max_len      = args.max_len,
                cfg          = cfg,
                device       = device,
                dtype        = dtype,
                pad_token_id = pad_token_id,
                verbose      = False,
            )
        except Exception as e:
            response = f"[ERROR: {e}]"

        elapsed = time.time() - t0
        scores  = score_response(item["prompt"], response)

        if scores["coherent"]: n_coherent  += 1
        if scores["on_topic"]: n_on_topic  += 1
        total_words += scores["n_words"]

        coh = "✓" if scores["coherent"] else "✗"
        ont = "✓" if scores["on_topic"] else "✗"
        print(f"{item['id']:<14} {item['category']:<12} {scores['n_words']:>6}  "
              f"{coh:>4}  {ont:>4}  {elapsed:>5.1f}s")

        results.append({
            **item,
            "response": response,
            "scores":   scores,
            "elapsed":  round(elapsed, 2),
        })

    n_total  = len(EVAL_PROMPTS)
    avg_words = total_words / n_total

    print(f"\nCoerenza: {n_coherent}/{n_total} ({100*n_coherent//n_total}%)")
    print(f"On-topic: {n_on_topic}/{n_total} ({100*n_on_topic//n_total}%)")
    print(f"Parole medie: {avg_words:.1f}")

    # Stampa risposte complete
    print(f"\n{'─'*70}")
    print("RISPOSTE COMPLETE")
    print(f"{'─'*70}")
    for r in results:
        print(f"\n[{r['id']}] {r['category']}")
        print(f"Prompt:   {r['prompt']}")
        resp_display = r['response'][:400] + ("..." if len(r['response']) > 400 else "")
        print(f"Response: {resp_display}")

    summary = {
        "n_total":    n_total,
        "n_coherent": n_coherent,
        "n_on_topic": n_on_topic,
        "avg_words":  round(avg_words, 1),
        "pct_coherent": round(100 * n_coherent / n_total, 1),
        "pct_on_topic": round(100 * n_on_topic / n_total, 1),
    }

    # Salva JSON
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, f"step_{step:07d}.json")
    with open(json_path, "w") as f:
        json.dump({
            "step":       step,
            "checkpoint": checkpoint,
            "cfg_scale":  args.cfg_scale,
            "max_len":    args.max_len,
            "n_steps":    args.max_steps,
            "summary":    summary,
            "results":    results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nJSON salvato: {json_path}")

    # Salva Markdown
    md_content = results_to_markdown(
        results, checkpoint, step, summary, args.cfg_scale, args.max_steps,
    )
    md_path = os.path.join(out_dir, f"step_{step:07d}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Markdown salvato: {md_path}")

    # Dealloca solo se il modello è stato caricato qui (non se passato dall'esterno)
    if not _model_provided:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {"step": step, "summary": summary, "json_path": json_path, "md_path": md_path}


# ─────────────────────────────────────────────────────────────────────────────
# API per chiamata diretta da train.py (senza reload del modello)
# ─────────────────────────────────────────────────────────────────────────────

def run_eval_on_model(
    model,
    tokenizer,
    pad_token_id: int,
    step:         int,
    out_dir:      str,
    max_steps:    int = 16,
    cfg_scale:    float = 3.0,
    device:       str = "cuda",
) -> dict:
    """
    Lancia l'eval qualitativa usando il modello già in memoria.

    Chiamata da train.py dopo ogni checkpoint periodico — evita il doppio
    load in VRAM che causerebbe OOM durante il training.

    Args:
        model:        Harold già caricato in VRAM
        tokenizer:    tokenizer HuggingFace
        pad_token_id: id del padding token
        step:         step corrente (usato per nominare i file di output)
        out_dir:      directory di output per JSON e Markdown
        max_steps:    step di denoising per il sampler
        cfg_scale:    scala CFG
        device:       device string

    Returns:
        dict con summary e path dei file generati
    """
    import argparse
    # Costruisce un Namespace minimale compatibile con evaluate_checkpoint
    args = argparse.Namespace(
        device      = device,
        min_steps   = max(8, max_steps // 2),
        max_steps   = max_steps,
        cfg_scale   = cfg_scale,
        max_len     = 200,
        out_dir     = out_dir,
    )
    # Mette il modello in eval — train.py lo riporterà in train dopo
    was_training = model.training
    model.eval()
    try:
        result = evaluate_checkpoint(
            checkpoint   = f"step_{step:07d}",  # usato solo per il log
            tokenizer    = tokenizer,
            pad_token_id = pad_token_id,
            args         = args,
            step         = step,
            model        = model,               # bypass del load
        )
    finally:
        if was_training:
            model.train()
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Modalità auto: gira su tutti i checkpoint in ordine
# ─────────────────────────────────────────────────────────────────────────────

def run_auto(args: argparse.Namespace, tokenizer, pad_token_id: int) -> None:
    """
    [v0.7-E2] Modalità automatica: scansiona checkpoint_dir, gira su ognuno,
    produce un report comparativo finale.
    """
    pattern = os.path.join(args.checkpoint_dir, "*.pt")
    ckpts   = sorted(glob.glob(pattern))

    if not ckpts:
        print(f"Nessun checkpoint trovato in {args.checkpoint_dir}")
        return

    print(f"Trovati {len(ckpts)} checkpoint in {args.checkpoint_dir}")

    all_results = []
    for ckpt in ckpts:
        # Estrai step dal nome file (es. harold_v07_0010000.pt → 10000)
        basename = os.path.basename(ckpt).replace(".pt", "")
        parts    = basename.split("_")
        try:
            step = int(parts[-1])
        except ValueError:
            step = 0

        r = evaluate_checkpoint(ckpt, tokenizer, pad_token_id, args, step=step)
        all_results.append(r)

    # Report comparativo
    _write_comparison_report(all_results, args.out_dir)


def _write_comparison_report(all_results: list, out_dir: str) -> None:
    """Genera un Markdown comparativo tra tutti i checkpoint valutati."""
    if not all_results:
        return

    lines = ["# Harold v0.7 — Confronto qualitativo tra checkpoint", ""]
    lines.append(f"Generato: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")
    lines.append("| Step | Coerenza | On-topic | Parole medie |")
    lines.append("|---|---|---|---|")

    for r in sorted(all_results, key=lambda x: x["step"]):
        s = r["summary"]
        lines.append(
            f"| {r['step']:,} "
            f"| {s['n_coherent']}/{s['n_total']} ({s['pct_coherent']}%) "
            f"| {s['n_on_topic']}/{s['n_total']} ({s['pct_on_topic']}%) "
            f"| {s['avg_words']} |"
        )

    lines.append("")
    lines.append("## Note di interpretazione")
    lines.append("")
    lines.append("- **Coerenza** < 30%: il modello genera rumore — normale < 20k iter")
    lines.append("- **Coerenza** 30-60%: sintassi emergente — tipico 20-50k iter")
    lines.append("- **Coerenza** > 60%: testo leggibile — buon segnale per un diffusion LM")
    lines.append("- **Parole medie** < 5: risposte troppo corte, aumentare max_len o n_steps")

    path = os.path.join(out_dir, "comparison.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nReport comparativo salvato: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Harold v0.7 — Generation Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi:
  # Singolo checkpoint
  python eval/eval_generation.py --checkpoint /workspace/checkpoints/v0.7/harold_v07_best.pt

  # Tutti i checkpoint in una directory
  python eval/eval_generation.py --checkpoint_dir /workspace/checkpoints/v0.7 --auto

  # Con parametri custom
  python eval/eval_generation.py --checkpoint harold_v07_best.pt --cfg_scale 2.0 --max_steps 32
        """,
    )
    parser.add_argument("--checkpoint",     type=str, default=None,
                        help="Path a un singolo checkpoint .pt")
    parser.add_argument("--checkpoint_dir", type=str,
                        default="/workspace/checkpoints/v0.7",
                        help="Directory con i checkpoint (per --auto)")
    parser.add_argument("--auto",           action="store_true",
                        help="Gira su tutti i checkpoint in --checkpoint_dir")
    parser.add_argument("--out_dir",        type=str,
                        default="eval/results",
                        help="Directory di output per JSON e Markdown")
    parser.add_argument("--tokenizer",      type=str,
                        default="NousResearch/Llama-2-7b-hf")
    parser.add_argument("--max_len",        type=int,   default=200)
    parser.add_argument("--cfg_scale",      type=float, default=3.0)
    parser.add_argument("--min_steps",      type=int,   default=16)
    parser.add_argument("--max_steps",      type=int,   default=32)
    parser.add_argument("--device",         type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Carico tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_token_id = int(tokenizer.pad_token_id)

    if args.auto:
        run_auto(args, tokenizer, pad_token_id)
    elif args.checkpoint:
        # Estrai step dal nome se possibile
        basename = os.path.basename(args.checkpoint).replace(".pt", "")
        parts    = basename.split("_")
        try:
            step = int(parts[-1])
        except ValueError:
            step = 0
        evaluate_checkpoint(args.checkpoint, tokenizer, pad_token_id, args, step=step)
    else:
        print("Specifica --checkpoint o usa --auto con --checkpoint_dir")
        raise SystemExit(1)


if __name__ == "__main__":
    main()