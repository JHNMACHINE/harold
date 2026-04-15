"""
eval_generation.py — Harold v0.6
==================================
Campiona risposte a un set standard di prompt per valutazione qualitativa.
Produce output formattato per il paper (tabella LaTeX con esempi).

Avvio:
  python eval/eval_generation.py
  python eval/eval_generation.py --checkpoint harold-v0.6-1B-sft.safetensors
"""

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Compatibilità con import legacy in model.py
import sys as _sys
from core import config as _core_config
from core import model as _core_model
_sys.modules.setdefault("config", _core_config)
_sys.modules.setdefault("model", _core_model)

from core.sampler import load_model, sample
from transformers import AutoTokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Prompt standard per valutazione
# ─────────────────────────────────────────────────────────────────────────────

EVAL_PROMPTS = [
    # Factual
    {"id": "factual_1",   "category": "Factual",   "prompt": "What is the capital of France?"},
    {"id": "factual_2",   "category": "Factual",   "prompt": "Who wrote Romeo and Juliet?"},
    {"id": "factual_3",   "category": "Factual",   "prompt": "What is the speed of light?"},

    # Reasoning
    {"id": "reasoning_1", "category": "Reasoning", "prompt": "If a train travels 60 miles per hour for 2 hours, how far does it travel?"},
    {"id": "reasoning_2", "category": "Reasoning", "prompt": "What comes next in the sequence: 2, 4, 8, 16, ...?"},

    # Instruction following
    {"id": "instruct_1",  "category": "Instruction", "prompt": "List three benefits of regular exercise."},
    {"id": "instruct_2",  "category": "Instruction", "prompt": "Explain machine learning in simple terms."},
    {"id": "instruct_3",  "category": "Instruction", "prompt": "Write a short poem about the ocean."},

    # Coding
    {"id": "code_1",      "category": "Coding",    "prompt": "Write a Python function that returns the factorial of a number."},
    {"id": "code_2",      "category": "Coding",    "prompt": "What is a dictionary in Python?"},
]


# ─────────────────────────────────────────────────────────────────────────────
# Valutazione qualitativa semplice
# ─────────────────────────────────────────────────────────────────────────────

def score_response(prompt: str, response: str) -> dict:
    """
    Valutazione euristica della risposta:
    - coherent: la risposta contiene parole reali (non solo rumore)
    - on_topic: la risposta sembra rispondere al prompt
    - length:   lunghezza in caratteri
    """
    words = [w for w in response.split() if len(w) > 1 and w.isalpha()]
    n_words = len(words)
    coherent = n_words > 5 and len(set(words)) > 3

    # Keyword matching basilare
    prompt_words = set(prompt.lower().split())
    resp_words   = set(response.lower().split())
    overlap      = len(prompt_words & resp_words)
    on_topic     = overlap >= 1 or n_words > 10

    return {
        "coherent":  coherent,
        "on_topic":  on_topic,
        "n_words":   n_words,
        "n_chars":   len(response),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    device = args.device
    dtype  = torch.bfloat16 if device.startswith("cuda") else torch.float32

    print("=" * 70)
    print("Harold v0.6 — Generation Evaluation")
    print("=" * 70)

    # Carica tokenizer
    print(f"\nCarico tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_token_id = int(tokenizer.pad_token_id)

    # Carica modello
    print(f"Carico modello: {args.checkpoint}")
    model = load_model(args.checkpoint, device=device)
    model = model.to(dtype)

    results = []
    n_coherent = 0
    n_on_topic = 0

    print(f"\n{'ID':<12} {'Category':<12} {'Words':>6}  {'Coh':>4}  {'OnT':>4}")
    print("-" * 50)

    for item in EVAL_PROMPTS:
        t0 = time.time()
        try:
            response = sample(
                model        = model,
                tokenizer    = tokenizer,
                prompt       = item["prompt"],
                max_len      = args.max_len,
                cfg_scale    = args.cfg_scale,
                min_steps    = args.min_steps,
                max_steps    = args.max_steps,
                device       = device,
                dtype        = dtype,
                pad_token_id = pad_token_id,
                verbose      = False,
            )
        except Exception as e:
            response = f"[ERROR: {e}]"

        elapsed = time.time() - t0
        scores  = score_response(item["prompt"], response)

        if scores["coherent"]:
            n_coherent += 1
        if scores["on_topic"]:
            n_on_topic += 1

        print(f"{item['id']:<12} {item['category']:<12} {scores['n_words']:>6}  "
              f"{'✓' if scores['coherent'] else '✗':>4}  "
              f"{'✓' if scores['on_topic'] else '✗':>4}  "
              f"({elapsed:.1f}s)")

        results.append({
            **item,
            "response": response,
            "scores":   scores,
            "elapsed":  elapsed,
        })

    n_total = len(EVAL_PROMPTS)
    print(f"\nCoherence: {n_coherent}/{n_total} ({100*n_coherent/n_total:.0f}%)")
    print(f"On-topic:  {n_on_topic}/{n_total} ({100*n_on_topic/n_total:.0f}%)")

    # Output dettagliato
    print("\n\n--- Responses ---")
    for r in results:
        print(f"\n[{r['id']}] {r['category']}")
        print(f"Prompt:   {r['prompt']}")
        print(f"Response: {r['response'][:200]}{'...' if len(r['response']) > 200 else ''}")

    # LaTeX per il paper
    print("\n\n--- LaTeX Examples (selected) ---")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\small")
    print("\\begin{tabular}{p{4cm}p{8cm}}")
    print("\\toprule")
    print("Prompt & Response \\\\")
    print("\\midrule")
    for r in results[:4]:  # primi 4 esempi
        prompt_esc   = r["prompt"].replace("_", "\\_").replace("&", "\\&")
        response_esc = r["response"][:120].replace("_", "\\_").replace("&", "\\&")
        if len(r["response"]) > 120:
            response_esc += "..."
        print(f"{prompt_esc} & {response_esc} \\\\")
        print("\\midrule")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\caption{Harold v0.6 generation examples (SFT checkpoint, 8-20 denoising steps).}")
    print("\\end{table}")

    # Salva JSON
    out_path = "eval/results_generation.json"
    os.makedirs("eval", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "checkpoint": args.checkpoint,
            "cfg_scale":  args.cfg_scale,
            "max_len":    args.max_len,
            "summary": {
                "n_total":    n_total,
                "n_coherent": n_coherent,
                "n_on_topic": n_on_topic,
            },
            "results": results,
        }, f, indent=2)
    print(f"\nRisultati salvati in {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Harold v0.6 — Generation Evaluation")
    parser.add_argument("--checkpoint", type=str, default="harold-v0.6-1B-sft.safetensors")
    parser.add_argument("--tokenizer",  type=str, default="NousResearch/Llama-2-7b-hf")
    parser.add_argument("--max_len",    type=int,   default=128)
    parser.add_argument("--cfg_scale",  type=float, default=3.0)
    parser.add_argument("--min_steps",  type=int,   default=8)
    parser.add_argument("--max_steps",  type=int,   default=20)
    parser.add_argument("--device",     type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())