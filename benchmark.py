"""
Harold v0.4 — benchmark.py
===========================
Valutazione quantitativa per il paper:

  1. Perplexity (PPL) su WikiText-103
     Calcolata via CE loss ausiliaria a t basso.

  2. BERTScore
     Similarità semantica tra testo generato e riferimento.
     Riportiamo F1 medio su n_samples campioni.

  3. MAUVE
     Qualità distribuzionale del testo generato.
     Score in [0,1], più alto = meglio.

Uso:
    python benchmark.py --checkpoint checkpoints_v4/harold_v04_best.pt
    python benchmark.py --checkpoint checkpoints_v4/harold_v04_best.pt --n_samples 200
"""

import argparse
import json
import math
import os
import sys
import time
from typing import List, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer

from config import ModelConfig
from model import Harold, build_model
from sampler import HaroldSampler


def load_harold(
    ckpt_path: str,
    device:    str,
) -> Tuple[Harold, ModelConfig, PreTrainedTokenizer]:
    print(f"Carico checkpoint: {ckpt_path}")
    state     = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_cfg = state["model_cfg"]
    model     = build_model(model_cfg)
    model.load_state_dict(state["model_state"], strict=False)
    model     = model.eval().to(device)
    del state

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Harold v0.4 — {n_params:.1f}M parametri su {device}")
    return model, model_cfg, tokenizer


@torch.no_grad()
def compute_perplexity(
    model:     Harold,
    tokenizer: PreTrainedTokenizer,
    device:    str,
    n_chunks:  int   = 500,
    seq_len:   int   = 1024,
    t_eval:    float = 0.1,
) -> dict:
    """
    Perplexity su WikiText-103 test set.
    Approssimazione via CE loss ausiliaria a t_eval basso.
    PPL = exp(CE_loss)
    """
    print("\n1. PERPLEXITY su WikiText-103")

    from datasets import load_dataset
    ds   = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    text = " ".join(doc for doc in ds["text"] if doc.strip())

    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=False,
        add_special_tokens=False,
    )
    ids = enc.input_ids[0]

    _pad = tokenizer.pad_token_id
    pad_token_id: int = _pad if isinstance(_pad, int) else 0

    print(f"WikiText-103 test: {len(ids):,} token")
    print(f"Valuto su {n_chunks} chunk di {seq_len} token a t={t_eval}")

    total_ce = 0.0
    total_n  = 0
    t_tensor = torch.full((1,), t_eval, dtype=torch.float32, device=device)

    torch.manual_seed(42)
    max_start = max(len(ids) - seq_len - 1, 1)
    starts    = torch.randint(0, max_start, (n_chunks,))

    for start in tqdm(starts, desc="PPL"):
        chunk = ids[start : start + seq_len].unsqueeze(0).to(device)
        mask  = (chunk != pad_token_id)

        if mask.sum() == 0:
            continue

        x0_emb     = model.token_emb(chunk)
        x_t, _     = model.schedule.add_noise(x0_emb, t_tensor)
        _, ce_logits, _ = model(x_t, t_tensor)

        ce = F.cross_entropy(
            ce_logits[0][mask[0]],
            chunk[0][mask[0]],
            reduction="mean",
        )

        total_ce += ce.item() * int(mask.sum().item())
        total_n  += int(mask.sum().item())

    avg_ce = total_ce / max(total_n, 1)
    ppl    = math.exp(avg_ce)

    result = {
        "perplexity": round(ppl, 2),
        "ce_loss":    round(avg_ce, 4),
        "n_tokens":   total_n,
        "t_eval":     t_eval,
    }
    print(f"CE loss: {avg_ce:.4f}  |  PPL: {ppl:.2f}  |  Token: {total_n:,}")
    return result


def generate_samples(
    model:     Harold,
    tokenizer: PreTrainedTokenizer,
    device:    str,
    prompts:   List[str],
    gen_len:   int   = 128,
    steps:     int   = 20,
    cfg_scale: float = 3.0,
) -> List[str]:
    """
    Genera testi condizionati sui prompt usando HaroldSampler.
    """
    sampler   = HaroldSampler(model, tokenizer, device=device)
    generated = []

    for prompt in tqdm(prompts, desc="Generazione"):
        text = sampler.generate_conditioned(
            context=prompt,
            gen_len=gen_len,
            steps=steps,
            cfg_scale=cfg_scale,
            mode="argmax",
            use_self_cond=True,
        )
        generated.append(text)

    return generated


def compute_mauve(
    generated:  List[str],
    references: List[str],
) -> dict:
    print("\n2. MAUVE")

    try:
        import mauve
    except ImportError:
        print("Installo mauve-text...")
        os.system("pip install mauve-text --break-system-packages -q")
        import mauve

    pairs     = [(g, r) for g, r in zip(generated, references) if g.strip() and r.strip()]
    gen_texts = [p[0] for p in pairs]
    ref_texts = [p[1] for p in pairs]

    if len(gen_texts) < 10:
        print("Troppo pochi campioni validi per MAUVE.")
        return {"mauve": None, "error": "insufficient_samples"}

    print(f"Calcolo MAUVE su {len(gen_texts)} campioni...")

    result = mauve.compute_mauve(
        p_text=gen_texts,
        q_text=ref_texts,
        device_id=0 if torch.cuda.is_available() else -1,
        max_text_length=256,
        verbose=False,
        batch_size=8,
    )

    score = round(float(result.mauve), 4)
    print(f"MAUVE: {score:.4f}")
    return {"mauve": score, "n_pairs": len(gen_texts)}


def compute_bertscore(
    generated:  List[str],
    references: List[str],
) -> dict:
    print("\n3. BERTScore")

    try:
        from bert_score import score as bert_score
    except ImportError:
        print("Installo bert-score...")
        os.system("pip install bert-score --break-system-packages -q")
        from bert_score import score as bert_score

    pairs     = [(g, r) for g, r in zip(generated, references) if g.strip() and r.strip()]
    gen_texts = [p[0] for p in pairs]
    ref_texts = [p[1] for p in pairs]

    if len(gen_texts) < 5:
        print("Troppo pochi campioni validi per BERTScore.")
        return {"bertscore_f1": None, "error": "insufficient_samples"}

    print(f"Calcolo BERTScore su {len(gen_texts)} campioni...")

    P, R, F1 = bert_score(
        gen_texts,
        ref_texts,
        lang="en",
        model_type="bert-base-uncased",
        device="cuda" if torch.cuda.is_available() else "cpu",
        batch_size=16,
        verbose=False,
    )

    result = {
        "bertscore_precision": round(float(P.mean()), 4),
        "bertscore_recall":    round(float(R.mean()), 4),
        "bertscore_f1":        round(float(F1.mean()), 4),
        "n_pairs":             len(gen_texts),
    }
    print(
        f"P: {result['bertscore_precision']:.4f}  "
        f"R: {result['bertscore_recall']:.4f}  "
        f"F1: {result['bertscore_f1']:.4f}"
    )
    return result


def load_prompts_and_references(
    n_samples: int = 300,
) -> Tuple[List[str], List[str]]:
    """
    Carica prompt e riferimenti da WikiText-103 test set.
    Prompt = primo terzo della frase, reference = resto.
    """
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")

    prompts    = []
    references = []

    for doc in ds["text"]:
        doc = doc.strip()
        if len(doc) < 50:
            continue
        parts = [s.strip() for s in doc.split(". ") if len(s.strip()) > 30]
        for sent in parts:
            words = sent.split()
            if len(words) < 10:
                continue
            mid = len(words) // 3
            prompts.append(" ".join(words[:mid]))
            references.append(" ".join(words[mid:]))
            if len(prompts) >= n_samples:
                break
        if len(prompts) >= n_samples:
            break

    print(f"Caricati {len(prompts)} prompt/reference da WikiText-103")
    return prompts, references


def main() -> None:
    parser = argparse.ArgumentParser(description="Harold v0.4 — Benchmark")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints_v4/harold_v04_best.pt",
    )
    parser.add_argument("--n_samples",  type=int,   default=300)
    parser.add_argument("--ppl_chunks", type=int,   default=500)
    parser.add_argument("--gen_steps",  type=int,   default=20)
    parser.add_argument("--cfg_scale",  type=float, default=3.0)
    parser.add_argument("--output",     type=str,   default="benchmark_results.json")
    parser.add_argument("--skip_ppl",   action="store_true")
    parser.add_argument("--skip_gen",   action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  |  Checkpoint: {args.checkpoint}")

    model, model_cfg, tokenizer = load_harold(args.checkpoint, device)

    results: dict = {
        "checkpoint": args.checkpoint,
        "timestamp":  time.strftime("%Y-%m-%d %H:%M:%S"),
        "device":     device,
        "n_samples":  args.n_samples,
        "cfg_scale":  args.cfg_scale,
    }

    if not args.skip_ppl:
        results["perplexity"] = compute_perplexity(
            model, tokenizer, device,
            n_chunks=args.ppl_chunks,
        )
    else:
        print("Perplexity: saltata (--skip_ppl)")

    gen_cache = args.output.replace(".json", "_generated.json")

    if not args.skip_gen:
        prompts, references = load_prompts_and_references(args.n_samples)
        generated = generate_samples(
            model, tokenizer, device,
            prompts=prompts,
            gen_len=128,
            steps=args.gen_steps,
            cfg_scale=args.cfg_scale,
        )
        with open(gen_cache, "w") as f:
            json.dump(
                {"prompts": prompts, "references": references, "generated": generated},
                f, indent=2,
            )
        print(f"Campioni salvati in: {gen_cache}")
        print("\nEsempi:")
        for i in range(min(3, len(generated))):
            print(f"  Prompt:    {prompts[i][:80]}...")
            print(f"  Reference: {references[i][:80]}...")
            print(f"  Generated: {generated[i][:80]}...")
    else:
        if not os.path.exists(gen_cache):
            print(f"File non trovato: {gen_cache}. Rimuovi --skip_gen.")
            sys.exit(1)
        with open(gen_cache) as f:
            cache = json.load(f)
        prompts    = cache["prompts"]
        references = cache["references"]
        generated  = cache["generated"]
        print(f"Campioni caricati da: {gen_cache}")

    results["mauve"]     = compute_mauve(generated, references)
    results["bertscore"] = compute_bertscore(generated, references)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print("\nRIEPILOGO BENCHMARK HAROLD v0.4")
    if "perplexity" in results:
        print(f"  PPL (WikiText-103): {results['perplexity']['perplexity']:.2f}")
    if results.get("mauve", {}).get("mauve"):
        print(f"  MAUVE:              {results['mauve']['mauve']:.4f}")
    if results.get("bertscore", {}).get("bertscore_f1"):
        print(f"  BERTScore F1:       {results['bertscore']['bertscore_f1']:.4f}")
    print(f"Risultati salvati in: {args.output}")


if __name__ == "__main__":
    main()