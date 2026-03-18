"""
Harold v3 — benchmark.py
=========================
Valutazione quantitativa per il paper:

  1. Perplexity (PPL) su WikiText-103
     - Misura quanto bene il modello predice testo reale
     - Calcolata via CE loss ausiliaria sul test set
     - Comparabile con altri LM della stessa dimensione

  2. BERTScore
     - Similarità semantica tra testo generato e riferimento
     - Usa BERT embeddings per confronto token-level
     - Riportiamo F1 medio su 500 campioni

  3. MAUVE
     - Misura la qualità distribuzionale del testo generato
     - Confronta la distribuzione del testo generato con quella umana
     - Score in [0,1], più alto = meglio

Uso:
    python benchmark.py --checkpoint checkpoints_sft/harold_sft_s1_best.pt
    python benchmark.py --checkpoint checkpoints_sft/harold_sft_s1_best.pt --n_samples 200
"""

import argparse
import json
import math
import os
import sys
import time
from typing import List, Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer

# Import Harold
from config import ModelConfig
from model import Harold, build_model
from train_sft import SFTConfig  # per caricare checkpoint SFT


# ─────────────────────────────────────────────────────────────────────────────
# Caricamento modello
# ─────────────────────────────────────────────────────────────────────────────

def load_harold(ckpt_path: str, device: str) -> tuple[Harold, ModelConfig, AutoTokenizer]:
    print(f"Carico checkpoint: {ckpt_path}")
    state     = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_cfg = state["model_cfg"]
    model     = build_model(model_cfg)
    model.load_state_dict(state["model_state"], strict=False)
    model = model.eval().to(device)
    del state

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    n_params  = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Harold v3 — {n_params:.1f}M parametri caricati su {device}")
    return model, model_cfg, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# 1. Perplexity su WikiText-103
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_perplexity(
    model:     Harold,
    tokenizer: AutoTokenizer,
    device:    str,
    n_chunks:  int  = 500,
    seq_len:   int  = 256,
    t_eval:    float = 0.1,   # timestep basso = poco rumore = più vicino alla CE reale
) -> dict:
    """
    Perplexity su WikiText-103 test set.

    Approssimazione via CE loss ausiliaria a t_eval basso.
    A t→0: x_t ≈ x0 → la CE loss è una buona proxy della PPL.

    PPL = exp(CE_loss)
    """
    print("\n" + "="*60)
    print("1. PERPLEXITY su WikiText-103")
    print("="*60)

    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")

    # Concatena tutto il testo e tokenizza
    text = " ".join(doc for doc in ds["text"] if doc.strip())
    ids  = tokenizer(
        text, return_tensors="pt", truncation=False,
        add_special_tokens=False,
    )["input_ids"][0] # type: ignore

    print(f"WikiText-103 test: {len(ids):,} token totali")
    print(f"Valuto su {n_chunks} chunk di {seq_len} token a t={t_eval}")

    total_ce = 0.0
    total_n  = 0
    t_tensor = torch.full((1,), t_eval, dtype=torch.float32, device=device)

    # Campiona chunk casuali dal test set
    torch.manual_seed(42)
    max_start = len(ids) - seq_len - 1
    starts    = torch.randint(0, max_start, (n_chunks,))

    for start in tqdm(starts, desc="PPL chunks"):
        chunk = ids[start:start + seq_len].unsqueeze(0).to(device)  # (1, L)
        mask  = (chunk != 0)

        if mask.sum() == 0:
            continue

        # Forward process a t_eval
        with torch.no_grad():
            x0_emb = model.token_emb(chunk)                         # (1, L, D)
            x_t, _ = model.schedule.add_noise(x0_emb, t_tensor)

            _, ce_logits, _ = model(x_t, t_tensor)
            ce = F.cross_entropy(
                ce_logits[0][mask[0]],
                chunk[0][mask[0]],
                reduction="mean",
            )

        total_ce += ce.item() * mask.sum().item()
        total_n  += mask.sum().item()

    avg_ce = total_ce / max(total_n, 1)
    ppl    = math.exp(avg_ce)

    result = {
        "perplexity": round(ppl, 2),
        "ce_loss":    round(avg_ce, 4),
        "n_tokens":   total_n,
        "t_eval":     t_eval,
    }
    print(f"\nRisultati PPL:")
    print(f"  CE loss medio:  {avg_ce:.4f}")
    print(f"  Perplexity:     {ppl:.2f}")
    print(f"  Token valutati: {total_n:,}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 2. Generazione testi per MAUVE e BERTScore
# ─────────────────────────────────────────────────────────────────────────────

def generate_samples(
    model:     Harold,
    tokenizer: AutoTokenizer,
    device:    str,
    prompts:   List[str],
    gen_len:   int   = 128,
    steps:     int   = 50,
    cfg_scale: float = 3.0,
) -> List[str]:
    """
    Genera testi condizionati sui prompt usando CFG.
    Usa la stessa logica di HaroldSampler.generate_conditioned().
    """
    import re
    d_model = model.config.d_model

    # Build vocab mask (allowlist ASCII)
    vocab   = tokenizer.get_vocab() # type: ignore
    V       = model.emb_vocab
    allowed = re.compile(r"^([a-z]+|[0-9]+|[.,!?:;'\"\-\(\)\[\]/@#&\*\+=%<>\\]+|##[a-z0-9]+)$")
    vmask   = torch.zeros(V, dtype=torch.bool, device=device)
    for tok, idx in vocab.items():
        if idx < V and not allowed.match(tok.lower()):
            vmask[idx] = True

    generated = []

    for prompt in tqdm(prompts, desc="Generazione campioni"):
        # Encoda contesto
        ctx_ids  = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)["input_ids"].to(device) # type: ignore
        pad_mask = (ctx_ids != 0).float()
        emb      = model.token_emb(ctx_ids)
        n        = pad_mask.sum(dim=1, keepdim=True).clamp(min=1)
        ctx_emb  = (emb * pad_mask.unsqueeze(-1)).sum(dim=1) / n  # (1, D)

        x_t       = torch.randn(1, gen_len, d_model, device=device)
        self_cond = None
        dt        = -1.0 / steps

        with torch.no_grad():
            for step in range(steps):
                t        = 1.0 - step / steps
                t_tensor = torch.full((1,), t, dtype=torch.float32, device=device)

                eps_cond,   ce_logits, _ = model(x_t, t_tensor, self_cond=self_cond, ctx_emb=ctx_emb)
                eps_uncond, _,          _ = model(x_t, t_tensor, self_cond=self_cond, ctx_emb=None)
                eps = eps_uncond + cfg_scale * (eps_cond - eps_uncond)

                beta_t     = model.schedule.get_beta(t_tensor)
                _, sigma_t = model.schedule.get_alpha_sigma(t_tensor)
                score      = -eps / sigma_t.view(-1, 1, 1).clamp(min=1e-8)
                b          = beta_t.view(-1, 1, 1)
                noise      = torch.randn_like(x_t) if t > 1e-5 else torch.zeros_like(x_t)
                x_t        = x_t + (-0.5*b*x_t - b*score)*dt + b.sqrt()*noise*math.sqrt(abs(dt))

                # Self-cond da token puliti
                ce_clean = ce_logits.clone()
                ce_clean[..., vmask] = float("-inf")
                clean_ids = ce_clean.argmax(dim=-1)
                self_cond = model.token_emb(clean_ids).mean(dim=1).detach()

            # Decodifica finale
            t_final = torch.full((1,), 1.0/steps, dtype=torch.float32, device=device)
            _, ce_final, _ = model(x_t, t_final, self_cond=self_cond, ctx_emb=ctx_emb)
            ce_final[..., vmask] = float("-inf")
            token_ids = ce_final.argmax(dim=-1)[0]

        text = tokenizer.decode(token_ids.tolist(), skip_special_tokens=True) # type: ignore
        generated.append(text)

    return generated


# ─────────────────────────────────────────────────────────────────────────────
# 3. MAUVE
# ─────────────────────────────────────────────────────────────────────────────

def compute_mauve(
    generated:  List[str],
    references: List[str],
) -> dict:
    """
    MAUVE score — qualità distribuzionale del testo generato.
    Confronta p(generated) con p(references) nello spazio delle feature.
    Score in [0,1], più alto = meglio.
    """
    print("\n" + "="*60)
    print("2. MAUVE")
    print("="*60)

    try:
        import mauve # type: ignore
    except ImportError:
        print("Installo mauve-text...")
        os.system("pip install mauve-text --break-system-packages -q")
        import mauve # type: ignore

    print(f"Calcolo MAUVE su {len(generated)} campioni...")

    # Filtra testi vuoti
    pairs = [(g, r) for g, r in zip(generated, references) if g.strip() and r.strip()]
    gen_texts = [p[0] for p in pairs]
    ref_texts = [p[1] for p in pairs]

    if len(gen_texts) < 10:
        print("Troppo pochi campioni validi per MAUVE.")
        return {"mauve": None, "error": "insufficient_samples"}

    result = mauve.compute_mauve(
        p_text=gen_texts,
        q_text=ref_texts,
        device_id=0 if torch.cuda.is_available() else -1,
        max_text_length=256,
        verbose=False,
        batch_size=8,
    )

    score = round(float(result.mauve), 4)
    print(f"\nRisultati MAUVE:")
    print(f"  MAUVE score: {score:.4f}")
    return {"mauve": score, "n_pairs": len(gen_texts)}


# ─────────────────────────────────────────────────────────────────────────────
# 4. BERTScore
# ─────────────────────────────────────────────────────────────────────────────

def compute_bertscore(
    generated:  List[str],
    references: List[str],
) -> dict:
    """
    BERTScore F1 — similarità semantica token-level via BERT.
    Riportiamo la media di P, R, F1 su tutti i campioni.
    """
    print("\n" + "="*60)
    print("3. BERTScore")
    print("="*60)

    try:
        from bert_score import score as bert_score # type: ignore
    except ImportError: 
        print("Installo bert-score...")
        os.system("pip install bert-score --break-system-packages -q")
        from bert_score import score as bert_score # type: ignore

    # Filtra testi vuoti
    pairs = [(g, r) for g, r in zip(generated, references) if g.strip() and r.strip()]
    gen_texts = [p[0] for p in pairs]
    ref_texts = [p[1] for p in pairs]

    if len(gen_texts) < 5:
        print("Troppo pochi campioni validi per BERTScore.")
        return {"bertscore_f1": None, "error": "insufficient_samples"}

    print(f"Calcolo BERTScore su {len(gen_texts)} campioni...")

    P, R, F1 = bert_score(
        gen_texts, ref_texts,
        lang="en",
        model_type="bert-base-uncased",  # coerente col modello
        device="cuda" if torch.cuda.is_available() else "cpu",
        batch_size=16,
        verbose=False,
    )

    result = {
        "bertscore_precision": round(float(P.mean()), 4),
        "bertscore_recall":    round(float(R.mean()), 4),
        "bertscore_f1":        round(float(F1.mean()), 4),
        "n_pairs": len(gen_texts),
    }
    print(f"\nRisultati BERTScore:")
    print(f"  Precision: {result['bertscore_precision']:.4f}")
    print(f"  Recall:    {result['bertscore_recall']:.4f}")
    print(f"  F1:        {result['bertscore_f1']:.4f}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Prompt e reference per generazione
# ─────────────────────────────────────────────────────────────────────────────

def load_prompts_and_references(
    tokenizer: AutoTokenizer,
    n_samples: int = 300,
) -> tuple[List[str], List[str]]:
    """
    Carica prompt e riferimenti da WikiText-103.
    Usa le prime n_samples frasi come prompt e le successive come riferimento.
    Questo permette di valutare quanto la generazione è coerente
    con il tipo di testo del corpus.
    """
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")

    sentences = []
    for doc in ds["text"]:
        doc = doc.strip()
        if len(doc) > 50:
            # Divide in frasi approssimative
            parts = [s.strip() for s in doc.split(". ") if len(s.strip()) > 30]
            sentences.extend(parts)

    # Prompt = prima parte della frase, reference = frase completa
    prompts    = []
    references = []
    for sent in sentences[:n_samples * 2]:
        words = sent.split()
        if len(words) < 10:
            continue
        mid     = len(words) // 3
        prompt  = " ".join(words[:mid])
        ref     = " ".join(words[mid:])
        if prompt and ref:
            prompts.append(prompt)
            references.append(ref)
        if len(prompts) >= n_samples:
            break

    print(f"Caricati {len(prompts)} prompt/reference da WikiText-103 test")
    return prompts, references


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Harold v3 — Benchmark")
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints_sft/harold_sft_s1_best.pt",
                        help="Percorso del checkpoint da valutare")
    parser.add_argument("--n_samples", type=int, default=300,
                        help="Numero di campioni per MAUVE e BERTScore")
    parser.add_argument("--ppl_chunks", type=int, default=500,
                        help="Numero di chunk per la perplexity")
    parser.add_argument("--gen_steps", type=int, default=50,
                        help="Passi di denoising per la generazione")
    parser.add_argument("--cfg_scale", type=float, default=3.0,
                        help="Scala CFG per la generazione condizionata")
    parser.add_argument("--output", type=str, default="benchmark_results.json",
                        help="File di output per i risultati")
    parser.add_argument("--skip_ppl", action="store_true",
                        help="Salta il calcolo della perplexity")
    parser.add_argument("--skip_gen", action="store_true",
                        help="Salta la generazione (usa risultati precedenti)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Campioni: {args.n_samples}")

    # Carica modello
    model, model_cfg, tokenizer = load_harold(args.checkpoint, device)

    results = {
        "checkpoint": args.checkpoint,
        "timestamp":  time.strftime("%Y-%m-%d %H:%M:%S"),
        "device":     device,
        "n_samples":  args.n_samples,
        "cfg_scale":  args.cfg_scale,
    }

    # ── 1. Perplexity ────────────────────────────────────────────────────────
    if not args.skip_ppl:
        ppl_results = compute_perplexity(
            model, tokenizer, device,
            n_chunks=args.ppl_chunks,
        )
        results["perplexity"] = ppl_results
    else:
        print("\nPerplexity: saltata (--skip_ppl)")

    # ── Generazione campioni ─────────────────────────────────────────────────
    gen_cache_path = args.output.replace(".json", "_generated.json")

    if not args.skip_gen:
        print(f"\n{'='*60}")
        print("GENERAZIONE CAMPIONI")
        print(f"{'='*60}")

        prompts, references = load_prompts_and_references(
            tokenizer, n_samples=args.n_samples
        )

        generated = generate_samples(
            model, tokenizer, device,
            prompts=prompts,
            gen_len=128,
            steps=args.gen_steps,
            cfg_scale=args.cfg_scale,
        )

        # Salva campioni generati per debug/ispezione
        with open(gen_cache_path, "w") as f:
            json.dump({
                "prompts":    prompts,
                "references": references,
                "generated":  generated,
            }, f, indent=2)
        print(f"\nCampioni salvati in: {gen_cache_path}")

        # Mostra alcuni esempi
        print("\nEsempi di generazione:")
        for i in range(min(3, len(generated))):
            print(f"\n  Prompt:    {prompts[i][:80]}...")
            print(f"  Reference: {references[i][:80]}...")
            print(f"  Generated: {generated[i][:80]}...")

    else:
        print(f"\nCarico campioni precedenti da: {gen_cache_path}")
        if not os.path.exists(gen_cache_path):
            print("File non trovato. Rimuovi --skip_gen per generare.")
            sys.exit(1)
        with open(gen_cache_path) as f:
            cache = json.load(f)
        prompts    = cache["prompts"]
        references = cache["references"]
        generated  = cache["generated"]

    # ── 2. MAUVE ─────────────────────────────────────────────────────────────
    mauve_results = compute_mauve(generated, references)
    results["mauve"] = mauve_results

    # ── 3. BERTScore ─────────────────────────────────────────────────────────
    bert_results = compute_bertscore(generated, references)
    results["bertscore"] = bert_results

    # ── Salva risultati finali ────────────────────────────────────────────────
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    # ── Riepilogo ─────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("RIEPILOGO BENCHMARK HAROLD v3")
    print("="*60)
    if "perplexity" in results:
        print(f"  Perplexity (WikiText-103):  {results['perplexity']['perplexity']:.2f}")
    if "mauve" in results and results["mauve"].get("mauve"):
        print(f"  MAUVE score:                {results['mauve']['mauve']:.4f}")
    if "bertscore" in results and results["bertscore"].get("bertscore_f1"):
        print(f"  BERTScore F1:               {results['bertscore']['bertscore_f1']:.4f}")
    print(f"\nRisultati completi salvati in: {args.output}")
    print("="*60)


if __name__ == "__main__":
    main()