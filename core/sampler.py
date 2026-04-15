"""
Harold v0.6 — sampler.py
=========================
Inferenza con Flow Matching + CFG + self-conditioning.

Pipeline:
  1. Tokenizza il prompt
  2. Calcola ctx_emb (mean pooling degli embedding del prompt)
  3. Denoisa x_T ~ N(0, I) iterativamente da t=1 a t=0
     - Self-conditioning: forward senza cond → usa vel come self_cond
     - CFG: vel_guided = vel_uncond + cfg_scale * (vel_cond - vel_uncond)
     - Passi dinamici: 8-20 step in base alla complessità stimata
  4. Decodifica i token con nearest-neighbor sugli embedding

Avvio:
  python sampler.py --prompt "Spiega cosa è la diffusione"
  python sampler.py --prompt "..." --steps 16 --cfg_scale 3.0 --checkpoint path/to/ckpt.pt
"""

import argparse
import math
import os
import sys
import time
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

# Aggiungi la root del progetto al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.config import SFTConfig
from core.model import Harold, build_model


# ─────────────────────────────────────────────────────────────────────────────
# Costanti
# ─────────────────────────────────────────────────────────────────────────────

MIN_STEPS        = 8
MAX_STEPS        = 20
PROBE_STEPS      = 4       # step fissi prima di stimare la complessità
VAR_THRESHOLD_LO = 0.05    # sotto → semplice → 8 step
VAR_THRESHOLD_HI = 0.15    # sopra → complesso → 20 step


# ─────────────────────────────────────────────────────────────────────────────
# Caricamento modello
# ─────────────────────────────────────────────────────────────────────────────

def load_model(
    checkpoint_path: str,
    device:          str = "cuda",
) -> tuple[Harold, dict]:
    """
    Carica il modello dal checkpoint (pretraining o SFT).
    Ritorna (model, model_cfg).
    """
    print(f"Carico checkpoint: {checkpoint_path}")
    state     = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_cfg = state["model_cfg"]
    model     = build_model(model_cfg).to(device)
    model.load_state_dict(state["model_state"], strict=False)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Harold v0.6 — {n_params/1000:.2f}B parametri" if n_params >= 1000
          else f"Harold v0.6 — {n_params:.1f}M parametri")
    print(f"Device: {device}  |  dtype: {next(model.parameters()).dtype}")
    return model, model_cfg


# ─────────────────────────────────────────────────────────────────────────────
# Encode context (prompt → ctx_emb)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def encode_context(
    model:        Harold,
    prompt_ids:   torch.Tensor,
    pad_token_id: int,
) -> torch.Tensor:
    """
    Mean pooling degli embedding del prompt → ctx_emb (1, D).
    Identico a encode_context in train_sft.py.
    """
    pad_mask = (prompt_ids != pad_token_id).float()
    emb      = model.token_emb(prompt_ids)
    n_tokens = pad_mask.sum(dim=1, keepdim=True).clamp(min=1)
    return (emb * pad_mask.unsqueeze(-1)).sum(dim=1) / n_tokens


# ─────────────────────────────────────────────────────────────────────────────
# Stima complessità
# ─────────────────────────────────────────────────────────────────────────────

def estimate_complexity(vel_history: list[torch.Tensor]) -> int:
    """
    Stima il numero ottimale di step in base alla varianza della velocità
    osservata durante i primi PROBE_STEPS step.

    vel_history: lista di tensori (1, L, D) — uno per step probe
    Ritorna: numero di step totali (MIN_STEPS..MAX_STEPS)
    """
    if not vel_history:
        return MIN_STEPS

    # Varianza media sulla dimensione del modello
    stacked = torch.stack([v.float() for v in vel_history], dim=0)  # (P, 1, L, D)
    var     = stacked.var(dim=0).mean().item()

    if var < VAR_THRESHOLD_LO:
        n_steps = MIN_STEPS
    elif var > VAR_THRESHOLD_HI:
        n_steps = MAX_STEPS
    else:
        # Interpolazione lineare
        ratio   = (var - VAR_THRESHOLD_LO) / (VAR_THRESHOLD_HI - VAR_THRESHOLD_LO)
        n_steps = int(MIN_STEPS + ratio * (MAX_STEPS - MIN_STEPS))

    return n_steps


# ─────────────────────────────────────────────────────────────────────────────
# Sampler principale
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def sample(
    model:        Harold,
    tokenizer,
    prompt:       str,
    max_len:      int   = 128,
    cfg_scale:    float = 3.0,
    min_steps:    int   = MIN_STEPS,
    max_steps:    int   = MAX_STEPS,
    device:       str   = "cuda",
    dtype:        torch.dtype = torch.bfloat16,
    pad_token_id: int   = 0,
    verbose:      bool  = True,
) -> str:
    """
    Genera una risposta al prompt usando Flow Matching con:
    - CFG (Classifier-Free Guidance)
    - Self-conditioning
    - Numero dinamico di step di denoising

    Args:
        prompt:    testo del prompt
        max_len:   lunghezza massima della risposta in token
        cfg_scale: scala CFG (1.0 = no guidance, 3.0 = default)
        min_steps: step minimi di denoising
        max_steps: step massimi di denoising
    """
    t_start = time.time()

    # ── 1. Tokenizza il prompt ────────────────────────────────────────────────
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        add_special_tokens=False,
    )
    prompt_ids = enc["input_ids"].to(device)

    # ── 2. Calcola ctx_emb ────────────────────────────────────────────────────
    ctx_emb  = encode_context(model, prompt_ids, pad_token_id).to(dtype)
    null_emb = torch.zeros_like(ctx_emb)  # unconditional

    # ── 3. Inizializza x_T ~ N(0, I) nello spazio degli embedding ────────────
    d_model = model.d_model
    x_t     = torch.randn(1, max_len, d_model, device=device, dtype=dtype)

    # ── 4. Denoising con step dinamici ────────────────────────────────────────
    vel_history: list[torch.Tensor] = []
    n_steps     = min_steps          # verrà aggiornato dopo PROBE_STEPS
    self_cond:  Optional[torch.Tensor] = None
    step        = 0

    if verbose:
        print(f"Denoising — step iniziali: {n_steps} (adattativi)")

    while step < n_steps:
        t_val = 1.0 - step / n_steps               # da 1 a ~0
        dt    = 1.0 / n_steps
        t     = torch.full((1,), t_val, device=device, dtype=torch.float32)

        # Self-conditioning: forward senza cond → usa vel come self_cond
        if self_cond is None and step > 0:
            vel_sc, _, _ = model.forward(x_t, t, self_cond=None, ctx_emb=ctx_emb)
            self_cond    = vel_sc.mean(dim=1, keepdim=False).detach()

        # Forward condizionato
        with torch.autocast(device_type=device.split(":")[0], dtype=dtype):
            vel_cond,   _, _ = model.forward(x_t, t, self_cond=self_cond, ctx_emb=ctx_emb)
            vel_uncond, _, _ = model.forward(x_t, t, self_cond=self_cond, ctx_emb=null_emb)

        # CFG
        vel = vel_uncond + cfg_scale * (vel_cond - vel_uncond)

        # Aggiorna self_cond per il prossimo step
        self_cond = vel.mean(dim=1, keepdim=False).detach()

        # Accumula storia per stima complessità
        vel_history.append(vel.detach().cpu())

        # Stima complessità dopo PROBE_STEPS
        if step == PROBE_STEPS - 1:
            n_steps = estimate_complexity(vel_history)
            n_steps = max(min_steps, min(max_steps, n_steps))
            if verbose:
                var_est = torch.stack([v.float() for v in vel_history]).var(dim=0).mean().item()
                print(f"  Complessità stimata: var={var_est:.4f} → {n_steps} step totali")

        # Euler step: x_{t-dt} = x_t - vel * dt
        x_t = x_t - vel * dt
        step += 1

    # ── 5. Decodifica token ───────────────────────────────────────────────────
    token_ids = model.decode_tokens(x_t)  # (1, L)

    # Rimuovi padding e token speciali
    ids = token_ids[0].tolist()
    if tokenizer.eos_token_id in ids:
        ids = ids[:ids.index(tokenizer.eos_token_id)]

    text = tokenizer.decode(ids, skip_special_tokens=True).strip()

    elapsed = time.time() - t_start
    if verbose:
        print(f"  Generato in {elapsed:.2f}s — {step} step — {len(ids)} token")

    return text


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Harold v0.6 — Sampler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi:
  python sampler.py --prompt "Cos'è il machine learning?"
  python sampler.py --prompt "Scrivi un algoritmo di ordinamento" --steps 20
  python sampler.py --prompt "Ciao" --cfg_scale 1.0 --max_len 64
  python sampler.py --checkpoint checkpoints_sft_v6/harold_v06_sft_s1_best.pt --prompt "..."
        """,
    )
    parser.add_argument("--prompt",     type=str, required=True,  help="Prompt di input")
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints_sft_v6/harold_v06_sft_s1_best.pt",
                        help="Path del checkpoint")
    parser.add_argument("--max_len",    type=int,   default=128,  help="Lunghezza max risposta in token")
    parser.add_argument("--cfg_scale",  type=float, default=3.0,  help="Scala CFG")
    parser.add_argument("--min_steps",  type=int,   default=MIN_STEPS, help="Step minimi denoising")
    parser.add_argument("--max_steps",  type=int,   default=MAX_STEPS, help="Step massimi denoising")
    parser.add_argument("--device",     type=str,   default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tokenizer",  type=str,   default="NousResearch/Llama-2-7b-hf")
    parser.add_argument("--quiet",      action="store_true", help="Disabilita output verboso")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = args.device
    dtype  = torch.bfloat16 if device.startswith("cuda") else torch.float32

    # Carica tokenizer
    print(f"Carico tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_token_id = int(tokenizer.pad_token_id)

    # Carica modello
    model, _ = load_model(args.checkpoint, device=device)
    model     = model.to(dtype)

    print(f"\nPrompt: {args.prompt!r}")
    print("-" * 60)

    response = sample(
        model        = model,
        tokenizer    = tokenizer,
        prompt       = args.prompt,
        max_len      = args.max_len,
        cfg_scale    = args.cfg_scale,
        min_steps    = args.min_steps,
        max_steps    = args.max_steps,
        device       = device,
        dtype        = dtype,
        pad_token_id = pad_token_id,
        verbose      = not args.quiet,
    )

    print("-" * 60)
    print(f"Risposta:\n{response}")


if __name__ == "__main__":
    main()