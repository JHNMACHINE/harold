"""
Harold v0.7 — sampler.py
=========================
Inferenza con Flow Matching, x0-prediction, iterative decoding, CFG e self-conditioning.

Cambiamenti rispetto a v0.6:

  [v0.7-S1] x0-prediction
             Il modello predice x0_emb invece della velocità del flusso.
             La velocità è ricavata via: v = (x_t - x0_pred) / t

  [v0.7-S2] Iterative decoding (ispirato a MDLM/PLAID)
             Ad ogni step di denoising i token ad alta confidenza vengono
             "congelati" — discretizzati e non più denoised. Solo i token
             incerti continuano a ricevere rumore e vengono raffinati.
             Parametri chiave: freeze_threshold, freeze_ratio.

  [v0.7-S3] Euler + Heun
             ODE solver Euler (default) o Heun (use_heun=True, 2x forward/step).

  [v0.7-S4] Step dinamici
             Invariato da v0.6: stima della complessità dalla varianza di
             x0_pred nei primi PROBE_STEPS step.

  [v0.7-S5] generate_with_trajectory
             Metodo di debug che registra quanti token vengono congelati
             per step — utile per fare eval visiva durante il training.

Invariato da v0.6:
  - load_model con fallback HuggingFace (.safetensors → .pt)
  - encode_context (mean pooling del prompt)
  - CLI (argparse)

Avvio:
  python sampler.py --prompt "Spiega cosa è la diffusione"
  python sampler.py --prompt "..." --steps 32 --cfg_scale 3.0 --checkpoint path/to/ckpt.pt
  python sampler.py --prompt "..." --no_iterative  # denoising uniforme, come v0.6
"""

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import Harold, build_model
from config import get_model_config
from safetensors.torch import load_file as sf_load


# ─────────────────────────────────────────────────────────────────────────────
# Costanti
# ─────────────────────────────────────────────────────────────────────────────

MIN_STEPS        = 16
MAX_STEPS        = 50
PROBE_STEPS      = 4
VAR_THRESHOLD_LO = 0.05
VAR_THRESHOLD_HI = 0.15

HF_REPO_ID      = "JHN-MACHINE/harold-v0.7"
HF_SFT_PT       = "harold-v0.7-3B-sft.pt"
HF_SFT_SF       = "harold-v0.7-3B-sft.safetensors"
DEFAULT_CKPT_SF = "harold-v0.7-3B-sft.safetensors"
DEFAULT_CKPT_PT = "harold-v0.7-3B-sft.pt"


# ─────────────────────────────────────────────────────────────────────────────
# Caricamento modello
# ─────────────────────────────────────────────────────────────────────────────

def _download_from_hf(filename: str, local_path: str) -> bool:
    """Scarica un file da HuggingFace. Ritorna ``True`` se riuscito."""
    try:
        from huggingface_hub import hf_hub_download
        hf_token = os.environ.get("HF_TOKEN")
        print(f"  Scarico {filename} da {HF_REPO_ID}...")
        downloaded = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=filename,
            token=hf_token,
            local_dir=os.path.dirname(local_path) or ".",
        )
        if downloaded != local_path:
            import shutil
            shutil.move(downloaded, local_path)
        print(f"  Scaricato -> {local_path}")
        return True
    except Exception as e:
        print(f"  Download fallito: {e}")
        return False


def load_model(
    checkpoint_path: str         = DEFAULT_CKPT_SF,
    device:          str         = "cuda",
    dtype:           torch.dtype = torch.bfloat16,
) -> Harold:
    """
    Carica Harold dal checkpoint.

    Ordine di ricerca:
      1. Path locale fornito (se esiste)
      2. .safetensors locale (DEFAULT_CKPT_SF)
      3. Download .safetensors da HuggingFace
      4. .pt locale (DEFAULT_CKPT_PT)
      5. Download .pt da HuggingFace
    """
    path_to_use = checkpoint_path

    if not os.path.isfile(path_to_use):
        if os.path.isfile(DEFAULT_CKPT_SF):
            path_to_use = DEFAULT_CKPT_SF
        elif _download_from_hf(HF_SFT_SF, DEFAULT_CKPT_SF):
            path_to_use = DEFAULT_CKPT_SF
        elif os.path.isfile(DEFAULT_CKPT_PT):
            path_to_use = DEFAULT_CKPT_PT
        elif _download_from_hf(HF_SFT_PT, DEFAULT_CKPT_PT):
            path_to_use = DEFAULT_CKPT_PT
        else:
            raise FileNotFoundError(
                f"Checkpoint non trovato: {checkpoint_path}\n"
                "Imposta HF_TOKEN per scaricare automaticamente da HuggingFace."
            )

    print(f"Carico checkpoint: {path_to_use}")

    if path_to_use.endswith(".safetensors"):
        model_cfg = get_model_config()
        model     = build_model(model_cfg).to(device)
        weights   = sf_load(path_to_use, device="cpu")
        model.load_state_dict(weights, strict=False)
    else:
        state     = torch.load(path_to_use, map_location="cpu", weights_only=False)
        model_cfg = state.get("model_cfg", get_model_config())
        model     = build_model(model_cfg).to(device)
        model.load_state_dict(state["model_state"], strict=False)

    model = model.to(dtype).eval()

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    label    = f"{n_params/1000:.2f}B" if n_params >= 1000 else f"{n_params:.1f}M"
    print(f"Harold v0.7 -- {label} parametri")
    print(f"Device: {device}  |  dtype: {next(model.parameters()).dtype}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def encode_context(
    model:        Harold,
    prompt_ids:   torch.Tensor,
    pad_token_id: int,
) -> torch.Tensor:
    r"""encode_context(model, prompt_ids, pad_token_id) -> Tensor

    Mean pooling degli embedding del prompt.

    Args:
        model: istanza di :class:`Harold`
        prompt_ids (LongTensor): token id del prompt, shape :math:`(1, L\_prompt)`
        pad_token_id (int): id del token di padding

    Returns:
        Tensor: context embedding, shape :math:`(1, d\_model)`
    """
    pad_mask = (prompt_ids != pad_token_id).float()
    emb      = model.token_emb(prompt_ids)
    n_tokens = pad_mask.sum(dim=1, keepdim=True).clamp(min=1)
    return (emb * pad_mask.unsqueeze(-1)).sum(dim=1) / n_tokens


def x0_to_velocity(
    x_t:     torch.Tensor,
    x0_pred: torch.Tensor,
    t:       torch.Tensor,
    eps:     float = 1e-4,
) -> torch.Tensor:
    r"""x0_to_velocity(x_t, x0_pred, t, eps=1e-4) -> Tensor

    Converte :math:`\hat{x}_0` in velocita del flusso:
    :math:`\hat{v} = (x_t - \hat{x}_0) / t`

    Args:
        x_t (Tensor): embedding rumoroso, shape :math:`(B, L, d\_model)`
        x0_pred (Tensor): predizione di :math:`\hat{x}_0`, shape :math:`(B, L, d\_model)`
        t (Tensor): timestep corrente, shape :math:`(B,)`
        eps (float, optional): clamp minimo su ``t``. Default: ``1e-4``

    Returns:
        Tensor: velocita del flusso, shape :math:`(B, L, d\_model)`
    """
    return (x_t - x0_pred) / t.clamp(min=eps).view(-1, 1, 1)


def confidence_scores(ce_logits: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    r"""confidence_scores(ce_logits, temperature=0.5) -> Tensor

    Probabilita massima del softmax per token.

    Args:
        ce_logits (Tensor): logits, shape :math:`(B, L, V+1)`
        temperature (float, optional): temperatura. Default: ``0.5``

    Returns:
        Tensor: confidenza in :math:`[0, 1]`, shape :math:`(B, L)`
    """
    return F.softmax(ce_logits / temperature, dim=-1).max(dim=-1).values


def estimate_complexity(x0_history: list) -> int:
    r"""estimate_complexity(x0_history) -> int

    Stima il numero ottimale di step dalla varianza di ``x0_pred``
    nei primi ``PROBE_STEPS`` step.

    Args:
        x0_history (list of Tensor): lista di tensori :math:`(1, L, d\_model)`

    Returns:
        int: step stimati in :math:`[\text{MIN\_STEPS}, \text{MAX\_STEPS}]`
    """
    if not x0_history:
        return MIN_STEPS
    stacked = torch.stack([v.float() for v in x0_history], dim=0)
    var     = stacked.var(dim=0).mean().item()
    if var < VAR_THRESHOLD_LO:
        return MIN_STEPS
    if var > VAR_THRESHOLD_HI:
        return MAX_STEPS
    ratio = (var - VAR_THRESHOLD_LO) / (VAR_THRESHOLD_HI - VAR_THRESHOLD_LO)
    return int(MIN_STEPS + ratio * (MAX_STEPS - MIN_STEPS))


# ─────────────────────────────────────────────────────────────────────────────
# Configurazione
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SamplerConfig:
    r"""Configurazione per :func:`sample`.

    Args:
        min_steps (int): step minimi di denoising. Default: ``16``
        max_steps (int): step massimi di denoising. Default: ``50``
        cfg_scale (float): scala CFG. ``1.0`` = no guidance. Default: ``3.0``
        self_cond (bool): usa self-conditioning. Default: ``True``
        iterative (bool): abilita iterative decoding. Default: ``True``
        freeze_threshold (float): confidenza minima per congelare un token.
            Default: ``0.9``
        freeze_ratio (float, optional): frazione massima token da congelare
            per step. ``None`` = nessun limite. Default: ``None``
        confidence_temp (float): temperatura per la confidenza. Default: ``0.5``
        use_heun (bool): usa Heun invece di Euler. Default: ``False``
        t_min (float): timestep minimo. Default: ``1e-3``
    """
    min_steps:        int            = MIN_STEPS
    max_steps:        int            = MAX_STEPS
    cfg_scale:        float          = 3.0
    self_cond:        bool           = True
    iterative:        bool           = True
    freeze_threshold: float          = 0.9
    freeze_ratio:     Optional[float] = None
    confidence_temp:  float          = 0.5
    use_heun:         bool           = False
    t_min:            float          = 1e-3


# ─────────────────────────────────────────────────────────────────────────────
# Sampler principale
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def sample(
    model:        Harold,
    tokenizer,
    prompt:       str,
    max_len:      int                    = 256,
    cfg:          Optional[SamplerConfig] = None,
    device:       str                    = "cuda",
    dtype:        torch.dtype            = torch.bfloat16,
    pad_token_id: int                    = 0,
    verbose:      bool                   = True,
) -> str:
    r"""sample(model, tokenizer, prompt, max_len=256, ...) -> str

    Genera una risposta al prompt con Flow Matching, x0-prediction e
    (opzionalmente) iterative decoding.

    Args:
        model: istanza di :class:`Harold` in modalita eval
        tokenizer: tokenizer HuggingFace compatibile
        prompt (str): testo del prompt
        max_len (int, optional): lunghezza massima risposta in token. Default: ``256``
        cfg (:class:`SamplerConfig`, optional): configurazione. Default: ``None``
        device (str, optional): device. Default: ``"cuda"``
        dtype (torch.dtype, optional): dtype di compute. Default: ``torch.bfloat16``
        pad_token_id (int, optional): id del padding token. Default: ``0``
        verbose (bool, optional): stampa progress. Default: ``True``

    Returns:
        str: testo generato, senza token speciali
    """
    if cfg is None:
        cfg = SamplerConfig()

    t_start        = time.time()
    autocast_device = device.split(":")[0]

    # 1. Tokenizza e calcola ctx_emb
    enc        = tokenizer(prompt, return_tensors="pt", truncation=True,
                           max_length=512, add_special_tokens=False)
    prompt_ids = enc["input_ids"].to(device)
    ctx_emb    = encode_context(model, prompt_ids, pad_token_id).to(dtype)
    null_emb   = torch.zeros_like(ctx_emb)

    # 2. Inizializza x_T ~ N(0, I)
    B, L, D  = 1, max_len, model.d_model
    x_t      = torch.randn(B, L, D, device=device, dtype=dtype)

    # 3. Stato iterative decoding
    frozen   = torch.zeros(B, L, dtype=torch.bool, device=device)
    x_frozen = torch.zeros_like(x_t)

    # 4. Loop di denoising
    n_steps     = cfg.min_steps
    x0_history: list = []
    self_cond:  Optional[torch.Tensor] = None
    step        = 0

    if verbose:
        mode = "iterative" if cfg.iterative else "uniforme"
        print(f"Denoising [{mode}] -- step iniziali: {n_steps} (adattativi)")

    while step < n_steps:
        t_val  = 1.0 - step / n_steps
        t_next = 1.0 - (step + 1) / n_steps
        t      = torch.full((B,), t_val,  device=device, dtype=torch.float32)
        t_n    = torch.full((B,), t_next, device=device, dtype=torch.float32)

        if cfg.iterative and frozen.any():
            x_t[frozen] = x_frozen[frozen]

        # Forward: condizionato e incondizionato per CFG
        with torch.autocast(device_type=autocast_device, dtype=dtype):
            x0_cond,   ce_logits, _ = model(x_t, t, self_cond=self_cond, ctx_emb=ctx_emb)
            x0_uncond, _,         _ = model(x_t, t, self_cond=self_cond, ctx_emb=null_emb)

        # CFG su x0_pred
        x0_pred   = x0_uncond + cfg.cfg_scale * (x0_cond - x0_uncond)
        ce_logits = model.ce_head(x0_pred)

        if cfg.self_cond:
            self_cond = x0_pred.mean(dim=1).detach()

        # Stima complessita dopo PROBE_STEPS
        x0_history.append(x0_pred.detach().cpu())
        if step == PROBE_STEPS - 1:
            n_steps = max(cfg.min_steps, min(cfg.max_steps, estimate_complexity(x0_history)))
            if verbose:
                var_est = torch.stack([v.float() for v in x0_history]).var(dim=0).mean().item()
                print(f"  Complessita: var={var_est:.4f} -> {n_steps} step totali")

        # Iterative decoding: congela token ad alta confidenza
        if cfg.iterative:
            conf      = confidence_scores(ce_logits, cfg.confidence_temp)
            candidate = (conf >= cfg.freeze_threshold) & ~frozen

            if cfg.freeze_ratio is not None and candidate.any():
                max_freeze = max(1, int(cfg.freeze_ratio * L))
                for b in range(B):
                    cand_idx = candidate[b].nonzero(as_tuple=False).view(-1)
                    if cand_idx.numel() > max_freeze:
                        top_idx         = conf[b][cand_idx].topk(max_freeze).indices
                        mask            = torch.zeros_like(candidate[b])
                        mask[cand_idx[top_idx]] = True
                        candidate[b]    = mask

            if candidate.any():
                best_tokens         = ce_logits.argmax(dim=-1)
                x_frozen[candidate] = model.token_emb(best_tokens)[candidate].detach()
                frozen             |= candidate

        # ODE step
        if step < n_steps - 1:
            vel = x0_to_velocity(x_t, x0_pred, t)
            dt  = (t_n - t).view(-1, 1, 1)

            if cfg.use_heun and step < n_steps - 2:
                x_euler = x_t + dt * vel
                with torch.autocast(device_type=autocast_device, dtype=dtype):
                    x0_2, _, _ = model(x_euler, t_n, self_cond=self_cond, ctx_emb=ctx_emb)
                vel_2   = x0_to_velocity(x_euler, x0_2, t_n)
                x_t     = x_t + dt * (0.5 * (vel + vel_2))
            else:
                x_t = x_t + dt * vel

            if cfg.iterative and frozen.any():
                x_t[frozen] = x_frozen[frozen]

        step += 1

    # 5. Decodifica
    x_final = x_t.clone()
    if cfg.iterative and frozen.any():
        x_final[frozen] = x_frozen[frozen]

    token_ids = model.decode_tokens(x_final)
    ids       = token_ids[0].tolist()
    if tokenizer.eos_token_id is not None and tokenizer.eos_token_id in ids:
        ids = ids[:ids.index(tokenizer.eos_token_id)]

    text    = tokenizer.decode(ids, skip_special_tokens=True).strip()
    elapsed = time.time() - t_start

    if verbose:
        n_frozen = frozen.sum().item()
        print(f"  Generato in {elapsed:.2f}s -- {step} step -- {len(ids)} token"
              f" -- {n_frozen}/{L} congelati ({100.0 * n_frozen / L:.0f}%)")

    return text


# ─────────────────────────────────────────────────────────────────────────────
# generate_with_trajectory (debug)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_with_trajectory(
    model:   Harold,
    seq_len: int,
    cfg:     Optional[SamplerConfig] = None,
    device:  Optional[str]           = None,
    dtype:   torch.dtype             = torch.bfloat16,
) -> Tuple[torch.Tensor, list]:
    r"""generate_with_trajectory(model, seq_len, ...) -> (tokens, trajectory)

    Come :func:`sample` ma senza prompt, ritorna la traiettoria di denoising.
    Utile per debugging e visualizzazione del freezing durante il training.

    Args:
        model: istanza di :class:`Harold`
        seq_len (int): lunghezza della sequenza
        cfg (:class:`SamplerConfig`, optional): configurazione
        device (str, optional): device
        dtype (torch.dtype, optional): dtype di compute

    Returns:
        tuple:
            - **tokens** (*LongTensor*) -- token finali, shape :math:`(1, L)`
            - **trajectory** (*list of dict*) -- un dict per step con chiavi
              ``"step"``, ``"t"``, ``"n_frozen"``, ``"pct_frozen"``,
              ``"tokens_so_far"``
    """
    if cfg is None:
        cfg = SamplerConfig()
    if device is None:
        device = str(next(model.parameters()).device)

    B, L, D  = 1, seq_len, model.d_model
    x_t      = torch.randn(B, L, D, device=device, dtype=dtype)
    frozen   = torch.zeros(B, L, dtype=torch.bool, device=device)
    x_frozen = torch.zeros_like(x_t)
    self_cond: Optional[torch.Tensor] = None
    trajectory = []
    n_steps    = cfg.min_steps

    for step in range(n_steps):
        t_val  = 1.0 - step / n_steps
        t_next = 1.0 - (step + 1) / n_steps
        t      = torch.full((B,), t_val,  device=device, dtype=torch.float32)
        t_n    = torch.full((B,), t_next, device=device, dtype=torch.float32)

        if frozen.any():
            x_t[frozen] = x_frozen[frozen]

        x0_pred, ce_logits, _ = model(x_t, t, self_cond=self_cond)

        if cfg.self_cond:
            self_cond = x0_pred.mean(dim=1).detach()

        conf      = confidence_scores(ce_logits, cfg.confidence_temp)
        candidate = (conf >= cfg.freeze_threshold) & ~frozen
        if candidate.any():
            best_tokens         = ce_logits.argmax(dim=-1)
            x_frozen[candidate] = model.token_emb(best_tokens)[candidate].detach()
            frozen             |= candidate

        n_frozen = frozen.sum().item()
        trajectory.append({
            "step":         step,
            "t":            t_val,
            "n_frozen":     n_frozen,
            "pct_frozen":   100.0 * n_frozen / L,
            "tokens_so_far": model.decode_tokens(
                torch.where(frozen.unsqueeze(-1), x_frozen, x_t)
            ).cpu(),
        })

        if step < n_steps - 1:
            vel = x0_to_velocity(x_t, x0_pred, t)
            x_t = x_t + (t_n - t).view(-1, 1, 1) * vel
            if frozen.any():
                x_t[frozen] = x_frozen[frozen]

    x_final = x_t.clone()
    if frozen.any():
        x_final[frozen] = x_frozen[frozen]

    return model.decode_tokens(x_final), trajectory


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Harold v0.7 -- Sampler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi:
  python sampler.py --prompt "Cos'e il machine learning?"
  python sampler.py --prompt "Scrivi un algoritmo di ordinamento" --max_steps 32
  python sampler.py --prompt "Ciao" --cfg_scale 1.0 --max_len 128
  python sampler.py --prompt "..." --no_iterative         # denoising uniforme
  python sampler.py --prompt "..." --use_heun             # ODE Heun (piu lento)
  python sampler.py --checkpoint checkpoints_v7/harold_v07_final.pt --prompt "..."
        """,
    )
    parser.add_argument("--prompt",           type=str,   required=True)
    parser.add_argument("--checkpoint",       type=str,   default=DEFAULT_CKPT_SF)
    parser.add_argument("--max_len",          type=int,   default=256)
    parser.add_argument("--cfg_scale",        type=float, default=3.0)
    parser.add_argument("--min_steps",        type=int,   default=MIN_STEPS)
    parser.add_argument("--max_steps",        type=int,   default=MAX_STEPS)
    parser.add_argument("--freeze_threshold", type=float, default=0.9)
    parser.add_argument("--freeze_ratio",     type=float, default=None,
                        help="Frazione max token da congelare per step (default: nessun limite)")
    parser.add_argument("--confidence_temp",  type=float, default=0.5)
    parser.add_argument("--no_iterative",     action="store_true",
                        help="Disabilita iterative decoding (come v0.6)")
    parser.add_argument("--use_heun",         action="store_true",
                        help="Usa ODE solver Heun invece di Euler (2x forward/step)")
    parser.add_argument("--device",           type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tokenizer",        type=str,
                        default="JHN-MACHINE/harold")
    parser.add_argument("--quiet",            action="store_true")
    return parser.parse_args()


def main() -> None:
    args  = parse_args()
    device = args.device
    dtype  = torch.bfloat16 if device.startswith("cuda") else torch.float32

    print(f"Carico tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_token_id = int(tokenizer.pad_token_id)

    model = load_model(args.checkpoint, device=device, dtype=dtype)

    cfg = SamplerConfig(
        min_steps        = args.min_steps,
        max_steps        = args.max_steps,
        cfg_scale        = args.cfg_scale,
        iterative        = not args.no_iterative,
        freeze_threshold = args.freeze_threshold,
        freeze_ratio     = args.freeze_ratio,
        confidence_temp  = args.confidence_temp,
        use_heun         = args.use_heun,
    )

    print(f"\nPrompt: {args.prompt!r}")
    print("-" * 60)

    response = sample(
        model        = model,
        tokenizer    = tokenizer,
        prompt       = args.prompt,
        max_len      = args.max_len,
        cfg          = cfg,
        device       = device,
        dtype        = dtype,
        pad_token_id = pad_token_id,
        verbose      = not args.quiet,
    )

    print("-" * 60)
    print(f"Risposta:\n{response}")


if __name__ == "__main__":
    main()