"""
Harold Sampler — sampler.py
============================
Loop di denoising iterativo per Harold (DiffusionMoE).

Modalità disponibili:
  - argmax      : greedy deterministico (debug / baseline)
  - sample      : stochastic con temperature (più vario)
  - confidence  : decodifica solo i token ad alta certezza per step
                  (qualità migliore, simile a MaskGIT)

Uso base:
    sampler = HaroldSampler(model, tokenizer, device="cuda")
    text = sampler.generate("Once upon a time", steps=32, mode="confidence")
    print(text)
"""

import math
import torch
import torch.nn.functional as F
from typing import Literal
from transformers import PreTrainedTokenizer

from model import Harold
from config import ModelConfig


SamplingMode = Literal["argmax", "sample", "confidence"]


class HaroldSampler:
    """
    Sampler per Harold.

    Il loop di denoising parte da una sequenza completamente mascherata
    (o parzialmente, se viene fornito un prompt) e la raffina iterativamente
    in `steps` passi, da t=T a t=0.

    Args:
        model:       istanza Harold già caricata e in eval mode
        tokenizer:   tokenizer compatibile (es. bert-base-uncased)
        device:      "cuda" o "cpu"
        mask_token_id: id del token [MASK] (default: da model.config)
    """

    def __init__(
        self,
        model:         Harold,
        tokenizer:     PreTrainedTokenizer,
        device:        str = "cuda",
        mask_token_id: int | None = None,
    ):
        self.model    = model.eval().to(device)
        self.tokenizer = tokenizer
        self.device   = device
        self.mask_id  = mask_token_id or model.config.mask_token_id
        self.T        = model.config.diffusion_T

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _encode_prompt(self, prompt: str, max_len: int) -> torch.Tensor:
        """
        Tokenizza il prompt e lo piazza all'inizio della sequenza.
        Il resto è mascherato.
        """
        ids = self.tokenizer.encode(
            prompt,
            add_special_tokens=True,
            truncation=True,
            max_length=max_len,
        )
        seq = torch.full((max_len,), self.mask_id, dtype=torch.long)
        seq[: len(ids)] = torch.tensor(ids, dtype=torch.long)
        return seq  # (L,)

    def _timestep_tensor(self, t_int: int, batch: int) -> torch.Tensor:
        return torch.full((batch,), t_int, dtype=torch.long, device=self.device)

    # ── Modalità di decodifica ───────────────────────────────────────────────

    @staticmethod
    def _decode_argmax(logits: torch.Tensor) -> torch.Tensor:
        """Greedy: prende il token con probabilità massima."""
        return logits.argmax(dim=-1)  # (B, L)

    @staticmethod
    def _decode_sample(
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """
        Stochastic sampling con temperature e nucleus (top-p).
        """
        B, L, V = logits.shape
        logits_flat = logits.view(B * L, V)

        # Temperature scaling
        scaled = logits_flat / max(temperature, 1e-5)

        # Top-p (nucleus) filtering
        probs = F.softmax(scaled, dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
        cum_probs = sorted_probs.cumsum(dim=-1)
        # Rimuovi i token che superano la soglia top_p
        remove_mask = cum_probs - sorted_probs > top_p
        sorted_probs[remove_mask] = 0.0
        sorted_probs.div_(sorted_probs.sum(dim=-1, keepdim=True) + 1e-9)

        # Campiona
        sampled_idx = torch.multinomial(sorted_probs, num_samples=1)  # (B*L, 1)
        tokens = sorted_idx.gather(-1, sampled_idx).squeeze(-1)       # (B*L,)
        return tokens.view(B, L)

    @staticmethod
    def _decode_confidence(
        logits:      torch.Tensor,
        xt:          torch.Tensor,
        mask_id:     int,
        unmask_frac: float,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Confidence-based decoding (stile MaskGIT):
          1. Calcola la probabilità massima per ogni token mascherato
          2. Decodifica solo i top-unmask_frac token (i più certi)
          3. Re-maschera il resto

        Returns:
            x_new:    sequenza aggiornata
            unmasked: maschera bool True dove abbiamo decodificato
        """
        B, L, V = logits.shape
        mask_positions = (xt == mask_id)  # (B, L)

        # Probabilità del token più probabile per ogni posizione
        probs = F.softmax(logits / max(temperature, 1e-5), dim=-1)
        confidence, candidates = probs.max(dim=-1)  # (B, L)

        # Considera solo le posizioni mascherate
        confidence = confidence.masked_fill(~mask_positions, -1.0)

        # Numero di token da decodificare in questo step
        n_masked   = mask_positions.sum(dim=-1).float()         # (B,)
        n_unmask   = (n_masked * unmask_frac).ceil().long()     # (B,)
        n_unmask   = n_unmask.clamp(min=1)

        x_new    = xt.clone()
        unmasked = torch.zeros_like(xt, dtype=torch.bool)

        for b in range(B):
            k = n_unmask[b].item()
            if k == 0:
                continue
            topk_idx = confidence[b].topk(k).indices # type: ignore
            x_new[b, topk_idx]    = candidates[b, topk_idx]
            unmasked[b, topk_idx] = True

        return x_new, unmasked

    # ── Loop principale ──────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        prompt:      str   = "",
        gen_len:     int   = 64,
        steps:       int   = 32,
        mode:        SamplingMode = "confidence",
        temperature: float = 1.0,
        top_p:       float = 0.9,
        guidance:    float = 0.0,   # classifier-free guidance strength (sperimentale)
        verbose:     bool  = False,
    ) -> str:
        """
        Genera testo a partire da un prompt.

        Args:
            prompt:      testo iniziale (sarà freezato durante il sampling)
            gen_len:     lunghezza totale della sequenza (prompt + generazione)
            steps:       numero di passi di denoising (< diffusion_T)
            mode:        "argmax" | "sample" | "confidence"
            temperature: temperatura per sample e confidence mode
            top_p:       nucleus threshold per sample mode
            guidance:    forza del conditional guidance (0 = disabilitato)
            verbose:     stampa lo stato ad ogni step

        Returns:
            testo generato (stringa decodificata)
        """
        assert steps <= self.T, f"steps ({steps}) deve essere ≤ diffusion_T ({self.T})"

        # ── Inizializzazione ────────────────────────────────────────────────
        prompt_ids = self.tokenizer.encode(
            prompt, add_special_tokens=True, truncation=True, max_length=gen_len
        ) if prompt else []
        prompt_len = len(prompt_ids)

        # Sequenza: [prompt_ids | MASK MASK ... MASK]
        xt = torch.full((1, gen_len), self.mask_id, dtype=torch.long, device=self.device)
        if prompt_len > 0:
            xt[0, :prompt_len] = torch.tensor(prompt_ids, dtype=torch.long, device=self.device)

        # Maschera statica per il prompt: non verrà mai modificata
        prompt_mask = torch.zeros(1, gen_len, dtype=torch.bool, device=self.device)
        prompt_mask[0, :prompt_len] = True

        # Timestep schedule: da T a 1 in `steps` passi
        t_schedule = torch.linspace(self.T, 1, steps).long()

        # Frazione di token da decodificare per step in modalità confidence
        # (cosine schedule: più aggressivo all'inizio, più cauto alla fine)
        def unmask_frac(step_idx: int) -> float:
            cos = math.cos(math.pi * step_idx / (2 * steps))
            return float(1.0 - cos + 1.0 / steps)

        # ── Loop di denoising ───────────────────────────────────────────────
        for step_idx, t_val in enumerate(t_schedule):
            t_tensor = self._timestep_tensor(t_val.item(), batch=1) # type: ignore

            logits, _ = self.model(xt, t_tensor)  # (1, L, V)

            # Forza il prompt a non essere modificato: azzera la logit corretta
            # sulle posizioni del prompt (non necessario ma buona pratica)

            if mode == "argmax":
                x_pred = self._decode_argmax(logits)
                # Aggiorna solo le posizioni mascherate
                update_mask = (xt == self.mask_id) & ~prompt_mask
                xt = torch.where(update_mask, x_pred, xt)

            elif mode == "sample":
                x_pred = self._decode_sample(logits, temperature=temperature, top_p=top_p)
                # Re-maschera le posizioni poco certe se non siamo all'ultimo step
                if step_idx < steps - 1:
                    # Mantieni mascherato una frazione che decresce nel tempo
                    frac_to_keep_masked = 1.0 - (step_idx + 1) / steps
                    n_gen = gen_len - prompt_len
                    n_remask = max(0, int(n_gen * frac_to_keep_masked))
                    gen_positions = torch.arange(prompt_len, gen_len, device=self.device)
                    perm = torch.randperm(len(gen_positions), device=self.device)
                    remask_positions = gen_positions[perm[:n_remask]]
                    x_pred[0, remask_positions] = self.mask_id
                update_mask = (xt == self.mask_id) & ~prompt_mask
                xt = torch.where(update_mask, x_pred, xt)

            elif mode == "confidence":
                frac = unmask_frac(step_idx)
                x_new, unmasked = self._decode_confidence(
                    logits, xt, self.mask_id, frac, temperature=temperature
                )
                # Non toccare il prompt
                xt = torch.where(prompt_mask, xt, x_new)

            else:
                raise ValueError(f"Modalità sconosciuta: {mode!r}")

            if verbose:
                n_masked = (xt == self.mask_id).sum().item()
                decoded_so_far = self.tokenizer.decode(
                    xt[0].tolist(), skip_special_tokens=False
                )
                print(f"Step {step_idx+1:3d}/{steps} | t={t_val:4d} | masked={n_masked:3d} | {decoded_so_far[:80]}")

        # ── Decodifica finale ───────────────────────────────────────────────
        # Eventuali token rimasti mascherati: forziamo un argmax a t=1
        still_masked = (xt == self.mask_id).any()
        if still_masked:
            t_final = self._timestep_tensor(1, batch=1)
            logits, _ = self.model(xt, t_final)
            x_pred    = self._decode_argmax(logits)
            update    = (xt == self.mask_id) & ~prompt_mask
            xt        = torch.where(update, x_pred, xt)

        token_ids = xt[0].tolist()
        text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        return text # type: ignore

    @torch.no_grad()
    def generate_batch(
        self,
        prompts:     list[str],
        gen_len:     int   = 64,
        steps:       int   = 32,
        mode:        SamplingMode = "confidence",
        temperature: float = 1.0,
        top_p:       float = 0.9,
    ) -> list[str]:
        """
        Genera un batch di sequenze in parallelo.
        Più efficiente di chiamare generate() in loop.
        """
        B = len(prompts)

        # Tokenizza tutti i prompt con padding
        enc = self.tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=gen_len,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(self.device)  # (B, L) # type: ignore

        # Identifica le posizioni reali del prompt (non padding)
        attention_mask = enc["attention_mask"].to(self.device).bool()  # (B, L) # type: ignore

        # Sequenza iniziale: prompt + MASK
        xt = input_ids.clone()
        xt[~attention_mask] = self.mask_id  # Maschera il padding
        # Le posizioni dopo il prompt (che sono padding) diventano MASK da generare
        # Le posizioni del prompt rimangono freezate
        prompt_mask = attention_mask  # (B, L): True = prompt, non modificare

        t_schedule = torch.linspace(self.T, 1, steps).long()

        def unmask_frac(step_idx: int) -> float:
            cos = math.cos(math.pi * step_idx / (2 * steps))
            return float(1.0 - cos + 1.0 / steps)

        for step_idx, t_val in enumerate(t_schedule):
            t_tensor = self._timestep_tensor(t_val.item(), batch=B) # type: ignore
            logits, _ = self.model(xt, t_tensor)

            if mode == "argmax":
                x_pred = self._decode_argmax(logits)
                update = (xt == self.mask_id) & ~prompt_mask
                xt = torch.where(update, x_pred, xt)

            elif mode == "sample":
                x_pred = self._decode_sample(logits, temperature=temperature, top_p=top_p)
                if step_idx < steps - 1:
                    frac = 1.0 - (step_idx + 1) / steps
                    gen_len_actual = (~prompt_mask).sum(dim=-1).float()
                    for b in range(B):
                        n_remask = max(0, int(gen_len_actual[b].item() * frac))
                        gen_pos = (~prompt_mask[b]).nonzero(as_tuple=True)[0]
                        perm = torch.randperm(len(gen_pos), device=self.device)
                        remask_pos = gen_pos[perm[:n_remask]]
                        x_pred[b, remask_pos] = self.mask_id
                update = (xt == self.mask_id) & ~prompt_mask
                xt = torch.where(update, x_pred, xt)

            elif mode == "confidence":
                frac = unmask_frac(step_idx)
                x_new, _ = self._decode_confidence(
                    logits, xt, self.mask_id, frac, temperature=temperature
                )
                xt = torch.where(prompt_mask, xt, x_new)

        # Cleanup finale
        still_masked = (xt == self.mask_id).any()
        if still_masked:
            t_final = self._timestep_tensor(1, batch=B)
            logits, _ = self.model(xt, t_final)
            x_pred    = self._decode_argmax(logits)
            update    = (xt == self.mask_id) & ~prompt_mask
            xt        = torch.where(update, x_pred, xt)

        return [
            self.tokenizer.decode(xt[b].tolist(), skip_special_tokens=True)
            for b in range(B)
        ] # type: ignore


# ─────────────────────────────────────────────────────────────────────────────
# Uso standalone — test rapido senza chatbot
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from transformers import AutoTokenizer
    from config import ModelConfig, get_model_config
    from model import build_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Carica config POC ridotta
    cfg = ModelConfig(
        d_model=512,
        n_layers=8,
        n_heads=8,
        n_kv_heads=2,
        d_ff=2048,
        moe_n_routed_experts=4,
        moe_top_k=2,
        ds_moe_n_shared_experts=1,
        max_seq_len=512,
        diffusion_T=64,
        vocab_size=30522,       # bert-base-uncased
        mask_token_id=103,      # [MASK] in bert
    )

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model     = build_model(cfg)

    # Carica checkpoint se disponibile
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else None
    if ckpt_path:
        state = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state["model_state"])
        print(f"Checkpoint caricato: {ckpt_path}")
    else:
        print("Nessun checkpoint — pesi random (solo per test struttura)")

    sampler = HaroldSampler(model, tokenizer, device=device)

    prompt = "Once upon a time"
    print(f"\nPrompt: {prompt!r}")

    for mode in ("argmax", "sample", "confidence"):
        out = sampler.generate(
            prompt=prompt,
            gen_len=64,
            steps=16,
            mode=mode,
            temperature=0.8,
            verbose=False,
        )
        print(f"[{mode:10s}] {out}")