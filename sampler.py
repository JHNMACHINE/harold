"""
Harold v3 — sampler_v3.py
==========================
Sampler per continuous diffusion:
  - Input/output nello spazio embedding invece di token discreti
  - Nearest neighbor lookup per decodifica finale
  - Self-conditioning attivo durante il sampling
  - Top-k dinamico (threshold t-dipendente) per il MoE in inferenza
  - MLA KV cache: salva c_kv latente invece di (k,v) separati

Modalità disponibili:
  - confidence  : decodifica i token più certi per step (cosine schedule)
  - sample      : stochastic con temperature e nucleus
  - argmax      : greedy deterministico (debug)

Uso base:
    sampler = HaroldSamplerV3(model, tokenizer, device="cuda")
    text = sampler.generate("Once upon a time", steps=32)
    print(text)
"""

import math
import torch
import torch.nn.functional as F
from typing import Literal
from transformers import PreTrainedTokenizer

from model import Harold
import sys
from transformers import AutoTokenizer
from config import ModelConfig
from model import build_model


SamplingMode = Literal["argmax", "sample", "confidence"]


class HaroldSampler:
    """
    Sampler per Harold v3 — continuous diffusion.

    Differenze chiave rispetto a HaroldSampler (v2):
      - xt è sempre un embedding (B, L, d_model), non token IDs
      - Ogni step: Harold predice x0_pred (embedding), non logit
      - Confidence calcolata via cosine similarity con il vocabolario
      - Self-conditioning attivo: ogni step passa x0_pred al successivo
      - KV cache MLA: c_kv latente invece di (k,v)
    """

    def __init__(
        self,
        model:         Harold,
        tokenizer:     PreTrainedTokenizer,
        device:        str = "cuda",
        mask_token_id: int | None = None,
    ):
        self.model     = model.eval().to(device)
        self.tokenizer = tokenizer
        self.device    = device
        self.mask_id   = mask_token_id or model.config.mask_token_id
        self.T         = model.config.diffusion_T
        self.d_model   = model.config.d_model

        # Embedding del token [MASK] — usato per inizializzare la sequenza
        with torch.no_grad():
            mask_id_tensor = torch.tensor([self.mask_id], device=device)
            self.mask_emb  = model.token_emb(mask_id_tensor)  # (1, d_model)

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _timestep_tensor(self, t_int: int, batch: int) -> torch.Tensor:
        return torch.full((batch,), t_int, dtype=torch.long, device=self.device)

    def _tokens_to_emb(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Converte token IDs in embedding. token_ids: (B, L) → (B, L, d_model)"""
        with torch.no_grad():
            return self.model.token_emb(token_ids)

    def _emb_to_tokens(self, x0_pred: torch.Tensor) -> torch.Tensor:
        """
        Nearest neighbor lookup: trova il token più vicino per ogni posizione.
        x0_pred: (B, L, d_model) → token_ids: (B, L)
        Usa cosine similarity per efficienza e invarianza alla scala.
        """
        return self.model.decode_tokens(x0_pred)

    # ── Modalità di decodifica ───────────────────────────────────────────────

    def _decode_confidence(
        self,
        x0_pred:     torch.Tensor,   # (B, L, d_model)
        xt_emb:      torch.Tensor,   # (B, L, d_model)
        is_masked:   torch.Tensor,   # (B, L) bool — posizioni ancora "sporche"
        unmask_frac: float,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Confidence-based decoding per continuous diffusion.

        La confidenza non viene da un softmax su logit ma dalla
        cosine similarity tra x0_pred e il nearest neighbor nel vocabolario.

        Decodifica i top-unmask_frac token più certi tra quelli mascherati.
        Ritorna l'embedding aggiornato e la maschera dei token decodificati.
        """
        B, L, D = x0_pred.shape

        # Cosine similarity con tutto il vocabolario → confidenza
        emb_norm  = F.normalize(self.model.token_emb.weight, dim=-1)  # (V, D)
        pred_norm = F.normalize(x0_pred / max(temperature, 1e-5), dim=-1)  # (B, L, D)
        sim       = torch.einsum("bld,vd->blv", pred_norm, emb_norm)  # (B, L, V)

        # Confidence = max cosine similarity (quanto è sicuro il nearest neighbor)
        confidence, best_tokens = sim.max(dim=-1)  # (B, L)

        # Considera solo le posizioni mascherate
        confidence = confidence.masked_fill(~is_masked, -1.0)

        # Numero di token da decodificare in questo step
        n_masked  = is_masked.sum(dim=-1).float()          # (B,)
        n_unmask  = (n_masked * unmask_frac).ceil().long().clamp(min=1)

        x_new    = xt_emb.clone()
        unmasked = torch.zeros(B, L, dtype=torch.bool, device=self.device)

        for b in range(B):
            k        = int(n_unmask[b].item())
            topk_idx = confidence[b].topk(k).indices
            # Sostituisci l'embedding mascherato con il nearest neighbor predetto
            x_new[b, topk_idx]    = self.model.token_emb(best_tokens[b, topk_idx])
            unmasked[b, topk_idx] = True

        return x_new, unmasked

    def _decode_sample(
        self,
        x0_pred:   torch.Tensor,   # (B, L, d_model)
        xt_emb:    torch.Tensor,   # (B, L, d_model)
        is_masked: torch.Tensor,   # (B, L) bool
        temperature: float = 1.0,
        top_p:       float = 0.9,
    ) -> torch.Tensor:
        """
        Stochastic sampling con temperature e nucleus.
        Campiona token dalle probabilità derivate dalla cosine similarity.
        """
        B, L, D = x0_pred.shape

        emb_norm  = F.normalize(self.model.token_emb.weight, dim=-1)
        pred_norm = F.normalize(x0_pred / max(temperature, 1e-5), dim=-1)
        sim       = torch.einsum("bld,vd->blv", pred_norm, emb_norm)  # (B, L, V)

        # Softmax sulla similarity → distribuzione sui token
        probs = F.softmax(sim, dim=-1)   # (B, L, V)
        V     = probs.shape[-1]

        # Top-p filtering
        probs_flat  = probs.view(B * L, V)
        sp, si      = torch.sort(probs_flat, descending=True, dim=-1)
        cum         = sp.cumsum(dim=-1)
        remove      = cum - sp > top_p
        sp[remove]  = 0.0
        sp.div_(sp.sum(dim=-1, keepdim=True) + 1e-9)

        sampled_idx = si.gather(-1, torch.multinomial(sp, 1)).view(B, L)

        # Aggiorna solo le posizioni mascherate
        x_new = xt_emb.clone()
        for b in range(B):
            mask_pos = is_masked[b].nonzero(as_tuple=True)[0]
            if mask_pos.numel() > 0:
                x_new[b, mask_pos] = self.model.token_emb(sampled_idx[b, mask_pos])

        return x_new

    def _decode_argmax(
        self,
        x0_pred:   torch.Tensor,   # (B, L, d_model)
        xt_emb:    torch.Tensor,   # (B, L, d_model)
        is_masked: torch.Tensor,   # (B, L) bool
    ) -> torch.Tensor:
        """Greedy: prende il nearest neighbor per ogni posizione mascherata."""
        token_ids = self._emb_to_tokens(x0_pred)  # (B, L)
        x_new     = xt_emb.clone()
        x_new[is_masked] = self.model.token_emb(token_ids[is_masked])
        return x_new

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
        use_self_cond: bool = True,   # self-conditioning attivo
        verbose:     bool  = False,
    ) -> str:
        """
        Genera testo a partire da un prompt.

        Args:
            prompt:        testo iniziale (freezato durante il sampling)
            gen_len:       lunghezza totale sequenza (prompt + generazione)
            steps:         numero di passi di denoising
            mode:          "confidence" | "sample" | "argmax"
            temperature:   temperatura per confidence e sample mode
            top_p:         nucleus threshold per sample mode
            use_self_cond: usa self-conditioning (consigliato)
            verbose:       stampa lo stato ad ogni step

        Returns:
            testo generato (stringa decodificata)
        """
        assert steps <= self.T, f"steps ({steps}) deve essere ≤ diffusion_T ({self.T})"

        # ── Inizializzazione ────────────────────────────────────────────────
        prompt_ids = self.tokenizer.encode(
            prompt, add_special_tokens=True, truncation=True, max_length=gen_len
        ) if prompt else []
        prompt_len = len(prompt_ids)

        # Sequenza iniziale: embedding del prompt + embedding del [MASK]
        xt_emb = self.mask_emb.expand(1, gen_len, self.d_model).clone()
        if prompt_len > 0:
            prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=self.device)
            xt_emb[0, :prompt_len] = self.model.token_emb(prompt_tensor)

        # Maschera statica: posizioni del prompt non vengono mai modificate
        prompt_mask = torch.zeros(1, gen_len, dtype=torch.bool, device=self.device)
        prompt_mask[0, :prompt_len] = True

        # Posizioni ancora da decodificare
        is_masked = ~prompt_mask  # (1, L)

        # Schedule timestep: da T a 1 in steps passi
        t_schedule = torch.linspace(self.T, 1, steps).long()

        def unmask_frac(step_idx: int) -> float:
            cos = math.cos(math.pi * step_idx / (2 * steps))
            return float(1.0 - cos + 1.0 / steps)

        # Self-conditioning: parte con None, aggiornato ad ogni step
        self_cond: torch.Tensor | None = None

        # ── Loop di denoising ───────────────────────────────────────────────
        for step_idx, t_val in enumerate(t_schedule):
            t_tensor = self._timestep_tensor(int(t_val.item()), batch=1)

            # Forward: predice x0 nello spazio embedding
            x0_pred, ce_logits, _ = self.model(
                xt_emb, t_tensor,
                self_cond=self_cond,
            )

            # Aggiorna self-conditioning per il prossimo step
            if use_self_cond:
                self_cond = x0_pred.mean(dim=1).detach()  # (1, d_model)

            # Decodifica in base alla modalità
            if mode == "argmax":
                xt_emb_new = self._decode_argmax(x0_pred, xt_emb, is_masked)
                xt_emb     = torch.where(
                    prompt_mask.unsqueeze(-1), xt_emb, xt_emb_new
                )
                is_masked  = torch.zeros_like(is_masked)  # tutto decodificato

            elif mode == "sample":
                xt_emb_new = self._decode_sample(
                    x0_pred, xt_emb, is_masked, temperature, top_p
                )
                # Re-masking graduale: mantieni mascherato una frazione decrescente
                if step_idx < steps - 1:
                    frac      = 1.0 - (step_idx + 1) / steps
                    n_gen     = gen_len - prompt_len
                    n_remask  = max(0, int(n_gen * frac))
                    gen_pos   = torch.arange(prompt_len, gen_len, device=self.device)
                    perm      = torch.randperm(len(gen_pos), device=self.device)
                    remask_pos = gen_pos[perm[:n_remask]]
                    xt_emb_new[0, remask_pos] = self.mask_emb.squeeze(0)
                    is_masked[0, remask_pos]  = True
                    is_masked[0, :prompt_len] = False
                xt_emb   = torch.where(prompt_mask.unsqueeze(-1), xt_emb, xt_emb_new)
                is_masked = is_masked & ~prompt_mask

            elif mode == "confidence":
                frac = unmask_frac(step_idx)
                xt_emb_new, newly_unmasked = self._decode_confidence(
                    x0_pred, xt_emb, is_masked, frac, temperature
                )
                xt_emb    = torch.where(prompt_mask.unsqueeze(-1), xt_emb, xt_emb_new)
                is_masked = is_masked & ~newly_unmasked & ~prompt_mask

            if verbose:
                n_masked_left = is_masked.sum().item()
                # Decodifica corrente per stampa
                cur_tokens = self._emb_to_tokens(xt_emb)
                cur_text   = self.tokenizer.decode(
                    cur_tokens[0].tolist(), skip_special_tokens=False
                )
                print(f"Step {step_idx+1:3d}/{steps} | t={t_val:4d} | "
                      f"masked={n_masked_left:3d} | {cur_text[:80]}")

            # Termina prima se non ci sono più token mascherati
            if is_masked.sum() == 0:
                break

        # ── Decodifica finale ───────────────────────────────────────────────
        # Posizioni ancora mascherate: forza un ultimo forward a t=1
        if is_masked.any():
            t_final = self._timestep_tensor(1, batch=1)
            x0_pred, _, _ = self.model(xt_emb, t_final, self_cond=self_cond)
            final_tokens  = self._emb_to_tokens(x0_pred)
            final_emb     = self.model.token_emb(final_tokens)
            xt_emb        = torch.where(
                (is_masked & ~prompt_mask).unsqueeze(-1),
                final_emb, xt_emb,
            )

        # Nearest neighbor sull'intera sequenza → token IDs → testo
        token_ids = self._emb_to_tokens(xt_emb)[0].tolist()
        text      = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        return text # type: ignore

    @torch.no_grad()
    def generate_batch(
        self,
        prompts:       list[str],
        gen_len:       int   = 64,
        steps:         int   = 32,
        mode:          SamplingMode = "confidence",
        temperature:   float = 1.0,
        top_p:         float = 0.9,
        use_self_cond: bool  = True,
    ) -> list[str]:
        """
        Genera un batch di sequenze in parallelo.
        Più efficiente di chiamare generate() in loop.
        """
        B = len(prompts)

        enc = self.tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=gen_len,
            return_tensors="pt",
        )
        input_ids      = enc["input_ids"].to(self.device)           # type: ignore # (B, L)
        attention_mask = enc["attention_mask"].to(self.device).bool()  # type: ignore # (B, L)

        # Inizializza: prompt embedding + mask embedding
        xt_emb = self.mask_emb.expand(B, gen_len, self.d_model).clone()
        xt_emb[attention_mask] = self.model.token_emb(input_ids[attention_mask])

        prompt_mask = attention_mask   # (B, L): True = prompt
        is_masked   = ~prompt_mask

        t_schedule = torch.linspace(self.T, 1, steps).long()

        def unmask_frac(i: int) -> float:
            return float(1.0 - math.cos(math.pi * i / (2 * steps)) + 1.0 / steps)

        self_cond: torch.Tensor | None = None

        for step_idx, t_val in enumerate(t_schedule):
            t_tensor = self._timestep_tensor(int(t_val.item()), batch=B)

            x0_pred, _, _ = self.model(xt_emb, t_tensor, self_cond=self_cond)

            if use_self_cond:
                self_cond = x0_pred.mean(dim=1).detach()

            if mode == "argmax":
                xt_emb_new = self._decode_argmax(x0_pred, xt_emb, is_masked)
                xt_emb     = torch.where(prompt_mask.unsqueeze(-1), xt_emb, xt_emb_new)
                is_masked  = torch.zeros_like(is_masked)

            elif mode == "sample":
                xt_emb_new = self._decode_sample(
                    x0_pred, xt_emb, is_masked, temperature, top_p
                )
                if step_idx < steps - 1:
                    frac = 1.0 - (step_idx + 1) / steps
                    for b in range(B):
                        gen_pos   = (~prompt_mask[b]).nonzero(as_tuple=True)[0]
                        n_remask  = max(0, int(len(gen_pos) * frac))
                        perm      = torch.randperm(len(gen_pos), device=self.device)
                        remask_pos = gen_pos[perm[:n_remask]]
                        xt_emb_new[b, remask_pos] = self.mask_emb.squeeze(0)
                        is_masked[b, remask_pos]  = True
                xt_emb    = torch.where(prompt_mask.unsqueeze(-1), xt_emb, xt_emb_new)
                is_masked = is_masked & ~prompt_mask

            elif mode == "confidence":
                frac = unmask_frac(step_idx)
                xt_emb_new, newly_unmasked = self._decode_confidence(
                    x0_pred, xt_emb, is_masked, frac, temperature
                )
                xt_emb    = torch.where(prompt_mask.unsqueeze(-1), xt_emb, xt_emb_new)
                is_masked = is_masked & ~newly_unmasked & ~prompt_mask

            if is_masked.sum() == 0:
                break

        # Cleanup finale
        if is_masked.any():
            t_final   = self._timestep_tensor(1, batch=B)
            x0_pred, _, _ = self.model(xt_emb, t_final, self_cond=self_cond)
            final_tokens  = self._emb_to_tokens(x0_pred)
            final_emb     = self.model.token_emb(final_tokens)
            xt_emb        = torch.where(
                (is_masked & ~prompt_mask).unsqueeze(-1),
                final_emb, xt_emb,
            )

        return [
            self.tokenizer.decode(
                self._emb_to_tokens(xt_emb)[b].tolist(),
                skip_special_tokens=True,
            )
            for b in range(B)
        ] # type: ignore


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else None

    state     = torch.load(ckpt_path, map_location="cpu", weights_only=False) # type: ignore
    model_cfg = state["model_cfg"]
    model     = build_model(model_cfg)
    model.load_state_dict(state["model_state"])
    print(f"Checkpoint caricato: {ckpt_path}")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    sampler   = HaroldSampler(model, tokenizer, device=device)

    prompt = "Once upon a time"
    print(f"\nPrompt: {prompt!r}\n")

    for temp in (0.8, 1.0, 1.3):
        out = sampler.generate(
            prompt=prompt,
            gen_len=64,
            steps=32,
            mode="confidence",
            temperature=temp,
            use_self_cond=True,
        )
        print(f"[confidence t={temp}] {out}")

    out = sampler.generate(
        prompt=prompt,
        gen_len=64,
        steps=32,
        mode="sample",
        temperature=1.0,
        top_p=0.9,
    )
    print(f"[sample     t=1.0] {out}")