"""
Harold v0.5 — sampler.py
============================
Sampler per Flow Matching continuous diffusion.

Cambiamenti rispetto a v0.4:
  [S-FM1] _dpm_step e _reverse_sde_step rimossi — usavano VP-SDE schedule
          (alpha/sigma/beta) che non esiste più in v0.5.

  [S-FM2] _flow_step — integrazione ODE con Euler (1° ordine):
            x_{t-1} = x_t - dt * vel_pred
          Con Flow Matching le traiettorie sono rette, quindi Euler
          con 10-20 step è sufficiente senza DPM-Solver++.

  [S-FM3] _flow_step_cfg — come _flow_step ma con due forward pass
          separati (cond + uncond) per CFG guidance:
            vel_guided = vel_uncond + cfg_scale * (vel_cond - vel_uncond)

  [S-FM4] self_cond aggiornato: usa vel_pred invece di eps_pred.

  [S-FM5] Tokenizer aggiornato a LLaMA nel standalone test.

Mantenuti da v0.4:
  _build_unused_mask, _mask_unused_tokens, _apply_repetition_penalty,
  _decode, _anchor_confident_tokens, _encode_context,
  generate, generate_batch, generate_conditioned
"""

import math
import sys
import torch
import torch.nn.functional as F
from typing import Any, Literal, Optional, List, Tuple, cast
from transformers import PreTrainedTokenizer, AutoTokenizer

from model import Harold, build_model


SamplingMode = Literal["argmax", "sample", "confidence"]


class HaroldSampler:
    """
    Sampler per Harold v0.5 — Flow Matching ODE integration + CFG.

    Loop di generazione (generate / generate_batch):
      1. Inizializza x_1 ~ N(0,I) per le posizioni da generare
      2. Per ogni step t da 1 a 0 con Euler ODE (10-20 step sufficienti):
           a. Forward: predice vel e ce_logits
           b. x_{t-1} = x_t - dt * vel_pred
           c. Ripristina posizioni del prompt
           d. Aggiorna self_cond e locked_mask (anchoring)
      3. Decodifica finale con ce_logits dell'ultimo step

    generate_conditioned usa CFG con due forward pass separati per step.

    Modalità di decodifica:
      "argmax"     — greedy, deterministico
      "sample"     — top-p con temperature, stocastico
      "confidence" — ancora progressivamente i token ad alta confidenza
    """

    def __init__(
        self,
        model:     Harold,
        tokenizer: PreTrainedTokenizer,
        device:    str = "cuda",
    ):
        self.model     = model.eval().to(device)
        self.tokenizer = tokenizer
        self.device    = device
        self.d_model   = model.config.d_model

    def _t_tensor(self, t_float: float, batch: int) -> torch.Tensor:
        return torch.full((batch,), t_float, dtype=torch.float32, device=self.device)

    def _sample_tokens(
        self,
        ce_logits:   torch.Tensor,
        temperature: float,
        top_p:       float,
    ) -> torch.Tensor:
        B, L, V = ce_logits.shape
        scaled  = ce_logits / max(temperature, 1e-5)
        probs   = F.softmax(scaled, dim=-1).view(B * L, V)

        sp, si = torch.sort(probs, descending=True, dim=-1)
        cum    = sp.cumsum(dim=-1)
        sp[cum - sp > top_p] = 0.0
        sp.div_(sp.sum(dim=-1, keepdim=True) + 1e-9)

        return si.gather(-1, torch.multinomial(sp, 1)).view(B, L)

    def _build_unused_mask(self) -> torch.Tensor:
        """
        Maschera token da escludere dalla decodifica.
        Con LLaMA: mascheriamo solo mask_token_id e pad aggiunto da Harold.
        """
        if hasattr(self, "_unused_mask_cache"):
            return self._unused_mask_cache

        V    = self.model.emb_vocab
        mask = torch.zeros(V, dtype=torch.bool, device=self.device)

        mask_token_id = self.model.mask_token_id
        if mask_token_id < V:
            mask[mask_token_id] = True

        pad_id = self.model.config.vocab_size
        if pad_id < V:
            mask[pad_id] = True

        self._unused_mask_cache = mask
        print(f"[sampler] Token mascherati: {int(mask.sum().item())} (mask + pad)")
        return mask

    def _mask_unused_tokens(self, ce_logits: torch.Tensor) -> torch.Tensor:
        mask = self._build_unused_mask()
        ce_logits = ce_logits.clone()
        ce_logits[..., mask] = float("-inf")
        return ce_logits

    def _apply_repetition_penalty(
        self,
        ce_logits:     torch.Tensor,
        penalty:       float = 1.5,
        generated_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if penalty == 1.0:
            return ce_logits

        B, L, V   = ce_logits.shape
        ce_logits = ce_logits.clone()
        token_ids = generated_ids if generated_ids is not None else ce_logits.argmax(dim=-1)

        for b in range(B):
            seen  = token_ids[b].unique()
            score = ce_logits[b, :, seen]
            score = torch.where(score > 0, score / penalty, score * penalty)
            ce_logits[b, :, seen] = score

        return ce_logits

    def _decode(
        self,
        ce_logits:          torch.Tensor,
        mode:               SamplingMode,
        temperature:        float,
        top_p:              float,
        mask_unused:        bool  = True,
        repetition_penalty: float = 1.5,
        generated_ids:      Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if mask_unused:
            ce_logits = self._mask_unused_tokens(ce_logits)
        if repetition_penalty != 1.0:
            ce_logits = self._apply_repetition_penalty(
                ce_logits, repetition_penalty, generated_ids
            )
        if mode == "argmax" or mode == "confidence" or temperature <= 0:
            return ce_logits.argmax(dim=-1)
        return self._sample_tokens(ce_logits, temperature, top_p)

    def _anchor_confident_tokens(
        self,
        x_next:      torch.Tensor,
        ce_logits:   torch.Tensor,
        prompt_mask: torch.Tensor,
        threshold:   float,
        t:           float,
        locked_mask: Optional[torch.Tensor] = None,
        prev_conf:   Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Confidence-guided anchoring con soglia dinamica esponenziale
        e smoothing della confidenza.
        """
        ce_logits_clean = self._mask_unused_tokens(ce_logits)
        probs      = F.softmax(ce_logits_clean, dim=-1)
        conf, toks = probs.max(dim=-1)

        conf_smooth = 0.7 * conf + 0.3 * prev_conf if prev_conf is not None else conf

        k = 4.0
        dynamic_threshold = min(1.0 - (1.0 - threshold) * math.exp(-k * (1.0 - t)), 0.99)
        to_anchor_new     = (conf_smooth > dynamic_threshold) & ~prompt_mask

        locked_mask = to_anchor_new if locked_mask is None else locked_mask | to_anchor_new

        if locked_mask.any():
            anchored_emb = self.model.token_emb(toks)
            x_next = torch.where(locked_mask.unsqueeze(-1), anchored_emb, x_next)

        return x_next, locked_mask, conf_smooth

    def _flow_step(
        self,
        x_t:       torch.Tensor,
        t:         float,
        dt:        float,
        self_cond: Optional[torch.Tensor],
        ctx_emb:   Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        [S-FM2] Un passo di integrazione ODE con Euler (Flow Matching).

        x_{t-dt} = x_t - dt * vel_pred

        Con traiettorie lineari, Euler è esatto sulla traiettoria retta —
        l'errore deriva solo dalla non-linearità del modello, non dal solver.

        Ritorna (x_next, ce_logits, vel_pred).
        """
        B        = x_t.shape[0]
        t_tensor = self._t_tensor(t, B)

        vel_pred, ce_logits, _ = self.model(
            x_t, t_tensor, self_cond=self_cond, ctx_emb=ctx_emb
        )

        x_next = x_t - dt * vel_pred
        return x_next, ce_logits, vel_pred

    def _flow_step_cfg(
        self,
        x_t:       torch.Tensor,
        t:         float,
        dt:        float,
        self_cond: Optional[torch.Tensor],
        ctx_emb:   torch.Tensor,
        cfg_scale: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        [S-FM3] Un passo ODE con CFG guidance.

        vel_guided = vel_uncond + cfg_scale * (vel_cond - vel_uncond)
        x_{t-dt}   = x_t - dt * vel_guided

        Ritorna (x_next, ce_logits_cond, vel_guided).
        """
        B        = x_t.shape[0]
        t_tensor = self._t_tensor(t, B)

        vel_cond,   ce_logits, _ = self.model(
            x_t, t_tensor, self_cond=self_cond, ctx_emb=ctx_emb
        )
        vel_uncond, _, _ = self.model(
            x_t, t_tensor, self_cond=self_cond, ctx_emb=None
        )

        vel_guided = vel_uncond + cfg_scale * (vel_cond - vel_uncond)
        x_next     = x_t - dt * vel_guided
        return x_next, ce_logits, vel_guided

    def _encode_context(self, context: str) -> torch.Tensor:
        """Mean pooling degli embedding del contesto. Returns: ctx_emb (1, D)"""
        tokenized = self.tokenizer(
            context, return_tensors="pt",
            truncation=True, max_length=128,
        )
        ids = tokenized.input_ids.to(self.device)

        def _extract_int(val: Any) -> int:
            if val is None:
                return 0
            if isinstance(val, (int, str)):
                try:
                    return int(val)
                except (ValueError, TypeError):
                    return 0
            if isinstance(val, (list, tuple)) and len(val) > 0:
                return _extract_int(val[0])
            return 0

        pad_id   = _extract_int(self.tokenizer.pad_token_id)
        ids      = ids.long()
        pad_mask = (ids != pad_id).to(torch.float32)
        emb      = self.model.token_emb(ids)
        n        = pad_mask.sum(dim=1, keepdim=True).clamp(min=1)
        return (emb * pad_mask.unsqueeze(-1)).sum(dim=1) / n

    @torch.no_grad()
    def generate(
        self,
        prompt:               str          = "",
        gen_len:              int          = 64,
        steps:                int          = 20,
        mode:                 SamplingMode = "argmax",
        temperature:          float        = 1.0,
        top_p:                float        = 0.9,
        confidence_threshold: float        = 0.7,
        anchor_every:         int          = 1,
        use_self_cond:        bool         = True,
        verbose:              bool         = False,
    ) -> str:
        """
        Genera testo con Flow Matching ODE (Euler, 20 step default).

        Args:
            prompt:               testo iniziale (posizioni fissate)
            gen_len:              lunghezza totale (prompt + generazione)
            steps:                passi ODE Euler (default 20)
            mode:                 "argmax" | "sample" | "confidence"
            temperature:          per mode="sample"
            top_p:                nucleus threshold per mode="sample"
            confidence_threshold: soglia base per mode="confidence"
            anchor_every:         ogni quanti step applicare l'anchoring
            use_self_cond:        passa hint al passo successivo
            verbose:              stampa decodifica intermedia ogni 5 step
        """
        prompt_ids = (
            self.tokenizer.encode(
                prompt, add_special_tokens=True,
                truncation=True, max_length=gen_len,
            ) if prompt else []
        )
        prompt_len = len(prompt_ids)

        # Inizializza da rumore puro — t=1
        x_t = torch.randn(1, gen_len, self.d_model, device=self.device)
        if prompt_len > 0:
            prompt_tensor       = torch.tensor(prompt_ids, dtype=torch.long, device=self.device)
            x_t[0, :prompt_len] = self.model.token_emb(prompt_tensor)

        prompt_mask = torch.zeros(1, gen_len, dtype=torch.bool, device=self.device)
        prompt_mask[0, :prompt_len] = True

        self_cond:   Optional[torch.Tensor] = None
        locked_mask: Optional[torch.Tensor] = None
        prev_conf:   Optional[torch.Tensor] = None
        ce_logits:   Optional[torch.Tensor] = None

        # [S-FM2] Integrazione ODE da t=1 a t=0
        ts = [1.0 - i / steps for i in range(steps)]
        dt = 1.0 / steps

        for step, t in enumerate(ts):
            x_next, ce_logits, vel_cur = self._flow_step(
                x_t, t, dt, self_cond
            )

            x_next = torch.where(prompt_mask.unsqueeze(-1), x_t, x_next)

            if mode == "confidence" and step % anchor_every == 0:
                x_next, locked_mask, prev_conf = self._anchor_confident_tokens(
                    x_next, ce_logits, prompt_mask,
                    threshold=confidence_threshold, t=t,
                    locked_mask=locked_mask, prev_conf=prev_conf,
                )

            if use_self_cond:
                clean_logits = self._mask_unused_tokens(ce_logits)
                clean_logits = self._apply_repetition_penalty(clean_logits, penalty=1.5)
                clean_ids    = clean_logits.argmax(dim=-1)
                # [S-FM4] self_cond da vel_cur invece di eps_pred
                self_cond    = vel_cur.mean(dim=1).detach()

            x_t = x_next

            if verbose and step % 5 == 0:
                tokens     = self._mask_unused_tokens(ce_logits).argmax(dim=-1)[0].tolist()
                text       = self.tokenizer.decode(tokens, skip_special_tokens=True)
                n_anchored = 0
                if mode == "confidence" and locked_mask is not None:
                    n_anchored = int((locked_mask[0] & ~prompt_mask[0]).sum().item())
                print(f"  step {step+1:3d}/{steps} | t={t:.3f} | anchored={n_anchored} | {text[:70]}")

        assert ce_logits is not None
        token_ids = self._decode(ce_logits, mode, temperature, top_p)[0]
        return cast(str, self.tokenizer.decode(token_ids.tolist(), skip_special_tokens=True))

    @torch.no_grad()
    def generate_batch(
        self,
        prompts:              List[str],
        gen_len:              int          = 64,
        steps:                int          = 20,
        mode:                 SamplingMode = "argmax",
        temperature:          float        = 1.0,
        top_p:                float        = 0.9,
        confidence_threshold: float        = 0.7,
        anchor_every:         int          = 1,
        use_self_cond:        bool         = True,
    ) -> List[str]:
        """Versione batch di generate — ODE vettorializzato su B sequenze."""
        B   = len(prompts)
        enc = self.tokenizer(
            prompts, padding="max_length", truncation=True,
            max_length=gen_len, return_tensors="pt",
        )
        input_ids      = cast(torch.Tensor, enc["input_ids"]).to(self.device)
        attention_mask = cast(torch.Tensor, enc["attention_mask"]).to(self.device).bool()

        # Inizializza da rumore puro, poi fixa le posizioni del prompt
        x_t = torch.randn(B, gen_len, self.d_model, device=self.device)
        x_t[attention_mask] = self.model.token_emb(input_ids[attention_mask])

        prompt_mask  = attention_mask
        self_cond:   Optional[torch.Tensor] = None
        locked_mask: Optional[torch.Tensor] = None
        prev_conf:   Optional[torch.Tensor] = None
        ce_logits:   Optional[torch.Tensor] = None

        ts = [1.0 - i / steps for i in range(steps)]
        dt = 1.0 / steps

        for step, t in enumerate(ts):
            x_next, ce_logits, vel_cur = self._flow_step(x_t, t, dt, self_cond)

            x_next = torch.where(prompt_mask.unsqueeze(-1), x_t, x_next)

            if mode == "confidence" and step % anchor_every == 0:
                x_next, locked_mask, prev_conf = self._anchor_confident_tokens(
                    x_next, ce_logits, prompt_mask,
                    threshold=confidence_threshold, t=t,
                    locked_mask=locked_mask, prev_conf=prev_conf,
                )

            if use_self_cond:
                clean_logits = self._mask_unused_tokens(ce_logits)
                clean_ids    = clean_logits.argmax(dim=-1)
                self_cond    = vel_cur.mean(dim=1).detach()

            x_t = x_next

        assert ce_logits is not None
        token_ids_batch = self._decode(ce_logits, mode, temperature, top_p)

        return [
            cast(str, self.tokenizer.decode(token_ids_batch[b].tolist(), skip_special_tokens=True))
            for b in range(B)
        ]

    @torch.no_grad()
    def generate_conditioned(
        self,
        context:            str,
        gen_len:            int   = 64,
        steps:              int   = 20,
        cfg_scale:          float = 3.0,
        mode:               SamplingMode = "argmax",
        temperature:        float = 1.0,
        top_p:              float = 0.9,
        repetition_penalty: float = 1.5,
        use_self_cond:      bool  = True,
        verbose:            bool  = False,
    ) -> str:
        """
        Genera testo condizionato usando CFG + Flow Matching ODE.

        [S-FM3] Ogni step fa due forward pass (cond + uncond) per CFG.
        Compatibile con generate() — stessa qualità, aggiunge guidance.
        """
        ctx_emb   = self._encode_context(context)
        x_t       = torch.randn(1, gen_len, self.d_model, device=self.device)
        self_cond: Optional[torch.Tensor] = None
        ce_logits: Optional[torch.Tensor] = None

        ts = [1.0 - i / steps for i in range(steps)]
        dt = 1.0 / steps

        for step, t in enumerate(ts):
            x_t, ce_logits, vel_guided = self._flow_step_cfg(
                x_t, t, dt, self_cond, ctx_emb, cfg_scale
            )

            if use_self_cond:
                clean_logits = self._mask_unused_tokens(ce_logits)
                clean_logits = self._apply_repetition_penalty(clean_logits, 1.5)
                self_cond    = vel_guided.mean(dim=1).detach()

            if verbose and step % 5 == 0:
                tokens = self._mask_unused_tokens(ce_logits).argmax(dim=-1)[0].tolist()
                text   = self.tokenizer.decode(tokens, skip_special_tokens=False)
                print(f"  step {step+1:3d}/{steps} | t={t:.3f} | cfg={cfg_scale} | {text[:70]}")

        assert ce_logits is not None
        intermediate = self._mask_unused_tokens(ce_logits).argmax(dim=-1)
        token_ids    = self._decode(
            ce_logits, mode, temperature, top_p,
            repetition_penalty=repetition_penalty,
            generated_ids=intermediate,
        )[0]

        return cast(str, self.tokenizer.decode(token_ids.tolist(), skip_special_tokens=True))


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if len(sys.argv) < 2:
        print("Usage: python sampler.py <checkpoint_path>")
        sys.exit(1)

    ckpt_path = sys.argv[1]
    state     = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_cfg = state["model_cfg"]
    model     = build_model(model_cfg)
    model.load_state_dict(state["model_state"])
    print(f"Checkpoint caricato: {ckpt_path}")

    # [S-FM5] LLaMA tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    sampler = HaroldSampler(model, tokenizer, device=device)

    prompt = "Once upon a time"
    print(f"\nPrompt: {prompt!r}\n")

    out = sampler.generate(
        prompt=prompt, gen_len=64, steps=20,
        mode="argmax", use_self_cond=True, verbose=True,
    )
    print(f"[argmax] {out}\n")

    out = sampler.generate(
        prompt=prompt, gen_len=64, steps=20,
        mode="confidence", confidence_threshold=0.7,
        use_self_cond=True,
    )
    print(f"[confidence] {out}\n")

    for temp in (0.8, 1.0, 1.3):
        out = sampler.generate(
            prompt=prompt, gen_len=64, steps=20,
            mode="sample", temperature=temp, top_p=0.9,
            use_self_cond=True,
        )
        print(f"[sample t={temp}] {out}")

    print("\n--- CFG Conditioned ---")
    for cfg_prompt in [
        "What is the capital of France?",
        "Explain what a neural network is.",
    ]:
        print(f"\n[prompt] {cfg_prompt}")
        for scale in (1.0, 3.0, 5.0):
            out = sampler.generate_conditioned(
                context=cfg_prompt, gen_len=64, steps=20,
                cfg_scale=scale, mode="argmax",
                repetition_penalty=1.3,
            )
            print(f"  [cfg={scale:.1f}] {out}")

    print("\n--- Batch ---")
    outs = sampler.generate_batch(
        prompts=["Once upon a time", "The quick brown fox"],
        gen_len=64, steps=20, mode="confidence",
        confidence_threshold=0.7,
    )
    for p, o in zip(["Once upon a time", "The quick brown fox"], outs):
        print(f"[{p!r}] {o}")