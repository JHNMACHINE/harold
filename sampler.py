"""
Harold v0.4 — sampler.py
============================
Sampler per VP-SDE continuous diffusion.

Novità v0.4:
  [S8] DPM-Solver++ al posto di Euler-Maruyama
       - 2° ordine: stesso risultato in 20 step invece di 100 (5× più veloce)
       - _dpm_step() sostituisce _reverse_sde_step() nel loop principale
       - _reverse_sde_step() rimane disponibile per CFG (generate_conditioned)
         dove serve accesso diretto a eps separati cond/uncond
  [S9] Anchoring più robusto:
       - soglia dinamica esponenziale invece di lineare — meno token
         vengono ancorati in anticipo quando t è alto
       - smoothing della confidenza con media pesata tra passo corrente
         e precedente — riduce oscillazioni durante il sampling stocastico
       - locked_mask: una volta ancorato, un token non viene più rilasciato

Fix rispetto a v0.3:
  [FIX S1] Tuple_ import spostato in cima — era in fondo al file, dopo l'uso
  [FIX S2] _emb_to_tokens_ce rimosso, si usa ce_logits direttamente
  [FIX S3] self_cond aggiornato con eps_pred.mean invece di x_next.mean
  [FIX S4] top-k dinamico MoE ripristinato in inferenza
  [FIX S5] generate() e generate_batch() logica decodifica unificata
  [FIX S6] temperature=1.0 con top_p=1.0 ora fa argmax
  [FIX S7] Prompt injection con embedding reali, non rumore
"""

import math
import sys
import torch
import torch.nn.functional as F
from typing import Literal, Optional, List, Tuple
from transformers import PreTrainedTokenizer, AutoTokenizer

from model import Harold, build_model

# Import opzionale di SFTConfig — necessario per caricare checkpoint SFT
# senza questo torch.load fallisce perché SFTConfig è serializzata nel checkpoint
try:
    from train_sft import SFTConfig  # noqa: F401
except ImportError:
    pass


SamplingMode = Literal["argmax", "sample", "confidence"]


class HaroldSampler:
    """
    Sampler per Harold v0.4 — DPM-Solver++ (2° ordine) + Euler-Maruyama CFG.

    Loop di generazione (generate / generate_batch):
      1. Inizializza x_1 ~ N(0,I) per le posizioni da generare
         (le posizioni del prompt sono fissate ai loro embedding)
      2. Per ogni step t da 1 a 0 con DPM-Solver++ (20 step sufficienti):
           a. Forward: predice ε e ce_logits
           b. Aggiorna x con lo schema DPM-Solver++ 2° ordine
           c. Ripristina posizioni del prompt (invarianti)
           d. Aggiorna self_cond e locked_mask (anchoring)
      3. Decodifica finale con ce_logits (argmax o top-p sampling)

    generate_conditioned usa Euler-Maruyama perché richiede due forward
    pass separati (cond + uncond) che non si integrano nativamente con
    il multi-step DPM-Solver++.

    Modalità di decodifica:
      "argmax"     — greedy, deterministico
      "sample"     — top-p con temperature, stocastico
      "confidence" — ancora progressivamente i token ad alta confidenza
                     durante il loop (ibrido continuo/discreto)
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

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _t_tensor(self, t_float: float, batch: int) -> torch.Tensor:
        return torch.full((batch,), t_float, dtype=torch.float32, device=self.device)

    def _sample_tokens(
        self,
        ce_logits:   torch.Tensor,  # (B, L, V)
        temperature: float,
        top_p:       float,
    ) -> torch.Tensor:
        """Top-p sampling con temperature. Ritorna token_ids (B, L)."""
        B, L, V = ce_logits.shape
        scaled  = ce_logits / max(temperature, 1e-5)
        probs   = F.softmax(scaled, dim=-1).view(B * L, V)

        sp, si  = torch.sort(probs, descending=True, dim=-1)
        cum     = sp.cumsum(dim=-1)
        sp[cum - sp > top_p] = 0.0
        sp.div_(sp.sum(dim=-1, keepdim=True) + 1e-9)

        return si.gather(-1, torch.multinomial(sp, 1)).view(B, L)

    def _build_unused_mask(self) -> torch.Tensor:
        """
        Costruisce una maschera dei token da escludere dalla decodifica.

        GPT-2 non ha token [unused] o token speciali problematici —
        il vocabolario è byte-level BPE quindi tutti i token sono validi.

        Maschiamo solo:
        - Il token di padding aggiunto da Harold (idx = vocab_size = 50257)
        - <|endoftext|> (idx 50256) usato come mask token durante la diffusione
          ma che non deve apparire nell'output finale

        Questo è molto più semplice della versione BERT che richiedeva
        una allowlist ASCII per filtrare [unused], [CLS], [SEP], ecc.
        """
        if hasattr(self, "_unused_mask_cache"):
            return self._unused_mask_cache

        V    = self.model.emb_vocab
        mask = torch.zeros(V, dtype=torch.bool, device=self.device)

        # Maschera il mask token (usato durante la diffusion, non nell'output)
        mask_token_id = self.model.mask_token_id
        if mask_token_id < V:
            mask[mask_token_id] = True

        # Maschera il padding token aggiunto da Harold (vocab_size)
        pad_id = self.model.config.vocab_size
        if pad_id < V:
            mask[pad_id] = True

        self._unused_mask_cache = mask
        n_masked = int(mask.sum().item())
        print(f"[sampler] Token mascherati: {n_masked} (mask + pad)")
        return mask

    def _mask_unused_tokens(self, ce_logits: torch.Tensor) -> torch.Tensor:
        """
        Azzera i logit dei token [unused] e subword rumorosi.
        Usa il tokenizer per identificare tutti i token indesiderati.
        """
        mask = self._build_unused_mask()                    # (V,) bool
        ce_logits = ce_logits.clone()
        ce_logits[..., mask] = float("-inf")
        return ce_logits

    def _apply_repetition_penalty(
        self,
        ce_logits:     torch.Tensor,   # (B, L, V)
        penalty:       float = 1.5,
        generated_ids: Optional[torch.Tensor] = None,  # (B, L) token già generati
    ) -> torch.Tensor:
        """
        Penalizza i token già presenti nella sequenza generata.
        penalty > 1.0: riduce la probabilità dei token già visti
        penalty = 1.0: nessun effetto

        Se generated_ids è fornito, penalizza quei token specifici.
        Altrimenti usa argmax come proxy dei token correnti.

        Implementazione standard (HuggingFace style):
          logit > 0 → logit / penalty
          logit < 0 → logit * penalty
        """
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
        """Decodifica ce_logits in token_ids (B, L) secondo la modalità."""
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
        x_next:      torch.Tensor,            # (B, L, D) stato SDE aggiornato
        ce_logits:   torch.Tensor,            # (B, L, V) logit del passo corrente
        prompt_mask: torch.Tensor,            # (B, L) bool — posizioni del prompt
        threshold:   float,                   # soglia base di confidenza [0,1]
        t:           float,                   # timestep corrente in (0,1]
        locked_mask: Optional[torch.Tensor] = None,  # (B, L) bool — già ancorati
        prev_conf:   Optional[torch.Tensor] = None,  # (B, L) confidenza passo prec.
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Confidence-guided anchoring migliorato (v0.4).

        Miglioramenti rispetto a v0.3:
          1. Soglia dinamica esponenziale:
               dyn = 1 - (1 - threshold) * exp(-k * (1 - t))
             Cresce più lentamente a t alto → meno falsi ancoraggi precoci.
             A t=1 (rumore puro): dyn ≈ threshold (quasi niente ancorato)
             A t=0 (segnale pulito): dyn → 0.99 (quasi tutto ancorato)

          2. Smoothing della confidenza:
               conf_smooth = 0.7 * conf_cur + 0.3 * prev_conf
             Riduce le oscillazioni nella stima della confidenza durante
             il sampling stocastico — evita di ancorare token per un picco
             accidentale di confidenza in un singolo step.

          3. Locked mask — once locked, always locked:
             I token già ancorati nei passi precedenti vengono mantenuti
             ancorati indipendentemente dalla confidenza corrente. Questo
             previene il "flickering" (token che entrano ed escono dall'anchor).

        Args:
            x_next:      stato SDE dopo il passo corrente
            ce_logits:   logit ausiliari del modello
            prompt_mask: maschera del prompt (mai ancorato)
            threshold:   soglia base di confidenza
            t:           timestep corrente
            locked_mask: posizioni già ancorate nei passi precedenti
            prev_conf:   confidenza dal passo precedente (per smoothing)

        Returns:
            (x_next aggiornato, locked_mask aggiornata, conf_smooth corrente)
        """
        ce_logits_clean = self._mask_unused_tokens(ce_logits)
        probs      = F.softmax(ce_logits_clean, dim=-1)   # (B, L, V)
        conf, toks = probs.max(dim=-1)                     # (B, L)

        # Smoothing: media pesata con la confidenza del passo precedente
        if prev_conf is not None:
            conf_smooth = 0.7 * conf + 0.3 * prev_conf
        else:
            conf_smooth = conf

        # Soglia esponenziale: lenta a t alto, rapida a t basso
        # k=4 calibrato per 20 step DPM-Solver++ (threshold≈0.7 tipico)
        k = 4.0
        dynamic_threshold = min(1.0 - (1.0 - threshold) * math.exp(-k * (1.0 - t)), 0.99)

        # Token da ancorare: confidenza sopra soglia E non nel prompt
        to_anchor_new = (conf_smooth > dynamic_threshold) & ~prompt_mask

        # Aggiorna la locked mask — mai rilasciare un token già ancorato
        if locked_mask is None:
            locked_mask = to_anchor_new
        else:
            locked_mask = locked_mask | to_anchor_new

        if locked_mask.any():
            anchored_emb = self.model.token_emb(toks)     # (B, L, D)
            x_next = torch.where(locked_mask.unsqueeze(-1), anchored_emb, x_next)

        return x_next, locked_mask, conf_smooth

    # ── Reverse SDE step (Euler-Maruyama) ────────────────────────────────────

    def _reverse_sde_step(
        self,
        x_t:       torch.Tensor,           # (B, L, D)
        t:         float,                   # timestep corrente in (0,1]
        dt:        float,                   # passo negativo (-1/steps)
        self_cond: Optional[torch.Tensor], # (B, D) detached o None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Un passo di reverse SDE (Euler-Maruyama).

        Reverse SDE (Anderson 1982):
          dx = [-0.5*β(t)*x - β(t)*score(x,t)] dt + √β(t) dW

        score(x,t) = ∇ log p_t(x) ≈ -ε_θ(x,t) / σ(t)

        Ritorna (x_next, ce_logits, eps_pred).
        """
        B        = x_t.shape[0]
        t_tensor = self._t_tensor(t, B)

        # Forward del modello
        # [FIX S4] t_normalized viene calcolato internamente in Harold.forward
        #          dal valore di t — il top-k dinamico del MoE è automatico
        eps_pred, ce_logits, _ = self.model(x_t, t_tensor, self_cond=self_cond)

        # Parametri SDE al tempo t
        beta_t           = self.model.schedule.get_beta(t_tensor)           # (B,)
        _, sigma_t       = self.model.schedule.get_alpha_sigma(t_tensor)    # (B,)

        # Score: ∇ log p_t(x) = -ε / σ(t)
        score     = -eps_pred / sigma_t.view(-1, 1, 1).clamp(min=1e-8)

        # Reverse SDE drift e diffusion
        b         = beta_t.view(-1, 1, 1)
        drift     = -0.5 * b * x_t - b * score
        diffusion = b.sqrt()

        # Nessun rumore all'ultimo step (t ≈ 0) per decodifica stabile
        noise  = torch.randn_like(x_t) if t > 1e-5 else torch.zeros_like(x_t)
        x_next = x_t + drift * dt + diffusion * noise * math.sqrt(abs(dt))

        return x_next, ce_logits, eps_pred

    # ── DPM-Solver++ step (2° ordine) ────────────────────────────────────────

    def _dpm_step(
        self,
        x_t:        torch.Tensor,            # (B, L, D) stato corrente
        t:          float,                   # timestep corrente in (0,1]
        t_next:     float,                   # timestep successivo in [0,1)
        self_cond:  Optional[torch.Tensor],  # (B, D) hint o None
        eps_prev:   Optional[torch.Tensor] = None,  # (B, L, D) pred. passo prec.
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        DPM-Solver++ 2° ordine (Lu et al. 2022, Algo. 2).

        Vantaggi su Euler-Maruyama:
          - 2° ordine: errore O(h²) invece di O(h) → stessa qualità in
            ~1/5 degli step (20 invece di 100)
          - Deterministico: nessun termine di rumore → più stabile e
            riproducibile (no dW)

        Schema:
          Primo step (o eps_prev=None):
            x_{t-1} = (σ_{t-1}/σ_t) * x_t
                    - α_{t-1} * (e^{-h} - 1) * eps_θ(x_t, t)
            → equivalente a Euler su log-SNR (1° ordine)

          Step successivi (eps_prev disponibile):
            D = (1 + 1/(2r)) * eps_cur - (1/(2r)) * eps_prev
            x_{t-1} = (σ_{t-1}/σ_t) * x_t - α_{t-1} * (e^{-h} - 1) * D
            dove r = h_{prev}/h  (rapporto degli step in log-SNR)
            → 2° ordine con correzione multistep

        Note:
          - Lavora in log-SNR (lambda = log α/σ) come da paper
          - eps_prev viene aggiornato a eps_cur per il passo successivo

        Args:
            x_t:       stato corrente
            t:         timestep corrente
            t_next:    timestep successivo (< t)
            self_cond: self-conditioning hint
            eps_prev:  eps_pred del passo precedente (None al primo step)

        Returns:
            (x_next, ce_logits, eps_cur)
        """
        B        = x_t.shape[0]
        t_tensor = self._t_tensor(t,      B)
        tn_tensor= self._t_tensor(t_next, B)

        eps_cur, ce_logits, _ = self.model(x_t, t_tensor, self_cond=self_cond)

        alpha_t,  sigma_t  = self.model.schedule.get_alpha_sigma(t_tensor)
        alpha_tn, sigma_tn = self.model.schedule.get_alpha_sigma(tn_tensor)

        # log-SNR: λ = log(α/σ)
        lam_t  = torch.log(alpha_t  / sigma_t.clamp(min=1e-8)).mean().item()
        lam_tn = torch.log(alpha_tn / sigma_tn.clamp(min=1e-8)).mean().item()
        h      = lam_tn - lam_t   # negativo: λ decresce con t

        # Scalari per update — shape (B,) → (B,1,1)
        s_ratio  = (sigma_tn / sigma_t.clamp(min=1e-8)).view(-1, 1, 1)
        a_tn     = alpha_tn.view(-1, 1, 1)
        exp_mh   = math.exp(-h)   # e^{-h}

        if eps_prev is None:
            # Primo step — 1° ordine
            D = eps_cur
        else:
            # Step successivi — 2° ordine (r = h_prev/h ≈ 1 per step uniformi)
            r = 0.5   # semplificazione: assume step uniformi in log-SNR
            D = (1.0 + 1.0 / (2.0 * r)) * eps_cur - (1.0 / (2.0 * r)) * eps_prev

        x_next = s_ratio * x_t - a_tn * (exp_mh - 1.0) * D

        return x_next, ce_logits, eps_cur

    # ── Generate ─────────────────────────────────────────────────────────────

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
        Genera testo con DPM-Solver++ (2° ordine).

        Args:
            prompt:               testo iniziale (posizioni fissate, non modificate)
            gen_len:              lunghezza totale della sequenza (prompt + generazione)
            steps:                passi DPM-Solver++ — default 20 (era 100 con E-M)
            mode:                 "argmax" | "sample" | "confidence"
            temperature:          temperatura per mode="sample"
            top_p:                nucleus threshold per mode="sample"
            confidence_threshold: soglia base per mode="confidence" [0,1]
                                  valori tipici: 0.6-0.8
            anchor_every:         ogni quanti step applicare l'anchoring
                                  (default 1 — DPM ha meno step quindi anchor sempre)
            use_self_cond:        passa hint al passo successivo
            verbose:              stampa decodifica intermedia ogni 5 step
        """
        # ── Tokenizza il prompt ─────────────────────────────────────────────
        prompt_ids = (
            self.tokenizer.encode(
                prompt, add_special_tokens=True,
                truncation=True, max_length=gen_len,
            ) if prompt else []
        )
        prompt_len = len(prompt_ids)

        # ── Inizializza da rumore puro ──────────────────────────────────────
        x_t = torch.zeros(1, gen_len, self.d_model, device=self.device)
        if prompt_len < gen_len:
            x_t[0, prompt_len:] = torch.randn(
                gen_len - prompt_len, self.d_model, device=self.device
            )

        if prompt_len > 0:
            prompt_tensor       = torch.tensor(prompt_ids, dtype=torch.long, device=self.device)
            x_t[0, :prompt_len] = self.model.token_emb(prompt_tensor)

        prompt_mask = torch.zeros(1, gen_len, dtype=torch.bool, device=self.device)
        prompt_mask[0, :prompt_len] = True

        # ── Loop DPM-Solver++ ────────────────────────────────────────────────
        self_cond:   Optional[torch.Tensor] = None
        eps_prev:    Optional[torch.Tensor] = None
        locked_mask: Optional[torch.Tensor] = None
        prev_conf:   Optional[torch.Tensor] = None

        for step in range(steps):
            t      = 1.0 - step / steps
            t_next = 1.0 - (step + 1) / steps

            x_next, ce_logits, eps_cur = self._dpm_step(
                x_t, t, t_next, self_cond, eps_prev
            )

            # Ripristina le posizioni del prompt
            x_next = torch.where(prompt_mask.unsqueeze(-1), x_t, x_next)

            # Confidence anchoring con locked mask e smoothing
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
                self_cond    = self.model.token_emb(clean_ids).mean(dim=1).detach()

            eps_prev = eps_cur
            x_t      = x_next

            if verbose and step % 5 == 0:
                tokens = clean_logits.argmax(dim=-1)[0].tolist() if use_self_cond else \
                         self._mask_unused_tokens(ce_logits).argmax(dim=-1)[0].tolist()
                text   = self.tokenizer.decode(tokens, skip_special_tokens=True)
                n_anchored = 0
                if mode == "confidence" and locked_mask is not None:
                    n_anchored = int((locked_mask[0] & ~prompt_mask[0]).sum().item())
                print(f"  step {step+1:3d}/{steps} | t={t:.3f} | anchored={n_anchored} | {text[:70]}")

        # ── Decodifica finale ───────────────────────────────────────────────
        t_final        = self._t_tensor(1.0 / steps, batch=1)
        _, ce_final, _ = self.model(x_t, t_final, self_cond=self_cond)
        token_ids      = self._decode(ce_final, mode, temperature, top_p)[0]

        return self.tokenizer.decode(token_ids.tolist(), skip_special_tokens=True)  # type: ignore

    @torch.no_grad()
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
        """
        Versione batch di generate — DPM-Solver++ vettorializzato su B sequenze.
        Stessa logica di generate() ma processa tutti i prompt in parallelo.
        """
        B   = len(prompts)
        enc = self.tokenizer(
            prompts, padding="max_length", truncation=True,
            max_length=gen_len, return_tensors="pt",
        )
        input_ids      = enc["input_ids"].to(self.device)             # type: ignore
        attention_mask = enc["attention_mask"].to(self.device).bool() # type: ignore

        x_t = torch.zeros(B, gen_len, self.d_model, device=self.device)
        noise_mask = ~attention_mask
        x_t[noise_mask] = torch.randn(
            int(noise_mask.sum().item()), self.d_model, device=self.device
        )
        x_t[attention_mask] = self.model.token_emb(input_ids[attention_mask])

        prompt_mask  = attention_mask
        self_cond:   Optional[torch.Tensor] = None
        eps_prev:    Optional[torch.Tensor] = None
        locked_mask: Optional[torch.Tensor] = None
        prev_conf:   Optional[torch.Tensor] = None

        ts = [1.0 - i / steps for i in range(steps)]

        for step, (t, t_next) in enumerate(zip(ts, ts[1:] + [1.0 / steps])):

            x_next, ce_logits, eps_cur = self._dpm_step(
                x_t, t, t_next, self_cond, eps_prev
            )
            eps_prev = eps_cur

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
                self_cond    = self.model.token_emb(clean_ids).mean(dim=1).detach()

            x_t = x_next

        t_final         = self._t_tensor(1.0 / steps, batch=B)
        _, ce_final, _  = self.model(x_t, t_final, self_cond=self_cond)
        token_ids_batch = self._decode(ce_final, mode, temperature, top_p)

        return [
            self.tokenizer.decode(token_ids_batch[b].tolist(), skip_special_tokens=True)
            for b in range(B)
        ]  # type: ignore

    # ── CFG conditioned generation ───────────────────────────────────────────

    def _to_tensor(self, tokenized, device=None) -> torch.Tensor:
        """Converte l'output del tokenizer in un tensor sul dispositivo corretto."""
        # Estrai input_ids
        if hasattr(tokenized, "input_ids"):
            ids = tokenized.input_ids
        elif isinstance(tokenized, dict):
            ids = tokenized.get("input_ids")
        else:
            ids = tokenized
        
        # Converti in tensor se necessario
        if not isinstance(ids, torch.Tensor):
            ids = torch.tensor(ids)
        
        # Sposta sul dispositivo
        if device is not None and hasattr(ids, "to"):
            ids = ids.to(device)
        
        return ids

    def _encode_context(self, context: str) -> torch.Tensor:
        """
        Encoda un testo di contesto come mean pooling degli embedding.
        Identico a encode_context() in train_sft.py.

        Returns:
            ctx_emb: (1, D)
        """
        tokenized = self.tokenizer(
            context, return_tensors="pt",
            truncation=True, max_length=128,
        )
        
        ids = self._to_tensor(tokenized, self.device)  # (1, L)
        
        pad_mask = (ids != 0).float()                         # (1, L)
        emb      = self.model.token_emb(ids)                  # (1, L, D)
        n        = pad_mask.sum(dim=1, keepdim=True).clamp(min=1)
        return (emb * pad_mask.unsqueeze(-1)).sum(dim=1) / n  # (1, D)
    @torch.no_grad()
    def generate_conditioned(
        self,
        context:            str,
        gen_len:            int   = 64,
        steps:              int   = 50,
        cfg_scale:          float = 3.0,
        mode:               SamplingMode = "argmax",
        temperature:        float = 1.0,
        top_p:              float = 0.9,
        repetition_penalty: float = 1.5,
        use_self_cond:      bool  = True,
        verbose:            bool  = False,
    ) -> str:
        """
        Genera testo condizionato sul contesto usando CFG.

        A differenza di generate(), qui:
          - Il contesto NON viene iniettato come token nella sequenza
          - Il contesto viene encodato come ctx_emb e usato per guidare la SDE
          - Ogni step fa DUE forward pass: uno condizionato, uno no
          - La guida è: ε_guided = ε_uncond + cfg_scale * (ε_cond - ε_uncond)

        Usare questo metodo per testare il SFT con CFG.
        Usare generate() per generazione non condizionata (pretraining).

        Args:
            context:     prompt conversazionale (es. "What is the capital of France?")
            gen_len:     lunghezza della risposta
            steps:       passi di integrazione (50 è sufficiente per test)
            cfg_scale:   forza del conditioning. 1.0 = nessuna guida (come uncond).
                         Valori tipici: 2.0-5.0. Troppo alto → testo ripetitivo.
            mode:        modalità decodifica finale
            temperature: per mode="sample"
            top_p:       per mode="sample"
            use_self_cond: usa self-conditioning
            verbose:     stampa progresso ogni 10 step
        """
        # Encoda il contesto — usato per entrambi i forward pass
        ctx_emb   = self._encode_context(context)  # (1, D)

        # Parti da rumore puro — nessun token della sequenza è noto
        x_t       = torch.randn(1, gen_len, self.d_model, device=self.device)
        self_cond: Optional[torch.Tensor] = None
        dt        = -1.0 / steps

        for step in range(steps):
            t        = 1.0 - step / steps
            t_tensor = self._t_tensor(t, batch=1)

            # Forward condizionato (con ctx_emb)
            eps_cond, ce_logits, _ = self.model(
                x_t, t_tensor, self_cond=self_cond, ctx_emb=ctx_emb
            )
            # Forward incondizionato (ctx_emb=None)
            eps_uncond, _, _ = self.model(
                x_t, t_tensor, self_cond=self_cond, ctx_emb=None
            )

            # CFG: interpola tra condizionato e incondizionato
            eps = eps_uncond + cfg_scale * (eps_cond - eps_uncond)

            # Parametri SDE
            beta_t        = self.model.schedule.get_beta(t_tensor)
            _, sigma_t    = self.model.schedule.get_alpha_sigma(t_tensor)
            score         = -eps / sigma_t.view(-1, 1, 1).clamp(min=1e-8)
            b             = beta_t.view(-1, 1, 1)
            noise         = torch.randn_like(x_t) if t > 1e-5 else torch.zeros_like(x_t)
            x_t           = x_t + (-0.5*b*x_t - b*score)*dt + b.sqrt()*noise*math.sqrt(abs(dt))

            if use_self_cond:
                # Self-cond: embedding dei token puliti, non eps_pred grezzo
                clean_logits_sc = self._mask_unused_tokens(ce_logits)
                clean_logits_sc = self._apply_repetition_penalty(clean_logits_sc, 1.5)
                clean_ids_sc    = clean_logits_sc.argmax(dim=-1)
                self_cond       = self.model.token_emb(clean_ids_sc).mean(dim=1).detach()

            if verbose and step % 10 == 0:
                tokens = self._mask_unused_tokens(ce_logits).argmax(dim=-1)[0].tolist()
                text   = self.tokenizer.decode(tokens, skip_special_tokens=False)
                print(f"  step {step+1:3d}/{steps} | t={t:.3f} | cfg={cfg_scale} | {text[:70]}")

        # Decodifica finale — forward condizionato a t≈0
        t_final           = self._t_tensor(1.0 / steps, batch=1)
        _, ce_final, _    = self.model(x_t, t_final, self_cond=self_cond, ctx_emb=ctx_emb)
        # Usa i token intermedi come riferimento per la repetition penalty
        intermediate_ids  = self._mask_unused_tokens(ce_final).argmax(dim=-1)
        token_ids         = self._decode(
            ce_final, mode, temperature, top_p,
            repetition_penalty=repetition_penalty,
            generated_ids=intermediate_ids,
        )[0]

        return self.tokenizer.decode(token_ids.tolist(), skip_special_tokens=True)  # type: ignore


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────────────────────────────────────

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

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    # GPT-2 non ha pad token di default — lo aggiungiamo
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    sampler   = HaroldSampler(model, tokenizer, device=device)

    prompt = "Once upon a time"
    print(f"\nPrompt: {prompt!r}\n")

    # Greedy
    out = sampler.generate(
        prompt=prompt, gen_len=64, steps=20,
        mode="argmax", use_self_cond=True, verbose=True,
    )
    print(f"[argmax] {out}\n")

    # Confidence-guided
    out = sampler.generate(
        prompt=prompt, gen_len=64, steps=20,
        mode="confidence", confidence_threshold=0.7, anchor_every=1,
        use_self_cond=True,
    )
    print(f"[confidence] {out}\n")

    # Top-p sampling
    for temp in (0.8, 1.0, 1.3):
        out = sampler.generate(
            prompt=prompt, gen_len=64, steps=20,
            mode="sample", temperature=temp, top_p=0.9,
            use_self_cond=True,
        )
        print(f"[sample t={temp}] {out}")

    # CFG conditioned — per testare il SFT
    # Confronta cfg_scale=1.0 (no guidance) vs 3.0 vs 5.0
    # Se il SFT funziona, le risposte devono essere tematicamente coerenti col prompt
    print("\n--- CFG Conditioned (SFT test) ---")
    cfg_prompts = [
        "What is the capital of France?",
        "Tell me a joke about programming.",
        "Explain what a neural network is in simple terms.",
    ]
    for cfg_prompt in cfg_prompts:
        print(f"\n[prompt] {cfg_prompt}")
        for scale in (1.0, 3.0, 5.0):
            out = sampler.generate_conditioned(
                context=cfg_prompt, gen_len=64, steps=50,
                cfg_scale=scale, mode="argmax",
                repetition_penalty=1.3,
            )
            print(f"  [cfg={scale:.1f}] {out}")

    # Batch non condizionato
    print("\n--- Batch ---")
    outs = sampler.generate_batch(
        prompts=["Once upon a time", "The quick brown fox"],
        gen_len=64, steps=20, mode="confidence",
        confidence_threshold=0.7, anchor_every=1,
    )
    for p, o in zip(["Once upon a time", "The quick brown fox"], outs):
        print(f"[{p!r}] {o}")