"""
Harold v3 — sampler.py
==========================
Sampler per VP-SDE continuous diffusion con reverse SDE (Euler-Maruyama).
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
    Sampler per Harold v3 — reverse SDE con Euler-Maruyama.

    Loop di generazione:
      1. Inizializza x_1 ~ N(0,I) per le posizioni da generare
         (le posizioni del prompt sono fissate ai loro embedding)
      2. Per ogni step t da 1 a 0:
           a. Forward: predice ε e ce_logits
           b. Calcola score = -ε / o(t)
           c. Integra reverse SDE: dx = drift*dt + diffusion*dW
           d. Ripristina posizioni del prompt (invarianti)
           e. Aggiorna self_cond = eps_pred.mean(dim=1)
      3. Decodifica finale con ce_logits (argmax o top-p sampling)

    Modalità di decodifica:
      "argmax"     — greedy, deterministico
      "sample"     — top-p con temperature, stocastico
      "confidence" — ancora progressivamente i token ad alta confidenza
                     durante il loop SDE (ibrido continuo/discreto)
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
        x_next:      torch.Tensor,   # (B, L, D) stato SDE aggiornato
        ce_logits:   torch.Tensor,   # (B, L, V) logit del passo corrente
        prompt_mask: torch.Tensor,   # (B, L) bool — posizioni del prompt
        threshold:   float,          # soglia di confidenza [0,1]
        t:           float,          # timestep corrente — usato per threshold dinamica
    ) -> torch.Tensor:
        """
        Confidence-guided anchoring per VP-SDE.

        Per i token con confidenza (max softmax prob) sopra la soglia,
        sostituisce l'embedding SDE con l'embedding discreto del token
        più probabile. Questi token non evolveranno più con la SDE nei
        passi successivi perché la loro posizione viene "congelata".

        La soglia è dinamica rispetto a t:
          - t alto (tanto rumore) → soglia alta → ancora solo i token
            già molto certi (pochi)
          - t basso (poco rumore) → soglia bassa → ancora sempre più token

        Questo ibrida il processo continuo VP-SDE con il decoding
        progressivo stile MDLM/MaskGIT.

        Args:
            x_next:      stato SDE dopo il passo corrente
            ce_logits:   logit ausiliari del modello
            prompt_mask: maschera del prompt (non viene mai ancorato)
            threshold:   soglia base di confidenza
            t:           timestep corrente in (0,1]

        Returns:
            x_next con le posizioni ad alta confidenza ancorate
        """
        # Confidenza = max probabilità del softmax
        # Maschera token [unused] prima di calcolare la confidenza
        ce_logits_clean = self._mask_unused_tokens(ce_logits)
        probs      = F.softmax(ce_logits_clean, dim=-1)         # (B, L, V)
        conf, toks = probs.max(dim=-1)                    # (B, L) ciascuno

        # Soglia dinamica: cresce con t, così a t alto ancora solo i token
        # veramente certi e a t basso ancora la maggioranza
        # range: [threshold, min(threshold * 2, 0.99)]
        dynamic_threshold = min(threshold + t * threshold, 0.99)

        # Maschera dei token da ancorare: alta confidenza E non nel prompt
        to_anchor = (conf > dynamic_threshold) & ~prompt_mask  # (B, L)

        if to_anchor.any():
            # Embedding discreto dei token più probabili
            anchored_emb = self.model.token_emb(toks)    # (B, L, D)
            x_next = torch.where(
                to_anchor.unsqueeze(-1),
                anchored_emb,
                x_next,
            )

        return x_next

    # ── Reverse SDE step ─────────────────────────────────────────────────────

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

    # ── Generate ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        prompt:               str          = "",
        gen_len:              int          = 64,
        steps:                int          = 100,
        mode:                 SamplingMode = "argmax",
        temperature:          float        = 1.0,
        top_p:                float        = 0.9,
        confidence_threshold: float        = 0.7,
        anchor_every:         int          = 5,
        use_self_cond:        bool         = True,
        verbose:              bool         = False,
    ) -> str:
        """
        Genera testo con reverse SDE.

        Args:
            prompt:               testo iniziale (posizioni fissate, non modificate)
            gen_len:              lunghezza totale della sequenza (prompt + generazione)
            steps:                passi Euler-Maruyama (più = qualità migliore, più lento)
            mode:                 "argmax" | "sample" | "confidence"
            temperature:          temperatura per mode="sample"
            top_p:                nucleus threshold per mode="sample"
            confidence_threshold: soglia base per mode="confidence" [0,1]
                                  valori tipici: 0.6-0.8
                                  più basso = ancora più token per step
            anchor_every:         ogni quanti step applicare l'anchoring
                                  (default 5 = ogni 5 step SDE)
            use_self_cond:        passa eps_pred.mean come hint al passo successivo
            verbose:              stampa decodifica intermedia ogni 10 step
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
        # [FIX] Solo le posizioni da generare partono da rumore gaussiano.
        # Il prompt viene iniettato SENZA rumore con la sua scala naturale.
        # Prima: torch.randn su tutta la sequenza → il rumore (std=1) schiaccia
        # gli embedding del prompt (std≈0.02), rendendo il prompt invisibile.
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

        # ── Loop reverse SDE ────────────────────────────────────────────────
        self_cond: Optional[torch.Tensor] = None
        dt = -1.0 / steps

        for step in range(steps):
            t = 1.0 - step / steps

            x_next, ce_logits, eps_pred = self._reverse_sde_step(x_t, t, dt, self_cond)

            # Ripristina le posizioni del prompt
            x_next = torch.where(prompt_mask.unsqueeze(-1), x_t, x_next)

            # Confidence anchoring: congela i token certi ogni anchor_every step
            if mode == "confidence" and step % anchor_every == 0:
                x_next = self._anchor_confident_tokens(
                    x_next, ce_logits, prompt_mask,
                    threshold=confidence_threshold, t=t,
                )

            if use_self_cond:
                # Self-cond: usa gli embedding dei token più probabili (puliti)
                # invece di eps_pred grezzo — così i token bias non inquinano
                # la traiettoria SDE attraverso il self-conditioning
                clean_logits = self._mask_unused_tokens(ce_logits)
                clean_logits = self._apply_repetition_penalty(clean_logits, penalty=1.5)
                clean_ids    = clean_logits.argmax(dim=-1)           # (B, L)
                self_cond    = self.model.token_emb(clean_ids).mean(dim=1).detach()  # (B, D)

            x_t = x_next

            if verbose and step % 10 == 0:
                tokens = clean_logits.argmax(dim=-1)[0].tolist() if use_self_cond else \
                         self._mask_unused_tokens(ce_logits).argmax(dim=-1)[0].tolist()
                text   = self.tokenizer.decode(tokens, skip_special_tokens=True)
                n_anchored = 0
                if mode == "confidence":
                    probs = F.softmax(self._mask_unused_tokens(ce_logits), dim=-1)
                    conf  = probs.max(dim=-1).values
                    dyn_t = min(confidence_threshold + t * confidence_threshold, 0.99)
                    n_anchored = int(((conf[0] > dyn_t) & ~prompt_mask[0]).sum().item())
                print(f"  step {step+1:3d}/{steps} | t={t:.3f} | anchored={n_anchored} | {text[:70]}")

        # ── Decodifica finale ───────────────────────────────────────────────
        t_final        = self._t_tensor(1.0 / steps, batch=1)
        _, ce_final, _ = self.model(x_t, t_final, self_cond=self_cond)
        token_ids      = self._decode(ce_final, mode, temperature, top_p)[0]

        return self.tokenizer.decode(token_ids.tolist(), skip_special_tokens=True)  # type: ignore

    @torch.no_grad()
    def generate_batch(
        self,
        prompts:              List[str],
        gen_len:              int          = 64,
        steps:                int          = 100,
        mode:                 SamplingMode = "argmax",
        temperature:          float        = 1.0,
        top_p:                float        = 0.9,
        confidence_threshold: float        = 0.7,
        anchor_every:         int          = 5,
        use_self_cond:        bool         = True,
    ) -> List[str]:
        """
        Versione batch di generate — più efficiente di chiamare generate() in loop.
        Stessa logica di generate() ma vettorializzata su B sequenze.
        """
        B   = len(prompts)
        enc = self.tokenizer(
            prompts, padding="max_length", truncation=True,
            max_length=gen_len, return_tensors="pt",
        )
        input_ids      = enc["input_ids"].to(self.device)             # type: ignore
        attention_mask = enc["attention_mask"].to(self.device).bool() # type: ignore

        # [FIX] Inizializza solo le posizioni non-prompt con rumore.
        # Le posizioni del prompt vengono iniettate con i loro embedding reali
        # senza rumore, così il modello le vede alla loro scala naturale.
        x_t = torch.zeros(B, gen_len, self.d_model, device=self.device)
        # Rumore solo sulle posizioni non-prompt (attention_mask=0)
        noise_mask = ~attention_mask   # (B, L) True = da generare
        x_t[noise_mask] = torch.randn(
            int(noise_mask.sum().item()), self.d_model, device=self.device
        )
        # Inietta embedding del prompt
        x_t[attention_mask] = self.model.token_emb(input_ids[attention_mask])

        prompt_mask             = attention_mask
        self_cond: Optional[torch.Tensor] = None
        dt = -1.0 / steps

        for step in range(steps):
            t = 1.0 - step / steps

            x_next, ce_logits, eps_pred = self._reverse_sde_step(x_t, t, dt, self_cond)

            x_next = torch.where(prompt_mask.unsqueeze(-1), x_t, x_next)

            if mode == "confidence" and step % anchor_every == 0:
                x_next = self._anchor_confident_tokens(
                    x_next, ce_logits, prompt_mask,
                    threshold=confidence_threshold, t=t,
                )

            if use_self_cond:
                self_cond = eps_pred.mean(dim=1).detach()

            x_t = x_next

        t_final         = self._t_tensor(1.0 / steps, batch=B)
        _, ce_final, _  = self.model(x_t, t_final, self_cond=self_cond)
        token_ids_batch = self._decode(ce_final, mode, temperature, top_p)

        return [
            self.tokenizer.decode(token_ids_batch[b].tolist(), skip_special_tokens=True)
            for b in range(B)
        ]  # type: ignore

    # ── CFG conditioned generation ───────────────────────────────────────────

    def _encode_context(self, context: str) -> torch.Tensor:
        """
        Encoda un testo di contesto come mean pooling degli embedding.
        Identico a encode_context() in train_sft.py.

        Returns:
            ctx_emb: (1, D)
        """
        ids      = self.tokenizer(
            context, return_tensors="pt",
            truncation=True, max_length=128,
        )["input_ids"].to(self.device)                       # type: ignore # (1, L)
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
        print("Usage: python sampler_v3.py <checkpoint_path>")
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
        prompt=prompt, gen_len=64, steps=100,
        mode="argmax", use_self_cond=True, verbose=True,
    )
    print(f"[argmax] {out}\n")

    # Confidence-guided
    out = sampler.generate(
        prompt=prompt, gen_len=64, steps=100,
        mode="confidence", confidence_threshold=0.7, anchor_every=5,
        use_self_cond=True,
    )
    print(f"[confidence] {out}\n")

    # Top-p sampling
    for temp in (0.8, 1.0, 1.3):
        out = sampler.generate(
            prompt=prompt, gen_len=64, steps=100,
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
        gen_len=64, steps=100, mode="confidence",
        confidence_threshold=0.7, anchor_every=5,
    )
    for p, o in zip(["Once upon a time", "The quick brown fox"], outs):
        print(f"[{p!r}] {o}")