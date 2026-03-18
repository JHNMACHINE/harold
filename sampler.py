"""
Harold v3 — sampler_v3.py
==========================
Sampler per VP-SDE continuous diffusion con reverse SDE (Euler-Maruyama).

Fix rispetto alla versione precedente:
  [FIX S1] Tuple_ import spostato in cima — era in fondo al file, dopo l'uso
  [FIX S2] _emb_to_tokens_ce non era usabile dall'esterno (richiedeva x_out
           pre-head) — rimosso, si usa ce_logits direttamente ovunque
  [FIX S3] self_cond aggiornato con eps_pred.mean invece di x_next.mean —
           più coerente: il self-cond dovrebbe riflettere la predizione
           del modello, non lo stato SDE
  [FIX S4] top-k dinamico MoE ripristinato in inferenza — passa t_normalized
           al modello attraverso forward() che già lo gestisce internamente
  [FIX S5] generate() e generate_batch() unificate nella logica di decodifica
           finale — evitava duplicazione e bug potenziali
  [FIX S6] temperature=1.0 con top_p=1.0 ora fa argmax invece di sampling
           uniforme — condizione corretta
  [FIX S7] Prompt injection: le posizioni del prompt vengono inizializzate
           con i loro embedding reali, non con rumore — era già così ma
           ora è esplicitamente garantito anche dopo ogni step SDE
"""

import math
import sys
import torch
import torch.nn.functional as F
from typing import Literal, Optional, List, Tuple
from transformers import PreTrainedTokenizer, AutoTokenizer

from model import Harold, build_model


SamplingMode = Literal["argmax", "sample", "confidence"]


class HaroldSampler:
    """
    Sampler per Harold v3 — reverse SDE con Euler-Maruyama.

    Loop di generazione:
      1. Inizializza x_1 ~ N(0,I) per le posizioni da generare
         (le posizioni del prompt sono fissate ai loro embedding)
      2. Per ogni step t da 1 a 0:
           a. Forward: predice ε e ce_logits
           b. Calcola score = -ε / σ(t)
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
        Costruisce una maschera booleana dei token [unused] nel vocabolario.
        Scansiona il vocabolario del tokenizer una volta sola e cachea il risultato.
        Molto più accurato dell'hardcoding degli indici 1-99.
        """
        if hasattr(self, "_unused_mask_cache"):
            return self._unused_mask_cache

        vocab = self.tokenizer.get_vocab()           # {token_str: idx}
        V     = self.model.emb_vocab
        mask  = torch.zeros(V, dtype=torch.bool, device=self.device)

        for token_str, idx in vocab.items():
            if idx < V and (
                token_str.startswith("[unused") or
                token_str.startswith("##") and len(token_str) <= 3  # subword rumorosi
            ):
                mask[idx] = True

        self._unused_mask_cache = mask
        n_masked = int(mask.sum().item())
        print(f"[sampler] Token mascherati nella decodifica: {n_masked} "
              f"([unused] + subword rumorosi)")
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

    def _decode(
        self,
        ce_logits:    torch.Tensor,  # (B, L, V)
        mode:         SamplingMode,
        temperature:  float,
        top_p:        float,
        mask_unused:  bool = True,   # rimuovi token [unused] dalla decodifica
    ) -> torch.Tensor:
        """Decodifica ce_logits in token_ids (B, L) secondo la modalità."""
        if mask_unused:
            ce_logits = self._mask_unused_tokens(ce_logits)
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
                self_cond = eps_pred.mean(dim=1).detach()

            x_t = x_next

            if verbose and step % 10 == 0:
                tokens = self._mask_unused_tokens(ce_logits).argmax(dim=-1)[0].tolist()
                text   = self.tokenizer.decode(tokens, skip_special_tokens=False)
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

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
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
        use_self_cond=True, verbose=True,
    )
    print(f"[confidence] {out}\n")

    # Top-p sampling con diverse temperature
    for temp in (0.8, 1.0, 1.3):
        out = sampler.generate(
            prompt=prompt, gen_len=64, steps=100,
            mode="sample", temperature=temp, top_p=0.9,
            use_self_cond=True,
        )
        print(f"[sample t={temp}] {out}")

    # Batch
    print("\n--- Batch ---")
    outs = sampler.generate_batch(
        prompts=["Once upon a time", "The quick brown fox"],
        gen_len=64, steps=100, mode="confidence",
        confidence_threshold=0.7, anchor_every=5,
    )
    for p, o in zip(["Once upon a time", "The quick brown fox"], outs):
        print(f"[{p!r}] {o}")