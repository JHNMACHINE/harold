"""
Harold v3 — model.py
=====================
Architettura 1B con stack completa:

  Attention:
    - MLA  (Multi-head Latent Attention) — KV cache compressa 4-8x
    - DSA  (sparse attention) — window locale + global tokens
    - GQA  — query heads > kv heads
    - RoPE — positional encoding rotazionale

  FFN:
    - DeepSeekMoE — shared SwiGLU (sempre attivi) + routed GELU (top-k)
    - Router t-condizionato (Mixture of Diffusions)
    - Top-k fisso in training, dinamico (threshold) in inferenza

  Conditioning:
    - AdaLN — adaptive layer norm condizionata su t_emb + self_cond
    - Self-conditioning — hint dalla predizione al timestep precedente

  Diffusion:
    - Continuous diffusion — predice embedding invece di logit discreti
    - Token-weighted schedule — maschera prima i token informativi
    - Cosine schedule base invariato

  Modifiche rispetto a v2:
    - Gate tanh rimosse (erano inizializzate a 1, ma più pulito senza)
    - lm_head proietta su d_model invece di vocab_size (continuous)
    - Harold.forward accetta xt_emb (B,L,d_model) invece di xt (B,L)
    - Loss: MSE embedding + 0.1 * CE ausiliaria per stabilità
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import ModelConfig


# ─────────────────────────────────────────────────────────────────────────────
# Expert e SharedExpert — invariati da v2
# ─────────────────────────────────────────────────────────────────────────────

class Expert(nn.Module):
    """MLP expert con attivazione GELU."""
    def __init__(self, n_embd: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim, n_embd, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SharedExpert(nn.Module):
    """Shared expert con SwiGLU — sempre attivo su tutti i token."""
    def __init__(self, n_embd: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.w1      = nn.Linear(n_embd, hidden_dim, bias=False)
        self.w3      = nn.Linear(n_embd, hidden_dim, bias=False)
        self.w2      = nn.Linear(hidden_dim, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.dropout(F.silu(self.w1(x)) * self.w3(x)))


# ─────────────────────────────────────────────────────────────────────────────
# DeepSeekMoELayer v3 — router t-condizionato + top-k dinamico
# ─────────────────────────────────────────────────────────────────────────────

class DeepSeekMoELayer(nn.Module):
    """
    DeepSeek-style MoE v3:
      - N shared experts SwiGLU (sempre attivi)
      - E routed experts GELU (top-k selezionati per token)
      - Router t-condizionato: input = [x, t_emb] → Mixture of Diffusions
      - Top-k fisso durante training (efficienza GPU)
      - Top-k dinamico con threshold durante inferenza (flessibilità)
      - Router bias adattivo per load balancing
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_routed_experts  = config.moe_n_routed_experts
        self.top_k             = config.moe_top_k
        self.bias_update_gamma = 1e-3
        self.threshold_base    = 0.3
        self.threshold_min     = 0.15
        self.top_k_min         = 1

        self.shared_experts = nn.ModuleList([
            SharedExpert(config.d_model, config.d_ff // 2, dropout=config.dropout)
            for _ in range(config.ds_moe_n_shared_experts)
        ])
        self.routed_experts = nn.ModuleList([
            Expert(config.d_model, config.d_ff // 4, dropout=config.dropout)
            for _ in range(self.n_routed_experts)
        ])

        # Router t-condizionato: input = concat(x, t_emb) → 2 * d_model
        # Inizializzazione con std piccolo per stabilità
        self.router = nn.Linear(config.d_model * 2, self.n_routed_experts, bias=False)
        nn.init.normal_(self.router.weight, std=0.01)

        self.register_buffer("router_bias", torch.zeros(self.n_routed_experts))
        self.router_indices: torch.Tensor | None = None

    def _affinity(self, x_flat: torch.Tensor, t_emb_flat: torch.Tensor) -> torch.Tensor:
        """Router condizionato su x e t — Mixture of Diffusions."""
        router_input = torch.cat([x_flat, t_emb_flat], dim=-1)  # (N, 2*d_model)
        return torch.sigmoid(self.router(router_input).float())

    def _compute_threshold(self, t_normalized: float) -> float:
        """
        Threshold t-dipendente per top-k dinamico:
          t alto (molto rumore) → threshold più bassa → più esperti attivi
          t basso (poco rumore) → threshold più alta  → meno esperti attivi
        """
        return self.threshold_base - (
            (self.threshold_base - self.threshold_min) * t_normalized
        )

    def forward(
        self,
        x:           torch.Tensor,
        t_emb:       torch.Tensor,
        t_normalized: float | None = None,  # None = training (top-k fisso)
    ) -> torch.Tensor:
        B, T, C = x.shape
        x_flat  = x.view(-1, C)

        # Espandi t_emb su tutti i token: (B, d_model) → (B*T, d_model)
        t_emb_flat = t_emb.unsqueeze(1).expand(B, T, C).reshape(-1, C)

        # Shared experts (sempre attivi)
        shared_out = torch.zeros_like(x_flat)
        for expert in self.shared_experts:
            shared_out += expert(x_flat)

        # Router t-condizionato
        s          = self._affinity(x_flat, t_emb_flat)
        sel_scores = s + self.router_bias.to(s.device)  # type: ignore

        # Selezione esperti: top-k fisso in training, dinamico in inferenza
        if self.training or t_normalized is None:
            topk_indices = torch.topk(sel_scores, self.top_k, dim=-1).indices
        else:
            threshold = self._compute_threshold(t_normalized)
            active    = sel_scores > threshold
            n_active  = active.sum(dim=-1).clamp(self.top_k_min, self.n_routed_experts)
            k_max     = int(n_active.max().item())
            topk_indices = torch.topk(sel_scores, k_max, dim=-1).indices
            # Maschera esperti sotto threshold
            topk_scores = sel_scores.gather(1, topk_indices)
            invalid     = topk_scores <= threshold
            topk_indices = topk_indices.masked_fill(invalid, -1)

        self.router_indices = topk_indices.detach()

        # Gating normalizzato
        valid_mask = topk_indices >= 0
        s_sel = s.gather(dim=1, index=topk_indices.clamp(min=0))
        s_sel = s_sel * valid_mask.to(s_sel.dtype)
        denom = s_sel.sum(dim=1, keepdim=True)
        gates = torch.where(
            denom > 1e-9,
            s_sel / (denom + 1e-9),
            torch.full_like(s_sel, 1.0 / self.top_k),
        ).to(x.dtype)

        # Routed experts
        routed_out = torch.zeros_like(x_flat)
        for i in range(self.n_routed_experts):
            mask             = (topk_indices == i)
            row_idx, which_k = mask.nonzero(as_tuple=True)
            if row_idx.numel() == 0:
                continue
            expert_out  = self.routed_experts[i](x_flat.index_select(0, row_idx))
            gate_values = gates[row_idx, which_k].unsqueeze(1)
            routed_out.index_add_(0, row_idx, expert_out * gate_values)

        return (shared_out + routed_out).view(B, T, C)

    @torch.no_grad()
    def update_bias(self):
        if self.router_indices is None:
            return
        valid   = self.router_indices[self.router_indices >= 0]
        if valid.numel() == 0:
            return
        counts  = torch.bincount(valid.view(-1), minlength=self.n_routed_experts)
        counts  = counts.float().to(self.router_bias.device)  # type: ignore
        self.router_bias += self.bias_update_gamma * (counts.mean() - counts).sign()
        self.router_indices = None


# ─────────────────────────────────────────────────────────────────────────────
# RoPE — invariato da v2
# ─────────────────────────────────────────────────────────────────────────────

class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10000.0):
        super().__init__()
        freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t     = torch.arange(max_seq_len, dtype=torch.float32)
        emb   = torch.outer(t, freqs)
        self.register_buffer("cos_cached", torch.cat([torch.cos(emb), torch.cos(emb)], dim=-1))
        self.register_buffer("sin_cached", torch.cat([torch.sin(emb), torch.sin(emb)], dim=-1))

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        return torch.cat([-x[..., half:], x[..., :half]], dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor, offset: int = 0):
        T   = q.shape[2]
        cos = self.cos_cached[offset:offset + T].to(q.dtype).unsqueeze(0).unsqueeze(0)  # type: ignore
        sin = self.sin_cached[offset:offset + T].to(q.dtype).unsqueeze(0).unsqueeze(0)  # type: ignore
        return q * cos + self._rotate_half(q) * sin, k * cos + self._rotate_half(k) * sin


# ─────────────────────────────────────────────────────────────────────────────
# BlockCausalAttention v3 — MLA + DSA
# ─────────────────────────────────────────────────────────────────────────────

class BlockCausalAttention(nn.Module):
    """
    Attenzione v3 con MLA e DSA:

    MLA (Multi-head Latent Attention):
      - Comprimi K e V in spazio latente (latent_dim << n_kv_heads * head_dim)
      - KV cache salva solo c_kv (latent) → 4-8x meno memoria
      - K e V vengono ricostruiti al momento dell'uso

    DSA (Sparse Attention):
      - Ogni token vede una finestra locale di window_size token
      - Ogni dsa_global_every token è un global token che vede tutto
      - Complessità: O(n × window) invece di O(n²)
      - Mantiene la block-causal semantics originale (OR con sparse mask)
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        assert config.n_heads % config.n_kv_heads == 0

        self.n_heads      = config.n_heads
        self.n_kv_heads   = config.n_kv_heads
        self.n_kv_groups  = config.n_heads // config.n_kv_heads
        self.head_dim     = config.d_model // config.n_heads
        self.block_size   = config.block_size
        self.dropout      = config.dropout

        # MLA: proiezioni latenti invece di k_proj/v_proj diretti
        self.latent_dim = config.mla_latent_dim
        self.q_proj   = nn.Linear(config.d_model, config.n_heads * self.head_dim, bias=False)
        self.kv_down  = nn.Linear(config.d_model, self.latent_dim, bias=False)
        self.k_up     = nn.Linear(self.latent_dim, config.n_kv_heads * self.head_dim, bias=False)
        self.v_up     = nn.Linear(self.latent_dim, config.n_kv_heads * self.head_dim, bias=False)
        self.o_proj   = nn.Linear(config.n_heads * self.head_dim, config.d_model, bias=False)

        # DSA: parametri per window e global tokens
        self.window_size  = config.dsa_window_size
        self.global_every = config.dsa_global_every

        self.rope       = RotaryEmbedding(self.head_dim, config.max_seq_len, config.rope_theta)
        self.resid_drop = nn.Dropout(config.dropout)

    def _build_sparse_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Maschera sparse: combina DSA (window + global) con block-causal.

        mask[i,j] = True → posizione j NON visibile da i (attention = -inf)

        Visibile se:
          - j è nella finestra locale di i (|i-j| <= window_size)
          - j è un global token (j % global_every == 0)
          - block(j) <= block(i) (block-causal: non vedere il futuro inter-block)
        """
        idx       = torch.arange(seq_len, device=device)
        block_idx = idx // self.block_size

        # Maschera block-causal: True se j è in blocco futuro rispetto a i
        future_block = block_idx.unsqueeze(0) > block_idx.unsqueeze(1)

        # Maschera window locale: True se j è fuori finestra
        dist = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
        outside_window = dist > self.window_size

        # Token globali: visibili sempre (override window)
        is_global = (idx % self.global_every == 0)
        global_visible = is_global.unsqueeze(0).expand(seq_len, seq_len)

        # Visibile se: (nella finestra OR global token) AND non blocco futuro
        mask = (outside_window & ~global_visible) | future_block
        return mask  # bool

    def forward(
        self,
        x:         torch.Tensor,
        past_kv:   torch.Tensor | None = None,   # MLA: cache di c_kv invece di (k,v)
        use_cache: bool = False,
        kv_offset: int  = 0,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T, C = x.shape

        # Query — invariata
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # MLA: comprimi in spazio latente
        c_kv = self.kv_down(x)  # (B, T, latent_dim) — questo va in cache
        k    = self.k_up(c_kv).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v    = self.v_up(c_kv).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # RoPE su q e k
        q, k = self.rope(q, k, offset=kv_offset)

        # KV cache MLA: concatena c_kv latente (molto più piccolo di k,v)
        if past_kv is not None:
            c_kv_full = torch.cat([past_kv, c_kv], dim=1)   # (B, past+T, latent_dim)
            k = self.k_up(c_kv_full).view(B, -1, self.n_kv_heads, self.head_dim).transpose(1, 2)
            v = self.v_up(c_kv_full).view(B, -1, self.n_kv_heads, self.head_dim).transpose(1, 2)
        else:
            c_kv_full = c_kv

        full_len = k.shape[2]

        # GQA: espandi k/v
        k_exp = k.repeat_interleave(self.n_kv_groups, dim=1)
        v_exp = v.repeat_interleave(self.n_kv_groups, dim=1)

        # DSA: maschera sparse
        mask = self._build_sparse_mask(full_len, x.device)
        if T < full_len:
            mask = mask[-T:, :]

        attn_bias = torch.zeros(T, full_len, device=x.device, dtype=x.dtype)
        attn_bias = attn_bias.masked_fill(mask, float("-inf"))
        attn_bias = attn_bias.unsqueeze(0).unsqueeze(0)

        y = F.scaled_dot_product_attention(
            q, k_exp, v_exp,
            attn_mask=attn_bias,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )

        y   = y.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_drop(self.o_proj(y))

        # Salva c_kv latente nella cache (4x più piccolo di k,v separati)
        present_kv = c_kv_full if use_cache else None
        return out, present_kv


# ─────────────────────────────────────────────────────────────────────────────
# AdaLN v3 — con self-conditioning
# ─────────────────────────────────────────────────────────────────────────────

class AdaLN(nn.Module):
    """
    Adaptive Layer Norm v3 con self-conditioning:
      out = norm(x) * (1 + scale) + shift
      scale, shift = proj(t_emb + self_cond_emb)

    self_cond_emb è la predizione del timestep precedente proiettata
    nello stesso spazio di t_emb — permette al modello di usare
    il proprio output precedente come hint per il passo corrente.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.proj = nn.Linear(d_model, 2 * d_model, bias=True)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        t_emb può già incorporare il self-conditioning se sommato a monte
        in Harold.forward — AdaLN non ha bisogno di saperlo.
        """
        scale, shift = self.proj(t_emb).chunk(2, dim=-1)
        return self.norm(x) * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# ─────────────────────────────────────────────────────────────────────────────
# Block v3
# ─────────────────────────────────────────────────────────────────────────────

class Block(nn.Module):
    """
    Transformer block v3:
      - AdaLN con self-conditioning (incorporato in t_emb a monte)
      - BlockCausalAttention v3 (MLA + DSA)
      - DeepSeekMoELayer v3 (router t-condizionato)
      - Residual standard (gate tanh rimossi — erano fonte di instabilità)
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ada_ln_1 = AdaLN(config.d_model)
        self.ada_ln_2 = AdaLN(config.d_model)
        self.attn     = BlockCausalAttention(config)
        self.moe      = DeepSeekMoELayer(config)

    def forward(
        self,
        x:            torch.Tensor,
        t_emb:        torch.Tensor,
        past_kv:      torch.Tensor | None = None,
        use_cache:    bool  = False,
        kv_offset:    int   = 0,
        t_normalized: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:

        attn_out, present_kv = self.attn(
            self.ada_ln_1(x, t_emb),
            past_kv=past_kv,
            use_cache=use_cache,
            kv_offset=kv_offset,
        )
        x = x + attn_out
        x = x + self.moe(self.ada_ln_2(x, t_emb), t_emb, t_normalized=t_normalized)
        return x, present_kv


# ─────────────────────────────────────────────────────────────────────────────
# MaskDiffusionSchedule v3 — token-weighted schedule
# ─────────────────────────────────────────────────────────────────────────────

class MaskDiffusionSchedule(nn.Module):
    """
    Cosine noise schedule v3 con token-weighted masking.

    Token-weighted: la probabilità di masking è modulata per token:
      - Token rari (informativi) → mascherati prima (peso alto)
      - Token comuni ("the","a") → mascherati dopo (peso basso)

    token_weights è una lookup table precomputata dalle frequenze
    del dataset. Se non fornita, tutti i token hanno peso uguale
    (comportamento identico a v2).
    """
    def __init__(self, config: ModelConfig, token_weights: torch.Tensor | None = None):
        super().__init__()
        self.T             = config.diffusion_T
        self.mask_token_id = config.mask_token_id

        t      = torch.linspace(0, self.T, self.T + 1)
        alphas = torch.cos((t / self.T) * math.pi / 2) ** 2
        self.register_buffer("alphas", alphas)

        # Token weights: (vocab_size,) float in [0, 1]
        if token_weights is not None:
            w = token_weights.float()
            w = (w - w.min()) / (w.max() - w.min() + 1e-8)
        else:
            w = torch.ones(config.vocab_size)
        self.register_buffer("token_weights", w)

    def q_sample(
        self,
        x0: torch.Tensor,   # (B, L) token IDs
        t:  torch.Tensor,   # (B,)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        alpha_t    = self.alphas[t]          # type: ignore  (B,)
        base_prob  = 1.0 - alpha_t.unsqueeze(1)  # (B, 1)

        # Peso per ogni token nella sequenza
        tok_w = self.token_weights[x0]       # type: ignore  (B, L)

        # Scala la probabilità: token importanti → più facile da mascherare
        # range: [base_prob * 0.5, base_prob * 1.5]
        scaled_prob = base_prob * (0.5 + tok_w)
        scaled_prob = scaled_prob.clamp(0.0, 1.0)

        mask = torch.bernoulli(scaled_prob).bool()
        xt   = x0.clone()
        xt[mask] = self.mask_token_id
        return xt, mask

    def get_alpha(self, t: torch.Tensor) -> torch.Tensor:
        return self.alphas[t]  # type: ignore


# ─────────────────────────────────────────────────────────────────────────────
# Harold v3 — continuous diffusion + self-conditioning
# ─────────────────────────────────────────────────────────────────────────────

class Harold(nn.Module):
    """
    Harold v3 — DiffusionMoE con continuous diffusion e self-conditioning.

    Differenze chiave rispetto a v2:

    Continuous diffusion:
      - Input: xt_emb (B, L, d_model) embedding continui invece di token IDs
      - Output: x0_pred (B, L, d_model) embedding predetti invece di logit
      - Loss: MSE(x0_pred, emb(x0)) + 0.1 * CE(x0_pred @ W_emb^T, x0)
      - Nearest neighbor per decodifica: argmin dist(x0_pred, W_emb)

    Self-conditioning:
      - Con prob 0.5 durante training: passa la predizione precedente come hint
      - self_cond_proj proietta x0_prev nello spazio di t_emb
      - t_emb_sc = t_emb + self_cond_proj(x0_prev.mean(dim=1))
      - Init a zero → no-op all'inizio del training

    forward accetta:
      xt_emb:    (B, L, d_model) — embedding corrotti
      t:         (B,)            — timestep
      self_cond: (B, d_model) | None — embedding medio della predizione precedente
    """
    def __init__(self, config: ModelConfig, token_weights: torch.Tensor | None = None):
        super().__init__()
        self.config        = config
        self.d_model       = config.d_model
        self.mask_token_id = config.mask_token_id

        # Vocabulary size per embedding e nearest neighbor lookup
        emb_vocab = max(config.vocab_size, config.mask_token_id) + 1
        self.emb_vocab = emb_vocab

        # Token embedding — usato per:
        #   1. Convertire token ID in embedding (input al modello)
        #   2. Nearest neighbor lookup nella decodifica
        self.token_emb = nn.Embedding(emb_vocab, config.d_model, padding_idx=0)
        self.pos_emb   = nn.Embedding(config.max_seq_len, config.d_model)

        # Timestep MLP — invariato
        self.time_emb = nn.Sequential(
            nn.Linear(config.d_model, config.d_model * 4),
            nn.SiLU(),
            nn.Linear(config.d_model * 4, config.d_model),
        )

        # Self-conditioning projection
        # Proietta x0_prev (media degli embedding) nello spazio di t_emb
        # Init a zero: no-op all'inizio, il modello impara quando usarlo
        self.self_cond_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        nn.init.zeros_(self.self_cond_proj.weight)

        # Frequenze sinusoidali per il timestep
        half  = config.d_model // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, dtype=torch.float32) / half
        )
        self.register_buffer("t_freqs", freqs)

        # Transformer blocks
        self.blocks   = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.norm_out = nn.LayerNorm(config.d_model)

        # lm_head v3: proietta su d_model invece di vocab_size (continuous diffusion)
        # NON weight-tied: predice embedding, non logit
        self.lm_head = nn.Linear(config.d_model, config.d_model, bias=False)

        # Proiezione ausiliaria per la CE loss di stabilizzazione
        # Weight-tied con token_emb per coerenza
        self.ce_head = nn.Linear(config.d_model, emb_vocab, bias=False)
        self.ce_head.weight = self.token_emb.weight

        # Noise schedule
        self.schedule = MaskDiffusionSchedule(config, token_weights)

        self._init_weights()

    def _init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                std = (0.02 / math.sqrt(2 * self.config.n_layers)
                       if any(name.endswith(s) for s in ("o_proj", "w2", "v_up"))
                       else 0.02)
                nn.init.normal_(module.weight, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
        # self_cond_proj rimane a zero (init sopra)

    def get_timestep_embedding(self, t: torch.Tensor) -> torch.Tensor:
        args = t[:, None].float() * self.t_freqs[None]  # type: ignore
        emb  = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.d_model % 2:
            emb = F.pad(emb, (0, 1))
        return emb

    def forward(
        self,
        xt_emb:          torch.Tensor,                  # (B, L, d_model) embedding corrotti
        t:               torch.Tensor,                  # (B,)
        self_cond:       torch.Tensor | None = None,    # (B, d_model) predizione precedente
        past_key_values: list[torch.Tensor] | None = None,
        use_cache:       bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, list | None]:
        """
        Returns:
            x0_pred:  (B, L, d_model) embedding predetti
            ce_logits: (B, L, V)      logit ausiliari per CE loss
            present_key_values: lista di c_kv latenti (MLA cache)
        """
        B, L, _ = xt_emb.shape
        device  = xt_emb.device

        kv_offset = past_key_values[0].shape[1] if past_key_values is not None else 0

        # Posizione encoding
        pos = torch.arange(kv_offset, kv_offset + L, device=device).unsqueeze(0).expand(B, -1)
        x   = xt_emb + self.pos_emb(pos)

        # Timestep embedding
        t_emb = self.time_emb(self.get_timestep_embedding(t))  # (B, d_model)

        # Self-conditioning: aggiungi hint dalla predizione precedente
        if self_cond is not None:
            sc_emb = self.self_cond_proj(self_cond)  # (B, d_model)
            t_emb  = t_emb + sc_emb

        # t normalizzato per top-k dinamico in inferenza
        t_normalized = (t.float().mean() / self.config.diffusion_T).item() if not self.training else None

        present_kvs = [] if use_cache else None

        for i, block in enumerate(self.blocks):
            past_kv = past_key_values[i] if past_key_values is not None else None
            x, present_kv = block(
                x, t_emb,
                past_kv=past_kv,
                use_cache=use_cache,
                kv_offset=kv_offset,
                t_normalized=t_normalized,
            )
            if use_cache:
                present_kvs.append(present_kv)  # type: ignore

        x_out     = self.norm_out(x)
        x0_pred   = self.lm_head(x_out)       # (B, L, d_model) — embedding predetti
        ce_logits = self.ce_head(x_out)        # (B, L, V)       — logit ausiliari

        return x0_pred, ce_logits, present_kvs

    def decode_tokens(self, x0_pred: torch.Tensor) -> torch.Tensor:
        """
        Nearest neighbor lookup: trova il token più vicino per ogni posizione.
        x0_pred: (B, L, d_model) → token_ids: (B, L)
        """
        # Distanza euclidea tra predizione e tutti gli embedding del vocabolario
        # Usiamo prodotto scalare normalizzato (cosine similarity) per efficienza
        emb_norm  = F.normalize(self.token_emb.weight, dim=-1)   # (V, d_model)
        pred_norm = F.normalize(x0_pred, dim=-1)                  # (B, L, d_model)
        sim       = torch.einsum("bld,vd->blv", pred_norm, emb_norm)  # (B, L, V)
        return sim.argmax(dim=-1)  # (B, L)

    def compute_loss(
        self,
        x0_pred:   torch.Tensor,  # (B, L, d_model)
        ce_logits: torch.Tensor,  # (B, L, V)
        x0:        torch.Tensor,  # (B, L) token IDs originali
        mask:      torch.Tensor,  # (B, L) bool
        ce_weight: float = 0.1,
    ) -> tuple[torch.Tensor, dict]:
        """
        Loss ibrida: MSE embedding + CE ausiliaria per stabilità.

        La CE ausiliaria previene il collasso dello spazio embedding
        nelle prime fasi del training, quando la MSE da sola è instabile.
        """
        if mask.sum() == 0:
            zero = torch.tensor(0.0, device=x0_pred.device, requires_grad=True)
            return zero, {"mse": 0.0, "ce": 0.0, "total": 0.0}

        # Target: embedding dei token originali
        emb_target = self.token_emb(x0)  # (B, L, d_model)

        # MSE loss sulle posizioni mascherate
        loss_mse = F.mse_loss(x0_pred[mask], emb_target[mask].detach())

        # CE loss ausiliaria per stabilità
        loss_ce = F.cross_entropy(
            ce_logits[mask],
            x0[mask],
            reduction="mean",
        )

        total = loss_mse + ce_weight * loss_ce

        return total, {
            "mse":   loss_mse.item(),
            "ce":    loss_ce.item(),
            "total": total.item(),
        }

    @torch.no_grad()
    def update_router_biases(self):
        for block in self.blocks:
            block.moe.update_bias()  # type: ignore


def build_model(model_cfg: ModelConfig, token_weights: torch.Tensor | None = None) -> Harold:
    return Harold(model_cfg, token_weights)