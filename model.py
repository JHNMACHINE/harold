"""
DiffusionMoE v2 — model.py
==========================
Integra il meglio di DiffusionMoE (originale) e LLaDA2.1:

  Tuo originale (mantenuto):
    - DeepSeekMoELayer  (shared experts SwiGLU + routed GELU + bias adattivo)
    - AdaLN             (timestep conditioning per ogni block)
    - MaskDiffusionSchedule  (cosine schedule)

  LLaDA2.1 (integrato e corretto):
    - BlockCausalAttention con RoPE + GQA  (bidirezionale intra-blocco,
                                             causale inter-blocco)
    - Signature forward(xt, t) mantenuta

  Correzioni applicate al codice LLaDA2.1:
    - Block-causal mask corretta (intra-block = bidirezionale)
    - KV-cache salva k/v completi, non solo l'ultimo chunk
    - RoPE aggiunto (essenziale a max_seq_len 4096+)
    - GQA repeat_interleave spostato nel posto corretto
    - AdaLN collegato al block (mancava completamente in LLaDA2.1)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import ModelConfig


# ─────────────────────────────────────────────────────────────────────────────
# MoE  (dal tuo originale, invariato)
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


class DeepSeekMoELayer(nn.Module):
    """
    DeepSeek-style MoE:
      - N shared experts (SwiGLU, sempre attivi)
      - E routed experts (GELU, top-k selezionati per token)
      - Router bias adattivo per bilanciare il carico
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_routed_experts  = config.moe_n_routed_experts
        self.top_k             = config.moe_top_k
        self.bias_update_gamma = 1e-3

        self.shared_experts = nn.ModuleList([
            SharedExpert(config.d_model, config.d_ff // 2, dropout=config.dropout)
            for _ in range(config.ds_moe_n_shared_experts)
        ])
        self.routed_experts = nn.ModuleList([
            Expert(config.d_model, config.d_ff // 4, dropout=config.dropout)
            for _ in range(self.n_routed_experts)
        ])
        self.router = nn.Linear(config.d_model, self.n_routed_experts, bias=False)
        self.register_buffer("router_bias", torch.zeros(self.n_routed_experts))
        self.router_indices: torch.Tensor | None = None

    def _affinity(self, x_flat: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.router(x_flat).float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        x_flat  = x.view(-1, C)

        # Shared experts (sempre attivi)
        shared_out = torch.zeros_like(x_flat)
        for expert in self.shared_experts:
            shared_out += expert(x_flat)

        # Router
        s            = self._affinity(x_flat)
        sel_scores   = s + self.router_bias.to(s.device) # type: ignore
        topk_indices = torch.topk(sel_scores, self.top_k, dim=-1).indices

        self.router_indices = topk_indices.detach()

        # Gating normalizzato sulle affinità reali
        s_sel = s.gather(dim=1, index=topk_indices)
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
        """
        Aggiorna il router_bias dopo ogni optimizer step:
          expert sovraccarichi → bias scende → meno favoriti
          expert scarichi      → bias sale   → più favoriti
        """
        if self.router_indices is None:
            return
        counts = torch.bincount(
            self.router_indices.view(-1),
            minlength=self.n_routed_experts,
        ).float().to(self.router_bias.device) # type: ignore
        self.router_bias += self.bias_update_gamma * (counts.mean() - counts).sign()
        self.router_indices = None


# ─────────────────────────────────────────────────────────────────────────────
# RoPE
# ─────────────────────────────────────────────────────────────────────────────

class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).
    Pre-computa le frequenze e le applica a q e k.
    """
    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10000.0):
        super().__init__()
        freqs = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2).float() / head_dim)
        )
        t    = torch.arange(max_seq_len, dtype=torch.float32)
        emb  = torch.outer(t, freqs)           # (max_seq_len, head_dim/2)
        cos  = torch.cos(emb)                  # (max_seq_len, head_dim/2)
        sin  = torch.sin(emb)
        # Duplica per coprire tutti i canali della head
        self.register_buffer("cos_cached", torch.cat([cos, cos], dim=-1))  # (L, D)
        self.register_buffer("sin_cached", torch.cat([sin, sin], dim=-1))

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([-x2, x1], dim=-1)

    def forward(
        self,
        q: torch.Tensor,   # (B, n_heads, T, head_dim)
        k: torch.Tensor,   # (B, n_kv_heads, T, head_dim)
        offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        T   = q.shape[2]
        # (T, D)
        cos = self.cos_cached[offset : offset + T].to(q.dtype)   # type: ignore
        sin = self.sin_cached[offset : offset + T].to(q.dtype) # type: ignore

        # Broadcast su (1, 1, T, D)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin
        return q_rot, k_rot


# ─────────────────────────────────────────────────────────────────────────────
# Block-Causal Attention con GQA + RoPE  (LLaDA2.1 corretto)
# ─────────────────────────────────────────────────────────────────────────────

class BlockCausalAttention(nn.Module):
    """
    Attenzione bidirezionale intra-blocco, causale inter-blocco.

    Vantaggi rispetto alla bidirezionale pura:
      - KV cache utilizzabile durante la generazione per nuovi blocchi
      - Scaling migliore su sequenze molto lunghe
      - Mantiene la bidirezionalità che serve al diffusion model
        (all'interno di ogni blocco)

    Correzioni rispetto a LLaDA2.1:
      - Block-causal mask corretta (bidirezionale intra-blocco)
      - KV cache: salva i KV dell'intera sequenza, non solo l'ultimo chunk
      - RoPE integrato
      - GQA repeat_interleave applicato correttamente
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        assert config.n_heads % config.n_kv_heads == 0

        self.n_heads        = config.n_heads
        self.n_kv_heads     = config.n_kv_heads
        self.n_kv_groups    = config.n_heads // config.n_kv_heads
        self.head_dim       = config.d_model // config.n_heads
        self.block_size     = config.block_size
        self.dropout        = config.dropout

        self.q_proj  = nn.Linear(config.d_model, config.n_heads    * self.head_dim, bias=False)
        self.k_proj  = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.v_proj  = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.o_proj  = nn.Linear(config.n_heads * self.head_dim, config.d_model,    bias=False)

        self.rope        = RotaryEmbedding(self.head_dim, config.max_seq_len, config.rope_theta)
        self.resid_drop  = nn.Dropout(config.dropout)

    @staticmethod
    def _build_block_causal_mask(seq_len: int, block_size: int, device: torch.device) -> torch.Tensor:
        """
        Crea la maschera block-causal:
          - True  = posizione da mascherare (attention = -inf)
          - False = posizione visibile

        Regola:
          token i può attendere token j se:
            block(j) < block(i)   → blocco precedente: sempre visibile
            block(j) == block(i)  → stesso blocco: sempre visibile (bidirezionale)
            block(j) > block(i)   → blocco futuro: mascherato
        """
        idx    = torch.arange(seq_len, device=device)
        block  = idx // block_size                          # (seq_len,)
        # mask[i,j] = True se block(j) > block(i)  (futuro → mascherato)
        mask   = block.unsqueeze(0) > block.unsqueeze(1)    # (seq_len, seq_len)  — FIX: era <
        return mask  # bool

    def forward(
        self,
        x:           torch.Tensor,
        t_emb:       torch.Tensor | None = None,   # non usato qui, già applicato in AdaLN
        past_kv:     tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache:   bool = False,
        kv_offset:   int  = 0,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads,    self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # RoPE (applicato prima di concatenare la KV cache)
        q, k = self.rope(q, k, offset=kv_offset)

        # KV cache
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)   # dim=2 = seq dim dopo transpose
            v = torch.cat([past_v, v], dim=2)

        full_len = k.shape[2]

        # GQA: espandi k/v per corrispondere al numero di query heads
        k_exp = k.repeat_interleave(self.n_kv_groups, dim=1)  # (B, n_heads, full_len, head_dim)
        v_exp = v.repeat_interleave(self.n_kv_groups, dim=1)

        # Block-causal mask (costruita sulla lunghezza completa, query su T corrente)
        mask = self._build_block_causal_mask(full_len, self.block_size, x.device)
        # Se stiamo processando solo T token nuovi, prendi le ultime T righe della mask
        if T < full_len:
            mask = mask[-T:, :]  # (T, full_len)

        attn_bias = torch.zeros(T, full_len, device=x.device, dtype=x.dtype)
        attn_bias = attn_bias.masked_fill(mask, float("-inf"))
        attn_bias = attn_bias.unsqueeze(0).unsqueeze(0)  # (1, 1, T, full_len)

        y = F.scaled_dot_product_attention(
            q, k_exp, v_exp,
            attn_mask=attn_bias,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,   # usiamo la nostra mask esplicita
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_drop(self.o_proj(y))

        # Salva i KV *completi* (FIX: LLaDA2.1 salvava solo l'ultimo chunk)
        present_kv = (k, v) if use_cache else None
        return out, present_kv


# ─────────────────────────────────────────────────────────────────────────────
# AdaLN  (dal tuo originale, invariato)
# ─────────────────────────────────────────────────────────────────────────────

class AdaLN(nn.Module):
    """
    Adaptive Layer Norm: out = norm(x) * (1 + scale) + shift
    scale e shift sono proiettati dal timestep embedding.
    Init a zero → si comporta come LN standard all'inizio del training.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.proj = nn.Linear(d_model, 2 * d_model, bias=True)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        scale, shift = self.proj(t_emb).chunk(2, dim=-1)
        scale = scale.unsqueeze(1)
        shift = shift.unsqueeze(1)
        return self.norm(x) * (1.0 + scale) + shift


# ─────────────────────────────────────────────────────────────────────────────
# Block  (combina BlockCausalAttention + DeepSeekMoE + AdaLN)
# ─────────────────────────────────────────────────────────────────────────────

class Block(nn.Module):
    """
    Transformer block v2:
      - AdaLN per timestep conditioning (tuo originale)
      - BlockCausalAttention + GQA + RoPE (LLaDA2.1 corretto)
      - DeepSeekMoELayer (tuo originale)
      - Gated residuals (tuo originale)
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ada_ln_1  = AdaLN(config.d_model)
        self.ada_ln_2  = AdaLN(config.d_model)
        self.attn      = BlockCausalAttention(config)
        self.moe       = DeepSeekMoELayer(config)
        self.attn_gate = nn.Parameter(torch.zeros(1))
        self.moe_gate  = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        x:         torch.Tensor,
        t_emb:     torch.Tensor,
        past_kv:   tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
        kv_offset: int  = 0,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        attn_out, present_kv = self.attn(
            self.ada_ln_1(x, t_emb),
            past_kv=past_kv,
            use_cache=use_cache,
            kv_offset=kv_offset,
        )
        x = x + self.attn_gate.tanh() * attn_out
        x = x + self.moe_gate.tanh()  * self.moe(self.ada_ln_2(x, t_emb))
        return x, present_kv


# ─────────────────────────────────────────────────────────────────────────────
# Noise schedule  (dal tuo originale, invariato)
# ─────────────────────────────────────────────────────────────────────────────

class MaskDiffusionSchedule:
    """
    Cosine noise schedule per masked diffusion.
    alpha_t = probabilità che un token NON sia mascherato al tempo t.
    alpha: 1 (t=0, nessun mask) → 0 (t=T, tutto mascherato).
    """
    def __init__(self, config: ModelConfig):
        self.T             = config.diffusion_T
        self.mask_token_id = config.mask_token_id

        t       = torch.linspace(0, self.T, self.T + 1)
        alphas  = torch.cos((t / self.T) * math.pi / 2) ** 2
        self.alphas = alphas / alphas[0]

    def q_sample(
        self,
        x0: torch.Tensor,
        t:  torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        alpha_t   = self.alphas[t].to(x0.device)
        mask_prob = 1.0 - alpha_t.unsqueeze(1)
        mask      = torch.bernoulli(mask_prob.expand_as(x0.float())).bool()
        xt        = x0.clone()
        xt[mask]  = self.mask_token_id
        return xt, mask

    def get_alpha(self, t: torch.Tensor) -> torch.Tensor:
        return self.alphas[t]


# ─────────────────────────────────────────────────────────────────────────────
# DiffusionMoE v2  (modello principale)
# ─────────────────────────────────────────────────────────────────────────────

class DiffusionMoE(nn.Module):
    """
    DiffusionMoE v2 — predice x0 dato (xt, t).

    Architettura:
      token_emb  + pos_emb (apprendibili)
      timestep MLP → t_emb (condiziona AdaLN in ogni block)
      N × Block(BlockCausalAttention + DeepSeekMoE + AdaLN)
      LayerNorm + lm_head (weight-tied con token_emb)
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config        = config
        self.d_model       = config.d_model
        self.mask_token_id = config.mask_token_id

        # Embeddings
        # emb_vocab copre sia i token normali (0 … vocab_size-1)
        # sia il mask token (mask_token_id), qualunque sia il suo valore.
        emb_vocab = max(config.vocab_size, config.mask_token_id) + 1
        self.emb_vocab = emb_vocab
        self.token_emb = nn.Embedding(emb_vocab, config.d_model, padding_idx=0)
        self.pos_emb   = nn.Embedding(config.max_seq_len, config.d_model)

        # Timestep MLP
        self.time_emb = nn.Sequential(
            nn.Linear(config.d_model, config.d_model * 4),
            nn.SiLU(),
            nn.Linear(config.d_model * 4, config.d_model),
        )

        # Frequenze sinusoidali per il timestep (precompute)
        half  = config.d_model // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, dtype=torch.float32) / half
        )
        self.register_buffer("t_freqs", freqs)

        # Transformer blocks
        self.blocks   = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.norm_out = nn.LayerNorm(config.d_model)
        self.lm_head  = nn.Linear(config.d_model, emb_vocab, bias=False)

        # Init weights, poi weight tying
        self._init_weights()
        self.lm_head.weight = self.token_emb.weight

    def _init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                std = 0.02 / math.sqrt(2 * self.config.n_layers) if name.endswith(("o_proj", "c_proj", "w2")) else 0.02
                nn.init.normal_(module.weight, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def get_timestep_embedding(self, t: torch.Tensor) -> torch.Tensor:
        args = t[:, None].float() * self.t_freqs[None] # type: ignore
        emb  = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.d_model % 2:
            emb = F.pad(emb, (0, 1))
        return emb

    def forward(
        self,
        xt:              torch.Tensor,
        t:               torch.Tensor,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        use_cache:       bool = False,
    ) -> tuple[torch.Tensor, list | None]:
        """
        xt  : (B, L)  — sequenza corrotta
        t   : (B,)    — timestep per ogni sample
        Returns: logits (B, L, V), present_key_values (opzionale)
        """
        B, L   = xt.shape
        device = xt.device

        # Offset per RoPE/KV cache (se usiamo past_key_values)
        kv_offset = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        pos  = torch.arange(kv_offset, kv_offset + L, device=device).unsqueeze(0).expand(B, -1)
        x    = self.token_emb(xt) + self.pos_emb(pos)
        t_emb = self.time_emb(self.get_timestep_embedding(t))

        present_kvs = [] if use_cache else None

        for i, block in enumerate(self.blocks):
            past_kv = past_key_values[i] if past_key_values is not None else None
            x, present_kv = block(
                x, t_emb,
                past_kv=past_kv,
                use_cache=use_cache,
                kv_offset=kv_offset,
            )
            if use_cache:
                present_kvs.append(present_kv) # type: ignore

        logits = self.lm_head(self.norm_out(x))
        return logits, present_kvs

    @torch.no_grad()
    def update_router_biases(self):
        """Chiama update_bias() su tutti i MoE layer dopo ogni optimizer step."""
        for block in self.blocks:
            block.moe.update_bias() # type: ignore

def build_model(model_cfg: ModelConfig) -> DiffusionMoE:
    return DiffusionMoE(model_cfg)