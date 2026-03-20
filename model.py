"""
Harold v3 — model.py
=====================
Architettura con VP-SDE continuous diffusion.

  Diffusion:
    - VP-SDE (Variance Preserving SDE) con beta_min/beta_max
    - Modello predice il rumore ε (epsilon prediction)
    - Score matching loss sui token non-padding
    - CE ausiliaria weight-tied per stabilizzazione e decoding
    - compute_loss centralizzato nel modello (gestisce self-cond internamente)

  Fix rispetto alla versione precedente:
    - [FIX #1] compute_loss: score matching (MSE su ε) + CE ausiliaria
               accetta fixed_t per valutazione per-timestep
               gestisce self_cond_prob internamente
    - [FIX #2] VP-SDE schedule sostituisce masking discreto
    - [FIX #3] shared_out diviso fuori dal loop
    - [FIX #4] self_cond sempre .detach()
    - [FIX #5] eps_pred con init standard std=0.02
    - [FIX #6] modello predice ε invece di x0
    - [FIX #7] get_timestep_embedding riscalato *1000 per t in [0,1]
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
from config import ModelConfig


# ─────────────────────────────────────────────────────────────────────────────
# Expert e SharedExpert
# ─────────────────────────────────────────────────────────────────────────────

class Expert(nn.Module):
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
    def __init__(self, n_embd: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.w1      = nn.Linear(n_embd, hidden_dim, bias=False)
        self.w3      = nn.Linear(n_embd, hidden_dim, bias=False)
        self.w2      = nn.Linear(hidden_dim, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.dropout(F.silu(self.w1(x)) * self.w3(x)))


# ─────────────────────────────────────────────────────────────────────────────
# DeepSeekMoELayer
# ─────────────────────────────────────────────────────────────────────────────

class DeepSeekMoELayer(nn.Module):
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

        self.router = nn.Linear(config.d_model * 2, self.n_routed_experts, bias=False)
        nn.init.normal_(self.router.weight, std=0.01)

        self.register_buffer("router_bias", torch.zeros(self.n_routed_experts))
        self.router_indices: Optional[torch.Tensor] = None

    def _affinity(self, x_flat: torch.Tensor, t_emb_flat: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.router(torch.cat([x_flat, t_emb_flat], dim=-1)).float())

    def _compute_threshold(self, t_normalized: float) -> float:
        return self.threshold_base - (self.threshold_base - self.threshold_min) * t_normalized

    def forward(
        self,
        x:            torch.Tensor,
        t_emb:        torch.Tensor,
        t_normalized: Optional[float] = None,
    ) -> torch.Tensor:
        B, T, C    = x.shape
        x_flat     = x.view(-1, C)
        t_emb_flat = t_emb.unsqueeze(1).expand(B, T, C).reshape(-1, C)

        # [FIX #3] divide UNA VOLTA fuori dal loop
        shared_out = torch.zeros_like(x_flat)
        for expert in self.shared_experts:
            shared_out += expert(x_flat)
        shared_out = shared_out / len(self.shared_experts)

        s          = self._affinity(x_flat, t_emb_flat)
        sel_scores = s + self.router_bias.to(s.device)  # type: ignore

        if self.training or t_normalized is None:
            topk_indices = torch.topk(sel_scores, self.top_k, dim=-1).indices
        else:
            threshold    = self._compute_threshold(t_normalized)
            k_max        = int(
                (sel_scores > threshold).sum(dim=-1)
                .clamp(self.top_k_min, self.n_routed_experts).max().item()
            )
            topk_indices = torch.topk(sel_scores, k_max, dim=-1).indices
            topk_scores  = sel_scores.gather(1, topk_indices)
            topk_indices = topk_indices.masked_fill(topk_scores <= threshold, -1)

        self.router_indices = topk_indices.detach()

        valid_mask = topk_indices >= 0
        s_sel      = s.gather(dim=1, index=topk_indices.clamp(min=0)) * valid_mask.to(s.dtype)
        denom      = s_sel.sum(dim=1, keepdim=True)
        gates      = torch.where(
            denom > 1e-9,
            s_sel / (denom + 1e-9),
            torch.full_like(s_sel, 1.0 / self.top_k),
        ).to(x.dtype)

        routed_out = torch.zeros_like(x_flat)
        for i in range(self.n_routed_experts):
            row_idx, which_k = (topk_indices == i).nonzero(as_tuple=True)
            if row_idx.numel() == 0:
                continue
            expert_out = self.routed_experts[i](x_flat.index_select(0, row_idx))
            routed_out.index_add_(0, row_idx, expert_out * gates[row_idx, which_k].unsqueeze(1))

        return ((shared_out + routed_out) / (len(self.shared_experts) + self.top_k)).view(B, T, C)

    @torch.no_grad()
    def update_bias(self):
        if self.router_indices is None:
            return
        valid = self.router_indices[self.router_indices >= 0]
        if valid.numel() == 0:
            return
        counts = torch.bincount(valid.view(-1), minlength=self.n_routed_experts).float()
        counts = counts.to(self.router_bias.device)  # type: ignore
        self.router_bias += self.bias_update_gamma * (counts.mean() - counts).sign()
        self.router_indices = None


# ─────────────────────────────────────────────────────────────────────────────
# RoPE
# ─────────────────────────────────────────────────────────────────────────────

class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10000.0):
        super().__init__()
        freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        emb   = torch.outer(torch.arange(max_seq_len, dtype=torch.float32), freqs)
        self.register_buffer("cos_cached", torch.cat([torch.cos(emb), torch.cos(emb)], dim=-1))
        self.register_buffer("sin_cached", torch.cat([torch.sin(emb), torch.sin(emb)], dim=-1))

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        h = x.shape[-1] // 2
        return torch.cat([-x[..., h:], x[..., :h]], dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor, offset: int = 0):
        T   = q.shape[2]
        cos = self.cos_cached[offset:offset + T].to(q.dtype).unsqueeze(0).unsqueeze(0)  # type: ignore
        sin = self.sin_cached[offset:offset + T].to(q.dtype).unsqueeze(0).unsqueeze(0)  # type: ignore
        return q * cos + self._rotate_half(q) * sin, k * cos + self._rotate_half(k) * sin


# ─────────────────────────────────────────────────────────────────────────────
# BlockCausalAttention — MLA + DSA
# ─────────────────────────────────────────────────────────────────────────────

class BlockCausalAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        assert config.n_heads % config.n_kv_heads == 0

        self.n_heads     = config.n_heads
        self.n_kv_groups = config.n_heads // config.n_kv_heads
        self.head_dim    = config.d_model // config.n_heads
        self.block_size  = config.block_size
        self.dropout     = config.dropout

        self.latent_dim = config.mla_latent_dim
        self.q_proj  = nn.Linear(config.d_model, config.n_heads * self.head_dim, bias=False)
        self.kv_down = nn.Linear(config.d_model, self.latent_dim, bias=False)
        self.k_up    = nn.Linear(self.latent_dim, config.n_kv_heads * self.head_dim, bias=False)
        self.v_up    = nn.Linear(self.latent_dim, config.n_kv_heads * self.head_dim, bias=False)
        self.o_proj  = nn.Linear(config.n_heads * self.head_dim, config.d_model, bias=False)

        self.window_size  = config.dsa_window_size
        self.global_every = config.dsa_global_every
        self.rope         = RotaryEmbedding(self.head_dim, config.max_seq_len, config.rope_theta)
        self.resid_drop   = nn.Dropout(config.dropout)

    def _build_sparse_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        idx            = torch.arange(seq_len, device=device)
        block_idx      = idx // self.block_size
        future_block   = block_idx.unsqueeze(0) > block_idx.unsqueeze(1)
        outside_window = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs() > self.window_size
        global_visible = (idx % self.global_every == 0).unsqueeze(0).expand(seq_len, seq_len)
        return (outside_window & ~global_visible) | future_block

    def forward(
        self,
        x:         torch.Tensor,
        past_kv:   Optional[torch.Tensor] = None,
        use_cache: bool = False,
        kv_offset: int  = 0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T, C = x.shape

        q    = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        c_kv = self.kv_down(x)
        k    = self.k_up(c_kv).view(B, T, -1, self.head_dim).transpose(1, 2)
        v    = self.v_up(c_kv).view(B, T, -1, self.head_dim).transpose(1, 2)

        q, k = self.rope(q, k, offset=kv_offset)

        if past_kv is not None:
            c_kv_full = torch.cat([past_kv, c_kv], dim=1)
            k = self.k_up(c_kv_full).view(B, -1, -1, self.head_dim).transpose(1, 2)
            v = self.v_up(c_kv_full).view(B, -1, -1, self.head_dim).transpose(1, 2)
        else:
            c_kv_full = c_kv

        full_len  = k.shape[2]
        k_exp     = k.repeat_interleave(self.n_kv_groups, dim=1)
        v_exp     = v.repeat_interleave(self.n_kv_groups, dim=1)

        mask      = self._build_sparse_mask(full_len, x.device)
        if T < full_len:
            mask  = mask[-T:, :]
        attn_bias = torch.zeros(T, full_len, device=x.device, dtype=x.dtype)
        attn_bias = attn_bias.masked_fill(mask, float("-inf")).unsqueeze(0).unsqueeze(0)

        y = F.scaled_dot_product_attention(
            q, k_exp, v_exp, attn_mask=attn_bias,
            dropout_p=self.dropout if self.training else 0.0, is_causal=False,
        )
        out = self.resid_drop(self.o_proj(y.transpose(1, 2).contiguous().view(B, T, C)))
        return out, (c_kv_full if use_cache else None)


# ─────────────────────────────────────────────────────────────────────────────
# AdaLN
# ─────────────────────────────────────────────────────────────────────────────

class AdaLN(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.proj = nn.Linear(d_model, 2 * d_model, bias=True)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        scale, shift = self.proj(t_emb).chunk(2, dim=-1)
        return self.norm(x) * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# ─────────────────────────────────────────────────────────────────────────────
# Block
# ─────────────────────────────────────────────────────────────────────────────

class Block(nn.Module):
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
        past_kv:      Optional[torch.Tensor] = None,
        use_cache:    bool = False,
        kv_offset:    int  = 0,
        t_normalized: Optional[float] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        attn_out, present_kv = self.attn(
            self.ada_ln_1(x, t_emb),
            past_kv=past_kv, use_cache=use_cache, kv_offset=kv_offset,
        )
        x       = x + attn_out
        moe_out = self.moe(self.ada_ln_2(x, t_emb), t_emb, t_normalized=t_normalized)
        moe_out = moe_out / (moe_out.norm(dim=-1, keepdim=True).clamp(min=1.0))
        return x + moe_out, present_kv


# ─────────────────────────────────────────────────────────────────────────────
# VP-SDE Schedule
# ─────────────────────────────────────────────────────────────────────────────

class VPSDESchedule(nn.Module):
    """
    Variance Preserving SDE schedule.

    Forward process: x_t = α(t)*x0 + σ(t)*ε,  ε ~ N(0,I)

      β(t)  = β_min + t*(β_max - β_min)
      α²(t) = exp(-(β_min*t + 0.5*(β_max-β_min)*t²))
      σ²(t) = 1 - α²(t)

    t ∈ [0,1]:
      t=0 → α≈1, σ≈0  (dato pulito)
      t=1 → α≈0, σ≈1  (rumore puro)
    """
    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0):
        super().__init__()
        self.beta_min = beta_min
        self.beta_max = beta_max

    def get_beta(self, t: torch.Tensor) -> torch.Tensor:
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def get_alpha_sigma(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        beta_int = self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t ** 2
        alpha_sq = torch.exp(-beta_int)
        alpha    = torch.sqrt(alpha_sq.clamp(min=0.0))
        sigma    = torch.sqrt((1.0 - alpha_sq).clamp(min=1e-8))
        return alpha, sigma

    def add_noise(
        self,
        x0: torch.Tensor,   # (B, L, D)
        t:  torch.Tensor,   # (B,) in [0,1]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """x_t = α(t)*x0 + σ(t)*ε. Ritorna (x_t, ε)."""
        alpha, sigma = self.get_alpha_sigma(t)
        eps = torch.randn_like(x0)
        x_t = alpha.view(-1, 1, 1) * x0 + sigma.view(-1, 1, 1) * eps
        return x_t, eps

    def get_snr(self, t: torch.Tensor) -> torch.Tensor:
        alpha, sigma = self.get_alpha_sigma(t)
        return (alpha / sigma.clamp(min=1e-8)) ** 2


# ─────────────────────────────────────────────────────────────────────────────
# Harold v3 — VP-SDE Continuous Diffusion
# ─────────────────────────────────────────────────────────────────────────────

class Harold(nn.Module):
    """
    Harold v0.4 — Continuous Diffusion con VP-SDE.

    Cambiamenti rispetto a v0.3:
      - Tokenizer: GPT-2 BPE (50,257 vocab, case-sensitive, byte-level)
                   invece di BERT-uncased (30,522 vocab, lowercase)
      - padding_idx rimosso — GPT-2 non ha token di padding dedicato,
        usiamo la mask esplicita nel compute_loss

    Pipeline training:
      1. x0_emb = token_emb(x0)
      2. x_t, ε = schedule.add_noise(x0_emb, t)    t ~ U[0,1]
      3. ε_pred = forward(x_t, t, self_cond)
      4. loss   = MSE(ε_pred, ε)[mask] + λ*CE(ce_logits, x0)[mask]

    Pipeline inferenza:
      - Parti da rumore puro x_1 ~ N(0,I)
      - Integra reverse SDE con Euler-Maruyama da t=1 a t=0
      - Decodifica finale con ce_logits.argmax
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config        = config
        self.d_model       = config.d_model
        emb_vocab          = config.vocab_size + 1
        self.emb_vocab     = emb_vocab
        self.mask_token_id = config.vocab_size

        # GPT-2: nessun padding_idx fisso — il padding è gestito dalla mask
        self.token_emb = nn.Embedding(emb_vocab, config.d_model)
        self.pos_emb   = nn.Embedding(config.max_seq_len, config.d_model)

        self.time_emb = nn.Sequential(
            nn.Linear(config.d_model, config.d_model * 4),
            nn.SiLU(),
            nn.Linear(config.d_model * 4, config.d_model),
        )

        self.self_cond_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        nn.init.normal_(self.self_cond_proj.weight, std=0.02)

        # CFG (Classifier-Free Guidance) — conditioning sul contesto conversazionale
        # Proietta il context embedding (mean pooling del prompt) nello spazio di t_emb
        # Init a zero: no-op all'inizio, il modello impara gradualmente a usarlo
        # Viene aggiunto solo durante il SFT, durante il pretraining ctx_emb=None
        self.cfg_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        nn.init.zeros_(self.cfg_proj.weight)

        # [FIX #7] Frequenze per sinusoidi — t verrà riscalato *1000
        half  = config.d_model // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32) / half)
        self.register_buffer("t_freqs", freqs)

        self.blocks   = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.norm_out = nn.LayerNorm(config.d_model)

        # [FIX #6] Predice ε (rumore) — non x0 né logit
        self.eps_pred = nn.Linear(config.d_model, config.d_model, bias=False)

        # CE head ausiliaria weight-tied con token_emb
        self.ce_head        = nn.Linear(config.d_model, emb_vocab, bias=False)
        self.ce_head.weight = self.token_emb.weight

        # [FIX #2] VP-SDE schedule
        self.schedule = VPSDESchedule(
            beta_min=config.diffusion_beta_min,
            beta_max=config.diffusion_beta_max,
        )

        self._init_weights()

    def _init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                std = (0.02 / math.sqrt(2 * self.config.n_layers)
                       if any(name.endswith(s) for s in ("o_proj", "w2", "v_up", "eps_pred"))
                       else 0.02)
                nn.init.normal_(module.weight, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def get_timestep_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """
        [FIX #7] Riscala t ∈ [0,1] → [0,1000] prima delle sinusoidi.
        Senza questo, gli argomenti sarebbero quasi-zero → embedding
        quasi costante → il modello non distingue i timestep.
        """
        args = (t.float() * 1000.0)[:, None] * self.t_freqs[None]  # type: ignore
        emb  = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.d_model % 2:
            emb = F.pad(emb, (0, 1))
        return emb

    def forward(
        self,
        x_t:             torch.Tensor,                   # (B, L, D)
        t:               torch.Tensor,                   # (B,) float in [0,1]
        self_cond:       Optional[torch.Tensor] = None,  # (B, D) già detached
        ctx_emb:         Optional[torch.Tensor] = None,  # (B, D) context embedding per CFG
        past_key_values: Optional[List[torch.Tensor]] = None,
        use_cache:       bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[List]]:
        B, L, _ = x_t.shape
        device  = x_t.device

        kv_offset = past_key_values[0].shape[1] if past_key_values is not None else 0
        pos       = torch.arange(kv_offset, kv_offset + L, device=device).unsqueeze(0).expand(B, -1)
        x         = x_t + self.pos_emb(pos)

        t_emb = self.time_emb(self.get_timestep_embedding(t))

        # Self-conditioning — sempre detached
        if self_cond is not None:
            t_emb = t_emb + self.self_cond_proj(self_cond.detach())

        # CFG conditioning — context embedding del prompt
        # Durante il pretraining ctx_emb=None → nessun effetto
        # Durante il SFT ctx_emb è il mean pooling degli embedding del prompt
        # Con p_uncond=0.1 ctx_emb viene azzerato → training unconditional
        if ctx_emb is not None:
            t_emb = t_emb + self.cfg_proj(ctx_emb.detach())

        t_normalized = t.float().mean().item() if not self.training else None

        present_kvs = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            past_kv = past_key_values[i] if past_key_values is not None else None
            x, present_kv = block(
                x, t_emb,
                past_kv=past_kv, use_cache=use_cache,
                kv_offset=kv_offset, t_normalized=t_normalized,
            )
            if use_cache:
                present_kvs.append(present_kv)  # type: ignore

        x_out    = self.norm_out(x)
        eps_pred = self.eps_pred(x_out)     # (B, L, D)
        ce_logits = self.ce_head(x_out)     # (B, L, V)

        return eps_pred, ce_logits, present_kvs

    def compute_loss(
        self,
        x0:             torch.Tensor,                   # (B, L) token IDs
        mask:           torch.Tensor,                   # (B, L) bool — token non-padding
        ce_weight:      float = 0.1,
        fixed_t:        Optional[torch.Tensor] = None,  # (B,) in [0,1]
        self_cond_prob: float = 0.0,
        ctx_emb:        Optional[torch.Tensor] = None,  # (B, D) context per CFG
        p_uncond:       float = 0.0,                    # prob di zerare ctx_emb (CFG)
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Score matching loss per VP-SDE con SNR weighting.

        Struttura:
          1. t ~ U[0,1]  (o fixed_t per valutazione per-timestep)
          2. x0_emb = token_emb(x0)
          3. x_t, ε = schedule.add_noise(x0_emb, t)
          4. (opz.) self-cond: primo forward senza hint → hint detached
          5. ε_pred = forward(x_t, t, self_cond)
          6. loss_score = SNR-weighted MSE(ε_pred, ε)[mask]
          7. loss_ce    = CE(ce_logits[mask], x0[mask])
          8. total = loss_score + ce_weight * loss_ce

        SNR weighting (Min-SNR, Hang et al. 2023):
          w(t) = SNR(t) / (SNR(t) + 1)  ∈ (0, 1)

          - t alto (tanto rumore, SNR≈0) → w≈0 → contribuisce poco
          - t basso (poco rumore, SNR>>1) → w→1 → contribuisce normalmente

          Razionale: a t alto la loss MSE è naturalmente rumorosa perché
          il segnale è quasi completamente corrotto. Pesarla meno stabilizza
          i gradienti e fa convergere più velocemente i timestep bassi
          (quelli che contano di più per la qualità finale della generazione).
        """
        device = x0.device

        if mask.sum() == 0:
            zero = torch.tensor(0.0, device=device, requires_grad=True)
            return zero, {"score": 0.0, "ce": 0.0, "total": 0.0}

        # ── Timestep ────────────────────────────────────────────────────────
        B = x0.shape[0]
        t = fixed_t if fixed_t is not None else torch.rand(B, device=device)

        # ── Forward process ──────────────────────────────────────────────────
        with torch.no_grad():
            x0_emb = self.token_emb(x0)             # (B, L, D)
        x_t, eps = self.schedule.add_noise(x0_emb, t)

        # ── Self-conditioning ────────────────────────────────────────────────
        self_cond: Optional[torch.Tensor] = None
        if self_cond_prob > 0 and torch.rand(1).item() < self_cond_prob:
            with torch.no_grad():
                eps_prev, _, _ = self.forward(x_t, t, self_cond=None, ctx_emb=None)
            self_cond = eps_prev.mean(dim=1).detach()   # (B, D)

        # ── CFG dropout ──────────────────────────────────────────────────────
        # Con prob p_uncond azzera il context embedding → training unconditional
        # Necessario per abilitare CFG in inferenza: guida = uncond + scale*(cond-uncond)
        cfg_emb: Optional[torch.Tensor] = None
        if ctx_emb is not None:
            if p_uncond > 0 and torch.rand(1).item() < p_uncond:
                cfg_emb = torch.zeros_like(ctx_emb)   # unconditional
            else:
                cfg_emb = ctx_emb

        # ── Predizione ───────────────────────────────────────────────────────
        eps_pred, ce_logits, _ = self.forward(x_t, t, self_cond=self_cond, ctx_emb=cfg_emb)

        # ── SNR weighting ────────────────────────────────────────────────────
        # w(t) = SNR(t) / (SNR(t) + 1)
        # Clampato a snr_clip per evitare che t≈0 domini con peso enorme
        snr_clip = 5.0
        snr      = self.schedule.get_snr(t).clamp(max=snr_clip)   # (B,)
        snr_w    = (snr / (snr + 1.0)).to(eps_pred.dtype)         # (B,) ∈ (0, 1)

        # ── Score matching con SNR weighting ─────────────────────────────────
        # MSE per token: (B, L, D) → media su D → (B, L)
        per_token_mse = F.mse_loss(eps_pred, eps, reduction="none").mean(dim=-1)

        # Applica peso SNR per batch item, poi media sui token mascherati
        # snr_w: (B,) → (B, 1) per broadcasting su L
        weighted_mse = per_token_mse * snr_w.unsqueeze(1)         # (B, L)
        loss_score   = weighted_mse[mask].mean()

        # ── CE ausiliaria ─────────────────────────────────────────────────────
        loss_ce = F.cross_entropy(ce_logits[mask], x0[mask], reduction="mean")

        total = loss_score + ce_weight * loss_ce

        return total, {
            "score": loss_score.item(),
            "ce":    loss_ce.item(),
            "total": total.item(),
        }

    @torch.no_grad()
    def decode_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Nearest neighbor lookup: cosine similarity con il vocabolario."""
        emb_norm = F.normalize(self.token_emb.weight, dim=-1)
        x_norm   = F.normalize(x, dim=-1)
        return torch.einsum("bld,vd->blv", x_norm, emb_norm).argmax(dim=-1)

    @torch.no_grad()
    def update_router_biases(self):
        for block in self.blocks:
            block.moe.update_bias()  # type: ignore


def build_model(model_cfg: ModelConfig) -> Harold:
    return Harold(model_cfg)