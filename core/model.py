"""
Harold v0.7 — model.py
========================
Principali cambiamenti rispetto a v0.6:

  [v0.7-X1] x0-prediction invece di v-prediction (vel_pred → x0_pred)
             Il modello predice direttamente x0_emb nello spazio degli embedding,
             invece della velocità del flusso v = noise - x0.
             Vantaggi:
               - Target più stabile: errore diretto nello spazio degli embedding
               - CE loss consistente: ce_head(x0_pred) invece di ce_head(x_out)
               - Prerequisito per iterative decoding (v0.7-X2, futuro)
             Formula di conversione x0 → velocità per ODE step nel sampler:
               noise_pred = (x_t - (1 - t) * x0_pred) / t.clamp(min=1e-4)
               v = noise_pred - x0_pred

  [v0.7-X2] FlowMatchingSchedule: rimosso target_velocity, aggiunto target_x0
             La traiettoria lineare è invariata, cambia solo il target di regressione.

  [v0.7-X3] Harold.forward: (vel_pred, ce_logits, present_kvs)
                           → (x0_pred, ce_logits, present_kvs)
             ce_logits ora calcolati da ce_head(x0_pred) per consistenza.

  [v0.7-X4] Harold.compute_loss: MSE(x0_pred, x0_emb) invece di MSE(vel_pred, v_target)

  [v0.7-X5] self_cond_proj ora condiziona su x0_pred.mean(dim=1) invece di vel_prev.mean(dim=1)
             Semantica coerente con la predizione corrente.

  [v0.7-T1] Logit-Normal timestep sampling (default: logit_normal, std=0.5)
             Previene velocity collapse privilegiando t~0.5 dove il gradiente
             e piu informativo. Configurabile via ModelConfig.t_sampling:
             'logit_normal' (default), 'cosine', 'uniform' (baseline v0.6).

  [v0.7-M4] x0_pred norm monitoring in compute_loss
             Aggiunge 'x0_norm_mean', 'x0_norm_std', 'x0_var_tokens' ai metrics.
             norm_mean < 0.1 -> collapse verso zero; x0_var_tokens ~ 0 -> mode collapse.

  [v0.7-P4] Monitoring x0_pred norm in compute_loss
             compute_loss ritorna ora anche:
               x0_norm_mean:  norma media di x0_pred sui token validi
                              se << 1.0 → collapse verso zero
               x0_norm_std:   std della norma — se ~0 → mode collapse
               x0_var_tokens: varianza di x0_pred tra token
                              se ~0 → tutti i token predetti identici

  [v0.7-M3] Mamba2Block → Mamba3Block
             Sostituzione del mixer SSM con Mamba3 (Lahoti et al., ICLR 2026).
             Tre miglioramenti chiave rispetto a Mamba2:
               1. Ricorrenza exponential-trapezoidal: più espressiva di exponential-Euler
               2. State update complex-valued: abilita state tracking (risolve parity problem)
               3. MIMO (multi-input, multi-output): +espressività senza aumentare latenza decode
             Configurazione Harold:
               is_mimo=True, mimo_rank=4, chunk_size=16 (ottimale per bfloat16)
               is_outproj_norm=False (il JambaBlock genitore applica già AdaLN)
             I/O invariati: (B, T, d_model) → (B, T, d_model)

Invariato da v0.6:
  [v0.6-J1] Architettura Jamba ibrida SSM + Attention (pattern [SSM,SSM,SSM,Attn] x 9)
  [v0.6-J3] JambaBlock
  [v0.6-M1] MoE: 1 shared + 8 routed (top-2)
  [v0.5-M2] Loss senza Min-SNR weighting
  [v0.5-M4] Scaling 1.51B (d_model=1280, n_layers=36, n_heads=20, d_ff=3584)
  [OPT-*]   Tutte le ottimizzazioni precedenti (routing vettorializzato, sparse mask cache,
             RoPE bfloat16, AdaLN unbind, shared experts stack+mean, add_noise empty+normal_)

Dipendenze:
  pip install mamba-ssm causal-conv1d
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Callable, cast
from core.config import ModelConfig


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


class DeepSeekMoELayer(nn.Module):
    r"""DeepSeek-style Mixture-of-Experts FFN layer.

    Combina ``ds_moe_n_shared_experts`` shared experts (sempre attivi, SwiGLU)
    con ``moe_n_routed_experts`` routed experts (top-k per token, routing sigmoid
    condizionato su :math:`[x; t\_emb]`).

    Durante l'inferenza, una soglia adattiva sostituisce il top-k fisso:
    gli expert con affinity sopra ``threshold(t)`` vengono attivati, con
    la soglia interpolata tra ``threshold_base`` e ``threshold_min`` in funzione
    di :math:`t`.

    Args:
        config (:class:`ModelConfig`): configurazione del modello. Campi rilevanti:
            ``moe_n_routed_experts``, ``moe_top_k``, ``ds_moe_n_shared_experts``,
            ``d_model``, ``d_ff``, ``dropout``

    Shape:
        - Input ``x``: :math:`(B, L, d\_model)`
        - Input ``t_emb``: :math:`(B, d\_model)`
        - Output: :math:`(B, L, d\_model)`
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
            SharedExpert(config.d_model, config.moe_shared_hidden, dropout=config.dropout)
            for _ in range(config.ds_moe_n_shared_experts)
        ])
        self.routed_experts = nn.ModuleList([
            Expert(config.d_model, config.moe_routed_hidden, dropout=config.dropout)
            for _ in range(self.n_routed_experts)
        ])

        # [v0.7-OPT7] Router FiLM: separa condizionamento x e t_emb.
        # Invece di cat([x, t_emb]) → Linear(2d, E), usa:
        #   logit = W_x @ x + (W_t @ t_emb)  broadcast su T
        # Vantaggi:
        #   - Elimina repeat_interleave di t_emb (era (B*T, C) per concat)
        #   - W_t @ t_emb calcolato una volta per batch (B, E) → broadcast su T
        #   - Stessi parametri totali: 2*d*E = d*E + d*E
        # Breaking change rispetto a v0.6: shape router cambia — checkpoint incompatibili.
        self.router   = nn.Linear(config.d_model, self.n_routed_experts, bias=False)
        self.router_t = nn.Linear(config.d_model, self.n_routed_experts, bias=False)
        nn.init.normal_(self.router.weight,   std=0.01)
        nn.init.normal_(self.router_t.weight, std=0.01)

        self.register_buffer("router_bias", torch.zeros(self.n_routed_experts))
        self.router_indices: Optional[torch.Tensor] = None

    def _affinity(self, x_flat: torch.Tensor, t_bias_flat: torch.Tensor) -> torch.Tensor:
        # [v0.7-OPT7b] FiLM con broadcast puro — nessun repeat_interleave.
        # router(x_flat): (B*T, E)
        # t_bias_flat:    (B*T, E) — pre-calcolato in forward con repeat_interleave(T)
        # Su E=16 (invece di C=1792) il repeat è 112x più piccolo.
        return torch.sigmoid((self.router(x_flat) + t_bias_flat).float())

    def _compute_threshold(self, t_normalized: float) -> float:
        return self.threshold_base - (self.threshold_base - self.threshold_min) * t_normalized

    def _get_expert_outputs_vectorized(
        self,
        x_flat:       torch.Tensor,
        topk_indices: torch.Tensor,
        gates:        torch.Tensor,
    ) -> torch.Tensor:
        # [v0.7-OPT2] Dispatch senza Python loop con .item() — nessuna CPU-GPU sync.
        # Vecchia versione: loop su boundaries con .item() ad ogni expert
        #   → n_active_experts sync CPU per forward pass.
        # Nuova versione: loop su n_routed_experts con maschera booleana CUDA
        #   → 0 sync CPU, torch.compile fonde le op elementwise per expert.
        N, k   = topk_indices.shape
        d      = x_flat.shape[-1]
        output = torch.zeros(N, d, dtype=x_flat.dtype, device=x_flat.device)

        flat_indices = topk_indices.view(-1)
        flat_gates   = gates.view(-1)
        token_ids    = torch.arange(N, device=x_flat.device)                            .unsqueeze(1).expand(N, k).reshape(-1)

        valid_mask = flat_indices >= 0
        if not valid_mask.any():
            return output

        flat_indices_v = flat_indices[valid_mask]
        flat_gates_v   = flat_gates[valid_mask]
        token_ids_v    = token_ids[valid_mask]
        x_selected     = x_flat[token_ids_v]

        # [v0.7-OPT8] Expert dispatch con sort — una sola comparazione per expert.
        # Invece di (flat_indices_v == exp_id) per ogni exp_id — O(M*E) comparazioni —
        # ordiniamo per expert_id una volta (già fatto) e usiamo searchsorted O(log M).
        # Con M=B*T*k=32768 e E=16: da 524k a 16*log2(32768)≈240 operazioni.
        sort_idx       = flat_indices_v.argsort(stable=True)
        sorted_exp     = flat_indices_v[sort_idx]
        sorted_tok     = token_ids_v[sort_idx]
        sorted_x       = x_selected[sort_idx]
        sorted_g       = flat_gates_v[sort_idx]

        # Boundaries tra gruppi di expert — searchsorted senza .item()
        exp_ids_range  = torch.arange(self.n_routed_experts + 1,
                                      device=x_flat.device, dtype=sorted_exp.dtype)
        boundaries     = torch.searchsorted(sorted_exp.contiguous(),
                                            exp_ids_range.contiguous())  # (E+1,)

        for exp_id in range(self.n_routed_experts):
            s = boundaries[exp_id].item()
            e = boundaries[exp_id + 1].item()
            if s == e:
                continue
            tok_ids = sorted_tok[s:e]
            x_in    = sorted_x[s:e]
            g       = sorted_g[s:e].unsqueeze(1).to(x_flat.dtype)
            exp_out = self.routed_experts[exp_id](x_in)
            output.index_add_(0, tok_ids, (exp_out * g).to(output.dtype))

        return output

    def forward(
        self,
        x:            torch.Tensor,
        t_emb:        torch.Tensor,
        t_normalized: Optional[float] = None,
    ) -> torch.Tensor:
        B, T, C = x.shape
        N       = B * T
        x_flat  = x.view(N, C)

        # [v0.7-OPT7b] Router FiLM: t_bias (B, E) → (B*T, E) via expand+reshape.
        # expand è zero-copy (stride trick), reshape su tensor contiguo è zero-copy.
        # Su E=16: 112x meno memoria di repeat_interleave su C=1792.
        t_bias      = self.router_t(t_emb)                              # (B, E)
        t_bias_flat = t_bias.unsqueeze(1).expand(B, T, -1).reshape(N, -1)  # (B*T, E) zero-copy

        # [v0.7-OPT2b] SharedExpert: somma diretta invece di stack+mean.
        if len(self.shared_experts) == 1:
            shared_out = self.shared_experts[0](x_flat)
        else:
            shared_out = self.shared_experts[0](x_flat)
            for e in self.shared_experts[1:]:
                shared_out = shared_out + e(x_flat)
            shared_out = shared_out * (1.0 / len(self.shared_experts))

        s          = self._affinity(x_flat, t_bias_flat)
        sel_scores = s + self.router_bias

        if self.training or t_normalized is None:
            topk_vals    = torch.topk(sel_scores, self.top_k, dim=-1)
            topk_indices = topk_vals.indices
        else:
            threshold    = self._compute_threshold(t_normalized)
            k_max        = int(
                (sel_scores > threshold).sum(dim=-1)
                .clamp(self.top_k_min, self.n_routed_experts).max().item()
            )
            topk_vals    = torch.topk(sel_scores, k_max, dim=-1)
            topk_indices = topk_vals.indices
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

        routed_out = self._get_expert_outputs_vectorized(x_flat, topk_indices, gates)
        return ((shared_out + routed_out) / (len(self.shared_experts) + self.top_k)).view(B, T, C)

    @torch.no_grad()
    def update_bias(self):
        r"""Update router bias based on recent expert usage.

        Aggiorna ``router_bias`` con un passo di segno proporzionale alla
        differenza tra l'utilizzo medio e quello per singolo expert, incentivando
        il bilanciamento del carico. Deve essere chiamato una volta per optimizer
        step durante il training.

        .. note::
            Noop se ``router_indices`` è ``None`` (nessun forward eseguito dall'ultimo update).
        """
        if self.router_indices is None:
            return
        valid = self.router_indices[self.router_indices >= 0]
        if valid.numel() == 0:
            return
        counts = torch.bincount(valid.view(-1), minlength=self.n_routed_experts).float()
        self.router_bias += self.bias_update_gamma * (counts.mean() - counts).sign()
        self.router_indices = None


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_dim:             int,
        max_seq_len:          int,
        theta:                float = 10000.0,
        original_max_seq_len: int   = 1024,
        scale_factor:         float = 1.0,
        beta_fast:            int   = 32,
        beta_slow:            int   = 1,
    ):
        super().__init__()
        freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))

        if scale_factor > 1.0:
            orig_len     = float(original_max_seq_len)
            beta         = head_dim / (2 * math.pi * freqs * orig_len)
            mask_high    = beta > beta_fast
            mask_low     = beta < beta_slow
            freqs_ntk    = freqs
            freqs_linear = freqs / scale_factor
            blend        = (beta - beta_slow) / (beta_fast - beta_slow + 1e-8)
            freqs_mid    = freqs_linear * blend + freqs_ntk * (1.0 - blend)
            freqs  = torch.where(mask_high, freqs_linear,
                     torch.where(mask_low,  freqs_ntk, freqs_mid))
            mscale = 0.1 * math.log(scale_factor) + 1.0
        else:
            mscale = 1.0

        self.mscale: torch.Tensor
        self.cos_cached: torch.Tensor
        self.sin_cached: torch.Tensor
        self.register_buffer("mscale", torch.tensor(mscale, dtype=torch.float32))
        emb = torch.outer(torch.arange(max_seq_len, dtype=torch.float32), freqs)
        cache_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        self.register_buffer("cos_cached",
            torch.cat([torch.cos(emb), torch.cos(emb)], dim=-1).to(cache_dtype))
        self.register_buffer("sin_cached",
            torch.cat([torch.sin(emb), torch.sin(emb)], dim=-1).to(cache_dtype))

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        h = x.shape[-1] // 2
        return torch.cat([-x[..., h:], x[..., :h]], dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor, offset: int = 0):
        T      = q.shape[2]
        mscale = self.mscale.to(q.dtype)
        cos    = self.cos_cached[offset:offset + T].to(q.dtype).unsqueeze(0).unsqueeze(0)
        sin    = self.sin_cached[offset:offset + T].to(q.dtype).unsqueeze(0).unsqueeze(0)
        q_rot  = (q * cos + self._rotate_half(q) * sin) * mscale
        k_rot  = (k * cos + self._rotate_half(k) * sin) * mscale
        return q_rot, k_rot


class BlockCausalAttention(nn.Module):
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
        self.window_size  = config.dsa_window_size
        self.global_every = config.dsa_global_every

        self.latent_dim = config.mla_latent_dim
        self.q_proj  = nn.Linear(config.d_model, config.n_heads * self.head_dim, bias=False)
        self.kv_down = nn.Linear(config.d_model, self.latent_dim, bias=False)
        self.k_up    = nn.Linear(self.latent_dim, config.n_kv_heads * self.head_dim, bias=False)
        self.v_up    = nn.Linear(self.latent_dim, config.n_kv_heads * self.head_dim, bias=False)
        self.o_proj  = nn.Linear(config.n_heads * self.head_dim, config.d_model, bias=False)

        self.rope = RotaryEmbedding(
            head_dim             = self.head_dim,
            max_seq_len          = config.max_seq_len,
            theta                = config.rope_theta,
            original_max_seq_len = config.rope_original_max_seq_len,
            scale_factor         = config.rope_scale_factor,
        )
        self.resid_drop = nn.Dropout(config.dropout)
        self._sparse_mask_cache: Dict[Tuple[int, str], torch.Tensor] = {}

        self._use_flash = False
        self._flash_attn_func: Optional[Callable] = None
        if config.use_flash_attention:
            try:
                from flash_attn import flash_attn_func  # type: ignore
                self._flash_attn_func = flash_attn_func
                self._use_flash = True
            except ImportError:
                pass

    def _build_sparse_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        key = (seq_len, str(device))
        if key in self._sparse_mask_cache:
            return self._sparse_mask_cache[key]

        idx            = torch.arange(seq_len, device=device)
        block_idx      = idx // self.block_size
        future_block   = block_idx.unsqueeze(0) > block_idx.unsqueeze(1)
        outside_window = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs() > self.window_size
        global_visible = (idx % self.global_every == 0).unsqueeze(0).expand(seq_len, seq_len)
        mask = (outside_window & ~global_visible) | future_block
        self._sparse_mask_cache[key] = mask
        return mask

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

        full_len = k.shape[2]

        if self._use_flash and past_kv is None:
            k_exp = k.repeat_interleave(self.n_kv_groups, dim=1)
            v_exp = v.repeat_interleave(self.n_kv_groups, dim=1)
            assert self._flash_attn_func is not None
            y = self._flash_attn_func(
                q.transpose(1, 2),
                k_exp.transpose(1, 2),
                v_exp.transpose(1, 2),
                dropout_p=self.dropout if self.training else 0.0,
                causal=False,
                window_size=(self.window_size, self.window_size),
            ).transpose(1, 2)
        else:
            k_exp     = k.repeat_interleave(self.n_kv_groups, dim=1)
            v_exp     = v.repeat_interleave(self.n_kv_groups, dim=1)
            mask      = self._build_sparse_mask(full_len, x.device)
            if T < full_len:
                mask  = mask[-T:, :]
            attn_bias = torch.zeros(T, full_len, device=x.device, dtype=x.dtype)
            attn_bias.masked_fill_(mask, float("-inf"))
            attn_bias = attn_bias.unsqueeze(0).unsqueeze(0)

            y = F.scaled_dot_product_attention(
                q, k_exp, v_exp, attn_mask=attn_bias,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,
            )

        out = self.resid_drop(self.o_proj(y.transpose(1, 2).contiguous().view(B, T, C)))
        return out, (c_kv_full if use_cache else None)


class AdaLN(nn.Module):
    # [v0.7-OPT3] Forward fused: split manuale invece di chunk(),
    # poi mul_ e add_ in-place fusi da torch.compile.
    # Elimina 1 kernel chunk + 2 unsqueeze separati rispetto alla versione v0.6.
    def __init__(self, d_model: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.proj = nn.Linear(d_model, 2 * d_model, bias=True)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        proj  = self.proj(t_emb)                    # (B, 2*d)
        d     = proj.shape[-1] // 2
        scale = proj[..., :d].unsqueeze(1)          # (B, 1, d)
        shift = proj[..., d:].unsqueeze(1)          # (B, 1, d)
        return self.norm(x).mul_(1.0 + scale).add_(shift)


class Mamba3Block(nn.Module):
    r"""Wrapper attorno a :class:`mamba_ssm.Mamba3` con dropout residuale.

    Sostituisce :class:`Mamba2Block` a partire da v0.7 (`[v0.7-M3]`).
    L'input viene già normalizzato da :class:`AdaLN` nel :class:`JambaBlock` genitore.

    Mamba3 introduce tre miglioramenti rispetto a Mamba2 (Lahoti et al., ICLR 2026):

    1. **Exponential-trapezoidal discretization** — ricorrenza più espressiva
       di exponential-Euler, con migliore approssimazione dell'ODE sottostante.
    2. **Complex-valued state update** — abilita state tracking ricco, risolvendo
       task come il parity problem che Mamba2 non riesce a risolvere.
    3. **MIMO (multi-input, multi-output)** — aumenta l'espressività del modello
       senza incrementare la dimensione dello stato né la latenza di decoding.

    .. rubric:: Configurazione Harold

    ``headdim`` è allineato a ``d_model // n_heads`` per consistenza con
    :class:`BlockCausalAttention`. Con MIMO abilitato (``is_mimo=True``),
    ``chunk_size=16`` è il valore ottimale per bfloat16 (``64 // mimo_rank``).
    ``is_outproj_norm=False`` perché il :class:`JambaBlock` genitore applica
    già :class:`AdaLN` sia prima che dopo il mixer.

    Args:
        config (:class:`ModelConfig`): configurazione del modello. Campi rilevanti:
            ``d_model``, ``mamba_d_state``, ``mamba_mimo_rank``, ``n_heads``, ``dropout``

    Shape:
        - Input: :math:`(B, L, d\_model)` — già normalizzato da AdaLN
        - Output: :math:`(B, L, d\_model)`

    .. note::
        Richiede ``mamba-ssm >= 2.x`` con supporto Mamba3:
        ``pip install mamba-ssm causal-conv1d``

    .. _Lahoti et al. ICLR 2026:
        https://arxiv.org/abs/2603.15569
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        # [v0.7-M3] Mamba3 con fallback automatico su Mamba2.
        # Mamba3 non è ancora disponibile su PyPI (aprile 2026) e richiede
        # kernel CUDA compilati per l'architettura target. Se non disponibile,
        # Mamba2 è un sostituto equivalente per convergenza e scaling law check.
        try:
            from mamba_ssm import Mamba3  # type: ignore
            self.mamba = Mamba3(
                d_model         = config.d_model,
                d_state         = config.mamba_d_state,
                headdim         = config.d_model // config.n_heads,
                is_mimo         = False,
                is_outproj_norm = False,
                dtype           = torch.bfloat16,
            )
            self._using_mamba3 = True
        except (ImportError, Exception):
            from mamba_ssm import Mamba2  # type: ignore
            import warnings
            warnings.warn(
                "Mamba3 non disponibile — fallback su Mamba2. "
                "Per il full run installare mamba3-release dal branch GitHub.",
                stacklevel=2,
            )
            self.mamba = Mamba2(
                d_model  = config.d_model,
                d_state  = config.mamba_d_state,
                headdim  = config.d_model // config.n_heads,
                expand   = 1,
            )
            self._using_mamba3 = False
        self.resid_drop = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""forward(x) -> Tensor

        Args:
            x (Tensor): input già normalizzato da AdaLN, shape :math:`(B, L, d\_model)`

        Returns:
            Tensor: output SSM, shape :math:`(B, L, d\_model)`

        Examples::

            >>> cfg = ModelConfig()
            >>> block = Mamba3Block(cfg).cuda().to(torch.bfloat16)
            >>> x = torch.randn(2, 512, cfg.d_model, device="cuda", dtype=torch.bfloat16)
            >>> block(x).shape
            torch.Size([2, 512, 1280])
        """
        return self.resid_drop(self.mamba(x))


class JambaBlock(nn.Module):
    r"""Blocco ibrido Jamba: mixer (Mamba2 o Attention) seguito da MoE.

    Ogni blocco applica uno dei due mixer in base a ``is_attn_layer``:

    - ``False`` (3 blocchi su 4): :class:`Mamba3Block` — SSM a complessità lineare,
      nessuna KV cache
    - ``True`` (ogni ``jamba_attn_every``-esimo layer): :class:`BlockCausalAttention`
      con MLA + sparse mask — supporta KV cache

    In entrambi i casi il mixer è preceduto da :class:`AdaLN` condizionato su ``t_emb``,
    e l'output passa attraverso :class:`DeepSeekMoELayer` con residual normalizzato.

    Args:
        config (:class:`ModelConfig`): configurazione del modello
        is_attn_layer (bool): se ``True``, usa :class:`BlockCausalAttention`;
            altrimenti usa :class:`Mamba2Block`

    Shape:
        - Input: :math:`(B, L, d\_model)`
        - Output: :math:`(B, L, d\_model)`
    """

    def __init__(self, config: ModelConfig, is_attn_layer: bool):
        super().__init__()
        self.is_attn_layer = is_attn_layer

        self.ada_ln_1 = AdaLN(config.d_model)
        self.ada_ln_2 = AdaLN(config.d_model)

        if is_attn_layer:
            self.mixer: nn.Module = BlockCausalAttention(config)
        else:
            self.mixer = Mamba3Block(config)

        self.moe = DeepSeekMoELayer(config)

    def forward(
        self,
        x:            torch.Tensor,
        t_emb:        torch.Tensor,
        past_kv:      Optional[torch.Tensor] = None,
        use_cache:    bool = False,
        kv_offset:    int  = 0,
        t_normalized: Optional[float] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        r"""forward(x, t_emb, ...) -> (x, present_kv)

        Args:
            x (Tensor): input, shape :math:`(B, L, d\_model)`
            t_emb (Tensor): timestep embedding, shape :math:`(B, d\_model)`
            past_kv (Tensor, optional): KV cache dal forward precedente (solo per
                layer attention). Default: ``None``
            use_cache (bool, optional): se ``True``, ritorna il KV state corrente.
                Default: ``False``
            kv_offset (int, optional): offset di posizione per RoPE (generazione
                incrementale). Default: ``0``
            t_normalized (float, optional): timestep normalizzato in :math:`[0, 1]`
                per la soglia adattiva del router MoE (solo inference). Default: ``None``

        Returns:
            tuple:
                - **x** (*Tensor*) – output del blocco, shape :math:`(B, L, d\_model)`
                - **present_kv** (*Tensor or None*) – KV state se ``use_cache=True``
                  e layer attention, altrimenti ``None``
        """
        normed = self.ada_ln_1(x, t_emb)

        if self.is_attn_layer:
            mixer_out, present_kv = cast(
                Tuple[torch.Tensor, Optional[torch.Tensor]],
                self.mixer(normed, past_kv=past_kv, use_cache=use_cache, kv_offset=kv_offset),
            )
        else:
            mixer_out = cast(torch.Tensor, self.mixer(normed))
            present_kv = None

        x       = x + mixer_out
        moe_out = self.moe(self.ada_ln_2(x, t_emb), t_emb, t_normalized=t_normalized)
        # [v0.7-OPT3b] F.normalize con max_norm semantics invece di norm+clamp+div.
        # Equivalente: divide solo se norm > 1, lascia invariato altrimenti.
        # F.normalize è un kernel fused — backward più efficiente della versione manuale.
        norm = moe_out.norm(dim=-1, keepdim=True).clamp(min=1.0)
        moe_out = moe_out / norm
        return x + moe_out, present_kv


class FlowMatchingSchedule(nn.Module):
    r"""Linear Conditional Flow Matching schedule (Lipman et al., 2022).

    Definisce una traiettoria lineare tra dati :math:`x_0` e rumore
    :math:`\varepsilon \sim \mathcal{N}(0, I)`:

    .. math::
        x_t = (1 - t_\mathrm{eff}) \, x_0 + t_\mathrm{eff} \, \varepsilon, \quad
        t_\mathrm{eff} = (1 - \sigma_\mathrm{min}) \, t + \sigma_\mathrm{min}

    .. rubric:: [v0.7-X2] Target: x0-prediction

    A differenza di v0.6 (che usava ``target_velocity``), il modello predice
    direttamente :math:`x_0`. Il target è quindi:

    .. math::
        x_0^* = x_0

    La velocità del flusso può essere ricavata dal sampler quando necessario:

    .. math::
        \hat{\varepsilon} = \frac{x_t - (1 - t) \hat{x}_0}{t}, \quad
        \hat{v} = \hat{\varepsilon} - \hat{x}_0

    Args:
        sigma_min (float, optional): rumore residuo a :math:`t=0` per stabilità
            numerica. Default: ``1e-4``

    .. _Lipman et al. 2022:
        https://arxiv.org/abs/2210.02747
    """

    def __init__(self, sigma_min: float = 1e-4, t_sampling: str = "logit_normal",
                 t_logit_normal_std: float = 0.5):
        super().__init__()
        self.sigma_min          = sigma_min
        self.t_sampling         = t_sampling
        self.t_logit_normal_std = t_logit_normal_std

    def sample_t(self, B: int, device: torch.device) -> torch.Tensor:
        r"""sample_t(B, device) -> Tensor

        Campiona i timestep di training secondo la strategia configurata.

        .. rubric:: Strategie disponibili

        - ``"uniform"``: :math:`t \sim U[0, 1]` — baseline, campionamento flat
        - ``"logit_normal"``: :math:`t = \sigma(z),\; z \sim \mathcal{N}(0, s^2)` —
          concentra i campioni intorno a :math:`t=0.5` dove il gradiente è più
          informativo. Usato in SD3 e altri modelli FM recenti. Previene il
          velocity collapse privilegiando i timestep difficili.
        - ``"cosine"``: densità proporzionale a :math:`\sin(\pi t)` — simile a
          logit_normal ma con tails più pesanti verso :math:`t=0` e :math:`t=1`.

        Args:
            B (int): batch size
            device (torch.device): device di output

        Returns:
            Tensor: timestep in :math:`[0, 1]`, shape :math:`(B,)`
        """
        if self.t_sampling == "logit_normal":
            u = torch.randn(B, device=device) * self.t_logit_normal_std
            return torch.sigmoid(u)
        elif self.t_sampling == "cosine":
            u = torch.rand(B, device=device)
            return 0.5 * (1.0 - torch.cos(math.pi * u))
        else:  # uniform
            return torch.rand(B, device=device)

    def interpolate(
        self,
        x0:    torch.Tensor,
        noise: torch.Tensor,
        t:     torch.Tensor,
    ) -> torch.Tensor:
        r"""interpolate(x0, noise, t) -> Tensor

        Calcola :math:`x_t` lungo la traiettoria lineare.

        Args:
            x0 (Tensor): embedding dei token originali, shape :math:`(B, L, d\_model)`
            noise (Tensor): rumore gaussiano, shape :math:`(B, L, d\_model)`
            t (Tensor): timestep, shape :math:`(B,)`

        Returns:
            Tensor: embedding interpolato :math:`x_t`, shape :math:`(B, L, d\_model)`
        """
        t_eff = (1.0 - self.sigma_min) * t + self.sigma_min
        t_eff = t_eff.view(-1, 1, 1)
        return (1.0 - t_eff) * x0 + t_eff * noise

    def target_x0(self, x0: torch.Tensor) -> torch.Tensor:
        r"""target_x0(x0) -> Tensor

        Ritorna il target di regressione per x0-prediction.

        Il target è :math:`x_0` stesso — l'embedding del token originale.
        Identità esplicita per chiarezza semantica e compatibilità con il
        sampler che usa questa firma per ricavare :math:`\hat{v}`.

        .. note::
            Sostituisce ``target_velocity`` di v0.6. La traiettoria lineare
            è invariata; cambia solo il target di regressione.

        Args:
            x0 (Tensor): embedding dei token originali, shape :math:`(B, L, d\_model)`

        Returns:
            Tensor: target x0, shape :math:`(B, L, d\_model)` — uguale all'input
        """
        return x0

    def add_noise(
        self,
        x0: torch.Tensor,
        t:  torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""add_noise(x0, t) -> (x_t, noise)

        Campiona rumore e calcola :math:`x_t` lungo la traiettoria.

        Args:
            x0 (Tensor): embedding dei token originali, shape :math:`(B, L, d\_model)`
            t (Tensor): timestep, shape :math:`(B,)`

        Returns:
            tuple:
                - **x_t** (*Tensor*) – embedding rumoroso, shape :math:`(B, L, d\_model)`
                - **noise** (*Tensor*) – rumore campionato :math:`\varepsilon`,
                  shape :math:`(B, L, d\_model)`
        """
        noise = torch.empty_like(x0).normal_()
        x_t   = self.interpolate(x0, noise, t)
        return x_t, noise


class Harold(nn.Module):
    r"""Harold v0.7 — diffusion language model ibrido Jamba con x0-prediction e Mamba3.

    Architettura:
        - 36 layer Jamba [Mamba3 × 3, Attention] × 9
        - MoE DeepSeek (1 shared + 8 routed top-2) su ogni layer
        - Flow Matching con traiettoria lineare
        - **[v0.7-X1]** x0-prediction: predice direttamente :math:`\hat{x}_0`
          nello spazio degli embedding invece della velocità del flusso
        - **[v0.7-M3]** :class:`Mamba3Block` come mixer SSM al posto di Mamba2,
          con MIMO abilitato (``mimo_rank=4``)

    Args:
        config (:class:`ModelConfig`): configurazione completa del modello

    .. note::
        Il checkpoint v0.6 non è direttamente compatibile per due ragioni:
        ``vel_pred`` è rinominato ``x0_pred``, e ``mamba`` internamente usa
        Mamba3 invece di Mamba2 (architettura diversa nei pesi SSM).
        Usare uno script di migrazione per i pesi non-SSM (attention, MoE, ecc.).
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config        = config
        self.d_model       = config.d_model
        emb_vocab          = config.vocab_size + 1
        self.emb_vocab     = emb_vocab
        self.mask_token_id = config.vocab_size

        self.token_emb = nn.Embedding(emb_vocab, config.d_model)

        self.time_emb = nn.Sequential(
            nn.Linear(config.d_model, config.d_model * 4),
            nn.SiLU(),
            nn.Linear(config.d_model * 4, config.d_model),
        )

        # [v0.7-X5] self_cond_proj condiziona su x0_pred.mean(dim=1) (era vel_prev.mean)
        self.self_cond_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        nn.init.normal_(self.self_cond_proj.weight, std=0.02)

        self.cfg_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        nn.init.zeros_(self.cfg_proj.weight)

        half  = config.d_model // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32) / half)
        self.t_freqs: torch.Tensor
        self.register_buffer("t_freqs", freqs)

        # [v0.7-M3] Mamba3Block per layer SSM (pattern [Mamba3, Mamba3, Mamba3, Attn] x 9)
        # layer_idx % jamba_attn_every == jamba_attn_every - 1  →  attention
        # Con n_layers=36, jamba_attn_every=4: idx 3,7,11,15,19,23,27,31,35 → 9 attention
        self.blocks = nn.ModuleList([
            JambaBlock(
                config,
                is_attn_layer=((i % config.jamba_attn_every) == (config.jamba_attn_every - 1)),
            )
            for i in range(config.n_layers)
        ])

        self.norm_out = nn.LayerNorm(config.d_model)

        # [v0.7-X1] x0_pred: predice x0_emb direttamente (era vel_pred → velocità)
        self.x0_pred = nn.Linear(config.d_model, config.d_model, bias=False)

        # [v0.7-X3] ce_head applicato a x0_pred invece di x_out per consistenza
        self.ce_head        = nn.Linear(config.d_model, emb_vocab, bias=False)
        self.ce_head.weight = self.token_emb.weight

        self.schedule = FlowMatchingSchedule(
            sigma_min           = config.flow_sigma_min,
            t_sampling          = config.t_sampling,
            t_logit_normal_std  = config.t_logit_normal_std,
        )

        self._init_weights()

    def _init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                std = (0.02 / math.sqrt(2 * self.config.n_layers)
                       if any(name.endswith(s) for s in ("o_proj", "w2", "v_up", "x0_pred"))
                       else 0.02)
                nn.init.normal_(module.weight, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def get_timestep_embedding(self, t: torch.Tensor) -> torch.Tensor:
        r"""get_timestep_embedding(t) -> Tensor

        Embedding sinusoidale del timestep di diffusione.

        [v0.7-OPT6] Usa stack+view invece di cat([cos, sin]) per evitare
        una delle due allocazioni intermedie. stack produce (B, half, 2),
        view lo riorganizza a (B, d_model) senza copia — zero-copy reshape.

        Args:
            t (Tensor): timestep in :math:`[0, 1]`, shape :math:`(B,)`

        Returns:
            Tensor: embedding shape :math:`(B, d\_model)`
        """
        args = (t.float() * 1000.0)[:, None] * self.t_freqs[None]  # (B, half)
        emb  = torch.stack([torch.cos(args), torch.sin(args)], dim=-1)  # (B, half, 2)
        emb  = emb.view(t.shape[0], -1)                                  # (B, d_model)
        if self.d_model % 2:
            emb = F.pad(emb, (0, 1))
        return emb

    def forward(
        self,
        x_t:             torch.Tensor,
        t:               torch.Tensor,
        self_cond:       Optional[torch.Tensor] = None,
        ctx_emb:         Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.Tensor]] = None,
        use_cache:       bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[List]]:
        r"""forward(x_t, t, self_cond=None, ...) -> (x0_pred, ce_logits, present_kvs)

        Forward pass completo attraverso il backbone Jamba.

        Condiziona su ``t`` via embedding sinusoidale + MLP, poi processa
        attraverso :attr:`blocks` (Mamba2 e :class:`BlockCausalAttention` alternati,
        ciascuno seguito da :class:`DeepSeekMoELayer`).

        .. rubric:: [v0.7-X3] Output semantics

        L'head di output predice direttamente :math:`\hat{x}_0` nello spazio
        degli embedding. I logits CE sono calcolati da ``ce_head(x0_pred)``
        invece di ``ce_head(x_out)``, rendendo le due predizioni consistenti.

        Args:
            x_t (Tensor): embedding rumorosi, shape :math:`(B, L, d\_model)`
            t (Tensor): timestep in :math:`[0, 1]`, shape :math:`(B,)`
            self_cond (Tensor, optional): segnale di self-conditioning
                :math:`\hat{x}_0` medio del passo precedente,
                shape :math:`(B, d\_model)`. Default: ``None``
            ctx_emb (Tensor, optional): context embedding per CFG,
                shape :math:`(B, d\_model)`. Default: ``None``
            past_key_values (list of Tensor, optional): KV cache per layer
                attention. Default: ``None``
            use_cache (bool, optional): se ``True``, ritorna i KV state correnti.
                Default: ``False``

        Returns:
            tuple:
                - **x0_pred** (*Tensor*) – predizione di :math:`\hat{x}_0`,
                  shape :math:`(B, L, d\_model)`
                - **ce_logits** (*Tensor*) – logits sul vocabolario,
                  shape :math:`(B, L, V+1)`
                - **present_kvs** (*list or None*) – KV cache se ``use_cache=True``,
                  altrimenti ``None``

        Examples::

            >>> cfg = ModelConfig()
            >>> model = Harold(cfg).cuda()
            >>> x_t = torch.randn(2, 512, cfg.d_model).cuda()
            >>> t = torch.rand(2).cuda()
            >>> x0_pred, logits, _ = model(x_t, t)
            >>> x0_pred.shape, logits.shape
            (torch.Size([2, 512, 1280]), torch.Size([2, 512, 32001]))
        """
        kv_offset = past_key_values[0].shape[1] if past_key_values is not None else 0
        x = x_t

        t_emb = self.time_emb(self.get_timestep_embedding(t))

        if self_cond is not None:
            t_emb = t_emb + self.self_cond_proj(self_cond.detach())

        if ctx_emb is not None:
            t_emb = t_emb + self.cfg_proj(ctx_emb.detach())

        t_normalized = t.float().mean().item() if not self.training else None

        present_kvs: Optional[List] = [] if use_cache else None

        for i, block in enumerate(self.blocks):
            past_kv = past_key_values[i] if past_key_values is not None else None

            x, present_kv = block(
                x, t_emb,
                past_kv=past_kv, use_cache=use_cache,
                kv_offset=kv_offset, t_normalized=t_normalized,
            )

            if present_kvs is not None:
                present_kvs.append(present_kv)

        x_out    = self.norm_out(x)

        # [v0.7-X1] Predice x0_emb direttamente (non la velocità)
        x0_pred  = self.x0_pred(x_out)

        # [v0.7-X3] CE logits da x0_pred per consistenza con la predizione principale
        ce_logits = self.ce_head(x0_pred)

        return x0_pred, ce_logits, present_kvs

    def compute_loss(
        self,
        x0:             torch.Tensor,
        mask:           torch.Tensor,
        ce_weight:      float = 0.1,
        fixed_t:        Optional[torch.Tensor] = None,
        self_cond_prob: float = 0.0,
        ctx_emb:        Optional[torch.Tensor] = None,
        p_uncond:       float = 0.0,
    ) -> Tuple[torch.Tensor, Dict]:
        r"""compute_loss(x0, mask, ce_weight=0.1, ...) -> (loss, metrics)

        Flow Matching loss con x0-prediction.

        Campiona :math:`t \sim U[0,1]` per esempio, interpola gli embedding lungo
        la traiettoria lineare :math:`x_t = (1-t) x_0 + t \varepsilon`, poi calcola:

        .. math::
            \mathcal{L} = \underbrace{\mathrm{MSE}(\hat{x}_0,\, x_0)}_{\text{score loss}}
                        + w_\mathrm{CE} \cdot \mathrm{CE}(\hat{x}_0 W_\mathrm{emb}^\top,\, y)

        .. rubric:: [v0.7-X4] Differenze rispetto a v0.6

        - Target MSE: ``x0_emb`` invece di ``v_target = noise - x0_emb``
        - CE logits: da ``ce_head(x0_pred)`` — già incluso nel forward
        - Self-conditioning: condiziona su ``x0_prev.mean(dim=1)`` invece di
          ``vel_prev.mean(dim=1)``

        Nessun Min-SNR weighting — target uniforme su tutti i :math:`t`.

        Args:
            x0 (LongTensor): token id originali, shape :math:`(B, L)`
            mask (BoolTensor): posizioni che contribuiscono alla loss,
                shape :math:`(B, L)`
            ce_weight (float, optional): peso della cross-entropy ausiliaria.
                Default: ``0.1``
            fixed_t (Tensor, optional): timestep fisso shape :math:`(B,)`.
                Se ``None``, campiona :math:`t \sim U[0,1]`. Default: ``None``
            self_cond_prob (float, optional): probabilità di self-conditioning
                ad ogni step. Default: ``0.0``
            ctx_emb (Tensor, optional): context embedding per CFG,
                shape :math:`(B, d\_model)`. Default: ``None``
            p_uncond (float, optional): probabilità di drop del contesto (CFG training).
                Default: ``0.0``

        Returns:
            tuple:
                - **loss** (*Tensor*) – loss scalare totale
                - **metrics** (*dict*) – chiavi: ``"score"``, ``"ce"``, ``"total"``.
                  Se ``fixed_t`` è impostato, include anche ``"total_per_sample"``,
                  ``"score_per_sample"``, ``"ce_per_sample"``.

        Examples::

            >>> loss, metrics = model.compute_loss(tokens, mask)
            >>> loss.backward()
            >>> metrics["score"], metrics["ce"]
            (0.312, 4.21)
        """
        device = x0.device

        if mask.sum() == 0:
            zero = torch.tensor(0.0, device=device, requires_grad=True)
            return zero, {"score": 0.0, "ce": 0.0, "total": 0.0}

        B = x0.shape[0]
        t = fixed_t if fixed_t is not None else self.schedule.sample_t(B, device)

        with torch.no_grad():
            x0_emb = self.token_emb(x0)

        x_t, _ = self.schedule.add_noise(x0_emb, t)

        # [v0.7-X2] Target è x0_emb direttamente — target_x0() è un'identità,
        # usiamo x0_emb direttamente per evitare l'alias non necessario.
        # [v0.7-OPT5] Rimosso target_x0() call — x0_target is x0_emb per definizione.

        # [v0.7-X5] Self-conditioning su x0_prev invece di vel_prev
        self_cond: Optional[torch.Tensor] = None
        if self_cond_prob > 0 and torch.rand(1).item() < self_cond_prob:
            with torch.no_grad():
                x0_prev, _, _ = self.forward(x_t, t, self_cond=None, ctx_emb=None)
            self_cond = x0_prev.mean(dim=1).detach()

        cfg_emb: Optional[torch.Tensor] = None
        if ctx_emb is not None:
            if p_uncond > 0 and torch.rand(1).item() < p_uncond:
                cfg_emb = torch.zeros_like(ctx_emb)
            else:
                cfg_emb = ctx_emb

        x0_pred, ce_logits, _ = self.forward(x_t, t, self_cond=self_cond, ctx_emb=cfg_emb)

        # [v0.7-X4] MSE su x0_emb invece di v_target
        per_token_mse = F.mse_loss(x0_pred, x0_emb, reduction="none").mean(dim=-1)  # [v0.7-OPT5]
        loss_score    = per_token_mse[mask].mean()

        x0_for_ce = x0.masked_fill(~mask, -100)
        loss_ce   = F.cross_entropy(
            ce_logits.view(-1, ce_logits.size(-1)),
            x0_for_ce.view(-1),
            ignore_index=-100,
            reduction="mean",
        )
        total = loss_score + ce_weight * loss_ce

        extra: Dict = {}
        if fixed_t is not None:
            per_sample_score = per_token_mse.mean(dim=-1)
            per_sample_ce    = F.cross_entropy(
                ce_logits.view(-1, ce_logits.size(-1)),
                x0_for_ce.view(-1),
                ignore_index=-100,
                reduction="none",
            ).view(B, -1).mean(dim=-1)
            per_sample_total = per_sample_score + ce_weight * per_sample_ce
            extra = {
                "total_per_sample": per_sample_total.detach(),
                "score_per_sample": per_sample_score.detach(),
                "ce_per_sample":    per_sample_ce.detach(),
            }

        # [v0.7-P4] Monitoring x0_pred norm — rileva velocity/x0 collapse
        # - x0_pred_norm_mean << 1.0: collapse verso zero
        # - x0_pred_norm_std  ~  0.0: mode collapse (tutti i token → stesso embedding)
        # Calcolato solo su token validi (mask=True) per efficienza
        with torch.no_grad():
            x0_masked     = x0_pred[mask]                              # (N_valid, D)
            x0_norms      = x0_masked.norm(dim=-1)                     # (N_valid,)
            x0_norm_mean  = x0_norms.mean().item()
            x0_norm_std   = x0_norms.std().item() if x0_norms.numel() > 1 else 0.0
            # Varianza media tra token — se ~0 tutti i token predetti sono uguali
            x0_var_tokens = x0_pred.var(dim=1).mean().item()

        return total, {
            "score":           loss_score.item(),
            "ce":              loss_ce.item(),
            "total":           total.item(),
            "x0_norm_mean":    round(x0_norm_mean,  4),
            "x0_norm_std":     round(x0_norm_std,   4),
            "x0_var_tokens":   round(x0_var_tokens, 4),
            **extra,
        }

    @torch.no_grad()
    def decode_tokens(self, x: torch.Tensor) -> torch.Tensor:
        r"""decode_tokens(x) -> LongTensor

        Decodifica embedding continui in token id via cosine nearest-neighbor
        lookup nella embedding table.

        Args:
            x (Tensor): embedding, shape :math:`(B, L, d\_model)`

        Returns:
            LongTensor: token id, shape :math:`(B, L)`
        """
        emb_norm = F.normalize(self.token_emb.weight, dim=-1)
        x_norm   = F.normalize(x, dim=-1)
        return torch.einsum("bld,vd->blv", x_norm, emb_norm).argmax(dim=-1)

    @torch.no_grad()
    def update_router_biases(self):
        r"""Update all MoE router bias terms based on recent expert usage.

        Chiama :meth:`DeepSeekMoELayer.update_bias` su ogni blocco. Deve essere
        chiamato una volta per optimizer step durante il training.
        """
        for block in self.blocks:
            if isinstance(block.moe, DeepSeekMoELayer):
                block.moe.update_bias()


def build_model(model_cfg: ModelConfig) -> Harold:
    return Harold(model_cfg)