"""
Harold v0.5 — model.py
========================
Principali cambiamenti rispetto a v0.4:

  [v0.5-M1] VPSDESchedule → FlowMatchingSchedule
             Traiettoria lineare: x_t = (1-t)*x0 + t*noise
             Target: v = noise - x0  (velocità costante lungo la traiettoria)
             Nessuna schedule beta, nessun alpha/sigma.

  [v0.5-M2] compute_loss: Min-SNR weighting rimosso
             Il target è uniforme su tutti i timestep — non serve pesare.
             Loss = MSE(vel_pred, v_target) + ce_weight * CE

  [v0.5-M3] eps_pred → vel_pred
             Il modello predice la velocità del flusso, non il rumore.
             Stessa architettura, diverso significato dell'output.

  [v0.5-M4] Scaling a 1B parametri
             d_model=1280, n_layers=36, n_heads=20, d_ff=3584

Ottimizzazioni mantenute da v0.4:
  [OPT-M1] MoE routing vettorializzato
  [OPT-M2] Sparse mask cache on-device
  [OPT-M3] Gradient checkpointing
  [OPT-M5] RoPE cache in bfloat16
  [OPT-M6] AdaLN con unbind
  [OPT-M7] Shared experts stack+mean
  [OPT-M8] add_noise con empty+normal_
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as gradient_checkpoint
from typing import Optional, Tuple, List, Dict, Callable, cast
from config import ModelConfig


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

    def _get_expert_outputs_vectorized(
        self,
        x_flat:       torch.Tensor,
        topk_indices: torch.Tensor,
        gates:        torch.Tensor,
    ) -> torch.Tensor:
        N, k = topk_indices.shape
        d    = x_flat.shape[-1]
        output = torch.zeros(N, d, dtype=x_flat.dtype, device=x_flat.device)

        flat_indices = topk_indices.view(-1)
        flat_gates   = gates.view(-1)
        token_ids    = torch.arange(N, device=x_flat.device).unsqueeze(1).expand(N, k).reshape(-1)

        valid_mask = flat_indices >= 0
        if not valid_mask.any():
            return output

        flat_indices_v = flat_indices[valid_mask]
        flat_gates_v   = flat_gates[valid_mask]
        token_ids_v    = token_ids[valid_mask]
        x_selected     = x_flat[token_ids_v]

        sort_idx   = flat_indices_v.argsort(stable=True)
        sorted_exp = flat_indices_v[sort_idx]
        sorted_tok = token_ids_v[sort_idx]
        sorted_x   = x_selected[sort_idx]
        sorted_g   = flat_gates_v[sort_idx]

        boundaries = torch.cat([
            torch.tensor([-1], device=x_flat.device),
            (sorted_exp[1:] != sorted_exp[:-1]).nonzero(as_tuple=False).view(-1),
            torch.tensor([sorted_exp.numel() - 1], device=x_flat.device),
        ])

        for b in range(len(boundaries) - 1):
            start   = int((boundaries[b] + 1).item())
            end     = int((boundaries[b + 1] + 1).item())
            exp_id  = int(sorted_exp[start].item())
            chunk_x = sorted_x[start:end]
            chunk_g = sorted_g[start:end].unsqueeze(1).to(x_flat.dtype)
            chunk_t = sorted_tok[start:end]
            exp_out = self.routed_experts[exp_id](chunk_x)
            output.index_add_(0, chunk_t, (exp_out * chunk_g).to(output.dtype))

        return output

    def forward(
        self,
        x:            torch.Tensor,
        t_emb:        torch.Tensor,
        t_normalized: Optional[float] = None,
    ) -> torch.Tensor:
        B, T, C    = x.shape
        x_flat     = x.view(-1, C)
        t_emb_flat = t_emb.unsqueeze(1).expand(B, T, C).reshape(-1, C)

        if len(self.shared_experts) == 1:
            shared_out = self.shared_experts[0](x_flat)
        else:
            shared_out = torch.stack(
                [e(x_flat) for e in self.shared_experts], dim=0
            ).mean(dim=0)

        s          = self._affinity(x_flat, t_emb_flat)
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
    def __init__(self, d_model: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.proj = nn.Linear(d_model, 2 * d_model, bias=True)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        scale, shift = self.proj(t_emb).chunk(2, dim=-1)
        return self.norm(x) * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


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


# [v0.5-M1] FlowMatchingSchedule — sostituisce VPSDESchedule
class FlowMatchingSchedule(nn.Module):
    """
    Conditional Flow Matching con traiettoria lineare (Lipman et al. 2022).

    Forward process:
        x_t = (1 - t) * x0 + t * noise,  t in [0, 1]

    A t=0: x_t = x0  (dato pulito)
    A t=1: x_t = noise  (rumore puro)

    Il modello predice la velocità del flusso:
        v = noise - x0  (costante lungo la traiettoria)

    Questo è equivalente a predire la direzione da x0 verso noise,
    che è indipendente da t — target molto più stabile di epsilon-prediction.

    sigma_min: piccola quantità di rumore residuo a t=0 per stabilità numerica.
    """

    def __init__(self, sigma_min: float = 1e-4):
        super().__init__()
        self.sigma_min = sigma_min

    def interpolate(
        self,
        x0:    torch.Tensor,
        noise: torch.Tensor,
        t:     torch.Tensor,
    ) -> torch.Tensor:
        """
        Interpola linearmente tra x0 e noise al timestep t.
        x_t = (1 - t_eff) * x0 + t_eff * noise
        dove t_eff = (1 - sigma_min) * t + sigma_min
        per garantire un minimo di rumore anche a t=0.
        """
        t_eff = (1.0 - self.sigma_min) * t + self.sigma_min
        t_eff = t_eff.view(-1, 1, 1)
        return (1.0 - t_eff) * x0 + t_eff * noise

    def target_velocity(
        self,
        x0:    torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """
        Velocità target del flusso: v = noise - x0.
        Costante lungo la traiettoria — indipendente da t.
        """
        return noise - x0

    def add_noise(
        self,
        x0: torch.Tensor,
        t:  torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Aggiunge rumore secondo la traiettoria lineare.
        Ritorna (x_t, noise) per compatibilità con il vecchio VPSDESchedule.
        """
        noise = torch.empty_like(x0).normal_()
        x_t   = self.interpolate(x0, noise, t)
        return x_t, noise


# [v0.5] Harold — Flow Matching 1B
class Harold(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config        = config
        self.d_model       = config.d_model
        emb_vocab          = config.vocab_size + 1
        self.emb_vocab     = emb_vocab
        self.mask_token_id = config.vocab_size

        self.token_emb = nn.Embedding(emb_vocab, config.d_model)
        self.pos_emb   = nn.Embedding(config.max_seq_len, config.d_model)

        self.time_emb = nn.Sequential(
            nn.Linear(config.d_model, config.d_model * 4),
            nn.SiLU(),
            nn.Linear(config.d_model * 4, config.d_model),
        )

        self.self_cond_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        nn.init.normal_(self.self_cond_proj.weight, std=0.02)

        self.cfg_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        nn.init.zeros_(self.cfg_proj.weight)

        half  = config.d_model // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32) / half)
        self.t_freqs: torch.Tensor
        self.register_buffer("t_freqs", freqs)

        self.blocks   = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.norm_out = nn.LayerNorm(config.d_model)

        # [v0.5-M3] vel_pred invece di eps_pred — predice velocità del flusso
        self.vel_pred = nn.Linear(config.d_model, config.d_model, bias=False)

        self.ce_head        = nn.Linear(config.d_model, emb_vocab, bias=False)
        self.ce_head.weight = self.token_emb.weight

        # [v0.5-M1] FlowMatchingSchedule invece di VPSDESchedule
        self.schedule = FlowMatchingSchedule(
            sigma_min=config.flow_sigma_min,
        )

        self.use_gradient_checkpointing = getattr(config, "gradient_checkpointing", False)
        self._init_weights()

    def _init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                std = (0.02 / math.sqrt(2 * self.config.n_layers)
                       if any(name.endswith(s) for s in ("o_proj", "w2", "v_up", "vel_pred"))
                       else 0.02)
                nn.init.normal_(module.weight, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def get_timestep_embedding(self, t: torch.Tensor) -> torch.Tensor:
        args = (t.float() * 1000.0)[:, None] * self.t_freqs[None]
        emb  = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.d_model % 2:
            emb = F.pad(emb, (0, 1))
        return emb

    def _block_forward(
        self,
        block:        Block,
        x:            torch.Tensor,
        t_emb:        torch.Tensor,
        past_kv:      Optional[torch.Tensor],
        use_cache:    bool,
        kv_offset:    int,
        t_normalized: Optional[float],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return block(x, t_emb, past_kv=past_kv, use_cache=use_cache,
                     kv_offset=kv_offset, t_normalized=t_normalized)

    def forward(
        self,
        x_t:             torch.Tensor,
        t:               torch.Tensor,
        self_cond:       Optional[torch.Tensor] = None,
        ctx_emb:         Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.Tensor]] = None,
        use_cache:       bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[List]]:
        B, L, _ = x_t.shape
        device  = x_t.device

        kv_offset = past_key_values[0].shape[1] if past_key_values is not None else 0
        pos       = torch.arange(kv_offset, kv_offset + L, device=device).unsqueeze(0).expand(B, -1)
        x         = x_t + self.pos_emb(pos)

        t_emb = self.time_emb(self.get_timestep_embedding(t))

        if self_cond is not None:
            t_emb = t_emb + self.self_cond_proj(self_cond.detach())

        if ctx_emb is not None:
            t_emb = t_emb + self.cfg_proj(ctx_emb.detach())

        t_normalized = t.float().mean().item() if not self.training else None

        present_kvs: Optional[List] = [] if use_cache else None

        for i, block in enumerate(self.blocks):
            past_kv = past_key_values[i] if past_key_values is not None else None

            if self.use_gradient_checkpointing and self.training and not use_cache:
                x, present_kv = cast(
                    Tuple[torch.Tensor, Optional[torch.Tensor]],
                    gradient_checkpoint(
                        self._block_forward,
                        block, x, t_emb, past_kv, use_cache, kv_offset, t_normalized,
                        use_reentrant=False,
                    )
                )
            else:
                x, present_kv = block(
                    x, t_emb,
                    past_kv=past_kv, use_cache=use_cache,
                    kv_offset=kv_offset, t_normalized=t_normalized,
                )

            if present_kvs is not None:
                present_kvs.append(present_kv)

        x_out     = self.norm_out(x)
        # [v0.5-M3] predice velocità del flusso invece del rumore
        vel_pred  = self.vel_pred(x_out)
        ce_logits = self.ce_head(x_out)

        return vel_pred, ce_logits, present_kvs

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
        """
        [v0.5-M2] Flow Matching loss — nessun Min-SNR weighting.

        Target: v = noise - x0  (velocità costante lungo la traiettoria)
        Loss:   MSE(vel_pred, v_target) + ce_weight * CE

        Il target è uniforme su tutti i timestep — nessun peso necessario.
        """
        device = x0.device

        if mask.sum() == 0:
            zero = torch.tensor(0.0, device=device, requires_grad=True)
            return zero, {"score": 0.0, "ce": 0.0, "total": 0.0}

        B = x0.shape[0]
        t = fixed_t if fixed_t is not None else torch.rand(B, device=device)

        with torch.no_grad():
            x0_emb = self.token_emb(x0)

        # [v0.5-M1] Traiettoria lineare
        x_t, noise = self.schedule.add_noise(x0_emb, t)

        # [v0.5-M1] Target: velocità del flusso
        v_target = self.schedule.target_velocity(x0_emb, noise)

        self_cond: Optional[torch.Tensor] = None
        if self_cond_prob > 0 and torch.rand(1).item() < self_cond_prob:
            with torch.no_grad():
                vel_prev, _, _ = self.forward(x_t, t, self_cond=None, ctx_emb=None)
            self_cond = vel_prev.mean(dim=1).detach()

        cfg_emb: Optional[torch.Tensor] = None
        if ctx_emb is not None:
            if p_uncond > 0 and torch.rand(1).item() < p_uncond:
                cfg_emb = torch.zeros_like(ctx_emb)
            else:
                cfg_emb = ctx_emb

        vel_pred, ce_logits, _ = self.forward(x_t, t, self_cond=self_cond, ctx_emb=cfg_emb)

        # [v0.5-M2] Loss senza weighting — target uniforme su tutti i t
        per_token_mse = F.mse_loss(vel_pred, v_target, reduction="none").mean(dim=-1)
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

        return total, {
            "score": loss_score.item(),
            "ce":    loss_ce.item(),
            "total": total.item(),
            **extra,
        }

    @torch.no_grad()
    def decode_tokens(self, x: torch.Tensor) -> torch.Tensor:
        emb_norm = F.normalize(self.token_emb.weight, dim=-1)
        x_norm   = F.normalize(x, dim=-1)
        return torch.einsum("bld,vd->blv", x_norm, emb_norm).argmax(dim=-1)

    @torch.no_grad()
    def update_router_biases(self):
        for block in self.blocks:
            if isinstance(block.moe, DeepSeekMoELayer):
                block.moe.update_bias()


def build_model(model_cfg: ModelConfig) -> Harold:
    return Harold(model_cfg)