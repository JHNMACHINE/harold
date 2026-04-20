"""Attention mixer and rotary positional embedding.

Contains the block-causal attention used in the attention-track of the Jamba
layer stack, with Multi-head Latent Attention (MLA) for KV compression and
YaRN-style RoPE scaling.
"""

import math
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.config import ModelConfig


__all__ = ["RotaryEmbedding", "BlockCausalAttention"]


class RotaryEmbedding(nn.Module):
    r"""Rotary Position Embedding with optional YaRN-style frequency interpolation.

    Precomputes ``cos`` and ``sin`` tables up to ``max_seq_len`` and applies them
    to queries and keys at forward time. When ``scale_factor > 1``, uses the
    YaRN blending scheme to preserve short-range frequency behaviour while
    extending context.

    Args:
        head_dim: per-head dimension (must be even)
        max_seq_len: maximum sequence length to precompute tables for
        theta: RoPE base frequency
        original_max_seq_len: the context length at which the model was
            originally trained, used to derive the YaRN scaling curve
        scale_factor: context-extension factor (1.0 = vanilla RoPE)
        beta_fast: YaRN high-frequency cutoff wavelength
        beta_slow: YaRN low-frequency cutoff wavelength
    """

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int,
        theta: float = 10000.0,
        original_max_seq_len: int = 1024,
        scale_factor: float = 1.0,
        beta_fast: int = 32,
        beta_slow: int = 1,
    ):
        super().__init__()
        freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))

        if scale_factor > 1.0:
            orig_len = float(original_max_seq_len)
            beta = head_dim / (2 * math.pi * freqs * orig_len)
            mask_high = beta > beta_fast
            mask_low = beta < beta_slow
            freqs_ntk = freqs
            freqs_linear = freqs / scale_factor
            blend = (beta - beta_slow) / (beta_fast - beta_slow + 1e-8)
            freqs_mid = freqs_linear * blend + freqs_ntk * (1.0 - blend)
            freqs = torch.where(
                mask_high, freqs_linear,
                torch.where(mask_low, freqs_ntk, freqs_mid),
            )
            mscale = 0.1 * math.log(scale_factor) + 1.0
        else:
            mscale = 1.0

        self.mscale: torch.Tensor
        self.cos_cached: torch.Tensor
        self.sin_cached: torch.Tensor
        self.register_buffer("mscale", torch.tensor(mscale, dtype=torch.float32))

        emb = torch.outer(torch.arange(max_seq_len, dtype=torch.float32), freqs)
        cache_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        self.register_buffer(
            "cos_cached",
            torch.cat([torch.cos(emb), torch.cos(emb)], dim=-1).to(cache_dtype),
        )
        self.register_buffer(
            "sin_cached",
            torch.cat([torch.sin(emb), torch.sin(emb)], dim=-1).to(cache_dtype),
        )

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        h = x.shape[-1] // 2
        return torch.cat([-x[..., h:], x[..., :h]], dim=-1)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        offset: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        T = q.shape[2]
        mscale = self.mscale.to(q.dtype)
        cos = self.cos_cached[offset:offset + T].to(q.dtype).unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[offset:offset + T].to(q.dtype).unsqueeze(0).unsqueeze(0)
        q_rot = (q * cos + self._rotate_half(q) * sin) * mscale
        k_rot = (k * cos + self._rotate_half(k) * sin) * mscale
        return q_rot, k_rot


class BlockCausalAttention(nn.Module):
    r"""Block-causal attention with MLA, sparse windowing, and optional FlashAttention.

    .. rubric:: Multi-head Latent Attention (MLA)

    Keys and values are compressed into a shared latent of dimension
    ``mla_latent_dim``, then independently projected back to the per-head KV
    dimensions. Reduces KV cache size compared to grouped-query attention at
    the cost of extra projections.

    .. rubric:: Block-causal + sparse mask

    Positions in the same block attend to each other fully; across blocks the
    mask is causal. A fixed sliding window and periodic global tokens (every
    ``dsa_global_every`` positions) further sparsify long-range attention.

    .. rubric:: FlashAttention fallback

    When FlashAttention is available and no KV cache is being used, attention
    falls through to :func:`flash_attn_func`. Otherwise the SDPA kernel with
    an explicit additive bias is used.

    Args:
        config (:class:`ModelConfig`): model configuration. Relevant fields:
            ``d_model``, ``n_heads``, ``n_kv_heads``, ``mla_latent_dim``,
            ``block_size``, ``dsa_window_size``, ``dsa_global_every``,
            ``dropout``, ``use_flash_attention``, ``max_seq_len``,
            ``rope_theta``, ``rope_original_max_seq_len``, ``rope_scale_factor``.

    Shape:
        - Input ``x``: :math:`(B, T, d\_model)`
        - Input ``past_kv``: :math:`(B, T_\mathrm{past}, \mathrm{mla\_latent\_dim})`
          or ``None``
        - Output ``out``: :math:`(B, T, d\_model)`
        - Output ``present_kv``: :math:`(B, T_\mathrm{past}+T, \mathrm{mla\_latent\_dim})`
          if ``use_cache`` else ``None``
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        assert config.n_heads % config.n_kv_heads == 0

        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_kv_groups = config.n_heads // config.n_kv_heads
        self.head_dim = config.d_model // config.n_heads
        self.block_size = config.block_size
        self.dropout = config.dropout
        self.window_size = config.dsa_window_size
        self.global_every = config.dsa_global_every

        self.latent_dim = config.mla_latent_dim
        self.q_proj = nn.Linear(config.d_model, config.n_heads * self.head_dim, bias=False)
        self.kv_down = nn.Linear(config.d_model, self.latent_dim, bias=False)
        self.k_up = nn.Linear(self.latent_dim, config.n_kv_heads * self.head_dim, bias=False)
        self.v_up = nn.Linear(self.latent_dim, config.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_heads * self.head_dim, config.d_model, bias=False)

        self.rope = RotaryEmbedding(
            head_dim=self.head_dim,
            max_seq_len=config.max_seq_len,
            theta=config.rope_theta,
            original_max_seq_len=config.rope_original_max_seq_len,
            scale_factor=config.rope_scale_factor,
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

        idx = torch.arange(seq_len, device=device)
        block_idx = idx // self.block_size
        future_block = block_idx.unsqueeze(0) > block_idx.unsqueeze(1)
        outside_window = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs() > self.window_size
        global_visible = (idx % self.global_every == 0).unsqueeze(0).expand(seq_len, seq_len)
        mask = (outside_window & ~global_visible) | future_block
        self._sparse_mask_cache[key] = mask
        return mask

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        kv_offset: int = 0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        c_kv = self.kv_down(x)
        k = self.k_up(c_kv).view(B, T, -1, self.head_dim).transpose(1, 2)
        v = self.v_up(c_kv).view(B, T, -1, self.head_dim).transpose(1, 2)

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
            k_exp = k.repeat_interleave(self.n_kv_groups, dim=1)
            v_exp = v.repeat_interleave(self.n_kv_groups, dim=1)
            mask = self._build_sparse_mask(full_len, x.device)
            if T < full_len:
                mask = mask[-T:, :]
            attn_bias = torch.zeros(T, full_len, device=x.device, dtype=x.dtype)
            attn_bias.masked_fill_(mask, float("-inf"))
            attn_bias = attn_bias.unsqueeze(0).unsqueeze(0)

            y = F.scaled_dot_product_attention(
                q, k_exp, v_exp, attn_mask=attn_bias,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,
            )

        out = self.resid_drop(
            self.o_proj(y.transpose(1, 2).contiguous().view(B, T, C))
        )
        return out, (c_kv_full if use_cache else None)