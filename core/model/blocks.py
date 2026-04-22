"""Jamba hybrid block — the architectural orchestrator.

A :class:`JambaBlock` combines one mixer (either :class:`BlockCausalAttention`
or :class:`Mamba3Block`) with a Mixture-of-Experts FFN
(:class:`DeepSeekMoELayer` or :class:`HashMoELayer`). Both sub-layers are
wrapped in :class:`AdaLN` for timestep conditioning.
"""

from typing import Optional, Tuple, cast

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as ckpt_fn

from core.config import ModelConfig

from .attention import BlockCausalAttention
from .moe import DeepSeekMoELayer, HashMoELayer
from .norm import AdaLN
from .ssm import Mamba3Block


__all__ = ["JambaBlock"]


class JambaBlock(nn.Module):
    r"""Jamba hybrid block: mixer (Mamba or Attention) followed by MoE.

    Each block applies one of two mixers based on ``is_attn_layer``:

    - ``False`` (3 out of every 4 blocks in the default Harold configuration):
      :class:`Mamba3Block` — linear-complexity SSM, no KV cache.
    - ``True`` (every ``jamba_attn_every``-th layer): :class:`BlockCausalAttention`
      with MLA and sparse windowing — supports KV cache.

    In both cases the mixer is preceded by :class:`AdaLN` conditioned on
    ``t_emb``, and its output passes through a Mixture-of-Experts layer
    (:class:`DeepSeekMoELayer` by default, or :class:`HashMoELayer` when
    ``config.use_hash_moe=True``) whose residual is norm-clamped.

    .. rubric:: Gradient checkpointing

    When ``config.use_gradient_checkpointing=True``, both the mixer and the
    MoE are checkpointed via instance methods ``_attn_forward`` /
    ``_moe_forward`` rather than closures. With 36 layers this saves dozens of
    closure object allocations per step.

    Args:
        config (:class:`ModelConfig`): model configuration
        is_attn_layer: if ``True``, use :class:`BlockCausalAttention`;
            otherwise use :class:`Mamba3Block`

    Shape:
        - Input ``x``: :math:`(B, L, d\_model)`
        - Output: :math:`(B, L, d\_model)`
    """

    def __init__(self, config: ModelConfig, is_attn_layer: bool):
        super().__init__()
        self.is_attn_layer = is_attn_layer
        self.use_gc = getattr(config, "use_gradient_checkpointing", False)

        self.ada_ln_1 = AdaLN(config.d_model)
        self.ada_ln_2 = AdaLN(config.d_model)

        if is_attn_layer:
            self.mixer: nn.Module = BlockCausalAttention(config)
        else:
            self.mixer = Mamba3Block(config)

        if getattr(config, "use_hash_moe", False):
            self.moe: nn.Module = HashMoELayer(config)
        else:
            self.moe = DeepSeekMoELayer(config)

        # Transient state used by _attn_forward / _moe_forward when
        # gradient checkpointing is active. Avoids closure allocation
        # per forward call.
        self._fwd_t_normalized: Optional[float] = None
        self._fwd_use_cache: bool = False
        self._fwd_kv_offset: int = 0

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        past_kv: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        kv_offset: int = 0,
        t_normalized: Optional[float] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        r"""Forward pass through the Jamba block.

        Args:
            x: input, shape :math:`(B, L, d\_model)`
            t_emb: timestep embedding, shape :math:`(B, d\_model)`
            past_kv: KV cache from a previous forward (attention layers only).
            use_cache: if ``True``, return the current KV state.
            kv_offset: position offset for RoPE (incremental generation).
            t_normalized: timestep normalized to :math:`[0, 1]` used by the
                adaptive router threshold (inference only).

        Returns:
            A tuple ``(x, present_kv)`` where ``present_kv`` is the updated KV
            state for attention layers with ``use_cache=True``, and ``None``
            otherwise.
        """
        normed = self.ada_ln_1(x, t_emb)

        _gc = self.use_gc and self.training and not use_cache

        # Persist transient state for the checkpointed helpers.
        self._fwd_t_normalized = t_normalized
        self._fwd_use_cache = use_cache
        self._fwd_kv_offset = kv_offset

        if self.is_attn_layer:
            if _gc and past_kv is None:
                mixer_out, present_kv = cast(
                    Tuple[torch.Tensor, Optional[torch.Tensor]],
                    ckpt_fn(self._attn_forward, normed, past_kv, use_reentrant=False),
                )
            else:
                mixer_out, present_kv = cast(
                    Tuple[torch.Tensor, Optional[torch.Tensor]],
                    self.mixer(
                        normed,
                        past_kv=past_kv,
                        use_cache=use_cache,
                        kv_offset=kv_offset,
                    ),
                )
        else:
            # [v0.7-GC-fix] Gradient checkpointing disabilitato su Mamba3Block.
            # Mamba3 ha gia il suo checkpoint interno nei kernel Triton —
            # sovrapporre ckpt_fn causa CheckpointError nel backward
            # (unpack triggered twice su ctx.saved_tensors).
            mixer_out = cast(torch.Tensor, self.mixer(normed))
            present_kv = None

        x = x + mixer_out

        if _gc:
            moe_out = cast(
                torch.Tensor,
                ckpt_fn(self._moe_forward, x, t_emb, use_reentrant=False),
            )
        else:
            moe_out = self.moe(
                self.ada_ln_2(x, t_emb),
                t_emb,
                t_normalized=t_normalized,
            )

        norm = moe_out.norm(dim=-1, keepdim=True).clamp(min=1.0)
        moe_out = moe_out / norm
        return x + moe_out, present_kv

    def _attn_forward(
        self,
        normed: torch.Tensor,
        past_kv: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Instance method used by gradient checkpointing (avoids inline closures)."""
        return cast(
            Tuple[torch.Tensor, Optional[torch.Tensor]],
            self.mixer(
                normed,
                past_kv=past_kv,
                use_cache=self._fwd_use_cache,
                kv_offset=self._fwd_kv_offset,
            ),
        )

    def _moe_forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Instance method used by gradient checkpointing (avoids inline closures)."""
        return self.moe(
            self.ada_ln_2(x, t_emb),
            t_emb,
            t_normalized=self._fwd_t_normalized,
        )