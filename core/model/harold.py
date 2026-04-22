"""Top-level Harold model — hybrid Jamba diffusion language model.

The :class:`Harold` class ties together the token embeddings, timestep
embeddings, Jamba block stack, and output heads. The training loss
(:meth:`Harold.compute_loss`) implements Flow Matching with x0-prediction
and an auxiliary cross-entropy term.
"""

import math
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.config import ModelConfig

from .blocks import JambaBlock
from .quantization import FP8Linear
from .schedule import FlowMatchingSchedule


__all__ = ["Harold", "build_model"]


class Harold(nn.Module):
    r"""Hybrid Jamba diffusion language model with x0-prediction.

    .. rubric:: Architecture summary

    - 36 Jamba layers in a :math:`[\text{Mamba3} \times 3, \text{Attention}]
      \times 9` pattern.
    - DeepSeek-style MoE (1 shared + 8 routed top-2) on every layer.
    - Flow Matching with linear trajectory.
    - **x0-prediction**: the model predicts :math:`\hat{x}_0` directly in the
      embedding space, not the flow velocity. The velocity can be recovered
      analytically by the sampler.
    - :class:`Mamba3Block` as the SSM mixer, with MIMO enabled and
      automatic fallback to Mamba2 if Mamba3 is unavailable.

    Args:
        config (:class:`ModelConfig`): full model configuration

    .. note::
        v0.6 checkpoints are not directly compatible: the output head is
        renamed (``vel_pred`` → ``x0_pred``) and the SSM mixer uses different
        weight shapes (Mamba3 vs Mamba2). Use a migration script for the
        non-SSM weights (attention, MoE, etc.).
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        emb_vocab = config.vocab_size + 1
        self.emb_vocab = emb_vocab
        self.mask_token_id = config.vocab_size

        self.token_emb = nn.Embedding(emb_vocab, config.d_model)

        self.time_emb = nn.Sequential(
            nn.Linear(config.d_model, config.d_model * 4),
            nn.SiLU(),
            nn.Linear(config.d_model * 4, config.d_model),
        )

        # self_cond projects the averaged x0 prediction from the previous
        # sampling step back into t_emb space.
        self.self_cond_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        nn.init.normal_(self.self_cond_proj.weight, std=0.02)

        # Classifier-free-guidance context projection (zero-init so uncond
        # behaviour matches the unconditional schedule at init time).
        self.cfg_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        nn.init.zeros_(self.cfg_proj.weight)

        half = config.d_model // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, dtype=torch.float32) / half
        )
        self.t_freqs: torch.Tensor
        self.register_buffer("t_freqs", freqs)

        # Pattern: layer idx % jamba_attn_every == jamba_attn_every - 1 → attention.
        # With n_layers=36 and jamba_attn_every=4 this yields attention at
        # indices 3, 7, 11, 15, 19, 23, 27, 31, 35 — nine attention layers total.
        self.blocks = nn.ModuleList([
            JambaBlock(
                config,
                is_attn_layer=(
                    (i % config.jamba_attn_every) == (config.jamba_attn_every - 1)
                ),
            )
            for i in range(config.n_layers)
        ])

        self.norm_out = nn.LayerNorm(config.d_model)

        # x0 head: predicts x0_emb directly.
        self.x0_pred = nn.Linear(config.d_model, config.d_model, bias=False)

        # CE head tied to the token embedding; applied on top of x0_pred
        # so that both predictions are consistent.
        self.ce_head = nn.Linear(config.d_model, emb_vocab, bias=False)
        self.ce_head.weight = self.token_emb.weight

        self.schedule = FlowMatchingSchedule(
            sigma_min=config.flow_sigma_min,
            t_sampling=config.t_sampling,
            t_logit_normal_std=config.t_logit_normal_std,
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                std = (
                    0.02 / math.sqrt(2 * self.config.n_layers)
                    if any(name.endswith(s) for s in ("o_proj", "w2", "v_up", "x0_pred"))
                    else 0.02
                )
                nn.init.normal_(module.weight, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def get_timestep_embedding(self, t: torch.Tensor) -> torch.Tensor:
        r"""Sinusoidal embedding of the diffusion timestep.

        Uses ``stack + view`` instead of ``cat([cos, sin])`` to avoid one of
        two intermediate allocations. The view is a zero-copy reshape.

        Args:
            t: timesteps in :math:`[0, 1]`, shape :math:`(B,)`

        Returns:
            Embedding tensor, shape :math:`(B, d\_model)`.
        """
        if t.numel() == 0:
            # Batch vuoto: restituisci embedding vuoto
            return torch.empty(0, self.d_model, device=t.device, dtype=torch.float32)
        if t.dim() == 0:
            t = t.view(1)                     # scalare -> (1,)
        elif t.dim() > 1:
            t = t.view(-1)                    # appiattisci se ha più dimensioni

        # Ora t ha sicuramente forma (B,)
        args = (t.float() * 1000.0).unsqueeze(1) * self.t_freqs.unsqueeze(0)  # (B, half)
        emb = torch.stack([torch.cos(args), torch.sin(args)], dim=-1)         # (B, half, 2)
        emb = emb.view(t.shape[0], -1)                                        # (B, d_model)
        if self.d_model % 2:
            emb = F.pad(emb, (0, 1))
        return emb

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        self_cond: Optional[torch.Tensor] = None,
        ctx_emb: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[List]]:
        r"""Forward pass through the Jamba backbone.

        Conditions on ``t`` via sinusoidal embedding + MLP, then processes
        the sequence through :attr:`blocks` (interleaved Mamba3 and
        :class:`BlockCausalAttention`, each followed by a MoE layer).

        The output head predicts :math:`\hat{x}_0` directly. The CE logits
        are computed from ``ce_head(x0_pred)`` rather than an intermediate
        hidden state, keeping the two predictions consistent.

        Args:
            x_t: noisy embeddings, shape :math:`(B, L, d\_model)`
            t: timesteps in :math:`[0, 1]`, shape :math:`(B,)`
            self_cond: averaged :math:`\hat{x}_0` from the previous sampling
                step, shape :math:`(B, d\_model)`, or ``None``.
            ctx_emb: context embedding for classifier-free guidance,
                shape :math:`(B, d\_model)`, or ``None``.
            past_key_values: KV cache per attention layer, or ``None``.
            use_cache: if ``True``, return current KV states.

        Returns:
            A tuple ``(x0_pred, ce_logits, present_kvs)`` where:
                - ``x0_pred`` has shape :math:`(B, L, d\_model)`
                - ``ce_logits`` has shape :math:`(B, L, V+1)`
                - ``present_kvs`` is a list of KV states or ``None``
        """
        emb = self.get_timestep_embedding(t)
        assert emb.dim() == 2, f"emb shape {emb.shape} is not 2D"
        t_emb = self.time_emb(emb)
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

        x_out = self.norm_out(x)
        x0_pred = self.x0_pred(x_out)
        ce_logits = self.ce_head(x0_pred)

        return x0_pred, ce_logits, present_kvs

    def compute_loss(
        self,
        x0: torch.Tensor,
        mask: torch.Tensor,
        ce_weight: float = 0.1,
        fixed_t: Optional[torch.Tensor] = None,
        self_cond_prob: float = 0.0,
        ctx_emb: Optional[torch.Tensor] = None,
        p_uncond: float = 0.0,
    ) -> Tuple[torch.Tensor, Dict]:
        r"""Flow Matching loss with x0-prediction.

        Samples :math:`t \sim U[0,1]` per example, interpolates the embeddings
        along the linear trajectory :math:`x_t = (1-t) x_0 + t \varepsilon`,
        then computes:

        .. math::
            \mathcal{L} = \underbrace{\mathrm{MSE}(\hat{x}_0,\, x_0)}_{\text{score loss}}
                        + w_\mathrm{CE} \cdot
                          \mathrm{CE}(\hat{x}_0 W_\mathrm{emb}^\top,\, y)

        No Min-SNR weighting — the target is uniform across all :math:`t`.

        .. rubric:: Monitoring

        The returned metrics dict includes ``x0_norm_mean``, ``x0_norm_std``
        and ``x0_var_tokens`` for detecting the two common failure modes:

        - ``x0_norm_mean`` ≪ 1.0 → collapse toward zero.
        - ``x0_var_tokens`` ≈ 0 → mode collapse (all tokens predicted identical).

        Args:
            x0: original token ids, shape :math:`(B, L)`
            mask: positions that contribute to the loss, shape :math:`(B, L)`
            ce_weight: weight of the auxiliary cross-entropy term
            fixed_t: fixed timestep of shape :math:`(B,)`. If ``None``, sample
                :math:`t \sim U[0,1]`.
            self_cond_prob: probability of self-conditioning per step
            ctx_emb: context embedding for classifier-free guidance,
                shape :math:`(B, d\_model)`
            p_uncond: probability of dropping the context (CFG training)

        Returns:
            A tuple ``(loss, metrics)``. The metrics dict has keys ``"score"``,
            ``"ce"``, ``"total"``, ``"x0_norm_mean"``, ``"x0_norm_std"``,
            ``"x0_var_tokens"``; if ``fixed_t`` is provided it also includes
            ``"total_per_sample"``, ``"score_per_sample"``, ``"ce_per_sample"``.

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

        # Target is x0_emb itself — target_x0() would be an identity call.

        # Self-conditioning on previous x0 prediction (not velocity).
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

        x0_pred, ce_logits, _ = self.forward(
            x_t, t, self_cond=self_cond, ctx_emb=cfg_emb,
        )

        # MSE directly against x0_emb.
        per_token_mse = F.mse_loss(x0_pred, x0_emb, reduction="none").mean(dim=-1)
        loss_score = per_token_mse[mask].mean()

        x0_for_ce = x0.masked_fill(~mask, -100)
        loss_ce = F.cross_entropy(
            ce_logits.view(-1, ce_logits.size(-1)),
            x0_for_ce.view(-1),
            ignore_index=-100,
            reduction="mean",
        )
        total = loss_score + ce_weight * loss_ce

        extra: Dict = {}
        if fixed_t is not None:
            per_sample_score = per_token_mse.mean(dim=-1)
            per_sample_ce = F.cross_entropy(
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

        # x0_pred norm monitoring — only over valid (masked) tokens.
        with torch.no_grad():
            x0_masked = x0_pred[mask]                    # (N_valid, D)
            x0_norms = x0_masked.norm(dim=-1)             # (N_valid,)
            x0_norm_mean = x0_norms.mean().item()
            x0_norm_std = x0_norms.std().item() if x0_norms.numel() > 1 else 0.0
            x0_var_tokens = x0_pred.var(dim=1).mean().item()

        return total, {
            "score":         loss_score.item(),
            "ce":            loss_ce.item(),
            "total":         total.item(),
            "x0_norm_mean":  round(x0_norm_mean,  4),
            "x0_norm_std":   round(x0_norm_std,   4),
            "x0_var_tokens": round(x0_var_tokens, 4),
            **extra,
        }

    @torch.no_grad()
    def decode_tokens(self, x: torch.Tensor) -> torch.Tensor:
        r"""Decode continuous embeddings to token ids via cosine NN lookup.

        Args:
            x: embeddings, shape :math:`(B, L, d\_model)`

        Returns:
            Token ids, shape :math:`(B, L)`.
        """
        emb_norm = F.normalize(self.token_emb.weight, dim=-1)
        x_norm = F.normalize(x, dim=-1)
        return torch.einsum("bld,vd->blv", x_norm, emb_norm).argmax(dim=-1)

    @torch.no_grad()
    def update_router_biases(self) -> None:
        """Update all MoE router biases based on recent expert usage.

        Calls ``update_bias()`` on every block — no-op for :class:`HashMoELayer`.
        """
        for block in self.blocks:
            moe = block.moe
            update_fn = getattr(moe, "update_bias", None)
            if callable(update_fn):
                update_fn()

    @torch.no_grad()
    def update_fp8_weights(self) -> None:
        """Re-quantize every :class:`FP8Linear` after an optimizer step.

        bf16 master weights are updated by the optimizer but the ``weight_fp8``
        and ``scale_w`` buffers are stale until this call re-aligns them.
        No-op when FP8 is disabled (no :class:`FP8Linear` modules).
        """
        for m in self.modules():
            if isinstance(m, FP8Linear):
                m.update_weight_fp8()


def build_model(model_cfg: ModelConfig) -> Harold:
    """Factory for a :class:`Harold` model from a :class:`ModelConfig`."""
    return Harold(model_cfg)