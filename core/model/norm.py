"""Normalization layers used throughout Harold."""

import torch
import torch.nn as nn


__all__ = ["AdaLN"]


class AdaLN(nn.Module):
    r"""Adaptive LayerNorm conditioned on a timestep embedding.

    Computes :math:`\mathrm{LayerNorm}(x) \cdot (1 + \gamma(t)) + \beta(t)` where
    :math:`\gamma` and :math:`\beta` are produced by a single linear projection
    of the timestep embedding.

    The forward is fused: a manual split replaces ``chunk()`` and the
    subsequent ``mul_`` / ``add_`` are in-place, allowing :func:`torch.compile`
    to fuse them into a single kernel. Compared to an unfused version, this
    saves one ``chunk`` kernel and two ``unsqueeze`` ops per call.

    Args:
        d_model: embedding dimension

    Shape:
        - Input ``x``: :math:`(B, L, d\_model)`
        - Input ``t_emb``: :math:`(B, d\_model)`
        - Output: :math:`(B, L, d\_model)`
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.proj = nn.Linear(d_model, 2 * d_model, bias=True)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        proj = self.proj(t_emb)                      # (B, 2*d)
        d = proj.shape[-1] // 2
        scale = proj[..., :d].unsqueeze(1)           # (B, 1, d)
        shift = proj[..., d:].unsqueeze(1)           # (B, 1, d)
        return self.norm(x).mul_(1.0 + scale).add_(shift)