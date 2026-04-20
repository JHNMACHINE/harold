"""FP8 quantization primitives.

Contains :class:`FP8Linear` — a drop-in replacement for :class:`torch.nn.Linear`
using :func:`torch._scaled_mm` for the forward pass and a straight-through
estimator for the backward pass.

Requirements:
    - PyTorch 2.1+
    - Hopper (SM90) or Blackwell (SM120) GPU for native FP8 support
"""

from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["FP8Linear", "maybe_fp8"]


class FP8Linear(nn.Module):
    r"""FP8 drop-in replacement for :class:`torch.nn.Linear` via :func:`torch._scaled_mm`.

    Uses ``float8_e4m3fn`` for weights with dynamic scaling for both inputs and
    weights. Does not support bias.

    .. rubric:: Scale semantics

    ``torch._scaled_mm`` expects *de-quantization* scales:

    .. math::
        \mathrm{output} = (x_\mathrm{fp8} \cdot s_a) @ (w_\mathrm{fp8} \cdot s_b)^\top

    Therefore ``x_fp8`` and ``w_fp8`` must be normalized into :math:`[-1, 1]`
    before FP8 conversion, and ``scale_x`` / ``scale_w`` are the de-quantization
    factors that undo this normalization.

    .. rubric:: Backward pass

    ``_scaled_mm`` has no backward implementation in PyTorch 2.10. A
    straight-through estimator is used: forward uses FP8, backward uses
    :func:`F.linear` on the bfloat16 weights. Gradients flow through the
    bfloat16 weights, not through the FP8 matmul. This is the standard pattern
    for FP8 training without TransformerEngine.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample

    Shape:
        - Input: :math:`(*, H_\mathrm{in})`
        - Output: :math:`(*, H_\mathrm{out})`
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Weights kept in bf16 — converted to FP8 on-the-fly in forward.
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=torch.bfloat16)
        )
        nn.init.normal_(self.weight, std=0.02)

        # Type annotations help Pylance: register_buffer returns None but
        # buffers are accessible as Tensor at runtime.
        self.scale_x: torch.Tensor
        self.scale_w: torch.Tensor
        self.weight_fp8: torch.Tensor
        self.fp8_initialized: torch.Tensor

        # scale_x: de-quantization scale for input (updated every forward).
        # scale_w: de-quantization scale for weights (updated after optimizer step).
        self.register_buffer("scale_x", torch.ones(1, dtype=torch.float32))
        self.register_buffer("scale_w", torch.ones(1, dtype=torch.float32))

        # Pre-quantized weights — updated by update_weight_fp8().
        self.register_buffer(
            "weight_fp8",
            torch.zeros(out_features, in_features, dtype=torch.float8_e4m3fn),
        )

        # Bool flag as buffer — survives checkpoint save/load so that after
        # resume, weight_fp8 is already populated and we avoid re-initialization.
        self.register_buffer("fp8_initialized", torch.zeros(1, dtype=torch.bool))

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"fp8={bool(self.fp8_initialized.item())}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.fp8_initialized.item():
            self.update_weight_fp8()

        with torch.no_grad():
            fp8_max = 448.0
            # .detach() prevents float() from breaking the autograd graph.
            x_amax = x.detach().float().abs().max().clamp(min=1e-12)
            in_scale = float(fp8_max / x_amax)
            self.scale_x.fill_(float(x_amax / fp8_max))
            x_fp8 = (x.detach() * in_scale).to(torch.float8_e4m3fn)
            fp8_out = torch._scaled_mm(
                x_fp8,
                self.weight_fp8.t(),
                scale_a=self.scale_x,
                scale_b=self.scale_w,
                out_dtype=torch.bfloat16,
            )

        # Straight-through estimator:
        #   forward value: fp8_out
        #   backward gradient: F.linear(x, self.weight) in bf16
        bf16_out = F.linear(x, self.weight)
        return fp8_out + (bf16_out - bf16_out.detach())

    @torch.no_grad()
    def update_weight_fp8(self) -> None:
        """Quantize bf16 weights to FP8 and update ``scale_w``.

        Must be called:
            1. Once after init (lazy, on first forward).
            2. After every optimizer step (bf16 weights have changed).
        """
        fp8_max = 448.0
        w_amax = self.weight.float().abs().max().clamp(min=1e-12)
        w_scale = float(fp8_max / w_amax)  # quantization scale
        self.scale_w.fill_(float(w_amax / fp8_max))  # de-quantization scale
        self.weight_fp8.copy_((self.weight * w_scale).to(torch.float8_e4m3fn))
        self.fp8_initialized.fill_(True)


def maybe_fp8(in_f: int, out_f: int, use_fp8: bool) -> nn.Module:
    """Return an :class:`FP8Linear` if ``use_fp8`` else a plain :class:`nn.Linear` (no bias)."""
    if use_fp8:
        return FP8Linear(in_f, out_f)
    return nn.Linear(in_f, out_f, bias=False)