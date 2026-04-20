"""State-space model mixer (Mamba3 with Mamba2 fallback)."""

import warnings

import torch
import torch.nn as nn

from core.config import ModelConfig


__all__ = ["Mamba3Block"]


class Mamba3Block(nn.Module):
    r"""Wrapper around :class:`mamba_ssm.Mamba3` with residual dropout.

    Mamba3 (Lahoti et al., ICLR 2026) improves over Mamba2 in three ways:

    1. **Exponential-trapezoidal discretization** — more expressive than
       exponential-Euler, with a better approximation of the underlying ODE.
    2. **Complex-valued state update** — enables rich state tracking, solving
       tasks like the parity problem that Mamba2 cannot.
    3. **MIMO (multi-input, multi-output)** — increases expressivity without
       enlarging the state or adding decode latency.

    .. rubric:: Harold configuration

    ``headdim`` is aligned with ``d_model // n_heads`` for consistency with
    :class:`BlockCausalAttention`. When MIMO is enabled (``is_mimo=True``),
    ``chunk_size=16`` is optimal for bfloat16 (``64 // mimo_rank``).
    ``is_outproj_norm=False`` because the parent :class:`JambaBlock` applies
    :class:`AdaLN` both before and after the mixer.

    The input is already normalized by :class:`AdaLN` in the parent
    :class:`JambaBlock`, so this wrapper only adds the Mamba core plus a
    residual dropout.

    .. rubric:: Fallback

    Mamba3 is not yet available on PyPI at the time of writing (April 2026)
    and requires CUDA kernels compiled for the target architecture. If
    unavailable, the block falls back to Mamba2, which is equivalent for
    convergence and scaling-law validation purposes.

    Args:
        config (:class:`ModelConfig`): model configuration. Relevant fields:
            ``d_model``, ``mamba_d_state``, ``n_heads``, ``dropout``.

    Shape:
        - Input: :math:`(B, L, d\_model)` — already normalized by AdaLN
        - Output: :math:`(B, L, d\_model)`

    .. note::
        Requires ``mamba-ssm >= 2.x`` with Mamba3 support::

            pip install mamba-ssm causal-conv1d

    .. _Lahoti et al. ICLR 2026: https://arxiv.org/abs/2603.15569
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        try:
            from mamba_ssm import Mamba3  # type: ignore
            self.mamba = Mamba3(
                d_model=config.d_model,
                d_state=config.mamba_d_state,
                headdim=config.d_model // config.n_heads,
                is_mimo=False,
                is_outproj_norm=False,
                dtype=torch.bfloat16,
            )
            self._using_mamba3 = True
        except (ImportError, Exception):
            from mamba_ssm import Mamba2  # type: ignore
            warnings.warn(
                "Mamba3 not available — falling back to Mamba2. "
                "Install mamba3-release from GitHub for the full run.",
                stacklevel=2,
            )
            self.mamba = Mamba2(
                d_model=config.d_model,
                d_state=config.mamba_d_state,
                headdim=config.d_model // config.n_heads,
                expand=1,
            )
            self._using_mamba3 = False
        self.resid_drop = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Forward pass.

        Args:
            x: input already normalized by :class:`AdaLN`, shape :math:`(B, L, d\_model)`

        Returns:
            SSM output, shape :math:`(B, L, d\_model)`.

        Examples::

            >>> cfg = ModelConfig()
            >>> block = Mamba3Block(cfg).cuda().to(torch.bfloat16)
            >>> x = torch.randn(2, 512, cfg.d_model, device="cuda", dtype=torch.bfloat16)
            >>> block(x).shape
            torch.Size([2, 512, 1280])
        """
        return self.resid_drop(self.mamba(x))