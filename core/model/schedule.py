"""Flow Matching noise schedule for diffusion training.

The schedule is architecturally independent from the rest of the model — it
defines the forward/reverse diffusion trajectory and could be reused with a
different backbone.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


__all__ = ["FlowMatchingSchedule"]


class FlowMatchingSchedule(nn.Module):
    r"""Linear Conditional Flow Matching schedule (Lipman et al., 2022).

    Defines a linear trajectory between data :math:`x_0` and noise
    :math:`\varepsilon \sim \mathcal{N}(0, I)`:

    .. math::
        x_t = (1 - t_\mathrm{eff}) \, x_0 + t_\mathrm{eff} \, \varepsilon, \quad
        t_\mathrm{eff} = (1 - \sigma_\mathrm{min}) \, t + \sigma_\mathrm{min}

    .. rubric:: x0-prediction target

    The model predicts :math:`x_0` directly (not the flow velocity). The
    regression target is therefore :math:`x_0^* = x_0`. The velocity can be
    recovered by the sampler when needed:

    .. math::
        \hat{\varepsilon} = \frac{x_t - (1 - t) \hat{x}_0}{t}, \quad
        \hat{v} = \hat{\varepsilon} - \hat{x}_0

    .. rubric:: Timestep sampling strategies

    - ``"uniform"``: :math:`t \sim U[0, 1]` — flat baseline
    - ``"logit_normal"``: :math:`t = \sigma(z),\; z \sim \mathcal{N}(0, s^2)` —
      concentrates samples around :math:`t=0.5` where the gradient is most
      informative. Used in SD3 and recent FM models. Prevents velocity collapse
      by prioritizing difficult timesteps.
    - ``"cosine"``: density proportional to :math:`\sin(\pi t)` — similar to
      logit_normal but with heavier tails near :math:`t=0` and :math:`t=1`.

    Args:
        sigma_min: residual noise at :math:`t=0` for numerical stability
        t_sampling: one of ``"logit_normal"`` | ``"cosine"`` | ``"uniform"``
        t_logit_normal_std: standard deviation of the Gaussian for
            ``"logit_normal"`` sampling

    .. _Lipman et al. 2022: https://arxiv.org/abs/2210.02747
    """

    def __init__(
        self,
        sigma_min: float = 1e-4,
        t_sampling: str = "logit_normal",
        t_logit_normal_std: float = 0.5,
    ):
        super().__init__()
        self.sigma_min = sigma_min
        self.t_sampling = t_sampling
        self.t_logit_normal_std = t_logit_normal_std
        self._t_buffer: Optional[torch.Tensor] = None
        self._t_buf_idx: int = 0
        self._t_buf_size: int = 0

    def warmup_buffer(self, size: int = 4096, device: str = "cuda") -> None:
        """Pre-sample a buffer of timesteps to avoid per-step allocation in the training loop."""
        dev = torch.device(device)
        if self.t_sampling == "logit_normal":
            u = torch.randn(size, device=dev) * self.t_logit_normal_std
            self._t_buffer = torch.sigmoid(u)
        elif self.t_sampling == "cosine":
            u = torch.rand(size, device=dev)
            self._t_buffer = 0.5 * (1.0 - torch.cos(math.pi * u))
        else:
            self._t_buffer = torch.rand(size, device=dev)
        self._t_buf_idx = 0
        self._t_buf_size = size

    def sample_t(self, B: int, device: torch.device) -> torch.Tensor:
        r"""Sample training timesteps according to the configured strategy.

        Uses the pre-filled buffer when available (zero allocation in the hot
        path); falls back to on-the-fly sampling otherwise.

        Args:
            B: batch size
            device: output device

        Returns:
            Tensor of timesteps in :math:`[0, 1]`, shape :math:`(B,)`.
        """
        if self._t_buffer is not None and self._t_buffer.device == device:
            idx = self._t_buf_idx
            if idx + B <= self._t_buf_size:
                self._t_buf_idx += B
                return self._t_buffer[idx:idx + B]
            self.warmup_buffer(self._t_buf_size, str(device))
            self._t_buf_idx = B
            return self._t_buffer[:B]

        # Fallback: on-the-fly sampling
        if self.t_sampling == "logit_normal":
            u = torch.randn(B, device=device) * self.t_logit_normal_std
            return torch.sigmoid(u)
        elif self.t_sampling == "cosine":
            u = torch.rand(B, device=device)
            return 0.5 * (1.0 - torch.cos(math.pi * u))
        else:
            return torch.rand(B, device=device)

    def interpolate(
        self,
        x0: torch.Tensor,
        noise: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        r"""Compute :math:`x_t` along the linear trajectory.

        Args:
            x0: original token embeddings, shape :math:`(B, L, d\_model)`
            noise: Gaussian noise, shape :math:`(B, L, d\_model)`
            t: timesteps, shape :math:`(B,)`

        Returns:
            Interpolated embedding :math:`x_t`, shape :math:`(B, L, d\_model)`.
        """
        t_eff = (1.0 - self.sigma_min) * t + self.sigma_min
        t_eff = t_eff.view(-1, 1, 1)
        return (1.0 - t_eff) * x0 + t_eff * noise

    def target_x0(self, x0: torch.Tensor) -> torch.Tensor:
        r"""Return the x0-prediction regression target.

        The target is :math:`x_0` itself. Kept as an explicit method for
        semantic clarity and sampler API compatibility — the sampler uses this
        signature to recover :math:`\hat{v}` from :math:`\hat{x}_0`.

        Args:
            x0: original token embeddings, shape :math:`(B, L, d\_model)`

        Returns:
            Target x0, same shape as input.
        """
        return x0

    def add_noise(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Sample noise and compute :math:`x_t` along the trajectory.

        Args:
            x0: original token embeddings, shape :math:`(B, L, d\_model)`
            t: timesteps, shape :math:`(B,)`

        Returns:
            A tuple ``(x_t, noise)`` where both tensors have shape
            :math:`(B, L, d\_model)`.
        """
        noise = torch.empty_like(x0).normal_()
        x_t = self.interpolate(x0, noise, t)
        return x_t, noise