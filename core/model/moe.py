"""Mixture-of-Experts FFN layers.

Two MoE variants share the same API:
    - :class:`DeepSeekMoELayer`: learned router, sigmoid affinity, adaptive
      top-k at inference time, shared+routed expert split.
    - :class:`HashMoELayer`: deterministic hash-based routing (THOR-style),
      no learnable router, perfectly balanced load by construction.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.config import ModelConfig

from .quantization import maybe_fp8


__all__ = ["Expert", "SharedExpert", "DeepSeekMoELayer", "HashMoELayer"]


class Expert(nn.Module):
    """GELU FFN used as a routed expert in MoE layers."""

    def __init__(
        self,
        n_embd: int,
        hidden_dim: int,
        dropout: float = 0.0,
        use_fp8: bool = False,
    ):
        super().__init__()
        self.net = nn.Sequential(
            maybe_fp8(n_embd, hidden_dim, use_fp8),
            nn.GELU(),
            maybe_fp8(hidden_dim, n_embd, use_fp8),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SharedExpert(nn.Module):
    """SwiGLU FFN used as a shared (always-active) expert in MoE layers."""

    def __init__(
        self,
        n_embd: int,
        hidden_dim: int,
        dropout: float = 0.0,
        use_fp8: bool = False,
    ):
        super().__init__()
        self.w1 = maybe_fp8(n_embd, hidden_dim, use_fp8)
        self.w3 = maybe_fp8(n_embd, hidden_dim, use_fp8)
        self.w2 = maybe_fp8(hidden_dim, n_embd, use_fp8)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.dropout(F.silu(self.w1(x)) * self.w3(x)))


class DeepSeekMoELayer(nn.Module):
    r"""DeepSeek-style Mixture-of-Experts FFN layer.

    Combines ``ds_moe_n_shared_experts`` shared experts (always active, SwiGLU)
    with ``moe_n_routed_experts`` routed experts (top-k per token, sigmoid
    routing conditioned on :math:`[x; t\_emb]`).

    At inference time, an adaptive threshold replaces the fixed top-k: experts
    whose affinity exceeds ``threshold(t)`` are activated, with the threshold
    interpolated between ``threshold_base`` and ``threshold_min`` as a function
    of :math:`t`.

    .. rubric:: Router FiLM

    Instead of concatenating ``[x, t_emb]`` before the router projection, the
    conditioning is additive:

    .. math::
        \mathrm{logit} = W_x x + W_t \, t_\mathrm{emb}

    with ``W_t @ t_emb`` computed once per batch and broadcast across the
    sequence. This avoids an expensive ``repeat_interleave`` on the large
    ``d_model`` axis and matches the total parameter count of the concatenated
    variant.

    .. rubric:: Expert dispatch

    Tokens are sorted by assigned expert id (stable sort), then
    :func:`torch.searchsorted` finds the slice boundaries in a single call.
    For each of ``n_routed_experts`` experts we run one forward pass on its
    assigned tokens — no CPU-GPU sync other than the two ``.item()`` calls that
    read the slice boundaries.

    Args:
        config (:class:`ModelConfig`): model configuration. Relevant fields:
            ``moe_n_routed_experts``, ``moe_top_k``, ``ds_moe_n_shared_experts``,
            ``d_model``, ``moe_shared_hidden``, ``moe_routed_hidden``, ``dropout``.

    Shape:
        - Input ``x``: :math:`(B, L, d\_model)`
        - Input ``t_emb``: :math:`(B, d\_model)`
        - Output: :math:`(B, L, d\_model)`
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_routed_experts = config.moe_n_routed_experts
        self.top_k = config.moe_top_k
        self.bias_update_gamma = 1e-3
        self.threshold_base = 0.3
        self.threshold_min = 0.15
        self.top_k_min = 1

        _fp8 = getattr(config, "use_fp8", False)
        self.shared_experts = nn.ModuleList([
            SharedExpert(
                config.d_model, config.moe_shared_hidden,
                dropout=config.dropout, use_fp8=_fp8,
            )
            for _ in range(config.ds_moe_n_shared_experts)
        ])
        self.routed_experts = nn.ModuleList([
            Expert(
                config.d_model, config.moe_routed_hidden,
                dropout=config.dropout, use_fp8=_fp8,
            )
            for _ in range(self.n_routed_experts)
        ])

        # FiLM router: separate projections for x and t_emb.
        # Same total params as concat-then-project (2*d*E = d*E + d*E).
        self.router = nn.Linear(config.d_model, self.n_routed_experts, bias=False)
        self.router_t = nn.Linear(config.d_model, self.n_routed_experts, bias=False)
        nn.init.normal_(self.router.weight, std=0.01)
        nn.init.normal_(self.router_t.weight, std=0.01)

        self.register_buffer("router_bias", torch.zeros(self.n_routed_experts))
        self.router_indices: Optional[torch.Tensor] = None

    def _affinity(
        self,
        x_flat: torch.Tensor,
        t_bias_flat: torch.Tensor,
    ) -> torch.Tensor:
        # router(x_flat):  (B*T, E)
        # t_bias_flat:     (B*T, E) — broadcast from (B, E)
        return torch.sigmoid((self.router(x_flat) + t_bias_flat).float())

    def _compute_threshold(self, t_normalized: float) -> float:
        return (
            self.threshold_base
            - (self.threshold_base - self.threshold_min) * t_normalized
        )

    def _get_expert_outputs_vectorized(
        self,
        x_flat: torch.Tensor,
        topk_indices: torch.Tensor,
        gates: torch.Tensor,
    ) -> torch.Tensor:
        N, k = topk_indices.shape
        d = x_flat.shape[-1]
        output = torch.zeros(N, d, dtype=x_flat.dtype, device=x_flat.device)

        flat_indices = topk_indices.view(-1)
        flat_gates = gates.view(-1)
        token_ids = (
            torch.arange(N, device=x_flat.device)
            .unsqueeze(1)
            .expand(N, k)
            .reshape(-1)
        )

        valid_mask = flat_indices >= 0
        if not valid_mask.any():
            return output

        flat_indices_v = flat_indices[valid_mask]
        flat_gates_v = flat_gates[valid_mask]
        token_ids_v = token_ids[valid_mask]
        x_selected = x_flat[token_ids_v]

        # Sort once, find expert boundaries with a single searchsorted call.
        # With M = B*T*k and E experts this is O(log M) vs O(M*E) for a naive
        # per-expert equality mask.
        sort_idx = flat_indices_v.argsort(stable=True)
        sorted_exp = flat_indices_v[sort_idx]
        sorted_tok = token_ids_v[sort_idx]
        sorted_x = x_selected[sort_idx]
        sorted_g = flat_gates_v[sort_idx]

        exp_ids_range = torch.arange(
            self.n_routed_experts + 1,
            device=x_flat.device,
            dtype=sorted_exp.dtype,
        )
        boundaries = torch.searchsorted(
            sorted_exp.contiguous(),
            exp_ids_range.contiguous(),
        )  # (E+1,)

        for exp_id in range(self.n_routed_experts):
            s = boundaries[exp_id].item()
            e = boundaries[exp_id + 1].item()
            if s == e:
                continue
            tok_ids = sorted_tok[s:e]
            x_in = sorted_x[s:e]
            g = sorted_g[s:e].unsqueeze(1).to(x_flat.dtype)
            exp_out = self.routed_experts[exp_id](x_in)
            output.index_add_(0, tok_ids, (exp_out * g).to(output.dtype))

        return output

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        t_normalized: Optional[float] = None,
    ) -> torch.Tensor:
        B, T, C = x.shape
        N = B * T
        x_flat = x.view(N, C)

        # t_bias: (B, E) → (B*T, E) via expand+reshape (zero-copy).
        t_bias = self.router_t(t_emb)                           # (B, E)
        t_bias_flat = t_bias.unsqueeze(1).expand(B, T, -1).reshape(N, -1)

        # Shared experts: direct sum instead of stack+mean.
        if len(self.shared_experts) == 1:
            shared_out = self.shared_experts[0](x_flat)
        else:
            shared_out = self.shared_experts[0](x_flat)
            for e in self.shared_experts[1:]:
                shared_out = shared_out + e(x_flat)
            shared_out = shared_out * (1.0 / len(self.shared_experts))

        s = self._affinity(x_flat, t_bias_flat)
        sel_scores = s + self.router_bias

        if self.training or t_normalized is None:
            topk_vals = torch.topk(sel_scores, self.top_k, dim=-1)
            topk_indices = topk_vals.indices
        else:
            threshold = self._compute_threshold(t_normalized)
            k_max = int(
                (sel_scores > threshold).sum(dim=-1)
                .clamp(self.top_k_min, self.n_routed_experts).max().item()
            )
            topk_vals = torch.topk(sel_scores, k_max, dim=-1)
            topk_indices = topk_vals.indices
            topk_scores = sel_scores.gather(1, topk_indices)
            topk_indices = topk_indices.masked_fill(topk_scores <= threshold, -1)

        self.router_indices = topk_indices.detach()

        valid_mask = topk_indices >= 0
        s_sel = s.gather(dim=1, index=topk_indices.clamp(min=0)) * valid_mask.to(s.dtype)
        denom = s_sel.sum(dim=1, keepdim=True)
        gates = torch.where(
            denom > 1e-9,
            s_sel / (denom + 1e-9),
            torch.full_like(s_sel, 1.0 / self.top_k),
        ).to(x.dtype)

        routed_out = self._get_expert_outputs_vectorized(x_flat, topk_indices, gates)
        return (
            (shared_out + routed_out)
            / (len(self.shared_experts) + self.top_k)
        ).view(B, T, C)

    @torch.no_grad()
    def update_bias(self) -> None:
        r"""Update ``router_bias`` based on recent expert usage.

        Applies a sign step proportional to the deviation between mean usage
        and per-expert usage, incentivising balanced load. Must be called once
        per optimizer step during training.

        No-op if ``router_indices`` is ``None`` (no forward since last call).
        """
        if self.router_indices is None:
            return
        valid = self.router_indices[self.router_indices >= 0]
        if valid.numel() == 0:
            return
        counts = torch.bincount(
            valid.view(-1),
            minlength=self.n_routed_experts,
        ).float()
        self.router_bias += self.bias_update_gamma * (counts.mean() - counts).sign()
        self.router_indices = None


class HashMoELayer(nn.Module):
    r"""THOR-style Mixture-of-Experts with deterministic hash routing.

    Each token is assigned to ``top_k`` experts via a deterministic hash of
    its global batch index:

    .. math::
        \mathrm{expert\_id} = (\mathrm{token\_idx} \cdot p + k \cdot 7919)
                              \bmod n_\mathrm{routed\_experts}

    where :math:`p` is the smallest prime :math:`\geq n_\mathrm{routed\_experts} + 1`.

    .. rubric:: Pros

    - No router parameters — eliminates ``router.weight`` and ``router_t.weight``.
    - No topk, no searchsorted decision path.
    - Perfectly balanced load by construction.

    .. rubric:: Cons

    - No semantic specialization — experts cannot learn differentiated roles.
    - API identical to :class:`DeepSeekMoELayer` for drop-in swapping.

    Args:
        config (:class:`ModelConfig`): model configuration. See
            :class:`DeepSeekMoELayer` for relevant fields.

    Shape:
        - Input ``x``: :math:`(B, L, d\_model)`
        - Input ``t_emb``: :math:`(B, d\_model)` — unused but kept for API parity
        - Output: :math:`(B, L, d\_model)`
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_routed_experts = config.moe_n_routed_experts
        self.top_k = config.moe_top_k
        self.router_indices: Optional[torch.Tensor] = None

        _fp8 = getattr(config, "use_fp8", False)
        self.shared_experts = nn.ModuleList([
            SharedExpert(
                config.d_model, config.moe_shared_hidden,
                dropout=config.dropout, use_fp8=_fp8,
            )
            for _ in range(config.ds_moe_n_shared_experts)
        ])
        self.routed_experts = nn.ModuleList([
            Expert(
                config.d_model, config.moe_routed_hidden,
                dropout=config.dropout, use_fp8=_fp8,
            )
            for _ in range(self.n_routed_experts)
        ])
        self._hash_prime = self._find_prime(self.n_routed_experts + 1)

    @staticmethod
    def _find_prime(n: int) -> int:
        def is_prime(x: int) -> bool:
            if x < 2:
                return False
            for i in range(2, int(x ** 0.5) + 1):
                if x % i == 0:
                    return False
            return True
        while not is_prime(n):
            n += 1
        return n

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        t_normalized: Optional[float] = None,
    ) -> torch.Tensor:
        B, T, C = x.shape
        N = B * T
        x_flat = x.view(N, C)

        if len(self.shared_experts) == 1:
            shared_out = self.shared_experts[0](x_flat)
        else:
            shared_out = self.shared_experts[0](x_flat)
            for e in self.shared_experts[1:]:
                shared_out = shared_out + e(x_flat)
            shared_out = shared_out * (1.0 / len(self.shared_experts))

        token_ids = torch.arange(N, device=x.device)
        output = torch.zeros(N, C, dtype=x.dtype, device=x.device)

        for k in range(self.top_k):
            exp_ids = (token_ids * self._hash_prime + k * 7919) % self.n_routed_experts
            self.router_indices = exp_ids.detach()

            # Sort by expert id to exploit locality in the per-expert forward.
            sort_idx = exp_ids.argsort(stable=True)
            sorted_exp = exp_ids[sort_idx]
            sorted_tok = token_ids[sort_idx]
            sorted_x = x_flat[sort_idx]
            boundaries = torch.searchsorted(
                sorted_exp.contiguous(),
                torch.arange(
                    self.n_routed_experts + 1,
                    device=x.device,
                    dtype=sorted_exp.dtype,
                ).contiguous(),
            )
            for exp_id in range(self.n_routed_experts):
                s = boundaries[exp_id].item()
                e = boundaries[exp_id + 1].item()
                if s == e:
                    continue
                exp_out = self.routed_experts[exp_id](sorted_x[s:e])
                output.index_add_(0, sorted_tok[s:e], exp_out.to(output.dtype))

        output = output / self.top_k
        return (
            (shared_out + output) / (len(self.shared_experts) + 1)
        ).view(B, T, C)

    @torch.no_grad()
    def update_bias(self) -> None:
        """No-op — Hash MoE has no router bias."""
        pass