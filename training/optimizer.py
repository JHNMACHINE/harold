"""
Harold v0.6 — optimizer.py
===========================
MuonAdamW: Muon per i pesi 2D (matrici), AdamW per tutto il resto.

Muon usa orthogonalizzazione polare (Polar Express) per mantenere i gradienti
delle matrici di peso sulla varietà delle matrici ortogonali, con NorMuon
variance reduction e cautious weight decay.

Basato su: Karpathy / Kostrikov MuonAdamW
Adattato per Harold v0.6 (Jamba + MoE + Flow Matching).

Utilizzo in train.py:
    from optimizer import build_optimizer
    optimizer = build_optimizer(model, train_cfg)

Se train_cfg.use_muon=False, ritorna AdamW standard (comportamento v0.5).
"""

import torch
import torch.nn as nn
from typing import List, TYPE_CHECKING

from utils.ddp import is_main

if TYPE_CHECKING:
    from core.config import TrainConfig
    from core.model import Harold


# ---------------------------------------------------------------------------
# Coefficienti Polar Express per orthogonalizzazione iterativa
# (precalcolati per ns_steps fino a 10)
# ---------------------------------------------------------------------------

polar_express_coeffs = [
    (8.156554524902461,  -22.48329292557795,  15.878769915207462),
    (4.042929935166739,   -2.808917465908714,   0.5000178451051316),
    (3.8916678022926607,  -2.772484153217685,   0.5060648178503393),
    (3.285753657755655,   -2.3681294933425376,  0.46449024233003106),
    (2.3465413258596377,  -1.7097828382687081,  0.42323551169305323),
]


# ---------------------------------------------------------------------------
# Kernel fusi compilati una volta sola al primo utilizzo
# ---------------------------------------------------------------------------

@torch.compile(dynamic=False, fullgraph=True)
def adamw_step_fused(p, grad, exp_avg, exp_avg_sq,
                     step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t):
    p.mul_(1 - lr_t * wd_t)
    exp_avg.lerp_(grad, 1 - beta1_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    bias1     = 1 - beta1_t ** step_t
    bias2     = 1 - beta2_t ** step_t
    denom     = (exp_avg_sq / bias2).sqrt() + eps_t
    step_size = lr_t / bias1
    p.add_(exp_avg / denom, alpha=-step_size)


def muon_step_fused(stacked_grads, stacked_params,
                    momentum_buffer, second_momentum_buffer,
                    momentum_t, lr_t, wd_t, beta2_t, ns_steps, red_dim):
    # Nesterov momentum
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)

    # Polar Express orthogonalizzazione
    X = g.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    if g.size(-2) > g.size(-1):
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    g = X

    # NorMuon variance reduction
    beta2       = beta2_t.to(g.dtype)
    v_mean      = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq   = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm      = v_norm_sq.sqrt()
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
    step_size   = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new  = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)

    # Cautious weight decay + parameter update
    lr   = lr_t.to(g.dtype)
    wd   = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)


# ---------------------------------------------------------------------------
# MuonAdamW
# ---------------------------------------------------------------------------

class MuonAdamW(torch.optim.Optimizer):
    """
    Optimizer combinato per Harold v0.6.

    - **Muon** (con Polar Express + NorMuon): per tutti i parametri 2D
      (matrici di proiezione attention, expert weights, router, Mamba2 proiezioni).
    - **AdamW**: per tutto il resto (embedding, LayerNorm, bias, vettori 1D).

    I param_groups devono avere il campo ``kind``:
        - ``'muon'``:  gruppo Muon — tutti i params devono avere la stessa shape
        - ``'adamw'``: gruppo AdamW — shape arbitraria
    """

    def __init__(self, param_groups):
        super().__init__(param_groups, defaults={})
        # Tensori 0-D su CPU per evitare ricompilazioni di torch.compile
        self._adamw_step_t  = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t    = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t   = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t    = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t       = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t       = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t    = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    def _step_adamw(self, group):
        for p in group["params"]:
            if p.grad is None:
                continue
            grad  = p.grad
            state = self.state[p]
            if not state:
                state["step"]        = 0
                state["exp_avg"]     = torch.zeros_like(p)
                state["exp_avg_sq"]  = torch.zeros_like(p)
            state["step"] += 1
            self._adamw_step_t.fill_(state["step"])
            self._adamw_lr_t.fill_(group["lr"])
            self._adamw_beta1_t.fill_(group["betas"][0])
            self._adamw_beta2_t.fill_(group["betas"][1])
            self._adamw_eps_t.fill_(group["eps"])
            self._adamw_wd_t.fill_(group["weight_decay"])
            adamw_step_fused(
                p, grad, state["exp_avg"], state["exp_avg_sq"],
                self._adamw_step_t, self._adamw_lr_t,
                self._adamw_beta1_t, self._adamw_beta2_t,
                self._adamw_eps_t, self._adamw_wd_t,
            )

    def _step_muon(self, group):
        params = group["params"]
        if not params:
            return

        # Raggruppa per shape — ogni shape vuole il suo buffer e step fused
        from collections import defaultdict
        by_shape: dict = defaultdict(list)
        for p in params:
            if p.grad is not None:
                by_shape[p.shape].append(p)

        for shape, ps in by_shape.items():
            p0    = ps[0]
            state = self.state[p0]
            n     = len(ps)

            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros(
                    n, *shape, dtype=p0.dtype, device=p0.device)
            if "second_momentum_buffer" not in state:
                sb_shape = (n, shape[-2], 1) if shape[-2] >= shape[-1] else (n, 1, shape[-1])
                state["second_momentum_buffer"] = torch.zeros(
                    sb_shape, dtype=p0.dtype, device=p0.device)

            red_dim       = -1 if shape[-2] >= shape[-1] else -2
            stacked_grads  = torch.stack([p.grad for p in ps])
            stacked_params = torch.stack(ps)

            # lr scalata per aspect ratio (Muon standard)
            lr_scaled = group["lr"] * max(1.0, shape[-2] / shape[-1]) ** 0.5

            self._muon_momentum_t.fill_(group["momentum"])
            self._muon_beta2_t.fill_(group["beta2"])
            self._muon_lr_t.fill_(lr_scaled)
            self._muon_wd_t.fill_(group["weight_decay"])

            muon_step_fused(
                stacked_grads, stacked_params,
                state["momentum_buffer"], state["second_momentum_buffer"],
                self._muon_momentum_t, self._muon_lr_t, self._muon_wd_t,
                self._muon_beta2_t, group["ns_steps"], red_dim,
            )
            torch._foreach_copy_(ps, list(stacked_params.unbind(0)))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            if group["kind"] == "adamw":
                self._step_adamw(group)
            elif group["kind"] == "muon":
                self._step_muon(group)


# ---------------------------------------------------------------------------
# build_optimizer — entry point per train.py
# ---------------------------------------------------------------------------

def _is_muon_eligible(name: str, param: torch.Tensor) -> bool:
    """
    Determina se un parametro va nel gruppo Muon o AdamW.

    Muon richiede matrici 2D con entrambe le dimensioni >= 2.
    Esclusi esplicitamente:
      - token_emb, pos_emb  (embedding — semantica diversa dalle proiezioni)
      - ce_head             (tied con token_emb)
      - t_freqs             (buffer sinusoidale, non trainabile)
      - LayerNorm / bias    (già 1D, verrebbero esclusi dal dim check)
    """
    if param.dim() != 2:
        return False
    if param.shape[0] < 2 or param.shape[1] < 2:
        return False
    # Escludi embedding e output head (tied weights)
    excluded = ("token_emb", "pos_emb", "ce_head")
    if any(e in name for e in excluded):
        return False
    return True


def build_optimizer(
    model: nn.Module,
    train_cfg: "TrainConfig",
) -> torch.optim.Optimizer:
    """
    Costruisce MuonAdamW o AdamW in base a ``train_cfg.use_muon``.

    Args:
        model: Harold (o DDP-wrapped Harold — gestisce entrambi)
        train_cfg: configurazione training

    Returns:
        MuonAdamW se ``train_cfg.use_muon=True``, altrimenti AdamW.
    """
    # Unwrap DDP se necessario
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        raw: nn.Module = model.module
    else:
        raw = model

    if not train_cfg.use_muon:
        return torch.optim.AdamW(
            raw.parameters(),
            lr=train_cfg.lr,
            betas=train_cfg.adamw_betas,
            eps=train_cfg.adamw_eps,
            weight_decay=train_cfg.adamw_wd,
            fused=True,
        )

    muon_params: List[torch.nn.Parameter] = []
    adamw_params: List[torch.nn.Parameter] = []

    for name, param in raw.named_parameters():
        if not param.requires_grad:
            continue
        if _is_muon_eligible(name, param):
            muon_params.append(param)
        else:
            adamw_params.append(param)

    n_muon  = sum(p.numel() for p in muon_params)
    n_adamw = sum(p.numel() for p in adamw_params)
    if is_main():
        print(f"MuonAdamW — Muon: {n_muon/1e6:.1f}M params  |  AdamW: {n_adamw/1e6:.1f}M params")

    param_groups = [
        {
            "kind":         "muon",
            "params":       muon_params,
            "lr":           train_cfg.lr,
            "momentum":     train_cfg.muon_momentum,
            "beta2":        train_cfg.muon_beta2,
            "ns_steps":     train_cfg.muon_ns_steps,
            "weight_decay": train_cfg.muon_wd,
        },
        {
            "kind":         "adamw",
            "params":       adamw_params,
            "lr":           train_cfg.lr,
            "betas":        train_cfg.adamw_betas,
            "eps":          train_cfg.adamw_eps,
            "weight_decay": train_cfg.adamw_wd,
        },
    ]

    return MuonAdamW(param_groups)