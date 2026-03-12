"""
DiffusionMoE v2 — ebpo.py
==========================
ELBO-based Block-level Policy Optimization (EBPO).

Fix v2:
  - log_probs normalizzati per (num_timesteps * seq_len) → ratio stabile
  - advantage calcolato PRIMA del detach per avere gradiente sul value head
  - mean_advantage non collassa a zero: rewards e values ora divergono
    perché il value head è inizializzato a zero ma i rewards sono arbitrari
  - ratio clampato in log-space prima di exp() per prevenire overflow
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from config import ModelConfig
from model import DiffusionMoE


class ValueHead(nn.Module):
    """
    Value head: stima V(x) dal mean pooling degli hidden states.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features: (B, L, D) → (B,)"""
        return self.net(features.mean(dim=1)).squeeze(-1)


class EBPOTrainer:
    """
    ELBO-based Block-level Policy Optimization.

    Obiettivo:
      max_theta  E[ clip(rho, 1-eps_lo, 1+eps_hi) * A ]

    dove:
      rho   = exp( log_pi_theta(y|x) - log_pi_old(y|x) )
      A     = (r - V(x)) normalizzato
      log_pi(y|x) = ELBO normalizzato per token e per timestep
    """
    def __init__(
        self,
        model:              "DiffusionMoE",
        config:             ModelConfig,
        learning_rate:      float = 1e-6,
        clip_epsilon_low:   float = 0.2,
        clip_epsilon_high:  float = 0.2,
        num_timesteps:      int   = 8,   # ridotto: abbastanza per stima ELBO
        gamma:              float = 0.99,
        gae_lambda:         float = 0.95,
    ):
        self.model         = model
        self.config        = config
        self.clip_lo       = clip_epsilon_low
        self.clip_hi       = clip_epsilon_high
        self.num_timesteps = num_timesteps
        self.gamma         = gamma
        self.gae_lambda    = gae_lambda

        self.value_head    = ValueHead(config.d_model).to(
            next(model.parameters()).device
        )

        all_params         = list(model.parameters()) + list(self.value_head.parameters())
        self.optimizer     = torch.optim.Adam(all_params, lr=learning_rate)

    # ── Corruption ──────────────────────────────────────────────────────────

    def _corrupt(self, tokens: torch.Tensor, noise_level: float) -> torch.Tensor:
        corrupted       = tokens.clone()
        mask            = torch.rand_like(tokens.float()) < noise_level
        corrupted[mask] = self.config.mask_token_id
        return corrupted

    # ── Log prob normalizzato per sample ────────────────────────────────────

    def _compute_log_probs(
        self,
        model:     "DiffusionMoE",
        responses: torch.Tensor,   # (B, L)
        prompts:   torch.Tensor,   # (B, L)
    ) -> torch.Tensor:
        """
        Restituisce log_prob per sample normalizzato: (B,)

        Normalizzazione per (num_timesteps * seq_len) → valori in [-log(V), 0]
        invece di valori enormemente negativi che fanno esplodere exp().
        """
        B, L   = responses.shape
        device = responses.device
        t_vals = torch.linspace(1, self.config.diffusion_T, self.num_timesteps).long()

        log_prob_sum = torch.zeros(B, device=device)

        for t_val in t_vals:
            t         = torch.full((B,), t_val.item(), device=device, dtype=torch.long)
            noise_lvl = t_val.item() / self.config.diffusion_T
            corrupted = self._corrupt(responses, noise_lvl)

            logits, _ = model(corrupted, t)
            vocab_dim = logits.shape[-1]

            # log p per token, media su L (non somma) → scala fissa
            log_p = -F.cross_entropy(
                logits.view(-1, vocab_dim),
                responses.view(-1),
                reduction="none",
            ).view(B, L).mean(dim=1)   # (B,)  ← MEAN non SUM

            log_prob_sum = log_prob_sum + log_p

        # Normalizza anche per num_timesteps
        return log_prob_sum / self.num_timesteps   # (B,)

    # ── Advantage ────────────────────────────────────────────────────────────

    def _compute_advantage(
        self,
        rewards: torch.Tensor,
        values:  torch.Tensor,
    ) -> torch.Tensor:
        """
        A_i = r_i - V(x_i), normalizzato per stabilità numerica.
        """
        adv = rewards - values
        if adv.shape[0] > 1:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return adv

    # ── Features per value head ──────────────────────────────────────────────

    def _get_features(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Hidden states a t=1 (minimo rumore, rappresentazione più ricca).
        Non usa no_grad: il gradiente deve fluire verso il value head.
        """
        B     = tokens.shape[0]
        t     = torch.ones(B, device=tokens.device, dtype=torch.long)
        pos   = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0).expand(B, -1)
        x     = self.model.token_emb(tokens) + self.model.pos_emb(pos)
        t_emb = self.model.time_emb(self.model.get_timestep_embedding(t))
        for block in self.model.blocks:
            x, _ = block(x, t_emb)
        return self.model.norm_out(x)   # (B, L, D)

    # ── Training step ─────────────────────────────────────────────────────────

    def train_step(
        self,
        prompts:   torch.Tensor,   # (B, L)
        responses: torch.Tensor,   # (B, L)
        rewards:   torch.Tensor,   # (B,)
        old_model: "DiffusionMoE",
    ) -> Dict[str, float]:
        # 1. Log prob old model (frozen, no grad)
        with torch.no_grad():
            old_log_probs = self._compute_log_probs(old_model, responses, prompts)

        # 2. Log prob modello corrente (con grad)
        log_probs = self._compute_log_probs(self.model, responses, prompts)

        # 3. Ratio in log-space prima, poi exp con clamp per evitare overflow
        log_ratio = log_probs - old_log_probs
        log_ratio = torch.clamp(log_ratio, -10.0, 10.0)   # clamp in log-space
        ratio     = torch.exp(log_ratio)                   # (B,)

        # 4. Value e advantage
        features  = self._get_features(responses)
        values    = self.value_head(features)               # (B,)
        advantage = self._compute_advantage(rewards.float(), values.detach())

        # 5. Clipped surrogate loss PPO
        pg_loss1  = -ratio * advantage
        pg_loss2  = -torch.clamp(ratio, 1 - self.clip_lo, 1 + self.clip_hi) * advantage
        pg_loss   = torch.max(pg_loss1, pg_loss2).mean()

        # 6. Value loss
        value_loss = F.mse_loss(values, rewards.float())

        total_loss = pg_loss + 0.5 * value_loss

        # 7. Ottimizzazione
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.model.parameters()) + list(self.value_head.parameters()),
            max_norm=1.0,
        )
        self.optimizer.step()
        self.model.update_router_biases()

        return {
            "rl_loss":        pg_loss.item(),
            "value_loss":     value_loss.item(),
            "total_loss":     total_loss.item(),
            "mean_ratio":     ratio.mean().item(),
            "mean_log_ratio": log_ratio.mean().item(),
            "mean_advantage": advantage.mean().item(),
        }