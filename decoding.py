"""
DiffusionMoE v2 — decoding.py
"""

import torch
import torch.nn.functional as F
from typing import Optional
from config import ModelConfig


class ThresholdDecoding:
    def __init__(self, config: ModelConfig):
        self.num_steps      = config.diffusion_T
        self.mask_token_id  = config.mask_token_id
        self.mask_threshold = config.mask_threshold
        self.edit_threshold = config.edit_threshold

    def get_update_sets(self, x_t, logits, tau_mask, tau_edit):
        probs = F.softmax(logits, dim=-1)
        top_probs, top_tokens = probs.max(dim=-1)
        is_masked = (x_t == self.mask_token_id)
        gamma = is_masked & (top_probs > tau_mask)
        delta = (~is_masked) & (top_tokens != x_t) & (top_probs > tau_edit)
        return gamma, delta

    def step(self, x_t, logits, tau_mask, tau_edit):
        gamma, delta = self.get_update_sets(x_t, logits, tau_mask, tau_edit)
        top_tokens   = logits.argmax(dim=-1)
        x_next       = x_t.clone()
        x_next[gamma | delta] = top_tokens[gamma | delta]
        return x_next

    @torch.no_grad()
    def decode(
        self,
        model,
        initial_tokens: torch.Tensor,
        mode:           str  = "S",
        use_mbe:        bool = False,
        num_blocks:     int  = 4,
        num_steps:      Optional[int] = None,
    ) -> torch.Tensor:
        if mode == "S":
            tau_mask_start, tau_mask_end = 0.1, 0.4
            tau_edit_start, tau_edit_end = 0.5, 0.7
        else:
            tau_mask_start, tau_mask_end = 0.3, 0.7
            tau_edit_start, tau_edit_end = 0.7, 0.9

        steps  = num_steps or self.num_steps
        x      = initial_tokens.clone()
        B      = x.shape[0]
        device = x.device

        for step_idx in range(steps):
            progress = step_idx / max(steps - 1, 1)
            tau_mask = tau_mask_start + (tau_mask_end - tau_mask_start) * progress
            tau_edit = tau_edit_start + (tau_edit_end - tau_edit_start) * progress
            t_val    = max(1, int(self.num_steps * (1.0 - progress)))
            t        = torch.full((B,), t_val, device=device, dtype=torch.long)

            logits, _ = model(x, t)
            x = self.step(x, logits, tau_mask, tau_edit)

        # ── Force decode incondizionato ───────────────────────────────────
        # Sempre eseguito: rimpiazza tutti i [MASK] rimasti con argmax.
        t_final     = torch.zeros(B, device=device, dtype=torch.long)
        logits, _   = model(x, t_final)
        forced      = logits.argmax(dim=-1)          # (B, L)
        mask_pos    = (x == self.mask_token_id)      # (B, L) bool
        x[mask_pos] = forced[mask_pos]               # in-place, no if needed

        if use_mbe and num_blocks > 1:
            x = self.multi_block_editing(model, x, num_blocks)

        return x

    @torch.no_grad()
    def multi_block_editing(self, model, tokens, num_blocks):
        B, L   = tokens.shape
        bs     = max(1, L // num_blocks)
        edited = tokens.clone()
        device = tokens.device

        for block_idx in range(num_blocks - 1, -1, -1):
            start = block_idx * bs
            end   = min(start + bs, L)
            # MODIFICA ANCHE QUI: da t=1 a t=0
            t     = torch.zeros(B, device=device, dtype=torch.long)  # <-- CAMBIA DA ones A zeros

            logits, _             = model(edited, t)
            probs                 = F.softmax(logits, dim=-1)
            top_probs, top_tokens = probs.max(dim=-1, keepdim=True)  # <-- AGGIUNGI keepdim=True
            top_probs = top_probs.squeeze(-1)  # (B, L)
            top_tokens = top_tokens.squeeze(-1)  # (B, L)

            block_mask            = torch.zeros(B, L, dtype=torch.bool, device=device)
            block_mask[:, start:end] = True

            edit_cond             = (top_probs > self.edit_threshold) & block_mask
            edited_next = edited.clone()  # <-- AGGIUNGI clone per evitare problemi
            edited_next[edit_cond]     = top_tokens[edit_cond]
            edited = edited_next

        return edited