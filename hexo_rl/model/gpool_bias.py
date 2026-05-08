"""§170 P3 — gpool-bias side-branch for A1 (K-cluster + min/max).

Adds a KataGo-style global-pool summary as an additive, K-invariant bias to
both heads of the canonical A1 architecture WITHOUT perturbing the load-
bearing per-cluster scatter-max value semantics. Reuses
``GlobalTokenEncoder`` verbatim — same canvas-mask plumbing, same
KataGo gpool operator (mean ⊕ size-aware mean ⊕ max).

Side-branch outputs:
  - value_bias:  (B, 256) — added to the value head's post-ReLU(value_fc1)
                 hidden activation BEFORE the final value_fc2 → tanh.
  - policy_bias: (B, n_actions) — added to the raw per-cluster policy_fc
                 logits BEFORE log_softmax. Same bias broadcast to every
                 cluster window — the bot-side scatter-max-on-prob then
                 operates on softmax(logits + bias), which is equivalent to
                 the model adding the same bias to every cluster.

Both biases are multiplied by a learnable scalar ``gate`` (init 0.0). At
init the gate is exactly zero → the branch contributes nothing → the
model is BYTE-EXACT A1. Only as the gradient grows the gate does the
global summary earn weight.

Spec references:
  - audit/encoding_spikes/s2_global_pooling.md (KataGo gpool operator)
  - docs/07_PHASE4_SPRINT_LOG.md §170 P3 (additive bias rationale; A3 verdict
    that PMA replaced load-bearing min-pool semantics)
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from hexo_rl.model.global_token import GlobalTokenEncoder


class GpoolBiasBranch(nn.Module):
    """K-invariant additive bias branch for A1's value + policy heads.

    Args:
        filters:      trunk filter count; sizes the GlobalTokenEncoder output.
        n_actions:    policy head output size (cluster cells + pass slot).
        value_hidden: width of the value head's hidden layer (matches
                      network.py's value_fc1 out_features). Default 256.
        in_channels:  global-crop plane count (cur / opp / canvas mask).
                      Pinned at 3 to match GlobalTokenEncoder's contract.
    """

    def __init__(
        self,
        filters: int,
        n_actions: int,
        value_hidden: int = 256,
        in_channels: int = 3,
    ) -> None:
        super().__init__()
        self.filters = int(filters)
        self.n_actions = int(n_actions)
        self.value_hidden = int(value_hidden)
        self.encoder = GlobalTokenEncoder(
            in_channels=in_channels,
            out_dim=filters,
        )
        self.value_proj = nn.Linear(filters, self.value_hidden)
        self.policy_proj = nn.Linear(filters, self.n_actions)
        # Gate scalar — init 0.0 so the branch is byte-exact A1 at construction.
        # Stored as a (1,) parameter for trivial state_dict round-trip.
        self.gate = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))

    def forward(
        self, global_crop: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode the global crop and return (value_bias, policy_bias).

        Args:
            global_crop: (B, 3, H, W) — typically (B, 3, 32, 32). Plane 2 is
                         the canvas-realness mask consumed by KataGPool.

        Returns:
            value_bias:  (B, value_hidden) gated additive bias.
            policy_bias: (B, n_actions)   gated additive bias.
        """
        g = self.encoder(global_crop)                       # (B, filters)
        gated_v = self.value_proj(g)                        # (B, value_hidden)
        gated_p = self.policy_proj(g)                       # (B, n_actions)
        scale = self.gate.to(g.dtype)
        return scale * gated_v, scale * gated_p

    def gate_value(self) -> float:
        """Return the current scalar gate value as a Python float.

        Mirrors ``PMAGlobalPool.gate_value()`` for training-loop logging —
        surfaces whether the bias branch earned weight or stayed at init.
        """
        return float(self.gate.detach().cpu().item())
