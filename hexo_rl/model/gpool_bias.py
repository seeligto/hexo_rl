"""§170 P3/P4 — gpool-bias side-branch for A1 (K-cluster + min/max).

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

The policy bias is multiplied by a learnable scalar ``gate`` (init 0.0). At
init the gate is exactly zero → the branch contributes nothing → the
model is BYTE-EXACT A1. Only as the gradient grows the gate does the
global summary earn weight.

§170 P4 — ``policy_only=True`` decouples the value head from the bias
branch by construction:
  - value_bias is returned as a constant zero tensor with NO gradient
    path back to the encoder / value_proj / gate. The value head therefore
    receives no signal from the global crop and value_proj.weight.grad is
    None (the parameter is preserved purely for state-dict round-trip with
    P3 checkpoints).
  - The gate parameter (existing P3 ``gate``) drives the policy path only.
P3 falsified the bilateral-bias bet (argmax +7.5 pp, MCTS-64 −15 pp); P4
tests whether confining injection to the policy head preserves the policy
lift while the value head stays anchored at A1.

Spec references:
  - audit/encoding_spikes/s2_global_pooling.md (KataGo gpool operator)
  - docs/07_PHASE4_SPRINT_LOG.md §170 P3 (FALSIFIED) / §170 P4 (policy-only).
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
        policy_only:  §170 P4 — when True, ``forward()`` returns a constant
                      zero ``value_bias`` with NO gradient path back to the
                      encoder / value_proj / gate. ``value_proj`` is still
                      constructed (state-dict round-trip with §170 P3
                      checkpoints) but never invoked at forward time.
    """

    def __init__(
        self,
        filters: int,
        n_actions: int,
        value_hidden: int = 256,
        in_channels: int = 3,
        policy_only: bool = False,
    ) -> None:
        super().__init__()
        self.filters = int(filters)
        self.n_actions = int(n_actions)
        self.value_hidden = int(value_hidden)
        self.policy_only: bool = bool(policy_only)
        self.encoder = GlobalTokenEncoder(
            in_channels=in_channels,
            out_dim=filters,
        )
        self.value_proj = nn.Linear(filters, self.value_hidden)
        self.policy_proj = nn.Linear(filters, self.n_actions)
        # Gate scalar — init 0.0 so the branch is byte-exact A1 at construction.
        # Under policy_only=True this gate drives the policy path only.
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
            value_bias:  (B, value_hidden) gated additive bias. Under
                         ``policy_only=True`` this is a fresh zero tensor
                         with no autograd connection — the value head is
                         frozen at A1 by construction.
            policy_bias: (B, n_actions) gated additive bias.
        """
        g = self.encoder(global_crop)                       # (B, filters)
        gated_p = self.policy_proj(g)                       # (B, n_actions)
        scale = self.gate.to(g.dtype)
        policy_bias = scale * gated_p
        if self.policy_only:
            # Construct a fresh zero tensor (NOT a view of any computation
            # graph node) so backward through value_bias terminates here.
            # value_proj is intentionally not invoked: its weights remain
            # for state-dict round-trip with §170 P3 ckpts but contribute
            # no signal and no gradient flows through them.
            value_bias = torch.zeros(
                global_crop.size(0),
                self.value_hidden,
                dtype=g.dtype,
                device=g.device,
            )
        else:
            gated_v = self.value_proj(g)                    # (B, value_hidden)
            value_bias = scale * gated_v
        return value_bias, policy_bias

    def gate_value(self) -> float:
        """Return the current scalar gate value as a Python float.

        Mirrors ``PMAGlobalPool.gate_value()`` for training-loop logging —
        surfaces whether the bias branch earned weight or stayed at init.
        """
        return float(self.gate.detach().cpu().item())
