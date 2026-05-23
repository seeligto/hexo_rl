"""§176 P24 — min_max policy + value per-window head helpers.

Extracted from ``hexo_rl/model/network.py`` to dedupe the per-window
policy + value computation that was inline in both
``HexTacToeNet.forward`` (single-window B) and
``HexTacToeNet.aggregated_forward_K`` (K-cluster scatter-pool).

§S181 FU-2 A2 (2026-05-23): value head's coverage-blind GMP half was
replaced with a multi-scale avg-pool aggregation
(:func:`multi_scale_avg_pool`). ``VALUE_FC1_MULTIPLIER`` is the single
source of truth for the value_fc1 input dim — both the ctor in
``network.py`` and this helper read it. Pre-A2 checkpoints (GAP+GMP,
2*filters) are state-dict incompatible; see the A2 shape guard in
``hexo_rl/eval/checkpoint_loader.py``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# §S181 FU-2 A2 — value_fc1 input dim multiplier (was 2 for GAP+GMP, now
# 5 for GAP (1C) + 2×2 adaptive avg pool (4C)). Single source of truth:
# the ctor in network.py reads this constant, as does
# multi_scale_avg_pool below. Bump together if the concat shape changes.
VALUE_FC1_MULTIPLIER: int = 5


def multi_scale_avg_pool(out: torch.Tensor) -> torch.Tensor:
    """§S181 FU-2 A2 — multi-scale avg-pool replacing GAP+GMP value-head input.

    Two scales of pure average pooling preserve coverage information at
    multiple resolutions (T2 §7 A2). The audit found the prior GMP half
    was coverage-blind (T2 §1.3: GMP(colony) ≡ GMP(extension) for matched
    peaks, exact max|diff|=0) and the PRIMARY permissive element of the
    §S178/§S180b colony attractor.

    - scale 1 (1C): global avg pool ``out.mean(dim=(2,3))`` — whole-board
      coverage.
    - scale 2 (4C): ``F.adaptive_avg_pool2d(out, 2).flatten(1)`` — quadrant
      coverage. Distinguishes single-quadrant extensions from a colony
      whose mass is distributed across multiple quadrants. (Note: 2×2
      adaptive pool of an odd 19×19 feature map uses uneven kernel sizes
      per axis — semantics are "average within each of 4 equal-area
      quadrants of the feature map".)

    Returns ``(N, VALUE_FC1_MULTIPLIER * C)``.
    """
    v_gap = out.mean(dim=(2, 3))                       # (N, C)
    v_2x2 = F.adaptive_avg_pool2d(out, 2).flatten(1)   # (N, 4C)
    return torch.cat([v_gap, v_2x2], dim=1)            # (N, 5C)


def min_max_window_head(
    out: torch.Tensor,
    *,
    policy_conv: nn.Conv2d,
    policy_fc: nn.Linear,
    value_fc1: nn.Linear,
    value_fc2: nn.Linear,
    policy_bias: Optional[torch.Tensor] = None,
    value_bias: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-window min_max head for ``has_pass_slot=true`` encodings.

    Args:
        out:        ``(N, C, H, W)`` trunk output. ``N`` is ``B`` when called
                    from ``HexTacToeNet.forward`` (one window per board) or
                    ``K`` when called from ``aggregated_forward_K`` (K cluster
                    windows for a single board).
        policy_conv / policy_fc / value_fc1 / value_fc2:
                    The trained head layers attached to ``HexTacToeNet`` —
                    passed by reference so state-dict keys stay rooted at the
                    parent module.
        policy_bias: Optional ``(N, n_actions)`` additive logit bias from the
                    §170 P3 ``GpoolBiasBranch``. ``None`` ⇒ A1 byte-exact.
        value_bias:  Optional ``(N, value_hidden)`` additive bias on the
                    ``value_fc1`` post-ReLU hidden. ``None`` ⇒ A1 byte-exact.

    Returns:
        ``(log_policy, value, v_logit)`` per-window:
          * ``log_policy``: ``(N, n_actions)`` log-softmax probabilities.
          * ``value``:      ``(N, 1)`` tanh of v_logit.
          * ``v_logit``:    ``(N, 1)`` raw pre-tanh logit (for BCE loss).
    """
    p = F.relu(policy_conv(out))
    p = p.flatten(1)
    p_logits = policy_fc(p)
    if policy_bias is not None:
        p_logits = p_logits + policy_bias.to(p_logits.dtype)
    log_policy = F.log_softmax(p_logits, dim=1)

    v = multi_scale_avg_pool(out)                       # (N, 5C)
    v = F.relu(value_fc1(v))
    if value_bias is not None:
        v = v + value_bias.to(v.dtype)
    v_logit = value_fc2(v)
    value = torch.tanh(v_logit)
    return log_policy, value, v_logit
