"""§176 P24 — min_max policy + value per-window head helpers.

Extracted from ``hexo_rl/model/network.py`` to dedupe the per-window
policy + value computation that was inline in both
``HexTacToeNet.forward`` (single-window B) and
``HexTacToeNet.aggregated_forward_K`` (K-cluster scatter-pool).

Design notes
------------

* The trained ``nn.Conv2d`` / ``nn.Linear`` layers stay attached as direct
  attributes of ``HexTacToeNet`` (``policy_conv``, ``policy_fc``,
  ``value_fc1``, ``value_fc2``). This file exports plain functions taking
  layer references as args — state-dict keys are unchanged. Loading any
  pre-existing v6 / v6w25 / v7full / v7 / v7e30 / v7mw checkpoint is byte-exact.

* ``min_max_window_head`` runs the per-window head once and returns the
  ``(log_policy, value, v_logit)`` triple. ``forward()`` calls it on the
  whole (B, C, H, W) trunk output; ``aggregated_forward_K`` calls it on
  the per-cluster (K, C, H, W) trunk output and feeds the per-K result
  into ``MinMaxPool``.

* The optional ``policy_bias`` / ``value_bias`` args carry the §170 P3
  gpool-bias side-branch contribution (None when the branch is inactive,
  preserving exact A1 byte-parity).

Cycle 3 Wave 8 Batch D (2026-05-17): renamed from ``min_max_v6_head`` /
``network_v6_head.py`` to generic ``min_max_window_head`` /
``network_min_max_head.py``. The head is shared across every
``has_pass_slot=true`` single-window registry encoding (v6, v6w25,
v7full, v7, v7e30, v7mw) — the historical ``v6`` prefix was
encoding-version legacy.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def min_max_window_head(
    out: torch.Tensor,
    *,
    policy_conv: nn.Conv2d,
    policy_fc: nn.Linear,
    value_fc1: nn.Linear,
    value_fc2: nn.Linear,
    policy_bias: Optional[torch.Tensor] = None,
    value_bias: Optional[torch.Tensor] = None,
    value_head_type: str = "scalar",
    value_fc2_bins: Optional[nn.Linear] = None,
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
        ``(log_policy, value, value_aux)`` per-window:
          * ``log_policy``:  ``(N, n_actions)`` log-softmax probabilities.
          * ``value``:       ``(N, 1)`` scalar in [-1, 1]:
                               scalar mode = tanh(v_logit);
                               dist65 mode = E[softmax(bin_logits) · support].
          * ``value_aux``:   scalar mode → ``(N, 1)`` raw pre-tanh v_logit;
                             dist65 mode → ``(N, 65)`` bin logits.
    """
    # Policy branch — 1×1 conv → ReLU → flatten → FC → optional bias →
    # log_softmax. Identical math to the pre-refactor inline form.
    p = F.relu(policy_conv(out))
    p = p.flatten(1)
    p_logits = policy_fc(p)
    if policy_bias is not None:
        p_logits = p_logits + policy_bias.to(p_logits.dtype)
    log_policy = F.log_softmax(p_logits, dim=1)

    # Value branch — global avg + max pool → cat → FC → ReLU → optional
    # bias → FC → tanh. Mirrors the pre-refactor inline form exactly.
    v_avg = out.mean(dim=(2, 3))
    v_max = out.amax(dim=(2, 3))
    v = torch.cat([v_avg, v_max], dim=1)
    v = F.relu(value_fc1(v))
    if value_bias is not None:
        v = v + value_bias.to(v.dtype)
    if value_head_type == "dist65":
        assert value_fc2_bins is not None, "value_fc2_bins required for dist65 head"
        # Local import avoids a hard model→training circular dependency at module
        # load time; the import itself is cheap (module already in sys.modules
        # after first call).
        from hexo_rl.training.binned_value import decode_binned_value  # noqa: PLC0415
        bin_logits = value_fc2_bins(v)                    # (N, 65)
        value = decode_binned_value(bin_logits)           # (N, 1) in [-1,1]
        return log_policy, value, bin_logits
    v_logit = value_fc2(v)
    value = torch.tanh(v_logit)
    return log_policy, value, v_logit
