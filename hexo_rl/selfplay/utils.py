"""Shared constants and utility functions for the self-play pipeline."""

from __future__ import annotations

from typing import Any, Dict, Tuple

# DEPRECATED — module-level BOARD_SIZE / N_ACTIONS are v6-only legacy. Pass
# `EncodingSpec` (`hexo_rl.encoding`) and read `spec.board_size` /
# `spec.policy_logit_count` (or the `n_actions` property) instead. Kept for
# backward compat with eval/probe call sites that still hard-code v6.
from hexo_rl.encoding import EncodingSpec  # §172 A4.2 canonical
from hexo_rl.utils.constants import BOARD_SIZE

N_ACTIONS: int = BOARD_SIZE * BOARD_SIZE + 1  # 362  (v6 only — DEPRECATED)


def get_temperature(ply: int, mode: str, config: Dict[str, Any]) -> float:
    """Return the MCTS sampling temperature for the current game state.

    Args:
        ply:    Total half-moves played so far (board.ply).
        mode:   "training"   → tau=1.0 for first N plies, tau=0.1 after.
                "evaluation" → tau=0.0 (argmax, deterministic).
                "bootstrap"  → tau=0.5 (moderate, for minimax corpus games).
        config: Config dict.  Reads ``temperature_threshold_ply`` from the
                ``mcts`` sub-dict if present, else top-level, else default 30.

    Returns:
        Sampling temperature as a float.
    """
    if mode == "evaluation":
        return 0.0
    if mode == "bootstrap":
        return 0.5
    # Training mode: ply-based schedule.
    mcts_cfg = config.get("mcts", config)
    threshold = int(mcts_cfg.get("temperature_threshold_ply",
                                  config.get("temperature_threshold_ply", 30)))
    return 1.0 if ply < threshold else 0.1


def flat_to_coords(flat: int) -> Tuple[int, int]:
    """Convert a flat board index back to (q, r) axial coordinates (v6 only).

    DEPRECATED — caller assumes BOARD_SIZE=19. For non-v6 encodings call
    `flat_to_coords_for(flat, spec)` instead.
    """
    half = (BOARD_SIZE - 1) // 2   # 9
    q = flat // BOARD_SIZE - half
    r = flat %  BOARD_SIZE - half
    return q, r


def flat_to_coords_for(flat: int, spec: EncodingSpec) -> Tuple[int, int]:
    """§172 A4.2 — encoding-aware flat→(q,r) conversion.

    Uses `spec.board_size` so v6 (19), v6w25 (25), v8 (25) all resolve correctly.
    """
    bs = spec.board_size
    half = (bs - 1) // 2
    q = flat // bs - half
    r = flat % bs - half
    return q, r


def n_actions_for(spec: EncodingSpec) -> int:
    """§172 A4.2 — return policy logit count for an encoding spec.

    Trivial wrapper around `spec.policy_logit_count`; exists so call sites
    that read the legacy `N_ACTIONS` module constant migrate to a discoverable
    registry-driven helper.
    """
    return spec.policy_logit_count
