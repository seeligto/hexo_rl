"""Shared constants and utility functions for the self-play pipeline."""

from __future__ import annotations

from typing import Any, Dict

# DEPRECATED — module-level BOARD_SIZE / N_ACTIONS are v6-only legacy. Pass
# `EncodingSpec` (`hexo_rl.encoding`) and read `spec.board_size` /
# `spec.policy_logit_count` (or the `n_actions` property) instead. Kept for
# backward compat with eval/probe call sites that still hard-code v6.
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
