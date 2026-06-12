"""Shared constants and utility functions for the self-play pipeline."""

from __future__ import annotations

import math
from typing import Any, Dict

# DEPRECATED — module-level BOARD_SIZE / N_ACTIONS are v6-only legacy. Pass
# `EncodingSpec` (`hexo_rl.encoding`) and read `spec.board_size` /
# `spec.policy_logit_count` (or the `n_actions` property) instead. Kept for
# backward compat with eval/probe call sites that still hard-code v6.
from hexo_rl.encoding import lookup as _lookup_encoding

BOARD_SIZE: int = _lookup_encoding("v6").board_size

N_ACTIONS: int = BOARD_SIZE * BOARD_SIZE + 1  # 362  (v6 only — DEPRECATED)


def quarter_cosine_temperature(compound_move: int, threshold: int, temp_min: float) -> float:
    """Within-game quarter-cosine temperature — the single shared mechanism.

    Mirrors the Rust training-path ``compute_move_temperature``
    (``engine/src/game_runner/worker_loop/rotate.rs``) exactly:

        tau(cm) = max(temp_min, cos(pi/2 * cm / threshold))   for cm < threshold
                = temp_min                                     for cm >= threshold

    ``threshold == 0`` => the schedule is OFF: a constant ``temp_min`` at every
    compound move (no div-by-zero — the divide lives inside the ``cm < threshold``
    branch, which ``cm < 0`` never enters). (D-TEMPDECAY C4, 2026-06-12.)
    """
    if threshold > 0 and compound_move < threshold:
        return max(temp_min, math.cos(math.pi / 2 * compound_move / threshold))
    return temp_min


def get_temperature(ply: int, mode: str, config: Dict[str, Any]) -> float:
    """Return the MCTS sampling temperature for the current game state.

    Args:
        ply:    Total half-moves played so far (board.ply).
        mode:   "training"   → compound-turn quarter-cosine (see
                               :func:`quarter_cosine_temperature`); identical
                               shape + clock as the Rust training path.
                "evaluation" → tau=0.0 (argmax, deterministic).
                "bootstrap"  → tau=0.5 (moderate, for minimax corpus games).
        config: Config dict. The "training" branch reads
                ``temperature_threshold_compound_moves`` + ``temp_min`` from the
                ``mcts`` sub-dict (else top-level). The legacy
                ``temperature_threshold_ply`` is honoured as an eval/bot alias,
                auto-converted plies → compound-turns. Missing keys ⇒ schedule
                OFF (threshold 0 ⇒ constant ``temp_min``, default 0.5).

    Returns:
        Sampling temperature as a float.
    """
    if mode == "evaluation":
        return 0.0
    if mode == "bootstrap":
        return 0.5
    # Training / exploration mode: shared compound-turn quarter-cosine.
    mcts_cfg = config.get("mcts", config)

    def _get(key: str) -> Any:
        return mcts_cfg.get(key, config.get(key))

    threshold = _get("temperature_threshold_compound_moves")
    if threshold is None:
        # Legacy eval/bot alias: ply-clock threshold → compound-turns ((ply+1)//2).
        legacy_ply = _get("temperature_threshold_ply")
        threshold = (int(legacy_ply) + 1) // 2 if legacy_ply is not None else 0
    threshold = int(threshold)

    tm = _get("temp_min")
    temp_min = float(tm if tm is not None else 0.5)
    compound_move = 0 if ply == 0 else (ply + 1) // 2
    return quarter_cosine_temperature(compound_move, threshold, temp_min)
