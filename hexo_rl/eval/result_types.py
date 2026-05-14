"""Typed return shape for :meth:`hexo_rl.eval.eval_pipeline.EvalPipeline.run_evaluation`.

Cataloged 2026-05-14 (§176 P35) from ``eval_pipeline.py``.  ``total=False``
because most opponent-arms can be stride-skipped or disabled, so any single
round emits a subset of the keys.  The TypedDict is purely a static-type
annotation — no runtime validation, no schema enforcement.

Consumers (currently ``hexo_rl/training/step_coordinator.py:658``) get IDE
completion + mypy coverage instead of opaque ``dict[str, Any]``.

Field catalog (see ``run_evaluation`` source for exact write sites):

Always-set (entry into ``run_evaluation``):
  - ``step``: training step the evaluation was triggered at
  - ``promoted``: whether the round passed all promotion gates
  - ``eval_games``: total games played this round across all opponents
  - ``value_fc2_weight_abs_max``: §174 G4 value-head |max| measurement
  - ``g4_value_head_band_pass``: §174 G4 band flag

Conditional — vs Random arm:
  - ``wr_random``, ``ci_random``, ``colony_wins_random``

Conditional — vs SealBot arm:
  - ``wr_sealbot``, ``ci_sealbot``, ``colony_wins_sealbot``,
    ``sealbot_gate_passed``

Conditional — vs SealBot argmax-only (§170 P4 P1 DRIFT detector):
  - ``wr_argmax_n``, ``ci_argmax_n``, ``colony_wins_argmax_n``

Conditional — vs bootstrap_anchor (§155 T2 floor):
  - ``wr_bootstrap_anchor``, ``ci_bootstrap_anchor``,
    ``colony_wins_bootstrap_anchor``

Conditional — vs best_checkpoint:
  - ``wr_best``, ``ci_best``, ``colony_wins_best``

Conditional — Bradley-Terry ratings block:
  - ``ratings``: ``{player_name: {"rating": float, "ci": (lo, hi)}}``
  - ``elo_estimate``: current checkpoint's BT rating point estimate

Error path — set by ``step_coordinator._run_eval`` when the threaded
evaluation raises:
  - ``error``: bool flag for crash-on-eval
"""

from __future__ import annotations

from typing import TypedDict


class EvalRoundResult(TypedDict, total=False):
    """Return shape of :meth:`EvalPipeline.run_evaluation`.

    ``total=False`` — every field is optional because opponent arms can be
    disabled or stride-skipped.  Keys ``step`` / ``promoted`` /
    ``eval_games`` / ``value_fc2_weight_abs_max`` / ``g4_value_head_band_pass``
    are always set on the happy path, but the error fallback in
    ``step_coordinator._run_eval`` returns only ``{"promoted", "error",
    "step"}`` so total=False is the honest annotation.
    """

    # Always-set on the happy path
    step: int
    promoted: bool
    eval_games: int
    value_fc2_weight_abs_max: float
    g4_value_head_band_pass: bool

    # vs Random
    wr_random: float
    ci_random: tuple[float, float]
    colony_wins_random: int

    # vs SealBot
    wr_sealbot: float
    ci_sealbot: tuple[float, float]
    colony_wins_sealbot: int
    sealbot_gate_passed: bool

    # vs SealBot argmax-only (§170 P4 P1 DRIFT)
    wr_argmax_n: float
    ci_argmax_n: tuple[float, float]
    colony_wins_argmax_n: int

    # vs bootstrap_anchor (§155 T2 floor)
    wr_bootstrap_anchor: float
    ci_bootstrap_anchor: tuple[float, float]
    colony_wins_bootstrap_anchor: int

    # vs best_checkpoint
    wr_best: float
    ci_best: tuple[float, float]
    colony_wins_best: int

    # Bradley-Terry ratings block
    ratings: dict[str, dict[str, object]]
    elo_estimate: float

    # Error path (step_coordinator._run_eval fallback)
    error: bool


__all__ = ["EvalRoundResult"]
