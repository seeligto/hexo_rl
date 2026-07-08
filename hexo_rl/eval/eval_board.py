"""Curriculum-aware eval board construction (D-SHRIMP S4b fix, 2026-07-08).

Every in-loop eval / promotion path builds its game board via
``Board.with_encoding_name``, which binds the registry's FIXED
``legal_move_radius`` for the encoding and structurally forbids the plain
``set_legal_move_radius`` setter once an encoding is bound. When a run uses a
``legal_move_radius_schedule`` curriculum (§174), self-play plays under the
curriculum-current radius while eval was pinned to the registry default — a
train/eval regime mismatch that biases the promotion gate (deploy_strength +
the off-window robustness sub-gate).

This module is the single seam that fixes it:

* :func:`make_eval_board` threads the curriculum-current radius into the eval
  board via the engine's ``override_legal_move_radius`` bypass (the guard is
  intentionally skipped — see ``engine/src/pyo3/board.rs``).
* :func:`resolve_eval_radius` keeps eval CONNECTED to self-play by construction:
  by default the eval radius IS the resolved self-play curriculum radius (the
  same value ``StepCoordinator._resolve_radius`` produces from the one
  ``legal_move_radius_schedule``), with an explicit scalar override for the
  fixed-yardstick case. There is no second radius source to drift from.
"""
from __future__ import annotations

from engine import Board


def make_eval_board(encoding_name: str, legal_move_radius: int | None = None) -> Board:
    """Build an eval board on ``encoding_name``, honouring the curriculum radius.

    ``legal_move_radius=None`` (default) leaves the registry's per-encoding radius
    untouched — back-compatible for non-curriculum runs. A non-None value applies
    the curriculum-current (or explicitly pinned) radius via the engine bypass.
    """
    board = Board.with_encoding_name(encoding_name)
    if legal_move_radius is not None:
        board.override_legal_move_radius(int(legal_move_radius))
    return board


def resolve_eval_radius(
    curriculum_radius: int | None,
    override: int | None = None,
) -> int | None:
    """Resolve the radius eval should run under.

    Args:
        curriculum_radius: the run's self-play curriculum-current radius
            (``StepCoordinator._current_radius`` — resolved from the single
            ``legal_move_radius_schedule``). ``None`` when the run has no
            schedule (eval then keeps the registry default).
        override: optional configurable pin (``evaluation.legal_move_radius``).
            ``None`` → track the curriculum (default; eval and self-play share
            one source and cannot drift). An int → fix eval at that radius (the
            stable-yardstick option), regardless of the curriculum stage.

    Returns:
        The radius to pass to :func:`make_eval_board`, or ``None`` to keep the
        registry default.
    """
    if override is not None:
        return int(override)
    return curriculum_radius
