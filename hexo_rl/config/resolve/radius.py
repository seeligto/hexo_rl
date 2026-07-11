"""CONFRES 6c/6d — the radius resolution AUTHORITY (design §4 eval_radius, §8 B6, N3).

One rule module for the legal_move_radius curriculum, delegated to by both invocation surfaces:

- self-play / in-loop eval — ``StepCoordinator._resolve_radius`` (schedule@step) +
  ``_resolve_eval_radius`` (curriculum vs explicit pin) already funnel through
  ``eval_board.resolve_eval_radius``. This module re-exports that same rule + adds
  :func:`resolve_radius_from_schedule` so the schedule-scan logic is single-sourced.
- offline eval fleet — ``scripts/evalfair/core.radius_from_checkpoint`` drives the SAME
  schedule-scan over a checkpoint's baked config, and :func:`require_offline_radius` HARD-ERRORS
  (B6) when the ckpt carries no baked schedule AND no explicit ``--radius-stage`` override, instead
  of the pre-CONFRES silent registry-default fall-through.

Import constraint (§8 N3): ``eval_board`` is imported LAZILY inside the wrapper (not at module
import) so ``hexo_rl.config.resolve.radius`` does not pull ``hexo_rl.eval`` at import time and
cycle via ``hexo_rl/eval/__init__.py``.

Design: docs/designs/confres_design.md §4 (eval_radius ← curriculum), §8 (offline HARD-ERROR B6).
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence


class OfflineRadiusUnresolvableError(ValueError):
    """An offline instrument could not resolve a radius (B6, design §4/§8).

    The checkpoint carries no baked ``legal_move_radius_schedule`` AND no explicit
    ``--radius-stage`` was supplied. Pre-CONFRES this fell through to the registry default silently
    — a per-stage book read at the wrong radius biases Series B toward a false plateau. HARD-ERROR
    instead: the sanctioned weights-only strip must PRESERVE the curriculum stage in the ckpt
    metadata/config, or the operator must pass ``--radius-stage``.
    """


def resolve_radius_from_schedule(
    schedule: Sequence[Mapping[str, Any]] | None,
    step: int,
) -> int | None:
    """Resolve the curriculum-current radius at ``step`` from a ``legal_move_radius_schedule``.

    The single schedule-scan rule (was inlined at ``StepCoordinator._resolve_radius`` and re-driven
    over a checkpoint's config in ``evalfair.core.radius_from_checkpoint``). ``None`` when no
    schedule is configured (the caller then keeps the registry per-encoding default). Entries are
    ordered by ``step``; the last entry whose ``step`` is ``<=`` the query step wins.
    """
    if not schedule:
        return None
    current: int | None = None
    for entry in schedule:
        if step >= entry["step"]:
            current = entry["radius"]
    return current


def resolve_eval_radius(curriculum_radius: int | None, override: int | None = None) -> int | None:
    """Resolve the radius eval/promotion boards run under (delegates to ``eval_board``).

    Thin re-export of ``hexo_rl.eval.eval_board.resolve_eval_radius`` (the dffd5aa template,
    design §4 law #4) so the resolve package is the ONE import surface for radius resolution.
    Lazy import (§8 N3 cycle guard). ``override`` (an int, from ``evaluation.legal_move_radius``)
    pins a fixed yardstick; ``None`` tracks the curriculum.
    """
    from hexo_rl.eval.eval_board import resolve_eval_radius as _resolve

    return _resolve(curriculum_radius, override)


def require_offline_radius(
    resolved: int | None,
    radius_stage_override: int | None,
    *,
    ckpt_label: str = "<checkpoint>",
) -> int:
    """Offline HARD-ERROR gate (B6): return a concrete radius or raise.

    Precedence: an explicit ``--radius-stage`` override wins; else the schedule-resolved radius; if
    BOTH are ``None`` → :class:`OfflineRadiusUnresolvableError` naming the checkpoint + the fix
    (preserve the stage in the strip, or pass ``--radius-stage``). This is the offline-fleet
    replacement for the silent ``None`` → registry-default fall-through.
    """
    if radius_stage_override is not None:
        return int(radius_stage_override)
    if resolved is not None:
        return int(resolved)
    raise OfflineRadiusUnresolvableError(
        f"cannot resolve legal_move_radius for {ckpt_label}: the checkpoint carries no baked "
        "legal_move_radius_schedule and no --radius-stage was supplied. A per-stage book read at "
        "the wrong radius biases the measurement; refusing to silently fall back to the registry "
        "default. Fix: preserve the curriculum stage in the checkpoint (weights-only strip must "
        "keep config['selfplay']['legal_move_radius_schedule'] + step, or metadata radius_stage), "
        "or pass --radius-stage <int> explicitly."
    )
