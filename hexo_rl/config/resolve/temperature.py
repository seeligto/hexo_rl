"""CONFRES P4 — temperature resolution authority (self-play schedule + eval constant).

Co-locates the two LIVE temperature resolvers as one authority module (design §6 P4). They
resolve DIFFERENT quantities for DIFFERENT contexts and numerically agree at defaults (tau=0.5),
so they stay separate accessors — not a merged value:

- ``resolve_selfplay_temperature`` — self-play move-temperature schedule params
  ``(temp_threshold_compound_moves, temp_min)``, with the L9/§156 cosine-BAN pinned into the
  fallback ``(0, 0.5)``.
- ``resolve_eval_temperature`` — the live-eval constant PUCT policy temperature (default 0.5).

The legacy Python worker's mode-based ``get_temperature`` (``selfplay/utils.py``) is a THIRD,
off-Rust-training-path resolver left as-is (``temperature_threshold_ply`` is dead for the Rust
path). stdlib-only import — safe from ``hexo_rl.eval`` / ``hexo_rl.selfplay`` without a cycle (§8 N3).
"""
from __future__ import annotations

from typing import Any, Dict

# The ONE live-eval constant temperature default (was ``defaults.DEFAULT_EVAL_TEMPERATURE``).
EVAL_TEMPERATURE_DEFAULT: float = 0.5


def resolve_selfplay_temperature(pc: Dict[str, Any]) -> tuple[int, float]:
    """Resolve ``(temp_threshold_compound_moves, temp_min)`` from a ``playout_cap`` dict.

    Fallback = cosine-OFF ``(0, 0.5)`` — mirrors the Rust ``SelfPlayRunnerConfig`` default and
    the documented production posture: a variant that omits these keys inherits a constant
    tau=0.5, and must NOT silently re-arm the §156/L9 draw-collapse cosine (the legacy fallback
    was the toxic 15 / 0.05). Schedule-ON values (e.g. a D-TEMPDECAY probe/smoke arm) pass
    through unchanged. (D-TEMPDECAY C1, 2026-06-12.)
    """
    thr = pc.get("temperature_threshold_compound_moves")
    tmin = pc.get("temp_min")
    return (
        int(thr) if thr is not None else 0,       # absent OR explicit null -> OFF
        float(tmin) if tmin is not None else 0.5,
    )


def resolve_eval_temperature(cfg_value: float | None) -> float:
    """Resolve the live-eval constant PUCT policy temperature. Config value wins; else 0.5."""
    if cfg_value is not None:
        return float(cfg_value)
    return EVAL_TEMPERATURE_DEFAULT
