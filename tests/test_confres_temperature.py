"""CONFRES P4 — temperature resolution authority (self-play schedule + eval constant).

Byte-pure consolidation: co-locates the two LIVE temperature resolvers in one module.
- self-play move-temperature schedule (was pool.resolve_playout_cap_temperature) — the L9/§156
  cosine-ban authority (fallback pinned OFF (0, 0.5), never the toxic legacy 15/0.05).
- eval constant PUCT policy temperature (was scattered DEFAULT_EVAL_TEMPERATURE=0.5 + 2 sites).
Both keep their exact values (tau=0.5) — no behavior change.
"""
from __future__ import annotations

from hexo_rl.config.resolve.temperature import (
    EVAL_TEMPERATURE_DEFAULT,
    resolve_eval_temperature,
    resolve_selfplay_temperature,
)


def test_eval_temperature_default_is_0_5():
    assert resolve_eval_temperature(None) == 0.5
    assert EVAL_TEMPERATURE_DEFAULT == 0.5


def test_eval_temperature_cfg_value_wins():
    assert resolve_eval_temperature(0.3) == 0.3


def test_selfplay_temperature_l9_ban_default_off():
    # Absent keys -> cosine OFF (0, 0.5); must NOT re-arm the toxic legacy 15/0.05 cosine.
    assert resolve_selfplay_temperature({}) == (0, 0.5)


def test_selfplay_temperature_explicit_null_stays_off():
    assert resolve_selfplay_temperature(
        {"temperature_threshold_compound_moves": None, "temp_min": None}
    ) == (0, 0.5)


def test_selfplay_temperature_explicit_schedule_passes_through():
    # Explicit opt-in (e.g. a D-TEMPDECAY probe arm) is allowed — the ban is on the DEFAULT.
    assert resolve_selfplay_temperature(
        {"temperature_threshold_compound_moves": 15, "temp_min": 0.05}
    ) == (15, 0.05)


def test_pool_reexports_the_selfplay_resolver():
    # Back-compat: the old name still imports from pool.py and IS the moved resolver.
    from hexo_rl.selfplay.pool import resolve_playout_cap_temperature

    assert resolve_playout_cap_temperature is resolve_selfplay_temperature


def test_direct_evaluator_eval_temperature_is_0_5():
    from hexo_rl.eval.evaluator import Evaluator

    class _DummyModel:
        pass

    ev = Evaluator(_DummyModel(), "cpu", {})
    assert ev._eval_temperature == 0.5
