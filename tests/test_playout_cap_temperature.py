"""C1 (D-TEMPDECAY 2026-06-12) — playout_cap within-game temperature resolution.

The production footgun fixed here: a variant whose config omits the
``playout_cap`` temperature keys must inherit cosine-OFF (threshold 0, floor
0.5) — the documented production posture and the §156/L9 draw-collapse guard —
NOT the legacy toxic 15 / 0.05 cosine that ``pool.py`` used to fall back to.

The resolver mirrors the Rust ``SelfPlayRunnerConfig`` default
(``temp_threshold_compound_moves=0``, ``temp_min=0.5``); the two are pinned
together by ``inv19_*`` (Rust) + ``default_config_schedule_is_off_constant_floor``
(Rust) + these tests (Python).
"""

from __future__ import annotations

import pytest

from hexo_rl.selfplay.pool import resolve_playout_cap_temperature


def test_playout_cap_temperature_defaults_to_cosine_off():
    # Variant omits the keys entirely -> cosine OFF (constant tau=temp_min).
    thr, tmin = resolve_playout_cap_temperature({})
    assert thr == 0, "missing threshold must default to 0 (schedule OFF)"
    assert tmin == pytest.approx(0.5), "missing temp_min must default to 0.5 (anti-colony floor)"


def test_playout_cap_temperature_reads_explicit_values():
    # Explicit schedule-ON values (e.g. a D-TEMPDECAY probe arm) pass through.
    thr, tmin = resolve_playout_cap_temperature(
        {"temperature_threshold_compound_moves": 12, "temp_min": 0.30}
    )
    assert thr == 12
    assert tmin == pytest.approx(0.30)


def test_playout_cap_temperature_partial_keys():
    # Threshold set, temp_min omitted -> threshold honored, floor defaults to 0.5.
    thr, tmin = resolve_playout_cap_temperature(
        {"temperature_threshold_compound_moves": 15}
    )
    assert thr == 15
    assert tmin == pytest.approx(0.5)
