"""PIPE-4 — dashboard draw-fraction band derived from live config, not stale.

`buffer_composition()` counted buffer outcomes in a hardcoded [-0.6, -0.4) band
to report `draw_target_fraction`. But the live `draw_value` (-0.1) and
`ply_cap_value` (0.0) both fall OUTSIDE that band, so the metric read ~0 for
every current run (monitoring-only bug). Derive the band from the live config
values (±epsilon) so it actually tracks draw-like outcomes.
"""
from hexo_rl.selfplay.pool import _draw_outcome_band


def test_band_covers_live_draw_and_ply_cap_outcomes():
    lo, hi = _draw_outcome_band(draw_value=-0.1, ply_cap_value=0.0)
    assert lo <= -0.1 <= hi, "band must include the draw_value outcome"
    assert lo <= 0.0 <= hi, "band must include the ply_cap_value outcome"
    # decisive ±1 wins must be excluded
    assert not (lo <= 1.0 <= hi)
    assert not (lo <= -1.0 <= hi)


def test_band_tracks_other_config_values():
    # A different split (organic draw -0.5, ply-cap 0.0) is still covered.
    lo, hi = _draw_outcome_band(draw_value=-0.5, ply_cap_value=0.0)
    assert lo <= -0.5 <= hi
    assert lo <= 0.0 <= hi
    assert not (lo <= 1.0 <= hi)
