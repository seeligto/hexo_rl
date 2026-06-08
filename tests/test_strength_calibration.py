"""D-EVALFOUND C5 (§2.5) — calibrate the strength-abort knobs from a banked-ladder RR.

Pre-registered method (design §2.5):
  strength_abort_floor = aggregate s.t. >=95% of the "healthy" rungs pass AND >=80% of
    post-peak rungs cluster below it (a separation, not a guess).
  strength_cycle_density_max = max(0.15, 75th-pct of the ladder's 3-cycle density)
    anchored to the measured 0.073 (M-CYC).
These are pure functions over per-rung aggregates; the heavy RR that produces those
aggregates is operator-gated.
"""
from __future__ import annotations

import pytest

from hexo_rl.eval.strength_calibration import (
    calibrate_cycle_density_max,
    calibrate_strength_floor,
)


def test_floor_separates_healthy_from_post_peak():
    # healthy rungs sit high (~0.55-0.62), post-peak collapse (~0.30-0.40)
    healthy = [0.58, 0.61, 0.55, 0.60, 0.57]
    post_peak = [0.40, 0.35, 0.30, 0.38, 0.33]
    floor = calibrate_strength_floor(healthy, post_peak)
    # floor must let >=95% healthy pass (>= floor) and >=80% post-peak fall below
    assert sum(h >= floor for h in healthy) / len(healthy) >= 0.95
    assert sum(p < floor for p in post_peak) / len(post_peak) >= 0.80
    # and it must sit strictly between the two clusters
    assert max(post_peak) <= floor <= min(healthy)


def test_floor_returns_none_when_clusters_overlap_unseparably():
    # no value can satisfy both constraints → calibration must refuse, not guess
    healthy = [0.40, 0.45, 0.50]
    post_peak = [0.42, 0.48, 0.55]
    assert calibrate_strength_floor(healthy, post_peak) is None


def test_cycle_density_max_floored_at_015():
    # observed densities all small (≈0.073) → max(0.15, p75) = 0.15
    densities = [0.05, 0.07, 0.073, 0.08, 0.06]
    assert calibrate_cycle_density_max(densities) == pytest.approx(0.15)


def test_cycle_density_max_tracks_high_p75():
    # a markedly more tangled ladder → p75 dominates the 0.15 floor
    densities = [0.10, 0.20, 0.25, 0.30, 0.35]
    val = calibrate_cycle_density_max(densities)
    assert val > 0.15
