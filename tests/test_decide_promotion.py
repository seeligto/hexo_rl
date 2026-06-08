"""D-EVALFOUND C4/C5 — promotion conjunction (decisions 3+4).

PROMOTE iff strength_ok AND robustness_ok. Strength uses the fixed-reference aggregate
when present (decision 4: replace wr_best), else falls back to the existing wr_best+CI
gate result. Robustness (decision 3: gates PROMOTE) blocks when the off-window rate
exceeds the bar; MISSING robustness data is a pass (no false block when the monitor is
disabled — the review gotcha).
"""
from __future__ import annotations

from hexo_rl.eval.gate_logic import decide_promotion


def test_promote_when_wr_best_passes_and_no_new_signals():
    # back-compat: no strength aggregate, no robustness measurement → wr_best gate decides
    d = decide_promotion(wr_best_promoted=True, strength_aggregate=None, strength_floor=0.45,
                         robustness_rate=None, robustness_threshold=0.06)
    assert d.promoted is True


def test_block_when_wr_best_fails_and_no_new_signals():
    d = decide_promotion(wr_best_promoted=False, strength_aggregate=None, strength_floor=0.45,
                         robustness_rate=None, robustness_threshold=0.06)
    assert d.promoted is False


def test_robustness_blocks_promotion_even_if_wr_best_passes():
    d = decide_promotion(wr_best_promoted=True, strength_aggregate=None, strength_floor=0.45,
                         robustness_rate=0.235, robustness_threshold=0.06)
    assert d.promoted is False
    assert "robust" in d.reason.lower()


def test_robustness_pass_allows_promotion():
    d = decide_promotion(wr_best_promoted=True, strength_aggregate=None, strength_floor=0.45,
                         robustness_rate=0.04, robustness_threshold=0.06)
    assert d.promoted is True


def test_strength_aggregate_replaces_wr_best_when_present():
    # decision 4: when the fixed-ref aggregate is present it REPLACES wr_best as the
    # strength criterion. Here wr_best "passed" but the aggregate is below floor → block.
    d = decide_promotion(wr_best_promoted=True, strength_aggregate=0.30, strength_floor=0.45,
                         robustness_rate=None, robustness_threshold=0.06)
    assert d.promoted is False
    assert "strength" in d.reason.lower()


def test_strength_aggregate_above_floor_promotes():
    d = decide_promotion(wr_best_promoted=False, strength_aggregate=0.60, strength_floor=0.45,
                         robustness_rate=None, robustness_threshold=0.06)
    assert d.promoted is True  # aggregate replaces wr_best; wr_best result ignored


def test_both_axes_required():
    # strength ok but robustness fails → block
    d = decide_promotion(wr_best_promoted=True, strength_aggregate=0.60, strength_floor=0.45,
                         robustness_rate=0.20, robustness_threshold=0.06)
    assert d.promoted is False
