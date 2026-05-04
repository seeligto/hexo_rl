"""Phase B' Class-4 stride-5 detector — unit tests.

Locks in `_compute_stride5_metrics` behaviour against synthetic patterns
covering all three hex axes plus boundary cases.  The detector is
color-blind by design: it measures stone-on-row geometry, not same-color
sub-runs.
"""
from __future__ import annotations

from hexo_rl.selfplay.pool import _compute_stride5_metrics


def test_empty_history():
    assert _compute_stride5_metrics([]) == (0, 0)


def test_single_stone():
    assert _compute_stride5_metrics([(0, 0)]) == (0, 1)


def test_q_axis_stride5_run_three():
    # Same r, q at {0, 5, 10} → stride-5 chain of length 3.
    moves = [(0, 0), (5, 0), (10, 0)]
    sr, rm = _compute_stride5_metrics(moves)
    assert sr == 3
    assert rm == 3


def test_q_axis_stride5_run_four_with_gap():
    # Same r, q at {0, 5, 10, 15, 21} → length-4 stride-5 chain plus an off-stride extra.
    moves = [(0, 0), (5, 0), (10, 0), (15, 0), (21, 0)]
    sr, rm = _compute_stride5_metrics(moves)
    assert sr == 4
    assert rm == 5


def test_r_axis_stride5_run_three():
    # Same q, r at {0, 5, 10} → row keyed by q, position = r.
    moves = [(0, 0), (0, 5), (0, 10)]
    sr, rm = _compute_stride5_metrics(moves)
    assert sr == 3
    assert rm == 3


def test_s_axis_stride5_run_three():
    # axis_s row: s = -q-r constant. Pattern (0,0), (5,-5), (10,-10) → s=0
    # for all three; positions along q = 0, 5, 10 → stride-5 chain length 3.
    moves = [(0, 0), (5, -5), (10, -10)]
    sr, rm = _compute_stride5_metrics(moves)
    assert sr == 3
    assert rm == 3


def test_no_stride5_chain_distance_4():
    # Same r, q at {0, 4, 8} → no stride-5 chain (each row-pair has 1 stone
    # connection at stride 4, not 5).
    moves = [(0, 0), (4, 0), (8, 0)]
    sr, rm = _compute_stride5_metrics(moves)
    assert sr == 1  # singleton chains for each stone
    assert rm == 3


def test_color_blind_mixed_color_chain():
    # Stride-5 chain spanning P1/P2 stones (ply rule alternates) — detector
    # should report length 4 regardless of color.
    moves = [(0, 0), (5, 0), (10, 0), (15, 0)]
    sr, _ = _compute_stride5_metrics(moves)
    assert sr == 4


def test_row_max_density_picks_densest_axis():
    # 4 stones in r-axis row (same r), 2 stones in q-axis row.
    moves = [(0, 0), (1, 0), (2, 0), (3, 0), (0, 1)]
    _, rm = _compute_stride5_metrics(moves)
    assert rm == 4


def test_diagnosis_report_smoke_sample():
    # Synthetic cap-bound game — 30-long stride-5 chain along east-west.
    # Mirrors the diagnosis-report shape but reduced; ensures the detector
    # scales linearly to long chains without off-by-one.
    moves = [(5 * k, 3) for k in range(30)]
    sr, rm = _compute_stride5_metrics(moves)
    assert sr == 30
    assert rm == 30
