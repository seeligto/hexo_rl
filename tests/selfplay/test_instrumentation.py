"""Unit tests for PoolInstrumentation (§162 M2)."""
from __future__ import annotations

import threading
from collections import deque

import pytest

from hexo_rl.selfplay.instrumentation import (
    PoolInstrumentation,
    _compute_colony_extension,
    _compute_longest_line,
    _compute_n_components,
    _compute_stride5_metrics,
)


def _lock() -> threading.Lock:
    return threading.Lock()


def _make_instr(log_metrics: bool = True) -> PoolInstrumentation:
    return PoolInstrumentation(log_investigation_metrics=log_metrics)


def _game_complete(instr, lock, *, winner_code=1, move_history=None, worker_id=0,
                   terminal_reason=0, mv_min=0, mv_max=0, mv_distinct=1, stride5_run=0,
                   cluster_threshold=5):
    return instr.on_game_complete(
        lock, winner_code, move_history or [], worker_id,
        terminal_reason, mv_min, mv_max, mv_distinct, stride5_run,
        cluster_threshold,
    )


# ── on_game_complete ─────────────────────────────────────────────────────────

def test_draws_counter_increments_on_draw_terminal():
    instr = _make_instr()
    lk = _lock()
    _game_complete(instr, lk, winner_code=0, worker_id=0)  # draw
    _game_complete(instr, lk, winner_code=1, worker_id=0)  # win
    rates = instr.per_worker_draw_rates(lk)
    assert 0 in rates
    assert abs(rates[0] - 0.5) < 1e-9


def test_terminal_reasons_histogram_accumulates():
    instr = _make_instr()
    lk = _lock()
    _game_complete(instr, lk, terminal_reason=0)  # six
    _game_complete(instr, lk, terminal_reason=0)  # six
    _game_complete(instr, lk, terminal_reason=2)  # cap
    counts = instr.terminal_reason_counts(lk)
    assert counts["six_in_a_row"] == 2
    assert counts["ply_cap"] == 1
    assert counts["colony"] == 0


def test_model_version_tracking_threadsafe():
    instr = _make_instr()
    lk = _lock()
    _game_complete(instr, lk, mv_min=10, mv_max=20, mv_distinct=3, winner_code=1)
    summary = instr.model_version_summary(lk)
    assert summary["n"] == 1
    assert summary["median_range"] == 10


def test_move_histories_ring_buffer_capacity():
    instr = _make_instr()
    lk = _lock()
    moves = [(0, 0), (1, 0)]
    for _ in range(110):
        _game_complete(instr, lk, move_history=moves)
    hist = instr.recent_move_histories(lk)
    assert len(hist) == 100  # maxlen=100 ring buffer


def test_stride5_p90_passive_calculation():
    instr = _make_instr()
    lk = _lock()
    for run in range(10):
        _, _, _, p90, _, _, _ = _game_complete(instr, lk, stride5_run=run)
    # P90 of [0..9] (10 values): index = max(0, int(10*0.9)-1) = 8 → value 8
    assert p90 == 8


def test_colony_extension_pure_function():
    # P1 at (0,0); P2 far away at (50,50)
    moves = [(0, 0), (50, 50)]
    count, total = _compute_colony_extension(moves)
    assert total == 2
    assert count == 2  # both stones far from any opponent stone


# ── B3a structural metrics (longest_line + n_components) ─────────────────────
# Fixtures cross-checked byte-for-byte against scripts/d1m_replay_analyzer.py
# (longest_line, longest_line_fraction, n_components all match the analyzer).

# Ply->player rule: ply0=P1, [1,2]=P2, [3,4]=P1, [5,6]=P2, [7,8]=P1, [9,10]=P2,
# [11]=P1.  Six P1 plies (0,3,4,7,8,11) -> a 6-in-a-row on the r=0 q-axis;
# P2 filler far away so it never interferes.
_SIX_IN_A_ROW_P1 = [
    (0, 0),            # ply0  P1
    (50, 50), (51, 50),  # ply1,2 P2
    (1, 0), (2, 0),    # ply3,4 P1
    (52, 50), (53, 50),  # ply5,6 P2
    (3, 0), (4, 0),    # ply7,8 P1
    (54, 50), (55, 50),  # ply9,10 P2
    (5, 0),            # ply11 P1  -> P1 = (0..5, 0)
]

# P1 = two disjoint pairs: {(0,0),(1,0)} and {(20,0),(21,0)}; gap 19 > thresh 5.
_TWO_CLUSTER_P1 = [
    (0, 0),            # ply0  P1   clusterA
    (50, 50), (51, 50),  # ply1,2 P2
    (1, 0), (20, 0),   # ply3,4 P1  clusterA, clusterB
    (52, 50), (53, 50),  # ply5,6 P2
    (21, 0),           # ply7  P1   clusterB
]


def test_longest_line_six_in_a_row():
    # Winner = P1 (winner_code=1): 6-in-a-row -> longest_line==6, fraction==1.0
    # (6 stones, all on one line).
    ll, frac = _compute_longest_line(_SIX_IN_A_ROW_P1, 5, 1)
    assert ll == 6
    assert abs(frac - 1.0) < 1e-9


def test_longest_line_capped_at_six():
    # 7 collinear P1 stones (plies 0,3,4,7,8,11,12 -> q=0..6 on r=0); raw run 7
    # but reported longest_line==6 (capped at WIN_LENGTH). fraction = 6/7.
    seven_collinear = [
        (0, 0),            # ply0  P1
        (50, 50), (51, 50),  # ply1,2 P2
        (1, 0), (2, 0),    # ply3,4 P1
        (52, 50), (53, 50),  # ply5,6 P2
        (3, 0), (4, 0),    # ply7,8 P1
        (54, 50), (55, 50),  # ply9,10 P2
        (5, 0), (6, 0),    # ply11,12 P1 -> P1 = (0..6, 0), raw run 7
    ]
    ll, frac = _compute_longest_line(seven_collinear, 5, 1)
    assert ll == 6  # 7 raw -> capped at WIN_LENGTH
    assert abs(frac - 6.0 / 7.0) < 1e-9


def test_n_components_two_disjoint_clusters():
    # Winner = P1: two pairs separated by 19 > threshold(5) -> 2 components.
    nc = _compute_n_components(_TWO_CLUSTER_P1, 5, 1)
    assert nc == 2


def test_n_components_cluster_threshold_honored():
    # Same two pairs; gap 19. threshold>=19 merges them into 1 component.
    assert _compute_n_components(_TWO_CLUSTER_P1, 5, 1) == 2
    assert _compute_n_components(_TWO_CLUSTER_P1, 19, 1) == 1


def test_structural_metrics_empty():
    assert _compute_longest_line([], 5, 1) == (0, 0.0)
    assert _compute_n_components([], 5, 1) == 0


def test_structural_metrics_via_on_game_complete():
    # End-to-end: pool path returns the 7-tuple with structural fields populated.
    instr = _make_instr(log_metrics=True)
    lk = _lock()
    out = _game_complete(instr, lk, winner_code=1, move_history=_SIX_IN_A_ROW_P1,
                         cluster_threshold=5)
    (_ext_c, _ext_t, _ext_f, _p90, longest_line, ll_frac, n_comp) = out
    assert longest_line == 6
    assert abs(ll_frac - 1.0) < 1e-9
    assert n_comp == 1  # all six P1 stones one connected line


def test_structural_metrics_off_when_log_disabled():
    # log_investigation_metrics False -> structural fields zeroed (gate respected).
    instr = _make_instr(log_metrics=False)
    lk = _lock()
    out = _game_complete(instr, lk, winner_code=1, move_history=_SIX_IN_A_ROW_P1,
                         cluster_threshold=5)
    (_ext_c, _ext_t, _ext_f, _p90, longest_line, ll_frac, n_comp) = out
    assert (longest_line, ll_frac, n_comp) == (0, 0.0, 0)


# ── _compute_stride5_metrics (§176 P16, formerly in pool.py) ─────────────────


def test_stride5_metrics_empty_history():
    assert _compute_stride5_metrics([]) == (0, 0)


def test_stride5_metrics_chain_along_r_row():
    # Four stones on r=0 at q ∈ {3, 8, 13, 18} → stride-5 chain of length 4.
    moves = [(3, 0), (8, 0), (13, 0), (18, 0)]
    stride5_max, row_max = _compute_stride5_metrics(moves)
    assert stride5_max == 4
    assert row_max == 4


def test_stride5_metrics_no_stride5_pattern():
    # Adjacent stones — row_max counts them; stride5_max reads each stone as
    # a degenerate "chain of length 1" (no stride-5 follow-on in row).
    moves = [(0, 0), (1, 0), (2, 0)]
    stride5_max, row_max = _compute_stride5_metrics(moves)
    assert stride5_max == 1
    assert row_max == 3
