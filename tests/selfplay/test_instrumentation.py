"""Unit tests for PoolInstrumentation (§162 M2)."""
from __future__ import annotations

import threading
from collections import deque

import pytest

from hexo_rl.selfplay.instrumentation import PoolInstrumentation, _compute_colony_extension


def _lock() -> threading.Lock:
    return threading.Lock()


def _make_instr(log_metrics: bool = True) -> PoolInstrumentation:
    return PoolInstrumentation(log_investigation_metrics=log_metrics)


def _game_complete(instr, lock, *, winner_code=1, move_history=None, worker_id=0,
                   terminal_reason=0, mv_min=0, mv_max=0, mv_distinct=1, stride5_run=0):
    return instr.on_game_complete(
        lock, winner_code, move_history or [], worker_id,
        terminal_reason, mv_min, mv_max, mv_distinct, stride5_run,
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
        _, _, _, p90 = _game_complete(instr, lk, stride5_run=run)
    # P90 of [0..9] (10 values): index = max(0, int(10*0.9)-1) = 8 → value 8
    assert p90 == 8


def test_colony_extension_pure_function():
    # P1 at (0,0); P2 far away at (50,50)
    moves = [(0, 0), (50, 50)]
    count, total = _compute_colony_extension(moves)
    assert total == 2
    assert count == 2  # both stones far from any opponent stone
