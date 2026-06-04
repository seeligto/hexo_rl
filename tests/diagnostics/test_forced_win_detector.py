"""Unit tests for the shared forced-win detector (§EVALGATE-B).

ONE source of the §PRELONG forced-win / off-window / win-from-policy detector,
factored out of the three ``scripts/structural_diagnosis/prelong_*.py`` copies so the
eval gate, dashboard, logs, and future probes share a single definition (no metric
drift).  All readouts derive from already-recorded games (the ``GameRecorder`` jsonl
schema) — no hot-path pass.  Geometry comes from the encoding spec (zero literals).
"""
from __future__ import annotations

from unittest.mock import Mock

from hexo_rl.diagnostics import forced_win_detector as fw


def _winning_game_moves():
    """P1 (side 1) builds a 5-in-a-row on axis r=0 then completes 6 → P1 win.

    Validated against the engine: every move is legal under legal_move_radius=5 and
    the final stone (5,0) completes a 6-in-a-row (first stone of P1's last turn).
    """
    return [
        (0, 0),                # P1 opening
        (0, 1), (0, 2),        # P2 (near, off-axis)
        (1, 0), (2, 0),        # P1
        (1, 1), (2, 1),        # P2
        (3, 0), (4, 0),        # P1  → (0,0),(1,0),(2,0),(3,0),(4,0) = 5 in a row
        (3, 1), (1, 2),        # P2
        (5, 0),                # P1 completes 6-in-a-row → win
    ]


# ── geometry (zero-literal helpers) ──────────────────────────────────────────

def test_trunc2_truncates_toward_zero():
    assert fw.trunc2(5) == 2
    assert fw.trunc2(-5) == -2
    assert fw.trunc2(4) == 2


def test_window_center_is_bbox_midpoint():
    assert fw.window_center([(0, 0), (4, 0), (0, 4)]) == (2, 2)


def test_cheb_distance():
    assert fw.cheb((0, 0), (3, -5)) == 5


# ── detector (validated fixture) ─────────────────────────────────────────────

def test_depth1_wins_finds_both_completion_cells():
    from engine import Board
    b = Board.with_encoding_name("v6_live2")
    for (q, r) in _winning_game_moves()[:-1]:   # stop before the winning move
        b.apply_move(q, r)
    wins = {tuple(c) for c in fw.depth1_wins(b, 1)}
    assert (5, 0) in wins
    assert (-1, 0) in wins


# ── flags (derived from the engine, not hardcoded geometry) ──────────────────

def test_is_off_window_flag_near_vs_far():
    from engine import Board
    from hexo_rl.encoding import lookup
    spec = lookup("v6_live2")
    b = Board.with_encoding_name("v6_live2")
    b.apply_move(0, 0)
    assert fw.is_off_window(b, (0, 0), spec) is False
    assert fw.is_off_window(b, (50, 50), spec) is True


# ── per-game analysis (offline, from a recorded move list) ───────────────────

def test_analyze_recorded_game_converted_in_window_forced_win():
    summary = fw.analyze_recorded_game(
        _winning_game_moves(), "x_win", encoding="v6_live2", mover_side=1,
    )
    assert summary.forced_win_turns >= 1
    assert summary.converted >= 1
    assert summary.off_window_forced_turns == 0      # near-origin line is in-window
    assert summary.path == "single-window"


def test_analyze_recorded_game_no_forced_win_short_game():
    summary = fw.analyze_recorded_game(
        [(0, 0), (0, 1), (1, 0)], "draw", encoding="v6_live2", mover_side=1,
    )
    assert summary.forced_win_turns == 0
    assert summary.converted == 0


def test_analyze_recorded_game_path_label_is_explicit():
    summary = fw.analyze_recorded_game(
        _winning_game_moves(), "x_win", encoding="v6_live2", mover_side=1,
        path="scatter",
    )
    assert summary.path == "scatter"


# ── smoothed trend (EMA, labelled path + n + smoothing) ──────────────────────

def test_forced_win_trend_ema_labels_path_n_smoothing():
    trend = fw.ForcedWinTrend(path="single-window", smoothing=0.5)
    trend.update(fw.GameForcedWinSummary(
        path="single-window", forced_win_turns=2, off_window_forced_turns=1, converted=1))
    trend.update(fw.GameForcedWinSummary(
        path="single-window", forced_win_turns=4, off_window_forced_turns=0, converted=4))
    snap = trend.snapshot()
    assert snap["path"] == "single-window"
    assert snap["n"] == 2
    assert snap["smoothing"] == 0.5
    assert 0.0 <= snap["off_window_forced_win_rate"] <= 1.0
    assert 0.0 <= snap["forced_win_conversion"] <= 1.0


def test_forced_win_trend_skips_games_with_no_forced_win():
    trend = fw.ForcedWinTrend(path="single-window", smoothing=0.5)
    trend.update(fw.GameForcedWinSummary(
        path="single-window", forced_win_turns=0, off_window_forced_turns=0, converted=0))
    # a game with no forced win does not inform the rate → n stays 0
    assert trend.snapshot()["n"] == 0


# ── structlog emission (carried even with the web dashboard OFF) ──────────────

def test_emit_forced_win_trend_logs_event_with_path_label():
    logger = Mock()
    trend = fw.ForcedWinTrend(path="single-window", smoothing=0.5)
    trend.update(fw.GameForcedWinSummary(
        path="single-window", forced_win_turns=1, off_window_forced_turns=0, converted=1))
    fw.emit_forced_win_trend(trend, logger=logger)
    logger.info.assert_called_once()
    args, kwargs = logger.info.call_args
    assert args[0] == "forced_win_trend"
    assert kwargs["path"] == "single-window"
    assert kwargs["n"] == 1
    assert "smoothing" in kwargs
    assert "off_window_forced_win_rate" in kwargs


# ── offline file reader (GameRecorder jsonl) ─────────────────────────────────

def test_analyze_replay_file_aggregates_trend(tmp_path):
    import json
    p = tmp_path / "games_test.jsonl"
    rec = {
        "moves": _winning_game_moves(),
        "outcome": "x_win",
        "game_length": len(_winning_game_moves()),
        "checkpoint_step": 30000,
    }
    p.write_text(json.dumps(rec) + "\n")
    trend = fw.analyze_replay_file(
        p, encoding="v6_live2", mover_side=1, smoothing=0.5, path="single-window",
    )
    snap = trend.snapshot()
    assert snap["path"] == "single-window"
    assert snap["n"] == 1
    assert snap["forced_win_conversion"] == 1.0
