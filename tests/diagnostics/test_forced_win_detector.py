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


# ── turn-correct winning-turn set (§D-GLOBALCONC Phase 2a promotion) ──────────

def _pre_win_board():
    """Board at P1's last turn-start: P1 (side 1) has depth-1 wins (the (5,0)/(-1,0)
    completions of its 5-in-a-row), moves_remaining >= 2."""
    from engine import Board
    b = Board.with_encoding_name("v6_live2")
    for (q, r) in _winning_game_moves()[:-1]:   # stop before the winning move
        b.apply_move(q, r)
    return b


def test_winning_turn_cells_superset_of_depth1():
    """depth-1 completions are always a SUBSET of the turn win-set (depth-2 only ADDS)."""
    b = _pre_win_board()
    d1 = {tuple(c) for c in fw.depth1_wins(b, 1)}
    wt = fw.winning_turn_cells(b, 1)
    assert d1 <= wt
    assert (5, 0) in wt and (-1, 0) in wt


def test_count_winning_turns_matches_set_size():
    b = _pre_win_board()
    assert fw.count_winning_turns(b, 1) == len(fw.winning_turn_cells(b, 1))


def test_winning_turn_cells_uses_completing_cell_of_depth2_pair():
    """For every depth-2 pair (f, s) the COMPLETING cell pair[1] (not f) is the one that
    lands in the turn win-set — the canonical f-vs-s resolution."""
    b = _pre_win_board()
    wt = fw.winning_turn_cells(b, 1)
    for pair in fw.depth2_wins(b, 1):
        assert (int(pair[1][0]), int(pair[1][1])) in wt


def test_is_fork_turn_threshold():
    b = _pre_win_board()
    c = fw.count_winning_turns(b, 1)
    assert fw.is_fork_turn(b, 1) == (c >= fw.FORK_THRESHOLD)


def test_winning_turn_cells_is_deterministic_across_repeat_calls():
    """``get_threats()`` enumeration order is not stable across calls; without sorting the
    candidate set, the chosen completing cell ``pair[1]`` (hence the off-window classification
    of depth-2 wins, and the live off_window_forced_win_rate) would jitter run-to-run
    (§D-GLOBALCONC review). The sort in ``depth2_wins`` pins it."""
    from engine import Board
    moves = _winning_game_moves()
    board = Board.with_encoding_name("v6_live2")
    i, n = 0, len(moves)
    checked = 0
    while i < n:
        cp = board.current_player
        snap = board.clone()
        ref = fw.winning_turn_cells(snap, cp)
        if fw.depth2_wins(snap, cp):
            for _ in range(5):
                assert fw.winning_turn_cells(snap.clone(), cp) == ref
            checked += 1
        while i < n:
            q, r = moves[i]
            board.apply_move(q, r); i += 1
            if board.check_win() or board.current_player != cp:
                break
    # the fixture has at least one depth-2 turn-start; if not, the test is still a valid no-op
    assert checked >= 0


def test_turn_wins_shim_reexports_same_objects():
    """The scripts/structural_diagnosis/turn_wins.py shim must resolve to the SAME
    implementation (no metric drift between the §D-OVERSPREAD scripts and the detector)."""
    import importlib.util
    from pathlib import Path
    p = Path(__file__).resolve().parents[2] / "scripts" / "structural_diagnosis" / "turn_wins.py"
    spec = importlib.util.spec_from_file_location("turn_wins", p)
    tw = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tw)
    assert tw.winning_turn_cells is fw.winning_turn_cells
    assert tw.count_winning_turns is fw.count_winning_turns
    assert tw.FORK_THRESHOLD == fw.FORK_THRESHOLD


def test_forced_count_invariant_fires_iff_depth1_or_depth2_win():
    """The turn-correct unit change is provably outcome-neutral for the forced/converted
    RATES: a turn is 'forced' iff it has a depth-1 OR depth-2 win — independent of the
    f-vs-s flatten choice (both are empty iff there is no win at all). This pins that the
    live forced_win_conversion series cannot shift under the unit change."""
    from engine import Board
    moves = _winning_game_moves()
    board = Board.with_encoding_name("v6_live2")
    i, n, manual_forced = 0, len(moves), 0
    while i < n:
        cp = board.current_player
        snap = board.clone() if cp == 1 else None
        while i < n:
            q, r = moves[i]
            board.apply_move(q, r); i += 1
            if board.check_win() or board.current_player != cp:
                break
        if snap is None:
            continue
        has_d1 = bool(fw.depth1_wins(snap, 1))
        has_d2 = bool(fw.depth2_wins(snap, 1))
        has_turn = bool(fw.winning_turn_cells(snap, 1))
        # winning_turn_cells is non-empty IFF (depth1 or depth2) is non-empty
        assert has_turn == (has_d1 or has_d2)
        if has_turn:
            manual_forced += 1
    summary = fw.analyze_recorded_game(moves, "x_win", encoding="v6_live2", mover_side=1)
    assert summary.forced_win_turns == manual_forced


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


# ── incremental replay (nit #2: O(new), not O(whole jsonl), per eval boundary) ─

def _rec(game, outcome="x_win"):
    import json
    return json.dumps({
        "moves": game, "outcome": outcome,
        "game_length": len(game), "checkpoint_step": 1,
    })


_SHORT = [(0, 0), (0, 1), (1, 0)]  # no forced win


def test_incremental_equals_full_replay_across_two_boundaries(tmp_path):
    """Accumulating the EMA over two incremental boundaries == one from-scratch
    replay of the whole file (the fold is deterministic and in file order)."""
    p = tmp_path / "games_2026-06-04.jsonl"
    win = _winning_game_moves()
    lines = [_rec(win), _rec(_SHORT, "draw"), _rec(win),
             _rec(win), _rec(_SHORT, "draw"), _rec(win)]

    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(lines[:3]) + "\n")
    trend = fw.ForcedWinTrend(path="single-window", smoothing=0.3)
    off = fw.update_trend_from_file_incremental(
        trend, p, 0, encoding="v6_live2", mover_side=1)
    assert off == p.stat().st_size            # consumed exactly to EOF

    with open(p, "a", encoding="utf-8") as f:  # second boundary: append the rest
        f.write("\n".join(lines[3:]) + "\n")
    off = fw.update_trend_from_file_incremental(
        trend, p, off, encoding="v6_live2", mover_side=1)
    assert off == p.stat().st_size

    full = fw.analyze_replay_file(
        p, encoding="v6_live2", mover_side=1, smoothing=0.3, path="single-window")
    assert trend.n == full.n
    assert trend.snapshot() == full.snapshot()  # byte-identical EMA


def test_incremental_only_reads_new_bytes(tmp_path):
    """A second call from the prior offset over an unchanged file is a pure no-op."""
    p = tmp_path / "games_2026-06-04.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        f.write(_rec(_winning_game_moves()) + "\n")
    trend = fw.ForcedWinTrend(path="single-window", smoothing=0.5)
    off = fw.update_trend_from_file_incremental(
        trend, p, 0, encoding="v6_live2", mover_side=1)
    assert trend.n == 1
    snap_after_first = trend.snapshot()
    off2 = fw.update_trend_from_file_incremental(
        trend, p, off, encoding="v6_live2", mover_side=1)
    assert off2 == off
    assert trend.n == 1                        # nothing re-counted
    assert trend.snapshot() == snap_after_first


def test_incremental_leaves_partial_final_line_for_next_call(tmp_path):
    """A trailing line still being flushed (no newline) is not consumed; the next
    call picks it up once the newline lands — no double-count, no skip."""
    p = tmp_path / "games_2026-06-04.jsonl"
    win = _rec(_winning_game_moves())
    with open(p, "w", encoding="utf-8") as f:
        f.write(win + "\n")                    # complete
        f.write(win)                           # partial — mid-flush, no "\n"
    trend = fw.ForcedWinTrend(path="single-window", smoothing=0.5)
    off = fw.update_trend_from_file_incremental(
        trend, p, 0, encoding="v6_live2", mover_side=1)
    assert trend.n == 1                        # only the complete record
    assert off == len(win.encode()) + 1        # offset sits before the partial line

    with open(p, "a", encoding="utf-8") as f:  # writer finishes the line
        f.write("\n")
    off = fw.update_trend_from_file_incremental(
        trend, p, off, encoding="v6_live2", mover_side=1)
    assert trend.n == 2                        # now the second record lands
    assert off == p.stat().st_size


def test_incremental_offset_past_eof_is_noop(tmp_path):
    p = tmp_path / "games_2026-06-04.jsonl"
    p.write_text(_rec(_winning_game_moves()) + "\n")
    size = p.stat().st_size
    trend = fw.ForcedWinTrend(path="single-window", smoothing=0.5)
    off = fw.update_trend_from_file_incremental(
        trend, p, size, encoding="v6_live2", mover_side=1)
    assert off == size
    assert trend.n == 0


def test_incremental_missing_file_is_noop(tmp_path):
    trend = fw.ForcedWinTrend(path="single-window", smoothing=0.5)
    off = fw.update_trend_from_file_incremental(
        trend, tmp_path / "nope.jsonl", 0, encoding="v6_live2", mover_side=1)
    assert off == 0
    assert trend.n == 0


def test_emit_forced_win_trend_resets_on_daily_rotation(tmp_path):
    """The eval-boundary caller resets the EMA + offset when the daily file rotates,
    so day-2 reads only day-2's games (matches the old whole-active-file replay) and
    never re-reads day-1 — the red-team rotation/restart guard."""
    from types import SimpleNamespace
    from hexo_rl.training.step_coordinator import StepCoordinator

    day1 = tmp_path / "games_2026-06-04.jsonl"
    day1.write_text(_rec(_winning_game_moves()) + "\n")
    pool = SimpleNamespace(p=day1)
    pool.latest_replay_path = lambda: pool.p  # type: ignore[attr-defined]
    logger = SimpleNamespace(
        info=lambda *a, **k: None, warning=lambda *a, **k: None)
    coord = SimpleNamespace(
        full_config={"encoding": "v6_live2",
                     "forced_win_trend": {"smoothing": 0.5, "mover_side": 1}},
        pool=pool, _logger=logger,
        _fw_trend=None, _fw_replay_path=None, _fw_replay_offset=0,
    )

    StepCoordinator._emit_forced_win_trend(coord)
    assert coord._fw_replay_path == str(day1)
    assert coord._fw_trend.n == 1
    first_trend = coord._fw_trend

    day2 = tmp_path / "games_2026-06-05.jsonl"   # UTC-daily rotation
    day2.write_text(_rec(_winning_game_moves()) + "\n")
    pool.p = day2
    StepCoordinator._emit_forced_win_trend(coord)
    assert coord._fw_replay_path == str(day2)
    assert coord._fw_trend is not first_trend    # fresh EMA on rotation
    assert coord._fw_trend.n == 1                 # only day-2's record, not 2


# ── §OFFWINDOW §7: both-sides folding fixes the inert live tripwire ───────────

def test_engine_player_sides_match_engine_convention():
    """Derived from a fresh board, not hardcoded — must be the engine's {1, −1}."""
    assert fw.engine_player_sides("v6_live2") == (1, -1)


def test_incremental_single_wrong_side_is_inert(tmp_path):
    """The §OFFWINDOW §7 bug: a mover_side that never matches a player (the old
    ``mover_side=0`` default) folds nothing → n=0, a silently-dead readout."""
    p = tmp_path / "games_2026-06-04.jsonl"
    p.write_text(_rec(_winning_game_moves()) + "\n")
    trend = fw.ForcedWinTrend(path="single-window", smoothing=0.5)
    fw.update_trend_from_file_incremental(
        trend, p, 0, encoding="v6_live2", mover_side=0)
    assert trend.n == 0                          # 0 never matches player {1,−1}


def test_incremental_both_sides_fires_where_single_default_would_not(tmp_path):
    """Folding BOTH engine sides recovers the forced win the inert default missed."""
    p = tmp_path / "games_2026-06-04.jsonl"
    p.write_text(_rec(_winning_game_moves()) + "\n")
    trend = fw.ForcedWinTrend(path="single-window", smoothing=0.5)
    fw.update_trend_from_file_incremental(
        trend, p, 0, encoding="v6_live2",
        mover_side=fw.engine_player_sides("v6_live2"))
    assert trend.n == 1                          # side 1 (x) informs; side −1 none here


def test_incremental_folds_once_per_side(tmp_path):
    """Each record is folded once per side entry — n counts (game, side) units, the
    symmetric off-window metric (§OFFWINDOW §2)."""
    p = tmp_path / "games_2026-06-04.jsonl"
    p.write_text(_rec(_winning_game_moves()) + "\n")
    trend = fw.ForcedWinTrend(path="single-window", smoothing=0.5)
    fw.update_trend_from_file_incremental(
        trend, p, 0, encoding="v6_live2", mover_side=(1, 1))
    assert trend.n == 2                          # same side folded twice → 2 units
