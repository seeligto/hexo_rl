"""Unit tests for selfplay axis-distribution metric (§axis_dist)."""
from __future__ import annotations

import numpy as np
import pytest

from hexo_rl.training.axis_distribution import (
    AXIS_LABELS,
    _assign_colors,
    _AXES,
    compute_axis_fractions,
    compute_axis_fractions_from_states,
)


# ── _assign_colors ─────────────────────────────────────────────────────────────

def test_assign_colors_single_move():
    result = _assign_colors([(0, 0)])
    assert result == {(0, 0): 1}  # ply 0 → P1


def test_assign_colors_alternating():
    # ply 0 → P1; ply 1,2 → P2; ply 3,4 → P1; ply 5,6 → P2
    moves = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0)]
    colors = _assign_colors(moves)
    assert colors[(0, 0)] == 1   # ply 0
    assert colors[(1, 0)] == -1  # ply 1, compound_idx=0 (even) → P2
    assert colors[(2, 0)] == -1  # ply 2, compound_idx=0 (even) → P2
    assert colors[(3, 0)] == 1   # ply 3, compound_idx=1 (odd) → P1
    assert colors[(4, 0)] == 1   # ply 4, compound_idx=1 (odd) → P1
    assert colors[(5, 0)] == -1  # ply 5, compound_idx=2 (even) → P2
    assert colors[(6, 0)] == -1  # ply 6, compound_idx=2 (even) → P2


# ── compute_axis_fractions ─────────────────────────────────────────────────────

def test_empty_games_returns_zeros():
    result = compute_axis_fractions([])
    assert result["axis_q"] == 0.0
    assert result["axis_r"] == 0.0
    assert result["axis_s"] == 0.0
    assert result["axis_max"] in AXIS_LABELS


def test_single_stone_game_returns_zeros():
    result = compute_axis_fractions([[(0, 0)]])
    assert result["axis_q"] == 0.0
    assert result["axis_r"] == 0.0
    assert result["axis_s"] == 0.0


def test_all_same_color_q_axis():
    # 5 P1 stones in a row along q axis (no opponent stones adjacent)
    # ply 0 → P1, ply 1,2 → P2 (isolated), ply 3,4 → P1
    # Build a game where P1 occupies q=0..4 on the q axis (r=0)
    # and P2 stones are far away.
    # P1: plies 0, 3, 4; P2: plies 1, 2
    moves = [(0, 0), (0, 10), (1, 10), (1, 0), (2, 0)]
    colors = _assign_colors(moves)
    # P1 stones: (0,0), (1,0), (2,0) — three in a row along q axis
    # P2 stones: (0,10), (1,10) — far away, not adjacent to P1 stones
    result = compute_axis_fractions([moves])
    # Along q axis (dq=1,dr=0): pairs (0,0)-(1,0) and (1,0)-(2,0) → same color (P1)
    # P2 pair: (0,10)-(1,10) → same color (P2)
    # All 3 adjacent pairs are same color → axis_q = 1.0
    assert result["axis_q"] == pytest.approx(1.0)
    assert result["axis_max"] == "axis_q"


def test_perfectly_interleaved_q_axis():
    # P1 and P2 alternate along q axis: P1, P2, P1, P2
    # plies 0 (P1), 1 (P2), 2 (P2, but at ply=2 compound=0, still P2), ...
    # Actually let's just build a known coloring directly.
    # Build moves so P1 is at q=0,2 and P2 at q=1,3 along r=0
    # ply 0 → P1 at (0,0)
    # ply 1,2 → P2 at (1,0), (2,0)  [P2 places 2 stones]
    # Wait, that puts P2 at (1,0) AND (2,0), not alternating.
    # Let's place P2 at (1,0) and somewhere else, P1 at (2,0)
    moves = [
        (0, 0),   # ply 0 → P1
        (1, 0),   # ply 1 → P2 (compound_idx=0 even)
        (10, 10), # ply 2 → P2
        (2, 0),   # ply 3 → P1 (compound_idx=1 odd)
        (20, 20), # ply 4 → P1
    ]
    # Along q axis: pair (0,0)-(1,0): P1-P2 → different; pair (1,0)-(2,0): P2-P1 → different
    result = compute_axis_fractions([moves])
    assert result["axis_q"] == pytest.approx(0.0)


def test_axis_max_label_correct():
    # Build a game with all stones along r axis (same color)
    # P1 at (0,0), (0,1), (0,2); P2 far away
    moves = [
        (0, 0),    # ply 0 → P1
        (10, 10),  # ply 1 → P2
        (20, 20),  # ply 2 → P2
        (0, 1),    # ply 3 → P1
        (0, 2),    # ply 4 → P1
    ]
    result = compute_axis_fractions([moves])
    # r axis pairs: (0,0)-(0,1) same P1; (0,1)-(0,2) same P1 → axis_r = 1.0
    assert result["axis_r"] == pytest.approx(1.0)
    assert result["axis_max"] == "axis_r"


def test_multiple_games_aggregated():
    # Game 1: two P1 stones adjacent on q axis → 1 same-color pair
    # Game 2: P1 and P2 adjacent on q axis → 0 same-color pairs
    game1 = [(0, 0), (10, 10), (20, 20), (1, 0), (2, 0)]
    game2 = [(0, 0), (1, 0), (10, 10), (2, 0), (20, 20)]

    colors2 = _assign_colors(game2)
    # In game2: (0,0)=P1, (1,0)=P2, (2,0)=P1 along q-axis
    # pairs along q: (0,0)-(1,0) → P1-P2 (diff); (1,0)-(2,0) → P2-P1 (diff)
    # So game2 contributes 0 same, 2 total on q axis
    # game1: (0,0)=P1, (1,0)=P1, (2,0)=P1 along q-axis → 2 same, 2 total
    result = compute_axis_fractions([game1, game2])
    # Combined: 2 same / 4 total = 0.5
    assert result["axis_q"] == pytest.approx(0.5)


def test_s_axis_pairs():
    # s-axis (dq=1, dr=-1): stones at (0,1), (1,0) are adjacent
    moves = [
        (0, 1),    # ply 0 → P1
        (10, 10),  # ply 1 → P2
        (20, 20),  # ply 2 → P2
        (1, 0),    # ply 3 → P1
        (2, -1),   # ply 4 → P1
    ]
    result = compute_axis_fractions([moves])
    # s-axis (1,-1): (0,1)→(1,0) same P1; (1,0)→(2,-1) same P1
    assert result["axis_s"] == pytest.approx(1.0)
    assert result["axis_max"] == "axis_s"


# ── compute_axis_fractions_from_states ────────────────────────────────────────

def _make_state(cur_positions: list[tuple[int, int]], opp_positions: list[tuple[int, int]], H: int = 19) -> np.ndarray:
    """Build a (18, H, H) state with stones at specified grid positions."""
    state = np.zeros((18, H, H), dtype=np.float32)
    for (i, j) in cur_positions:
        state[0, i, j] = 1.0
    for (i, j) in opp_positions:
        state[8, i, j] = 1.0
    return state


def test_from_states_empty_board():
    states = np.zeros((5, 18, 19, 19), dtype=np.float32)
    result = compute_axis_fractions_from_states(states)
    assert result["axis_q"] == 0.0
    assert result["axis_r"] == 0.0
    assert result["axis_s"] == 0.0


def test_from_states_same_color_q_row():
    # Two current-player stones adjacent along q axis (row 0, cols 0 and 1)
    # In state space: q = row index, r = col index
    # axis q (dq=1, dr=0) → adjacent in row direction
    state = _make_state(cur_positions=[(0, 0), (1, 0)], opp_positions=[])
    states = state[np.newaxis]  # (1, 18, 19, 19)
    result = compute_axis_fractions_from_states(states)
    # Only pair is along q axis (rows 0 and 1, col 0) — same color
    assert result["axis_q"] == pytest.approx(1.0)
    assert result["axis_r"] == 0.0  # no r-axis neighbors
    assert result["axis_s"] == 0.0  # (0,0)-(1,-1) = (1,-1) → out of bounds at col-1


def test_from_states_different_color_q_pair():
    state = _make_state(cur_positions=[(0, 0)], opp_positions=[(1, 0)])
    states = state[np.newaxis]
    result = compute_axis_fractions_from_states(states)
    assert result["axis_q"] == pytest.approx(0.0)


def test_from_states_multiple_positions():
    # Position 1: two same-color q-pairs; Position 2: one diff-color q-pair
    s1 = _make_state(cur_positions=[(0, 0), (1, 0)], opp_positions=[])
    s2 = _make_state(cur_positions=[(0, 0)], opp_positions=[(1, 0)])
    states = np.stack([s1, s2], axis=0)  # (2, 18, 19, 19)
    result = compute_axis_fractions_from_states(states)
    # q-axis: 1 same pair + 0 same pairs / 1 total + 1 total = 0.5
    assert result["axis_q"] == pytest.approx(0.5)


def test_from_states_r_axis():
    # Stones at (0,0) and (0,1) → adjacent along r axis (dr=1)
    state = _make_state(cur_positions=[(0, 0), (0, 1)], opp_positions=[])
    states = state[np.newaxis]
    result = compute_axis_fractions_from_states(states)
    assert result["axis_r"] == pytest.approx(1.0)
    assert result["axis_q"] == 0.0


def test_from_states_s_axis():
    # Stones at (0,1) and (1,0) → adjacent along s axis (dq=1, dr=-1)
    state = _make_state(cur_positions=[(0, 1), (1, 0)], opp_positions=[])
    states = state[np.newaxis]
    result = compute_axis_fractions_from_states(states)
    assert result["axis_s"] == pytest.approx(1.0)


def test_from_states_float16_input():
    state = _make_state(cur_positions=[(0, 0), (1, 0)], opp_positions=[])
    states = state[np.newaxis].astype(np.float16)
    result = compute_axis_fractions_from_states(states)
    assert result["axis_q"] == pytest.approx(1.0)


def test_axis_labels_exhaustive():
    assert len(AXIS_LABELS) == 3
    assert "axis_q" in AXIS_LABELS
    assert "axis_r" in AXIS_LABELS
    assert "axis_s" in AXIS_LABELS
    assert len(_AXES) == 3
