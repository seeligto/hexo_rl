"""Unit tests for Q13 chain-length plane computation.

Validates the post-placement semantics: cell value = 1 + pos_run + neg_run for
both empty and own-stone cells, 0 for opponent cells, capped at 6. Window edges
are opaque (terminate runs). Covers the 12 cases enumerated in the plan file.
"""
from __future__ import annotations
import numpy as np
import pytest

from hexo_rl.env.game_state import (
    _CHAIN_CAP,
    _HEX_AXES,
    _chain_plane_for_axis,
    _compute_chain_planes,
    _shift_zero_pad,
)
from hexo_rl.utils.constants import BOARD_SIZE

# Axis plane indices in the 6-plane block.
AX0_CUR, AX0_OPP = 0, 1  # (1, 0) E/W
AX1_CUR, AX1_OPP = 2, 3  # (0, 1) NE/SW
AX2_CUR, AX2_OPP = 4, 5  # (1, -1) SE/NW


def _blank() -> tuple[np.ndarray, np.ndarray]:
    cur = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    opp = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    return cur, opp


def _set_cur(cur: np.ndarray, q: int, r: int) -> None:
    cur[q + 9, r + 9] = 1.0


def _set_opp(opp: np.ndarray, q: int, r: int) -> None:
    opp[q + 9, r + 9] = 1.0


def _at(plane: np.ndarray, q: int, r: int) -> int:
    return int(plane[q + 9, r + 9])


# ---------------------------------------------------------------------------
# Case 1 — empty board
# ---------------------------------------------------------------------------
def test_empty_board_all_zeros():
    cur, opp = _blank()
    planes = _compute_chain_planes(cur, opp)
    assert planes.shape == (6, BOARD_SIZE, BOARD_SIZE)
    assert planes.dtype == np.int8
    assert np.all(planes == 0)


# ---------------------------------------------------------------------------
# Case 2 — single current-player stone: value 1 on all 3 axes through that cell
# ---------------------------------------------------------------------------
def test_single_stone_value_one_on_all_axes():
    cur, opp = _blank()
    _set_cur(cur, 0, 0)
    planes = _compute_chain_planes(cur, opp)
    # Cur planes through the stone
    assert _at(planes[AX0_CUR], 0, 0) == 1
    assert _at(planes[AX1_CUR], 0, 0) == 1
    assert _at(planes[AX2_CUR], 0, 0) == 1
    # Opp planes should be empty everywhere
    assert np.all(planes[AX0_OPP] == 0)
    assert np.all(planes[AX1_OPP] == 0)
    assert np.all(planes[AX2_OPP] == 0)


# ---------------------------------------------------------------------------
# Case 3 — open-3 horizontal along axis0 (1,0)
# Stones X at (0,0), (1,0), (2,0). Empty flanks at (-1,0) and (3,0).
# Each stone should show 3, each flanking empty should show 4.
# ---------------------------------------------------------------------------
def test_open_three_axis0():
    cur, opp = _blank()
    for q in (0, 1, 2):
        _set_cur(cur, q, 0)
    planes = _compute_chain_planes(cur, opp)
    p = planes[AX0_CUR]
    assert _at(p, 0, 0) == 3
    assert _at(p, 1, 0) == 3
    assert _at(p, 2, 0) == 3
    assert _at(p, -1, 0) == 4
    assert _at(p, 3, 0) == 4
    # Perpendicular-axis planes through the stones see value 1 each.
    assert _at(planes[AX1_CUR], 1, 0) == 1
    assert _at(planes[AX2_CUR], 1, 0) == 1


# ---------------------------------------------------------------------------
# Case 4 — blocked-3 diagonal along axis1 (0,1)
# Stones at (0,0),(0,1),(0,2). Opponent at (0,3). Empty at (0,-1).
# Stones show 3, open flank (0,-1) shows 4, blocked side (0,3) is opp → 0.
# ---------------------------------------------------------------------------
def test_blocked_three_axis1():
    cur, opp = _blank()
    for r in (0, 1, 2):
        _set_cur(cur, 0, r)
    _set_opp(opp, 0, 3)
    planes = _compute_chain_planes(cur, opp)
    p = planes[AX1_CUR]
    assert _at(p, 0, 0) == 3
    assert _at(p, 0, 1) == 3
    assert _at(p, 0, 2) == 3
    assert _at(p, 0, -1) == 4  # open flank
    assert _at(p, 0, 3) == 0  # opponent cell, cur plane is zero there
    # Opponent plane should light up at (0,3): that opp stone has run length 1.
    assert _at(planes[AX1_OPP], 0, 3) == 1


# ---------------------------------------------------------------------------
# Case 5 — open-4 horizontal
# Stones 4,4,4,4; flanks 5,5.
# ---------------------------------------------------------------------------
def test_open_four_axis0():
    cur, opp = _blank()
    for q in range(4):
        _set_cur(cur, q, 0)
    planes = _compute_chain_planes(cur, opp)
    p = planes[AX0_CUR]
    for q in range(4):
        assert _at(p, q, 0) == 4
    assert _at(p, -1, 0) == 5
    assert _at(p, 4, 0) == 5


# ---------------------------------------------------------------------------
# Case 6 — double-3 fork (row-3 and col-3 sharing a point)
# Planes are per-axis independent; cross-axis does not bleed.
# ---------------------------------------------------------------------------
def test_double_three_fork_independent_axes():
    cur, opp = _blank()
    # Row-3 along axis0
    for q in (0, 1, 2):
        _set_cur(cur, q, 0)
    # Col-3 along axis1 (sharing (0,0))
    for r in (1, 2):
        _set_cur(cur, 0, r)
    planes = _compute_chain_planes(cur, opp)
    # Intersection (0,0) sees row-3 on axis0 and col-3 on axis1
    assert _at(planes[AX0_CUR], 0, 0) == 3
    assert _at(planes[AX1_CUR], 0, 0) == 3
    # Axis2 through (0,0) sees only the isolated stone
    assert _at(planes[AX2_CUR], 0, 0) == 1
    # Row-3 tail (2,0) sees only axis0
    assert _at(planes[AX0_CUR], 2, 0) == 3
    assert _at(planes[AX1_CUR], 2, 0) == 1


# ---------------------------------------------------------------------------
# Case 7 — window-edge clipping: run against the east edge truncates, no phantom
# ---------------------------------------------------------------------------
def test_window_edge_opaque_clipping():
    cur, opp = _blank()
    # Place 3 stones along axis0 flush against the east edge.
    # Axial coord q ∈ [-9, 9]; place at q=7,8,9, r=0.
    for q in (7, 8, 9):
        _set_cur(cur, q, 0)
    planes = _compute_chain_planes(cur, opp)
    p = planes[AX0_CUR]
    # All 3 stones should show 3 (no phantom extension beyond q=9).
    assert _at(p, 7, 0) == 3
    assert _at(p, 8, 0) == 3
    assert _at(p, 9, 0) == 3
    # The empty west flank at q=6 should see 4 (hypothetical placement extends run).
    assert _at(p, 6, 0) == 4


# ---------------------------------------------------------------------------
# Case 8 — opponent-blocked run: X X · Y X with Y=opp
# ---------------------------------------------------------------------------
def test_opponent_blocks_run_axis0():
    cur, opp = _blank()
    _set_cur(cur, 0, 0)
    _set_cur(cur, 1, 0)
    _set_opp(opp, 2, 0)
    _set_cur(cur, 3, 0)
    planes = _compute_chain_planes(cur, opp)
    p = planes[AX0_CUR]
    # Left cluster (0,0),(1,0): opaque block at (2,0) terminates.
    assert _at(p, 0, 0) == 2
    assert _at(p, 1, 0) == 2
    # Isolated right stone (3,0): both sides are opp/empty.
    assert _at(p, 3, 0) == 1


# ---------------------------------------------------------------------------
# Case 9 — XX_XXX: empty cell value = 6 (2 + 3 + 1), clipped at cap
# ---------------------------------------------------------------------------
def test_xx_empty_xxx_pattern():
    cur, opp = _blank()
    # Stones at q = 0, 1, (empty at 2), 3, 4, 5; all r=0
    for q in (0, 1, 3, 4, 5):
        _set_cur(cur, q, 0)
    planes = _compute_chain_planes(cur, opp)
    p = planes[AX0_CUR]
    # Empty cell between: 2 left + 3 right + 1 = 6.
    assert _at(p, 2, 0) == 6
    # Left XX: pos_run=1 blocked by empty, neg_run=0 or 1.
    assert _at(p, 0, 0) == 2  # pos_run=1, neg_run=0
    assert _at(p, 1, 0) == 2  # pos_run=0 (blocked by empty), neg_run=1
    # Right XXX stones.
    assert _at(p, 3, 0) == 3
    assert _at(p, 4, 0) == 3
    assert _at(p, 5, 0) == 3


# ---------------------------------------------------------------------------
# Case 10 — already-6 run → cap saturates correctly
# ---------------------------------------------------------------------------
def test_cap_at_six_along_axis0():
    cur, opp = _blank()
    for q in range(6):
        _set_cur(cur, q, 0)
    planes = _compute_chain_planes(cur, opp)
    p = planes[AX0_CUR]
    for q in range(6):
        assert _at(p, q, 0) == 6  # every stone in the 6-run sees 6


def test_cap_saturates_above_six():
    """A run of 7 stones still caps at 6 on every cell (no overflow)."""
    cur, opp = _blank()
    for q in range(-3, 4):  # 7 stones
        _set_cur(cur, q, 0)
    planes = _compute_chain_planes(cur, opp)
    p = planes[AX0_CUR]
    for q in range(-3, 4):
        assert _at(p, q, 0) == 6


# ---------------------------------------------------------------------------
# Case 11 — multi-axis overlap at intersection
# ---------------------------------------------------------------------------
def test_multi_axis_intersection():
    cur, opp = _blank()
    # Row-3 along axis0
    for q in (0, 1, 2):
        _set_cur(cur, q, 0)
    # Col-3 along axis1, disjoint except at (0,0)
    for r in (0, 1, 2):
        _set_cur(cur, 0, r)
    planes = _compute_chain_planes(cur, opp)
    assert _at(planes[AX0_CUR], 0, 0) == 3
    assert _at(planes[AX1_CUR], 0, 0) == 3
    assert _at(planes[AX2_CUR], 0, 0) == 1


# ---------------------------------------------------------------------------
# Case 12 — perspective swap: swapping (cur, opp) swaps the plane pairs
# ---------------------------------------------------------------------------
def test_perspective_swap():
    cur, opp = _blank()
    # Mixed position: some cur, some opp stones.
    for q in (0, 1, 2):
        _set_cur(cur, q, 0)
    for r in (0, 1):
        _set_opp(opp, 3, r)
    planes_ab = _compute_chain_planes(cur, opp)
    planes_ba = _compute_chain_planes(opp, cur)
    # Swapping arguments should swap each axis's cur/opp plane pair byte-exact.
    for axis_idx in range(3):
        cur_idx = 2 * axis_idx
        opp_idx = 2 * axis_idx + 1
        assert np.array_equal(planes_ab[cur_idx], planes_ba[opp_idx])
        assert np.array_equal(planes_ab[opp_idx], planes_ba[cur_idx])


# ---------------------------------------------------------------------------
# Ancillary: output dtype/shape and normalization range invariant
# ---------------------------------------------------------------------------
def test_output_dtype_shape_and_range():
    cur, opp = _blank()
    for q in range(-9, 10):
        _set_cur(cur, q, 0)  # very long run, triggers cap
    planes = _compute_chain_planes(cur, opp)
    assert planes.shape == (6, BOARD_SIZE, BOARD_SIZE)
    assert planes.dtype == np.int8
    assert planes.min() >= 0
    assert planes.max() <= _CHAIN_CAP


# ---------------------------------------------------------------------------
# Shift helper sanity: no wrap-around (would indicate accidental np.roll usage)
# ---------------------------------------------------------------------------
def test_shift_no_wraparound_axis0():
    arr = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
    arr[0, 0] = 1
    shifted_neg = _shift_zero_pad(arr, -1, 0)
    # Value at (0,0) should now be zero; (-1,0) is off-grid so no source.
    assert shifted_neg[0, 0] == 0
    # Nothing else should be set — no wrap to bottom row.
    assert shifted_neg.sum() == 0


def test_shift_positive_axis1():
    arr = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
    arr[5, 5] = 1
    shifted = _shift_zero_pad(arr, 0, 1)  # shift +1 along r (column)
    assert shifted[5, 6] == 1
    assert shifted[5, 5] == 0
    assert shifted.sum() == 1


def test_shift_diagonal_axis2():
    arr = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
    arr[5, 5] = 1
    shifted = _shift_zero_pad(arr, 1, -1)  # axis2 direction
    assert shifted[6, 4] == 1
    assert shifted.sum() == 1
