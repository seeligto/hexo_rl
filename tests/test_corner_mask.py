"""
Python-side smoke for the engine corner-mask toggle (§153 T2).

Verifies the flag round-trips and that `Board.to_tensor()` produces masked
stone planes at the corners when the toggle is on, and unmasked otherwise.
"""
from __future__ import annotations

import numpy as np
import pytest

import engine
from engine import Board


def _to_tensor_18planes(b: Board) -> np.ndarray:
    """Wrap `Board.to_tensor()` (flat list, 18 * 19 * 19) into a (18, 19, 19)
    numpy array."""
    flat = np.asarray(b.to_tensor(), dtype=np.float32)
    assert flat.shape == (18 * 19 * 19,)
    return flat.reshape(18, 19, 19)


@pytest.fixture(autouse=True)
def restore_corner_mask_flag():
    """Snapshot + restore the flag around each test so they are isolated."""
    before = engine.corner_mask_enabled()
    yield
    engine.set_corner_mask_enabled(before)


def test_corner_mask_default_off():
    assert engine.corner_mask_enabled() is False


def test_corner_mask_toggle_round_trip():
    assert engine.set_corner_mask_enabled(True) is False
    assert engine.corner_mask_enabled() is True
    assert engine.set_corner_mask_enabled(False) is True
    assert engine.corner_mask_enabled() is False


def test_to_tensor_corner_planes_change_with_flag():
    """Stones inside vs outside hex_dist > 9 from window centre — only
    the corner stones should toggle on the flag."""
    b = Board()
    # Place a small cluster near the origin so the window centre lands at
    # window-coord (9, 9) (board origin maps to window cell (9, 9)).
    b.apply_move(0, 0)   # cur P1 — 1st move
    b.apply_move(1, 0)   # opp P2
    b.apply_move(0, 1)   # opp P2 (compound rule)

    # With the mask off: each stone's window-frame cell carries 1.0 in the
    # owning player's plane. The cells (0,0), (1,0), (0,1) all sit AT or
    # near window centre — well inside the central hexagon — so their
    # values do NOT change under the mask.
    engine.set_corner_mask_enabled(False)
    t_off = _to_tensor_18planes(b)
    n_my_off = int((t_off[0] > 0).sum())
    n_opp_off = int((t_off[8] > 0).sum())

    engine.set_corner_mask_enabled(True)
    t_on = _to_tensor_18planes(b)
    n_my_on = int((t_on[0] > 0).sum())
    n_opp_on = int((t_on[8] > 0).sum())

    # All three stones are well inside the central hexagon — no change.
    assert n_my_off == n_my_on
    assert n_opp_off == n_opp_on


def test_to_tensor_corner_stone_zeroed_when_flag_on():
    """Place a stone far enough from origin that the window centre tracks
    the bbox centroid — the stone at the window's CORNER is then zeroed by
    the mask."""
    b = Board()
    # Large bbox: stones at (0, 0) and (10, 0) → centroid (5, 0); window
    # spans q ∈ [-4, 14] in board coords. The cell (0, 0) lands at window
    # cell (-4 + 9, 0 + 9) = (5, 9). hex_dist((5,9),(9,9)) = 4 → in mask.
    # Now place a stone at (-4, 0) — that lands at window cell
    # (-4 - 5 + 9, 0 + 9) = (0, 9). hex_dist((0,9),(9,9)) = 9 → ON the
    # mask boundary, kept. Push further to (-5, 0): wq=-1, OFF window.
    # Easier path: rely on the small test in state.rs for the exact
    # boundary; here just sanity-check that toggling the flag changes the
    # tensor for a board that places stones near a corner.
    b.apply_move(-9, -9)         # P1
    b.apply_move(-9 + 1, -9)     # P2
    b.apply_move(9, 9)           # P1
    b.apply_move(9, 9 - 1)       # P2

    engine.set_corner_mask_enabled(False)
    t_off = _to_tensor_18planes(b)

    engine.set_corner_mask_enabled(True)
    t_on = _to_tensor_18planes(b)

    # The diagonal extreme stones land in the parallelogram corners after
    # the window centres on the bbox centroid; flipping the mask must
    # change at least one stone-plane cell.
    diff = np.where(t_off != t_on)
    assert diff[0].size > 0, (
        "Expected the mask to change at least one cell for a board with "
        "stones near the parallelogram corners"
    )
    # Differences must come from stone planes 0 or 8 only — the mask
    # must NOT touch broadcast / history planes.
    plane_indices = set(int(p) for p in diff[0])
    assert plane_indices.issubset({0, 8}), (
        f"Mask leaked to non-stone planes: {plane_indices}"
    )
