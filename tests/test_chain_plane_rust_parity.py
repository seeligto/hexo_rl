"""F2 guard — Python `_compute_chain_planes` vs Rust `engine.compute_chain_planes`.

Two independent implementations of the Q13 chain-plane computation exist:
  - Python: `hexo_rl/env/game_state.py:_compute_chain_planes` (numpy-vectorised).
  - Rust:   `engine/src/board/state.rs:encode_chain_planes` (pure loop over
            axes + axial cells, exposed as `engine.compute_chain_planes`).

Both must produce byte-exact outputs for every input; any drift is a
silent training-data bug since self-play (Rust) and pretrain / Python path
(`GameState.to_tensor()`) both feed into the same replay buffer.

Q22 (chain-plane promotion to Rust) will delete the Python path. Until then
this test is the regression guard that lets the two coexist.
"""
from __future__ import annotations

import numpy as np
import pytest

import engine
from hexo_rl.env.game_state import _CHAIN_CAP, _compute_chain_planes
from hexo_rl.utils.constants import BOARD_SIZE

HALF = (BOARD_SIZE - 1) // 2  # 9


def _empty() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32),
        np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32),
    )


def _set(mask: np.ndarray, q: int, r: int) -> None:
    mask[q + HALF, r + HALF] = 1.0


def _single_stone_origin() -> tuple[np.ndarray, np.ndarray]:
    cur, opp = _empty()
    _set(cur, 0, 0)
    return cur, opp


def _open_three_axis0() -> tuple[np.ndarray, np.ndarray]:
    cur, opp = _empty()
    for q in (-1, 0, 1):
        _set(cur, q, 0)
    return cur, opp


def _open_four_axis1() -> tuple[np.ndarray, np.ndarray]:
    cur, opp = _empty()
    for r in (-1, 0, 1, 2):
        _set(cur, 0, r)
    return cur, opp


def _blocked_three_axis0() -> tuple[np.ndarray, np.ndarray]:
    cur, opp = _empty()
    for q in (0, 1, 2):
        _set(cur, q, 0)
    _set(opp, 3, 0)
    _set(opp, -1, 0)
    return cur, opp


def _xx_gap_xxx_axis0() -> tuple[np.ndarray, np.ndarray]:
    cur, opp = _empty()
    for q in (0, 1, 3, 4, 5):
        _set(cur, q, 0)
    return cur, opp


def _double_three_fork() -> tuple[np.ndarray, np.ndarray]:
    cur, opp = _empty()
    for q in (-1, 0, 1):
        _set(cur, q, 0)
    for r in (-1, 1):
        _set(cur, 0, r)
    return cur, opp


def _triple_axis_intersection() -> tuple[np.ndarray, np.ndarray]:
    cur, opp = _empty()
    _set(cur, 0, 0)
    _set(cur, 1, 0)   # axis0
    _set(cur, 0, 1)   # axis1
    _set(cur, 1, -1)  # axis2
    return cur, opp


def _broken_four_with_gap() -> tuple[np.ndarray, np.ndarray]:
    """XX.X.XX along axis0 — runs must NOT combine across a second gap."""
    cur, opp = _empty()
    for q in (0, 1, 3, 5, 6):
        _set(cur, q, 0)
    return cur, opp


def _cap_saturation() -> tuple[np.ndarray, np.ndarray]:
    cur, opp = _empty()
    for q in range(-3, 4):  # 7 stones along axis 0
        _set(cur, q, 0)
    return cur, opp


def _perspective_opponent_run() -> tuple[np.ndarray, np.ndarray]:
    cur, opp = _empty()
    for r in (-1, 0, 1):
        _set(opp, 2, r)
    _set(cur, 0, 0)
    return cur, opp


def _window_edge_axis0() -> tuple[np.ndarray, np.ndarray]:
    cur, opp = _empty()
    for q in (7, 8, 9):
        _set(cur, q, 0)
    return cur, opp


def _window_edge_axis1_neg() -> tuple[np.ndarray, np.ndarray]:
    cur, opp = _empty()
    for r in (-9, -8, -7):
        _set(cur, 0, r)
    return cur, opp


def _mid_game_asymmetric() -> tuple[np.ndarray, np.ndarray]:
    cur, opp = _empty()
    for q, r in [(0, 0), (1, 0), (2, 0), (3, -1), (2, 1), (-1, -1), (-2, -1)]:
        _set(cur, q, r)
    for q, r in [(4, 0), (4, -1), (-3, 0), (0, 2), (0, -2), (-1, 2)]:
        _set(opp, q, r)
    return cur, opp


def _mid_game_dense() -> tuple[np.ndarray, np.ndarray]:
    cur, opp = _empty()
    cur_cells = [(0, 0), (1, 0), (2, 0), (3, 0), (-1, 1), (0, 1), (1, 1),
                 (2, -1), (3, -2), (-2, 2), (-1, 2), (0, 2), (4, -3), (5, -3)]
    opp_cells = [(0, -1), (1, -1), (2, -2), (-1, 0), (-2, 0), (-3, 1),
                 (3, 1), (4, 0), (2, 2), (1, 2), (-1, 3), (0, 3)]
    for q, r in cur_cells:
        _set(cur, q, r)
    for q, r in opp_cells:
        _set(opp, q, r)
    return cur, opp


def _two_axes_blocked_by_opp() -> tuple[np.ndarray, np.ndarray]:
    cur, opp = _empty()
    for q in (-2, -1, 0):
        _set(cur, q, 0)
    _set(opp, 1, 0)
    for r in (0, 1, 2):
        _set(cur, -2, r)
    return cur, opp


def _sparse_multi_colony() -> tuple[np.ndarray, np.ndarray]:
    cur, opp = _empty()
    for q, r in [(-6, -6), (-5, -6), (-4, -6)]:
        _set(cur, q, r)
    for q, r in [(5, 5), (6, 5), (7, 5)]:
        _set(cur, q, r)
    for q, r in [(-6, 5), (-5, 5)]:
        _set(opp, q, r)
    for q, r in [(5, -6), (5, -5)]:
        _set(opp, q, r)
    return cur, opp


def _near_five_almost_win() -> tuple[np.ndarray, np.ndarray]:
    cur, opp = _empty()
    for q in (-2, -1, 0, 1, 2):
        _set(cur, q, 0)
    _set(opp, 3, 0)
    return cur, opp


def _axis2_only_run() -> tuple[np.ndarray, np.ndarray]:
    cur, opp = _empty()
    for k in (-2, -1, 0, 1):
        _set(cur, k, -k)  # axis 2 direction (1, -1)
    return cur, opp


def _mixed_all_three_axes_runs() -> tuple[np.ndarray, np.ndarray]:
    cur, opp = _empty()
    for q in (-1, 0, 1):
        _set(cur, q, 0)
    for r in (2, 3, 4):
        _set(cur, 0, r)
    for k in (-4, -3, -2):
        _set(cur, k, -k + 3)
    return cur, opp


def _opp_dominant_position() -> tuple[np.ndarray, np.ndarray]:
    cur, opp = _empty()
    for q in (-2, -1, 0, 1, 2):
        _set(opp, q, 0)
    for r in (-1, 0, 1):
        _set(opp, 3, r)
    for q, r in [(-3, 0), (-3, 1)]:
        _set(cur, q, r)
    return cur, opp


POSITIONS = [
    ("empty_board", _empty),
    ("single_stone_origin", _single_stone_origin),
    ("open_three_axis0", _open_three_axis0),
    ("open_four_axis1", _open_four_axis1),
    ("blocked_three_axis0", _blocked_three_axis0),
    ("xx_gap_xxx_axis0", _xx_gap_xxx_axis0),
    ("double_three_fork", _double_three_fork),
    ("triple_axis_intersection", _triple_axis_intersection),
    ("broken_four_with_gap", _broken_four_with_gap),
    ("cap_saturation", _cap_saturation),
    ("perspective_opponent_run", _perspective_opponent_run),
    ("window_edge_axis0", _window_edge_axis0),
    ("window_edge_axis1_neg", _window_edge_axis1_neg),
    ("mid_game_asymmetric", _mid_game_asymmetric),
    ("mid_game_dense", _mid_game_dense),
    ("two_axes_blocked_by_opp", _two_axes_blocked_by_opp),
    ("sparse_multi_colony", _sparse_multi_colony),
    ("near_five_almost_win", _near_five_almost_win),
    ("axis2_only_run", _axis2_only_run),
    ("mixed_all_three_axes_runs", _mixed_all_three_axes_runs),
    ("opp_dominant_position", _opp_dominant_position),
]


def _python_reference(cur: np.ndarray, opp: np.ndarray) -> np.ndarray:
    """Python path: int8 planes normalised by /CHAIN_CAP, as float32."""
    planes = _compute_chain_planes(cur, opp)  # int8 (6, 19, 19)
    return planes.astype(np.float32) / float(_CHAIN_CAP)


@pytest.mark.parametrize("name,pos_fn", POSITIONS)
def test_compute_chain_planes_python_rust_byte_exact(name, pos_fn):
    cur, opp = pos_fn()
    expected = _python_reference(cur, opp)
    actual = engine.compute_chain_planes(cur, opp)
    assert expected.shape == actual.shape == (6, BOARD_SIZE, BOARD_SIZE)
    assert expected.dtype == actual.dtype == np.float32
    if not np.array_equal(expected, actual):
        diff = np.where(expected != actual)
        first = tuple(int(x[0]) for x in diff)
        raise AssertionError(
            f"[{name}] Rust vs Python chain-plane divergence at plane/row/col "
            f"{first}: py={float(expected[first]):.6f} rs={float(actual[first]):.6f} "
            f"(total differing cells: {len(diff[0])})"
        )
