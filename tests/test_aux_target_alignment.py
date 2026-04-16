"""ReplayBuffer per-row aux target alignment (A1 fix verification).

These tests guard against regressions in the per-row ownership + winning_line
columns added by Rust commit faafc43 and threaded through Python in §85.
"""
from __future__ import annotations

import numpy as np
import pytest

from engine import ReplayBuffer

CHANNELS   = 18
BOARD_SIZE = 19
N_ACTIONS  = BOARD_SIZE * BOARD_SIZE + 1   # 362
AUX_STRIDE = BOARD_SIZE * BOARD_SIZE       # 361


def _make_state() -> np.ndarray:
    return np.zeros((CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float16)


def _make_chain() -> np.ndarray:
    return np.zeros((6, BOARD_SIZE, BOARD_SIZE), dtype=np.float16)


def _make_policy() -> np.ndarray:
    p = np.zeros(N_ACTIONS, dtype=np.float32)
    p[0] = 1.0
    return p


def test_aux_target_round_trip_no_augment():
    """augment=False: u8 aux planes round-trip exactly via sample_batch."""
    buf = ReplayBuffer(capacity=16)

    own = np.ones(AUX_STRIDE, dtype=np.uint8)            # all empty
    own[[10, 11, 12, 13, 14, 15]] = 2                    # six P1 stones
    wl = np.zeros(AUX_STRIDE, dtype=np.uint8)
    wl[[10, 11, 12, 13, 14, 15]] = 1                     # six winning-line cells

    buf.push(_make_state(), _make_chain(), _make_policy(), 1.0, own, wl)
    _s, _chain, _p, _o, own_b, wl_b = buf.sample_batch(1, augment=False)

    assert own_b.shape == (1, BOARD_SIZE, BOARD_SIZE)
    assert wl_b.shape  == (1, BOARD_SIZE, BOARD_SIZE)
    assert own_b.dtype == np.uint8
    assert wl_b.dtype  == np.uint8

    # 6 P1 stones (encoding 2), 0 P2 stones (encoding 0), rest empty (encoding 1).
    assert int((own_b == 2).sum()) == 6
    assert int((own_b == 0).sum()) == 0
    assert int((own_b == 1).sum()) == AUX_STRIDE - 6

    # Winning line: exactly 6 ones, rest zeros.
    assert int(wl_b.sum()) == 6


def test_aux_target_augmentation_equivariance():
    """augment=True: cell counts are sym-invariant across all 12 hex symmetries."""
    buf = ReplayBuffer(capacity=16)

    own = np.ones(AUX_STRIDE, dtype=np.uint8)
    # Six P1 stones in the centre region — guaranteed to stay in-window under all syms.
    centre_idxs = [9 * BOARD_SIZE + c for c in range(7, 13)]
    own[centre_idxs] = 2
    wl = np.zeros(AUX_STRIDE, dtype=np.uint8)
    wl[centre_idxs] = 1

    buf.push(_make_state(), _make_chain(), _make_policy(), 1.0, own, wl)

    # Sample many times under augment=True; with one row in the buffer, every
    # call exercises a fresh random symmetry on the same row.
    for _ in range(50):
        _s, _chain, _p, _o, own_b, wl_b = buf.sample_batch(1, augment=True)
        # Cell count is sym-invariant (centre cells stay in-window).
        assert int((own_b == 2).sum()) == 6, "ownership P1 count must survive augmentation"
        assert int(wl_b.sum()) == 6,         "winning_line count must survive augmentation"
        # No P2 cells were pushed; augmentation must not introduce any.
        assert int((own_b == 0).sum()) == 0
