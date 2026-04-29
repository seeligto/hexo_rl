"""Uniform 12-fold augmentation across ALL batch slices — Option A guard.

Covers the RecentBuffer gap identified in docs/notes/augmentation_audit.md:
at late self-play ~67% of each batch was identity-only (no aug). This file
verifies the Python call-site fix in batch_assembly._augment_recent_rows.

  T1. augment=True / 256 rows: non-identity rate ≥ 82% on recent slice
      (expected 11/12 ≈ 91.7% under uniform-12 draws).
  T2. Same np.random seed → identical sym draws (determinism).
  T3. augment=False → identity on recent slice (same object references);
      corpus/selfplay sample_batch forwarded with augment=False.
  T4. assemble_mixed_batch with real RecentBuffer: n_recent_actual > 0,
      corpus/selfplay sample_batch called with correct augment flag, and
      recent slice is transformed when augment=True.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from hexo_rl.training.batch_assembly import (
    _augment_recent_rows,
    allocate_batch_buffers,
    assemble_mixed_batch,
)
from hexo_rl.training.recency_buffer import RecentBuffer
from hexo_rl.utils.constants import BOARD_SIZE, KEPT_PLANE_INDICES

N_ACTIONS = BOARD_SIZE * BOARD_SIZE + 1
AUX_STRIDE = BOARD_SIZE * BOARD_SIZE


# ── helpers ───────────────────────────────────────────────────────────────────

def _mk_buffer_return(n: int) -> tuple:
    """Mock Rust buffer sample_batch return — HEXB v6: 8-plane states."""
    return (
        np.zeros((n, 8,  19, 19), dtype=np.float16),
        np.zeros((n, 6,  19, 19), dtype=np.float16),
        np.zeros((n, N_ACTIONS),  dtype=np.float32),
        np.zeros(n,               dtype=np.float32),
        np.ones( (n, 19, 19),     dtype=np.uint8),
        np.zeros((n, 19, 19),     dtype=np.uint8),
        np.ones(n,                dtype=np.uint8),
    )


def _make_asymmetric_recent_buffer(n: int, seed: int = 0) -> tuple[RecentBuffer, np.ndarray]:
    """Fill a RecentBuffer with n clearly-asymmetric positions.

    Uses a mid-game Board position so all 12 syms produce distinct states.
    Returns (buffer, stacked_states) for comparison.
    """
    from engine import Board
    from hexo_rl.env.game_state import GameState, _compute_chain_planes

    rng = np.random.default_rng(seed)
    buf = RecentBuffer(capacity=max(n, 32))
    saved = np.zeros((n, 8, BOARD_SIZE, BOARD_SIZE), dtype=np.float16)  # HEXB v6

    board = Board()
    gs = GameState.from_board(board)
    for q, r in [(0, 0), (1, 0), (2, 0), (-1, 1), (-2, 2), (3, -1)]:
        gs = gs.apply_move(board, q, r)
    tensor, _ = gs.to_tensor()
    base18 = tensor[0].astype(np.float32)  # (18, 19, 19) from game engine
    base8  = base18[KEPT_PLANE_INDICES].astype(np.float16)  # (8, 19, 19) for buffer

    for i in range(n):
        s = base8.copy()
        saved[i] = s
        # chain from 18-plane stones 0 (cur) and 8 (opp) before slicing
        chain = (_compute_chain_planes(base18[0], base18[8]).astype(np.float32) / 6.0).astype(np.float16)
        p = rng.uniform(0, 1, size=N_ACTIONS).astype(np.float32)
        p /= p.sum()
        own = rng.choice([0, 1, 2], size=AUX_STRIDE).astype(np.uint8)
        wl  = (rng.uniform(size=AUX_STRIDE) < 0.05).astype(np.uint8)
        buf.push(s, chain, p, 0.0, own, wl, True)

    return buf, saved


# ── T1: non-identity frequency ────────────────────────────────────────────────

def test_augment_recent_nonidentity_frequency_256():
    """augment=True on 256 rows: ≥ 82% changed (expected 11/12 ≈ 91.7%)."""
    n = 256
    buf, _ = _make_asymmetric_recent_buffer(n)
    s_r, c_r, p_r, o_r, own_r, wl_r, _ = buf.sample(n)
    s_orig = s_r.copy()

    np.random.seed(101)
    s_aug, c_aug, p_aug, own_aug, wl_aug = _augment_recent_rows(
        s_r, c_r, p_r, own_r, wl_r, augment=True
    )

    n_changed = int(sum(not np.array_equal(s_aug[i], s_orig[i]) for i in range(n)))
    frac = n_changed / n
    # 11/12 ≈ 91.7%; allow ±10% margin (3-sigma binomial ≈ 5%)
    assert frac >= 0.82, (
        f"recent slice non-identity rate {frac:.1%} < 82% — augmentation not uniform"
    )


# ── T2: determinism ───────────────────────────────────────────────────────────

def test_determinism_same_seed_same_transforms():
    """Same np.random seed before sample+augment → identical output."""
    n = 64
    buf, _ = _make_asymmetric_recent_buffer(n)

    np.random.seed(42)
    s_r1, c_r1, p_r1, o_r1, own_r1, wl_r1, _ = buf.sample(n)
    s1, _, p1, _, _ = _augment_recent_rows(
        s_r1.copy(), c_r1.copy(), p_r1.copy(), own_r1.copy(), wl_r1.copy(), augment=True
    )

    np.random.seed(42)
    s_r2, c_r2, p_r2, o_r2, own_r2, wl_r2, _ = buf.sample(n)
    s2, _, p2, _, _ = _augment_recent_rows(
        s_r2.copy(), c_r2.copy(), p_r2.copy(), own_r2.copy(), wl_r2.copy(), augment=True
    )

    np.testing.assert_array_equal(s1, s2, err_msg="states: same seed → different transforms")
    np.testing.assert_array_equal(p1, p2, err_msg="policies: same seed → different transforms")


# ── T3: augment=False identity ────────────────────────────────────────────────

def test_augment_false_recent_slice_is_noop():
    """augment=False: _augment_recent_rows returns same array objects."""
    n = 32
    buf, _ = _make_asymmetric_recent_buffer(n)
    s_r, c_r, p_r, o_r, own_r, wl_r, _ = buf.sample(n)

    s_out, c_out, p_out, own_out, wl_out = _augment_recent_rows(
        s_r, c_r, p_r, own_r, wl_r, augment=False
    )

    assert s_out is s_r,     "states: expected same object for augment=False"
    assert c_out is c_r,     "chain: expected same object for augment=False"
    assert p_out is p_r,     "policies: expected same object for augment=False"
    assert own_out is own_r, "ownership: expected same object for augment=False"
    assert wl_out is wl_r,   "winning_line: expected same object for augment=False"


def test_augment_false_assemble_forwards_flag():
    """assemble_mixed_batch(augment=False): corpus + selfplay called with False."""
    batch_size = 32
    n_pre  = 8
    n_self = 24

    pretrained = MagicMock()
    pretrained.sample_batch = MagicMock(return_value=_mk_buffer_return(n_pre))
    selfplay = MagicMock()
    selfplay.sample_batch = MagicMock(return_value=_mk_buffer_return(n_self))

    bufs = allocate_batch_buffers(batch_size, N_ACTIONS)
    assemble_mixed_batch(
        pretrained_buffer=pretrained,
        buffer=selfplay,
        recent_buffer=None,
        n_pre=n_pre,
        n_self=n_self,
        batch_size=batch_size,
        batch_size_cfg=batch_size,
        recency_weight=0.0,
        bufs=bufs,
        train_step=0,
        augment=False,
    )

    pretrained.sample_batch.assert_called_once_with(n_pre, False)
    selfplay.sample_batch.assert_called_once_with(max(1, n_self), False)


# ── T4: assemble with real RecentBuffer ───────────────────────────────────────

@pytest.mark.parametrize("augment", [True, False])
def test_assemble_with_recent_buffer(augment: bool):
    """assemble_mixed_batch with real RecentBuffer: n_recent_actual > 0;
    corpus/selfplay receive correct augment flag; recent slice transformed
    iff augment=True."""
    batch_size   = 32
    n_pre        = 8
    n_self       = 24
    recency_weight = 0.75
    n_recent_req = max(1, int(round(n_self * recency_weight)))  # 18
    n_uniform    = n_self - n_recent_req                         # 6

    pretrained = MagicMock()
    pretrained.sample_batch = MagicMock(return_value=_mk_buffer_return(n_pre))
    selfplay = MagicMock()
    selfplay.sample_batch = MagicMock(return_value=_mk_buffer_return(n_uniform))

    buf, _ = _make_asymmetric_recent_buffer(n_recent_req + 5)
    bufs = allocate_batch_buffers(batch_size, N_ACTIONS)

    # Capture what sample() will draw so we can compare
    np.random.seed(55)
    s_raw, *_ = buf.sample(n_recent_req)

    np.random.seed(55)
    result = assemble_mixed_batch(
        pretrained_buffer=pretrained,
        buffer=selfplay,
        recent_buffer=buf,
        n_pre=n_pre,
        n_self=n_self,
        batch_size=batch_size,
        batch_size_cfg=batch_size,
        recency_weight=recency_weight,
        bufs=bufs,
        train_step=500,
        augment=augment,
    )
    states_batch, _, _, _, _, _, _, n_recent_actual = result

    # recent rows were drawn
    assert n_recent_actual == n_recent_req, (
        f"expected {n_recent_req} recent rows, got {n_recent_actual}"
    )

    # corpus/selfplay flags
    pretrained.sample_batch.assert_called_once_with(n_pre, augment)
    selfplay.sample_batch.assert_called_once_with(n_uniform, augment)

    # recent slice: compare to raw sample
    recent_slice = np.array(states_batch[n_pre:n_pre + n_recent_actual])
    if augment:
        # At least some rows should be transformed (P(all identity) ≈ (1/12)^18)
        n_changed = int(sum(
            not np.array_equal(recent_slice[i], s_raw[i]) for i in range(n_recent_actual)
        ))
        assert n_changed > 0, (
            "augment=True: all recent rows identity — _augment_recent_rows not called"
        )
    else:
        np.testing.assert_array_equal(
            recent_slice, s_raw,
            err_msg="augment=False: recent slice contains transforms (should be identity)",
        )
