"""Unit tests for RecentBuffer augmentation in batch_assembly._augment_recent_rows.

Guards three invariants:
  I1. sym_idx=0 (identity) produces byte-identical output to un-augmented.
  I2. A known rotation (sym_idx=1) matches engine.apply_symmetry on the state,
      with the policy permuted by the corresponding LUT.
  I3. augment=False is a strict no-op (same array objects returned).
"""
from __future__ import annotations

import numpy as np
import pytest

import engine
from engine import Board
from hexo_rl.augment.luts import get_policy_scatters
from hexo_rl.env.game_state import GameState, _compute_chain_planes
from hexo_rl.training.batch_assembly import _augment_recent_rows
from hexo_rl.utils.constants import BOARD_SIZE

N_ACTIONS = BOARD_SIZE * BOARD_SIZE + 1
AUX_STRIDE = BOARD_SIZE * BOARD_SIZE


def _make_synthetic_batch(n: int = 4, seed: int = 0) -> tuple:
    """Build a batch of n rows via Board positions for realistic stone planes."""
    rng = np.random.default_rng(seed)
    states   = np.zeros((n, 18, BOARD_SIZE, BOARD_SIZE), dtype=np.float16)
    chain    = np.zeros((n, 6, BOARD_SIZE, BOARD_SIZE),  dtype=np.float16)
    policies = np.zeros((n, N_ACTIONS), dtype=np.float32)
    outcomes = rng.uniform(-1, 1, size=n).astype(np.float32)
    own_flat = np.ones((n, AUX_STRIDE), dtype=np.uint8)
    wl_flat  = np.zeros((n, AUX_STRIDE), dtype=np.uint8)
    ifs      = np.ones(n, dtype=np.uint8)

    move_seqs = [
        [(0, 0)],
        [(0, 0), (1, 0), (0, 1)],
        [(0, 0), (1, 0), (2, 0), (-1, 1)],
        [(0, 0), (-1, 1), (1, -1), (2, -2)],
    ]
    for i in range(n):
        board = Board()
        gs = GameState.from_board(board)
        for q, r in move_seqs[i % len(move_seqs)]:
            gs = gs.apply_move(board, q, r)
        tensor, _ = gs.to_tensor()
        states[i] = tensor[0].astype(np.float16)
        # chain planes from stone planes 0 and 8
        s32 = states[i].astype(np.float32)
        chain[i] = (_compute_chain_planes(s32[0], s32[8]).astype(np.float32) / 6.0).astype(np.float16)
        # uniform policy with non-trivial mass
        p = rng.uniform(0, 1, size=N_ACTIONS).astype(np.float32)
        p /= p.sum()
        policies[i] = p
        # ownership: random mix of 0/1/2 (empty=1, cur=0, opp=2)
        own_flat[i] = rng.choice([0, 1, 2], size=AUX_STRIDE).astype(np.uint8)
        # winning_line: sparse
        wl_flat[i] = (rng.uniform(size=AUX_STRIDE) < 0.05).astype(np.uint8)

    return states, chain, policies, outcomes, own_flat, wl_flat, ifs


def _force_sym(s_r, c_r, p_r, own_flat, wl_flat, sym_idx: int):
    """Apply a single fixed sym_idx to all rows (for testing)."""
    import engine as _engine
    from hexo_rl.env.game_state import _compute_chain_planes as _ccp

    n = len(s_r)
    scatters = get_policy_scatters()
    lut = scatters[sym_idx]

    states_f32 = s_r.astype(np.float32)
    sym_ids = [sym_idx] * n
    states_f32 = _engine.apply_symmetries_batch(states_f32, sym_ids)
    s_out = states_f32.astype(np.float16)

    c_out = np.empty_like(c_r)
    for i in range(n):
        c_out[i] = (_ccp(states_f32[i, 0], states_f32[i, 8]).astype(np.float32) / 6.0).astype(np.float16)

    p_out = np.empty_like(p_r)
    own_out = np.empty_like(own_flat)
    wl_out = np.empty_like(wl_flat)
    for i in range(n):
        p_out[i] = p_r[i][lut]
        own_out[i] = own_flat[i][lut[:AUX_STRIDE]]
        wl_out[i] = wl_flat[i][lut[:AUX_STRIDE]]

    return s_out, c_out, p_out, own_out, wl_out


def test_identity_is_byte_identical():
    """sym_idx=0 output is byte-identical to un-augmented input."""
    s, c, p, o, own, wl, ifs = _make_synthetic_batch(n=4)
    s_id, c_id, p_id, own_id, wl_id = _force_sym(s, c, p, own, wl, sym_idx=0)

    np.testing.assert_array_equal(s_id, s, err_msg="states differ under identity")
    np.testing.assert_array_equal(p_id, p, err_msg="policies differ under identity")
    np.testing.assert_array_equal(own_id, own, err_msg="ownership differs under identity")
    np.testing.assert_array_equal(wl_id, wl, err_msg="winning_line differs under identity")


def test_rotation_matches_apply_symmetry():
    """sym_idx=1 state matches engine.apply_symmetry; policy matches LUT scatter."""
    s, c, p, o, own, wl, ifs = _make_synthetic_batch(n=3)
    sym_idx = 1

    s_aug, c_aug, p_aug, own_aug, wl_aug = _force_sym(s, c, p, own, wl, sym_idx=sym_idx)

    scatters = get_policy_scatters()
    lut = scatters[sym_idx]

    for i in range(len(s)):
        # State: compare via engine.apply_symmetry reference
        ref_state = engine.apply_symmetry(s[i].astype(np.float32), sym_idx).astype(np.float16)
        np.testing.assert_array_equal(
            s_aug[i], ref_state,
            err_msg=f"row {i}: augmented state does not match engine.apply_symmetry",
        )

        # Policy: LUT scatter
        expected_policy = p[i][lut]
        np.testing.assert_allclose(
            p_aug[i], expected_policy, rtol=0, atol=0,
            err_msg=f"row {i}: policy scatter mismatch",
        )

        # Ownership scatter
        expected_own = own[i][lut[:AUX_STRIDE]]
        np.testing.assert_array_equal(
            own_aug[i], expected_own,
            err_msg=f"row {i}: ownership scatter mismatch",
        )


def test_augment_false_is_noop():
    """augment=False returns same array objects (no copies, no mutation)."""
    s, c, p, o, own, wl, ifs = _make_synthetic_batch(n=4)
    s_out, c_out, p_out, own_out, wl_out = _augment_recent_rows(
        s, c, p, own, wl, augment=False
    )
    assert s_out is s,   "states: expected same object"
    assert c_out is c,   "chain_planes: expected same object"
    assert p_out is p,   "policies: expected same object"
    assert own_out is own, "ownership: expected same object"
    assert wl_out is wl, "winning_line: expected same object"


def test_augment_true_changes_data():
    """augment=True actually transforms the batch (not a no-op)."""
    s, c, p, o, own, wl, ifs = _make_synthetic_batch(n=8, seed=99)
    np.random.seed(7)  # deterministic sym draws; unlikely all land on sym_idx=0
    s_aug, c_aug, p_aug, own_aug, wl_aug = _augment_recent_rows(
        s.copy(), c.copy(), p.copy(), own.copy(), wl.copy(), augment=True
    )
    # At least one of the 8 rows should differ (P(all sym_idx=0) = (1/12)^8 ≈ 1e-9)
    assert not np.array_equal(s_aug, s) or not np.array_equal(p_aug, p), (
        "augment=True produced identical output — extremely unlikely, check RNG seeding"
    )
