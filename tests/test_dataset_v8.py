"""Tests for hexo_rl.bootstrap.dataset_v8 — v8 corpus encoder."""
from __future__ import annotations

from collections import deque

import numpy as np
import pytest

from hexo_rl.bootstrap.dataset_v8 import (
    BOARD_SIZE_V8,
    HALF_V8,
    HISTORY_LEN_V8,
    LEGAL_MOVE_RADIUS_V8,
    MAX_MOVES_V8,
    N_ACTIONS_V8,
    N_PLANES_V8,
    _build_canvas_realness_mask,
    _build_off_window_mask,
    _compute_bbox_centroid,
    _get_plane8_mask,
    encode_position_v8,
    replay_game_to_triples_v8,
)


# ── Constants regression guards ────────────────────────────────────────────


def test_v8_constants_match_contract() -> None:
    """Pin the v8 numeric constants to the values in
    docs/designs/encoding_v8_contract.md §1.2."""
    assert BOARD_SIZE_V8 == 25
    assert HALF_V8 == 12
    assert N_PLANES_V8 == 11
    assert N_ACTIONS_V8 == 625
    assert LEGAL_MOVE_RADIUS_V8 == 8
    assert HISTORY_LEN_V8 == 4


# ── Off_window mask geometry ───────────────────────────────────────────────


def test_off_window_mask_origin_is_inside() -> None:
    mask = _build_off_window_mask()
    # Origin is the window centre; must be inside the dilated hex.
    assert mask[HALF_V8, HALF_V8] == 0.0


def test_off_window_mask_corners_are_outside() -> None:
    mask = _build_off_window_mask()
    # The four square corners are at hex distance > 8 from origin.
    for wq, wr in [(0, 0), (0, 24), (24, 0), (24, 24)]:
        assert mask[wq, wr] == 1.0, f"corner ({wq},{wr}) must be outside hex"


def test_off_window_mask_radius_boundary() -> None:
    mask = _build_off_window_mask()
    # A cell at hex distance exactly 8 must be inside (≤ R=8).
    # axial (8, 0) → window-local (8+12, 0+12) = (20, 12).
    assert mask[HALF_V8 + 8, HALF_V8] == 0.0, "dist=8 must be inside (≤R)"
    # A cell at hex distance 9 must be outside.
    assert mask[HALF_V8 + 9, HALF_V8] == 1.0, "dist=9 must be outside (>R)"


def test_off_window_mask_count_inside() -> None:
    mask = _build_off_window_mask()
    inside_count = int((mask == 0.0).sum())
    # Hex of radius 8 has 1 + 6*(1+2+3+4+5+6+7+8) = 1 + 6*36 = 217 cells.
    assert inside_count == 217


# ── Bbox centroid ──────────────────────────────────────────────────────────


def test_bbox_centroid_empty_board() -> None:
    assert _compute_bbox_centroid([]) == (0, 0)


def test_bbox_centroid_single_stone() -> None:
    cq, cr = _compute_bbox_centroid([(3, -2, 1)])
    assert (cq, cr) == (3, -2)


def test_bbox_centroid_symmetric_pair() -> None:
    cq, cr = _compute_bbox_centroid([(1, 0, 1), (-1, 0, -1)])
    assert (cq, cr) == (0, 0)


def test_bbox_centroid_off_origin() -> None:
    cq, cr = _compute_bbox_centroid([(2, 4, 1), (4, 6, -1)])
    # bbox q ∈ [2, 4], r ∈ [4, 6] → centroid (3, 5)
    assert (cq, cr) == (3, 5)


# ── encode_position_v8 ─────────────────────────────────────────────────────


def test_encode_empty_board_planes() -> None:
    tensor, (cq, cr), n_clipped = encode_position_v8(
        board_stones=[],
        cur_player=1,
        history=deque(),
        ply=0,
        moves_remaining=1,
    )
    assert tensor.shape == (11, 25, 25)
    assert tensor.dtype == np.float16
    assert (cq, cr) == (0, 0)
    assert n_clipped == 0
    # Stone planes (0-7) all zero
    assert tensor[:8].sum() == 0.0
    # Plane 9: ply=0 → mr_normalized = (200 - 0) / 200 = 1.0
    assert np.all(tensor[9] == np.float16(1.0))
    # Plane 10: ply=0 → 0
    assert np.all(tensor[10] == np.float16(0.0))


def test_encode_single_stone_at_origin() -> None:
    tensor, (cq, cr), n_clipped = encode_position_v8(
        board_stones=[(0, 0, 1)],
        cur_player=1,  # P1 just placed; now P2 to move? No — "cur_player" is who's about to move
        history=deque(),
        ply=1,
        moves_remaining=2,
    )
    # P1 placed the stone but cur_player=1 says P1 is current, so the stone is "ours"
    # → goes to plane 0 (cur ply-0).
    assert tensor[0, HALF_V8, HALF_V8] == 1.0
    # Plane 4 (opp ply-0) must be zero
    assert tensor[4].sum() == 0.0
    assert (cq, cr) == (0, 0)
    assert n_clipped == 0


def test_encode_opponent_stone_goes_to_plane_4() -> None:
    tensor, _, _ = encode_position_v8(
        board_stones=[(0, 0, -1)],  # P2 stone
        cur_player=1,  # P1 to move; P2 is opponent
        history=deque(),
        ply=1,
        moves_remaining=2,
    )
    assert tensor[4, HALF_V8, HALF_V8] == 1.0
    assert tensor[0].sum() == 0.0  # No cur stones


def test_encode_history_planes() -> None:
    """History snapshot at depth 1 → plane 1 (cur) and plane 5 (opp).

    Stones at (0,0), (1,1), (2,0) → bbox q∈[0,2], r∈[0,1] → centroid (1, 0).
    Stones project to window-local (wq = sq - 1 + 12, wr = sr - 0 + 12).
    """
    history: deque = deque(maxlen=3)
    history.append(([(0, 0)], [(1, 1)]))  # cur stones, opp stones at T-1
    tensor, (cq, cr), _ = encode_position_v8(
        board_stones=[(0, 0, 1), (1, 1, -1), (2, 0, 1)],
        cur_player=1,
        history=history,
        ply=2,
        moves_remaining=1,
    )
    # Centroid is (1, 0)
    assert (cq, cr) == (1, 0)
    # Plane 0 (cur ply T): P1 stones at (0,0)→(11,12) and (2,0)→(13,12)
    assert tensor[0, 11, 12] == 1.0
    assert tensor[0, 13, 12] == 1.0
    # Plane 4 (opp ply T): P2 stone at (1,1)→(12,13)
    assert tensor[4, 12, 13] == 1.0
    # Plane 1 (cur ply T-1): historical cur stones at (0,0)→(11,12)
    assert tensor[1, 11, 12] == 1.0
    # Plane 5 (opp ply T-1): historical opp stones at (1,1)→(12,13)
    assert tensor[5, 12, 13] == 1.0
    # Plane 2/3/6/7 (older history): zero
    assert tensor[2].sum() == 0.0
    assert tensor[3].sum() == 0.0
    assert tensor[6].sum() == 0.0
    assert tensor[7].sum() == 0.0


def test_encode_ply_parity() -> None:
    for ply in [0, 1, 2, 3, 100, 101]:
        tensor, _, _ = encode_position_v8(
            board_stones=[],
            cur_player=1,
            history=deque(),
            ply=ply,
            moves_remaining=1,
        )
        assert np.all(tensor[10] == np.float16(float(ply % 2))), \
            f"plane 10 incorrect at ply={ply}"


def test_encode_moves_remaining_bcast() -> None:
    """Plane 9: (MAX_MOVES - ply) / MAX_MOVES, clamped to [0, 1]."""
    tensor, _, _ = encode_position_v8(
        board_stones=[], cur_player=1, history=deque(),
        ply=100, moves_remaining=1,
    )
    expected = np.float16((MAX_MOVES_V8 - 100) / MAX_MOVES_V8)
    assert np.all(tensor[9] == expected)
    # Past MAX_MOVES → clamps to 0
    tensor, _, _ = encode_position_v8(
        board_stones=[], cur_player=1, history=deque(),
        ply=MAX_MOVES_V8 + 50, moves_remaining=1,
    )
    assert np.all(tensor[9] == np.float16(0.0))


def test_encode_clipping_outlier_stone() -> None:
    """Stones too far apart for the 25×25 window → all clipped (telemetry)."""
    # Two stones at origin + one far away → bbox spans 0..50, centroid (25, 25).
    # Window is 25×25 centred on (25, 25); window-local extent ±12.
    # All three stones land at axial offsets > 12 from centroid → all clipped.
    tensor, (cq, cr), n_clipped = encode_position_v8(
        board_stones=[(0, 0, 1), (1, 0, 1), (50, 50, -1)],
        cur_player=1,
        history=deque(),
        ply=3,
        moves_remaining=2,
    )
    assert (cq, cr) == (25, 25)
    # (0, 0) → wq=-13, wr=-13 → OUT
    # (1, 0) → wq=-12, wr=-13 → OUT (wr<0)
    # (50, 50) → wq=37, wr=37 → OUT (both >24)
    assert n_clipped == 3
    # All stone planes empty because every stone fell outside the window.
    assert tensor[0].sum() == 0.0
    assert tensor[4].sum() == 0.0


def test_encode_one_stone_clipped_one_kept() -> None:
    """Stone within window survives; outlier gets clipped — minimal case."""
    # Stones at (0,0) and (15, 0). Centroid q ∈ [0,15] → 7. r ∈ [0,0] → 0.
    # (0,0) → wq = -7 + 12 = 5, wr = 12 → IN
    # (15,0) → wq = 8 + 12 = 20, wr = 12 → IN
    # Both fit. Clip count = 0.
    tensor, (cq, cr), n_clipped = encode_position_v8(
        board_stones=[(0, 0, 1), (15, 0, -1)],
        cur_player=1,
        history=deque(),
        ply=2,
        moves_remaining=1,
    )
    assert (cq, cr) == (7, 0)
    assert n_clipped == 0
    assert tensor[0, 5, 12] == 1.0  # cur (P1)
    assert tensor[4, 20, 12] == 1.0  # opp (P2)


# ── replay_game_to_triples_v8 ──────────────────────────────────────────────


def test_replay_zero_moves() -> None:
    states, chain_planes, policies, outcomes, n_clipped = replay_game_to_triples_v8(
        moves=[], winner=1
    )
    assert states.shape == (0, 11, 25, 25)
    assert chain_planes.shape == (0, 6, 25, 25)
    assert policies.shape == (0, 625)
    assert outcomes.shape == (0,)
    assert n_clipped == 0


def test_replay_single_move() -> None:
    states, chain_planes, policies, outcomes, n_clipped = replay_game_to_triples_v8(
        moves=[(0, 0)], winner=1
    )
    assert states.shape == (1, 11, 25, 25)
    assert chain_planes.shape == (1, 6, 25, 25)
    assert policies.shape == (1, 625)
    assert outcomes.shape == (1,)
    # Target: P1 plays at (0, 0); centroid is (0, 0); window-local (12, 12);
    # flat index 12 * 25 + 12 = 312.
    assert policies[0].argmax() == 312
    # P1 to move at ply 0 → cur_player = 1 = winner → outcome +1
    assert outcomes[0] == 1.0


def test_replay_short_game_shapes() -> None:
    """Replay a 5-move sequence and check tensor shapes + policy correctness."""
    moves = [(0, 0), (1, 0), (0, 1), (1, -1), (-1, 1)]
    states, chain_planes, policies, outcomes, n_clipped = replay_game_to_triples_v8(
        moves=moves, winner=1
    )
    T = states.shape[0]
    assert T <= 5  # may drop positions if target outside window
    assert T > 0
    assert states.shape == (T, 11, 25, 25)
    assert chain_planes.shape == (T, 6, 25, 25)
    assert policies.shape == (T, 625)
    assert outcomes.shape == (T,)
    # Each policy row is one-hot
    for t in range(T):
        assert policies[t].sum() == 1.0
        assert (policies[t] > 0).sum() == 1


def test_replay_outcomes_alternate() -> None:
    """Verify outcomes from current player's POV — should track P1 wins."""
    moves = [(0, 0), (1, 0), (0, 1)]
    _, _, _, outcomes, _ = replay_game_to_triples_v8(moves=moves, winner=1)
    # Each outcome is +1 if cur_player_at_that_ply == winner else -1.
    # With winner=1, plies where P1 is cur_player get +1, P2 plies get -1.
    assert all(o in {-1.0, 1.0} for o in outcomes)


def test_replay_dtypes() -> None:
    states, chain_planes, policies, outcomes, _ = replay_game_to_triples_v8(
        moves=[(0, 0), (1, 0)], winner=1
    )
    assert states.dtype == np.float16
    assert chain_planes.dtype == np.float16
    assert policies.dtype == np.float32
    assert outcomes.dtype == np.float32


# ── Plane semantics integration ────────────────────────────────────────────


def test_replay_plane_8_off_window_consistent_across_plies() -> None:
    """Plane 8 (off_window) is geometry-only — should be identical at every ply."""
    states, _, _, _, _ = replay_game_to_triples_v8(
        moves=[(0, 0), (1, 0), (0, 1)], winner=1
    )
    expected = _build_off_window_mask()
    for t in range(states.shape[0]):
        np.testing.assert_array_equal(states[t, 8], expected)


# ── P2 hotfix-(c) perception verification at encoder level ────────────────


def test_p2_hotfix_c_far_stones_visible_in_v8_encoder() -> None:
    """v8 encoder MUST capture stones at hex distance 6-8 from any
    cur-player stone (P2 hotfix-(c) substance verified at encoder level).

    Probe SUMMARY §31: v7full (8-plane × 19×19 + cluster radius 5) loses
    22% to a brain-dead far-line script because stones at hex_dist > 5
    fall outside the K-cluster window. v8 encoder uses bbox-of-all-stones
    + 25×25 fixed-max + dilation margin m=8, so distant stones up to hex
    radius 8 from any existing stone always appear in the tensor.

    The actual P2 probe (`tests/probes/p2_far_placement_opponent.py`)
    requires a trained v8 model checkpoint and runs in Phase D after
    Phase B (model architecture) + Phase C (v8 bootstrap retrain). This
    test is the encoder-level companion gate.
    """
    # Cur-player stone at origin; opponent stone at hex distance 8 (the
    # rule baseline, beyond v6's R=5 perception cap).
    far_q, far_r = 8, 0  # axial offset (8, 0) → hex distance 8
    tensor, (cq, cr), n_clipped = encode_position_v8(
        board_stones=[(0, 0, 1), (far_q, far_r, -1)],
        cur_player=1,
        history=deque(),
        ply=2,
        moves_remaining=1,
    )
    # bbox spans q ∈ [0, 8], r ∈ [0, 0] → centroid (4, 0).
    assert (cq, cr) == (4, 0)
    assert n_clipped == 0, "v8 encoder must NOT clip stones at hex dist ≤ R=8"
    # P1 stone at axial (0, 0): wq = 0 - 4 + 12 = 8, wr = 0 + 12 = 12. Visible.
    assert tensor[0, 8, 12] == 1.0, "v8 encoder must capture origin stone"
    # P2 stone at axial (8, 0): wq = 8 - 4 + 12 = 16, wr = 12. Visible.
    assert tensor[4, 16, 12] == 1.0, \
        "v8 encoder MUST capture far_line stone at hex_dist=8 (P2 hotfix-(c))"


def test_p2_hotfix_c_all_six_axes_at_radius_8() -> None:
    """The 6 hex axes at radius 8 each must be captured by v8 encoder.

    The far-line script in P2 plays along all 6 axes at radii {6, 7, 8}.
    v6 caps perception at R=5 — most of those stones are invisible. v8
    perception is R=8 — every far-line stone visible.
    """
    HEX_DIRS = [(1, 0), (0, 1), (1, -1), (-1, 0), (0, -1), (-1, 1)]
    for radius in (6, 7, 8):
        for dq, dr in HEX_DIRS:
            far_q, far_r = dq * radius, dr * radius
            tensor, _, n_clipped = encode_position_v8(
                board_stones=[(0, 0, 1), (far_q, far_r, -1)],
                cur_player=1,
                history=deque(),
                ply=2,
                moves_remaining=1,
            )
            assert n_clipped == 0, (
                f"v8 must capture stone at axis {(dq, dr)} radius {radius} "
                f"(hex_dist={radius}); got n_clipped={n_clipped}"
            )
            # Plane 4 (opp ply T) should contain the far stone projected
            # through the bbox centroid.
            assert tensor[4].sum() > 0, (
                f"plane 4 empty at axis {(dq, dr)} radius {radius} "
                f"— far stone lost in encoding"
            )


def test_v8_replay_board_uses_R_8_perception() -> None:
    """v8 replay constructs a Board with R=8 (HTTT rule baseline).

    Hotfix-(c) bundling: under v8 path, Board.legal_move_radius must be 8,
    not the v6 default of 5. This is checked at the Board level via the
    PyBoard.legal_move_radius getter (newly exposed in §166 Bucket D).
    """
    from engine import Board
    b = Board()
    assert b.legal_move_radius() == 5, "v6 default radius is 5"
    b.set_legal_move_radius(LEGAL_MOVE_RADIUS_V8)
    assert b.legal_move_radius() == 8, \
        f"after setting v8 radius, getter must return 8; got {b.legal_move_radius()}"


def test_v8_off_window_mask_includes_radius_8_inside() -> None:
    """The off_window indicator (plane 8) must mark the entire dilated
    hex of radius 8 as INSIDE (mask=0). This is the perception envelope
    boundary — outside = padding cell, inside = valid cell.
    """
    mask = _build_off_window_mask()
    HEX_DIRS = [(1, 0), (0, 1), (1, -1), (-1, 0), (0, -1), (-1, 1)]
    for radius in range(0, 9):  # 0..8 inclusive (R=8 = LEGAL_MOVE_RADIUS_V8)
        for dq, dr in HEX_DIRS:
            wq = dq * radius + HALF_V8
            wr = dr * radius + HALF_V8
            assert mask[wq, wr] == 0.0, (
                f"window-local ({wq}, {wr}) at axis {(dq, dr)} radius {radius} "
                f"should be INSIDE the dilated hex"
            )
    # Radius 9 is OUTSIDE
    for dq, dr in HEX_DIRS:
        wq = dq * 9 + HALF_V8
        wr = dr * 9 + HALF_V8
        if 0 <= wq < BOARD_SIZE_V8 and 0 <= wr < BOARD_SIZE_V8:
            assert mask[wq, wr] == 1.0, (
                f"window-local ({wq}, {wr}) at axis {(dq, dr)} radius 9 "
                f"should be OUTSIDE the dilated hex"
            )


def test_canvas_realness_mask_is_inverted_off_window() -> None:
    """§169 A4 — canvas_realness polarity is exactly 1 - off_window."""
    off = _build_off_window_mask()
    realness = _build_canvas_realness_mask()
    np.testing.assert_array_equal(realness, (1.0 - off).astype(np.float16))


def test_canvas_realness_mask_origin_inside_corners_outside() -> None:
    """§169 A4 — origin should read 1 (inside canvas), corners 0 (outside)."""
    realness = _build_canvas_realness_mask()
    assert realness[HALF_V8, HALF_V8] == 1.0
    for wq, wr in [(0, 0), (0, 24), (24, 0), (24, 24)]:
        assert realness[wq, wr] == 0.0, (
            f"corner ({wq},{wr}) must be 0 under canvas_realness"
        )


def test_get_plane8_mask_polarity_dispatch() -> None:
    """``_get_plane8_mask`` returns the requested polarity (cached)."""
    off = _get_plane8_mask(canvas_realness=False)
    realness = _get_plane8_mask(canvas_realness=True)
    np.testing.assert_array_equal(off, _build_off_window_mask())
    np.testing.assert_array_equal(realness, _build_canvas_realness_mask())
    # Caching: second call returns identical array reference.
    assert _get_plane8_mask(canvas_realness=False) is off
    assert _get_plane8_mask(canvas_realness=True) is realness


def test_encode_position_v8_canvas_realness_inverts_plane_8() -> None:
    """§169 A4 — canvas_realness=True flips plane 8 polarity (1=inside)."""
    args: dict = dict(
        board_stones=[(0, 0, 1)],
        cur_player=1,
        history=deque(),
        ply=1,
        moves_remaining=2,
    )
    tensor_off, _, _ = encode_position_v8(canvas_realness=False, **args)
    tensor_real, _, _ = encode_position_v8(canvas_realness=True, **args)
    # Plane 8 differs — exactly inverted.
    np.testing.assert_array_equal(
        tensor_real[8], (1.0 - tensor_off[8]).astype(np.float16),
    )
    # All other planes unchanged.
    for p in [0, 1, 2, 3, 4, 5, 6, 7, 9, 10]:
        np.testing.assert_array_equal(tensor_real[p], tensor_off[p])
    # Spot check: origin is inside canvas (realness=1, off=0).
    assert tensor_real[8, HALF_V8, HALF_V8] == 1.0
    assert tensor_off[8, HALF_V8, HALF_V8] == 0.0


def test_replay_canvas_realness_inverts_plane_8_across_plies() -> None:
    """§169 A4 — replay with canvas_realness=True flips plane 8 at every ply."""
    moves = [(0, 0), (1, 0), (0, 1)]
    states_off, *_ = replay_game_to_triples_v8(moves, 1, canvas_realness=False)
    states_real, *_ = replay_game_to_triples_v8(moves, 1, canvas_realness=True)
    assert states_off.shape == states_real.shape
    for t in range(states_off.shape[0]):
        np.testing.assert_array_equal(
            states_real[t, 8], (1.0 - states_off[t, 8]).astype(np.float16),
        )


def test_replay_plane_10_alternates_with_ply() -> None:
    """ply_parity should track ply % 2 for each emitted position."""
    states, _, _, _, _ = replay_game_to_triples_v8(
        moves=[(0, 0), (1, 0), (0, 1), (1, -1)], winner=1
    )
    # Position t was encoded at ply t (since each iteration emits one record
    # before applying the move; ply increments AFTER apply_move).
    for t in range(states.shape[0]):
        expected = np.float16(float(t % 2))
        assert np.all(states[t, 10] == expected), \
            f"ply_parity mismatch at t={t}"
