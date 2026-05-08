"""§169 A3 — global summary crop helper unit tests.

Covers the three planes (cur / opp / canvas-realness mask) and the
T2 §E.1 pitfall 2 invariant: zero-padding inside the 32×32 canvas must
be distinguishable from real-but-empty board cells via the mask.
"""
from __future__ import annotations

import numpy as np

from engine import Board
from hexo_rl.utils.global_crop import (
    CANVAS_SIZE,
    N_GLOBAL_PLANES,
    compute_global_crop,
    compute_global_crop_from_board,
)


def test_empty_board_returns_all_zero():
    """No stones → all-zero canvas (mask included). The encoder must
    handle this trivially (g token built from a degenerate input)."""
    out = compute_global_crop([], [])
    assert out.shape == (N_GLOBAL_PLANES, CANVAS_SIZE, CANVAS_SIZE)
    assert out.dtype == np.float16
    assert np.all(out == 0.0)


def test_single_stone_centers_in_canvas():
    """Single-stone bbox is 1×1; should embed at canvas center, mask 1.0
    on exactly that single cell, rest of canvas zero everywhere."""
    out = compute_global_crop([(0, 0)], [])
    cur, opp, mask = out[0], out[1], out[2]
    assert mask.sum() == 1.0, f"mask should mark exactly 1 cell; got {mask.sum()}"
    # Canvas size 32 ⇒ off=(32-1)//2=15.
    assert mask[15, 15] == 1.0
    assert cur[15, 15] == 1.0
    assert opp.sum() == 0.0


def test_canvas_mask_distinguishes_padding_from_empty():
    """T2 §E.1 pitfall 2 — the mask channel must be 0 in padding cells
    and 1 in active region cells, even when the active region's interior
    is mostly empty (no stones)."""
    # Bbox spans 4×4 (corners at (0,0) / (3,3)); only those two cells are stones.
    out = compute_global_crop([(0, 0)], [(3, 3)])
    cur, opp, mask = out[0], out[1], out[2]
    # Active region 4×4 ⇒ pooled 4×4 (s=1); off_q=14, off_r=14.
    active = mask[14:18, 14:18]
    assert active.sum() == 16.0, "mask should be 1.0 over the entire 4×4 active region"
    # Outside the active region: mask=0 (and stones=0 by construction).
    assert mask[0, 0] == 0.0 and mask[31, 31] == 0.0
    # Stone placement: top-left + bottom-right of the active region.
    assert cur[14, 14] == 1.0
    assert opp[17, 17] == 1.0
    # Real-but-empty cells inside the active region: cur=0, opp=0, mask=1.
    assert cur[15, 15] == 0.0 and opp[15, 15] == 0.0 and mask[15, 15] == 1.0


def test_negative_coordinates_handled_via_bbox_translation():
    """HTTT board is theoretically infinite; stones at negative axial
    coordinates must embed correctly (bbox-relative, not origin-relative)."""
    out = compute_global_crop([(-3, -2)], [(2, 3)])
    cur, opp, mask = out[0], out[1], out[2]
    # Bbox = 6×6 ⇒ pooled 6×6 (s=1); off=(32-6)//2=13.
    assert mask[13:19, 13:19].sum() == 36.0
    assert cur[13, 13] == 1.0
    assert opp[18, 18] == 1.0


def test_large_bbox_downsamples_to_canvas():
    """Bbox > 32 must downsample. Single stone at far corner forces an
    extreme bbox; the helper must still produce a valid 32×32 output
    with the canvas mask correctly marking the active pooled region."""
    # bbox spans 50×50 ⇒ s = ceil(50/32) = 2 ⇒ pooled 25×25 (fits).
    out = compute_global_crop([(0, 0)], [(49, 49)])
    cur, opp, mask = out[0], out[1], out[2]
    # Active region 25×25 ⇒ off=(32-25)//2=3.
    active = mask[3:28, 3:28]
    assert active.sum() == 625.0
    # Stones land at the corners of the active region (within ±1 cell).
    assert cur[3, 3] == 1.0
    # Far stone: floor((49-0)/2) = 24, plus off=3 ⇒ index 27.
    assert opp[27, 27] == 1.0


def test_canvas_size_and_dtype_are_stable():
    """Wire-format guard: canvas constants must stay (3, 32, 32) f16 so
    the corpus NPZ schema and the model's GlobalTokenEncoder agree."""
    assert CANVAS_SIZE == 32
    assert N_GLOBAL_PLANES == 3
    out = compute_global_crop([(0, 0)], [(1, 1)])
    assert out.shape == (3, 32, 32)
    assert out.dtype == np.float16


def test_replay_v6w25_emits_global_crop_per_ply():
    """dataset_v6w25.replay_game_to_triples_v6w25(..., with_global_crop=True)
    must emit a (T, 3, 32, 32) f16 array aligned with the per-ply states
    array, in the player-to-move's frame at each ply."""
    from hexo_rl.bootstrap.dataset_v6w25 import replay_game_to_triples_v6w25

    moves = [
        (0, 0),
        (0, 1), (1, 0),
        (1, 1), (2, 0),
        (2, 1), (3, 0),
    ]
    out = replay_game_to_triples_v6w25(moves, winner=1, with_global_crop=True)
    assert len(out) == 5, f"expected 5-tuple, got {len(out)}"
    states, _chain, _policies, _outcomes, global_crops = out
    T = states.shape[0]
    assert global_crops.shape == (T, 3, 32, 32)
    assert global_crops.dtype == np.float16
    # First ply: empty board ⇒ all-zero crop.
    assert np.all(global_crops[0] == 0.0)
    # Second ply: 1 stone (the P1 opener at (0,0)) ⇒ exactly 1 mask cell + 1 stone cell.
    # P2 is to move; the placed P1 stone shows up on the opp plane.
    assert global_crops[1, 2].sum() == 1.0   # mask
    assert global_crops[1, 1].sum() == 1.0   # opp stones (P1's stone in P2's frame)
    assert global_crops[1, 0].sum() == 0.0   # cur stones (P2 has none yet)


def test_from_board_partitions_by_current_player():
    """compute_global_crop_from_board must split stones by Board's
    current_player and yield the same crop as the manual partition."""
    b = Board()
    b.set_legal_move_radius(8)
    b.set_cluster_threshold(8)
    b.set_cluster_window_size(25)
    # P1 single, P2 turn (2 stones), P1 turn (2 stones).
    b.apply_move(0, 0)
    b.apply_move(0, 1)
    b.apply_move(1, 0)
    b.apply_move(1, 1)
    b.apply_move(2, 0)
    # Now it's P2's turn; current_player=-1.
    crop = compute_global_crop_from_board(b)
    cur_p = int(b.current_player)
    cur, opp = [], []
    for q, r, p in b.get_stones():
        if int(p) == cur_p:
            cur.append((int(q), int(r)))
        elif int(p) != 0:
            opp.append((int(q), int(r)))
    expected = compute_global_crop(cur, opp)
    assert np.array_equal(crop, expected)
