import pytest
import numpy as np
from engine import Board
from hexo_rl.env.game_state import GameState

def test_from_board_initial():
    b = Board()
    s = GameState.from_board(b)
    assert s.ply == 0
    assert s.current_player == 1
    assert s.moves_remaining == 1
    assert len(s.views) == 1
    assert s.views[0].shape == (2, 19, 19)
    assert len(s.centers) == 1
    assert s.centers[0] == (0, 0)

def test_from_board_after_moves():
    b = Board()
    b.apply_move(0, 0)
    b.apply_move(1, 0)
    s = GameState.from_board(b)
    assert s.ply == 2
    assert s.current_player == -1
    assert s.moves_remaining == 1

def test_board_array_p1_stone():
    b = Board()
    b.apply_move(0, 0)   # P1 stone at (0,0)
    # After ply0 it's now P2's turn. P1 is opponent → plane 1 of the 2-plane view.
    s = GameState.from_board(b)
    q_idx, r_idx = 9, 9
    assert s.views[0][1, q_idx, r_idx] == 1.0   # P1 stone in opponent plane (plane 1)

def test_board_array_p2_stone():
    b = Board()
    b.apply_move(0, 0)    # P1 ply0
    b.apply_move(-1, 0)   # P2 first stone
    b.apply_move( 1, 0)   # P2 second stone
    # Now P1's turn again. P2 is opponent → plane 1 of the 2-plane view.
    s = GameState.from_board(b)
    assert s.views[0][1, -1 + 9, 0 + 9] == 1.0  # P2 stone at (-1,0) in opponent plane (plane 1)

def test_board_array_empty_cells_are_zero():
    b = Board()
    b.apply_move(0, 0)
    s = GameState.from_board(b)
    assert s.views[0][0, 1 + 9, 0 + 9] == 0.0

def test_apply_move_returns_new_state():
    b = Board()
    s0 = GameState.from_board(b)
    s1 = s0.apply_move(b, 0, 0)
    assert s1.ply == 1
    assert s1.current_player == -1
    assert s1.moves_remaining == 2
    # It's now P2's turn. P1's stone at (0,0) is in opponent plane 1 (2-plane view).
    assert s1.views[0][1, 9, 9] == 1.0

def test_apply_move_does_not_mutate_history_board():
    b = Board()
    s0 = GameState.from_board(b)
    s1 = s0.apply_move(b, 0, 0)
    # The board stored in s0 should still be all zeros (no stones yet)
    assert np.all(s0.views[0] == 0)

def test_move_history_grows():
    b = Board()
    s = GameState.from_board(b)
    assert len(s.move_history) == 0
    s1 = s.apply_move(b, 0, 0)
    assert len(s1.move_history) == 1

def test_move_history_capped_at_8():
    b = Board()
    s = GameState.from_board(b)
    for i in range(10):
        s = s.apply_move(b, i, 0)
    assert len(s.move_history) == 8

def test_zobrist_hash_matches_rust():
    b = Board()
    b.apply_move(0, 0)
    s = GameState.from_board(b)
    assert s.zobrist_hash == b.zobrist_hash()

def test_zobrist_hash_used_for_python_hash():
    b = Board()
    b.apply_move(0, 0)
    s = GameState.from_board(b)
    # zobrist_hash is u128; Python's hash() reduces large ints to Py_hash_t width.
    assert hash(s) == hash(s.zobrist_hash)

def test_to_tensor_shape():
    b = Board()
    s = GameState.from_board(b)
    t, c = s.to_tensor()
    assert t.shape == (1, 24, 19, 19)

def test_to_tensor_empty_board_has_zero_stone_planes():
    b = Board()
    s = GameState.from_board(b)
    t, c = s.to_tensor()
    assert np.all(t[0, :16] == 0.0)

def test_to_tensor_stone_planes_after_moves():
    b = Board()
    b.apply_move(0, 0)   # P1 at (0,0); now P2's turn
    s = GameState.from_board(b)
    t, c = s.to_tensor()
    # current player is -1 (P2); opponent is P1
    # plane 8 should show opponent stone at (0,0)
    assert t[0, 8, 9, 9] == 1.0

def test_to_tensor_moves_remaining_channel():
    b = Board()
    s = GameState.from_board(b)
    t, c = s.to_tensor()
    assert np.all(t[0, 16] == 0.0)

def test_to_tensor_turn_parity_channel():
    b = Board()
    s = GameState.from_board(b)
    t, c = s.to_tensor()
    assert np.all(t[0, 17] == 0.0)

def test_equal_states_have_equal_hash():
    b1 = Board()
    b1.apply_move(0, 0)
    s1 = GameState.from_board(b1)
    
    b2 = Board()
    b2.apply_move(0, 0)
    s2 = GameState.from_board(b2)
    
    assert s1 == s2
    assert hash(s1) == hash(s2)

def test_different_states_are_not_equal():
    b1 = Board()
    b1.apply_move(0, 0)
    s1 = GameState.from_board(b1)

    b2 = Board()
    b2.apply_move(1, 1)
    s2 = GameState.from_board(b2)

    assert s1 != s2
    assert hash(s1) != hash(s2)

# ── History plane tests ───────────────────────────────────────────────────────

def test_history_planes_graceful_missing():
    """At ply 0 (no history), all history planes 1-7 and 9-15 must be zeros."""
    b = Board()
    s = GameState.from_board(b)
    t, _ = s.to_tensor()
    assert t.shape == (1, 24, 19, 19)
    assert np.all(t[0, 1:8]  == 0.0), "my-stone history planes should be zero at game start"
    assert np.all(t[0, 9:16] == 0.0), "opp-stone history planes should be zero at game start"

def test_history_planes_are_filled():
    """After 3 moves the t-1 history planes must be non-zero (prior positions present)."""
    b = Board()
    s = GameState.from_board(b)
    # ply 0: P1 at (0,0)
    s1 = s.apply_move(b, 0, 0)
    # ply 1+2: P2 at (1,0) and (2,0) — completes P2's turn, passes back to P1
    s2 = s1.apply_move(b, 1, 0)
    s3 = s2.apply_move(b, 2, 0)

    t, centers = s3.to_tensor()
    assert t.shape[1] == 24

    # plane 0 = current player's (P1) stones at t.
    # P1 placed a stone at (0,0). Window center at s3 = (1,0) (bbox [0,2]×[0,0]).
    # P1 stone at (0,0) → window pos wq = 0-1+9 = 8, wr = 0-0+9 = 9.
    cq, cr = centers[0]
    p1_wq = 0 - cq + 9
    p1_wr = 0 - cr + 9
    assert t[0, 0, p1_wq, p1_wr] == 1.0, "P1 stone at (0,0) should be in current my-stones plane"

    # plane 1 = prior my-stones at t-1 (= s2.views[k][0]).
    # s2 was P2's turn with stone at (1,0); s2.views[k][0] = P2 stones = non-zero.
    assert t[0, 1].any(), "t-1 my-stones plane should be non-zero (prior position recorded)"

def test_to_tensor_uses_cached_views():
    """to_tensor() must produce (K, 24, 19, 19) with correct stone planes using cached self.views."""
    b = Board()
    b.apply_move(0, 0)   # P1 at (0,0); now P2's turn
    s = GameState.from_board(b)
    t, c = s.to_tensor()
    assert t.shape == (1, 24, 19, 19)
    # Current player is P2; opponent (P1) stone at (0,0) is in tensor plane 8.
    assert t[0, 8, 9, 9] == 1.0, "opponent stone must appear in plane 8 of to_tensor output"


# ── Split-responsibility boundary tests ───────────────────────────────────────
# These tests verify that Python's to_tensor() correctly reads from the cached
# 2-plane views (split-responsibility: Rust supplies 2 planes, Python assembles 18).

def test_plane_0_current_my_stones_after_p1_move():
    """Given: P1 places at (0,0), P2 places at (1,0).
    When: to_tensor() is called (P1's turn).
    Then: plane 0 (my-stones at t) contains P1's stone at (0,0).
    """
    b = Board()
    s = GameState.from_board(b)
    s = s.apply_move(b, 0, 0)   # P1 ply 0 → P2's turn
    s = s.apply_move(b, 1, 0)   # P2 ply 1
    s = s.apply_move(b, 2, 0)   # P2 ply 2 → P1's turn

    t, centers = s.to_tensor()
    cq, cr = centers[0]
    # P1 stone at (0,0) → window idx
    wq = 0 - cq + 9
    wr = 0 - cr + 9
    assert t[0, 0, wq, wr] == 1.0, \
        "plane 0 (current my-stones) must contain P1's stone at (0,0)"


def test_plane_8_opponent_stones_after_p1_move():
    """Given: P1 places at (0,0), P2 places a pair.
    When: to_tensor() is called (P1's turn).
    Then: plane 8 (opponent's stones at t) contains P2's stones.
    """
    b = Board()
    s = GameState.from_board(b)
    s = s.apply_move(b, 0, 0)   # P1 ply 0 → P2's turn
    s = s.apply_move(b, 1, 0)   # P2 ply 1
    s = s.apply_move(b, 2, 0)   # P2 ply 2 → P1's turn

    t, centers = s.to_tensor()
    cq, cr = centers[0]
    # P2 stones at (1,0) and (2,0) must appear in plane 8 (opponent at t)
    wq1 = 1 - cq + 9
    wr1 = 0 - cr + 9
    wq2 = 2 - cq + 9
    wr2 = 0 - cr + 9
    assert t[0, 8, wq1, wr1] == 1.0, \
        "plane 8 (opponent my-stones) must contain P2's first stone at (1,0)"
    assert t[0, 8, wq2, wr2] == 1.0, \
        "plane 8 (opponent my-stones) must contain P2's second stone at (2,0)"
