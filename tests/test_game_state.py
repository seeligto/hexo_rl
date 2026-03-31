import pytest
import numpy as np
from native_core import Board
from python.env.game_state import GameState

def test_from_board_initial():
    b = Board()
    s = GameState.from_board(b)
    assert s.ply == 0
    assert s.current_player == 1
    assert s.moves_remaining == 1
    assert len(s.views) == 1
    assert s.views[0].shape == (18, 19, 19)
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
    # After ply0 it's now P2's turn. P1 is opponent → plane 8.
    s = GameState.from_board(b)
    q_idx, r_idx = 9, 9
    assert s.views[0][8, q_idx, r_idx] == 1.0   # P1 stone in opponent view (plane 8)

def test_board_array_p2_stone():
    b = Board()
    b.apply_move(0, 0)    # P1 ply0
    b.apply_move(-1, 0)   # P2 first stone
    b.apply_move( 1, 0)   # P2 second stone
    # Now P1's turn again. P2 is opponent → plane 8.
    s = GameState.from_board(b)
    assert s.views[0][8, -1 + 9, 0 + 9] == 1.0  # P2 stone at (-1,0) in opponent view (plane 8)

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
    # It's now P2's turn. P1's stone at (0,0) is in opponent plane 8.
    assert s1.views[0][8, 9, 9] == 1.0

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
    assert t.shape == (1, 18, 19, 19)

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
