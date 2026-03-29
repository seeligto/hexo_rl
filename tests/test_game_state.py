"""
Phase 0 tests for the GameState Python dataclass.

Run with: pytest tests/test_game_state.py -v
"""
import pytest
import numpy as np
from native_core import Board
from python.env import GameState


# ── Construction ──────────────────────────────────────────────────────────────

def test_from_board_initial():
    b = Board()
    s = GameState.from_board(b)
    assert s.ply == 0
    assert s.current_player == 1
    assert s.moves_remaining == 1
    assert s.board.shape == (19, 19)
    assert s.board.dtype == np.int8
    assert np.all(s.board == 0)
    assert len(s.move_history) == 0


def test_from_board_after_moves():
    b = Board()
    b.apply_move(0, 0)   # P1 ply0
    b.apply_move(1, 0)   # P2 first of 2
    s = GameState.from_board(b)
    assert s.ply == 2
    assert s.current_player == -1
    assert s.moves_remaining == 1   # P2 has 1 move remaining


# ── Board array values ────────────────────────────────────────────────────────

def test_board_array_p1_stone():
    b = Board()
    b.apply_move(0, 0)   # P1 stone at (0,0)
    # After ply0 it's now P2's turn
    s = GameState.from_board(b)
    q_idx, r_idx = 0 + 9, 0 + 9
    assert s.board[q_idx, r_idx] == 1   # P1 stone should be +1


def test_board_array_p2_stone():
    # Use symmetric P2 stones so the window centre stays at (0,0).
    # bbox after (0,0),(-1,0),(1,0): min_q=-1,max_q=1 → centre_q=0; same for r.
    b = Board()
    b.apply_move(0, 0)    # P1 ply0
    b.apply_move(-1, 0)   # P2 first stone  (symmetric)
    b.apply_move( 1, 0)   # P2 second stone (symmetric — centre stays at (0,0))
    # Now P1's turn again
    s = GameState.from_board(b)
    assert s.board[-1 + 9, 0 + 9] == -1  # P2 stone at (-1,0)
    assert s.board[ 1 + 9, 0 + 9] == -1  # P2 stone at (1,0)


def test_board_array_empty_cells_are_zero():
    b = Board()
    b.apply_move(0, 0)
    s = GameState.from_board(b)
    assert s.board[1 + 9, 0 + 9] == 0
    assert s.board[-9 + 9, -9 + 9] == 0


# ── apply_move updates state ──────────────────────────────────────────────────

def test_apply_move_returns_new_state():
    b = Board()
    s0 = GameState.from_board(b)
    s1 = s0.apply_move(b, 0, 0)
    assert s1.ply == 1
    assert s1.current_player == -1
    assert s1.moves_remaining == 2
    assert s1.board[9, 9] == 1   # P1 stone at (0,0)


def test_apply_move_does_not_mutate_history_board():
    b = Board()
    s0 = GameState.from_board(b)
    s1 = s0.apply_move(b, 0, 0)
    # The board stored in s0 should still be all zeros
    assert np.all(s0.board == 0)


def test_move_history_grows():
    b = Board()
    s = GameState.from_board(b)
    s = s.apply_move(b, 0, 0)   # ply 1
    assert len(s.move_history) == 1
    s = s.apply_move(b, 1, 0)   # ply 2
    assert len(s.move_history) == 2


def test_move_history_capped_at_8():
    b = Board()
    s = GameState.from_board(b)
    # Place 9 stones to overflow history
    # ply0: P1 single
    s = s.apply_move(b, 0, 0)
    # Then 8 more (4 full turns of 2 each, alternating P2/P1)
    moves = [(q, 1) for q in range(-8, 0)]
    for q, r in moves:
        s = s.apply_move(b, q, r)
    assert len(s.move_history) == 8


# ── Zobrist hash ──────────────────────────────────────────────────────────────

def test_zobrist_hash_matches_rust():
    b = Board()
    b.apply_move(0, 0)
    s = GameState.from_board(b)
    assert s.zobrist_hash == b.zobrist_hash()


def test_zobrist_hash_used_for_python_hash():
    b = Board()
    b.apply_move(0, 0)
    s = GameState.from_board(b)
    assert hash(s) == s.zobrist_hash


# ── to_tensor ─────────────────────────────────────────────────────────────────

def test_to_tensor_shape():
    b = Board()
    s = GameState.from_board(b)
    t = s.to_tensor()
    assert t.shape == (18, 19, 19)
    assert t.dtype == np.float16


def test_to_tensor_empty_board_has_zero_stone_planes():
    b = Board()
    s = GameState.from_board(b)
    t = s.to_tensor()
    # All stone planes (0–15) should be zero on empty board
    assert np.all(t[:16] == 0.0)


def test_to_tensor_stone_planes_after_moves():
    b = Board()
    b.apply_move(0, 0)   # P1 at (0,0); now P2's turn
    s = GameState.from_board(b)
    t = s.to_tensor()
    # current player is -1 (P2); opponent is P1
    # plane 7 (latest snapshot) should show opponent stone at (0,0)
    assert t[8 + 7, 9, 9] == 1.0   # P1's stone visible in opponent plane
    assert t[7, 9, 9] == 0.0        # P2 has no stone there


def test_to_tensor_moves_remaining_channel():
    b = Board()
    s = GameState.from_board(b)
    t = s.to_tensor()
    # ply0: moves_remaining=1 → channel 16 = 0.0
    assert np.all(t[16] == 0.0)

    s = s.apply_move(b, 0, 0)  # after P1's single, P2 has 2 remaining
    t = s.to_tensor()
    assert np.all(t[16] == 1.0)


def test_to_tensor_turn_parity_channel():
    b = Board()
    s = GameState.from_board(b)
    t = s.to_tensor()
    assert np.all(t[17] == 0.0)   # ply 0 is even

    s = s.apply_move(b, 0, 0)
    t = s.to_tensor()
    assert np.all(t[17] == 1.0)   # ply 1 is odd


# ── Equality and hashing ──────────────────────────────────────────────────────

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
    b2.apply_move(1, 0)
    s2 = GameState.from_board(b2)

    assert s1 != s2


# ── legal_moves raises ────────────────────────────────────────────────────────

def test_legal_moves_raises():
    b = Board()
    s = GameState.from_board(b)
    with pytest.raises(NotImplementedError):
        s.legal_moves()
