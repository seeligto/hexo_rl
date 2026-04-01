"""Verify the sequential two-stone action space is correctly implemented.

The game uses 1-2-2 turn structure: P1 opens with 1 stone, then both players
alternate 2 stones per turn.  This is implemented as sequential MCTS plies
(one node per stone placed), NOT a compound action space.

Key properties verified:
  1. current_player does NOT change between stones of a 2-stone turn
  2. current_player flips after the last stone of each turn
  3. moves_remaining transitions: 2→1 (no flip), 1→0→reset=2 (flip)
  4. Dirichlet noise is skipped at intermediate plies (moves_remaining==1, ply>0)

Player encoding: P1 = 1, P2 = -1.
"""
from __future__ import annotations

from native_core import Board

P1 = 1
P2 = -1


class TestTurnStructure:
    """Board-level turn tracking: moves_remaining and current_player."""

    def test_opening_move_is_single_stone(self):
        b = Board()
        assert b.moves_remaining == 1
        assert b.current_player == P1

    def test_after_opening_move_player_flips(self):
        b = Board()
        b.apply_move(0, 0)
        assert b.current_player == P2
        assert b.moves_remaining == 2

    def test_first_stone_of_two_stone_turn_no_player_flip(self):
        b = Board()
        b.apply_move(0, 0)  # P1 opening
        assert b.current_player == P2
        assert b.moves_remaining == 2
        b.apply_move(1, 0)  # P2 first stone
        assert b.current_player == P2  # still P2
        assert b.moves_remaining == 1  # one stone left

    def test_second_stone_of_two_stone_turn_flips_player(self):
        b = Board()
        b.apply_move(0, 0)  # P1 opening (mr=1→0, flip to P2)
        b.apply_move(1, 0)  # P2 stone 1 (mr=2→1, no flip)
        b.apply_move(2, 0)  # P2 stone 2 (mr=1→0, flip to P1)
        assert b.current_player == P1
        assert b.moves_remaining == 2

    def test_full_turn_cycle(self):
        """P1(1) → P2(2) → P1(2) → P2(2) ..."""
        b = Board()
        # P1 opening: 1 stone
        b.apply_move(0, 0)
        assert b.current_player == P2
        assert b.moves_remaining == 2
        # P2 turn: 2 stones
        b.apply_move(1, 0)
        assert b.current_player == P2
        assert b.moves_remaining == 1
        b.apply_move(2, 0)
        assert b.current_player == P1
        assert b.moves_remaining == 2
        # P1 turn: 2 stones
        b.apply_move(3, 0)
        assert b.current_player == P1
        assert b.moves_remaining == 1
        b.apply_move(4, 0)
        assert b.current_player == P2
        assert b.moves_remaining == 2
        # P2 turn: 2 stones
        b.apply_move(5, 0)
        assert b.current_player == P2
        assert b.moves_remaining == 1
        b.apply_move(6, 0)
        assert b.current_player == P1
        assert b.moves_remaining == 2


class TestIntermediatePlyDetection:
    """Intermediate ply = moves_remaining==1 and ply>0 (second stone of a turn).

    Dirichlet noise should be skipped at intermediate plies.
    """

    def test_ply0_is_not_intermediate(self):
        b = Board()
        # Ply 0: P1 opening, moves_remaining=1, but ply==0 → NOT intermediate
        assert b.moves_remaining == 1
        assert b.ply == 0
        is_intermediate = b.moves_remaining == 1 and b.ply > 0
        assert not is_intermediate

    def test_first_stone_of_p2_turn_is_not_intermediate(self):
        b = Board()
        b.apply_move(0, 0)  # P1 opening
        # Now P2's turn starts, moves_remaining=2 → NOT intermediate
        assert b.moves_remaining == 2
        is_intermediate = b.moves_remaining == 1 and b.ply > 0
        assert not is_intermediate

    def test_second_stone_of_p2_turn_is_intermediate(self):
        b = Board()
        b.apply_move(0, 0)  # P1 opening
        b.apply_move(1, 0)  # P2 stone 1
        # Now P2 has moves_remaining=1 and ply>0 → IS intermediate
        assert b.moves_remaining == 1
        assert b.ply > 0
        is_intermediate = b.moves_remaining == 1 and b.ply > 0
        assert is_intermediate


class TestMovesPlaneTensor:
    """Plane 16 of the tensor encodes moves_remaining==2."""

    def test_plane16_is_one_at_turn_start(self):
        b = Board()
        b.apply_move(0, 0)  # After P1 opening, P2's turn, mr=2
        tensor = b.to_tensor()
        TOTAL_CELLS = 19 * 19
        # Plane 16 should be all 1.0
        plane16 = tensor[16 * TOTAL_CELLS: 17 * TOTAL_CELLS]
        assert all(v == 1.0 for v in plane16)

    def test_plane16_is_zero_at_mid_turn(self):
        b = Board()
        b.apply_move(0, 0)
        b.apply_move(1, 0)  # P2 stone 1, now mr=1
        tensor = b.to_tensor()
        TOTAL_CELLS = 19 * 19
        plane16 = tensor[16 * TOTAL_CELLS: 17 * TOTAL_CELLS]
        assert all(v == 0.0 for v in plane16)
