"""GameState — immutable snapshot of a Hex Tac Toe board position.

Wraps the Rust `Board` and carries everything the network and MCTS need:
  - board: 19×19 numpy array (0=empty, 1=P1, -1=P2)
  - current_player: 1 or -1
  - moves_remaining: 1 or 2
  - move_history: tuple of up to 8 previous board arrays (oldest first)
  - zobrist_hash: incremental 64-bit hash (as a Python int)
  - ply: total half-moves placed

Usage
-----
    from native_core import Board
    from python.env import GameState

    rust_board = Board()
    state = GameState.from_board(rust_board)
    next_state = state.apply_move(rust_board, q, r)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from native_core import Board

# Number of historical board planes carried in move_history.
HISTORY_LEN: int = 8
BOARD_SIZE: int = 19
HALF: int = 9  # coordinate range: -HALF .. +HALF


def _board_to_array(rust_board: Board) -> np.ndarray:
    """Convert the Rust board to a (19, 19) int8 array using the sliding view window.

    Values: 0 = empty, 1 = P1, -1 = P2.
    Axis layout: arr[wq, wr] where (wq, wr) are window-relative indices
    centred on the bounding-box centroid of all placed stones.
    When the board is empty (or centroid == (0,0)) this is identical to the
    old absolute layout arr[q+9, r+9].
    """
    planes = np.array(rust_board.view_window(BOARD_SIZE), dtype=np.float32).reshape(2, BOARD_SIZE, BOARD_SIZE)
    # plane 0: current player's stones, plane 1: opponent's stones
    cp = rust_board.current_player  # +1 or -1
    arr = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
    arr += (planes[0] * cp).astype(np.int8)   # current player's stones
    arr += (planes[1] * -cp).astype(np.int8)  # opponent's stones
    return arr


@dataclass(frozen=True)
class GameState:
    """Immutable snapshot of a board position."""

    board: np.ndarray          # shape (19, 19), int8: 0/+1/-1
    current_player: int        # +1 or -1
    moves_remaining: int       # 1 or 2
    move_history: Tuple[np.ndarray, ...]  # up to HISTORY_LEN previous boards
    zobrist_hash: int          # 64-bit hash as Python int
    ply: int                   # total half-moves placed

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def from_board(rust_board: Board, history: Tuple[np.ndarray, ...] = ()) -> "GameState":
        """Build a GameState from a Rust Board (with optional prior history)."""
        return GameState(
            board=_board_to_array(rust_board),
            current_player=rust_board.current_player,
            moves_remaining=rust_board.moves_remaining,
            move_history=history,
            zobrist_hash=rust_board.zobrist_hash(),
            ply=rust_board.ply,
        )

    def apply_move(self, rust_board: Board, q: int, r: int) -> "GameState":
        """Apply a move to *rust_board* (mutates it) and return the new GameState.

        The caller is responsible for passing the same Board instance that
        produced this GameState.  After this call, `rust_board` is in the
        post-move state.
        """
        rust_board.apply_move(q, r)
        new_history = (self.move_history + (self.board,))[-HISTORY_LEN:]
        return GameState.from_board(rust_board, history=new_history)

    # ------------------------------------------------------------------
    # Tensor encoding for the neural network
    # ------------------------------------------------------------------

    def to_tensor(self) -> np.ndarray:
        """Encode the state as a float16 tensor of shape (18, 19, 19).

        Channel layout:
          0–7:  current player's stones in last 8 board states (binary, oldest first)
          8–15: opponent's stones in last 8 board states (binary, oldest first)
          16:   moves_remaining broadcast — 0.0 (1 move left) or 1.0 (2 moves left)
          17:   turn parity — 0.0 if ply even, 1.0 if ply odd
        """
        cp = self.current_player
        channels = np.zeros((18, BOARD_SIZE, BOARD_SIZE), dtype=np.float16)

        # Build the ordered list of boards: history + current (up to 8 total)
        boards = list(self.move_history) + [self.board]
        boards = boards[-HISTORY_LEN:]
        # Pad with zeros if fewer than HISTORY_LEN boards available
        while len(boards) < HISTORY_LEN:
            boards.insert(0, np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8))

        for i, b in enumerate(boards):
            channels[i]     = (b == cp).astype(np.float16)    # current player
            channels[8 + i] = (b == -cp).astype(np.float16)   # opponent

        channels[16] = np.float16(0.0 if self.moves_remaining == 1 else 1.0)
        channels[17] = np.float16(self.ply % 2)

        return channels

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def legal_moves(self) -> list:
        """Delegate to Rust board is not possible here (board is a snapshot).

        Callers should call rust_board.legal_moves() directly.
        """
        raise NotImplementedError(
            "Call rust_board.legal_moves() directly — GameState holds a snapshot, "
            "not a live board reference."
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GameState):
            return NotImplemented
        return (
            self.zobrist_hash == other.zobrist_hash
            and self.ply == other.ply
            and self.current_player == other.current_player
            and self.moves_remaining == other.moves_remaining
            and np.array_equal(self.board, other.board)
        )

    def __hash__(self) -> int:
        return self.zobrist_hash
