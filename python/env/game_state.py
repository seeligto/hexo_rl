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


def _board_to_arrays(rust_board: Board) -> Tuple[np.ndarray, np.ndarray]:
    """Convert the Rust board to two (19, 19) int8 arrays (local and global)."""
    local_flat, global_flat = rust_board.get_dual_state()
    
    local_planes = np.array(local_flat, dtype=np.float32).reshape(2, BOARD_SIZE, BOARD_SIZE)
    global_planes = np.array(global_flat, dtype=np.float32).reshape(2, BOARD_SIZE, BOARD_SIZE)

    cp = rust_board.current_player
    
    local_arr = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
    local_arr += (local_planes[0] * cp).astype(np.int8)
    local_arr += (local_planes[1] * -cp).astype(np.int8)
    
    global_arr = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
    global_arr += (global_planes[0] * cp).astype(np.int8)
    global_arr += (global_planes[1] * -cp).astype(np.int8)
    
    return local_arr, global_arr


@dataclass(frozen=True)
class GameState:
    """Immutable snapshot of a board position."""

    board: np.ndarray          # shape (19, 19), int8: 0/+1/-1 (local)
    global_board: np.ndarray   # shape (19, 19), int8: 0/+1/-1 (global)
    current_player: int        # +1 or -1
    moves_remaining: int       # 1 or 2
    move_history: Tuple[Tuple[np.ndarray, np.ndarray], ...]  # up to HISTORY_LEN previous boards (local, global)
    zobrist_hash: int          # 64-bit hash as Python int
    ply: int                   # total half-moves placed

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def from_board(rust_board: Board, history: Tuple[Tuple[np.ndarray, np.ndarray], ...] = ()) -> "GameState":
        """Build a GameState from a Rust Board (with optional prior history)."""
        local_arr, global_arr = _board_to_arrays(rust_board)
        return GameState(
            board=local_arr,
            global_board=global_arr,
            current_player=rust_board.current_player,
            moves_remaining=rust_board.moves_remaining,
            move_history=history,
            zobrist_hash=rust_board.zobrist_hash(),
            ply=rust_board.ply,
        )

    def apply_move(self, rust_board: Board, q: int, r: int) -> "GameState":
        """Apply a move to *rust_board* (mutates it) and return the new GameState."""
        rust_board.apply_move(q, r)
        new_history = (self.move_history + ((self.board, self.global_board),))[-HISTORY_LEN:]
        return GameState.from_board(rust_board, history=new_history)

    # ------------------------------------------------------------------
    # Tensor encoding for the neural network
    # ------------------------------------------------------------------

    def to_tensor(self) -> np.ndarray:
        """Encode the state as a float16 tensor of shape (36, 19, 19).
        Channels 0-17: Local map history and metadata
        Channels 18-35: Global map history and metadata
        """
        cp = self.current_player
        local_channels = np.zeros((18, BOARD_SIZE, BOARD_SIZE), dtype=np.float16)
        global_channels = np.zeros((18, BOARD_SIZE, BOARD_SIZE), dtype=np.float16)

        # Build the ordered list of boards: history + current (up to 8 total)
        hist_list = list(self.move_history) + [(self.board, self.global_board)]
        hist_list = hist_list[-HISTORY_LEN:]
        
        # Pad with zeros if fewer than HISTORY_LEN boards available
        while len(hist_list) < HISTORY_LEN:
            hist_list.insert(0, (np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8), 
                                 np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)))

        for i, (l_b, g_b) in enumerate(hist_list):
            local_channels[i]     = (l_b == cp).astype(np.float16)    # current player
            local_channels[8 + i] = (l_b == -cp).astype(np.float16)   # opponent
            global_channels[i]     = (g_b == cp).astype(np.float16)
            global_channels[8 + i] = (g_b == -cp).astype(np.float16)

        local_channels[16] = np.float16(0.0 if self.moves_remaining == 1 else 1.0)
        local_channels[17] = np.float16(self.ply % 2)
        
        global_channels[16] = local_channels[16]
        global_channels[17] = local_channels[17]

        return np.concatenate([local_channels, global_channels], axis=0)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def legal_moves(self) -> list:
        """Delegate to Rust board is not possible here (board is a snapshot)."""
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
            and np.array_equal(self.global_board, other.global_board)
        )

    def __hash__(self) -> int:
        return self.zobrist_hash
