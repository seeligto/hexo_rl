"""GameState — immutable snapshot of a Hex Tac Toe board position."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple, List
import numpy as np
from native_core import Board

BOARD_SIZE: int = 19
HISTORY_LEN: int = 8

@dataclass(frozen=True)
class GameState:
    current_player: int
    moves_remaining: int
    zobrist_hash: int
    ply: int
    move_history: Tuple[GameState, ...] = field(default_factory=tuple)
    views: List[np.ndarray] = field(default_factory=list)
    centers: List[Tuple[int, int]] = field(default_factory=list)

    @staticmethod
    def from_board(rust_board: Board, history: Tuple[GameState, ...] = ()) -> "GameState":
        views_flat, centers = rust_board.get_cluster_views()
        # Convert Rust-extracted views to explicitly C-contiguous numpy arrays.
        # This prevents potential deadlocks when these views are passed back
        # to Rust functions that expect safe slice boundaries.
        # get_cluster_views returns 18-plane encoded views (18 * 19 * 19 = 6498 floats).
        views = [np.ascontiguousarray(v, dtype=np.float32).reshape(18, BOARD_SIZE, BOARD_SIZE) for v in views_flat]
        if not views:
            views = [np.zeros((18, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)]
            centers = [(0, 0)]
            
        return GameState(
            current_player=rust_board.current_player,
            moves_remaining=rust_board.moves_remaining,
            zobrist_hash=rust_board.zobrist_hash(),
            ply=rust_board.ply,
            move_history=history,
            views=views,
            centers=centers
        )

    def apply_move(self, rust_board: Board, q: int, r: int) -> "GameState":
        rust_board.apply_move(q, r)
        new_history = (self.move_history + (self,))[-HISTORY_LEN:]
        return GameState.from_board(rust_board, history=new_history)

    def to_tensor(self, rust_board: Board = None) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """Encode the state into K tensors of shape (18, 19, 19).

        get_cluster_views() already returns fully-encoded 18-plane views:
          plane  0:     current player's stones
          planes 1-7:   history slots (zeros until history is wired up)
          plane  8:     opponent's stones
          planes 9-15:  opponent history slots (zeros until history is wired up)
          plane 16:     moves_remaining == 2 flag
          plane 17:     ply parity
        """
        # If rust_board is provided, re-extract views. Otherwise use cached ones.
        if rust_board is not None:
            views_flat, centers = rust_board.get_cluster_views()
            views = [np.ascontiguousarray(v, dtype=np.float32).reshape(18, BOARD_SIZE, BOARD_SIZE)
                     for v in views_flat]
            if not views:
                views = [np.zeros((18, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)]
                centers = [(0, 0)]
        else:
            views = self.views
            centers = self.centers

        K = len(centers)
        tensor = np.empty((K, 18, 19, 19), dtype=np.float16)
        for k in range(K):
            tensor[k] = views[k]

        return tensor, centers

    def __hash__(self) -> int:
        # zobrist_hash is u128; Python's hash() reduces it to Py_hash_t width.
        return hash(self.zobrist_hash)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GameState):
            return False
        return self.zobrist_hash == other.zobrist_hash and self.ply == other.ply
