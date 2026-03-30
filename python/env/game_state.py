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
        views = [np.ascontiguousarray(v, dtype=np.float32).reshape(2, BOARD_SIZE, BOARD_SIZE) for v in views_flat]
        if not views:
            views = [np.zeros((2, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)]
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
        """Encode the state into K tensors of shape (18, 19, 19)."""
        # If rust_board is provided, we re-extract views. Otherwise we use cached ones.
        if rust_board is not None:
            views_flat, centers = rust_board.get_cluster_views()
            views = [np.array(v, dtype=np.float32).reshape(2, BOARD_SIZE, BOARD_SIZE) for v in views_flat]
            if not views:
                views = [np.zeros((2, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)]
                centers = [(0, 0)]
        else:
            views = self.views
            centers = self.centers
            
        K = len(centers)
        tensor = np.zeros((K, 18, 19, 19), dtype=np.float16)
        
        for k in range(K):
            planes = views[k]
            tensor[k, 0] = planes[0]
            tensor[k, 8] = planes[1]
            tensor[k, 16] = 0.0 if self.moves_remaining == 1 else 1.0
            tensor[k, 17] = float(self.ply % 2)
            
        return tensor, centers

    def __hash__(self) -> int:
        return self.zobrist_hash

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GameState):
            return False
        return self.zobrist_hash == other.zobrist_hash and self.ply == other.ply
