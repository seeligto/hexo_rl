"""GameState — immutable snapshot of a Hex Tac Toe board position."""

from __future__ import annotations
from dataclasses import dataclass
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

    @staticmethod
    def from_board(rust_board: Board) -> "GameState":
        return GameState(
            current_player=rust_board.current_player,
            moves_remaining=rust_board.moves_remaining,
            zobrist_hash=rust_board.zobrist_hash(),
            ply=rust_board.ply,
        )

    def apply_move(self, rust_board: Board, q: int, r: int) -> "GameState":
        rust_board.apply_move(q, r)
        return GameState.from_board(rust_board)

    def to_tensor(self, rust_board: Board) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """Encode the state into K tensors of shape (18, 19, 19)."""
        views, centers = rust_board.get_cluster_views()
        K = len(centers)
        if K == 0:
            K = 1
            views = [[0.0] * (2 * 19 * 19)]
            centers = [(0, 0)]
            
        tensor = np.zeros((K, 18, 19, 19), dtype=np.float16)
        
        for k in range(K):
            planes = np.array(views[k], dtype=np.float32).reshape(2, 19, 19)
            tensor[k, 0] = planes[0]
            tensor[k, 8] = planes[1]
            tensor[k, 16] = 0.0 if self.moves_remaining == 1 else 1.0
            tensor[k, 17] = float(self.ply % 2)
            
        return tensor, centers
