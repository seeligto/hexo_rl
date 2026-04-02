"""GameState — immutable snapshot of a Hex Tac Toe board position."""

from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple
import numpy as np
from engine import Board
from python.utils.constants import BOARD_SIZE
HISTORY_LEN: int = 8  # AlphaZero uses 8 timesteps (current + 7 prior)

@dataclass(frozen=True)
class GameState:
    current_player: int
    moves_remaining: int
    zobrist_hash: int
    ply: int
    # deque(maxlen=HISTORY_LEN): most-recent state is at the right (index -1).
    move_history: Deque["GameState"] = field(
        default_factory=lambda: deque(maxlen=HISTORY_LEN)
    )
    # Each view is shape (2, BOARD_SIZE, BOARD_SIZE): plane 0 = current player's
    # stones, plane 1 = opponent's stones.  This is what Rust's get_cluster_views()
    # returns — 2 planes, not 18.  to_tensor() assembles the full 18-plane tensor
    # by stacking the current snapshot with historical snapshots.
    views: List[np.ndarray] = field(default_factory=list)
    centers: List[Tuple[int, int]] = field(default_factory=list)

    @staticmethod
    def from_board(
        rust_board: Board,
        history: Optional[Deque["GameState"]] = None,
    ) -> "GameState":
        if history is None:
            history = deque(maxlen=HISTORY_LEN)
        # get_cluster_views returns (list of (2,19,19) float32 numpy arrays, list of centers).
        # Arrays are C-contiguous, created zero-copy in Rust via the numpy crate.
        views, centers = rust_board.get_cluster_views()
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
            centers=centers,
        )

    def apply_move(self, rust_board: Board, q: int, r: int) -> "GameState":
        rust_board.apply_move(q, r)
        new_history: Deque["GameState"] = deque(self.move_history, maxlen=HISTORY_LEN)
        new_history.append(self)
        return GameState.from_board(rust_board, history=new_history)

    def to_tensor(self) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """Encode the state into K tensors of shape (18, 19, 19) float16.

        The 18-plane layout follows AlphaZero's 8-step history encoding:

          plane  0:    current player's stones at t           (views[k][0])
          planes 1–7:  current player's stones at t-1 … t-7  (from move_history)
          plane  8:    opponent's stones at t                 (views[k][1])
          planes 9–15: opponent's stones at t-1 … t-7        (from move_history)
          plane 16:    moves_remaining == 2 flag (0.0 or 1.0, broadcast)
          plane 17:    ply parity (ply % 2, broadcast)

        Planes for timesteps earlier than the start of the game are zeros.

        The Rust self-play loop (game_runner.rs) has no Python history, so it
        expands 2-plane views to 18 planes via encode_18_planes_to_buffer, leaving
        history planes as zeros.  Full history is only available on the Python path
        (worker.py, evaluator.py, pretrain.py) which uses this method.
        """
        current_views = self.views
        centers = self.centers

        K = len(centers)
        tensor = np.zeros((K, 18, BOARD_SIZE, BOARD_SIZE), dtype=np.float16)

        # Scalar planes are identical across all clusters.
        mr_val = np.float16(0.0 if self.moves_remaining == 1 else 1.0)
        ply_val = np.float16(float(self.ply % 2))
        tensor[:, 16, :, :] = mr_val
        tensor[:, 17, :, :] = ply_val

        history = self.move_history  # deque of prior GameStates, oldest first

        for k in range(K):
            # Current timestep
            tensor[k, 0] = current_views[k][0]
            tensor[k, 8] = current_views[k][1]

            # Historical timesteps t-1 … t-7 (deque[-1]=most recent, deque[-t]=t steps back)
            for t in range(1, HISTORY_LEN):
                if t > len(history):
                    break  # no more history; remaining planes stay zero
                prior = history[-t]
                if k < len(prior.views):
                    tensor[k, t]     = prior.views[k][0]  # prior my-stones
                    tensor[k, 8 + t] = prior.views[k][1]  # prior opp-stones
                # if the prior state had fewer clusters, leave the planes as zero

        return tensor, centers

    def __hash__(self) -> int:
        # zobrist_hash is u128; Python's hash() reduces large ints to Py_hash_t width.
        return hash(self.zobrist_hash)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GameState):
            return False
        return self.zobrist_hash == other.zobrist_hash and self.ply == other.ply
