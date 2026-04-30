"""Bootstrap dataset: converts raw game records to (tensor, policy, value) triples."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import structlog

from engine import Board
from hexo_rl.env.game_state import GameState, BOARD_SIZE, _compute_chain_planes

log = structlog.get_logger()

_POLICY_SIZE = BOARD_SIZE * BOARD_SIZE + 1  # 362


def replay_game_to_triples(
    moves: List[Tuple[int, int]],
    winner: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Replay a move sequence and return pre-allocated training arrays.

    Args:
        moves:  Ordered (q, r) sequence for the complete game.
        winner: +1 if player 1 won, -1 if player 2 won.

    Returns:
        states:       float16 array of shape (T, 18, 19, 19)
        chain_planes: float16 array of shape (T, 6, 19, 19)
        policies:     float32 array of shape (T, 362)  — one-hot on move played
        outcomes:     float32 array of shape (T,)       — ±1 from current player's POV
    """
    max_len = len(moves)
    states       = np.zeros((max_len, 18, BOARD_SIZE, BOARD_SIZE), dtype=np.float16)
    chain_planes = np.zeros((max_len,  6, BOARD_SIZE, BOARD_SIZE), dtype=np.float16)
    policies     = np.zeros((max_len, _POLICY_SIZE),               dtype=np.float32)
    outcomes     = np.zeros(max_len,                               dtype=np.float32)
    t = 0

    board = Board()
    state = GameState.from_board(board)

    for q, r in moves:
        tensor, centers = state.to_tensor()  # (K, 18, 19, 19) float16
        target_k = target_idx = -1
        for k, (cq, cr) in enumerate(centers):
            wq = q - cq + 9
            wr = r - cr + 9
            if 0 <= wq < BOARD_SIZE and 0 <= wr < BOARD_SIZE:
                target_k, target_idx = k, wq * BOARD_SIZE + wr
                break

        if target_k >= 0:
            states[t]               = tensor[target_k]   # direct write, no .copy()
            # Chain planes: computed from most-recent stone planes (plane 0 = cur, plane 8 = opp).
            chain_planes[t]         = _compute_chain_planes(
                tensor[target_k, 0].astype(np.float32),
                tensor[target_k, 8].astype(np.float32),
            ).astype(np.float16) / 6.0
            policies[t, target_idx] = 1.0
            outcomes[t]             = 1.0 if state.current_player == winner else -1.0
            t += 1

        try:
            state = state.apply_move(board, q, r)
        except Exception:
            break

    return states[:t], chain_planes[:t], policies[:t], outcomes[:t]
