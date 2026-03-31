"""Bootstrap dataset: converts raw game records to (tensor, policy, value) triples."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
import structlog
from tqdm import tqdm

from native_core import Board
from python.env.game_state import GameState, BOARD_SIZE

log = structlog.get_logger()

_POLICY_SIZE = BOARD_SIZE * BOARD_SIZE + 1  # 362


def replay_game_to_triples(
    moves: List[Tuple[int, int]],
    winner: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Replay a move sequence and return pre-allocated training arrays.

    Args:
        moves:  Ordered (q, r) sequence for the complete game.
        winner: +1 if player 1 won, -1 if player 2 won.

    Returns:
        states:   float16 array of shape (T, 18, 19, 19)
        policies: float32 array of shape (T, 362)  — one-hot on move played
        outcomes: float32 array of shape (T,)       — ±1 from current player's POV
    """
    max_len = len(moves)
    states   = np.zeros((max_len, 18, BOARD_SIZE, BOARD_SIZE), dtype=np.float16)
    policies = np.zeros((max_len, _POLICY_SIZE),               dtype=np.float32)
    outcomes = np.zeros(max_len,                               dtype=np.float32)
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
            states[t]              = tensor[target_k]   # direct write, no .copy()
            policies[t, target_idx] = 1.0
            outcomes[t]            = 1.0 if state.current_player == winner else -1.0
            t += 1

        try:
            state = state.apply_move(board, q, r)
        except Exception:
            break

    return states[:t], policies[:t], outcomes[:t]


def convert_to_dataset(
    games: List[List[Tuple[int, int]]]
) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    """Convert a list of move sequences into (state_tensor, policy, value) triples.

    Each game is replayed from scratch. The policy for each position is a
    one-hot on the move that was actually played. The value is determined
    retrospectively once the game outcome is known.
    """
    log.info("converting_to_dataset", n_games=len(games))
    dataset = []

    for moves in tqdm(games, desc="Converting"):
        board = Board()
        state = GameState.from_board(board)
        history = []

        for q, r in moves:
            tensor, centers = state.to_tensor()

            # Find which cluster window contains the move being played.
            target_k = -1
            target_local_idx = -1
            for k, (cq, cr) in enumerate(centers):
                wq = q - cq + 9
                wr = r - cr + 9
                if 0 <= wq < 19 and 0 <= wr < 19:
                    target_k = k
                    target_local_idx = wq * 19 + wr
                    break

            if target_k != -1:
                policy = np.zeros(19 * 19 + 1, dtype=np.float32)
                policy[target_local_idx] = 1.0
                history.append((tensor[target_k], policy, board.current_player))

            try:
                state = state.apply_move(board, q, r)
            except Exception:
                break

        winner = board.winner()
        outcome = 0.0
        if winner is not None:
            outcome = float(winner)

        for s_t, p, player in history:
            val = 1.0 if player == outcome else (-1.0 if outcome != 0.0 else 0.0)
            dataset.append((s_t, p, val))

    return dataset


class BootstrapDataset(torch.utils.data.Dataset):
    """PyTorch Dataset wrapping (state_tensor, policy, value) triples."""

    def __init__(self, data: List[Tuple[np.ndarray, np.ndarray, float]]) -> None:
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s, p, v = self.data[idx]
        return torch.from_numpy(s), torch.from_numpy(p), torch.tensor(v, dtype=torch.float32)
