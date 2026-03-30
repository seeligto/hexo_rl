"""Policy projection helpers for multi-cluster self-play targets."""

from __future__ import annotations

import numpy as np


def project_global_policy_to_local(
    board,
    center: tuple[int, int],
    global_policy: np.ndarray,
    board_size: int = 19,
) -> np.ndarray:
    """Project a global policy distribution onto one cluster-local board window."""
    half = (board_size - 1) // 2
    n_actions = board_size * board_size + 1
    local = np.zeros((n_actions,), dtype=np.float32)

    cq, cr = center
    for i in range(board_size):
        for j in range(board_size):
            wq = i - half + cq
            wr = j - half + cr
            mcts_idx = board.to_flat(wq, wr)
            if 0 <= mcts_idx < n_actions - 1:
                local_idx = i * board_size + j
                local[local_idx] = float(global_policy[mcts_idx])

    # Keep pass-move target identical across cluster views.
    local[-1] = float(global_policy[-1])
    s = float(local.sum())
    if s > 1e-9:
        local /= s
    else:
        local.fill(1.0 / n_actions)
    return local
