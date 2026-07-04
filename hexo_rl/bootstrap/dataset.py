"""Bootstrap dataset: converts raw game records to (tensor, policy, value) triples."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import structlog

from engine import Board
from hexo_rl.env.game_state import GameState, BOARD_SIZE, HISTORY_LEN, _compute_chain_planes

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
                tensor[target_k, HISTORY_LEN].astype(np.float32),
            ).astype(np.float16) / 6.0
            policies[t, target_idx] = 1.0
            outcomes[t]             = 1.0 if state.current_player == winner else -1.0
            t += 1

        try:
            state = state.apply_move(board, q, r)
        except Exception:
            break

    return states[:t], chain_planes[:t], policies[:t], outcomes[:t]


def replay_game_to_triples_ls(
    moves: List[Tuple[int, int]],
    winner: int,
    *,
    kept_plane_indices: Sequence[int],
    policy_size: int,
    k_max: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Legal-set (``v6_live2_ls``) per-cluster-row replay — NO off-window drop.

    §D-MULTICLUSTER-S0 Tier-2 variant-(b) (dmulticluster_362_legalset_design.md
    §4): scatter the played move across **ALL containing windows** — one
    dense-``policy_size`` one-hot row per cluster window that geometrically
    contains the move. This is the corpus-side mirror of the Rust
    ``legal_set_scatter_max`` per-cluster-local-362 buffer rows.

    Semantic delta vs :func:`replay_game_to_triples` (the v6/v6_live2 path):
    the v6 path emits ONE row per ply — the FIRST containing window — and
    drops every other window's view of the move (off-window-DROP supervision:
    zero probability mass on the move's cell in every non-first window frame).
    Here a move near/beyond one window's 19×19 extent still gets probability
    mass in whichever window(s) contain it. A ply whose move lies outside ALL
    cluster windows is still skipped (no representable dense target — matches
    the v6 path).

    Args:
        moves:  Ordered (q, r) sequence for the complete game.
        winner: +1 if player 1 won, -1 if player 2 won.
        kept_plane_indices: registry ``kept_plane_indices`` — slice of the
            18-plane tensor to emit (v6_live2_ls = [0, 8, 16, 17]).
        policy_size: registry ``policy_logit_count`` (v6_live2_ls = 362).
        k_max: registry ``k_max`` — cap on cluster views considered per ply
            (v6_live2_ls = 8; mirrors the multi-window bundle's per-leaf cap).

    Returns:
        states:    float16 (R, len(kept_plane_indices), S, S) — one row per
                   (ply, containing-window) pair, R >= number of emitted plies
        policies:  float32 (R, policy_size) — one-hot at the window-LOCAL cell
        outcomes:  float32 (R,) — ±1 from current player's POV
        ply_index: int32   (R,) — ORIGINAL ply index of each row (rows for the
                   same ply are consecutive; use this to select sampled plies —
                   unlike the v6 path there is no positional row/ply identity)
    """
    kept = list(kept_plane_indices)
    states_rows: List[np.ndarray] = []
    target_rows: List[int] = []
    outcome_rows: List[float] = []
    ply_rows: List[int] = []

    board = Board()
    state = GameState.from_board(board)

    for ply, (q, r) in enumerate(moves):
        tensor, centers = state.to_tensor()  # (K, 18, S, S) float16
        _, _, H, W = tensor.shape
        half = (H - 1) // 2
        outcome = 1.0 if state.current_player == winner else -1.0
        for k, (cq, cr) in enumerate(centers[:k_max]):
            wq = q - cq + half
            wr = r - cr + half
            if 0 <= wq < H and 0 <= wr < W:
                states_rows.append(tensor[k][kept])  # slice 18→len(kept) planes
                target_rows.append(wq * W + wr)
                outcome_rows.append(outcome)
                ply_rows.append(ply)

        try:
            state = state.apply_move(board, q, r)
        except Exception:
            break

    n_rows = len(states_rows)
    if n_rows == 0:
        s_dim = BOARD_SIZE
        return (
            np.zeros((0, len(kept), s_dim, s_dim), dtype=np.float16),
            np.zeros((0, policy_size), dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.int32),
        )

    states = np.stack(states_rows).astype(np.float16, copy=False)
    policies = np.zeros((n_rows, policy_size), dtype=np.float32)
    policies[np.arange(n_rows), np.asarray(target_rows, dtype=np.int64)] = 1.0
    outcomes = np.asarray(outcome_rows, dtype=np.float32)
    ply_index = np.asarray(ply_rows, dtype=np.int32)
    return states, policies, outcomes, ply_index
