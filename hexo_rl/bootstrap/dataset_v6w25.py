"""Bootstrap dataset (v6w25 encoding): K-cluster windows at matched-perception
R=8 perception + 25×25 cluster windows + cluster_threshold=8.

Wire format identical to v6 (8 KEPT_PLANE_INDICES planes + pass-slot policy)
but spatial extent is 25×25 instead of 19×19. Used as the matched-perception
A/B baseline against v8 single-bbox in §168 T3.

Replays raw human game moves on a Rust Board configured for v6w25
(`set_cluster_window_size(25)` + `set_cluster_threshold(8)` +
`set_legal_move_radius(8)`) and emits per-ply (8, 25, 25) tensors aligned
with the played move's cluster window — same alignment policy as v6.

§169 A3 — optional ``with_global_crop`` flag emits an additional
``(T, 3, 32, 32)`` float16 array (cur stones / opp stones / canvas-realness
mask) per replayed ply, in the current-to-move player's frame. Used as the
input to the global-summary token branch in PMA-with-global pooling.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import structlog

from engine import Board
from hexo_rl.env.game_state import GameState, _compute_chain_planes
from hexo_rl.utils.constants import KEPT_PLANE_INDICES
from hexo_rl.utils.global_crop import (
    CANVAS_SIZE as GLOBAL_CANVAS_SIZE,
    N_GLOBAL_PLANES,
    compute_global_crop_from_board,
)

log = structlog.get_logger()

# v6w25 dimensions (mirrors engine/src/replay_buffer/sym_tables.rs:96-104).
BOARD_SIZE_V6W25: int = 25
HALF_V6W25: int = (BOARD_SIZE_V6W25 - 1) // 2  # 12
N_PLANES_V6W25: int = 8
N_CELLS_V6W25: int = BOARD_SIZE_V6W25 * BOARD_SIZE_V6W25  # 625
N_ACTIONS_V6W25: int = N_CELLS_V6W25 + 1  # 626 (cells + pass; v6 wire-format compat)
CLUSTER_THRESHOLD_V6W25: int = 8
LEGAL_MOVE_RADIUS_V6W25: int = 8


def _make_v6w25_board() -> Board:
    """Construct a fresh Board configured for v6w25 cluster encoding."""
    board = Board()
    board.set_legal_move_radius(LEGAL_MOVE_RADIUS_V6W25)
    board.set_cluster_threshold(CLUSTER_THRESHOLD_V6W25)
    board.set_cluster_window_size(BOARD_SIZE_V6W25)
    return board


def replay_game_to_triples_v6w25(
    moves: List[Tuple[int, int]],
    winner: int,
    *,
    with_global_crop: bool = False,
) -> Tuple[np.ndarray, ...]:
    """Replay a move sequence and return v6w25 training arrays.

    Args:
        moves:  Ordered (q, r) sequence for the complete game.
        winner: +1 if player 1 won, -1 if player 2 won.
        with_global_crop: §169 A3 — when True, also return a per-ply
            ``(T, 3, 32, 32)`` float16 global-summary crop in the
            current-to-move player's frame. Channels = (cur stones,
            opp stones, canvas-realness mask).

    Returns:
        Default 4-tuple ``(states, chain_planes, policies, outcomes)``:
        states:       float16 array of shape (T, 8, 25, 25) — KEPT_PLANE_INDICES
                      slice of the full 18-plane tensor.
        chain_planes: float16 array of shape (T, 6, 25, 25) — Q13 chain planes.
        policies:     float32 array of shape (T, 626) — one-hot on move played.
        outcomes:     float32 array of shape (T,)     — ±1 from current player's POV.

        With ``with_global_crop=True``, returns the 5-tuple
        ``(states, chain_planes, policies, outcomes, global_crops)`` where
        global_crops is float16 of shape ``(T, 3, 32, 32)``.
    """
    max_len = len(moves)
    states = np.zeros(
        (max_len, N_PLANES_V6W25, BOARD_SIZE_V6W25, BOARD_SIZE_V6W25),
        dtype=np.float16,
    )
    chain_planes = np.zeros(
        (max_len, 6, BOARD_SIZE_V6W25, BOARD_SIZE_V6W25), dtype=np.float16
    )
    policies = np.zeros((max_len, N_ACTIONS_V6W25), dtype=np.float32)
    outcomes = np.zeros(max_len, dtype=np.float32)
    global_crops: Optional[np.ndarray] = None
    if with_global_crop:
        global_crops = np.zeros(
            (max_len, N_GLOBAL_PLANES, GLOBAL_CANVAS_SIZE, GLOBAL_CANVAS_SIZE),
            dtype=np.float16,
        )
    t = 0

    board = _make_v6w25_board()
    state = GameState.from_board(board)

    for q, r in moves:
        full_tensor, centers = state.to_tensor()  # (K, 18, 25, 25) float16
        target_k = -1
        target_idx = -1
        for k, (cq, cr) in enumerate(centers):
            wq = q - cq + HALF_V6W25
            wr = r - cr + HALF_V6W25
            if 0 <= wq < BOARD_SIZE_V6W25 and 0 <= wr < BOARD_SIZE_V6W25:
                target_k = k
                target_idx = wq * BOARD_SIZE_V6W25 + wr
                break

        if target_k >= 0:
            # Slice 18 → 8 planes (KEPT_PLANE_INDICES, v6 wire format).
            states[t] = full_tensor[target_k, KEPT_PLANE_INDICES]
            chain_planes[t] = (
                _compute_chain_planes(
                    full_tensor[target_k, 0].astype(np.float32),
                    full_tensor[target_k, 8].astype(np.float32),
                ).astype(np.float16)
                / 6.0
            )
            policies[t, target_idx] = 1.0
            outcomes[t] = 1.0 if state.current_player == winner else -1.0
            if global_crops is not None:
                # Global crop is computed BEFORE applying the move; uses the
                # live Board's (q, r, player) stones in the current-to-move
                # player's frame so cur/opp matches the cluster-window planes.
                global_crops[t] = compute_global_crop_from_board(board)
            t += 1

        try:
            state = state.apply_move(board, q, r)
        except Exception:
            break

    if global_crops is not None:
        return states[:t], chain_planes[:t], policies[:t], outcomes[:t], global_crops[:t]
    return states[:t], chain_planes[:t], policies[:t], outcomes[:t]
