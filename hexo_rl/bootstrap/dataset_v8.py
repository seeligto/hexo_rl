"""Bootstrap dataset (v8 encoding): replays a game and emits 11-plane bbox
tensors for v8 pretrain.

v8 encoding (Path β, locked at §165 + §166):
- 25×25 fixed-max bbox-of-all-stones spatial extent
- 11 planes (8 stone-history + off_window + 2 broadcast scalars)
- R=8 perception (legal_move_radius=8 — bundles P2 hotfix-(c))
- N_ACTIONS=625 (no pass slot)
- K-aggregation REMOVED (single bbox per ply replaces 1-6 cluster windows)

This module is a parallel implementation of `hexo_rl.bootstrap.dataset`;
the v6 dataset module is unchanged. Routing happens in
`hexo_rl.bootstrap.pretrain` based on `EncodingSpec.version`.

Contract: `docs/designs/encoding_v8_contract.md` §2 (plane schema), §3
(NPZ schema v8).
"""
from __future__ import annotations

from collections import deque
from typing import Deque, List, Tuple

import numpy as np
import structlog

from engine import Board
from hexo_rl.env.game_state import _compute_chain_planes
from hexo_rl.utils import constants as _c

log = structlog.get_logger()

# v8 dimensions (mirrors hexo_rl.utils.constants.* but inlined to avoid runtime
# encoding-spec resolution in the hot encode path).
BOARD_SIZE_V8: int = _c.BOARD_SIZE_V8  # 25
HALF_V8: int = (BOARD_SIZE_V8 - 1) // 2  # 12
N_PLANES_V8: int = _c.BUFFER_CHANNELS_V8  # 11
N_CELLS_V8: int = _c.NUM_CELLS_V8  # 625
N_ACTIONS_V8: int = _c.N_ACTIONS_V8  # 625 (no pass slot)
LEGAL_MOVE_RADIUS_V8: int = _c.LEGAL_MOVE_RADIUS_V8  # 8

# v8 history depth: 4 plies (ply T, T-1, T-2, T-3) for both current player and
# opponent → 8 stone planes (matches v6 KEPT_PLANE_INDICES depth).
HISTORY_LEN_V8: int = 4

# Game-length cap used to normalize the moves_remaining_bcast scalar plane.
# Matches selfplay.yaml mcts.bench_max_moves and roughly the typical max ply
# for a complete HTTT game; the exact value is a free parameter the model
# learns to handle. v8 contract §2 plane 9: (MAX_MOVES − ply) / MAX_MOVES.
MAX_MOVES_V8: int = 200

# Plane 8 off_window mask is precomputed once per module load (it depends only
# on bbox-window geometry, not stone positions or ply).
_OFF_WINDOW_MASK: np.ndarray = None  # type: ignore[assignment]


def _build_off_window_mask() -> np.ndarray:
    """Precompute the off_window indicator: 1.0 outside dilated hex of radius
    LEGAL_MOVE_RADIUS_V8 around the window centre, 0.0 inside.
    """
    mask = np.ones((BOARD_SIZE_V8, BOARD_SIZE_V8), dtype=np.float16)
    for wq in range(BOARD_SIZE_V8):
        for wr in range(BOARD_SIZE_V8):
            lq = wq - HALF_V8
            lr = wr - HALF_V8
            ls = -(lq + lr)
            dist = max(abs(lq), abs(lr), abs(ls))
            if dist <= LEGAL_MOVE_RADIUS_V8:
                mask[wq, wr] = 0.0
    return mask


def _get_off_window_mask() -> np.ndarray:
    global _OFF_WINDOW_MASK
    if _OFF_WINDOW_MASK is None:
        _OFF_WINDOW_MASK = _build_off_window_mask()
    return _OFF_WINDOW_MASK


def _compute_bbox_centroid(
    stones: List[Tuple[int, int, int]],
) -> Tuple[int, int]:
    """Integer-truncated centroid of the stone-set bbox.

    Empty board → (0, 0) per contract §2.1 step 5 (window centred on origin).
    """
    if not stones:
        return 0, 0
    qs = [s[0] for s in stones]
    rs = [s[1] for s in stones]
    cq = (min(qs) + max(qs)) // 2
    cr = (min(rs) + max(rs)) // 2
    return cq, cr


def _scatter_stones_to_plane(
    plane: np.ndarray,
    stones: List[Tuple[int, int]],
    cq: int,
    cr: int,
) -> int:
    """Project stones at axial (q, r) into a 25×25 window centred on (cq, cr).

    Returns the count of stones that fell outside the window (telemetry for
    the bbox_clip_fired event, contract §2.1 outlier handling).
    """
    clipped = 0
    for sq, sr in stones:
        wq = sq - cq + HALF_V8
        wr = sr - cr + HALF_V8
        if 0 <= wq < BOARD_SIZE_V8 and 0 <= wr < BOARD_SIZE_V8:
            plane[wq, wr] = 1.0
        else:
            clipped += 1
    return clipped


def encode_position_v8(
    board_stones: List[Tuple[int, int, int]],
    cur_player: int,
    history: Deque[Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]],
    ply: int,
    moves_remaining: int,
) -> Tuple[np.ndarray, Tuple[int, int], int]:
    """Encode one position into an (11, 25, 25) v8 tensor.

    Args:
        board_stones: full stone list at this ply, `[(q, r, player), ...]`.
        cur_player: ±1, player about to move at this ply.
        history: deque of `(cur_stones_xy, opp_stones_xy)` snapshots — one
            entry per prior ply, in chronological order (oldest left, most
            recent right). Up to `HISTORY_LEN_V8 - 1 = 3` entries used.
        ply: current ply index (0 = pre-first-move).
        moves_remaining: `Board.moves_remaining` for plane 9 normalization.

    Returns:
        `(tensor, (cq, cr), n_clipped)`:
        - tensor: (11, 25, 25) float16 v8 encoding
        - (cq, cr): bbox centroid used for window centring (caller projects
          target moves through the same offset)
        - n_clipped: count of stones that fell outside the 25×25 envelope
          (contract §2.1: trigger `bbox_clip_fired` telemetry if > 1% of
          plies in a sample)
    """
    cur_stones = [(sq, sr) for (sq, sr, p) in board_stones if p == cur_player]
    opp_stones = [(sq, sr) for (sq, sr, p) in board_stones if p != cur_player]

    cq, cr = _compute_bbox_centroid(board_stones)

    tensor = np.zeros((N_PLANES_V8, BOARD_SIZE_V8, BOARD_SIZE_V8), dtype=np.float16)
    n_clipped = 0

    # Plane 0: current player's stones at ply T
    n_clipped += _scatter_stones_to_plane(tensor[0], cur_stones, cq, cr)

    # Plane 4: opponent's stones at ply T
    n_clipped += _scatter_stones_to_plane(tensor[4], opp_stones, cq, cr)

    # Planes 1-3: current-player history at T-1, T-2, T-3
    # Planes 5-7: opponent-history at T-1, T-2, T-3
    # history[-1] = ply T-1 (most recent), history[-2] = T-2, history[-3] = T-3
    for depth in range(1, HISTORY_LEN_V8):
        if depth > len(history):
            break
        h_cur, h_opp = history[-depth]
        n_clipped += _scatter_stones_to_plane(tensor[depth], h_cur, cq, cr)
        n_clipped += _scatter_stones_to_plane(tensor[4 + depth], h_opp, cq, cr)

    # Plane 8: off_window indicator (1.0 outside dilated hex, 0.0 inside)
    np.copyto(tensor[8], _get_off_window_mask())

    # Plane 9: moves_remaining_bcast (normalized scalar broadcast)
    # Contract §2 plane 9: (MAX_MOVES − ply) / MAX_MOVES, clamped to [0, 1].
    mr_normalized = max(0.0, min(1.0, (MAX_MOVES_V8 - ply) / MAX_MOVES_V8))
    tensor[9, :, :] = np.float16(mr_normalized)

    # Plane 10: ply_parity_bcast (0/1 broadcast)
    tensor[10, :, :] = np.float16(float(ply % 2))

    return tensor, (cq, cr), n_clipped


def replay_game_to_triples_v8(
    moves: List[Tuple[int, int]],
    winner: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Replay a move sequence and return v8-encoded training arrays.

    Args:
        moves:  Ordered (q, r) sequence for the complete game.
        winner: +1 if player 1 won, -1 if player 2 won.

    Returns:
        states:       float16 array of shape (T, 11, 25, 25)
        chain_planes: float16 array of shape (T,  6, 25, 25)
        policies:     float32 array of shape (T, 625)  — one-hot on move played
        outcomes:     float32 array of shape (T,)       — ±1 from current player's POV
        n_clipped:    total count of stones clipped by the 25×25 envelope
                      across the whole game (telemetry; should be ≪ T × stone_count)
    """
    max_len = len(moves)
    states = np.zeros(
        (max_len, N_PLANES_V8, BOARD_SIZE_V8, BOARD_SIZE_V8), dtype=np.float16
    )
    chain_planes = np.zeros(
        (max_len, 6, BOARD_SIZE_V8, BOARD_SIZE_V8), dtype=np.float16
    )
    policies = np.zeros((max_len, N_ACTIONS_V8), dtype=np.float32)
    outcomes = np.zeros(max_len, dtype=np.float32)
    t = 0
    total_clipped = 0

    # legal_move_radius irrelevant for replay (apply_move accepts any
     # in-bounds cell regardless of MCTS legal-move radius). R=8 wiring lives
     # in Bucket D self-play / eval paths, not in the corpus encoder.
    board = Board()
    history: Deque[Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]] = deque(
        maxlen=HISTORY_LEN_V8 - 1
    )

    for q, r in moves:
        board_stones = board.get_stones()
        cur_player = board.current_player
        ply = board.ply
        moves_rem = board.moves_remaining

        tensor, (cq, cr), n_clipped = encode_position_v8(
            board_stones, cur_player, history, ply, moves_rem
        )
        total_clipped += n_clipped

        # Project target move (q, r) into the same 25×25 window
        wq = q - cq + HALF_V8
        wr = r - cr + HALF_V8
        if 0 <= wq < BOARD_SIZE_V8 and 0 <= wr < BOARD_SIZE_V8:
            target_idx = wq * BOARD_SIZE_V8 + wr
            states[t] = tensor
            # Chain planes from current/opponent stone planes (plane 0, plane 4)
            chain_planes[t] = (
                _compute_chain_planes(
                    tensor[0].astype(np.float32),
                    tensor[4].astype(np.float32),
                ).astype(np.float16)
                / 6.0
            )
            policies[t, target_idx] = 1.0
            outcomes[t] = 1.0 if cur_player == winner else -1.0
            t += 1
        # If target out of window, drop position (rare under R=8 + 25×25)

        # Snapshot pre-move state into history for the NEXT iteration
        cur_stones_xy = [(sq, sr) for (sq, sr, p) in board_stones if p == cur_player]
        opp_stones_xy = [(sq, sr) for (sq, sr, p) in board_stones if p != cur_player]
        history.append((cur_stones_xy, opp_stones_xy))

        # Apply move; bail on illegal-move errors (matches v6 dataset behavior)
        try:
            board.apply_move(q, r)
        except Exception:
            break

    return states[:t], chain_planes[:t], policies[:t], outcomes[:t], total_clipped
