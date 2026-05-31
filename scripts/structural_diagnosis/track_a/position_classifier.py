"""§S181 Track A — position classifier shared across A1/A3/A5 subtasks.

Classifies a single position as 'colony', 'extension', or 'neither'.

Classifier definition (EXPLICIT, reproducible):

  - 'extension': there exists a player whose stones include an open run of
    >= 4 same-color stones along one of the 3 hex axes, with both flanking
    cells empty (off-window cells are treated as empty — the hex board is
    infinite; the 19x19 plane is a cropped window).

  - 'colony': total stone count >= MIN_STONES (default 8), no open 4+ run
    for either player, and the mean hex distance from the stone centroid
    to each stone is <= MAX_MEAN_HEX_DIST (default 2.5). The threshold is
    calibrated on the T3 40-position bank: every constructed colony
    (stages 6-44) lands at <= 2.5; every extension construction (run+far
    filler) lands at >= 4.5. Boundary is generous (~2 sigma) on bank.

  - 'neither': otherwise.

Reads v6 / v7 / v7full state planes shape (8, 19, 19): plane 0 is
current-player stones (t0), plane 4 is opponent stones (t0). The (i, j)
plane index maps to axial (q, r) via q = i - 9, r = j - 9 (engine
flat_idx convention, trunk_sz=19, half=9). The 3 distinct hex axes are
(1, 0), (1, -1), (0, -1).
"""
from __future__ import annotations
from typing import Tuple

import numpy as np

HEX_AXES: tuple[tuple[int, int], ...] = ((1, 0), (1, -1), (0, -1))
TRUNK_SZ = 19
HALF = 9
MIN_STONES = 8
MAX_MEAN_HEX_DIST = 2.7
OPEN_RUN_LEN = 4


def _find_max_open_run(player_mask: np.ndarray,
                       occ_mask: np.ndarray) -> Tuple[int, bool]:
    """Longest open run of `player_mask` cells along 3 hex axes.

    Returns (length, both_ends_open). A run is "open" iff both flanking
    cells (one step further along the axis, in each direction) are empty.
    Off-window cells count as empty (infinite hex board)."""
    H, W = player_mask.shape
    max_len = 0
    max_open = False
    for di, dj in HEX_AXES:
        visited = np.zeros((H, W), dtype=bool)
        for i in range(H):
            for j in range(W):
                if not player_mask[i, j] or visited[i, j]:
                    continue
                # walk back to find start of run
                si, sj = i, j
                while (0 <= si - di < H and 0 <= sj - dj < W
                       and player_mask[si - di, sj - dj]):
                    si -= di
                    sj -= dj
                # walk forward marking visited, count length
                length = 0
                ei, ej = si, sj
                while (0 <= ei < H and 0 <= ej < W and player_mask[ei, ej]):
                    visited[ei, ej] = True
                    length += 1
                    ei += di
                    ej += dj
                # ei,ej = one past last stone; si,sj = first stone
                before_i, before_j = si - di, sj - dj
                after_i, after_j = ei, ej
                # Off-window → treated as empty (infinite hex board).
                before_open = (not (0 <= before_i < H and 0 <= before_j < W)
                               or not occ_mask[before_i, before_j])
                after_open = (not (0 <= after_i < H and 0 <= after_j < W)
                              or not occ_mask[after_i, after_j])
                is_open = before_open and after_open
                # Prefer longer runs; among equal-length, prefer open ones.
                if length > max_len or (length == max_len and is_open
                                         and not max_open):
                    max_len = length
                    max_open = is_open
    return max_len, max_open


def _mean_hex_dist_from_centroid(occ_mask: np.ndarray) -> float:
    """Mean hex distance from stone centroid (axial coords)."""
    rows, cols = np.where(occ_mask)
    if len(rows) == 0:
        return 0.0
    cent_i = rows.mean()
    cent_j = cols.mean()
    dq = rows - cent_i
    dr = cols - cent_j
    dists = (np.abs(dq) + np.abs(dr) + np.abs(dq + dr)) / 2.0
    return float(dists.mean())


def classify_state(state: np.ndarray,
                   min_stones: int = MIN_STONES,
                   max_mean_hex_dist: float = MAX_MEAN_HEX_DIST,
                   open_run_len: int = OPEN_RUN_LEN,
                   cur_slot: int = 0,
                   opp_slot: int = 4) -> str:
    """Classify a single (n_planes, H, W) state plane tensor.

    `cur_slot` / `opp_slot` are the current/opponent t0 kept-slot indices.
    They default to v6 (0 / 4); pass the registry-derived slots for a
    non-v6 array (v6_live2 opp is at slot 1, not 4 — §P5-CT L65 class).

    Decision order:
      1. COLONY first: tight cluster (mean hex dist <= max_mean_hex_dist)
         AND >= min_stones. This dominates over incidental internal runs
         that exist inside a packed blob (a tight blob with an internal
         4-run is structurally colony — the run is flanked by other
         blob stones, not free space).
      2. EXTENSION otherwise: any player has open run >= open_run_len
         along a hex axis with empty flanks.
      3. NEITHER if neither test fires."""
    cp_mask = np.asarray(state[cur_slot]) > 0.5
    op_mask = np.asarray(state[opp_slot]) > 0.5
    occ_mask = cp_mask | op_mask
    n_stones = int(occ_mask.sum())
    # COLONY test (precedes extension to handle blob-with-internal-run case)
    if n_stones >= min_stones:
        mean_dist = _mean_hex_dist_from_centroid(occ_mask)
        if mean_dist <= max_mean_hex_dist:
            return 'colony'
    # EXTENSION test
    for player_mask in (cp_mask, op_mask):
        max_run, max_open = _find_max_open_run(player_mask, occ_mask)
        if max_run >= open_run_len and max_open:
            return 'extension'
    return 'neither'


def classify_batch(states: np.ndarray, **kwargs) -> np.ndarray:
    """Classify a batch of states. states: (B, n_planes, H, W).
    Returns array of dtype=object with strings."""
    out = np.empty(states.shape[0], dtype=object)
    for i in range(states.shape[0]):
        out[i] = classify_state(states[i], **kwargs)
    return out


# ── Adapter for T3-bank Board-based positions ────────────────────────────
def classify_board(board, min_stones: int = MIN_STONES,
                   max_mean_hex_dist: float = MAX_MEAN_HEX_DIST,
                   open_run_len: int = OPEN_RUN_LEN) -> str:
    """Classify directly from an `engine.Board` instance.

    Builds occupancy + per-player masks from `board.get_stones()`. Same
    decision order as `classify_state`: colony first, then extension."""
    cp_mask = np.zeros((TRUNK_SZ, TRUNK_SZ), dtype=bool)
    op_mask = np.zeros((TRUNK_SZ, TRUNK_SZ), dtype=bool)
    occ_mask = np.zeros((TRUNK_SZ, TRUNK_SZ), dtype=bool)
    cur_player = board.current_player
    for q, r, p in board.get_stones():
        i, j = q + HALF, r + HALF
        # Off-window stones (board is infinite) are dropped from the mask.
        if not (0 <= i < TRUNK_SZ and 0 <= j < TRUNK_SZ):
            continue
        occ_mask[i, j] = True
        if p == cur_player:
            cp_mask[i, j] = True
        else:
            op_mask[i, j] = True
    n_in_window = int(occ_mask.sum())
    # COLONY first (subsumes blob-with-internal-run)
    if n_in_window >= min_stones:
        mean_dist = _mean_hex_dist_from_centroid(occ_mask)
        if mean_dist <= max_mean_hex_dist:
            return 'colony'
    # EXTENSION
    for player_mask in (cp_mask, op_mask):
        max_run, max_open = _find_max_open_run(player_mask, occ_mask)
        if max_run >= open_run_len and max_open:
            return 'extension'
    return 'neither'


__all__ = [
    "classify_state", "classify_board", "classify_batch",
    "HEX_AXES", "TRUNK_SZ", "HALF", "MIN_STONES",
    "MAX_MEAN_HEX_DIST", "OPEN_RUN_LEN",
]
