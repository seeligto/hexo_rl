"""GameState — immutable snapshot of a Hex Tac Toe board position."""

from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple
import numpy as np
from engine import Board
from hexo_rl.utils.constants import BOARD_SIZE
HISTORY_LEN: int = 8  # AlphaZero uses 8 timesteps (current + 7 prior)

# Chain-length plane encoding (Q13). Mirror of engine/src/board/state.rs:48-52.
# Plane layout within the 6-plane block: [a0_cur, a0_opp, a1_cur, a1_opp, a2_cur, a2_opp].
_HEX_AXES: Tuple[Tuple[int, int], ...] = ((1, 0), (0, 1), (1, -1))
_CHAIN_CAP: int = 6  # win target; also the saturation cap per the literature review


def _shift_zero_pad(arr: np.ndarray, dq: int, dr: int) -> np.ndarray:
    """Translate a (H,W) array by (dq,dr) with zero padding. NOT np.roll — no wrap.

    Window edges behave as opaque: cells translated off-grid become 0, which
    terminates any run counting walk that reaches them (Q13 spec §"opaque edges").
    """
    H, W = arr.shape
    out = np.zeros_like(arr)
    qs, qe = max(0, dq), min(H, H + dq)
    rs, re = max(0, dr), min(W, W + dr)
    if qs >= qe or rs >= re:
        return out
    src_qs, src_qe = qs - dq, qe - dq
    src_rs, src_re = rs - dr, re - dr
    out[qs:qe, rs:re] = arr[src_qs:src_qe, src_rs:src_re]
    return out


def _run_batched(
    stones: np.ndarray, dq: int, dr: int, scratch: np.ndarray
) -> np.ndarray:
    """Batched run count over a (2, H, W) bool stack (cur, opp layers).

    Returns (2, H, W) int8: for each layer, each cell holds the count of
    consecutive own-layer stones at positions (q+k·dq, r+k·dr) for
    k = 1.._CHAIN_CAP−1, stopping at first non-own (including window edge).
    `scratch` is a pre-allocated (2, H, W) bool buffer.
    """
    H = BOARD_SIZE
    W = BOARD_SIZE
    out = np.zeros((2, H, W), dtype=np.int8)
    alive = np.ones((2, H, W), dtype=bool)
    for step in range(1, _CHAIN_CAP):
        sdq = step * dq
        sdr = step * dr
        scratch.fill(False)
        qs = 0 if sdq >= 0 else -sdq
        qe = H - sdq if sdq >= 0 else H
        rs = 0 if sdr >= 0 else -sdr
        re = W - sdr if sdr >= 0 else W
        if qs < qe and rs < re:
            scratch[:, qs:qe, rs:re] = stones[
                :, qs + sdq : qe + sdq, rs + sdr : re + sdr
            ]
        alive &= scratch
        out += alive
    return out


def _chain_plane_for_axis(
    own: np.ndarray,
    opp: np.ndarray,
    dq: int,
    dr: int,
    scratch: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute one chain-length plane for a single (axis, player) combination.

    Retained as a standalone helper for unit tests; production path goes
    through the batched `_compute_chain_planes`. See docstring there.
    """
    if scratch is None:
        scratch = np.empty((2, BOARD_SIZE, BOARD_SIZE), dtype=bool)
    stones = np.empty((2, BOARD_SIZE, BOARD_SIZE), dtype=bool)
    stones[0] = own > 0
    stones[1] = opp > 0

    pos_run = _run_batched(stones, dq, dr, scratch)
    neg_run = _run_batched(stones, -dq, -dr, scratch)

    # We only need layer 0 (own) here.
    total = pos_run[0] + neg_run[0]
    value = total.copy()
    eligible = stones[0] | (total > 0)
    value[eligible] += 1
    np.minimum(value, np.int8(_CHAIN_CAP), out=value)
    value[stones[1]] = 0
    return value


def _compute_chain_planes(
    cur_stones: np.ndarray, opp_stones: np.ndarray
) -> np.ndarray:
    """Compute the 6 chain-length planes for one cluster window.

    Layout: planes[0,1] = axis (1,0) cur/opp; planes[2,3] = axis (0,1) cur/opp;
    planes[4,5] = axis (1,-1) cur/opp. Matches Rust HEX_AXES order in
    engine/src/board/state.rs:48-52. Values are int8 in [0, _CHAIN_CAP];
    callers normalize by /_CHAIN_CAP at tensor cast time.

    Batched implementation: for each of the 3 axes, compute pos/neg runs for
    both players simultaneously via a (2, H, W) stone stack. Avoids per-plane
    allocation overhead and per-step array creation.
    """
    H = BOARD_SIZE
    planes = np.zeros((6, H, H), dtype=np.int8)
    stones = np.empty((2, H, H), dtype=bool)
    stones[0] = cur_stones > 0
    stones[1] = opp_stones > 0
    scratch = np.empty((2, H, H), dtype=bool)
    # "Opponent mask per layer" for zeroing out opponent cells at the end:
    # layer 0 (cur-chain) zeroes where stones[1] is True; layer 1 (opp-chain)
    # zeroes where stones[0] is True.
    opp_mask = np.empty((2, H, H), dtype=bool)
    opp_mask[0] = stones[1]
    opp_mask[1] = stones[0]

    for axis_idx, (dq, dr) in enumerate(_HEX_AXES):
        pos_run = _run_batched(stones, dq, dr, scratch)
        neg_run = _run_batched(stones, -dq, -dr, scratch)
        total = pos_run
        total += neg_run  # in-place: total now holds sum
        eligible = stones | (total > 0)
        total[eligible] += 1
        np.minimum(total, np.int8(_CHAIN_CAP), out=total)
        total[opp_mask] = 0
        planes[2 * axis_idx] = total[0]
        planes[2 * axis_idx + 1] = total[1]

    return planes

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
    # returns — 2 planes, not 24.  to_tensor() assembles the full 24-plane tensor
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
        """Encode the state into K tensors of shape (24, 19, 19) float16.

        24-plane layout:

          plane  0:     current player's stones at t           (views[k][0])
          planes 1-7:   current player's stones at t-1 … t-7   (from move_history)
          plane  8:     opponent's stones at t                 (views[k][1])
          planes 9-15:  opponent's stones at t-1 … t-7         (from move_history)
          plane 16:     moves_remaining == 2 flag (broadcast)
          plane 17:     ply parity (ply % 2, broadcast)
          planes 18-23: Q13 chain-length planes, computed from current-step
                        stones only (no temporal stacking). Layout
                        [a0_cur, a0_opp, a1_cur, a1_opp, a2_cur, a2_opp],
                        /_CHAIN_CAP-normalised so values lie in [0, 1].

        Planes for timesteps earlier than the start of the game are zeros.

        The Rust self-play loop (game_runner.rs) has no Python history, so it
        expands 2-plane views to 24 planes via encode_state_to_buffer, leaving
        history planes as zeros. Full history is only available on the Python
        path (worker.py, evaluator.py, pretrain.py) which uses this method.
        """
        current_views = self.views
        centers = self.centers

        K = len(centers)
        tensor = np.zeros((K, 24, BOARD_SIZE, BOARD_SIZE), dtype=np.float16)

        # Scalar planes are identical across all clusters.
        mr_val = np.float16(0.0 if self.moves_remaining == 1 else 1.0)
        ply_val = np.float16(float(self.ply % 2))
        tensor[:, 16, :, :] = mr_val
        tensor[:, 17, :, :] = ply_val

        history = self.move_history  # deque of prior GameStates, oldest first

        for k in range(K):
            # Current timestep
            cur_stones = current_views[k][0]
            opp_stones = current_views[k][1]
            tensor[k, 0] = cur_stones
            tensor[k, 8] = opp_stones

            # Historical timesteps t-1 … t-7 (deque[-1]=most recent, deque[-t]=t steps back)
            for t in range(1, HISTORY_LEN):
                if t > len(history):
                    break  # no more history; remaining planes stay zero
                prior = history[-t]
                if k < len(prior.views):
                    tensor[k, t]     = prior.views[k][0]  # prior my-stones
                    tensor[k, 8 + t] = prior.views[k][1]  # prior opp-stones
                # if the prior state had fewer clusters, leave the planes as zero

            # Q13 chain-length planes — current-step only, no temporal stacking.
            chain_i8 = _compute_chain_planes(cur_stones, opp_stones)
            tensor[k, 18:24] = chain_i8.astype(np.float16) / np.float16(_CHAIN_CAP)

        return tensor, centers

    def __hash__(self) -> int:
        # zobrist_hash is u128; Python's hash() reduces large ints to Py_hash_t width.
        return hash(self.zobrist_hash)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GameState):
            return False
        return self.zobrist_hash == other.zobrist_hash and self.ply == other.ply
