"""Rolling tensor buffer for zero-allocation history assembly during self-play."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from python.env.game_state import GameState, HISTORY_LEN
from python.selfplay.utils import BOARD_SIZE

_TOTAL_PLANES = 18


class TensorBuffer:
    """Pre-allocated rolling buffer for assembling the (K, 18, 19, 19) network input.

    On first call (or when K changes), allocates and fills from scratch.
    On subsequent calls with the same K, performs an in-place circular shift
    of the history planes and writes only the new current planes 0 and 8,
    avoiding both np.zeros allocation and the 7-step history loop.
    """

    def __init__(self) -> None:
        self._buf: Optional[np.ndarray] = None  # shape (K, 18, 19, 19) float16
        self._K: int = 0

    def reset(self) -> None:
        """Discard the buffer. Call at the start of each new game."""
        self._buf = None
        self._K = 0

    def assemble(self, state: GameState) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """Return the current (K, 18, 19, 19) float16 tensor and cluster centers.

        The returned array is owned by this buffer and will be overwritten on
        the next call. Copy individual slices if you need to store them.
        """
        K = len(state.views)
        centers = state.centers

        if self._buf is None or K != self._K:
            # Full rebuild: new game or cluster count changed.
            self._K = K
            buf = np.zeros((K, _TOTAL_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=np.float16)
            history = state.move_history
            for k in range(K):
                for t in range(1, HISTORY_LEN):
                    if t > len(history):
                        break
                    prior = history[-t]
                    if k < len(prior.views):
                        buf[k, t]     = prior.views[k][0]
                        buf[k, 8 + t] = prior.views[k][1]
            self._buf = buf
        else:
            # Circular shift: push all history planes one step older.
            # planes 1..7 ← 0..6  (my-stones history)
            # planes 9..15 ← 8..14 (opp-stones history)
            buf = self._buf
            buf[:, 1:8, :, :] = buf[:, 0:7, :, :]
            buf[:, 9:16, :, :] = buf[:, 8:15, :, :]

        buf = self._buf
        # Write current-timestep planes (always overwritten).
        for k in range(K):
            buf[k, 0] = state.views[k][0]
            buf[k, 8] = state.views[k][1]

        # Scalar planes (broadcast across K and spatial dims).
        buf[:, 16, :, :] = np.float16(0.0 if state.moves_remaining == 1 else 1.0)
        buf[:, 17, :, :] = np.float16(float(state.ply % 2))

        return buf, centers
