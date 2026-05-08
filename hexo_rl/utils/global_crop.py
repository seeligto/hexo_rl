"""§169 A3 — global summary crop for the K-cluster + PMA + global token branch.

Produces a fixed-size ``(N_GLOBAL_PLANES, CANVAS_SIZE, CANVAS_SIZE)`` view
of the entire stone configuration on the (theoretically infinite) HTTT
board. Used as the input to the ``GlobalTokenEncoder`` branch alongside
the K cluster windows.

Three planes:

  - ``cur``        : 1.0 in cells whose pooled-bbox region contains a
                     current-player stone.
  - ``opp``        : 1.0 in cells whose pooled-bbox region contains an
                     opponent stone.
  - ``canvas_mask``: 1.0 in cells that map to the active bbox region,
                     0.0 in padding cells outside the bbox embedding.
                     This is the §169 P3 prompt's "real-canvas indicator
                     channel" — without it, zero padding inside the
                     32×32 fixed canvas is indistinguishable from
                     "real but empty board cells", and the model would
                     learn off-board features as signal (T2 §E.1
                     pitfall 2 — padding leak).

Algorithm (single canvas size, deterministic):

  bbox = (q_min..q_max, r_min..r_max) over all placed stones.
  side = max(q_max-q_min+1, r_max-r_min+1, 1)
  s    = max(1, ceil(side / canvas_size))      # downsample factor
  pooled_h, pooled_w = ceil(height/s), ceil(width/s)
  # Center the pooled bbox in the (canvas_size × canvas_size) canvas.
  off_q = (canvas_size - pooled_h) // 2
  off_r = (canvas_size - pooled_w) // 2

For early/mid-game positions (the 99% case) ``s=1`` and the bbox embeds
directly. For pathological super-late-game positions where the bbox
grows past 32 the helper max-pools the stone density into the canvas;
the canvas mask still accurately marks active vs padding cells.
"""
from __future__ import annotations

import math
from typing import Sequence, Tuple

import numpy as np


CANVAS_SIZE: int = 32
N_GLOBAL_PLANES: int = 3   # cur, opp, canvas_mask


def compute_global_crop(
    cur_stones: Sequence[Tuple[int, int]],
    opp_stones: Sequence[Tuple[int, int]],
    canvas_size: int = CANVAS_SIZE,
) -> np.ndarray:
    """Build a ``(3, canvas_size, canvas_size)`` global summary tensor.

    Args:
        cur_stones: ``(q, r)`` pairs for the current-to-move player.
        opp_stones: ``(q, r)`` pairs for the other player.
        canvas_size: side length of the fixed canvas. Default 32.

    Returns:
        ``np.ndarray`` of shape ``(3, canvas_size, canvas_size)`` in
        float16 — channels = (cur, opp, canvas_mask).
    """
    out = np.zeros((N_GLOBAL_PLANES, canvas_size, canvas_size), dtype=np.float16)
    if not cur_stones and not opp_stones:
        # Empty board — no canvas region defined; mask stays 0 so the
        # encoder treats the whole canvas as padding.
        return out

    qs = [q for q, _ in cur_stones] + [q for q, _ in opp_stones]
    rs = [r for _, r in cur_stones] + [r for _, r in opp_stones]
    q_min, q_max = min(qs), max(qs)
    r_min, r_max = min(rs), max(rs)
    height = q_max - q_min + 1
    width = r_max - r_min + 1
    side = max(height, width, 1)

    s = max(1, math.ceil(side / canvas_size))
    pooled_h = max(1, math.ceil(height / s))
    pooled_w = max(1, math.ceil(width / s))
    # Clamp in case pooling math drifts past canvas (shouldn't happen given
    # the s ceiling but defensive against off-by-one at canvas_size=1).
    pooled_h = min(pooled_h, canvas_size)
    pooled_w = min(pooled_w, canvas_size)

    off_q = (canvas_size - pooled_h) // 2
    off_r = (canvas_size - pooled_w) // 2

    cur_plane = out[0]
    opp_plane = out[1]
    mask_plane = out[2]

    for q, r in cur_stones:
        wq = off_q + min(pooled_h - 1, (q - q_min) // s)
        wr = off_r + min(pooled_w - 1, (r - r_min) // s)
        cur_plane[wq, wr] = 1.0
    for q, r in opp_stones:
        wq = off_q + min(pooled_h - 1, (q - q_min) // s)
        wr = off_r + min(pooled_w - 1, (r - r_min) // s)
        opp_plane[wq, wr] = 1.0
    mask_plane[off_q:off_q + pooled_h, off_r:off_r + pooled_w] = 1.0
    return out


def compute_global_crop_from_board(
    rust_board,
    canvas_size: int = CANVAS_SIZE,
) -> np.ndarray:
    """Convenience wrapper: pull stones from a live ``engine.Board`` and
    build the global crop in the current-player's frame.

    Reads ``board.get_stones()`` (returns ``[(q, r, player_label)]`` where
    ``player_label`` is +1 / -1) and partitions by ``board.current_player``.
    """
    cur_p = int(rust_board.current_player)
    cur: list[Tuple[int, int]] = []
    opp: list[Tuple[int, int]] = []
    for q, r, p in rust_board.get_stones():
        if int(p) == cur_p:
            cur.append((int(q), int(r)))
        elif int(p) != 0:
            opp.append((int(q), int(r)))
    return compute_global_crop(cur, opp, canvas_size=canvas_size)
