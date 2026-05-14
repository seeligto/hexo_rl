"""Policy-scatter LUTs for 12-fold hex symmetry augmentation.

`scatter[dst_flat] = src_flat` — apply via `new_policy = policy[scatter]`.
Matches the Rust SymTables convention: axial (q, r) with q = flat // N - half,
r = flat % N - half; reflect swaps (q, r) → (r, q); rotate applies
(q, r) → (−r, q+r) n_rot times. sym_idx in [0, 12): syms 0-5 are pure
rotations (0–5×60°); syms 6-11 are reflect-then-rotate.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from hexo_rl.encoding import lookup as _lookup_encoding

BOARD_SIZE: int = _lookup_encoding("v6").board_size

_POLICY_SCATTERS: Optional[List[np.ndarray]] = None


def get_policy_scatters(
    board_size: int = BOARD_SIZE,
    has_pass: bool = True,
) -> List[np.ndarray]:
    """Return 12 policy-scatter index arrays.

    Each array has length `board_size**2 + (1 if has_pass else 0)` and dtype
    int64. When `has_pass` is True (v6 default), index `board_size**2` is
    the pass move and is invariant under all symmetries. Under v8
    (`has_pass=False`, no pass slot), the pass-row scatter entry is omitted
    entirely. Cells whose source maps outside the window fall back to
    `src_flat = board_size**2 - 1` (consistent with the Rust sample_batch
    scatter for out-of-window cells).

    Cached on first call for the canonical default config (v6:
    board_size=BOARD_SIZE, has_pass=True). v8 callers do not hit the cache.
    """
    global _POLICY_SCATTERS
    is_canonical_default = board_size == BOARD_SIZE and has_pass
    if _POLICY_SCATTERS is not None and is_canonical_default:
        return _POLICY_SCATTERS

    N = board_size
    half = (N - 1) // 2
    n_cells = N * N
    n_actions = n_cells + (1 if has_pass else 0)
    scatters: List[np.ndarray] = []

    for sym_idx in range(12):
        reflect = sym_idx >= 6
        n_rot = sym_idx % 6
        scatter = np.full(n_actions, n_cells - 1, dtype=np.int64)
        if has_pass:
            scatter[n_cells] = n_cells  # pass is invariant
        for src in range(n_cells):
            q = src // N - half
            r = src % N - half
            if reflect:
                q, r = r, q
            for _ in range(n_rot):
                q, r = -r, q + r
            dq = q + half
            dr = r + half
            if 0 <= dq < N and 0 <= dr < N:
                scatter[dq * N + dr] = src
        scatters.append(scatter)

    if is_canonical_default:
        _POLICY_SCATTERS = scatters
    return scatters
