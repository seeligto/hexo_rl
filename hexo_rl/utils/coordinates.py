"""Hex coordinate helpers shared across Python callers.

Mirrors the Rust counterparts in `engine/src/replay_buffer/sym_tables.rs`
(`from_flat`, `to_flat`) and `engine/src/board/state.rs` (`hex_distance`).
Keep these pure — no numpy, no hexo_rl imports — so that any script or
test can use them without dragging in the full project surface.

## Window-local coordinates

`flat_to_axial` and `axial_to_flat` operate on **window-local** axial
coordinates with origin at the window centre (q, r both in
`[-half, half]`, where `half = (board_size - 1) // 2`). They are the
window-local inverse of each other and match the Rust scatter table
exactly. For cluster-centred global coordinates, compose with a centre
offset at the call site:

    # Global axial → window-local flat for a cluster at (cq, cr):
    flat = axial_to_flat(q - cq, r - cr, board_size)

## Distance

`axial_distance` is the hex Manhattan distance between two axial points.
Accepts int or float tuples (float supports centroid-vs-centroid checks in
`colony_detection`).
"""
from __future__ import annotations

from typing import Optional, Tuple


def flat_to_axial(flat_idx: int, board_size: int) -> Tuple[int, int]:
    """Window-local flat index → axial `(q, r)`.

    Inverse of `axial_to_flat`. Assumes the window is centred on `(0, 0)`
    with extent `[-half, half]` on each axis. Matches
    `SymTables::from_flat` in engine/src/replay_buffer/sym_tables.rs
    byte-exact.

    Args:
        flat_idx: integer in `[0, board_size * board_size)`.
        board_size: odd window side length (19 for the standard board).

    Returns:
        `(q, r)` with `q, r` in `[-half, half]`.
    """
    half = (board_size - 1) // 2
    q = flat_idx // board_size - half
    r = flat_idx % board_size - half
    return q, r


def axial_to_flat(q: int, r: int, board_size: int) -> Optional[int]:
    """Axial `(q, r)` → window-local flat index, or `None` if outside window.

    Inverse of `flat_to_axial`. Returns `None` when the cell lies outside
    the `[-half, half]` axial window, matching the Rust `to_flat` contract.
    """
    half = (board_size - 1) // 2
    wq = q + half
    wr = r + half
    if 0 <= wq < board_size and 0 <= wr < board_size:
        return wq * board_size + wr
    return None


def cell_to_flat(cell_str: str, board_size: int) -> int:
    """Parse a ``"q,r"`` cell string into a window-local flat index.

    Accepts optional surrounding whitespace and parentheses (e.g. ``"(0,0)"``,
    ``" 3, -4 "``). Raises `ValueError` on invalid format or out-of-window
    coordinates. Note this differs from `axial_to_flat` by raising rather
    than returning `None` because a string literal represents caller intent
    that a specific cell exists.
    """
    tok = cell_str.strip().strip("()")
    parts = tok.split(",")
    if len(parts) != 2:
        raise ValueError(f"expected 'q,r', got {cell_str!r}")
    q = int(parts[0].strip())
    r = int(parts[1].strip())
    flat = axial_to_flat(q, r, board_size)
    if flat is None:
        raise ValueError(
            f"cell ({q}, {r}) outside window of size {board_size}"
        )
    return flat


def axial_distance(
    a: Tuple[float, float], b: Tuple[float, float]
) -> int:
    """Hex Manhattan distance between two axial points.

    Equivalent to `max(|dq|, |dr|, |dq + dr|)` — this is the three-axis
    form of the standard axial distance and matches
    `engine/src/board/state.rs::hex_distance` (which uses the
    `(|dq| + |dr| + |ds|) / 2` identity form).

    Accepts `int` or `float` tuples; returns an `int`. For float inputs
    the result is the ceiling of the exact distance — callers that need
    sub-unit precision should compute `max(|dq|, |dr|, |dq + dr|)` directly.
    """
    dq = abs(a[0] - b[0])
    dr = abs(a[1] - b[1])
    ds = abs((a[0] + a[1]) - (b[0] + b[1]))
    return int(max(dq, dr, ds))
