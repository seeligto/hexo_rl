"""Pure geometry/scoring helpers for the off-window adversary (D-EXPLOIT Phase 1).

No engine Board, no NN — these reason over plain stone sets and axial coords so the
exploit-construction logic (build a 6-line whose completion is off-window) is honestly
unit-tested. Window-centre / off-window flags come from the engine (via
``hexo_rl.diagnostics.forced_win_detector``) — NOT reimplemented here.

EVAL-PATH ONLY: imported solely by the adversary bot + its tests; never by the engine
hot path or training (pinned by ``tests/test_offwindow_adversary_eval_path_only.py``).
"""
from __future__ import annotations

from typing import Callable, Sequence

# Hex-topology axis basis (q, r) — fixed by the game grid, not tunable geometry.
HEX_AXES = [(1, 0), (0, 1), (1, -1)]

Cell = tuple[int, int]


def prefer_offwindow(cells: Sequence[Cell], is_off: Callable[[Cell], bool]) -> Cell:
    """Pick an off-window cell from ``cells`` if any (the exploit win), else the first.

    ``is_off`` is injected (the bot binds it to the engine's
    ``forced_win_detector.is_off_window`` on the live board) so the selection logic is
    tested without reconstructing the off-window geometry.
    """
    for c in cells:
        if is_off(c):
            return tuple(c)  # type: ignore[return-value]
    return tuple(cells[0])  # type: ignore[return-value]


def longest_line(my_stones: set[Cell], cell: Cell) -> tuple[int, list[Cell]]:
    """Length of my best same-colour run if I place a stone at ``cell``, plus the two
    cells that would extend that run (one past each end), for the best of the 3 axes.

    ``my_stones`` is my existing stones (excluding ``cell``). Returns
    ``(run_length, [back_end, forward_end])`` for the axis giving the longest run.
    """
    cq, cr = cell
    best_len = 0
    best_ends: list[Cell] = []
    for dq, dr in HEX_AXES:
        q, r = cq + dq, cr + dr
        n_fwd = 0
        while (q, r) in my_stones:
            n_fwd += 1
            q += dq
            r += dr
        fwd_end = (q, r)  # first empty cell past the forward run
        q, r = cq - dq, cr - dr
        n_bwd = 0
        while (q, r) in my_stones:
            n_bwd += 1
            q -= dq
            r -= dr
        bwd_end = (q, r)
        length = 1 + n_fwd + n_bwd
        if length > best_len:
            best_len = length
            best_ends = [bwd_end, fwd_end]
    return best_len, best_ends
