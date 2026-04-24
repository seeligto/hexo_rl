"""Selfplay axis-distribution metric.

For each hex axis, computes the fraction of adjacent stone pairs that share
the same color.  Values near 0.5 = balanced opponent interleaving; near 1.0 =
same-color clustering along that axis — a potential degenerate-strategy signal.

Three axes (matching game_state._HEX_AXES order):
  axis_q  E-W      (dq=+1, dr= 0)
  axis_r  NW-SE    (dq= 0, dr=+1)
  axis_s  NE-SW    (dq=+1, dr=-1)
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

# Mirrors game_state._HEX_AXES — do not reorder.
_AXES = ((1, 0), (0, 1), (1, -1))
AXIS_LABELS = ("axis_q", "axis_r", "axis_s")


def _assign_colors(
    move_history: list[tuple[int, int]],
) -> dict[tuple[int, int], int]:
    """Return {(q, r): color} for a game's move list.

    color = +1 for P1, -1 for P2.  Assignment matches pool.py compound-move
    rule: ply 0 → P1; compound_idx = (ply - 1) // 2; P2 when even, P1 odd.
    """
    stone_color: dict[tuple[int, int], int] = {}
    for ply, pos in enumerate(move_history):
        is_p1 = (ply == 0) or (((ply - 1) // 2) % 2 == 1)
        stone_color[pos] = 1 if is_p1 else -1
    return stone_color


def compute_axis_fractions(
    games: Sequence[list[tuple[int, int]]],
) -> dict[str, float]:
    """Compute axis-distribution fractions from completed self-play games.

    Aggregates total same-color adjacent pairs / total adjacent pairs across
    all games for each axis.  Returns 0.0 on empty input.

    Returns dict with keys: axis_q, axis_r, axis_s (floats in [0,1]),
    axis_max (label of the max-fraction axis).
    """
    same = [0, 0, 0]
    total = [0, 0, 0]

    for game in games:
        if len(game) < 2:
            continue
        stone_color = _assign_colors(game)
        for i, (dq, dr) in enumerate(_AXES):
            for (q, r), color in stone_color.items():
                nbr = (q + dq, r + dr)
                if nbr in stone_color:
                    total[i] += 1
                    if stone_color[nbr] == color:
                        same[i] += 1

    fracs = [s / t if t > 0 else 0.0 for s, t in zip(same, total)]
    if any(t > 0 for t in total):
        max_idx = int(np.argmax(fracs))
    else:
        max_idx = 0
    return {
        "axis_q": fracs[0],
        "axis_r": fracs[1],
        "axis_s": fracs[2],
        "axis_max": AXIS_LABELS[max_idx],
    }


def compute_axis_fractions_from_states(states: np.ndarray) -> dict[str, float]:
    """Compute axis fractions from a corpus state array.

    Args:
        states: (N, ≥9, H, W) array.  Plane 0 = current player stones,
                plane 8 = opponent stones.  Accepts float16 or float32.

    Returns dict with same keys as compute_axis_fractions.
    Aggregates pairs across all N positions.
    """
    cur = (states[:, 0] > 0).astype(np.uint8)  # (N, H, W)
    opp = (states[:, 8] > 0).astype(np.uint8)  # (N, H, W)
    H, W = cur.shape[1], cur.shape[2]

    fracs: list[float] = []
    for dq, dr in _AXES:
        if dq > 0:
            q_a: slice = slice(0, H - dq)
            q_b: slice = slice(dq, H)
        elif dq < 0:
            q_a = slice(-dq, H)
            q_b = slice(0, H + dq)
        else:
            q_a = q_b = slice(None)

        if dr > 0:
            r_a: slice = slice(0, W - dr)
            r_b: slice = slice(dr, W)
        elif dr < 0:
            r_a = slice(-dr, W)
            r_b = slice(0, W + dr)
        else:
            r_a = r_b = slice(None)

        cur_a = cur[:, q_a, r_a]
        cur_b = cur[:, q_b, r_b]
        opp_a = opp[:, q_a, r_a]
        opp_b = opp[:, q_b, r_b]

        has_stone_a = cur_a | opp_a  # (N, H', W')
        has_stone_b = cur_b | opp_b
        has_pair = (has_stone_a & has_stone_b).astype(bool)
        is_same = (
            (cur_a.astype(bool) & cur_b.astype(bool))
            | (opp_a.astype(bool) & opp_b.astype(bool))
        )

        t = int(has_pair.sum())
        s = int((has_pair & is_same).sum())
        fracs.append(s / t if t > 0 else 0.0)

    max_idx = int(np.argmax(fracs))
    return {
        "axis_q": fracs[0],
        "axis_r": fracs[1],
        "axis_s": fracs[2],
        "axis_max": AXIS_LABELS[max_idx],
    }
