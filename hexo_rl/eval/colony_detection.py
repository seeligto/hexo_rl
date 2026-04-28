"""Colony-win detection for the evaluation pipeline.

A "colony win" is a game where the winning player's stones form 2+
disconnected components with centroid distance >= threshold (axial distance).
This tracks SealBot's known blind spot with island openings.

Q11 fix: colony check is scoped to the winning 6-in-a-row's connected
component plus any OTHER substantial clusters (size >= 2). Single orphan
stones placed early and never developed are excluded, preventing false
positives when a player scatters exploratory stones but wins normally.
"""

from __future__ import annotations

import logging
from collections import deque

from hexo_rl.utils.coordinates import axial_distance as _axial_distance

log = logging.getLogger(__name__)

# Hex axial neighbours (6 directions)
_HEX_DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]

# 3 undirected hex axes; each checked in + direction only (start-of-run filter handles -)
_HEX_AXES = [(1, 0), (0, 1), (1, -1)]

_WIN_LENGTH = 6


def _connected_components(stones: set[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    """BFS flood-fill to find connected components on the sparse hex board."""
    visited: set[tuple[int, int]] = set()
    components: list[list[tuple[int, int]]] = []

    for start in stones:
        if start in visited:
            continue
        component: list[tuple[int, int]] = []
        queue = deque([start])
        visited.add(start)
        while queue:
            q, r = queue.popleft()
            component.append((q, r))
            for dq, dr in _HEX_DIRS:
                nb = (q + dq, r + dr)
                if nb in stones and nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        components.append(component)

    return components


def _centroid(component: list[tuple[int, int]]) -> tuple[float, float]:
    """Compute the centroid of a set of hex coordinates."""
    n = len(component)
    sq = sum(q for q, _ in component)
    sr = sum(r for _, r in component)
    return sq / n, sr / n


def _find_winning_line(
    winner_stones: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    """Scan winner's stones for a _WIN_LENGTH-in-a-row along any hex axis.

    Iterates each stone that is the *start* of a run on each axis (i.e. has no
    predecessor in that direction) and counts how far the run extends.  Returns
    the full run as an ordered list, or None if no winning line exists.
    """
    for q, r in winner_stones:
        for dq, dr in _HEX_AXES:
            # Skip if not the start of this run
            if (q - dq, r - dr) in winner_stones:
                continue
            run_len = 1
            nq, nr = q + dq, r + dr
            while (nq, nr) in winner_stones:
                run_len += 1
                nq, nr = nq + dq, nr + dr
            if run_len >= _WIN_LENGTH:
                return [(q + dq * i, r + dr * i) for i in range(run_len)]
    return None


def is_colony_win(
    stones: list[tuple[int, int, int]],
    winner: int,
    centroid_threshold: float = 6.0,
    winning_line: list[tuple[int, int]] | None = None,
) -> bool:
    """Determine if a game result is a colony win.

    Args:
        stones: List of (q, r, player) tuples from Board.get_stones().
        winner: The winning player (1 or -1).
        centroid_threshold: Minimum axial distance between any two component
            centroids to qualify as a colony win.
        winning_line: Optional pre-computed winning 6-in-a-row as (q, r) pairs.
            If None, computed from stones.  Exposed for testing edge cases.

    Returns:
        True if the winner's winning cluster and any OTHER substantial clusters
        (>= 2 stones) have at least one pair of centroids >= centroid_threshold
        apart.  Single orphan stones are excluded from the colony check.
        Returns False (conservative) if no winning line can be found.
    """
    winner_stones = {(q, r) for q, r, p in stones if p == winner}

    if len(winner_stones) < 2:
        return False

    if winning_line is None:
        winning_line = _find_winning_line(winner_stones)

    if winning_line is None:
        log.warning("colony_detection_no_winning_line winner=%s", winner)
        return False

    # Guard: winning line must be contiguous (each consecutive pair within radius 1).
    # A gap indicates a stale or synthetic line — conservatively not a colony win.
    for i in range(len(winning_line) - 1):
        if _axial_distance(winning_line[i], winning_line[i + 1]) > 1:
            log.warning("colony_detection_gapped_line winner=%s", winner)
            return False

    components = _connected_components(winner_stones)

    winning_set = set(winning_line)
    main_component: list[tuple[int, int]] | None = None
    other_components: list[list[tuple[int, int]]] = []
    for c in components:
        if any(s in winning_set for s in c):
            main_component = c
        else:
            other_components.append(c)

    if main_component is None:
        log.warning("colony_detection_line_not_in_stones winner=%s", winner)
        return False

    # Exclude single-stone orphans: only count other clusters with >= 2 stones.
    substantial_others = [c for c in other_components if len(c) >= 2]
    retained = [main_component] + substantial_others

    if len(retained) < 2:
        return False

    centroids = [_centroid(c) for c in retained]
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            if _axial_distance(centroids[i], centroids[j]) >= centroid_threshold:
                return True

    return False
