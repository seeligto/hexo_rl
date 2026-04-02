"""Colony-win detection for the evaluation pipeline.

A "colony win" is a game where the winning player's stones form 2+
disconnected components with centroid distance >= threshold (axial distance).
This tracks SealBot's known blind spot with island openings.
"""

from __future__ import annotations

import math
from collections import deque


# Hex axial neighbours (6 directions)
_HEX_DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]


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


def _axial_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Axial (hex Manhattan) distance between two points."""
    dq = abs(a[0] - b[0])
    dr = abs(a[1] - b[1])
    ds = abs((a[0] + a[1]) - (b[0] + b[1]))  # s = -q - r
    return max(dq, dr, ds)


def is_colony_win(
    stones: list[tuple[int, int, int]],
    winner: int,
    centroid_threshold: float = 6.0,
) -> bool:
    """Determine if a game result is a colony win.

    Args:
        stones: List of (q, r, player) tuples from Board.get_stones().
        winner: The winning player (1 or -1).
        centroid_threshold: Minimum axial distance between any two component
            centroids to qualify as a colony win.

    Returns:
        True if the winner's stones form 2+ disconnected components with
        at least one pair of centroids >= centroid_threshold apart.
    """
    winner_stones = {(q, r) for q, r, p in stones if p == winner}

    if len(winner_stones) < 2:
        return False

    components = _connected_components(winner_stones)

    if len(components) < 2:
        return False

    # Check if any pair of component centroids are far enough apart
    centroids = [_centroid(c) for c in components]
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            if _axial_distance(centroids[i], centroids[j]) >= centroid_threshold:
                return True

    return False
