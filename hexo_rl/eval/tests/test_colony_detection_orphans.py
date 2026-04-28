"""Tests for Q11 — colony detection orphan over-inclusion fix.

Demonstrates and guards the fix: is_colony_win must check the winning
6-in-a-row's cluster only, ignoring single orphan stones placed early.
"""

from __future__ import annotations

from hexo_rl.eval.colony_detection import is_colony_win


def _p1(coords: list[tuple[int, int]]) -> list[tuple[int, int, int]]:
    return [(q, r, 1) for q, r in coords]


def _p2(coords: list[tuple[int, int]]) -> list[tuple[int, int, int]]:
    return [(q, r, -1) for q, r in coords]


def test_true_colony_win_two_islands() -> None:
    """Winner has 6-in-a-row in island A + substantial second island B far away → True."""
    # Island A: 6-in-a-row at (0,0)-(5,0), centroid (2.5, 0)
    island_a = [(i, 0) for i in range(6)]
    # Island B: 3 stones at (15,0)-(17,0), centroid (16, 0)
    # axial_distance((2.5,0), (16,0)) = max(13.5, 0, 13.5) = 13.5 > 6.0
    island_b = [(15, 0), (16, 0), (17, 0)]
    stones = _p1(island_a) + _p1(island_b) + _p2([(8, 5)])
    assert is_colony_win(stones, winner=1, centroid_threshold=6.0) is True


def test_non_colony_win_orphan_excluded() -> None:
    """6-in-a-row + single orphan stone far away → False (orphan filtered, size=1).

    This is the regression case from Q11: the old code returned True because
    the orphan created a second connected component with centroid >= 6.0 away.
    """
    winning_line = [(i, 0) for i in range(6)]  # centroid (2.5, 0)
    orphan = [(20, 0)]  # 1 stone, axial distance 17.5 from centroid — large but size=1
    stones = _p1(winning_line) + _p1(orphan) + _p2([(3, 5)])
    # After fix: orphan (size=1) excluded from colony check → 1 retained component → False
    assert is_colony_win(stones, winner=1, centroid_threshold=6.0) is False


def test_non_colony_win_multiple_orphans_still_excluded() -> None:
    """Multiple single orphan stones far away — each size=1 — still excluded → False."""
    winning_line = [(i, 0) for i in range(6)]
    orphans = [(20, 0), (25, 5), (30, -3)]  # three isolated stones
    stones = _p1(winning_line) + _p1(orphans) + _p2([(3, 5)])
    assert is_colony_win(stones, winner=1, centroid_threshold=6.0) is False


def test_winning_line_crosses_gap_returns_false() -> None:
    """Winning line with a gap (consecutive pair > radius 1) → False (conservative).

    Exercises the adjacency validation guard added in Q11.  This path is hit
    when a synthetic or externally-supplied winning_line contains non-adjacent
    cells — the function returns False rather than producing a spurious result.
    """
    winning_line = [(i, 0) for i in range(6)]  # (0,0)-(5,0), all adjacent
    # Insert a gap: replace (3,0) with (4,0) shifted — skip (3,0), go straight to (4,0).
    # Synthetic gapped line: (0,0),(1,0),(2,0),(4,0),(5,0),(6,0) — gap at (3,0).
    gapped_line = [(0, 0), (1, 0), (2, 0), (4, 0), (5, 0), (6, 0)]
    stones = _p1([(0, 0), (1, 0), (2, 0), (4, 0), (5, 0), (6, 0)])
    assert is_colony_win(stones, winner=1, winning_line=gapped_line) is False


def test_true_colony_boundary_centroid_exactly_at_threshold() -> None:
    """Second island centroid exactly at threshold → True (>= is inclusive)."""
    # Island A: 6-in-a-row at (0,0)-(5,0), centroid (2.5, 0)
    island_a = [(i, 0) for i in range(6)]
    # Island B: 2 stones at (8,0),(9,0), centroid (8.5, 0)
    # axial_distance((2.5,0),(8.5,0)) = 6.0 exactly
    island_b = [(8, 0), (9, 0)]
    stones = _p1(island_a) + _p1(island_b)
    assert is_colony_win(stones, winner=1, centroid_threshold=6.0) is True


def test_second_island_just_below_threshold_false() -> None:
    """Second island centroid at distance 5.0 < 6.0 → False."""
    island_a = [(i, 0) for i in range(6)]  # centroid (2.5, 0)
    # (5,2),(6,2) not adjacent to island_a; centroid (5.5, 2)
    # axial_distance((2.5,0),(5.5,2)): dq=3, dr=2, ds=5 → max=5.0 < 6.0
    island_b = [(5, 2), (6, 2)]
    stones = _p1(island_a) + _p1(island_b)
    assert is_colony_win(stones, winner=1, centroid_threshold=6.0) is False
