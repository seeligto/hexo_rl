"""Tests for the opening taxonomy classifier.

Covers all named opening families, rotation invariance, reflection
invariance, and edge cases.
"""

from __future__ import annotations

import math
import pytest

from hexo_rl.bootstrap.opening_classifier import (
    angular_gap_sectors,
    classify_opening,
    hex_dist,
    OPENING_FAMILIES,
)


# -- Hex geometry unit tests ------------------------------------------------

class TestHexDist:
    def test_origin(self):
        assert hex_dist(0, 0, 0, 0) == 0

    def test_neighbors(self):
        # All 6 neighbors of origin are distance 1
        neighbors = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
        for q, r in neighbors:
            assert hex_dist(0, 0, q, r) == 1

    def test_ring_2(self):
        assert hex_dist(0, 0, 2, 0) == 2
        assert hex_dist(0, 0, 0, 2) == 2
        assert hex_dist(0, 0, -2, 2) == 2

    def test_symmetry(self):
        assert hex_dist(3, -1, 0, 2) == hex_dist(0, 2, 3, -1)


class TestAngularGap:
    def test_same_direction(self):
        # (1,0) and (2,0) — same direction
        assert angular_gap_sectors(1, 0, 2, 0) == 0

    def test_adjacent_sectors(self):
        # (1,0) and (0,1) — 60 degrees apart
        assert angular_gap_sectors(1, 0, 0, 1) == 1

    def test_two_sectors(self):
        # (1,0) and (-1,1) — 120 degrees
        assert angular_gap_sectors(1, 0, -1, 1) == 2

    def test_opposite(self):
        # (1,0) and (-1,0) — 180 degrees
        assert angular_gap_sectors(1, 0, -1, 0) == 3

    def test_symmetric_wrap(self):
        # (1,0) and (0,-1) are 4 positions apart on ring A
        # min(4, 6-4) = 2 sectors
        gap = angular_gap_sectors(1, 0, 0, -1)
        assert gap == 2

    def test_one_sector_gap(self):
        # (1,0) and (1,-1) — adjacent directions, 60 degrees
        gap = angular_gap_sectors(1, 0, 1, -1)
        assert gap == 1


# -- Hex rotation / reflection helpers for tests ----------------------------

def _rotate_60(q: int, r: int) -> tuple[int, int]:
    """Rotate (q, r) by +60 degrees around origin."""
    # In cube coords: (x,y,z) -> (-z,-x,-y)
    # Axial: x=q, z=-(q+r) => new_q = q+r, new_r = -q
    return (q + r, -q)


def _reflect_q(q: int, r: int) -> tuple[int, int]:
    """Reflect across the q axis (r -> -r, adjust for axial)."""
    # In cube: (x,y,z) -> (x,z,y) => axial: (q, -(q+r))
    return (q, -(q + r))


def _rotate_moves(moves: list[tuple[int, int]], n: int) -> list[tuple[int, int]]:
    """Rotate all moves n times by 60 degrees around moves[0]."""
    if not moves:
        return moves
    cq, cr = moves[0]
    result = []
    for q, r in moves:
        rq, rr = q - cq, r - cr
        for _ in range(n % 6):
            rq, rr = _rotate_60(rq, rr)
        result.append((rq + cq, rr + cr))
    return result


def _reflect_moves(moves: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Reflect all moves across the q axis through moves[0]."""
    if not moves:
        return moves
    cq, cr = moves[0]
    result = []
    for q, r in moves:
        rq, rr = q - cq, r - cr
        rq, rr = _reflect_q(rq, rr)
        result.append((rq + cq, rr + cr))
    return result


# -- Canonical opening sequences (from BKE → axial) -------------------------
# P1 opens at (0,0). P2 places 2 stones. These are the defining plies.

# Ring A positions (distance 1 from center):
#   A0=(1,0) A1=(0,1) A2=(-1,1) A3=(-1,0) A4=(0,-1) A5=(1,-1)
# Ring B positions (distance 2 from center):
#   B0=(2,0) B1=(1,1) B2=(0,2) B3=(-1,2) B4=(-2,2) B5=(-2,1)
#   B6=(-2,0) B7=(-1,-1) B8=(0,-2) B9=(1,-2) B10=(2,-2) B11=(2,-1)
# Ring C positions (distance 3):
#   C0=(3,0) C1=(2,1) ...

PAIR_MOVES = [(0, 0), (1, 0), (0, 1)]           # A0 A1
CLOSED_MOVES = [(0, 0), (1, 0), (-1, 1)]        # A0 A2 (gap=2, 120 deg)
MOVE_101 = [(0, 0), (1, 0), (-1, 0)]            # A0 A3 (gap=3, 180 deg)
PISTOL_MOVES = [(0, 0), (1, 0), (0, 2)]         # A0 B2
SHOTGUN_MOVES = [(0, 0), (1, 0), (-2, 3)]       # A0 C5
OPEN_GAME_MOVES = [(0, 0), (2, 0), (-2, 2)]     # B0 B4 (gap=2)
HORSESHOE_MOVES = [(0, 0), (2, 0), (-2, 0)]     # B0 B6 (gap=3, 180 deg)
MARGE_MOVES = [(0, 0), (2, 0), (0, 2)]          # B0 B2 (close on ring B)
NEAR_ISLAND_MOVES = [(0, 0), (3, 0), (2, 1)]    # C0 C1
ISLAND_MOVES = [(0, 0), (5, 0), (4, 1)]         # E0, close to E1


# -- Opening family tests ---------------------------------------------------

class TestClassifyOpeningFamilies:
    def test_pair(self):
        assert classify_opening(PAIR_MOVES) == "pair"

    def test_closed_game(self):
        assert classify_opening(CLOSED_MOVES) == "closed_game"

    def test_101(self):
        assert classify_opening(MOVE_101) == "101"

    def test_pistol(self):
        assert classify_opening(PISTOL_MOVES) == "pistol"

    def test_shotgun(self):
        assert classify_opening(SHOTGUN_MOVES) == "shotgun"

    def test_open_game(self):
        assert classify_opening(OPEN_GAME_MOVES) == "open_game"

    def test_horseshoe(self):
        assert classify_opening(HORSESHOE_MOVES) == "horseshoe"

    def test_marge(self):
        assert classify_opening(MARGE_MOVES) == "marge"

    def test_near_island_gambit(self):
        assert classify_opening(NEAR_ISLAND_MOVES) == "near_island_gambit"

    def test_island_gambit(self):
        assert classify_opening(ISLAND_MOVES) == "island_gambit"


# -- Edge cases --------------------------------------------------------------

class TestEdgeCases:
    def test_empty_moves(self):
        assert classify_opening([]) == "unknown"

    def test_one_move(self):
        assert classify_opening([(0, 0)]) == "unknown"

    def test_two_moves(self):
        assert classify_opening([(0, 0), (1, 0)]) == "unknown"

    def test_extra_moves_ignored(self):
        """Only first 3 plies matter for classification."""
        extended = CLOSED_MOVES + [(2, -1), (3, -2)]
        assert classify_opening(extended) == "closed_game"

    def test_non_center_opening(self):
        """P1 doesn't have to open at (0,0) — classifier uses relative coords."""
        # Pair = A0 A1 = (1,0),(0,1) relative. Shifted to center (5,3):
        shifted = [(5, 3), (6, 3), (5, 4)]
        assert classify_opening(shifted) == "pair"


# -- Rotation invariance (60-degree rotations) --------------------------------

ALL_CANONICAL = [
    ("pair", PAIR_MOVES),
    ("closed_game", CLOSED_MOVES),
    ("101", MOVE_101),
    ("pistol", PISTOL_MOVES),
    ("shotgun", SHOTGUN_MOVES),
    ("open_game", OPEN_GAME_MOVES),
    ("horseshoe", HORSESHOE_MOVES),
    ("marge", MARGE_MOVES),
    ("near_island_gambit", NEAR_ISLAND_MOVES),
    ("island_gambit", ISLAND_MOVES),
]


class TestRotationInvariance:
    @pytest.mark.parametrize("expected,moves", ALL_CANONICAL,
                             ids=[c[0] for c in ALL_CANONICAL])
    @pytest.mark.parametrize("rot", range(6), ids=[f"rot{r*60}deg" for r in range(6)])
    def test_rotation(self, expected: str, moves: list, rot: int):
        rotated = _rotate_moves(moves, rot)
        assert classify_opening(rotated) == expected, (
            f"{expected} rotated {rot*60}deg: {rotated}"
        )


# -- Reflection invariance ---------------------------------------------------

class TestReflectionInvariance:
    @pytest.mark.parametrize("expected,moves", ALL_CANONICAL,
                             ids=[c[0] for c in ALL_CANONICAL])
    def test_reflection(self, expected: str, moves: list):
        reflected = _reflect_moves(moves)
        assert classify_opening(reflected) == expected, (
            f"{expected} reflected: {reflected}"
        )

    @pytest.mark.parametrize("expected,moves", ALL_CANONICAL,
                             ids=[c[0] for c in ALL_CANONICAL])
    @pytest.mark.parametrize("rot", range(6), ids=[f"rot{r*60}deg" for r in range(6)])
    def test_reflect_then_rotate(self, expected: str, moves: list, rot: int):
        """Full 12-fold symmetry: reflect then rotate."""
        transformed = _rotate_moves(_reflect_moves(moves), rot)
        assert classify_opening(transformed) == expected


# -- P2 stone order invariance -----------------------------------------------

class TestP2OrderInvariance:
    @pytest.mark.parametrize("expected,moves", ALL_CANONICAL,
                             ids=[c[0] for c in ALL_CANONICAL])
    def test_swap_p2_stones(self, expected: str, moves: list):
        """Swapping P2's two stones should give the same classification."""
        swapped = [moves[0], moves[2], moves[1]] + list(moves[3:])
        assert classify_opening(swapped) == expected


# -- Return value validity ---------------------------------------------------

class TestOutputValidity:
    @pytest.mark.parametrize("expected,moves", ALL_CANONICAL,
                             ids=[c[0] for c in ALL_CANONICAL])
    def test_result_in_families(self, expected: str, moves: list):
        result = classify_opening(moves)
        assert result in OPENING_FAMILIES
