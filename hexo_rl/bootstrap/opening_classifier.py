"""Opening taxonomy classifier for Hex Tac Toe games.

Classifies games into opening families based on the geometric relationship
of P2's first compound move (2 stones) relative to P1's opening stone.

All classification is rotation- and reflection-invariant: it uses hex
distances, ring numbers, and angular gaps — never absolute coordinates.

Opening families:
    pair              P2 stones adjacent (d=1), both ring 1
    closed_game       Both ring 1, d=2, angular gap = 2 sectors (120 deg)
    101               Both ring 1, d=2, angular gap = 3 sectors (180 deg)
    marge             Both ring 2, d <= 2 (close together on ring B)
    pistol            One ring 1, one ring 2
    shotgun           One ring 1, one ring 3+
    open_game         Both ring 2, d >= 3
    horseshoe         Both ring 2, d >= 3, angular gap = 3 sectors (180 deg)
    near_island_gambit  Both ring 3, stones close together
    island_gambit     Both stones ring >= 4
    unknown           Anything else or < 3 plies
"""

from __future__ import annotations

import math
from typing import Sequence


# -- Hex geometry helpers ---------------------------------------------------

def hex_dist(q1: int, r1: int, q2: int, r2: int) -> int:
    """Hex (axial) distance between two cells."""
    dq = q1 - q2
    dr = r1 - r2
    return max(abs(dq), abs(dr), abs(dq + dr))


def _hex_angle(q: int, r: int) -> float:
    """Angle (radians) of axial coord (q, r) from origin in cartesian space."""
    x = q + r * 0.5
    y = r * (math.sqrt(3) / 2)
    return math.atan2(y, x)


def angular_gap_sectors(q1: int, r1: int, q2: int, r2: int) -> int:
    """Angular gap between two vectors from origin, in 60-degree sectors (0-3).

    Returns the minimum sector count (wraps at 3 due to hex symmetry of the
    gap itself: 4 sectors apart == 2 sectors apart via the other direction).
    """
    a1 = _hex_angle(q1, r1)
    a2 = _hex_angle(q2, r2)
    diff = abs(a1 - a2)
    if diff > math.pi:
        diff = 2 * math.pi - diff
    # Each sector = pi/3 radians (60 degrees)
    sectors = round(diff / (math.pi / 3))
    return min(sectors, 6 - sectors)


# -- Classifier -------------------------------------------------------------

OPENING_FAMILIES = [
    "pair",
    "marge",
    "closed_game",
    "open_game",
    "pistol",
    "horseshoe",
    "island_gambit",
    "near_island_gambit",
    "101",
    "shotgun",
    "unknown",
]


def classify_opening(moves: Sequence[tuple[int, int]]) -> str:
    """Classify a game's opening family from its move sequence.

    Args:
        moves: Ordered (q, r) axial coordinate tuples. Index 0 is P1's
               opening stone; indices 1-2 are P2's first compound move.

    Returns:
        One of the OPENING_FAMILIES strings.
    """
    if len(moves) < 3:
        return "unknown"

    # P1's opening stone = reference center
    cq, cr = moves[0]
    # P2's two stones (first compound move)
    aq, ar = moves[1][0] - cq, moves[1][1] - cr
    bq, br = moves[2][0] - cq, moves[2][1] - cr

    ra = hex_dist(0, 0, aq, ar)
    rb = hex_dist(0, 0, bq, br)
    dab = hex_dist(aq, ar, bq, br)

    # Sort so ra <= rb (canonical order)
    if ra > rb:
        ra, rb = rb, ra
        aq, ar, bq, br = bq, br, aq, ar

    # --- Classification rules (most specific first) ---

    # Island gambit: both stones far (ring >= 4)
    if ra >= 4:
        return "island_gambit"

    # Near-island gambit: both ring 3
    if ra == 3 and rb == 3:
        return "near_island_gambit"

    # Both ring 1
    if ra == 1 and rb == 1:
        if dab == 1:
            return "pair"
        if dab == 2:
            gap = angular_gap_sectors(aq, ar, bq, br)
            if gap == 3:
                return "101"
            # gap == 2 is the standard closed game (A0 A2 pattern)
            return "closed_game"
        # dab > 2 impossible on ring 1
        return "unknown"

    # One ring 1 + one ring 2 → pistol
    if ra == 1 and rb == 2:
        return "pistol"

    # One ring 1 + ring 3+ → shotgun
    if ra == 1 and rb >= 3:
        return "shotgun"

    # Both ring 2
    if ra == 2 and rb == 2:
        if dab >= 3:
            gap = angular_gap_sectors(aq, ar, bq, br)
            if gap == 3:
                return "horseshoe"
            return "open_game"
        # Close together on ring B
        return "marge"

    # Mixed ring 2 + ring 3
    if ra == 2 and rb == 3:
        if dab <= 2:
            return "near_island_gambit"
        return "unknown"

    # Ring 3 + ring 3 with large gap
    if ra == 3 and rb >= 3:
        return "near_island_gambit"

    return "unknown"


# -- Batch analysis ----------------------------------------------------------

def classify_corpus(
    records: Sequence,
    *,
    min_plies: int = 3,
) -> dict[str, list[str]]:
    """Classify all games in a corpus and return per-family game ID lists.

    Args:
        records: Iterable of objects with ``.moves`` (list of (q,r) tuples)
                 and ``.game_id_str`` attributes (i.e. GameRecord).
        min_plies: Skip games shorter than this.

    Returns:
        Dict mapping opening family name → list of game_id_str.
    """
    from collections import defaultdict

    by_family: dict[str, list[str]] = defaultdict(list)
    for rec in records:
        if len(rec.moves) < min_plies:
            by_family["unknown"].append(rec.game_id_str)
            continue
        family = classify_opening(rec.moves)
        by_family[family].append(rec.game_id_str)
    return dict(by_family)


def opening_distribution(
    records: Sequence,
) -> dict[str, int]:
    """Return {family: count} for a corpus."""
    from collections import Counter

    counts: Counter[str] = Counter()
    for rec in records:
        family = classify_opening(rec.moves)
        counts[family] += 1
    return dict(counts.most_common())
