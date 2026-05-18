"""INV27 — `find_winning_line` surfaces the winning 6-line on
colony-classified game endings (§S178 F-fix-1).

Pre-fix mechanism (`engine/src/board/moves.rs::find_winning_line`):
scanned outward only from `last_move`. Under the HTT 2-moves-per-turn rule
the **first** move of a turn can complete a 6-line and the **second** move
overwrites `last_move` off-line. `winner()` finds the win via its
all-stones fallback (`player_wins` moves.rs:184-189) so `terminal_reason`
classifies the game as colony; the threat-head target column in the
per-row aux buffer is then all-zero on every position of that game — the
threat head learns "no 6-line exists" and the threat-detection circuit
gradient-suppresses. SA-A verdict 6 (reports/s178_pre_design_investigation.md).

Fix: `find_winning_line` falls back to scanning all stones when last_move
doesn't surface a line, mirroring `player_wins`'s fallback.

Parametrized over the 4 game-end outcomes. Cell 2 is the new invariant.
"""
from __future__ import annotations

import pytest

from engine import Board


# (cell_id, move_sequence, expected_winner, expected_line_len, expected_cells_on_line)
# move_sequence: list of (q, r); applied verbatim via Board.apply_move.
# expected_cells_on_line: set of (q, r) the returned line must equal (when non-empty).
_P1_LINE = {(q, 0) for q in range(6)}
_CASES = [
    # Cell 1 — fast path: last_move completes the 6-line.
    (
        "cell_1_winner_last_move_on_line",
        [
            (0, 0), (10, 0), (15, 0),
            (1, 0), (2, 0),
            (10, 5), (15, 5),
            (3, 0), (4, 0),
            (20, 0), (20, 5),
            (5, 0),  # P1 ply 11 completes 6-line; last_move ON line
        ],
        1,                  # winner = P1
        6,                  # line length
        _P1_LINE,
    ),
    # Cell 2 — NEW INVARIANT: winner exists via fallback, line via fallback.
    (
        "cell_2_winner_last_move_off_line_colony",
        [
            (0, 0), (10, 0), (15, 0),
            (1, 0), (2, 0),
            (10, 5), (15, 5),
            (3, 0), (4, 0),
            (20, 0), (20, 5),
            (5, 0),                   # P1 completes 6-line at ply 11
            (-1, -1),                 # P1 off-line second move
            (25, 0), (25, 5),         # P2 turn; last_move OFF line, NOT P1
        ],
        1,                  # winner = P1 via player_wins fallback
        6,                  # line length — requires F-fix-1 fallback
        _P1_LINE,
    ),
    # Cell 3 — organic draw (no 6-line, ply < max).
    (
        "cell_3_no_winner_organic_draw",
        [(0, 0), (5, 5), (-3, 4), (2, 2), (-4, -4)],
        None,
        0,
        set(),
    ),
    # Cell 4 — ply cap analog (still no 6-line, find_winning_line returns []).
    (
        "cell_4_no_winner_ply_cap_equivalent",
        [(0, 0), (10, 10), (11, 10)],
        None,
        0,
        set(),
    ),
]


@pytest.mark.parametrize(
    "cell_id, moves, expected_winner, expected_line_len, expected_cells",
    _CASES,
    ids=[c[0] for c in _CASES],
)
def test_inv27_threat_target_colony(
    cell_id: str,
    moves: list[tuple[int, int]],
    expected_winner: int | None,
    expected_line_len: int,
    expected_cells: set[tuple[int, int]],
) -> None:
    board = Board()
    for q, r in moves:
        board.apply_move(q, r)

    assert board.winner() == expected_winner, (
        f"{cell_id}: winner mismatch — got {board.winner()!r}, "
        f"expected {expected_winner!r}"
    )

    line = board.find_winning_line()
    assert len(line) == expected_line_len, (
        f"{cell_id}: find_winning_line length mismatch — "
        f"got {len(line)} ({line!r}), expected {expected_line_len}. "
        f"(Pre-§S178 F-fix-1 Cell 2 returned [] because find_winning_line only "
        f"scanned from last_move.)"
    )

    if expected_line_len > 0:
        assert set(line) == expected_cells, (
            f"{cell_id}: line cells mismatch — got {set(line)!r}, "
            f"expected {expected_cells!r}"
        )
