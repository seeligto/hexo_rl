"""
Phase 0 test suite for the Rust Board exposed via PyO3.

Run with: pytest tests/test_board.py -v
"""
import pytest
import numpy as np
from engine import Board


# ── Basic construction ────────────────────────────────────────────────────────

def test_new_board_is_empty():
    b = Board()
    assert b.ply == 0
    assert b.current_player == 1
    assert b.moves_remaining == 1
    assert not b.check_win()
    assert b.winner() is None


def test_legal_moves_on_empty_board():
    b = Board()
    moves = b.legal_moves()
    # Empty board uses a 5×5 init region (25 cells) to keep MCTS branching
    # factor manageable — first move location is strategically arbitrary.
    assert len(moves) == 25
    assert b.legal_move_count() == 25


# ── Turn structure ────────────────────────────────────────────────────────────

def test_first_move_single_passes_turn():
    """Player 1 opens with exactly 1 move; turn then passes to player 2."""
    b = Board()
    b.apply_move(0, 0)
    assert b.current_player == -1
    assert b.moves_remaining == 2
    assert b.ply == 1


def test_player2_uses_two_moves():
    b = Board()
    b.apply_move(0, 0)          # P1 ply0
    b.apply_move(1, 0)          # P2 first
    assert b.current_player == -1
    assert b.moves_remaining == 1
    b.apply_move(2, 0)          # P2 second — turn passes
    assert b.current_player == 1
    assert b.moves_remaining == 2


def test_turn_alternation_over_several_turns():
    b = Board()
    b.apply_move(0, 0)   # P1 single (ply 0)
    assert b.current_player == -1
    b.apply_move(1, 0)   # P2
    b.apply_move(2, 0)   # P2  → P1's turn
    assert b.current_player == 1
    b.apply_move(3, 0)   # P1
    b.apply_move(4, 0)   # P1  → P2's turn
    assert b.current_player == -1
    b.apply_move(5, 0)   # P2
    b.apply_move(6, 0)   # P2  → P1's turn
    assert b.current_player == 1


# ── Illegal moves ─────────────────────────────────────────────────────────────

def test_occupied_cell_raises():
    b = Board()
    b.apply_move(0, 0)        # P1 places at origin
    with pytest.raises(ValueError):
        b.apply_move(0, 0)    # occupied — should raise


def test_legal_move_count_decrements():
    # §145 Option α': legal moves = all cells within hex-ball radius 5 of any
    # stone (was 8), minus occupied cells.
    # After 1 stone at (0,0): hex ball radius 5 = 91 cells, minus 1 occupied = 90.
    # After 2nd stone at (1,0): union of two adjacent radius-5 balls
    # (102 cells) minus 2 occupied = 100.
    b = Board()
    b.apply_move(0, 0)
    assert b.legal_move_count() == 90
    b.apply_move(1, 0)
    assert b.legal_move_count() == 100


# ── Win detection — all three axes ───────────────────────────────────────────

def _play_game(moves_p1, moves_p2):
    """
    Play a sequence of moves, alternating turns according to the turn structure.
    moves_p1 / moves_p2 are lists of (q, r) tuples; P1 goes first (single).
    Returns the Board after all moves are applied.
    """
    b = Board()
    # Interleave into a flat list respecting turn structure:
    # P1 ply 0 = 1 move, then 2 each.
    all_moves = []
    # P1 single first
    all_moves.append((1, moves_p1[0]))
    p1_rest = moves_p1[1:]
    p2_rest = moves_p2[:]
    # Now alternate: P2 gets 2, P1 gets 2, repeat
    p1_idx = 0
    p2_idx = 0
    while p1_idx < len(p1_rest) or p2_idx < len(p2_rest):
        for _ in range(2):
            if p2_idx < len(p2_rest):
                all_moves.append((-1, p2_rest[p2_idx]))
                p2_idx += 1
        for _ in range(2):
            if p1_idx < len(p1_rest):
                all_moves.append((1, p1_rest[p1_idx]))
                p1_idx += 1
    b = Board()
    for _, (q, r) in all_moves:
        try:
            b.apply_move(q, r)
        except ValueError:
            pass
    return b


def test_no_win_on_empty():
    b = Board()
    assert not b.check_win()
    assert b.winner() is None


def test_five_in_row_not_win():
    """5 stones in a row is not a win; needs 6."""
    b = Board()
    b.apply_move(0, 0)                         # P1 single
    b.apply_move(-9, 9).apply_move if False else None
    # P2 fillers on the opposite side
    p2_fillers = [(-9, r) for r in range(-9, 1)]
    moves = [(-9, 9), (-8, 9)]  # P2 start
    b2 = Board()
    b2.apply_move(0, 0)  # P1
    # P2: fill up two per turn
    p2 = [(-9, r) for r in range(0, 6, 2)]  # scattered P2
    p1 = [(q, 9) for q in range(1, 7)]       # P1 along E at r=9
    # Build the game manually
    bm = Board()
    bm.apply_move(0, 9)        # P1 ply0
    bm.apply_move(-9, 0)       # P2
    bm.apply_move(-9, 1)       # P2
    bm.apply_move(1, 9)        # P1
    bm.apply_move(2, 9)        # P1
    bm.apply_move(-9, 2)       # P2
    bm.apply_move(-9, 3)       # P2
    bm.apply_move(3, 9)        # P1
    bm.apply_move(4, 9)        # P1
    # P1 now has 5 in a row: q=0,1,2,3,4 at r=9 — should not be a win
    assert not bm.check_win()


def test_win_e_axis_player1():
    """P1 wins with 6 consecutive along E (dq=+1)."""
    b = Board()
    # P1 at q=0..5, r=0; P2 fillers on different rows
    b.apply_move(0, 0)
    b.apply_move(-9, 5); b.apply_move(-9, 6)
    b.apply_move(1, 0); b.apply_move(2, 0)
    b.apply_move(-9, 7); b.apply_move(-9, 8)
    b.apply_move(3, 0); b.apply_move(4, 0)
    b.apply_move(-9, -5); b.apply_move(-9, -6)
    b.apply_move(5, 0)
    assert b.check_win()
    assert b.winner() == 1


def test_win_ne_axis_player1():
    """P1 wins with 6 consecutive along NE (dr=+1)."""
    b = Board()
    # P1 at q=0, r=0..5; P2 fillers scattered
    b.apply_move(0, 0)
    b.apply_move(-9, -9); b.apply_move(-8, -9)
    b.apply_move(0, 1); b.apply_move(0, 2)
    b.apply_move(-7, -9); b.apply_move(-6, -9)
    b.apply_move(0, 3); b.apply_move(0, 4)
    b.apply_move(-5, -9); b.apply_move(-4, -9)
    b.apply_move(0, 5)
    assert b.check_win()
    assert b.winner() == 1


def test_win_nw_axis_player1():
    """P1 wins along NW axis (dq=-1, dr=+1)."""
    b = Board()
    # NW: (0,0),(-1,1),(-2,2),(-3,3),(-4,4),(-5,5)
    b.apply_move(0, 0)
    b.apply_move(9, -9); b.apply_move(8, -9)
    b.apply_move(-1, 1); b.apply_move(-2, 2)
    b.apply_move(7, -9); b.apply_move(6, -9)
    b.apply_move(-3, 3); b.apply_move(-4, 4)
    b.apply_move(5, -9); b.apply_move(4, -9)
    b.apply_move(-5, 5)
    assert b.check_win()
    assert b.winner() == 1


def test_win_player2_e_axis():
    """P2 wins with 6 consecutive along E."""
    b = Board()
    b.apply_move(0, 0)                          # P1 single first move
    b.apply_move(0, -1); b.apply_move(1, -1)   # P2
    b.apply_move(0, 3); b.apply_move(0, 4)     # P1 fillers (NE axis, not 6-in-row)
    b.apply_move(2, -1); b.apply_move(3, -1)   # P2
    b.apply_move(0, 5); b.apply_move(0, 6)     # P1 fillers
    b.apply_move(4, -1); b.apply_move(5, -1)   # P2 — win
    assert b.check_win()
    assert b.winner() == -1


def test_win_at_board_corners():
    """P1 wins with 6 in a row near the left edge of the sliding window."""
    # P1: (-9,0)..(-4,0) along E.  P2 fillers are scattered rightward.
    b = Board()
    b.apply_move(-9, 0)                         # P1 single first move
    b.apply_move(0, 1); b.apply_move(0, 2)     # P2 fillers
    b.apply_move(-8, 0); b.apply_move(-7, 0)   # P1
    b.apply_move(1, 1); b.apply_move(2, 2)     # P2 fillers
    b.apply_move(-6, 0); b.apply_move(-5, 0)   # P1
    b.apply_move(3, 3); b.apply_move(4, 4)     # P2 fillers (no 6-in-row)
    b.apply_move(-4, 0)                         # P1 — 6th stone, win
    assert b.check_win()
    assert b.winner() == 1


# ── Zobrist hash ──────────────────────────────────────────────────────────────

def test_zobrist_changes_each_move():
    b = Board()
    hashes = {b.zobrist_hash()}
    b.apply_move(0, 0)
    hashes.add(b.zobrist_hash())
    b.apply_move(1, 0)
    hashes.add(b.zobrist_hash())
    b.apply_move(2, 0)
    hashes.add(b.zobrist_hash())
    assert len(hashes) == 4, "each move must produce a distinct hash"


def test_zobrist_different_positions_different_hashes():
    """Two boards reached via different move sequences must have different hashes."""
    b1 = Board()
    b1.apply_move(0, 0)   # P1
    b1.apply_move(1, 0)   # P2
    b1.apply_move(2, 0)   # P2

    b2 = Board()
    b2.apply_move(0, 0)   # P1
    b2.apply_move(2, 0)   # P2
    b2.apply_move(1, 0)   # P2

    # Same stones on the board but reached in different order for P2 — hash should differ
    # because the board states are actually IDENTICAL (same stones), so they SHOULD be equal
    # Actually order of same-player stones doesn't affect the board state.
    # Let's instead compare genuinely different boards.
    b3 = Board()
    b3.apply_move(0, 0)   # P1
    b3.apply_move(3, 0)   # P2 (different cell)
    b3.apply_move(4, 0)   # P2

    assert b1.zobrist_hash() != b3.zobrist_hash()


def test_zobrist_identical_positions_same_hash():
    """Same board state reached via different paths has the same Zobrist hash."""
    # P2 places at (1,0) then (2,0) vs (2,0) then (1,0) — same position
    b1 = Board()
    b1.apply_move(0, 0)
    b1.apply_move(1, 0)
    b1.apply_move(2, 0)

    b2 = Board()
    b2.apply_move(0, 0)
    b2.apply_move(2, 0)
    b2.apply_move(1, 0)

    assert b1.zobrist_hash() == b2.zobrist_hash()


# ── to_tensor ─────────────────────────────────────────────────────────────────

def test_tensor_shape_and_dtype():
    b = Board()
    t = b.to_tensor()
    assert isinstance(t, list)
    assert len(t) == 18 * 19 * 19


def test_tensor_as_numpy():
    b = Board()
    b.apply_move(0, 0)  # P1 at origin; turn passes to P2
    arr = np.array(b.to_tensor(), dtype=np.float32).reshape(18, 19, 19)
    # Current player is P2 (-1).  Plane 0 = current player's stones = P2 = none yet.
    # Plane 8 = opponent's stones = P1's stones = origin (q=0, r=0).
    assert arr.shape == (18, 19, 19)
    # Plane 8 should have a 1 at the origin cell: flat index idx(0,0) = 9*19+9 = 180
    plane8 = arr[8].flatten()
    assert plane8[180] == 1.0
    # Plane 0 should be all zeros (P2 has no stones yet)
    assert arr[0].sum() == 0.0


def test_tensor_current_player_plane():
    b = Board()
    b.apply_move(1, 1)   # P1 at (1,1)
    # Turn is P2 now. Place P2 stone.
    b.apply_move(-1, -1)  # P2 at (-1,-1)
    b.apply_move(-2, -2)  # P2's 2nd move — turn passes to P1
    arr = np.array(b.to_tensor(), dtype=np.float32).reshape(18, 19, 19)
    flat = arr.reshape(18, -1)
    # Current player is P1 (1). Plane 0 = P1's stones = (1,1).
    p1_idx = (1 + 9) * 19 + (1 + 9)   # idx(1,1)
    assert flat[0, p1_idx] == 1.0
    # Plane 8 = P2's stones = (-1,-1) and (-2,-2)
    p2_idx1 = (-1 + 9) * 19 + (-1 + 9)
    p2_idx2 = (-2 + 9) * 19 + (-2 + 9)
    assert flat[8, p2_idx1] == 1.0
    assert flat[8, p2_idx2] == 1.0


# ── Board size property ───────────────────────────────────────────────────────

def test_board_size():
    b = Board()
    assert b.size == 19


# ── Full random game ──────────────────────────────────────────────────────────

def test_random_game_runs_to_completion():
    """A random game must run without errors.

    Capped at 100 moves to keep runtime under 1s.  The uncapped version
    took ~44s because legal_moves() grows with the stone frontier on the
    infinite board.  Worker-pool tests already exercise full-length games.
    """
    import random
    rng = random.Random(42)
    b = Board()
    for _ in range(100):
        if b.check_win():
            break
        moves = b.legal_moves()
        if not moves:
            break
        q, r = rng.choice(moves)
        b.apply_move(q, r)


def _build_p1_win(p1_cells: list) -> "Board":
    """
    Build a board where P1 occupies exactly `p1_cells` (a 6-in-a-row run).

    P2 fillers are drawn from a sparse grid (step=3) centred on P1's bounding
    box.  Spacing of 3 guarantees no two filler stones are consecutive along
    any hex axis, so P2 can never accidentally form a 6-in-a-row.

    Turn structure: P1 opens with 1 move, then 2 each.
    """
    p1_set = set(p1_cells)
    p1_qs = [q for q, r in p1_cells]
    p1_rs = [r for q, r in p1_cells]
    lo_q = min(p1_qs) - 3
    hi_q = max(p1_qs) + 3
    lo_r = min(p1_rs) - 3
    hi_r = max(p1_rs) + 3

    # Step-3 grid: adjacent candidates are ≥3 apart on every axis → no 6-in-a-row.
    p2_candidates = [
        (lo_q + i * 3, lo_r + j * 3)
        for i in range((hi_q - lo_q) // 3 + 3)
        for j in range((hi_r - lo_r) // 3 + 3)
        if (lo_q + i * 3, lo_r + j * 3) not in p1_set
    ]

    b = Board()
    b.apply_move(*p1_cells[0])  # P1 ply0 (single)
    p1_rest = p1_cells[1:]
    p2_placed: set = set()
    p2_idx = 0
    p1_idx = 0
    while p1_idx < len(p1_rest):
        for _ in range(2):
            # Find next candidate that is currently valid (in-window, unoccupied)
            while p2_idx < len(p2_candidates):
                cq, cr = p2_candidates[p2_idx]
                p2_idx += 1
                if (cq, cr) in p2_placed:
                    continue
                try:
                    b.apply_move(cq, cr)
                    p2_placed.add((cq, cr))
                    break
                except ValueError:
                    continue
        for _ in range(2):
            if p1_idx < len(p1_rest):
                b.apply_move(*p1_rest[p1_idx])
                p1_idx += 1
    return b


# ── Compound turn — sequential action space ──────────────────────────────────

def test_compound_turn_state_transitions():
    """A complete 2-stone compound turn must:
    1. Keep the same current_player for both stones.
    2. Decrement moves_remaining from 2 → 1 after the first stone.
    3. Flip current_player and reset moves_remaining to 2 after the second stone.
    """
    b = Board()
    b.apply_move(0, 0)          # P1 ply 0 (single)
    assert b.current_player == -1
    assert b.moves_remaining == 2

    # P2 first stone of compound turn
    b.apply_move(1, 0)
    assert b.current_player == -1, "player must NOT flip mid-turn"
    assert b.moves_remaining == 1
    assert b.ply == 2

    # P2 second stone — completes compound turn
    b.apply_move(2, 0)
    assert b.current_player == 1, "player must flip after full compound turn"
    assert b.moves_remaining == 2
    assert b.ply == 3


def test_intermediate_ply_detection():
    """Verify the condition used to skip Dirichlet noise at intermediate plies.

    Intermediate ply = moves_remaining == 1 AND ply > 0.
    Ply 0 has moves_remaining == 1 but is NOT intermediate (it's P1's full turn).
    """
    b = Board()
    # Ply 0: moves_remaining=1, ply=0 → NOT intermediate
    assert not (b.moves_remaining == 1 and b.ply > 0)

    b.apply_move(0, 0)  # P1 ply 0
    # Now P2's turn start: moves_remaining=2 → NOT intermediate
    assert not (b.moves_remaining == 1 and b.ply > 0)

    b.apply_move(1, 0)  # P2 first stone
    # Intermediate ply: moves_remaining=1, ply=2 → IS intermediate
    assert b.moves_remaining == 1 and b.ply > 0

    b.apply_move(2, 0)  # P2 second stone
    # P1's turn start: moves_remaining=2 → NOT intermediate
    assert not (b.moves_remaining == 1 and b.ply > 0)

    b.apply_move(3, 0)  # P1 first stone
    # Intermediate ply again
    assert b.moves_remaining == 1 and b.ply > 0

    b.apply_move(4, 0)  # P1 second stone
    # P2's turn start
    assert not (b.moves_remaining == 1 and b.ply > 0)


_WINNING_POSITIONS = [
    # (description, q_start, r_start, dq, dr)
    ("E  center",     0,  0,  1,  0),
    ("E  top row",    0,  9,  1,  0),
    ("E  bottom row", 0, -9,  1,  0),
    ("E  low q",     -9,  0,  1,  0),
    ("E  high q",     4,  0,  1,  0),
    ("E  corner",    -9,  9,  1,  0),
    ("NE center",     0, -3,  0,  1),
    ("NE low q",     -9, -3,  0,  1),
    ("NE high q",     9, -3,  0,  1),
    ("NE low r",      0, -9,  0,  1),
    ("NE high r",     0,  4,  0,  1),
    ("NE corner",    -9, -9,  0,  1),
    ("SE center",    -3,  3,  1, -1),
    ("SE top r",     -3,  9,  1, -1),
    ("SE low r",     -3, -4,  1, -1),
    ("SE low q",     -9,  9,  1, -1),
    ("SE high q",     4,  9,  1, -1),
    ("NW center",     3, -3, -1,  1),
    ("NW corner",     9, -9, -1,  1),
    ("NW top",        9,  4, -1,  1),
]


@pytest.mark.parametrize(
    "desc,q0,r0,dq,dr",
    _WINNING_POSITIONS,
    ids=[p[0].strip() for p in _WINNING_POSITIONS],
)
def test_hand_crafted_winning_position(desc: str, q0: int, r0: int, dq: int, dr: int) -> None:
    """Given: P1 plays 6 in a row at a specific location and axis.
    When: the board is evaluated.
    Then: check_win() is True and winner() is 1.

    Covers E, NE, SE/NW axes at 20 distinct board locations including corners and edges.
    """
    cells = [(q0 + i * dq, r0 + i * dr) for i in range(6)]
    valid = all(-9 <= q <= 9 and -9 <= r <= 9 for q, r in cells)
    if not valid:
        pytest.skip(f"position '{desc}' has out-of-bounds cells: {cells}")
    b = _build_p1_win(cells)
    assert b.check_win() and b.winner() == 1, \
        f"Expected P1 win at '{desc}': cells={cells}"
