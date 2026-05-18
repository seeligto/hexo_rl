//! INV27 — `find_winning_line` surfaces the winning 6-line on
//! colony-classified game endings (§S178 F-fix-1).
//!
//! Pre-fix mechanism (`engine/src/board/moves.rs::find_winning_line`):
//! the function only scanned outward from `last_move`. When two stones are
//! placed per turn (HTT rule) the **first** move of a turn can complete the
//! 6-line, but `last_move` is then overwritten by the **second** move which
//! is typically off-line. `winner()` still finds the win via its all-stones
//! fallback (`player_wins` at moves.rs:184-189), so `terminal_reason`
//! classifies the game as colony (= "winner exists, no winning_line"). The
//! threat-head target column in the per-row aux buffer is then all-zero on
//! every position of that game — the threat head learns "no 6-line exists"
//! and gradient-suppresses the threat-detection circuit. SA-A verdict 6 in
//! `reports/s178_pre_design_investigation.md`.
//!
//! Fix: `find_winning_line` falls back to scanning all stones (mirroring
//! `player_wins`'s fallback) when last_move doesn't surface a line.
//!
//! Cells:
//!   1. winner exists, last_move ON the winning line → 6 cells (fast path).
//!   2. winner exists, last_move OFF the winning line (opponent's stone)
//!      → 6 cells via fallback (NEW INVARIANT).
//!   3. no winner (draw, ply < max_moves) → empty.
//!   4. no winner (ply cap, ply == max_moves) → empty.

use engine::board::{Board, Player};

/// Cell 1 — Fast path. Last move completes a 6-line; line returned from
/// last_move's perspective.
#[test]
fn inv27_cell_1_winner_last_move_on_line() {
    let mut board = Board::new();
    // P1 line along r=0: (0,0), (1,0), (2,0), (3,0), (4,0), (5,0).
    // P2 scattered to never form a 6-line.
    board.apply_move(0, 0).unwrap();    // ply 0  P1 (opener)
    board.apply_move(10, 0).unwrap();   // ply 1  P2
    board.apply_move(15, 0).unwrap();   // ply 2  P2 — turn flip
    board.apply_move(1, 0).unwrap();    // ply 3  P1
    board.apply_move(2, 0).unwrap();    // ply 4  P1 — turn flip
    board.apply_move(10, 5).unwrap();   // ply 5  P2
    board.apply_move(15, 5).unwrap();   // ply 6  P2 — turn flip
    board.apply_move(3, 0).unwrap();    // ply 7  P1
    board.apply_move(4, 0).unwrap();    // ply 8  P1 — turn flip
    board.apply_move(20, 0).unwrap();   // ply 9  P2
    board.apply_move(20, 5).unwrap();   // ply 10 P2 — turn flip
    board.apply_move(5, 0).unwrap();    // ply 11 P1 — completes 6-line; last_move ON line

    assert_eq!(board.winner(), Some(Player::One), "P1 must be the winner");
    let line = board.find_winning_line();
    assert_eq!(line.len(), 6, "Cell 1: fast path must return 6 cells");
    let on_line: std::collections::HashSet<(i32, i32)> = line.into_iter().collect();
    let expected: std::collections::HashSet<(i32, i32)> =
        (0..6i32).map(|q| (q, 0)).collect();
    assert_eq!(on_line, expected, "Cell 1: line must be {{(0..6, 0)}}");
}

/// Cell 2 — Fallback path (THE NEW INVARIANT). Winning line exists, but
/// last_move is the OPPONENT's stone — fast path returns empty, fallback
/// scan must still find the line.
#[test]
fn inv27_cell_2_winner_last_move_off_line_colony_classification() {
    let mut board = Board::new();
    // Same P1 6-line at r=0, but two more P2 turns AFTER P1's win,
    // leaving last_move = P2 stone (no 6-line touching last_move).
    board.apply_move(0, 0).unwrap();    // ply 0  P1
    board.apply_move(10, 0).unwrap();   // ply 1  P2
    board.apply_move(15, 0).unwrap();   // ply 2  P2 — turn flip
    board.apply_move(1, 0).unwrap();    // ply 3  P1
    board.apply_move(2, 0).unwrap();    // ply 4  P1 — turn flip
    board.apply_move(10, 5).unwrap();   // ply 5  P2
    board.apply_move(15, 5).unwrap();   // ply 6  P2 — turn flip
    board.apply_move(3, 0).unwrap();    // ply 7  P1
    board.apply_move(4, 0).unwrap();    // ply 8  P1 — turn flip
    board.apply_move(20, 0).unwrap();   // ply 9  P2
    board.apply_move(20, 5).unwrap();   // ply 10 P2 — turn flip
    board.apply_move(5, 0).unwrap();    // ply 11 P1 — 6-line complete
    board.apply_move(-1, -1).unwrap();  // ply 12 P1 — off-line second move (turn flip)
    board.apply_move(25, 0).unwrap();   // ply 13 P2
    board.apply_move(25, 5).unwrap();   // ply 14 P2 — last_move OFF line, NOT P1

    // winner() finds P1 via the all-stones fallback in player_wins.
    assert_eq!(board.winner(), Some(Player::One),
        "Cell 2: winner() must find P1 via fallback scan");

    // find_winning_line MUST also find the line — this is the §S178 F-fix-1 invariant.
    // Without the fix this returned vec![]; the threat-head target then
    // had every cell == 0 across every row of the game.
    let line = board.find_winning_line();
    assert_eq!(line.len(), 6,
        "Cell 2 INVARIANT: find_winning_line must return 6 cells when last_move \
         is off the winning line (was empty pre-§S178 F-fix-1; threat target then \
         all-zero on colony games — SA-A verdict 6 in reports/s178_pre_design_investigation.md)");
    let on_line: std::collections::HashSet<(i32, i32)> = line.into_iter().collect();
    let expected: std::collections::HashSet<(i32, i32)> =
        (0..6i32).map(|q| (q, 0)).collect();
    assert_eq!(on_line, expected, "Cell 2: line must be {{(0..6, 0)}}");
}

/// Cell 3 — No winner, ply < max_moves (organic draw): empty line.
#[test]
fn inv27_cell_3_no_winner_organic_draw_empty_line() {
    let mut board = Board::new();
    // A handful of scattered moves, no 6-line.
    board.apply_move(0, 0).unwrap();    // ply 0 P1
    board.apply_move(5, 5).unwrap();    // ply 1 P2
    board.apply_move(-3, 4).unwrap();   // ply 2 P2 — turn flip
    board.apply_move(2, 2).unwrap();    // ply 3 P1
    board.apply_move(-4, -4).unwrap();  // ply 4 P1 — turn flip

    assert_eq!(board.winner(), None, "Cell 3: no winner expected");
    assert!(board.find_winning_line().is_empty(),
        "Cell 3: no winner → find_winning_line returns empty");
}

/// Cell 4 — No winner, ply cap reached (equivalent to Cell 3 for
/// find_winning_line — both depend on `winner == None` returning empty).
#[test]
fn inv27_cell_4_no_winner_ply_cap_empty_line() {
    // For find_winning_line's contract, ply cap vs organic draw are
    // indistinguishable — both return empty because no 6-line exists.
    // This cell pins that contract.
    let mut board = Board::new();
    board.apply_move(0, 0).unwrap();    // ply 0 P1
    board.apply_move(10, 10).unwrap();  // ply 1 P2
    board.apply_move(11, 10).unwrap();  // ply 2 P2 — turn flip

    assert_eq!(board.winner(), None);
    assert!(board.find_winning_line().is_empty());
}
