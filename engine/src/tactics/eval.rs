//! Static pattern eval — `3^6 = 729` ternary 6-cell pattern table (D-PFIT P2
//! increment 2; SealBot port of `board.h:185 _move_delta` / `:63 _eval_score`).
//!
//! Each length-6 window over a hex axis maps to a ternary index
//! `pi = Σ_j cell_j · 3^j` (`cell ∈ {0 empty, 1 mine, 2 theirs}`, side-to-move
//! perspective); `PATTERN_TABLE[pi]` is the precomputed window weight. A window
//! with BOTH colours is dead (0 — blocked, neither side can complete six there);
//! a pure run scores a run-length-weighted potential, antisymmetric in colour.
//!
//! # SOUNDNESS INVARIANT — the load-bearing property
//! The static eval is for move ORDERING and NON-PROOF heuristic-leaf scores
//! ONLY. A heuristic-leaf score is reported as **UNKNOWN, never as a proof**:
//! `search::solve` feeds it through `Scored::heuristic`/`clamp_heuristic`, which
//! pins it strictly inside `(-WIN_THRESHOLD, WIN_THRESHOLD)`, so it can NEVER
//! read as a mate. Only terminal backups (`terminal_value_to_move`, CF-1) and the
//! stone-count shortcuts in `search.rs` declare WIN/LOSS; the value head is never
//! read. Alpha-beta cutoffs the eval enables cannot change a root verdict
//! (root = full window, exact) nor a proven LOSS (concluded only on a fully
//! examined, non-cutoff node — see `search::solve` soundness note).
//!
//! # DEFERRED (later perf increments — NOT built here)
//! - the INCREMENTAL `_eval_score` accumulator (make/undo delta); this is a
//!   per-leaf full re-scan — correct-first, the incremental hoist is increment 3+.
//! - eval-tie-break ORDERING wiring (`ordering.rs` step 6) + net-policy ordering.
//! - PVS / LMR / aspiration / killers / history.

#![allow(dead_code)]

use fxhash::FxHashSet;

use crate::board::{Board, Cell, Player, HEX_AXES};

use super::WIN_THRESHOLD;

/// `3^6` — one entry per length-6 ternary window.
const N_PATTERNS: usize = 729;

/// Run-length potential weights `W[count]` for a PURE (single-colour) length-6
/// window. Strongly superlinear so the search prefers longer runs / earlier
/// cutoffs; symmetric for the opponent (negated). Ordering/heuristic ONLY — the
/// exact values are not load-bearing for soundness (every leaf is clamped).
const RUN_WEIGHT: [i32; 7] = [0, 1, 5, 25, 125, 625, 3125];

/// The 729-entry ternary pattern table, built once. `PATTERN_TABLE[pi]` =
/// `0` if the window holds BOTH colours (dead), else `W[mine] - W[theirs]`.
fn pattern_table() -> &'static [i32; N_PATTERNS] {
    use std::sync::OnceLock;
    static TABLE: OnceLock<[i32; N_PATTERNS]> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut t = [0i32; N_PATTERNS];
        for (pi, slot) in t.iter_mut().enumerate() {
            let (mut mine, mut theirs) = (0usize, 0usize);
            let mut x = pi;
            for _ in 0..6 {
                match x % 3 {
                    1 => mine += 1,
                    2 => theirs += 1,
                    _ => {}
                }
                x /= 3;
            }
            // Mixed window => dead (blocked): neither side can make six in it.
            *slot = if mine > 0 && theirs > 0 { 0 } else { RUN_WEIGHT[mine] - RUN_WEIGHT[theirs] };
        }
        t
    })
}

#[inline]
fn mine_cell(p: Player) -> Cell {
    match p {
        Player::One => Cell::P1,
        Player::Two => Cell::P2,
    }
}

/// Heuristic static score for `board` from the side-to-move's perspective,
/// summed over every length-6 window that overlaps a stone (each window scored
/// ONCE). `None` on an empty board (nothing to score). The result is clamped
/// strictly inside the proof region so a caller can never mistake it for a mate.
///
/// Move-ordering hint / UNKNOWN-leaf value ONLY — NEVER a proof. Wiring this in
/// MUST NOT change any WIN/LOSS proof in `search.rs` (it does not: every leaf is
/// re-clamped by `clamp_heuristic`, and α-β preserves the exact root verdict).
pub(crate) fn static_eval(board: &Board) -> Option<i32> {
    if board.cells.is_empty() {
        return None;
    }
    let me = mine_cell(board.current_player);
    let table = pattern_table();

    // Distinct windows overlapping a stone: anchor = stone shifted back 0..5 along
    // each axis. The HashSet de-dups so every window is scored exactly once.
    let mut windows: FxHashSet<(i32, i32, usize)> = FxHashSet::default();
    for &(sq, sr) in board.cells.keys() {
        for (ai, &(dq, dr)) in HEX_AXES.iter().enumerate() {
            for k in 0..6i32 {
                windows.insert((sq - k * dq, sr - k * dr, ai));
            }
        }
    }

    let mut score: i64 = 0;
    for (aq, ar, ai) in windows {
        let (dq, dr) = HEX_AXES[ai];
        let mut pi = 0usize;
        let mut pow = 1usize;
        for k in 0..6i32 {
            let v = match board.get(aq + k * dq, ar + k * dr) {
                Cell::Empty => 0,
                c if c == me => 1,
                _ => 2,
            };
            pi += v * pow;
            pow *= 3;
        }
        score += table[pi] as i64;
    }

    let bound = (WIN_THRESHOLD - 1) as i64;
    Some(score.clamp(-bound, bound) as i32)
}

/// Heuristic value for a NON-PROOF leaf (horizon / quiet node), side-to-move
/// perspective. `None` (empty board) => `0`. The caller (`search::solve`) CLAMPS
/// this strictly inside the proof region (`clamp_heuristic`), so it can NEVER
/// masquerade as a mate — the soundness invariant: a heuristic leaf is never a
/// proof.
#[inline]
pub(crate) fn heuristic_leaf(board: &Board) -> i32 {
    static_eval(board).unwrap_or(0)
}

// ── Tests ──────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    /// Build a ternary pattern index from 6 cell codes (0 empty, 1 mine, 2 opp).
    fn idx(cells: [usize; 6]) -> usize {
        let mut pi = 0usize;
        let mut pow = 1usize;
        for c in cells {
            pi += c * pow;
            pow *= 3;
        }
        pi
    }

    fn board_with(stones: &[((i32, i32), Cell)], stm: Player) -> Board {
        let mut b = Board::new();
        for &((q, r), c) in stones {
            b.cells.insert((q, r), c);
        }
        b.has_stones = !stones.is_empty();
        b.current_player = stm;
        b
    }

    #[test]
    fn pattern_table_dead_window_is_zero() {
        // A window with BOTH colours is blocked => 0 (neither can complete six).
        let t = pattern_table();
        assert_eq!(t[idx([1, 2, 0, 0, 0, 0])], 0, "mine+opp window must be dead");
        assert_eq!(t[idx([1, 1, 1, 2, 0, 0])], 0, "any mixed window must be dead");
        assert_eq!(t[idx([0, 0, 0, 0, 0, 0])], 0, "empty window scores 0");
    }

    #[test]
    fn pattern_table_run_length_monotonic_and_antisymmetric() {
        let t = pattern_table();
        // Pure-mine potential strictly increases with run length; pure-opp mirrors
        // it negative; and swapping colours negates the weight (antisymmetry).
        let mut prev = i32::MIN;
        for n in 0..=6usize {
            let mut mine = [0usize; 6];
            let mut opp = [0usize; 6];
            for c in mine.iter_mut().take(n) {
                *c = 1;
            }
            for c in opp.iter_mut().take(n) {
                *c = 2;
            }
            let vm = t[idx(mine)];
            let vo = t[idx(opp)];
            assert!(vm > prev, "pure-mine weight must increase with run length at n={n}");
            prev = vm;
            assert_eq!(vo, -vm, "colour swap must negate the weight at n={n}");
        }
    }

    #[test]
    fn static_eval_none_on_empty_board() {
        assert_eq!(static_eval(&Board::new()), None, "empty board has nothing to score");
    }

    #[test]
    fn static_eval_sign_follows_side_to_move_advantage() {
        // P1 holds an open length-4 run; the rest empty. From P1-to-move the eval
        // is positive (P1 has the potential), from P2-to-move it is the negation.
        let stones: Vec<((i32, i32), Cell)> =
            (0..4).map(|q| ((q, 0), Cell::P1)).collect();
        let as_p1 = static_eval(&board_with(&stones, Player::One)).unwrap();
        let as_p2 = static_eval(&board_with(&stones, Player::Two)).unwrap();
        assert!(as_p1 > 0, "side-with-the-run must score positive, got {as_p1}");
        assert_eq!(as_p2, -as_p1, "flipping the side-to-move must negate the eval");
    }

    #[test]
    fn static_eval_never_reaches_proof_region() {
        // SOUNDNESS: a heuristic leaf can NEVER masquerade as a mate, even on a
        // dense board with many long runs. `heuristic_leaf` stays strictly inside
        // the proof region for every constructed position.
        let dense: Vec<((i32, i32), Cell)> = (0..20)
            .map(|q| ((q, 0), if q % 2 == 0 { Cell::P1 } else { Cell::P2 }))
            .chain((0..20).map(|q| ((q, 1), Cell::P1)))
            .collect();
        for b in [
            board_with(&(0..6).map(|q| ((q, 0), Cell::P1)).collect::<Vec<_>>(), Player::One),
            board_with(&dense, Player::One),
            board_with(&dense, Player::Two),
        ] {
            let v = heuristic_leaf(&b);
            assert!(v.abs() < WIN_THRESHOLD, "heuristic leaf {v} leaked into the proof region");
        }
    }
}
