//! Candidate generation + ordering for the threat-space proof.
//!
//! FOUNDATION: the threat-guided narrow candidate set (port of
//! `solver.py::_candidates`) — keeps branching ~4-15 so deep FORCING mates are
//! reachable cheaply. Built on the V4-fuzzed engine primitives
//! (`winning_moves`/`threat_moves`), in-engine so each call is one pass over the
//! legal set, not a clone per cell.
//!
//! # DEFERRED (the AlphaZero ordering ingredient — `NATIVE_RUST_SOLVER_design.md` §3.4)
//! The full best-first pipeline is NOT built here:
//!   1. TT best move
//!   2. winning_moves (immediate)            <- present (via in-check must_block)
//!   3. threat_moves + responses (forcing)   <- present
//!   4. NET-POLICY prior (the learned ordering, the SealBot-lacks lever) <- TODO
//!   5. killers, then history                 <- TODO
//!   6. static `_move_delta` tie-break (eval.rs 3^6 table) <- TODO
//! Net-policy ordering only ACCELERATES (earlier cutoffs); the threat
//! enumeration here GUARANTEES a 0-prior refuter is still searched (T0: 67% of
//! refuters have ~0 deploy prior). Ordering never affects soundness.
//!
//! Crucially this is the THREAT-ONLY set = the measured **8% ceiling**: it omits
//! the QUIET developmental moves that start most mates. Adding the quiet-move
//! alpha-beta body (the hard, load-bearing part) is its own focused effort.

use fxhash::FxHashSet;

use crate::board::{Board, Player};

/// Threat-guided candidate set for the side-to-move `stm` (`solver.py::_candidates`).
///
/// - `stm` in check (opponent threatens an immediate win): block the threat(s)
///   (`winning_moves(opp)` — the must-hit cells) ∪ `stm`'s counter-threats. A
///   quiet move loses to the standing threat, so omitting non-responses cannot
///   turn a real escape into a false LOSS.
/// - `stm` not in check: `stm`'s threat-creating moves (attack) ∪ the
///   opponent's threat-creating cells (defensive pre-emption). Restricting to
///   the threat region is the standard TSS prune — guarded by the soundness
///   fuzz.
///
/// Deterministic ordering: must-block (sorted) first, then threats in
/// legal-sorted order. Capped at `cand_cap` (must-block kept first).
pub(crate) fn candidates(
    board: &Board,
    stm: Player,
    opp: Player,
    cand_cap: usize,
    neighbor_dist: Option<i32>,
) -> Vec<(i32, i32)> {
    let must_block = board.winning_moves(opp);
    if !must_block.is_empty() {
        // IN CHECK: the threat-only set (blocks ∪ counter-threats) is already
        // complete (a non-response loses to the standing threat) — NOT widened.
        let mut seen: FxHashSet<(i32, i32)> = must_block.iter().copied().collect();
        let mut out = must_block;
        for m in board.threat_moves(stm) {
            if seen.insert(m) {
                out.push(m);
            }
        }
        out.truncate(cand_cap);
        return out;
    }

    // NOT IN CHECK. Threat-creating moves first (forcing — searched regardless of
    // any prior, the 0-prior-refuter guarantee), then — under the quiet-move body
    // — the developmental neighbour cells the threat-only set omits.
    let mut seen: FxHashSet<(i32, i32)> = FxHashSet::default();
    let mut out: Vec<(i32, i32)> = Vec::new();
    for m in board.threat_moves(stm).into_iter().chain(board.threat_moves(opp)) {
        if seen.insert(m) {
            out.push(m);
        }
    }

    // Quiet-move widening (Track 3, the lever past the 8% ceiling). Append every
    // empty legal cell within cheb-distance `d` of a stone, in sorted order
    // (deterministic). When `d` covers the legal radius the set becomes the FULL
    // legal set, so a not-in-check LOSS satisfies the R3 guard's exhaustiveness
    // branch (`moves_len >= legal_move_count`) and is proven soundly. Quiet cells
    // come AFTER threats so forcing moves are still ordered first.
    if let Some(d) = neighbor_dist {
        for c in board.legal_moves() {
            if !seen.contains(&c)
                && board
                    .cells
                    .keys()
                    .any(|&(sq, sr)| (c.0 - sq).abs().max((c.1 - sr).abs()) <= d)
            {
                seen.insert(c);
                out.push(c);
            }
        }
    }

    out.truncate(cand_cap);
    out
}
