//! Static pattern eval — DEFERRED STUB (move-ordering only, never a proof).
//!
//! This file is a scaffold hook for the SealBot-grade `3^6 = 729` ternary
//! 6-cell pattern table (`NATIVE_RUST_SOLVER_design.md` §1.2 / §3.1 / §3.5):
//! each length-6 window maps to an index `pi = Σ cell·3^j`, weight `_pv[pi]`,
//! maintained incrementally on make/undo.
//!
//! # SOUNDNESS INVARIANT — read before wiring this in
//! The static eval is for move ORDERING (and, later, non-proof heuristic-leaf
//! scores) ONLY. A heuristic-leaf score is reported as **UNKNOWN, never as a
//! proof**. Only terminal backups (`terminal_value_to_move`, CF-1) and the
//! stone-count shortcuts in `search.rs` may declare WIN/LOSS. The net value head
//! is never read either. This keeps the "0 soundness violations" property the
//! A1 instrument measured.
//!
//! # DEFERRED (the ~2M-NPS perf work — NOT built in this FOUNDATION dispatch)
//! - the 729-entry ternary pattern table + incremental `_eval_score` accumulator
//! - the quiet-move alpha-beta BODY this eval would order/tie-break
//! - PVS / LMR / aspiration / killers / history
//! Until then the threat/double-threat proof core in `search.rs` is sound on its
//! own (it is the measured 8% ceiling — threat-only, no quiet moves).

#![allow(dead_code)]

use crate::board::Board;

/// Heuristic static score for `board` from the side-to-move's perspective.
///
/// DEFERRED: returns `None` (no static eval). Callers MUST treat `None` (and any
/// future `Some`) as a move-ordering hint / UNKNOWN leaf — NEVER as a proof.
/// Wiring a real value here MUST NOT change any WIN/LOSS proof in `search.rs`.
pub(crate) fn static_eval(_board: &Board) -> Option<i32> {
    // TODO(track3-perf): port the 3^6 ternary pattern table (eval.rs of the
    // SealBot fork) + incremental accumulator; use ONLY for ordering tie-break
    // (ordering.rs step 6) and heuristic non-proof leaves (reported UNKNOWN).
    None
}
