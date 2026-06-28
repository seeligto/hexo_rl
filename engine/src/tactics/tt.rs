//! Transposition table — proven-result cache.
//!
//! FOUNDATION: a basic game-theoretic cache of PROVEN outcomes keyed by
//! `(zobrist_hash, side_to_move, moves_remaining)` — depth-INDEPENDENT (a
//! proven win/loss is a proven win/loss regardless of the depth it was found
//! at). Mirrors `solver.py`'s `tt` dict.
//!
//! Only LOSS is cached (see the WIN-PV note in `search.rs::solve`): a WIN cache
//! hit returns an empty principal variation and could truncate the A1 override
//! line, so WINs are always reconstructed fresh. LOSS needs no line and is the
//! expensive case (must refute every defense) — so it carries the node savings.
//! UNKNOWN is never cached (budget/depth-dependent, not game-theoretic).
//!
//! # DEFERRED (the perf TT — `NATIVE_RUST_SOLVER_design.md` §1.3 / §3.1)
//! The 2-slot generation-aged bucket, depth-preferred + always-replace slots,
//! `int16` mate-distance-encoded score quantization, 64-byte alignment, and the
//! best-move-for-ordering field are NOT built here. This is a plain `FxHashMap`
//! proven-result cache. When the quiet-move alpha-beta body lands, replace this
//! with the packed aged TT and store the EXACT/LOWER/UPPER flag + best move.

use fxhash::FxHashMap;

use super::Outcome;

/// Key: (zobrist u128, side-to-move as i8, moves_remaining u8).
pub type TtKey = (u128, i8, u8);

/// Proven-outcome cache. Stores only game-theoretic WIN/LOSS (currently LOSS
/// only, per the WIN-PV note above); never UNKNOWN.
pub struct ProofTt {
    map: FxHashMap<TtKey, Outcome>,
}

impl ProofTt {
    pub fn new() -> Self {
        ProofTt { map: FxHashMap::default() }
    }

    #[inline]
    pub fn get(&self, key: TtKey) -> Option<Outcome> {
        self.map.get(&key).copied()
    }

    /// Insert a PROVEN outcome. UNKNOWN is rejected (debug-asserted) — caching it
    /// would be unsound (it is budget/depth-dependent, not game-theoretic).
    #[inline]
    pub fn insert(&mut self, key: TtKey, outcome: Outcome) {
        debug_assert_ne!(outcome, Outcome::Unknown, "UNKNOWN must never be cached (not game-theoretic)");
        self.map.insert(key, outcome);
    }
}

impl Default for ProofTt {
    fn default() -> Self {
        Self::new()
    }
}
