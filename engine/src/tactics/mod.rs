//! `engine::tactics` — native Rust in-window-offense tactical proof solver.
//!
//! D-DECODE Track 3 FOUNDATION. Ports the AND-OR threat-space proof skeleton
//! from the Python reference `scripts/dtactical/solver.py` onto the native
//! `Board`, with **zero board clone per node** (`apply_move_tracked` /
//! `undo_move` + incremental u128 zobrist).
//!
//! # Scope (D-DECODE simplification)
//! **IN-WINDOW-OFFENSE-ONLY.** Off-window defense is owned by the Track-2
//! multi-window decoding fix (commit 0338385) — this solver DROPS the
//! off-window-capability requirement. A WIN proof whose played move lands
//! outside the perception window (`window_half`, default 9) is suppressed
//! (downgraded to UNKNOWN); Track-2 owns that band.
//!
//! # Soundness invariant (the load-bearing property)
//! The net value head is **NEVER read inside the search**. A proof is ONLY a
//! terminal backup (`Board::terminal_value_to_move`, CF-1, the single
//! engine-owned sign) or a forced-win / double-threat stone-count shortcut.
//! Never a heuristic eval. The `eval.rs` static eval (deferred) is for move
//! ORDERING only and reports UNKNOWN, never a proof — preserving the
//! "0 soundness violations" property the A1 instrument measured. The
//! `#[cfg(test)]` soundness fuzz cross-checks every LOSS claim against an
//! independent exhaustive `brute_solve` (port of `solver.py::_brute_solve`)
//! and asserts 0 false-LOSS.
//!
//! # What is DEFERRED (FOUNDATION-ONLY dispatch — see file TODOs)
//! - `eval.rs`: the 3^6 ternary static pattern eval (perf/ordering tie-break).
//! - `tt.rs`: the 2-slot aged depth-preferred quantized TT (only a basic
//!   proven-result cache is built now).
//! - `ordering.rs`: net-policy ordering + killers + history (only the
//!   threat-guided candidate set is built now).
//! - The quiet-move alpha-beta BODY + threat-quiescence tail + PVS/LMR/
//!   aspiration. The current proof core is threat/double-threat-based =
//!   the measured **8% ceiling**; the quiet-move body is its own effort.

pub mod eval;
pub mod ordering;
pub mod search;
pub mod tt;

use crate::board::Board;

/// 3-valued proof result for the side-to-move (mirrors `solver.py` WIN/LOSS/UNKNOWN).
/// UNKNOWN = not determined within depth/budget (NOT a draw, NOT a proof).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Outcome {
    /// Side-to-move has a proven forced win in the explored (threat) subtree.
    Win,
    /// Side-to-move is in a proven forced loss in the explored subtree.
    Loss,
    /// Unresolved within depth/budget — never a proof.
    Unknown,
}

impl Outcome {
    /// Flip WIN<->LOSS; UNKNOWN stays UNKNOWN (negamax for HTTT compound turns).
    #[inline]
    pub fn negate(self) -> Self {
        match self {
            Outcome::Win => Outcome::Loss,
            Outcome::Loss => Outcome::Win,
            Outcome::Unknown => Outcome::Unknown,
        }
    }

    /// Int mapping matching `solver.py`: WIN=1, LOSS=-1, UNKNOWN=0.
    /// The A1 `solver_probe` DI shim maps WIN -> score >= WIN_THRESHOLD.
    #[inline]
    pub fn to_i32(self) -> i32 {
        match self {
            Outcome::Win => 1,
            Outcome::Loss => -1,
            Outcome::Unknown => 0,
        }
    }
}

/// Node-budget meter (board expansions). The honesty axis vs the ~150-sim
/// deploy search — mirrors `solver.py::Budget` exactly: `cap` ticks pass
/// (nodes 1..=cap), the `cap+1`-th sets `exhausted` and fails.
pub struct Budget {
    cap: u64,
    pub nodes: u64,
    pub exhausted: bool,
}

impl Budget {
    pub fn new(cap: u64) -> Self {
        Budget { cap, nodes: 0, exhausted: false }
    }

    /// Charge one node. Returns false (and latches `exhausted`) once over cap.
    #[inline]
    pub fn tick(&mut self) -> bool {
        self.nodes += 1;
        if self.nodes > self.cap {
            self.exhausted = true;
            false
        } else {
            true
        }
    }
}

/// Solver configuration. `window_half`/`cand_cap` default to the v6
/// single-window band (9) and the `solver.py` candidate cap (40).
#[derive(Clone, Copy, Debug)]
pub struct TacticalConfig {
    /// Threat-guided candidate cap per node (matches `solver.py` cand_cap=40).
    pub cand_cap: usize,
    /// In-window offense guard: `Some(h)` suppresses a WIN whose played move is
    /// cheb-distance > `h` from the window center. `None` disables the guard
    /// (full game-theoretic result). Default `Some(9)` for v6 single-window.
    pub window_half: Option<i32>,
}

impl Default for TacticalConfig {
    fn default() -> Self {
        TacticalConfig { cand_cap: 40, window_half: Some(9) }
    }
}

/// Result of a `prove` call: the 3-valued outcome, the principal variation
/// (the move LINE — populated for WIN; `line[0]`/`line[1]` are the side-to-
/// move's two stones for a 2-stone-turn forcing win, the A1 override path),
/// the node count, and whether the budget was exhausted.
#[derive(Clone, Debug)]
pub struct ProofResult {
    pub result: Outcome,
    pub line: Vec<(i32, i32)>,
    pub nodes: u64,
    pub budget_exhausted: bool,
}

/// The native tactical solver. NET-FREE proof core; net only ever ORDERS
/// (ordering wiring deferred — see `ordering.rs`).
pub struct TacticalSolver {
    config: TacticalConfig,
}

impl TacticalSolver {
    pub fn new(config: TacticalConfig) -> Self {
        TacticalSolver { config }
    }

    /// Try to prove the side-to-move at `board`. Clones the board ONCE at entry
    /// (per-CALL, not per-node) then runs zero-clone descent. Leaves `board`
    /// untouched. Generalizes `solver.py::prove_loss`: `result` carries the
    /// full WIN/LOSS/UNKNOWN (A1 offense overrides on WIN + `line`).
    pub fn prove(&self, board: &Board, max_depth: u32, node_budget: u64) -> ProofResult {
        let mut scratch = board.clone();
        self.prove_in_place(&mut scratch, max_depth, node_budget)
    }

    /// Zero-clone entry for the future deploy root hook: searches in place via
    /// `apply_move_tracked`/`undo_move` and restores `board` exactly on return.
    pub fn prove_in_place(
        &self,
        board: &mut Board,
        max_depth: u32,
        node_budget: u64,
    ) -> ProofResult {
        let mut budget = Budget::new(node_budget);
        let mut tt = tt::ProofTt::new();
        let solved = search::solve(board, max_depth as i32, &mut budget, &self.config, &mut tt);

        let mut result = solved.outcome;
        let mut line = solved.line;

        // In-window offense guard: the move we would PLAY (line[0]) must be in
        // the perception window. Off-window proofs are Track-2's scope; suppress
        // here so the A1 override never fires off-window (mirrors
        // `solver_backup_bot.py::_is_off_window`).
        if result == Outcome::Win {
            if let Some(half) = self.config.window_half {
                if line.first().is_some_and(|&first| is_off_window(board, first, half)) {
                    result = Outcome::Unknown;
                    line = Vec::new();
                }
            }
        }

        ProofResult {
            result,
            line,
            nodes: budget.nodes,
            budget_exhausted: budget.exhausted,
        }
    }
}

/// True if `mv` is off the single GLOBAL perception window: cheb-distance from
/// the bbox-centroid window center exceeds `half`. Mirrors the engine
/// `window_center`/`to_flat` off-window test and `solver_backup_bot.py`'s
/// `_is_off_window` (both truncate-toward-zero the centroid).
pub(crate) fn is_off_window(board: &Board, mv: (i32, i32), half: i32) -> bool {
    let (cq, cr) = board.window_center();
    (mv.0 - cq).abs().max((mv.1 - cr).abs()) > half
}
