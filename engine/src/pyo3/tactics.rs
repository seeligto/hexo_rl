//! Python-visible `TacticalSolver` wrapper.
//!
//! Binds `engine::tactics::TacticalSolver` so the offline A1 harness can drive
//! the native solver with no new perf build, and the future `solver_probe` DI
//! hook (`hexo_rl/eval/solver_backup_bot.py`) can swap SealBot for this engine.
//!
//! # A1 / `ProbeFn` contract
//! `prove(board, depth, node_budget) -> (result:int, line:list[(q,r)], nodes:int)`
//! where `result` is WIN=1 / LOSS=-1 / UNKNOWN=0 (matches `solver.py`). The A1
//! `ProbeFn` is `(state, board) -> (List[Move], score)` with a proven mate at
//! `score >= WIN_THRESHOLD`. The (deferred, step-6) Python shim is 4 lines:
//!
//! ```python
//! WIN_THRESHOLD = 99_999_000  # solver_backup_bot.py
//! def solver_probe(state, board):
//!     result, line, _nodes = solver.prove(board, depth, node_budget)
//!     score = 1e8 if result == 1 else (-1e8 if result == -1 else 0.0)
//!     return ([tuple(m) for m in line], score)
//! ```
//!
//! A `result == 1` (WIN) maps to `score >= WIN_THRESHOLD`, so the EXISTING A1
//! override path (`if result and last_score >= self._thr`) fires UNCHANGED, with
//! `line[0]` = the move to play and `line[1]` = the cached 2nd stone of the turn.

use pyo3::prelude::*;

use crate::pyo3::board::PyBoard;
use crate::tactics::{TacticalConfig, TacticalSolver};

/// Native in-window-offense tactical proof solver (D-DECODE Track 3).
///
/// `window_half`: in-window offense guard — a WIN whose played move is
/// cheb-distance > `window_half` from the window center is suppressed
/// (downgraded to UNKNOWN); `None` disables the guard. Default `9` (v6
/// single-window). `cand_cap`: threat-guided candidate cap (default 40).
#[pyclass(name = "TacticalSolver")]
pub struct PyTacticalSolver {
    inner: TacticalSolver,
}

#[pymethods]
impl PyTacticalSolver {
    #[new]
    #[pyo3(signature = (window_half = Some(9), cand_cap = 40, neighbor_dist = None))]
    pub fn new(window_half: Option<i32>, cand_cap: usize, neighbor_dist: Option<i32>) -> Self {
        PyTacticalSolver {
            inner: TacticalSolver::new(TacticalConfig { cand_cap, window_half, neighbor_dist }),
        }
    }

    /// Prove the side-to-move at `board` within `depth` plies and `node_budget`
    /// board expansions.
    ///
    /// Returns `(result, line, nodes)`:
    ///   - `result`: 1 = WIN (side-to-move has a proven forced win),
    ///     -1 = LOSS, 0 = UNKNOWN (unresolved / off-window-suppressed).
    ///   - `line`: principal variation as `[(q, r), ...]` — populated for WIN;
    ///     `line[0]` is the move to play, `line[1]` (if present) is the 2nd
    ///     stone of a 2-stone forcing turn.
    ///   - `nodes`: board expansions charged (the honesty axis vs the deploy
    ///     search). NET-FREE: the value head is never read inside the proof.
    #[pyo3(signature = (board, depth, node_budget))]
    pub fn prove(
        &self,
        board: &PyBoard,
        depth: u32,
        node_budget: u64,
    ) -> (i32, Vec<(i32, i32)>, u64) {
        let res = self.inner.prove(board.inner_ref(), depth, node_budget);
        (res.result.to_i32(), res.line, res.nodes)
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTacticalSolver>()?;
    Ok(())
}
