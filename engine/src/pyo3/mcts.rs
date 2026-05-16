//! Python-visible PUCT MCTSTree wrapper.
//!
//! Extracted from `engine/src/lib.rs` at §178 Wave 5b Commit 4. Byte-identical
//! move (struct + #[pymethods] block); only the surrounding `use` lines, the
//! file-level doc comment, and the `register()` registration helper are new.

use pyo3::prelude::*;
use numpy::{IntoPyArray, PyArray1};

use crate::board::{self, BOARD_SIZE};
use crate::mcts::MCTSTree;
use crate::pyo3::board::PyBoard;

/// Per-root-child info returned by `get_root_children_info`:
/// `((q, r), pool_idx, prior, visits, q_value)`. Used by the policy viewer
/// to drive Gumbel Sequential Halving from Python.
type RootChildInfo = ((i32, i32), u32, f32, u32, f32);

/// Single-threaded PUCT MCTS tree exposed to Python.
///
/// Usage (Python):
///
/// ```python
/// tree = MCTSTree(c_puct=1.5)
/// tree.new_game(board)
/// for _ in range(n_simulations):
///     boards = tree.select_leaves(1)
///     policies = [[...]]   # list of float lists, length = board_size^2 + 1
///     values   = [0.5]     # list of scalars
///     tree.expand_and_backup(policies, values)
/// policy = tree.get_policy(temperature=1.0, board_size=9)
/// visits = tree.root_visits()
/// ```
#[pyclass(name = "MCTSTree")]
pub struct PyMCTSTree {
    inner: MCTSTree,
    board_size: usize,
}

#[pymethods]
impl PyMCTSTree {
    /// Args:
    ///     c_puct: exploration constant (default 1.5).
    ///     virtual_loss: fixed penalty (default 1.0).
    ///     fpu_reduction: KataGo-style dynamic FPU base (default 0.25).
    ///         FPU for unvisited children = parent_q - fpu_reduction * sqrt(explored_mass).
    ///         Set to 0.0 to disable (classical Q=0 for unvisited).
    ///     quiescence_enabled: override leaf value when forced win/loss is proven (default True).
    ///     quiescence_blend_2: blend amount for the 2-winning-moves case (default 0.3).
    #[new]
    #[pyo3(signature = (c_puct = 1.5, virtual_loss = 1.0, fpu_reduction = 0.25, quiescence_enabled = true, quiescence_blend_2 = 0.3))]
    pub fn new(c_puct: f32, virtual_loss: f32, fpu_reduction: f32, quiescence_enabled: bool, quiescence_blend_2: f32) -> Self {
        let mut inner = MCTSTree::new_full(c_puct, virtual_loss, fpu_reduction);
        inner.quiescence_enabled = quiescence_enabled;
        inner.quiescence_blend_2 = quiescence_blend_2;
        PyMCTSTree {
            inner,
            board_size: board::BOARD_SIZE,
        }
    }

    /// Total quiescence value overrides/blends since last `new_game()`.
    #[getter]
    pub fn get_quiescence_fire_count(&self) -> u64 {
        self.inner.quiescence_fire_count.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Search statistics accumulated since the last `new_game()`.
    ///
    /// Returns `(mean_depth, root_concentration)`:
    /// - `mean_depth`: average leaf depth across all simulations this game/search
    /// - `root_concentration`: max child visits / total root visits ∈ [0.0, 1.0]
    ///
    /// Both 0.0 before any simulations. Call after search completes, not during.
    pub fn last_search_stats(&self) -> (f32, f32) {
        self.inner.last_search_stats()
    }

    /// Reset the tree for a new game starting from `board`.
    ///
    /// This re-uses the pre-allocated pool — no heap allocation.
    pub fn new_game(&mut self, board: &PyBoard) {
        self.board_size = BOARD_SIZE;
        self.inner.new_game(board.inner_ref().clone());
    }

    /// Select up to `n` distinct leaves for neural-network evaluation.
    ///
    /// Returns a list of Board objects (one per unique leaf).
    /// Always call `expand_and_backup` with the same number of results
    /// before the next call to `select_leaves`.
    pub fn select_leaves(&mut self, py: Python<'_>, n: usize) -> PyResult<Vec<Py<PyBoard>>> {
        let boards = py.detach(|| self.inner.select_leaves(n));
        boards
            .into_iter()
            .map(|b| Py::new(py, PyBoard::from_inner(b)))
            .collect()
    }

    /// Expand leaves and backup values from the last `select_leaves` call.
    ///
    /// Args:
    ///     policies: list of policy vectors (one per leaf).
    ///               Each vector has length `board_size * board_size + 1`.
    ///     values:   list of scalar values in [-1, 1] (one per leaf),
    ///               from the current player's perspective at that leaf.
    pub fn expand_and_backup(
        &mut self,
        py: Python<'_>,
        policies: Vec<Vec<f32>>,
        values: Vec<f32>,
    ) -> PyResult<()> {
        py.detach(|| self.inner.expand_and_backup(&policies, &values));
        Ok(())
    }

    /// Return the visit-count policy at the root.
    ///
    /// Args:
    ///     temperature: sampling temperature (0 = argmax).
    ///     board_size:  spatial dimension (default: size from last `new_game`).
    ///
    /// Returns a list of length `board_size * board_size + 1`.
    #[pyo3(signature = (temperature = 1.0, board_size = None))]
    pub fn get_policy<'py>(
        &self,
        py: Python<'py>,
        temperature: f32,
        board_size: Option<usize>,
    ) -> Bound<'py, PyArray1<f32>> {
        let bs = board_size.unwrap_or(self.board_size);
        // §P2: inner API now takes `n_actions` (= policy_stride) instead of
        // `board_size`. PyO3 surface unchanged — Python callers still pass
        // board_size kwarg. Python-side selfplay path (if any non-Rust
        // selfplay still uses MCTSTree) is v6-only today, so bs²+1 is correct.
        // audit: P14 follow-up — read n_actions from self.inner.root_board.encoding
        // §P77 — zero-copy return via `IntoPyArray`.
        let n_actions = bs * bs + 1;
        self.inner.get_policy(temperature, n_actions).into_pyarray(py)
    }

    /// Total visit count at the root (= number of simulations run).
    pub fn root_visits(&self) -> u32 {
        self.inner.root_visits()
    }

    /// Reset the tree to its root state (for benchmarking / reuse).
    pub fn reset(&mut self) {
        self.inner.reset();
    }

    /// Run `n` simulations using uniform priors and value=0 (no neural network).
    /// Used for CPU-only MCTS throughput benchmarking.
    pub fn run_simulations_cpu_only(&mut self, n: usize) {
        self.inner.run_simulations_cpu_only(n);
    }

    /// Mix Dirichlet noise into the root node's priors (self-play only).
    ///
    /// Call after the first expand_and_backup (which expands the root).
    /// On the Python side, generate `noise` with:
    ///     noise = np.random.dirichlet([alpha] * tree.root_n_children()).tolist()
    ///
    /// Args:
    ///     noise:   list of floats, length == root_n_children().
    ///     epsilon: mixing weight (default 0.25 per AlphaZero).
    #[pyo3(signature = (noise, epsilon = 0.25))]
    pub fn apply_dirichlet_to_root(&mut self, noise: Vec<f32>, epsilon: f32) {
        self.inner.apply_dirichlet_to_root(&noise, epsilon);
    }

    /// Number of children at the root (0 if not yet expanded).
    /// Use this to determine the noise vector length before calling
    /// apply_dirichlet_to_root.
    pub fn root_n_children(&self) -> usize {
        self.inner.root_n_children()
    }

    /// Top-N children of root by visit count.
    /// Returns list of ((q, r), visits, prior, q_value) sorted by visits descending.
    ///
    /// §P34 — coord is a `(i32, i32)` axial tuple instead of a pre-formatted
    /// `"(q,r)"` string. Saves per-child `format!(...)` heap String alloc on
    /// viewer/analyze-API paths. Python callers format with f-strings at the
    /// call site.
    pub fn get_top_visits(&self, n: usize) -> Vec<((i32, i32), u32, f32, f32)> {
        self.inner.get_top_visits(n)
    }

    /// Value estimate at root from perspective of player to move.
    pub fn root_value(&self) -> f32 {
        self.inner.root_value()
    }

    // ── Policy viewer accessors ──────────────────────────────────────────────

    /// Get/set forced root child for Gumbel Sequential Halving.
    /// Set to a child pool index to restrict select_leaves to that subtree.
    /// Set to None to restore normal PUCT selection.
    #[getter]
    pub fn forced_root_child(&self) -> Option<u32> {
        self.inner.forced_root_child
    }

    #[setter]
    pub fn set_forced_root_child(&mut self, val: Option<u32>) {
        self.inner.forced_root_child = val;
    }

    /// Returns list of ((q, r), pool_idx, prior, visits, q_value) for each root child.
    /// Used by the policy viewer to drive Gumbel Sequential Halving from Python.
    ///
    /// §P34 — coord is a `(i32, i32)` axial tuple instead of a pre-formatted
    /// `"(q,r)"` string. Same alloc-saving rationale as `get_top_visits`.
    pub fn get_root_children_info(&self) -> Vec<RootChildInfo> {
        let children = self.inner.get_root_children_info();
        let q_sign: f32 = if self.inner.pool[0].moves_remaining == 1 { -1.0 } else { 1.0 };
        children.into_iter().map(|(pool_idx, prior)| {
            let child = &self.inner.pool[pool_idx as usize];
            let visits = child.n_visits;
            let q_value = if visits > 0 { q_sign * child.w_value / visits as f32 } else { 0.0 };
            let val = child.action_idx;
            let aq = (val >> 16) as i32 - 32768;
            let ar = (val & 0xFFFF) as i32 - 32768;
            ((aq, ar), pool_idx, prior, visits, q_value)
        }).collect()
    }

    /// Compute improved policy targets using Gumbel completed Q-values
    /// (Danihelka et al., ICLR 2022). Used by the policy viewer for
    /// Gumbel-mode analysis overlay.
    #[pyo3(signature = (board_size = None, c_visit = 50.0, c_scale = 1.0))]
    pub fn get_improved_policy<'py>(
        &self,
        py: Python<'py>,
        board_size: Option<usize>,
        c_visit: f32,
        c_scale: f32,
    ) -> Bound<'py, PyArray1<f32>> {
        let bs = board_size.unwrap_or(self.board_size);
        // §P2: inner API now takes `n_actions` (= policy_stride) instead of
        // `board_size`. Same pattern as get_policy above.
        // audit: P14 follow-up — read n_actions from self.inner.root_board.encoding
        // §P77 — zero-copy return via `IntoPyArray`.
        let n_actions = bs * bs + 1;
        self.inner
            .get_improved_policy(n_actions, c_visit, c_scale)
            .into_pyarray(py)
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMCTSTree>()?;
    Ok(())
}
