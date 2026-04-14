/// engine — PyO3 extension module.
///
/// Exposes to Python:
///   from engine import Board, MCTSTree

pub mod board;
#[cfg(feature = "debug_prior_trace")]
pub mod debug_trace;
pub mod formations;
pub mod game_runner;
pub mod inference_bridge;
pub mod mcts;
pub mod replay_buffer;

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::{PyArray1, PyArray3, PyArrayMethods};

use board::{Board as RustBoard, Player, BOARD_SIZE};
use game_runner::SelfPlayRunner;
use inference_bridge::InferenceBatcher;
use replay_buffer::ReplayBuffer;

// ── Python-visible Board wrapper ──────────────────────────────────────────────

/// A Hex Tac Toe board.
///
/// Coordinate system: axial (q, r) with -9 ≤ q, r ≤ 9 for a 19×19 grid.
///
/// Turn structure:
///   - Player 1 opens with exactly 1 move (ply 0).
///   - After that, each player places 2 stones per turn.
#[pyclass(name = "Board")]
pub struct PyBoard {
    inner: RustBoard,
}

#[pymethods]
impl PyBoard {
    /// Create a new empty board.
    #[new]
    pub fn new() -> Self {
        PyBoard { inner: RustBoard::new() }
    }

    /// Place a stone at (q, r) for the current player.
    /// Raises ValueError if the move is illegal.
    pub fn apply_move(&mut self, q: i32, r: i32) -> PyResult<()> {
        self.inner.apply_move(q, r).map_err(|e| PyValueError::new_err(e))
    }

    /// Returns True if either player has 6 in a row.
    pub fn check_win(&self) -> bool {
        self.inner.check_win()
    }

    /// Returns the winning player (1 or -1) or None.
    pub fn winner(&self) -> Option<i8> {
        self.inner.winner().map(|p| match p {
            Player::One => 1,
            Player::Two => -1,
        })
    }

    /// List of all legal moves as list of (q, r) tuples.
    pub fn legal_moves(&self) -> Vec<(i32, i32)> {
        self.inner.legal_moves()
    }

    /// Number of legal moves (number of empty cells).
    pub fn legal_move_count(&self) -> usize {
        self.inner.legal_move_count()
    }

    /// Returns the cell value at (q, r): 0=empty, 1=P1, -1=P2.
    pub fn get(&self, q: i32, r: i32) -> i8 {
        match self.inner.get(q, r) {
            board::Cell::Empty => 0,
            board::Cell::P1    => 1,
            board::Cell::P2    => -1,
        }
    }

    /// Current player: 1 for player 1, -1 for player 2.
    #[getter]
    pub fn current_player(&self) -> i8 {
        match self.inner.current_player {
            Player::One => 1,
            Player::Two => -1,
        }
    }

    /// How many moves the current player still has to place this turn.
    #[getter]
    pub fn moves_remaining(&self) -> u8 {
        self.inner.moves_remaining
    }

    /// Total half-moves played (stones placed).
    #[getter]
    pub fn ply(&self) -> u32 {
        self.inner.ply
    }

    /// Incremental Zobrist hash of the current position.
    pub fn zobrist_hash(&self) -> u128 {
        self.inner.zobrist_hash
    }

    /// Encode the board as a flat list of floats for the 24 tensor planes
    /// (shape conceptually [24, 19, 19], returned as a flat list of length 24×361=8664):
    ///   plane 0: current player's stones
    ///   plane 8: opponent's stones
    ///   plane 16: moves_remaining == 2 ? 1.0 : 0.0
    ///   plane 17: ply % 2
    ///   planes 18..23: Q13 chain-length planes, 3 hex axes × 2 players, /6.0-normalized.
    ///
    /// Use numpy.array(board.to_tensor(), dtype=numpy.float32).reshape(24, 19, 19).
    pub fn to_tensor(&self) -> Vec<f32> {
        self.inner.to_planes()
    }

    /// Same as `to_tensor` but makes the sliding-window semantics explicit.
    /// `size` is ignored — always 19×19.
    pub fn view_window(&self, _size: usize) -> Vec<f32> {
        self.inner.to_planes()
    }

    /// Returns a tuple of (list of NumPy arrays, list of (q, r) centers) for each cluster.
    ///
    /// Each NumPy array has shape (2, 19, 19), dtype float32:
    ///   plane 0 = current player's stones, plane 1 = opponent's stones.
    ///
    /// Arrays are created via zero-copy transfer of ownership from Rust allocations.
    pub fn get_cluster_views<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Vec<Py<PyArray3<f32>>>, Vec<(i32, i32)>)> {
        let (views, centers) = self.inner.get_cluster_views();
        let py_views: PyResult<Vec<_>> = views
            .into_iter()
            .map(|v| {
                // Transfer Vec ownership to NumPy (zero-copy), then reshape to (2, 19, 19).
                PyArray1::from_vec(py, v)
                    .reshape([2_usize, board::BOARD_SIZE, board::BOARD_SIZE])
                    .map(|arr| arr.unbind())
            })
            .collect();
        Ok((py_views?, centers))
    }

    /// Window-relative flat index for axial (q, r).
    /// Used by selfplay workers to convert legal-move coords to policy indices.
    pub fn to_flat(&self, q: i32, r: i32) -> usize {
        self.inner.window_flat_idx(q, r)
    }

    /// (cq, cr) centre of the current 19×19 view window.
    pub fn window_center(&self) -> (i32, i32) {
        self.inner.window_center()
    }

    /// Whether (q, r) is inside the current 19×19 view window.
    pub fn in_window(&self, q: i32, r: i32) -> bool {
        self.inner.in_window(q, r)
    }

    /// Board size (cells per axis = 19).
    #[getter]
    pub fn size(&self) -> usize {
        BOARD_SIZE
    }

    /// Human-readable string showing the 19×19 view window (for debugging).
    
    /// Returns a list of all stones on the board as (q, r, player).
    /// Count how many empty cells, if occupied by `player`, would complete a 6-in-a-row.
    ///
    /// Used for the MCTS quiescence check: ≥3 winning moves is a forced win because
    /// the opponent can block at most 2 cells per turn.
    ///
    /// `player`: 1 for player 1, -1 for player 2.  Returns 0 for an empty board.
    pub fn count_winning_moves(&self, player: i8) -> u32 {
        let rust_player = match player {
            1  => board::Player::One,
            -1 => board::Player::Two,
            _  => return 0,
        };
        self.inner.count_winning_moves(rust_player)
    }

    /// Returns threat cells as list of (q, r, level, player) tuples.
    /// Threats are EMPTY cells within threatening windows. Viewer only.
    pub fn get_threats(&self) -> Vec<(i32, i32, u8, u8)> {
        let mut stones = std::collections::HashMap::new();
        for (&(q, r), &cell) in self.inner.cells.iter() {
            let player = match cell {
                board::Cell::P1 => 0u8,
                board::Cell::P2 => 1u8,
                board::Cell::Empty => continue,
            };
            stones.insert((q, r), player);
        }
        board::threats::get_threats(&stones)
            .into_iter()
            .map(|t| (t.q, t.r, t.level, t.player))
            .collect()
    }

    /// Returns a list of all stones on the board as (q, r, player).
    pub fn get_stones(&self) -> Vec<(i32, i32, i8)> {
        self.inner.cells.iter().map(|(&(q, r), &cell)| {
            let p = match cell {
                board::Cell::Empty => 0,
                board::Cell::P1 => 1,
                board::Cell::P2 => -1,
            };
            (q, r, p)
        }).collect()
    }

    pub fn __repr__(&self) -> String {
        let mut s = format!(
            "Board(ply={}, player={}, moves_remaining={})\n",
            self.inner.ply,
            match self.inner.current_player { Player::One => 1, Player::Two => -1 },
            self.inner.moves_remaining,
        );
        let (cq, cr) = self.inner.window_center();
        // wr=18 is top row visually; wq=0 is left column
        for wr in (0..board::BOARD_SIZE).rev() {
            for wq in 0..board::BOARD_SIZE {
                let q = wq as i32 - board::HALF + cq;
                let r = wr as i32 - board::HALF + cr;
                let c = match self.inner.get(q, r) {
                    board::Cell::Empty => '.',
                    board::Cell::P1    => 'X',
                    board::Cell::P2    => 'O',
                };
                s.push(c);
                s.push(' ');
            }
            s.push('\n');
        }
        s
    }
}

impl PyBoard {
    /// Construct a PyBoard directly from a Rust Board (used by PyMCTSTree).
    pub fn from_inner(inner: board::Board) -> Self {
        PyBoard { inner }
    }
}

// ── PyMCTSTree ────────────────────────────────────────────────────────────────

use mcts::MCTSTree;

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
    ///     vl_adaptive: enable scaled virtual loss (default False).
    ///     fpu_reduction: KataGo-style dynamic FPU base (default 0.25).
    ///         FPU for unvisited children = parent_q - fpu_reduction * sqrt(explored_mass).
    ///         Set to 0.0 to disable (classical Q=0 for unvisited).
    ///     quiescence_enabled: override leaf value when forced win/loss is proven (default True).
    ///     quiescence_blend_2: blend amount for the 2-winning-moves case (default 0.3).
    #[new]
    #[pyo3(signature = (c_puct = 1.5, virtual_loss = 1.0, vl_adaptive = false, fpu_reduction = 0.25, quiescence_enabled = true, quiescence_blend_2 = 0.3))]
    pub fn new(c_puct: f32, virtual_loss: f32, vl_adaptive: bool, fpu_reduction: f32, quiescence_enabled: bool, quiescence_blend_2: f32) -> Self {
        let mut inner = MCTSTree::new_full(c_puct, virtual_loss, fpu_reduction);
        inner.vl_adaptive = vl_adaptive;
        inner.quiescence_enabled = quiescence_enabled;
        inner.quiescence_blend_2 = quiescence_blend_2;
        PyMCTSTree {
            inner,
            board_size: board::BOARD_SIZE,
        }
    }

    #[getter]
    pub fn fpu_reduction(&self) -> f32 {
        self.inner.fpu_reduction
    }

    #[setter]
    pub fn set_fpu_reduction(&mut self, val: f32) {
        self.inner.fpu_reduction = val;
    }

    #[getter]
    pub fn quiescence_enabled(&self) -> bool {
        self.inner.quiescence_enabled
    }

    #[setter]
    pub fn set_quiescence_enabled(&mut self, val: bool) {
        self.inner.quiescence_enabled = val;
    }

    #[getter]
    pub fn quiescence_blend_2(&self) -> f32 {
        self.inner.quiescence_blend_2
    }

    #[setter]
    pub fn set_quiescence_blend_2(&mut self, val: f32) {
        self.inner.quiescence_blend_2 = val;
    }

    #[getter]
    pub fn vl_adaptive(&self) -> bool {
        self.inner.vl_adaptive
    }

    #[setter]
    pub fn set_vl_adaptive(&mut self, val: bool) {
        self.inner.vl_adaptive = val;
    }

    #[getter]
    pub fn virtual_loss(&self) -> f32 {
        self.inner.virtual_loss
    }

    #[setter]
    pub fn set_virtual_loss(&mut self, val: f32) {
        self.inner.virtual_loss = val;
    }

    #[getter]
    pub fn selection_overlap_count(&self) -> u32 {
        self.inner.selection_overlap_count
    }

    #[getter]
    pub fn max_depth_observed(&self) -> u32 {
        self.inner.max_depth_observed
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
        self.inner.new_game(board.inner.clone());
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
    pub fn get_policy(
        &self,
        temperature: f32,
        board_size: Option<usize>,
    ) -> Vec<f32> {
        let bs = board_size.unwrap_or(self.board_size);
        self.inner.get_policy(temperature, bs)
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
    /// Returns list of (coord_str, visits, prior, q_value) sorted by visits descending.
    pub fn get_top_visits(&self, n: usize) -> Vec<(String, u32, f32, f32)> {
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

    /// Returns list of (coord_str, pool_idx, prior, visits, q_value) for each root child.
    /// Used by the policy viewer to drive Gumbel Sequential Halving from Python.
    pub fn get_root_children_info(&self) -> Vec<(String, u32, f32, u32, f32)> {
        let children = self.inner.get_root_children_info();
        children.into_iter().map(|(pool_idx, prior)| {
            let child = &self.inner.pool[pool_idx as usize];
            let visits = child.n_visits;
            let q_value = if visits > 0 { child.w_value / visits as f32 } else { 0.0 };
            let val = child.action_idx;
            let aq = (val >> 16) as i32 - 32768;
            let ar = (val & 0xFFFF) as i32 - 32768;
            (format!("({},{})", aq, ar), pool_idx, prior, visits, q_value)
        }).collect()
    }

    /// Compute improved policy targets using Gumbel completed Q-values
    /// (Danihelka et al., ICLR 2022). Used by the policy viewer for
    /// Gumbel-mode analysis overlay.
    #[pyo3(signature = (board_size = None, c_visit = 50.0, c_scale = 1.0))]
    pub fn get_improved_policy(
        &self,
        board_size: Option<usize>,
        c_visit: f32,
        c_scale: f32,
    ) -> Vec<f32> {
        let bs = board_size.unwrap_or(self.board_size);
        self.inner.get_improved_policy(bs, c_visit, c_scale)
    }
}

// ── Module registration ───────────────────────────────────────────────────────

#[pymodule]
fn engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBoard>()?;
    m.add_class::<PyMCTSTree>()?;
    m.add_class::<InferenceBatcher>()?;
    m.add_class::<SelfPlayRunner>()?;
    m.add_class::<ReplayBuffer>()?;
    Ok(())
}
