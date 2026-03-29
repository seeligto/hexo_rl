/// native_core — PyO3 extension module.
///
/// Exposes to Python:
///   from native_core import Board, MCTSTree

pub mod board;
pub mod formations;
pub mod mcts;

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

use board::{Board as RustBoard, Player, BOARD_SIZE};

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
    pub fn zobrist_hash(&self) -> u64 {
        self.inner.zobrist_hash
    }

    /// Encode the board as a flat list of floats for the two "current" tensor planes
    /// (shape conceptually [2, 19, 19], returned as a flat list of length 2×361=722):
    ///   plane 0: current player's stones
    ///   plane 1: opponent's stones
    ///
    /// Use numpy.array(board.to_tensor(), dtype=numpy.float32).reshape(2, 19, 19).
    pub fn to_tensor(&self) -> Vec<f32> {
        self.inner.to_planes()
    }

    /// Same as `to_tensor` but makes the sliding-window semantics explicit.
    /// `size` is ignored — always 19×19.
    pub fn view_window(&self, _size: usize) -> Vec<f32> {
        self.inner.to_planes()
    }

    /// Returns a tuple of list of flat planes and list of (q, r) centers for each cluster.
    pub fn get_cluster_views(&self) -> (Vec<Vec<f32>>, Vec<(i32, i32)>) {
        self.inner.get_cluster_views()
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
    /// Create a new MCTS tree.
    ///
    /// Args:
    ///     c_puct: exploration constant (default 1.5).
    #[new]
    #[pyo3(signature = (c_puct = 1.5))]
    pub fn new(c_puct: f32) -> Self {
        PyMCTSTree {
            inner: MCTSTree::new(c_puct),
            board_size: board::BOARD_SIZE,
        }
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
        let boards = self.inner.select_leaves(n);
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
        policies: Vec<Vec<f32>>,
        values: Vec<f32>,
    ) -> PyResult<()> {
        self.inner.expand_and_backup(&policies, &values);
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
}

// ── Module registration ───────────────────────────────────────────────────────

#[pymodule]
fn native_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBoard>()?;
    m.add_class::<PyMCTSTree>()?;
    Ok(())
}
