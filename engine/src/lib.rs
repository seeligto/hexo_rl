/// engine — PyO3 extension module.
///
/// Exposes to Python:
///   from engine import Board, MCTSTree

pub mod board;
#[cfg(feature = "debug_prior_trace")]
pub mod debug_trace;
pub mod encoding;
pub mod game_runner;
pub mod inference_bridge;
pub mod mcts;
pub mod replay_buffer;

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::{
    IntoPyArray, PyArray1, PyArray3, PyArray4, PyArrayMethods,
    PyReadonlyArray4, PyUntypedArrayMethods,
};

use board::{Board as RustBoard, Player, BOARD_SIZE};
use crate::encoding::EncodingSpec as RustEncodingSpec;
use crate::encoding::RegistrySpec as RustRegistrySpec;
use game_runner::SelfPlayRunner;
use inference_bridge::InferenceBatcher;
use replay_buffer::ReplayBuffer;
use replay_buffer::sample::apply_symmetry_state;
use replay_buffer::sym_tables::{SymTables, N_SYMS};

// ── Python-visible EncodingSpec ───────────────────────────────────────────────

/// Python-visible EncodingSpec. Pass to `Board.with_encoding(spec)` to
/// construct a Board with non-default cluster window / threshold / radius.
//
// TODO(post-§172): pyo3 0.28 deprecated automatic `FromPyObject` derivation
// for `#[pyclass]` types implementing `Clone`. Build emits one warning here
// and at PyRegistrySpec below; both will become hard errors on pyo3 1.0.
// Migration is mechanical: add `#[pyclass(from_py_object)]` to opt-in OR
// `#[pyclass(skip_from_py_object)]` to skip. No timeline — gated by pyo3
// upgrade prioritization. See engine/src/lib.rs:115 for the second site.
#[pyclass(name = "EncodingSpec")]
#[derive(Clone)]
pub struct PyEncodingSpec {
    inner: RustEncodingSpec,
}

#[pymethods]
impl PyEncodingSpec {
    #[new]
    #[pyo3(signature = (*, cluster_window_size, cluster_threshold, legal_move_radius, board_size))]
    pub fn new(
        cluster_window_size: usize,
        cluster_threshold: i32,
        legal_move_radius: i32,
        board_size: usize,
    ) -> PyResult<Self> {
        let inner = RustEncodingSpec {
            cluster_window_size,
            cluster_threshold,
            legal_move_radius,
            board_size,
        };
        inner.validate().map_err(PyValueError::new_err)?;
        Ok(PyEncodingSpec { inner })
    }

    #[getter] pub fn cluster_window_size(&self) -> usize { self.inner.cluster_window_size }
    #[getter] pub fn cluster_threshold(&self) -> i32 { self.inner.cluster_threshold }
    #[getter] pub fn legal_move_radius(&self) -> i32 { self.inner.legal_move_radius }
    #[getter] pub fn board_size(&self) -> usize { self.inner.board_size }

    pub fn __repr__(&self) -> String {
        format!(
            "EncodingSpec(cluster_window_size={}, cluster_threshold={}, legal_move_radius={}, board_size={})",
            self.inner.cluster_window_size, self.inner.cluster_threshold,
            self.inner.legal_move_radius, self.inner.board_size,
        )
    }

    pub fn __eq__(&self, other: &PyEncodingSpec) -> bool { self.inner == other.inner }
    pub fn __hash__(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.inner.hash(&mut h);
        h.finish()
    }

    /// §172 A10 T8b — registry-backed lookup. Returns a `PyRegistrySpec`
    /// (full-schema record incl. policy_logit_count + n_planes) so v8 callers
    /// can derive feature_len / policy_len at the PyO3 boundary without
    /// re-implementing the Python-side schema. The legacy 4-field
    /// `EncodingSpec(...)` constructor cannot represent v8 (no cluster
    /// window), so `from_registry("v8")` is the v8 entry point.
    #[classmethod]
    pub fn from_registry(_cls: &Bound<'_, pyo3::types::PyType>, name: &str) -> PyResult<PyRegistrySpec> {
        match crate::encoding::lookup(name) {
            Some(spec) => Ok(PyRegistrySpec { inner: spec }),
            None => {
                let mut known: Vec<&str> =
                    crate::encoding::all_specs().map(|s| s.name).collect();
                known.sort();
                Err(PyValueError::new_err(format!(
                    "EncodingSpec.from_registry: unknown encoding {:?}; registered: {:?}",
                    name, known
                )))
            }
        }
    }
}

// ── Python-visible RegistrySpec (read-only registry record) ───────────────────

/// Python-visible RegistrySpec — wraps `&'static crate::encoding::RegistrySpec`.
/// Returned by `EncodingSpec.from_registry(name)`. Carries derived shape
/// accessors (`state_stride()`, `policy_stride()`) so PyO3 callers
/// constructing `SelfPlayRunner` / `InferenceBatcher` can derive
/// `feature_len` / `policy_len` from the canonical registry instead of
/// duplicating the per-encoding shape table.
///
/// Read-only — clone is `Copy` (just the &'static pointer).
#[pyclass(name = "RegistrySpec")]
#[derive(Clone, Copy)]
pub struct PyRegistrySpec {
    inner: &'static RustRegistrySpec,
}

#[pymethods]
impl PyRegistrySpec {
    #[getter] pub fn name(&self) -> &'static str { self.inner.name }
    #[getter] pub fn board_size(&self) -> usize { self.inner.board_size }
    #[getter] pub fn trunk_size(&self) -> usize { self.inner.trunk_size }
    #[getter] pub fn n_planes(&self) -> usize { self.inner.n_planes }
    #[getter] pub fn policy_logit_count(&self) -> usize { self.inner.policy_logit_count }
    #[getter] pub fn has_pass_slot(&self) -> bool { self.inner.has_pass_slot }
    #[getter] pub fn is_multi_window(&self) -> bool { self.inner.is_multi_window }
    /// §173 A3 — physical source-plane indices retained by wire format.
    #[getter] pub fn kept_plane_indices(&self) -> Vec<usize> {
        self.inner.kept_plane_indices.to_vec()
    }
    /// §173 A3 — source tensor plane count before kept_plane_indices slice.
    #[getter] pub fn n_source_planes(&self) -> usize { self.inner.n_source_planes }

    /// Cells per trunk input tensor = trunk_size². §173 A3 semantic: trunk_size, not board_size.
    pub fn n_cells(&self) -> usize { self.inner.n_cells() }
    /// State plane stride = n_planes × n_cells.
    pub fn state_stride(&self) -> usize { self.inner.state_stride() }
    /// Chain plane stride = N_CHAIN_PLANES × n_cells.
    pub fn chain_stride(&self) -> usize { self.inner.chain_stride() }
    /// Aux plane stride = n_cells (single aux plane).
    pub fn aux_stride(&self) -> usize { self.inner.aux_stride() }
    /// Policy logit count = `policy_logit_count` (mirror of the field).
    pub fn policy_stride(&self) -> usize { self.inner.policy_stride() }

    pub fn __repr__(&self) -> String {
        format!(
            "RegistrySpec(name={:?}, board_size={}, n_planes={}, policy_logit_count={}, is_multi_window={})",
            self.inner.name, self.inner.board_size, self.inner.n_planes,
            self.inner.policy_logit_count, self.inner.is_multi_window,
        )
    }
}

impl PyRegistrySpec {
    /// Crate-internal accessor — used by `SelfPlayRunner::new` /
    /// `InferenceBatcher::new` to read the static pointer.
    pub(crate) fn inner(&self) -> &'static RustRegistrySpec {
        self.inner
    }

    /// §173 A5a — test helper: construct from a `&'static RegistrySpec`
    /// reference (e.g. returned by `lookup_or_panic`). Allows Rust integration
    /// tests to pass a `PyRegistrySpec` to `SelfPlayRunner::new` without
    /// going through the Python boundary.
    pub fn from_static(spec: &'static RustRegistrySpec) -> Self {
        PyRegistrySpec { inner: spec }
    }
}

impl PyEncodingSpec {
    /// Crate-internal accessor: copy out the underlying `RustEncodingSpec`.
    /// Used by `SelfPlayRunner::new` (game_runner/mod.rs) to thread the
    /// spec through to the Rust-owned worker thread Boards.
    pub(crate) fn to_inner(&self) -> RustEncodingSpec {
        self.inner
    }

    /// Crate-internal constructor: wrap a `RustEncodingSpec` in the
    /// PyO3-visible PyEncodingSpec. Used by the `SelfPlayRunner.encoding`
    /// `#[getter]` and by Rust-side test sites that previously poked the
    /// (now-private) `inner` field via struct-literal construction.
    pub(crate) fn from_inner(inner: RustEncodingSpec) -> Self {
        PyEncodingSpec { inner }
    }
}

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

    /// §171 P2 reopen — construct a Board with a non-default EncodingSpec.
    /// Equivalent to calling Board() then set_cluster_window_size /
    /// set_cluster_threshold / set_legal_move_radius, but bundles them into a
    /// single validated call. Used by v6w25 self-play workers.
    #[staticmethod]
    pub fn with_encoding(encoding: &PyEncodingSpec) -> Self {
        PyBoard { inner: RustBoard::with_encoding(&encoding.inner) }
    }

    /// §172 A4.1 — registry-resolved Board ctor. Looks the encoding up by
    /// name in `engine/src/encoding/registry.toml` and binds the resulting
    /// `RegistrySpec` to the new Board. Preferred over `with_encoding`
    /// (which takes the legacy 4-field struct) for new consumers — adding
    /// a new encoding becomes a single TOML edit.
    ///
    /// Raises `ValueError` if `name` is not a registered encoding.
    /// `Board.size`, `Board.to_tensor()`, and the multi-window guard at
    /// `to_tensor` honor the registry record.
    #[staticmethod]
    pub fn with_encoding_name(name: &str) -> PyResult<Self> {
        let spec = crate::encoding::lookup(name).ok_or_else(|| {
            PyValueError::new_err(format!(
                "unknown encoding {:?}; see engine/src/encoding/registry.toml",
                name
            ))
        })?;
        Ok(PyBoard { inner: RustBoard::with_registry_spec(spec) })
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

    /// Override the per-Board legal-move radius cap.
    ///
    /// Default is `DEFAULT_LEGAL_MOVE_RADIUS = 5` (v6 path; §145 Option α').
    /// v8 callers (encoding migration §166 Path β) and per-game radius
    /// jitter (§152 Q2) override via this setter at construction time.
    /// HTTT rule baseline is r=8.
    ///
    /// §173 A6 — raises `ValueError` when the board was constructed via
    /// `Board.with_encoding_name` (encoding bound). Callers should use
    /// the registry entry instead of overriding post-construction.
    pub fn set_legal_move_radius(&mut self, radius: i32) -> PyResult<()> {
        if self.inner.encoding.is_some() {
            return Err(PyValueError::new_err(
                "set_legal_move_radius after with_encoding_name is not supported; \
                 use registry (Board.with_encoding_name) instead of overriding post-construction"
            ));
        }
        self.inner.set_legal_move_radius(radius);
        Ok(())
    }

    /// §174 — curriculum radius override.  Works with encoding.
    ///
    /// Use this for training-time radius scheduling, NOT for general board
    /// setup.  Bypasses the `set_legal_move_radius` guard intentionally.
    pub fn override_legal_move_radius(&mut self, radius: i32) -> PyResult<()> {
        self.inner.override_legal_move_radius(radius);
        Ok(())
    }

    /// Read the current per-Board legal-move radius cap.
    pub fn legal_move_radius(&self) -> i32 {
        self.inner.legal_move_radius()
    }

    /// Incremental Zobrist hash of the current position.
    pub fn zobrist_hash(&self) -> u128 {
        self.inner.zobrist_hash
    }

    /// Encode the board as a flat list of floats for the 18 tensor planes
    /// (shape conceptually [18, board_size, board_size] where board_size
    /// comes from the encoding bound at construction — default 19 = v6,
    /// flat length 18×361=6498; v8 = 25, flat length 18×625=11250).
    ///   plane 0: current player's stones
    ///   plane 8: opponent's stones
    ///   plane 16: moves_remaining == 2 ? 1.0 : 0.0
    ///   plane 17: ply % 2
    ///   (chain-length planes moved to replay-buffer aux sub-buffer post-§97)
    ///
    /// §172 A4.1: panics for multi-window encodings (v6w25 etc.); use
    /// `get_cluster_views()` for those. Multi-window selfplay deferred to α
    /// — see `docs/designs/encoding_alpha_multiwindow_selfplay.md`.
    ///
    /// Use numpy.array(board.to_tensor(), dtype=numpy.float32)
    ///   .reshape(18, board.size, board.size).
    pub fn to_tensor(&self) -> Vec<f32> {
        self.inner.to_planes()
    }

    /// Returns a tuple of (list of NumPy arrays, list of (q, r) centers) for each cluster.
    ///
    /// Each NumPy array has shape `(2, S, S)` where `S = self.cluster_window_size`
    /// (default 19 = v6 wire format; v6w25 callers `set_cluster_window_size(25)`
    /// per §168 Gate 3). Plane 0 = current player's stones, plane 1 = opponent's
    /// stones.  Arrays are created via zero-copy transfer from Rust allocations.
    pub fn get_cluster_views<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Vec<Py<PyArray3<f32>>>, Vec<(i32, i32)>)> {
        let window_size = self.inner.cluster_window_size();
        let (views, centers) = self.inner.get_cluster_views();
        let py_views: PyResult<Vec<_>> = views
            .into_iter()
            .map(|v| {
                // Transfer Vec ownership to NumPy (zero-copy), then reshape.
                PyArray1::from_vec(py, v)
                    .reshape([2_usize, window_size, window_size])
                    .map(|arr| arr.unbind())
            })
            .collect();
        Ok((py_views?, centers))
    }

    /// §168 Gate 3 — set the cluster connectivity threshold (default 5).
    /// Used by v6w25 corpus generation to widen cluster reach to 8 in
    /// proportion to the larger 25×25 cluster window. Affects only
    /// `get_clusters()` / `get_cluster_views()`; legal-move expansion is
    /// independent (controlled by `set_legal_move_radius`).
    ///
    /// §173 A6 — raises `ValueError` when the board was constructed via
    /// `Board.with_encoding_name` (encoding bound). Use registry entry instead.
    pub fn set_cluster_threshold(&mut self, threshold: i32) -> PyResult<()> {
        if self.inner.encoding.is_some() {
            return Err(PyValueError::new_err(
                "set_cluster_threshold after with_encoding_name is not supported; \
                 use registry (Board.with_encoding_name) instead of overriding post-construction"
            ));
        }
        self.inner.set_cluster_threshold(threshold);
        Ok(())
    }

    /// Current cluster threshold (default 5).
    pub fn cluster_threshold(&self) -> i32 {
        self.inner.cluster_threshold()
    }

    /// §168 Gate 3 — set the cluster window side length (default 19).
    /// Used by v6w25 corpus generation to produce 25×25 cluster windows.
    /// Caller must use an odd value >= 7. Returns ValueError on bad input.
    ///
    /// §173 A6 — raises `ValueError` when the board was constructed via
    /// `Board.with_encoding_name` (encoding bound). Use registry entry instead.
    pub fn set_cluster_window_size(&mut self, size: usize) -> PyResult<()> {
        if self.inner.encoding.is_some() {
            return Err(PyValueError::new_err(
                "set_cluster_window_size after with_encoding_name is not supported; \
                 use registry (Board.with_encoding_name) instead of overriding post-construction"
            ));
        }
        if size < 7 || size % 2 == 0 {
            return Err(PyValueError::new_err(format!(
                "cluster_window_size must be odd and >= 7; got {}", size
            )));
        }
        self.inner.set_cluster_window_size(size);
        Ok(())
    }

    /// Current cluster window side length (default 19).
    pub fn cluster_window_size(&self) -> usize {
        self.inner.cluster_window_size()
    }

    /// Window-relative flat index for axial (q, r).
    /// Used by selfplay workers to convert legal-move coords to policy indices.
    pub fn to_flat(&self, q: i32, r: i32) -> usize {
        self.inner.window_flat_idx(q, r)
    }

    /// Board size (cells per axis). Default 19 (v6 wire format); honors
    /// the encoding bound at construction via `with_encoding_name` (e.g.
    /// 25 for v8). §172 A4.1.
    #[getter]
    pub fn size(&self) -> usize {
        self.inner.encoding.map_or(BOARD_SIZE, |s| s.board_size)
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

    /// Return a deep clone of this board. Used by Python-side MCTS for v8
    /// (Rust MCTSTree is v6-only — see hexo_rl/eval/v8_mcts_bot.py).
    pub fn clone(&self) -> PyBoard {
        PyBoard { inner: self.inner.clone() }
    }

    /// Python copy.copy() / copy.deepcopy() support.
    pub fn __copy__(&self) -> PyBoard {
        PyBoard { inner: self.inner.clone() }
    }

    pub fn __deepcopy__(&self, _memo: pyo3::Py<pyo3::PyAny>) -> PyBoard {
        PyBoard { inner: self.inner.clone() }
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
        let q_sign: f32 = if self.inner.pool[0].moves_remaining == 1 { -1.0 } else { 1.0 };
        children.into_iter().map(|(pool_idx, prior)| {
            let child = &self.inner.pool[pool_idx as usize];
            let visits = child.n_visits;
            let q_value = if visits > 0 { q_sign * child.w_value / visits as f32 } else { 0.0 };
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

// ── Symmetry + chain-plane bindings (Q13 pretrain parity) ────────────────────
//
// These expose the exact Rust kernels used by the ReplayBuffer sampling path
// and by `Board.to_tensor()` so Python callers (pretrain collate, parity
// tests) can never diverge. Thread-local SymTables avoids per-call allocation.

thread_local! {
    static SYM_TABLES_TLS: SymTables = SymTables::new();
}

/// Batched hex-dihedral symmetry scatter.
///
/// Plane-count-generic: any positive `C` works (8 for HEXB v6 buffer planes,
/// 18 for legacy inference / corpus tensors). State planes do not permute
/// under hex dihedral symmetry — only cell coordinates do — so a single
/// scatter table applies to any plane count.
///
/// Args:
///     states:      (N, C, 19, 19) float32 numpy array.
///     sym_indices: (N,) integer sym_idx per state, values in [0, 12).
///
/// Returns a newly-allocated (N, C, 19, 19) float32 numpy array.
#[pyfunction]
fn apply_symmetries_batch<'py>(
    py: Python<'py>,
    states: PyReadonlyArray4<'py, f32>,
    sym_indices: Vec<usize>,
) -> PyResult<Bound<'py, PyArray4<f32>>> {
    let shape = states.shape();
    if shape.len() != 4 || shape[2] != BOARD_SIZE || shape[3] != BOARD_SIZE {
        return Err(PyValueError::new_err(format!(
            "expected states shape (N, C, {}, {}); got {:?}",
            BOARD_SIZE, BOARD_SIZE, shape
        )));
    }
    let n = shape[0];
    let n_planes = shape[1];
    if sym_indices.len() != n {
        return Err(PyValueError::new_err(format!(
            "sym_indices length {} != batch size {}",
            sym_indices.len(), n
        )));
    }
    for (i, &s) in sym_indices.iter().enumerate() {
        if s >= N_SYMS {
            return Err(PyValueError::new_err(format!(
                "sym_indices[{}] = {} out of range (expected 0..{})",
                i, s, N_SYMS
            )));
        }
    }
    let stride = n_planes * BOARD_SIZE * BOARD_SIZE;
    let src = states.as_slice()?;
    let mut dst = vec![0.0f32; n * stride];
    SYM_TABLES_TLS.with(|tables| {
        for b in 0..n {
            let src_b = &src[b * stride..(b + 1) * stride];
            let dst_b = &mut dst[b * stride..(b + 1) * stride];
            apply_symmetry_state::<f32>(src_b, dst_b, sym_indices[b], tables);
        }
    });
    dst.into_pyarray(py).reshape([n, n_planes, BOARD_SIZE, BOARD_SIZE])
}

/// Read the process-wide MCTS pool-overflow counter without resetting.
///
/// Pool overflow events fabricate a terminal value at the leaf and let
/// it propagate through `backup()`, biasing visit counts and
/// policy/value training targets. The counter is global (all trees
/// across all worker threads share it) — bench drops contaminated
/// runs by reading deltas across measurement windows; production
/// training loops should sample it periodically and alarm if it
/// grows.
#[pyfunction]
fn mcts_pool_overflow_count() -> u64 {
    mcts::pool_overflow_count()
}

/// Atomically read-and-reset the pool-overflow counter. Returns the
/// previous value. Used by the bench harness to bracket per-run
/// measurement windows and detect contamination.
#[pyfunction]
fn take_mcts_pool_overflow_count() -> u64 {
    mcts::take_pool_overflow_count()
}

// ── Module registration ───────────────────────────────────────────────────────

#[pymodule]
fn engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBoard>()?;
    m.add_class::<PyEncodingSpec>()?;
    m.add_class::<PyRegistrySpec>()?;
    m.add_class::<PyMCTSTree>()?;
    m.add_class::<InferenceBatcher>()?;
    m.add_class::<SelfPlayRunner>()?;
    m.add_class::<ReplayBuffer>()?;
    m.add_function(wrap_pyfunction!(apply_symmetries_batch, m)?)?;
    m.add_function(wrap_pyfunction!(mcts_pool_overflow_count, m)?)?;
    m.add_function(wrap_pyfunction!(take_mcts_pool_overflow_count, m)?)?;
    Ok(())
}
