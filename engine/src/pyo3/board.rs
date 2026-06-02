//! Python-visible Board wrapper.
//!
//! Extracted from `engine/src/lib.rs` at §178 Wave 5b Commit 3. Byte-identical
//! move (struct + #[pymethods] block + inherent impl); only the surrounding
//! `use` lines, the file-level doc comment, and the `register()` registration
//! helper are new.

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::{IntoPyArray, PyArray1, PyArray3, PyArrayMethods};

use crate::board::{self, Board as RustBoard, Player, BOARD_SIZE};

/// Return tuple of `get_cluster_views`: a list of `(2, S, S)` view arrays
/// (current-player + opponent stones) paired with the axial (q, r) centre
/// of each cluster window.
type ClusterViewsOut = (Vec<Py<PyArray3<f32>>>, Vec<(i32, i32)>);

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

impl Default for PyBoard {
    /// Equivalent to `PyBoard::new()` — empty board, v6 encoding default.
    fn default() -> Self {
        Self::new()
    }
}

#[pymethods]
impl PyBoard {
    /// Create a new empty board.
    #[new]
    pub fn new() -> Self {
        PyBoard { inner: RustBoard::new() }
    }

    /// §172 A4.1 — registry-resolved Board ctor. Looks the encoding up by
    /// name in `engine/src/encoding/registry.toml` and binds the resulting
    /// `RegistrySpec` to the new Board. §P3.2 retired the legacy 4-field
    /// `with_encoding(PyEncodingSpec)` static method — registry path is the
    /// single supported entry point for non-default encoding construction.
    /// Adding a new encoding becomes a single TOML edit.
    ///
    /// Raises `ValueError` if `name` is not a registered encoding.
    /// `Board.size`, `Board.to_tensor()`, and the multi-window guard at
    /// `to_tensor` honor the registry record.
    #[staticmethod]
    pub fn with_encoding_name(name: &str) -> PyResult<Self> {
        let spec = crate::encoding::lookup(name).ok_or_else(|| {
            PyValueError::new_err(format!(
                "unknown encoding {name:?}; see engine/src/encoding/registry.toml"
            ))
        })?;
        Ok(PyBoard { inner: RustBoard::with_registry_spec(spec) })
    }

    /// Place a stone at (q, r) for the current player.
    /// Raises ValueError if the move is illegal.
    pub fn apply_move(&mut self, q: i32, r: i32) -> PyResult<()> {
        self.inner.apply_move(q, r).map_err(PyValueError::new_err)
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

    /// Returns the 6 cells forming the winning line, or empty list if no win.
    ///
    /// §S178 F-fix-1: scans all stones when last_move doesn't yield a 6-line,
    /// so a win found by `winner()`'s fallback path (HTT 2-moves-per-turn
    /// shifting last_move off the line) still surfaces the winning cells.
    pub fn find_winning_line(&self) -> Vec<(i32, i32)> {
        self.inner.find_winning_line()
    }

    /// CF-1 terminal value from the side-to-move's perspective, valid at a
    /// `check_win()` leaf: `+1.0` when `moves_remaining == 1` (first-stone win,
    /// winner still to move), `-1.0` when `moves_remaining == 2` (turn-final
    /// win, flipped to the loser). Engine-owned SoT for the CF-1 sign so Python
    /// eval-bot MCTS need not re-derive it (SCATTER-1). Mirrors the inline
    /// derivation in `mcts/backup.rs::expand_and_backup_single`.
    pub fn terminal_value_to_move(&self) -> f32 {
        self.inner.terminal_value_to_move()
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
    /// §P76 — zero-copy return via `IntoPyArray`. Python callers spell:
    ///   board.to_tensor().reshape(18, board.size, board.size)
    /// `Bound<PyArray1<f32>>` is a NumPy view over the Vec the Rust side
    /// just allocated — `into_pyarray(py)` transfers ownership, no copy.
    pub fn to_tensor<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        self.inner.to_planes().into_pyarray(py)
    }

    /// Returns a tuple of (list of NumPy arrays, list of (q, r) centers) for each cluster.
    ///
    /// Each NumPy array has shape `(2, S, S)` where `S = self.cluster_window_size`
    /// (default 19 = v6 wire format; v6w25 callers `set_cluster_window_size(25)`
    /// per §168 Gate 3). Plane 0 = current player's stones, plane 1 = opponent's
    /// stones.  Arrays are created via zero-copy transfer from Rust allocations.
    pub fn get_cluster_views(
        &self,
        py: Python<'_>,
    ) -> PyResult<ClusterViewsOut> {
        let window_size = self.inner.cluster_window_size();
        let (views, centers) = self.inner.get_cluster_views();
        let py_views: PyResult<Vec<_>> = views
            .into_iter()
            .map(|v| {
                // Transfer Vec ownership to NumPy (zero-copy), then reshape.
                PyArray1::from_vec(py, v)
                    .reshape([2_usize, window_size, window_size])
                    .map(pyo3::Bound::unbind)
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
        if size < 7 || size.is_multiple_of(2) {
            return Err(PyValueError::new_err(format!(
                "cluster_window_size must be odd and >= 7; got {size}"
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
        for (&(q, r), &cell) in &self.inner.cells {
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

    pub fn __deepcopy__(&self, _memo: Py<PyAny>) -> PyBoard {
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
    ///
    /// §178 Wave 5b SD3: was `pub fn` in lib.rs; demoted to `pub(crate)`
    /// because the sole caller (PyMCTSTree::select_leaves) lives in
    /// crate::pyo3::mcts (sibling). No external callers verified by
    /// `rg "PyBoard::from_inner" engine/tests/ engine/benches/ engine/src/`.
    pub(crate) fn from_inner(inner: board::Board) -> Self {
        PyBoard { inner }
    }

    /// §178 Wave 5b SD3 — crate-internal accessor for the wrapped Rust Board.
    /// Used by PyMCTSTree::new_game to clone the underlying board across the
    /// PyO3 boundary. PyBoard's `inner` field is private; sibling-module
    /// callers (crate::pyo3::mcts) cannot read it directly, so route through
    /// this accessor.
    pub(crate) fn inner_ref(&self) -> &RustBoard {
        &self.inner
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBoard>()?;
    Ok(())
}
