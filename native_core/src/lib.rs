/// native_core — PyO3 extension module.
///
/// Exposes the Rust Board to Python:
///   from native_core import Board

pub mod board;

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

    /// Board size (cells per axis = 19).
    #[getter]
    pub fn size(&self) -> usize {
        BOARD_SIZE
    }

    /// Human-readable string showing the board (for debugging).
    pub fn __repr__(&self) -> String {
        let mut s = format!(
            "Board(ply={}, player={}, moves_remaining={})\n",
            self.inner.ply,
            match self.inner.current_player { Player::One => 1, Player::Two => -1 },
            self.inner.moves_remaining,
        );
        // Print a simple grid: q increases left-to-right, r increases top-to-bottom
        for r in (-9i32..=9).rev() {
            for q in -9i32..=9 {
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

// ── Module registration ───────────────────────────────────────────────────────

#[pymodule]
fn native_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBoard>()?;
    Ok(())
}
