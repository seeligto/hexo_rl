/// Sparse axial hex board with sliding 19×19 view window.
///
/// Coordinate system: axial (q, r).
///   E:  (+1,  0)   W:  (-1,  0)
///   NE: ( 0, +1)   SW: ( 0, -1)
///   NW: (-1, +1)   SE: (+1, -1)
///
/// Storage: HashMap<(q,r), Cell> — unbounded.
///
/// View window: fixed 19×19 tensor centred on the bounding-box centroid of all
/// placed stones.  On an empty board the window is centred at (0,0).
/// The window slides as play drifts; it never clips stones.
///
/// Legal moves: empty cells within bounding_box + 2 margin, clipped to the
/// current 19×19 window.  On an empty board all 361 window cells are legal.
///
/// Win condition: 6 stones of the same player in a row along one of the three
/// hex axes (E/W, NE/SW, NW/SE).
///
/// Turn structure:
///   ply 0 (first move ever): player 1 places exactly 1 stone.
///   ply 1+: each player places exactly 2 stones before the turn passes.

pub mod bitboard;
pub mod zobrist;

use std::collections::HashMap;
use zobrist::ZobristTable;

/// Board size (cells per axis of the view window).
pub const BOARD_SIZE: usize = 19;
/// Half-width: window covers [-HALF, HALF] relative to its centre.
pub const HALF: i32 = (BOARD_SIZE as i32 - 1) / 2; // 9
/// Total cells in the 19×19 view window.
pub const TOTAL_CELLS: usize = BOARD_SIZE * BOARD_SIZE; // 361
/// Stones in a row required to win.
const WIN_LENGTH: usize = 6;

/// The three hex axis directions (positive direction only; win scan uses ±).
pub const HEX_AXES: [(i32, i32); 3] = [
    (1, 0),  // E / W
    (0, 1),  // NE / SW
    (1, -1), // SE / NW
];

// ── Player ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i8)]
pub enum Player {
    One = 1,
    Two = -1,
}

impl Player {
    pub fn other(self) -> Self {
        match self {
            Player::One => Player::Two,
            Player::Two => Player::One,
        }
    }
}

// ── Cell ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(i8)]
pub enum Cell {
    #[default]
    Empty = 0,
    P1 = 1,
    P2 = -1,
}

// ── Board ─────────────────────────────────────────────────────────────────────

/// Sparse game board.  All state needed to continue a game from any position.
#[derive(Debug, Clone)]
pub struct Board {
    /// Sparse stone map: (q, r) → Cell.
    pub(crate) cells: HashMap<(i32, i32), Cell>,
    /// Whose turn it is.
    pub current_player: Player,
    /// How many moves the current player still has to place this turn.
    /// Starts at 1 on ply 0 (P1's single first move), then 2 for every turn.
    pub moves_remaining: u8,
    /// Total half-moves placed so far.
    pub ply: u32,
    /// Incremental Zobrist hash.
    pub zobrist_hash: u64,
    /// The move most recently applied (used for fast win detection).
    last_move: Option<(i32, i32)>,
    /// Bounding box of all placed stones (maintained incrementally).
    min_q: i32,
    max_q: i32,
    min_r: i32,
    max_r: i32,
    /// True once at least one stone has been placed.
    has_stones: bool,
}

impl Board {
    /// Create an empty board ready for the first move.
    pub fn new() -> Self {
        Board {
            cells: HashMap::new(),
            current_player: Player::One,
            moves_remaining: 1,
            ply: 0,
            zobrist_hash: 0,
            last_move: None,
            min_q: 0,
            max_q: 0,
            min_r: 0,
            max_r: 0,
            has_stones: false,
        }
    }

    // ── Window ────────────────────────────────────────────────────────────────

    /// Centre of the 19×19 view window: centroid of the bounding box.
    /// Defaults to (0, 0) on an empty board.
    ///
    /// Uses Rust truncating-toward-zero integer division, which matches the
    /// Python-side `board.window_center()` call used in `game_state.py`.
    pub fn window_center(&self) -> (i32, i32) {
        if !self.has_stones {
            return (0, 0);
        }
        let cq = (self.min_q + self.max_q) / 2;
        let cr = (self.min_r + self.max_r) / 2;
        (cq, cr)
    }

    /// Window-relative flat index for axial (q, r).
    ///
    /// Result is in [0, TOTAL_CELLS).  Does not panic; will produce garbage for
    /// out-of-window coords (callers should call `in_window` first).
    #[inline]
    pub fn window_flat_idx(&self, q: i32, r: i32) -> usize {
        let (cq, cr) = self.window_center();
        let wq = q - cq + HALF;
        let wr = r - cr + HALF;
        wq as usize * BOARD_SIZE + wr as usize
    }

    /// Returns the cell at (q, r).
    pub fn get_cell(&self, q: i32, r: i32) -> Cell {
        self.cells.get(&(q, r)).copied().unwrap_or(Cell::Empty)
    }

    /// Axial coordinates (q, r) from a window-relative flat index.
    #[inline]
    pub fn window_coords(&self, flat: usize) -> (i32, i32) {
        let (cq, cr) = self.window_center();
        let wq = (flat / BOARD_SIZE) as i32;
        let wr = (flat % BOARD_SIZE) as i32;
        (wq - HALF + cq, wr - HALF + cr)
    }

    /// Whether (q, r) is inside the current 19×19 view window.
    #[inline]
    pub fn in_window(&self, q: i32, r: i32) -> bool {
        let (cq, cr) = self.window_center();
        let wq = q - cq + HALF;
        let wr = r - cr + HALF;
        wq >= 0 && wq < BOARD_SIZE as i32 && wr >= 0 && wr < BOARD_SIZE as i32
    }

    // ── Queries ───────────────────────────────────────────────────────────────

    /// Cell at (q, r).  Returns Empty for unoccupied or out-of-window cells.
    #[inline]
    pub fn get(&self, q: i32, r: i32) -> Cell {
        self.cells.get(&(q, r)).copied().unwrap_or(Cell::Empty)
    }

    /// All legal moves: empty cells within bounding_box + 2 margin, clipped to
    /// the current 19×19 window.  On an empty board returns all 361 cells.
    pub fn legal_moves(&self) -> Vec<(i32, i32)> {
        let (cq, cr) = self.window_center();
        let (lo_q, hi_q, lo_r, hi_r) = if self.has_stones {
            (
                (self.min_q - 2).max(cq - HALF),
                (self.max_q + 2).min(cq + HALF),
                (self.min_r - 2).max(cr - HALF),
                (self.max_r + 2).min(cr + HALF),
            )
        } else {
            (cq - HALF, cq + HALF, cr - HALF, cr + HALF)
        };

        let cap = ((hi_q - lo_q + 1) * (hi_r - lo_r + 1)) as usize;
        let mut moves = Vec::with_capacity(cap);
        for q in lo_q..=hi_q {
            for r in lo_r..=hi_r {
                if !self.cells.contains_key(&(q, r)) {
                    moves.push((q, r));
                }
            }
        }
        moves
    }

    /// Number of legal moves.
    pub fn legal_move_count(&self) -> usize {
        self.legal_moves().len()
    }

    // ── Move application ──────────────────────────────────────────────────────

    /// Apply a move at (q, r) for the current player.
    ///
    /// Returns an error if the cell is outside the current 19×19 window or is
    /// already occupied.  Any in-window empty cell is accepted (the player is
    /// not restricted to the bbox+2 margin — that is enforced by MCTS via
    /// `legal_moves`, not by the board itself).
    ///
    /// After a successful move:
    /// - `moves_remaining` decrements.
    /// - When it reaches 0 the turn passes: `current_player` flips and
    ///   `moves_remaining` resets to 2.
    pub fn apply_move(&mut self, q: i32, r: i32) -> Result<(), &'static str> {
        if !self.in_window(q, r) {
            return Err("move out of window");
        }
        if self.cells.contains_key(&(q, r)) {
            return Err("cell already occupied");
        }

        // Update bounding box FIRST so that window_flat_idx uses the final
        // bounding box — this keeps the Zobrist hash position-deterministic
        // (same stone set → same bbox → same centre → same hash).
        if !self.has_stones {
            self.min_q = q;
            self.max_q = q;
            self.min_r = r;
            self.max_r = r;
            self.has_stones = true;
        } else {
            if q < self.min_q { self.min_q = q; }
            if q > self.max_q { self.max_q = q; }
            if r < self.min_r { self.min_r = r; }
            if r > self.max_r { self.max_r = r; }
        }

        let player_idx = match self.current_player { Player::One => 0, Player::Two => 1 };

        let cell = match self.current_player {
            Player::One => Cell::P1,
            Player::Two => Cell::P2,
        };
        self.cells.insert((q, r), cell);
        // Use absolute (q, r) for Zobrist — position-independent, no window dependency.
        self.zobrist_hash ^= ZobristTable::get_for_pos(q, r, player_idx);
        self.ply += 1;
        self.last_move = Some((q, r));

        // Advance turn structure
        self.moves_remaining -= 1;
        if self.moves_remaining == 0 {
            self.current_player = self.current_player.other();
            self.moves_remaining = 2;
        }

        Ok(())
    }

    // ── Win detection ─────────────────────────────────────────────────────────

    /// Returns true if either player has 6 in a row (checks last move only).
    pub fn check_win(&self) -> bool {
        match self.last_move {
            None => false,
            Some((q, r)) => {
                let cell = *self.cells.get(&(q, r)).unwrap();
                self.count_in_line(q, r, cell) >= WIN_LENGTH
            }
        }
    }

    /// Returns the winning player, if any.
    pub fn winner(&self) -> Option<Player> {
        if self.player_wins(Player::One) {
            Some(Player::One)
        } else if self.player_wins(Player::Two) {
            Some(Player::Two)
        } else {
            None
        }
    }

    /// Returns true if `player` has 6 stones in a row along any hex axis.
    pub fn player_wins(&self, player: Player) -> bool {
        let cell = match player {
            Player::One => Cell::P1,
            Player::Two => Cell::P2,
        };
        // Fast path: only the player who just moved can have just won.
        if let Some((lq, lr)) = self.last_move {
            if self.cells.get(&(lq, lr)).copied() == Some(cell) {
                return self.count_in_line(lq, lr, cell) >= WIN_LENGTH;
            }
        }
        // Fallback: scan all stones of this player (reached when player != last mover).
        for (&(q, r), &c) in &self.cells {
            if c == cell && self.count_in_line(q, r, cell) >= WIN_LENGTH {
                return true;
            }
        }
        false
    }

    /// Maximum consecutive run through (q, r) for stones of type `cell`,
    /// checked along all three hex axes.
    fn count_in_line(&self, q: i32, r: i32, cell: Cell) -> usize {
        let mut best = 0;
        for &(dq, dr) in &HEX_AXES {
            let count = 1
                + self.count_direction(q, r, dq, dr, cell)
                + self.count_direction(q, r, -dq, -dr, cell);
            if count > best {
                best = count;
            }
        }
        best
    }

    /// Count consecutive stones of `cell` starting from (q, r) in direction
    /// (dq, dr), not counting (q, r) itself.
    pub(crate) fn count_direction(&self, mut q: i32, mut r: i32, dq: i32, dr: i32, cell: Cell) -> usize {
        let mut count = 0;
        loop {
            q += dq;
            r += dr;
            if self.cells.get(&(q, r)).copied() != Some(cell) {
                break;
            }
            count += 1;
        }
        count
    }

    // ── Tensor encoding ───────────────────────────────────────────────────────

    /// Encode the board as a flat f32 array of length `2 * TOTAL_CELLS`
    /// representing shape [2, BOARD_SIZE, BOARD_SIZE]:
    ///   plane 0: current player's stones (window-relative indexing)
    ///   plane 1: opponent's stones
    ///
    /// Stones outside the current 19×19 window are silently omitted.
    pub fn to_planes(&self) -> Vec<f32> {
        let mut out = vec![0.0f32; 2 * TOTAL_CELLS];
        let (my_cell, opp_cell) = match self.current_player {
            Player::One => (Cell::P1, Cell::P2),
            Player::Two => (Cell::P2, Cell::P1),
        };
        let (cq, cr) = self.window_center();
        for (&(q, r), &cell) in &self.cells {
            let wq = q - cq + HALF;
            let wr = r - cr + HALF;
            if wq >= 0 && wq < BOARD_SIZE as i32 && wr >= 0 && wr < BOARD_SIZE as i32 {
                let flat = wq as usize * BOARD_SIZE + wr as usize;
                if cell == my_cell {
                    out[flat] = 1.0;
                } else if cell == opp_cell {
                    out[TOTAL_CELLS + flat] = 1.0;
                }
            }
        }
        out
    }

    /// Same as `to_planes` — present to make the sliding-window semantics
    /// explicit in the PyO3 interface.  `size` is ignored (always 19×19).
    pub fn view_window(&self, _size: usize) -> Vec<f32> {
        self.to_planes()
    }
}

impl Default for Board {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_board_no_win() {
        let b = Board::new();
        assert!(!b.check_win());
        assert!(b.winner().is_none());
    }

    #[test]
    fn first_move_is_single_for_player_one() {
        let mut b = Board::new();
        assert_eq!(b.moves_remaining, 1);
        assert_eq!(b.current_player, Player::One);
        b.apply_move(0, 0).unwrap();
        assert_eq!(b.current_player, Player::Two);
        assert_eq!(b.moves_remaining, 2);
    }

    #[test]
    fn subsequent_turns_have_two_moves() {
        let mut b = Board::new();
        b.apply_move(0, 0).unwrap(); // P1 ply 0
        b.apply_move(1, 0).unwrap(); // P2 first
        assert_eq!(b.current_player, Player::Two);
        assert_eq!(b.moves_remaining, 1);
        b.apply_move(2, 0).unwrap(); // P2 second — turn passes
        assert_eq!(b.current_player, Player::One);
        assert_eq!(b.moves_remaining, 2);
    }

    #[test]
    fn out_of_window_rejected() {
        let mut b = Board::new();
        // Empty board: window = [-9,9]×[-9,9]
        assert!(b.apply_move(10, 0).is_err());
        assert!(b.apply_move(0, -10).is_err());
    }

    #[test]
    fn occupied_cell_rejected() {
        let mut b = Board::new();
        b.apply_move(0, 0).unwrap();
        assert!(b.apply_move(0, 0).is_err());
    }

    #[test]
    fn legal_moves_counts_empty_cells() {
        let mut b = Board::new();
        assert_eq!(b.legal_move_count(), TOTAL_CELLS); // empty → full window
        b.apply_move(0, 0).unwrap();
        // bbox+2 margin = [-2,2]×[-2,2] = 25 cells, minus 1 occupied = 24
        assert_eq!(b.legal_move_count(), 24);
    }

    #[test]
    fn zobrist_changes_on_each_move() {
        let mut b = Board::new();
        let h0 = b.zobrist_hash;
        b.apply_move(0, 0).unwrap();
        let h1 = b.zobrist_hash;
        b.apply_move(1, 0).unwrap();
        let h2 = b.zobrist_hash;
        assert_ne!(h0, h1);
        assert_ne!(h1, h2);
        assert_ne!(h0, h2);
    }

    #[test]
    fn win_e_axis_player_one() {
        // P1: (0,0)…(5,0). P2 fillers on different rows.
        let mut b = Board::new();
        b.apply_move(0, 0).unwrap();
        b.apply_move(-9, 5).unwrap(); b.apply_move(-9, 6).unwrap();
        b.apply_move(1, 0).unwrap(); b.apply_move(2, 0).unwrap();
        b.apply_move(-9, 7).unwrap(); b.apply_move(-9, 8).unwrap();
        b.apply_move(3, 0).unwrap(); b.apply_move(4, 0).unwrap();
        b.apply_move(-9, -5).unwrap(); b.apply_move(-9, -6).unwrap();
        b.apply_move(5, 0).unwrap();
        assert!(b.player_wins(Player::One), "P1 should win along E axis");
        assert!(!b.player_wins(Player::Two), "P2 fillers must not win");
    }

    #[test]
    fn win_ne_axis_player_one() {
        // NE axis: (0,0),(0,1),(0,2),(0,3),(0,4),(0,5)
        let mut b = Board::new();
        b.apply_move(0, 0).unwrap();
        b.apply_move(-1, 0).unwrap(); b.apply_move(-2, 0).unwrap();
        b.apply_move(0, 1).unwrap(); b.apply_move(0, 2).unwrap();
        b.apply_move(-3, 0).unwrap(); b.apply_move(-4, 0).unwrap();
        b.apply_move(0, 3).unwrap(); b.apply_move(0, 4).unwrap();
        b.apply_move(-5, 0).unwrap(); b.apply_move(-6, 0).unwrap();
        b.apply_move(0, 5).unwrap();
        assert!(b.player_wins(Player::One), "P1 should win along NE axis");
    }

    #[test]
    fn win_nw_axis_player_one() {
        // NW axis: (0,0),(-1,1),(-2,2),(-3,3),(-4,4),(-5,5)
        let mut b = Board::new();
        b.apply_move(0, 0).unwrap();
        b.apply_move(1, 0).unwrap(); b.apply_move(2, 0).unwrap();
        b.apply_move(-1, 1).unwrap(); b.apply_move(-2, 2).unwrap();
        b.apply_move(3, 0).unwrap(); b.apply_move(4, 0).unwrap();
        b.apply_move(-3, 3).unwrap(); b.apply_move(-4, 4).unwrap();
        b.apply_move(5, 0).unwrap(); b.apply_move(6, 0).unwrap();
        b.apply_move(-5, 5).unwrap();
        assert!(b.player_wins(Player::One), "P1 should win along NW axis");
    }

    #[test]
    fn five_in_row_is_not_win() {
        let mut b = Board::new();
        b.apply_move(0, 0).unwrap();
        b.apply_move(-1, -1).unwrap(); b.apply_move(-2, -2).unwrap();
        b.apply_move(1, 0).unwrap(); b.apply_move(2, 0).unwrap();
        b.apply_move(-3, -3).unwrap(); b.apply_move(-4, -4).unwrap();
        b.apply_move(3, 0).unwrap(); b.apply_move(4, 0).unwrap();
        // P1 has (0,0),(1,0),(2,0),(3,0),(4,0) = 5 in a row — not a win
        assert!(!b.check_win(), "5 in a row should not be a win");
    }

    #[test]
    fn win_player_two() {
        // P2 builds 6 along E: (0,-1)..(5,-1).  P1 fillers at r=3..6.
        let mut b = Board::new();
        b.apply_move(0, 0).unwrap(); // P1 single first move
        b.apply_move(0, -1).unwrap(); b.apply_move(1, -1).unwrap();
        b.apply_move(0, 3).unwrap(); b.apply_move(0, 4).unwrap();
        b.apply_move(2, -1).unwrap(); b.apply_move(3, -1).unwrap();
        b.apply_move(0, 5).unwrap(); b.apply_move(0, 6).unwrap();
        b.apply_move(4, -1).unwrap(); b.apply_move(5, -1).unwrap();
        assert!(b.player_wins(Player::Two), "P2 should win along E axis");
        assert!(!b.player_wins(Player::One), "P1 fillers must not win");
    }

    #[test]
    fn win_at_board_edge() {
        // P2 builds 6 along NE (q=8, r=-2..3) near right edge of view window.
        // P1 fillers at q=-1,-2,-3,-4 (E axis, only 4 stones).
        let mut b = Board::new();
        b.apply_move(0, 0).unwrap(); // P1 single first move
        b.apply_move(8, -2).unwrap(); b.apply_move(8, -1).unwrap();
        b.apply_move(-1, 0).unwrap(); b.apply_move(-2, 0).unwrap();
        b.apply_move(8, 0).unwrap(); b.apply_move(8, 1).unwrap();
        b.apply_move(-3, 0).unwrap(); b.apply_move(-4, 0).unwrap();
        b.apply_move(8, 2).unwrap(); b.apply_move(8, 3).unwrap();
        assert!(b.player_wins(Player::Two), "P2 wins near right window edge at q=8");
        assert!(!b.player_wins(Player::One), "P1 fillers (4 in a row) must not win");
    }

    // ── New sliding-window tests ───────────────────────────────────────────────

    #[test]
    fn empty_view_window_is_all_zeros() {
        let b = Board::new();
        let planes = b.to_planes();
        assert_eq!(planes.len(), 2 * TOTAL_CELLS);
        assert!(planes.iter().all(|&x| x == 0.0), "empty board planes must be all zero");
    }

    #[test]
    fn window_slides_not_clips() {
        // After P1@(0,0) and P2@(8,0) the window must slide right.
        // Both stones must remain visible; the left side must also be accessible.
        let mut b = Board::new();
        b.apply_move(0, 0).unwrap(); // P1
        b.apply_move(8, 0).unwrap(); // P2 — forces window right
        // Bounding box: [0,8]×[0,0]; centre = (4,0)
        assert_eq!(b.window_center(), (4, 0));
        // (0,0) and (8,0) must both be within the 19×19 window
        assert!(b.in_window(0, 0), "left stone must remain in window");
        assert!(b.in_window(8, 0), "right stone must remain in window");
        // Left edge of window is now 4-9 = -5; right edge is 4+9 = 13
        assert!(b.in_window(-5, 0), "left window edge must be reachable");
        assert!(!b.in_window(-6, 0), "one beyond left edge must be out-of-window");
    }

    #[test]
    fn legal_grows_with_bounding_box() {
        let mut b = Board::new();
        b.apply_move(0, 0).unwrap(); // P1 ply0: bbox+2=[-2,2]×[-2,2], 25-1=24
        assert_eq!(b.legal_move_count(), 24);
        b.apply_move(5, 0).unwrap(); // P2: bbox=[0,5], margin=[-2,7]×[-2,2]=10×5-2=48
        assert_eq!(b.legal_move_count(), 48);
    }
}
