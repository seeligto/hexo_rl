/// Axial hex board with bitboard win detection, legal move generation,
/// move application, and Zobrist hashing.
///
/// Coordinate system: axial (q, r).
///   E:  (+1,  0)   W:  (-1,  0)
///   NE: ( 0, +1)   SW: ( 0, -1)
///   NW: (-1, +1)   SE: (+1, -1)
///
/// Board is a fixed 19×19 window centred at (0,0), so valid coordinates
/// satisfy -9 ≤ q ≤ 9 and -9 ≤ r ≤ 9.  Flat index = (q + 9) * 19 + (r + 9).
///
/// Win condition: 6 stones of the same player in a row along one of the three
/// hex axes (E/W, NE/SW, NW/SE).
///
/// Turn structure:
///   ply 0 (first move ever): player 1 places exactly 1 stone.
///   ply 1+: each player places exactly 2 stones before the turn passes.

pub mod bitboard;
pub mod zobrist;

use bitboard::Bitboard;
use zobrist::ZobristTable;

/// Board size (cells per axis, square window).
pub const BOARD_SIZE: usize = 19;
/// Half-width offset: coordinate range is [-HALF, HALF].
pub const HALF: i32 = 9;
/// Total cells in the 19×19 grid.
pub const TOTAL_CELLS: usize = BOARD_SIZE * BOARD_SIZE;

/// Flat index from axial coordinates. Panics if out of range.
#[inline(always)]
pub fn idx(q: i32, r: i32) -> usize {
    debug_assert!(q >= -HALF && q <= HALF, "q={q} out of range");
    debug_assert!(r >= -HALF && r <= HALF, "r={r} out of range");
    ((q + HALF) as usize) * BOARD_SIZE + ((r + HALF) as usize)
}

/// Axial coordinates from flat index.
#[inline(always)]
pub fn coords(i: usize) -> (i32, i32) {
    let q = (i / BOARD_SIZE) as i32 - HALF;
    let r = (i % BOARD_SIZE) as i32 - HALF;
    (q, r)
}

/// Whether (q, r) is inside the 19×19 window.
#[inline(always)]
pub fn in_bounds(q: i32, r: i32) -> bool {
    q >= -HALF && q <= HALF && r >= -HALF && r <= HALF
}

/// The three hex axis directions (only one direction per axis; the other is the negative).
/// (dq, dr) for positive direction; win detection checks both ways from a cell.
pub const HEX_AXES: [(i32, i32); 3] = [
    (1, 0),   // E / W axis
    (0, 1),   // NE / SW axis
    (1, -1),  // SE / NW axis
];

// ── Player ──────────────────────────────────────────────────────────────────

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

// ── Cell state ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(i8)]
pub enum Cell {
    #[default]
    Empty = 0,
    P1 = 1,
    P2 = -1,
}

// ── Board ────────────────────────────────────────────────────────────────────

/// Game board. All state needed to continue a game from any position.
#[derive(Debug, Clone)]
pub struct Board {
    /// Cell occupancy: indexed by `idx(q, r)`.
    cells: [Cell; TOTAL_CELLS],
    /// Bitboard for player 1's stones.
    bb_p1: Bitboard,
    /// Bitboard for player 2's stones.
    bb_p2: Bitboard,
    /// Whose turn it is.
    pub current_player: Player,
    /// How many moves the current player still has to make this turn.
    /// Starts at 1 on the very first ply, then 2 for every subsequent turn.
    pub moves_remaining: u8,
    /// Total half-moves played so far (one per stone placed).
    pub ply: u32,
    /// Incremental Zobrist hash of the current position.
    pub zobrist_hash: u64,
}

impl Board {
    /// Create an empty board ready for the first move.
    pub fn new() -> Self {
        Board {
            cells: [Cell::Empty; TOTAL_CELLS],
            bb_p1: Bitboard::empty(),
            bb_p2: Bitboard::empty(),
            current_player: Player::One,
            moves_remaining: 1, // player 1 opens with exactly 1 move
            ply: 0,
            zobrist_hash: 0,
        }
    }

    // ── Queries ──────────────────────────────────────────────────────────────

    /// Returns the cell state at (q, r). Panics if out of bounds.
    #[inline]
    pub fn get(&self, q: i32, r: i32) -> Cell {
        self.cells[idx(q, r)]
    }

    /// Whether (q, r) is empty. Returns false for out-of-bounds.
    #[inline]
    pub fn is_empty(&self, q: i32, r: i32) -> bool {
        in_bounds(q, r) && self.cells[idx(q, r)] == Cell::Empty
    }

    /// List of all legal moves as (q, r) pairs.
    /// Every empty cell in the 19×19 window is legal.
    pub fn legal_moves(&self) -> Vec<(i32, i32)> {
        let mut moves = Vec::with_capacity(TOTAL_CELLS);
        for i in 0..TOTAL_CELLS {
            if self.cells[i] == Cell::Empty {
                moves.push(coords(i));
            }
        }
        moves
    }

    /// Number of legal moves (= number of empty cells).
    pub fn legal_move_count(&self) -> usize {
        self.cells.iter().filter(|&&c| c == Cell::Empty).count()
    }

    // ── Move application ─────────────────────────────────────────────────────

    /// Apply a move at (q, r) for the current player. Returns an error string
    /// if the move is illegal (out of bounds or cell occupied).
    ///
    /// After a successful move:
    /// - `moves_remaining` decrements.
    /// - When it reaches 0, the turn passes: `current_player` flips and
    ///   `moves_remaining` is reset to 2.
    pub fn apply_move(&mut self, q: i32, r: i32) -> Result<(), &'static str> {
        if !in_bounds(q, r) {
            return Err("move out of bounds");
        }
        let i = idx(q, r);
        if self.cells[i] != Cell::Empty {
            return Err("cell already occupied");
        }

        // Place stone
        match self.current_player {
            Player::One => {
                self.cells[i] = Cell::P1;
                self.bb_p1.set(i);
                self.zobrist_hash ^= ZobristTable::get(i, 0);
            }
            Player::Two => {
                self.cells[i] = Cell::P2;
                self.bb_p2.set(i);
                self.zobrist_hash ^= ZobristTable::get(i, 1);
            }
        }
        self.ply += 1;

        // Advance turn structure
        self.moves_remaining -= 1;
        if self.moves_remaining == 0 {
            self.current_player = self.current_player.other();
            self.moves_remaining = 2;
        }

        Ok(())
    }

    // ── Win detection ────────────────────────────────────────────────────────

    /// Returns true if either player has 6 in a row.
    pub fn check_win(&self) -> bool {
        self.player_wins(Player::One) || self.player_wins(Player::Two)
    }

    /// Returns the winning player if any.
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
        let bb = match player {
            Player::One => &self.bb_p1,
            Player::Two => &self.bb_p2,
        };
        bb.has_six_in_row()
    }

    // ── Tensor encoding ──────────────────────────────────────────────────────

    /// Encode the current position as a flat f32 array of shape
    /// `[2, BOARD_SIZE, BOARD_SIZE]`:
    ///   plane 0: current player's stones (1.0 / 0.0)
    ///   plane 1: opponent's stones (1.0 / 0.0)
    ///
    /// Caller stacks history planes on top; this produces the two "latest" planes.
    pub fn to_planes(&self) -> Vec<f32> {
        let mut out = vec![0.0f32; 2 * TOTAL_CELLS];
        let (my_cell, opp_cell) = match self.current_player {
            Player::One => (Cell::P1, Cell::P2),
            Player::Two => (Cell::P2, Cell::P1),
        };
        for i in 0..TOTAL_CELLS {
            if self.cells[i] == my_cell {
                out[i] = 1.0;
            } else if self.cells[i] == opp_cell {
                out[TOTAL_CELLS + i] = 1.0;
            }
        }
        out
    }
}

impl Default for Board {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

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
        // Turn should have passed to player 2 with 2 moves
        assert_eq!(b.current_player, Player::Two);
        assert_eq!(b.moves_remaining, 2);
    }

    #[test]
    fn subsequent_turns_have_two_moves() {
        let mut b = Board::new();
        b.apply_move(0, 0).unwrap(); // P1 ply 0
        // P2 first move — still P2's turn
        b.apply_move(1, 0).unwrap();
        assert_eq!(b.current_player, Player::Two);
        assert_eq!(b.moves_remaining, 1);
        // P2 second move — turn passes to P1
        b.apply_move(2, 0).unwrap();
        assert_eq!(b.current_player, Player::One);
        assert_eq!(b.moves_remaining, 2);
    }

    #[test]
    fn out_of_bounds_rejected() {
        let mut b = Board::new();
        assert!(b.apply_move(10, 0).is_err());
        assert!(b.apply_move(0, -10).is_err());
    }

    #[test]
    fn occupied_cell_rejected() {
        let mut b = Board::new();
        b.apply_move(0, 0).unwrap();
        // Now it's P2's turn; try to play on the same cell — should fail
        assert!(b.apply_move(0, 0).is_err());
    }

    #[test]
    fn legal_moves_counts_empty_cells() {
        let mut b = Board::new();
        assert_eq!(b.legal_move_count(), TOTAL_CELLS);
        b.apply_move(0, 0).unwrap();
        assert_eq!(b.legal_move_count(), TOTAL_CELLS - 1);
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
        // P1 builds 6 in a row along E: (0,0),(1,0),(2,0),(3,0),(4,0),(5,0).
        // P2 fillers are on different rows each time so P2 never gets 6-in-a-row.
        let mut b = Board::new();
        b.apply_move(0, 0).unwrap(); // P1 ply0 (single)
        b.apply_move(-9, 5).unwrap(); // P2 filler row r=5
        b.apply_move(-9, 6).unwrap(); // P2 filler row r=6
        b.apply_move(1, 0).unwrap(); // P1
        b.apply_move(2, 0).unwrap(); // P1
        b.apply_move(-9, 7).unwrap(); // P2 filler row r=7
        b.apply_move(-9, 8).unwrap(); // P2 filler row r=8
        b.apply_move(3, 0).unwrap(); // P1
        b.apply_move(4, 0).unwrap(); // P1
        b.apply_move(-9, -5).unwrap(); // P2 filler
        b.apply_move(-9, -6).unwrap(); // P2 filler
        b.apply_move(5, 0).unwrap(); // P1 — 6th stone, win!
        assert!(b.player_wins(Player::One), "P1 should win along E axis");
        assert!(!b.player_wins(Player::Two), "P2 fillers must not create a win");
    }

    #[test]
    fn win_ne_axis_player_one() {
        // NE axis: direction (0, +1). Place (0,0),(0,1),(0,2),(0,3),(0,4),(0,5)
        let mut b = Board::new();
        b.apply_move(0, 0).unwrap(); // P1 ply0
        b.apply_move(-1, 0).unwrap(); // P2
        b.apply_move(-2, 0).unwrap(); // P2
        b.apply_move(0, 1).unwrap(); // P1
        b.apply_move(0, 2).unwrap(); // P1
        b.apply_move(-3, 0).unwrap(); // P2
        b.apply_move(-4, 0).unwrap(); // P2
        b.apply_move(0, 3).unwrap(); // P1
        b.apply_move(0, 4).unwrap(); // P1
        b.apply_move(-5, 0).unwrap(); // P2
        b.apply_move(-6, 0).unwrap(); // P2
        b.apply_move(0, 5).unwrap(); // P1 — 6th, win
        assert!(b.player_wins(Player::One), "P1 should win along NE axis");
    }

    #[test]
    fn win_nw_axis_player_one() {
        // NW axis: direction (-1, +1). Place (0,0),(-1,1),(-2,2),(-3,3),(-4,4),(-5,5)
        let mut b = Board::new();
        b.apply_move(0, 0).unwrap(); // P1 ply0
        b.apply_move(1, 0).unwrap(); // P2
        b.apply_move(2, 0).unwrap(); // P2
        b.apply_move(-1, 1).unwrap(); // P1
        b.apply_move(-2, 2).unwrap(); // P1
        b.apply_move(3, 0).unwrap(); // P2
        b.apply_move(4, 0).unwrap(); // P2
        b.apply_move(-3, 3).unwrap(); // P1
        b.apply_move(-4, 4).unwrap(); // P1
        b.apply_move(5, 0).unwrap(); // P2
        b.apply_move(6, 0).unwrap(); // P2 — wait, 6 is fine (within ±9)
        b.apply_move(-5, 5).unwrap(); // P1 — 6th, win
        assert!(b.player_wins(Player::One), "P1 should win along NW axis");
    }

    #[test]
    fn five_in_row_is_not_win() {
        let mut b = Board::new();
        // P1 gets 5 in a row (ply 0 + turns)
        b.apply_move(0, 0).unwrap();
        b.apply_move(-1, -1).unwrap(); // P2 off to side
        b.apply_move(-2, -2).unwrap(); // P2
        b.apply_move(1, 0).unwrap(); // P1
        b.apply_move(2, 0).unwrap(); // P1
        b.apply_move(-3, -3).unwrap(); // P2
        b.apply_move(-4, -4).unwrap(); // P2
        b.apply_move(3, 0).unwrap(); // P1
        b.apply_move(4, 0).unwrap(); // P1
        // P1 has (0,0),(1,0),(2,0),(3,0),(4,0) = 5 in a row
        assert!(!b.check_win(), "5 in a row should not be a win");
    }

    #[test]
    fn win_player_two() {
        // P2 builds 6 along E: (0,-1)..(5,-1). P1 fillers are at r=5.
        let mut b = Board::new();
        b.apply_move(9, 9).unwrap(); // P1 single first move
        // P2 turn 1
        b.apply_move(0, -1).unwrap(); b.apply_move(1, -1).unwrap();
        // P1 turn 1: fillers at r=5
        b.apply_move(-9, 5).unwrap(); b.apply_move(-8, 5).unwrap();
        // P2 turn 2
        b.apply_move(2, -1).unwrap(); b.apply_move(3, -1).unwrap();
        // P1 turn 2: fillers
        b.apply_move(-7, 5).unwrap(); b.apply_move(-6, 5).unwrap();
        // P2 turn 3 — win
        b.apply_move(4, -1).unwrap(); b.apply_move(5, -1).unwrap();
        assert!(b.player_wins(Player::Two), "P2 should win along E axis");
        assert!(!b.player_wins(Player::One), "P1 fillers must not create a win");
    }

    #[test]
    fn win_at_board_edge() {
        // Win along top edge: q=9, r=-4..1  (NE axis, q fixed, r varies)
        // Actually NE direction is (0,+1), so fixed q, varying r works.
        let mut b = Board::new();
        b.apply_move(-9, -9).unwrap(); // P1 single first move
        // Build P2 six in a row along NE: (0, -4)..(0, 1) — but we want near edge
        // Use q=9, r=-4..-4+5 => r=-4,-3,-2,-1,0,1 all ≤ HALF=9 ✓
        b.apply_move(9, -4).unwrap(); b.apply_move(9, -3).unwrap(); // P2
        b.apply_move(-8, -8).unwrap(); b.apply_move(-7, -7).unwrap(); // P1
        b.apply_move(9, -2).unwrap(); b.apply_move(9, -1).unwrap(); // P2
        b.apply_move(-6, -6).unwrap(); b.apply_move(-5, -5).unwrap(); // P1
        b.apply_move(9, 0).unwrap(); b.apply_move(9, 1).unwrap(); // P2 — win
        assert!(b.player_wins(Player::Two), "P2 wins at q=9 edge");
    }
}
