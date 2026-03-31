/// Sparse axial hex board with sliding 19×19 view window.
///
/// Coordinate system: axial (q, r).
///   E:  (+1,  0)   W:  (-1,  0)
///   NE: ( 0, +1)   SW: ( 0, -1)
///   NW: (-1, +1)   SE: (+1, -1)
///
/// Storage: FxHashMap<(q,r), Cell> — unbounded.
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

use fxhash::FxHashMap;
use zobrist::ZobristTable;

// ── MoveDiff ──────────────────────────────────────────────────────────────────

/// Captures everything mutated by one `apply_move_tracked` call so that
/// `undo_move` can reverse it in O(1) without any HashMap scan.
///
/// All fields are private; the only way to construct a `MoveDiff` is through
/// `apply_move_tracked`, and the only way to consume it is through `undo_move`.
#[derive(Debug, Clone)]
pub struct MoveDiff {
    pub(crate) q: i32,
    pub(crate) r: i32,
    pub(crate) player: Player,
    // Previous full Zobrist hash state.
    prev_zobrist_hash: u128,
    // Turn-structure state before the move.
    prev_moves_remaining: u8,
    prev_current_player: Player,
    prev_ply: u32,
    // Win-detection state before the move.
    prev_last_move: Option<(i32, i32)>,
    // Bounding-box state before the move (needed for O(1) bbox undo).
    prev_min_q: i32,
    prev_max_q: i32,
    prev_min_r: i32,
    prev_max_r: i32,
    prev_has_stones: bool,
    // Action anchors state before the move.
    prev_action_anchors: [(i32, i32); 4],
    prev_action_anchors_count: usize,
}

/// Board size (cells per axis of the view window).
pub const BOARD_SIZE: usize = 19;
/// Half-width: window covers [-HALF, HALF] relative to its centre.
pub const HALF: i32 = (BOARD_SIZE as i32 - 1) / 2; // 9
/// Total cells in the 19×19 view window.
pub const TOTAL_CELLS: usize = BOARD_SIZE * BOARD_SIZE; // 361
/// Stones in a row required to win.
const WIN_LENGTH: usize = 6;

/// The three hex axis directions (positive direction only; win scan uses ±).
pub fn hex_distance(q1: i32, r1: i32, q2: i32, r2: i32) -> i32 {
    ((q1 - q2).abs() + (q1 + r1 - q2 - r2).abs() + (r1 - r2).abs()) / 2
}

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
    pub(crate) cells: FxHashMap<(i32, i32), Cell>,
    /// Whose turn it is.
    pub current_player: Player,
    /// How many moves the current player still has to place this turn.
    /// Starts at 1 on ply 0 (P1's single first move), then 2 for every turn.
    pub moves_remaining: u8,
    /// Total half-moves placed so far.
    pub ply: u32,
    /// Incremental Zobrist hash.
    pub zobrist_hash: u128,
    /// The move most recently applied (used for fast win detection).
    last_move: Option<(i32, i32)>,
    /// Bounding box of all placed stones (maintained incrementally).
    min_q: i32,
    max_q: i32,
    min_r: i32,
    max_r: i32,
    /// True once at least one stone has been placed.
    has_stones: bool,
    /// Last 4 stones placed (q, r).
    pub(crate) action_anchors: [(i32, i32); 4],
    pub(crate) action_anchors_count: usize,
}

impl Board {
    /// Create an empty board ready for the first move.
    pub fn new() -> Self {
        Board {
            cells: FxHashMap::default(),
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
            action_anchors: [(0, 0); 4],
            action_anchors_count: 0,
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
    /// Result is in [0, TOTAL_CELLS). Returns usize::MAX for out-of-window coords.
    #[inline]
    pub fn window_flat_idx(&self, q: i32, r: i32) -> usize {
        let (cq, cr) = self.window_center();
        Self::window_flat_idx_at(q, r, cq, cr)
    }

    /// Window-relative flat index for axial (q, r) at a specific center.
    #[inline]
    pub fn window_flat_idx_at(q: i32, r: i32, cq: i32, cr: i32) -> usize {
        let wq = q - cq + HALF;
        let wr = r - cr + HALF;
        if wq >= 0 && wq < BOARD_SIZE as i32 && wr >= 0 && wr < BOARD_SIZE as i32 {
            (wq as usize * BOARD_SIZE) + wr as usize
        } else {
            usize::MAX
        }
    }

    /// Returns the cell at (q, r).
    pub fn get_cell(&self, q: i32, r: i32) -> Cell {
        self.cells.get(&(q, r)).map(|r| *r).unwrap_or(Cell::Empty)
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
        self.cells.get(&(q, r)).map(|r| *r).unwrap_or(Cell::Empty)
    }

    /// All legal moves: empty cells within bounding_box + 2 margin, clipped to
    /// the current 19×19 window.  On an empty board returns all 361 cells.
    pub fn legal_moves(&self) -> Vec<(i32, i32)> {
        let mut moves = std::collections::HashSet::new();
        let clusters = self.get_clusters();
        
        if clusters.is_empty() {
            let (cq, cr) = (0, 0);
            let lo_q = cq - HALF;
            let hi_q = cq + HALF;
            let lo_r = cr - HALF;
            let hi_r = cr + HALF;
            for q in lo_q..=hi_q {
                for r in lo_r..=hi_r {
                    moves.insert((q, r));
                }
            }
        } else {
            for cluster in clusters {
                let mut min_q = i32::MAX;
                let mut max_q = i32::MIN;
                let mut min_r = i32::MAX;
                let mut max_r = i32::MIN;
                for &(q, r) in &cluster {
                    min_q = min_q.min(q);
                    max_q = max_q.max(q);
                    min_r = min_r.min(r);
                    max_r = max_r.max(r);
                }
                
                let cq = (min_q + max_q) / 2;
                let cr = (min_r + max_r) / 2;
                
                let lo_q = (min_q - 2).max(cq - HALF);
                let hi_q = (max_q + 2).min(cq + HALF);
                let lo_r = (min_r - 2).max(cr - HALF);
                let hi_r = (max_r + 2).min(cr + HALF);
                
                for q in lo_q..=hi_q {
                    for r in lo_r..=hi_r {
                        if !self.cells.contains_key(&(q, r)) {
                            let wq = q - cq + HALF;
                            let wr = r - cr + HALF;
                            if wq >= 0 && wq < BOARD_SIZE as i32 && wr >= 0 && wr < BOARD_SIZE as i32 {
                                moves.insert((q, r));
                            }
                        }
                    }
                }
            }
        }
        
        let mut moves_vec: Vec<(i32, i32)> = moves.into_iter().collect();
        moves_vec.sort_unstable();
        moves_vec
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

        // Update action anchors (last 4 stones).
        if self.action_anchors_count < 4 {
            self.action_anchors[self.action_anchors_count] = (q, r);
            self.action_anchors_count += 1;
        } else {
            self.action_anchors[0] = self.action_anchors[1];
            self.action_anchors[1] = self.action_anchors[2];
            self.action_anchors[2] = self.action_anchors[3];
            self.action_anchors[3] = (q, r);
        }

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

    /// Apply a move and return a reversible state diff for O(1) undo.
    pub fn apply_move_tracked(&mut self, q: i32, r: i32) -> Result<MoveDiff, &'static str> {
        let diff = MoveDiff {
            q,
            r,
            player: self.current_player,
            prev_zobrist_hash: self.zobrist_hash,
            prev_moves_remaining: self.moves_remaining,
            prev_current_player: self.current_player,
            prev_ply: self.ply,
            prev_last_move: self.last_move,
            prev_min_q: self.min_q,
            prev_max_q: self.max_q,
            prev_min_r: self.min_r,
            prev_max_r: self.max_r,
            prev_has_stones: self.has_stones,
            prev_action_anchors: self.action_anchors,
            prev_action_anchors_count: self.action_anchors_count,
        };

        self.apply_move(q, r)?;
        Ok(diff)
    }

    /// Undo a move previously applied by `apply_move_tracked`.
    pub fn undo_move(&mut self, diff: MoveDiff) {
        if let Some(cell) = self.cells.remove(&(diff.q, diff.r)) {
            debug_assert_eq!(
                cell,
                match diff.player {
                    Player::One => Cell::P1,
                    Player::Two => Cell::P2,
                },
                "undo_move removed a stone with mismatched player",
            );
        } else {
            debug_assert!(false, "undo_move expected placed stone to exist");
        }

        self.zobrist_hash = diff.prev_zobrist_hash;
        self.moves_remaining = diff.prev_moves_remaining;
        self.current_player = diff.prev_current_player;
        self.ply = diff.prev_ply;
        self.last_move = diff.prev_last_move;

        self.min_q = diff.prev_min_q;
        self.max_q = diff.prev_max_q;
        self.min_r = diff.prev_min_r;
        self.max_r = diff.prev_max_r;
        self.has_stones = diff.prev_has_stones;

        self.action_anchors = diff.prev_action_anchors;
        self.action_anchors_count = diff.prev_action_anchors_count;
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
            if self.cells.get(&(lq, lr)).map(|r| *r) == Some(cell) {
                return self.count_in_line(lq, lr, cell) >= WIN_LENGTH;
            }
        }
        // Fallback: scan all stones of this player (reached when player != last mover).
        for (&(q, r), &c) in self.cells.iter() {
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
            if self.cells.get(&(q, r)).map(|r| *r) != Some(cell) {
                break;
            }
            count += 1;
        }
        count
    }

    // ── Tensor encoding ───────────────────────────────────────────────────────

    /// Encode 18 planes for the neural network.
    /// out must have length 18 * TOTAL_CELLS.
    pub fn encode_18_planes_to_buffer(
        &self,
        planes_2: &[f32], // The 2-plane [my, opp] view
        out: &mut [f32]
    ) {
        // Plane 0: my stones
        for i in 0..TOTAL_CELLS {
            out[i] = planes_2[i];
        }
        // Plane 8: opp stones
        for i in 0..TOTAL_CELLS {
            out[8 * TOTAL_CELLS + i] = planes_2[TOTAL_CELLS + i];
        }
        // Plane 16: moves_remaining == 2 ? 1.0 : 0.0
        let mr_val = if self.moves_remaining == 2 { 1.0 } else { 0.0 };
        for i in 0..TOTAL_CELLS {
            out[16 * TOTAL_CELLS + i] = mr_val;
        }
        // Plane 17: ply % 2
        let ply_val = (self.ply % 2) as f32;
        for i in 0..TOTAL_CELLS {
            out[17 * TOTAL_CELLS + i] = ply_val;
        }
        // Planes 1..7 and 9..15 are left as 0.0 (placeholder for history if needed later)
    }

    /// Encode the board as a flat f32 array of length `18 * TOTAL_CELLS`
    /// representing shape [18, BOARD_SIZE, BOARD_SIZE]:
    ///   plane 0: current player's stones
    ///   plane 8: opponent's stones
    ///   plane 16: moves_remaining == 2 ? 1.0 : 0.0
    ///   plane 17: ply % 2
    ///
    /// Stones outside the current 19×19 window are silently omitted.
    pub fn to_planes(&self) -> Vec<f32> {
        let mut planes_2 = vec![0.0f32; 2 * TOTAL_CELLS];
        let (my_cell, opp_cell) = match self.current_player {
            Player::One => (Cell::P1, Cell::P2),
            Player::Two => (Cell::P2, Cell::P1),
        };
        for (&(q, r), &cell) in self.cells.iter() {
            let flat = self.window_flat_idx(q, r);
            if flat < TOTAL_CELLS {
                if cell == my_cell {
                    planes_2[flat] = 1.0;
                } else if cell == opp_cell {
                    planes_2[TOTAL_CELLS + flat] = 1.0;
                }
            }
        }

        let mut out = vec![0.0f32; 18 * TOTAL_CELLS];
        self.encode_18_planes_to_buffer(&planes_2, &mut out);
        out
    }

    /// Same as `to_planes` — present to make the sliding-window semantics
    /// explicit in the PyO3 interface.  `size` is ignored (always 19×19).
    pub fn view_window(&self, _size: usize) -> Vec<f32> {
        self.to_planes()
    }

    pub fn get_clusters(&self) -> Vec<Vec<(i32, i32)>> {
        let mut clusters: Vec<Vec<(i32, i32)>> = Vec::new();
        if self.cells.is_empty() {
            return clusters;
        }
        
        let stones: Vec<(i32, i32)> = self.cells.keys().cloned().collect();
        let mut visited = vec![false; stones.len()];
        
        for i in 0..stones.len() {
            if visited[i] { continue; }
            let mut cluster = Vec::new();
            let mut queue = vec![i];
            visited[i] = true;
            
            while let Some(curr) = queue.pop() {
                cluster.push(stones[curr]);
                for j in 0..stones.len() {
                    if !visited[j] && crate::board::hex_distance(stones[curr].0, stones[curr].1, stones[j].0, stones[j].1) <= 8 {
                        visited[j] = true;
                        queue.push(j);
                    }
                }
            }
            clusters.push(cluster);
        }
        
        clusters
    }

    /// Returns **2-plane views** (2 × TOTAL_CELLS = 722 floats each) and the
    /// window centre (cq, cr) for each cluster.
    ///
    /// Plane 0 = current player's stones in this 19×19 window.
    /// Plane 1 = opponent's stones in this 19×19 window.
    ///
    /// # Why 2 planes, not 18?
    ///
    /// The full AlphaZero input has 18 planes:
    ///   - planes  0–7:  current player's stones at t, t-1, … t-7
    ///   - planes  8–15: opponent's stones        at t, t-1, … t-7
    ///   - planes 16–17: game-state scalars (moves_remaining, ply parity)
    ///
    /// Assembling all 18 planes requires the full move history — which only
    /// Python's `GameState.move_history` possesses.  Encoding 18 planes in
    /// Rust would mean crossing the PyO3 boundary with 6 498 floats per
    /// cluster (18 × 361) while 14 of them are always zero, a 9× overhead.
    ///
    /// Instead:
    ///   - `get_cluster_views` returns the 2-plane current snapshot (722 floats).
    ///   - `GameState.to_tensor()` stacks the current snapshot with historical
    ///     snapshots from `move_history` to form the final (18, 19, 19) tensor.
    ///
    /// The Rust self-play hot-path (`game_runner.rs`) has no Python history.
    /// It expands the 2-plane view to 18 planes via `encode_18_planes_to_buffer`
    /// in-place — the network receives planes 1–7 and 9–15 as zeros for
    /// Rust-driven self-play, which is an acceptable approximation for the
    /// warmup / RL phases before Python-side history is plumbed in.
    ///
    /// `to_planes()` / `Board.to_tensor()` (the single-board Python binding) still
    /// use the full 18-plane encoding via `encode_18_planes_to_buffer`.
    pub fn get_cluster_views(&self) -> (Vec<Vec<f32>>, Vec<(i32, i32)>) {
        let clusters = self.get_clusters();
        let mut final_centers = Vec::new();

        if clusters.is_empty() {
            final_centers.push((0, 0));
        } else {
            let threat_anchors = self.get_threat_anchors();
            let action_anchors = &self.action_anchors[..self.action_anchors_count];

            for cluster in clusters {
                let mut min_q = i32::MAX;
                let mut max_q = i32::MIN;
                let mut min_r = i32::MAX;
                let mut max_r = i32::MIN;
                for &(q, r) in &cluster {
                    min_q = min_q.min(q);
                    max_q = max_q.max(q);
                    min_r = min_r.min(r);
                    max_r = max_r.max(r);
                }

                let span_q = max_q - min_q;
                let span_r = max_r - min_r;

                if span_q <= 15 && span_r <= 15 {
                    // Small Clusters: single 19x19 window centered on geometric middle
                    final_centers.push(((min_q + max_q) / 2, (min_r + max_r) / 2));
                } else {
                    // Massive Clusters: window centered on each Action and Threat anchor in the cluster
                    let mut cluster_anchors = Vec::new();
                    
                    // Action anchors in this cluster
                    for &anchor in action_anchors {
                        if cluster.contains(&anchor) {
                            cluster_anchors.push(anchor);
                        }
                    }
                    
                    // Threat anchors in this cluster
                    for &anchor in &threat_anchors {
                        if cluster.contains(&anchor) {
                            cluster_anchors.push(anchor);
                        }
                    }

                    if cluster_anchors.is_empty() {
                        // Fallback if no anchors found
                        final_centers.push(((min_q + max_q) / 2, (min_r + max_r) / 2));
                    } else {
                        // Deduplicate anchors: radius 5
                        let mut deduped: Vec<(i32, i32)> = Vec::new();
                        for &a in &cluster_anchors {
                            if !deduped.iter().any(|&d| hex_distance(a.0, a.1, d.0, d.1) <= 5) {
                                deduped.push(a);
                            }
                        }
                        final_centers.extend(deduped);
                    }
                }
            }
        }

        let mut views = Vec::with_capacity(final_centers.len());
        let (my_cell, opp_cell) = match self.current_player {
            Player::One => (Cell::P1, Cell::P2),
            Player::Two => (Cell::P2, Cell::P1),
        };
        
        for &(cq, cr) in &final_centers {
            let mut planes_2 = vec![0.0f32; 2 * TOTAL_CELLS];
            for (&(q, r), &cell) in self.cells.iter() {
                let flat = Self::window_flat_idx_at(q, r, cq, cr);
                if flat < TOTAL_CELLS {
                    if cell == my_cell {
                        planes_2[flat] = 1.0;
                    } else if cell == opp_cell {
                        planes_2[TOTAL_CELLS + flat] = 1.0;
                    }
                }
            }
            views.push(planes_2);
        }
        (views, final_centers)
    }

    /// Returns the (q, r) centers of any open 3-in-a-row or 4-in-a-row formations.
    /// A formation is considered "open" if at least one of its ends is empty.
    ///
    /// Performance: Optimized O(Stones) by skipping redundant scans.
    pub fn get_threat_anchors(&self) -> Vec<(i32, i32)> {
        let mut anchors = Vec::with_capacity(8);
        if self.cells.is_empty() {
            return anchors;
        }

        // Pre-collect stones by player to avoid repeated HashMap scans.
        let mut p1_stones = Vec::with_capacity(self.cells.len());
        let mut p2_stones = Vec::with_capacity(self.cells.len());
        for (&pos, &cell) in self.cells.iter() {
            match cell {
                Cell::P1 => p1_stones.push(pos),
                Cell::P2 => p2_stones.push(pos),
                Cell::Empty => {}
            }
        }

        self.append_threat_anchors_for_player(&p1_stones, Cell::P1, &mut anchors);
        self.append_threat_anchors_for_player(&p2_stones, Cell::P2, &mut anchors);

        anchors
    }

    fn append_threat_anchors_for_player(
        &self,
        stones: &[(i32, i32)],
        player_cell: Cell,
        anchors: &mut Vec<(i32, i32)>,
    ) {
        // Track visited (pos, axis) to avoid redundant scans for the same sequence.
        // There are 3 axes, so we can pack (q, r, axis_idx) into a single key if needed,
        // but for simplicity and to avoid allocations, we just check if the previous
        // cell in the direction is the same color.
        for &(q, r) in stones {
            for (dq, dr) in HEX_AXES {
                // Efficiency: Only start scanning if this stone is the *start* of a sequence
                // in this direction. This ensures each sequence is scanned exactly once.
                if self.get(q - dq, r - dr) == player_cell {
                    continue;
                }

                let mut count = 1;
                while self.get(q + dq * count, r + dr * count) == player_cell {
                    count += 1;
                }

                if count == 3 || count == 4 {
                    // Check if open at either end.
                    let open_start = self.get(q - dq, r - dr) == Cell::Empty;
                    let open_end = self.get(q + dq * count, r + dr * count) == Cell::Empty;

                    if open_start || open_end {
                        let center_idx = count / 2;
                        anchors.push((q + dq * center_idx, r + dr * center_idx));
                    }
                }
            }
        }
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

    fn recompute_zobrist(board: &Board) -> u128 {
        let mut hash = 0u128;
        for (&(q, r), &cell) in board.cells.iter() {
            let player_idx = match cell {
                Cell::P1 => 0,
                Cell::P2 => 1,
                Cell::Empty => continue,
            };
            hash ^= super::zobrist::ZobristTable::get_for_pos(q, r, player_idx);
        }
        hash
    }

    fn next_u64(seed: &mut u64) -> u64 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *seed
    }

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
        assert_eq!(planes.len(), 18 * TOTAL_CELLS);
        assert!(planes.iter().all(|x| *x == 0.0), "empty board planes must be all zero");
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

    #[test]
    fn test_action_anchors_tracking() {
        let mut board = Board::new();
        assert_eq!(board.action_anchors_count, 0);

        board.apply_move(0, 0).unwrap();
        assert_eq!(board.action_anchors_count, 1);
        assert_eq!(board.action_anchors[0], (0, 0));

        board.apply_move(1, 1).unwrap();
        board.apply_move(2, 2).unwrap();
        board.apply_move(3, 3).unwrap();
        assert_eq!(board.action_anchors_count, 4);
        assert_eq!(board.action_anchors[0], (0, 0));
        assert_eq!(board.action_anchors[3], (3, 3));

        board.apply_move(4, 4).unwrap();
        assert_eq!(board.action_anchors_count, 4);
        assert_eq!(board.action_anchors[0], (1, 1)); // (0,0) was evicted
        assert_eq!(board.action_anchors[3], (4, 4));
    }

    #[test]
    fn test_threat_anchors_identification() {
        let mut board = Board::new();
        // Place a 3-in-a-row for P1 along the E axis: (0,0), (1,0), (2,0)
        board.cells.insert((0, 0), Cell::P1);
        board.cells.insert((1, 0), Cell::P1);
        board.cells.insert((2, 0), Cell::P1);
        
        // It's open at both ends ((-1,0) and (3,0) are Empty)
        let anchors = board.get_threat_anchors();
        assert_eq!(anchors.len(), 1);
        assert_eq!(anchors[0], (1, 0)); // The center stone

        // Now place another 4-in-a-row for P2 along the NE axis: (5,5), (5,6), (5,7), (5,8)
        board.cells.insert((5, 5), Cell::P2);
        board.cells.insert((5, 6), Cell::P2);
        board.cells.insert((5, 7), Cell::P2);
        board.cells.insert((5, 8), Cell::P2);
        
        let anchors = board.get_threat_anchors();
        assert_eq!(anchors.len(), 2);
        // Centers for 4-in-a-row: count / 2 = 2. So (5, 5 + 2) = (5, 7).
        assert!(anchors.contains(&(1, 0)));
        assert!(anchors.contains(&(5, 7)));
    }

    #[test]
    fn test_apply_undo_symmetry() {
        let mut board = Board::new();
        let mut diffs = Vec::new();
        let mut seed = 0x5eed_1234_5678_90abu64;

        for _ in 0..10 {
            let legal = board.legal_moves();
            assert!(!legal.is_empty(), "expected at least one legal move");
            let idx = (next_u64(&mut seed) as usize) % legal.len();
            let (q, r) = legal[idx];

            let diff = board.apply_move_tracked(q, r).expect("move should be legal");
            diffs.push(diff);

            assert_eq!(board.zobrist_hash, recompute_zobrist(&board));
            assert!(board.has_stones);
            assert!(board.min_q <= board.max_q);
            assert!(board.min_r <= board.max_r);
            assert!(q >= board.min_q && q <= board.max_q);
            assert!(r >= board.min_r && r <= board.max_r);
        }

        while let Some(diff) = diffs.pop() {
            board.undo_move(diff);
        }

        let empty = Board::new();
        assert_eq!(board.cells.len(), empty.cells.len());
        assert_eq!(board.zobrist_hash, empty.zobrist_hash);
        assert_eq!(board.min_q, empty.min_q);
        assert_eq!(board.max_q, empty.max_q);
        assert_eq!(board.min_r, empty.min_r);
        assert_eq!(board.max_r, empty.max_r);
        assert_eq!(board.has_stones, empty.has_stones);
        assert_eq!(board.current_player, empty.current_player);
        assert_eq!(board.moves_remaining, empty.moves_remaining);
        assert_eq!(board.ply, empty.ply);
        assert_eq!(board.last_move, empty.last_move);
        assert_eq!(board.action_anchors_count, empty.action_anchors_count);
    }

    #[test]
    fn cluster_views_returns_two_planes() {
        let mut b = Board::new();
        b.apply_move(0, 0).unwrap(); // P1 at origin; turn passes to P2
        let (views, centers) = b.get_cluster_views();
        assert_eq!(views.len(), 1, "one cluster expected");
        assert_eq!(centers.len(), 1);
        assert_eq!(
            views[0].len(),
            2 * TOTAL_CELLS,
            "get_cluster_views must return 2-plane views (2 * 361 = 722 floats)"
        );
        // Current player is P2. Plane 0 = P2 (current, no stones), Plane 1 = P1 (opponent).
        // P1 stone at origin → flat index = HALF * BOARD_SIZE + HALF = 9*19+9 = 180.
        let origin_flat = (HALF as usize) * BOARD_SIZE + (HALF as usize);
        assert_eq!(views[0][TOTAL_CELLS + origin_flat], 1.0,
            "P1 stone should be in opponent plane (offset TOTAL_CELLS)");
        assert_eq!(views[0][origin_flat], 0.0,
            "current player (P2) has no stones yet");
    }
}
