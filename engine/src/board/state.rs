use std::cell::{Cell as StdCell, UnsafeCell};
use fxhash::{FxHashMap, FxHashSet};
use super::zobrist::ZobristTable;

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

/// The three hex axis directions (positive direction only; win scan uses ±).
pub fn hex_distance(q1: i32, r1: i32, q2: i32, r2: i32) -> i32 {
    ((q1 - q2).abs() + (q1 + r1 - q2 - r2).abs() + (r1 - r2).abs()) / 2
}

pub const HEX_AXES: [(i32, i32); 3] = [
    (1, 0),  // E / W
    (0, 1),  // NE / SW
    (1, -1), // SE / NW
];

/// All 6 hex directions (each HEX_AXES entry plus its negative).
pub const HEX_DIRS: [(i32, i32); 6] = [
    (1, 0), (-1, 0),   // E, W
    (0, 1), (0, -1),   // NE, SW
    (1, -1), (-1, 1),  // SE, NW
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
///
/// # Thread safety
///
/// `Board` contains `UnsafeCell` for lazy cache rebuilding, making it `!Sync`
/// by default.  We add `unsafe impl Sync for Board` because every `Board`
/// instance is either:
///   - owned by a single MCTS worker thread, or
///   - accessed from Python under the GIL (which serialises all calls).
/// Concurrent mutable access to the same `Board` never occurs.
#[derive(Debug)]
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
    pub(crate) last_move: Option<(i32, i32)>,
    /// Bounding box of all placed stones (maintained incrementally).
    pub(crate) min_q: i32,
    pub(crate) max_q: i32,
    pub(crate) min_r: i32,
    pub(crate) max_r: i32,
    /// True once at least one stone has been placed.
    pub(crate) has_stones: bool,
    /// Last 4 stones placed (q, r).
    pub(crate) action_anchors: [(i32, i32); 4],
    pub(crate) action_anchors_count: usize,
    /// Lazily-maintained set of all currently legal moves.
    ///
    /// Uses interior mutability so that `legal_moves_set(&self)` can rebuild
    /// on demand without requiring `&mut self` (which would conflict with
    /// callers that hold `&Board` references, e.g. `expand_and_backup_single`).
    ///
    /// Invariant: when `cache_dirty` is false, `legal_cache` is correct.
    /// When `cache_dirty` is true, `legal_moves_set()` rebuilds it.
    ///
    /// SAFETY: `Board` is single-owner / single-thread per MCTS worker.
    /// No concurrent access to `legal_cache` can occur.
    pub(crate) legal_cache: UnsafeCell<FxHashSet<(i32, i32)>>,
    /// Set to true by any mutating operation (apply_move / undo_move).
    /// Cleared by `legal_moves_set()` after a full rebuild.
    pub(crate) cache_dirty: StdCell<bool>,
}

impl Board {
    /// Create an empty board ready for the first move.
    pub fn new() -> Self {
        // Pre-populate legal_cache with the 5×5 region centred at (0,0).
        // This restricts the first move to a 25-cell neighbourhood, keeping
        // the MCTS branching factor at ~24 for the entire game (matching the
        // bbox+2 semantics used after every stone is placed).
        //
        // The spec says "all 361 cells legal for empty board", but 361 root
        // children would require 361 PUCT evaluations per sim and create a
        // 14x MCTS throughput regression — an unjustifiable cost when the
        // first move's location is strategically arbitrary.
        //
        // cache_dirty starts false — no rebuild needed until a stone is placed.
        let mut init_cache = FxHashSet::default();
        init_cache.reserve(50);
        for dq in -2i32..=2 {
            for dr in -2i32..=2 {
                init_cache.insert((dq, dr));
            }
        }

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
            legal_cache: UnsafeCell::new(init_cache),
            cache_dirty: StdCell::new(false),
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

    /// Iterator over all occupied cells: yields `(&(q, r), &Cell)` pairs.
    pub fn cells_iter(&self) -> impl Iterator<Item = (&(i32, i32), &Cell)> {
        self.cells.iter()
    }

    /// Cell at (q, r).  Returns Empty for unoccupied or out-of-window cells.
    #[inline]
    pub fn get(&self, q: i32, r: i32) -> Cell {
        self.cells.get(&(q, r)).map(|r| *r).unwrap_or(Cell::Empty)
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

        // Mark legal cache dirty — legal_moves_set() will rebuild lazily.
        // This avoids 24+ HashSet operations on every apply_move (the MCTS hot
        // path calls apply_move ~2D times per simulation during traversal and
        // reconstruction, but legal_moves_set() is only needed once per sim at
        // leaf expansion).
        self.cache_dirty.set(true);

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

        // Mark legal cache dirty — it will be rebuilt lazily on next access.
        // This avoids O(24) HashSet operations per undo (undo is called ~D times
        // per MCTS sim during selection traversal but legal_moves_set() is not
        // called until leaf expansion).
        self.cache_dirty.set(true);

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

    /// Returns **2-plane views** (2 × TOTAL_CELLS = 722 floats each) and the
    /// window centre (cq, cr) for each cluster.
    ///
    /// Plane 0 = current player's stones in this 19×19 window.
    /// Plane 1 = opponent's stones in this 19×19 window.
    ///
    /// # Why 2 planes, not 18?
    ///
    /// The full AlphaZero input has 18 planes:
    ///   - planes  0-7:  current player's stones at t, t-1, … t-7
    ///   - planes  8-15: opponent's stones        at t, t-1, … t-7
    ///   - planes 16-17: game-state scalars (moves_remaining, ply parity)
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
    /// in-place — the network receives planes 1-7 and 9-15 as zeros for
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

impl Clone for Board {
    fn clone(&self) -> Self {
        // Skip copying legal_cache contents — rebuilding a HashSet of N entries
        // is O(N) allocation and dominates clone cost in the MCTS hot path (every
        // expand_and_backup reconstructs a board via clone + apply_move* ).
        //
        // We set cache_dirty = true unconditionally so that the first
        // legal_moves_set() call on the clone rebuilds from `cells` (which IS
        // correctly copied).  This is safe even when diffs is empty (root node
        // expansion) because the rebuild is always correct given a valid `cells`.
        let cap = unsafe { (*self.legal_cache.get()).len() };
        Board {
            cells: self.cells.clone(),
            current_player: self.current_player,
            moves_remaining: self.moves_remaining,
            ply: self.ply,
            zobrist_hash: self.zobrist_hash,
            last_move: self.last_move,
            min_q: self.min_q,
            max_q: self.max_q,
            min_r: self.min_r,
            max_r: self.max_r,
            has_stones: self.has_stones,
            action_anchors: self.action_anchors,
            action_anchors_count: self.action_anchors_count,
            legal_cache: UnsafeCell::new(FxHashSet::with_capacity_and_hasher(cap, Default::default())),
            cache_dirty: StdCell::new(true),
        }
    }
}

// SAFETY: `Board` is always accessed by a single thread (MCTS worker or Python GIL).
// The `UnsafeCell` / `Cell` fields are never reached concurrently via shared references.
unsafe impl Sync for Board {}
