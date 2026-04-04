use fxhash::FxHashSet;
use super::state::{Board, Cell, Player, HEX_AXES, hex_distance};

/// Stones in a row required to win.
const WIN_LENGTH: usize = 6;

/// Maximum hex distance from any existing stone at which a new stone may be
/// placed, per the official hexo.did.science rules ("at most 8 cells apart").
const LEGAL_MOVE_RADIUS: i32 = 8;

impl Board {
    /// Zero-allocation reference to the lazily-maintained legal move set.
    ///
    /// If the cache is dirty (any `apply_move` or `undo_move` since the last
    /// call), rebuilds by iterating a hex ball of radius `LEGAL_MOVE_RADIUS`
    /// (8) centred on every existing stone and collecting unoccupied cells.
    /// This matches the official rule: "a new hex can be placed at most 8
    /// cells apart from any other hex."
    ///
    /// Prefer this over `legal_moves()` in MCTS expansion — it avoids a Vec
    /// allocation.  During tree traversal, `apply_move_tracked` / `undo_move`
    /// are cheap (just a dirty-flag set); the rebuild cost is paid only once
    /// per leaf expansion.
    ///
    /// # Safety
    ///
    /// Uses `UnsafeCell` for interior mutability.  Safe because `Board` is
    /// single-owner / single-thread per MCTS worker — no concurrent access.
    pub fn legal_moves_set(&self) -> &FxHashSet<(i32, i32)> {
        if self.cache_dirty.get() {
            // SAFETY: Board is single-owner; no concurrent mutation possible.
            let cache = unsafe { &mut *self.legal_cache.get() };
            cache.clear();
            if self.cells.is_empty() {
                // Empty board: 5×5 region, same as Board::new() init.
                for dq in -2i32..=2 {
                    for dr in -2i32..=2 {
                        cache.insert((dq, dr));
                    }
                }
            } else {
                // For every placed stone, emit all empty cells within
                // LEGAL_MOVE_RADIUS hex steps (official 8-cell rule).
                // The hex ball in axial coords is the set of (dq, dr) satisfying:
                //   |dq| ≤ R, |dr| ≤ R, |dq + dr| ≤ R
                // which translates to dr ∈ [max(-R, -R-dq), min(R, R-dq)].
                let r = LEGAL_MOVE_RADIUS;
                for &(sq, sr) in self.cells.keys() {
                    for dq in -r..=r {
                        let dr_min = (-r).max(-r - dq);
                        let dr_max = r.min(r - dq);
                        for dr in dr_min..=dr_max {
                            let pos = (sq + dq, sr + dr);
                            if !self.cells.contains_key(&pos) {
                                cache.insert(pos);
                            }
                        }
                    }
                }
            }
            self.cache_dirty.set(false);
        }
        // SAFETY: cache is now valid and we hold &self for its lifetime.
        unsafe { &*self.legal_cache.get() }
    }

    /// All legal moves as a sorted Vec.  Delegates to `legal_moves_set()`.
    ///
    /// Use `legal_moves_set()` in performance-critical paths.
    pub fn legal_moves(&self) -> Vec<(i32, i32)> {
        let mut moves_vec: Vec<(i32, i32)> = self.legal_moves_set().iter().cloned().collect();
        moves_vec.sort_unstable();
        moves_vec
    }

    /// Number of legal moves — O(1) when cache is clean, O(n²+bbox) on first
    /// call after a mutating operation (same cost as the original legal_moves()).
    pub fn legal_move_count(&self) -> usize {
        self.legal_moves_set().len()
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

    /// Returns true if `player` has at least `min_len` consecutive stones along
    /// any of the three hex axes.  Used as a cheap pre-check before the more
    /// expensive `count_winning_moves`: a winning move requires ≥ WIN_LENGTH-1
    /// consecutive stones, so if no such run exists the full count is unnecessary.
    ///
    /// O(player_stones × 3 × avg_run_length) — much cheaper than O(legal_moves)
    /// because legal_moves grows with hex-ball-8 radius while stone count is fixed.
    pub fn has_player_long_run(&self, player: Player, min_len: usize) -> bool {
        let cell = match player {
            Player::One => Cell::P1,
            Player::Two => Cell::P2,
        };
        for (&(q, r), &c) in self.cells.iter() {
            if c != cell {
                continue;
            }
            for &(dq, dr) in &HEX_AXES {
                let run = 1
                    + self.count_direction(q, r, dq, dr, cell)
                    + self.count_direction(q, r, -dq, -dr, cell);
                if run >= min_len {
                    return true;
                }
            }
        }
        false
    }

    /// Count how many empty cells, if occupied by `player`, would give `player`
    /// a completed 6-in-a-row (a winning move).
    ///
    /// Scans the legal-move set (cells within LEGAL_MOVE_RADIUS of any stone),
    /// which is O(legal_moves).  Each cell is checked by computing the run length
    /// through it along all three hex axes — without actually placing a stone —
    /// using the existing `count_direction` helper.
    ///
    /// Used by the MCTS quiescence check: if `count >= 3` the current player has
    /// a provably forced win because the opponent can block at most 2 per turn.
    pub fn count_winning_moves(&self, player: Player) -> u32 {
        let cell = match player {
            Player::One => Cell::P1,
            Player::Two => Cell::P2,
        };

        let legal = self.legal_moves_set();
        let mut count = 0u32;

        for &(q, r) in legal {
            for &(dq, dr) in &HEX_AXES {
                let run = 1
                    + self.count_direction(q, r, dq, dr, cell)
                    + self.count_direction(q, r, -dq, -dr, cell);
                if run >= WIN_LENGTH {
                    count += 1;
                    break; // count each cell at most once
                }
            }
        }

        count
    }

    // ── Cluster helpers ───────────────────────────────────────────────────────

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
                    if !visited[j] && hex_distance(stones[curr].0, stones[curr].1, stones[j].0, stones[j].1) <= 8 {
                        visited[j] = true;
                        queue.push(j);
                    }
                }
            }
            clusters.push(cluster);
        }

        clusters
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_count_winning_moves_empty_board() {
        let board = Board::new();
        // No stones → no winning moves for either player.
        assert_eq!(board.count_winning_moves(Player::One), 0);
        assert_eq!(board.count_winning_moves(Player::Two), 0);
    }

    #[test]
    fn test_count_winning_moves_five_in_row() {
        // P1 has 5 stones in a row along E axis: q=0..4 at r=0.
        // Placing at q=-1 or q=5 completes 6-in-a-row → 2 winning moves.
        let mut board = Board::new();
        board.apply_move(0, 0).unwrap(); // P1 ply0 (single)
        // Place 4 more P1 stones (need P2 filler moves between each pair)
        // P2 fillers go far away; we just need P1 to have 5 in a row.
        board.apply_move(0, 9).unwrap(); board.apply_move(0, 8).unwrap(); // P2 turn
        board.apply_move(1, 0).unwrap(); board.apply_move(2, 0).unwrap(); // P1 turn
        board.apply_move(0, 7).unwrap(); board.apply_move(0, 6).unwrap(); // P2 turn
        board.apply_move(3, 0).unwrap(); board.apply_move(4, 0).unwrap(); // P1 turn

        // It's P2's turn, but we can count P1's winning moves directly.
        let p1_wins = board.count_winning_moves(Player::One);
        // Cells q=-1,r=0 and q=5,r=0 both complete 5-in-a-row to 6.
        assert_eq!(p1_wins, 2, "5-in-a-row should have exactly 2 winning moves");
    }

    #[test]
    fn test_count_winning_moves_five_blocked_one_end() {
        // P1: 5 in a row q=0..4 at r=0.
        // P2 blocker at q=-1,r=0 → only q=5 is a winning cell.
        let mut board = Board::new();
        board.apply_move(0, 0).unwrap(); // P1
        board.apply_move(-1, 0).unwrap(); board.apply_move(0, 9).unwrap(); // P2 (blocker + filler)
        board.apply_move(1, 0).unwrap(); board.apply_move(2, 0).unwrap(); // P1
        board.apply_move(0, 8).unwrap(); board.apply_move(0, 7).unwrap(); // P2
        board.apply_move(3, 0).unwrap(); board.apply_move(4, 0).unwrap(); // P1

        let p1_wins = board.count_winning_moves(Player::One);
        assert_eq!(p1_wins, 1, "one end blocked → 1 winning move");
    }

    #[test]
    fn test_count_winning_moves_zero_when_early_game() {
        // After a few scattered moves no one has 5-in-a-row.
        let mut board = Board::new();
        board.apply_move(0, 0).unwrap(); // P1
        board.apply_move(3, 3).unwrap(); board.apply_move(4, 4).unwrap(); // P2
        board.apply_move(0, 5).unwrap(); board.apply_move(5, 0).unwrap(); // P1

        assert_eq!(board.count_winning_moves(Player::One), 0);
        assert_eq!(board.count_winning_moves(Player::Two), 0);
    }

    #[test]
    fn test_count_winning_moves_three_independent_winning_cells() {
        // P1 has three separate 5-in-a-row threats, each with one open end.
        // Axis E at r=0: stones q=0..4; winning cell at q=5 (q=-1 blocked by P2)
        // Axis NE at q=0: stones r=0..4; winning cell at r=5 (r=-1 blocked by P2)
        // Axis NW: stones (0,0),(-1,1),(-2,2),(-3,3),(-4,4); winning cell at (-5,5)

        let mut board = Board::new();

        // We'll manually insert stones to avoid dealing with turn structure.
        // Use cells directly and set ply / player appropriately.
        // Simpler: insert directly into the board's cells map.
        for q in 0..5i32 {
            board.cells.insert((q, 0), Cell::P1);
        }
        // Blocker for E-axis west end
        board.cells.insert((-1, 0), Cell::P2);

        for r in 1..5i32 {  // r=0 already placed above
            board.cells.insert((0, r), Cell::P1);
        }
        // Blocker for NE-axis south end
        board.cells.insert((0, -1), Cell::P2);

        for i in 1..5i32 {  // (0,0) already placed
            board.cells.insert((-i, i), Cell::P1);
        }
        // Blocker for NW-axis south end
        board.cells.insert((1, -1), Cell::P2);

        // Rebuild legal cache.
        board.has_stones = true;
        board.cache_dirty.set(true);

        let p1_wins = board.count_winning_moves(Player::One);
        // E-axis: q=5 (1 cell); NE-axis: (0,5) (1 cell); NW-axis: (-5,5) (1 cell)
        assert!(p1_wins >= 3,
            "expected ≥3 winning moves for three blocked 5-in-a-row threats, got {p1_wins}");
    }

    #[test]
    fn test_has_player_long_run_empty_board() {
        let board = Board::new();
        assert!(!board.has_player_long_run(Player::One, 5));
        assert!(!board.has_player_long_run(Player::Two, 5));
    }

    #[test]
    fn test_has_player_long_run_detects_five_in_row() {
        let mut board = Board::new();
        for q in 0..5i32 {
            board.cells.insert((q, 0), Cell::P1);
        }
        board.has_stones = true;
        board.cache_dirty.set(true);
        assert!(board.has_player_long_run(Player::One, 5),
            "5 consecutive P1 stones should be detected");
        assert!(!board.has_player_long_run(Player::Two, 5),
            "P2 has no long run");
    }

    #[test]
    fn test_has_player_long_run_returns_false_for_scattered() {
        let mut board = Board::new();
        // Scattered P1 stones — no run of 3 along any axis.
        board.cells.insert((0, 0), Cell::P1);
        board.cells.insert((5, 0), Cell::P1);
        board.cells.insert((0, 5), Cell::P1);
        board.has_stones = true;
        board.cache_dirty.set(true);
        assert!(!board.has_player_long_run(Player::One, 3));
    }
}
