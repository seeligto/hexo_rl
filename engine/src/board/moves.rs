use fxhash::FxHashSet;
use super::state::{Board, Cell, Player, HEX_AXES, hex_distance};

/// Stones in a row required to win.
const WIN_LENGTH: usize = 6;

/// Maximum hex distance from any existing stone at which a new stone may be
/// placed, per the official [site-redacted] rules ("at most 8 cells apart").
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
