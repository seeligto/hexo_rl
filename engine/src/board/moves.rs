use std::cell::RefCell;

use fxhash::FxHashSet;
use super::state::{Board, Cell, Player, HEX_AXES, hex_distance};

/// Per-thread reusable scratch buffers for `get_clusters()` BFS partition.
///
/// §P10 (Wave 4): pre-cycle-2, `get_clusters` allocated `stones`, `visited`,
/// and `queue` Vecs on every call. Per MCTS leaf expansion + per record this
/// produced churn proportional to leaves/move × clusters/board on the hot
/// path. Reusing per-thread scratch eliminates those three allocations; the
/// returned `Vec<Vec<(i32,i32)>>` is still owned-per-cluster (return-type
/// contract is unchanged).
///
/// Thread-local rather than per-Board: `Board::clone` is on the MCTS hot path
/// (`expand_and_backup` reconstructs a board per leaf) and the existing
/// pattern (cycle-1 legal_cache lesson at `state/core.rs:619-651`) is to skip
/// cache-shaped fields in Clone. A per-Board scratch would need the same
/// skip-on-clone treatment AND would not survive across distinct Board calls
/// (each leaf is a fresh clone). Thread-local is shared across all leaves on
/// the same worker thread — strictly more reuse and zero Clone footprint.
struct ClusterScratch {
    stones: Vec<(i32, i32)>,
    visited: Vec<bool>,
    queue: Vec<usize>,
}

impl ClusterScratch {
    fn new() -> Self {
        Self {
            stones: Vec::new(),
            visited: Vec::new(),
            queue: Vec::new(),
        }
    }
}

thread_local! {
    static CLUSTER_SCRATCH_TLS: RefCell<ClusterScratch> = RefCell::new(ClusterScratch::new());
}

/// Stones in a row required to win.
const WIN_LENGTH: usize = 6;

/// Default maximum hex distance from any existing stone at which a new stone
/// may be placed.  Official HTTT rule is 8, but self-play with bootstrap-v6
/// fragments the board past the 19×19 encoding window (§142, §144 W4C); §145
/// Option α' caps the practical radius at 5 to bound game extent without
/// changing the network architecture or buffer schema.  Real games (corpus +
/// SealBot) never exceed radius 5 between consecutive plies, so the rule cap
/// is a no-op for in-distribution play and only suppresses fragmentation in
/// self-play.
///
/// Phase B' v8 (§152 Q2) introduced a per-Board override
/// (`Board::legal_move_radius`) so `SelfPlayRunner` can jitter r ∈ {4, 5, 6}
/// per game.  This constant remains the canonical default for fresh Boards
/// (eval, bots, tests, corpus replay).
pub const DEFAULT_LEGAL_MOVE_RADIUS: i32 = 5;

/// Default maximum hex distance between stones that share a single cluster
/// for the `get_clusters()` partition.  Originally 8 to match the legal-move
/// radius (one cluster per "reachable neighborhood").  Phase B δ.c (§151)
/// lowers this to 5 so that legal_radius == cluster_radius, removing the
/// cluster / move-radius mismatch that produced extra-wide cluster windows
/// under `get_cluster_views()`.  §168 Gate 3 (v6w25 plumbing) introduces a
/// per-Board override `Board.cluster_threshold` so corpus generation can
/// widen this to 8 alongside `cluster_window_size = 25` for matched-
/// perception A/B vs v8 — without disturbing the v6 default.
pub const DEFAULT_CLUSTER_THRESHOLD: i32 = 5;

impl Board {
    /// Zero-allocation reference to the lazily-maintained legal move set.
    ///
    /// If the cache is dirty (any `apply_move` or `undo_move` since the last
    /// call), rebuilds by iterating a hex ball of radius `LEGAL_MOVE_RADIUS`
    /// centred on every existing stone and collecting unoccupied cells.
    /// §145 Option α' caps the radius at 5 to bound self-play game extent
    /// (was 8 per the official HTTT rule).
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
                // For every placed stone, emit all empty cells within the
                // Board's per-game `legal_move_radius` (default 5 from §145
                // Option α'; jittered per game in Phase B' v8 §152 Q2).
                // The hex ball in axial coords is the set of (dq, dr) satisfying:
                //   |dq| ≤ R, |dr| ≤ R, |dq + dr| ≤ R
                // which translates to dr ∈ [max(-R, -R-dq), min(R, R-dq)].
                let r = self.legal_move_radius;
                // Bbox-based upper bound on the legal-move set size. Two
                // independent valid upper bounds, take the min (tighter):
                //  (a) bbox area — every legal cell lies in the axial rectangle
                //      [min_q-r, max_q+r] × [min_r-r, max_r+r]; each cell at
                //      most once a legal move. Tight late-game (high density).
                //  (b) cells.len() · hex-ball-area — every stone contributes at
                //      most a radius-r ball of legal positions; the union can't
                //      exceed the sum. Tight early-game (low density).
                // O(1) integer math, no allocation. saturating ops are
                // defensive against extreme board sizes. Reserving a proven
                // upper bound lets the insert loop grow the table in a single
                // allocation — zero in-loop hashbrown rehash cascade.
                let ru = r.max(0) as usize;
                let w_q = (self.max_q.saturating_sub(self.min_q) as usize)
                    .saturating_add(1)
                    .saturating_add(2 * ru);
                let w_r = (self.max_r.saturating_sub(self.min_r) as usize)
                    .saturating_add(1)
                    .saturating_add(2 * ru);
                let bbox_area = w_q.saturating_mul(w_r);
                // Hex ball area in axial coords = 3r² + 3r + 1.
                let ball_area = 3 * ru * ru + 3 * ru + 1;
                let combo_bound = self.cells.len().saturating_mul(ball_area);
                let upper_bound = bbox_area.min(combo_bound);
                // reserve relative to current capacity — no-op if already sized.
                cache.reserve(upper_bound.saturating_sub(cache.len()));
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
        let mut moves_vec: Vec<(i32, i32)> = self.legal_moves_set().iter().copied().collect();
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
                // Invariant: `apply_move` (state/core.rs) atomically inserts the
                // cell into `self.cells` (line 518) and sets `self.last_move`
                // (line 541). `check_win` is only meaningful after `apply_move`,
                // so when `last_move == Some((q, r))` the cell at (q, r) is
                // guaranteed present → `.unwrap()` is sound.
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
        for (&(q, r), &c) in &self.cells {
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

    /// Returns the cells forming the winning 6-in-a-row, or an empty Vec if no win.
    ///
    /// Fast path: checks from the last placed stone along all three hex axes.
    /// Fallback (§S178 F-fix-1): if last_move doesn't yield a 6-line, scans
    /// all placed stones for any 6-in-a-row. Mirrors `player_wins`'s fallback
    /// at line 184-189 — both must agree on outcome or threat-head target
    /// goes empty on `winner=Some(_)` games where the winning line was not
    /// completed by the immediately-prior move (HTT 2-moves-per-turn rule:
    /// first move of a turn can complete a 6-line then the second move sits
    /// off-line; `winner()` finds it via fallback, this fn must too).
    pub fn find_winning_line(&self) -> Vec<(i32, i32)> {
        if let Some((lq, lr)) = self.last_move {
            if let Some(&cell) = self.cells.get(&(lq, lr)) {
                for &(dq, dr) in &HEX_AXES {
                    let pos_count = self.count_direction(lq, lr, dq, dr, cell) as i32;
                    let neg_count = self.count_direction(lq, lr, -dq, -dr, cell) as i32;
                    let total = (1 + pos_count + neg_count) as usize;
                    if total >= WIN_LENGTH {
                        let mut line = Vec::with_capacity(total);
                        for i in -neg_count..=pos_count {
                            line.push((lq + dq * i, lr + dr * i));
                        }
                        return line;
                    }
                }
            }
        }
        // Fallback scan (§S178 F-fix-1). Sort stones by (q, r) so the choice
        // of returned line is deterministic across HashMap iteration orders.
        // Per-game-end call (not hot path) — sort cost is negligible.
        let mut stones: Vec<((i32, i32), Cell)> =
            self.cells.iter().map(|(&k, &v)| (k, v)).collect();
        stones.sort_unstable_by_key(|&((q, r), _)| (q, r));
        for &((q, r), cell) in &stones {
            for &(dq, dr) in &HEX_AXES {
                // Only count from the start of a run (no predecessor of same colour).
                if self.cells.get(&(q - dq, r - dr)).copied() == Some(cell) {
                    continue;
                }
                let pos_count = self.count_direction(q, r, dq, dr, cell) as i32;
                let total = (1 + pos_count) as usize;
                if total >= WIN_LENGTH {
                    let mut line = Vec::with_capacity(total);
                    for i in 0..=pos_count {
                        line.push((q + dq * i, r + dr * i));
                    }
                    return line;
                }
            }
        }
        vec![]
    }

    // ── Cluster helpers ───────────────────────────────────────────────────────

    /// Partition all placed stones (both colours) into clusters where two
    /// stones share a cluster iff their `hex_distance` is at most
    /// `self.cluster_threshold` (default `DEFAULT_CLUSTER_THRESHOLD = 5`,
    /// runtime-overridable via `set_cluster_threshold`).  Used by
    /// `get_cluster_views()` to emit one window-sized view per cluster.
    pub fn get_clusters(&self) -> Vec<Vec<(i32, i32)>> {
        let mut clusters: Vec<Vec<(i32, i32)>> = Vec::new();
        if self.cells.is_empty() {
            return clusters;
        }

        let threshold = self.cluster_threshold;

        CLUSTER_SCRATCH_TLS.with(|scratch| {
            let mut s = scratch.borrow_mut();
            let ClusterScratch { stones, visited, queue } = &mut *s;

            // §P10: refill the thread-local scratch from this Board's stones.
            // `clear` + `extend` reuses the existing allocation when capacity
            // suffices; `visited.resize(.., false)` zeroes only the in-use
            // prefix (still O(n) but no allocation when n ≤ capacity).
            stones.clear();
            stones.extend(self.cells.keys().copied());
            visited.clear();
            visited.resize(stones.len(), false);
            queue.clear();

            for i in 0..stones.len() {
                if visited[i] { continue; }
                let mut cluster = Vec::new();
                queue.push(i);
                visited[i] = true;

                while let Some(curr) = queue.pop() {
                    cluster.push(stones[curr]);
                    for j in 0..stones.len() {
                        if !visited[j] && hex_distance(stones[curr].0, stones[curr].1, stones[j].0, stones[j].1) <= threshold {
                            visited[j] = true;
                            queue.push(j);
                        }
                    }
                }
                clusters.push(cluster);
            }
        });

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
