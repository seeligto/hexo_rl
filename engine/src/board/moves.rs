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

/// Stones in a row required to win. C1 (2026-06-02): `pub(crate)` + re-exported
/// from `board` so `mcts/backup.rs` uses `WIN_LENGTH - 1` instead of a bare `5`.
pub(crate) const WIN_LENGTH: usize = 6;

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

    /// CF-1 terminal value from the side-to-move's perspective at a `check_win`
    /// leaf. `apply_move` flips the player ONLY on a turn-final stone
    /// (`moves_remaining` 1→0→flip→2); a first-stone win keeps the winner to
    /// move (`moves_remaining` 2→1, no flip). So at a won terminal:
    ///   `moves_remaining == 1` ⇒ the winner is still to move ⇒ **+1.0**
    ///   `moves_remaining == 2` ⇒ the player flipped to the loser ⇒ **-1.0**
    ///
    /// This is the single engine-owned surface for the CF-1 sign; it mirrors the
    /// inline derivation in `mcts/backup.rs::expand_and_backup_single` (the
    /// canonical self-play path) so Python eval bots (`k_cluster_mcts_bot`,
    /// `v8_mcts_bot`) need not re-derive the sign (SCATTER-1). Only meaningful
    /// when `check_win()` is true.
    #[inline]
    pub fn terminal_value_to_move(&self) -> f32 {
        if self.moves_remaining == 1 { 1.0 } else { -1.0 }
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

    /// All empty legal cells that complete a 6-in-a-row for `player`, sorted.
    ///
    /// The enumerating variant of `count_winning_moves` — same per-cell run
    /// test, returns the cells (not just the count). The threat/defense
    /// enumeration primitive for an offline threat-space search: the cells a
    /// player can play to win NOW (own threats), and equivalently the cells the
    /// opponent must occupy to deny them (defenses). Deterministic (sorted).
    pub fn winning_moves(&self, player: Player) -> Vec<(i32, i32)> {
        let cell = match player {
            Player::One => Cell::P1,
            Player::Two => Cell::P2,
        };
        let legal = self.legal_moves_set();
        let mut wins: Vec<(i32, i32)> = Vec::new();
        for &(q, r) in legal {
            for &(dq, dr) in &HEX_AXES {
                let run = 1
                    + self.count_direction(q, r, dq, dr, cell)
                    + self.count_direction(q, r, -dq, -dr, cell);
                if run >= WIN_LENGTH {
                    wins.push((q, r));
                    break; // each cell at most once
                }
            }
        }
        wins.sort_unstable();
        wins
    }

    /// Cells that, if `player` plays them, give `player` ≥1 immediate winning move
    /// afterward — i.e. moves that CREATE a must-answer threat (an open-4 → win-in-1).
    /// The threat-creating move set behind a threat-space search; lets the search
    /// narrow branching to forcing moves and reach deep mates cheaply.
    ///
    /// Computed in-engine (a single scratch clone, mutate-probe per legal cell) so a
    /// Python caller pays ONE FFI hop, not one clone per candidate. O(legal²) worst
    /// case but gated to the legal set. Sorted/deterministic.
    pub fn threat_moves(&self, player: Player) -> Vec<(i32, i32)> {
        let pcell = match player {
            Player::One => Cell::P1,
            Player::Two => Cell::P2,
        };
        let mut legal: Vec<(i32, i32)> = self.legal_moves_set().iter().copied().collect();
        legal.sort_unstable();
        let mut out = Vec::new();
        for &(cq, cr) in &legal {
            // c is threat-creating iff some length-6 window THROUGH c has exactly 5
            // player stones (counting c) + 1 empty — that empty is a NEW win-in-1 cell
            // c just created. Requiring the window to contain c excludes pre-existing
            // open-fives (those would not be new threats). Clone-free; O(legal·3·6·6).
            let mut is_threat = false;
            'axis: for &(dq, dr) in &HEX_AXES {
                for s in -5..=0i32 {
                    let mut pcount = 0;
                    let mut empties = 0;
                    let mut dead = false;
                    for i in 0..6i32 {
                        let (q, r) = (cq + (s + i) * dq, cr + (s + i) * dr);
                        let occ = if (q, r) == (cq, cr) {
                            Some(pcell)
                        } else {
                            self.cells.get(&(q, r)).copied()
                        };
                        match occ {
                            Some(c) if c == pcell => pcount += 1,
                            None => empties += 1,
                            _ => { dead = true; break; } // opponent stone kills the window
                        }
                    }
                    if !dead && pcount == 5 && empties == 1 {
                        is_threat = true;
                        break 'axis;
                    }
                }
            }
            if is_threat {
                out.push((cq, cr));
            }
        }
        out
    }

    /// Returns the lexicographically-first empty legal cell that completes a
    /// 6-in-a-row for `player`, or `None`. Deterministic across `FxHashSet`
    /// iteration order (legal set is sorted). Mirrors `count_winning_moves`'s
    /// per-cell run test but returns the cell — the primitive behind the O1
    /// forced-win one-hot POLICY target (depth-1 detection).
    pub fn first_winning_move(&self, player: Player) -> Option<(i32, i32)> {
        let cell = match player {
            Player::One => Cell::P1,
            Player::Two => Cell::P2,
        };
        let legal = self.legal_moves_set();
        let mut cells: Vec<(i32, i32)> = legal.iter().copied().collect();
        cells.sort_unstable();
        for (q, r) in cells {
            for &(dq, dr) in &HEX_AXES {
                let run = 1
                    + self.count_direction(q, r, dq, dr, cell)
                    + self.count_direction(q, r, -dq, -dr, cell);
                if run >= WIN_LENGTH {
                    return Some((q, r));
                }
            }
        }
        None
    }

    /// Returns the immediate move (for the SIDE TO MOVE) that proves a
    /// within-turn forced win, or `None`. O1 (SootyOwl-validated) forced-win →
    /// one-hot POLICY target detector. Rides the same winning-move primitive as
    /// the quiescence VALUE override (`count_winning_moves` / `count_direction`)
    /// but returns a single hard target move:
    ///
    /// * `depth >= 1` (any turn-phase): a move completing 6-in-a-row *now*.
    /// * `depth >= 2` AND `moves_remaining == 2`: a first placement that leaves
    ///   the SAME player an immediate win on the SECOND stone of this turn.
    ///   Both stones of a turn are placed before the opponent replies, so such a
    ///   setup is a *proven* forced win. Turn-phase is read from
    ///   `moves_remaining` — NOT ply parity (CF-1/CF-6 discipline). At `mr == 1`
    ///   the opponent moves before the second stone, so depth-2 is deliberately
    ///   suppressed.
    ///
    /// Cheap-pre-gated by `has_player_long_run` (a one-move win needs ≥5, a
    /// two-move win ≥4 consecutive own stones) so the O(legal) / O(legal²) scans
    /// run only on genuine threats. Called once per move at training-target
    /// extraction — NOT in the per-simulation MCTS hot path. Recall note: the
    /// depth-2 ≥4-run pre-gate skips rare two-gap setups (e.g. `XX__XX`,
    /// max-run 2); precision over recall, matching the reference's
    /// candidate-set-only detection.
    pub fn forced_win_move(&self, depth: u8) -> Option<(i32, i32)> {
        if depth == 0 {
            return None;
        }
        let player = self.current_player;

        // depth-1: an immediate 6-completing move (valid at any moves_remaining).
        if self.has_player_long_run(player, WIN_LENGTH - 1) {
            if let Some(mv) = self.first_winning_move(player) {
                return Some(mv);
            }
        }

        // depth-2: a first placement that sets up an immediate win on the same
        // turn's second stone. Only when the player still holds both placements
        // (mr == 2) — otherwise the opponent replies before the second stone.
        if depth >= 2
            && self.moves_remaining == 2
            && self.has_player_long_run(player, WIN_LENGTH - 2)
        {
            let legal = self.legal_moves_set();
            let mut cells: Vec<(i32, i32)> = legal.iter().copied().collect();
            cells.sort_unstable();
            for (q, r) in cells {
                let mut probe = self.clone();
                if probe.apply_move(q, r).is_err() {
                    continue;
                }
                // mr 2→1, no flip: `probe.current_player == player`. A win
                // available now == the same player completes 6 on the 2nd stone.
                if probe.has_player_long_run(player, WIN_LENGTH - 1)
                    && probe.first_winning_move(player).is_some()
                {
                    return Some((q, r));
                }
            }
        }

        None
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

    // ── O1: forced-win → one-hot policy target detector ──────────────────────
    // `forced_win_move(depth)` returns the immediate move (for the SIDE TO MOVE)
    // that proves a within-turn forced win: depth-1 = a move completing 6 now;
    // depth-2 (only at moves_remaining==2) = a first placement that leaves the
    // SAME player an immediate win on the second placement (opponent never moves
    // between the two stones of one turn). Turn-phase is read from
    // moves_remaining — never ply parity (CF-1/CF-6 discipline).

    /// Build a static position with explicit side-to-move + turn-phase. bbox is
    /// set so a depth-2 clone+`apply_move` stays internally consistent.
    fn fwm_board(stones: &[((i32, i32), Cell)], player: Player, mr: u8) -> Board {
        let mut b = Board::new();
        let (mut lq, mut hq, mut lr, mut hr) = (i32::MAX, i32::MIN, i32::MAX, i32::MIN);
        for &((q, r), c) in stones {
            b.cells.insert((q, r), c);
            lq = lq.min(q); hq = hq.max(q); lr = lr.min(r); hr = hr.max(r);
        }
        b.has_stones = true;
        b.min_q = lq; b.max_q = hq; b.min_r = lr; b.max_r = hr;
        b.cache_dirty.set(true);
        b.current_player = player;
        b.moves_remaining = mr;
        b.ply = stones.len() as u32;
        b
    }

    #[test]
    fn test_first_winning_move_returns_completing_cell() {
        // P1 5-in-a-row (q=0..4, r=0): (-1,0) and (5,0) each complete 6.
        let stones: Vec<_> = (0..5).map(|q| ((q, 0), Cell::P1)).collect();
        let b = fwm_board(&stones, Player::One, 2);
        let mv = b.first_winning_move(Player::One).expect("a winning move exists");
        let mut b2 = b.clone();
        b2.apply_move(mv.0, mv.1).unwrap();
        assert!(b2.check_win(), "first_winning_move must complete 6, got {mv:?}");
        assert_eq!(b.first_winning_move(Player::Two), None, "P2 has no winning move");
    }

    #[test]
    fn test_forced_win_move_depth1_completes_six() {
        // P1 5-in-a-row, P1 to move (mr=2): depth-1 returns a 6-completing move.
        let stones: Vec<_> = (0..5).map(|q| ((q, 0), Cell::P1)).collect();
        let b = fwm_board(&stones, Player::One, 2);
        let mv = b.forced_win_move(1).expect("depth-1 forced win exists");
        let mut b2 = b.clone();
        b2.apply_move(mv.0, mv.1).unwrap();
        assert!(b2.check_win(), "depth-1 move must complete 6, got {mv:?}");
    }

    #[test]
    fn test_forced_win_move_depth1_fires_at_mr1() {
        // Depth-1 (immediate win) is valid regardless of turn-phase (mr==1 too).
        let stones: Vec<_> = (0..5).map(|q| ((q, 0), Cell::P1)).collect();
        let b = fwm_board(&stones, Player::One, 1);
        let mv = b.forced_win_move(2).expect("depth-1 must fire at mr==1");
        let mut b2 = b.clone();
        b2.apply_move(mv.0, mv.1).unwrap();
        assert!(b2.check_win(), "got {mv:?}");
    }

    #[test]
    fn test_forced_win_move_depth2_sets_up_within_turn_win() {
        // P1 4-in-a-row (q=0..3, r=0): no single move wins (depth-1 None). At
        // mr==2, a first placement leaves P1 an immediate win for the SECOND
        // stone of the same turn → depth-2 fires and the returned move proves it.
        let stones: Vec<_> = (0..4).map(|q| ((q, 0), Cell::P1)).collect();
        let b = fwm_board(&stones, Player::One, 2);
        assert_eq!(b.forced_win_move(1), None, "4-in-a-row has no immediate win");

        let mv = b.forced_win_move(2).expect("depth-2 forced win exists");
        let mut b2 = b.clone();
        b2.apply_move(mv.0, mv.1).unwrap();
        // After the first placement the SAME player is still to move (mr 2→1, no
        // flip) and now has an immediate win — that is the within-turn forced win.
        assert_eq!(b2.current_player, Player::One,
            "first placement of a 2-move turn keeps the same player");
        assert!(b2.first_winning_move(Player::One).is_some(),
            "after the depth-2 setup P1 must have an immediate win, setup={mv:?}");
    }

    #[test]
    fn test_forced_win_move_depth2_guarded_at_mr1() {
        // Same 4-in-a-row but P1 has only its LAST placement this turn (mr==1):
        // after it the OPPONENT moves and can block → NOT forced. The turn-phase
        // guard must suppress depth-2 at mr==1.
        let stones: Vec<_> = (0..4).map(|q| ((q, 0), Cell::P1)).collect();
        let b = fwm_board(&stones, Player::One, 1);
        assert_eq!(b.forced_win_move(2), None,
            "depth-2 must be guarded off at mr==1 (opponent blocks before 2nd stone)");
    }

    #[test]
    fn test_forced_win_move_targets_side_to_move_only() {
        // P2 holds the 5-in-a-row but it is P1 to move: must NOT return the
        // opponent's winning cell.
        let stones: Vec<_> = (0..5).map(|q| ((q, 0), Cell::P2)).collect();
        let b = fwm_board(&stones, Player::One, 2);
        assert_eq!(b.forced_win_move(2), None, "only the side-to-move's wins count");
    }

    #[test]
    fn test_forced_win_move_none_without_threat() {
        let stones = [((0, 0), Cell::P1), ((5, 0), Cell::P1), ((0, 5), Cell::P1)];
        let b = fwm_board(&stones, Player::One, 2);
        assert_eq!(b.forced_win_move(2), None);
    }

    #[test]
    fn test_terminal_value_to_move_cf1_sign() {
        // SCATTER-1: engine-owned CF-1 terminal sign. At a won terminal,
        // mr==1 (first-stone win, winner still to move) ⇒ +1.0; mr==2
        // (turn-final win, flipped to loser) ⇒ -1.0. Mirrors backup.rs.
        let stones: Vec<_> = (0..5).map(|q| ((q, 0), Cell::P1)).collect();
        let b1 = fwm_board(&stones, Player::One, 1);
        assert_eq!(b1.terminal_value_to_move(), 1.0, "mr==1 ⇒ +1.0");
        let b2 = fwm_board(&stones, Player::One, 2);
        assert_eq!(b2.terminal_value_to_move(), -1.0, "mr==2 ⇒ -1.0");
    }
}
