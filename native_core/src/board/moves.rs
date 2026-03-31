use super::state::{Board, Cell, Player, HEX_AXES, HALF, BOARD_SIZE, hex_distance};

/// Stones in a row required to win.
const WIN_LENGTH: usize = 6;

impl Board {
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
