/// PUCT selection and tree traversal.

use crate::board::{Board, MoveDiff};
use super::MCTSTree;

impl MCTSTree {
    /// PUCT score for `child_idx`, evaluated from `parent_idx`'s player perspective.
    ///
    /// `fpu_value`: pre-computed first-play urgency value for unvisited children.
    /// For visited children this is ignored — their actual Q is used instead.
    #[inline]
    pub(crate) fn puct_score(&self, child_idx: u32, parent_idx: u32, parent_n: f32, fpu_value: f32) -> f32 {
        let child  = &self.pool[child_idx  as usize];
        let parent = &self.pool[parent_idx as usize];

        let q = if child.n_visits == 0 && child.virtual_loss_count == 0 {
            // Unvisited node: use dynamic FPU value.
            // fpu_value is already negated relative to parent when moves_remaining == 1,
            // so we use it directly (the caller computes it from the parent's Q).
            fpu_value
        } else if parent.moves_remaining == 1 {
            -child.q_value_vl(self.virtual_loss, self.vl_adaptive)
        } else {
            child.q_value_vl(self.virtual_loss, self.vl_adaptive)
        };

        let u = self.c_puct * child.prior * parent_n.sqrt()
            / (1.0 + child.n_visits as f32 + child.virtual_loss_count as f32);
        q + u
    }

    /// Walk the tree via PUCT until an unexpanded (or terminal) leaf is found.
    /// Applies virtual loss to every node on the path.
    pub(crate) fn select_one_leaf(&mut self, board: &mut Board, diffs: &mut Vec<MoveDiff>) -> u32 {
        let mut cur: u32 = 0;
        let mut depth = 0;
        loop {
            self.pool[cur as usize].virtual_loss_count += 1;

            let node = &self.pool[cur as usize];
            if node.is_terminal || !node.is_expanded() {
                if depth > self.max_depth_observed {
                    self.max_depth_observed = depth;
                }
                return cur;
            }

            let parent_n = (node.n_visits + node.virtual_loss_count) as f32;
            let first    = node.first_child as usize;
            let n_ch     = node.n_children  as usize;

            // KataGo-style dynamic FPU: value estimate for unvisited children.
            // explored_mass = sum of priors for all children that have been visited.
            // fpu_value = parent_q - fpu_reduction * sqrt(explored_mass)
            // When fpu_reduction == 0.0 this collapses to 0.0 (legacy behaviour).
            let fpu_value = if self.fpu_reduction > 0.0 {
                let parent_q = if node.n_visits > 0 {
                    node.w_value / node.n_visits as f32
                } else {
                    0.0
                };
                let explored_mass: f32 = (first..first + n_ch)
                    .filter(|&i| {
                        let c = &self.pool[i];
                        c.n_visits > 0 || c.virtual_loss_count > 0
                    })
                    .map(|i| self.pool[i].prior)
                    .sum();
                parent_q - self.fpu_reduction * explored_mass.sqrt()
            } else {
                0.0
            };

            // Gumbel MCTS: at root (cur==0), skip PUCT and use the forced child.
            let best = if cur == 0 {
                if let Some(forced) = self.forced_root_child {
                    forced
                } else {
                    (first..first + n_ch)
                        .max_by(|&a, &b| {
                            let sa = self.puct_score(a as u32, cur, parent_n, fpu_value);
                            let sb = self.puct_score(b as u32, cur, parent_n, fpu_value);
                            sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .unwrap() as u32
                }
            } else {
                (first..first + n_ch)
                    .max_by(|&a, &b| {
                        let sa = self.puct_score(a as u32, cur, parent_n, fpu_value);
                        let sb = self.puct_score(b as u32, cur, parent_n, fpu_value);
                        sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .unwrap() as u32
            };

            let val = self.pool[best as usize].action_idx;
            let q = (val >> 16) as i32 - 32768;
            let r = (val & 0xFFFF) as i32 - 32768;

            let diff = board
                .apply_move_tracked(q, r)
                .expect("selected move should always be legal");
            diffs.push(diff);

            cur = best;
            depth += 1;
        }
    }

    /// Select up to `n` distinct leaves for evaluation.
    pub fn select_leaves(&mut self, n: usize) -> Vec<Board> {
        self.pending.clear();
        let mut boards = Vec::with_capacity(n);
        let mut board = self.root_board.clone();
        let mut diffs = Vec::with_capacity(32);

        let mut i = 0;
        let mut attempts = 0;
        let max_attempts = n * 4;

        while i < n && attempts < max_attempts {
            attempts += 1;
            diffs.clear();
            let leaf_idx = self.select_one_leaf(&mut board, &mut diffs);

            if self.pending.iter().any(|(idx, _)| *idx == leaf_idx) {
                self.undo_virtual_loss(leaf_idx);
                self.selection_overlap_count += 1;
                while let Some(diff) = diffs.pop() {
                    board.undo_move(diff);
                }
                continue;
            }

            if let Some(entry) = self.transposition_table.get(&board.zobrist_hash) {
                let policy = entry.policy.clone();
                let value = entry.value;
                self.expand_and_backup_single(leaf_idx, &board, &policy, value);
                while let Some(diff) = diffs.pop() {
                    board.undo_move(diff);
                }
                continue;
            }

            boards.push(board.clone());
            self.pending.push((leaf_idx, diffs.clone()));

            while let Some(diff) = diffs.pop() {
                board.undo_move(diff);
            }
            i += 1;
        }

        debug_assert_eq!(board.zobrist_hash, self.root_board.zobrist_hash);
        debug_assert_eq!(board.ply, self.root_board.ply);

        boards
    }

    /// Reverse virtual loss on all nodes from `node_idx` to the root.
    pub(crate) fn undo_virtual_loss(&mut self, mut node_idx: u32) {
        loop {
            let node = &mut self.pool[node_idx as usize];
            if node.virtual_loss_count > 0 {
                node.virtual_loss_count -= 1;
            }
            let parent = node.parent;
            if parent == u32::MAX {
                break;
            }
            node_idx = parent;
        }
    }
}
