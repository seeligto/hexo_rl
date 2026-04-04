/// Expansion and backup for the MCTS tree.

use crate::board::Board;
use super::node::Node;
use super::MCTSTree;

impl MCTSTree {
    /// Apply quiescence correction to a NN value at a non-terminal leaf.
    ///
    /// Game theorem: each turn places 2 stones, so the opponent can block at most 2
    /// winning cells per response.  If the current player has ≥3 winning moves,
    /// the win is forced regardless of opponent play → override to +1.0.
    /// Conversely, if the opponent has ≥3 winning moves the current player cannot
    /// prevent a loss on the next turn → override to -1.0.
    ///
    /// The 2-winning-moves case is strong but unproven; we blend the NN value
    /// toward the win/loss boundary by `quiescence_blend_2` (clamped to ±1.0).
    ///
    /// This check runs ONLY at leaf evaluation (value correction only).
    /// The NN policy is still used for MCTS expansion so the network continues
    /// to learn about these positions.
    #[inline]
    pub(crate) fn apply_quiescence(&self, board: &Board, value: f32) -> f32 {
        if !self.quiescence_enabled {
            return value;
        }

        // Cheap pre-check: count_winning_moves is O(legal_moves), which is
        // expensive with hex-ball-8 rules (hundreds of cells even in early/mid
        // game). A winning move requires ≥5 consecutive stones, so skip the
        // full count entirely when neither player has a run of that length.
        // has_player_long_run is O(stones × 3 × avg_run) — much cheaper because
        // stone count << legal_move count.
        let current_player = board.current_player;
        let opponent = current_player.other();
        // WIN_LENGTH = 6, so a winning move needs a run of WIN_LENGTH - 1 = 5.
        let current_may_threat = board.has_player_long_run(current_player, 5);
        let opponent_may_threat = board.has_player_long_run(opponent, 5);

        if !current_may_threat && !opponent_may_threat {
            return value;
        }

        let current_wins = if current_may_threat {
            board.count_winning_moves(current_player)
        } else {
            0
        };
        if current_wins >= 3 {
            return 1.0;
        }
        let opponent_wins = if opponent_may_threat {
            board.count_winning_moves(opponent)
        } else {
            0
        };
        if opponent_wins >= 3 {
            return -1.0;
        }
        if current_wins == 2 {
            return (value + self.quiescence_blend_2).min(1.0);
        }
        if opponent_wins == 2 {
            return (value - self.quiescence_blend_2).max(-1.0);
        }
        value
    }

    /// Expand a single leaf node and backup its value.
    pub(crate) fn expand_and_backup_single(&mut self, leaf_idx: u32, board: &Board, policy: &[f32], value: f32) {
        if self.pool[leaf_idx as usize].is_terminal {
            let tv = self.pool[leaf_idx as usize].terminal_value;
            self.backup(leaf_idx, tv);
            return;
        }
        if self.pool[leaf_idx as usize].is_expanded() {
            // TT-hit path: node already expanded by a previous leaf visit.
            // Still apply quiescence so repeated TT-backed values are corrected.
            let corrected = self.apply_quiescence(board, value);
            self.backup(leaf_idx, corrected);
            return;
        }

        if board.check_win() {
            self.pool[leaf_idx as usize].is_terminal    = true;
            self.pool[leaf_idx as usize].terminal_value = -1.0;
            self.backup(leaf_idx, -1.0);
            return;
        }

        let legal_moves = board.legal_moves_set();
        if legal_moves.is_empty() {
            self.pool[leaf_idx as usize].is_terminal    = true;
            self.pool[leaf_idx as usize].terminal_value = 0.0;
            self.backup(leaf_idx, 0.0);
            return;
        }

        let n_ch         = legal_moves.len();
        let first_child  = self.next_free;
        if first_child as usize + n_ch > self.pool.len() {
            self.backup(leaf_idx, value);
            return;
        }
        self.next_free += n_ch as u32;

        let leaf_mr      = self.pool[leaf_idx as usize].moves_remaining;
        let child_mr: u8 = if leaf_mr == 1 { 2 } else { 1 };

        self.pool[leaf_idx as usize].first_child = first_child;
        self.pool[leaf_idx as usize].n_children  = n_ch as u16;

        for (j, &(q, r)) in legal_moves.iter().enumerate() {
            let ci          = first_child as usize + j;
            let action_flat = board.window_flat_idx(q, r);
            let prior = if action_flat < policy.len() {
                policy[action_flat]
            } else {
                1.0 / n_ch as f32
            };

            let action_encoded = (((q + 128) as u16) << 8) | ((r + 128) as u16);

            self.pool[ci] = Node {
                parent:              leaf_idx,
                action_idx:          action_encoded,
                n_visits:            0,
                w_value:             0.0,
                prior,
                first_child:         u32::MAX,
                n_children:          0,
                moves_remaining:     child_mr,
                is_terminal:         false,
                terminal_value:      0.0,
                virtual_loss_count:  0,
            };
        }

        let corrected = self.apply_quiescence(board, value);
        self.backup(leaf_idx, corrected);
    }

    /// Expand all pending leaves and backup values to the root.
    pub fn expand_and_backup(&mut self, policies: &[Vec<f32>], values: &[f32]) {
        let pending: Vec<(u32, Vec<crate::board::MoveDiff>)> = std::mem::take(&mut self.pending);
        let n = pending.len().min(policies.len()).min(values.len());

        for i in 0..n {
            let (leaf_idx, ref diffs) = pending[i];
            let policy = &policies[i];
            let value  = values[i];

            let mut board = self.root_board.clone();
            for diff in diffs {
                board.apply_move(diff.q, diff.r).expect("moves in diffs must be legal");
            }

            self.transposition_table.insert(board.zobrist_hash, super::node::TTEntry {
                policy: policy.clone(),
                value,
            });

            self.expand_and_backup_single(leaf_idx, &board, policy, value);
        }
    }

    /// Propagate `value` from `node_idx` to the root (negamax with VL reversal).
    pub(crate) fn backup(&mut self, mut node_idx: u32, mut value: f32) {
        loop {
            let node = &mut self.pool[node_idx as usize];
            node.n_visits += 1;
            node.w_value  += value;
            if node.virtual_loss_count > 0 {
                node.virtual_loss_count -= 1;
            }

            let parent = node.parent;
            if parent == u32::MAX {
                break;
            }
            if self.pool[parent as usize].moves_remaining == 1 {
                value = -value;
            }
            node_idx = parent;
        }
    }
}
