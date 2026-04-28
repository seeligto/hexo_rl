/// Expansion and backup for the MCTS tree.

use std::sync::atomic::{AtomicU64, Ordering};
use fxhash::FxHashSet;
use crate::board::Board;
use super::node::Node;
use super::{MCTSTree, MAX_CHILDREN_PER_NODE};

/// Process-wide counter of pool-overflow events.
///
/// Should always read 0 in production: `MAX_CHILDREN_PER_NODE` caps children
/// per node, and `MAX_NODES` is sized so `n_sims × leaf_batch × K` fits with
/// headroom. Any non-zero value indicates either a hand-crafted small pool
/// (test fixtures) or a configuration outside the design envelope.
///
/// On overflow the leaf expansion **panics** rather than fabricate a
/// terminal value; this counter increments immediately before the panic so
/// telemetry can attribute the crash. Earlier behaviour (silently marking
/// the leaf terminal with a quiescence-corrected value) was removed because
/// it corrupted training targets without surfacing the issue.
pub static POOL_OVERFLOW_COUNT: AtomicU64 = AtomicU64::new(0);

/// Read-and-reset the global overflow counter atomically. Returns the
/// previous value. Used by bench to bracket measurement windows.
pub fn take_pool_overflow_count() -> u64 {
    POOL_OVERFLOW_COUNT.swap(0, Ordering::Relaxed)
}

/// Read the current overflow count without resetting. Used by training
/// loops that want a running tally rather than a per-window delta.
pub fn pool_overflow_count() -> u64 {
    POOL_OVERFLOW_COUNT.load(Ordering::Relaxed)
}

/// Pick up to `MAX_CHILDREN_PER_NODE` children for a leaf expansion.
///
/// Returns `(chosen, sort_used)`:
/// * `chosen` — `Vec<((q, r), prior)>`, length `min(legal_moves.len(), K)`.
/// * `sort_used` — `true` when the slow path (sort + truncate) ran,
///   `false` for the fast path. Exposed only so unit tests can assert
///   the fast path is taken when `legal_moves.len() <= K`.
///
/// Fast path (`n_legal <= K`): emit every legal move in HashSet iteration
/// order with its policy prior (or `1/n_ch` fallback for out-of-window
/// cells). Identical to the pre-cap behaviour.
///
/// Sort path (`n_legal > K`): order by `(prior desc, window_flat_idx asc)`
/// and take the top K. The flat-index tie-break is what makes the truncated
/// set deterministic regardless of `FxHashSet` iteration order.
///
/// Out-of-window cells (`window_flat_idx == usize::MAX`) get sort prior
/// `0.0` so they sink to the bottom of the slow path; in the fast path
/// they keep the legacy `1/n_ch` fallback prior. Out-of-window legal cells
/// are vanishingly rare given the centred 19×19 view + radius-8 hex ball.
pub(crate) fn pick_topk_children(
    legal_moves: &FxHashSet<(i32, i32)>,
    cq: i32,
    cr: i32,
    policy: &[f32],
) -> (Vec<((i32, i32), f32)>, bool) {
    let n_legal = legal_moves.len();
    let n_ch = n_legal.min(MAX_CHILDREN_PER_NODE);

    if n_legal <= MAX_CHILDREN_PER_NODE {
        let chosen: Vec<((i32, i32), f32)> = legal_moves
            .iter()
            .map(|&(q, r)| {
                let flat = Board::window_flat_idx_at(q, r, cq, cr);
                let prior = if flat < policy.len() {
                    policy[flat]
                } else {
                    1.0 / n_ch as f32
                };
                ((q, r), prior)
            })
            .collect();
        return (chosen, false);
    }

    let mut all: Vec<((i32, i32), f32, usize)> = legal_moves
        .iter()
        .map(|&(q, r)| {
            let flat = Board::window_flat_idx_at(q, r, cq, cr);
            let sort_prior = if flat < policy.len() { policy[flat] } else { 0.0 };
            ((q, r), sort_prior, flat)
        })
        .collect();

    all.sort_unstable_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.2.cmp(&b.2))
    });
    all.truncate(MAX_CHILDREN_PER_NODE);

    let chosen: Vec<((i32, i32), f32)> = all
        .into_iter()
        .map(|((q, r), _sort_prior, flat)| {
            let prior = if flat < policy.len() {
                policy[flat]
            } else {
                1.0 / n_ch as f32
            };
            ((q, r), prior)
        })
        .collect();

    (chosen, true)
}

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

        // Cheap pre-checks — two tiers, ordered by cost:
        //
        // Tier 1 (free): ply gate.
        //   P1 places stones at ply 0, 3-4, 7-8, …; P1 first reaches 5 stones
        //   at ply=8. P2 at ply=9.  With < 8 total half-moves on the board no
        //   player can have 5 consecutive stones → count_winning_moves = 0.
        //   Benchmarks start from an empty board, so this single comparison
        //   eliminates quiescence overhead for virtually all MCTS benchmark
        //   leaves and for early-game leaves during self-play.
        if board.ply < 8 {
            return value;
        }

        // Tier 2 (O(stones × 3 × avg_run)): long-run check.
        //   A winning move requires ≥5 consecutive stones.  Skip the expensive
        //   count_winning_moves (O(legal_moves) with hex-ball-8 rules) for any
        //   player that has no such run.  stone_count << legal_move_count so
        //   this check is much cheaper than count_winning_moves.
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
            self.quiescence_fire_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return 1.0;
        }
        let opponent_wins = if opponent_may_threat {
            board.count_winning_moves(opponent)
        } else {
            0
        };
        if opponent_wins >= 3 {
            self.quiescence_fire_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return -1.0;
        }
        if current_wins == 2 {
            self.quiescence_fire_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return (value + self.quiescence_blend_2).min(1.0);
        }
        if opponent_wins == 2 {
            self.quiescence_fire_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
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

        // Top-K cap on leaf children (see MAX_CHILDREN_PER_NODE doc).
        let (cq, cr) = board.window_center();
        let (chosen, _sort_used) = pick_topk_children(legal_moves, cq, cr, policy);
        let n_ch        = chosen.len();
        let first_child = self.next_free;

        if first_child as usize + n_ch > self.pool.len() {
            // Should be unreachable in production: pool is sized for
            // n_simulations × leaf_batch × MAX_CHILDREN_PER_NODE. Increment
            // the counter for telemetry attribution, then panic — never
            // fabricate a terminal value here, that silently corrupts
            // training targets (the prior `is_terminal=true` shortcut).
            POOL_OVERFLOW_COUNT.fetch_add(1, Ordering::Relaxed);
            panic!(
                "MCTS pool overflow: next_free={} n_ch={} pool_len={} K={}. \
                 Pool sizing assumption violated — increase MAX_NODES or \
                 reduce n_simulations × leaf_batch.",
                first_child, n_ch, self.pool.len(), MAX_CHILDREN_PER_NODE
            );
        }
        self.next_free += n_ch as u32;

        let leaf_mr      = self.pool[leaf_idx as usize].moves_remaining;
        let child_mr: u8 = if leaf_mr == 1 { 2 } else { 1 };

        self.pool[leaf_idx as usize].first_child = first_child;
        self.pool[leaf_idx as usize].n_children  = n_ch as u16;

        for (j, &((q, r), prior)) in chosen.iter().enumerate() {
            let ci             = first_child as usize + j;
            let action_encoded = (((q + 32768) as u32) << 16) | ((r + 32768) as u32 & 0xFFFF);

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
