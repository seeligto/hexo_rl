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
///
/// §173 A8'': `trunk_sz` + `half` are pre-extracted scalars matching the
/// spec-derived NN-input frame geometry (`spec.trunk_size` /
/// `Board::cluster_window_size`). For v6 single-window callers pass 19 / 9;
/// for v6w25 multi-window pass 25 / 12. Per-MCTS-sim hot path — kernel call
/// is `#[inline]` to fold the bounds check + index math into this function.
#[inline]
pub(crate) fn pick_topk_children(
    legal_moves: &FxHashSet<(i32, i32)>,
    cq: i32,
    cr: i32,
    policy: &[f32],
    trunk_sz: i32,
    half: i32,
) -> (Vec<((i32, i32), f32)>, bool) {
    let n_legal = legal_moves.len();
    let n_ch = n_legal.min(MAX_CHILDREN_PER_NODE);

    if n_legal <= MAX_CHILDREN_PER_NODE {
        let chosen: Vec<((i32, i32), f32)> = legal_moves
            .iter()
            .map(|&(q, r)| {
                let flat = Board::window_flat_idx_at_geom(q, r, cq, cr, trunk_sz, half);
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

    // §P5: collapse the prior double-allocation (sort buffer + final `chosen`)
    // into a single pre-sized `chosen` Vec. Sort path now allocates exactly one
    // Vec of `((q,r), sort_prior, flat)` triples (needed for sort + tie-break),
    // sorts + truncates, then drains in order into a single K-sized `chosen`
    // Vec built up-front. No intermediate `.collect()` re-allocation.
    let mut all: Vec<((i32, i32), f32, usize)> = legal_moves
        .iter()
        .map(|&(q, r)| {
            let flat = Board::window_flat_idx_at_geom(q, r, cq, cr, trunk_sz, half);
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

    let mut chosen: Vec<((i32, i32), f32)> = Vec::with_capacity(all.len());
    for ((q, r), _sort_prior, flat) in all {
        let prior = if flat < policy.len() {
            policy[flat]
        } else {
            1.0 / n_ch as f32
        };
        chosen.push(((q, r), prior));
    }

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

        // §P32: consolidate 4 fetch_add(1) sites into a single fetch_add at
        // function end. Bit-equivalent final counter value; saves 3 atomic ops
        // per quiescence fire. SD4: no concurrent reader of
        // quiescence_fire_count mid-apply_quiescence — all loads are
        // telemetry/test-side after game/search completion.
        let mut fired: u64 = 0;
        let current_wins = if current_may_threat {
            board.count_winning_moves(current_player)
        } else {
            0
        };
        let result = if current_wins >= 3 {
            fired = 1;
            1.0
        } else {
            let opponent_wins = if opponent_may_threat {
                board.count_winning_moves(opponent)
            } else {
                0
            };
            if opponent_wins >= 3 {
                fired = 1;
                -1.0
            } else if current_wins == 2 {
                fired = 1;
                (value + self.quiescence_blend_2).min(1.0)
            } else if opponent_wins == 2 {
                fired = 1;
                (value - self.quiescence_blend_2).max(-1.0)
            } else {
                value
            }
        };
        if fired > 0 {
            self.quiescence_fire_count.fetch_add(fired, std::sync::atomic::Ordering::Relaxed);
        }
        result
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
        // §173 A8'': read trunk_sz from Board's cached cluster_window_size
        // (set at Board construction from spec.cluster_window_size for
        // multi-window encodings; falls back to spec.board_size / BOARD_SIZE
        // for single-window). One field access — cheaper than the §173 A5b
        // RegistrySpec-by-value antipattern.
        let (cq, cr) = board.window_center();
        let trunk_sz = board.cluster_window_size() as i32;
        let half = (trunk_sz - 1) / 2;
        let (chosen, _sort_used) = pick_topk_children(legal_moves, cq, cr, policy, trunk_sz, half);
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
        // §P9: pending now owns the leaf `Board` (§P6 tuple shape change).
        // Per-leaf `root_board.clone() + N × apply_move` re-walk eliminated;
        // each leaf board is consumed directly. SD3 disclosure: P9 + P6
        // share the pending tuple shape change (Vec<MoveDiff> → Board).
        let pending: Vec<(u32, crate::board::Board)> = std::mem::take(&mut self.pending);
        let n = pending.len().min(policies.len()).min(values.len());

        // §P7: TTEntry.policy is `Arc<Vec<f32>>`. We allocate one Arc per
        // first-touch insertion (cheaper than the prior per-hit clone path
        // because hits dominate at high TT-hit rate); the policy slice is
        // still threaded through `expand_and_backup_single` as `&[f32]`.
        for i in 0..n {
            let (leaf_idx, board) = &pending[i];
            let policy = &policies[i];
            let value  = values[i];

            self.transposition_table.insert(board.zobrist_hash, super::node::TTEntry {
                policy: std::sync::Arc::new(policy.clone()),
                value,
            });

            self.expand_and_backup_single(*leaf_idx, board, policy, value);
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
