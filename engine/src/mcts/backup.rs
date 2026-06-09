/// Expansion and backup for the MCTS tree.

use std::sync::atomic::{AtomicU64, Ordering};
use fxhash::FxHashSet;
use crate::board::{Board, WIN_LENGTH};
use crate::game_runner::records::LegalSetPolicy;
use super::node::{CachedPolicy, Node};
use super::{MCTSTree, MAX_CHILDREN_PER_NODE};

/// Output of `pick_topk_children`: `(chosen, sort_used)` where `chosen` is a
/// vector of `((q, r), prior)` entries and `sort_used` flags whether the
/// slow sort path ran (vs the fast no-sort path).
pub(crate) type TopKChildren = (Vec<((i32, i32), f32)>, bool);

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
/// Returns `(chosen, topk_truncated)`:
/// * `chosen` — `Vec<((q, r), prior)>`, length `min(legal_moves.len(), K)`,
///   ALWAYS ordered by `(prior desc, window_flat_idx asc)`.
/// * `topk_truncated` — `true` when `n_legal > K` and the Top-K cap dropped
///   the lowest-prior moves; `false` otherwise. (`false` for `n_legal <= K`,
///   so unit tests asserting the no-truncation case still hold.)
///
/// Canonical order — independent of `FxHashSet` iteration order. `legal_moves`
/// is an `FxHashSet`; its iteration order is a hashbrown table-layout artifact
/// (capacity + insertion order), NOT a semantic move order. Emitting children
/// in raw iteration order leaked that layout into the MCTS child array and
/// hence into `pick_best_puct` tie-breaking ("first equal score wins"). §S182's
/// `legal_moves_set` capacity-reserve changed the layout and silently shifted
/// search behaviour (mcts_mean_depth 3.4 -> 2.5 from the bootstrap anchor).
/// Both the small-set and the truncated case now sort by
/// `(prior desc, window_flat_idx asc)`; the flat-index tie-break makes the
/// child array fully deterministic regardless of `FxHashSet` capacity.
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
) -> TopKChildren {
    let n_legal = legal_moves.len();
    let n_ch = n_legal.min(MAX_CHILDREN_PER_NODE);

    // Single canonical path for every node size. Collect `((q,r), sort_prior,
    // flat)` triples, sort by `(prior desc, flat asc)`, truncate to the Top-K
    // cap (a no-op when `n_legal <= MAX_CHILDREN_PER_NODE`), then drain in
    // order into the final `chosen` Vec. Sorting unconditionally is what makes
    // the child array independent of `FxHashSet` iteration order — see the
    // fn-doc: the previous `n_legal <= K` fast path emitted children in raw
    // hash order, so §S182's capacity-reserve perturbed search behaviour.
    // The sort is O(K log K), K <= MAX_CHILDREN_PER_NODE — negligible beside
    // the per-leaf NN forward that dominates expansion cost.
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

    (chosen, n_legal > MAX_CHILDREN_PER_NODE)
}

/// §D-MULTICLUSTER-S0 legal-set counterpart of `pick_topk_children`: reads each
/// child's prior from the ragged `ls` BY COORD (in-window cells from `ls.dense`,
/// the fast array path identical to the dense variant; covered off-window cells
/// from `ls.overflow`; no-coverage cells from the `1/n_ch` floor). Tie-break is
/// the packed (q,r) key — `window_flat_idx` is `usize::MAX` for ALL off-window
/// cells so it is not a stable tiebreak here. Truncates by TRUE prior.
/// `cq`/`cr` are the leaf's global window centre (the centre `ls.dense` was
/// indexed with), passed through to `ls.get`.
#[inline]
pub(crate) fn pick_topk_children_ls(
    legal_moves: &FxHashSet<(i32, i32)>,
    cq: i32,
    cr: i32,
    ls: &LegalSetPolicy,
    trunk_sz: i32,
    half: i32,
) -> TopKChildren {
    let n_legal = legal_moves.len();
    let n_ch = n_legal.min(MAX_CHILDREN_PER_NODE);
    let floor = 1.0 / n_ch as f32;

    let mut all: Vec<((i32, i32), f32, u32)> = legal_moves
        .iter()
        .map(|&(q, r)| {
            let prior = ls.get(q, r, cq, cr, trunk_sz, half, floor);
            // packed (q,r) — unique, total-orderable, deterministic tiebreak.
            let key = (((q + 32768) as u32) << 16) | ((r + 32768) as u32 & 0xFFFF);
            ((q, r), prior, key)
        })
        .collect();

    all.sort_unstable_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.2.cmp(&b.2))
    });
    all.truncate(MAX_CHILDREN_PER_NODE);

    let chosen: Vec<((i32, i32), f32)> = all.into_iter().map(|((q, r), prior, _)| ((q, r), prior)).collect();
    (chosen, n_legal > MAX_CHILDREN_PER_NODE)
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
        // A winning move needs a run of WIN_LENGTH - 1 (C1: no bare 5).
        let current_may_threat = board.has_player_long_run(current_player, WIN_LENGTH - 1);
        let opponent_may_threat = board.has_player_long_run(opponent, WIN_LENGTH - 1);

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
            // CF-1: derive the terminal sign from the leaf's side-to-move, not
            // a hardcoded -1.0. `apply_move` flips the player only on a
            // turn-final stone (mr 1→0→flip→2); a stone-1 win keeps the same
            // player (mr 2→1, no flip). So `mr==1` ⇒ the winner is still to
            // move ⇒ +1.0; `mr==2` ⇒ the player flipped to the loser ⇒ -1.0.
            // The old hardcode scored a first-stone win as a loss, biasing the
            // policy target toward filler-first move orders.
            let tv = if board.moves_remaining == 1 { 1.0 } else { -1.0 };
            self.pool[leaf_idx as usize].is_terminal    = true;
            self.pool[leaf_idx as usize].terminal_value = tv;
            self.backup(leaf_idx, tv);
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
        self.finish_expansion(leaf_idx, board, chosen, value);
    }

    /// Shared tail of `expand_and_backup_single`[`_ls`]: materialise the chosen
    /// children into the pool, apply quiescence to the leaf value, and backup.
    /// Policy-representation-agnostic (operates on the already-picked `chosen`
    /// list), so the dense and legal-set expansion paths share it verbatim —
    /// keeping the dense path's behaviour byte-identical.
    fn finish_expansion(&mut self, leaf_idx: u32, board: &Board, chosen: Vec<((i32, i32), f32)>, value: f32) {
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

    /// §D-MULTICLUSTER-S0 legal-set counterpart of `expand_and_backup_single`.
    /// Pre-checks identical (terminal / TT-hit / win / no-legal); the only
    /// difference is the prior source — `pick_topk_children_ls` reads the ragged
    /// `ls` by coord (off-window covered cells get real priors, no uniform sink).
    /// Shares `finish_expansion`.
    pub(crate) fn expand_and_backup_single_ls(&mut self, leaf_idx: u32, board: &Board, ls: &LegalSetPolicy, value: f32) {
        if self.pool[leaf_idx as usize].is_terminal {
            let tv = self.pool[leaf_idx as usize].terminal_value;
            self.backup(leaf_idx, tv);
            return;
        }
        if self.pool[leaf_idx as usize].is_expanded() {
            let corrected = self.apply_quiescence(board, value);
            self.backup(leaf_idx, corrected);
            return;
        }
        if board.check_win() {
            let tv = if board.moves_remaining == 1 { 1.0 } else { -1.0 };
            self.pool[leaf_idx as usize].is_terminal    = true;
            self.pool[leaf_idx as usize].terminal_value = tv;
            self.backup(leaf_idx, tv);
            return;
        }
        let legal_moves = board.legal_moves_set();
        if legal_moves.is_empty() {
            self.pool[leaf_idx as usize].is_terminal    = true;
            self.pool[leaf_idx as usize].terminal_value = 0.0;
            self.backup(leaf_idx, 0.0);
            return;
        }
        let (cq, cr) = board.window_center();
        let trunk_sz = board.cluster_window_size() as i32;
        let half = (trunk_sz - 1) / 2;
        let (chosen, _sort_used) = pick_topk_children_ls(legal_moves, cq, cr, ls, trunk_sz, half);
        self.finish_expansion(leaf_idx, board, chosen, value);
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
                policy: CachedPolicy::Dense(std::sync::Arc::new(policy.clone())),
                value,
            });

            self.expand_and_backup_single(*leaf_idx, board, policy, value);
        }
    }

    /// §D-MULTICLUSTER-S0 legal-set counterpart of `expand_and_backup`. Caches
    /// the ragged `LegalSetPolicy` in the TT (`CachedPolicy::Ls`) so a TT-hit
    /// re-expansion (selection.rs) replays the same ragged prior.
    pub fn expand_and_backup_ls(&mut self, policies: &[LegalSetPolicy], values: &[f32]) {
        let pending: Vec<(u32, crate::board::Board)> = std::mem::take(&mut self.pending);
        let n = pending.len().min(policies.len()).min(values.len());
        for i in 0..n {
            let (leaf_idx, board) = &pending[i];
            let ls = &policies[i];
            let value = values[i];

            self.transposition_table.insert(board.zobrist_hash, super::node::TTEntry {
                policy: CachedPolicy::Ls(std::sync::Arc::new(ls.clone())),
                value,
            });

            self.expand_and_backup_single_ls(*leaf_idx, board, ls, value);
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

#[cfg(test)]
mod ls_prior_tests {
    //! §D-MULTICLUSTER-S0 — pick_topk_children_ls reads priors by coord.
    use super::*;

    #[test]
    fn test_pick_topk_children_ls_reads_dense_and_overflow() {
        // window centre (0,0), trunk 19, half 9. In-window cells read ls.dense;
        // the off-window (28,0) reads ls.overflow; chosen is sorted by TRUE prior.
        let mut legal: FxHashSet<(i32, i32)> = FxHashSet::default();
        legal.insert((0, 0)); // wq=9,wr=9 → flat 9*19+9 = 180 (in-window)
        legal.insert((1, 0)); // wq=10,wr=9 → flat 10*19+9 = 199 (in-window)
        legal.insert((28, 0)); // wq=37 ≥ 19 → off-window (usize::MAX)

        let mut dense = vec![0.0f32; 19 * 19 + 1];
        dense[180] = 0.2;
        dense[199] = 0.3;
        let mut overflow: fxhash::FxHashMap<(i32, i32), f32> = fxhash::FxHashMap::default();
        overflow.insert((28, 0), 0.5);
        let ls = LegalSetPolicy { dense, overflow };

        let (chosen, truncated) = pick_topk_children_ls(&legal, 0, 0, &ls, 19, 9);
        assert!(!truncated);
        assert_eq!(chosen.len(), 3);
        // sorted by prior desc: (28,0)=0.5 (overflow), (1,0)=0.3, (0,0)=0.2 (dense)
        assert_eq!(chosen[0], ((28, 0), 0.5), "off-window prior read from overflow, ranks first");
        assert_eq!(chosen[1], ((1, 0), 0.3));
        assert_eq!(chosen[2], ((0, 0), 0.2));
    }
}
