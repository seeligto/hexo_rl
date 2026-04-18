/// Single-threaded PUCT MCTS tree with a flat pre-allocated node pool.
///
/// Design notes:
/// - Virtual loss (Phase 2): when `select_one_leaf` descends through a node
///   it increments `virtual_loss_count`. This decreases the effective Q seen
///   by subsequent selections on the same path, so batched calls to
///   `select_leaves(n)` naturally spread across different branches. The
///   penalty is reversed during `backup`, leaving `w_value` / `n_visits`
///   correct after the full round-trip.
/// - No per-node heap allocation: all nodes live in `pool: Vec<Node>`.
/// - Negamax value convention: `node.w_value` accumulates values from THAT
///   node's player-to-move perspective. Backup flips sign when the player
///   changes (which happens when `parent.moves_remaining == 1`).
/// - PUCT formula (AlphaZero):
///   `Q(s,a) + c_puct · P(s,a) · √N(s) / (1 + N(s,a))`
///   where Q is from the *parent's* perspective.

pub mod dirichlet;
pub mod node;
mod selection;
mod backup;

pub use node::{Node, TTEntry, MAX_NODES, VIRTUAL_LOSS_PENALTY};

use crate::board::{Board, MoveDiff, BOARD_SIZE};
use fxhash::FxHashMap;
use std::sync::atomic::{AtomicU64, Ordering};

// ── Tree ─────────────────────────────────────────────────────────────────────

pub struct MCTSTree {
    pub pool: Vec<Node>,
    pub(crate) next_free: u32,
    pub root_board: Board,
    pub(crate) c_puct: f32,
    pub(crate) virtual_loss: f32,
    pub(crate) vl_adaptive: bool,
    /// KataGo-style dynamic FPU base. FPU for unvisited children is computed as:
    ///   fpu_value = parent_q - fpu_reduction * sqrt(explored_policy_mass)
    /// where explored_policy_mass = sum of prior for all visited children.
    /// Set to 0.0 to disable (classical fixed-FPU behaviour: Q=0 for unvisited).
    pub(crate) fpu_reduction: f32,
    pub selection_overlap_count: u32,
    pub max_depth_observed: u32,
    pub(crate) pending: Vec<(u32, Vec<MoveDiff>)>,
    pub transposition_table: FxHashMap<u128, TTEntry>,
    /// Enable quiescence value override at leaf nodes.
    /// When true, if the current player has ≥3 winning moves the value is
    /// overridden to +1.0 (forced win); if the opponent has ≥3, to -1.0.
    /// This is a game-specific theorem: each turn places 2 stones, so the
    /// opponent can block at most 2 winning cells per turn.
    pub(crate) quiescence_enabled: bool,
    /// Value blend amount for the 2-winning-moves case (strong but unproven).
    /// The NN value is nudged by ±quiescence_blend_2 toward ±1.0.
    pub(crate) quiescence_blend_2: f32,
    /// When set, `select_one_leaf` skips PUCT at the root and descends
    /// directly to this child pool index. Used by Sequential Halving
    /// (Gumbel MCTS) to force simulations into a specific candidate's subtree.
    pub(crate) forced_root_child: Option<u32>,
    /// Accumulated leaf depth across all simulations since last `new_game()`.
    /// Divide by `sim_count` to get mean depth per simulation.
    pub(crate) depth_accum: u64,
    /// Number of simulations (calls to `select_one_leaf`) since last `new_game()`.
    pub(crate) sim_count: u32,
    /// Cumulative count of quiescence value overrides/blends since last `new_game()`.
    /// Tracks all 4 firing branches: ≥3 current wins (+1.0), ≥3 opponent wins (-1.0),
    /// 2 current wins (blend up), 2 opponent wins (blend down).
    /// Atomic (not Cell) because MCTSTree is wrapped in `#[pyclass] PyMCTSTree`,
    /// which requires `Send + Sync`. Cell is `!Sync` and would break the pyo3 bound.
    pub quiescence_fire_count: AtomicU64,
}

impl MCTSTree {
    pub fn new(c_puct: f32) -> Self {
        MCTSTree::new_with_vl(c_puct, VIRTUAL_LOSS_PENALTY)
    }

    pub fn new_with_vl(c_puct: f32, virtual_loss: f32) -> Self {
        MCTSTree::new_full(c_puct, virtual_loss, 0.0)
    }

    /// Create with a custom pool capacity. Used in tests to force pool overflow.
    pub fn new_with_capacity(c_puct: f32, pool_size: usize) -> Self {
        let mut t = MCTSTree::new_full(c_puct, VIRTUAL_LOSS_PENALTY, 0.0);
        t.pool = vec![Node::uninit(); pool_size];
        t
    }

    pub fn new_full(c_puct: f32, virtual_loss: f32, fpu_reduction: f32) -> Self {
        let pool = vec![Node::uninit(); MAX_NODES];
        MCTSTree {
            pool,
            next_free: 1,
            root_board: Board::new(),
            c_puct,
            virtual_loss,
            vl_adaptive: false,
            fpu_reduction,
            selection_overlap_count: 0,
            max_depth_observed: 0,
            depth_accum: 0,
            sim_count: 0,
            pending: Vec::new(),
            transposition_table: FxHashMap::default(),
            quiescence_enabled: true,
            quiescence_blend_2: 0.3,
            forced_root_child: None,
            quiescence_fire_count: AtomicU64::new(0),
        }
    }

    // ── Game lifecycle ────────────────────────────────────────────────────────

    pub fn new_game(&mut self, board: Board) {
        let mr = board.moves_remaining;
        self.root_board = board;
        self.pool[0] = Node::uninit();
        self.pool[0].moves_remaining = mr;
        self.next_free = 1;
        self.pending.clear();
        self.selection_overlap_count = 0;
        self.max_depth_observed = 0;
        self.depth_accum = 0;
        self.sim_count = 0;
        self.quiescence_fire_count.store(0, Ordering::Relaxed);
        self.forced_root_child = None;
        // Clear TT between games — positions don't repeat across games and
        // Vec<f32> policy entries accumulate unboundedly without this.
        self.transposition_table.clear();
    }

    // ── Policy extraction ─────────────────────────────────────────────────────

    pub fn get_policy(&self, temperature: f32, board_size: usize) -> Vec<f32> {
        let n_actions = board_size * board_size + 1;
        let mut policy = vec![0.0f32; n_actions];

        let root = &self.pool[0];
        if !root.is_expanded() {
            return policy;
        }

        let first = root.first_child as usize;
        let n_ch  = root.n_children  as usize;

        if temperature == 0.0 {
            if let Some(best) = (first..first + n_ch)
                .max_by_key(|&i| self.pool[i].n_visits)
            {
                let val = self.pool[best].action_idx;
                let q = (val >> 16) as i32 - 32768;
                let r = (val & 0xFFFF) as i32 - 32768;
                let action = self.root_board.window_flat_idx(q, r);

                if action < n_actions {
                    policy[action] = 1.0;
                }
            }
        } else {
            let visits: Vec<f32> = (first..first + n_ch)
                .map(|i| (self.pool[i].n_visits as f32).powf(1.0 / temperature))
                .collect();
            let total: f32 = visits.iter().sum();
            if total > 0.0 {
                for (j, &v) in visits.iter().enumerate() {
                    let val = self.pool[first + j].action_idx;
                    let q = (val >> 16) as i32 - 32768;
                    let r = (val & 0xFFFF) as i32 - 32768;
                    let action = self.root_board.window_flat_idx(q, r);

                    if action < n_actions {
                        policy[action] = v / total;
                    }
                }
            }
        }

        policy
    }

    /// Compute improved policy targets using Gumbel completed Q-values
    /// (Danihelka et al., Gumbel AlphaZero, ICLR 2022 §4, Appendix D Eq. 33).
    ///
    /// Returns a `(board_size * board_size + 1)`-dim probability distribution that
    /// incorporates MCTS Q-values into the prior, giving useful policy signal even
    /// at low simulation counts.
    pub fn get_improved_policy(
        &self,
        board_size: usize,
        c_visit: f32,
        c_scale: f32,
    ) -> Vec<f32> {
        let n_actions = board_size * board_size + 1;
        let mut policy = vec![0.0f32; n_actions];

        let root = &self.pool[0];
        if !root.is_expanded() {
            return policy;
        }

        let first = root.first_child as usize;
        let n_ch = root.n_children as usize;

        // Children store w_value in their own player-to-move perspective (backup.rs negamax).
        // When root.moves_remaining==1 the children belong to the opponent, so negate their Q
        // to bring them into root's perspective before computing completed-Q targets.
        let q_sign: f32 = if self.pool[0].moves_remaining == 1 { -1.0 } else { 1.0 };

        // Collect per-child data: (flat_action_idx, visits, prior, q_value)
        let mut child_data: Vec<(usize, u32, f32, f32)> = Vec::with_capacity(n_ch);
        let mut sum_n: u32 = 0;
        let mut max_n: u32 = 0;
        let mut visited_prior_sum: f32 = 0.0;
        let mut policy_weighted_q: f32 = 0.0;

        for j in 0..n_ch {
            let child = &self.pool[first + j];
            let val = child.action_idx;
            let q = (val >> 16) as i32 - 32768;
            let r = (val & 0xFFFF) as i32 - 32768;
            let action = self.root_board.window_flat_idx(q, r);
            if action >= n_actions {
                continue;
            }

            let visits = child.n_visits;
            let prior = child.prior;
            let q_val = if visits > 0 {
                q_sign * child.w_value / visits as f32
            } else {
                0.0
            };

            sum_n += visits;
            if visits > max_n {
                max_n = visits;
            }
            if visits > 0 {
                visited_prior_sum += prior;
                policy_weighted_q += prior * q_val;
            }

            child_data.push((action, visits, prior, q_val));
        }

        // Edge case: no visits at all — return prior distribution
        if sum_n == 0 {
            let mut total_prior = 0.0f32;
            for &(action, _, prior, _) in &child_data {
                policy[action] = prior;
                total_prior += prior;
            }
            if total_prior > 0.0 {
                for p in policy.iter_mut() {
                    *p /= total_prior;
                }
            }
            return policy;
        }

        // v_hat: root value estimate (W/N from root node)
        let v_hat = root.w_value / root.n_visits as f32;

        // v_mix: mixed value estimate for unvisited actions (paper Eq. 33).
        // Note: child.prior values come from softmax and are assumed to sum to
        // ~1.0 over legal actions. No normalization step needed unless expansion
        // pruning is added in the future.
        let v_mix = if visited_prior_sum > 1e-8 {
            let sum_n_f = sum_n as f32;
            (1.0 / (1.0 + sum_n_f))
                * (v_hat + (sum_n_f / visited_prior_sum) * policy_weighted_q)
        } else {
            v_hat
        };

        // Build completed Q-values and compute pi_improved = softmax(log_prior + sigma(completedQ))
        let sigma_scale = (c_visit + max_n as f32) * c_scale;
        let mut logits = vec![f32::NEG_INFINITY; n_actions];

        for &(action, visits, prior, q_val) in &child_data {
            let completed_q = if visits > 0 {
                q_val.clamp(-1.0, 1.0)
            } else {
                v_mix.clamp(-1.0, 1.0)
            };
            let log_prior = (prior.max(1e-8)).ln();
            logits[action] = log_prior + sigma_scale * completed_q;
        }

        // Numerically stable softmax over legal actions only
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        if max_logit == f32::NEG_INFINITY {
            return policy;
        }

        let mut sum_exp = 0.0f32;
        for &l in &logits {
            if l > f32::NEG_INFINITY {
                sum_exp += (l - max_logit).exp();
            }
        }
        if sum_exp <= 0.0 {
            return policy;
        }

        for i in 0..n_actions {
            if logits[i] > f32::NEG_INFINITY {
                policy[i] = (logits[i] - max_logit).exp() / sum_exp;
            }
        }

        // Note: policy pruning is applied only in the Python training loop
        // (prune_policy_targets in trainer.py) to avoid double-pruning.

        policy
    }

    /// Returns (child_pool_index, prior) for each root child.
    /// Used by Gumbel MCTS to build the candidate list after root expansion.
    pub fn get_root_children_info(&self) -> Vec<(u32, f32)> {
        let root = &self.pool[0];
        if !root.is_expanded() {
            return Vec::new();
        }
        let first = root.first_child as usize;
        let n_ch = root.n_children as usize;
        (first..first + n_ch)
            .map(|i| (i as u32, self.pool[i].prior))
            .collect()
    }

    pub fn root_visits(&self) -> u32 {
        self.pool[0].n_visits
    }

    pub fn reset(&mut self) {
        let mr = self.pool[0].moves_remaining;
        let board = self.root_board.clone();
        self.new_game(board);
        self.pool[0].moves_remaining = mr;
    }

    // ── Dirichlet noise ───────────────────────────────────────────────────────

    pub fn apply_dirichlet_to_root(&mut self, noise: &[f32], epsilon: f32) {
        let root = &self.pool[0];
        if !root.is_expanded() {
            return;
        }
        let n_ch = root.n_children as usize;
        if noise.len() < n_ch {
            return;
        }
        let first = root.first_child as usize;

        // Snapshot pre-priors for the debug_prior_trace feature. Compiled out
        // in default builds; zero cost.
        #[cfg(feature = "debug_prior_trace")]
        let pre_priors: Vec<f32> = (0..n_ch)
            .map(|j| self.pool[first + j].prior)
            .collect();

        for j in 0..n_ch {
            let child = &mut self.pool[first + j];
            child.prior = (1.0 - epsilon) * child.prior + epsilon * noise[j];
        }

        #[cfg(feature = "debug_prior_trace")]
        {
            let post_priors: Vec<f32> = (0..n_ch)
                .map(|j| self.pool[first + j].prior)
                .collect();
            crate::debug_trace::record_dirichlet(
                epsilon,
                &pre_priors,
                &noise[..n_ch],
                &post_priors,
            );
        }
    }

    /// Top-N children of root by visit count.
    /// Returns Vec<(coord_string, visits, prior, q_value)> sorted by visits descending.
    pub fn get_top_visits(&self, n: usize) -> Vec<(String, u32, f32, f32)> {
        let root = &self.pool[0];
        if !root.is_expanded() {
            return Vec::new();
        }
        let first = root.first_child as usize;
        let n_ch = root.n_children as usize;

        let mut children: Vec<(usize, u32)> = (first..first + n_ch)
            .map(|i| (i, self.pool[i].n_visits))
            .collect();
        children.sort_unstable_by(|a, b| b.1.cmp(&a.1));
        children.truncate(n);

        let q_sign: f32 = if root.moves_remaining == 1 { -1.0 } else { 1.0 };
        children.into_iter().map(|(i, visits)| {
            let node = &self.pool[i];
            let val = node.action_idx;
            let q = (val >> 16) as i32 - 32768;
            let r = (val & 0xFFFF) as i32 - 32768;
            let q_value = if visits > 0 { q_sign * node.w_value / visits as f32 } else { 0.0 };
            (format!("({},{})", q, r), visits, node.prior, q_value)
        }).collect()
    }

    /// Value estimate at root from perspective of player to move.
    /// Returns Q = w_value / n_visits, or 0.0 if no visits.
    pub fn root_value(&self) -> f32 {
        let root = &self.pool[0];
        if root.n_visits == 0 {
            0.0
        } else {
            root.w_value / root.n_visits as f32
        }
    }

    pub fn root_n_children(&self) -> usize {
        if self.pool[0].is_expanded() {
            self.pool[0].n_children as usize
        } else {
            0
        }
    }

    /// Return search statistics accumulated since the last `new_game()`.
    ///
    /// Returns `(mean_depth, root_concentration)`:
    /// - `mean_depth`: average depth descended per simulation (leaf depth)
    /// - `root_concentration`: max child visits / root total visits ∈ [0.0, 1.0]
    ///
    /// Both values are 0.0 when no simulations have been run.
    /// This is a once-per-search aggregation — NOT called from the inner sim loop.
    pub fn last_search_stats(&self) -> (f32, f32) {
        let mean_depth = if self.sim_count > 0 {
            self.depth_accum as f32 / self.sim_count as f32
        } else {
            0.0
        };
        let root_conc = {
            let root = &self.pool[0];
            let total = root.n_visits;
            if total == 0 || !root.is_expanded() {
                0.0
            } else {
                let first = root.first_child as usize;
                let n = root.n_children as usize;
                let max_v = (first..first + n)
                    .map(|i| self.pool[i].n_visits)
                    .max()
                    .unwrap_or(0);
                max_v as f32 / total as f32
            }
        };
        (mean_depth, root_conc)
    }

    /// Run `n` simulations using uniform priors (no NN). For benchmarking.
    pub fn run_simulations_cpu_only(&mut self, n: usize) {
        let uniform_prior = 1.0 / (BOARD_SIZE * BOARD_SIZE + 1) as f32;
        let uniform_policy = vec![uniform_prior; BOARD_SIZE * BOARD_SIZE + 1];
        for _ in 0..n {
            let boards = self.select_leaves(1);
            if boards.is_empty() {
                continue;
            }
            let policies: Vec<Vec<f32>> = (0..boards.len())
                .map(|_| uniform_policy.clone())
                .collect();
            let values: Vec<f32> = vec![0.0; boards.len()];
            self.expand_and_backup(&policies, &values);
        }
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn setup_two_child_tree(c_puct: f32) -> (MCTSTree, u32, u32) {
        let mut tree = MCTSTree::new(c_puct);
        tree.pool[0].moves_remaining = 2;

        let first_child = 1u32;
        tree.next_free = 3;
        tree.pool[0].first_child = first_child;
        tree.pool[0].n_children  = 2;

        let action_a = ((0u32 + 32768) << 16) | (0u32 + 32768);
        let action_b = ((0u32 + 32768) << 16) | (1u32 + 32768);

        tree.pool[1] = Node {
            parent: 0, action_idx: action_a, n_visits: 0, w_value: 0.0,
            prior: 0.7, first_child: u32::MAX, n_children: 0,
            moves_remaining: 1, is_terminal: false, terminal_value: 0.0,
            virtual_loss_count: 0,
        };
        tree.pool[2] = Node {
            parent: 0, action_idx: action_b, n_visits: 0, w_value: 0.0,
            prior: 0.3, first_child: u32::MAX, n_children: 0,
            moves_remaining: 1, is_terminal: false, terminal_value: 0.0,
            virtual_loss_count: 0,
        };
        (tree, 1, 2)
    }

    #[test]
    fn test_puct_prefers_higher_prior_when_unvisited() {
        let (mut tree, child_a, child_b) = setup_two_child_tree(1.5);
        tree.pool[0].n_visits = 1;
        // fpu_value=0.0: both children unvisited, Q=0 for both; prior drives selection.
        let score_a = tree.puct_score(child_a, 0, 1.0, 0.0);
        let score_b = tree.puct_score(child_b, 0, 1.0, 0.0);
        assert!(score_a > score_b,
            "child with prior 0.7 should score higher than 0.3: {score_a:.4} vs {score_b:.4}");
    }

    #[test]
    fn test_puct_visits_reduce_exploration() {
        let (mut tree, child_a, child_b) = setup_two_child_tree(1.5);
        tree.pool[0].n_visits = 100;
        tree.pool[child_a as usize].n_visits = 99;
        tree.pool[child_a as usize].w_value  = 0.0;

        // child_a is visited (n_visits=99), child_b is unvisited → fpu_value applies.
        // With fpu_value=0.0 the unvisited child still looks like Q=0, but has higher U.
        let score_a = tree.puct_score(child_a, 0, 100.0, 0.0);
        let score_b = tree.puct_score(child_b, 0, 100.0, 0.0);
        assert!(score_b > score_a,
            "less-visited child_b should be preferred: {score_b:.4} vs {score_a:.4}");
    }

    #[test]
    fn test_backup_single_value_reaches_root() {
        let mut tree = MCTSTree::new(1.5);
        tree.pool[0].moves_remaining = 1;
        tree.pool[1] = Node {
            parent: 0, action_idx: (32768u32 << 16) | 32768u32, n_visits: 0, w_value: 0.0,
            prior: 1.0, first_child: u32::MAX, n_children: 0,
            moves_remaining: 2, is_terminal: false, terminal_value: 0.0,
            virtual_loss_count: 0,
        };
        tree.pool[2] = Node {
            parent: 1, action_idx: (32768u32 << 16) | 32769u32, n_visits: 0, w_value: 0.0,
            prior: 1.0, first_child: u32::MAX, n_children: 0,
            moves_remaining: 1, is_terminal: false, terminal_value: 0.0,
            virtual_loss_count: 0,
        };
        tree.next_free = 3;

        tree.backup(2, 1.0);

        assert_eq!(tree.pool[2].w_value, 1.0);
        assert_eq!(tree.pool[2].n_visits, 1);
        assert_eq!(tree.pool[1].w_value, 1.0, "child should not flip (mr=2)");
        assert_eq!(tree.pool[1].n_visits, 1);
        assert_eq!(tree.pool[0].w_value, -1.0, "root should flip (mr=1)");
        assert_eq!(tree.pool[0].n_visits, 1);
    }

    #[test]
    fn test_backup_negamax_player_change() {
        let mut tree = MCTSTree::new(1.5);
        tree.pool[0].moves_remaining = 1;
        tree.pool[1] = Node {
            parent: 0, action_idx: (32768u32 << 16) | 32768u32, n_visits: 0, w_value: 0.0,
            prior: 1.0, first_child: u32::MAX, n_children: 0,
            moves_remaining: 2, is_terminal: false, terminal_value: 0.0,
            virtual_loss_count: 0,
        };
        tree.next_free = 2;

        tree.backup(1, 0.6);

        assert!((tree.pool[1].w_value - 0.6).abs() < 1e-6, "child w = 0.6");
        assert!((tree.pool[0].w_value - (-0.6)).abs() < 1e-6,
            "root w should be -0.6, got {}", tree.pool[0].w_value);
    }

    #[test]
    fn test_get_policy_proportional_to_visits() {
        let (mut tree, child_a, child_b) = setup_two_child_tree(1.5);
        tree.pool[0].n_visits = 10;
        tree.pool[child_a as usize].n_visits = 7;
        tree.pool[child_b as usize].n_visits = 3;

        let policy = tree.get_policy(1.0, BOARD_SIZE);
        let pa = policy[180];
        let pb = policy[181];
        assert!((pa - 0.7).abs() < 1e-5, "action 0 should get 70%: {pa}");
        assert!((pb - 0.3).abs() < 1e-5, "action 1 should get 30%: {pb}");
        assert!((pa + pb - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_get_policy_argmax_temperature_zero() {
        let (mut tree, child_a, child_b) = setup_two_child_tree(1.5);
        tree.pool[0].n_visits = 10;
        tree.pool[child_a as usize].n_visits = 7;
        tree.pool[child_b as usize].n_visits = 3;

        let policy = tree.get_policy(0.0, BOARD_SIZE);
        assert_eq!(policy[180], 1.0);
        assert_eq!(policy[181], 0.0);
    }

    #[test]
    fn test_select_leaves_returns_root_when_empty() {
        let mut tree = MCTSTree::new(1.5);
        let board = Board::new();
        tree.new_game(board.clone());

        let leaves = tree.select_leaves(1);
        assert_eq!(leaves.len(), 1);
        assert_eq!(leaves[0].ply, board.ply);
        assert_eq!(leaves[0].moves_remaining, board.moves_remaining);
    }

    #[test]
    fn test_expand_and_backup_creates_children() {
        let mut tree = MCTSTree::new(1.5);
        let board = Board::new();
        tree.new_game(board);

        let leaves = tree.select_leaves(1);
        let n_legal = leaves[0].legal_move_count();

        let policy = vec![1.0 / (BOARD_SIZE * BOARD_SIZE + 1) as f32; BOARD_SIZE * BOARD_SIZE + 1];
        tree.expand_and_backup(&[policy], &[0.0]);

        let root = &tree.pool[0];
        assert!(root.is_expanded());
        assert_eq!(root.n_children as usize, n_legal);
        assert_eq!(root.n_visits, 1);
    }

    #[test]
    fn test_full_search_runs_n_simulations() {
        let mut tree = MCTSTree::new(1.5);
        let board = Board::new();
        tree.new_game(board);

        let n_sims = 20;
        let uniform = vec![1.0 / (BOARD_SIZE * BOARD_SIZE + 1) as f32; BOARD_SIZE * BOARD_SIZE + 1];

        for _ in 0..n_sims {
            let leaves = tree.select_leaves(1);
            let n = leaves.len();
            let policies: Vec<Vec<f32>> = (0..n).map(|_| uniform.clone()).collect();
            let values: Vec<f32> = (0..n).map(|_| 0.0).collect();
            tree.expand_and_backup(&policies, &values);
        }

        assert_eq!(tree.root_visits(), n_sims as u32);
    }

    #[test]
    fn test_policy_sums_to_one_after_search() {
        let mut tree = MCTSTree::new(1.5);
        let board = Board::new();
        tree.new_game(board);

        let n_sims = 10;
        let uniform = vec![1.0 / (BOARD_SIZE * BOARD_SIZE + 1) as f32; BOARD_SIZE * BOARD_SIZE + 1];
        for _ in 0..n_sims {
            let leaves = tree.select_leaves(1);
            let n = leaves.len();
            let policies: Vec<Vec<f32>> = (0..n).map(|_| uniform.clone()).collect();
            let values = vec![0.0f32; n];
            tree.expand_and_backup(&policies, &values);
        }

        let policy = tree.get_policy(1.0, BOARD_SIZE);
        let sum: f32 = policy.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "policy should sum to 1.0, got {sum}");
    }

    #[test]
    fn test_virtual_loss_applied_during_select() {
        let mut tree = MCTSTree::new(1.5);
        tree.new_game(Board::new());
        let _leaves = tree.select_leaves(1);
        assert_eq!(tree.pool[0].virtual_loss_count, 1);
    }

    #[test]
    fn test_virtual_loss_reversed_after_backup() {
        let mut tree = MCTSTree::new(1.5);
        tree.new_game(Board::new());

        let leaves = tree.select_leaves(1);
        let n = leaves.len();
        let uniform = vec![1.0 / (BOARD_SIZE * BOARD_SIZE + 1) as f32; BOARD_SIZE * BOARD_SIZE + 1];
        let policies: Vec<Vec<f32>> = (0..n).map(|_| uniform.clone()).collect();
        tree.expand_and_backup(&policies, &vec![0.0; n]);

        for i in 0..tree.next_free as usize {
            assert_eq!(tree.pool[i].virtual_loss_count, 0,
                "node {i} should have virtual_loss_count=0 after backup");
        }
    }

    #[test]
    fn test_virtual_loss_causes_path_divergence() {
        let (mut tree, child_a, child_b) = setup_two_child_tree(1.5);
        tree.pool[0].n_visits = 1;

        let batch = tree.select_leaves(2);
        assert_eq!(batch.len(), 2);

        let dummy = vec![0.5f32; BOARD_SIZE * BOARD_SIZE + 1];
        let policies: Vec<Vec<f32>> = (0..2).map(|_| dummy.clone()).collect();
        tree.expand_and_backup(&policies, &vec![0.0; 2]);

        assert_eq!(tree.pool[0].virtual_loss_count, 0);
        assert_eq!(tree.pool[child_a as usize].virtual_loss_count, 0);
        assert_eq!(tree.pool[child_b as usize].virtual_loss_count, 0);
    }

    #[test]
    fn test_virtual_loss_q_adjustment() {
        let node = Node {
            parent: u32::MAX, action_idx: (32768u32 << 16) | 32768u32,
            n_visits: 4, w_value: 2.0,
            prior: 0.5, first_child: u32::MAX, n_children: 0,
            moves_remaining: 1, is_terminal: false, terminal_value: 0.0,
            virtual_loss_count: 2,
        };
        let q = node.q_value_vl(VIRTUAL_LOSS_PENALTY, false);
        assert!(q.abs() < 1e-6, "Q should be 0.0: got {q}");
    }

    #[test]
    fn test_dirichlet_ignored_before_root_expanded() {
        // Serialize against test_dirichlet_trace_roundtrip (see debug_trace::TEST_TRACE_LOCK).
        #[cfg(feature = "debug_prior_trace")]
        let _guard = crate::debug_trace::TEST_TRACE_LOCK.lock().unwrap();
        let mut tree = MCTSTree::new(1.5);
        tree.new_game(Board::new());
        tree.apply_dirichlet_to_root(&[0.5, 0.5], 0.25);
        assert_eq!(tree.root_n_children(), 0);
    }

    #[test]
    fn test_dirichlet_mixes_priors_correctly() {
        // Serialize against test_dirichlet_trace_roundtrip (see debug_trace::TEST_TRACE_LOCK).
        #[cfg(feature = "debug_prior_trace")]
        let _guard = crate::debug_trace::TEST_TRACE_LOCK.lock().unwrap();
        let (mut tree, child_a, child_b) = setup_two_child_tree(1.5);
        assert!(tree.pool[0].is_expanded());

        let noise   = [0.9f32, 0.1f32];
        let epsilon = 0.25f32;
        tree.apply_dirichlet_to_root(&noise, epsilon);

        let expected_a = (1.0 - epsilon) * 0.7 + epsilon * 0.9;
        let expected_b = (1.0 - epsilon) * 0.3 + epsilon * 0.1;

        let prior_a = tree.pool[child_a as usize].prior;
        let prior_b = tree.pool[child_b as usize].prior;

        assert!((prior_a - expected_a).abs() < 1e-6,
            "child_a prior: expected {expected_a:.6}, got {prior_a:.6}");
        assert!((prior_b - expected_b).abs() < 1e-6,
            "child_b prior: expected {expected_b:.6}, got {prior_b:.6}");
    }

    /// Verifies the compile-time-gated JSONL trace wrapper around
    /// `apply_dirichlet_to_root` writes exactly one well-formed record.
    /// Only compiled with `--features debug_prior_trace`.
    #[cfg(feature = "debug_prior_trace")]
    #[test]
    fn test_dirichlet_trace_roundtrip() {
        use std::fs;
        use std::io::{BufRead, BufReader};

        // Serialize against other tests that call apply_dirichlet_to_root —
        // they share the same global TRACE_FILE sink, so parallel execution
        // would cause cross-test writes into our JSONL file.
        let _guard = crate::debug_trace::TEST_TRACE_LOCK.lock().unwrap();

        let path = std::env::temp_dir().join(format!(
            "hexo_dbg_trace_{}.jsonl", std::process::id()
        ));
        let _ = fs::remove_file(&path);
        let path_str = path.to_str().expect("tmp path utf8");

        crate::debug_trace::set_sink_for_test(path_str);
        crate::debug_trace::reset_counters_for_test();

        let (mut tree, _, _) = setup_two_child_tree(1.5);
        assert!(tree.pool[0].is_expanded());
        let noise = [0.9f32, 0.1f32];
        tree.apply_dirichlet_to_root(&noise, 0.25);

        // Tear the sink down BEFORE releasing the guard so no subsequent
        // test (from either this file or another) can write into our tmp.
        crate::debug_trace::clear_sink_for_test();

        let file = fs::File::open(&path).expect("trace file should exist");
        let reader = BufReader::new(file);
        let lines: Vec<String> = reader.lines().filter_map(Result::ok).collect();

        assert_eq!(lines.len(), 1, "expected exactly one JSONL record, got {}", lines.len());
        let record = &lines[0];
        assert!(record.contains(r#""site":"apply_dirichlet_to_root""#),
            "record missing site marker: {record}");
        assert!(record.contains(r#""n_children":2"#),
            "record missing n_children=2: {record}");
        assert!(record.contains(r#""epsilon":0.250000"#),
            "record missing epsilon=0.25: {record}");
        assert!(record.contains(r#""noise":[0.900000,0.100000]"#),
            "record missing noise vector: {record}");

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_dynamic_fpu_reduces_unvisited_q() {
        // Dynamic FPU: unvisited children should receive parent_q - reduction*sqrt(mass).
        let (mut tree, child_a, child_b) = setup_two_child_tree(1.5);
        tree.fpu_reduction = 0.25;
        tree.pool[0].n_visits = 10;
        tree.pool[0].w_value  = 3.0; // parent Q = 0.3

        // child_a: visited (n_visits=1), Q from w_value/n_visits.
        tree.pool[child_a as usize].n_visits = 1;
        tree.pool[child_a as usize].w_value  = 0.2;

        // child_b: unvisited → should get fpu_value.
        // explored_mass = prior of child_a = 0.7  → reduction = 0.25*sqrt(0.7) ≈ 0.209
        // fpu_value = parent_q(0.3) - 0.209 ≈ 0.091
        let explored_mass: f32 = 0.7;
        let expected_fpu = (3.0f32 / 10.0) - 0.25 * explored_mass.sqrt();

        // child_b uses fpu_value; child_a (visited) uses its own Q.
        let score_b_fpu = tree.puct_score(child_b, 0, 10.0, expected_fpu);
        let score_b_zero_fpu = tree.puct_score(child_b, 0, 10.0, 0.0);
        // With parent_q > 0, FPU value < 0.3 but > 0.0 — so dynamic FPU raises score
        // compared to the legacy Q=0 baseline when parent_q is positive.
        assert!(
            score_b_fpu > score_b_zero_fpu || expected_fpu < 0.0,
            "dynamic FPU should raise unvisited score when parent_q > 0: \
             fpu_score={score_b_fpu:.4} vs zero_fpu={score_b_zero_fpu:.4} (fpu={expected_fpu:.4})"
        );
    }

    // ── Quiescence tests ──────────────────────────────────────────────────────


    #[test]
    fn test_quiescence_overrides_value_for_3_winning_moves() {
        // Board where P1 (current player to move next, but we'll evaluate from P1's perspective)
        // has ≥3 winning moves → value should be overridden to 1.0.
        let mut tree = MCTSTree::new(1.5);
        tree.quiescence_enabled = true;
        tree.quiescence_blend_2 = 0.3;

        // Build a board where P1 has exactly 3 winning cells:
        // Two from (0,0)..(4,0): q=-1 and q=5
        // One from (20,0)..(24,0): q=19 (west end blocked, east end free)
        let mut board = Board::new();
        for q in 0..5i32 {
            board.cells.insert((q, 0), crate::board::Cell::P1);
        }
        // Block west end of first threat so it has only 1 winning cell
        board.cells.insert((-1, 0), crate::board::Cell::P2);
        // Second threat (unblocked both ends → 2 winning cells: q=19 and q=25)
        for q in 20..25i32 {
            board.cells.insert((q, 0), crate::board::Cell::P1);
        }
        board.has_stones = true;
        board.cache_dirty.set(true);
        board.current_player = crate::board::Player::One;
        // ply must be ≥ 8 so the early-game ply gate does not short-circuit.
        board.ply = 20;

        // Current player is P1; P1 has 1 + 2 = 3 winning cells → forced win.
        let wins = board.count_winning_moves(crate::board::Player::One);
        assert!(wins >= 3, "expected ≥3 winning moves for P1, got {wins}");

        let corrected = tree.apply_quiescence(&board, 0.0);
        assert_eq!(corrected, 1.0,
            "quiescence should override to 1.0 for 3+ winning moves");
    }

    #[test]
    fn test_quiescence_overrides_value_for_3_opponent_winning_moves() {
        let mut tree = MCTSTree::new(1.5);
        tree.quiescence_enabled = true;
        tree.quiescence_blend_2 = 0.3;

        // Current player is P1, but opponent (P2) has 3 winning moves → value = -1.0
        let mut board = Board::new();
        // P2 stones: (0,0)..(4,0) unblocked (2 cells) + (20,0)..(24,0) east-only blocked (1 cell)
        for q in 0..5i32 {
            board.cells.insert((q, 0), crate::board::Cell::P2);
        }
        board.cells.insert((-1, 0), crate::board::Cell::P1); // block west end
        for q in 20..25i32 {
            board.cells.insert((q, 0), crate::board::Cell::P2);
        }
        board.has_stones = true;
        board.cache_dirty.set(true);
        board.current_player = crate::board::Player::One;
        // ply must be ≥ 8 so the early-game ply gate does not short-circuit.
        board.ply = 20;

        let opp_wins = board.count_winning_moves(crate::board::Player::Two);
        assert!(opp_wins >= 3, "expected ≥3 winning moves for P2, got {opp_wins}");

        let corrected = tree.apply_quiescence(&board, 0.0);
        assert_eq!(corrected, -1.0,
            "quiescence should override to -1.0 when opponent has 3+ winning moves");
    }

    #[test]
    fn test_quiescence_blend_for_2_winning_moves() {
        let mut tree = MCTSTree::new(1.5);
        tree.quiescence_enabled = true;
        tree.quiescence_blend_2 = 0.3;

        // P1 has exactly 2 winning moves (unblocked 5-in-a-row along E axis)
        let mut board = Board::new();
        for q in 0..5i32 {
            board.cells.insert((q, 0), crate::board::Cell::P1);
        }
        board.has_stones = true;
        board.cache_dirty.set(true);
        board.current_player = crate::board::Player::One;
        // ply must be ≥ 8 so the early-game ply gate does not short-circuit.
        board.ply = 10;

        let wins = board.count_winning_moves(crate::board::Player::One);
        assert_eq!(wins, 2, "unblocked 5-in-a-row should have exactly 2 winning moves");

        let nn_value = 0.5f32;
        let corrected = tree.apply_quiescence(&board, nn_value);
        let expected = (nn_value + 0.3).min(1.0);
        assert!((corrected - expected).abs() < 1e-6,
            "blend for 2 winning moves: expected {expected}, got {corrected}");
    }

    #[test]
    fn test_quiescence_disabled_does_not_change_value() {
        let mut tree = MCTSTree::new(1.5);
        tree.quiescence_enabled = false;

        let mut board = Board::new();
        // Give P1 a huge number of winning moves
        for q in 0..5i32 {
            board.cells.insert((q, 0), crate::board::Cell::P1);
        }
        board.has_stones = true;
        board.cache_dirty.set(true);

        let nn_value = 0.42f32;
        let corrected = tree.apply_quiescence(&board, nn_value);
        assert_eq!(corrected, nn_value, "disabled quiescence must not change value");
    }

    #[test]
    fn test_quiescence_fire_count_increments_and_resets() {
        let mut tree = MCTSTree::new_full(1.5, 0.0, 0.0);
        tree.quiescence_enabled = true;
        tree.quiescence_blend_2 = 0.3;

        // Board setup: P1 has ≥3 winning moves (same as test_quiescence_overrides_value_for_current_wins).
        let mut board = Board::new();
        for q in 0..5i32 {
            board.cells.insert((q, 0), crate::board::Cell::P1);
        }
        board.cells.insert((-1, 0), crate::board::Cell::P2); // block west end of first threat
        for q in 20..25i32 {
            board.cells.insert((q, 0), crate::board::Cell::P1);
        }
        board.has_stones = true;
        board.cache_dirty.set(true);
        board.current_player = crate::board::Player::One;
        board.ply = 20;

        let wins = board.count_winning_moves(crate::board::Player::One);
        assert!(wins >= 3, "expected ≥3 winning moves for P1, got {wins}");

        // Counter starts at 0.
        assert_eq!(tree.quiescence_fire_count.load(Ordering::Relaxed), 0);

        // First call fires → counter = 1.
        let result = tree.apply_quiescence(&board, 0.5);
        assert_eq!(result, 1.0, "forced win should override to 1.0");
        assert_eq!(tree.quiescence_fire_count.load(Ordering::Relaxed), 1,
            "counter should be 1 after one firing call");

        // Second call fires again → counter = 2.
        tree.apply_quiescence(&board, 0.5);
        assert_eq!(tree.quiescence_fire_count.load(Ordering::Relaxed), 2,
            "counter should accumulate across calls");

        // new_game() resets counter to 0.
        tree.new_game(Board::new());
        assert_eq!(tree.quiescence_fire_count.load(Ordering::Relaxed), 0,
            "counter should reset to 0 after new_game()");
    }

    #[test]
    fn test_quiescence_no_override_in_early_game() {
        let mut tree = MCTSTree::new(1.5);
        tree.quiescence_enabled = true;
        tree.quiescence_blend_2 = 0.3;

        // Early game — no threatening formations.
        let board = Board::new();
        let nn_value = 0.123f32;
        let corrected = tree.apply_quiescence(&board, nn_value);
        assert_eq!(corrected, nn_value, "early game should not trigger quiescence");
    }

    #[test]
    fn test_no_forced_win_short_circuit_in_expansion() {
        use crate::board::Player;

        let mut board = Board::new();
        for r in 0..4 {
            board.cells.insert((0, r), crate::board::Cell::P1);
        }
        board.ply = 7;
        board.current_player = Player::One;

        assert!(!board.check_win(), "test setup: board should not be a terminal win");

        let mut tree = MCTSTree::new(1.5);
        tree.new_game(board);

        let n_actions = BOARD_SIZE * BOARD_SIZE + 1;
        let uniform_policy = vec![1.0 / n_actions as f32; n_actions];
        let nn_value = 0.5;

        tree.expand_and_backup(&[uniform_policy], &[nn_value]);

        let root = &tree.pool[0];
        assert!(!root.is_terminal,
            "root must NOT be marked terminal for a forced-win formation");
    }

    // ── Gumbel MCTS tests ────────────────────────────────────────────────────

    fn setup_expanded_root() -> MCTSTree {
        let mut tree = MCTSTree::new(1.5);
        let board = Board::new();
        tree.new_game(board);

        // Expand root with uniform priors.
        let _leaves = tree.select_leaves(1);
        let n_actions = BOARD_SIZE * BOARD_SIZE + 1;
        let policy = vec![1.0 / n_actions as f32; n_actions];
        tree.expand_and_backup(&[policy], &[0.0]);
        tree
    }

    #[test]
    fn test_get_root_children_info_returns_correct_count() {
        let tree = setup_expanded_root();
        let info = tree.get_root_children_info();
        assert_eq!(info.len(), tree.root_n_children());
        // All priors should be > 0 (from uniform policy over full action space).
        for &(idx, prior) in &info {
            assert!(prior > 0.0, "prior for child {idx} should be > 0");
        }
        assert!(!info.is_empty(), "root should have children after expansion");
    }

    #[test]
    fn test_forced_root_child_selection() {
        let mut tree = setup_expanded_root();
        let root = &tree.pool[0];
        assert!(root.is_expanded());
        let first = root.first_child;
        let n_ch = root.n_children as usize;
        assert!(n_ch >= 2, "need at least 2 children");

        // Force selection to second child.
        let target_child = first + 1;
        tree.forced_root_child = Some(target_child);

        // Run several simulations — all should go through the forced child.
        let n_sims = 10;
        let n_actions = BOARD_SIZE * BOARD_SIZE + 1;
        let uniform = vec![1.0 / n_actions as f32; n_actions];
        for _ in 0..n_sims {
            let leaves = tree.select_leaves(1);
            let n = leaves.len();
            let policies: Vec<Vec<f32>> = (0..n).map(|_| uniform.clone()).collect();
            let values = vec![0.0f32; n];
            tree.expand_and_backup(&policies, &values);
        }

        // The forced child should have gotten all visits (minus root expansion).
        let forced_visits = tree.pool[target_child as usize].n_visits;
        assert!(forced_visits >= n_sims as u32 - 1,
            "forced child should have >= {} visits, got {}", n_sims - 1, forced_visits);

        // First child (not forced) should have 0 visits.
        let other_visits = tree.pool[first as usize].n_visits;
        assert_eq!(other_visits, 0,
            "non-forced child should have 0 visits, got {other_visits}");

        tree.forced_root_child = None;
    }

    #[test]
    fn test_forced_root_none_uses_puct() {
        // With forced_root_child = None, PUCT selects normally.
        let mut tree = setup_expanded_root();
        tree.forced_root_child = None;

        let n_sims = 20;
        let n_actions = BOARD_SIZE * BOARD_SIZE + 1;
        let uniform = vec![1.0 / n_actions as f32; n_actions];
        for _ in 0..n_sims {
            let leaves = tree.select_leaves(1);
            let n = leaves.len();
            let policies: Vec<Vec<f32>> = (0..n).map(|_| uniform.clone()).collect();
            let values = vec![0.0f32; n];
            tree.expand_and_backup(&policies, &values);
        }

        // Multiple children should have visits (PUCT spreads them).
        let first = tree.pool[0].first_child as usize;
        let n_ch = tree.pool[0].n_children as usize;
        let visited_count = (first..first + n_ch)
            .filter(|&i| tree.pool[i].n_visits > 0)
            .count();
        assert!(visited_count >= 2,
            "PUCT should visit multiple children, only {visited_count} visited");
    }

    #[test]
    fn test_gumbel_disabled_no_behavior_change() {
        // When forced_root_child is None, behavior is identical to pre-Gumbel code.
        // Verify by running search twice with same setup and checking same results.
        let run_search = || -> Vec<u32> {
            let mut tree = MCTSTree::new(1.5);
            let board = Board::new();
            tree.new_game(board);
            tree.forced_root_child = None; // explicitly None

            let n_actions = BOARD_SIZE * BOARD_SIZE + 1;
            let uniform = vec![1.0 / n_actions as f32; n_actions];
            for _ in 0..10 {
                let leaves = tree.select_leaves(1);
                let n = leaves.len();
                let policies: Vec<Vec<f32>> = (0..n).map(|_| uniform.clone()).collect();
                let values = vec![0.0f32; n];
                tree.expand_and_backup(&policies, &values);
            }

            // Extract visit counts for root children
            let first = tree.pool[0].first_child as usize;
            let n_ch = tree.pool[0].n_children as usize;
            (first..first + n_ch).map(|i| tree.pool[i].n_visits).collect()
        };

        let visits_a = run_search();
        let visits_b = run_search();
        // With deterministic input (uniform policy, value=0), results should match.
        assert_eq!(visits_a, visits_b,
            "search with forced_root_child=None should be deterministic");
    }

    #[test]
    fn test_nonroot_uses_puct_when_root_forced() {
        // Verify that non-root selection still uses PUCT (spreads visits)
        // even when root selection is forced.
        let mut tree = setup_expanded_root();
        let first_child = tree.pool[0].first_child;
        tree.forced_root_child = Some(first_child);

        let n_actions = BOARD_SIZE * BOARD_SIZE + 1;
        let uniform = vec![1.0 / n_actions as f32; n_actions];

        // Run enough sims to expand the forced child and go deeper.
        for _ in 0..30 {
            let leaves = tree.select_leaves(1);
            let n = leaves.len();
            let policies: Vec<Vec<f32>> = (0..n).map(|_| uniform.clone()).collect();
            let values = vec![0.0f32; n];
            tree.expand_and_backup(&policies, &values);
        }

        // The forced child should now be expanded with multiple children.
        let fc = &tree.pool[first_child as usize];
        if fc.is_expanded() && fc.n_children > 1 {
            // Multiple grandchildren should have visits (PUCT below root).
            let gc_first = fc.first_child as usize;
            let gc_n = fc.n_children as usize;
            let gc_visited = (gc_first..gc_first + gc_n)
                .filter(|&i| tree.pool[i].n_visits > 0)
                .count();
            assert!(gc_visited >= 2,
                "PUCT at non-root should visit multiple grandchildren, got {gc_visited}");
        }
        // If forced child isn't expanded (e.g., terminal), the test still passes.
        tree.forced_root_child = None;
    }

    // ── get_improved_policy tests ────────────────────────────────────────────

    /// Helper: set up a tree with N root children having specified visits, w_values, and priors.
    /// Each child is placed at action (0, j) for j in 0..N.
    fn setup_improved_policy_tree(
        children: &[(u32, f32, f32)], // (visits, w_value, prior)
    ) -> MCTSTree {
        let mut tree = MCTSTree::new(1.5);
        let board = Board::new();
        tree.root_board = board;
        let n = children.len();
        tree.pool[0].first_child = 1;
        tree.pool[0].n_children = n as u16;
        tree.pool[0].moves_remaining = 2;

        let mut total_visits = 0u32;
        let mut total_w = 0.0f32;
        for (j, &(visits, w_value, prior)) in children.iter().enumerate() {
            let q = 0i32;
            let r = j as i32;
            let action_idx = ((q as u32).wrapping_add(32768) << 16) | (r as u32).wrapping_add(32768);
            tree.pool[1 + j] = Node {
                parent: 0,
                action_idx,
                n_visits: visits,
                w_value,
                prior,
                first_child: u32::MAX,
                n_children: 0,
                moves_remaining: 1,
                is_terminal: false,
                terminal_value: 0.0,
                virtual_loss_count: 0,
            };
            total_visits += visits;
            total_w += w_value;
        }
        tree.pool[0].n_visits = total_visits;
        tree.pool[0].w_value = total_w;
        tree.next_free = 1 + n as u32;
        tree
    }

    #[test]
    fn test_improved_policy_sums_to_one() {
        // Three children with different visits and Q values.
        let tree = setup_improved_policy_tree(&[
            (10, 5.0, 0.5),   // Q=0.5
            (8, -2.0, 0.3),   // Q=-0.25
            (2, 0.4, 0.2),    // Q=0.2
        ]);
        let policy = tree.get_improved_policy(BOARD_SIZE, 50.0, 1.0);
        let sum: f32 = policy.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "policy should sum to 1.0, got {sum}");
    }

    #[test]
    fn test_improved_policy_no_visits_returns_prior() {
        // All children unvisited — should return normalized priors.
        let tree = setup_improved_policy_tree(&[
            (0, 0.0, 0.6),
            (0, 0.0, 0.4),
        ]);
        let policy = tree.get_improved_policy(BOARD_SIZE, 50.0, 1.0);
        let sum: f32 = policy.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "prior fallback should sum to 1.0, got {sum}");

        // The two non-zero entries should roughly reflect priors.
        let nonzero: Vec<f32> = policy.iter().copied().filter(|&p| p > 0.0).collect();
        assert_eq!(nonzero.len(), 2);
        assert!(nonzero[0] > nonzero[1], "higher prior should get higher prob");
    }

    #[test]
    fn test_improved_policy_q_ordering() {
        // Two children: one clearly winning (Q=+0.9), one losing (Q=-0.9).
        // Equal priors — the improved policy should favor the winning child.
        let tree = setup_improved_policy_tree(&[
            (50, 45.0, 0.5),   // Q=+0.9
            (50, -45.0, 0.5),  // Q=-0.9
        ]);
        let policy = tree.get_improved_policy(BOARD_SIZE, 50.0, 1.0);

        // Find the two non-zero actions.
        let (cq, cr) = tree.root_board.window_center();
        let idx_good = Board::window_flat_idx_at(0, 0, cq, cr);
        let idx_bad = Board::window_flat_idx_at(0, 1, cq, cr);

        assert!(policy[idx_good] > policy[idx_bad],
            "Q=+0.9 child should get more probability than Q=-0.9: {} vs {}",
            policy[idx_good], policy[idx_bad]);
    }

    // Note: policy_prune_frac test removed — pruning now lives only in
    // Python's prune_policy_targets (trainer.py) to avoid double-pruning.

    #[test]
    fn test_last_search_stats_bounds_after_sims() {
        let mut tree = MCTSTree::new(1.5);
        tree.new_game(Board::new());

        let n_sims = 10;
        let n_actions = BOARD_SIZE * BOARD_SIZE + 1;
        let uniform = vec![1.0 / n_actions as f32; n_actions];
        for _ in 0..n_sims {
            let leaves = tree.select_leaves(1);
            let n = leaves.len();
            let policies: Vec<Vec<f32>> = (0..n).map(|_| uniform.clone()).collect();
            let values = vec![0.0f32; n];
            tree.expand_and_backup(&policies, &values);
        }

        let (mean_depth, root_concentration) = tree.last_search_stats();
        assert!(mean_depth >= 0.0,
            "mean_depth must be >= 0.0, got {mean_depth}");
        assert!(root_concentration >= 0.0 && root_concentration <= 1.0,
            "root_concentration must be in [0.0, 1.0], got {root_concentration}");
    }

    #[test]
    fn test_improved_policy_illegal_actions_stay_zero() {
        let tree = setup_improved_policy_tree(&[
            (10, 5.0, 0.7),
            (5, 1.0, 0.3),
        ]);
        let policy = tree.get_improved_policy(BOARD_SIZE, 50.0, 1.0);

        // Only 2 actions should be non-zero out of board_size*board_size+1.
        let nonzero_count = policy.iter().filter(|&&p| p > 0.0).count();
        assert_eq!(nonzero_count, 2, "only legal actions should have non-zero prob, got {nonzero_count}");
    }
}
