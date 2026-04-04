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

pub mod node;
mod selection;
mod backup;

pub use node::{Node, TTEntry, MAX_NODES, VIRTUAL_LOSS_PENALTY};

use crate::board::{Board, MoveDiff, BOARD_SIZE};
use fxhash::FxHashMap;

// ── Tree ─────────────────────────────────────────────────────────────────────

pub struct MCTSTree {
    pub(crate) pool: Vec<Node>,
    pub(crate) next_free: u32,
    pub(crate) root_board: Board,
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
}

impl MCTSTree {
    pub fn new(c_puct: f32) -> Self {
        MCTSTree::new_with_vl(c_puct, VIRTUAL_LOSS_PENALTY)
    }

    pub fn new_with_vl(c_puct: f32, virtual_loss: f32) -> Self {
        MCTSTree::new_full(c_puct, virtual_loss, 0.0)
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
            pending: Vec::new(),
            transposition_table: FxHashMap::default(),
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
                let q = (val >> 8) as i32 - 128;
                let r = (val & 0xFF) as i32 - 128;
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
                    let q = (val >> 8) as i32 - 128;
                    let r = (val & 0xFF) as i32 - 128;
                    let action = self.root_board.window_flat_idx(q, r);

                    if action < n_actions {
                        policy[action] = v / total;
                    }
                }
            }
        }

        policy
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
        for j in 0..n_ch {
            let child = &mut self.pool[first + j];
            child.prior = (1.0 - epsilon) * child.prior + epsilon * noise[j];
        }
    }

    /// Top-N children of root by visit count.
    /// Returns Vec<(coord_string, visits, prior)> sorted by visits descending.
    pub fn get_top_visits(&self, n: usize) -> Vec<(String, u32, f32)> {
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

        children.into_iter().map(|(i, visits)| {
            let val = self.pool[i].action_idx;
            let q = (val >> 8) as i32 - 128;
            let r = (val & 0xFF) as i32 - 128;
            (format!("({},{})", q, r), visits, self.pool[i].prior)
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

        let action_a = ((0 + 128) << 8) | (0 + 128);
        let action_b = ((0 + 128) << 8) | (1 + 128);

        tree.pool[1] = Node {
            parent: 0, action_idx: action_a as u16, n_visits: 0, w_value: 0.0,
            prior: 0.7, first_child: u32::MAX, n_children: 0,
            moves_remaining: 1, is_terminal: false, terminal_value: 0.0,
            virtual_loss_count: 0,
        };
        tree.pool[2] = Node {
            parent: 0, action_idx: action_b as u16, n_visits: 0, w_value: 0.0,
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
            parent: 0, action_idx: ((128 << 8) | 128) as u16, n_visits: 0, w_value: 0.0,
            prior: 1.0, first_child: u32::MAX, n_children: 0,
            moves_remaining: 2, is_terminal: false, terminal_value: 0.0,
            virtual_loss_count: 0,
        };
        tree.pool[2] = Node {
            parent: 1, action_idx: ((128 << 8) | 129) as u16, n_visits: 0, w_value: 0.0,
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
            parent: 0, action_idx: ((128 << 8) | 128) as u16, n_visits: 0, w_value: 0.0,
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
            parent: u32::MAX, action_idx: ((128 << 8) | 128) as u16,
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
        let mut tree = MCTSTree::new(1.5);
        tree.new_game(Board::new());
        tree.apply_dirichlet_to_root(&[0.5, 0.5], 0.25);
        assert_eq!(tree.root_n_children(), 0);
    }

    #[test]
    fn test_dirichlet_mixes_priors_correctly() {
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
}
