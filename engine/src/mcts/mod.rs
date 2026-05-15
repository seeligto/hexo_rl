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
pub mod policy;

pub use node::{Node, TTEntry, MAX_NODES, VIRTUAL_LOSS_PENALTY};
pub use backup::{pool_overflow_count, take_pool_overflow_count};

/// Maximum children created per leaf expansion.
///
/// Caps `expand_and_backup_single`: when the position has more than this many
/// legal moves, only the top-K by NN policy prior are expanded (tie-break by
/// `window_flat_idx` for determinism). Fewer-than-K positions take a fast
/// path with no sort.
///
/// Bound: pool nodes consumed per search ≈ n_simulations × leaf_batch × K.
/// At n_sims=400, leaf_batch=8, K=192 → ~614k slots, fits MAX_NODES=1M with
/// headroom for transposition-table re-expansions and root re-rooting (Q40).
///
/// Captures wide-board exploration during early training (legal_moves can
/// balloon past 1k cells once the board has 100+ stones spread out). Can drop
/// to 128 post-training-stabilisation if threat-probe shows no regression.
///
/// Q40 (subtree reuse) interaction: K is per-node, not per-tree. Children are
/// stable identity across re-roots since the chosen top-K set is determined
/// by local policy + flat_idx, both invariant under root rotation.
pub const MAX_CHILDREN_PER_NODE: usize = 192;

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
    /// Convenience constructor used by benches + tests; production hot path
    /// constructs via `new_full` directly (see `SelfPlayRunner` worker init).
    pub fn new(c_puct: f32) -> Self {
        MCTSTree::new_full(c_puct, VIRTUAL_LOSS_PENALTY, 0.0)
    }

    pub fn new_full(c_puct: f32, virtual_loss: f32, fpu_reduction: f32) -> Self {
        let pool = vec![Node::uninit(); MAX_NODES];
        MCTSTree {
            pool,
            next_free: 1,
            root_board: Board::new(),
            c_puct,
            virtual_loss,
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

    pub fn root_visits(&self) -> u32 {
        self.pool[0].n_visits
    }

    /// First unallocated pool slot. Useful for tests/audits that want to
    /// scan only the live portion of the node pool without iterating
    /// `MAX_NODES` zeroed slots.
    pub fn next_free_slot(&self) -> u32 {
        self.next_free
    }

    pub fn reset(&mut self) {
        let mr = self.pool[0].moves_remaining;
        let board = self.root_board.clone();
        self.new_game(board);
        self.pool[0].moves_remaining = mr;
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

    pub(super) fn setup_two_child_tree(c_puct: f32) -> (MCTSTree, u32, u32) {
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
        let q = node.q_value_vl(VIRTUAL_LOSS_PENALTY);
        assert!(q.abs() < 1e-6, "Q should be 0.0: got {q}");
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

    pub(super) fn setup_expanded_root() -> MCTSTree {
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

    // ── Top-K leaf cap tests ─────────────────────────────────────────────────

    #[test]
    fn test_topk_truncates_at_max_children() {
        use fxhash::FxHashSet;
        use crate::board::HALF;
        use super::backup::pick_topk_children;

        // 600 unique cells split between 200 in-window (high priors) and
        // 400 out-of-window (sort prior 0.0). Top K will be drawn from the
        // 200 in-window cells since out-of-window sinks under sort.
        let mut cells: FxHashSet<(i32, i32)> = FxHashSet::default();
        'iw: for q in -HALF..=HALF {
            for r in -HALF..=HALF {
                cells.insert((q, r));
                if cells.len() == 200 { break 'iw; }
            }
        }
        'ow: for q in 30..=60 {
            for r in 30..=60 {
                cells.insert((q, r));
                if cells.len() == 600 { break 'ow; }
            }
        }
        assert_eq!(cells.len(), 600, "test setup must produce 600 cells");

        // Strictly increasing prior with flat_idx → unique priors for
        // every in-window cell.
        let n_actions = BOARD_SIZE * BOARD_SIZE + 1;
        let policy: Vec<f32> =
            (0..n_actions).map(|i| (i + 1) as f32 / n_actions as f32).collect();

        let (chosen, sort_used) = pick_topk_children(&cells, 0, 0, &policy, BOARD_SIZE as i32, HALF);
        assert!(sort_used, "600 > K must take sort path");
        assert_eq!(chosen.len(), MAX_CHILDREN_PER_NODE,
            "chosen must equal K, got {}", chosen.len());

        // Top K should all be in-window since out-of-window sort_prior=0.0
        // and 200 in-window cells with policy > 0 dominate.
        for &((q, r), prior) in &chosen {
            let flat = Board::window_flat_idx_at(q, r, 0, 0);
            assert!(flat < n_actions,
                "top-K must be drawn from in-window cells, got flat={flat} for ({q},{r})");
            assert!(prior > 0.0,
                "in-window cell prior must be >0 for this fixture, got {prior}");
        }

        // With all-in-window selection and priors monotonic in flat, the
        // chosen Vec's priors must be non-increasing.
        for w in chosen.windows(2) {
            assert!(w[0].1 >= w[1].1,
                "priors must be non-increasing: {} then {}", w[0].1, w[1].1);
        }
    }

    #[test]
    fn test_topk_tie_break_by_flat_idx() {
        use fxhash::FxHashSet;
        use crate::board::HALF;
        use super::backup::pick_topk_children;

        // K + 1 cells inside window with identical priors → exactly one is
        // dropped. Tie-break = flat_idx asc, so the cell with the largest
        // flat_idx is the one dropped.
        let target = MAX_CHILDREN_PER_NODE + 1;
        let mut cells: FxHashSet<(i32, i32)> = FxHashSet::default();
        let mut flats_inserted: Vec<usize> = Vec::new();
        'outer: for q in -HALF..=HALF {
            for r in -HALF..=HALF {
                let flat = Board::window_flat_idx_at(q, r, 0, 0);
                cells.insert((q, r));
                flats_inserted.push(flat);
                if cells.len() == target { break 'outer; }
            }
        }
        assert_eq!(cells.len(), target);

        let n_actions = BOARD_SIZE * BOARD_SIZE + 1;
        let uniform_high = vec![0.5_f32; n_actions];

        let (chosen, sort_used) = pick_topk_children(&cells, 0, 0, &uniform_high, BOARD_SIZE as i32, HALF);
        assert!(sort_used);
        assert_eq!(chosen.len(), MAX_CHILDREN_PER_NODE);

        let chosen_flats: std::collections::HashSet<usize> = chosen
            .iter()
            .map(|&((q, r), _)| Board::window_flat_idx_at(q, r, 0, 0))
            .collect();

        let max_flat = *flats_inserted.iter().max().unwrap();
        assert!(!chosen_flats.contains(&max_flat),
            "highest flat_idx must be the dropped cell under tie (max_flat={max_flat})");
        assert_eq!(chosen_flats.len(), MAX_CHILDREN_PER_NODE);
    }

    #[test]
    fn test_topk_fast_path_keeps_all_when_under_cap() {
        use fxhash::FxHashSet;
        use crate::board::HALF;
        use super::backup::pick_topk_children;

        // 50 cells, K=192 → fast path; all cells must appear in the output
        // and `sort_used` is false.
        let mut cells: FxHashSet<(i32, i32)> = FxHashSet::default();
        'outer: for q in -3..=4 {
            for r in -3..=4 {
                cells.insert((q, r));
                if cells.len() == 50 { break 'outer; }
            }
        }
        assert_eq!(cells.len(), 50);

        let n_actions = BOARD_SIZE * BOARD_SIZE + 1;
        let policy = vec![1.0 / n_actions as f32; n_actions];

        let (chosen, sort_used) = pick_topk_children(&cells, 0, 0, &policy, BOARD_SIZE as i32, HALF);
        assert!(!sort_used, "fast path expected when n_legal <= K");
        assert_eq!(chosen.len(), 50);

        let chosen_set: std::collections::HashSet<(i32, i32)> =
            chosen.iter().map(|&(coord, _)| coord).collect();
        assert_eq!(chosen_set, cells.iter().copied().collect(),
            "fast path must include every legal move");
    }
}
