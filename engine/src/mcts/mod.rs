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

use crate::board::{Board, BOARD_SIZE};
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
    /// §P6+§P9: pending leaves carry the fully-replayed leaf `Board` itself
    /// (zobrist + ply state captured at `select_one_leaf` exit), eliminating
    /// the per-leaf `root_board.clone() + N × apply_move` re-walk previously
    /// done inside `expand_and_backup`. MoveDiff Vec retired (its
    /// prev_zobrist/bbox/anchors fields were never consumed downstream).
    pub(crate) pending: Vec<(u32, Board)>,
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
    ///
    /// §P2 — n_actions sourced from root_board's encoding spec when present;
    /// audit: legacy-v6-fallback when encoding is None (bench harness path
    /// has no spec wired today). Pre-P2 unconditionally used `BOARD_SIZE² + 1`
    /// which would phantom-pass-slot a v8 board if a future bench wraps one.
    ///
    /// Cycle 3 Wave 8 Batch C (FF.10): the parallel fallback arms in
    /// `engine/src/game_runner/mod.rs` and `engine/src/inference_bridge.rs`
    /// retired to `PyValueError` so production callers must supply
    /// `encoding_name`. This bench-harness arm is the sole survivor and is
    /// **comment-and-keep** by operator pre-decision — `MCTSTree::new()` /
    /// `Board::new()` produces a tree with no `encoding_spec`, which is the
    /// canonical bench-harness construction shape. Migrating the bench
    /// harness to thread an explicit name would expand scope past the FF.10
    /// anchor with no production-side benefit.
    pub fn run_simulations_cpu_only(&mut self, n: usize) {
        let n_actions = match self.root_board.encoding_spec() {
            Some(spec) => spec.policy_stride(),
            // audit: bench-harness-only
            //
            // Bench harness (`MCTSTree::new()` + `Board::new()`) constructs
            // trees with no `encoding_spec`. Retained as the v6 fallback so
            // the bench keeps compiling against multiple registry versions.
            // Production callers — `SelfPlayRunner::new` /
            // `InferenceBatcher::new` — now `PyValueError` on missing
            // `encoding_name`, so this arm is unreachable from production
            // paths.
            None => BOARD_SIZE * BOARD_SIZE + 1, // audit: bench-harness-only
        };
        let uniform_prior = 1.0 / n_actions as f32;
        let uniform_policy = vec![uniform_prior; n_actions];
        // §P38: bench-fidelity — hoist policies/values slot vecs ONCE outside the
        // outer loop and resize per iteration. select_leaves(1) returns at most
        // one board, so capacity 1 suffices. Production self-play does NOT enter
        // this path — worker_loop uses a different MCTS entry. The change removes
        // bench-loop allocation noise so the MCTS sim/s metric better reflects
        // algorithm cost. Bit-equivalent algorithm.
        let mut policies: Vec<Vec<f32>> = Vec::with_capacity(1);
        let mut values: Vec<f32> = Vec::with_capacity(1);
        for _ in 0..n {
            let boards = self.select_leaves(1);
            if boards.is_empty() {
                continue;
            }
            policies.clear();
            for _ in 0..boards.len() {
                policies.push(uniform_policy.clone());
            }
            values.clear();
            values.resize(boards.len(), 0.0);
            self.expand_and_backup(&policies, &values);
        }
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests;
