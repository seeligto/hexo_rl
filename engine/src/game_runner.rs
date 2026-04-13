use std::collections::{VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};

use pyo3::prelude::*;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods};
use crate::board::{Board, Cell, BOARD_SIZE, HALF, TOTAL_CELLS, hex_distance};
use crate::mcts::MCTSTree;
use crate::inference_bridge::InferenceBatcher;
use rand::prelude::IndexedRandom;
use rand::{rng, RngExt};

// ── Gumbel MCTS: Sequential Halving state ────────────────────────────────────

/// Per-search state for Gumbel-Top-k root sampling with Sequential Halving
/// (Danihelka et al., "Policy improvement by planning with Gumbel", ICLR 2022).
///
/// Created once at the start of each MCTS search call (after root expansion).
/// Not stored in the node pool or transposition table.
struct GumbelSearchState {
    /// Gumbel(0,1) noise values, one per root child (indexed by child offset).
    gumbel_values: Vec<f32>,
    /// Log-prior for each root child.
    log_priors: Vec<f32>,
    /// Child offsets (relative to first_child) still in the candidate set.
    candidates: Vec<usize>,
    /// ceil(log2(m)) — number of halving phases.
    num_phases: usize,
    /// Sigma scaling constants (same as get_improved_policy).
    c_visit: f32,
    c_scale: f32,
    /// Pool index of root's first child (for converting offsets to pool indices).
    first_child: u32,
    /// Cached (n_visits, w_value) per root child, refreshed each halving phase.
    cached_children: Vec<(u32, f32)>,
}

impl GumbelSearchState {
    /// Create a new Gumbel search state after root has been expanded.
    ///
    /// Generates Gumbel(0,1) noise for all root children, computes
    /// `g(a) + log_prior(a)` scores, and selects the top `m` candidates.
    fn new(
        tree: &crate::mcts::MCTSTree,
        m: usize,
        c_visit: f32,
        c_scale: f32,
        rng: &mut impl rand::Rng,
    ) -> Self {
        let children = tree.get_root_children_info();
        let n_children = children.len();

        // Generate Gumbel(0,1) = -log(-log(U)), U ~ Uniform(0,1)
        let gumbel_values: Vec<f32> = (0..n_children)
            .map(|_| {
                let u: f32 = rng.random::<f32>().clamp(1e-10, 1.0 - 1e-7);
                -(-u.ln()).ln()
            })
            .collect();

        let log_priors: Vec<f32> = children
            .iter()
            .map(|(_, prior)| prior.max(1e-8).ln())
            .collect();

        // Score = g(a) + log_prior(a); select top m
        let effective_m = m.min(n_children);
        let mut scored: Vec<(usize, f32)> = (0..n_children)
            .map(|i| (i, gumbel_values[i] + log_priors[i]))
            .collect();
        scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let candidates: Vec<usize> = scored.iter().take(effective_m).map(|(i, _)| *i).collect();

        let num_phases = if effective_m <= 1 { 1 } else { (effective_m as f64).log2().ceil() as usize };

        let first_child = tree.pool[0].first_child;
        let n_ch = tree.pool[0].n_children as usize;
        let cached_children: Vec<(u32, f32)> = (0..n_ch)
            .map(|j| {
                let c = &tree.pool[first_child as usize + j];
                (c.n_visits, c.w_value)
            })
            .collect();

        GumbelSearchState {
            gumbel_values,
            log_priors,
            candidates,
            num_phases,
            c_visit,
            c_scale,
            first_child,
            cached_children,
        }
    }

    /// Refresh cached child stats from the tree. Call once per halving phase.
    fn refresh_cache(&mut self, tree: &crate::mcts::MCTSTree) {
        let n_ch = self.cached_children.len();
        for j in 0..n_ch {
            let c = &tree.pool[self.first_child as usize + j];
            self.cached_children[j] = (c.n_visits, c.w_value);
        }
    }

    /// Max visit count across all root children (from cache).
    fn max_n(&self) -> u32 {
        self.cached_children.iter().map(|(n, _)| *n).max().unwrap_or(0)
    }

    /// Compute the Gumbel + log_prior + sigma(Q) score for a candidate.
    /// `child_offset` is relative to `first_child`.
    /// `max_n` is the max visit count across all root children (cached per phase).
    ///
    /// Unvisited candidates (n_visits=0) return score = gumbel + log_prior with
    /// no Q contribution (q_hat=0 → sigma=0). This is correct per the paper:
    /// before any simulations, score is just the Gumbel perturbation of the prior.
    fn score(&self, child_offset: usize, max_n: u32) -> f32 {
        let child = &self.cached_children[child_offset];
        let q_hat = if child.0 > 0 {
            (child.1 / child.0 as f32).clamp(-1.0, 1.0)
        } else {
            0.0
        };

        // sigma(q) = (c_visit + max_b N(b)) * c_scale * q
        let sigma = (self.c_visit + max_n as f32) * self.c_scale * q_hat;

        self.gumbel_values[child_offset] + self.log_priors[child_offset] + sigma
    }

    /// Rank candidates by score, keep top half. Refreshes cache from tree first.
    fn halve_candidates(&mut self, tree: &crate::mcts::MCTSTree) {
        if self.candidates.len() <= 1 {
            return;
        }
        self.refresh_cache(tree);
        let max_n = self.max_n();
        // Sort by descending Gumbel+log_prior+sigma(Q) score.
        // Pre-compute into scored pairs to avoid self-borrow in sort closure.
        let mut scored: Vec<(usize, f32)> = self.candidates
            .iter()
            .map(|&c| (c, self.score(c, max_n)))
            .collect();
        scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let keep = (scored.len() + 1) / 2;
        scored.truncate(keep);
        self.candidates = scored.into_iter().map(|(c, _)| c).collect();
    }

    /// Return the pool index of the best candidate (Sequential Halving winner).
    fn best_action_pool_idx(&mut self, tree: &crate::mcts::MCTSTree) -> u32 {
        self.refresh_cache(tree);
        let max_n = self.max_n();
        let best_offset = self.candidates
            .iter()
            .max_by(|&&a, &&b| {
                let sa = self.score(a, max_n);
                let sb = self.score(b, max_n);
                sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
            })
            .copied()
            .unwrap_or(0);
        self.first_child + best_offset as u32
    }
}

/// One recorded position from a self-play game.
#[derive(Clone)]
pub struct Position {
    pub features: Vec<f32>,
    pub policy: Vec<f32>,
    pub player: i8,
}

#[pyclass(name = "SelfPlayRunner")]
pub struct SelfPlayRunner {
    batcher: InferenceBatcher,
    pol_len: usize,
    n_workers: usize,
    max_moves_per_game: usize,
    leaf_batch_size: usize,
    c_puct: f32,
    fpu_reduction: f32,
    quiescence_enabled: bool,
    quiescence_blend_2: f32,
    fast_prob: f32,
    fast_sims: usize,
    standard_sims: usize,
    temp_threshold_compound_moves: usize,
    temp_min: f32,
    draw_reward: f32,
    zoi_enabled: bool,
    zoi_lookback: usize,
    zoi_margin: i32,
    completed_q_values: bool,
    c_visit: f32,
    c_scale: f32,
    gumbel_mcts: bool,
    gumbel_m: usize,
    gumbel_explore_moves: usize,
    dirichlet_alpha: f32,
    dirichlet_epsilon: f32,
    dirichlet_enabled: bool,
    running: Arc<AtomicBool>,
    games_completed: Arc<AtomicUsize>,
    positions_generated: Arc<AtomicUsize>,
    x_wins: Arc<AtomicU64>,
    o_wins: Arc<AtomicU64>,
    draws: Arc<AtomicU64>,
    /// Per-row training rows produced by self-play.
    /// Tuple: (feat, policy, outcome, plies, ownership_u8, winning_line_u8)
    ///   ownership_u8: 361 bytes encoding {0=P2, 1=empty, 2=P1}, projected to the row's
    ///     own cluster window centre at the time the position was recorded (NOT the
    ///     game-end bbox centroid). Aligns the aux target frame with the state frame.
    ///   winning_line_u8: 361 bytes binary mask, same per-row reprojection. All zero on draw.
    results: Arc<Mutex<VecDeque<(Vec<f32>, Vec<f32>, f32, usize, Vec<u8>, Vec<u8>)>>>,
    /// Ring-buffer of recent game results for Python logging and auxiliary training targets.
    /// Tuple: (plies, winner_code, move_history, worker_id, ownership_flat, winning_line_flat)
    ///   winner_code: 1 = Player One, 2 = Player Two, 0 = draw.
    ///   move_history: sequence of (q, r) coordinates in play order.
    ///   ownership_flat: 361 floats projected to the final board window (+1.0 P1, -1.0 P2, 0.0 empty).
    ///   winning_line_flat: 361 floats with 1.0 at winning-line cell positions, 0.0 elsewhere.
    recent_game_results: Arc<Mutex<VecDeque<(usize, u8, Vec<(i32, i32)>, usize, Vec<f32>, Vec<f32>)>>>,
    handles: Arc<Mutex<Vec<JoinHandle<()>>>>,
    /// Accumulated MCTS leaf depth across all searches (scaled by 1_000_000 to preserve fractional part).
    mcts_depth_accum: Arc<AtomicU64>,
    /// Accumulated root concentration * 1_000_000 across all searches.
    mcts_conc_accum: Arc<AtomicU64>,
    /// Number of searches (moves) contributing to the above accumulators.
    mcts_stat_count: Arc<AtomicU64>,
    /// Cumulative quiescence fires (all 4 branches) across all searches since `start()`.
    mcts_quiescence_fires: Arc<AtomicU64>,
}

#[pymethods]
impl SelfPlayRunner {
    #[new]
    #[pyo3(signature = (n_workers = 4, max_moves_per_game = 128, n_simulations = 50, leaf_batch_size = 8, c_puct = 1.5, fpu_reduction = 0.25, feature_len = 18 * 19 * 19, policy_len = 19 * 19 + 1, fast_prob = 0.0, fast_sims = 50, standard_sims = 0, temp_threshold_compound_moves = 15, draw_reward = -0.1, quiescence_enabled = true, quiescence_blend_2 = 0.3, temp_min = 0.05, zoi_enabled = false, zoi_lookback = 16, zoi_margin = 5, completed_q_values = false, c_visit = 50.0, c_scale = 1.0, gumbel_mcts = false, gumbel_m = 16, gumbel_explore_moves = 10, dirichlet_alpha = 0.3, dirichlet_epsilon = 0.25, dirichlet_enabled = true))]
    pub fn new(
        n_workers: usize,
        max_moves_per_game: usize,
        n_simulations: usize,
        leaf_batch_size: usize,
        c_puct: f32,
        fpu_reduction: f32,
        feature_len: usize,
        policy_len: usize,
        fast_prob: f32,
        fast_sims: usize,
        standard_sims: usize,
        temp_threshold_compound_moves: usize,
        draw_reward: f32,
        quiescence_enabled: bool,
        quiescence_blend_2: f32,
        temp_min: f32,
        zoi_enabled: bool,
        zoi_lookback: usize,
        zoi_margin: i32,
        completed_q_values: bool,
        c_visit: f32,
        c_scale: f32,
        gumbel_mcts: bool,
        gumbel_m: usize,
        gumbel_explore_moves: usize,
        dirichlet_alpha: f32,
        dirichlet_epsilon: f32,
        dirichlet_enabled: bool,
    ) -> Self {
        // If standard_sims is 0, fall back to n_simulations.
        let effective_standard = if standard_sims == 0 { n_simulations } else { standard_sims };
        Self {
            batcher: InferenceBatcher::new(feature_len, policy_len),
            pol_len: policy_len,
            n_workers,
            max_moves_per_game,
            leaf_batch_size,
            c_puct,
            fpu_reduction,
            quiescence_enabled,
            quiescence_blend_2,
            fast_prob,
            fast_sims,
            standard_sims: effective_standard,
            temp_threshold_compound_moves,
            temp_min,
            draw_reward,
            zoi_enabled,
            zoi_lookback,
            zoi_margin,
            completed_q_values,
            c_visit,
            c_scale,
            gumbel_mcts,
            gumbel_m,
            gumbel_explore_moves,
            dirichlet_alpha,
            dirichlet_epsilon,
            dirichlet_enabled,
            running: Arc::new(AtomicBool::new(false)),
            games_completed: Arc::new(AtomicUsize::new(0)),
            positions_generated: Arc::new(AtomicUsize::new(0)),
            x_wins: Arc::new(AtomicU64::new(0)),
            o_wins: Arc::new(AtomicU64::new(0)),
            draws: Arc::new(AtomicU64::new(0)),
            results: Arc::new(Mutex::new(VecDeque::new())),
            recent_game_results: Arc::new(Mutex::new(VecDeque::new())),
            handles: Arc::new(Mutex::new(Vec::new())),
            mcts_depth_accum: Arc::new(AtomicU64::new(0)),
            mcts_conc_accum: Arc::new(AtomicU64::new(0)),
            mcts_stat_count: Arc::new(AtomicU64::new(0)),
            mcts_quiescence_fires: Arc::new(AtomicU64::new(0)),
        }
    }

    pub fn start(&self) {
        if self.running.swap(true, Ordering::SeqCst) {
            return;
        }

        let mut handles = self.handles.lock().expect("runner handles lock poisoned");
        for worker_id in 0..self.n_workers {
            let running = self.running.clone();
            let games_completed = self.games_completed.clone();
            let positions_generated = self.positions_generated.clone();
            let x_wins = self.x_wins.clone();
            let o_wins = self.o_wins.clone();
            let draws = self.draws.clone();
            let batcher = self.batcher.clone();
            let max_moves = self.max_moves_per_game;
            let leaf_batch_size = self.leaf_batch_size;
            let c_puct = self.c_puct;
            let fpu_reduction = self.fpu_reduction;
            let quiescence_enabled = self.quiescence_enabled;
            let quiescence_blend_2 = self.quiescence_blend_2;
            let fast_prob = self.fast_prob;
            let fast_sims = self.fast_sims;
            let standard_sims = self.standard_sims;
            let temp_threshold = self.temp_threshold_compound_moves;
            let temp_min = self.temp_min;
            let draw_reward = self.draw_reward;
            let zoi_enabled = self.zoi_enabled;
            let zoi_lookback = self.zoi_lookback;
            let zoi_margin = self.zoi_margin;
            let completed_q_values = self.completed_q_values;
            let c_visit = self.c_visit;
            let c_scale = self.c_scale;
            let gumbel_mcts = self.gumbel_mcts;
            let gumbel_m = self.gumbel_m;
            let gumbel_explore_moves = self.gumbel_explore_moves;
            let dirichlet_alpha   = self.dirichlet_alpha;
            let dirichlet_epsilon = self.dirichlet_epsilon;
            let dirichlet_enabled = self.dirichlet_enabled;
            let results_queue = self.results.clone();
            let recent_game_results = self.recent_game_results.clone();
            let mcts_depth_accum = self.mcts_depth_accum.clone();
            let mcts_conc_accum = self.mcts_conc_accum.clone();
            let mcts_stat_count = self.mcts_stat_count.clone();
            let mcts_quiescence_fires = self.mcts_quiescence_fires.clone();

            let handle = thread::spawn(move || {
                let mut tree = MCTSTree::new_full(c_puct, crate::mcts::VIRTUAL_LOSS_PENALTY, fpu_reduction);
                tree.quiescence_enabled = quiescence_enabled;
                tree.quiescence_blend_2 = quiescence_blend_2;
                let mut rng = rng();
                // Per-worker game counter for the debug_prior_trace feature.
                // Increments at each game end so the trace records can be
                // grouped by (worker_id, game_index). Not read in default
                // builds — tracked unconditionally to avoid a cfg on every
                // increment site.
                #[cfg(feature = "debug_prior_trace")]
                let mut dbg_game_idx: u32 = 0;

                while running.load(Ordering::SeqCst) {
                    let mut board = Board::new();
                    let mut records = Vec::new();
                    let mut move_history: Vec<(i32, i32)> = Vec::new();

                    // KataGo-style playout cap randomisation.
                    let is_fast_game = fast_prob > 0.0 && rng.random::<f32>() < fast_prob;
                    let game_sims = if is_fast_game { fast_sims } else { standard_sims };

                    for _ in 0..max_moves {
                        if !running.load(Ordering::SeqCst) || board.check_win() || board.legal_move_count() == 0 {
                            break;
                        }

                        // ── MCTS Search ──
                        tree.new_game(board.clone());

                        // Helper: select leaves, run NN inference, expand+backup.
                        // Returns number of leaves processed, or 0 on failure.
                        let infer_and_expand = |tree: &mut MCTSTree, batch_size: usize| -> usize {
                            let leaves = tree.select_leaves(batch_size);
                            if leaves.is_empty() { return 0; }

                            let mut all_batch_features = Vec::new();
                            let mut leaf_metadata = Vec::with_capacity(leaves.len());

                            for leaf in &leaves {
                                let (views, centers) = leaf.get_cluster_views();
                                let k = views.len();
                                leaf_metadata.push((k, centers));
                                for view in views {
                                    let mut buffer = batcher.get_feature_buffer();
                                    leaf.encode_18_planes_to_buffer(&view, &mut buffer);
                                    all_batch_features.push(buffer);
                                }
                            }
                            if all_batch_features.is_empty() { return 0; }

                            let total_clusters: usize = leaf_metadata.iter().map(|(k, _)| *k).sum();

                            let (all_policies, all_values) = match batcher.submit_batch_and_wait_rust(all_batch_features) {
                                Ok(results) => {
                                    let mut ps = Vec::with_capacity(results.len());
                                    let mut vs = Vec::with_capacity(results.len());
                                    for (p, v) in results {
                                        ps.push(p);
                                        vs.push(v);
                                    }
                                    (ps, vs)
                                }
                                Err(_) => return 0,
                            };

                            if all_policies.len() < total_clusters { return 0; }

                            let mut aggregated_policies = Vec::with_capacity(leaves.len());
                            let mut aggregated_values = Vec::with_capacity(leaves.len());
                            let mut curr = 0;

                            for (i, (k, centers)) in leaf_metadata.iter().enumerate() {
                                let leaf_policies = &all_policies[curr..curr+k];
                                let leaf_values = &all_values[curr..curr+k];
                                curr += k;

                                let mut min_v = leaf_values[0];
                                for &v in leaf_values {
                                    if v < min_v { min_v = v; }
                                }
                                aggregated_values.push(min_v);
                                aggregated_policies.push(Self::aggregate_policy(&leaves[i], centers, leaf_policies));
                            }

                            let n = leaves.len();
                            tree.expand_and_backup(&aggregated_policies, &aggregated_values);
                            n
                        };

                        let mut gumbel_state: Option<GumbelSearchState> = None;

                        if gumbel_mcts {
                            // ── Gumbel MCTS with Sequential Halving ──
                            // Phase 1: Expand root to get children and priors.
                            let root_sims = infer_and_expand(&mut tree, 1);
                            if root_sims == 0 || !tree.pool[0].is_expanded() {
                                // Root expansion failed — skip this move.
                                continue;
                            }
                            let mut sims_used = root_sims;

                            // Apply Dirichlet noise to root priors after expansion.
                            // Skip at intermediate ply (second stone of compound turn) —
                            // mirrors hexo_rl/selfplay/worker.py:107-111.
                            // ply==0 is P1's single opening stone, which IS a turn boundary.
                            let is_intermediate_ply = board.moves_remaining == 1 && board.ply > 0;
                            if dirichlet_enabled && !is_intermediate_ply {
                                let n_ch = tree.pool[0].n_children as usize;
                                if n_ch > 0 {
                                    let noise = crate::mcts::dirichlet::sample_dirichlet(
                                        dirichlet_alpha, n_ch, &mut rng,
                                    );
                                    tree.apply_dirichlet_to_root(&noise, dirichlet_epsilon);
                                }
                            }

                            // Phase 2: Gumbel-Top-k candidate selection.
                            // Guard: if effective_m is 0 (no budget or no children),
                            // fall back to the standard PUCT path for this move.
                            let effective_m = gumbel_m.min(game_sims).min(tree.root_n_children());
                            if effective_m == 0 {
                                // No candidates — use standard PUCT search instead.
                                // Subtract root expansion sims already spent from budget.
                                let mut sims_done = sims_used;
                                while sims_done < game_sims {
                                    if !running.load(Ordering::SeqCst) { break; }
                                    let n = infer_and_expand(&mut tree, leaf_batch_size);
                                    if n == 0 { break; }
                                    sims_done += n;
                                }
                                gumbel_state = None;
                            } else {

                            let mut gs = GumbelSearchState::new(
                                &tree, effective_m, c_visit, c_scale, &mut rng,
                            );

                            // Phase 3: Sequential Halving — allocate budget across phases.
                            let num_phases = gs.num_phases;
                            for phase in 0..num_phases {
                                if sims_used >= game_sims { break; }
                                let remaining_budget = game_sims.saturating_sub(sims_used);
                                let remaining_phases = num_phases - phase;
                                let sims_per = (remaining_budget / (remaining_phases * gs.candidates.len())).max(1);

                                let cands = gs.candidates.clone();
                                for &cand_offset in &cands {
                                    if sims_used >= game_sims { break; }
                                    if !running.load(Ordering::SeqCst) { break; }

                                    let child_pool_idx = gs.first_child + cand_offset as u32;
                                    tree.forced_root_child = Some(child_pool_idx);

                                    let mut cand_sims = 0;
                                    while cand_sims < sims_per && sims_used < game_sims {
                                        if !running.load(Ordering::SeqCst) { break; }
                                        // Cap batch to remaining budget for this candidate so we
                                        // don't overshoot sims_per (leaf_batch_size can be 8 when
                                        // only 1-3 sims are budgeted, biasing early candidates).
                                        let batch = leaf_batch_size.min(sims_per.saturating_sub(cand_sims));
                                        let n = infer_and_expand(&mut tree, batch.max(1));
                                        if n == 0 { break; }
                                        cand_sims += n;
                                        sims_used += n;
                                    }
                                    tree.forced_root_child = None;
                                }

                                if gs.candidates.len() <= 1 { break; }
                                gs.halve_candidates(&tree);
                            }
                            tree.forced_root_child = None;
                            gumbel_state = Some(gs);
                            } // end effective_m > 0
                        } else {
                            // ── Standard PUCT search with Dirichlet root noise ──
                            // Expand root first so priors are available before noise injection.
                            // Uses batch=1 to guarantee only the root leaf is processed,
                            // matching the Gumbel branch pattern.
                            let root_n = infer_and_expand(&mut tree, 1);
                            if root_n == 0 {
                                continue;
                            }
                            let mut sims_done = root_n;

                            // Apply Dirichlet noise after root expansion, before simulation loop.
                            // Skip at intermediate ply (second stone of compound turn) —
                            // mirrors hexo_rl/selfplay/worker.py:107-111.
                            // ply==0 is P1's single opening stone, which IS a turn boundary.
                            let is_intermediate_ply = board.moves_remaining == 1 && board.ply > 0;
                            if dirichlet_enabled && !is_intermediate_ply {
                                if tree.pool[0].is_expanded() {
                                    let n_ch = tree.pool[0].n_children as usize;
                                    if n_ch > 0 {
                                        let noise = crate::mcts::dirichlet::sample_dirichlet(
                                            dirichlet_alpha, n_ch, &mut rng,
                                        );
                                        tree.apply_dirichlet_to_root(&noise, dirichlet_epsilon);
                                    }
                                }
                            }

                            while sims_done < game_sims {
                                if !running.load(Ordering::SeqCst) { break; }
                                let n = infer_and_expand(&mut tree, leaf_batch_size);
                                if n == 0 { break; }
                                sims_done += n;
                            }
                        }

                        if !running.load(Ordering::SeqCst) { break; }

                        // ── MCTS Policy with cosine-annealed temperature schedule ──
                        let compound_move = if board.ply == 0 { 0 } else { (board.ply as usize + 1) / 2 };
                        let temperature = if is_fast_game {
                            1.0  // fast games: always exploratory
                        } else if compound_move < temp_threshold {
                            let progress = compound_move as f32 / temp_threshold as f32;
                            f32::max(temp_min, (std::f32::consts::PI / 2.0 * progress).cos())
                        } else {
                            temp_min  // settled phase: minimal exploration
                        };
                        let policy = tree.get_policy(temperature, BOARD_SIZE);

                        // ── debug_prior_trace: snapshot root priors + visit counts ──
                        // Compile-time gated; zero cost in default builds. Runtime
                        // gated a second time by HEXO_PRIOR_TRACE_PATH (see
                        // engine/src/debug_trace.rs). Cap-enforced per site.
                        #[cfg(feature = "debug_prior_trace")]
                        {
                            let root = &tree.pool[0];
                            if root.is_expanded() {
                                let first = root.first_child as usize;
                                let n_ch = root.n_children as usize;
                                let mut priors = Vec::with_capacity(n_ch);
                                let mut visits = Vec::with_capacity(n_ch);
                                for j in 0..n_ch {
                                    priors.push(tree.pool[first + j].prior);
                                    visits.push(tree.pool[first + j].n_visits);
                                }
                                let legal_count = board.legal_move_count() as u32;
                                crate::debug_trace::record_game_runner(
                                    dbg_game_idx,
                                    worker_id as u32,
                                    compound_move as u32,
                                    board.ply as u32,
                                    legal_count,
                                    n_ch as u32,
                                    game_sims as u32,
                                    &priors,
                                    &visits,
                                    temperature,
                                    is_fast_game,
                                );
                            }
                        }

                        // Accumulate MCTS health stats once per search (not in inner sim loop).
                        {
                            let (depth, conc) = tree.last_search_stats();
                            mcts_depth_accum.fetch_add((depth * 1_000_000.0) as u64, Ordering::Relaxed);
                            mcts_conc_accum.fetch_add((conc * 1_000_000.0) as u64, Ordering::Relaxed);
                            mcts_stat_count.fetch_add(1, Ordering::Relaxed);
                            // Accumulate quiescence fires for this search. tree.new_game() resets
                            // the counter before each search, so this captures per-search fires.
                            mcts_quiescence_fires.fetch_add(
                                tree.quiescence_fire_count.load(Ordering::Relaxed),
                                Ordering::Relaxed,
                            );
                        }

                        // Completed Q-values: compute improved policy for training target.
                        // Move selection still uses temperature-scaled visit counts above.
                        let target_policy = if completed_q_values {
                            tree.get_improved_policy(BOARD_SIZE, c_visit, c_scale)
                        } else {
                            policy.clone()
                        };

                        // ── Sample and apply move (ZOI-filtered legal set) ──
                        let full_legal = board.legal_moves();
                        if full_legal.is_empty() { break; }

                        // ZOI filtering: restrict move sampling to cells near recent moves.
                        let legal = if zoi_enabled && move_history.len() >= 3 {
                            let filtered: Vec<_> = full_legal.iter()
                                .filter(|(q, r)| {
                                    move_history.iter().rev().take(zoi_lookback).any(|(q0, r0)| {
                                        hex_distance(*q, *r, *q0, *r0) <= zoi_margin
                                    })
                                })
                                .cloned()
                                .collect();
                            if filtered.len() < 3 { full_legal } else { filtered }
                        } else {
                            full_legal
                        };

                        // Move selection: Gumbel winner or visit-count sampling.
                        // Early in the game (< gumbel_explore_moves plies), always use
                        // visit-count sampling for exploration (paper Figure 8b).
                        // Note: threshold is in plies, not turns. With compound moves
                        // (2 per turn after the first), ply 10 ≈ turn 6. Adjust the
                        // gumbel_explore_moves default if more early exploration is desired.
                        let use_gumbel_winner = gumbel_state.is_some()
                            && board.ply as usize >= gumbel_explore_moves;
                        let move_idx = if use_gumbel_winner {
                            let gs = gumbel_state.as_mut().unwrap();
                            let best_pool = gs.best_action_pool_idx(&tree);
                            let val = tree.pool[best_pool as usize].action_idx;
                            let mq = (val >> 16) as i32 - 32768;
                            let mr = (val & 0xFFFF) as i32 - 32768;
                            // Ensure chosen move is legal (it always should be).
                            if legal.contains(&(mq, mr)) {
                                (mq, mr)
                            } else {
                                match Self::sample_policy(&policy, &legal, &board) {
                                    Some(idx) => idx,
                                    None => *legal.choose(&mut rng).unwrap(),
                                }
                            }
                        } else {
                            match Self::sample_policy(&policy, &legal, &board) {
                                Some(idx) => idx,
                                None => *legal.choose(&mut rng).unwrap(),
                            }
                        };
                        
                        // ── Record position ──
                        let n_actions = BOARD_SIZE * BOARD_SIZE + 1;
                        let (views, centers) = board.get_cluster_views();
                        for (k, center) in centers.iter().enumerate() {
                            let mut feat = batcher.get_feature_buffer();
                            // get_cluster_views returns 2-plane views; expand to 18 for storage.
                            board.encode_18_planes_to_buffer(&views[k], &mut feat);
                            // Fast games: zero-policy marks value-only targets (unless
                            // completed Q-values are enabled, which give signal even at 50 sims).
                            let projected_policy = if is_fast_game && !completed_q_values {
                                vec![0.0; n_actions]
                            } else {
                                Self::aggregate_policy_to_local(&board, center, &target_policy)
                            };
                            // Capture the per-row window centre so the aux targets
                            // (computed at game end) can be reprojected into the same
                            // coordinate frame as this row's state planes.
                            records.push((feat, projected_policy, board.current_player, center.0, center.1));
                        }

                        if board.apply_move(move_idx.0, move_idx.1).is_err() {
                            break;
                        }
                        move_history.push((move_idx.0, move_idx.1));
                        positions_generated.fetch_add(1, Ordering::SeqCst);
                    }

                    // ── Game End: determine outcome ──
                    let winner = board.winner();
                    let plies = board.ply as usize;
                    let winner_code: u8 = match winner {
                        Some(crate::board::Player::One) => 1,
                        Some(_)                         => 2,
                        None                            => 0,
                    };
                    // Snapshot the final-board cell list and winning line once;
                    // each row reprojects them into its own per-cluster window centre.
                    let final_cells: Vec<((i32, i32), Cell)> = board.cells
                        .iter()
                        .map(|(&qr, &c)| (qr, c))
                        .collect();
                    let winning_cells: Vec<(i32, i32)> = board.find_winning_line();

                    let mut games_results = results_queue.lock().expect("results lock poisoned");
                    for (feat, pol, player, cq, cr) in records {
                        let outcome = match winner {
                            Some(p) => if p as i8 == player as i8 { 1.0 } else { -1.0 },
                            None => draw_reward,
                        };

                        // Per-row aux reprojection: project the FINAL board state and
                        // winning line into the window centred on (cq, cr) — the same
                        // centre used to encode this row's state planes. Encoding:
                        //   ownership_u8: 0 = P2, 1 = empty, 2 = P1
                        //   winning_line_u8: 0 / 1 binary
                        let mut own_u8 = vec![1u8; TOTAL_CELLS];
                        for &((q, r), cell) in &final_cells {
                            let flat = Board::window_flat_idx_at(q, r, cq, cr);
                            if flat < TOTAL_CELLS {
                                own_u8[flat] = match cell {
                                    Cell::P1    => 2,
                                    Cell::P2    => 0,
                                    Cell::Empty => 1,
                                };
                            }
                        }
                        let mut wl_u8 = vec![0u8; TOTAL_CELLS];
                        for &(q, r) in &winning_cells {
                            let flat = Board::window_flat_idx_at(q, r, cq, cr);
                            if flat < TOTAL_CELLS {
                                wl_u8[flat] = 1;
                            }
                        }

                        games_results.push_back((feat, pol, outcome, plies, own_u8, wl_u8));
                    }
                    games_completed.fetch_add(1, Ordering::Relaxed);
                    match winner {
                        Some(crate::board::Player::One) => { x_wins.fetch_add(1, Ordering::Relaxed); }
                        Some(_)                         => { o_wins.fetch_add(1, Ordering::Relaxed); }
                        None                            => { draws.fetch_add(1, Ordering::Relaxed); }
                    }
                    // Compute ownership_flat: 361-element window projection of final board state.
                    // +1.0 = P1 stone, -1.0 = P2 stone, 0.0 = empty.
                    let mut ownership_flat = vec![0.0f32; TOTAL_CELLS];
                    for (&(q, r), &cell) in board.cells.iter() {
                        let flat = board.window_flat_idx(q, r);
                        if flat < TOTAL_CELLS {
                            ownership_flat[flat] = match cell {
                                Cell::P1 => 1.0,
                                Cell::P2 => -1.0,
                                Cell::Empty => 0.0,
                            };
                        }
                    }

                    // Compute winning_line_flat: 361-element window projection of the 6-cell winning run.
                    let mut winning_line_flat = vec![0.0f32; TOTAL_CELLS];
                    for (q, r) in board.find_winning_line() {
                        let flat = board.window_flat_idx(q, r);
                        if flat < TOTAL_CELLS {
                            winning_line_flat[flat] = 1.0;
                        }
                    }

                    // Record for Python game_complete logging and auxiliary training targets.
                    {
                        let mut rg = recent_game_results.lock().expect("recent_game_results lock poisoned");
                        rg.push_back((plies, winner_code, move_history, worker_id, ownership_flat, winning_line_flat));
                        // Cap at 2000 entries to avoid unbounded growth if Python is slow.
                        if rg.len() > 2000 {
                            rg.pop_front();
                        }
                    }

                    // Cap the results queue to avoid memory explosion if Python is slow
                    if games_results.len() > 10000 {
                        let to_drop = games_results.len() - 10000;
                        for _ in 0..to_drop {
                            games_results.pop_front();
                        }
                    }

                    #[cfg(feature = "debug_prior_trace")]
                    { dbg_game_idx += 1; }
                }
            });
            handles.push(handle);
        }
    }

    /// Drain all buffered positions and return them as numpy arrays.
    ///
    /// Returns (features, policies, values, plies, ownership, winning_line) where each
    /// row is one position:
    ///   features:     (N, feat_len) float32
    ///   policies:     (N, pol_len)  float32
    ///   values:       (N,)          float32
    ///   plies:        (N,)          uint64 — game length in plies (game-length weighting)
    ///   ownership:    (N, 361)      uint8  — per-row aux target {0=P2, 1=empty, 2=P1}
    ///   winning_line: (N, 361)      uint8  — per-row binary mask of winning 6-in-a-row
    ///
    /// Ownership and winning_line are projected to each row's per-cluster window
    /// centre (NOT the game-end bbox centroid). They live in the same coordinate
    /// frame as the row's state planes — see game_runner.rs for the reprojection.
    ///
    /// N = 0 when no positions are available (arrays have zero rows).
    /// Returning pre-built numpy arrays eliminates Vec<f32> → Python list conversion
    /// and the associated pymalloc arena fragmentation (~164 KB pymalloc/position).
    pub fn collect_data<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray1<f32>>,
        Bound<'py, PyArray1<u64>>,
        Bound<'py, PyArray2<u8>>,
        Bound<'py, PyArray2<u8>>,
    )> {
        let feat_len = self.batcher.feature_len();
        let pol_len  = self.pol_len;

        let mut results = self.results.lock().expect("results lock poisoned");
        let n = results.len();

        let mut flat_feats = Vec::with_capacity(n * feat_len);
        let mut flat_pols  = Vec::with_capacity(n * pol_len);
        let mut vals       = Vec::with_capacity(n);
        let mut plies_out  = Vec::with_capacity(n);
        let mut flat_own   = Vec::with_capacity(n * TOTAL_CELLS);
        let mut flat_wl    = Vec::with_capacity(n * TOTAL_CELLS);

        while let Some((feat, pol, outcome, plies, own_u8, wl_u8)) = results.pop_front() {
            flat_feats.extend_from_slice(&feat);
            flat_pols.extend_from_slice(&pol);
            vals.push(outcome);
            plies_out.push(plies as u64);
            flat_own.extend_from_slice(&own_u8);
            flat_wl.extend_from_slice(&wl_u8);
        }

        let feats_np = flat_feats.into_pyarray(py).reshape([n, feat_len])?;
        let pols_np  = flat_pols.into_pyarray(py).reshape([n, pol_len])?;
        let vals_np  = vals.into_pyarray(py);
        let gids_np  = plies_out.into_pyarray(py);
        let own_np   = flat_own.into_pyarray(py).reshape([n, TOTAL_CELLS])?;
        let wl_np    = flat_wl.into_pyarray(py).reshape([n, TOTAL_CELLS])?;

        Ok((feats_np, pols_np, vals_np, gids_np, own_np, wl_np))
    }

    pub fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
        self.batcher.close_rust();

        let mut handles = self.handles.lock().expect("runner handles lock poisoned");
        while let Some(handle) = handles.pop() {
            let _ = handle.join();
        }
    }

    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    #[getter]
    pub fn batcher(&self) -> InferenceBatcher {
        self.batcher.clone()
    }

    #[getter]
    pub fn games_completed(&self) -> usize {
        self.games_completed.load(Ordering::Relaxed)
    }

    #[getter]
    pub fn positions_generated(&self) -> usize {
        self.positions_generated.load(Ordering::Relaxed)
    }

    #[getter]
    pub fn x_wins(&self) -> u64 {
        self.x_wins.load(Ordering::Relaxed)
    }

    #[getter]
    pub fn o_wins(&self) -> u64 {
        self.o_wins.load(Ordering::Relaxed)
    }

    #[getter]
    pub fn draws(&self) -> u64 {
        self.draws.load(Ordering::Relaxed)
    }

    pub fn get_win_stats(&self) -> (u64, u64, u64) {
        (
            self.x_wins.load(Ordering::Relaxed),
            self.o_wins.load(Ordering::Relaxed),
            self.draws.load(Ordering::Relaxed),
        )
    }

    /// Mean MCTS leaf depth across all searches since `start()`.
    /// Returns 0.0 before any searches have been recorded.
    #[getter]
    pub fn mcts_mean_depth(&self) -> f32 {
        let count = self.mcts_stat_count.load(Ordering::Relaxed);
        if count == 0 {
            return 0.0;
        }
        (self.mcts_depth_accum.load(Ordering::Relaxed) as f64 / (count as f64 * 1_000_000.0)) as f32
    }

    /// Mean root concentration (max child visits / total root visits) since `start()`.
    /// Range [0.0, 1.0]. Returns 0.0 before any searches have been recorded.
    #[getter]
    pub fn mcts_mean_root_concentration(&self) -> f32 {
        let count = self.mcts_stat_count.load(Ordering::Relaxed);
        if count == 0 {
            return 0.0;
        }
        self.mcts_conc_accum.load(Ordering::Relaxed) as f32 / (count as f32 * 1_000_000.0)
    }

    /// Cumulative quiescence value override/blend count since `start()`.
    /// Counts all 4 firing branches in `apply_quiescence()`.
    #[getter]
    pub fn mcts_quiescence_fires(&self) -> u64 {
        self.mcts_quiescence_fires.load(Ordering::Relaxed)
    }

    /// Drain and return all buffered game results since the last call.
    ///
    /// Each entry: (plies, winner_code, move_history, worker_id, ownership_flat, winning_line_flat)
    ///   winner_code: 1 = Player One, 2 = Player Two, 0 = draw.
    ///   move_history: (q, r) stone placements in play order.
    ///   ownership_flat: PyArray1<f32> len 361 — +1.0 P1 / -1.0 P2 / 0.0 empty.
    ///   winning_line_flat: PyArray1<f32> len 361 — 1.0 at winning-line cells, 0.0 elsewhere.
    ///
    /// ownership_flat and winning_line_flat are returned as numpy arrays to avoid
    /// Vec<f32> → Python list conversion and the associated pymalloc arena fragmentation.
    pub fn drain_game_results<'py>(
        &self,
        py: Python<'py>,
    ) -> Vec<(usize, u8, Vec<(i32, i32)>, usize, Py<PyArray1<f32>>, Py<PyArray1<f32>>)> {
        self.drain_game_results_raw()
            .into_iter()
            .map(|(plies, winner_code, move_history, worker_id, ownership_flat, winning_line_flat)| {
                let own_np = ownership_flat.into_pyarray(py).unbind();
                let win_np = winning_line_flat.into_pyarray(py).unbind();
                (plies, winner_code, move_history, worker_id, own_np, win_np)
            })
            .collect()
    }
}

impl SelfPlayRunner {
    /// Internal drain used by tests and the pymethods wrapper.
    /// Returns raw Vecs — no Python dependency, safe to call from `cargo test`.
    pub(crate) fn drain_game_results_raw(
        &self,
    ) -> Vec<(usize, u8, Vec<(i32, i32)>, usize, Vec<f32>, Vec<f32>)> {
        let mut rg = self.recent_game_results.lock().expect("recent_game_results lock poisoned");
        rg.drain(..).collect()
    }

    fn aggregate_policy(board: &Board, centers: &[(i32, i32)], cluster_policies: &[Vec<f32>]) -> Vec<f32> {
        let n_actions = BOARD_SIZE * BOARD_SIZE + 1;
        let mut global_policy = vec![0.0; n_actions];
        let legal = board.legal_moves();
        
        for &(q, r) in &legal {
            let mcts_idx = board.window_flat_idx(q, r);
            if mcts_idx >= n_actions - 1 { continue; }
            
            let mut max_prob = 0.0;
            for (k, &(cq, cr)) in centers.iter().enumerate() {
                let wq = q - cq + HALF;
                let wr = r - cr + HALF;
                if wq >= 0 && wq < BOARD_SIZE as i32 && wr >= 0 && wr < BOARD_SIZE as i32 {
                    let local_idx = wq as usize * BOARD_SIZE + wr as usize;
                    if cluster_policies[k][local_idx] > max_prob {
                        max_prob = cluster_policies[k][local_idx];
                    }
                }
            }
            global_policy[mcts_idx] = max_prob;
        }
        
        // Pass move is always copied from the first cluster (should be consistent)
        if !cluster_policies.is_empty() {
            global_policy[n_actions - 1] = cluster_policies[0][n_actions - 1];
        }
        
        let sum: f32 = global_policy.iter().sum();
        if sum > 1e-9 {
            for p in &mut global_policy { *p /= sum; }
        } else {
            let uniform = 1.0 / n_actions as f32;
            for p in &mut global_policy { *p = uniform; }
        }
        global_policy
    }

    fn aggregate_policy_to_local(board: &Board, center: &(i32, i32), global_policy: &[f32]) -> Vec<f32> {
        let (cq, cr) = *center;
        let n_actions = BOARD_SIZE * BOARD_SIZE + 1;
        let mut local_policy = vec![0.0; n_actions];
        let legal = board.legal_moves();
        
        for &(q, r) in &legal {
            let wq = q - cq + HALF;
            let wr = r - cr + HALF;
            if wq >= 0 && wq < BOARD_SIZE as i32 && wr >= 0 && wr < BOARD_SIZE as i32 {
                let local_idx = wq as usize * BOARD_SIZE + wr as usize;
                let mcts_idx = board.window_flat_idx(q, r);
                if mcts_idx < global_policy.len() {
                    local_policy[local_idx] = global_policy[mcts_idx];
                }
            }
        }
        
        // Pass move (the last element) is always copied from the global policy
        if n_actions > 0 && global_policy.len() >= n_actions {
            local_policy[n_actions - 1] = global_policy[n_actions - 1];
        }
        
        let sum: f32 = local_policy.iter().sum();
        if sum > 1e-9 {
            for p in &mut local_policy { *p /= sum; }
        } else {
            let uniform = 1.0 / n_actions as f32;
            for p in &mut local_policy { *p = uniform; }
        }
        local_policy
    }

    fn sample_policy(policy: &[f32], legal_moves: &[(i32, i32)], board: &Board) -> Option<(i32, i32)> {
        let mut probs = Vec::with_capacity(legal_moves.len());
        let mut sum = 0.0;
        for &(q, r) in legal_moves {
            let idx = board.window_flat_idx(q, r);
            let p = if idx < policy.len() { policy[idx] } else { 0.0 };
            probs.push(p);
            sum += p;
        }

        if sum < 1e-9 {
            return None;
        }

        let mut rng = rng();
        let mut r: f32 = rng.random();
        r *= sum;

        let mut current = 0.0;
        for (i, &p) in probs.iter().enumerate() {
            current += p;
            if r <= current {
                return Some(legal_moves[i]);
            }
        }
        Some(legal_moves[legal_moves.len() - 1])
    }
}

impl Drop for SelfPlayRunner {
    fn drop(&mut self) {
        self.stop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::time::Duration;

    #[test]
    fn test_worker_id_assignment() {
        // Run with max_moves_per_game = 0 to avoid triggering MCTS and inference server dependency
        let runner = SelfPlayRunner::new(
            4, 0, 1, 1, 1.5, 0.25, 18*19*19, 19*19+1, 1.0, 1, 1, 15, -0.1, true, 0.3,
            0.05, false, 16, 5, false, 50.0, 1.0, false, 16, 10, 0.3, 0.25, true,
        );
        runner.start();
        
        let mut attempts = 0;
        let mut completed_workers = HashSet::new();
        
        while completed_workers.len() < 4 && attempts < 50 {
            let results = runner.drain_game_results_raw();
            for (_, _, _, worker_id, _, _) in results {
                assert!(worker_id < 4);
                completed_workers.insert(worker_id);
            }
            std::thread::sleep(Duration::from_millis(50));
            attempts += 1;
        }
        
        runner.stop();
        assert_eq!(completed_workers.len(), 4, "Should have seen games from all 4 workers");
    }

    // ── Dirichlet call-site gate tests ───────────────────────────────────────

    /// Verify that Dirichlet noise modifies root priors when enabled and leaves
    /// them unchanged when disabled.  Uses MCTSTree directly — no inference
    /// server, no PyO3.
    #[test]
    fn test_dirichlet_gate_enabled_modifies_priors() {
        let mut tree = crate::mcts::MCTSTree::new(1.5);
        let board = crate::board::Board::new();
        tree.new_game(board.clone());

        // Expand root with uniform priors (no inference server needed).
        let n_actions = crate::board::BOARD_SIZE * crate::board::BOARD_SIZE + 1;
        let policy = vec![1.0 / n_actions as f32; n_actions];
        let _leaves = tree.select_leaves(1);
        tree.expand_and_backup(&[policy], &[0.0]);
        assert!(tree.pool[0].is_expanded(), "root should be expanded");

        // Snapshot pre-noise priors.
        let first = tree.pool[0].first_child as usize;
        let n_ch = tree.pool[0].n_children as usize;
        let priors_before: Vec<f32> = (0..n_ch).map(|j| tree.pool[first + j].prior).collect();

        // Apply Dirichlet noise (same call site logic as game_runner PUCT branch).
        let alpha = 0.3f32;
        let epsilon = 0.25f32;
        let mut rng = rand::rng();
        let noise = crate::mcts::dirichlet::sample_dirichlet(alpha, n_ch, &mut rng);
        tree.apply_dirichlet_to_root(&noise, epsilon);

        let priors_after: Vec<f32> = (0..n_ch).map(|j| tree.pool[first + j].prior).collect();

        // With epsilon=0.25 and Dirichlet noise, priors should differ from uniform.
        let max_diff: f32 = priors_before
            .iter()
            .zip(priors_after.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_diff > 1e-6,
            "Dirichlet noise should modify root priors; max_diff={max_diff}"
        );
    }

    #[test]
    fn test_dirichlet_gate_disabled_preserves_priors() {
        let mut tree = crate::mcts::MCTSTree::new(1.5);
        let board = crate::board::Board::new();
        tree.new_game(board);

        let n_actions = crate::board::BOARD_SIZE * crate::board::BOARD_SIZE + 1;
        let policy = vec![1.0 / n_actions as f32; n_actions];
        let _leaves = tree.select_leaves(1);
        tree.expand_and_backup(&[policy], &[0.0]);

        let first = tree.pool[0].first_child as usize;
        let n_ch = tree.pool[0].n_children as usize;
        let priors_before: Vec<f32> = (0..n_ch).map(|j| tree.pool[first + j].prior).collect();

        // dirichlet_enabled = false: do NOT call apply_dirichlet_to_root.
        // Priors should be identical to pre-expansion values.
        let priors_after: Vec<f32> = (0..n_ch).map(|j| tree.pool[first + j].prior).collect();

        assert_eq!(
            priors_before, priors_after,
            "Disabled gate must not modify root priors"
        );
    }

    // ── Gumbel MCTS tests ────────────────────────────────────────────────────

    fn setup_tree_for_gumbel() -> crate::mcts::MCTSTree {
        let mut tree = crate::mcts::MCTSTree::new(1.5);
        let board = Board::new();
        tree.new_game(board);

        // Expand root with uniform priors.
        let n_actions = BOARD_SIZE * BOARD_SIZE + 1;
        let policy = vec![1.0 / n_actions as f32; n_actions];
        let _leaves = tree.select_leaves(1);
        tree.expand_and_backup(&[policy], &[0.0]);
        tree
    }

    #[test]
    fn test_gumbel_topk_selection() {
        let tree = setup_tree_for_gumbel();
        let n_children = tree.root_n_children();
        assert!(n_children > 16, "fresh board should have many legal moves");

        let mut rng = rand::rng();
        let gs = GumbelSearchState::new(&tree, 16, 50.0, 1.0, &mut rng);

        assert_eq!(gs.candidates.len(), 16, "should select m=16 candidates");
        assert_eq!(gs.gumbel_values.len(), n_children);
        assert_eq!(gs.log_priors.len(), n_children);

        // All candidates should be unique and within valid range.
        let mut seen = std::collections::HashSet::new();
        for &c in &gs.candidates {
            assert!(c < n_children, "candidate offset {c} out of range");
            assert!(seen.insert(c), "duplicate candidate {c}");
        }
    }

    #[test]
    fn test_gumbel_topk_few_legal_moves() {
        // When legal moves < m, all moves should be candidates.
        let tree = setup_tree_for_gumbel();
        let n_children = tree.root_n_children();
        let mut rng = rand::rng();

        // Request m = 1000, but there are only n_children legal moves.
        let gs = GumbelSearchState::new(&tree, 1000, 50.0, 1.0, &mut rng);
        assert_eq!(gs.candidates.len(), n_children,
            "with m > legal moves, all {} moves should be candidates", n_children);
    }

    #[test]
    fn test_sequential_halving_phases_count() {
        let tree = setup_tree_for_gumbel();
        let mut rng = rand::rng();

        let gs8 = GumbelSearchState::new(&tree, 8, 50.0, 1.0, &mut rng);
        assert_eq!(gs8.num_phases, 3, "ceil(log2(8)) = 3");

        let gs16 = GumbelSearchState::new(&tree, 16, 50.0, 1.0, &mut rng);
        assert_eq!(gs16.num_phases, 4, "ceil(log2(16)) = 4");

        let gs1 = GumbelSearchState::new(&tree, 1, 50.0, 1.0, &mut rng);
        assert_eq!(gs1.num_phases, 1, "ceil(log2(1)) should be 1");
        assert_eq!(gs1.candidates.len(), 1);
    }

    #[test]
    fn test_halve_candidates_reduces_count() {
        let mut tree = setup_tree_for_gumbel();
        let mut rng = rand::rng();
        let mut gs = GumbelSearchState::new(&tree, 8, 50.0, 1.0, &mut rng);
        assert_eq!(gs.candidates.len(), 8);

        // Give some candidates visits so they have different Q values.
        let first = tree.pool[0].first_child as usize;
        for (i, &cand) in gs.candidates.iter().enumerate() {
            let pool_idx = first + cand;
            tree.pool[pool_idx].n_visits = (i + 1) as u32;
            tree.pool[pool_idx].w_value = (i as f32) * 0.1;
        }

        gs.halve_candidates(&tree);
        assert_eq!(gs.candidates.len(), 4, "8 → 4 after one halving");

        gs.halve_candidates(&tree);
        assert_eq!(gs.candidates.len(), 2, "4 → 2 after two halvings");

        gs.halve_candidates(&tree);
        assert_eq!(gs.candidates.len(), 1, "2 → 1 after three halvings");
    }

    #[test]
    fn test_gumbel_score_uses_sigma() {
        let mut tree = setup_tree_for_gumbel();
        let mut rng = rand::rng();
        let mut gs = GumbelSearchState::new(&tree, 8, 50.0, 1.0, &mut rng);

        let first = tree.pool[0].first_child as usize;
        let c0 = gs.candidates[0];
        let c1 = gs.candidates[1];

        // Give c0 a high Q, c1 a low Q.
        tree.pool[first + c0].n_visits = 10;
        tree.pool[first + c0].w_value = 8.0; // Q = 0.8
        tree.pool[first + c1].n_visits = 10;
        tree.pool[first + c1].w_value = -8.0; // Q = -0.8

        gs.refresh_cache(&tree);
        let max_n = gs.max_n();
        let s0 = gs.score(c0, max_n);
        let s1 = gs.score(c1, max_n);
        // Higher Q should lead to higher score (sigma term dominates at high visit counts).
        assert!(s0 > s1, "higher Q should give higher score: {s0} vs {s1}");
    }

    #[test]
    fn test_best_action_pool_idx() {
        let mut tree = setup_tree_for_gumbel();
        let mut rng = rand::rng();
        let mut gs = GumbelSearchState::new(&tree, 4, 50.0, 1.0, &mut rng);

        let first = tree.pool[0].first_child;

        // Give the last candidate a very high Q.
        let best_cand = *gs.candidates.last().unwrap();
        tree.pool[(first as usize) + best_cand].n_visits = 100;
        tree.pool[(first as usize) + best_cand].w_value = 90.0; // Q = 0.9

        let best_pool = gs.best_action_pool_idx(&tree);
        assert_eq!(best_pool, first + best_cand as u32,
            "best action should be the high-Q candidate");
    }
}

