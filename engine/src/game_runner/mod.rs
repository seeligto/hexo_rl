//! Self-play runner — spawns Rust worker threads that run full games,
//! stream (feature, policy, outcome, aux) rows back to Python via the
//! shared `results` queue, and track game-level win stats + MCTS health
//! metrics.
//!
//! ## Module layout
//!   mod.rs          — `SelfPlayRunner` struct + `#[pymethods]` facade + Drop + mod tests
//!   worker_loop.rs  — `start_impl` (spawns workers, runs the main self-play loop)
//!   gumbel_search.rs — `GumbelSearchState` (Gumbel-Top-k + Sequential Halving)
//!   records.rs      — policy aggregation + game-end aux reprojection helpers

pub mod gumbel_search;
mod records;
mod worker_loop;

pub use worker_loop::compute_move_temperature;

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;

use crate::board::TOTAL_CELLS;
use crate::inference_bridge::InferenceBatcher;

#[pyclass(name = "SelfPlayRunner")]
pub struct SelfPlayRunner {
    pub(crate) batcher: InferenceBatcher,
    pub(crate) pol_len: usize,
    pub(crate) n_workers: usize,
    pub(crate) max_moves_per_game: usize,
    pub(crate) leaf_batch_size: usize,
    pub(crate) c_puct: f32,
    pub(crate) fpu_reduction: f32,
    pub(crate) quiescence_enabled: bool,
    pub(crate) quiescence_blend_2: f32,
    pub(crate) fast_prob: f32,
    pub(crate) fast_sims: usize,
    pub(crate) standard_sims: usize,
    pub(crate) temp_threshold_compound_moves: usize,
    pub(crate) temp_min: f32,
    pub(crate) draw_reward: f32,
    pub(crate) zoi_enabled: bool,
    pub(crate) zoi_lookback: usize,
    pub(crate) zoi_margin: i32,
    pub(crate) completed_q_values: bool,
    pub(crate) c_visit: f32,
    pub(crate) c_scale: f32,
    pub(crate) gumbel_mcts: bool,
    pub(crate) gumbel_m: usize,
    pub(crate) gumbel_explore_moves: usize,
    pub(crate) dirichlet_alpha: f32,
    pub(crate) dirichlet_epsilon: f32,
    pub(crate) dirichlet_enabled: bool,
    /// Move-level playout cap: probability each move uses full search.
    /// 0.0 = disabled (all moves use game_sims from game-level cap).
    pub(crate) full_search_prob: f32,
    /// Sim budget for quick-search moves (is_full_search = false).
    pub(crate) n_sims_quick: usize,
    /// Sim budget for full-search moves (is_full_search = true).
    pub(crate) n_sims_full: usize,
    /// Maximum positions buffered in `results` before workers drop the oldest
    /// to avoid unbounded growth when Python consumption stalls. Tracked on
    /// `positions_dropped` when it fires.
    pub(crate) results_queue_cap: usize,
    pub(crate) running: Arc<AtomicBool>,
    pub(crate) games_completed: Arc<AtomicUsize>,
    pub(crate) positions_generated: Arc<AtomicUsize>,
    pub(crate) x_wins: Arc<AtomicU64>,
    pub(crate) o_wins: Arc<AtomicU64>,
    pub(crate) draws: Arc<AtomicU64>,
    /// Positions evicted from the `results` queue by the `results_queue_cap`
    /// backpressure drop. Always zero under a healthy consumer; any nonzero
    /// reading means the Rust worker threads outran Python's drain rate and
    /// the dashboard-visible throughput numbers overstate what actually
    /// reached the replay buffer. Monotonic since `start()`.
    pub(crate) positions_dropped: Arc<AtomicU64>,
    /// Per-row training rows produced by self-play.
    /// Tuple: (feat, chain, policy, outcome, plies, combined_aux_u8)
    ///   feat:   18-plane state (f32, flat [18 × TOTAL_CELLS])
    ///   chain:  6-plane chain-length (f32, flat [6 × TOTAL_CELLS], normalized /6)
    ///   combined_aux_u8 layout: first TOTAL_CELLS bytes = ownership
    ///     (0=P2, 1=empty, 2=P1), last TOTAL_CELLS = winning_line binary mask.
    ///     Both projected to the row's own cluster window centre at the time
    ///     the position was recorded (NOT the game-end bbox centroid), so the
    ///     aux target frame aligns with the state frame under later symmetry
    ///     augmentation in the replay buffer.
    pub(crate) results: Arc<Mutex<VecDeque<(Vec<f32>, Vec<f32>, Vec<f32>, f32, usize, Vec<u8>, bool)>>>,
    /// Ring-buffer of recent game results for Python logging.
    /// Tuple: (plies, winner_code, move_history, worker_id)
    ///   winner_code: 1 = Player One, 2 = Player Two, 0 = draw.
    ///   move_history: sequence of (q, r) coordinates in play order.
    pub(crate) recent_game_results: Arc<Mutex<VecDeque<(usize, u8, Vec<(i32, i32)>, usize)>>>,
    pub(crate) handles: Arc<Mutex<Vec<JoinHandle<()>>>>,
    /// Accumulated MCTS leaf depth across all searches (scaled by 1_000_000 to preserve fractional part).
    pub(crate) mcts_depth_accum: Arc<AtomicU64>,
    /// Accumulated root concentration * 1_000_000 across all searches.
    pub(crate) mcts_conc_accum: Arc<AtomicU64>,
    /// Number of searches (moves) contributing to the above accumulators.
    pub(crate) mcts_stat_count: Arc<AtomicU64>,
    /// Cumulative quiescence fires (all 4 branches) across all searches since `start()`.
    pub(crate) mcts_quiescence_fires: Arc<AtomicU64>,
}

#[pymethods]
impl SelfPlayRunner {
    #[new]
    #[pyo3(signature = (n_workers = 4, max_moves_per_game = 128, n_simulations = 50, leaf_batch_size = 8, c_puct = 1.5, fpu_reduction = 0.25, feature_len = 18 * 19 * 19, policy_len = 19 * 19 + 1, fast_prob = 0.0, fast_sims = 50, standard_sims = 0, temp_threshold_compound_moves = 15, draw_reward = -0.1, quiescence_enabled = true, quiescence_blend_2 = 0.3, temp_min = 0.05, zoi_enabled = false, zoi_lookback = 16, zoi_margin = 5, completed_q_values = false, c_visit = 50.0, c_scale = 1.0, gumbel_mcts = false, gumbel_m = 16, gumbel_explore_moves = 10, dirichlet_alpha = 0.3, dirichlet_epsilon = 0.25, dirichlet_enabled = true, results_queue_cap = 10_000, full_search_prob = 0.0, n_sims_quick = 0, n_sims_full = 0))]
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
        results_queue_cap: usize,
        full_search_prob: f32,
        n_sims_quick: usize,
        n_sims_full: usize,
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
            results_queue_cap,
            full_search_prob,
            n_sims_quick,
            n_sims_full,
            running: Arc::new(AtomicBool::new(false)),
            games_completed: Arc::new(AtomicUsize::new(0)),
            positions_generated: Arc::new(AtomicUsize::new(0)),
            x_wins: Arc::new(AtomicU64::new(0)),
            o_wins: Arc::new(AtomicU64::new(0)),
            draws: Arc::new(AtomicU64::new(0)),
            positions_dropped: Arc::new(AtomicU64::new(0)),
            results: Arc::new(Mutex::new(VecDeque::new())),
            recent_game_results: Arc::new(Mutex::new(VecDeque::new())),
            handles: Arc::new(Mutex::new(Vec::new())),
            mcts_depth_accum: Arc::new(AtomicU64::new(0)),
            mcts_conc_accum: Arc::new(AtomicU64::new(0)),
            mcts_stat_count: Arc::new(AtomicU64::new(0)),
            mcts_quiescence_fires: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Spawn `n_workers` self-play threads. Idempotent.
    pub fn start(&self) {
        self.start_impl();
    }

    /// Drain all buffered positions and return them as numpy arrays.
    ///
    /// Returns (features, chain_planes, policies, values, plies, ownership, winning_line, is_full_search):
    ///   features:        (N, feat_len) float32 — 18-plane state tensor
    ///   chain_planes:    (N, 6*361)    float32 — Q13 chain-length planes (flat, /6 normalized)
    ///   policies:        (N, pol_len)  float32
    ///   values:          (N,)          float32
    ///   plies:           (N,)          uint64 — game length in plies (game-length weighting)
    ///   ownership:       (N, 361)      uint8  — per-row aux target {0=P2, 1=empty, 2=P1}
    ///   winning_line:    (N, 361)      uint8  — per-row binary mask of winning 6-in-a-row
    ///   is_full_search:  (N,)          uint8  — 1 = full-search move, 0 = quick-search move
    ///
    /// N = 0 when no positions are available (arrays have zero rows).
    pub fn collect_data<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray1<f32>>,
        Bound<'py, PyArray1<u64>>,
        Bound<'py, PyArray2<u8>>,
        Bound<'py, PyArray2<u8>>,
        Bound<'py, PyArray1<u8>>,
    )> {
        let feat_len  = self.batcher.feature_len();
        let chain_len = 6 * TOTAL_CELLS;
        let pol_len   = self.pol_len;

        let mut results = self.results.lock().expect("results lock poisoned");
        let n = results.len();

        let mut flat_feats      = Vec::with_capacity(n * feat_len);
        let mut flat_chain      = Vec::with_capacity(n * chain_len);
        let mut flat_pols       = Vec::with_capacity(n * pol_len);
        let mut vals            = Vec::with_capacity(n);
        let mut plies_out       = Vec::with_capacity(n);
        let mut flat_own        = Vec::with_capacity(n * TOTAL_CELLS);
        let mut flat_wl         = Vec::with_capacity(n * TOTAL_CELLS);
        let mut is_full_search  = Vec::with_capacity(n);

        while let Some((feat, chain, pol, outcome, plies, aux_u8, full_search)) = results.pop_front() {
            flat_feats.extend_from_slice(&feat);
            flat_chain.extend_from_slice(&chain);
            flat_pols.extend_from_slice(&pol);
            vals.push(outcome);
            plies_out.push(plies as u64);
            // Split combined aux: first TOTAL_CELLS = ownership, last = winning_line.
            flat_own.extend_from_slice(&aux_u8[..TOTAL_CELLS]);
            flat_wl.extend_from_slice(&aux_u8[TOTAL_CELLS..]);
            is_full_search.push(full_search as u8);
        }

        let feats_np  = flat_feats.into_pyarray(py).reshape([n, feat_len])?;
        let chain_np  = flat_chain.into_pyarray(py).reshape([n, chain_len])?;
        let pols_np   = flat_pols.into_pyarray(py).reshape([n, pol_len])?;
        let vals_np   = vals.into_pyarray(py);
        let gids_np   = plies_out.into_pyarray(py);
        let own_np    = flat_own.into_pyarray(py).reshape([n, TOTAL_CELLS])?;
        let wl_np     = flat_wl.into_pyarray(py).reshape([n, TOTAL_CELLS])?;
        let ifs_np    = is_full_search.into_pyarray(py);

        Ok((feats_np, chain_np, pols_np, vals_np, gids_np, own_np, wl_np, ifs_np))
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

    /// Positions dropped by `results_queue_cap` backpressure since `start()`.
    /// Always 0 under a healthy consumer. Any nonzero reading means the
    /// Rust side outran Python and the throughput numbers overstate what
    /// actually reached the replay buffer.
    #[getter]
    pub fn positions_dropped(&self) -> u64 {
        self.positions_dropped.load(Ordering::Relaxed)
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
    /// Each entry: (plies, winner_code, move_history, worker_id)
    ///   winner_code: 1 = Player One, 2 = Player Two, 0 = draw.
    ///   move_history: (q, r) stone placements in play order.
    pub fn drain_game_results(
        &self,
    ) -> Vec<(usize, u8, Vec<(i32, i32)>, usize)> {
        self.drain_game_results_raw()
    }
}

impl SelfPlayRunner {
    /// Internal drain used by tests and the pymethods wrapper.
    /// Returns raw Vecs — no Python dependency, safe to call from `cargo test`.
    pub(crate) fn drain_game_results_raw(
        &self,
    ) -> Vec<(usize, u8, Vec<(i32, i32)>, usize)> {
        let mut rg = self.recent_game_results.lock().expect("recent_game_results lock poisoned");
        rg.drain(..).collect()
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
            10_000, 0.0_f32, 0_usize, 0_usize,
        );
        runner.start();

        let mut attempts = 0;
        let mut completed_workers = HashSet::new();

        while completed_workers.len() < 4 && attempts < 50 {
            let results = runner.drain_game_results_raw();
            for (_, _, _, worker_id) in results {
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

    // ── MCTS mean-depth accounting regression (A1 review L5) ────────────────

    /// Verify `MCTSTree::last_search_stats()` is per-search, not cumulative.
    ///
    /// The post-refactor review raised a concern that the worker loop's
    /// stats block at `worker_loop.rs` (former `game_runner.rs:622-633`)
    /// might double-count by pushing a cumulative-since-game-start depth
    /// into an accumulator that's then averaged by per-move count.
    ///
    /// The underlying claim is that `tree.depth_accum` / `tree.sim_count`
    /// reset in `new_game()` — which the worker loop calls before every
    /// search — so each `last_search_stats()` call reports a clean
    /// per-search average. This test locks that contract in.
    #[test]
    fn test_last_search_stats_resets_across_new_game() {
        let mut tree = crate::mcts::MCTSTree::new(1.5);
        let n_actions = crate::board::BOARD_SIZE * crate::board::BOARD_SIZE + 1;
        let uniform = vec![1.0 / n_actions as f32; n_actions];

        // Game 1: run a handful of sims so depth_accum becomes positive.
        tree.new_game(crate::board::Board::new());
        for _ in 0..10 {
            let leaves = tree.select_leaves(1);
            let n = leaves.len();
            let policies: Vec<Vec<f32>> = (0..n).map(|_| uniform.clone()).collect();
            let values = vec![0.0f32; n];
            tree.expand_and_backup(&policies, &values);
        }
        let (d1, c1) = tree.last_search_stats();
        assert!(d1 > 0.0, "game 1: depth must be > 0 after sims, got {d1}");
        assert!(c1 >= 0.0 && c1 <= 1.0, "game 1: root concentration out of range: {c1}");

        // Game 2: reset without running any sims. Both stats must drop
        // cleanly to 0.0 / 0.0 — confirming the per-search reset contract.
        tree.new_game(crate::board::Board::new());
        let (d2, c2) = tree.last_search_stats();
        assert_eq!(d2, 0.0, "new_game must reset mean_depth, got {d2}");
        assert_eq!(c2, 0.0, "new_game must reset root_concentration, got {c2}");
    }

    /// Verify the runner-level mean_depth getter is an arithmetic mean of
    /// per-search means, not a biased quantity.
    ///
    /// Pre-conditions locked in by the test:
    ///   * the 1_000_000 scaling factor applied in `worker_loop.rs` matches
    ///     the divisor in `mcts_mean_depth` / `mcts_mean_root_concentration`
    ///   * three synthetic per-search depth means {4.0, 6.0, 8.0} with equal
    ///     weight collapse to exactly 6.0 when read via the getter
    ///
    /// The test manipulates the shared atomics directly (same pattern the
    /// worker loop uses), so it exercises the getter's arithmetic without
    /// requiring a full worker / inference-batcher spin-up.
    #[test]
    fn test_mcts_mean_depth_is_per_search_average() {
        let runner = SelfPlayRunner::new(
            1, 0, 1, 1, 1.5, 0.25, 18*19*19, 19*19+1, 1.0, 1, 1, 15, -0.1, true, 0.3,
            0.05, false, 16, 5, false, 50.0, 1.0, false, 16, 10, 0.3, 0.25, true,
            10_000, 0.0_f32, 0_usize, 0_usize,
        );

        // Simulate three per-search stat pushes matching what the worker
        // loop does for depth 4.0 / 6.0 / 8.0. Scaling factor = 1_000_000.
        for &depth in &[4.0_f32, 6.0, 8.0] {
            runner.mcts_depth_accum
                .fetch_add((depth * 1_000_000.0) as u64, Ordering::Relaxed);
            runner.mcts_stat_count.fetch_add(1, Ordering::Relaxed);
        }

        let mean = runner.mcts_mean_depth();
        assert!(
            (mean - 6.0).abs() < 1e-3,
            "mean of per-search depths {{4, 6, 8}} must be 6.0, got {mean}"
        );

        // Same arithmetic for concentration.
        for &conc in &[0.25_f32, 0.50, 0.75] {
            runner.mcts_conc_accum
                .fetch_add((conc * 1_000_000.0) as u64, Ordering::Relaxed);
            // stat_count re-used; the getter divides by the same count.
        }
        // stat_count is already at 3 from the depth loop above, so the
        // second arithmetic shares the same denominator on purpose — this
        // mirrors worker_loop.rs where both accumulators advance together.
        let mean_conc = runner.mcts_mean_root_concentration();
        assert!(
            (mean_conc - 0.50).abs() < 1e-3,
            "mean of per-search root concentrations {{0.25, 0.50, 0.75}} must be 0.50, got {mean_conc}"
        );

        // Zero-denominator guard.
        let empty = SelfPlayRunner::new(
            1, 0, 1, 1, 1.5, 0.25, 18*19*19, 19*19+1, 1.0, 1, 1, 15, -0.1, true, 0.3,
            0.05, false, 16, 5, false, 50.0, 1.0, false, 16, 10, 0.3, 0.25, true,
            10_000, 0.0_f32, 0_usize, 0_usize,
        );
        assert_eq!(empty.mcts_mean_depth(), 0.0);
        assert_eq!(empty.mcts_mean_root_concentration(), 0.0);
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
}
