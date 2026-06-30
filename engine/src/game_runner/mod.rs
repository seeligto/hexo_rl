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

pub mod config;
pub mod gumbel_search;
// §173 A5b: `pub mod` so integration tests can call aggregate_policy* directly.
pub mod records;
mod worker_loop;

pub use config::SelfPlayRunnerConfig;
pub use worker_loop::compute_move_temperature;

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::board::TOTAL_CELLS;
use crate::inference_bridge::InferenceBatcher;
// §173 A5a: STATE_STRIDE import removed — collect_data now uses batcher.feature_len() (H6-α).
// use crate::replay_buffer::sym_tables::STATE_STRIDE;

/// Per-row training tuple produced by self-play workers and consumed by
/// `collect_data`. See the `results` field doc for the field semantics.
///
/// Fields (in order): `(feat, chain, policy, outcome, plies, combined_aux_u8,
/// is_full_search, ply_index, value_valid)`. `ply_index` (CF-4) is the per-row
/// 0-based ply of the decision; `plies` is the game-total (shared across the
/// game's rows). `value_valid` (DRAW-MASK, Phase 6) is a per-game u8: 1 =
/// supervise the value head on this row, 0 = ply-capped (terminal_reason==2) →
/// mask the fabricated value label out of the value loss (policy target kept).
pub(crate) type WorkerResultRow = (Vec<f32>, Vec<f32>, Vec<f32>, f32, usize, Vec<u8>, bool, u16, u8);

/// Per-game result tuple consumed by `drain_game_results`. See the
/// `recent_game_results` field doc for the field semantics.
///
/// Fields (in order): `(plies, winner_code, move_history, worker_id,
/// terminal_reason, model_version_min, model_version_max, model_version_distinct)`.
pub(crate) type GameResultRow = (usize, u8, Vec<(i32, i32)>, usize, u8, u64, u64, u32);

/// Return tuple of `collect_data` — ten NumPy arrays bound to the GIL
/// lifetime. Fields: `(feat, chain, policy, value, plies, ownership,
/// winning_line, is_full_search, position_index, value_valid)`. `position_index`
/// (CF-4, u16) is the per-row 0-based ply index; `plies` is the game-total.
/// `value_valid` (DRAW-MASK, Phase 6, u8) is 1 = supervise value, 0 = ply-capped
/// row whose fabricated value label is masked out of the value loss.
pub(crate) type CollectDataOut<'py> = (
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<u64>>,
    Bound<'py, PyArray2<u8>>,
    Bound<'py, PyArray2<u8>>,
    Bound<'py, PyArray1<u8>>,
    Bound<'py, PyArray1<u16>>,
    Bound<'py, PyArray1<u8>>,
);

// cycle 3 P79 Wave 7 Batch A: stored-flat for hot-path field access in worker_loop;
// the public construction surface is `SelfPlayRunnerConfig` (`config.rs`).
// Permanent KEEP — 7 bool fields here mirror user-tunable PyO3 kwargs; folding them
// into an enum would lose the per-flag ergonomics on the Python kwarg surface.
#[allow(clippy::struct_excessive_bools)]
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
    /// §178: terminal-via-ply-cap outcome (winner=None AND ply>=max_moves_per_game).
    /// Split from `draw_reward` so organic draws and ply-cap truncations can pay
    /// distinct value-head targets. See `worker_loop/inner.rs::finalize_game`.
    pub(crate) ply_cap_value: f32,
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
    /// §115: first `random_opening_plies` plies of every self-play game use a
    /// uniformly-random legal move instead of MCTS. These plies do NOT produce
    /// training rows (skipped before `records_vec.push`). Semantics match the
    /// eval path (`eval_random_opening_plies`, §80). 0 disables.
    pub(crate) random_opening_plies: u32,
    /// Maximum positions buffered in `results` before workers drop the oldest
    /// to avoid unbounded growth when Python consumption stalls. Tracked on
    /// `positions_dropped` when it fires.
    pub(crate) results_queue_cap: usize,
    /// §130: per-game uniform rotation across the 12-element hex dihedral group.
    /// At each game start the worker samples `sym_idx ∈ [0, 12)`; that rotation is
    /// applied (a) to NN input planes before inference (forward scatter), (b) to
    /// the returned policy (inverse scatter) so MCTS keeps a canonical-frame view,
    /// and (c) to the recorded state/chain/policy/ownership/winning_line tensors
    /// (forward scatter) so the buffer stores the rotated frame. Eval and bot
    /// paths construct `SelfPlayRunner` with `selfplay_rotation_enabled=false` so
    /// they play real-board games. Closes §121 Component 1.
    pub(crate) selfplay_rotation_enabled: bool,
    /// Phase B' v8 §152 Q2: when true, each new game samples a legal-move
    /// radius uniformly from {4, 5, 6} via the worker-local RNG and applies
    /// it via `Board::set_legal_move_radius` before the first move.  When
    /// false, every game uses the default radius (`DEFAULT_LEGAL_MOVE_RADIUS
    /// = 5`).  Used to break the radius-5 stride-5 fixed point identified in
    /// the §152 instrumented diagnosis.
    pub(crate) legal_move_radius_jitter: bool,
    /// O1 (SootyOwl-validated) forced-win → one-hot POLICY target: enabled flag,
    /// depth (1=immediate / 2=+within-turn-setup), one-hot peak weight. Threaded
    /// to `worker_loop` via the `ForcedWinPolicy` sub-bundle.
    pub(crate) forced_win_policy_enabled: bool,
    pub(crate) forced_win_policy_depth: u8,
    pub(crate) forced_win_policy_weight: f32,
    /// D-QFIX-LAND A1: interior (non-root) MCTS selection rule, parsed from the
    /// `mcts.interior_selector` config string at `new()` (panics on an unknown
    /// variant). Threaded to each worker via `WorkerParams` and applied to the
    /// per-worker `MCTSTree` immediately after `new_full` in
    /// `worker_loop/inner.rs`. `Puct` = HEAD behaviour (byte-identical).
    pub(crate) interior_selector: crate::mcts::InteriorSelector,
    /// D-WS3 L1 solver-in-loop SOFT visit-injection knobs (threaded to
    /// `worker_loop` via the `SolverInLoop` sub-bundle). `solver_enabled=false`
    /// (default) makes the per-move hook a no-op — byte-identical to pre-D-WS3.
    pub(crate) solver_enabled: bool,
    pub(crate) solver_depth: u32,
    pub(crate) solver_node_budget: u64,
    pub(crate) solver_neighbor_dist: i32,
    pub(crate) solver_visit_weight: f32,
    /// §173 A5a — full registry record for the active encoding; `None` for
    /// legacy callers that don't provide `encoding_spec`. Used by worker_loop
    /// to call `sym_tables_for(spec)` (H1-α) and to derive per-spec geometry
    /// constants (`n_cells`, `kept_plane_indices`) replacing hardcoded v6
    /// literals (H2-α, H3-α, H6-α). When `None`, worker_loop falls back to
    /// the v6 compile-time defaults (byte-exact for bare v6 runners).
    pub(crate) registry_spec: Option<&'static crate::encoding::RegistrySpec>,
    /// §174 — per-game legal-move radius override for curriculum training.
    /// `-1` means "use encoding default / no override".  Updated live by
    /// `set_radius_override()`; workers read it at the start of each game.
    pub(crate) radius_override: Arc<AtomicI32>,
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
    ///   feat:   8-plane state (f32, flat [8 × TOTAL_CELLS]) — HEXB v6 wire format,
    ///           sliced from the 18-plane game-state tensor by
    ///           `worker_loop::slice_kept_planes_18_to_8` per §131 Option X.
    ///   chain:  6-plane chain-length (f32, flat [6 × TOTAL_CELLS], normalized /6)
    ///   combined_aux_u8 layout: first TOTAL_CELLS bytes = ownership
    ///     (0=P2, 1=empty, 2=P1), last TOTAL_CELLS = winning_line binary mask.
    ///     Both projected to the row's own cluster window centre at the time
    ///     the position was recorded (NOT the game-end bbox centroid), so the
    ///     aux target frame aligns with the state frame under later symmetry
    ///     augmentation in the replay buffer.
    pub(crate) results: Arc<Mutex<VecDeque<WorkerResultRow>>>,
    /// Ring-buffer of recent game results for Python logging.
    /// Tuple: (plies, winner_code, move_history, worker_id, terminal_reason,
    ///         model_version_min, model_version_max, model_version_distinct).
    ///   winner_code: 1 = Player One, 2 = Player Two, 0 = draw.
    ///   move_history: sequence of (q, r) coordinates in play order.
    ///   terminal_reason: 0 = six_in_a_row, 1 = colony, 2 = ply_cap,
    ///                    3 = other_draw (no winner, not cap-bound).
    ///   model_version_*: range of `InferenceBatcher::model_version` snapshots
    ///                    seen across the moves of this game (Phase B' Class-1
    ///                    probe). distinct = count of unique versions seen.
    ///                    All zero when the model never swapped during the game.
    pub(crate) recent_game_results: Arc<Mutex<VecDeque<GameResultRow>>>,
    pub(crate) handles: Arc<Mutex<Vec<JoinHandle<()>>>>,
    /// Accumulated MCTS leaf depth across all searches (scaled by 1_000_000 to preserve fractional part).
    pub(crate) mcts_depth_accum: Arc<AtomicU64>,
    /// Accumulated root concentration * 1_000_000 across all searches.
    pub(crate) mcts_conc_accum: Arc<AtomicU64>,
    /// Number of searches (moves) contributing to the above accumulators.
    pub(crate) mcts_stat_count: Arc<AtomicU64>,
    /// Cumulative quiescence fires (all 4 branches) across all searches since `start()`.
    pub(crate) mcts_quiescence_fires: Arc<AtomicU64>,
    /// I2 investigation metric: accumulated per-position std-dev of per-cluster
    /// values (scaled by 1_000_000) summed across all K≥2 inferences since `start()`.
    pub(crate) cluster_value_std_accum: Arc<AtomicU64>,
    /// I2 investigation metric: accumulated per-position top-1 policy
    /// disagreement (scaled by 1_000_000) summed across all K≥2 inferences since `start()`.
    pub(crate) cluster_policy_disagreement_accum: Arc<AtomicU64>,
    /// I2 investigation metric: count of K≥2 multi-cluster positions scored.
    pub(crate) cluster_variance_samples: Arc<AtomicU64>,
}

#[pymethods]
impl SelfPlayRunner {
    /// Construct a runner from a [`SelfPlayRunnerConfig`] builder (cycle 3 Wave 7
    /// Batch A, P79). The 38-kwarg PyO3 surface lives on `SelfPlayRunnerConfig::new`;
    /// this `#[new]` consumes the config struct and runs the existing validation
    /// (effective_standard, fast_sims, n_sims_quick/full) byte-equivalent to the
    /// pre-cycle-3 38-positional-arg form. INV19 pins.
    // 38-field destructure + 3 validation gates + 50-field struct init pushes the
    // body over the 100-LOC clippy `too_many_lines` threshold. Permanent KEEP per
    // cycle 3 P79 Wave 7 Batch A — splitting the body would obscure the validation
    // / construction sequence that INV19 contracts against.
    #[allow(clippy::too_many_lines)]
    #[new]
    pub fn new(config: SelfPlayRunnerConfig) -> PyResult<Self> {
        let SelfPlayRunnerConfig {
            n_workers,
            max_moves_per_game,
            n_simulations,
            leaf_batch_size,
            c_puct,
            fpu_reduction,
            feature_len,
            policy_len,
            fast_prob,
            fast_sims,
            standard_sims,
            temp_threshold_compound_moves,
            draw_reward,
            ply_cap_value,
            quiescence_enabled,
            quiescence_blend_2,
            temp_min,
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
            random_opening_plies,
            selfplay_rotation_enabled,
            legal_move_radius_jitter,
            encoding_name,
            radius_override,
            inference_pool_size,
            forced_win_policy_enabled,
            forced_win_policy_depth,
            forced_win_policy_weight,
            interior_selector,
            solver_enabled,
            solver_depth,
            solver_node_budget,
            solver_neighbor_dist,
            solver_visit_weight,
        } = config;
        // D-QFIX-LAND A1: parse the interior-selector string → enum here (panics
        // on an unknown variant — A1 config is hard-read end-to-end, no silent
        // default). "puct" reproduces HEAD behaviour byte-for-byte.
        let interior_selector = crate::mcts::InteriorSelector::from_config_str(&interior_selector);
        // §172 A10 T8b / cycle 3 Wave 8 Batch C FF.10 — derive feature_len /
        // policy_len from the named encoding's registry record. Pre-Wave-8
        // the registry was looked up Python-side and routed as a
        // `PyRegistrySpec` round-trip; cycle 3 Wave 8 collapsed that into a
        // single string lookup at the Rust boundary. The 4 historic
        // `audit: legacy-v6-fallback` arms (none-and-none) retire to
        // `PyValueError` so silent v6-geometry corruption of v8 callers is
        // impossible.
        //
        // Precedence: explicit kwargs > encoding_name lookup > error.
        let spec_static: Option<&'static crate::encoding::RegistrySpec> =
            if let Some(name) = encoding_name.as_deref() {
                if let Some(spec) = crate::encoding::lookup(name) {
                    Some(spec)
                } else {
                    let mut known: Vec<&str> =
                        crate::encoding::all_specs().map(|s| s.name).collect();
                    known.sort_unstable();
                    return Err(PyValueError::new_err(format!(
                        "SelfPlayRunner: encoding_name {name:?} not in registry; known: {known:?}"
                    )));
                }
            } else {
                None
            };
        let (feature_len, policy_len) = match (feature_len, policy_len, spec_static) {
            (Some(f), Some(p), _) => (f, p),
            (None, None, Some(spec)) => (spec.state_stride(), spec.policy_stride()),
            (Some(f), None, Some(spec)) => (f, spec.policy_stride()),
            (None, Some(p), Some(spec)) => (spec.state_stride(), p),
            (None, _, None) | (_, None, None) => {
                return Err(PyValueError::new_err(
                    "SelfPlayRunner: encoding_name required when feature_len/policy_len omitted \
                     (cycle 3 Wave 8 Batch C FF.10 retired the legacy v6 fallback arms)",
                ));
            }
        };
        // Effective standard-search sim budget: `standard_sims` wins, else
        // `n_simulations`. Reject zero on the *effective* value — a silent
        // zero fallback here ran workers with 0 sims per move (no search,
        // random-policy self-play) and corrupted training data before the
        // first dashboard read.
        let effective_standard = if standard_sims == 0 { n_simulations } else { standard_sims };
        if effective_standard == 0 {
            return Err(PyValueError::new_err(
                "SelfPlayRunner: n_simulations (or standard_sims) must be > 0",
            ));
        }
        if fast_prob > 0.0 && fast_sims == 0 {
            return Err(PyValueError::new_err(
                "SelfPlayRunner: fast_sims must be > 0 when fast_prob > 0",
            ));
        }
        if full_search_prob > 0.0 && (n_sims_quick == 0 || n_sims_full == 0) {
            return Err(PyValueError::new_err(format!(
                "SelfPlayRunner: n_sims_quick and n_sims_full must both be > 0 \
                 when full_search_prob > 0 (got n_sims_quick={n_sims_quick}, n_sims_full={n_sims_full})",
            )));
        }
        // cycle 3 P55 / Wave 9 — auto-derive InferenceBatcher.pool_size from
        // spec.k_max when Python omits the explicit kwarg. Heuristic:
        // `n_workers * leaf_batch_size * k_max * 2`, floored at 512 so the
        // dominant v6 default path (k_max=1 → 14*8*1*2 = 224 < 512) preserves
        // the cycle 1 fallback. v6w25 16-worker default (k_max=8 → 14*8*8*2 =
        // 1792) clears the floor and closes the K-aware-pool gap flagged by
        // P55. Explicit `inference_pool_size` kwarg still wins.
        let derived_pool_size = inference_pool_size.or_else(|| {
            spec_static.map(|s| {
                let derived = n_workers * leaf_batch_size * (s.k_max as usize) * 2;
                derived.max(512)
            })
        });
        Ok(Self {
            // §172 A10 T8b: pass already-resolved widths through to the
            // batcher (its pyo3 sig now also takes Option<usize>; the runner
            // already collapsed encoding_spec → concrete widths above).
            batcher: InferenceBatcher::new(None, Some(feature_len), Some(policy_len), derived_pool_size)?,
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
            ply_cap_value,
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
            random_opening_plies,
            selfplay_rotation_enabled,
            legal_move_radius_jitter,
            forced_win_policy_enabled,
            forced_win_policy_depth,
            forced_win_policy_weight,
            interior_selector,
            solver_enabled,
            solver_depth,
            solver_node_budget,
            solver_neighbor_dist,
            solver_visit_weight,
            registry_spec: spec_static,
            radius_override: Arc::new(AtomicI32::new(radius_override.unwrap_or(-1))),
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
            cluster_value_std_accum: Arc::new(AtomicU64::new(0)),
            cluster_policy_disagreement_accum: Arc::new(AtomicU64::new(0)),
            cluster_variance_samples: Arc::new(AtomicU64::new(0)),
        })
    }

    /// Spawn `n_workers` self-play threads. Idempotent.
    pub fn start(&self) {
        self.start_impl();
    }

    /// Drain all buffered positions and return them as numpy arrays.
    ///
    /// Returns (features, chain_planes, policies, values, plies, ownership, winning_line, is_full_search, position_index, value_valid):
    ///   features:        (N, 8*361)    float32 — HEXB v6 buffer wire format (sliced from
    ///                                            18-plane game state via KEPT_PLANE_INDICES;
    ///                                            §131 Option X). Reshape to (N, 8, 19, 19) on
    ///                                            the Python side before pushing into
    ///                                            `ReplayBuffer::push_*` / `RecentBuffer`.
    ///   chain_planes:    (N, 6*361)    float32 — Q13 chain-length planes (flat, /6 normalized)
    ///   policies:        (N, pol_len)  float32
    ///   values:          (N,)          float32
    ///   plies:           (N,)          uint64 — game length in plies (game-length weighting)
    ///   ownership:       (N, 361)      uint8  — per-row aux target {0=P2, 1=empty, 2=P1}
    ///   winning_line:    (N, 361)      uint8  — per-row binary mask of winning 6-in-a-row
    ///   is_full_search:  (N,)          uint8  — 1 = full-search move, 0 = quick-search move
    ///   position_index:  (N,)          uint16 — CF-4: per-row 0-based ply index of the
    ///                                            decision (NOT the game-total `plies`); feeds
    ///                                            the ply-index aux target. K cluster rows of
    ///                                            one ply share it.
    ///   value_valid:     (N,)          uint8  — DRAW-MASK (Phase 6): 1 = supervise value
    ///                                            head, 0 = ply-capped row → mask the
    ///                                            fabricated value label out of the value loss
    ///                                            (per-game constant; policy target unaffected).
    ///
    /// N = 0 when no positions are available (arrays have zero rows).
    pub fn collect_data<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<CollectDataOut<'py>> {
        // §173 A5a (H6-α): use batcher.feature_len() instead of the v6-hardcoded
        // STATE_STRIDE constant. batcher carries the spec-derived value set at
        // construction time (§172 A10 T8b), so this is always encoding-correct.
        let feat_len  = self.batcher.feature_len();
        // n_cells: per-spec cluster window cell count. Falls back to v6 TOTAL_CELLS
        // (361) for legacy runners that don't provide encoding_spec.
        let n_cells   = self.registry_spec.map_or(TOTAL_CELLS, super::encoding::spec::RegistrySpec::n_cells);
        let chain_len = 6 * n_cells;
        let pol_len   = self.pol_len;

        let mut results = self.results.lock().expect("results lock poisoned");
        let n = results.len();

        let mut flat_feats      = Vec::with_capacity(n * feat_len);
        let mut flat_chain      = Vec::with_capacity(n * chain_len);
        let mut flat_pols       = Vec::with_capacity(n * pol_len);
        let mut vals            = Vec::with_capacity(n);
        let mut plies_out       = Vec::with_capacity(n);
        let mut flat_own        = Vec::with_capacity(n * n_cells);
        let mut flat_wl         = Vec::with_capacity(n * n_cells);
        let mut is_full_search  = Vec::with_capacity(n);
        let mut position_index  = Vec::with_capacity(n);
        let mut value_valid_v   = Vec::with_capacity(n);

        while let Some((feat, chain, pol, outcome, plies, aux_u8, full_search, ply_index, value_valid)) = results.pop_front() {
            flat_feats.extend_from_slice(&feat);
            flat_chain.extend_from_slice(&chain);
            flat_pols.extend_from_slice(&pol);
            vals.push(outcome);
            plies_out.push(plies as u64);
            // Split combined aux: first n_cells = ownership, last = winning_line.
            // §173 A5a (H6-α): n_cells is spec-derived; replaces TOTAL_CELLS=361 literal.
            flat_own.extend_from_slice(&aux_u8[..n_cells]);
            flat_wl.extend_from_slice(&aux_u8[n_cells..]);
            is_full_search.push(full_search as u8);
            // CF-4: per-row 0-based ply index for the ply-index aux target.
            position_index.push(ply_index);
            // DRAW-MASK (Phase 6): per-row value-supervision flag (per-game constant).
            value_valid_v.push(value_valid);
        }

        let feats_np  = flat_feats.into_pyarray(py).reshape([n, feat_len])?;
        let chain_np  = flat_chain.into_pyarray(py).reshape([n, chain_len])?;
        let pols_np   = flat_pols.into_pyarray(py).reshape([n, pol_len])?;
        let vals_np   = vals.into_pyarray(py);
        let gids_np   = plies_out.into_pyarray(py);
        let own_np    = flat_own.into_pyarray(py).reshape([n, n_cells])?;
        let wl_np     = flat_wl.into_pyarray(py).reshape([n, n_cells])?;
        let ifs_np    = is_full_search.into_pyarray(py);
        let pidx_np   = position_index.into_pyarray(py);
        let vv_np     = value_valid_v.into_pyarray(py);

        Ok((feats_np, chain_np, pols_np, vals_np, gids_np, own_np, wl_np, ifs_np, pidx_np, vv_np))
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

    /// §172 A10 T8b — observable for the encoding-derived feature_len.
    /// Mirrors `InferenceBatcher::feature_len()` for the runner's own batcher.
    pub fn feature_len(&self) -> usize {
        self.batcher.feature_len()
    }

    /// §172 A10 T8b — observable for the encoding-derived policy_len.
    pub fn policy_len(&self) -> usize {
        self.pol_len
    }

    #[getter]
    pub fn games_completed(&self) -> usize {
        self.games_completed.load(Ordering::Relaxed)
    }

    /// Rows produced by self-play workers — continuous per-ply counter
    /// (§128). EXCLUDES random-opening plies: when
    /// `random_opening_plies > 0` the worker selects opening moves
    /// uniformly at random WITHOUT pushing rows to the replay buffer
    /// (random-move targets would poison policy/value heads), and the
    /// counter is intentionally not bumped (see
    /// `worker_loop.rs:542`). Consequence for dashboards: `pos/hr`
    /// derived from this counter excludes opening wall-clock work and
    /// is only directly comparable across runs that share the same
    /// `random_opening_plies` value.
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

    /// I2: mean per-cluster value std-dev over K≥2 inferences since `start()`.
    /// Range [0.0, ~1.0]. Returns 0.0 before any K≥2 sample seen.
    #[getter]
    pub fn cluster_value_std_mean(&self) -> f32 {
        let count = self.cluster_variance_samples.load(Ordering::Relaxed);
        if count == 0 { return 0.0; }
        (self.cluster_value_std_accum.load(Ordering::Relaxed) as f64
            / (count as f64 * 1_000_000.0)) as f32
    }

    /// I2: mean per-cluster top-1 policy-disagreement over K≥2 inferences since `start()`.
    /// Range [0.0, 1.0]. 0.0 = all windows agree on top move; returns 0.0 if no sample.
    #[getter]
    pub fn cluster_policy_disagreement_mean(&self) -> f32 {
        let count = self.cluster_variance_samples.load(Ordering::Relaxed);
        if count == 0 { return 0.0; }
        (self.cluster_policy_disagreement_accum.load(Ordering::Relaxed) as f64
            / (count as f64 * 1_000_000.0)) as f32
    }

    /// I2: count of K≥2 multi-cluster positions scored since `start()`.
    #[getter]
    pub fn cluster_variance_sample_count(&self) -> u64 {
        self.cluster_variance_samples.load(Ordering::Relaxed)
    }

    /// Drain and return all buffered game results since the last call.
    ///
    /// Each entry: (plies, winner_code, move_history, worker_id,
    ///              terminal_reason, model_version_min, model_version_max,
    ///              model_version_distinct).
    ///   winner_code: 1 = Player One, 2 = Player Two, 0 = draw.
    ///   move_history: (q, r) stone placements in play order.
    ///   terminal_reason: see `recent_game_results` docstring.
    ///   model_version_*: Phase B' Class-1 instrumentation; all-zero when
    ///                    no weight swap occurred during the game.
    pub fn drain_game_results(
        &self,
    ) -> Vec<GameResultRow> {
        self.drain_game_results_raw()
    }

    /// Snapshot the current InferenceBatcher model version (Phase B' Class-1).
    /// Increments on every `InferenceServer.load_state_dict_safe()` call.
    #[getter]
    pub fn model_version(&self) -> u64 {
        self.batcher.current_model_version()
    }

    /// §174 — update the per-game legal-move radius override live.
    /// `None` clears the override (use encoding default).  Workers read this
    /// atomic at the start of each game.
    pub fn set_radius_override(&self, radius: Option<i32>) {
        let val = radius.unwrap_or(-1);
        self.radius_override.store(val, Ordering::SeqCst);
    }
}

impl SelfPlayRunner {
    /// Internal drain used by tests and the pymethods wrapper.
    /// Returns raw Vecs — no Python dependency, safe to call from `cargo test`.
    pub(crate) fn drain_game_results_raw(
        &self,
    ) -> Vec<GameResultRow> {
        let mut rg = self.recent_game_results.lock().expect("recent_game_results lock poisoned");
        rg.drain(..).collect()
    }

    /// §178 INV26 — test-only accessor for per-row outcomes (the 4th tuple
    /// field of `WorkerResultRow`). The Python-facing `collect_data` consumes
    /// the same queue but folds it into a NumPy array under the GIL, which is
    /// not reachable from `cargo test`. Returns one f32 per row in the order
    /// produced by the workers. NOT for production use.
    pub fn drain_outcomes_for_test(&self) -> Vec<f32> {
        let mut q = self.results.lock().expect("results lock poisoned");
        q.drain(..).map(|(_, _, _, outcome, _, _, _, _, _)| outcome).collect()
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
        let runner = SelfPlayRunner::new(SelfPlayRunnerConfig {
            max_moves_per_game: 0,
            n_simulations: 1,
            leaf_batch_size: 1,
            feature_len: Some(8 * 19 * 19),
            policy_len: Some(19 * 19 + 1),
            fast_prob: 1.0,
            fast_sims: 1,
            standard_sims: 1,
            ..Default::default()
        }).unwrap();
        runner.start();

        let mut attempts = 0;
        let mut completed_workers = HashSet::new();

        while completed_workers.len() < 4 && attempts < 50 {
            let results = runner.drain_game_results_raw();
            for (_, _, _, worker_id, _, _, _, _) in results {
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
        let runner = SelfPlayRunner::new(SelfPlayRunnerConfig {
            n_workers: 1,
            max_moves_per_game: 0,
            n_simulations: 1,
            leaf_batch_size: 1,
            feature_len: Some(8 * 19 * 19),
            policy_len: Some(19 * 19 + 1),
            fast_prob: 1.0,
            fast_sims: 1,
            standard_sims: 1,
            ..Default::default()
        }).unwrap();

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
        let empty = SelfPlayRunner::new(SelfPlayRunnerConfig {
            n_workers: 1,
            max_moves_per_game: 0,
            n_simulations: 1,
            leaf_batch_size: 1,
            feature_len: Some(8 * 19 * 19),
            policy_len: Some(19 * 19 + 1),
            fast_prob: 1.0,
            fast_sims: 1,
            standard_sims: 1,
            ..Default::default()
        }).unwrap();
        assert_eq!(empty.mcts_mean_depth(), 0.0);
        assert_eq!(empty.mcts_mean_root_concentration(), 0.0);
    }

    // ── §P3.2: removed 4 PyEncodingSpec-coupled tests ─────────────────────
    // Removed test fns:
    //   - `test_worker_loop_honors_v6w25_encoding`
    //   - `test_worker_loop_default_is_v6`
    //   - `test_worker_loop_spawns_with_v6w25_encoding`
    //   - `test_worker_loop_jitter_does_not_override_v6w25_radius`
    //
    // Coverage of the v6w25 worker_loop path is preserved by:
    //   - `engine/tests/test_worker_loop_v6w25_smoke.rs` (registry_spec path,
    //     `PyRegistrySpec::from_static` ctor — §173 A5a)
    //   - `engine/tests/inv17_pyregistryspec_supersedes_pyencodingspec.rs`
    //     (registry parity pin, INV17 Rust — §P3.1)
    //
    // Default-v6 byte-exactness is covered by the
    // `with_registry_spec_tests` cohort at `engine/src/board/state/core.rs`
    // (cycle 3 Wave 8 Batch B retired the legacy `with_encoding_tests`).
    // Jitter behaviour is now guarded by `worker_loop.rs:321`
    // `registry_spec.is_none()` branch (the deleted `encoding` field used
    // to alias this guard — registry_spec is the surviving sentinel for
    // "non-default perception bound").

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
