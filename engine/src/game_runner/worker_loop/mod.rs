//! Self-play worker loop — spawns `n_workers` threads that each play full
//! games and push recorded (feature, policy, outcome, aux) rows into the
//! shared results queue.
//!
//! This is the hottest code path in the engine. The PyO3 `start()` facade in
//! `game_runner/mod.rs` is a thin wrapper over `SelfPlayRunner::start_impl`
//! defined here. End-of-game ownership / winning-line reprojection has been
//! extracted to `records::reproject_game_end_row` so the inner loop stays
//! under the per-file size budget.
//!
//! **Wave 10 Batch A split (cycle 3, 2026-05-17):** the pre-split
//! `worker_loop.rs` (1129 LOC) decomposes into 7 sibling modules:
//!   - `mod.rs` — `SelfPlayRunner::start_impl` orchestration head (this
//!     file); pub-uses `compute_move_temperature` so the outer
//!     `pub use worker_loop::compute_move_temperature` chain at
//!     `engine/src/game_runner/mod.rs:19` keeps working.
//!   - `rotate.rs` — 5 `#[inline]` rotation helpers + temperature schedule.
//!   - `params.rs` — `WorkerParams` + 3 Wave 7 Batch C themed flag bundles.
//!   - `channels.rs` — `WorkerChannels` (batcher + result queues).
//!   - `stats.rs` — `WorkerStats` (13 `Arc<AtomicU*>` accumulators).
//!   - `atomics.rs` — `WorkerAtomics` (running + radius_override).
//!   - `inner.rs` — `run_worker_thread` (per-worker hot loop body; named
//!     fn extracted from the pre-split closure body).
//!
//! Byte-identity-on-behavior preserved: Wave 7 Batch C destructure pattern
//! verbatim; all 5 `#[inline]` attributes intact; INV25 pin asserts the
//! split holds.

mod atomics;
mod channels;
mod inner;
mod params;
mod rotate;
mod stats;

pub use rotate::compute_move_temperature;

use std::sync::atomic::Ordering;
use std::thread;

use crate::board::BOARD_SIZE;
use crate::replay_buffer::sym_tables::{SymTables, sym_tables_for};

use super::SelfPlayRunner;

use atomics::WorkerAtomics;
use channels::WorkerChannels;
use params::{
    ExplorationFlags, MoveConstraintFlags, SearchFlags, WorkerGeometry, WorkerParams,
};
use stats::WorkerStats;

impl SelfPlayRunner {
    /// Spawn `n_workers` self-play threads. Idempotent: a second call while
    /// already running is a no-op.
    ///
    /// Each worker owns its own `MCTSTree`, RNG, and per-game state. Shared
    /// state is accessed through the `Arc` fields on `SelfPlayRunner`:
    /// `running` (kill switch), `games_completed` / `positions_generated` /
    /// `x_wins` / `o_wins` / `draws` / `results` / `recent_game_results` /
    /// `mcts_*_accum` (stats dashboards). All workers are joined by `stop()`.
    pub(crate) fn start_impl(&self) {
        if self.running.swap(true, Ordering::SeqCst) {
            return;
        }

        // Defense-in-depth mutex (§100): game-level (`fast_prob`) and
        // move-level (`full_search_prob`) playout-cap randomisers must not
        // both be active.  Python pool init raises first; this panic only
        // fires if `SelfPlayRunner` is driven from Rust or a path that
        // bypasses the Python validator.
        assert!(
            !(self.fast_prob > 0.0 && self.full_search_prob > 0.0),
            "playout-cap mutex violated: fast_prob={} and full_search_prob={} \
             are both > 0 (§100 — game-level and move-level caps are mutually \
             exclusive)",
            self.fast_prob,
            self.full_search_prob,
        );

        // §130: build the 12-fold dihedral scatter tables once and share via a
        // `&'static SymTables`. SymTables construction is O(N_CELLS × N_SYMS)
        // ≈ 4 µs; lazy-init means the FIRST `start()` per encoding pays it and
        // every subsequent `start()` reuses the same shared singleton (no Arc
        // allocation, no `Arc::clone` per worker spawn).
        //
        // §173 A5a (H1-α): when a registry spec is present, `sym_tables_for(spec)`
        // returns the spec-keyed singleton (size_19/{8|11}, size_25/{8|11}).
        // Cycle 3 Wave 8 Batch C (FF.10): legacy no-spec runners fall back to
        // the v6 registry spec singleton via `sym_tables_for(lookup_or_panic("v6"))`.
        // The historical `sym_tables_v6_default()` accessor retired alongside
        // the `audit: legacy-v6-fallback` arms in `SelfPlayRunner::new` /
        // `InferenceBatcher::new`.
        let sym_tables_static: &'static SymTables = match self.registry_spec {
            Some(spec) => sym_tables_for(spec),
            None       => sym_tables_for(crate::encoding::registry::lookup_or_panic("v6")),
        };

        // §P52: build the per-worker capture prototype ONCE before the spawn
        // loop. Each iteration calls `.clone()` on each sub-struct (cheap:
        // 1 `Arc::clone` per field for Arc-typed members, scalar copy for
        // Params), collapsing the historical 35-line per-spawn clone block to
        // 4 group clones. Body of the spawn closure destructures the bundle
        // back into the same identifier names so the hot-loop body is byte-
        // identical to the pre-P52 source.
        // Wave 10 Batch A: prototype build extracted to `build_worker_prototypes`
        // so `start_impl` stays under the clippy::too_many_lines threshold.
        let (stats_proto, atomics_proto, channels_proto, params_proto) =
            self.build_worker_prototypes();

        let mut handles = self.handles.lock().expect("runner handles lock poisoned");
        for worker_id in 0..self.n_workers {
            // §P52: per-worker bundle clone — 4 sub-struct clones replace the
            // historical 35-line `let x = self.x.clone();` block.
            let stats = stats_proto.clone();
            let atomics = atomics_proto.clone();
            let channels = channels_proto.clone();
            let params = params_proto.clone();

            // §173 A5a (H2-α, H3-α): per-spec geometry pre-extracted once before
            // thread spawn so workers don't re-derive on every hot iteration.
            // `n_cells` = trunk_size² (cluster window cells per view); used to
            // replace hardcoded SYM_N_CELLS=361 and TOTAL_CELLS=361 in rotation
            // helpers and buffer sizing. Falls back to v6 default (361) for
            // legacy runners that don't supply encoding_spec.
            // `kept_planes` = &'static slice of source-plane indices retained by
            // this encoding; replaces the hardcoded KEPT_PLANE_INDICES import.
            // Falls back to the v6 constant (len=8, [0,1,2,3,8,9,10,11]).
            let (n_cells, kept_planes): (usize, &'static [usize]) = if let Some(spec) = self.registry_spec { (spec.n_cells(), spec.kept_plane_indices) } else {
                use crate::replay_buffer::sym_tables::N_CELLS;
                // KEPT_PLANE_INDICES is a `const` array — &KEPT_PLANE_INDICES
                // promotes to &'static [usize] via const-to-static coercion.
                const KPI: &[usize] = &crate::replay_buffer::sym_tables::KEPT_PLANE_INDICES;
                (N_CELLS, KPI)
            };
            // §173 A5b (H4-α): encoding geometry pre-extracted once before thread
            // spawn so per-sim hot path passes cheap integer pairs instead of
            // copying the full RegistrySpec struct (~174 B) on every
            // aggregate_policy* call. `policy_stride` = n_actions per call site;
            // `agg_trunk_sz` = trunk_size as i32 for window-bound arithmetic.
            // §P2: `has_pass_slot` added so records::aggregate_policy* can gate
            // the pass-slot skip + zero-write at the tail index. v6/v6w25/v7full
            // = true; v8/v8_canvas_realness = false. v6 default fallback for
            // legacy SelfPlayRunner constructions without registry_spec.
            let policy_stride: usize = match self.registry_spec {
                Some(s) => s.policy_stride(),
                None => BOARD_SIZE * BOARD_SIZE + 1,
            };
            let agg_trunk_sz: i32 = match self.registry_spec {
                Some(s) => s.trunk_size as i32,
                None => BOARD_SIZE as i32,
            };
            let has_pass_slot: bool = match self.registry_spec {
                Some(s) => s.has_pass_slot,
                None => true, // v6 default
            };
            // Wave 10 Batch A: bundle 5 per-worker geometry scalars into
            // `WorkerGeometry` (`Copy`, ~32 B) so `run_worker_thread` arity
            // stays ≤ 7 (avoids clippy::too_many_arguments; preserves F1
            // 18 → 18 per U10/J.2.a). Destructured back to local scalars
            // at fn entry — per-sim hot path remains scalar-API.
            let geometry = WorkerGeometry {
                n_cells,
                kept_planes,
                policy_stride,
                agg_trunk_sz,
                has_pass_slot,
            };
            // §P51: `sym_tables_static` is `&'static SymTables` (Copy); each
            // closure captures it by `move` with zero allocation cost. No
            // `Arc::clone` per worker spawn (cycle 1 ran one Arc clone here).
            let sym_tables = sym_tables_static;

            let handle = thread::spawn(move || {
                inner::run_worker_thread(
                    worker_id,
                    stats,
                    atomics,
                    channels,
                    params,
                    sym_tables,
                    geometry,
                );
            });
            handles.push(handle);
        }
    }

    /// Build the §P52 4-bundle capture prototype for the worker spawn loop.
    /// Extracted from `start_impl` (Wave 10 Batch A) to keep `start_impl`
    /// under the clippy::too_many_lines threshold without a new suppression
    /// — preserves the Wave 9 close baseline lint-suppression count across
    /// the Wave 10 split per U10 binding (PREP §M.4).
    fn build_worker_prototypes(
        &self,
    ) -> (WorkerStats, WorkerAtomics, WorkerChannels, WorkerParams) {
        let stats_proto = WorkerStats {
            games_completed: self.games_completed.clone(),
            positions_generated: self.positions_generated.clone(),
            x_wins: self.x_wins.clone(),
            o_wins: self.o_wins.clone(),
            draws: self.draws.clone(),
            positions_dropped: self.positions_dropped.clone(),
            mcts_depth_accum: self.mcts_depth_accum.clone(),
            mcts_conc_accum: self.mcts_conc_accum.clone(),
            mcts_stat_count: self.mcts_stat_count.clone(),
            mcts_quiescence_fires: self.mcts_quiescence_fires.clone(),
            cluster_value_std_accum: self.cluster_value_std_accum.clone(),
            cluster_policy_disagreement_accum: self.cluster_policy_disagreement_accum.clone(),
            cluster_variance_samples: self.cluster_variance_samples.clone(),
        };
        let atomics_proto = WorkerAtomics {
            running: self.running.clone(),
            radius_override: self.radius_override.clone(),
        };
        let channels_proto = WorkerChannels {
            batcher: self.batcher.clone(),
            results_queue: self.results.clone(),
            recent_game_results: self.recent_game_results.clone(),
        };
        let params_proto = WorkerParams {
            max_moves: self.max_moves_per_game,
            leaf_batch_size: self.leaf_batch_size,
            c_puct: self.c_puct,
            fpu_reduction: self.fpu_reduction,
            quiescence_blend_2: self.quiescence_blend_2,
            fast_prob: self.fast_prob,
            fast_sims: self.fast_sims,
            standard_sims: self.standard_sims,
            temp_threshold: self.temp_threshold_compound_moves,
            temp_min: self.temp_min,
            draw_reward: self.draw_reward,
            ply_cap_value: self.ply_cap_value,
            zoi_lookback: self.zoi_lookback,
            zoi_margin: self.zoi_margin,
            c_visit: self.c_visit,
            c_scale: self.c_scale,
            gumbel_m: self.gumbel_m,
            gumbel_explore_moves: self.gumbel_explore_moves,
            dirichlet_alpha: self.dirichlet_alpha,
            dirichlet_epsilon: self.dirichlet_epsilon,
            results_queue_cap: self.results_queue_cap,
            full_search_prob: self.full_search_prob,
            n_sims_quick: self.n_sims_quick,
            n_sims_full: self.n_sims_full,
            random_opening_plies: self.random_opening_plies,
            registry_spec: self.registry_spec,
            search_flags: SearchFlags {
                quiescence_enabled: self.quiescence_enabled,
                completed_q_values: self.completed_q_values,
                gumbel_mcts: self.gumbel_mcts,
            },
            exploration_flags: ExplorationFlags {
                dirichlet_enabled: self.dirichlet_enabled,
                selfplay_rotation_enabled: self.selfplay_rotation_enabled,
            },
            move_constraint_flags: MoveConstraintFlags {
                zoi_enabled: self.zoi_enabled,
                legal_move_radius_jitter: self.legal_move_radius_jitter,
            },
        };
        (stats_proto, atomics_proto, channels_proto, params_proto)
    }
}
