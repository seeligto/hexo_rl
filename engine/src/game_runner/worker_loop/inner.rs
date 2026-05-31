//! Per-worker hot loop body (cycle 3 Wave 10 Batch A; sub-fn extraction
//! cycle 3 Wave 11 Batch B / J.2.b).
//!
//! `run_worker_thread` is the hottest code path in the engine. Extracted
//! verbatim from the pre-split `worker_loop.rs::start_impl` closure body
//! (pre-split L421-1124) via a named-fn boundary so:
//!   - stack traces surface the worker body by name (debugging benefit);
//!   - future inline `#[cfg(test)]` blocks (P69 Batch B+) can target the
//!     extracted fn's helpers.
//!
//! Byte-identity-on-behavior preserved (Wave 10 PREP §A.5):
//!   - Wave 7 Batch C destructure pattern verbatim at fn entry;
//!   - all 5 `#[inline]` rotate helpers called via `use super::rotate::*`
//!     (LLVM cross-module inlining on `--release` preserves the effect;
//!     pre-registered fragile claim L.4);
//!   - per-game loop body (~631 LOC) ports verbatim.
//!
//! Wave 11 Batch B (J.2.b) extraction (2026-05-17):
//!   - 7 sub-fns extracted: `init_per_game_board` (warm), `infer_and_expand`
//!     (HOT; was closure, now `#[inline]` fn), `run_mcts_search` (HOT),
//!     `play_one_move` (warm; per-move dispatcher), `record_position` (warm),
//!     `finalize_game` (warm), `run_one_game` (warm; per-game body).
//!   - 4 helper bundles (`InferContext`, `ClusterVarianceAtomics`,
//!     `MoveAccumulators`, `MovePlayContext`) keep sub-fn arities under the
//!     `clippy::too_many_arguments` threshold without adding new `#[allow]`
//!     attributes (preserves F1 retirement target per PREP §D.2).
//!   - Parent `#[allow(clippy::too_many_lines)]` (pre-Wave-11 at L52) RETIRED.
//!     Parent body now <100 LOC orchestrator under the clippy threshold.
//!   - INV25 byte-identity-on-behavior contract preserved: the destructure
//!     pattern at parent fn entry is untouched; sub-fns receive destructured
//!     scalar locals as explicit args.

use std::collections::VecDeque;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU64, AtomicUsize, Ordering};

use rand::prelude::IndexedRandom;
use rand::rngs::ThreadRng;
use rand::{rng, RngExt};

use crate::board::{Board, hex_distance};
use crate::board::encode_chain_planes;
use crate::inference_bridge::InferenceBatcher;
use crate::mcts::MCTSTree;
use crate::replay_buffer::sym_tables::{N_SYMS, SymTables};

use super::super::gumbel_search::GumbelSearchState;
use super::super::{GameResultRow, WorkerResultRow, records};

use super::atomics::WorkerAtomics;
use super::channels::WorkerChannels;
use super::params::{
    ExplorationFlags, MoveConstraintFlags, SearchFlags, WorkerGeometry, WorkerParams,
};
use super::rotate::{
    compute_move_temperature, inv_sym_idx, rotate_aux_inplace, rotate_chain_inplace,
    rotate_policy_inplace, rotate_state_inplace,
};
use super::stats::WorkerStats;

/// Per-sub-fn arg bundle: NN batcher + symmetry context.
///
/// Wave 11 Batch B helper. Bundles 4 pointer/scalar fields so the extracted
/// sub-fns can stay under the `clippy::too_many_arguments` threshold without
/// adding new `#[allow]` attributes (preserves F1 retirement target per
/// PREP §D.2). `Copy` semantics — passed by value with zero allocation.
#[derive(Clone, Copy)]
struct InferContext<'a> {
    batcher: &'a InferenceBatcher,
    sym_tables: &'static SymTables,
    sym_idx: usize,
    inv_idx: usize,
}

/// Per-sub-fn arg bundle: I2 cluster-variance accumulator triplet.
///
/// Wave 11 Batch B helper. Bundles the 3 `AtomicU64` refs used by
/// `infer_and_expand` for per-cluster value/policy variance accumulation
/// (Q2/Q27 attention-hijacking probe).
#[derive(Clone, Copy)]
struct ClusterVarianceAtomics<'a> {
    value_std_accum: &'a AtomicU64,
    policy_disagreement_accum: &'a AtomicU64,
    variance_samples: &'a AtomicU64,
}

/// Per-sub-fn arg bundle: per-move MCTS accumulators + positions_generated.
///
/// Wave 11 Batch B helper. Used by `play_one_move` to record per-move
/// search-health stats (depth, concentration, stat count, quiescence fires)
/// and increment the worker-level `positions_generated` counter.
#[derive(Clone, Copy)]
struct MoveAccumulators<'a> {
    mcts_depth_accum: &'a AtomicU64,
    mcts_conc_accum: &'a AtomicU64,
    mcts_stat_count: &'a AtomicU64,
    mcts_quiescence_fires: &'a AtomicU64,
    positions_generated: &'a AtomicUsize,
}

/// Type alias for per-position record tuple pushed into `records_vec`.
/// Matches the existing inline tuple at the call site in `record_position`.
type RecordTuple = (
    Vec<f32>,           // feat (state planes)
    Vec<f32>,           // chain (Q13 chain planes)
    Vec<f32>,           // projected_policy
    crate::board::Player, // player at move time
    i32,                // center q
    i32,                // center r
    bool,               // is_full_search
    u16,                // ply_index (CF-4): 0-based ply of this decision (board.ply
                        // pre-move; shared across the K cluster rows of one ply)
);

/// Per-game scalar context bundled to keep `play_one_move` / `run_one_game`
/// arity under `clippy::too_many_arguments`. All `Copy` scalars from the
/// parent fn's destructured locals; pure ergonomic packaging (no hot-path
/// field-access cost per `feedback_registryspec_by_ref_in_hotpath.md`).
#[derive(Clone, Copy)]
#[allow(clippy::struct_excessive_bools)] // mirror of WorkerParams flat layout (cycle 3 P79 KEEP)
struct MovePlayContext {
    leaf_batch_size: usize,
    temp_threshold: usize,
    temp_min: f32,
    zoi_lookback: usize,
    zoi_margin: i32,
    c_visit: f32,
    c_scale: f32,
    gumbel_m: usize,
    gumbel_explore_moves: usize,
    dirichlet_alpha: f32,
    dirichlet_epsilon: f32,
    full_search_prob: f32,
    n_sims_quick: usize,
    n_sims_full: usize,
    game_sims: usize,
    is_fast_game: bool,
    sym_idx: usize,
    completed_q_values: bool,
    gumbel_mcts: bool,
    dirichlet_enabled: bool,
    zoi_enabled: bool,
}

/// Per-game scalar context bundled to keep `run_one_game` arity under
/// `clippy::too_many_arguments`. Holds the per-game-init scalar outputs +
/// references needed for finalize.
#[derive(Clone, Copy)]
struct PerGameInitCtx {
    max_moves: usize,
    random_opening_plies: u32,
    legal_move_radius_jitter: bool,
    selfplay_rotation_enabled: bool,
    fast_prob: f32,
    fast_sims: usize,
    standard_sims: usize,
    n_cells: usize,
    draw_reward: f32,
    /// §178: terminal-via-ply-cap outcome (winner=None AND ply>=max_moves).
    /// Routed parallel to `draw_reward` so the value head sees distinct targets
    /// for organic draws vs ply-cap truncations.
    ply_cap_value: f32,
    results_queue_cap: usize,
    worker_id: usize,
}

/// Result of `play_one_move`: signals the parent loop how to proceed.
enum MoveOutcome {
    /// Move played successfully — continue inner for-loop iteration.
    Played,
    /// Move loop should break (terminal / failure / shutdown).
    Break,
    /// Move loop should `continue` to next iteration (search expansion failed).
    Continue,
}

/// Result of `run_mcts_search`: either the search completed (with optional
/// resulting GumbelSearchState) or root expansion failed (caller should
/// signal `continue` to the outer per-move loop).
enum McTSSearchResult {
    Completed(Option<GumbelSearchState>),
    RootExpansionFailed,
}

/// Per-worker hot loop. Extracted from the pre-split `start_impl`
/// `thread::spawn(move || { ... })` body; semantics byte-identical.
///
/// Wave 11 Batch B (J.2.b): body decomposed into 7 sub-fns + helper bundles.
/// Parent body is now <100 LOC orchestrator; the previously-required
/// `#[allow(clippy::too_many_lines)]` at this site is retired.
pub(super) fn run_worker_thread(
    worker_id: usize,
    stats: WorkerStats,
    atomics: WorkerAtomics,
    channels: WorkerChannels,
    params: WorkerParams,
    sym_tables_static: &'static SymTables,
    geometry: WorkerGeometry,
) {
    // §173 A5b: destructure geometry into local scalar bindings at fn entry.
    // Per feedback_registryspec_by_ref_in_hotpath.md the per-sim hot path must
    // see cheap integer locals, not field-access on a struct ref. WorkerGeometry
    // is `Copy` (~32 bytes) and exists purely to keep the fn arity ≤ 7.
    let WorkerGeometry { n_cells, kept_planes, policy_stride, agg_trunk_sz, has_pass_slot } =
        geometry;
    // §P52: destructure the captured groups back into the local names used by
    // the hot-loop body. Field order matches the pre-P52 capture order so the
    // body diff is empty. INV25 Cell 3 substring-anchors this pattern in
    // engine/tests/inv25_worker_loop_split_byte_identity.rs.
    let WorkerStats {
        games_completed,
        positions_generated,
        x_wins,
        o_wins,
        draws,
        positions_dropped,
        mcts_depth_accum,
        mcts_conc_accum,
        mcts_stat_count,
        mcts_quiescence_fires,
        cluster_value_std_accum,
        cluster_policy_disagreement_accum,
        cluster_variance_samples,
    } = stats;
    let WorkerAtomics { running, radius_override } = atomics;
    let WorkerChannels { batcher, results_queue, recent_game_results } = channels;
    let WorkerParams {
        max_moves,
        leaf_batch_size,
        c_puct,
        fpu_reduction,
        quiescence_blend_2,
        fast_prob,
        fast_sims,
        standard_sims,
        temp_threshold,
        temp_min,
        draw_reward,
        ply_cap_value,
        zoi_lookback,
        zoi_margin,
        c_visit,
        c_scale,
        gumbel_m,
        gumbel_explore_moves,
        dirichlet_alpha,
        dirichlet_epsilon,
        results_queue_cap,
        full_search_prob,
        n_sims_quick,
        n_sims_full,
        random_opening_plies,
        registry_spec: registry_spec_for_worker,
        search_flags: SearchFlags { quiescence_enabled, completed_q_values, gumbel_mcts },
        exploration_flags: ExplorationFlags { dirichlet_enabled, selfplay_rotation_enabled },
        move_constraint_flags: MoveConstraintFlags { zoi_enabled, legal_move_radius_jitter },
    } = params;

    let sym_tables = sym_tables_static;

    let mut tree = MCTSTree::new_full(c_puct, crate::mcts::VIRTUAL_LOSS_PENALTY, fpu_reduction);
    tree.quiescence_enabled = quiescence_enabled;
    tree.quiescence_blend_2 = quiescence_blend_2;
    let mut rng = rng();
    let mut version_seen: Vec<u64> = Vec::with_capacity(8);
    #[cfg(feature = "debug_prior_trace")]
    let mut dbg_game_idx: u32 = 0;
    let worker_registry_spec = registry_spec_for_worker;

    let variance_atomics = ClusterVarianceAtomics {
        value_std_accum: &cluster_value_std_accum,
        policy_disagreement_accum: &cluster_policy_disagreement_accum,
        variance_samples: &cluster_variance_samples,
    };
    let move_accumulators = MoveAccumulators {
        mcts_depth_accum: &mcts_depth_accum,
        mcts_conc_accum: &mcts_conc_accum,
        mcts_stat_count: &mcts_stat_count,
        mcts_quiescence_fires: &mcts_quiescence_fires,
        positions_generated: &positions_generated,
    };
    let init_ctx = PerGameInitCtx {
        max_moves, random_opening_plies, legal_move_radius_jitter,
        selfplay_rotation_enabled, fast_prob, fast_sims, standard_sims,
        n_cells, draw_reward, ply_cap_value, results_queue_cap, worker_id,
    };
    let search_scalars = (leaf_batch_size, temp_threshold, temp_min, zoi_lookback, zoi_margin);
    let gumbel_scalars = (c_visit, c_scale, gumbel_m, gumbel_explore_moves);
    let dirichlet_scalars = (dirichlet_alpha, dirichlet_epsilon, full_search_prob);
    let move_cap_scalars = (n_sims_quick, n_sims_full);
    let play_flags = (completed_q_values, gumbel_mcts, dirichlet_enabled, zoi_enabled);
    let finalize_counters: (&AtomicUsize, &AtomicU64, &AtomicU64, &AtomicU64, &AtomicU64) = (
        &games_completed, &x_wins, &o_wins, &draws, &positions_dropped,
    );

    while running.load(Ordering::Relaxed) {
        #[cfg(feature = "debug_prior_trace")]
        let dbg_game_idx_for_game = { let g = dbg_game_idx; dbg_game_idx += 1; g };
        #[cfg(not(feature = "debug_prior_trace"))]
        let dbg_game_idx_for_game: u32 = 0;
        run_one_game(
            &mut tree, &mut rng, &mut version_seen, &running, &radius_override,
            &batcher, sym_tables, worker_registry_spec, init_ctx, kept_planes,
            policy_stride, has_pass_slot, agg_trunk_sz, search_scalars,
            gumbel_scalars, dirichlet_scalars, move_cap_scalars, play_flags,
            variance_atomics, move_accumulators, &results_queue,
            &recent_game_results, finalize_counters, dbg_game_idx_for_game,
        );
    }
}

/// Per-game loop body (Wave 11 Batch B; warm/HOT path).
///
/// Init board + per-game state, run inner move loop, finalize game. Ports
/// verbatim from the pre-Wave-11 inner.rs L143-L772 (the body inside the
/// outer `while running.load()` loop). Returns when game ends or shutdown.
#[allow(clippy::too_many_arguments)] // per-game loop body; hot-path-by-value per §173 A5b (bundle would re-pack on each game)
fn run_one_game(
    tree: &mut MCTSTree,
    rng: &mut ThreadRng,
    version_seen: &mut Vec<u64>,
    running: &AtomicBool,
    radius_override: &AtomicI32,
    batcher: &InferenceBatcher,
    sym_tables: &'static SymTables,
    worker_registry_spec: Option<&'static crate::encoding::RegistrySpec>,
    init_ctx: PerGameInitCtx,
    kept_planes: &'static [usize],
    policy_stride: usize,
    has_pass_slot: bool,
    agg_trunk_sz: i32,
    search_scalars: (usize, usize, f32, usize, i32),
    gumbel_scalars: (f32, f32, usize, usize),
    dirichlet_scalars: (f32, f32, f32),
    move_cap_scalars: (usize, usize),
    play_flags: (bool, bool, bool, bool),
    variance_atomics: ClusterVarianceAtomics,
    move_accumulators: MoveAccumulators,
    results_queue: &Mutex<VecDeque<WorkerResultRow>>,
    recent_game_results: &Mutex<VecDeque<GameResultRow>>,
    finalize_counters: (
        &AtomicUsize, // games_completed
        &AtomicU64,   // x_wins
        &AtomicU64,   // o_wins
        &AtomicU64,   // draws
        &AtomicU64,   // positions_dropped
    ),
    dbg_game_idx: u32,
) {
    let (leaf_batch_size, temp_threshold, temp_min, zoi_lookback, zoi_margin) = search_scalars;
    let (c_visit, c_scale, gumbel_m, gumbel_explore_moves) = gumbel_scalars;
    let (dirichlet_alpha, dirichlet_epsilon, full_search_prob) = dirichlet_scalars;
    let (n_sims_quick, n_sims_full) = move_cap_scalars;
    let (completed_q_values, gumbel_mcts, dirichlet_enabled, zoi_enabled) = play_flags;

    let PerGameInit { mut board, mut records_vec, mut move_history,
        sym_idx, inv_idx, is_fast_game, game_sims } =
        init_per_game_board(worker_registry_spec, init_ctx, radius_override, rng, version_seen);

    let infer = InferContext { batcher, sym_tables, sym_idx, inv_idx };
    let play_ctx = MovePlayContext {
        leaf_batch_size, temp_threshold, temp_min, zoi_lookback, zoi_margin,
        c_visit, c_scale, gumbel_m, gumbel_explore_moves, dirichlet_alpha,
        dirichlet_epsilon, full_search_prob, n_sims_quick, n_sims_full,
        game_sims, is_fast_game, sym_idx,
        completed_q_values, gumbel_mcts, dirichlet_enabled, zoi_enabled,
    };

    for _ in 0..init_ctx.max_moves {
        if !running.load(Ordering::Relaxed) || board.check_win() || board.legal_move_count() == 0 {
            break;
        }

        // §115 random-opening plies: skip MCTS + recording for the first
        // `random_opening_plies` plies of every game.
        if board.ply < init_ctx.random_opening_plies {
            let legal = board.legal_moves();
            if legal.is_empty() { break; }
            let (mq, mr) = *legal.choose(rng).unwrap();
            if board.apply_move(mq, mr).is_err() { break; }
            move_history.push((mq, mr));
            continue;
        }

        match play_one_move(
            tree, &mut board, &mut records_vec, &mut move_history,
            version_seen, rng, running, play_ctx, kept_planes,
            init_ctx.n_cells, policy_stride, has_pass_slot, agg_trunk_sz,
            infer, variance_atomics, move_accumulators,
            init_ctx.worker_id, dbg_game_idx,
        ) {
            MoveOutcome::Played | MoveOutcome::Continue => {},
            MoveOutcome::Break => break,
        }
    }

    // §P22 — drain shutdown skip: if the move loop broke because `running`
    // was flipped false by `stop()`, the game is *in progress*, NOT terminal.
    // Returning here short-circuits to the outer `while running…` guard.
    if !running.load(Ordering::Relaxed) {
        return;
    }

    let (games_completed, x_wins, o_wins, draws, positions_dropped) = finalize_counters;
    finalize_game(
        &board, init_ctx.max_moves, records_vec, move_history, version_seen,
        sym_idx, sym_tables, init_ctx.n_cells, init_ctx.draw_reward,
        init_ctx.ply_cap_value, init_ctx.results_queue_cap, init_ctx.worker_id,
        results_queue, recent_game_results,
        games_completed, x_wins, o_wins, draws, positions_dropped,
    );
}

/// Per-game state outputs from `init_per_game_board`. Named struct avoids
/// the `clippy::type_complexity` warning on the 7-element return tuple.
struct PerGameInit {
    board: Board,
    records_vec: Vec<RecordTuple>,
    move_history: Vec<(i32, i32)>,
    sym_idx: usize,
    inv_idx: usize,
    is_fast_game: bool,
    game_sims: usize,
}

/// Per-game board + state initializer (Wave 11 Batch B; warm path).
///
/// Constructs the per-game `Board` via registry-spec or v6 default, pre-sizes
/// the per-game record vectors, samples per-game rotation symmetry, and
/// resolves the playout-cap (fast vs standard sims). Ports verbatim from the
/// pre-Wave-11 inner.rs L144-L201.
fn init_per_game_board(
    worker_registry_spec: Option<&'static crate::encoding::RegistrySpec>,
    init_ctx: PerGameInitCtx,
    radius_override: &AtomicI32,
    rng: &mut ThreadRng,
    version_seen: &mut Vec<u64>,
) -> PerGameInit {
    // §P3.2 — bind the registry-resolved spec at per-game Board construction.
    // v6w25/v7full/etc. carry their own perception (cluster window, threshold,
    // legal-move radius). `None` keeps byte-exact pre-§171 v6 defaults.
    let mut board = match worker_registry_spec {
        Some(spec) => Board::with_registry_spec(spec),
        None       => Board::new(),
    };
    // §P67: pre-size per-game record vectors against the upper-bound on moves
    // per game so the typical 64-128-move game never re-allocates.
    let records_vec = Vec::with_capacity(init_ctx.max_moves);
    let move_history: Vec<(i32, i32)> = Vec::with_capacity(init_ctx.max_moves);
    version_seen.clear();

    // §174: curriculum radius override takes precedence over encoding default
    // and jitter. Applied after encoding setup so it intentionally overrides
    // the spec's canonical radius.
    let ro = radius_override.load(Ordering::SeqCst);
    if ro >= 0 {
        board.override_legal_move_radius(ro);
    }

    // Phase B' v8 §152 Q2: per-game radius jitter ∈ {4, 5, 6}.
    // §P3.2 — guard with `worker_registry_spec.is_none()` so a v6w25 (or any
    // future registry-bound encoding) Board's legal_move_radius is not
    // overwritten by the v6-shaped jitter range. §174: jitter is skipped when
    // curriculum override is active.
    if init_ctx.legal_move_radius_jitter && worker_registry_spec.is_none() && ro < 0 {
        const JITTER_RADII: [i32; 3] = [4, 5, 6];
        let r = *JITTER_RADII.choose(rng).unwrap();
        board.set_legal_move_radius(r);
    }

    // §130: sample per-game rotation across the 12-element hex dihedral group
    // when self-play rotation is enabled.
    let sym_idx: usize = if init_ctx.selfplay_rotation_enabled {
        rng.random_range(0..N_SYMS)
    } else {
        0
    };
    let inv_idx = inv_sym_idx(sym_idx);

    // KataGo-style playout cap randomisation.
    let is_fast_game = init_ctx.fast_prob > 0.0 && rng.random::<f32>() < init_ctx.fast_prob;
    let game_sims = if is_fast_game { init_ctx.fast_sims } else { init_ctx.standard_sims };

    PerGameInit { board, records_vec, move_history, sym_idx, inv_idx, is_fast_game, game_sims }
}

/// Per-batch NN inference + leaf expansion (Wave 11 Batch B; HOT path).
///
/// Selects leaves, encodes per-cluster state, submits to the inference
/// batcher, forward/inverse-scatters under the per-game symmetry, accumulates
/// I2 cluster-variance metrics, aggregates per-leaf policies, and runs
/// `expand_and_backup`. `#[inline]` preserves the pre-Wave-11 closure inlining
/// behavior (L31 hazard mitigation).
///
/// Ports verbatim from the pre-Wave-11 inner.rs L243-L350 closure body.
#[inline]
#[allow(clippy::too_many_arguments)] // 5 of 9 args are spec-derived scalars passed by value (§173 A5b hot-path discipline)
fn infer_and_expand(
    tree: &mut MCTSTree,
    batch_size: usize,
    kept_planes: &'static [usize],
    n_cells: usize,
    policy_stride: usize,
    has_pass_slot: bool,
    agg_trunk_sz: i32,
    infer: InferContext,
    variance: ClusterVarianceAtomics,
) -> usize {
    let leaves = tree.select_leaves(batch_size);
    if leaves.is_empty() { return 0; }

    let mut all_batch_features = Vec::new();
    let mut leaf_metadata = Vec::with_capacity(leaves.len());

    for leaf in &leaves {
        let (views, centers) = leaf.get_cluster_views();
        let k = views.len();
        leaf_metadata.push((k, centers));
        for view in views {
            let mut buffer = infer.batcher.get_feature_buffer();
            leaf.encode_state_to_buffer_channels(&view, &mut buffer, kept_planes, n_cells);
            // §130: forward-scatter the input planes to the rotated frame so
            // the model sees a randomly-oriented view of this game.
            if infer.sym_idx != 0 {
                rotate_state_inplace(&mut buffer, infer.sym_idx, infer.sym_tables);
            }
            all_batch_features.push(buffer);
        }
    }
    if all_batch_features.is_empty() { return 0; }

    let total_clusters: usize = leaf_metadata.iter().map(|(k, _)| *k).sum();

    let (all_policies, all_values) = match infer.batcher.submit_batch_and_wait_rust(all_batch_features) {
        Ok(results) => {
            let mut ps = Vec::with_capacity(results.len());
            let mut vs = Vec::with_capacity(results.len());
            for (mut p, v) in results {
                // §130: inverse-scatter the policy back to canonical frame.
                if infer.sym_idx != 0 {
                    rotate_policy_inplace(&mut p, infer.inv_idx, infer.sym_tables, n_cells);
                }
                ps.push(p);
                vs.push(v);
            }
            (ps, vs)
        }
        Err(()) => return 0,
    };

    if all_policies.len() < total_clusters { return 0; }

    let mut aggregated_policies = Vec::with_capacity(leaves.len());
    let mut aggregated_values = Vec::with_capacity(leaves.len());
    let mut curr = 0;

    for (i, (k, centers)) in leaf_metadata.iter().enumerate() {
        let leaf_policies = &all_policies[curr..curr+k];
        let leaf_values = &all_values[curr..curr+k];
        curr += k;

        // I2 investigation metric: per-cluster value/policy variance (Q2/Q27).
        if *k >= 2 {
            let mean_v: f32 = leaf_values.iter().sum::<f32>() / *k as f32;
            let var_v: f32 = leaf_values.iter()
                .map(|&v| (v - mean_v).powi(2))
                .sum::<f32>() / *k as f32;
            let std_v = var_v.sqrt();
            let mut top1 = Vec::with_capacity(*k);
            for p in leaf_policies {
                let mut bi = 0usize;
                let mut bv = p[0];
                for (ii, &pv) in p.iter().enumerate() {
                    if pv > bv { bi = ii; bv = pv; }
                }
                top1.push(bi);
            }
            let mut max_c = 1usize;
            for &a in &top1 {
                let c = top1.iter().filter(|&&x| x == a).count();
                if c > max_c { max_c = c; }
            }
            let disagree = 1.0f32 - (max_c as f32 / *k as f32);
            variance.value_std_accum.fetch_add(
                (std_v * 1_000_000.0) as u64, Ordering::Relaxed);
            variance.policy_disagreement_accum.fetch_add(
                (disagree * 1_000_000.0) as u64, Ordering::Relaxed);
            variance.variance_samples.fetch_add(1, Ordering::Relaxed);
        }

        let mut min_v = leaf_values[0];
        for &v in leaf_values {
            if v < min_v { min_v = v; }
        }
        aggregated_values.push(min_v);
        aggregated_policies.push(records::aggregate_policy(policy_stride, has_pass_slot, agg_trunk_sz, &leaves[i], centers, leaf_policies));
    }

    let n = leaves.len();
    tree.expand_and_backup(&aggregated_policies, &aggregated_values);
    n
}

/// Per-move MCTS search dispatcher (Wave 11 Batch B; HOT path).
///
/// Two-branch dispatcher: Gumbel sequential-halving (when `gumbel_mcts=true`)
/// or standard PUCT with Dirichlet root noise (otherwise). Returns
/// `Some(Option<GumbelSearchState>)` on a normal search; outer `None` signals
/// the caller should `continue` (root expansion failed). Calls
/// `infer_and_expand` repeatedly inside.
///
/// Ports verbatim from the pre-Wave-11 inner.rs L352-L470.
#[allow(clippy::too_many_arguments)] // MCTS dispatcher; hot-path-by-value per §173 A5b (bundle would add field-access overhead on each sim)
fn run_mcts_search(
    tree: &mut MCTSTree,
    board: &Board,
    move_sims: usize,
    leaf_batch_size: usize,
    gumbel_mcts: bool,
    dirichlet_enabled: bool,
    dirichlet_alpha: f32,
    dirichlet_epsilon: f32,
    gumbel_m: usize,
    c_visit: f32,
    c_scale: f32,
    running: &AtomicBool,
    rng: &mut ThreadRng,
    kept_planes: &'static [usize],
    n_cells: usize,
    policy_stride: usize,
    has_pass_slot: bool,
    agg_trunk_sz: i32,
    infer: InferContext,
    variance: ClusterVarianceAtomics,
) -> McTSSearchResult {
    let mut gumbel_state: Option<GumbelSearchState> = None;

    if gumbel_mcts {
        // ── Gumbel MCTS with Sequential Halving ──
        let root_sims = infer_and_expand(tree, 1, kept_planes, n_cells, policy_stride, has_pass_slot, agg_trunk_sz, infer, variance);
        if root_sims == 0 || !tree.pool[0].is_expanded() {
            return McTSSearchResult::RootExpansionFailed;
        }
        let mut sims_used = root_sims;

        // Apply Dirichlet noise to root priors after expansion. Skip at
        // intermediate ply (second stone of compound turn). ply==0 is P1's
        // single opening stone, which IS a turn boundary.
        let is_intermediate_ply = board.moves_remaining == 1 && board.ply > 0;
        if dirichlet_enabled && !is_intermediate_ply {
            let n_ch = tree.pool[0].n_children as usize;
            if n_ch > 0 {
                let noise = crate::mcts::dirichlet::sample_dirichlet(dirichlet_alpha, n_ch, rng);
                tree.apply_dirichlet_to_root(&noise, dirichlet_epsilon);
            }
        }

        // Phase 2: Gumbel-Top-k candidate selection. Guard: if effective_m is
        // 0 (no budget or no children), fall back to the standard PUCT path.
        let effective_m = gumbel_m.min(move_sims).min(tree.root_n_children());
        if effective_m == 0 {
            let mut sims_done = sims_used;
            while sims_done < move_sims {
                if !running.load(Ordering::Relaxed) { break; }
                let n = infer_and_expand(tree, leaf_batch_size, kept_planes, n_cells, policy_stride, has_pass_slot, agg_trunk_sz, infer, variance);
                if n == 0 { break; }
                sims_done += n;
            }
            gumbel_state = None;
        } else {

        let mut gs = GumbelSearchState::new(tree, effective_m, c_visit, c_scale, rng);

        // Phase 3: Sequential Halving — allocate budget across phases.
        let num_phases = gs.num_phases;
        for phase in 0..num_phases {
            if sims_used >= move_sims { break; }
            let remaining_budget = move_sims.saturating_sub(sims_used);
            let remaining_phases = num_phases - phase;
            let sims_per = (remaining_budget / (remaining_phases * gs.candidates.len())).max(1);

            let cands = gs.candidates.clone();
            for &cand_offset in &cands {
                if sims_used >= move_sims { break; }
                if !running.load(Ordering::Relaxed) { break; }

                let child_pool_idx = gs.first_child + cand_offset as u32;
                tree.forced_root_child = Some(child_pool_idx);

                let mut cand_sims = 0;
                while cand_sims < sims_per && sims_used < move_sims {
                    if !running.load(Ordering::Relaxed) { break; }
                    // Cap batch to remaining budget for this candidate so we
                    // don't overshoot sims_per (leaf_batch_size can be 8 when
                    // only 1-3 sims are budgeted, biasing early candidates).
                    let batch = leaf_batch_size.min(sims_per.saturating_sub(cand_sims));
                    let n = infer_and_expand(tree, batch.max(1), kept_planes, n_cells, policy_stride, has_pass_slot, agg_trunk_sz, infer, variance);
                    if n == 0 { break; }
                    cand_sims += n;
                    sims_used += n;
                }
                tree.forced_root_child = None;
            }

            if gs.candidates.len() <= 1 { break; }
            gs.halve_candidates(tree);
        }
        tree.forced_root_child = None;
        gumbel_state = Some(gs);
        } // end effective_m > 0
    } else {
        // ── Standard PUCT search with Dirichlet root noise ──
        let root_n = infer_and_expand(tree, 1, kept_planes, n_cells, policy_stride, has_pass_slot, agg_trunk_sz, infer, variance);
        if root_n == 0 {
            return McTSSearchResult::RootExpansionFailed;
        }
        let mut sims_done = root_n;

        let is_intermediate_ply = board.moves_remaining == 1 && board.ply > 0;
        if dirichlet_enabled && !is_intermediate_ply
            && tree.pool[0].is_expanded() {
                let n_ch = tree.pool[0].n_children as usize;
                if n_ch > 0 {
                    let noise = crate::mcts::dirichlet::sample_dirichlet(dirichlet_alpha, n_ch, rng);
                    tree.apply_dirichlet_to_root(&noise, dirichlet_epsilon);
                }
            }

        while sims_done < move_sims {
            if !running.load(Ordering::Relaxed) { break; }
            let n = infer_and_expand(tree, leaf_batch_size, kept_planes, n_cells, policy_stride, has_pass_slot, agg_trunk_sz, infer, variance);
            if n == 0 { break; }
            sims_done += n;
        }
    }

    McTSSearchResult::Completed(gumbel_state)
}

/// Per-move dispatcher (Wave 11 Batch B; warm/HOT path).
///
/// Orchestrates one full move: playout-cap selection, MCTS search dispatch,
/// debug-trace snapshot, per-move stat accumulation, target-policy compute,
/// ZOI-filtered legal move sampling, position recording, and apply-move.
/// Returns `MoveOutcome` to signal the parent loop control flow.
///
/// Ports verbatim from the pre-Wave-11 inner.rs L225-L443.
#[allow(clippy::too_many_arguments)] // per-move dispatcher; hot-path-by-value per §173 A5b (bundle would re-pack on each move)
fn play_one_move(
    tree: &mut MCTSTree,
    board: &mut Board,
    records_vec: &mut Vec<RecordTuple>,
    move_history: &mut Vec<(i32, i32)>,
    version_seen: &mut Vec<u64>,
    rng: &mut ThreadRng,
    running: &AtomicBool,
    ctx: MovePlayContext,
    kept_planes: &'static [usize],
    n_cells: usize,
    policy_stride: usize,
    has_pass_slot: bool,
    agg_trunk_sz: i32,
    infer: InferContext,
    variance: ClusterVarianceAtomics,
    accumulators: MoveAccumulators,
    worker_id: usize,
    dbg_game_idx: u32,
) -> MoveOutcome {
    let _ = worker_id; // only consumed under debug_prior_trace cfg
    let _ = dbg_game_idx; // only consumed under debug_prior_trace cfg

    // Move-level playout cap (orthogonal to game-level fast_prob).
    let (move_is_full_search, move_sims) = if ctx.full_search_prob > 0.0 {
        let full = rng.random::<f32>() < ctx.full_search_prob;
        let sims = if full { ctx.n_sims_full } else { ctx.n_sims_quick };
        (full, sims)
    } else {
        (true, ctx.game_sims)
    };

    // ── MCTS Search ──
    tree.new_game(board.clone());

    let gumbel_state = match run_mcts_search(
        tree, board, move_sims, ctx.leaf_batch_size, ctx.gumbel_mcts,
        ctx.dirichlet_enabled, ctx.dirichlet_alpha, ctx.dirichlet_epsilon,
        ctx.gumbel_m, ctx.c_visit, ctx.c_scale, running, rng,
        kept_planes, n_cells, policy_stride, has_pass_slot, agg_trunk_sz,
        infer, variance,
    ) {
        McTSSearchResult::Completed(gs) => gs,
        McTSSearchResult::RootExpansionFailed => return MoveOutcome::Continue,
    };

    if !running.load(Ordering::Relaxed) { return MoveOutcome::Break; }

    // ── MCTS Policy with cosine-annealed temperature schedule ──
    let compound_move = if board.ply == 0 { 0 } else { (board.ply as usize).div_ceil(2) };
    let temperature = if ctx.is_fast_game {
        1.0  // fast games: always exploratory
    } else {
        compute_move_temperature(compound_move, ctx.temp_threshold, ctx.temp_min)
    };
    // §P2: pass `policy_stride` (= spec.policy_logit_count). Pre-P2 the inner
    // API computed bs²+1 unconditionally, producing phantom pass-slot vectors
    // for v8 (audit FD.4).
    let policy = tree.get_policy(temperature, policy_stride);

    // ── debug_prior_trace: snapshot root priors + visit counts ──
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
                crate::debug_trace::GameRunnerRecord {
                    game_index:          dbg_game_idx,
                    worker_id:           worker_id as u32,
                    compound_move:       compound_move as u32,
                    ply:                 board.ply as u32,
                    legal_move_count:    legal_count,
                    root_n_children:     n_ch as u32,
                    simulations_planned: ctx.game_sims as u32,
                    root_priors:         &priors,
                    root_visit_counts:   &visits,
                    temperature,
                    is_fast_game: ctx.is_fast_game,
                },
            );
        }
    }

    // Accumulate MCTS health stats once per search (not in inner sim loop).
    {
        let (depth, conc) = tree.last_search_stats();
        accumulators.mcts_depth_accum.fetch_add((depth * 1_000_000.0) as u64, Ordering::Relaxed);
        accumulators.mcts_conc_accum.fetch_add((conc * 1_000_000.0) as u64, Ordering::Relaxed);
        accumulators.mcts_stat_count.fetch_add(1, Ordering::Relaxed);
        accumulators.mcts_quiescence_fires.fetch_add(
            tree.quiescence_fire_count.load(Ordering::Relaxed),
            Ordering::Relaxed,
        );
    }

    // Phase B' Class-1: snapshot the batcher's model_version once per move
    // and dedup-insert into version_seen.
    {
        let v = infer.batcher.current_model_version();
        if !version_seen.contains(&v) {
            version_seen.push(v);
        }
    }

    // Completed Q-values: compute improved policy for training target.
    let target_policy = if ctx.completed_q_values {
        tree.get_improved_policy(policy_stride, ctx.c_visit, ctx.c_scale)
    } else {
        policy.clone()
    };

    // ── Sample and apply move (ZOI-filtered legal set) ──
    let Some(move_idx) = select_move(
        board, move_history, &policy, gumbel_state, ctx, agg_trunk_sz, tree, rng,
    ) else {
        return MoveOutcome::Break;
    };

    // ── Record position ──
    record_position(
        board, kept_planes, n_cells, agg_trunk_sz, ctx.is_fast_game,
        ctx.completed_q_values, policy_stride, has_pass_slot, &target_policy,
        ctx.sym_idx, infer.sym_tables, move_is_full_search, records_vec,
    );

    if board.apply_move(move_idx.0, move_idx.1).is_err() {
        return MoveOutcome::Break;
    }
    move_history.push((move_idx.0, move_idx.1));
    accumulators.positions_generated.fetch_add(1, Ordering::Relaxed);
    MoveOutcome::Played
}

/// Per-move legal-move sampler (Wave 11 Batch B; warm path).
///
/// Filters legal moves by ZOI when enabled, picks via Gumbel winner (post
/// exploration threshold) or visit-count sampling, falls back to uniform
/// random over the legal set. Returns `None` if no legal moves available
/// (caller should break the inner move loop).
///
/// Ports verbatim from the pre-Wave-11 inner.rs L559-L607.
#[allow(clippy::too_many_arguments)] // per-move sampler; hot-path-by-value per §173 A5b (bundle would re-pack on each move)
fn select_move(
    board: &Board,
    move_history: &[(i32, i32)],
    policy: &[f32],
    gumbel_state: Option<GumbelSearchState>,
    ctx: MovePlayContext,
    agg_trunk_sz: i32,
    tree: &MCTSTree,
    rng: &mut ThreadRng,
) -> Option<(i32, i32)> {
    let full_legal = board.legal_moves();
    if full_legal.is_empty() { return None; }

    // ZOI filtering: restrict move sampling to cells near recent moves.
    let legal = if ctx.zoi_enabled && move_history.len() >= 3 {
        let filtered: Vec<_> = full_legal.iter()
            .filter(|(q, r)| {
                move_history.iter().rev().take(ctx.zoi_lookback).any(|(q0, r0)| {
                    hex_distance(*q, *r, *q0, *r0) <= ctx.zoi_margin
                })
            })
            .copied()
            .collect();
        if filtered.len() < 3 { full_legal } else { filtered }
    } else {
        full_legal
    };

    // Move selection: Gumbel winner or visit-count sampling.
    let use_gumbel_winner = gumbel_state.is_some()
        && board.ply as usize >= ctx.gumbel_explore_moves;
    let move_idx = if use_gumbel_winner {
        let mut gs = gumbel_state.unwrap();
        let best_pool = gs.best_action_pool_idx(tree);
        let val = tree.pool[best_pool as usize].action_idx;
        let mq = (val >> 16) as i32 - 32768;
        let mr = (val & 0xFFFF) as i32 - 32768;
        if legal.contains(&(mq, mr)) {
            (mq, mr)
        } else {
            // §173 A8'': sample_policy now takes spec-derived trunk_sz.
            match records::sample_policy(policy, &legal, board, agg_trunk_sz) {
                Some(idx) => idx,
                None => *legal.choose(rng).unwrap(),
            }
        }
    } else {
        match records::sample_policy(policy, &legal, board, agg_trunk_sz) {
            Some(idx) => idx,
            None => *legal.choose(rng).unwrap(),
        }
    };
    Some(move_idx)
}

/// Per-move position recorder (Wave 11 Batch B; warm path).
///
/// Encodes per-cluster state/chain/policy buffers, forward-scatters under the
/// per-game symmetry, and pushes one record per cluster view into the per-game
/// `records_vec`. Ports verbatim from the pre-Wave-11 inner.rs L609-L655.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::fn_params_excessive_bools)] // ports per-move flag locals from caller; bundle would re-pack
fn record_position(
    board: &Board,
    kept_planes: &'static [usize],
    n_cells: usize,
    agg_trunk_sz: i32,
    is_fast_game: bool,
    completed_q_values: bool,
    policy_stride: usize,
    has_pass_slot: bool,
    target_policy: &[f32],
    sym_idx: usize,
    sym_tables: &'static SymTables,
    move_is_full_search: bool,
    records_vec: &mut Vec<RecordTuple>,
) {
    let (views, centers) = board.get_cluster_views();
    // §P11 (Wave 4): hoist legal_moves once across the K cluster scatters.
    let record_legal_moves = board.legal_moves();
    for (k, center) in centers.iter().enumerate() {
        let mut feat = vec![0.0f32; kept_planes.len() * n_cells];
        board.encode_state_to_buffer_channels(&views[k], &mut feat, kept_planes, n_cells);
        // Compute Q13 chain-length planes separately (not in state).
        let mut chain = vec![0.0f32; 6 * n_cells];
        encode_chain_planes(
            &views[k][..n_cells],
            &views[k][n_cells..2 * n_cells],
            &mut chain,
            n_cells,
            agg_trunk_sz,
        );
        // Fast games: zero-policy marks value-only targets (unless completed
        // Q-values are enabled, which give signal even at 50 sims).
        let mut projected_policy = if is_fast_game && !completed_q_values {
            vec![0.0; policy_stride]
        } else {
            records::aggregate_policy_to_local(policy_stride, has_pass_slot, agg_trunk_sz, board, center, target_policy, &record_legal_moves)
        };
        // §130: forward-scatter the recorded state, chain, and policy into
        // the rotated frame.
        if sym_idx != 0 {
            rotate_state_inplace(&mut feat, sym_idx, sym_tables);
            rotate_chain_inplace(&mut chain, sym_idx, sym_tables);
            rotate_policy_inplace(&mut projected_policy, sym_idx, sym_tables, n_cells);
        }
        // CF-4: record this decision's 0-based ply index. `record_position`
        // runs before `apply_move`, so `board.ply` is the ply of the position
        // being recorded; all K cluster rows of one ply share it. Fixes the
        // degenerate constant-0 self-play ply-index aux target (audit CF-4).
        records_vec.push((feat, chain, projected_policy, board.current_player, center.0, center.1, move_is_full_search, board.ply as u16));
    }
}

/// Per-game terminal handler (Wave 11 Batch B; warm path).
///
/// Classifies game outcome (winner, terminal_reason, version_seen range),
/// reprojects per-row aux targets, scatters them under per-game symmetry,
/// pushes all rows into the shared results queue, increments win/draw
/// counters, caps the queue at `results_queue_cap`, and pushes a single
/// `recent_game_results` metadata row.
///
/// Ports verbatim from the pre-Wave-11 inner.rs L680-L772.
#[allow(clippy::too_many_arguments)] // per-game finalize; hot-path-by-value per §173 A5b (bundle would add field-access overhead on results-queue push)
fn finalize_game(
    board: &Board,
    max_moves: usize,
    records_vec: Vec<RecordTuple>,
    move_history: Vec<(i32, i32)>,
    version_seen: &[u64],
    sym_idx: usize,
    sym_tables: &'static SymTables,
    n_cells: usize,
    draw_reward: f32,
    ply_cap_value: f32,
    results_queue_cap: usize,
    worker_id: usize,
    results_queue: &Mutex<VecDeque<WorkerResultRow>>,
    recent_game_results: &Mutex<VecDeque<GameResultRow>>,
    games_completed: &AtomicUsize,
    x_wins: &AtomicU64,
    o_wins: &AtomicU64,
    draws: &AtomicU64,
    positions_dropped: &AtomicU64,
) {
    // ── Game End: determine outcome ──
    let winner = board.winner();
    let plies = board.ply as usize;
    let winner_code: u8 = match winner {
        Some(crate::board::Player::One) => 1,
        Some(_)                         => 2,
        None                            => 0,
    };
    // Snapshot the final-board cell list and winning line once; each row
    // reprojects them into its own per-cluster window centre.
    let final_cells: Vec<((i32, i32), crate::board::Cell)> = board.cells
        .iter()
        .map(|(&qr, &c)| (qr, c))
        .collect();
    let winning_cells: Vec<(i32, i32)> = board.find_winning_line();

    // Phase B' Class-3 (terminal_reason) — discriminator for self-play
    // composition tracking. Encoding mirrors recent_game_results docstring.
    //   0 = six_in_a_row : winner exists AND winning_cells non-empty
    //   1 = colony       : winner exists AND no winning_line
    //   2 = ply_cap      : no winner AND ply == max_moves
    //   3 = other_draw   : no winner AND ply < max_moves
    let terminal_reason: u8 = match winner {
        Some(_) => u8::from(winning_cells.is_empty()),
        None    => if plies >= max_moves { 2 } else { 3 },
    };

    // Phase B' Class-1: collapse version_seen into (min, max, distinct).
    let (mv_min, mv_max, mv_distinct) = if version_seen.is_empty() {
        (0u64, 0u64, 0u32)
    } else {
        let mn = *version_seen.iter().min().unwrap();
        let mx = *version_seen.iter().max().unwrap();
        (mn, mx, version_seen.len() as u32)
    };

    // DRAW-MASK (Phase 6): per-GAME value-supervision flag. `terminal_reason == 2`
    // is the ply-cap branch (horizon truncation, no real outcome) — its fabricated
    // `ply_cap_value` label must be masked out of the value loss. 1 = supervise value
    // (default, all decisive/organic-draw games), 0 = capped → mask. Computed once
    // here (per-game constant); the per-move record carries no value_valid field.
    let value_valid: u8 = u8::from(terminal_reason != 2);
    let mut games_results = results_queue.lock().expect("results lock poisoned");
    for (feat, chain, pol, player, cq, cr, is_full_search, ply_index) in records_vec {
        // §178: split outcome by terminal_reason. `terminal_reason == 2` is the
        // ply-cap branch (winner=None AND ply>=max_moves) — pay `ply_cap_value`.
        // Any other winner=None path (organic draw, legal_move_count==0) pays
        // `draw_reward`. Default cfg has both equal (`-0.1`), preserving
        // pre-§178 behavior byte-for-byte (INV26 Cell 4 pins this).
        let outcome = match winner {
            Some(p) => if p as i8 == player as i8 { 1.0 } else { -1.0 },
            None => if terminal_reason == 2 { ply_cap_value } else { draw_reward },
        };

        // Per-row aux reprojection (ownership + winning_line) into this row's
        // per-cluster window centre. See records::reproject_game_end_row.
        let mut aux_u8 = records::reproject_game_end_row(
            &final_cells, &winning_cells, cq, cr, n_cells,
        );
        // §130: forward-scatter the aux pair into the same rotated frame as
        // state/chain/policy. Reproject + scatter compose because both are
        // pure permutations on cell indices.
        if sym_idx != 0 {
            rotate_aux_inplace(&mut aux_u8, sym_idx, sym_tables, n_cells);
        }

        games_results.push_back((feat, chain, pol, outcome, plies, aux_u8, is_full_search, ply_index, value_valid));
    }
    games_completed.fetch_add(1, Ordering::Relaxed);
    match winner {
        Some(crate::board::Player::One) => { x_wins.fetch_add(1, Ordering::Relaxed); }
        Some(_)                         => { o_wins.fetch_add(1, Ordering::Relaxed); }
        None                            => { draws.fetch_add(1, Ordering::Relaxed); }
    }
    // Per-row aux is already pushed into game_results above. The game-end
    // record here only carries metadata for Python's game_complete logging.
    {
        let mut rg = recent_game_results.lock().expect("recent_game_results lock poisoned");
        rg.push_back((
            plies, winner_code, move_history, worker_id,
            terminal_reason, mv_min, mv_max, mv_distinct,
        ));
        // Cap at 2000 entries to avoid unbounded growth if Python is slow.
        if rg.len() > 2000 {
            rg.pop_front();
        }
    }

    // Cap the results queue to avoid memory explosion if Python is slow.
    // Drop count is tracked on `positions_dropped` for dashboard visibility.
    if games_results.len() > results_queue_cap {
        let to_drop = games_results.len() - results_queue_cap;
        for _ in 0..to_drop {
            games_results.pop_front();
        }
        positions_dropped.fetch_add(to_drop as u64, Ordering::Relaxed);
    }
}

