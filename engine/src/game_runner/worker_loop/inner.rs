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
use crate::replay_buffer::hexg::{GraphRecord, MAX_VISITS};
use crate::replay_buffer::sym_tables::{N_SYMS, SymTables};

use super::super::gumbel_search::GumbelSearchState;
use super::super::{GameResultRow, WorkerResultRow, records};

use super::atomics::WorkerAtomics;
use super::channels::WorkerChannels;
use super::params::{
    ExplorationFlags, ForcedWinPolicy, MoveConstraintFlags, SearchFlags, SeedCorpus, SolverInLoop,
    WorkerGeometry, WorkerParams,
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

/// Per-sub-fn arg bundle: D-WS3V3 in-run solver fire-rate counter refs.
///
/// Bundles the 8 cumulative `AtomicU64` refs the seeding + solver-injection
/// hooks increment. `Copy` — passed by value with zero allocation. Incremented
/// ONLY under the `solver_enabled` / seeded branches so an OFF run leaves the
/// bench-gated hot path byte-identical.
#[derive(Clone, Copy)]
struct SolverCounters<'a> {
    moves_eligible: &'a AtomicU64,
    win_proven: &'a AtomicU64,
    injected: &'a AtomicU64,
    injected_offwindow: &'a AtomicU64,
    budget_exhausted: &'a AtomicU64,
    moves_eligible_seeded: &'a AtomicU64,
    injected_seeded: &'a AtomicU64,
    seeded_games_started: &'a AtomicU64,
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
    // O1 forced-win → one-hot POLICY target (per-move; read at target extraction).
    forced_win_enabled: bool,
    forced_win_depth: u8,
    forced_win_weight: f32,
    // D-WS3 L1 solver-in-loop SOFT visit-injection (per-move; read at target
    // extraction in `play_one_move`). Default-OFF = byte-identical hot path.
    solver_enabled: bool,
    solver_depth: u32,
    solver_node_budget: u64,
    solver_neighbor_dist: i32,
    solver_visit_weight: f32,
    // D-WS3V3 — per-game seeding state. `game_start_ply` = the absolute ply the
    // organic self-play begins at (0 for organic games; == seed prefix_len for a
    // seeded game). Gates Gumbel exploration RELATIVE to game start so a seeded
    // game (start ply 40-80) still explores instead of collapsing to argmax from
    // its first move (the D-ARGMAX dup trap). `seeded` scopes the `*_seeded`
    // solver counters. The temperature schedule stays on ABSOLUTE ply (matches
    // the organic regime).
    game_start_ply: usize,
    seeded: bool,
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

/// §B1 (2026-06-02) — per-move STATIC scalar config, built ONCE per worker at
/// proto-build and copied through the COLD layers (`run_one_game` →
/// `MovePlayContext`). Collapses the 6 anonymous transport tuples
/// (`search_scalars`/`gumbel_scalars`/`dirichlet_scalars`/`move_cap_scalars`/
/// `play_flags`/`forced_win_scalars`) that existed only to dodge
/// `clippy::too_many_arguments`.
///
/// A5b: this is NEVER passed into the per-sim hot path — `infer_and_expand` and
/// `run_mcts_search` signatures are unchanged; only the per-game/per-move cold
/// layers carry it, where the `Copy` is amortized over O(100) sims/move. The
/// fields/order mirror the static fields of `MovePlayContext`, so the per-game
/// destructure feeds that build unchanged.
#[derive(Clone, Copy)]
struct WorkerMoveCfg {
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
    completed_q_values: bool,
    gumbel_mcts: bool,
    dirichlet_enabled: bool,
    zoi_enabled: bool,
    forced_win_enabled: bool,
    forced_win_depth: u8,
    forced_win_weight: f32,
    // D-WS3 L1 solver-in-loop SOFT visit-injection static knobs (mirror the
    // MovePlayContext fields; built once per worker, copied through run_one_game).
    solver_enabled: bool,
    solver_depth: u32,
    solver_node_budget: u64,
    solver_neighbor_dist: i32,
    solver_visit_weight: f32,
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
    let WorkerGeometry {
        n_cells, kept_planes, policy_stride, agg_trunk_sz, has_pass_slot, legal_set, is_graph,
    } = geometry;
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
        solver_moves_eligible,
        solver_win_proven,
        solver_injected,
        solver_injected_offwindow,
        solver_budget_exhausted,
        solver_moves_eligible_seeded,
        solver_injected_seeded,
        seeded_games_started,
    } = stats;
    let WorkerAtomics { running, radius_override } = atomics;
    let WorkerChannels { batcher, results_queue, recent_game_results, graph_results_queue } = channels;
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
        forced_win_policy: ForcedWinPolicy {
            enabled: forced_win_policy_enabled,
            depth: forced_win_policy_depth,
            weight: forced_win_policy_weight,
        },
        solver_in_loop: SolverInLoop {
            enabled: solver_enabled,
            depth: solver_depth,
            node_budget: solver_node_budget,
            neighbor_dist: solver_neighbor_dist,
            visit_weight: solver_visit_weight,
        },
        seed_corpus,
        interior_selector,
    } = params;

    let sym_tables = sym_tables_static;

    let mut tree = MCTSTree::new_full(c_puct, crate::mcts::VIRTUAL_LOSS_PENALTY, fpu_reduction);
    tree.quiescence_enabled = quiescence_enabled;
    tree.quiescence_blend_2 = quiescence_blend_2;
    // D-QFIX-LAND A1: apply the interior (non-root) selection rule to this
    // worker's tree (lower blast radius than extending `new_full`). `Puct` =
    // HEAD behaviour (byte-identical); `GumbelImproved` is a wired placeholder.
    tree.interior_selector = interior_selector;
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
    let solver_counters = SolverCounters {
        moves_eligible: &solver_moves_eligible,
        win_proven: &solver_win_proven,
        injected: &solver_injected,
        injected_offwindow: &solver_injected_offwindow,
        budget_exhausted: &solver_budget_exhausted,
        moves_eligible_seeded: &solver_moves_eligible_seeded,
        injected_seeded: &solver_injected_seeded,
        seeded_games_started: &seeded_games_started,
    };
    let init_ctx = PerGameInitCtx {
        max_moves, random_opening_plies, legal_move_radius_jitter,
        selfplay_rotation_enabled, fast_prob, fast_sims, standard_sims,
        n_cells, draw_reward, ply_cap_value, results_queue_cap, worker_id,
    };
    // §B1: one Copy sub-config replaces the 6 anonymous per-move scalar tuples.
    // O1 forced-win knobs fold in as plain fields (no separate bundle).
    let move_cfg = WorkerMoveCfg {
        leaf_batch_size, temp_threshold, temp_min, zoi_lookback, zoi_margin,
        c_visit, c_scale, gumbel_m, gumbel_explore_moves,
        dirichlet_alpha, dirichlet_epsilon, full_search_prob,
        n_sims_quick, n_sims_full,
        completed_q_values, gumbel_mcts, dirichlet_enabled, zoi_enabled,
        forced_win_enabled: forced_win_policy_enabled,
        forced_win_depth: forced_win_policy_depth,
        forced_win_weight: forced_win_policy_weight,
        solver_enabled, solver_depth, solver_node_budget,
        solver_neighbor_dist, solver_visit_weight,
    };
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
            policy_stride, has_pass_slot, agg_trunk_sz, legal_set, is_graph, move_cfg,
            variance_atomics, move_accumulators, solver_counters, &seed_corpus,
            &results_queue, &graph_results_queue, &recent_game_results, finalize_counters,
            dbg_game_idx_for_game,
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
    legal_set: bool,
    is_graph: bool,
    move_cfg: WorkerMoveCfg,
    variance_atomics: ClusterVarianceAtomics,
    move_accumulators: MoveAccumulators,
    solver_counters: SolverCounters,
    seed: &SeedCorpus,
    results_queue: &Mutex<VecDeque<WorkerResultRow>>,
    graph_results_queue: &Mutex<VecDeque<GraphRecord>>,
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
    // §B1: destructure the Copy sub-config into the same local names the
    // MovePlayContext build below already uses (build unchanged). Replaces the
    // 6 anonymous tuple unpacks.
    let WorkerMoveCfg {
        leaf_batch_size, temp_threshold, temp_min, zoi_lookback, zoi_margin,
        c_visit, c_scale, gumbel_m, gumbel_explore_moves,
        dirichlet_alpha, dirichlet_epsilon, full_search_prob,
        n_sims_quick, n_sims_full,
        completed_q_values, gumbel_mcts, dirichlet_enabled, zoi_enabled,
        forced_win_enabled, forced_win_depth, forced_win_weight,
        solver_enabled, solver_depth, solver_node_budget,
        solver_neighbor_dist, solver_visit_weight,
    } = move_cfg;

    let PerGameInit { mut board, mut records_vec, mut graph_records, mut move_history,
        sym_idx, inv_idx, is_fast_game, game_sims, seeded, prefix_len } =
        init_per_game_board(worker_registry_spec, init_ctx, radius_override, rng, version_seen, seed);

    // D-WS3V3: count a seeded game once at start (the seed prefix was replayed in
    // `init_per_game_board`). `seeded == false` for organic games (byte-identical).
    if seeded {
        solver_counters.seeded_games_started.fetch_add(1, Ordering::Relaxed);
    }

    let infer = InferContext { batcher, sym_tables, sym_idx, inv_idx };
    let play_ctx = MovePlayContext {
        leaf_batch_size, temp_threshold, temp_min, zoi_lookback, zoi_margin,
        c_visit, c_scale, gumbel_m, gumbel_explore_moves, dirichlet_alpha,
        dirichlet_epsilon, full_search_prob, n_sims_quick, n_sims_full,
        game_sims, is_fast_game, sym_idx,
        completed_q_values, gumbel_mcts, dirichlet_enabled, zoi_enabled,
        forced_win_enabled, forced_win_depth, forced_win_weight,
        solver_enabled, solver_depth, solver_node_budget,
        solver_neighbor_dist, solver_visit_weight,
        game_start_ply: prefix_len, seeded,
    };

    // D-WS3V3: a seeded game starts `prefix_len` plies in; cap the organic move
    // budget so the total ply stays ~<= max_moves (game_lengths weighting stays
    // comparable). `.max(20)` guarantees a deep-seed game still plays out. Organic
    // games (prefix_len == 0) get the unchanged `max_moves` budget.
    let mut solver_fires: u32 = 0;
    let move_iters = if seeded {
        init_ctx.max_moves.saturating_sub(prefix_len).max(20)
    } else {
        init_ctx.max_moves
    };
    for _ in 0..move_iters {
        if !running.load(Ordering::Relaxed) || board.check_win() || board.legal_move_count() == 0 {
            break;
        }

        // §115 random-opening plies: skip MCTS + recording for the first
        // `random_opening_plies` plies of every game. D-WS3V3: skipped entirely
        // for a seeded game — the corpus prefix already supplies off-canonical
        // early-game diversity (and `prefix_len >= random_opening_plies` makes it
        // a natural no-op anyway).
        if !seeded && board.ply < init_ctx.random_opening_plies {
            let legal = board.legal_moves();
            if legal.is_empty() { break; }
            let (mq, mr) = *legal.choose(rng).unwrap();
            if board.apply_move(mq, mr).is_err() { break; }
            move_history.push((mq, mr));
            continue;
        }

        match play_one_move(
            tree, &mut board, &mut records_vec, &mut graph_records, &mut move_history,
            version_seen, rng, running, play_ctx, kept_planes,
            init_ctx.n_cells, policy_stride, has_pass_slot, agg_trunk_sz, legal_set, is_graph,
            infer, variance_atomics, move_accumulators, solver_counters,
            &mut solver_fires, init_ctx.worker_id, dbg_game_idx,
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
    // GNN-integration WP-5b commit A (R5): ONE hoisted branch at the finalize
    // call site — grid runs `finalize_game` verbatim (byte-identical, §2 point
    // 2 of the delta doc); `finalize_game_graph` is a new sibling fn with no
    // dense caller.
    if is_graph {
        finalize_game_graph(
            &board, init_ctx.max_moves, graph_records, move_history, version_seen,
            init_ctx.draw_reward, init_ctx.ply_cap_value, init_ctx.results_queue_cap,
            init_ctx.worker_id, seeded, solver_fires,
            graph_results_queue, recent_game_results,
            games_completed, x_wins, o_wins, draws, positions_dropped,
        );
    } else {
        finalize_game(
            &board, init_ctx.max_moves, records_vec, move_history, version_seen,
            sym_idx, sym_tables, init_ctx.n_cells, init_ctx.draw_reward,
            init_ctx.ply_cap_value, init_ctx.results_queue_cap, init_ctx.worker_id,
            seeded, solver_fires,
            results_queue, recent_game_results,
            games_completed, x_wins, o_wins, draws, positions_dropped,
        );
    }
}

/// Per-game state outputs from `init_per_game_board`. Named struct avoids
/// the `clippy::type_complexity` warning on the 7-element return tuple.
struct PerGameInit {
    board: Board,
    records_vec: Vec<RecordTuple>,
    /// GNN-integration WP-5b commit A (R3): per-game graph-record accumulator.
    /// `Vec::new()` — zero-capacity, no allocation — for every grid game; only
    /// grows (via push) on the `is_graph` record-dispatch branch.
    graph_records: Vec<GraphRecord>,
    move_history: Vec<(i32, i32)>,
    sym_idx: usize,
    inv_idx: usize,
    is_fast_game: bool,
    game_sims: usize,
    // D-WS3V3 — start-position seeding outputs. `seeded` = this game replayed a
    // corpus prefix; `prefix_len` = plies removed from the organic budget + the
    // game-start ply for the relative Gumbel-explore gate (0 for organic games).
    seeded: bool,
    prefix_len: usize,
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
    seed: &SeedCorpus,
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
    let mut move_history: Vec<(i32, i32)> = Vec::with_capacity(init_ctx.max_moves);
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

    // D-WS3V3 — start-position seeding (KataGo startPoses). HARD INVARIANT
    // (INV25/26-class): the rng is drawn ONLY when the corpus is non-empty, so the
    // DEFAULT path (empty corpus OR `seed_fraction == 0.0`) leaves the rng stream —
    // and therefore every downstream sym_idx / is_fast_game / MCTS draw — byte-
    // identical to pre-D-WS3V3. When it fires, a prefix is chosen uniformly and
    // dry-replayed onto the board (ctor-validated, so `apply_move` cannot fail at
    // runtime — a failure is a debug_assert + graceful unseeded fall-back). Seeded
    // prefix moves go into `move_history` (state is fully derived by replay:
    // current_player, moves_remaining, ply, window/bbox, zobrist) but NOT
    // `records_vec`, and do NOT bump `positions_generated` (the random_opening_plies
    // convention).
    let (seeded, prefix_len) = if !seed.corpus.is_empty() && seed.seed_fraction > 0.0
        && rng.random::<f32>() < seed.seed_fraction
    {
        let prefix = seed.corpus.choose(rng).expect("corpus non-empty checked above");
        let mut ok = true;
        for &(q, r) in prefix {
            if board.apply_move(q, r).is_err() {
                debug_assert!(false, "seed prefix replay failed at ({q},{r}) — corpus is ctor-validated");
                ok = false;
                break;
            }
            move_history.push((q, r));
        }
        if ok { (true, prefix.len()) } else { (false, move_history.len()) }
    } else {
        (false, 0)
    };

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

    PerGameInit {
        board, records_vec, graph_records: Vec::new(), move_history, sym_idx, inv_idx,
        is_fast_game, game_sims, seeded, prefix_len,
    }
}

/// Per-batch NN inference + leaf expansion (Wave 11 Batch B; HOT path).
///
/// §D-MULTICLUSTER-S0 per-move policy/target — either the dense scatter_max
/// vector (existing path, byte-identical) or the ragged legal-set policy. Lets
/// `select_move` / `record_position` keep one signature and match only at the
/// sample / per-cluster-project site.
#[derive(Clone)]
enum MovePolicy {
    Dense(Vec<f32>),
    Ls(records::LegalSetPolicy),
}

impl MovePolicy {
    /// Sample a move from `legal` proportional to this policy's mass at each
    /// coord (ragged variant uses the `1/n` no-coverage floor, matching the
    /// dense path's uniform fallback for unpriored cells).
    fn sample(&self, legal: &[(i32, i32)], board: &Board, trunk: i32) -> Option<(i32, i32)> {
        match self {
            MovePolicy::Dense(p) => records::sample_policy(p, legal, board, trunk),
            MovePolicy::Ls(ls) => {
                let floor = 1.0 / legal.len().max(1) as f32;
                records::sample_policy_ls(ls, legal, board, trunk, floor)
            }
        }
    }
}

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
    legal_set: bool,
    infer: InferContext,
    variance: ClusterVarianceAtomics,
) -> usize {
    // ── WP-3 step 6: graph-seam dispatch (seam design §3.2) ──────────────────
    // Hoisted at the worker boundary (NOT per-sim): a `representation="graph"`
    // batcher routes to the ragged GNN path; every grid batcher (all 11 dense
    // encodings) returns `false` here, so the dense instruction stream below is
    // byte-identical (one never-taken predicted branch per leaf-batch, the graph
    // fn is `#[cold]`/`#[inline(never)]` so it never bloats the inlined hot path).
    if infer.batcher.is_graph() {
        return infer_and_expand_graph(tree, batch_size, agg_trunk_sz, infer);
    }

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

    // One arm allocates, the other is an empty `Vec::new()` (no heap alloc) — the
    // run uses exactly one policy_pool, so only the matching buffer is filled.
    let mut aggregated_policies: Vec<Vec<f32>> =
        if legal_set { Vec::new() } else { Vec::with_capacity(leaves.len()) };
    let mut aggregated_policies_ls: Vec<records::LegalSetPolicy> =
        if legal_set { Vec::with_capacity(leaves.len()) } else { Vec::new() };
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
        if legal_set {
            aggregated_policies_ls.push(records::aggregate_policy_ls(policy_stride, has_pass_slot, agg_trunk_sz, &leaves[i], centers, leaf_policies));
        } else {
            aggregated_policies.push(records::aggregate_policy(policy_stride, has_pass_slot, agg_trunk_sz, &leaves[i], centers, leaf_policies));
        }
    }

    let n = leaves.len();
    if legal_set {
        tree.expand_and_backup_ls(&aggregated_policies_ls, &aggregated_values);
    } else {
        tree.expand_and_backup(&aggregated_policies, &aggregated_values);
    }
    n
}

/// GNN counterpart of `infer_and_expand` (seam design §3.2). Selected once per
/// leaf-batch when the batcher is `representation="graph"` (never per-sim; the
/// dispatch is hoisted at the top of `infer_and_expand`). Builds ONE axis graph
/// per evaluated leaf (§S186 — never a search-time delta), submits the batch
/// through the parallel graph queue, and expands each leaf's assembled ragged
/// `LegalSetPolicy` via `expand_and_backup_ls_at`.
///
/// **S2 (review forward-flag, F1 coord/slot class):** the assembled
/// `LegalSetPolicy.dense` slots are baked against the BUILDER's per-leaf
/// `window_center` (via `policy_dst_slot`), so the builder centre is captured
/// here (`g.window_center`) and threaded into `expand_and_backup_ls_at` — the
/// read frame is the SAME object the slots were baked with, not a coincident
/// `board.window_center()` re-derivation.
///
/// Whole-board graph → one value per leaf (no K-cluster min-pool). v1
/// inference-time symmetry is DEFAULT-OFF (coord pre-rotation is WP-5 aug), so
/// `infer.sym_idx` is NOT applied on this path. Returns 0 (skip the batch,
/// matching the dense inference-failure degradation) on a build-guard trip or a
/// graph-inference failure.
#[cold]
#[inline(never)]
fn infer_and_expand_graph(
    tree: &mut MCTSTree,
    batch_size: usize,
    agg_trunk_sz: i32,
    infer: InferContext,
) -> usize {
    let leaves = tree.select_leaves(batch_size);
    if leaves.is_empty() { return 0; }

    let mut graphs = Vec::with_capacity(leaves.len());
    let mut centers = Vec::with_capacity(leaves.len());
    for leaf in &leaves {
        // Stone list from the board's sparse cell map (order irrelevant — the
        // builder coordinate-sorts). `Cell`/`Player` are `#[repr(i8)]`
        // (P1/One = 1, P2/Two = -1); the cast lands ±1 as the builder expects.
        let mut stones: Vec<(i64, i64, i64)> = Vec::new();
        for (&(q, r), &cell) in leaf.cells_iter() {
            stones.push((i64::from(q), i64::from(r), cell as i64));
        }
        let current_player = leaf.current_player as i64;
        let moves_remaining = i64::from(leaf.moves_remaining);
        match infer.batcher.build_leaf_graph(&stones, current_player, moves_remaining) {
            Some(g) => {
                centers.push(g.window_center);
                graphs.push(g);
            }
            // Seam guard tripped (unreachable for a valid self-play board — coords
            // bounded, player/moves in range). Degrade like a dense inference
            // failure: skip the whole batch.
            None => return 0,
        }
    }

    let results = match infer.batcher.submit_batch_and_wait_graph_rust(graphs) {
        Ok(r) => r,
        Err(()) => return 0,
    };
    if results.len() < leaves.len() { return 0; }

    let mut aggregated_ls: Vec<records::LegalSetPolicy> = Vec::with_capacity(results.len());
    let mut aggregated_values: Vec<f32> = Vec::with_capacity(results.len());
    for (ls, v) in results {
        aggregated_ls.push(ls);
        aggregated_values.push(v);
    }

    let n = leaves.len();
    // Expand frame trunk = spec.trunk_size (`agg_trunk_sz`), which is the SAME
    // trunk the builder used for `policy_dst_slot` (batcher `graph_trunk_size`).
    // ALWAYS-ON (graph-only call site; one integer compare per leaf batch —
    // release builds strip debug_asserts, and a trunk mismatch here misreads
    // every in-window slot).
    assert_eq!(
        agg_trunk_sz,
        infer.batcher.graph_trunk_size(),
        "graph trunk mismatch: spec agg_trunk_sz vs batcher graph_trunk_size"
    );
    tree.expand_and_backup_ls_at(&aggregated_ls, &aggregated_values, &centers, agg_trunk_sz);
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
/// Dirichlet root noise is PUCT-only. Under Gumbel, Gumbel-Top-k IS the root
/// exploration mechanism (ICLR2022, §104) — stacking Dirichlet on top inflates
/// the completed-Q policy-target floor ~2x, so the Gumbel arm applies NONE.
/// The `dirichlet_enabled/alpha/epsilon` params remain because the PUCT arm
/// still consumes them.
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
    legal_set: bool,
    infer: InferContext,
    variance: ClusterVarianceAtomics,
) -> McTSSearchResult {
    let mut gumbel_state: Option<GumbelSearchState> = None;

    if gumbel_mcts {
        // ── Gumbel MCTS with Sequential Halving ──
        let root_sims = infer_and_expand(tree, 1, kept_planes, n_cells, policy_stride, has_pass_slot, agg_trunk_sz, legal_set, infer, variance);
        if root_sims == 0 || !tree.pool[0].is_expanded() {
            return McTSSearchResult::RootExpansionFailed;
        }
        let mut sims_used = root_sims;

        // NO Dirichlet root noise under Gumbel: Gumbel-Top-k IS the root
        // exploration mechanism (ICLR2022, §104). Stacking Dirichlet on top
        // doubles the completed-Q policy-target floor. Dirichlet lives in the
        // PUCT `else` arm only.

        // Phase 2: Gumbel-Top-k candidate selection. Guard: if effective_m is
        // 0 (no budget or no children), fall back to the standard PUCT path.
        let effective_m = gumbel_m.min(move_sims).min(tree.root_n_children());
        if effective_m == 0 {
            let mut sims_done = sims_used;
            while sims_done < move_sims {
                if !running.load(Ordering::Relaxed) { break; }
                let n = infer_and_expand(tree, leaf_batch_size, kept_planes, n_cells, policy_stride, has_pass_slot, agg_trunk_sz, legal_set, infer, variance);
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
                    let n = infer_and_expand(tree, batch.max(1), kept_planes, n_cells, policy_stride, has_pass_slot, agg_trunk_sz, legal_set, infer, variance);
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
        let root_n = infer_and_expand(tree, 1, kept_planes, n_cells, policy_stride, has_pass_slot, agg_trunk_sz, legal_set, infer, variance);
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
            let n = infer_and_expand(tree, leaf_batch_size, kept_planes, n_cells, policy_stride, has_pass_slot, agg_trunk_sz, legal_set, infer, variance);
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
    graph_records_vec: &mut Vec<GraphRecord>,
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
    legal_set: bool,
    is_graph: bool,
    infer: InferContext,
    variance: ClusterVarianceAtomics,
    accumulators: MoveAccumulators,
    solver_counters: SolverCounters,
    solver_fires: &mut u32,
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
        kept_planes, n_cells, policy_stride, has_pass_slot, agg_trunk_sz, legal_set,
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
    let policy = if legal_set {
        MovePolicy::Ls(tree.get_policy_ls(temperature, policy_stride))
    } else {
        MovePolicy::Dense(tree.get_policy(temperature, policy_stride))
    };

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
    let mut target_policy = if ctx.completed_q_values {
        if legal_set {
            MovePolicy::Ls(tree.get_improved_policy_ls(policy_stride, ctx.c_visit, ctx.c_scale))
        } else {
            MovePolicy::Dense(tree.get_improved_policy(policy_stride, ctx.c_visit, ctx.c_scale))
        }
    } else {
        policy.clone()
    };

    // O1 (SootyOwl-validated): forced-win → (near-)one-hot POLICY target. When
    // search proves a within-turn forced win the soft visit distribution
    // under-weights (depth-1 immediate / depth-2 setup), harden the *training
    // target* to a one-hot on the proven winning move. Rides the same
    // winning-move primitive as the quiescence VALUE override; fires once per
    // move at target extraction — NOT in the per-sim hot path. When O1 fires the
    // row is forced full-search so the ground-truth target always reaches
    // `compute_policy_loss` (PCR's `full_search_mask` would otherwise drop the
    // ~half of forced-win rows that PCR sampled as quick-search).
    let forced_win_fired = ctx.forced_win_enabled
        && match board.forced_win_move(ctx.forced_win_depth) {
            Some((wq, wr)) => match &mut target_policy {
                MovePolicy::Dense(t) => {
                    let action = board.window_flat_idx(wq, wr);
                    if action < policy_stride {
                        records::apply_forced_win_one_hot(t, action, ctx.forced_win_weight);
                        true
                    } else {
                        false
                    }
                }
                // §9.2a/§9.3: coverage-gated — a covered (in- or off-window) win
                // one-hots; an uncovered win is a no-op (no uniform-fallback
                // corruption). Coverage uses the SAME centers as the record-time
                // projection (`board` is the pre-move search root; §9.8).
                MovePolicy::Ls(ls) => {
                    let (bcq, bcr) = board.window_center();
                    let half = (agg_trunk_sz - 1) / 2;
                    let (_, centers) = board.get_cluster_views();
                    let covered = records::is_covered(wq, wr, &centers, agg_trunk_sz, half);
                    records::apply_forced_win_one_hot_ls(
                        ls, (wq, wr), ctx.forced_win_weight, covered, bcq, bcr, agg_trunk_sz, half,
                    )
                }
            },
            None => false,
        };

    // D-WS3 L1 — native solver-in-loop SOFT visit-injection. The net-free
    // `engine::tactics` solver proves the side-to-move's forced win and SOFT-
    // injects visit mass onto the proving move's first stone (`line[0]`) into the
    // POLICY training target. This is the L1 "teach the policy the saving move"
    // lever: a generalisation of the O1 forced-win one-hot to (a) the native deep
    // solver (recall past `forced_win_move`'s within-turn horizon — the ~80%-quiet
    // trap class needs `neighbor_dist` widening, A2/D-TACTICAL), (b) SOFT injection
    // (`visit_weight < 1` convex blend — NOT one-hot, which the dpfit probe showed
    // is collaterally destructive; graft A INJECT-not-reweight reaches the 67%
    // ~0-prior saving moves), (c) `window_half = None` so OFF-WINDOW forced wins
    // are surfaced and routed through the multi-window legal_set coverage gate into
    // the ragged target (the D-DECODE action-space fix; 60% of saves are off-
    // window). WIN proofs are SOUND BY CONSTRUCTION (terminal backup; the attacker
    // plays only threat moves → defender in-check at every node → `must_block`), so
    // the L1 policy signal needs no R3 LOSS guard — that guard governs the L2
    // value-z LOSS path (HELD), and this smoke writes NO LOSS-derived z. Fires once
    // per move at target extraction, NOT in the per-sim hot path; `solver_enabled
    // = false` (default) keeps the bench-gated hot path byte-identical. The O1 and
    // solver levers compose (the z2 smoke runs O1 off + solver on); both inject the
    // winning move — a double-fire is safe (additive convex blend, weight saturates
    // at 1.0; NOT idempotent — `1-(1-wA)(1-wB)`).
    let solver_fired = if ctx.solver_enabled {
        // D-WS3V3 in-run fire-rate: the solver runs on EVERY move under
        // `solver_enabled`, so every move reaching here is "eligible".
        solver_counters.moves_eligible.fetch_add(1, Ordering::Relaxed);
        if ctx.seeded {
            solver_counters.moves_eligible_seeded.fetch_add(1, Ordering::Relaxed);
        }
        let cfg = crate::tactics::TacticalConfig {
            cand_cap: 40,
            // legal_set (multi-window): None surfaces off-window forced wins; the
            // coverage gate (below) gives them a ragged-target slot — the saving moves
            // L1 must teach. DENSE (single-window): an off-window win has NO logit slot
            // (`window_flat_idx` -> usize::MAX -> dropped), so surfacing it would just
            // silently no-op (red-team F3); keep the single-window guard so the solver
            // only spends budget on the expressible action space.
            window_half: if legal_set { None } else { Some((agg_trunk_sz - 1) / 2) },
            // Quiet-move widening for recall on the quiet trap class. < 0 → None
            // (threat-only). The native in-tree solver carries the neighbor_dist
            // body (NOT the stale pre-neighbor_dist Python binding).
            neighbor_dist: if ctx.solver_neighbor_dist < 0 {
                None
            } else {
                Some(ctx.solver_neighbor_dist)
            },
        };
        let proof = crate::tactics::TacticalSolver::new(cfg).prove(
            board,
            ctx.solver_depth,
            ctx.solver_node_budget,
        );
        if proof.budget_exhausted {
            solver_counters.budget_exhausted.fetch_add(1, Ordering::Relaxed);
        }
        if proof.result == crate::tactics::Outcome::Win {
            solver_counters.win_proven.fetch_add(1, Ordering::Relaxed);
            match proof.line.first() {
                // (injected, off_window) — off_window is only reachable on the LS
                // path (the Dense path drops an off-window win: no logit slot).
                Some(&(wq, wr)) => {
                    let (injected, off_window) = match &mut target_policy {
                        MovePolicy::Dense(t) => {
                            let action = board.window_flat_idx(wq, wr);
                            if action < policy_stride {
                                records::apply_forced_win_one_hot(t, action, ctx.solver_visit_weight);
                                (true, false)
                            } else {
                                (false, false)
                            }
                        }
                        // §9.2a/§9.3 coverage-gated (mirrors the O1 LS path): a covered
                        // (in- or off-window) win injects; an uncovered win is a no-op
                        // (no uniform-fallback corruption). Coverage uses the SAME
                        // centers as the record-time projection (`board` is the pre-move
                        // search root).
                        MovePolicy::Ls(ls) => {
                            let (bcq, bcr) = board.window_center();
                            let half = (agg_trunk_sz - 1) / 2;
                            let (_, centers) = board.get_cluster_views();
                            let covered = records::is_covered(wq, wr, &centers, agg_trunk_sz, half);
                            let did = records::apply_forced_win_one_hot_ls(
                                ls, (wq, wr), ctx.solver_visit_weight, covered, bcq, bcr,
                                agg_trunk_sz, half,
                            );
                            // Off-window = injected into the ragged OVERFLOW target: the
                            // win maps outside the dense global window
                            // (`window_flat_idx_at_geom` == usize::MAX >= policy_stride).
                            let off = did
                                && Board::window_flat_idx_at_geom(wq, wr, bcq, bcr, agg_trunk_sz, half)
                                    >= policy_stride;
                            (did, off)
                        }
                    };
                    if injected {
                        solver_counters.injected.fetch_add(1, Ordering::Relaxed);
                        if ctx.seeded {
                            solver_counters.injected_seeded.fetch_add(1, Ordering::Relaxed);
                        }
                        if off_window {
                            solver_counters.injected_offwindow.fetch_add(1, Ordering::Relaxed);
                        }
                        *solver_fires += 1;
                    }
                    injected
                }
                None => false,
            }
        } else {
            false
        }
    } else {
        false
    };
    let record_full_search = move_is_full_search || forced_win_fired || solver_fired;

    // ── Sample and apply move (ZOI-filtered legal set) ──
    let Some(move_idx) = select_move(
        board, move_history, &policy, gumbel_state, ctx, agg_trunk_sz, tree, rng,
    ) else {
        return MoveOutcome::Break;
    };

    // ── Record position ──
    // GNN-integration WP-5b commit A (R4, delta doc §2 point 1): ONE hoisted
    // branch. `is_graph` is a `WorkerGeometry` Copy bool set once per worker
    // spawn; grid = false → this branch is never taken and `record_position`
    // runs unchanged. `record_position_graph_dispatch` is `#[cold]`/
    // `#[inline(never)]` so it never bloats the inlined dense record path.
    if is_graph {
        record_position_graph_dispatch(
            board, &target_policy, agg_trunk_sz, record_full_search, graph_records_vec,
        );
    } else {
        record_position(
            board, kept_planes, n_cells, agg_trunk_sz, ctx.is_fast_game,
            ctx.completed_q_values, policy_stride, has_pass_slot, &target_policy,
            ctx.sym_idx, infer.sym_tables, record_full_search, records_vec,
        );
    }

    if board.apply_move(move_idx.0, move_idx.1).is_err() {
        return MoveOutcome::Break;
    }
    move_history.push((move_idx.0, move_idx.1));
    accumulators.positions_generated.fetch_add(1, Ordering::Relaxed);
    MoveOutcome::Played
}

/// D-WS3V3 — Gumbel-explore gate on ply RELATIVE to game start.
///
/// Returns true once `ply - game_start_ply >= explore_moves` (the deterministic
/// argmax winner regime). `game_start_ply == 0` for organic games → the gate is
/// byte-identical to the pre-D-WS3V3 absolute-ply test. A seeded game
/// (`game_start_ply == prefix_len`) explores for `explore_moves` moves AFTER its
/// deep start instead of collapsing to argmax from move 1 (the D-ARGMAX dup trap).
#[inline]
fn relative_explore_gate(ply: usize, game_start_ply: usize, explore_moves: usize) -> bool {
    ply.saturating_sub(game_start_ply) >= explore_moves
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
    policy: &MovePolicy,
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
    // D-WS3V3: gate exploration on ply RELATIVE to game start (`game_start_ply` ==
    // seed prefix_len; 0 for organic games → byte-identical). A seeded game starts
    // deep (ply 40-80); an absolute-ply gate would take deterministic argmax from
    // its first move and collapse the fixed corpus into near-duplicate games (the
    // D-ARGMAX dup trap). The temperature schedule stays on ABSOLUTE ply (organic
    // regime) — see MovePlayContext.
    let use_gumbel_winner = gumbel_state.is_some()
        && relative_explore_gate(board.ply as usize, ctx.game_start_ply, ctx.gumbel_explore_moves);
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
            match policy.sample(&legal, board, agg_trunk_sz) {
                Some(idx) => idx,
                None => *legal.choose(rng).unwrap(),
            }
        }
    } else {
        match policy.sample(&legal, board, agg_trunk_sz) {
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
    target_policy: &MovePolicy,
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
        // O1 caveat (defense-in-depth): this `is_fast_game && !completed_q_values`
        // branch discards `target_policy` entirely, so an O1 forced-win one-hot
        // is NOT recorded here. Dormant under every O1 config — `is_fast_game`
        // needs `fast_prob>0`, which `pool.py` makes mutually exclusive with the
        // PCR `full_search_prob` path, and the O1 lineage runs `completed_q_values
        // =true`. If game-level fast games are ever re-enabled WITH O1, route the
        // one-hot through here too (or keep `completed_q_values=true`).
        let mut projected_policy = if is_fast_game && !completed_q_values {
            vec![0.0; policy_stride]
        } else {
            match target_policy {
                MovePolicy::Dense(t) => records::aggregate_policy_to_local(policy_stride, has_pass_slot, agg_trunk_sz, board, center, t, &record_legal_moves),
                MovePolicy::Ls(ls) => records::aggregate_policy_to_local_ls(policy_stride, has_pass_slot, agg_trunk_sz, board, center, ls, &record_legal_moves),
            }
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

/// GNN-integration WP-5b commit A (R4) — graph sibling of `record_position`.
/// Whole-board (NO K-cluster loop, NO dense planes) — pushes ONE compact
/// `GraphRecord` via the WP-5a `records::record_position_graph` primitive
/// (records.rs — no new correctness logic here, this fn is pure dispatch).
///
/// `target_policy` is guaranteed `MovePolicy::Ls` whenever this is called:
/// `is_graph` and `legal_set` are BOTH derived from `spec.is_graph()` (R1 —
/// `legal_set = spec.is_graph() || matches!(policy_pool, LegalSetScatterMax)`),
/// so a graph spec always forces `legal_set = true`, and `play_one_move` only
/// ever builds `MovePolicy::Dense` when `legal_set == false`. The `Dense` arm
/// is therefore structurally unreachable, not a runtime data condition — an
/// `unreachable!()` (always-on, not a stripped `debug_assert!`) is the
/// correct die-loud response per the delta doc §9 failure-modes table
/// ("legal_set=false for graph -> ... -> silent" is exactly the class this
/// forecloses at construction).
///
/// `#[cold]`/`#[inline(never)]` — mirrors the WP-3 step-6 graph dispatch
/// (`infer_and_expand_graph`) so this call never bloats the dense inlined
/// record path (§2 point 1 of the delta doc).
#[cold]
#[inline(never)]
fn record_position_graph_dispatch(
    board: &Board,
    target_policy: &MovePolicy,
    trunk_sz: i32,
    move_is_full_search: bool,
    graph_records_vec: &mut Vec<GraphRecord>,
) {
    let ls = match target_policy {
        MovePolicy::Ls(ls) => ls,
        MovePolicy::Dense(_) => unreachable!(
            "record_position_graph_dispatch: target_policy must be MovePolicy::Ls for a \
             graph spec — R1 forces legal_set=true whenever spec.is_graph() is true"
        ),
    };
    let current_player = board.current_player as i8;
    let moves_remaining = board.moves_remaining;
    let ply_index = board.ply as u16;
    let rec = records::record_position_graph(
        board, ls, trunk_sz, current_player, moves_remaining, ply_index,
        move_is_full_search, MAX_VISITS,
    );
    graph_records_vec.push(rec);
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
    seeded: bool,
    solver_fires: u32,
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
            u8::from(seeded), solver_fires,
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

/// GNN-integration WP-5b commit A (R5) — graph sibling of `finalize_game`.
///
/// Reuses the winner / `terminal_reason` / `version_seen` classification
/// verbatim (byte-identical duplication of `finalize_game`'s L1567-1601 —
/// grid's `finalize_game` is untouched, this is a new fn with no dense
/// caller). Stamps each buffered `GraphRecord`'s `outcome`/`value_valid` via
/// the WP-5a §178 KEEP-verbatim `records::finalize_graph_outcome` (INV26
/// transfers unchanged — records.rs docs), sets `game_length` (compound-move
/// count, the same `(plies+1)/2` convention `pool.py` applies to the dense
/// `plies` field before push), and drains into `graph_results_queue`. NO
/// cell-geometry reprojection (no ownership/winning_line aux — `GnnNet` has
/// no consuming head, design §1.3 DROP). Pushes the SAME
/// `recent_game_results` metadata row `finalize_game` does (representation-
/// agnostic — `pool.py::_run_stats_loop` processes `drain_game_results()`
/// identically for both representations) and increments the same win/draw
/// counters. Caps `graph_results_queue` at `results_queue_cap` (parity with
/// the dense backpressure drop — reuses the existing `positions_dropped`
/// counter; not a new correctness primitive, just the same hygiene pattern
/// applied to the new queue).
#[allow(clippy::too_many_arguments)] // per-game finalize; mirrors finalize_game's arity for the same reason
fn finalize_game_graph(
    board: &Board,
    max_moves: usize,
    graph_records: Vec<GraphRecord>,
    move_history: Vec<(i32, i32)>,
    version_seen: &[u64],
    draw_reward: f32,
    ply_cap_value: f32,
    results_queue_cap: usize,
    worker_id: usize,
    seeded: bool,
    solver_fires: u32,
    graph_results_queue: &Mutex<VecDeque<GraphRecord>>,
    recent_game_results: &Mutex<VecDeque<GameResultRow>>,
    games_completed: &AtomicUsize,
    x_wins: &AtomicU64,
    o_wins: &AtomicU64,
    draws: &AtomicU64,
    positions_dropped: &AtomicU64,
) {
    // ── Game End: determine outcome (verbatim twin of finalize_game) ──
    let winner = board.winner();
    let plies = board.ply as usize;
    let winner_code: u8 = match winner {
        Some(crate::board::Player::One) => 1,
        Some(_)                         => 2,
        None                            => 0,
    };
    let winning_cells: Vec<(i32, i32)> = board.find_winning_line();
    let terminal_reason: u8 = match winner {
        Some(_) => u8::from(winning_cells.is_empty()),
        None    => if plies >= max_moves { 2 } else { 3 },
    };
    let (mv_min, mv_max, mv_distinct) = if version_seen.is_empty() {
        (0u64, 0u64, 0u32)
    } else {
        let mn = *version_seen.iter().min().unwrap();
        let mx = *version_seen.iter().max().unwrap();
        (mn, mx, version_seen.len() as u32)
    };
    // Compound-move sampling weight — same `(plies+1)/2` (== `div_ceil(2)`,
    // matching `compound_move` above) convention `pool.py::_run_stats_loop`
    // applies to the dense `plies` field before `push_many` (the HEXG push
    // takes `game_length` directly, no Python-side conversion step in the
    // graph drain — delta doc §6).
    let game_length: u16 = plies.div_ceil(2).min(u16::MAX as usize) as u16;

    let mut gq = graph_results_queue.lock().expect("graph_results_queue lock poisoned");
    for mut rec in graph_records {
        // §178 KEEP-verbatim split (records::finalize_graph_outcome, WP-5a
        // unit-tested 4 cases) — reads rec.current_player/winner/terminal_reason
        // only, no cell geometry, so INV26 transfers unchanged.
        let (outcome, value_valid_u8) = records::finalize_graph_outcome(
            rec.current_player, winner, terminal_reason, ply_cap_value, draw_reward,
        );
        rec.outcome = outcome;
        rec.value_valid = value_valid_u8 != 0;
        rec.game_length = game_length;
        gq.push_back(rec);
    }
    games_completed.fetch_add(1, Ordering::Relaxed);
    match winner {
        Some(crate::board::Player::One) => { x_wins.fetch_add(1, Ordering::Relaxed); }
        Some(_)                         => { o_wins.fetch_add(1, Ordering::Relaxed); }
        None                            => { draws.fetch_add(1, Ordering::Relaxed); }
    }
    // Metadata-only game-complete row — SAME queue/schema `finalize_game`
    // pushes into; `pool.py::_run_stats_loop` drains it representation-blind.
    {
        let mut rg = recent_game_results.lock().expect("recent_game_results lock poisoned");
        rg.push_back((
            plies, winner_code, move_history, worker_id,
            terminal_reason, mv_min, mv_max, mv_distinct,
            u8::from(seeded), solver_fires,
        ));
        if rg.len() > 2000 {
            rg.pop_front();
        }
    }

    // Cap the graph results queue to avoid memory explosion if Python is slow
    // (parity with `finalize_game`'s dense backpressure drop).
    if gq.len() > results_queue_cap {
        let to_drop = gq.len() - results_queue_cap;
        for _ in 0..to_drop {
            gq.pop_front();
        }
        positions_dropped.fetch_add(to_drop as u64, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod seeding_tests {
    //! D-WS3V3 — start-position seeding + relative-explore gate + the dense
    //! proven-vs-injected classification predicate. The full solver-injection
    //! counter path fires only inside self-play (no per-move behavioural unit test
    //! exercises play_one_move end-to-end — same constraint as O1); these pin the
    //! load-bearing seeding semantics + the exact predicates the counter code uses.
    use super::*;
    use crate::board::Cell;
    use std::sync::Arc;

    fn test_init_ctx() -> PerGameInitCtx {
        PerGameInitCtx {
            max_moves: 150,
            random_opening_plies: 1,
            legal_move_radius_jitter: false,
            selfplay_rotation_enabled: false,
            fast_prob: 0.0,
            fast_sims: 64,
            standard_sims: 200,
            n_cells: 361,
            draw_reward: -0.1,
            ply_cap_value: -0.1,
            results_queue_cap: 10_000,
            worker_id: 0,
        }
    }

    #[test]
    fn test_seeded_prefix_replay_matches_manual_apply() {
        // (a) A seeded game's board == the same prefix applied stone-by-stone on a
        // fresh board (state fully derived by replay: ply, current_player,
        // moves_remaining, stones).
        let prefix = vec![(0, 0), (1, 0), (2, 0)];
        let seed = SeedCorpus { corpus: Arc::new(vec![prefix.clone()]), seed_fraction: 1.0 };
        let radius = AtomicI32::new(-1);
        let mut rng = rng();
        let mut version_seen = Vec::new();
        let init = init_per_game_board(None, test_init_ctx(), &radius, &mut rng, &mut version_seen, &seed);
        assert!(init.seeded, "seed_fraction=1.0 + non-empty corpus must seed");
        assert_eq!(init.prefix_len, 3);
        assert_eq!(init.move_history, prefix, "seed prefix goes into move_history");

        let mut manual = Board::new();
        for &(q, r) in &prefix {
            manual.apply_move(q, r).unwrap();
        }
        assert_eq!(init.board.ply, manual.ply, "ply derived by replay");
        assert_eq!(init.board.current_player, manual.current_player);
        assert_eq!(init.board.moves_remaining, manual.moves_remaining);
        assert_eq!(init.board.cells, manual.cells, "stones match manual replay");
    }

    #[test]
    fn test_seed_fraction_zero_never_seeds() {
        // (c) seed_fraction=0.0 short-circuits BEFORE the rng draw (`!empty &&
        // fraction>0 && rng<..` — `&&` is left-to-right) so the default path's rng
        // stream is untouched (INV25/26-class). Observable: never seeded.
        let seed = SeedCorpus { corpus: Arc::new(vec![vec![(0, 0), (1, 0)]]), seed_fraction: 0.0 };
        let radius = AtomicI32::new(-1);
        let mut rng = rng();
        let mut version_seen = Vec::new();
        let init = init_per_game_board(None, test_init_ctx(), &radius, &mut rng, &mut version_seen, &seed);
        assert!(!init.seeded, "seed_fraction=0.0 must never seed");
        assert_eq!(init.prefix_len, 0);
        assert_eq!(init.board.ply, 0, "unseeded game starts at ply 0");
    }

    #[test]
    fn test_empty_corpus_never_seeds() {
        // (c') empty corpus short-circuits before the rng draw even at fraction=1.
        let seed = SeedCorpus { corpus: Arc::new(Vec::new()), seed_fraction: 1.0 };
        let radius = AtomicI32::new(-1);
        let mut rng = rng();
        let mut version_seen = Vec::new();
        let init = init_per_game_board(None, test_init_ctx(), &radius, &mut rng, &mut version_seen, &seed);
        assert!(!init.seeded);
        assert_eq!(init.prefix_len, 0);
    }

    #[test]
    fn test_relative_explore_gate_uses_relative_ply() {
        // (d) organic (game_start_ply=0) is byte-identical to the absolute-ply test.
        assert!(!relative_explore_gate(9, 0, 10));
        assert!(relative_explore_gate(10, 0, 10));
        // A seeded game starting at ply 40 must still explore for `explore_moves`
        // moves AFTER its deep start — NOT collapse to argmax from move 1.
        assert!(!relative_explore_gate(40, 40, 10), "seeded first move must explore");
        assert!(!relative_explore_gate(49, 40, 10));
        assert!(relative_explore_gate(50, 40, 10), "gate fires 10 plies past the seed start");
    }

    #[test]
    fn test_dense_offwindow_win_is_proven_not_injected() {
        // (e) Dense path injection is gated on `window_flat_idx(wq,wr) <
        // policy_stride`. An off-window proven win maps to usize::MAX (>= stride) →
        // PROVEN but NOT injected (the win_proven vs injected counter split); an
        // in-window win maps to a real logit slot (injected).
        let mut board = Board::new();
        for q in 0..5i32 { board.cells.insert((q, 0), Cell::P1); }
        board.has_stones = true;
        board.min_q = -1; board.max_q = 5; board.min_r = 0; board.max_r = 0;
        board.cache_dirty.set(true);
        let stride = 19 * 19 + 1;
        assert!(board.window_flat_idx(5, 0) < stride, "in-window win maps to a logit slot (injected)");
        assert_eq!(board.window_flat_idx(100, 0), usize::MAX, "off-window win has no dense slot (proven-not-injected)");
    }
}

#[cfg(test)]
mod graph_finalize_tests {
    //! GNN-integration WP-5b commit A (R5, delta doc §10 test-plan row 2) —
    //! `finalize_game_graph` LIVE-LOOP wiring test. Drives the real fn
    //! end-to-end (not just the pure `records::finalize_graph_outcome` helper
    //! `records.rs` already unit-tests): mocked terminal boards for each
    //! `terminal_reason` class -> drained `GraphRecord.outcome`/`value_valid`
    //! must match, AND the `recent_game_results` metadata + win/draw counter
    //! side effects must fire exactly like `finalize_game`'s dense counterpart.
    //! `winner()`/`find_winning_line()` scan `board.cells` directly (no bbox /
    //! `legal_moves()` dependency — verified via source read), so a direct
    //! cell-insert is a faithful terminal board (mirrors `seeding_tests`'
    //! `test_dense_offwindow_win_is_proven_not_injected` pattern).
    use super::*;
    use crate::board::Cell;

    fn mk_record(current_player: i8) -> GraphRecord {
        GraphRecord {
            stones: vec![],
            visits: vec![],
            current_player,
            moves_remaining: 2,
            ply_index: 0,
            is_full_search: true,
            outcome: 0.0,      // placeholder — finalize_game_graph must overwrite
            value_valid: true, // placeholder — finalize_game_graph must overwrite
            game_length: 0,    // placeholder — finalize_game_graph must overwrite
        }
    }

    type Counters = (AtomicUsize, AtomicU64, AtomicU64, AtomicU64, AtomicU64);
    fn counters() -> Counters {
        (AtomicUsize::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0))
    }

    #[test]
    fn finalize_game_graph_win_loss_split_matches_178() {
        // 6-in-a-row for Player::One along the q axis.
        let mut board = Board::new();
        for q in 0..6i32 {
            board.cells.insert((q, 0), Cell::P1);
        }
        board.ply = 11; // winner short-circuits terminal_reason; game_length = (11+1)/2 = 6

        let gq: Mutex<VecDeque<GraphRecord>> = Mutex::new(VecDeque::new());
        let rg: Mutex<VecDeque<GameResultRow>> = Mutex::new(VecDeque::new());
        let (games_completed, x_wins, o_wins, draws, positions_dropped) = counters();

        // Two rows from the SAME game: one recorded when P1 (the eventual
        // winner) was to move — +1; one when P2 (the loser) was to move — −1.
        let records = vec![mk_record(1), mk_record(-1)];

        finalize_game_graph(
            &board, 150, records, vec![(0, 0), (1, 0)], &[],
            -0.5, -0.5, 10_000, 0, false, 0,
            &gq, &rg, &games_completed, &x_wins, &o_wins, &draws, &positions_dropped,
        );

        let drained: Vec<GraphRecord> = gq.lock().unwrap().drain(..).collect();
        assert_eq!(drained.len(), 2);
        let winner_row = drained.iter().find(|r| r.current_player == 1).unwrap();
        let loser_row = drained.iter().find(|r| r.current_player == -1).unwrap();
        assert_eq!((winner_row.outcome, winner_row.value_valid), (1.0, true));
        assert_eq!((loser_row.outcome, loser_row.value_valid), (-1.0, true));
        assert_eq!(winner_row.game_length, 6, "game_length = (ply+1)/2 compound moves");

        assert_eq!(games_completed.load(Ordering::Relaxed), 1);
        assert_eq!(x_wins.load(Ordering::Relaxed), 1);
        assert_eq!(o_wins.load(Ordering::Relaxed), 0);
        assert_eq!(draws.load(Ordering::Relaxed), 0);

        let meta = rg.lock().unwrap();
        assert_eq!(meta.len(), 1, "one recent_game_results metadata row per game");
        assert_eq!(meta[0].1, 1, "winner_code == 1 for Player::One");
    }

    #[test]
    fn finalize_game_graph_ply_cap_masks_value_valid() {
        let mut board = Board::new(); // empty — no winner
        board.ply = 150; // >= max_moves(150) -> terminal_reason 2 (ply-cap)

        let gq: Mutex<VecDeque<GraphRecord>> = Mutex::new(VecDeque::new());
        let rg: Mutex<VecDeque<GameResultRow>> = Mutex::new(VecDeque::new());
        let (games_completed, x_wins, o_wins, draws, positions_dropped) = counters();

        finalize_game_graph(
            &board, 150, vec![mk_record(1)], vec![], &[],
            /* draw_reward */ -0.1, /* ply_cap_value */ -0.5, 10_000, 0, false, 0,
            &gq, &rg, &games_completed, &x_wins, &o_wins, &draws, &positions_dropped,
        );

        let drained: Vec<GraphRecord> = gq.lock().unwrap().drain(..).collect();
        assert_eq!(drained.len(), 1);
        assert_eq!(
            (drained[0].outcome, drained[0].value_valid), (-0.5, false),
            "ply-cap row pays ply_cap_value and is MASKED (value_valid=false)"
        );
        assert_eq!(draws.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn finalize_game_graph_organic_draw_stays_supervised() {
        let mut board = Board::new(); // empty — no winner
        board.ply = 5; // < max_moves(150) -> terminal_reason 3 (organic draw)

        let gq: Mutex<VecDeque<GraphRecord>> = Mutex::new(VecDeque::new());
        let rg: Mutex<VecDeque<GameResultRow>> = Mutex::new(VecDeque::new());
        let (games_completed, x_wins, o_wins, draws, positions_dropped) = counters();

        finalize_game_graph(
            &board, 150, vec![mk_record(1)], vec![], &[],
            /* draw_reward */ -0.1, /* ply_cap_value */ -0.5, 10_000, 0, false, 0,
            &gq, &rg, &games_completed, &x_wins, &o_wins, &draws, &positions_dropped,
        );

        let drained: Vec<GraphRecord> = gq.lock().unwrap().drain(..).collect();
        assert_eq!(drained.len(), 1);
        assert_eq!(
            (drained[0].outcome, drained[0].value_valid), (-0.1, true),
            "organic-draw row pays draw_reward and stays SUPERVISED (value_valid=true)"
        );
        assert_eq!(draws.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn finalize_game_graph_caps_queue_at_results_queue_cap() {
        let board = Board::new();
        let gq: Mutex<VecDeque<GraphRecord>> = Mutex::new(VecDeque::new());
        let rg: Mutex<VecDeque<GameResultRow>> = Mutex::new(VecDeque::new());
        let (games_completed, x_wins, o_wins, draws, positions_dropped) = counters();

        let records: Vec<GraphRecord> = (0..5).map(|_| mk_record(1)).collect();
        finalize_game_graph(
            &board, 150, records, vec![], &[],
            -0.1, -0.5, /* results_queue_cap */ 3, 0, false, 0,
            &gq, &rg, &games_completed, &x_wins, &o_wins, &draws, &positions_dropped,
        );

        assert_eq!(gq.lock().unwrap().len(), 3, "backpressure drop caps the graph queue like the dense queue");
        assert_eq!(positions_dropped.load(Ordering::Relaxed), 2);
    }
}

