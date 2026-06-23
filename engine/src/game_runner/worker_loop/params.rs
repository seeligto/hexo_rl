//! `WorkerParams` + themed sub-flag bundles + per-worker geometry
//! (cycle 3 Wave 10 Batch A).
//!
//! Extracted verbatim from the pre-split `worker_loop.rs` (§P52 capture-bundle
//! prototype build at L329-368; Wave 7 Batch C themed sub-flag structs at
//! L200-217). The 3 themed `*Flags` sub-structs partition the 7 WorkerParams
//! bool fields into ≤3-bool groups so `clippy::struct_excessive_bools` no
//! longer fires on `WorkerParams` itself nor on any sub-struct.
//!
//! All 4 bundles are `#[derive(Clone)]` — cloned once per worker spawn from
//! the prototype; destructured at `inner::run_worker_thread` fn entry.
//! Field-level visibility is `pub(super)` to enable the cross-module
//! destructure pattern preservation (Wave 10 Batch A invariant).

/// Per-worker geometry scalars pre-extracted from `RegistrySpec` at spawn-loop
/// time (Wave 10 Batch A). 5 `Copy` scalars passed by value to
/// `inner::run_worker_thread`. Bundled to keep the fn arity ≤ 7 (avoids
/// the `clippy::too_many_arguments` lint without adding a new suppression;
/// preserves the Wave 9 close lint-suppression baseline per U10/J.2.a).
/// All fields are stack-cheap; per-sim hot-path access goes via fn-entry
/// destructure into local scalar bindings (per
/// `feedback_registryspec_by_ref_in_hotpath.md` — bundle exists as fn-arg
/// ergonomics, NOT as hot-path field-access cost).
#[derive(Clone, Copy)]
pub(super) struct WorkerGeometry {
    pub(super) n_cells: usize,
    pub(super) kept_planes: &'static [usize],
    pub(super) policy_stride: usize,
    pub(super) agg_trunk_sz: i32,
    pub(super) has_pass_slot: bool,
    /// §D-MULTICLUSTER-S0: when true the encoding's `policy_pool` is
    /// `LegalSetScatterMax` — the worker uses the ragged legal-set MCTS prior /
    /// improved-policy target (no off-window drop) instead of the dense path.
    pub(super) legal_set: bool,
}

#[derive(Clone)]
pub(super) struct SearchFlags {
    pub(super) quiescence_enabled: bool,
    pub(super) completed_q_values: bool,
    pub(super) gumbel_mcts: bool,
}

#[derive(Clone)]
pub(super) struct ExplorationFlags {
    pub(super) dirichlet_enabled: bool,
    pub(super) selfplay_rotation_enabled: bool,
}

#[derive(Clone)]
pub(super) struct MoveConstraintFlags {
    pub(super) zoi_enabled: bool,
    pub(super) legal_move_radius_jitter: bool,
}

/// O1 (SootyOwl-validated) forced-win → one-hot POLICY target knobs. Themed
/// sub-bundle (mirrors `SearchFlags`/`MoveConstraintFlags`) threaded through the
/// `WorkerParams` prototype to `inner::run_worker_thread`. `enabled == false`
/// (default) makes the per-move override a no-op — byte-identical to pre-O1.
#[derive(Clone, Copy)]
pub(super) struct ForcedWinPolicy {
    pub(super) enabled: bool,
    pub(super) depth: u8,
    pub(super) weight: f32,
}

#[derive(Clone)]
pub(super) struct WorkerParams {
    pub(super) max_moves: usize,
    pub(super) leaf_batch_size: usize,
    pub(super) c_puct: f32,
    pub(super) fpu_reduction: f32,
    pub(super) quiescence_blend_2: f32,
    pub(super) fast_prob: f32,
    pub(super) fast_sims: usize,
    pub(super) standard_sims: usize,
    pub(super) temp_threshold: usize,
    pub(super) temp_min: f32,
    pub(super) draw_reward: f32,
    /// §178: terminal-via-ply-cap outcome (winner=None AND ply>=max_moves).
    /// Split from `draw_reward` so the value head sees distinct targets for
    /// organic draws (`terminal_reason==3`) vs ply-cap truncations
    /// (`terminal_reason==2`). See `inner::finalize_game`.
    pub(super) ply_cap_value: f32,
    pub(super) zoi_lookback: usize,
    pub(super) zoi_margin: i32,
    pub(super) c_visit: f32,
    pub(super) c_scale: f32,
    pub(super) gumbel_m: usize,
    pub(super) gumbel_explore_moves: usize,
    pub(super) dirichlet_alpha: f32,
    pub(super) dirichlet_epsilon: f32,
    pub(super) results_queue_cap: usize,
    pub(super) full_search_prob: f32,
    pub(super) n_sims_quick: usize,
    pub(super) n_sims_full: usize,
    pub(super) random_opening_plies: u32,
    pub(super) registry_spec: Option<&'static crate::encoding::RegistrySpec>,
    pub(super) search_flags: SearchFlags,
    pub(super) exploration_flags: ExplorationFlags,
    pub(super) move_constraint_flags: MoveConstraintFlags,
    pub(super) forced_win_policy: ForcedWinPolicy,
    /// D-QFIX-LAND A1: interior (non-root) MCTS selection rule. Cloned from the
    /// runner prototype per worker spawn, applied to the per-worker `MCTSTree`
    /// after `new_full` in `inner::run_worker_thread`. `Copy` enum.
    pub(super) interior_selector: crate::mcts::InteriorSelector,
}
