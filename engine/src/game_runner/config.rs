//! `SelfPlayRunnerConfig` — fold of the 38-kwarg `SelfPlayRunner::new` parameter list
//! into a single `#[pyclass]` builder struct (cycle 3 Wave 7 Batch A, P79).
//!
//! Construction from Python:
//! ```python
//! from engine import SelfPlayRunner, SelfPlayRunnerConfig
//! runner = SelfPlayRunner(SelfPlayRunnerConfig(n_workers=4, max_moves_per_game=128, ...))
//! ```
//!
//! Construction from Rust (e.g. integration tests):
//! ```ignore
//! use engine::game_runner::{SelfPlayRunner, SelfPlayRunnerConfig};
//! let config = SelfPlayRunnerConfig::new(
//!     4, 128, 50, 8, 1.5, 0.25, Some(8 * 19 * 19), Some(19 * 19 + 1),
//!     /* ... remaining 30 args ... */
//! );
//! let runner = SelfPlayRunner::new(config)?;
//! ```
//!
//! The 38 fields mirror the prior `SelfPlayRunner::new` parameter list exactly;
//! INV19 (`engine/tests/inv19_selfplayrunner_config_builder_byte_equivalence.rs`)
//! pins the byte-equivalence contract.

use pyo3::prelude::*;

/// Configuration for [`crate::game_runner::SelfPlayRunner`] — single-struct fold of the
/// pre-cycle-3 38-kwarg constructor surface.
///
/// 9 bool fields mirror the PyO3 kwarg surface (each is a user-tunable flag, not
/// internal state); the `clippy::struct_excessive_bools` allow is a permanent KEEP
/// per the cycle-3 P79 anchor map rationale.
#[allow(clippy::struct_excessive_bools)] // PyO3 kwarg surface — 9 user-tunable flags; permanent KEEP per cycle 3 P79.
// `from_py_object` opt-in (PyO3 0.28+ requirement for #[pyclass] types that derive Clone
// and need FromPyObject — SelfPlayRunner::new(config) extracts this struct from Python).
#[pyclass(name = "SelfPlayRunnerConfig", from_py_object)]
#[derive(Clone)]
pub struct SelfPlayRunnerConfig {
    pub n_workers: usize,
    pub max_moves_per_game: usize,
    pub n_simulations: usize,
    pub leaf_batch_size: usize,
    pub c_puct: f32,
    pub fpu_reduction: f32,
    pub feature_len: Option<usize>,
    pub policy_len: Option<usize>,
    pub fast_prob: f32,
    pub fast_sims: usize,
    pub standard_sims: usize,
    pub temp_threshold_compound_moves: usize,
    pub draw_reward: f32,
    /// §178: terminal-via-ply-cap outcome (winner=None AND ply≥max_moves_per_game).
    /// Split from `draw_reward` so organic draws and ply-cap truncations can pay
    /// distinct value-head targets. Default `-0.1` matches `draw_reward` default
    /// for back-compat — pre-§178 callers see identical outcomes.
    pub ply_cap_value: f32,
    pub quiescence_enabled: bool,
    pub quiescence_blend_2: f32,
    pub temp_min: f32,
    pub zoi_enabled: bool,
    pub zoi_lookback: usize,
    pub zoi_margin: i32,
    pub completed_q_values: bool,
    pub c_visit: f32,
    pub c_scale: f32,
    pub gumbel_mcts: bool,
    pub gumbel_m: usize,
    pub gumbel_explore_moves: usize,
    pub dirichlet_alpha: f32,
    pub dirichlet_epsilon: f32,
    pub dirichlet_enabled: bool,
    pub results_queue_cap: usize,
    pub full_search_prob: f32,
    pub n_sims_quick: usize,
    pub n_sims_full: usize,
    pub random_opening_plies: u32,
    pub selfplay_rotation_enabled: bool,
    pub legal_move_radius_jitter: bool,
    /// Registry-form encoding name (e.g. "v6", "v6w25"). Routed to
    /// `crate::encoding::lookup` at `SelfPlayRunner::new` time to derive
    /// `&'static RegistrySpec`. Cycle 3 Wave 8 Batch C FF.10 collapsed the
    /// pre-refactor `encoding_spec: Option<PyRegistrySpec>` round-trip — the
    /// Rust side now owns the lookup, breaking the parallel Python wire-spec
    /// resolution shim that bridged into PyO3.
    pub encoding_name: Option<String>,
    pub radius_override: Option<i32>,
    pub inference_pool_size: Option<usize>,
    /// O1 (SootyOwl-validated) forced-win → one-hot POLICY target knobs.
    /// Added as `#[pyo3(get, set)]` attributes (set post-construction from
    /// Python) rather than `new()` positional kwargs so the 38-arg positional
    /// Rust ctor surface — pinned by INV19 + ~10 positional test call sites —
    /// is untouched. Defaulted OFF in `new()` (single-variable / back-compat
    /// discipline, mirroring `completed_q_values=false`); the operative values
    /// come from `configs/selfplay.yaml` via `pool.py`.
    #[pyo3(get, set)]
    pub forced_win_policy_enabled: bool,
    /// 1 = depth-1 only (immediate win); 2 = depth-1 + depth-2 (within-turn setup).
    #[pyo3(get, set)]
    pub forced_win_policy_depth: u8,
    /// One-hot peak weight: 1.0 = pure one-hot on the winning move; <1.0 = convex
    /// blend with the improved-policy target.
    #[pyo3(get, set)]
    pub forced_win_policy_weight: f32,
    /// D-QFIX-LAND A1: interior (non-root) MCTS selection rule, as a registry
    /// string ("puct" | "gumbel_improved"). Added as a `#[pyo3(get, set)]`
    /// attribute (set post-construction from Python) rather than a `new()`
    /// positional kwarg so the INV19-pinned 38-arg positional ctor surface is
    /// untouched. Parsed to `InteriorSelector` at `SelfPlayRunner::new`
    /// (panics on an unknown variant). Default `"puct"` = HEAD behaviour; the
    /// operative value is hard-read from `configs/selfplay.yaml` via `pool.py`.
    #[pyo3(get, set)]
    pub interior_selector: String,
}

/// §B1 (CONFIG-4, 2026-06-02) — semantic defaults SoT for Rust struct-literal
/// callers: `SelfPlayRunnerConfig { foo, ..Default::default() }` instead of the
/// 39-arg positional `::new(...)` (the positional-swap tax). The values MUST
/// match the `#[pyo3(signature = (...))]` defaults on `new()` below — INV19
/// Test 1 + `test_default_matches_pyo3_signature_defaults` pin the two mirrors
/// against drift. (A *derived* Default would give type-zeros, NOT these
/// semantic defaults, silently changing every caller — hence the manual impl.)
impl Default for SelfPlayRunnerConfig {
    fn default() -> Self {
        Self {
            n_workers: 4,
            max_moves_per_game: 128,
            n_simulations: 50,
            leaf_batch_size: 8,
            c_puct: 1.5,
            fpu_reduction: 0.25,
            feature_len: None,
            policy_len: None,
            fast_prob: 0.0,
            fast_sims: 50,
            standard_sims: 0,
            // D-TEMPDECAY C1 (2026-06-12): cosine-OFF default (was 15). A variant
            // omitting playout_cap must NOT re-arm the §156/L9 draw-collapse cosine.
            temp_threshold_compound_moves: 0,
            draw_reward: -0.1,
            ply_cap_value: -0.1,
            quiescence_enabled: true,
            quiescence_blend_2: 0.3,
            // D-TEMPDECAY C1: anti-colony constant floor (was 0.05). With
            // threshold=0 above, the schedule is a constant tau=0.5.
            temp_min: 0.5,
            zoi_enabled: false,
            zoi_lookback: 16,
            zoi_margin: 5,
            completed_q_values: false,
            c_visit: 50.0,
            c_scale: 1.0,
            gumbel_mcts: false,
            gumbel_m: 16,
            gumbel_explore_moves: 10,
            dirichlet_alpha: 0.3,
            dirichlet_epsilon: 0.25,
            dirichlet_enabled: true,
            results_queue_cap: 10_000,
            full_search_prob: 0.0,
            n_sims_quick: 0,
            n_sims_full: 0,
            random_opening_plies: 0,
            selfplay_rotation_enabled: false,
            legal_move_radius_jitter: false,
            encoding_name: None,
            radius_override: None,
            inference_pool_size: None,
            // O1 forced-win one-hot POLICY target — OFF by default (CONFIG-3
            // dissolve of the get/set surface is deferred until after the O1
            // smoke validates the current attr path).
            forced_win_policy_enabled: false,
            forced_win_policy_depth: 2,
            forced_win_policy_weight: 1.0,
            // D-QFIX-LAND A1: default "puct" = HEAD interior selection
            // (byte-identical). Operative value hard-read from yaml via pool.py.
            interior_selector: "puct".to_string(),
        }
    }
}

#[pymethods]
impl SelfPlayRunnerConfig {
    /// Construct a `SelfPlayRunnerConfig` from kwargs. Defaults match the prior
    /// `SelfPlayRunner::new` `#[pyo3(signature = ...)]` byte-for-byte; INV19 pins.
    #[allow(clippy::too_many_arguments, clippy::fn_params_excessive_bools)] // PyO3 kwarg builder — 38 user-tunable params, 9 of them bools; permanent KEEP per cycle 3 P79.
    #[new]
    #[pyo3(signature = (
        n_workers = 4,
        max_moves_per_game = 128,
        n_simulations = 50,
        leaf_batch_size = 8,
        c_puct = 1.5,
        fpu_reduction = 0.25,
        feature_len = None,
        policy_len = None,
        fast_prob = 0.0,
        fast_sims = 50,
        standard_sims = 0,
        temp_threshold_compound_moves = 0,
        draw_reward = -0.1,
        ply_cap_value = -0.1,
        quiescence_enabled = true,
        quiescence_blend_2 = 0.3,
        temp_min = 0.5,
        zoi_enabled = false,
        zoi_lookback = 16,
        zoi_margin = 5,
        completed_q_values = false,
        c_visit = 50.0,
        c_scale = 1.0,
        gumbel_mcts = false,
        gumbel_m = 16,
        gumbel_explore_moves = 10,
        dirichlet_alpha = 0.3,
        dirichlet_epsilon = 0.25,
        dirichlet_enabled = true,
        results_queue_cap = 10_000,
        full_search_prob = 0.0,
        n_sims_quick = 0,
        n_sims_full = 0,
        random_opening_plies = 0,
        selfplay_rotation_enabled = false,
        legal_move_radius_jitter = false,
        encoding_name = None,
        radius_override = None,
        inference_pool_size = None
    ))]
    pub fn new(
        n_workers: usize,
        max_moves_per_game: usize,
        n_simulations: usize,
        leaf_batch_size: usize,
        c_puct: f32,
        fpu_reduction: f32,
        feature_len: Option<usize>,
        policy_len: Option<usize>,
        fast_prob: f32,
        fast_sims: usize,
        standard_sims: usize,
        temp_threshold_compound_moves: usize,
        draw_reward: f32,
        ply_cap_value: f32,
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
        random_opening_plies: u32,
        selfplay_rotation_enabled: bool,
        legal_move_radius_jitter: bool,
        encoding_name: Option<String>,
        radius_override: Option<i32>,
        inference_pool_size: Option<usize>,
    ) -> Self {
        Self {
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
            // O1 forced-win knobs come from Default (false/2/1.0); Python still
            // sets the operative values as #[pyo3(get,set)] attributes from
            // configs/selfplay.yaml (CONFIG-3 ctor-dissolve deferred past the
            // O1 smoke). §B1: ..Default::default() DRYs the O1 defaults to one
            // place (the Default impl above).
            ..Default::default()
        }
    }
}
