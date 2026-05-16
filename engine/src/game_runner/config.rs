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

use crate::pyo3::encoding::PyRegistrySpec;

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
    pub encoding_spec: Option<PyRegistrySpec>,
    pub radius_override: Option<i32>,
    pub inference_pool_size: Option<usize>,
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
        temp_threshold_compound_moves = 15,
        draw_reward = -0.1,
        quiescence_enabled = true,
        quiescence_blend_2 = 0.3,
        temp_min = 0.05,
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
        encoding_spec = None,
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
        encoding_spec: Option<PyRegistrySpec>,
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
            encoding_spec,
            radius_override,
            inference_pool_size,
        }
    }
}
