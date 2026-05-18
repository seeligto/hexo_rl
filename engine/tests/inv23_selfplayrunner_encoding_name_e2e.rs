//! INV23 — `SelfPlayRunnerConfig.encoding_name` end-to-end registry-lookup pin
//! (cycle 3 Wave 8 Batch C FF.10).
//!
//! Pre-Batch-C the Rust runner accepted an `encoding_spec: Option<PyRegistrySpec>`
//! kwarg, paired Python-side with a `WireFormatSpec` shim and an additional
//! `legacy_spec_for_registry_name` lookup; the registry was effectively looked
//! up twice (Python → Rust). FF.10 collapsed the round-trip — the runner now
//! takes `encoding_name: Option<String>` and resolves the registry record at
//! `SelfPlayRunner::new` time. This pin contracts the new entry surface:
//!
//!   1. `encoding_name=Some("v6w25")` resolves to v6w25 geometry (state_stride
//!      8 × 625 = 5000, policy_stride 626).
//!   2. `encoding_name=Some("v6")` resolves to v6 geometry (2888, 362).
//!   3. `encoding_name=Some("not_a_real_encoding")` returns `PyValueError`
//!      (registry-lookup-not-found contract).
//!   4. `encoding_name=None` + no explicit feature_len/policy_len returns
//!      `PyValueError` — closes the legacy "silent v6 fallback for v8 caller"
//!      class of bug (FH.10 / §172 A10 T8b lineage).
//!
//! Companion: Python pin lives at
//! `tests/test_engine_encoding_spec.py::test_selfplay_runner_legacy_no_spec_keeps_v6_default`
//! (post-migration the test asserts loud-fail rather than silent-v6).

use engine::game_runner::{SelfPlayRunner, SelfPlayRunnerConfig};

/// Build a `SelfPlayRunnerConfig` carrying every-default fields except for
/// `encoding_name` / `feature_len` / `policy_len`. Workers spawn for 0 moves
/// (max_moves_per_game = 0) so we never enter the inference loop, keeping the
/// test runtime-cheap and free of an InferenceServer dependency.
fn cfg_with_encoding(
    encoding_name: Option<&str>,
    feature_len: Option<usize>,
    policy_len: Option<usize>,
) -> SelfPlayRunnerConfig {
    SelfPlayRunnerConfig::new(
        1,                                 //  1 n_workers
        0,                                 //  2 max_moves_per_game (no game loop)
        1,                                 //  3 n_simulations
        1,                                 //  4 leaf_batch_size
        1.5,                               //  5 c_puct
        0.25,                              //  6 fpu_reduction
        feature_len,                       //  7 feature_len
        policy_len,                        //  8 policy_len
        0.0,                               //  9 fast_prob
        1,                                 // 10 fast_sims
        1,                                 // 11 standard_sims
        15,                                // 12 temp_threshold_compound_moves
        -0.1,                              // 13 draw_reward
        -0.1,                              // 14 ply_cap_value (§178; back-compat = draw_reward)
        false,                             // 15 quiescence_enabled
        0.0,                               // 15 quiescence_blend_2
        0.05,                              // 16 temp_min
        false,                             // 17 zoi_enabled
        16,                                // 18 zoi_lookback
        5,                                 // 19 zoi_margin
        false,                             // 20 completed_q_values
        50.0,                              // 21 c_visit
        1.0,                               // 22 c_scale
        false,                             // 23 gumbel_mcts
        16,                                // 24 gumbel_m
        10,                                // 25 gumbel_explore_moves
        0.3,                               // 26 dirichlet_alpha
        0.25,                              // 27 dirichlet_epsilon
        false,                             // 28 dirichlet_enabled
        10_000,                            // 29 results_queue_cap
        0.0_f32,                           // 30 full_search_prob
        0_usize,                           // 31 n_sims_quick
        0_usize,                           // 32 n_sims_full
        0_u32,                             // 33 random_opening_plies
        false,                             // 34 selfplay_rotation_enabled
        false,                             // 35 legal_move_radius_jitter
        encoding_name.map(str::to_string), // 36 encoding_name
        None,                              // 37 radius_override
        None,                              // 38 inference_pool_size
    )
}

/// Test 1 — `encoding_name=Some("v6w25")` resolves through the registry and
/// the runner's feature_len / policy_len match v6w25's state_stride /
/// policy_stride. Verifies the FF.10 single-lookup contract end-to-end.
#[test]
fn test_inv23_encoding_name_v6w25_resolves_correctly() {
    let cfg = cfg_with_encoding(Some("v6w25"), None, None);
    let runner = SelfPlayRunner::new(cfg)
        .expect("v6w25 encoding_name must resolve via registry");
    // v6w25: trunk_size=25, n_planes=8 → state_stride = 8 × 625 = 5000.
    assert_eq!(
        runner.feature_len(),
        8 * 25 * 25,
        "encoding_name='v6w25' must resolve to state_stride=5000"
    );
    // v6w25: has_pass_slot=true, board_size=25 → policy_stride = 626.
    assert_eq!(
        runner.policy_len(),
        626,
        "encoding_name='v6w25' must resolve to policy_stride=626"
    );
    assert!(!runner.is_running(), "runner must not auto-start");
}

/// Test 2 — `encoding_name=Some("v6")` resolves to v6 geometry (2888 / 362).
/// Pre-FF.10 this was the "implicit default" reached via the legacy fallback
/// arm; under FF.10 it is the explicit happy-path for v6 callers.
#[test]
fn test_inv23_encoding_name_v6_resolves_correctly() {
    let cfg = cfg_with_encoding(Some("v6"), None, None);
    let runner = SelfPlayRunner::new(cfg)
        .expect("v6 encoding_name must resolve via registry");
    assert_eq!(
        runner.feature_len(),
        8 * 19 * 19,
        "encoding_name='v6' must resolve to state_stride=2888"
    );
    assert_eq!(
        runner.policy_len(),
        19 * 19 + 1,
        "encoding_name='v6' must resolve to policy_stride=362"
    );
    assert!(!runner.is_running());
}

/// Test 3 — unknown encoding name returns `PyValueError`. The lookup-error
/// message must surface both the bad name and the set of known names to make
/// downstream diagnostics actionable.
///
/// Formatting the `PyErr` requires a live Python interpreter (`PyErr::value`
/// / `Display::fmt` both `Python::with_gil(...)`); without one the formatter
/// panics inside pyo3. Acquire the GIL via `Python::with_gil` for the
/// assertion, mirroring the `engine/tests/test_radius_override.rs` pattern.
#[test]
fn test_inv23_encoding_name_unknown_raises() {
    use pyo3::prelude::*;
    Python::initialize();
    Python::attach(|_py| {
        let cfg = cfg_with_encoding(Some("not_a_real_encoding"), None, None);
        match SelfPlayRunner::new(cfg) {
            Ok(_) => panic!("unknown encoding_name must return PyValueError"),
            Err(err) => {
                let msg = format!("{err}");
                assert!(
                    msg.contains("not_a_real_encoding"),
                    "error must name the bad encoding_name; got: {msg}"
                );
                assert!(
                    msg.contains("encoding_name") || msg.contains("registry"),
                    "error must hint at the registry source; got: {msg}"
                );
            }
        }
    });
}

/// Test 4 — `encoding_name=None` AND no explicit feature_len/policy_len must
/// return `PyValueError`. Closes the FH.10 silent-v6-fallback hazard: pre-
/// Wave-8 a v8 caller who omitted everything silently inherited 2888/362,
/// corrupting wire-format on every push. Post-Wave-8 the omission loud-fails.
#[test]
fn test_inv23_no_encoding_no_explicit_shapes_raises() {
    use pyo3::prelude::*;
    Python::initialize();
    Python::attach(|_py| {
        let cfg = cfg_with_encoding(None, None, None);
        match SelfPlayRunner::new(cfg) {
            Ok(_) => panic!("missing encoding_name + no explicit shape must return PyValueError"),
            Err(err) => {
                let msg = format!("{err}");
                assert!(
                    msg.contains("encoding_name") || msg.contains("feature_len"),
                    "error must reference the missing surface; got: {msg}"
                );
            }
        }
    });
}
