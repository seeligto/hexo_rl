//! §173 A5a — v6w25 runner construction smoke test.
//!
//! Verifies that `SelfPlayRunner` built with the v6w25 registry spec:
//!   1. Constructs without panic.
//!   2. Exposes the correct encoding geometry (n_cells=625, feature_len=5000).
//!   3. The spec-keyed SymTables inside start_impl would be 25×25 (derived via
//!      SymTables::with_shape(spec.trunk_size, spec.n_planes)).
//!
//! Full end-to-end cold smoke (model server, 10-position selfplay) is covered
//! by §173 A8; this integration test exercises the construction path only
//! (no inference server dependency), ensuring the A5a spec-wiring reaches
//! SelfPlayRunner without a panic or shape mismatch.

use engine::encoding::registry::lookup_or_panic;
use engine::game_runner::SelfPlayRunner;
use engine::replay_buffer::sym_tables::SymTables;

/// v6w25 runner constructs with correct geometry and does not panic.
///
/// Validates:
///   - `feature_len()` == v6w25 state_stride = 8 × 625 = 5000 (H2-α spec-derived)
///   - `policy_len()` == v6w25 policy_stride = 626 (has_pass_slot=true, board_size=25)
///   - Runner is not running (start() not called — no inference server in tests).
#[test]
fn test_v6w25_runner_constructs_correct_geometry() {
    let spec = lookup_or_panic("v6w25");

    // v6w25: trunk_size=25, n_planes=8, has_pass_slot=true
    let expected_feature_len = spec.state_stride(); // 8 × 625 = 5000
    let expected_policy_len  = spec.policy_stride(); // 626

    let runner = SelfPlayRunner::new(
        1,                        // n_workers
        0,                        // max_moves_per_game (no game loop)
        1,                        // n_simulations
        1,                        // leaf_batch_size
        1.5,                      // c_puct
        0.25,                     // fpu_reduction
        Some(expected_feature_len), // feature_len (spec-derived)
        Some(expected_policy_len),  // policy_len (spec-derived)
        0.0,                      // fast_prob
        1,                        // fast_sims
        1,                        // standard_sims
        15,                       // temp_threshold
        -0.1,                     // draw_reward
        false,                    // quiescence_enabled
        0.0,                      // quiescence_blend_2
        0.05,                     // temp_min
        false,                    // zoi_enabled
        16,                       // zoi_lookback
        5,                        // zoi_margin
        false,                    // completed_q
        50.0,                     // c_visit
        1.0,                      // c_scale
        false,                    // gumbel_mcts
        16,                       // gumbel_m
        10,                       // gumbel_explore
        0.3,                      // dirichlet_alpha
        0.25,                     // dirichlet_eps
        false,                    // dirichlet_enabled
        10_000,                   // results_queue_cap
        0.0_f32,                  // full_search_prob
        0_usize,                  // n_sims_quick
        0_usize,                  // n_sims_full
        0_u32,                    // random_opening_plies
        false,                    // selfplay_rotation_enabled
        false,                    // legal_move_radius_jitter
        None,                     // encoding (legacy EncodingSpec — None for registry-path test)
        Some(engine::PyRegistrySpec::from_static(spec)), // encoding_spec (§173 A5a)
        None,                     // radius_override (§174)
    )
    .expect("v6w25 SelfPlayRunner must construct without panic");

    assert_eq!(
        runner.feature_len(),
        expected_feature_len,
        "feature_len must match v6w25 state_stride (8×625=5000)"
    );
    assert_eq!(
        runner.policy_len(),
        expected_policy_len,
        "policy_len must match v6w25 policy_stride (626)"
    );
    assert!(!runner.is_running(), "runner must not auto-start");
}

/// SymTables built with v6w25 geometry (trunk_size=25, n_planes=8) must
/// have 25×25=625 scatter entries for the identity symmetry.
///
/// This is the shape that start_impl would pass to SymTables::with_shape
/// for a v6w25 runner (H1-α fix). Checked here without constructing a live
/// runner to avoid the inference-server dependency.
#[test]
fn test_sym_tables_v6w25_shape() {
    let spec = lookup_or_panic("v6w25");
    let tables = SymTables::with_shape(spec.trunk_size, spec.n_planes);

    assert_eq!(tables.board_size, 25);
    assert_eq!(tables.n_cells,    625);
    assert_eq!(tables.n_planes,   8);

    // Identity scatter (sym_idx=0) must cover all 625 cells.
    assert_eq!(
        tables.scatter[0].len(), 625,
        "v6w25 identity scatter must produce 625 pairs"
    );
    // Every cell must map to itself under identity.
    for &(src, dst) in &tables.scatter[0] {
        assert_eq!(src, dst, "identity sym: cell {src} must map to itself");
    }
}

/// v6 SymTables from SymTables::new() must still be 19×19 (v6 byte-exact
/// regression guard — A5a must not change the v6 default path).
#[test]
fn test_v6_sym_tables_unchanged() {
    let tables = SymTables::new();
    assert_eq!(tables.board_size, 19);
    assert_eq!(tables.n_cells,    361);
    assert_eq!(tables.scatter[0].len(), 361);
}
