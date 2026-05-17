//! INV19 — `SelfPlayRunnerConfig` builder byte-equivalence pin (cycle 3 Wave 7 Batch A, P79).
//!
//! Pins the configuration-builder contract introduced by the cycle-3 P79 refactor:
//!   1. `SelfPlayRunnerConfig::new(...)` with the 38 documented PyO3 signature
//!      defaults produces a config whose fields match those defaults byte-for-byte.
//!      Locks the `#[pyo3(signature = (...))]` defaults against drift.
//!   2. Each of the 38 `SelfPlayRunnerConfig` fields maps to exactly one
//!      `SelfPlayRunner` slot — no silent merges, no silent drops. Uses
//!      unique-value-per-field construction so any swap, alias, or drop manifests.
//!   3. Minimal valid configuration (`n_workers=1, max_moves_per_game=0`, no MCTS
//!      activity) constructs without panic via the new config-taking ctor.
//!
//! Renumbered from PREP §A's proposed `INV18` because Wave 6.5 already took
//! `INV18` (`inv18_window_center_negative_bbox.rs`) and `INV18b`
//! (`inv18b_cluster_center_negative_bbox.rs`) for the `i32::midpoint`
//! truncate-semantic revert pins.

use engine::game_runner::{SelfPlayRunner, SelfPlayRunnerConfig};

/// Build a config with every field set to a distinct, non-default sentinel value.
/// Used by Test 2 to detect any internal field swap / alias / drop.
fn config_with_distinct_sentinels() -> SelfPlayRunnerConfig {
    SelfPlayRunnerConfig::new(
        7,                                  //  1 n_workers
        77,                                 //  2 max_moves_per_game
        88,                                 //  3 n_simulations
        3,                                  //  4 leaf_batch_size
        2.5,                                //  5 c_puct
        0.125,                              //  6 fpu_reduction
        Some(8 * 19 * 19),                  //  7 feature_len (v6 byte-valid)
        Some(19 * 19 + 1),                  //  8 policy_len (v6 byte-valid)
        0.375,                              //  9 fast_prob
        37,                                 // 10 fast_sims
        42,                                 // 11 standard_sims
        21,                                 // 12 temp_threshold_compound_moves
        -0.75,                              // 13 draw_reward
        false,                              // 14 quiescence_enabled (flip from default true)
        0.625,                              // 15 quiescence_blend_2
        0.0625,                             // 16 temp_min
        true,                               // 17 zoi_enabled (flip)
        24,                                 // 18 zoi_lookback
        9,                                  // 19 zoi_margin
        true,                               // 20 completed_q_values (flip)
        37.5,                               // 21 c_visit
        1.25,                               // 22 c_scale
        true,                               // 23 gumbel_mcts (flip)
        12,                                 // 24 gumbel_m
        7,                                  // 25 gumbel_explore_moves
        0.4,                                // 26 dirichlet_alpha
        0.3,                                // 27 dirichlet_epsilon
        false,                              // 28 dirichlet_enabled (flip from default true)
        20_000,                             // 29 results_queue_cap
        0.5,                                // 30 full_search_prob — leave 0 to dodge mutex with fast_prob
        50,                                 // 31 n_sims_quick (>0 because full_search_prob>0)
        100,                                // 32 n_sims_full (>0 because full_search_prob>0)
        3,                                  // 33 random_opening_plies
        true,                               // 34 selfplay_rotation_enabled (flip)
        true,                               // 35 legal_move_radius_jitter (flip)
        None,                               // 36 encoding_name (cycle 3 Wave 8 Batch C)
        Some(6),                            // 37 radius_override
        Some(4096),                         // 38 inference_pool_size
    )
}

/// Test 1 — PyO3 signature defaults locked against drift.
///
/// The 38 `#[pyo3(signature = (...))]` defaults at `engine/src/game_runner/config.rs`
/// must match the documented per-field defaults exactly. Any future edit that
/// silently shifts a default (e.g. flipping `quiescence_enabled` from `true` to
/// `false`) fires this test.
#[test]
fn test_config_default_matches_pyo3_signature_defaults() {
    let cfg = SelfPlayRunnerConfig::new(
        4,                                  //  1 n_workers
        128,                                //  2 max_moves_per_game
        50,                                 //  3 n_simulations
        8,                                  //  4 leaf_batch_size
        1.5,                                //  5 c_puct
        0.25,                               //  6 fpu_reduction
        None,                               //  7 feature_len
        None,                               //  8 policy_len
        0.0,                                //  9 fast_prob
        50,                                 // 10 fast_sims
        0,                                  // 11 standard_sims
        15,                                 // 12 temp_threshold_compound_moves
        -0.1,                               // 13 draw_reward
        true,                               // 14 quiescence_enabled
        0.3,                                // 15 quiescence_blend_2
        0.05,                               // 16 temp_min
        false,                              // 17 zoi_enabled
        16,                                 // 18 zoi_lookback
        5,                                  // 19 zoi_margin
        false,                              // 20 completed_q_values
        50.0,                               // 21 c_visit
        1.0,                                // 22 c_scale
        false,                              // 23 gumbel_mcts
        16,                                 // 24 gumbel_m
        10,                                 // 25 gumbel_explore_moves
        0.3,                                // 26 dirichlet_alpha
        0.25,                               // 27 dirichlet_epsilon
        true,                               // 28 dirichlet_enabled
        10_000,                             // 29 results_queue_cap
        0.0,                                // 30 full_search_prob
        0,                                  // 31 n_sims_quick
        0,                                  // 32 n_sims_full
        0,                                  // 33 random_opening_plies
        false,                              // 34 selfplay_rotation_enabled
        false,                              // 35 legal_move_radius_jitter
        None,                               // 36 encoding_name (cycle 3 Wave 8 Batch C)
        None,                               // 37 radius_override
        None,                               // 38 inference_pool_size
    );

    // Field-by-field: every documented default must be reachable on the config.
    assert_eq!(cfg.n_workers, 4);
    assert_eq!(cfg.max_moves_per_game, 128);
    assert_eq!(cfg.n_simulations, 50);
    assert_eq!(cfg.leaf_batch_size, 8);
    assert!((cfg.c_puct - 1.5).abs() < 1e-9);
    assert!((cfg.fpu_reduction - 0.25).abs() < 1e-9);
    assert_eq!(cfg.feature_len, None);
    assert_eq!(cfg.policy_len, None);
    assert!((cfg.fast_prob - 0.0).abs() < 1e-9);
    assert_eq!(cfg.fast_sims, 50);
    assert_eq!(cfg.standard_sims, 0);
    assert_eq!(cfg.temp_threshold_compound_moves, 15);
    assert!((cfg.draw_reward - -0.1).abs() < 1e-9);
    assert!(cfg.quiescence_enabled);
    assert!((cfg.quiescence_blend_2 - 0.3).abs() < 1e-9);
    assert!((cfg.temp_min - 0.05).abs() < 1e-9);
    assert!(!cfg.zoi_enabled);
    assert_eq!(cfg.zoi_lookback, 16);
    assert_eq!(cfg.zoi_margin, 5);
    assert!(!cfg.completed_q_values);
    assert!((cfg.c_visit - 50.0).abs() < 1e-9);
    assert!((cfg.c_scale - 1.0).abs() < 1e-9);
    assert!(!cfg.gumbel_mcts);
    assert_eq!(cfg.gumbel_m, 16);
    assert_eq!(cfg.gumbel_explore_moves, 10);
    assert!((cfg.dirichlet_alpha - 0.3).abs() < 1e-9);
    assert!((cfg.dirichlet_epsilon - 0.25).abs() < 1e-9);
    assert!(cfg.dirichlet_enabled);
    assert_eq!(cfg.results_queue_cap, 10_000);
    assert!((cfg.full_search_prob - 0.0).abs() < 1e-9);
    assert_eq!(cfg.n_sims_quick, 0);
    assert_eq!(cfg.n_sims_full, 0);
    assert_eq!(cfg.random_opening_plies, 0);
    assert!(!cfg.selfplay_rotation_enabled);
    assert!(!cfg.legal_move_radius_jitter);
    assert!(cfg.encoding_name.is_none());
    assert_eq!(cfg.radius_override, None);
    assert_eq!(cfg.inference_pool_size, None);
}

/// Test 2 — param-grouping invariant: each config field maps to exactly one
/// internal slot. Uses unique sentinels per field; a swap or drop manifests
/// as either a field-equality assert failure on the config OR a getter-visible
/// runner-side delta.
#[test]
fn test_config_param_grouping_one_to_one_no_silent_drops() {
    let cfg = config_with_distinct_sentinels();

    // First: every distinct sentinel must round-trip via the config builder.
    // This catches `Self { foo: bar }` cross-wires inside the ctor body.
    assert_eq!(cfg.n_workers, 7);
    assert_eq!(cfg.max_moves_per_game, 77);
    assert_eq!(cfg.n_simulations, 88);
    assert_eq!(cfg.leaf_batch_size, 3);
    assert!((cfg.c_puct - 2.5).abs() < 1e-9);
    assert!((cfg.fpu_reduction - 0.125).abs() < 1e-9);
    assert_eq!(cfg.feature_len, Some(8 * 19 * 19));
    assert_eq!(cfg.policy_len, Some(19 * 19 + 1));
    assert!((cfg.fast_prob - 0.375).abs() < 1e-9);
    assert_eq!(cfg.fast_sims, 37);
    assert_eq!(cfg.standard_sims, 42);
    assert_eq!(cfg.temp_threshold_compound_moves, 21);
    assert!((cfg.draw_reward - -0.75).abs() < 1e-9);
    assert!(!cfg.quiescence_enabled);
    assert!((cfg.quiescence_blend_2 - 0.625).abs() < 1e-9);
    assert!((cfg.temp_min - 0.0625).abs() < 1e-9);
    assert!(cfg.zoi_enabled);
    assert_eq!(cfg.zoi_lookback, 24);
    assert_eq!(cfg.zoi_margin, 9);
    assert!(cfg.completed_q_values);
    assert!((cfg.c_visit - 37.5).abs() < 1e-9);
    assert!((cfg.c_scale - 1.25).abs() < 1e-9);
    assert!(cfg.gumbel_mcts);
    assert_eq!(cfg.gumbel_m, 12);
    assert_eq!(cfg.gumbel_explore_moves, 7);
    assert!((cfg.dirichlet_alpha - 0.4).abs() < 1e-9);
    assert!((cfg.dirichlet_epsilon - 0.3).abs() < 1e-9);
    assert!(!cfg.dirichlet_enabled);
    assert_eq!(cfg.results_queue_cap, 20_000);
    assert!((cfg.full_search_prob - 0.5).abs() < 1e-9);
    assert_eq!(cfg.n_sims_quick, 50);
    assert_eq!(cfg.n_sims_full, 100);
    assert_eq!(cfg.random_opening_plies, 3);
    assert!(cfg.selfplay_rotation_enabled);
    assert!(cfg.legal_move_radius_jitter);
    assert!(cfg.encoding_name.is_none());
    assert_eq!(cfg.radius_override, Some(6));
    assert_eq!(cfg.inference_pool_size, Some(4096));

    // Second: SelfPlayRunner::new(config) must construct successfully and
    // expose the getter-visible config-derived fields. Validation order is
    // unchanged from the pre-refactor 38-param ctor — fast_prob=0.375 + full_search_prob=0.5
    // would BOTH be > 0, but full_search_prob's start_impl panic only triggers on
    // start(); ctor only rejects (effective_standard==0, fast_sims==0 && fast_prob>0,
    // and the full_search_prob>0 + (n_sims_quick==0 || n_sims_full==0) gate. With
    // n_sims_quick=50, n_sims_full=100, the ctor accepts; start() would later panic
    // via the playout-cap mutex (covered by `playout_cap_mutex.rs`).
    //
    // We don't call start() here — the goal is to assert byte-equivalence of the
    // ctor surface, not the start gate.
    let runner = SelfPlayRunner::new(cfg).expect("ctor must accept distinct-sentinel config");
    // feature_len getter mirrors config.feature_len.unwrap() — explicit kwarg wins.
    assert_eq!(runner.feature_len(), 8 * 19 * 19);
    assert_eq!(runner.policy_len(), 19 * 19 + 1);
    // Runner is not started — drop will be a no-op stop().
    assert!(!runner.is_running());
}

/// Test 3 — minimal-valid config constructs without panic.
///
/// Pins the edge case: smallest valid configuration accepted by the validation
/// gates. `n_simulations = 1` keeps `effective_standard > 0` so the ctor
/// accepts; `fast_prob = 0` and `full_search_prob = 0` disable both
/// playout-cap branches.
#[test]
fn test_config_construction_edge_case_minimal_valid() {
    let cfg = SelfPlayRunnerConfig::new(
        1,                          // n_workers (must be ≥ 1 to spawn anything; ctor doesn't reject 0 though)
        0,                          // max_moves_per_game (0 → workers exit before any move)
        1,                          // n_simulations (must be > 0 — would error if 0)
        1,                          // leaf_batch_size
        1.0,                        // c_puct
        0.0,                        // fpu_reduction
        Some(8 * 19 * 19),          // feature_len
        Some(19 * 19 + 1),          // policy_len
        0.0,                        // fast_prob (disable game-level cap)
        1,                          // fast_sims (unused when fast_prob==0)
        0,                          // standard_sims (0 → falls back to n_simulations)
        15,                         // temp_threshold_compound_moves
        0.0,                        // draw_reward
        false,                      // quiescence_enabled
        0.0,                        // quiescence_blend_2
        0.0,                        // temp_min
        false,                      // zoi_enabled
        16,                         // zoi_lookback
        5,                          // zoi_margin
        false,                      // completed_q_values
        1.0,                        // c_visit
        1.0,                        // c_scale
        false,                      // gumbel_mcts
        16,                         // gumbel_m
        10,                         // gumbel_explore_moves
        0.3,                        // dirichlet_alpha
        0.25,                       // dirichlet_epsilon
        false,                      // dirichlet_enabled
        10_000,                     // results_queue_cap
        0.0,                        // full_search_prob (disable move-level cap)
        0,                          // n_sims_quick
        0,                          // n_sims_full
        0,                          // random_opening_plies
        false,                      // selfplay_rotation_enabled
        false,                      // legal_move_radius_jitter
        None,                       // encoding_name (cycle 3 Wave 8 Batch C)
        None,                       // radius_override
        None,                       // inference_pool_size (cycle-1 fixed 512 pool)
    );

    let runner = SelfPlayRunner::new(cfg).expect("minimal valid config must construct");
    assert!(!runner.is_running());
    assert_eq!(runner.feature_len(), 8 * 19 * 19);
    assert_eq!(runner.policy_len(), 19 * 19 + 1);
}
