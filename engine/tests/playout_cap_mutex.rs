/// Regression test for A-009: Rust-side playout-cap mutex (defense-in-depth).
///
/// §100: game-level (`fast_prob`) and move-level (`full_search_prob`) playout-cap
/// randomisers are mutually exclusive. Python pool init is the primary gate;
/// `SelfPlayRunner::start()` panics if both arrive >0 through a path that
/// bypasses that gate.

use engine::game_runner::SelfPlayRunner;

#[test]
#[should_panic(expected = "playout-cap mutex violated")]
fn start_panics_when_both_caps_active() {
    let runner = SelfPlayRunner::new(
        1,                  // n_workers
        0,                  // max_moves_per_game (0 → workers exit immediately)
        1,                  // n_simulations
        1,                  // leaf_batch_size
        1.5,                // c_puct
        0.25,               // fpu_reduction
        18 * 19 * 19,       // feature_len
        19 * 19 + 1,        // policy_len
        0.5,                // fast_prob   (> 0)
        1,                  // fast_sims
        1,                  // standard_sims
        15,                 // temp_threshold_compound_moves
        -0.1,               // draw_reward
        true,               // quiescence_enabled
        0.3,                // quiescence_blend_2
        0.05,               // temp_min
        false,              // zoi_enabled
        16,                 // zoi_lookback
        5,                  // zoi_margin
        false,              // completed_q_values
        50.0,               // c_visit
        1.0,                // c_scale
        false,              // gumbel_mcts
        16,                 // gumbel_m
        10,                 // gumbel_explore_moves
        0.3,                // dirichlet_alpha
        0.25,               // dirichlet_epsilon
        true,               // dirichlet_enabled
        10_000,             // results_queue_cap
        0.25,               // full_search_prob (> 0 — mutex violation)
        50,                 // n_sims_quick
        100,                // n_sims_full
        0_u32,              // random_opening_plies
    )
    .expect("constructor succeeds; mutex is enforced at start()");

    runner.start();
}
