/// Regression test for A-009: Rust-side playout-cap mutex (defense-in-depth).
///
/// §100: game-level (`fast_prob`) and move-level (`full_search_prob`) playout-cap
/// randomisers are mutually exclusive. Python pool init is the primary gate;
/// `SelfPlayRunner::start()` panics if both arrive >0 through a path that
/// bypasses that gate.

use engine::game_runner::{SelfPlayRunner, SelfPlayRunnerConfig};

#[test]
#[should_panic(expected = "playout-cap mutex violated")]
fn start_panics_when_both_caps_active() {
    let runner = SelfPlayRunner::new(SelfPlayRunnerConfig {
        n_workers: 1,
        max_moves_per_game: 0, // 0 → workers exit immediately
        n_simulations: 1,
        leaf_batch_size: 1,
        feature_len: Some(8 * 19 * 19),
        policy_len: Some(19 * 19 + 1),
        fast_prob: 0.5, // > 0
        fast_sims: 1,
        standard_sims: 1,
        full_search_prob: 0.25, // > 0 — mutex violation
        n_sims_quick: 50,
        n_sims_full: 100,
        ..Default::default()
    })
    .expect("constructor succeeds; mutex is enforced at start()");

    runner.start();
}
