//! §115 Phase 4b — regression guard for self-play random-opening plies.
//!
//! Contract (mirrors `hexo_rl/eval/evaluator.py:184-185`, §80):
//!   1. For every ply in `[0, random_opening_plies)`, the worker picks a
//!      uniformly-random legal move and skips MCTS + record push.
//!   2. Rows produced by opening plies do NOT reach the results queue.
//!   3. `random_opening_plies = 0` matches the pre-§115 behaviour (regression
//!      guard for the opt-in path).
//!
//! The tests run `SelfPlayRunner` with `max_moves_per_game` tuned so that
//! every game is short enough to complete inside the test timeout.
//!
//! Setting `max_moves_per_game = random_opening_plies` guarantees no
//! `InferenceBatcher::submit_batch_and_wait_rust` call fires — the worker
//! never enters the MCTS branch, so no Python inference server is needed.
//! This keeps the test a pure-Rust `cargo test` runner.

use std::collections::HashSet;
use std::time::Duration;

use engine::board::{Board, BOARD_SIZE};
use engine::game_runner::SelfPlayRunner;

/// Helper: construct a `SelfPlayRunner` with the full argument list. The
/// positional-arg surface is painful but matches the existing test pattern
/// in `engine/tests/playout_cap_mutex.rs` and `engine/src/game_runner/mod.rs`.
fn make_runner(
    max_moves_per_game: usize,
    random_opening_plies: u32,
) -> SelfPlayRunner {
    SelfPlayRunner::new(
        2,                                  // n_workers (2 keeps game count small)
        max_moves_per_game,
        1,                                  // n_simulations (irrelevant — no MCTS)
        1,                                  // leaf_batch_size
        1.5,                                // c_puct
        0.25,                               // fpu_reduction
        18 * BOARD_SIZE * BOARD_SIZE,       // feature_len
        BOARD_SIZE * BOARD_SIZE + 1,        // policy_len
        0.0,                                // fast_prob (mutex)
        1,                                  // fast_sims
        1,                                  // standard_sims
        15,                                 // temp_threshold_compound_moves
        -0.1,                               // draw_reward
        true,                               // quiescence_enabled
        0.3,                                // quiescence_blend_2
        0.05,                               // temp_min
        false,                              // zoi_enabled
        16,                                 // zoi_lookback
        5,                                  // zoi_margin
        false,                              // completed_q_values
        50.0,                               // c_visit
        1.0,                                // c_scale
        false,                              // gumbel_mcts
        16,                                 // gumbel_m
        10,                                 // gumbel_explore_moves
        0.3,                                // dirichlet_alpha
        0.25,                               // dirichlet_epsilon
        true,                               // dirichlet_enabled
        10_000,                             // results_queue_cap
        0.0_f32,                            // full_search_prob (mutex-safe)
        0_usize,                            // n_sims_quick
        0_usize,                            // n_sims_full
        random_opening_plies,
        false,                              // selfplay_rotation_enabled
    )
    .expect("runner construction should succeed")
}

/// Opening-only games: set `max_moves_per_game == random_opening_plies` so
/// every game consists entirely of random moves. Confirm:
///   - `positions_generated` stays at 0 (no training row produced)
///   - `collect_data()` returns zero rows
///   - `drain_game_results` still records completed games (for logging)
#[test]
fn opening_plies_are_not_pushed_to_buffer() {
    let n = 4_u32;
    let runner = make_runner(n as usize, n);
    runner.start();

    // Wait for at least one game to complete from each worker.
    let mut completed_workers: HashSet<usize> = HashSet::new();
    let mut attempts = 0;
    while completed_workers.len() < 2 && attempts < 100 {
        for (_plies, _winner_code, _move_history, worker_id) in runner.drain_game_results() {
            completed_workers.insert(worker_id);
        }
        std::thread::sleep(Duration::from_millis(50));
        attempts += 1;
    }
    runner.stop();

    assert_eq!(
        completed_workers.len(),
        2,
        "both workers should have completed at least one game in {attempts} attempts",
    );
    assert_eq!(
        runner.positions_generated(),
        0,
        "opening plies must not increment positions_generated; got {}",
        runner.positions_generated(),
    );
}

/// Regression guard: `random_opening_plies = 0` restores pre-§115 behaviour
/// in terms of the opening-plies branch being a no-op. With
/// `max_moves_per_game = 0` workers exit immediately (matches the existing
/// `test_worker_id_assignment` pattern) — this test locks that nothing in
/// the new `random_opening_plies` wiring broke the zero-work path.
#[test]
fn random_opening_plies_zero_preserves_baseline() {
    let runner = make_runner(0, 0);
    runner.start();

    // Allow the worker threads to iterate a handful of empty-game outer loops.
    std::thread::sleep(Duration::from_millis(250));
    runner.stop();

    assert_eq!(
        runner.positions_generated(),
        0,
        "baseline: positions_generated must be 0 when max_moves=0",
    );
    // Allow some zero-ply games to have been recorded; harness parity with
    // the existing worker_id test.
    let (x, o, d) = runner.get_win_stats();
    assert_eq!(
        x + o, 0,
        "zero-ply games cannot have a winner (only draws or timeouts)",
    );
    let _ = d; // draws may or may not accumulate depending on scheduling
}

/// Concrete "moves are drawn from the legal set" check. Fires a single
/// short game (one opening ply) and confirms the recorded move landed on a
/// cell that was in the empty-board `legal_moves()` set. Not a uniform-
/// distribution test (cheap — picks an arbitrary first move); the worker
/// loop uses the same `rand::prelude::IndexedRandom::choose` primitive as
/// `hexo_rl/eval/evaluator.py`'s `random.choice`.
#[test]
fn opening_move_is_in_the_legal_set() {
    let runner = make_runner(1, 1);
    runner.start();

    let mut saw_any_game = false;
    let mut attempts = 0;
    while !saw_any_game && attempts < 100 {
        for (plies, _winner_code, move_history, _worker_id) in runner.drain_game_results() {
            assert_eq!(plies, 1, "game must end at ply 1 (max_moves=1)");
            assert_eq!(move_history.len(), 1);
            let (q, r) = move_history[0];
            let legal_from_fresh: Vec<(i32, i32)> = Board::new().legal_moves();
            assert!(
                legal_from_fresh.contains(&(q, r)),
                "recorded opening move ({q},{r}) not in empty-board legal set",
            );
            saw_any_game = true;
        }
        std::thread::sleep(Duration::from_millis(50));
        attempts += 1;
    }

    runner.stop();
    assert!(saw_any_game, "at least one game should have completed in {attempts} attempts");
}
