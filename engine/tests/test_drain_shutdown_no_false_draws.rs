//! §P22 — drain-shutdown regression: a `SelfPlayRunner` stopped mid-game must
//! NOT push false-draw records for the in-progress game.
//!
//! Pre-P22 the worker loop unconditionally fell through to the terminal block
//! (`board.winner()` / record-push) after the move loop, even when the move
//! loop had exited because `stop()` flipped `running` mid-game. `winner()`
//! returns `None` for an in-progress board, routing through
//! `terminal_reason = 3 (other_draw)` and pushing N partial-game rows into the
//! results queue with `outcome = draw_reward`. At training boundary or
//! eval-pivot this corrupted ~n_workers × max_moves rows per shutdown event.
//!
//! P22 adds `if !running.load(Ordering::SeqCst) { continue; }` AFTER the move
//! loop break, BEFORE the `let winner = board.winner()` block, so an
//! in-progress game on shutdown drops silently instead of recording as a draw.
//!
//! Strategy (pure-Rust, no inference server): drive workers with
//! `random_opening_plies = max_moves_per_game` so every move is a random
//! opening pick — the worker never enters MCTS, no `submit_batch_and_wait_rust`
//! call, no inference server needed (mirrors `random_opening_plies.rs`). The
//! P22 guard sits AT the terminal-block entry, AFTER the move loop break, so
//! it fires for *any* worker that exits its move loop with `running=false` —
//! whether the move loop exited because of MCTS-path break, random-opening
//! break, or natural game end. By forcing stop() while workers are inside the
//! random-opening loop, we exercise the same terminal-block branch P22 guards.
//!
//! The test asserts:
//!   - `terminal_reason == 3` never appears in `drain_game_results()` for any
//!     game pushed AFTER stop() begins draining workers. Under random play
//!     with max_moves=50 a natural reason-3 draw is impossible (legal move
//!     set never empties pre-cap on a 19×19 hex board), so any reason-3 row
//!     would be a P22 regression.

use std::time::Duration;

use engine::board::BOARD_SIZE;
use engine::game_runner::{SelfPlayRunner, SelfPlayRunnerConfig};

/// Construct a SelfPlayRunner where every move is a random opening pick.
/// Avoids the inference batcher entirely (per random_opening_plies.rs).
fn make_runner_random_only(max_moves: usize) -> SelfPlayRunner {
    SelfPlayRunner::new(SelfPlayRunnerConfig::new(
        4,                                  // n_workers (more workers → higher mid-game stop probability)
        max_moves,
        1,                                  // n_simulations (irrelevant — no MCTS)
        1,                                  // leaf_batch_size
        1.5,                                // c_puct
        0.25,                               // fpu_reduction
        Some(8 * BOARD_SIZE * BOARD_SIZE),  // feature_len
        Some(BOARD_SIZE * BOARD_SIZE + 1),  // policy_len
        0.0,                                // fast_prob
        1,                                  // fast_sims
        1,                                  // standard_sims
        15,                                 // temp_threshold_compound_moves
        -0.1,                               // draw_reward
        false,                              // quiescence_enabled
        0.0,                                // quiescence_blend_2
        0.05,                               // temp_min
        false, 16, 5,                       // zoi_enabled, zoi_lookback, zoi_margin
        false, 50.0, 1.0,                   // completed_q, c_visit, c_scale
        false, 16, 10,                      // gumbel_mcts, gumbel_m, gumbel_explore
        0.3, 0.25, false,                   // dirichlet alpha/eps/enabled
        10_000,                             // results_queue_cap
        0.0_f32, 0_usize, 0_usize,          // playout-cap mutex (all zero)
        max_moves as u32,                   // random_opening_plies == max_moves → never MCTS
        false,                              // selfplay_rotation_enabled
        false,                              // legal_move_radius_jitter
        None,                               // encoding_name (cycle 3 Wave 8 Batch C)
        None,                               // radius_override
        None,                               // inference_pool_size
    ))
    .expect("runner construction should succeed")
}

/// §P22: a runner stopped mid-game must NOT emit terminal_reason==3 from a
/// partial-game push. Under random play this would only fire if the legal-
/// move set emptied before max_moves AND running was still true — neither
/// holds on a 19×19 hex board with max_moves=50.
#[test]
fn test_drain_shutdown_no_false_draws() {
    let runner = make_runner_random_only(50);

    // Drain everything that was recorded prior to start (sanity baseline).
    let _baseline: Vec<_> = runner.drain_game_results();

    runner.start();

    // Allow at least one move to be played before stopping. Random-opening
    // path is fast — 20 ms is enough for the worker to enter its move loop
    // and place several moves.
    std::thread::sleep(Duration::from_millis(20));

    runner.stop();

    let drained = runner.drain_game_results();
    for (plies, _wc, _mh, worker_id, terminal_reason, _vmin, _vmax, _vd) in &drained {
        assert_ne!(
            *terminal_reason, 3,
            "P22 violated: worker {worker_id} pushed game with terminal_reason=3 \
             (other_draw, partial-game shutdown injection). plies={plies} of max_moves=50."
        );
    }
}

/// Companion check: at least one game must complete naturally during the
/// pre-stop window so the test isn't trivially vacuous on a slow scheduler.
/// Natural completion produces terminal_reason==2 (ply_cap) since max_moves
/// is reached without a winning line OR a legal-move-set exhaustion.
#[test]
fn test_drain_natural_completion_terminal_reason_2() {
    let runner = make_runner_random_only(20); // small max_moves so games finish fast

    runner.start();

    // Wait up to 5 s for at least 4 natural completions (one per worker).
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    let mut completed = Vec::new();
    while std::time::Instant::now() < deadline {
        let batch = runner.drain_game_results();
        completed.extend(batch);
        if completed.len() >= 4 {
            break;
        }
        std::thread::sleep(Duration::from_millis(50));
    }

    runner.stop();

    assert!(
        !completed.is_empty(),
        "at least one natural completion expected in 5s",
    );
    for (_plies, _wc, _mh, _wid, terminal_reason, _vmin, _vmax, _vd) in &completed {
        // Under random play with max_moves=20: every game reaches ply_cap
        // (no winning line, no legal-move exhaustion). terminal_reason==2.
        assert_eq!(
            *terminal_reason, 2,
            "natural completion under random play with max_moves=20 \
             should always hit ply_cap (reason 2), got {}",
            terminal_reason,
        );
    }
}
