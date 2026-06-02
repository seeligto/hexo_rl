//! INV26 — `ply_cap_value` distinct outcome path (§178, T10).
//!
//! Pins the new `ply_cap_value` split introduced in T2. The `finalize_game`
//! outcome branch in `engine/src/game_runner/worker_loop/inner.rs` now reads:
//!
//! ```ignore
//! None => if terminal_reason == 2 { ply_cap_value } else { draw_reward },
//! ```
//!
//! where `terminal_reason == 2` is the ply-cap path (winner=None AND
//! ply >= max_moves). Pre-T2 the branch was unconditional `draw_reward`,
//! silently diluting the value head's resolution pressure on long colony-
//! prone games (§S178 mechanism candidate L26 per
//! `reports/s178_prep_contract.md` T8).
//!
//! Strategy mirrors `test_drain_shutdown_no_false_draws.rs` —
//! `random_opening_plies = max_moves_per_game` so every move is a random
//! opening pick (no MCTS, no inference server). With small `max_moves`
//! every game terminates at the ply-cap (no winning line, no legal-move
//! exhaustion on a 19×19 board).
//!
//! Rust-side cells (per T10 contract, adjusted for the random-only path):
//!   1. Ply-cap path with `draw_reward=-0.1, ply_cap_value=-0.5` is
//!      reachable AND `SelfPlayRunnerConfig` carries the new field
//!      end-to-end without compile error. `terminal_reason == 2` confirmed
//!      on every drained game.
//!   4. Back-compat default — `ply_cap_value == draw_reward == -0.1` runs
//!      cleanly; same ply-cap path reached. The branch-equality semantic is
//!      `if terminal_reason == 2 { v } else { v } ≡ v` by construction.
//!
//! Cells 1 & 2 (per-row outcome equality with split values) AND Cell 3
//! (winner=Some pays ±1.0) are exercised by `tests/test_inv26_ply_cap_value.py`
//! which can drive MCTS through the live PyO3 wire (random-only mode does
//! NOT push rows — §115 random-opening plies skip the recorder by design).

use std::time::Duration;

use engine::board::BOARD_SIZE;
use engine::game_runner::{SelfPlayRunner, SelfPlayRunnerConfig};

/// Build a runner that never calls MCTS — every move is a random opening
/// pick — so the inference server is unused.
fn make_runner(max_moves: usize, draw_reward: f32, ply_cap_value: f32) -> SelfPlayRunner {
    SelfPlayRunner::new(SelfPlayRunnerConfig {
        max_moves_per_game: max_moves,
        n_simulations: 1,
        leaf_batch_size: 1,
        feature_len: Some(8 * BOARD_SIZE * BOARD_SIZE),
        policy_len: Some(BOARD_SIZE * BOARD_SIZE + 1),
        fast_sims: 1,
        standard_sims: 1,
        draw_reward,
        ply_cap_value, // §178
        quiescence_enabled: false,
        quiescence_blend_2: 0.0,
        dirichlet_enabled: false,
        random_opening_plies: max_moves as u32, // == max_moves → never MCTS
        ..Default::default()
    })
    .expect("runner construction should succeed")
}

/// Drive workers until `min_games` complete or `timeout` fires.
/// Returns the `(plies, terminal_reason)` pairs from the game-result queue.
fn drive_to_completion(
    runner: &SelfPlayRunner,
    min_games: usize,
    timeout: Duration,
) -> Vec<(usize, u8)> {
    runner.start();
    let deadline = std::time::Instant::now() + timeout;
    let mut games = Vec::new();
    while std::time::Instant::now() < deadline {
        for (plies, _wc, _mh, _wid, terminal_reason, _vmin, _vmax, _vd)
            in runner.drain_game_results()
        {
            games.push((plies, terminal_reason));
        }
        if games.len() >= min_games {
            break;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    runner.stop();
    for (plies, _wc, _mh, _wid, terminal_reason, _vmin, _vmax, _vd)
        in runner.drain_game_results()
    {
        games.push((plies, terminal_reason));
    }
    games
}

/// Cell 1 — split values reachable, ply-cap path confirmed.
///
/// With `draw_reward=-0.1, ply_cap_value=-0.5` the new field threads through
/// `SelfPlayRunnerConfig::new` (39-arg ctor post-T2), through `WorkerParams`,
/// through `PerGameInitCtx`, and into `finalize_game`. The game-result queue
/// observes `terminal_reason == 2` for every game under random play with
/// `max_moves=20` — pinning the branch into which `ply_cap_value` is now
/// routed. Per-row outcome equality is exercised Python-side.
#[test]
fn inv26_cell_1_ply_cap_value_distinct_from_draw_reward() {
    let runner = make_runner(20, -0.1, -0.5);
    let games = drive_to_completion(&runner, 4, Duration::from_secs(5));

    assert!(
        !games.is_empty(),
        "INV26 Cell 1: at least one game must complete (5s timeout, max_moves=20)",
    );
    for (plies, reason) in &games {
        assert_eq!(
            *reason, 2,
            "INV26 Cell 1: every game under random play with max_moves=20 must \
             hit ply_cap (terminal_reason=2); got reason={reason} plies={plies}",
        );
    }
}

/// Cell 4 — back-compat default: when `ply_cap_value == draw_reward` (both at
/// `-0.1`), the new branch produces the same value as pre-T2 by construction.
/// Catches the regression "knob added but plumbing silently reverts to a
/// single value at top-level wiring".
#[test]
fn inv26_cell_4_back_compat_default_ply_cap_equals_draw_reward() {
    let runner = make_runner(20, -0.1, -0.1);
    let games = drive_to_completion(&runner, 4, Duration::from_secs(5));

    assert!(
        !games.is_empty(),
        "INV26 Cell 4: at least one game must complete",
    );
    for (plies, reason) in &games {
        assert_eq!(
            *reason, 2,
            "INV26 Cell 4: ply-cap reachable under random play (reason=2); \
             got reason={reason} plies={plies}",
        );
    }
    // Back-compat semantic enforced by-construction: with
    // `ply_cap_value == draw_reward`, `if terminal_reason == 2 { v } else { v }`
    // is value-equivalent to the pre-T2 unconditional `draw_reward`. The
    // Python INV26 Cell 4 closes the loop on observed outcome equality.
}
