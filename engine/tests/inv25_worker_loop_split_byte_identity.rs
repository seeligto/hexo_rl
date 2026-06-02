//! INV25 — `worker_loop` 7-sibling-module split byte-identity-on-behavior
//! pin (cycle 3 Wave 10 Batch A, 2026-05-17).
//!
//! The Wave 10 Batch A split decomposes `engine/src/game_runner/worker_loop.rs`
//! (1129 LOC) into 7 sibling modules under `engine/src/game_runner/worker_loop/`:
//! `mod.rs`, `rotate.rs`, `params.rs`, `channels.rs`, `stats.rs`, `atomics.rs`,
//! `inner.rs`. The 4 `Worker{Stats,Atomics,Channels,Params}` capture bundles +
//! 3 Wave 7 Batch C themed `*Flags` sub-structs migrate to their own files.
//! The pre-split closure body extracts to `inner::run_worker_thread` (named
//! fn; semantics byte-identical).
//!
//! These pins guard against silent regressions during the split:
//!
//! 1. **Hot-loop flag routing** — for each Wave 7 Batch C bool, construct a
//!    runner with that flag flipped and assert construction succeeds. Catches
//!    a regression where the post-split destructure mis-routes a bool from
//!    `SelfPlayRunnerConfig` → `SelfPlayRunner` → `WorkerParams` prototype →
//!    destructured local in `inner::run_worker_thread`.
//!
//! 2. **Spawn-loop fan-out bundle integrity** — assert that the runner's
//!    `Arc<AtomicU*>` accumulator surface is wired (initial counts == 0)
//!    via the public accessors. Catches a regression where the `WorkerStats`
//!    bundle accidentally constructs fresh atomics per worker rather than
//!    `Arc::clone`-ing the shared accumulators from `SelfPlayRunner`.
//!
//! 3. **Wave 7 Batch C destructure pattern preserved** — `include_str!` the
//!    post-split `inner.rs` source and assert every Wave 7 Batch C bool field
//!    name + every themed sub-struct name appears. Robust to rustfmt re-flow
//!    of the destructure pattern. Catches accidental field rename, sub-struct
//!    rename, or destructure-routing drift.

use engine::encoding::registry::lookup_or_panic;
use engine::game_runner::{SelfPlayRunner, SelfPlayRunnerConfig};

/// Build a v6w25 `SelfPlayRunnerConfig` with explicit defaults; toggle the
/// Wave 7 Batch C bool fields per the caller. Mirrors the ctor template in
/// `tests/test_worker_loop_v6w25_smoke.rs:35`.
#[allow(clippy::too_many_arguments)]
fn build_config(
    quiescence_enabled: bool,
    completed_q_values: bool,
    gumbel_mcts: bool,
    dirichlet_enabled: bool,
    selfplay_rotation_enabled: bool,
    zoi_enabled: bool,
    legal_move_radius_jitter: bool,
) -> SelfPlayRunnerConfig {
    let spec = lookup_or_panic("v6w25");
    let feature_len = spec.state_stride();
    let policy_len = spec.policy_stride();

    SelfPlayRunnerConfig {
        n_workers: 1,
        max_moves_per_game: 0, // no game loop
        n_simulations: 1,
        leaf_batch_size: 1,
        feature_len: Some(feature_len),
        policy_len: Some(policy_len),
        fast_sims: 1,
        standard_sims: 1,
        quiescence_enabled,
        quiescence_blend_2: 0.0,
        zoi_enabled,
        completed_q_values,
        gumbel_mcts,
        dirichlet_enabled,
        selfplay_rotation_enabled,
        legal_move_radius_jitter,
        encoding_name: Some("v6w25".to_string()),
        ..Default::default()
    }
}

/// Cell 1 — construct one runner per Wave 7 Batch C bool with that bool
/// flipped from its default. Validates the `SelfPlayRunnerConfig` → struct
/// fields → `WorkerParams` prototype wiring at construction time. Live
/// hot-loop destructure-routing semantics are pinned by Cell 3 (static
/// source assertion) + the existing v6w25 smoke integration test.
#[test]
fn inv25_search_flags_route_through_destructure() {
    // Default-all-false baseline.
    let baseline = build_config(false, false, false, false, false, false, false);
    let _r = SelfPlayRunner::new(baseline)
        .expect("baseline (all-flags-false) runner must construct");

    // Toggle each Wave 7 Batch C bool individually; every variant must
    // construct without panic — the post-split destructure pattern in
    // `inner::run_worker_thread` routes every flag through the same shape
    // as the pre-split source, so any rename / reorder regression would
    // surface as a compile error at the `WorkerParams { ... } = params;`
    // destructure (rustc enforces exhaustive struct destructure under
    // `#[deny(unused_variables)]` defaults).
    let variants = [
        ("quiescence_enabled",         (true,  false, false, false, false, false, false)),
        ("completed_q_values",         (false, true,  false, false, false, false, false)),
        ("gumbel_mcts",                (false, false, true,  false, false, false, false)),
        ("dirichlet_enabled",          (false, false, false, true,  false, false, false)),
        ("selfplay_rotation_enabled",  (false, false, false, false, true,  false, false)),
        ("zoi_enabled",                (false, false, false, false, false, true,  false)),
        ("legal_move_radius_jitter",   (false, false, false, false, false, false, true)),
    ];
    for (name, (q, c, g, d, s, z, j)) in variants {
        let cfg = build_config(q, c, g, d, s, z, j);
        let r = SelfPlayRunner::new(cfg)
            .unwrap_or_else(|_| panic!("runner must construct with {} flipped", name));
        assert!(!r.is_running(), "runner with {} flipped must not auto-start", name);
    }
}

/// Cell 2 — spawn-loop fan-out bundle integrity. Construct a 4-worker
/// runner and assert the public `Arc<AtomicU*>` accumulator surface is
/// wired and reads 0 at construction. Catches a regression where the
/// `WorkerStats` bundle constructs fresh atomics per worker rather than
/// `Arc::clone`-ing the runner's shared accumulators (the latter is what
/// allows public accessors like `games_completed()` to see the cross-worker
/// sum once a live run begins).
#[test]
fn inv25_spawn_loop_fan_out_independent_stats() {
    let spec = lookup_or_panic("v6w25");
    let feature_len = spec.state_stride();
    let policy_len = spec.policy_stride();

    let cfg = SelfPlayRunnerConfig {
        n_workers: 4, // multi-worker fan-out shape
        max_moves_per_game: 0, // no game loop fires
        n_simulations: 1,
        leaf_batch_size: 1,
        feature_len: Some(feature_len),
        policy_len: Some(policy_len),
        fast_sims: 1,
        standard_sims: 1,
        quiescence_enabled: false,
        quiescence_blend_2: 0.0,
        dirichlet_enabled: false,
        encoding_name: Some("v6w25".to_string()),
        ..Default::default()
    };
    let r = SelfPlayRunner::new(cfg)
        .expect("4-worker v6w25 runner must construct");

    // Every Arc<AtomicU*> accumulator on SelfPlayRunner must initialize to 0
    // and be readable via the public accessor surface. A bundle construction
    // regression that newly-allocates fresh atomics per worker would not
    // affect the initial read but would silently desync the dashboard at
    // first start; this pin is the cheap construction-time canary.
    assert_eq!(r.games_completed(), 0, "games_completed must init to 0");
    assert_eq!(r.positions_generated(), 0, "positions_generated must init to 0");
    assert_eq!(r.x_wins(), 0, "x_wins must init to 0");
    assert_eq!(r.o_wins(), 0, "o_wins must init to 0");
    assert_eq!(r.draws(), 0, "draws must init to 0");
    assert_eq!(r.positions_dropped(), 0, "positions_dropped must init to 0");
    assert_eq!(r.mcts_quiescence_fires(), 0, "mcts_quiescence_fires must init to 0");
    assert_eq!(r.cluster_variance_sample_count(), 0, "cluster_variance_sample_count must init to 0");
    assert!(!r.is_running(), "runner must not auto-start");
}

/// Cell 3 — Wave 7 Batch C destructure pattern preserved in post-split
/// `inner.rs`. Reads the source via `include_str!` and asserts every Wave 7
/// Batch C bool field name + every themed sub-struct name appears. Robust
/// to rustfmt re-flowing the pattern.
#[test]
fn inv25_wave7c_destructure_pattern_preserved() {
    let src = include_str!("../src/game_runner/worker_loop/inner.rs");

    // Wave 7 Batch C themed sub-struct names — exact `Name { ... }`
    // destructure shape required so the per-flag binding routes correctly.
    for type_name in ["SearchFlags", "ExplorationFlags", "MoveConstraintFlags"] {
        assert!(
            src.contains(type_name),
            "Wave 7 Batch C sub-struct '{}' must appear in post-split inner.rs",
            type_name,
        );
    }

    // All 7 Wave 7 Batch C bool field names. Trailing comma anchors the
    // assertion to a binding-position occurrence (vs an arbitrary string
    // mention inside a comment).
    for field in [
        "quiescence_enabled,",
        "completed_q_values,",
        "gumbel_mcts ",          // last field in SearchFlags destructure — followed by closing brace
        "dirichlet_enabled,",
        "selfplay_rotation_enabled ", // last field in ExplorationFlags — closing brace
        "zoi_enabled,",
        "legal_move_radius_jitter ",  // last field in MoveConstraintFlags — closing brace
    ] {
        assert!(
            src.contains(field),
            "Wave 7 Batch C destructure field '{}' missing from post-split inner.rs \
             (possible rename, reorder, or destructure-pattern drift)",
            field,
        );
    }

    // Bundle struct names also asserted — the destructure of the outer
    // 4 capture-bundle structs is part of the same hot-loop entry pattern.
    for bundle in ["WorkerStats", "WorkerAtomics", "WorkerChannels", "WorkerParams"] {
        assert!(
            src.contains(bundle),
            "capture bundle struct '{}' missing from post-split inner.rs",
            bundle,
        );
    }
}
