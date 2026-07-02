//! INV — D-WS3 L1 native solver-in-loop SOFT visit-injection wiring pin.
//!
//! The L1 lever SOFT-injects the native `engine::tactics` solver's proven forced-
//! win move into the self-play POLICY training target (a deep-search, soft, off-
//! window-aware generalisation of the O1 forced-win one-hot). The full signal only
//! fires inside self-play (no per-move behavioural unit test exercises the wired
//! path end-to-end — same constraint as O1), so a refactor / static-analysis
//! "clean up" could silently strip any link in the chain — the solver call, the
//! soft blend, the off-window (`window_half: None`) surfacing, the legal_set
//! coverage gate, or the full-search forcing — with NO behavioural test catching
//! it. These source-presence pins (idiom: INV25 Cell 3 / the O1 wiring INV) assert
//! every load-bearing marker persists; robust to rustfmt re-flow (substring match).
//!
//! Behavioural correctness of the pieces is pinned elsewhere: the solver itself in
//! `engine/src/tactics/*` (`tactics::` tests), and the solver→soft-injection
//! composition in `engine/src/game_runner/records.rs` (`dws3_tests` — solver
//! surfaces the win, the soft blend is NOT a one-hot, reaches a ~0-prior move, and
//! the LS path injects end-to-end).

const INNER: &str = include_str!("../src/game_runner/worker_loop/inner.rs");
const PARAMS: &str = include_str!("../src/game_runner/worker_loop/params.rs");
const CONFIG: &str = include_str!("../src/game_runner/config.rs");

/// The per-move target-extraction override must (1) gate on the config flag,
/// (2) call the native solver, (3) SOFT-inject the proven win, (4) surface off-
/// window wins (`window_half: None`) for the multi-window coverage gate, and
/// (5) force the injected row full-search (PCR `full_search_mask` discipline).
#[test]
fn inv_dws3_solver_override_wired_in_inner() {
    assert!(
        INNER.contains("ctx.solver_enabled"),
        "solver-in-loop override must be gated on the config-driven enabled flag"
    );
    assert!(
        INNER.contains("TacticalSolver::new") && INNER.contains(".prove("),
        "native solver call removed from the training-target extraction path"
    );
    assert!(
        INNER.contains("ctx.solver_visit_weight"),
        "SOFT visit-injection weight removed (one-hot is collaterally destructive — \
         the soft blend is the load-bearing knob)"
    );
    assert!(
        INNER.contains("window_half: if legal_set"),
        "off-window surfacing removed — the legal_set (multi-window) path must run the solver \
         with window_half=None so off-window saving moves are surfaced (D-DECODE action-space \
         fix); the Dense path keeps the single-window guard (no off-window slot)"
    );
    assert!(
        INNER.contains("solver_fired") && INNER.contains("|| solver_fired"),
        "the solver-injected row must be forced full-search (record_full_search) so PCR's \
         full_search_mask cannot drop the injected policy target"
    );
}

/// The knob must thread end-to-end: config builder → WorkerParams sub-bundle →
/// the inner destructure. (config.rs get/set attrs → SolverInLoop → inner.)
#[test]
fn inv_dws3_solver_knobs_thread_end_to_end() {
    for field in [
        "solver_enabled",
        "solver_depth",
        "solver_node_budget",
        "solver_neighbor_dist",
        "solver_visit_weight",
    ] {
        assert!(
            CONFIG.contains(field),
            "D-WS3 solver config knob `{field}` removed from SelfPlayRunnerConfig"
        );
    }
    assert!(
        PARAMS.contains("struct SolverInLoop"),
        "the SolverInLoop WorkerParams sub-bundle was removed (knobs no longer reach workers)"
    );
    assert!(
        INNER.contains("solver_in_loop: SolverInLoop") || INNER.contains("SolverInLoop {"),
        "inner.rs no longer destructures the SolverInLoop bundle from WorkerParams"
    );
}

/// D-WS3V3 — the in-run fire-rate counters, start-position seeding, and the
/// relative-ply Gumbel-explore gate must stay wired end-to-end. Same rationale as
/// the L1 pins above: the full path fires only inside self-play, so a static
/// "clean up" could strip a counter increment / the seed replay / the relative
/// gate with no behavioural test catching it.
#[test]
fn inv_dws3v3_seeding_and_counters_wired() {
    // Counter increments live in the solver / seeding branches of inner.rs.
    for marker in [
        "solver_counters.moves_eligible",
        "solver_counters.win_proven",
        "solver_counters.injected",
        "solver_counters.injected_offwindow",
        "solver_counters.budget_exhausted",
        "solver_counters.seeded_games_started",
    ] {
        assert!(INNER.contains(marker), "D-WS3V3 counter increment `{marker}` removed from inner.rs");
    }
    // Start-position seeding: the corpus bundle + the relative-explore gate.
    assert!(
        INNER.contains("seed.corpus") && INNER.contains("seed.seed_fraction"),
        "the seed-corpus replay hook (rng drawn ONLY when corpus non-empty) was removed"
    );
    assert!(
        INNER.contains("relative_explore_gate"),
        "the relative-ply Gumbel-explore gate (D-ARGMAX dup-trap fix) was removed"
    );
    assert!(
        PARAMS.contains("struct SeedCorpus"),
        "the SeedCorpus WorkerParams sub-bundle was removed (seeding no longer reaches workers)"
    );
    for field in ["seed_fraction", "seed_corpus"] {
        assert!(
            CONFIG.contains(field),
            "D-WS3V3 seeding config knob `{field}` removed from SelfPlayRunnerConfig"
        );
    }
}
