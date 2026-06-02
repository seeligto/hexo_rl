//! INV — O1 forced-win → one-hot POLICY target wiring pin (2026-06-02).
//!
//! O1 hardens the *training* policy target to a (near-)one-hot on a proven
//! within-turn forced win that the soft visit distribution under-weights. It is
//! the policy-side analogue of the existing quiescence VALUE override. The full
//! signal only fires inside self-play (no per-move behavioral unit test can
//! exercise the wired path end-to-end), so a refactor or static-analysis "clean
//! up" could silently strip any link in the chain — detection, blend, or the
//! full-search forcing — with NO behavioral test catching it.
//!
//! These source-presence pins (idiom: INV25 Cell 3 / `test_nnue_eval_path_only`)
//! assert every load-bearing marker persists. They are intentionally robust to
//! rustfmt re-flow (substring match, not exact layout). Behavioral correctness
//! of the detector itself is pinned by the unit tests in
//! `engine/src/board/moves.rs` (`forced_win_move` depth-1/depth-2/turn-phase) and
//! `engine/src/game_runner/records.rs` (`apply_forced_win_one_hot` + the
//! aggregate-survival pin `test_one_hot_survives_aggregate_to_local`).

const INNER: &str = include_str!("../src/game_runner/worker_loop/inner.rs");
const RECORDS: &str = include_str!("../src/game_runner/records.rs");
const MOVES: &str = include_str!("../src/board/moves.rs");
const CONFIG: &str = include_str!("../src/game_runner/config.rs");

/// The per-move target-extraction override must (1) detect the forced win,
/// (2) blend the one-hot into the training target, and (3) force the hardened
/// row full-search so PCR's `full_search_mask` cannot drop the ground-truth
/// target. All three must coexist in `inner.rs::play_one_move`.
#[test]
fn inv_o1_target_override_wired_in_inner() {
    assert!(
        INNER.contains("forced_win_move"),
        "O1 forced-win detection removed from the training-target extraction path"
    );
    assert!(
        INNER.contains("apply_forced_win_one_hot"),
        "O1 one-hot blend removed from the training-target extraction path"
    );
    assert!(
        INNER.contains("record_full_search"),
        "O1 must force the hardened row full-search (else PCR's full_search_mask \
         silently drops ~half the forced-win one-hot targets from the policy loss)"
    );
    // The override must be gated by the config flag so a disabled O1 is a no-op.
    assert!(
        INNER.contains("ctx.forced_win_enabled"),
        "O1 override must be gated on the config-driven enabled flag"
    );
}

/// The detector + blend primitives must exist where the override expects them.
#[test]
fn inv_o1_primitives_present() {
    assert!(
        MOVES.contains("fn forced_win_move") && MOVES.contains("fn first_winning_move"),
        "O1 forced-win detector primitives removed from board/moves.rs"
    );
    assert!(
        RECORDS.contains("fn apply_forced_win_one_hot"),
        "O1 one-hot blend helper removed from records.rs"
    );
}

/// The config knobs must remain on the builder so YAML → pool.py → runner can
/// drive O1. Source-of-truth discipline: no literal weights/depths in Rust.
#[test]
fn inv_o1_config_knobs_present() {
    for field in [
        "forced_win_policy_enabled",
        "forced_win_policy_depth",
        "forced_win_policy_weight",
    ] {
        assert!(
            CONFIG.contains(field),
            "O1 config knob `{field}` removed from SelfPlayRunnerConfig"
        );
    }
}
