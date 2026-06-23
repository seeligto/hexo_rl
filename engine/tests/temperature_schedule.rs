//! Regression guard for the quarter-cosine temperature schedule used in the
//! self-play worker loop (F-006, reports/master_review_2026-04-18/F_tests_benches.md).
//!
//! Pins the quarter-cosine FORMULA contract: per-compound-move
//!   `max(temp_min, cos(π/2 · compound_move / temp_threshold))`, clamped at
//!   `temp_min` for `compound_move ≥ temp_threshold`.
//! The `(TEMP_THRESHOLD=15, TEMP_MIN=0.05)` constants below are illustrative
//! FORMULA parameters (the legacy §143 config), NOT the production default —
//! since D-TEMPDECAY C1 the shipped default is `0 / 0.5` (schedule OFF), pinned
//! by `default_config_schedule_is_off_constant_floor` + `inv19_*`.
//!
//! `compute_move_temperature` was extracted from the inline closure in
//! `worker_loop.rs:310-320` precisely so this integration test can pin the
//! formula without a full self-play smoke run.
//!
//! §70 C.1 flags docs-vs-code drift (half-cosine-per-ply in §36 vs
//! quarter-cosine-per-compound-move in worker_loop.rs). This test anchors the
//! *current* behaviour; if the formula is intentionally changed, this file
//! must be updated too, making the change visible to reviewers.

use engine::game_runner::{compute_move_temperature, SelfPlayRunnerConfig};

const TEMP_MIN: f32 = 0.05;
const TEMP_THRESHOLD: usize = 15; // compound moves, not plies

/// C1 (D-TEMPDECAY 2026-06-12) — the *default* config must leave the within-game
/// schedule OFF: a variant that omits `playout_cap` must inherit cosine-OFF
/// (constant `temp_min`), NOT silently re-arm the §156/L9 draw-collapse cosine.
/// Pins the default to `temp_threshold_compound_moves = 0` + `temp_min = 0.5`
/// (the shipped production posture; every live YAML already sets these).
#[test]
fn default_config_schedule_is_off_constant_floor() {
    let cfg = SelfPlayRunnerConfig::default();
    assert_eq!(
        cfg.temp_threshold_compound_moves, 0,
        "default threshold must be 0 (schedule OFF), got {}",
        cfg.temp_threshold_compound_moves
    );
    assert!(
        (cfg.temp_min - 0.5).abs() < 1e-9,
        "default temp_min must be 0.5 (anti-colony constant floor), got {}",
        cfg.temp_min
    );
    // Behavioral: with the default (threshold=0) the schedule is a constant
    // == temp_min at every compound move (incl. cm=0), i.e. bit-identical to a
    // fixed tau=0.5 — no schedule influence.
    for cm in [0_usize, 1, 5, 12, 15, 40, 150] {
        let t = compute_move_temperature(cm, cfg.temp_threshold_compound_moves, cfg.temp_min);
        assert!(
            (t - cfg.temp_min).abs() < 1e-6,
            "default schedule must be constant temp_min; got {t} at cm={cm}"
        );
    }
}

/// At the start of a game (compound_move=0, progress=0), cosine of 0 is 1.0.
#[test]
fn temperature_at_move_zero_is_one() {
    let t = compute_move_temperature(0, TEMP_THRESHOLD, TEMP_MIN);
    assert!(
        (t - 1.0).abs() < 1e-6,
        "temperature at compound_move=0 must be 1.0, got {t}"
    );
}

/// At the threshold boundary the floor kicks in: cos(π/2·1.0) = 0.0,
/// but max(0.05, 0.0) = 0.05 = TEMP_MIN.
#[test]
fn temperature_at_threshold_equals_floor() {
    let t = compute_move_temperature(TEMP_THRESHOLD, TEMP_THRESHOLD, TEMP_MIN);
    assert!(
        (t - TEMP_MIN).abs() < 1e-6,
        "temperature at compound_move=threshold must equal TEMP_MIN={TEMP_MIN}, got {t}"
    );
}

/// Beyond the threshold, temperature stays locked at TEMP_MIN.
#[test]
fn temperature_past_threshold_stays_at_floor() {
    for cm in [TEMP_THRESHOLD + 1, TEMP_THRESHOLD + 5, TEMP_THRESHOLD + 100] {
        let t = compute_move_temperature(cm, TEMP_THRESHOLD, TEMP_MIN);
        assert!(
            (t - TEMP_MIN).abs() < 1e-6,
            "temperature past threshold must stay at TEMP_MIN={TEMP_MIN}, got {t} at cm={cm}"
        );
    }
}

/// Temperature is monotonically non-increasing over [0, threshold].
/// cos is strictly decreasing on [0, π/2], so strict decrease is expected
/// unless the floor kicks in early (only if temp_min > 0).
#[test]
fn temperature_is_monotonically_non_increasing() {
    let mut prev = compute_move_temperature(0, TEMP_THRESHOLD, TEMP_MIN);
    for cm in 1..=TEMP_THRESHOLD {
        let curr = compute_move_temperature(cm, TEMP_THRESHOLD, TEMP_MIN);
        assert!(
            curr <= prev + 1e-6,
            "temperature must be non-increasing: t({}) = {curr} > t({}) = {prev}",
            cm, cm - 1,
        );
        prev = curr;
    }
}

/// Spot-check three intermediate values against the analytical formula
/// cos(π/2 · compound_move / threshold) to within 1e-6.
///
/// This pins the formula against future refactors that might swap cos for
/// sin, forget the FRAC_PI_2 factor, or change the denominator.
#[test]
fn temperature_matches_quarter_cosine_formula() {
    for cm in [3_usize, 7, 12] {
        let expected = f32::max(
            TEMP_MIN,
            (std::f32::consts::FRAC_PI_2 * cm as f32 / TEMP_THRESHOLD as f32).cos(),
        );
        let got = compute_move_temperature(cm, TEMP_THRESHOLD, TEMP_MIN);
        assert!(
            (got - expected).abs() < 1e-6,
            "cm={cm}: expected cos formula {expected}, got {got}"
        );
    }
}

/// One step before the threshold: temperature should be strictly above TEMP_MIN
/// (the cosine hasn't quite reached zero).
#[test]
fn temperature_one_before_threshold_is_above_floor() {
    let t = compute_move_temperature(TEMP_THRESHOLD - 1, TEMP_THRESHOLD, TEMP_MIN);
    assert!(
        t > TEMP_MIN,
        "temperature at compound_move=threshold-1 should exceed floor {TEMP_MIN}, got {t}"
    );
}
