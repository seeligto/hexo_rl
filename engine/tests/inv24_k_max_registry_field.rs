//! INV24 — `RegistrySpec::k_max` field is populated, positive, and golden per
//! per encoding (cycle 3 P55 Wave 9 Batch A, 2026-05-17).
//!
//! The `k_max` field is the multi-window cluster-count upper bound per
//! position emitted by `Board::get_cluster_views()` (per the §170 / §173 α
//! design doc — `docs/designs/encoding_alpha_multiwindow_selfplay_design.md`
//! §213, §245). The field gates the auto-derived `InferenceBatcher.pool_size`
//! default in `SelfPlayRunner::new` (closes the cycle 1 hardcoded 512
//! prefill flagged by P55).
//!
//! Pins:
//!
//! 1. **Field present + positive** — every `RegistrySpec` returned by
//!    `all_specs()` has `k_max >= 1`. (Redundant pin vs the validator rule
//!    at `engine/src/encoding/spec/validate.rs::validate`; cheap belt-and-
//!    braces against silent registry edits or validator regressions.)
//!
//! 2. **Golden snapshot per encoding** — locks the per-encoding `k_max`
//!    values per Wave 9 PREP §A.2. A silent operator tune of `v6w25.k_max`
//!    or `v7mw.k_max` to a different bound updates the pool-size auto-
//!    derive heuristic; the test forces an explicit golden-snapshot edit
//!    + commit-body disclosure rather than silent drift.
//!
//! 3. **Single-window discipline** — encodings with `is_multi_window = false`
//!    have `k_max = 1`. This is NOT a validator rule (a future
//!    canvas-augmented single-window arch may legitimately set `k_max > 1`
//!    — e.g. dual-view canvas + body), but a Wave 9 discipline check
//!    pinning the current registry state. Updating the test alongside any
//!    such future addition documents the intentional break.

use engine::encoding::{all_specs, lookup_or_panic};

#[test]
fn inv24_k_max_field_present_and_positive() {
    for s in all_specs() {
        assert!(
            s.k_max >= 1,
            "encoding {:?}: k_max must be >= 1, got {}",
            s.name,
            s.k_max
        );
    }
}

#[test]
fn inv24_k_max_values_per_encoding() {
    // Golden snapshot per Wave 9 PREP §A.2. Update both the TOML and this
    // table — never the TOML alone — when re-tuning a multi-window k_max.
    let expected: &[(&str, u32)] = &[
        ("v6", 1),
        ("v7full", 1),
        ("v7", 1),
        ("v7e30", 1),
        ("v6w25", 8),
        ("v7mw", 8),
        ("v8", 1),
        ("v8_canvas_realness", 1),
    ];
    for (name, k_max) in expected {
        let s = lookup_or_panic(name);
        assert_eq!(
            s.k_max, *k_max,
            "encoding {:?}: expected k_max={}, got {}",
            name, k_max, s.k_max
        );
    }
}

#[test]
fn inv24_single_window_implies_k_max_one() {
    // Discipline check (not a validator rule). If a future single-window
    // encoding intentionally adds multi-view emission (e.g. canvas-augmented
    // dual-view), update this test alongside the TOML and document why.
    for s in all_specs() {
        if !s.is_multi_window {
            assert_eq!(
                s.k_max, 1,
                "single-window encoding {:?} has k_max={} (expected 1; if you \
                 intentionally added a multi-view single-window arch, update this \
                 test alongside the TOML)",
                s.name, s.k_max
            );
        }
    }
}
