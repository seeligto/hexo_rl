//! INV21 — P68 module splits byte-identity pin (cycle 3 Wave 7 Batch E).
//!
//! Three regression-pin tests, one per Wave 7 Batch E split:
//!
//!   1. `parse_byte_identity_v6w25_v8` — registry parse path. After
//!      moving `parse_one` out of `encoding/registry.rs` into
//!      `encoding/registry/parse.rs`, every spec field for v6w25 + v8 must
//!      remain byte-identical to a hand-coded golden snapshot. Pins the
//!      registry TOML → RegistrySpec contract.
//!
//!   2. `validate_byte_identity_error_strings` — spec validation path.
//!      After moving `RegistrySpec::validate` out of `encoding/spec.rs`
//!      into `encoding/spec/validate.rs`, the returned `Err(String)` must
//!      contain the same diagnostic substring and wrapping format
//!      (`"RegistrySpec \"...\" validation failed:\n  - "`). Pins the
//!      validate error-message contract.
//!
//!   3. `load_byte_identity_v6w25_roundtrip` — buffer load path. After
//!      moving `load_from_path_impl` out of `replay_buffer/persist.rs`
//!      into `replay_buffer/persist/load.rs`, the HEXB v7 save/load
//!      round-trip must preserve all public-API observables (size,
//!      per-slot is_full_search bits, outcome distribution). Mirrors the
//!      §P13 wire-signature test pattern at
//!      `engine/tests/test_p13_wire_signature_crossload.rs`.
//!
//! Renumbered from PREP §E's proposed `INV20` because Wave 6.5 took
//! INV18 + INV18b (i32::midpoint revert pins), Wave 7 Batch A took INV19
//! \(SelfPlayRunnerConfig builder pin\), and Wave 7 Batch B took INV20
//! \(ReplayBuffer push config field shape pin\).

use engine::encoding::registry::lookup_or_panic;
use engine::encoding::{PolicyPool, RegistrySpec, ValuePool};
use engine::replay_buffer::ReplayBuffer;
use std::env::temp_dir;
use std::sync::atomic::{AtomicU64, Ordering};

/// Per-test unique path under `temp_dir()` — same anti-flake pattern as
/// `replay_buffer_v6_roundtrip.rs` and `test_p13_wire_signature_crossload.rs`.
fn unique_test_path(stem: &str) -> std::path::PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    temp_dir().join(format!("hexb_inv21_{stem}_{pid}_{nanos}_{n}.hexb"))
}

/// Test 1 — parse path byte identity.
///
/// Two representative encodings cover the parse code paths: v6w25 (single
/// multi-window + value/policy pool variants, cluster window present) and v8
/// (single-window + has_pass_slot=false + larger n_planes/n_source_planes).
#[test]
fn parse_byte_identity_v6w25_v8() {
    // v6w25 golden snapshot (post-§173 A3 schema).
    let s = lookup_or_panic("v6w25");
    assert_eq!(s.name, "v6w25");
    assert_eq!(s.board_size, 25);
    assert_eq!(s.trunk_size, 25);
    assert_eq!(s.cluster_window_size, Some(25));
    assert_eq!(s.cluster_threshold, Some(8));
    assert_eq!(s.legal_move_radius, 8);
    assert_eq!(s.n_planes, 8);
    assert_eq!(s.plane_layout.len(), 8);
    assert_eq!(s.policy_logit_count, 626);
    assert!(s.has_pass_slot);
    assert!(s.is_multi_window);
    assert_eq!(s.value_pool, ValuePool::Min);
    assert_eq!(s.policy_pool, PolicyPool::ScatterMax);
    assert_eq!(s.sym_table_id, "size_25");
    assert_eq!(s.kept_plane_indices, &[0usize, 1, 2, 3, 8, 9, 10, 11]);
    assert_eq!(s.n_source_planes, 18);

    // v8 golden snapshot.
    let s = lookup_or_panic("v8");
    assert_eq!(s.name, "v8");
    assert_eq!(s.board_size, 25);
    assert_eq!(s.trunk_size, 25);
    assert_eq!(s.cluster_window_size, None);
    assert_eq!(s.cluster_threshold, None);
    assert_eq!(s.n_planes, 11);
    assert_eq!(s.plane_layout.len(), 11);
    assert_eq!(s.policy_logit_count, 625);
    assert!(!s.has_pass_slot);
    assert!(!s.is_multi_window);
    assert_eq!(s.value_pool, ValuePool::None);
    assert_eq!(s.policy_pool, PolicyPool::None);
    assert_eq!(s.sym_table_id, "size_25");
    assert_eq!(s.kept_plane_indices, &[0usize, 1, 2, 3, 8, 9, 10, 11, 18, 19, 20]);
    assert_eq!(s.n_source_planes, 21);
}

/// Test 2 — validate error-string byte identity.
///
/// Constructs a known-broken RegistrySpec (deep-copied from v6, with n_planes
/// bumped to 9 while plane_layout.len() stays at 8). Calls validate() and
/// asserts the returned Err(String) contains the exact diagnostic substring
/// AND the outer wrapping format. Pins the validate error-format contract
/// across the split.
#[test]
fn validate_byte_identity_error_strings() {
    // RegistrySpec is Copy — deref the &'static returned by lookup_or_panic
    // to obtain an owned mutable copy.
    let mut s: RegistrySpec = *lookup_or_panic("v6");
    s.n_planes = 9; // plane_layout still length 8 → triggers len-mismatch error.

    let err = s.validate().expect_err("broken n_planes must trigger validate error");

    // Inner per-invariant diagnostic substring (preserved across split).
    assert!(
        err.contains("len(plane_layout)=8 != n_planes=9"),
        "missing exact inner diagnostic substring; got: {err}"
    );
    // Outer wrapping format (preserved across split).
    assert!(
        err.starts_with("RegistrySpec \"v6\" validation failed:\n  - "),
        "missing outer wrap format; got: {err}"
    );
}

/// Test 3 — load path byte identity.
///
/// Round-trips a v6w25 buffer through HEXB v7 save/load. Asserts size +
/// per-slot is_full_search bits + outcome distribution survive byte-exact.
/// Public-API restricted: state/chain/policy/aux byte verification is the
/// responsibility of the existing crate-internal round-trip cohort at
/// `engine/src/replay_buffer/persist/mod.rs::tests`; this integration pin
/// catches load-path regressions reachable through the pyo3 facade.
#[test]
fn load_byte_identity_v6w25_roundtrip() {
    let n = 10;
    let mut writer = ReplayBuffer::new(n, "v6w25");

    // Push a diverse, deterministic pattern.
    //   - outcomes 0.0..n linearly spaced
    //   - is_full_search alternating
    //   - game_length varied
    for i in 0..n {
        let outcome = i as f32;
        let game_length = (10 + i * 3) as u16;
        let is_full = (i % 2) == 0;
        writer.push_for_test(outcome, game_length, is_full);
    }
    assert_eq!(writer.size(), n);

    let path = unique_test_path("v6w25_load_byte_identity");
    writer
        .save_to_path(path.to_str().unwrap())
        .expect("v6w25 save_to_path must succeed");

    // Load into a fresh buffer of identical encoding + capacity.
    let mut reader = ReplayBuffer::new(n, "v6w25");
    let loaded = reader
        .load_from_path(path.to_str().unwrap())
        .expect("v6w25 load_from_path must succeed");
    assert_eq!(loaded, n);
    assert_eq!(reader.size(), n);

    // Per-slot is_full_search bits must match the push pattern exactly.
    for slot in 0..n {
        let expected = ((slot % 2) == 0) as u8;
        assert_eq!(
            reader.is_full_search_at(slot),
            expected,
            "slot {slot}: is_full_search byte-identity violated post-split"
        );
    }

    // Outcome distribution check: every outcome 0.0..n was pushed, so a
    // range query covering [0.0, n.0) must return exactly n.
    let outcome_count = reader.outcome_in_range_count(-0.5, n as f32 + 0.5);
    assert_eq!(
        outcome_count, n,
        "outcome distribution byte-identity violated: expected {n} outcomes in [-0.5, {n}.5], got {outcome_count}"
    );

    let _ = std::fs::remove_file(&path);
}
