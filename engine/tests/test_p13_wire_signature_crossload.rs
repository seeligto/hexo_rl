//! §P13 regression: cross-encoding HEXB load guard compares wire-format
//! signature `(n_planes, board_size, policy_logit_count, has_pass_slot,
//! sym_table_id)` instead of encoding name. Wire-identical encodings
//! auto-cross-load; shape-different encodings still hard-error.
//!
//! Coverage (integration scope — happy path only):
//!  1. v6 → v7full wire-identical crossload SUCCEEDS (was strict-name reject).
//!  2. v7full → v6 wire-identical crossload SUCCEEDS (symmetric).
//!
//! The mismatch-reject regression sits in `replay_buffer/persist.rs` unit
//! tests (`test_hexb_v7_encoding_mismatch_rejects` covers v6 → v6w25; the new
//! `test_hexb_v7_wire_signature_v6_to_v8_rejects` covers v6 → v8) — those
//! exercise `load_from_path_impl` directly so they don't need a Python
//! interpreter to construct a `PyValueError`.
//!
//! Per Wave 5a Batch C PREP §A: v6 / v7full / v7 / v7e30 / v7mw all share
//! `(8, 19, 362, true, "size_19")`. v8 / v8_canvas_realness share
//! `(11, 25, 625, false, "size_25")`. v6w25 stands alone at
//! `(8, 25, 362, true, "size_25")`.

use engine::replay_buffer::ReplayBuffer;
use std::env::temp_dir;
use std::sync::atomic::{AtomicU64, Ordering};

/// Per-test unique path under `temp_dir()` — same pattern as
/// `replay_buffer_v6_roundtrip.rs`. Triaged cycle 2 wave 5 pre-flight (Flake 3).
fn unique_test_path(stem: &str) -> std::path::PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    temp_dir().join(format!("hexb_p13_{stem}_{pid}_{nanos}_{n}.hexb"))
}

/// v6 → v7full wire-identical crossload (Wave 5a §P13 primary case).
#[test]
fn p13_wire_signature_v6_to_v7full_crossload_succeeds() {
    let mut writer = ReplayBuffer::new(8, "v6");
    for i in 0..6 {
        writer.push_for_test(i as f32, 10, i % 2 == 0);
    }

    let path = unique_test_path("v6_to_v7full");
    writer.save_to_path(path.to_str().unwrap()).unwrap();

    // Load v6-on-disk file into a v7full buffer — wire-identical → accept.
    let mut reader = ReplayBuffer::new(8, "v7full");
    let loaded = reader
        .load_from_path(path.to_str().unwrap())
        .expect("v6 -> v7full crossload must succeed (wire-identical)");
    assert_eq!(loaded, 6);
    assert_eq!(reader.size(), 6);
    for slot in 0..6 {
        assert_eq!(
            reader.is_full_search_at(slot),
            (slot % 2 == 0) as u8,
            "slot {slot}: is_full_search must survive cross-encoding load"
        );
    }

    let _ = std::fs::remove_file(&path);
}

/// v7full → v6 wire-identical crossload (symmetry of the primary case).
#[test]
fn p13_wire_signature_v7full_to_v6_crossload_succeeds() {
    let mut writer = ReplayBuffer::new(8, "v7full");
    for i in 0..5 {
        writer.push_for_test(i as f32, 25, false);
    }

    let path = unique_test_path("v7full_to_v6");
    writer.save_to_path(path.to_str().unwrap()).unwrap();

    let mut reader = ReplayBuffer::new(8, "v6");
    let loaded = reader
        .load_from_path(path.to_str().unwrap())
        .expect("v7full -> v6 crossload must succeed (wire-identical)");
    assert_eq!(loaded, 5);

    let _ = std::fs::remove_file(&path);
}

