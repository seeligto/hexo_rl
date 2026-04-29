//! Regression guard for HEXB v6 save/load round-trip of `is_full_search` and
//! `game_length`-derived weights (F-029, reports/master_review_2026-04-18/).
//! Originally written against v5; v6 reuses the same on-disk row layout aside
//! from the state-plane count (18 → 8 per §131), so the round-trip semantics
//! are unchanged.
//!
//! The existing unit test in `persist.rs::test_aux_hexb_v6_roundtrip` manipulates
//! internal state directly and covers a single position. This integration test:
//!   1. Uses the public `push_for_test` API (no direct field writes).
//!   2. Pushes N positions with mixed is_full_search values (0 and 1).
//!   3. Saves to HEXB v6, loads into a buffer with a *larger* capacity.
//!   4. Asserts is_full_search values survive byte-exact at every slot.
//!   5. Asserts game_length-derived weights survive (f32 round-trip within f16 precision).
//!
//! Failure modes this guards against:
//!   - A save-path that skips is_full_search bytes for certain positions.
//!   - A load-path that initialises is_full_search to the default (1) for all
//!     slots rather than restoring the saved bytes.
//!   - A capacity-change bug that shifts the slot mapping on load.

use engine::replay_buffer::ReplayBuffer;
use std::env::temp_dir;

#[test]
fn hexb_v6_is_full_search_survives_roundtrip_at_different_capacity() {
    let n = 8;
    let mut buf = ReplayBuffer::new(n);

    // Push positions with alternating is_full_search: even=1 (full), odd=0 (quick).
    // Use outcome as an identity tag so we can cross-check after load.
    for i in 0..n {
        let is_full = (i % 2) == 0;
        buf.push_for_test(i as f32, 0, is_full);
    }
    assert_eq!(buf.size(), n);

    // is_full_search before save.
    let expected_ifs: Vec<u8> = (0..n).map(|i| (i % 2 == 0) as u8).collect();
    for (slot, &want) in expected_ifs.iter().enumerate() {
        assert_eq!(
            buf.is_full_search_at(slot), want,
            "pre-save: slot {slot} is_full_search mismatch"
        );
    }

    // Save.
    let path = temp_dir().join("integration_v6_ifs_roundtrip.hexb");
    buf.save_to_path(path.to_str().unwrap()).unwrap();

    // Load into a fresh buffer with LARGER capacity (regression: slot mapping must be preserved).
    let mut buf2 = ReplayBuffer::new(n * 2);
    let loaded = buf2.load_from_path(path.to_str().unwrap()).unwrap();
    assert_eq!(loaded, n, "wrong position count after load");
    assert_eq!(buf2.size(), n);

    // is_full_search must be byte-exact for every loaded slot.
    for (slot, &want) in expected_ifs.iter().enumerate() {
        assert_eq!(
            buf2.is_full_search_at(slot), want,
            "post-load: slot {slot} is_full_search mismatch (want={want})"
        );
    }

    let _ = std::fs::remove_file(&path);
}

#[test]
fn hexb_v6_game_length_weight_survives_roundtrip() {
    // Push two groups with different game_length values; the weight schedule maps
    // them to different f16 weights. Verify the weights are preserved after save/load.
    //
    // Weights are stored as f16 bits, so round-trip precision is f16.
    let mut buf = ReplayBuffer::new(10);
    buf.set_weight_schedule(vec![10, 25], vec![0.15, 0.50], 1.0)
        .unwrap();

    // short game (game_length=5 → weight=0.15)
    for _ in 0..4 {
        buf.push_for_test(1.0, 5, true);
    }
    // long game (game_length=40 → weight=1.0, default)
    for _ in 0..4 {
        buf.push_for_test(-1.0, 40, false);
    }

    let path = temp_dir().join("integration_v6_weight_roundtrip.hexb");
    buf.save_to_path(path.to_str().unwrap()).unwrap();

    let mut buf2 = ReplayBuffer::new(20);
    buf2.set_weight_schedule(vec![10, 25], vec![0.15, 0.50], 1.0)
        .unwrap();
    let loaded = buf2.load_from_path(path.to_str().unwrap()).unwrap();
    assert_eq!(loaded, 8);

    for slot in 0..4 {
        let w = buf2.weight_at_f32(slot);
        assert!(
            (w - 0.15).abs() < 0.01,
            "slot {slot}: expected weight ≈0.15 (short game), got {w}"
        );
        assert_eq!(buf2.is_full_search_at(slot), 1, "slot {slot}: is_full_search should be 1");
    }
    for slot in 4..8 {
        let w = buf2.weight_at_f32(slot);
        assert!(
            (w - 1.0).abs() < 0.01,
            "slot {slot}: expected weight ≈1.0 (long game), got {w}"
        );
        assert_eq!(buf2.is_full_search_at(slot), 0, "slot {slot}: is_full_search should be 0");
    }

    let _ = std::fs::remove_file(&path);
}
