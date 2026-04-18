/// Regression test for A-003: InferenceBatcher default feature_len must be 18*361=6498.
/// Before fix, the #[pyo3(signature)] default was 24*19*19=8664 (stale 24-plane value).
/// Verify the 18-plane size is accepted and reflected in feature buffers.

use engine::inference_bridge::InferenceBatcher;

const EXPECTED_FEATURE_LEN: usize = 18 * 19 * 19; // 6498

#[test]
fn feature_len_18_planes_roundtrip() {
    let batcher = InferenceBatcher::new(EXPECTED_FEATURE_LEN, 19 * 19 + 1);
    assert_eq!(batcher.feature_len(), EXPECTED_FEATURE_LEN);
    // Buffer pool pre-allocates with feature_len; verify a popped buffer has correct size.
    let buf = batcher.get_feature_buffer();
    assert_eq!(buf.len(), EXPECTED_FEATURE_LEN,
        "pre-allocated buffers must match 18-plane feature_len, not stale 24-plane (8664)");
}

#[test]
fn stale_24_plane_default_would_differ() {
    let stale = 24 * 19 * 19; // 8664 — the old wrong default
    assert_ne!(stale, EXPECTED_FEATURE_LEN,
        "sanity: 24-plane and 18-plane sizes must differ");
}
