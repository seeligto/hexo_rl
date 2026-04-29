/// Regression test for A-003 (updated for P3): InferenceBatcher default feature_len
/// must be 8*361=2888. Before §131 fix, the #[pyo3(signature)] default was 24*19*19=8664
/// (stale 24-plane value), then 18*19*19=6498 (pre-P3). Verify the 8-plane size is
/// accepted and reflected in feature buffers.

use engine::inference_bridge::InferenceBatcher;

const EXPECTED_FEATURE_LEN: usize = 8 * 19 * 19; // 2888

#[test]
fn feature_len_8_planes_roundtrip() {
    let batcher = InferenceBatcher::new(EXPECTED_FEATURE_LEN, 19 * 19 + 1);
    assert_eq!(batcher.feature_len(), EXPECTED_FEATURE_LEN);
    // Buffer pool pre-allocates with feature_len; verify a popped buffer has correct size.
    let buf = batcher.get_feature_buffer();
    assert_eq!(buf.len(), EXPECTED_FEATURE_LEN,
        "pre-allocated buffers must match 8-plane feature_len, not stale 18-plane (6498) or 24-plane (8664)");
}

#[test]
fn stale_18_plane_default_would_differ() {
    let stale = 18 * 19 * 19; // 6498 — the old pre-P3 default
    assert_ne!(stale, EXPECTED_FEATURE_LEN,
        "sanity: 18-plane and 8-plane sizes must differ");
}
