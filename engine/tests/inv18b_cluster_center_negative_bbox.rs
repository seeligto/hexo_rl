//! INV18b — `Board::get_cluster_views` truncate-toward-zero pin on
//! negative-bbox cluster centres (§Wave 6.5 revert of cycle 2 Wave 6
//! Batch A commit `2b0dd08`).
//!
//! Companion to INV18 (`engine/tests/inv18_window_center_negative_bbox.rs`)
//! covering the cluster-centre code path at
//! `engine/src/board/state/cluster.rs:75,96` instead of the bbox-centroid
//! path at `engine/src/board/state/core.rs:365,366`.
//!
//! Cycle 2's `cargo clippy --fix` over `manual_midpoint` lint substituted
//! `(a + b) / 2` (truncate) with `i32::midpoint(a, b)` (floor toward -∞)
//! at the small-cluster centroid (L75) AND the massive-cluster fallback
//! (L96). The two semantics diverge by one cell whenever `(min + max)`
//! is negative-odd. Forensic at
//! `audit/rust-engine/cycle_3/00_i32_midpoint_forensic.md` §2 (Sites A1, A2)
//! identifies the small-cluster path (L75) as the higher-exposure call
//! site under normal HTTT play.
//!
//! Tests:
//!   1. `test_cluster_center_small_path_negative_q` — default v6 Board
//!      with stones placed at `q ∈ {-5, -4, -3, -2}, r=0`. All within
//!      hex-distance 1 of each other → single cluster. Span q=3 ≤
//!      threshold (window_size - 4 = 15 for v6) → small-cluster path
//!      (L75). Bbox q=[-5, -2], sum=-7 → centroid q under truncate=-3.
//!      Under `i32::midpoint` floor this would yield -4 — the test
//!      would have FAILED at commit `2b0dd08` and passes post-revert.
//!   2. `test_cluster_center_v6w25_kept_planes` — v6w25 spec (cluster
//!      window=25, threshold=21), single cluster on the r-axis with
//!      negative-odd r-sum (r ∈ {-5, -4, -3, -2}, q=0). Asserts the
//!      cluster's centre is `(0, -3)` (truncate) not `(0, -4)` (floor).
//!
//! Construction notes:
//!   - `apply_move(q, r)` validates only cell occupancy; negative coords
//!     + cells outside the default legal-move radius are accepted as
//!     direct writes (test-side construction pattern).
//!   - HTTT turn rhythm: P1 opens with 1 move, then P1+P2 alternate
//!     2-move turns. Cluster ownership is irrelevant — `get_clusters()`
//!     groups stones by hex-distance regardless of player.
//!   - Multiple clusters can be returned; tests use a `contains` check
//!     against `final_centers` rather than indexing the first slot,
//!     because cluster iteration order depends on the underlying
//!     HashMap iteration order.

use engine::board::Board;
use engine::encoding::lookup_or_panic;

/// Default v6 Board, single small cluster on q-axis with negative-odd q-sum.
/// Asserts the small-cluster path at `cluster.rs:75` produces centroid
/// `(-3, 0)` (truncate) — not `(-4, 0)` (floor).
#[test]
fn test_cluster_center_small_path_negative_q() {
    let mut b = Board::new();
    // Turn 1 (P1): 1 move.
    b.apply_move(-5, 0).expect("apply -5,0");
    // Turn 2 (P2): 2 moves — placed within the same cluster (within
    // hex-distance 5 of the P1 stone) so all 4 stones form one cluster.
    b.apply_move(-4, 0).expect("apply -4,0");
    b.apply_move(-3, 0).expect("apply -3,0");
    // Turn 3 (P1): 2 moves — last in-cluster stone + a far-away filler
    // to avoid the empty-clusters fallback edge.
    b.apply_move(-2, 0).expect("apply -2,0");
    b.apply_move(100, 100).expect("apply 100,100 (far filler)");

    let (_views, centers) = b.get_cluster_views();

    // The negative-q cluster bbox is q=[-5,-2], r=[0,0].
    // Truncate: ((-5) + (-2)) / 2 == -3 (NOT i32::midpoint's -4).
    let want = (-3, 0);
    assert!(
        centers.contains(&want),
        "small-cluster centroid must include {:?} (truncate semantic) — \
         i32::midpoint floor would give (-4, 0). Got centers: {:?}",
        want, centers,
    );
    // Negative regression: -4 must NOT appear as a centre on this bbox
    // (would indicate floor-semantic reintroduction).
    assert!(
        !centers.contains(&(-4, 0)),
        "centroid (-4, 0) would indicate i32::midpoint floor — got {:?}",
        centers,
    );
}

/// v6w25 spec (cluster_window_size=25, threshold=21), single small
/// cluster on r-axis with negative-odd r-sum. Asserts truncate centroid
/// `(0, -3)` (not floor `(0, -4)`). Pins the v6w25 K-cluster path under
/// the §174-retired-but-still-pinned encoding so future α multi-window
/// work doesn't silently reintroduce the floor semantic.
#[test]
fn test_cluster_center_v6w25_kept_planes() {
    let spec = lookup_or_panic("v6w25");
    let mut b = Board::with_registry_spec(spec);
    // Sanity: v6w25 multi-window cluster geometry must be present.
    assert_eq!(b.cluster_window_size(), 25, "v6w25 window_size precondition");

    // Same construction as test 1 but rotated to the r-axis to also
    // exercise the cr cluster-centre code path symmetrically.
    b.apply_move(0, -5).expect("apply 0,-5");
    b.apply_move(0, -4).expect("apply 0,-4");
    b.apply_move(0, -3).expect("apply 0,-3");
    b.apply_move(0, -2).expect("apply 0,-2");
    b.apply_move(100, 100).expect("apply 100,100 (far filler)");

    let (_views, centers) = b.get_cluster_views();

    // Cluster bbox: q=[0,0], r=[-5,-2]. Truncate: ((-5) + (-2)) / 2 == -3.
    // i32::midpoint floor would yield (0, -4).
    let want = (0, -3);
    assert!(
        centers.contains(&want),
        "v6w25 cluster centroid must include {:?} (truncate semantic) — \
         i32::midpoint floor would give (0, -4). Got centers: {:?}",
        want, centers,
    );
    assert!(
        !centers.contains(&(0, -4)),
        "centroid (0, -4) would indicate i32::midpoint floor reintroduction — got {:?}",
        centers,
    );
}
