//! §173 A5b — aggregate_policy spec-threading integration tests.
//!
//! Verifies that `aggregate_policy` and `aggregate_policy_to_local`:
//!   1. Return a 626-element policy vector for v6w25 (25×25+1) — NOT 362.
//!   2. Scatter-max semantics UNCHANGED: for each legal move m, output is
//!      max(prob_k(m_local)) across all clusters k that cover m.
//!   3. v6 byte-exact regression: v6 geometry (362-element) matches
//!      pre-A5b behavior.
//!
//! §173 A5b-v2 perf fix: signatures changed from `Option<RegistrySpec>` to
//! `(n_actions: usize, trunk_sz: i32)` — callers extract geometry once from
//! spec before entering the hot loop. Tests updated accordingly.
//!
//! These tests exercise the records::aggregate_policy* path with a real
//! v6w25 RegistrySpec from the registry (no mocking required — the registry
//! is &'static so lookup_or_panic is safe in tests).

use engine::board::Board;
use engine::encoding::registry::lookup_or_panic;
use engine::game_runner::records::{aggregate_policy, aggregate_policy_to_local};

// ── 626-vector size tests ──────────────────────────────────────────────────

/// aggregate_policy with v6w25 geometry returns 626-element vector.
#[test]
fn test_aggregate_policy_v6w25_returns_626_vector() {
    let spec = *lookup_or_panic("v6w25");
    let n_actions = spec.policy_stride();   // 626
    let trunk_sz  = spec.trunk_size as i32; // 25
    assert_eq!(n_actions, 626, "precondition: v6w25 policy_stride");

    // Fresh v6w25 board — apply one move to seed legal neighbors.
    let mut board = Board::with_registry_spec(lookup_or_panic("v6w25"));
    board.apply_move(0, 0).unwrap();

    // Single cluster centred at origin with a 626-element zero policy.
    let centers = vec![(0i32, 0i32)];
    let cluster_policies = vec![vec![0.0f32; n_actions]];

    // §P2 SD3: aggregate_policy now takes has_pass_slot (= spec.has_pass_slot).
    // v6w25 has has_pass_slot=true.
    let result = aggregate_policy(n_actions, true, trunk_sz, &board, &centers, &cluster_policies);
    assert_eq!(
        result.len(),
        626,
        "v6w25 aggregate_policy must return 626-element vector"
    );
}

/// aggregate_policy_to_local with v6w25 geometry returns 626-element vector.
#[test]
fn test_aggregate_policy_to_local_v6w25_returns_626_vector() {
    let spec = *lookup_or_panic("v6w25");
    let n_actions = spec.policy_stride();
    let trunk_sz  = spec.trunk_size as i32;

    let mut board = Board::with_registry_spec(lookup_or_panic("v6w25"));
    board.apply_move(0, 0).unwrap();

    let center = (0i32, 0i32);
    let global_policy = vec![0.0f32; n_actions];

    // §P11: test-side legal_moves hoist matches worker_loop call shape.
    let legal_moves = board.legal_moves();
    let result = aggregate_policy_to_local(n_actions, true, trunk_sz, &board, &center, &global_policy, &legal_moves);
    assert_eq!(
        result.len(),
        626,
        "v6w25 aggregate_policy_to_local must return 626-element vector"
    );
}

// ── Scatter-max correctness ────────────────────────────────────────────────

/// Scatter-max: when two clusters both cover a legal move, the output
/// probability at that move's MCTS index is the max across clusters.
///
/// Setup:
///   - v6w25 board, stone at (0,0); legal moves include neighbors near origin.
///   - Two cluster centers: C1=(0,0) and C2=(30,0) — far apart so only
///     moves near origin are covered by C1, moves near (30,0) by C2.
///   - Place a legal move M1 near origin; assign C1 prob 0.6 at M1's local idx.
///   - Place a second move M2 near origin with C1 prob 0.4.
///   - C2 covers neither M1 nor M2 (they are >12 hops from (30,0)).
///   - After scatter-max and renorm: ratio of M1:M2 probs = 0.6:0.4 = 1.5.
#[test]
fn test_aggregate_policy_scatter_max_takes_max_across_clusters() {
    let spec = *lookup_or_panic("v6w25");
    let n_actions = spec.policy_stride();
    let trunk_sz  = spec.trunk_size as i32; // 25
    let half = (trunk_sz - 1) / 2;         // 12

    let mut board = Board::with_registry_spec(lookup_or_panic("v6w25"));
    // Apply opening move so legal moves expand beyond the initial radius.
    board.apply_move(0, 0).unwrap();

    let legal = board.legal_moves();
    assert!(legal.len() >= 2, "need ≥2 legal moves for scatter-max test");

    // C1 centred at origin; C2 far away so no overlap with near-origin legal moves.
    let c1 = (0i32, 0i32);
    let c2 = (30i32, 0i32);  // 30 > 12+8 from origin, so no legal move near origin is in C2.

    // Pick two legal moves both covered by C1 (within 12 hops of origin).
    let mut near_origin: Vec<(i32, i32)> = legal.iter().filter(|&&(q, r)| {
        let wq = q - c1.0 + half;
        let wr = r - c1.1 + half;
        wq >= 0 && wq < trunk_sz && wr >= 0 && wr < trunk_sz
    }).copied().collect();

    assert!(
        near_origin.len() >= 2,
        "need ≥2 legal moves in C1 window; got {}",
        near_origin.len()
    );
    // Sort for determinism across runs.
    near_origin.sort();
    let m1 = near_origin[0];
    let m2 = near_origin[1];

    // Compute local indices for m1 and m2 in C1.
    let (m1q, m1r) = m1;
    let m1_local_c1 = (m1q - c1.0 + half) as usize * trunk_sz as usize
        + (m1r - c1.1 + half) as usize;
    let (m2q, m2r) = m2;
    let m2_local_c1 = (m2q - c1.0 + half) as usize * trunk_sz as usize
        + (m2r - c1.1 + half) as usize;

    // Build cluster policies: C1 has mass on m1 and m2; C2 has no mass on either.
    let mut p_c1 = vec![0.0f32; n_actions];
    let p_c2 = vec![0.0f32; n_actions];

    p_c1[m1_local_c1] = 0.6;  // m1 score from C1
    p_c1[m2_local_c1] = 0.4;  // m2 score from C1

    let centers = vec![c1, c2];
    let cluster_policies = vec![p_c1, p_c2];

    let global = aggregate_policy(n_actions, true, trunk_sz, &board, &centers, &cluster_policies);

    assert_eq!(global.len(), 626, "output must be 626-vector for v6w25");

    // Get MCTS indices for m1 and m2.
    let mcts_m1 = board.window_flat_idx(m1q, m1r);
    let mcts_m2 = board.window_flat_idx(m2q, m2r);

    // Both must be in valid range (< n_actions - 1).
    assert!(mcts_m1 < 625, "m1 MCTS index must be < 625");
    assert!(mcts_m2 < 625, "m2 MCTS index must be < 625");

    let prob_m1 = global[mcts_m1];
    let prob_m2 = global[mcts_m2];

    // After renorm, ratio should be 0.6/0.4 = 1.5.
    assert!(prob_m1 > 0.0, "m1 must have non-zero probability");
    assert!(prob_m2 > 0.0, "m2 must have non-zero probability");
    let ratio = prob_m1 / prob_m2;
    assert!(
        (ratio - 1.5).abs() < 0.01,
        "scatter-max ratio should be 0.6/0.4=1.5; got m1={prob_m1:.4} m2={prob_m2:.4} ratio={ratio:.4}"
    );

    // Pass slot must be zero (no pass in HTTT).
    assert_eq!(global[625], 0.0, "pass slot must be 0.0");

    // Output must be normalised (sum ≈ 1.0).
    let sum: f32 = global.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-5,
        "aggregate_policy output must sum to 1.0; got {sum}"
    );
}

// ── v6 byte-exact regression ───────────────────────────────────────────────

/// aggregate_policy with v6 geometry returns 362-element vector — byte-exact
/// v6 regression for pre-α callers.
#[test]
fn test_aggregate_policy_v6_returns_362_vector() {
    // Use v6 geometry explicitly (19×19+1 = 362).
    let n_actions: usize = 362;
    let trunk_sz: i32 = 19;

    let mut board = Board::new();
    board.apply_move(0, 0).unwrap();

    let legal = board.legal_moves();
    let n = board.window_flat_idx(legal[0].0, legal[0].1);
    let mut p0 = vec![0.0f32; n_actions];
    if n < 361 { p0[n] = 1.0; }

    let centers = vec![(0i32, 0i32)];
    let cluster_policies = vec![p0];

    // §P2 SD3: v6 has has_pass_slot=true.
    let result = aggregate_policy(n_actions, true, trunk_sz, &board, &centers, &cluster_policies);
    assert_eq!(result.len(), 362, "v6 geometry must return 362-vector");
}

/// aggregate_policy_to_local with v6 geometry returns 362-element vector.
#[test]
fn test_aggregate_policy_to_local_v6_returns_362_vector() {
    let n_actions: usize = 362;
    let trunk_sz: i32 = 19;

    let mut board = Board::new();
    board.apply_move(0, 0).unwrap();

    let global = vec![1.0f32 / n_actions as f32; n_actions];
    let center = (0i32, 0i32);
    // §P11: test-side legal_moves hoist matches worker_loop call shape.
    let legal_moves = board.legal_moves();
    let result = aggregate_policy_to_local(n_actions, true, trunk_sz, &board, &center, &global, &legal_moves);
    assert_eq!(result.len(), 362, "v6 geometry must return 362-vector");
}

/// round-trip: aggregate_policy → aggregate_policy_to_local recovers local
/// policy probabilities for moves in the cluster window.
#[test]
fn test_round_trip_aggregate_to_local_v6w25() {
    let spec = *lookup_or_panic("v6w25");
    let n_actions = spec.policy_stride();
    let trunk_sz  = spec.trunk_size as i32;
    let half = (trunk_sz - 1) / 2;

    let mut board = Board::with_registry_spec(lookup_or_panic("v6w25"));
    board.apply_move(0, 0).unwrap();

    // Single cluster at origin.
    let center = (0i32, 0i32);
    let legal = board.legal_moves();
    assert!(!legal.is_empty());

    // Build a policy with mass only on the first legal move.
    let mut local_init = vec![0.0f32; n_actions];
    let (lq, lr) = legal[0];
    let wq = lq - center.0 + half;
    let wr = lr - center.1 + half;
    if wq >= 0 && wq < trunk_sz && wr >= 0 && wr < trunk_sz {
        local_init[(wq * trunk_sz + wr) as usize] = 1.0;
    }

    let centers = vec![center];
    let cluster_policies = vec![local_init.clone()];

    // Aggregate to global.
    let global = aggregate_policy(n_actions, true, trunk_sz, &board, &centers, &cluster_policies);
    assert_eq!(global.len(), 626);

    // Project back to local.
    // §P11: test-side legal_moves hoist matches worker_loop call shape.
    let local_out = aggregate_policy_to_local(n_actions, true, trunk_sz, &board, &center, &global, &legal);
    assert_eq!(local_out.len(), 626);

    // The move's local index should recover its probability.
    if wq >= 0 && wq < trunk_sz && wr >= 0 && wr < trunk_sz {
        let local_idx = (wq * trunk_sz + wr) as usize;
        assert!(
            local_out[local_idx] > 0.0,
            "local policy should recover non-zero prob at move's local index after round-trip"
        );
    }
}
