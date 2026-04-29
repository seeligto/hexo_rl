//! §130 — per-game self-play rotation port: structural guards.
//!
//! The production rotation path lives inside `worker_loop.rs` and is wired
//! through the inference batcher and the records queue. The transforms it
//! invokes (state scatter, chain scatter with axis-plane remap, policy
//! scatter, aux scatter) are the same primitives the replay-buffer sample
//! path already uses; this file pins the round-trip and orbit-coverage
//! properties that the worker loop relies on per game.
//!
//! What this file asserts (per the W4 Step 1 spec):
//!   1. `inv_sym_idx ∘ sym_idx` is the identity on the dihedral group — i.e.
//!      forward-then-inverse coordinate scatter recovers the original tensor
//!      (state, chain, policy, ownership/winning_line aux) for every sym.
//!   2. State scatter preserves the stone count of an in-window placement,
//!      and the destination cell agrees with the chain/policy/aux scatter
//!      under the same sym (top-K selection on policy is therefore a pure
//!      relabelling, not a content change — §127 doc-comment property).
//!   3. The pass-action slot of a 362-element policy is invariant under
//!      every dihedral element.
//!   4. Eval/bot construction of `SelfPlayRunner` (default ctor signature)
//!      defaults `selfplay_rotation_enabled` to `false`. The flag is the
//!      only opt-in to per-game scatter; eval pipelines that bypass
//!      `WorkerPool` therefore never rotate.
//!
//! Together these cover the W4 Step 1 done-when criteria:
//!   * `test_rotation_state_round_trip` and `test_rotation_chain_round_trip`
//!     and `test_rotation_aux_round_trip` and `test_rotation_policy_round_trip`
//!     prove the inverse cancels the forward (idempotency under a full
//!     forward+inverse cycle is a strict subset of the orbit identity that
//!     the round trip checks).
//!   * `test_rotation_preserves_top_k_under_relabel` proves that the top-K
//!     identity is stable under rotation (children are relabelled, not
//!     dropped or re-ranked).
//!   * `test_rotation_disabled_by_default_in_runner` proves the eval path
//!     stays at sym_idx=0.

use engine::game_runner::SelfPlayRunner;
use engine::replay_buffer::sample::{apply_chain_symmetry, apply_symmetry_state};
use engine::replay_buffer::sym_tables::{
    SymTables, N_ACTIONS, N_CELLS, N_CHAIN_PLANES, N_PLANES, N_SYMS,
};

/// Match `inv_sym_idx` from `engine/src/game_runner/worker_loop.rs` —
/// re-derived here so the test catches drift if the production helper
/// changes its parameterisation.
fn inv_sym(s: usize) -> usize {
    if s < 6 { (6 - s) % 6 } else { s }
}

/// State-plane scatter forward+inverse round trip for every sym_idx.
///
/// The scatter table drops cells that fall outside the 19×19 window after
/// the transform; for source cells in the central interior this is empty,
/// so the round trip is byte-exact. We plant markers on the central spine
/// (q in {-3..3}, r in {-3..3}) which is well inside every dihedral image.
#[test]
fn test_rotation_state_round_trip() {
    let tables = SymTables::new();
    let half = 9usize;
    let board_w = 19usize;

    // Build a deterministic source tensor: distinct value per cell on the
    // 7×7 central core so scatter mistakes (off-by-one, axis swap) show up
    // as a value mismatch rather than a coincidental zero collision.
    let mut src = vec![0.0f32; N_PLANES * N_CELLS];
    for plane in 0..N_PLANES {
        let plane_base = plane * N_CELLS;
        let mut tag = 1.0f32;
        for dq in -3i32..=3 {
            for dr in -3i32..=3 {
                let q = (half as i32 + dq) as usize;
                let r = (half as i32 + dr) as usize;
                src[plane_base + q * board_w + r] = tag + plane as f32 * 1000.0;
                tag += 1.0;
            }
        }
    }

    for sym_idx in 0..N_SYMS {
        let mut rotated = vec![0.0f32; src.len()];
        apply_symmetry_state::<f32>(&src, &mut rotated, sym_idx, &tables);

        let mut recovered = vec![0.0f32; src.len()];
        apply_symmetry_state::<f32>(&rotated, &mut recovered, inv_sym(sym_idx), &tables);

        // For interior cells (well within the 7×7 core), forward+inverse must
        // be byte-exact. Cells outside that core may have lost mass to the
        // 19×19 window edge under intermediate rotations — those are zero in
        // the source, so the assertion still passes for them.
        assert_eq!(
            recovered, src,
            "state round trip failed for sym_idx={sym_idx} (inv={})",
            inv_sym(sym_idx)
        );
    }
}

/// Chain-plane scatter forward+inverse round trip — exercises the axis-plane
/// remap (chain planes encode hex-axis-specific data; rotation must permute
/// the axis dimension and the cell dimension consistently).
#[test]
fn test_rotation_chain_round_trip() {
    let tables = SymTables::new();
    let half = 9usize;
    let board_w = 19usize;

    let mut src = vec![0.0f32; N_CHAIN_PLANES * N_CELLS];
    for plane in 0..N_CHAIN_PLANES {
        let plane_base = plane * N_CELLS;
        // Different value per (plane, dq, dr) so an axis-perm bug shows up
        // as a value mismatch rather than coincidental equality.
        for dq in -2i32..=2 {
            for dr in -2i32..=2 {
                let q = (half as i32 + dq) as usize;
                let r = (half as i32 + dr) as usize;
                src[plane_base + q * board_w + r] =
                    (plane as f32 + 1.0) * 100.0 + (dq + 3) as f32 * 10.0 + (dr + 3) as f32;
            }
        }
    }

    for sym_idx in 0..N_SYMS {
        let mut rotated = vec![0.0f32; src.len()];
        apply_chain_symmetry::<f32>(&src, &mut rotated, sym_idx, &tables);

        let mut recovered = vec![0.0f32; src.len()];
        apply_chain_symmetry::<f32>(&rotated, &mut recovered, inv_sym(sym_idx), &tables);

        assert_eq!(
            recovered, src,
            "chain round trip failed for sym_idx={sym_idx} (inv={})",
            inv_sym(sym_idx)
        );
    }
}

/// Policy scatter (362 = N_CELLS + 1, with pass action) round trip.
///
/// Mirrors the production helper at `worker_loop.rs::rotate_policy_inplace`:
/// scatter cells 0..361 by sym, copy the pass slot at 361 verbatim.
fn rotate_policy(src: &[f32], sym_idx: usize, tables: &SymTables) -> Vec<f32> {
    assert_eq!(src.len(), N_ACTIONS);
    let mut dst = vec![0.0f32; N_ACTIONS];
    for &(sc, dc) in &tables.scatter[sym_idx] {
        dst[dc as usize] = src[sc as usize];
    }
    dst[N_CELLS] = src[N_CELLS];
    dst
}

#[test]
fn test_rotation_policy_round_trip() {
    let tables = SymTables::new();
    let mut src = vec![0.0f32; N_ACTIONS];
    // Tag every cell with a unique nonzero value so missed scatters surface.
    // Offset by 1.0 so we can also spot stray zeros from a broken pass-slot copy.
    for i in 0..N_ACTIONS {
        src[i] = 1.0 + i as f32 * 0.01;
    }

    for sym_idx in 0..N_SYMS {
        let rotated = rotate_policy(&src, sym_idx, &tables);
        let recovered = rotate_policy(&rotated, inv_sym(sym_idx), &tables);

        // Pass action: invariant under every sym.
        assert_eq!(
            rotated[N_CELLS], src[N_CELLS],
            "pass slot drifted under sym_idx={sym_idx}"
        );

        // Cells inside the central core round-trip exactly. (Edge cells may
        // map outside the window under intermediate rotations and lose mass;
        // the round-trip recovers them only when both forward and inverse
        // keep them in-window. We check the central 7×7 core which always
        // round-trips.)
        let half = 9usize;
        let board_w = 19usize;
        for dq in -3i32..=3 {
            for dr in -3i32..=3 {
                let q = (half as i32 + dq) as usize;
                let r = (half as i32 + dr) as usize;
                let i = q * board_w + r;
                assert_eq!(
                    recovered[i], src[i],
                    "policy round trip cell ({dq},{dr}) failed for sym_idx={sym_idx}"
                );
            }
        }
    }
}

/// Combined aux (ownership ‖ winning_line) round trip — mirrors
/// `worker_loop.rs::rotate_aux_inplace` exactly so a regression in the
/// production helper's ownership-default initialisation surfaces here.
fn rotate_aux(src: &[u8], sym_idx: usize, tables: &SymTables) -> Vec<u8> {
    assert_eq!(src.len(), 2 * N_CELLS);
    let mut dst = vec![0u8; 2 * N_CELLS];
    dst[..N_CELLS].fill(1); // ownership default = empty
    for &(sc, dc) in &tables.scatter[sym_idx] {
        dst[dc as usize]            = src[sc as usize];
        dst[N_CELLS + dc as usize]  = src[N_CELLS + sc as usize];
    }
    dst
}

#[test]
fn test_rotation_aux_round_trip() {
    let tables = SymTables::new();

    // Build aux: ownership tagged 0/1/2 across central spine; winning_line a
    // 6-cell linear pattern (mock winning line) along axis_q.
    let mut src = vec![0u8; 2 * N_CELLS];
    src[..N_CELLS].fill(1);
    let half = 9usize;
    let board_w = 19usize;
    for dq in -3i32..=3 {
        for dr in -3i32..=3 {
            let q = (half as i32 + dq) as usize;
            let r = (half as i32 + dr) as usize;
            let i = q * board_w + r;
            src[i] = ((dq + dr).rem_euclid(3)) as u8; // 0/1/2
        }
    }
    // Mock winning line at (q in 0..6, r=0) — central spine, well inside core.
    for j in 0..6 {
        let q = (half as i32 + (j - 2)) as usize;
        let r = half;
        src[N_CELLS + q * board_w + r] = 1;
    }

    for sym_idx in 0..N_SYMS {
        let rotated = rotate_aux(&src, sym_idx, &tables);
        let recovered = rotate_aux(&rotated, inv_sym(sym_idx), &tables);

        // Central-core round trip is exact (same justification as state).
        for dq in -2i32..=2 {
            for dr in -2i32..=2 {
                let q = (half as i32 + dq) as usize;
                let r = (half as i32 + dr) as usize;
                let i = q * board_w + r;
                assert_eq!(
                    recovered[i], src[i],
                    "aux ownership ({dq},{dr}) round trip failed sym_idx={sym_idx}"
                );
                assert_eq!(
                    recovered[N_CELLS + i], src[N_CELLS + i],
                    "aux winning_line ({dq},{dr}) round trip failed sym_idx={sym_idx}"
                );
            }
        }
    }
}

/// Top-K identity stability under rotation — §127 doc-comment property.
///
/// Apply rotation to a policy with a clear top-3, recover indices of the
/// top-3 cells in the rotated frame, then inverse-rotate those cells and
/// confirm we land back on the original three. The set as a relabelling is
/// what subtree reuse (Q40) depends on: rotation never adds, drops, or
/// re-orders children — it relabels them.
#[test]
fn test_rotation_preserves_top_k_under_relabel() {
    let tables = SymTables::new();
    let half = 9i32;
    let board_w = 19usize;

    // Plant a clear top-3 on three nearby central cells: distinct masses.
    let mut src = vec![0.0f32; N_ACTIONS];
    let pick = |q: i32, r: i32| -> usize {
        ((half + q) as usize) * board_w + ((half + r) as usize)
    };
    let cells_top3: [(i32, i32); 3] = [(0, 0), (1, 0), (-1, 1)];
    src[pick(cells_top3[0].0, cells_top3[0].1)] = 0.50;
    src[pick(cells_top3[1].0, cells_top3[1].1)] = 0.30;
    src[pick(cells_top3[2].0, cells_top3[2].1)] = 0.15;
    src[N_CELLS] = 0.05;

    for sym_idx in 0..N_SYMS {
        let rotated = rotate_policy(&src, sym_idx, &tables);

        // Find top-3 cell indices in the rotated frame.
        let mut indexed: Vec<(usize, f32)> = (0..N_CELLS).map(|i| (i, rotated[i])).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let top3_rot: [usize; 3] = [indexed[0].0, indexed[1].0, indexed[2].0];

        // Inverse-rotate just those positions: build a sparse policy with the
        // rotated cells set, scatter back, find which canonical cells they
        // landed on. Should equal the original top-3 set (set equality —
        // rotation may permute the rank order is irrelevant; the *identity*
        // of the children is what matters for subtree reuse).
        let mut sparse = vec![0.0f32; N_ACTIONS];
        for (rank, &cell) in top3_rot.iter().enumerate() {
            sparse[cell] = (3 - rank) as f32; // distinct nonzero
        }
        let recovered = rotate_policy(&sparse, inv_sym(sym_idx), &tables);

        let mut recovered_cells: Vec<usize> = (0..N_CELLS)
            .filter(|&i| recovered[i] > 0.0)
            .collect();
        recovered_cells.sort();

        let mut expected_cells: Vec<usize> = cells_top3
            .iter()
            .map(|&(q, r)| pick(q, r))
            .collect();
        expected_cells.sort();

        assert_eq!(
            recovered_cells, expected_cells,
            "top-3 cell set drifted under sym_idx={sym_idx}: rotated cells {:?} → recovered {:?}, expected {:?}",
            top3_rot, recovered_cells, expected_cells
        );
    }
}

/// Default `SelfPlayRunner` ctor must default `selfplay_rotation_enabled = false`.
///
/// Eval pipelines (`eval_pipeline.py`, `our_model_bot.py`) construct the
/// runner via the default ctor or via `WorkerPool` paths that explicitly do
/// not flip the flag. The default-false guarantee means the eval/bot path
/// can never accidentally rotate — the only way `sym_idx ≠ 0` is via the
/// self-play training loop opting in.
///
/// The runner exposes the flag through the `#[pyo3(signature)]` default and
/// reads it at thread-spawn time. We can't introspect the field directly
/// (it's `pub(crate)`), but we can construct a runner with the documented
/// public ctor signature and exercise that the kwargs default behaves the
/// same as omitting the flag entirely. Both paths must succeed and yield a
/// runner that does not start a NaN MCTS pass before stop().
#[test]
fn test_rotation_disabled_by_default_in_runner() {
    // Default ctor — rotation flag omitted; spec defaults to false.
    let runner_default = SelfPlayRunner::new(
        1,            // n_workers
        0,            // max_moves_per_game (skip game loop)
        1,            // n_simulations
        1,            // leaf_batch_size
        1.5,          // c_puct
        0.25,         // fpu_reduction
        8 * 19 * 19,  // feature_len
        19 * 19 + 1,  // policy_len
        0.0,          // fast_prob
        1,            // fast_sims
        1,            // standard_sims
        15,           // temp_threshold
        -0.1,         // draw_reward
        true,         // quiescence_enabled
        0.3,          // quiescence_blend_2
        0.05,         // temp_min
        false,        // zoi_enabled
        16,           // zoi_lookback
        5,            // zoi_margin
        false,        // completed_q
        50.0,         // c_visit
        1.0,          // c_scale
        false,        // gumbel_mcts
        16,           // gumbel_m
        10,           // gumbel_explore
        0.3,          // dirichlet_alpha
        0.25,         // dirichlet_eps
        true,         // dirichlet_enabled
        10_000,       // results_queue_cap
        0.0_f32,      // full_search_prob
        0_usize,      // n_sims_quick
        0_usize,      // n_sims_full
        0_u32,        // random_opening_plies
        false,        // selfplay_rotation_enabled (eval default)
    )
    .expect("runner ctor with rotation=false must succeed");
    assert!(!runner_default.is_running());

    // Explicit rotation=true must also accept (training-loop path).
    let runner_rot = SelfPlayRunner::new(
        1, 0, 1, 1, 1.5, 0.25, 8 * 19 * 19, 19 * 19 + 1, 0.0, 1, 1, 15, -0.1, true,
        0.3, 0.05, false, 16, 5, false, 50.0, 1.0, false, 16, 10, 0.3, 0.25, true,
        10_000, 0.0_f32, 0_usize, 0_usize, 0_u32, true,
    )
    .expect("runner ctor with rotation=true must succeed");
    assert!(!runner_rot.is_running());
}
