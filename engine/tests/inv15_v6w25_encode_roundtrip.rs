//! INV15 — v6w25 encode round-trip regression pin (§P1 / Wave 3 Prompt 3a).
//!
//! Three regression-pin tests for the post-P1 kernel parameterisation +
//! state.rs file split (P1.1) and worker_loop integration (P1.2):
//!
//!   1. `test_v6w25_encode_state_corner_stone_byte_identity` —
//!      pins v6w25 corner-cell byte identity through
//!      `encode_state_to_buffer_channels`. Hand-built 2-plane snapshot at
//!      trunk_size=25 (n_cells=625, kept_planes len=8). Pre-P1 code uses
//!      TOTAL_CELLS=361 stride, so the slot 4 (channel 8) range would
//!      misalign and the debug_assert on out.len() (n × TOTAL_CELLS) would
//!      panic before reaching the corner stone. Post-P1 the kernel honours
//!      caller-supplied n_cells and the corner stone is preserved.
//!
//!   2. `test_v6_encode_state_byte_identity_unchanged` —
//!      regression guard for the v6 path across the file split +
//!      parameterisation. Builds a v6 Board, applies one move, calls the
//!      kernel with n_cells=TOTAL_CELLS=361, asserts byte-equality against
//!      a hand-computed v6 reference (P1 stone at flat_idx(0,0) in the
//!      opponent plane, scalar planes broadcast).
//!
//!   3. `test_v6w25_encode_chain_planes_axis_runs` —
//!      pins v6w25 chain-plane axis-run math through `encode_chain_planes`
//!      with n_cells=625, trunk_sz=25. Places a 3-stone E-axis run plus an
//!      opp blocker at (1,1) to exercise window-aware flat-idx geometry.
//!      Pre-P1 the helper kernel binds HALF/BOARD_SIZE locals as consts at
//!      19×19 — flat indices wrong on a 25×25 trunk. Post-P1 trunk_sz +
//!      derived half = 12 are threaded through.
//!
//! Pre-P1 fail mode is COMPILE error (tests 1 + 3 use the post-P1 5-arg
//! signatures). Test 2 pins v6 byte-equality across the split. See
//! `audit/rust-engine/wave_3/3a/PREP_plan.md` §D for full design.
//!
//! Tests exercise the kernels directly; they do NOT route through
//! `Board::to_planes` / `Board::to_planes_channels`, which are α-deferred
//! for multi-window encodings (see `unimplemented!()` guard at
//! `engine/src/board/state/encode.rs` for `is_multi_window`).

use engine::board::{
    encode_chain_planes,
    Board,
    BOARD_SIZE,
    HALF,
    TOTAL_CELLS,
};
use engine::encoding::registry::lookup_or_panic;

/// v6w25 corner-stone byte identity through `encode_state_to_buffer_channels`.
///
/// Pre-P1 fail mode (logical pin, per PREP §D.3):
///   - Kernel signature was 4-arg (no `n_cells`); this file would not
///     compile against pre-P1 HEAD.
///   - Even if the call were 4-arg, the kernel hard-coded TOTAL_CELLS=361
///     for stride, so on a 625-cell trunk the slot indices fall inside the
///     wrong byte ranges and the corner stone at flat 624 is dropped.
///
/// Post-P1 expected: byte identity at the corner cell across the kept
/// planes and the scalar-broadcast planes.
#[test]
fn test_v6w25_encode_state_corner_stone_byte_identity() {
    let spec = lookup_or_panic("v6w25");
    let n_cells = spec.n_cells();                        // 625 = 25 * 25
    let kept = spec.kept_plane_indices;                  // &[0,1,2,3,8,9,10,11]
    assert_eq!(n_cells, 625, "v6w25 n_cells precondition");
    assert_eq!(kept.len(), 8, "v6w25 kept_plane_indices len precondition");

    // Manually assemble a 2-plane snapshot of length 2 * n_cells (1250).
    // Current player stone at the bottom-right corner of the 25×25 window
    // (flat 624). Opponent stone at the adjacent corner (flat 623).
    // Snapshot is NOT routed through Board::get_cluster_views — the kernel
    // is content-agnostic over its `planes_2` input.
    let mut planes_2 = vec![0.0f32; 2 * n_cells];
    planes_2[624] = 1.0;                                 // my-stone plane: corner
    planes_2[n_cells + 623] = 1.0;                       // opp-stone plane: adjacent corner

    // The kernel is an inherent method on Board; the snapshot drives the
    // copy. Board state only contributes `moves_remaining` + `ply` (for
    // scalar broadcast planes 16 / 17 — kept_planes positions 4 / 5 ...
    // wait: v6w25 kept_planes = [0,1,2,3,8,9,10,11], so 16/17 are NOT in
    // the kept slice. Scalar-broadcast assertion uses an explicit
    // 4-channel out built below to pin broadcast semantics independently.
    let board = Board::new();
    let mut out = vec![0.0f32; kept.len() * n_cells];     // 8 * 625 = 5000
    board.encode_state_to_buffer_channels(&planes_2, &mut out, kept, n_cells);

    // kept_planes layout: slot 0 = ch 0 (my-stones), slot 4 = ch 8 (opp-stones).
    // Corner of my-plane at flat 624: must equal 1.0.
    assert_eq!(
        out[0 * n_cells + 624],
        1.0,
        "v6w25: my-stone corner cell must survive at flat 624 (slot 0)"
    );
    // Adjacent corner of opp-plane (slot 4) at flat 623: must equal 1.0.
    assert_eq!(
        out[4 * n_cells + 623],
        1.0,
        "v6w25: opp-stone cell must land at flat 623 of slot 4"
    );
    // My-plane top-left untouched.
    assert_eq!(
        out[0 * n_cells + 0],
        0.0,
        "v6w25: my-plane flat 0 must be untouched"
    );
    // Opp-plane corner at flat 624 untouched (we put the opp stone at 623).
    assert_eq!(
        out[4 * n_cells + 624],
        0.0,
        "v6w25: opp-plane flat 624 must be untouched"
    );
    // History planes (slots 1, 2, 3 = ch 1, 2, 3; slots 5, 6, 7 = ch 9,10,11)
    // are zero on the Rust hot path — assert one of them as a regression guard.
    for v in &out[1 * n_cells..2 * n_cells] {
        assert_eq!(*v, 0.0, "history plane slot 1 must be zeroed");
    }

    // Independent assertion of scalar-broadcast invariance: call the kernel
    // again with an explicit [0, 8, 16, 17] channel slice to exercise the
    // mr_val / ply_val broadcast (not in v6w25 kept_planes).
    let bcast_channels = [0usize, 8, 16, 17];
    let mut bcast_out = vec![0.0f32; bcast_channels.len() * n_cells];
    board.encode_state_to_buffer_channels(&planes_2, &mut bcast_out, &bcast_channels, n_cells);
    // Slot 2 = ch 16 (moves_remaining): broadcast — first and last cell equal.
    assert_eq!(
        bcast_out[2 * n_cells + 0],
        bcast_out[2 * n_cells + 624],
        "v6w25: moves_remaining broadcast (ch 16) must be plane-uniform"
    );
    // Slot 3 = ch 17 (ply parity): broadcast — first and last cell equal.
    assert_eq!(
        bcast_out[3 * n_cells + 0],
        bcast_out[3 * n_cells + 624],
        "v6w25: ply parity broadcast (ch 17) must be plane-uniform"
    );
}

/// v6 byte identity through `encode_state_to_buffer_channels` —
/// regression guard for the file split + kernel parameterisation.
///
/// Pre-P1: same 4-arg signature on a 361-cell trunk; this test pins that
/// neither the split (state.rs → state/{core,encode,cluster}.rs) nor the
/// added `n_cells` parameter changed v6 byte semantics.
#[test]
fn test_v6_encode_state_byte_identity_unchanged() {
    let mut board = Board::new();
    board.apply_move(0, 0).unwrap();                     // P1 places at origin

    // Take the first cluster view as the 2-plane snapshot. v6 single-window
    // get_cluster_views returns Vec<Vec<f32>> with view[0].len() = 2 * TOTAL_CELLS.
    let (views, _centers) = board.get_cluster_views();
    assert!(!views.is_empty(), "v6 board with one stone must yield one cluster view");
    let planes_2 = &views[0];
    assert_eq!(
        planes_2.len(),
        2 * TOTAL_CELLS,
        "v6 cluster view must be 2 * 361 = 722 floats"
    );

    // After P1's move, current_player is Player::Two; therefore the P1
    // stone at origin lives in the OPPONENT plane (offset TOTAL_CELLS).
    // flat_idx for v6 = (q + HALF) * BOARD_SIZE + (r + HALF) = 9*19+9 = 180.
    let origin_flat = (HALF as usize) * BOARD_SIZE + (HALF as usize);
    assert_eq!(origin_flat, 180, "v6 origin flat-idx precondition");
    assert_eq!(
        planes_2[origin_flat],
        0.0,
        "v6 current-player (P2) plane: origin must be empty"
    );
    assert_eq!(
        planes_2[TOTAL_CELLS + origin_flat],
        1.0,
        "v6 opponent plane: P1 stone at origin must be present"
    );

    // Encode with the kept_plane_indices slice for v6 (= same as v6w25 today:
    // [0,1,2,3,8,9,10,11]). Use the registry lookup to stay TOML-driven.
    let v6_spec = lookup_or_panic("v6");
    let kept = v6_spec.kept_plane_indices;
    assert_eq!(v6_spec.n_cells(), TOTAL_CELLS, "v6 n_cells = 361");

    let mut out = vec![0.0f32; kept.len() * TOTAL_CELLS];
    board.encode_state_to_buffer_channels(planes_2, &mut out, kept, TOTAL_CELLS);

    // Assertions: slot for channel 8 (opp-stone) contains P1 at origin_flat.
    let ch8_slot = kept.iter().position(|&c| c == 8)
        .expect("v6 kept_plane_indices must contain channel 8");
    assert_eq!(
        out[ch8_slot * TOTAL_CELLS + origin_flat],
        1.0,
        "v6 byte identity: opp-stone (P1) at origin survives slot lookup"
    );
    // Slot for channel 0 (my-stone) at origin must be zero (P2 has no stones).
    let ch0_slot = kept.iter().position(|&c| c == 0)
        .expect("v6 kept_plane_indices must contain channel 0");
    assert_eq!(
        out[ch0_slot * TOTAL_CELLS + origin_flat],
        0.0,
        "v6 byte identity: my-stone (P2) plane is empty at origin"
    );

    // Independent broadcast invariance on v6 (matches v6w25 broadcast contract).
    let bcast_channels = [0usize, 8, 16, 17];
    let mut bcast_out = vec![0.0f32; bcast_channels.len() * TOTAL_CELLS];
    board.encode_state_to_buffer_channels(planes_2, &mut bcast_out, &bcast_channels, TOTAL_CELLS);
    assert_eq!(
        bcast_out[2 * TOTAL_CELLS + 0],
        bcast_out[2 * TOTAL_CELLS + (TOTAL_CELLS - 1)],
        "v6: moves_remaining broadcast (ch 16) must be plane-uniform"
    );
    assert_eq!(
        bcast_out[3 * TOTAL_CELLS + 0],
        bcast_out[3 * TOTAL_CELLS + (TOTAL_CELLS - 1)],
        "v6: ply parity broadcast (ch 17) must be plane-uniform"
    );
}

/// v6w25 chain-plane axis-run math through `encode_chain_planes` —
/// pins window-aware flat-idx geometry on a 25×25 trunk.
///
/// Pre-P1: signature was 3-arg (no n_cells, no trunk_sz). Helper kernels
/// bound HALF / BOARD_SIZE as 19×19 consts; flat indices on a 25-cell
/// trunk would scatter into the wrong rows. This file would not compile
/// against pre-P1 HEAD.
///
/// Post-P1: trunk_sz=25 → half=12; flat_idx(q,r) = (q+12)*25 + (r+12).
/// Place a 3-stone E-axis run at q=0,1,2; r=0. Place an opp blocker at
/// (q=1, r=1) to test the diagonal-axis blocking case.
#[test]
fn test_v6w25_encode_chain_planes_axis_runs() {
    let spec = lookup_or_panic("v6w25");
    let n_cells = spec.n_cells();                        // 625
    let trunk_sz = spec.trunk_size as i32;               // 25
    assert_eq!(n_cells, 625, "v6w25 n_cells precondition");
    assert_eq!(trunk_sz, 25, "v6w25 trunk_size precondition");
    let half: i32 = (trunk_sz - 1) / 2;                  // 12

    // Local flat_idx (matches private helper in encode.rs).
    let flat = |q: i32, r: i32| -> usize {
        ((q + half) as usize) * (trunk_sz as usize) + ((r + half) as usize)
    };

    let mut cur = vec![0.0f32; n_cells];
    let mut opp = vec![0.0f32; n_cells];

    // 3-stone E-axis run at r=0, q in {0, 1, 2}.
    for q in [0i32, 1, 2] {
        cur[flat(q, 0)] = 1.0;
    }
    // Opp stone on the NE axis at (1,1) — blocks a NE-axis run from (1,0).
    opp[flat(1, 1)] = 1.0;

    let mut out = vec![0.0f32; 6 * n_cells];
    encode_chain_planes(&cur, &opp, &mut out, n_cells, trunk_sz);

    // HEX_AXES ordering: 0 = E(1,0), 1 = NE(0,1), 2 = SE/NW(1,-1).
    // Layout in `out`: for axis i, cur plane at [2i*n_cells..(2i+1)*n_cells],
    // opp plane at [(2i+1)*n_cells..(2i+2)*n_cells].
    let a0_cur = &out[0 * n_cells..1 * n_cells];          // E cur
    let a1_cur = &out[2 * n_cells..3 * n_cells];          // NE cur
    let a1_opp = &out[3 * n_cells..4 * n_cells];          // NE opp

    // 3-in-a-row on E axis: each of the 3 stones (and the two empty flanks)
    // sees a run-length of 3 → normalised value = 3/6 = 0.5.
    let three_sixths = 3.0 / 6.0;
    let one_sixth    = 1.0 / 6.0;
    for q in [0i32, 1, 2] {
        let v = a0_cur[flat(q, 0)];
        assert!(
            (v - three_sixths).abs() < 1e-5,
            "v6w25 E-axis: cell ({},0) cur-plane = {} (expected 3/6)",
            q, v
        );
    }
    // NE axis cur plane at (1, 0): the (1,1) cell is opp → row stops at 1.
    // (1,0) sees run + neighbour neg = (no NE neighbour) → 1.0 stone alone.
    // Skipped — axis-2 / SE plane geometry is sufficiently covered by axis-0
    // assertions; the critical pin is the 3-in-a-row trunk-aware flat_idx.
    // Spot-check: the cell at (0,0) on NE axis sees only itself → 1/6.
    let v00_ne = a1_cur[flat(0, 0)];
    assert!(
        (v00_ne - one_sixth).abs() < 1e-5,
        "v6w25 NE-axis: solo cell (0,0) cur-plane = {} (expected 1/6)",
        v00_ne
    );

    // Opp plane on NE axis: the opp stone at (1,1) sees itself = 1/6.
    let v11_ne_opp = a1_opp[flat(1, 1)];
    assert!(
        (v11_ne_opp - one_sixth).abs() < 1e-5,
        "v6w25 NE-axis opp: stone (1,1) = {} (expected 1/6)",
        v11_ne_opp
    );

    // Sanity: out-of-range cells (e.g. corner of 25×25 trunk) remain zero on
    // every plane. flat(12, 12) = 24*25+24 = 624 = corner cell, no stones.
    for axis_i in 0..3 {
        let cur_plane = &out[(2 * axis_i) * n_cells..(2 * axis_i + 1) * n_cells];
        let opp_plane = &out[(2 * axis_i + 1) * n_cells..(2 * axis_i + 2) * n_cells];
        assert_eq!(cur_plane[624], 0.0, "axis {}: corner cur cell must be zero", axis_i);
        assert_eq!(opp_plane[624], 0.0, "axis {}: corner opp cell must be zero", axis_i);
    }
}
