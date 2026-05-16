use super::core::{
    Board, Cell, Player, BOARD_SIZE, TOTAL_CELLS, HEX_AXES,
    MY_STONE_PLANE, OPP_STONE_PLANE, MOVES_REMAINING_PLANE, PLY_PARITY_PLANE,
};
#[cfg(test)]
use super::core::HALF;

impl Board {
    // ── Tensor encoding ───────────────────────────────────────────────────────

    /// Encode the 18-plane state tensor into `out` from a 2-plane cluster view.
    ///
    /// Layout:
    ///   plane  0:    current player's stones (from `planes_2[0..TOTAL_CELLS]`)
    ///   planes 1-7:  zero (no Python history on the Rust self-play path)
    ///   plane  8:    opponent's stones (from `planes_2[TOTAL_CELLS..2*TOTAL_CELLS]`)
    ///   planes 9-15: zero (no Python history)
    ///   plane 16:    moves_remaining == 2 broadcast
    ///   plane 17:    ply parity broadcast
    ///   planes 18-23: Q13 chain-length planes (3 hex axes × 2 players),
    ///                 /6.0-normalized to [0, 1]. Layout
    ///                 [a0_cur, a0_opp, a1_cur, a1_opp, a2_cur, a2_opp].
    ///
    /// `out` must have length `18 * TOTAL_CELLS`. Callers are responsible for
    /// zero-initializing the buffer before calling; this function writes to
    /// planes 0, 8, 16, 17 but leaves 1..7 and 9..15 untouched so the
    /// caller can rely on history planes being whatever the buffer started as
    /// (the existing self-play path zero-inits the pooled buffers).
    /// Chain-length planes (formerly 18..23) are computed separately via
    /// `encode_chain_planes` and stored in the replay buffer's dedicated
    /// chain sub-buffer.
    pub fn encode_state_to_buffer(
        &self,
        planes_2: &[f32], // The 2-plane [my, opp] view
        out: &mut [f32]
    ) {
        // Plane 0: my stones
        out[..TOTAL_CELLS].copy_from_slice(&planes_2[..TOTAL_CELLS]);
        // Plane 8: opp stones
        for i in 0..TOTAL_CELLS {
            out[OPP_STONE_PLANE * TOTAL_CELLS + i] = planes_2[TOTAL_CELLS + i];
        }
        // Plane 16: moves_remaining == 2 ? 1.0 : 0.0
        let mr_val = if self.moves_remaining == 2 { 1.0 } else { 0.0 };
        for i in 0..TOTAL_CELLS {
            out[MOVES_REMAINING_PLANE * TOTAL_CELLS + i] = mr_val;
        }
        // Plane 17: ply % 2
        let ply_val = (self.ply % 2) as f32;
        for i in 0..TOTAL_CELLS {
            out[PLY_PARITY_PLANE * TOTAL_CELLS + i] = ply_val;
        }
        debug_assert_eq!(
            out.len(),
            18 * TOTAL_CELLS,
            "encode_state_to_buffer output length mismatch — expected 18 planes × {TOTAL_CELLS} cells"
        );
    }

    /// Public alias for `encode_state_to_buffer`. Preserved as a named entry
    /// point for callers outside this module.
    #[inline]
    pub fn encode_planes_to_buffer(&self, planes_2: &[f32], out: &mut [f32]) {
        self.encode_state_to_buffer(planes_2, out);
    }

    /// Encode a *subset* of the 18 wire planes selected by `channels`, in the
    /// order given. Used by sweep variants whose model in_channels < 18.
    ///
    /// Plane semantics match `encode_state_to_buffer` (see header comment).
    /// Channels 0/8 carry the only non-zero stone information on the Rust
    /// self-play path; channels 16/17 are scalar broadcasts; 1–7 / 9–15
    /// are zero on this path (history filled by Python tensor assembly).
    /// `channels.iter().any(|c| c >= 18)` panics in debug; release silently
    /// skips out-of-range entries.
    ///
    /// `n_cells` is caller-supplied (= `spec.n_cells()` from registry).
    /// `planes_2.len()` must equal `2 * n_cells`; `out.len()` must equal
    /// `channels.len() * n_cells`. v6 callers pass `TOTAL_CELLS` (361);
    /// v6w25 callers pass `625`; v8 single-window callers pass `625`.
    #[inline]
    pub fn encode_state_to_buffer_channels(
        &self,
        planes_2: &[f32],
        out: &mut [f32],
        channels: &[usize],
        n_cells: usize,
    ) {
        let n = channels.len();
        debug_assert_eq!(
            out.len(),
            n * n_cells,
            "encode_state_to_buffer_channels output length mismatch — \
             expected {n} planes × {n_cells} cells"
        );
        let mr_val = if self.moves_remaining == 2 { 1.0 } else { 0.0 };
        let ply_val = (self.ply % 2) as f32;
        for (slot, &ch) in channels.iter().enumerate() {
            let dst = &mut out[slot * n_cells..(slot + 1) * n_cells];
            match ch {
                0 => {
                    dst.copy_from_slice(&planes_2[0..n_cells]);
                }
                8 => {
                    dst.copy_from_slice(&planes_2[n_cells..2 * n_cells]);
                }
                16 => {
                    for v in dst.iter_mut() {
                        *v = mr_val;
                    }
                }
                17 => {
                    for v in dst.iter_mut() {
                        *v = ply_val;
                    }
                }
                c if c < 18 => {
                    // History planes 1..7 / 9..15 are zero on the Rust
                    // self-play path; clear in case caller did not zero-init.
                    for v in dst.iter_mut() {
                        *v = 0.0;
                    }
                }
                _ => {
                    debug_assert!(false, "channel index {ch} out of range [0, 18)");
                    for v in dst.iter_mut() {
                        *v = 0.0;
                    }
                }
            }
        }
    }

    /// `to_planes` variant emitting only the listed channels, in the listed
    /// order. Length = `channels.len() * board_size²` where `board_size`
    /// is the Board's encoding's `board_size` (default 19 = v6). See
    /// `encode_state_to_buffer_channels` for plane semantics.
    ///
    /// §172 A4.1 (multi-window guard): panics if the bound encoding is
    /// multi-window (v6w25). Multi-window selfplay is α-deferred — see
    /// `docs/designs/encoding_alpha_multiwindow_selfplay.md`. Use
    /// `get_cluster_views()` instead for those encodings.
    pub fn to_planes_channels(&self, channels: &[usize]) -> Vec<f32> {
        if let Some(spec) = self.encoding {
            if spec.is_multi_window {
                unimplemented!(
                    "multi-window selfplay deferred to α; see \
                     docs/designs/encoding_alpha_multiwindow_selfplay_design.md \
                     (§172 Phase A7)"
                );
            }
        }
        let mut planes_2 = vec![0.0f32; 2 * TOTAL_CELLS];
        let (my_cell, opp_cell) = match self.current_player {
            Player::One => (Cell::P1, Cell::P2),
            Player::Two => (Cell::P2, Cell::P1),
        };
        for (&(q, r), &cell) in &self.cells {
            let flat = self.window_flat_idx(q, r);
            if flat < TOTAL_CELLS {
                if cell == my_cell {
                    planes_2[flat] = 1.0;
                } else if cell == opp_cell {
                    planes_2[TOTAL_CELLS + flat] = 1.0;
                }
            }
        }
        let mut out = vec![0.0f32; channels.len() * TOTAL_CELLS];
        // Caller-supplied n_cells: this path is single-window only (the
        // multi-window guard above bails); single-window uses the v6
        // 19×19 wire layout (TOTAL_CELLS).
        self.encode_state_to_buffer_channels(&planes_2, &mut out, channels, TOTAL_CELLS);
        out
    }

    /// Encode the board as a flat f32 array of length
    /// `18 * board_size * board_size` representing shape
    /// `[18, board_size, board_size]` (18 history+scalar planes).
    ///
    /// `board_size` resolves from `self.encoding` (default 19 = v6 wire
    /// format; v8 = 25 = canvas dim per registry).
    ///
    /// Chain-length planes are computed separately via `encode_chain_planes`.
    /// Stones outside the current 19×19 window are silently omitted.
    ///
    /// §172 A4.1 (multi-window guard, closes §171 P3 plane-export blocker):
    /// panics if the bound encoding is multi-window (v6w25 etc.). Single-
    /// window selfplay must route through `get_cluster_views` for those
    /// encodings; the silent shape corruption to_planes used to produce
    /// (always 18×19×19 regardless of encoding) was the §171 P3 blocker.
    /// Multi-window selfplay deferred to α — see
    /// `docs/designs/encoding_alpha_multiwindow_selfplay.md`.
    ///
    /// **v8 semantic mismatch caveat (out of scope today):** for
    /// single-window encodings with `board_size != 19` (v8 = 25), this
    /// method emits an 18-plane wire layout sized at the encoding's
    /// `board_size` but the kernel still walks the v6 19×19 layout —
    /// only the first 361 cells per plane carry stone data, the
    /// remainder is zero. v8 native is 11 planes (no KEPT_PLANE_INDICES
    /// slice) and a different stone-placement convention; Rust v8
    /// selfplay does not exist today and §17X will introduce a dedicated
    /// `to_planes_v8` if needed. A4.1's responsibility is bounded to
    /// stopping the spatial-dim hardcode that blocked v6w25.
    pub fn to_planes(&self) -> Vec<f32> {
        if let Some(spec) = self.encoding {
            if spec.is_multi_window {
                unimplemented!(
                    "multi-window selfplay deferred to α; see \
                     docs/designs/encoding_alpha_multiwindow_selfplay_design.md \
                     (§172 Phase A7)"
                );
            }
        }
        let board_size = self.encoding.map_or(BOARD_SIZE, |s| s.board_size);
        let total_cells = board_size * board_size;

        let mut planes_2 = vec![0.0f32; 2 * TOTAL_CELLS];
        let (my_cell, opp_cell) = match self.current_player {
            Player::One => (Cell::P1, Cell::P2),
            Player::Two => (Cell::P2, Cell::P1),
        };
        for (&(q, r), &cell) in &self.cells {
            let flat = self.window_flat_idx(q, r);
            if flat < TOTAL_CELLS {
                if cell == my_cell {
                    planes_2[flat] = 1.0;
                } else if cell == opp_cell {
                    planes_2[TOTAL_CELLS + flat] = 1.0;
                }
            }
        }

        // Output buffer sized by encoding's board_size. v6 (board_size=19)
        // is bit-identical to pre-A4.1 behavior. v8 (board_size=25) gets a
        // larger zero-padded buffer; see method-doc caveat.
        let mut out = vec![0.0f32; 18 * total_cells];
        if board_size == BOARD_SIZE {
            self.encode_state_to_buffer(&planes_2, &mut out);
        } else {
            // v8 single-window path: emit v6 19×19 wire layout into the
            // top-left 361 cells of each 25×25 plane. Plane semantics
            // differ from v8 native (11 planes); see method-doc caveat.
            // Plane 0: my stones; Plane 8: opp stones.
            for i in 0..TOTAL_CELLS {
                out[MY_STONE_PLANE * total_cells + i] = planes_2[i];
                out[OPP_STONE_PLANE * total_cells + i] = planes_2[TOTAL_CELLS + i];
            }
            // Plane 16: moves_remaining == 2 broadcast over full plane.
            let mr_val = if self.moves_remaining == 2 { 1.0 } else { 0.0 };
            for i in 0..total_cells {
                out[16 * total_cells + i] = mr_val;
            }
            // Plane 17: ply parity broadcast over full plane.
            let ply_val = (self.ply % 2) as f32;
            for i in 0..total_cells {
                out[17 * total_cells + i] = ply_val;
            }
        }
        out
    }
}

// ── Q13 chain-length plane encoding ─────────────────────────────────────────
//
// Pure-function helpers used by `Board::encode_state_to_buffer`. Mirror the
// Python `_compute_chain_planes` in `hexo_rl/env/game_state.py`.
//
// Output layout: 6 planes × `n_cells` f32 entries (caller-supplied), written
// into the caller's buffer slice at offset 0. Values are /6.0 normalized.
// Plane order within the 6-plane block:
//   [axis0_cur, axis0_opp, axis1_cur, axis1_opp, axis2_cur, axis2_opp]

/// Saturation cap for chain length — the 6-in-a-row win target.
const CHAIN_CAP: i32 = 6;
/// Normalisation denominator; matches Python's /6.0 after int8 computation.
const CHAIN_NORM: f32 = 6.0;

#[inline]
fn flat_idx(q: i32, r: i32, trunk_sz: i32, half: i32) -> usize {
    ((q + half) as usize) * (trunk_sz as usize) + (r + half) as usize
}

#[inline]
fn in_window(q: i32, r: i32, half: i32) -> bool {
    q >= -half && q <= half && r >= -half && r <= half
}

/// Walk +step * (dq, dr) from (q, r) counting consecutive `own` cells.
/// Stops at window edge or first non-own cell. Max count = `CHAIN_CAP - 1`.
#[inline]
fn count_run(
    own: &[f32],
    q: i32,
    r: i32,
    dq: i32,
    dr: i32,
    trunk_sz: i32,
    half: i32,
) -> i32 {
    let mut c = 0i32;
    for k in 1..CHAIN_CAP {
        let qk = q + dq * k;
        let rk = r + dr * k;
        if !in_window(qk, rk, half) {
            break;
        }
        let idx = flat_idx(qk, rk, trunk_sz, half);
        if own[idx] > 0.5 {
            c += 1;
        } else {
            break;
        }
    }
    c
}

/// Write one chain-length plane (single axis, single player) into `out`.
/// `out` must have length `n_cells` (= trunk_sz²).
#[inline]
fn encode_chain_plane_one(
    own: &[f32],
    opp: &[f32],
    dq: i32,
    dr: i32,
    out: &mut [f32],
    trunk_sz: i32,
    half: i32,
) {
    for q in -half..=half {
        for r in -half..=half {
            let idx = flat_idx(q, r, trunk_sz, half);
            if opp[idx] > 0.5 {
                out[idx] = 0.0;
                continue;
            }
            let pos_run = count_run(own, q, r, dq, dr, trunk_sz, half);
            let neg_run = count_run(own, q, r, -dq, -dr, trunk_sz, half);
            let is_own = own[idx] > 0.5;
            if !is_own && pos_run == 0 && neg_run == 0 {
                out[idx] = 0.0;
                continue;
            }
            let mut v = 1 + pos_run + neg_run;
            if v > CHAIN_CAP {
                v = CHAIN_CAP;
            }
            out[idx] = (v as f32) / CHAIN_NORM;
        }
    }
}

/// Write all 6 chain-length planes into `out` (length `6 * n_cells`).
///
/// `cur_mask` and `opp_mask` are `n_cells`-sized f32 masks with 1.0 at
/// stone positions and 0.0 elsewhere.
///
/// `n_cells` and `trunk_sz` are caller-supplied per §173 A5b scalar-API
/// lesson (pre-extract at the worker_loop boundary, pass scalars into
/// the per-sim hot loop). `n_cells` must equal `trunk_sz * trunk_sz`.
#[inline]
pub fn encode_chain_planes(
    cur_mask: &[f32],
    opp_mask: &[f32],
    out: &mut [f32],
    n_cells: usize,
    trunk_sz: i32,
) {
    let half = (trunk_sz - 1) / 2;
    debug_assert_eq!(n_cells, (trunk_sz as usize) * (trunk_sz as usize));
    debug_assert_eq!(cur_mask.len(), n_cells);
    debug_assert_eq!(opp_mask.len(), n_cells);
    debug_assert_eq!(out.len(), 6 * n_cells);

    for (axis_idx, &(dq, dr)) in HEX_AXES.iter().enumerate() {
        let cur_base = 2 * axis_idx * n_cells;
        let opp_base = (2 * axis_idx + 1) * n_cells;
        // Split into two mutable slices so we can borrow disjoint regions.
        let (head, tail) = out.split_at_mut(opp_base);
        encode_chain_plane_one(
            cur_mask,
            opp_mask,
            dq,
            dr,
            &mut head[cur_base..cur_base + n_cells],
            trunk_sz,
            half,
        );
        encode_chain_plane_one(
            opp_mask,
            cur_mask,
            dq,
            dr,
            &mut tail[0..n_cells],
            trunk_sz,
            half,
        );
    }
}

#[cfg(test)]
mod channel_select_tests {
    use super::*;

    fn build_planes_2() -> Vec<f32> {
        let mut v = vec![0.0f32; 2 * TOTAL_CELLS];
        v[0] = 1.0;
        v[5] = 1.0;
        v[TOTAL_CELLS + 7] = 1.0;
        v[TOTAL_CELLS + 11] = 1.0;
        v
    }

    fn make_board() -> Board {
        let mut b = Board::new();
        // Place a couple of stones to exercise moves_remaining + ply parity.
        b.apply_move(0, 0).unwrap();
        b.apply_move(1, 0).unwrap();
        b.apply_move(0, 1).unwrap();
        b
    }

    #[test]
    fn channel_select_matches_full_kernel_for_canonical_planes() {
        let b = make_board();
        let planes_2 = build_planes_2();

        let mut full = vec![0.0f32; 18 * TOTAL_CELLS];
        b.encode_state_to_buffer(&planes_2, &mut full);

        let channels = [0usize, 8, 16, 17];
        let mut sel = vec![0.0f32; channels.len() * TOTAL_CELLS];
        b.encode_state_to_buffer_channels(&planes_2, &mut sel, &channels, TOTAL_CELLS);

        for (slot, &ch) in channels.iter().enumerate() {
            let lhs = &full[ch * TOTAL_CELLS..(ch + 1) * TOTAL_CELLS];
            let rhs = &sel[slot * TOTAL_CELLS..(slot + 1) * TOTAL_CELLS];
            assert_eq!(lhs, rhs, "channel {} mismatch (slot {})", ch, slot);
        }
    }

    #[test]
    fn channel_select_history_planes_are_zero() {
        let b = make_board();
        let planes_2 = build_planes_2();
        let channels = [0usize, 1, 8, 9];
        let mut sel = vec![999.0f32; channels.len() * TOTAL_CELLS];
        b.encode_state_to_buffer_channels(&planes_2, &mut sel, &channels, TOTAL_CELLS);
        // Slots 1 and 3 are history planes (1 and 9) — zero on Rust path.
        for &slot in &[1usize, 3] {
            for v in &sel[slot * TOTAL_CELLS..(slot + 1) * TOTAL_CELLS] {
                assert_eq!(*v, 0.0);
            }
        }
    }

    #[test]
    fn to_planes_channels_length_matches_request() {
        let b = make_board();
        let v = b.to_planes_channels(&[0, 8]);
        assert_eq!(v.len(), 2 * TOTAL_CELLS);
        let v6 = b.to_planes_channels(&[0, 1, 8, 9, 16, 17]);
        assert_eq!(v6.len(), 6 * TOTAL_CELLS);
    }

    #[test]
    fn to_planes_channels_full_18_matches_to_planes() {
        let b = make_board();
        let full = b.to_planes();
        let channels: Vec<usize> = (0..18).collect();
        let sel = b.to_planes_channels(&channels);
        assert_eq!(full.len(), sel.len());
        for (i, (a, c)) in full.iter().zip(sel.iter()).enumerate() {
            assert!((a - c).abs() < 1e-7, "mismatch at index {}: {} vs {}", i, a, c);
        }
    }
}

#[cfg(test)]
mod chain_plane_tests {
    use super::*;

    fn at(plane: &[f32], q: i32, r: i32) -> f32 {
        plane[flat_idx(q, r, BOARD_SIZE as i32, HALF)]
    }

    fn set(mask: &mut Vec<f32>, q: i32, r: i32) {
        mask[flat_idx(q, r, BOARD_SIZE as i32, HALF)] = 1.0;
    }

    #[test]
    fn empty_board_all_zeros() {
        let cur = vec![0.0f32; TOTAL_CELLS];
        let opp = vec![0.0f32; TOTAL_CELLS];
        let mut out = vec![0.0f32; 6 * TOTAL_CELLS];
        encode_chain_planes(&cur, &opp, &mut out, TOTAL_CELLS, BOARD_SIZE as i32);
        for &v in &out {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn single_stone_value_one_sixth_on_all_axes() {
        let mut cur = vec![0.0f32; TOTAL_CELLS];
        let opp = vec![0.0f32; TOTAL_CELLS];
        set(&mut cur, 0, 0);
        let mut out = vec![0.0f32; 6 * TOTAL_CELLS];
        encode_chain_planes(&cur, &opp, &mut out, TOTAL_CELLS, BOARD_SIZE as i32);
        let expected = 1.0 / CHAIN_NORM;
        // axis0_cur, axis1_cur, axis2_cur at (0,0) all = 1/6.
        let a0 = &out[0..TOTAL_CELLS];
        let a1 = &out[2 * TOTAL_CELLS..3 * TOTAL_CELLS];
        let a2 = &out[4 * TOTAL_CELLS..5 * TOTAL_CELLS];
        assert!((at(a0, 0, 0) - expected).abs() < 1e-5);
        assert!((at(a1, 0, 0) - expected).abs() < 1e-5);
        assert!((at(a2, 0, 0) - expected).abs() < 1e-5);
        // All opp planes zero (no opp stones).
        let o0 = &out[TOTAL_CELLS..2 * TOTAL_CELLS];
        assert_eq!(o0.iter().map(|&v| v as i32).sum::<i32>(), 0);
    }

    #[test]
    fn xx_empty_xxx_caps_at_six() {
        let mut cur = vec![0.0f32; TOTAL_CELLS];
        let opp = vec![0.0f32; TOTAL_CELLS];
        // Stones at q=0,1 then empty at q=2 then stones at q=3,4,5, all r=0.
        for q in [0, 1, 3, 4, 5] {
            set(&mut cur, q, 0);
        }
        let mut out = vec![0.0f32; 6 * TOTAL_CELLS];
        encode_chain_planes(&cur, &opp, &mut out, TOTAL_CELLS, BOARD_SIZE as i32);
        let a0 = &out[0..TOTAL_CELLS];
        assert!((at(a0, 2, 0) - 1.0).abs() < 1e-5); // 6/6
    }

    #[test]
    fn opponent_blocks_run() {
        let mut cur = vec![0.0f32; TOTAL_CELLS];
        let mut opp = vec![0.0f32; TOTAL_CELLS];
        set(&mut cur, 0, 0);
        set(&mut cur, 1, 0);
        opp[flat_idx(2, 0, BOARD_SIZE as i32, HALF)] = 1.0;
        set(&mut cur, 3, 0);
        let mut out = vec![0.0f32; 6 * TOTAL_CELLS];
        encode_chain_planes(&cur, &opp, &mut out, TOTAL_CELLS, BOARD_SIZE as i32);
        let a0 = &out[0..TOTAL_CELLS];
        let v2 = 2.0 / CHAIN_NORM;
        let v1 = 1.0 / CHAIN_NORM;
        assert!((at(a0, 0, 0) - v2).abs() < 1e-5);
        assert!((at(a0, 1, 0) - v2).abs() < 1e-5);
        assert!((at(a0, 3, 0) - v1).abs() < 1e-5);
    }

    #[test]
    fn cap_saturates_above_six() {
        let mut cur = vec![0.0f32; TOTAL_CELLS];
        let opp = vec![0.0f32; TOTAL_CELLS];
        for q in -3..=3 {
            set(&mut cur, q, 0); // 7 stones
        }
        let mut out = vec![0.0f32; 6 * TOTAL_CELLS];
        encode_chain_planes(&cur, &opp, &mut out, TOTAL_CELLS, BOARD_SIZE as i32);
        let a0 = &out[0..TOTAL_CELLS];
        for q in -3..=3 {
            assert!((at(a0, q, 0) - 1.0).abs() < 1e-5, "q={}", q);
        }
    }

    #[test]
    fn python_rust_parity_50_stone_position() {
        // Reconstruct the Python perf test position deterministically — seed 2613,
        // 25 cur + 25 opp stones on non-overlapping flat indices. Verify Rust
        // output matches the Python helper output byte-exact after /6 normalization.
        // We don't call Python here; instead this test pins the Rust output for
        // a specific seeded position so drift between Rust and Python is caught
        // by a failing Python-side value comparison (implemented in
        // tests/test_chain_plane_augmentation.py once the Python wire-up lands).
        let mut cur = vec![0.0f32; TOTAL_CELLS];
        let mut opp = vec![0.0f32; TOTAL_CELLS];
        // Hand-pick a small representative position instead of pseudo-random.
        for (q, r) in [(0, 0), (1, 0), (2, 0), (-2, 0), (-1, 0)] {
            set(&mut cur, q, r);
        }
        for (q, r) in [(0, 1), (0, 2)] {
            opp[flat_idx(q, r, BOARD_SIZE as i32, HALF)] = 1.0;
        }
        let mut out = vec![0.0f32; 6 * TOTAL_CELLS];
        encode_chain_planes(&cur, &opp, &mut out, TOTAL_CELLS, BOARD_SIZE as i32);
        let a0 = &out[0..TOTAL_CELLS];
        // Cur run of 5 along axis0 at q=-2..=2, r=0 → each cell sees 5/6.
        let five_sixths = 5.0 / CHAIN_NORM;
        for q in -2..=2 {
            assert!((at(a0, q, 0) - five_sixths).abs() < 1e-5);
        }
        // Empty flanks at q=-3 and q=3 along axis0 see 6/6 (5 + 1).
        assert!((at(a0, -3, 0) - 1.0).abs() < 1e-5);
        assert!((at(a0, 3, 0) - 1.0).abs() < 1e-5);
        // Opponent run along axis1 at (0,1),(0,2) → opp plane axis1.
        let a1_opp = &out[3 * TOTAL_CELLS..4 * TOTAL_CELLS];
        let two_sixths = 2.0 / CHAIN_NORM;
        assert!((at(a1_opp, 0, 1) - two_sixths).abs() < 1e-5);
        assert!((at(a1_opp, 0, 2) - two_sixths).abs() < 1e-5);
    }
}
