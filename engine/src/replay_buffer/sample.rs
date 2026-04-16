//! Read path for `ReplayBuffer` — public `sample_batch_impl` entry plus the
//! internal kernels that back it.
//!
//! Contains (1) the PyO3-facing `sample_batch_impl` that allocates output
//! numpy arrays and dispatches per-sample scatter, and (2) the pure-Rust
//! kernels — weight bucketing, rejection-based weighted index sampling, the
//! correlation-guard dedup loop, and the 12-fold hex symmetry scatter
//! (`apply_sym`). These were previously split across `mod.rs` and a separate
//! `sampling.rs`; keeping them in one file avoids a public-entry/private-kernel
//! naming trap now that `sample_batch` itself has moved out of `mod.rs`.

use std::collections::HashSet;

use half::f16;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyArray4, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::RngExt;

use super::sym_tables::*;
use super::ReplayBuffer;

/// Apply symmetry `sym_idx` to one 18-plane state tensor (pure coord scatter).
///
/// Generic over the element type `T: Copy` — callable with `f32` for the
/// Python-facing bindings and with `u16` (f16 bits) for the internal buffer
/// sampling path. Pure scatter; caller zeroes `dst` before invocation.
///
/// All 18 planes scatter by pure coordinate permutation (identity plane mapping).
#[inline]
pub fn apply_symmetry_state<T: Copy>(
    src: &[T],
    dst: &mut [T],
    sym_idx: usize,
    sym_tables: &SymTables,
) {
    debug_assert_eq!(src.len(), N_PLANES * N_CELLS);
    debug_assert_eq!(dst.len(), N_PLANES * N_CELLS);
    debug_assert!(sym_idx < N_SYMS);

    let scatter          = &sym_tables.scatter[sym_idx];
    let src_plane_lookup = &sym_tables.src_plane_lookup[sym_idx];

    // All 18 planes: src_plane_lookup is identity, so src_p == dst_p.
    for dst_p in 0..N_PLANES {
        let src_p    = src_plane_lookup[dst_p];
        let src_base = src_p * N_CELLS;
        let dst_base = dst_p * N_CELLS;
        let src_plane = &src[src_base..src_base + N_CELLS];
        let dst_plane = &mut dst[dst_base..dst_base + N_CELLS];
        for &(sc, dc) in scatter {
            dst_plane[dc as usize] = src_plane[sc as usize];
        }
    }
}

/// Apply symmetry `sym_idx` to one 6-plane chain-length tensor.
///
/// Generic over `T: Copy`. Uses `chain_src_lookup` for axis-plane remap plus
/// coordinate permutation. Caller zeroes `dst` before invocation.
#[inline]
pub fn apply_chain_symmetry<T: Copy>(
    src: &[T],
    dst: &mut [T],
    sym_idx: usize,
    sym_tables: &SymTables,
) {
    debug_assert_eq!(src.len(), N_CHAIN_PLANES * N_CELLS);
    debug_assert_eq!(dst.len(), N_CHAIN_PLANES * N_CELLS);
    debug_assert!(sym_idx < N_SYMS);

    let scatter           = &sym_tables.scatter[sym_idx];
    let chain_src_lookup  = &sym_tables.chain_src_lookup[sym_idx];

    for dst_p in 0..N_CHAIN_PLANES {
        let src_p    = chain_src_lookup[dst_p];
        let src_base = src_p * N_CELLS;
        let dst_base = dst_p * N_CELLS;
        let src_plane = &src[src_base..src_base + N_CELLS];
        let dst_plane = &mut dst[dst_base..dst_base + N_CELLS];
        for &(sc, dc) in scatter {
            dst_plane[dc as usize] = src_plane[sc as usize];
        }
    }
}

impl ReplayBuffer {
    /// Map an f16 weight (stored as bits) to a histogram bucket index.
    ///
    /// Bucket 0: weight < 0.30  (short-game tier, ~0.15)
    /// Bucket 1: 0.30 ≤ w < 0.75 (medium-game tier, ~0.50)
    /// Bucket 2: weight ≥ 0.75  (full-weight tier, ~1.0)
    #[inline]
    pub(crate) fn weight_bucket(w_bits: u16) -> usize {
        let w = f16::from_bits(w_bits).to_f32();
        if w < 0.30 { 0 }
        else if w < 0.75 { 1 }
        else { 2 }
    }

    /// Sample a single index using rejection sampling on stored weights.
    ///
    /// Rejection: draw uniform index, accept with probability weight/1.0.
    /// Max 32 attempts; falls back to accepting unconditionally to bound latency.
    /// No heap allocations.
    #[inline]
    pub(crate) fn weighted_sample_one(&mut self) -> usize {
        const MAX_REJECT: usize = 32;
        for _ in 0..MAX_REJECT {
            let idx = self.rng.random_range(0..self.size);
            let w = f16::from_bits(self.weights[idx]).to_f32();
            if w >= 1.0 || self.rng.random::<f32>() < w {
                return idx;
            }
        }
        // Fallback: accept whatever we drew last to avoid infinite loop.
        self.rng.random_range(0..self.size)
    }

    /// Sample `batch_size` slot indices, optionally deduplicating by game_id.
    /// Uses per-position weight-based rejection sampling.
    pub(crate) fn sample_indices(&mut self, batch_size: usize, use_dedup: bool) -> Vec<usize> {
        if !use_dedup {
            return (0..batch_size)
                .map(|_| self.weighted_sample_one())
                .collect();
        }

        // Correlation guard: at most one cluster per board position per batch.
        // Strategy: sample candidates, replace duplicates up to MAX_RETRIES times.
        const MAX_RETRIES: usize = 8;

        let mut indices: Vec<usize> = (0..batch_size)
            .map(|_| self.weighted_sample_one())
            .collect();

        for _ in 0..MAX_RETRIES {
            let mut seen: HashSet<i64> = HashSet::with_capacity(batch_size);
            let mut all_unique = true;
            for idx in indices.iter_mut() {
                let gid = self.game_ids[*idx];
                if gid == -1 || seen.insert(gid) {
                    // OK: untagged or first occurrence of this game_id.
                    continue;
                }
                // Duplicate — resample this slot.
                all_unique = false;
                let mut candidate = self.weighted_sample_one();
                // Try a few times to find a unique one; give up gracefully.
                for _ in 0..16 {
                    let cgid = self.game_ids[candidate];
                    if cgid == -1 || !seen.contains(&cgid) {
                        break;
                    }
                    candidate = self.weighted_sample_one();
                }
                *idx = candidate;
                // Only real game_ids (not the -1 untagged sentinel) should
                // enter the dedup set. Inserting -1 here poisoned `seen` so
                // every subsequent untagged position collided against it and
                // burned its inner 16-retry budget for nothing.
                let cgid = self.game_ids[candidate];
                if cgid != -1 {
                    seen.insert(cgid);
                }
            }
            if all_unique { break; }
        }

        indices
    }

    /// Apply symmetry `sym_idx` to one (state, chain, policy, ownership, winning_line) sample.
    ///
    /// Scatter-copies from `src_*` into `dst_*`. Cells that have no valid destination
    /// under the transform remain at the caller-zeroed default. Aux planes
    /// (`ownership`, `winning_line`) reuse the same scatter table as state.
    ///
    /// State (18 planes): pure coordinate scatter via `apply_symmetry_state`.
    /// Chain (6 planes): coordinate scatter + axis-plane remap via `apply_chain_symmetry`.
    #[inline]
    pub(crate) fn apply_sym(
        sym_idx:      usize,
        src_state:    &[u16],     // f16 bits, length N_PLANES × N_CELLS
        src_chain:    &[u16],     // f16 bits, length N_CHAIN_PLANES × N_CELLS
        src_policy:   &[f32],     // length N_ACTIONS
        src_own:      &[u8],      // length AUX_STRIDE (= N_CELLS)
        src_wl:       &[u8],      // length AUX_STRIDE
        dst_state:    &mut [u16], // f16 bits, length N_PLANES × N_CELLS  (zeroed by caller)
        dst_chain:    &mut [u16], // f16 bits, length N_CHAIN_PLANES × N_CELLS (zeroed by caller)
        dst_policy:   &mut [f32], // length N_ACTIONS                     (zeroed by caller)
        dst_own:      &mut [u8],  // length AUX_STRIDE                    (caller-initialised)
        dst_wl:       &mut [u8],  // length AUX_STRIDE                    (caller-initialised)
        tables:       &SymTables,
    ) {
        // State planes: 18-plane coord-scatter (identity plane mapping).
        apply_symmetry_state::<u16>(src_state, dst_state, sym_idx, tables);

        // Chain planes: coord-scatter + axis-plane remap.
        apply_chain_symmetry::<u16>(src_chain, dst_chain, sym_idx, tables);

        let scatter = &tables.scatter[sym_idx];

        // Policy + ownership + winning_line: all three scatter through the same
        // 361-cell hex permutation table. Fuse into one loop.
        for &(sc, dc) in scatter {
            let sc_u = sc as usize;
            let dc_u = dc as usize;
            dst_policy[dc_u] = src_policy[sc_u];
            dst_own   [dc_u] = src_own[sc_u];
            dst_wl    [dc_u] = src_wl[sc_u];
        }
        // Pass action (index 361) is always the identity (policy only).
        dst_policy[N_CELLS] = src_policy[N_CELLS];
    }

    /// Sample `batch_size` entries, optionally with random 12-fold hex augmentation.
    ///
    /// Applies the Multi-Window correlation guard: entries sharing the same `game_id`
    /// (i.e. different cluster windows of the same board position) are never placed
    /// in the same batch together.  Falls back to plain uniform sampling when all
    /// game_ids are −1 (e.g. data loaded without tagging).
    pub(crate) fn sample_batch_impl<'py>(
        &mut self,
        py:        Python<'py>,
        batch_size: usize,
        augment:    bool,
    ) -> PyResult<(
        Bound<'py, PyArray4<f16>>,
        Bound<'py, PyArray4<f16>>,
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray1<f32>>,
        Bound<'py, PyArray3<u8>>,
        Bound<'py, PyArray3<u8>>,
        Bound<'py, PyArray1<u8>>,
    )> {
        if self.size == 0 {
            return Err(PyValueError::new_err("Cannot sample from an empty replay buffer"));
        }

        // ── Index selection (with correlation guard) ──────────────────────────
        let use_dedup = self.game_ids[0] != -1;
        let indices   = self.sample_indices(batch_size, use_dedup);

        // ── Allocate output arrays (owned by Python after return) ─────────────
        // States and chain_planes as f16 bits (u16) — no type conversion during scatter.
        let mut out_states      = vec![0u16; batch_size * STATE_STRIDE];
        let mut out_chain       = vec![0u16; batch_size * CHAIN_STRIDE];
        let mut out_policies    = vec![0.0f32; batch_size * POLICY_STRIDE];
        let mut out_outcomes    = vec![0.0f32; batch_size];
        // Ownership default 1 = "empty" — cells outside the symmetry's destination
        // window stay at the same neutral value as the row's initial state.
        let mut out_ownership      = vec![1u8; batch_size * AUX_STRIDE];
        let mut out_winning_line   = vec![0u8; batch_size * AUX_STRIDE];
        // is_full_search is per-position metadata — no symmetry transform needed.
        let mut out_is_full_search = vec![0u8; batch_size];

        // ── Fill output ───────────────────────────────────────────────────────
        for (b, &idx) in indices.iter().enumerate() {
            let sym_idx = if augment { self.rng.random_range(0..N_SYMS) } else { 0 };

            let src_state  = &self.states      [idx * STATE_STRIDE..(idx + 1) * STATE_STRIDE];
            let src_chain  = &self.chain_planes [idx * CHAIN_STRIDE..(idx + 1) * CHAIN_STRIDE];
            let src_policy = &self.policies    [idx * POLICY_STRIDE..(idx + 1) * POLICY_STRIDE];
            let src_own    = &self.ownership   [idx * AUX_STRIDE   ..(idx + 1) * AUX_STRIDE];
            let src_wl     = &self.winning_line[idx * AUX_STRIDE   ..(idx + 1) * AUX_STRIDE];

            let dst_state  = &mut out_states      [b * STATE_STRIDE..(b + 1) * STATE_STRIDE];
            let dst_chain  = &mut out_chain        [b * CHAIN_STRIDE..(b + 1) * CHAIN_STRIDE];
            let dst_policy = &mut out_policies    [b * POLICY_STRIDE..(b + 1) * POLICY_STRIDE];
            let dst_own    = &mut out_ownership   [b * AUX_STRIDE   ..(b + 1) * AUX_STRIDE];
            let dst_wl     = &mut out_winning_line[b * AUX_STRIDE   ..(b + 1) * AUX_STRIDE];

            Self::apply_sym(
                sym_idx,
                src_state, src_chain, src_policy, src_own, src_wl,
                dst_state, dst_chain, dst_policy, dst_own, dst_wl,
                &self.sym_tables,
            );

            out_outcomes[b] = self.outcomes[idx];
            out_is_full_search[b] = self.is_full_search[idx];
        }

        // ── Transmute u16 Vecs to f16 Vecs and wrap as numpy arrays ───────────
        // Safety: f16 and u16 have the same size/alignment; every bit pattern is valid for u16,
        // and we only wrote bits that came from valid f16 values stored via push().
        let states_f16: Vec<f16> = unsafe {
            let mut v = std::mem::ManuallyDrop::new(out_states);
            Vec::from_raw_parts(v.as_mut_ptr() as *mut f16, v.len(), v.capacity())
        };
        let chain_f16: Vec<f16> = unsafe {
            let mut v = std::mem::ManuallyDrop::new(out_chain);
            Vec::from_raw_parts(v.as_mut_ptr() as *mut f16, v.len(), v.capacity())
        };

        let states_np = states_f16
            .into_pyarray(py)
            .reshape([batch_size, N_PLANES, BOARD_H, BOARD_W])?;
        let chain_np = chain_f16
            .into_pyarray(py)
            .reshape([batch_size, N_CHAIN_PLANES, BOARD_H, BOARD_W])?;
        let policies_np = out_policies
            .into_pyarray(py)
            .reshape([batch_size, N_ACTIONS])?;
        let outcomes_np = out_outcomes.into_pyarray(py);
        let ownership_np = out_ownership
            .into_pyarray(py)
            .reshape([batch_size, BOARD_H, BOARD_W])?;
        let winning_line_np = out_winning_line
            .into_pyarray(py)
            .reshape([batch_size, BOARD_H, BOARD_W])?;
        let is_full_search_np = out_is_full_search.into_pyarray(py);

        Ok((states_np, chain_np, policies_np, outcomes_np, ownership_np, winning_line_np, is_full_search_np))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use std::sync::atomic::AtomicU64;

    /// Push positions from short (5 moves), medium (15), and long (40) games,
    /// sample 10,000 times, and verify short-game positions appear ~0.15x as
    /// often as long-game positions.
    #[test]
    fn test_weighted_sampling_distribution() {
        let default_w = f16::from_f32(1.0).to_bits();
        let mut buf = ReplayBuffer {
            capacity: 300,
            size: 0,
            head: 0,
            states:          vec![0u16; 300 * STATE_STRIDE],
            chain_planes:    vec![0u16; 300 * CHAIN_STRIDE],
            policies:        vec![0.0f32; 300 * POLICY_STRIDE],
            outcomes:        vec![0.0f32; 300],
            game_ids:        vec![-1i64; 300],
            weights:         vec![default_w; 300],
            ownership:       vec![1u8; 300 * AUX_STRIDE],
            winning_line:    vec![0u8; 300 * AUX_STRIDE],
            is_full_search:  vec![1u8; 300],
            sym_tables: SymTables::new(),
            weight_schedule: WeightSchedule::uniform(),
            next_game_id: 0,
            rng: StdRng::seed_from_u64(42),
            weight_buckets: [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)],
        };

        // Configure schedule: <10 → 0.15, <25 → 0.50, ≥25 → 1.0
        buf.weight_schedule = WeightSchedule {
            brackets: vec![
                WeightBracket { max_moves: 10, weight: f16::from_f32(0.15).to_bits() },
                WeightBracket { max_moves: 25, weight: f16::from_f32(0.50).to_bits() },
            ],
            default_weight: f16::from_f32(1.0).to_bits(),
        };

        // Push 100 positions from each category with distinct outcomes for identification.
        // Short games (length 5) → outcome = 1.0
        for _ in 0..100 { buf.push_raw(1.0, 5); }
        // Medium games (length 15) → outcome = 2.0
        for _ in 0..100 { buf.push_raw(2.0, 15); }
        // Long games (length 40) → outcome = 3.0
        for _ in 0..100 { buf.push_raw(3.0, 40); }

        assert_eq!(buf.size, 300);

        // Sample 10,000 indices and count by category.
        let n_samples = 10_000;
        let mut count_short: usize = 0;
        let mut count_medium: usize = 0;
        let mut count_long: usize = 0;

        for _ in 0..n_samples {
            let idx = buf.weighted_sample_one();
            match buf.outcomes[idx] as u32 {
                1 => count_short += 1,
                2 => count_medium += 1,
                3 => count_long += 1,
                _ => panic!("unexpected outcome"),
            }
        }

        // With equal counts in buffer (100 each), expected sampling ratios:
        //   short : long  ≈ 0.15 : 1.0
        //   medium : long ≈ 0.50 : 1.0
        // Allow generous tolerance (factor of 2) for statistical noise.
        let ratio_short_long = count_short as f64 / count_long as f64;
        let ratio_medium_long = count_medium as f64 / count_long as f64;

        assert!(
            ratio_short_long < 0.30,
            "short/long ratio {:.3} should be < 0.30 (expected ~0.15)",
            ratio_short_long
        );
        assert!(
            ratio_short_long > 0.05,
            "short/long ratio {:.3} should be > 0.05 (expected ~0.15)",
            ratio_short_long
        );
        assert!(
            ratio_medium_long < 0.80,
            "medium/long ratio {:.3} should be < 0.80 (expected ~0.50)",
            ratio_medium_long
        );
        assert!(
            ratio_medium_long > 0.25,
            "medium/long ratio {:.3} should be > 0.25 (expected ~0.50)",
            ratio_medium_long
        );
    }

    /// Verify that weight_for returns the correct bracket.
    #[test]
    fn test_weight_schedule_lookup() {
        let schedule = WeightSchedule {
            brackets: vec![
                WeightBracket { max_moves: 10, weight: f16::from_f32(0.15).to_bits() },
                WeightBracket { max_moves: 25, weight: f16::from_f32(0.50).to_bits() },
            ],
            default_weight: f16::from_f32(1.0).to_bits(),
        };

        let w5  = f16::from_bits(schedule.weight_for(5)).to_f32();
        let w10 = f16::from_bits(schedule.weight_for(10)).to_f32();
        let w15 = f16::from_bits(schedule.weight_for(15)).to_f32();
        let w25 = f16::from_bits(schedule.weight_for(25)).to_f32();
        let w40 = f16::from_bits(schedule.weight_for(40)).to_f32();

        assert!((w5 - 0.15).abs() < 0.01, "game_length=5 → {w5}");
        assert!((w10 - 0.50).abs() < 0.01, "game_length=10 → {w10}");
        assert!((w15 - 0.50).abs() < 0.01, "game_length=15 → {w15}");
        assert!((w25 - 1.0).abs() < 0.01, "game_length=25 → {w25}");
        assert!((w40 - 1.0).abs() < 0.01, "game_length=40 → {w40}");
    }

    /// Uniform schedule (default) must accept all positions equally.
    #[test]
    fn test_uniform_schedule_all_weight_one() {
        let schedule = WeightSchedule::uniform();
        let w = f16::from_bits(schedule.weight_for(5)).to_f32();
        assert!((w - 1.0).abs() < 0.01);
        let w = f16::from_bits(schedule.weight_for(100)).to_f32();
        assert!((w - 1.0).abs() < 0.01);
    }

    /// Augmentation equivariance — for each of the 12 hex symmetries, the
    /// same scatter table must apply to state, policy, ownership, and
    /// winning_line. We plant a single marker in each plane at the same source
    /// cell and assert all four destination indices agree (or are all
    /// out-of-window for that symmetry).
    #[test]
    fn test_aux_augment_equivariance() {
        let tables = SymTables::new();

        // Test multiple source cells to exercise both in-window and
        // edge-mapping regimes across all 12 symmetries.
        for &marker_src in &[0usize, 200, 180, 360] {
            for sym_idx in 0..N_SYMS {
                let mut src_state = vec![0u16; N_PLANES * N_CELLS];
                let mut src_chain = vec![0u16; N_CHAIN_PLANES * N_CELLS];
                let mut src_pol   = vec![0.0f32; N_ACTIONS];
                let mut src_own   = vec![1u8; AUX_STRIDE];
                let mut src_wl    = vec![0u8; AUX_STRIDE];

                src_state[marker_src] = f16::from_f32(7.0).to_bits();
                src_pol[marker_src]   = 7.0;
                src_own[marker_src]   = 2;   // P1
                src_wl[marker_src]    = 1;

                let mut dst_state = vec![0u16; N_PLANES * N_CELLS];
                let mut dst_chain = vec![0u16; N_CHAIN_PLANES * N_CELLS];
                let mut dst_pol   = vec![0.0f32; N_ACTIONS];
                let mut dst_own   = vec![1u8; AUX_STRIDE];
                let mut dst_wl    = vec![0u8; AUX_STRIDE];

                ReplayBuffer::apply_sym(
                    sym_idx,
                    &src_state, &src_chain, &src_pol, &src_own, &src_wl,
                    &mut dst_state, &mut dst_chain, &mut dst_pol, &mut dst_own, &mut dst_wl,
                    &tables,
                );

                let dst_state_idx = (0..N_CELLS).find(|&i| dst_state[i] != 0);
                let dst_pol_idx   = (0..N_CELLS).find(|&i| dst_pol[i]   != 0.0);
                let dst_own_idx   = (0..AUX_STRIDE).find(|&i| dst_own[i] == 2);
                let dst_wl_idx    = (0..AUX_STRIDE).find(|&i| dst_wl[i]  == 1);

                assert_eq!(
                    dst_state_idx, dst_pol_idx,
                    "sym {sym_idx} src {marker_src}: state vs policy mismatch"
                );
                assert_eq!(
                    dst_state_idx, dst_own_idx,
                    "sym {sym_idx} src {marker_src}: state vs ownership mismatch"
                );
                assert_eq!(
                    dst_state_idx, dst_wl_idx,
                    "sym {sym_idx} src {marker_src}: state vs winning_line mismatch"
                );
            }
        }
    }
}
