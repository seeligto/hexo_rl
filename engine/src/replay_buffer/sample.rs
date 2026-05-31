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
use numpy::{IntoPyArray, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::RngExt;

use super::sym_tables::{N_CHAIN_PLANES, N_SYMS, SymTables};
#[cfg(test)]
use super::sym_tables::{
    AUX_STRIDE, CHAIN_STRIDE, N_ACTIONS, N_CELLS, N_PLANES,
    POLICY_STRIDE, STATE_STRIDE, WeightBracket, WeightSchedule,
};
use super::ReplayBuffer;

/// Apply symmetry `sym_idx` to a state tensor (pure coord scatter).
///
/// Plane-count-generic: deduces `n_planes = src.len() / sym_tables.n_cells`.
/// Identical scatter is applied to every plane (state planes do not permute
/// under any hex dihedral symmetry — only cell coordinates do). Callable on
/// the 8-plane buffer wire format (HEXB v6), the 18-plane legacy inference /
/// corpus tensor, and the 11-plane v8 wire format with the same code path
/// (the cell count is read from `sym_tables.n_cells`, not a global constant,
/// so v6 and v8 SymTables instances dispatch to the same kernel).
///
/// Generic over the element type `T: Copy` — callable with `f32` for the
/// Python-facing bindings and with `u16` (f16 bits) for the internal buffer
/// sampling path. Pure scatter; caller zeroes `dst` before invocation.
#[inline]
pub fn apply_symmetry_state<T: Copy>(
    src: &[T],
    dst: &mut [T],
    sym_idx: usize,
    sym_tables: &SymTables,
) {
    debug_assert_eq!(src.len(), dst.len());
    debug_assert!(sym_idx < N_SYMS);
    let n_cells = sym_tables.n_cells;
    debug_assert_eq!(src.len() % n_cells, 0,
        "state tensor length {} not a multiple of {} cells", src.len(), n_cells);
    let n_planes = src.len() / n_cells;

    let scatter = &sym_tables.scatter[sym_idx];
    for p in 0..n_planes {
        let base = p * n_cells;
        let src_plane = &src[base..base + n_cells];
        let dst_plane = &mut dst[base..base + n_cells];
        for &(sc, dc) in scatter {
            dst_plane[dc as usize] = src_plane[sc as usize];
        }
    }
}

/// Apply symmetry `sym_idx` to one 6-plane chain-length tensor.
///
/// Generic over `T: Copy`. Uses `chain_src_lookup` for axis-plane remap plus
/// coordinate permutation. Cell count is read from `sym_tables.n_cells`
/// (v6: 361, v8: 625). Caller zeroes `dst` before invocation.
#[inline]
pub fn apply_chain_symmetry<T: Copy>(
    src: &[T],
    dst: &mut [T],
    sym_idx: usize,
    sym_tables: &SymTables,
) {
    let n_cells = sym_tables.n_cells;
    debug_assert_eq!(src.len(), N_CHAIN_PLANES * n_cells);
    debug_assert_eq!(dst.len(), N_CHAIN_PLANES * n_cells);
    debug_assert!(sym_idx < N_SYMS);

    let scatter           = &sym_tables.scatter[sym_idx];
    let chain_src_lookup  = &sym_tables.chain_src_lookup[sym_idx];

    for dst_p in 0..N_CHAIN_PLANES {
        let src_p    = chain_src_lookup[dst_p];
        let src_base = src_p * n_cells;
        let dst_base = dst_p * n_cells;
        let src_plane = &src[src_base..src_base + n_cells];
        let dst_plane = &mut dst[dst_base..dst_base + n_cells];
        for &(sc, dc) in scatter {
            dst_plane[dc as usize] = src_plane[sc as usize];
        }
    }
}

/// Source slice bundle for `ReplayBuffer::apply_sym` (read-only views).
pub(crate) struct ApplySymSrc<'a> {
    pub state:  &'a [u16],
    pub chain:  &'a [u16],
    pub policy: &'a [f32],
    pub own:    &'a [u8],
    pub wl:     &'a [u8],
}

/// Destination slice bundle for `ReplayBuffer::apply_sym` (mutable views).
pub(crate) struct ApplySymDst<'a> {
    pub state:  &'a mut [u16],
    pub chain:  &'a mut [u16],
    pub policy: &'a mut [f32],
    pub own:    &'a mut [u8],
    pub wl:     &'a mut [u8],
}

/// Combined slice + sym-tables bundle for `ReplayBuffer::apply_sym`.
pub(crate) struct ApplySymSlices<'a> {
    pub src:    ApplySymSrc<'a>,
    pub dst:    ApplySymDst<'a>,
    pub tables: &'a SymTables,
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

        // §P45: HashSet allocation hoisted outside the retry loop; we
        // `.clear()` at iteration top so each retry sees a fresh empty set
        // (bit-equivalent to the previous per-iteration alloc).
        let mut seen: HashSet<i64> = HashSet::with_capacity(batch_size);
        for _ in 0..MAX_RETRIES {
            seen.clear();
            let mut all_unique = true;
            for idx in &mut indices {
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
    /// State (8 planes, HEXB v6): pure coordinate scatter via `apply_symmetry_state`.
    /// Chain (6 planes): coordinate scatter + axis-plane remap via `apply_chain_symmetry`.
    #[inline]
    pub(crate) fn apply_sym(sym_idx: usize, slices: ApplySymSlices<'_>) {
        let ApplySymSlices {
            src: ApplySymSrc {
                state:  src_state,
                chain:  src_chain,
                policy: src_policy,
                own:    src_own,
                wl:     src_wl,
            },
            dst: ApplySymDst {
                state:  dst_state,
                chain:  dst_chain,
                policy: dst_policy,
                own:    dst_own,
                wl:     dst_wl,
            },
            tables,
        } = slices;

        // State planes: pure coordinate scatter (identity plane mapping).
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
        // Pass action (index n_cells) is always the identity (policy only).
        // H5-α guard: skip for encodings without a pass slot (e.g. v8 where
        // has_pass_slot=false and policy_logit_count=n_cells exactly — index
        // n_cells would be one past the end of the policy slice).
        // §173 A4 closes HAZARD H5-α.
        if tables.n_cells < dst_policy.len() {
            dst_policy[tables.n_cells] = src_policy[tables.n_cells];
        }
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
    ) -> PyResult<super::SampleBatchOut<'py>> {
        if self.size == 0 {
            return Err(PyValueError::new_err("Cannot sample from an empty replay buffer"));
        }

        // ── Index selection (with correlation guard) ──────────────────────────
        // Always run the dedup path: `sample_indices` treats the -1 untagged
        // sentinel as "skip this slot" per-sample, so mixed buffers (some
        // positions tagged, others not) are handled correctly.  The previous
        // slot-0 heuristic disabled dedup for the entire batch whenever slot
        // 0 happened to land in an untagged region, silently defeating the
        // Multi-Window correlation guard on the rest of the batch.  Cost of
        // dedup on a fully-untagged batch is one `HashSet` alloc plus
        // `batch_size` skips — measured negligible vs. the rest of the
        // scatter path.
        let indices = self.sample_indices(batch_size, true);

        let state_stride  = self.encoding.state_stride();
        let chain_stride  = self.encoding.chain_stride();
        let policy_stride = self.encoding.policy_stride();
        let aux_stride    = self.encoding.aux_stride();

        // ── Allocate output arrays (owned by Python after return) ─────────────
        // States and chain_planes as f16 bits (u16) — no type conversion during scatter.
        let mut out_states      = vec![0u16; batch_size * state_stride];
        let mut out_chain       = vec![0u16; batch_size * chain_stride];
        let mut out_policies    = vec![0.0f32; batch_size * policy_stride];
        let mut out_outcomes    = vec![0.0f32; batch_size];
        // Ownership default 1 = "empty" — cells outside the symmetry's destination
        // window stay at the same neutral value as the row's initial state.
        let mut out_ownership      = vec![1u8; batch_size * aux_stride];
        let mut out_winning_line   = vec![0u8; batch_size * aux_stride];
        // is_full_search is per-position metadata — no symmetry transform needed.
        let mut out_is_full_search = vec![0u8; batch_size];
        // DRAW-MASK (Phase 6) — per-row value-supervision flag; per-position
        // metadata, no symmetry transform. Mirrors out_is_full_search.
        let mut out_value_valid = vec![0u8; batch_size];
        // Note: sample_batch_impl does NOT emit position_indices to keep its
        // hot-path output stable pre/post §S181 Wave 4 4B-impl-1.
        // Callers needing per-row ply index use `sample_batch_with_pos_impl`.

        // ── Fill output ───────────────────────────────────────────────────────
        for (b, &idx) in indices.iter().enumerate() {
            let sym_idx = if augment { self.rng.random_range(0..N_SYMS) } else { 0 };

            let src_state  = &self.states      [idx * state_stride ..(idx + 1) * state_stride];
            let src_chain  = &self.chain_planes [idx * chain_stride ..(idx + 1) * chain_stride];
            let src_policy = &self.policies    [idx * policy_stride..(idx + 1) * policy_stride];
            let src_own    = &self.ownership   [idx * aux_stride   ..(idx + 1) * aux_stride];
            let src_wl     = &self.winning_line[idx * aux_stride   ..(idx + 1) * aux_stride];

            let dst_state  = &mut out_states      [b * state_stride ..(b + 1) * state_stride];
            let dst_chain  = &mut out_chain        [b * chain_stride ..(b + 1) * chain_stride];
            let dst_policy = &mut out_policies    [b * policy_stride..(b + 1) * policy_stride];
            let dst_own    = &mut out_ownership   [b * aux_stride   ..(b + 1) * aux_stride];
            let dst_wl     = &mut out_winning_line[b * aux_stride   ..(b + 1) * aux_stride];

            Self::apply_sym(sym_idx, ApplySymSlices {
                src: ApplySymSrc {
                    state:  src_state,
                    chain:  src_chain,
                    policy: src_policy,
                    own:    src_own,
                    wl:     src_wl,
                },
                dst: ApplySymDst {
                    state:  dst_state,
                    chain:  dst_chain,
                    policy: dst_policy,
                    own:    dst_own,
                    wl:     dst_wl,
                },
                tables: self.sym_tables,
            });

            out_outcomes[b] = self.outcomes[idx];
            out_is_full_search[b] = self.is_full_search[idx];
            out_value_valid[b] = self.value_target_valid[idx];
        }

        // ── Transmute u16 Vecs to f16 Vecs and wrap as numpy arrays ───────────
        // Safety: f16 and u16 have the same size/alignment; every bit pattern is valid for u16,
        // and we only wrote bits that came from valid f16 values stored via push().
        let states_f16: Vec<f16> = unsafe {
            let mut v = std::mem::ManuallyDrop::new(out_states);
            Vec::from_raw_parts(v.as_mut_ptr().cast::<f16>(), v.len(), v.capacity())
        };
        let chain_f16: Vec<f16> = unsafe {
            let mut v = std::mem::ManuallyDrop::new(out_chain);
            Vec::from_raw_parts(v.as_mut_ptr().cast::<f16>(), v.len(), v.capacity())
        };

        let n_planes       = self.encoding.n_planes;
        let trunk_size     = self.encoding.trunk_size;
        let n_logits       = self.encoding.policy_logit_count;
        let n_chain_planes = self.encoding.n_chain_planes();

        let states_np = states_f16
            .into_pyarray(py)
            .reshape([batch_size, n_planes, trunk_size, trunk_size])?;
        let chain_np = chain_f16
            .into_pyarray(py)
            .reshape([batch_size, n_chain_planes, trunk_size, trunk_size])?;
        let policies_np = out_policies
            .into_pyarray(py)
            .reshape([batch_size, n_logits])?;
        let outcomes_np = out_outcomes.into_pyarray(py);
        let ownership_np = out_ownership
            .into_pyarray(py)
            .reshape([batch_size, trunk_size, trunk_size])?;
        let winning_line_np = out_winning_line
            .into_pyarray(py)
            .reshape([batch_size, trunk_size, trunk_size])?;
        let is_full_search_np = out_is_full_search.into_pyarray(py);
        let value_valid_np = out_value_valid.into_pyarray(py);

        Ok((states_np, chain_np, policies_np, outcomes_np, ownership_np, winning_line_np, is_full_search_np, value_valid_np))
    }

    /// §S181-AUDIT Wave 4 4B-impl-1 — extended sampling that includes per-row
    /// position_indices for the ply-to-end aux head. Mirrors `sample_batch_impl`
    /// but returns 8-tuple.
    pub(crate) fn sample_batch_with_pos_impl<'py>(
        &mut self,
        py:        Python<'py>,
        batch_size: usize,
        augment:    bool,
    ) -> PyResult<super::SampleBatchWithPosOut<'py>> {
        if self.size == 0 {
            return Err(PyValueError::new_err("Cannot sample from an empty replay buffer"));
        }

        let indices = self.sample_indices(batch_size, true);

        let state_stride  = self.encoding.state_stride();
        let chain_stride  = self.encoding.chain_stride();
        let policy_stride = self.encoding.policy_stride();
        let aux_stride    = self.encoding.aux_stride();

        let mut out_states      = vec![0u16; batch_size * state_stride];
        let mut out_chain       = vec![0u16; batch_size * chain_stride];
        let mut out_policies    = vec![0.0f32; batch_size * policy_stride];
        let mut out_outcomes    = vec![0.0f32; batch_size];
        let mut out_ownership      = vec![1u8; batch_size * aux_stride];
        let mut out_winning_line   = vec![0u8; batch_size * aux_stride];
        let mut out_is_full_search = vec![0u8; batch_size];
        let mut out_position_indices = vec![0u16; batch_size];
        // DRAW-MASK (Phase 6) — per-row value-supervision flag (mirrors is_full_search).
        let mut out_value_valid = vec![0u8; batch_size];

        for (b, &idx) in indices.iter().enumerate() {
            let sym_idx = if augment { self.rng.random_range(0..N_SYMS) } else { 0 };

            let src_state  = &self.states      [idx * state_stride ..(idx + 1) * state_stride];
            let src_chain  = &self.chain_planes [idx * chain_stride ..(idx + 1) * chain_stride];
            let src_policy = &self.policies    [idx * policy_stride..(idx + 1) * policy_stride];
            let src_own    = &self.ownership   [idx * aux_stride   ..(idx + 1) * aux_stride];
            let src_wl     = &self.winning_line[idx * aux_stride   ..(idx + 1) * aux_stride];

            let dst_state  = &mut out_states      [b * state_stride ..(b + 1) * state_stride];
            let dst_chain  = &mut out_chain        [b * chain_stride ..(b + 1) * chain_stride];
            let dst_policy = &mut out_policies    [b * policy_stride..(b + 1) * policy_stride];
            let dst_own    = &mut out_ownership   [b * aux_stride   ..(b + 1) * aux_stride];
            let dst_wl     = &mut out_winning_line[b * aux_stride   ..(b + 1) * aux_stride];

            Self::apply_sym(sym_idx, ApplySymSlices {
                src: ApplySymSrc {
                    state:  src_state,
                    chain:  src_chain,
                    policy: src_policy,
                    own:    src_own,
                    wl:     src_wl,
                },
                dst: ApplySymDst {
                    state:  dst_state,
                    chain:  dst_chain,
                    policy: dst_policy,
                    own:    dst_own,
                    wl:     dst_wl,
                },
                tables: self.sym_tables,
            });

            out_outcomes[b] = self.outcomes[idx];
            out_is_full_search[b] = self.is_full_search[idx];
            out_position_indices[b] = self.position_indices[idx];
            out_value_valid[b] = self.value_target_valid[idx];
        }

        let states_f16: Vec<f16> = unsafe {
            let mut v = std::mem::ManuallyDrop::new(out_states);
            Vec::from_raw_parts(v.as_mut_ptr().cast::<f16>(), v.len(), v.capacity())
        };
        let chain_f16: Vec<f16> = unsafe {
            let mut v = std::mem::ManuallyDrop::new(out_chain);
            Vec::from_raw_parts(v.as_mut_ptr().cast::<f16>(), v.len(), v.capacity())
        };

        let n_planes       = self.encoding.n_planes;
        let trunk_size     = self.encoding.trunk_size;
        let n_logits       = self.encoding.policy_logit_count;
        let n_chain_planes = self.encoding.n_chain_planes();

        let states_np = states_f16
            .into_pyarray(py)
            .reshape([batch_size, n_planes, trunk_size, trunk_size])?;
        let chain_np = chain_f16
            .into_pyarray(py)
            .reshape([batch_size, n_chain_planes, trunk_size, trunk_size])?;
        let policies_np = out_policies
            .into_pyarray(py)
            .reshape([batch_size, n_logits])?;
        let outcomes_np = out_outcomes.into_pyarray(py);
        let ownership_np = out_ownership
            .into_pyarray(py)
            .reshape([batch_size, trunk_size, trunk_size])?;
        let winning_line_np = out_winning_line
            .into_pyarray(py)
            .reshape([batch_size, trunk_size, trunk_size])?;
        let is_full_search_np = out_is_full_search.into_pyarray(py);
        let position_indices_np = out_position_indices.into_pyarray(py);
        let value_valid_np = out_value_valid.into_pyarray(py);

        Ok((states_np, chain_np, policies_np, outcomes_np, ownership_np, winning_line_np, is_full_search_np, position_indices_np, value_valid_np))
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
        let v6_spec = crate::encoding::registry::lookup_or_panic("v6");
        let default_w = f16::from_f32(1.0).to_bits();
        let mut buf = ReplayBuffer {
            capacity: 300,
            size: 0,
            head: 0,
            encoding: v6_spec,
            states:          vec![0u16; 300 * STATE_STRIDE],
            chain_planes:    vec![0u16; 300 * CHAIN_STRIDE],
            policies:        vec![0.0f32; 300 * POLICY_STRIDE],
            outcomes:        vec![0.0f32; 300],
            game_ids:        vec![-1i64; 300],
            weights:         vec![default_w; 300],
            ownership:       vec![1u8; 300 * AUX_STRIDE],
            winning_line:    vec![0u8; 300 * AUX_STRIDE],
            is_full_search:  vec![1u8; 300],
            value_target_valid: vec![1u8; 300],
            position_indices: vec![0u16; 300],
            sym_tables: crate::replay_buffer::sym_tables::sym_tables_for(v6_spec),
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
                let src_chain     = vec![0u16; N_CHAIN_PLANES * N_CELLS];
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

                ReplayBuffer::apply_sym(sym_idx, ApplySymSlices {
                    src: ApplySymSrc {
                        state:  &src_state,
                        chain:  &src_chain,
                        policy: &src_pol,
                        own:    &src_own,
                        wl:     &src_wl,
                    },
                    dst: ApplySymDst {
                        state:  &mut dst_state,
                        chain:  &mut dst_chain,
                        policy: &mut dst_pol,
                        own:    &mut dst_own,
                        wl:     &mut dst_wl,
                    },
                    tables: &tables,
                });

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
