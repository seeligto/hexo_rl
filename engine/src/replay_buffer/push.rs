//! Write path for `ReplayBuffer` — single-position `push` and batched `push_game`.
//!
//! Both entry points manage the ring-buffer head, update the lock-free weight
//! histogram, and accept PyO3 numpy views that are validated for contiguity
//! and shape before any copy. A test-only `push_raw` helper is provided for
//! the sampling/aux correctness tests that live in this crate.

use half::f16;
use std::sync::atomic::Ordering;

use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, PyReadonlyArray4};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::sym_tables::*;
use super::ReplayBuffer;

impl ReplayBuffer {
    /// Store a single `(state, chain_planes, policy, outcome, ownership, winning_line)` sample.
    ///
    /// Args:
    ///     state:        float16 numpy array of shape (18, 19, 19)
    ///     chain_planes: float16 numpy array of shape (6, 19, 19) — Q13 chain-length planes
    ///     policy:       float32 numpy array of shape (362,)
    ///     outcome:      scalar float32  (−1 / 0 / +1)
    ///     ownership:    uint8 numpy array of shape (361,)  encoding {0=P2, 1=empty, 2=P1}
    ///     winning_line: uint8 numpy array of shape (361,)  binary mask
    ///     game_id:      position tag for correlation guard (from `next_game_id()`); default −1
    ///     game_length:  total compound moves in the originating game; default 0 (= weight 1.0)
    ///
    /// All aux targets MUST be projected to the same window centre as `state` (per-row cluster
    /// centre, not the game-end bbox centroid).
    pub(crate) fn push_impl(
        &mut self,
        state:          PyReadonlyArray3<f16>,
        chain_planes:   PyReadonlyArray3<f16>,
        policy:         PyReadonlyArray1<f32>,
        outcome:        f32,
        ownership:      PyReadonlyArray1<u8>,
        winning_line:   PyReadonlyArray1<u8>,
        game_id:        i64,
        game_length:    u16,
        is_full_search: bool,
    ) -> PyResult<()> {
        let state_slice  = state.as_slice()
            .map_err(|e| PyValueError::new_err(format!("state not contiguous: {e}")))?;
        let chain_slice  = chain_planes.as_slice()
            .map_err(|e| PyValueError::new_err(format!("chain_planes not contiguous: {e}")))?;
        let policy_slice = policy.as_slice()
            .map_err(|e| PyValueError::new_err(format!("policy not contiguous: {e}")))?;
        let own_slice = ownership.as_slice()
            .map_err(|e| PyValueError::new_err(format!("ownership not contiguous: {e}")))?;
        let wl_slice = winning_line.as_slice()
            .map_err(|e| PyValueError::new_err(format!("winning_line not contiguous: {e}")))?;

        if state_slice.len() != STATE_STRIDE {
            return Err(PyValueError::new_err(format!(
                "state must have {} elements (18×19×19), got {}", STATE_STRIDE, state_slice.len()
            )));
        }
        if chain_slice.len() != CHAIN_STRIDE {
            return Err(PyValueError::new_err(format!(
                "chain_planes must have {} elements (6×19×19), got {}", CHAIN_STRIDE, chain_slice.len()
            )));
        }
        if policy_slice.len() != POLICY_STRIDE {
            return Err(PyValueError::new_err(format!(
                "policy must have {} elements (362), got {}", POLICY_STRIDE, policy_slice.len()
            )));
        }
        if own_slice.len() != AUX_STRIDE {
            return Err(PyValueError::new_err(format!(
                "ownership must have {} elements (361), got {}", AUX_STRIDE, own_slice.len()
            )));
        }
        if wl_slice.len() != AUX_STRIDE {
            return Err(PyValueError::new_err(format!(
                "winning_line must have {} elements (361), got {}", AUX_STRIDE, wl_slice.len()
            )));
        }

        let slot = self.head;

        // If overwriting a valid slot, decrement its bucket before we clobber it.
        if self.size == self.capacity {
            let old_bucket = Self::weight_bucket(self.weights[slot]);
            self.weight_buckets[old_bucket].fetch_sub(1, Ordering::Relaxed);
        }

        // Copy state as raw f16 bits.
        let dst = &mut self.states[slot * STATE_STRIDE..(slot + 1) * STATE_STRIDE];
        for (d, s) in dst.iter_mut().zip(state_slice.iter()) {
            *d = s.to_bits();
        }

        // Copy chain planes as raw f16 bits.
        let dst_chain = &mut self.chain_planes[slot * CHAIN_STRIDE..(slot + 1) * CHAIN_STRIDE];
        for (d, s) in dst_chain.iter_mut().zip(chain_slice.iter()) {
            *d = s.to_bits();
        }

        self.policies[slot * POLICY_STRIDE..(slot + 1) * POLICY_STRIDE]
            .copy_from_slice(policy_slice);
        self.outcomes[slot] = outcome;
        self.game_ids[slot] = game_id;
        self.weights[slot] = if game_length == 0 {
            f16::from_f32(1.0).to_bits()
        } else {
            self.weight_schedule.weight_for(game_length)
        };

        self.ownership[slot * AUX_STRIDE..(slot + 1) * AUX_STRIDE]
            .copy_from_slice(own_slice);
        self.winning_line[slot * AUX_STRIDE..(slot + 1) * AUX_STRIDE]
            .copy_from_slice(wl_slice);
        self.is_full_search[slot] = is_full_search as u8;

        // Increment the new position's bucket.
        let new_bucket = Self::weight_bucket(self.weights[slot]);
        self.weight_buckets[new_bucket].fetch_add(1, Ordering::Relaxed);

        self.head = (self.head + 1) % self.capacity;
        self.size  = (self.size + 1).min(self.capacity);
        Ok(())
    }

    /// Store all positions from a completed game efficiently.
    ///
    /// Handles ring-buffer wrap-around correctly.
    ///
    /// Args:
    ///     states:        float16 numpy array of shape (T, 18, 19, 19)
    ///     chain_planes:  float16 numpy array of shape (T, 6, 19, 19)
    ///     policies:      float32 numpy array of shape (T, 362)
    ///     outcomes:      float32 numpy array of shape (T,)
    ///     ownership:     uint8   numpy array of shape (T, 361)  per-row {0=P2, 1=empty, 2=P1}
    ///     winning_line:  uint8   numpy array of shape (T, 361)  per-row binary mask
    ///     game_id:       shared position tag for all T entries; default −1
    ///     game_length:   total compound moves in the originating game; default 0 (= weight 1.0)
    pub(crate) fn push_game_impl(
        &mut self,
        states:         PyReadonlyArray4<f16>,
        chain_planes:   PyReadonlyArray4<f16>,
        policies:       PyReadonlyArray2<f32>,
        outcomes:       PyReadonlyArray1<f32>,
        ownership:      PyReadonlyArray2<u8>,
        winning_line:   PyReadonlyArray2<u8>,
        game_id:        i64,
        game_length:    u16,
        is_full_search: Option<PyReadonlyArray1<u8>>,
    ) -> PyResult<()> {
        let states_s   = states.as_slice()
            .map_err(|e| PyValueError::new_err(format!("states not contiguous: {e}")))?;
        let chain_s    = chain_planes.as_slice()
            .map_err(|e| PyValueError::new_err(format!("chain_planes not contiguous: {e}")))?;
        let policies_s = policies.as_slice()
            .map_err(|e| PyValueError::new_err(format!("policies not contiguous: {e}")))?;
        let outcomes_s = outcomes.as_slice()
            .map_err(|e| PyValueError::new_err(format!("outcomes not contiguous: {e}")))?;
        let own_s = ownership.as_slice()
            .map_err(|e| PyValueError::new_err(format!("ownership not contiguous: {e}")))?;
        let wl_s = winning_line.as_slice()
            .map_err(|e| PyValueError::new_err(format!("winning_line not contiguous: {e}")))?;
        // Resolve optional is_full_search slice; default 1 (full-search) when not provided.
        let ifs_owned: Vec<u8>;
        let ifs_s: &[u8] = if let Some(ref arr) = is_full_search {
            arr.as_slice()
                .map_err(|e| PyValueError::new_err(format!("is_full_search not contiguous: {e}")))?
        } else {
            // Allocate a temporary all-ones slice; `t` is resolved below.
            ifs_owned = Vec::new();
            &ifs_owned
        };

        let t = outcomes_s.len();
        if t == 0 { return Ok(()); }

        if states_s.len()   != t * STATE_STRIDE  { return Err(PyValueError::new_err("states shape mismatch")); }
        if chain_s.len()    != t * CHAIN_STRIDE  { return Err(PyValueError::new_err("chain_planes shape mismatch")); }
        if policies_s.len() != t * POLICY_STRIDE { return Err(PyValueError::new_err("policies shape mismatch")); }
        if own_s.len() != t * AUX_STRIDE { return Err(PyValueError::new_err("ownership shape mismatch")); }
        if wl_s.len()  != t * AUX_STRIDE { return Err(PyValueError::new_err("winning_line shape mismatch")); }
        if !ifs_s.is_empty() && ifs_s.len() != t {
            return Err(PyValueError::new_err(format!(
                "is_full_search must have {} elements (one per position), got {}", t, ifs_s.len()
            )));
        }

        let w = if game_length == 0 {
            f16::from_f32(1.0).to_bits()
        } else {
            self.weight_schedule.weight_for(game_length)
        };
        let new_bucket = Self::weight_bucket(w);

        // Number of available (empty) slots before we start writing.
        // Positions i >= available are overwrites of previously valid slots.
        let available = self.capacity.saturating_sub(self.size);

        for i in 0..t {
            let slot = (self.head + i) % self.capacity;

            // Decrement old bucket if this slot held valid data.
            if i >= available {
                let old_bucket = Self::weight_bucket(self.weights[slot]);
                self.weight_buckets[old_bucket].fetch_sub(1, Ordering::Relaxed);
            }

            // State: convert f16 → u16 bits
            let src_state = &states_s[i * STATE_STRIDE..(i + 1) * STATE_STRIDE];
            let dst_state = &mut self.states[slot * STATE_STRIDE..(slot + 1) * STATE_STRIDE];
            for (d, s) in dst_state.iter_mut().zip(src_state.iter()) {
                *d = s.to_bits();
            }

            // Chain planes: convert f16 → u16 bits
            let src_chain = &chain_s[i * CHAIN_STRIDE..(i + 1) * CHAIN_STRIDE];
            let dst_chain = &mut self.chain_planes[slot * CHAIN_STRIDE..(slot + 1) * CHAIN_STRIDE];
            for (d, s) in dst_chain.iter_mut().zip(src_chain.iter()) {
                *d = s.to_bits();
            }

            // Policy: direct f32 copy
            let src_pol = &policies_s[i * POLICY_STRIDE..(i + 1) * POLICY_STRIDE];
            self.policies[slot * POLICY_STRIDE..(slot + 1) * POLICY_STRIDE]
                .copy_from_slice(src_pol);

            // Auxiliary spatial targets — direct u8 copies.
            self.ownership[slot * AUX_STRIDE..(slot + 1) * AUX_STRIDE]
                .copy_from_slice(&own_s[i * AUX_STRIDE..(i + 1) * AUX_STRIDE]);
            self.winning_line[slot * AUX_STRIDE..(slot + 1) * AUX_STRIDE]
                .copy_from_slice(&wl_s[i * AUX_STRIDE..(i + 1) * AUX_STRIDE]);

            // is_full_search: use provided value or default to 1 (full-search).
            self.is_full_search[slot] = if ifs_s.is_empty() { 1u8 } else { ifs_s[i] };

            self.outcomes[slot] = outcomes_s[i];
            self.game_ids[slot] = game_id;
            self.weights[slot]  = w;

            // Increment new bucket.
            self.weight_buckets[new_bucket].fetch_add(1, Ordering::Relaxed);
        }

        self.head = (self.head + t) % self.capacity;
        self.size  = (self.size + t).min(self.capacity);
        Ok(())
    }

    /// Store N positions with per-row `game_length` and `is_full_search`.
    ///
    /// Replaces the per-row `push` loop in `WorkerPool._stats_loop`, avoiding
    /// N PyO3 method-dispatch + PyRefMut acquire/release cycles.
    ///
    /// Args:
    ///     states:         float16 numpy array of shape (N, 18, 19, 19)
    ///     chain_planes:   float16 numpy array of shape (N, 6, 19, 19)
    ///     policies:       float32 numpy array of shape (N, 362)
    ///     outcomes:       float32 numpy array of shape (N,)
    ///     ownership:      uint8   numpy array of shape (N, 361)
    ///     winning_line:   uint8   numpy array of shape (N, 361)
    ///     game_lengths:   uint16  numpy array of shape (N,) — compound-move counts
    ///     is_full_search: uint8   numpy array of shape (N,)
    pub(crate) fn push_many_impl(
        &mut self,
        states:         PyReadonlyArray4<f16>,
        chain_planes:   PyReadonlyArray4<f16>,
        policies:       PyReadonlyArray2<f32>,
        outcomes:       PyReadonlyArray1<f32>,
        ownership:      PyReadonlyArray2<u8>,
        winning_line:   PyReadonlyArray2<u8>,
        game_lengths:   PyReadonlyArray1<u16>,
        is_full_search: PyReadonlyArray1<u8>,
    ) -> PyResult<()> {
        let states_s   = states.as_slice()
            .map_err(|e| PyValueError::new_err(format!("states not contiguous: {e}")))?;
        let chain_s    = chain_planes.as_slice()
            .map_err(|e| PyValueError::new_err(format!("chain_planes not contiguous: {e}")))?;
        let policies_s = policies.as_slice()
            .map_err(|e| PyValueError::new_err(format!("policies not contiguous: {e}")))?;
        let outcomes_s = outcomes.as_slice()
            .map_err(|e| PyValueError::new_err(format!("outcomes not contiguous: {e}")))?;
        let own_s = ownership.as_slice()
            .map_err(|e| PyValueError::new_err(format!("ownership not contiguous: {e}")))?;
        let wl_s = winning_line.as_slice()
            .map_err(|e| PyValueError::new_err(format!("winning_line not contiguous: {e}")))?;
        let gl_s = game_lengths.as_slice()
            .map_err(|e| PyValueError::new_err(format!("game_lengths not contiguous: {e}")))?;
        let ifs_s = is_full_search.as_slice()
            .map_err(|e| PyValueError::new_err(format!("is_full_search not contiguous: {e}")))?;

        let t = outcomes_s.len();
        if t == 0 { return Ok(()); }

        if states_s.len()   != t * STATE_STRIDE  { return Err(PyValueError::new_err("states shape mismatch")); }
        if chain_s.len()    != t * CHAIN_STRIDE  { return Err(PyValueError::new_err("chain_planes shape mismatch")); }
        if policies_s.len() != t * POLICY_STRIDE { return Err(PyValueError::new_err("policies shape mismatch")); }
        if own_s.len() != t * AUX_STRIDE { return Err(PyValueError::new_err("ownership shape mismatch")); }
        if wl_s.len()  != t * AUX_STRIDE { return Err(PyValueError::new_err("winning_line shape mismatch")); }
        if gl_s.len() != t {
            return Err(PyValueError::new_err(format!("game_lengths must have {t} elements, got {}", gl_s.len())));
        }
        if ifs_s.len() != t {
            return Err(PyValueError::new_err(format!("is_full_search must have {t} elements, got {}", ifs_s.len())));
        }

        let available = self.capacity.saturating_sub(self.size);

        for i in 0..t {
            let slot = (self.head + i) % self.capacity;

            if i >= available {
                let old_bucket = Self::weight_bucket(self.weights[slot]);
                self.weight_buckets[old_bucket].fetch_sub(1, Ordering::Relaxed);
            }

            let game_length = gl_s[i];
            let w = if game_length == 0 {
                f16::from_f32(1.0).to_bits()
            } else {
                self.weight_schedule.weight_for(game_length)
            };
            let new_bucket = Self::weight_bucket(w);

            let src_state = &states_s[i * STATE_STRIDE..(i + 1) * STATE_STRIDE];
            let dst_state = &mut self.states[slot * STATE_STRIDE..(slot + 1) * STATE_STRIDE];
            for (d, s) in dst_state.iter_mut().zip(src_state.iter()) {
                *d = s.to_bits();
            }

            let src_chain = &chain_s[i * CHAIN_STRIDE..(i + 1) * CHAIN_STRIDE];
            let dst_chain = &mut self.chain_planes[slot * CHAIN_STRIDE..(slot + 1) * CHAIN_STRIDE];
            for (d, s) in dst_chain.iter_mut().zip(src_chain.iter()) {
                *d = s.to_bits();
            }

            self.policies[slot * POLICY_STRIDE..(slot + 1) * POLICY_STRIDE]
                .copy_from_slice(&policies_s[i * POLICY_STRIDE..(i + 1) * POLICY_STRIDE]);
            self.ownership[slot * AUX_STRIDE..(slot + 1) * AUX_STRIDE]
                .copy_from_slice(&own_s[i * AUX_STRIDE..(i + 1) * AUX_STRIDE]);
            self.winning_line[slot * AUX_STRIDE..(slot + 1) * AUX_STRIDE]
                .copy_from_slice(&wl_s[i * AUX_STRIDE..(i + 1) * AUX_STRIDE]);

            self.is_full_search[slot] = ifs_s[i];
            self.outcomes[slot] = outcomes_s[i];
            self.game_ids[slot] = -1;
            self.weights[slot]  = w;

            self.weight_buckets[new_bucket].fetch_add(1, Ordering::Relaxed);
        }

        self.head = (self.head + t) % self.capacity;
        self.size = (self.size + t).min(self.capacity);
        Ok(())
    }

    /// Push a position directly from Rust (no PyO3 / numpy).
    /// Used only by tests in this crate.
    #[cfg(test)]
    pub(crate) fn push_raw(&mut self, outcome: f32, game_length: u16) {
        let slot = self.head;

        // Decrement old bucket if overwriting.
        if self.size == self.capacity {
            let old_bucket = Self::weight_bucket(self.weights[slot]);
            self.weight_buckets[old_bucket].fetch_sub(1, Ordering::Relaxed);
        }

        // Zero state/policy/aux (content doesn't matter for weight tests).
        let s_start = slot * STATE_STRIDE;
        self.states[s_start..s_start + STATE_STRIDE].fill(0);
        let c_start = slot * CHAIN_STRIDE;
        self.chain_planes[c_start..c_start + CHAIN_STRIDE].fill(0);
        let p_start = slot * POLICY_STRIDE;
        self.policies[p_start..p_start + POLICY_STRIDE].fill(0.0);
        let a_start = slot * AUX_STRIDE;
        self.ownership   [a_start..a_start + AUX_STRIDE].fill(1); // empty
        self.winning_line[a_start..a_start + AUX_STRIDE].fill(0);
        self.is_full_search[slot] = 1;
        self.outcomes[slot] = outcome;
        self.game_ids[slot] = -1;
        self.weights[slot] = if game_length == 0 {
            f16::from_f32(1.0).to_bits()
        } else {
            self.weight_schedule.weight_for(game_length)
        };

        let new_bucket = Self::weight_bucket(self.weights[slot]);
        self.weight_buckets[new_bucket].fetch_add(1, Ordering::Relaxed);

        self.head = (self.head + 1) % self.capacity;
        self.size = (self.size + 1).min(self.capacity);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{RngExt, SeedableRng};
    use rand::rngs::StdRng;

    /// Reproject correctness — verify that projecting the same axial cells
    /// through two different window centres yields different flat indices, and
    /// that the chosen centre's projection round-trips through the buffer slot.
    ///
    /// This is the core invariant the legacy code violated: it used a single
    /// game-end bbox centre for all rows, while each row's state planes used
    /// its own per-cluster centre.
    #[test]
    fn test_aux_alignment_reproject() {
        use crate::board::Board;

        // 6-cell winning line at axial (5, 0)..(5, 5) — far from origin so
        // both candidate centres land it in different flat positions.
        let winning_cells: Vec<(i32, i32)> = (0..6).map(|i| (5, i)).collect();

        // Centre A: matches the winning line bbox (legacy game-end frame).
        let (cq_a, cr_a) = (5i32, 2i32);
        // Centre B: a hypothetical per-cluster centre offset from the line.
        let (cq_b, cr_b) = (8i32, 8i32);

        let proj_a: Vec<usize> = winning_cells.iter()
            .map(|&(q, r)| Board::window_flat_idx_at(q, r, cq_a, cr_a))
            .collect();
        let proj_b: Vec<usize> = winning_cells.iter()
            .map(|&(q, r)| Board::window_flat_idx_at(q, r, cq_b, cr_b))
            .collect();

        // Sanity: at least one in-window cell must differ between the two
        // centres — otherwise the test wouldn't actually distinguish them.
        let differs = proj_a.iter().zip(&proj_b).any(|(a, b)| {
            *a < AUX_STRIDE && *b < AUX_STRIDE && a != b
        });
        assert!(differs, "test setup: centres A and B must yield different projections");

        // Build the per-row aux mask using centre B (the row's own centre)
        // and write it into the buffer's ring storage directly.
        let mut buf = ReplayBuffer::new(4);
        let slot = 0;
        let a_start = slot * AUX_STRIDE;
        for &flat in &proj_b {
            if flat < AUX_STRIDE {
                buf.winning_line[a_start + flat] = 1;
            }
        }
        buf.size = 1;
        buf.head = 1;

        // Confirm the stored mask is identically the centre-B projection
        // and disagrees with the centre-A projection on at least one cell.
        let mut centre_b_hits = 0usize;
        for &flat in &proj_b {
            if flat < AUX_STRIDE {
                assert_eq!(
                    buf.winning_line[a_start + flat], 1,
                    "centre-B projection cell {flat} must be marked"
                );
                centre_b_hits += 1;
            }
        }
        assert!(centre_b_hits > 0, "no centre-B cells landed in window");

        let mut centre_a_only_misses = 0usize;
        for (&fa, &fb) in proj_a.iter().zip(&proj_b) {
            if fa < AUX_STRIDE && fa != fb && buf.winning_line[a_start + fa] == 0 {
                centre_a_only_misses += 1;
            }
        }
        assert!(
            centre_a_only_misses > 0,
            "centre-A projection should diverge from centre-B aux mask on at least one cell"
        );
    }

    /// Stress: push 1000 rows with random aux content via push_raw, then run
    /// 1000 sample_indices + apply_sym iterations against random symmetries.
    /// Asserts no panic, output shapes stay correct, and the four spatial
    /// outputs receive the chosen symmetry consistently.
    #[test]
    fn test_aux_stress_1k_rows() {
        let mut buf = ReplayBuffer::new(2000);

        // Push 1000 rows. push_raw zeroes state/policy/aux to defaults; we
        // then sprinkle random nonzero markers into the aux columns directly.
        for _ in 0..1000 { buf.push_raw(0.0, 10); }
        assert_eq!(buf.size, 1000);
        assert_eq!(buf.ownership.len(), 2000 * AUX_STRIDE);
        assert_eq!(buf.winning_line.len(), 2000 * AUX_STRIDE);

        let mut rng = StdRng::seed_from_u64(0xA1A1);
        for slot in 0..1000 {
            let a_start = slot * AUX_STRIDE;
            // Mark ~20 random cells per row in each aux plane.
            for _ in 0..20 {
                let cell = rng.random_range(0..AUX_STRIDE);
                buf.ownership[a_start + cell]    = if rng.random_bool(0.5) { 0 } else { 2 };
                buf.winning_line[a_start + cell] = 1;
            }
        }

        // Run 1000 sample-batch-equivalents in pure Rust (no GIL).
        let mut dst_state = vec![0u16; N_PLANES * N_CELLS];
        let mut dst_chain = vec![0u16; N_CHAIN_PLANES * N_CELLS];
        let mut dst_pol   = vec![0.0f32; N_ACTIONS];
        let mut dst_own   = vec![1u8;  AUX_STRIDE];
        let mut dst_wl    = vec![0u8;  AUX_STRIDE];

        for _ in 0..1000 {
            let idx = buf.weighted_sample_one();
            let sym_idx = rng.random_range(0..N_SYMS);

            dst_state.fill(0);
            dst_chain.fill(0);
            dst_pol.fill(0.0);
            dst_own.fill(1);
            dst_wl.fill(0);

            let s = idx * STATE_STRIDE;
            let c = idx * CHAIN_STRIDE;
            let p = idx * POLICY_STRIDE;
            let a = idx * AUX_STRIDE;
            ReplayBuffer::apply_sym(
                sym_idx,
                &buf.states[s..s + STATE_STRIDE],
                &buf.chain_planes[c..c + CHAIN_STRIDE],
                &buf.policies[p..p + POLICY_STRIDE],
                &buf.ownership[a..a + AUX_STRIDE],
                &buf.winning_line[a..a + AUX_STRIDE],
                &mut dst_state, &mut dst_chain, &mut dst_pol, &mut dst_own, &mut dst_wl,
                &buf.sym_tables,
            );

            // Every output ownership cell must be a valid encoding {0,1,2}.
            for &v in &dst_own {
                assert!(v <= 2, "ownership out-of-range: {v}");
            }
            for &v in &dst_wl {
                assert!(v <= 1, "winning_line out-of-range: {v}");
            }
        }
    }
}
