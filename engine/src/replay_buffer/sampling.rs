//! Weighted sampling and symmetry application for `ReplayBuffer`.

use half::f16;
use rand::RngExt;
use std::collections::HashSet;

use super::ReplayBuffer;
use super::sym_tables::*;

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
                seen.insert(self.game_ids[candidate]);
            }
            if all_unique { break; }
        }

        indices
    }

    /// Apply symmetry `sym_idx` to one (state, policy, ownership, winning_line) sample.
    ///
    /// Scatter-copies from `src_*` into `dst_*`. Cells that have no valid destination
    /// under the transform remain at the caller-zeroed default. Aux planes
    /// (`ownership`, `winning_line`) reuse the same scatter table as state — they live
    /// in the same window coordinate frame, just with u8 lanes.
    ///
    /// Loop order — outer = planes, inner = scatter pairs — keeps each 722-byte
    /// plane (361 × u16) resident in L1 cache during the inner scatter pass.
    #[inline]
    pub(crate) fn apply_sym(
        sym_idx:    usize,
        src_state:  &[u16],     // f16 bits, length N_PLANES × N_CELLS
        src_policy: &[f32],     // length N_ACTIONS
        src_own:    &[u8],      // length AUX_STRIDE (= N_CELLS)
        src_wl:     &[u8],      // length AUX_STRIDE
        dst_state:  &mut [u16], // f16 bits, length N_PLANES × N_CELLS  (zeroed by caller)
        dst_policy: &mut [f32], // length N_ACTIONS                     (zeroed by caller)
        dst_own:    &mut [u8],  // length AUX_STRIDE                    (caller-initialised)
        dst_wl:     &mut [u8],  // length AUX_STRIDE                    (caller-initialised)
        tables:     &SymTables,
    ) {
        let scatter = &tables.scatter[sym_idx];

        // State: for each plane, scatter the 361-cell slice.
        // Plane stride = N_CELLS = 361 u16 = 722 bytes (fits in a few cache lines).
        for p in 0..N_PLANES {
            let base = p * N_CELLS;
            let src_plane = &src_state[base..base + N_CELLS];
            let dst_plane = &mut dst_state[base..base + N_CELLS];
            for &(sc, dc) in scatter {
                dst_plane[dc as usize] = src_plane[sc as usize];
            }
        }

        // Policy: scatter the 361 spatial logits.
        for &(sc, dc) in scatter {
            dst_policy[dc as usize] = src_policy[sc as usize];
        }
        // Pass action (index 361) is always the identity.
        dst_policy[N_CELLS] = src_policy[N_CELLS];

        // Ownership + winning_line: single 361-cell u8 planes, same scatter table.
        for &(sc, dc) in scatter {
            dst_own[dc as usize] = src_own[sc as usize];
            dst_wl [dc as usize] = src_wl [sc as usize];
        }
    }

    /// Push a position directly from Rust (no PyO3 / numpy).
    /// Used only by tests.
    #[cfg(test)]
    pub(crate) fn push_raw(&mut self, outcome: f32, game_length: u16) {
        use std::sync::atomic::Ordering;

        let slot = self.head;

        // Decrement old bucket if overwriting.
        if self.size == self.capacity {
            let old_bucket = Self::weight_bucket(self.weights[slot]);
            self.weight_buckets[old_bucket].fetch_sub(1, Ordering::Relaxed);
        }

        // Zero state/policy/aux (content doesn't matter for weight tests).
        let s_start = slot * STATE_STRIDE;
        for v in &mut self.states[s_start..s_start + STATE_STRIDE] { *v = 0; }
        let p_start = slot * POLICY_STRIDE;
        for v in &mut self.policies[p_start..p_start + POLICY_STRIDE] { *v = 0.0; }
        let a_start = slot * AUX_STRIDE;
        for v in &mut self.ownership   [a_start..a_start + AUX_STRIDE] { *v = 1; } // empty
        for v in &mut self.winning_line[a_start..a_start + AUX_STRIDE] { *v = 0; }
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
