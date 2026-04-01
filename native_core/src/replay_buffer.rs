//! RustReplayBuffer — pre-allocated ring buffer with vectorized 12-fold hex augmentation.
//!
//! Stores (state, policy, outcome) tuples from self-play.  On `sample_batch` the data
//! is scatter-augmented in Rust and returned as owned numpy arrays (zero extra copy).
//!
//! ## Memory layout (flat, row-major)
//!   states   : Vec<u16>  — f16 bits, logical shape [capacity, 18, 361]
//!   policies : Vec<f32>  — logical shape [capacity, 362]
//!   outcomes : Vec<f32>  — logical shape [capacity]
//!   game_ids : Vec<i64>  — logical shape [capacity]; -1 = untagged
//!
//! ## Symmetry tables
//! 12 symmetries = 6 rotations × 2 (with/without prior reflection).
//! Each symmetry is stored as a Vec<(u16, u16)> of valid (src_cell, dst_cell) scatter pairs
//! (cells that map out of the 19×19 window are dropped, matching the Python implementation).

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::{
    IntoPyArray,
    PyArray1, PyArray2, PyArray4,
    PyArrayMethods,
    PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, PyReadonlyArray4,
};
use half::f16;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use std::collections::HashSet;

// ── Constants ─────────────────────────────────────────────────────────────────

const N_PLANES:  usize = 18;
const BOARD_H:   usize = 19;
const BOARD_W:   usize = 19;
const N_CELLS:   usize = BOARD_H * BOARD_W; // 361
const N_ACTIONS: usize = N_CELLS + 1;       // 362 (pass move at index 361)
const N_SYMS:    usize = 12;
const HALF:      i32   = 9; // (BOARD_H - 1) / 2

// State stride per buffer slot (f16 bits)
const STATE_STRIDE:  usize = N_PLANES * N_CELLS;
const POLICY_STRIDE: usize = N_ACTIONS;

// ── SymTables ─────────────────────────────────────────────────────────────────

/// Precomputed scatter tables for all 12 hexagonal symmetries.
///
/// `scatter[s]` is the list of `(src_cell, dst_cell)` pairs for symmetry `s`.
/// Cells that fall outside the 19×19 window after transformation are omitted —
/// the corresponding output cells remain zero (matching the Python behaviour).
struct SymTables {
    scatter: [Vec<(u16, u16)>; N_SYMS],
}

impl SymTables {
    fn new() -> Self {
        // Axial → flat index.  Returns None if the result is out of the 19×19 window.
        let to_flat = |q: i32, r: i32| -> Option<u16> {
            let qi = q + HALF;
            let ri = r + HALF;
            if qi >= 0 && qi < BOARD_H as i32 && ri >= 0 && ri < BOARD_W as i32 {
                Some((qi as usize * BOARD_W + ri as usize) as u16)
            } else {
                None
            }
        };

        // Flat index → axial coordinates.
        let from_flat = |flat: usize| -> (i32, i32) {
            ((flat / BOARD_W) as i32 - HALF, (flat % BOARD_W) as i32 - HALF)
        };

        // Each symmetry gets its own Vec.  We use a const-size array with a dummy
        // initialiser then overwrite each element.
        const EMPTY: Vec<(u16, u16)> = Vec::new();
        let mut scatter = [EMPTY; N_SYMS];

        for sym_idx in 0..N_SYMS {
            let reflect = sym_idx >= 6;
            let n_rot   = sym_idx % 6;
            let mut pairs: Vec<(u16, u16)> = Vec::with_capacity(N_CELLS);

            for src in 0..N_CELLS {
                let (mut q, mut r) = from_flat(src);

                // Optional reflection first (swap axes).
                if reflect {
                    (q, r) = (r, q);
                }

                // Apply n_rot × 60° rotations: (q, r) → (−r, q+r).
                for _ in 0..n_rot {
                    (q, r) = (-r, q + r);
                }

                if let Some(dst) = to_flat(q, r) {
                    pairs.push((src as u16, dst));
                }
            }

            scatter[sym_idx] = pairs;
        }

        SymTables { scatter }
    }
}

// ── RustReplayBuffer ──────────────────────────────────────────────────────────

/// Ring-buffer replay buffer with 12-fold hexagonal augmentation, exposed to Python.
///
/// Drop-in replacement for `python/training/replay_buffer.py::ReplayBuffer`.
///
/// Construction pre-allocates all storage.  No heap allocation happens after `__new__`.
#[pyclass(name = "RustReplayBuffer")]
pub struct RustReplayBuffer {
    capacity:     usize,
    size:         usize,
    head:         usize, // next write slot

    states:   Vec<u16>,  // f16 bits; flat [capacity × N_PLANES × N_CELLS]
    policies: Vec<f32>,  // flat [capacity × N_ACTIONS]
    outcomes: Vec<f32>,  // flat [capacity]
    game_ids: Vec<i64>,  // flat [capacity]; −1 = untagged

    sym_tables:   SymTables,
    next_game_id: i64,
    rng:          StdRng,
}

#[pymethods]
impl RustReplayBuffer {
    /// Create a new buffer with the given `capacity` (number of positions).
    ///
    /// Pre-allocates all storage.  Building the symmetry tables is O(N_CELLS × N_SYMS) ≈ 4 µs.
    #[new]
    pub fn new(capacity: usize) -> Self {
        RustReplayBuffer {
            capacity,
            size: 0,
            head: 0,
            states:   vec![0u16; capacity * STATE_STRIDE],
            policies: vec![0.0f32; capacity * POLICY_STRIDE],
            outcomes: vec![0.0f32; capacity],
            game_ids: vec![-1i64; capacity],
            sym_tables: SymTables::new(),
            next_game_id: 0,
            rng: StdRng::from_entropy(),
        }
    }

    // ── Monotonic ID counter ──────────────────────────────────────────────────

    /// Return a fresh position ID and advance the internal counter.
    ///
    /// Call once per board position (not once per cluster).  Pass the returned
    /// ID to every cluster's `push()` so the sampler can enforce the
    /// Multi-Window correlation guard.
    pub fn next_game_id(&mut self) -> i64 {
        let id = self.next_game_id;
        self.next_game_id += 1;
        id
    }

    // ── Write ─────────────────────────────────────────────────────────────────

    /// Store a single (state, policy, outcome) triple.
    ///
    /// Args:
    ///     state:   float16 numpy array of shape (18, 19, 19)
    ///     policy:  float32 numpy array of shape (362,)
    ///     outcome: scalar float32  (−1 / 0 / +1)
    ///     game_id: position tag for correlation guard (from `next_game_id()`); default −1
    #[pyo3(signature = (state, policy, outcome, game_id = -1))]
    pub fn push(
        &mut self,
        state:   PyReadonlyArray3<f16>,
        policy:  PyReadonlyArray1<f32>,
        outcome: f32,
        game_id: i64,
    ) -> PyResult<()> {
        let state_slice  = state.as_slice()
            .map_err(|e| PyValueError::new_err(format!("state not contiguous: {e}")))?;
        let policy_slice = policy.as_slice()
            .map_err(|e| PyValueError::new_err(format!("policy not contiguous: {e}")))?;

        if state_slice.len() != STATE_STRIDE {
            return Err(PyValueError::new_err(format!(
                "state must have {} elements (18×19×19), got {}", STATE_STRIDE, state_slice.len()
            )));
        }
        if policy_slice.len() != POLICY_STRIDE {
            return Err(PyValueError::new_err(format!(
                "policy must have {} elements (362), got {}", POLICY_STRIDE, policy_slice.len()
            )));
        }

        let slot = self.head;

        // Copy state as raw f16 bits.
        let dst = &mut self.states[slot * STATE_STRIDE..(slot + 1) * STATE_STRIDE];
        for (d, s) in dst.iter_mut().zip(state_slice.iter()) {
            *d = s.to_bits();
        }

        self.policies[slot * POLICY_STRIDE..(slot + 1) * POLICY_STRIDE]
            .copy_from_slice(policy_slice);
        self.outcomes[slot] = outcome;
        self.game_ids[slot] = game_id;

        self.head = (self.head + 1) % self.capacity;
        self.size  = (self.size + 1).min(self.capacity);
        Ok(())
    }

    /// Store all positions from a completed game efficiently.
    ///
    /// Handles ring-buffer wrap-around correctly.
    ///
    /// Args:
    ///     states:   float16 numpy array of shape (T, 18, 19, 19)
    ///     policies: float32 numpy array of shape (T, 362)
    ///     outcomes: float32 numpy array of shape (T,)
    ///     game_id:  shared position tag for all T entries; default −1
    #[pyo3(signature = (states, policies, outcomes, game_id = -1))]
    pub fn push_game(
        &mut self,
        states:   PyReadonlyArray4<f16>,
        policies: PyReadonlyArray2<f32>,
        outcomes: PyReadonlyArray1<f32>,
        game_id:  i64,
    ) -> PyResult<()> {
        let states_s   = states.as_slice()
            .map_err(|e| PyValueError::new_err(format!("states not contiguous: {e}")))?;
        let policies_s = policies.as_slice()
            .map_err(|e| PyValueError::new_err(format!("policies not contiguous: {e}")))?;
        let outcomes_s = outcomes.as_slice()
            .map_err(|e| PyValueError::new_err(format!("outcomes not contiguous: {e}")))?;

        let t = outcomes_s.len();
        if t == 0 { return Ok(()); }

        if states_s.len()   != t * STATE_STRIDE  { return Err(PyValueError::new_err("states shape mismatch")); }
        if policies_s.len() != t * POLICY_STRIDE { return Err(PyValueError::new_err("policies shape mismatch")); }

        for i in 0..t {
            let slot = (self.head + i) % self.capacity;

            // State: convert f16 → u16 bits
            let src_state = &states_s[i * STATE_STRIDE..(i + 1) * STATE_STRIDE];
            let dst_state = &mut self.states[slot * STATE_STRIDE..(slot + 1) * STATE_STRIDE];
            for (d, s) in dst_state.iter_mut().zip(src_state.iter()) {
                *d = s.to_bits();
            }

            // Policy: direct f32 copy
            let src_pol = &policies_s[i * POLICY_STRIDE..(i + 1) * POLICY_STRIDE];
            self.policies[slot * POLICY_STRIDE..(slot + 1) * POLICY_STRIDE]
                .copy_from_slice(src_pol);

            self.outcomes[slot] = outcomes_s[i];
            self.game_ids[slot] = game_id;
        }

        self.head = (self.head + t) % self.capacity;
        self.size  = (self.size + t).min(self.capacity);
        Ok(())
    }

    // ── Read ──────────────────────────────────────────────────────────────────

    /// Sample `batch_size` entries, optionally with random 12-fold hex augmentation.
    ///
    /// Applies the Multi-Window correlation guard: entries sharing the same `game_id`
    /// (i.e. different cluster windows of the same board position) are never placed
    /// in the same batch together.  Falls back to plain uniform sampling when all
    /// game_ids are −1 (e.g. data loaded without tagging).
    ///
    /// Returns:
    ///     states:   float16 numpy array of shape (batch_size, 18, 19, 19)
    ///     policies: float32 numpy array of shape (batch_size, 362)
    ///     outcomes: float32 numpy array of shape (batch_size,)
    ///
    /// States are returned as float16 to match the Python ReplayBuffer interface and
    /// minimise output bandwidth.  Convert to float32 on the GPU if needed.
    pub fn sample_batch<'py>(
        &mut self,
        py:        Python<'py>,
        batch_size: usize,
        augment:    bool,
    ) -> PyResult<(
        Bound<'py, PyArray4<f16>>,
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray1<f32>>,
    )> {
        if self.size == 0 {
            return Err(PyValueError::new_err("Cannot sample from an empty replay buffer"));
        }

        // ── Index selection (with correlation guard) ──────────────────────────
        let use_dedup = self.game_ids[0] != -1;
        let indices   = self.sample_indices(batch_size, use_dedup);

        // ── Allocate output arrays (owned by Python after return) ─────────────
        // States as f16 bits (u16) — no type conversion needed during scatter.
        let mut out_states   = vec![0u16; batch_size * STATE_STRIDE];
        let mut out_policies = vec![0.0f32; batch_size * POLICY_STRIDE];
        let mut out_outcomes = vec![0.0f32; batch_size];

        // ── Fill output ───────────────────────────────────────────────────────
        for (b, &idx) in indices.iter().enumerate() {
            let sym_idx = if augment { self.rng.gen_range(0..N_SYMS) } else { 0 };

            let src_state  = &self.states  [idx * STATE_STRIDE ..(idx + 1) * STATE_STRIDE];
            let src_policy = &self.policies[idx * POLICY_STRIDE..(idx + 1) * POLICY_STRIDE];

            let dst_state  = &mut out_states  [b * STATE_STRIDE..(b + 1) * STATE_STRIDE];
            let dst_policy = &mut out_policies[b * POLICY_STRIDE..(b + 1) * POLICY_STRIDE];

            Self::apply_sym(sym_idx, src_state, src_policy, dst_state, dst_policy, &self.sym_tables);

            out_outcomes[b] = self.outcomes[idx];
        }

        // ── Transmute u16 Vec to f16 Vec and wrap as numpy arrays ─────────────
        // Safety: f16 and u16 have the same size/alignment; every bit pattern is valid for u16,
        // and we only wrote bits that came from valid f16 values stored via push().
        let out_f16: Vec<f16> = unsafe {
            let mut v = std::mem::ManuallyDrop::new(out_states);
            Vec::from_raw_parts(v.as_mut_ptr() as *mut f16, v.len(), v.capacity())
        };

        let states_np = out_f16
            .into_pyarray(py)
            .reshape([batch_size, N_PLANES, BOARD_H, BOARD_W])?;
        let policies_np = out_policies
            .into_pyarray(py)
            .reshape([batch_size, N_ACTIONS])?;
        let outcomes_np = out_outcomes.into_pyarray(py);

        Ok((states_np, policies_np, outcomes_np))
    }

    // ── Resize ─────────────────────────────────────────────────────────────────

    /// Grow the buffer to `new_capacity` positions, preserving all existing data.
    ///
    /// The ring buffer is linearised in-place (oldest entry → slot 0) before
    /// extending.  Raises `ValueError` if `new_capacity <= self.capacity`.
    pub fn resize(&mut self, new_capacity: usize) -> PyResult<()> {
        if new_capacity <= self.capacity {
            return Err(PyValueError::new_err(format!(
                "resize: new_capacity ({}) must be greater than current capacity ({})",
                new_capacity, self.capacity,
            )));
        }

        // Linearise the ring buffer when it has wrapped around.
        if self.size == self.capacity && self.head != 0 {
            self.states[..self.capacity * STATE_STRIDE]
                .rotate_left(self.head * STATE_STRIDE);
            self.policies[..self.capacity * POLICY_STRIDE]
                .rotate_left(self.head * POLICY_STRIDE);
            self.outcomes[..self.capacity]
                .rotate_left(self.head);
            self.game_ids[..self.capacity]
                .rotate_left(self.head);
        }

        // Extend storage to new capacity.
        self.states.resize(new_capacity * STATE_STRIDE, 0u16);
        self.policies.resize(new_capacity * POLICY_STRIDE, 0.0f32);
        self.outcomes.resize(new_capacity, 0.0f32);
        self.game_ids.resize(new_capacity, -1i64);

        self.head = self.size;
        self.capacity = new_capacity;
        Ok(())
    }

    // ── Properties ────────────────────────────────────────────────────────────

    #[getter]
    pub fn size(&self) -> usize { self.size }

    #[getter]
    pub fn capacity(&self) -> usize { self.capacity }
}

// ── Private helpers ───────────────────────────────────────────────────────────

impl RustReplayBuffer {
    /// Sample `batch_size` slot indices, optionally deduplicating by game_id.
    fn sample_indices(&mut self, batch_size: usize, use_dedup: bool) -> Vec<usize> {
        if !use_dedup {
            return (0..batch_size)
                .map(|_| self.rng.gen_range(0..self.size))
                .collect();
        }

        // Correlation guard: at most one cluster per board position per batch.
        // Strategy: sample candidates, replace duplicates up to MAX_RETRIES times.
        const MAX_RETRIES: usize = 8;

        let mut indices: Vec<usize> = (0..batch_size)
            .map(|_| self.rng.gen_range(0..self.size))
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
                let mut candidate = self.rng.gen_range(0..self.size);
                // Try a few times to find a unique one; give up gracefully.
                for _ in 0..16 {
                    let cgid = self.game_ids[candidate];
                    if cgid == -1 || !seen.contains(&cgid) {
                        break;
                    }
                    candidate = self.rng.gen_range(0..self.size);
                }
                *idx = candidate;
                seen.insert(self.game_ids[candidate]);
            }
            if all_unique { break; }
        }

        indices
    }

    /// Apply symmetry `sym_idx` to one (state, policy) sample.
    ///
    /// Scatter-copies from `src_state` / `src_policy` into `dst_state` / `dst_policy`.
    /// Cells that have no valid destination under the transform remain zero (caller-zeroed).
    ///
    /// Loop order — outer = planes, inner = scatter pairs — keeps each 722-byte
    /// plane (361 × u16) resident in L1 cache during the inner scatter pass.
    #[inline]
    fn apply_sym(
        sym_idx:    usize,
        src_state:  &[u16],     // f16 bits, length N_PLANES × N_CELLS
        src_policy: &[f32],     // length N_ACTIONS
        dst_state:  &mut [u16], // f16 bits, length N_PLANES × N_CELLS  (zeroed by caller)
        dst_policy: &mut [f32], // length N_ACTIONS                     (zeroed by caller)
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
    }
}

