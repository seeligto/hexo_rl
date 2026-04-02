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
use std::sync::atomic::{AtomicU64, Ordering};

// ── Weight schedule ──────────────────────────────────────────────────────────

/// A single threshold bracket: games with length < `max_moves` get `weight`.
/// Brackets are evaluated in order; the first match wins.
#[derive(Clone, Debug)]
struct WeightBracket {
    max_moves: u16,   // exclusive upper bound (game_length < max_moves)
    weight:    u16,   // f16-as-u16 bits
}

/// Config-driven weight schedule for game-length-based sampling.
/// Default: all positions have weight 1.0 (uniform sampling).
#[derive(Clone, Debug)]
struct WeightSchedule {
    brackets: Vec<WeightBracket>,
    default_weight: u16, // f16 bits for weight when no bracket matches
}

impl WeightSchedule {
    fn uniform() -> Self {
        WeightSchedule {
            brackets: Vec::new(),
            default_weight: f16::from_f32(1.0).to_bits(),
        }
    }

    /// Look up the weight (as f16 bits) for a given game length.
    #[inline]
    fn weight_for(&self, game_length: u16) -> u16 {
        for b in &self.brackets {
            if game_length < b.max_moves {
                return b.weight;
            }
        }
        self.default_weight
    }
}

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
    weights:  Vec<u16>,  // f16-as-u16 bits; flat [capacity]; sampling weight per position

    sym_tables:      SymTables,
    weight_schedule: WeightSchedule,
    next_game_id:    i64,
    rng:             StdRng,

    /// Lock-free weight histogram for O(1) dashboard stats.
    ///
    /// Bucket boundaries (f32 weight):
    ///   [0] < 0.30  → short-game tier  (~0.15)
    ///   [1] 0.30–0.75 → medium-game tier (~0.50)
    ///   [2] ≥ 0.75  → full-weight tier  (~1.0)
    ///
    /// Incremented on push, decremented on overwrite.
    /// Read with Relaxed ordering — approximate counts for display only.
    weight_buckets: [AtomicU64; 3],
}

#[pymethods]
impl RustReplayBuffer {
    /// Create a new buffer with the given `capacity` (number of positions).
    ///
    /// Pre-allocates all storage.  Building the symmetry tables is O(N_CELLS × N_SYMS) ≈ 4 µs.
    #[new]
    pub fn new(capacity: usize) -> Self {
        let default_w = f16::from_f32(1.0).to_bits();
        RustReplayBuffer {
            capacity,
            size: 0,
            head: 0,
            states:   vec![0u16; capacity * STATE_STRIDE],
            policies: vec![0.0f32; capacity * POLICY_STRIDE],
            outcomes: vec![0.0f32; capacity],
            game_ids: vec![-1i64; capacity],
            weights:  vec![default_w; capacity],
            sym_tables: SymTables::new(),
            weight_schedule: WeightSchedule::uniform(),
            next_game_id: 0,
            rng: StdRng::from_os_rng(),
            weight_buckets: [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)],
        }
    }

    // ── Dashboard stats ───────────────────────────────────────────────────────

    /// Return (size, capacity, weight_histogram) for dashboard display.
    ///
    /// `weight_histogram` is a length-3 list: counts of positions in each
    /// weight tier (low/medium/full).  Reads lock-free atomic counters — O(1),
    /// never blocks push() or sample_batch().
    pub fn get_buffer_stats(&self) -> (usize, usize, Vec<u64>) {
        let histogram = vec![
            self.weight_buckets[0].load(Ordering::Relaxed),
            self.weight_buckets[1].load(Ordering::Relaxed),
            self.weight_buckets[2].load(Ordering::Relaxed),
        ];
        (self.size, self.capacity, histogram)
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
    ///     state:       float16 numpy array of shape (18, 19, 19)
    ///     policy:      float32 numpy array of shape (362,)
    ///     outcome:     scalar float32  (−1 / 0 / +1)
    ///     game_id:     position tag for correlation guard (from `next_game_id()`); default −1
    ///     game_length: total compound moves in the originating game; default 0 (= weight 1.0)
    #[pyo3(signature = (state, policy, outcome, game_id = -1, game_length = 0))]
    pub fn push(
        &mut self,
        state:       PyReadonlyArray3<f16>,
        policy:      PyReadonlyArray1<f32>,
        outcome:     f32,
        game_id:     i64,
        game_length: u16,
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

        self.policies[slot * POLICY_STRIDE..(slot + 1) * POLICY_STRIDE]
            .copy_from_slice(policy_slice);
        self.outcomes[slot] = outcome;
        self.game_ids[slot] = game_id;
        self.weights[slot] = if game_length == 0 {
            f16::from_f32(1.0).to_bits()
        } else {
            self.weight_schedule.weight_for(game_length)
        };

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
    ///     states:      float16 numpy array of shape (T, 18, 19, 19)
    ///     policies:    float32 numpy array of shape (T, 362)
    ///     outcomes:    float32 numpy array of shape (T,)
    ///     game_id:     shared position tag for all T entries; default −1
    ///     game_length: total compound moves in the originating game; default 0 (= weight 1.0)
    #[pyo3(signature = (states, policies, outcomes, game_id = -1, game_length = 0))]
    pub fn push_game(
        &mut self,
        states:      PyReadonlyArray4<f16>,
        policies:    PyReadonlyArray2<f32>,
        outcomes:    PyReadonlyArray1<f32>,
        game_id:     i64,
        game_length: u16,
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

            // Policy: direct f32 copy
            let src_pol = &policies_s[i * POLICY_STRIDE..(i + 1) * POLICY_STRIDE];
            self.policies[slot * POLICY_STRIDE..(slot + 1) * POLICY_STRIDE]
                .copy_from_slice(src_pol);

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
            let sym_idx = if augment { self.rng.random_range(0..N_SYMS) } else { 0 };

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
            self.weights[..self.capacity]
                .rotate_left(self.head);
        }

        // Extend storage to new capacity.
        let default_w = f16::from_f32(1.0).to_bits();
        self.states.resize(new_capacity * STATE_STRIDE, 0u16);
        self.policies.resize(new_capacity * POLICY_STRIDE, 0.0f32);
        self.outcomes.resize(new_capacity, 0.0f32);
        self.game_ids.resize(new_capacity, -1i64);
        self.weights.resize(new_capacity, default_w);

        self.head = self.size;
        self.capacity = new_capacity;
        Ok(())
    }

    /// Set the game-length weight schedule from Python config.
    ///
    /// Args:
    ///     thresholds: list of exclusive upper bounds (must be sorted ascending)
    ///     weights:    list of f32 weights, same length as thresholds
    ///     default_weight: weight for games >= all thresholds (typically 1.0)
    ///
    /// Example (from training.yaml):
    ///     buf.set_weight_schedule([10, 25], [0.15, 0.50], 1.0)
    ///     # game < 10 moves → 0.15, 10–24 → 0.50, 25+ → 1.0
    pub fn set_weight_schedule(
        &mut self,
        thresholds:     Vec<u16>,
        weights:        Vec<f32>,
        default_weight: f32,
    ) -> PyResult<()> {
        if thresholds.len() != weights.len() {
            return Err(PyValueError::new_err(
                "thresholds and weights must have the same length"
            ));
        }
        let brackets: Vec<WeightBracket> = thresholds.iter().zip(weights.iter())
            .map(|(&t, &w)| WeightBracket {
                max_moves: t,
                weight: f16::from_f32(w).to_bits(),
            })
            .collect();
        self.weight_schedule = WeightSchedule {
            brackets,
            default_weight: f16::from_f32(default_weight).to_bits(),
        };
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
    /// Map an f16 weight (stored as bits) to a histogram bucket index.
    ///
    /// Bucket 0: weight < 0.30  (short-game tier, ~0.15)
    /// Bucket 1: 0.30 ≤ w < 0.75 (medium-game tier, ~0.50)
    /// Bucket 2: weight ≥ 0.75  (full-weight tier, ~1.0)
    #[inline]
    fn weight_bucket(w_bits: u16) -> usize {
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
    fn weighted_sample_one(&mut self) -> usize {
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
    fn sample_indices(&mut self, batch_size: usize, use_dedup: bool) -> Vec<usize> {
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

    /// Push a position directly from Rust (no PyO3 / numpy).
    /// Used only by tests.
    #[cfg(test)]
    fn push_raw(&mut self, outcome: f32, game_length: u16) {
        let slot = self.head;

        // Decrement old bucket if overwriting.
        if self.size == self.capacity {
            let old_bucket = Self::weight_bucket(self.weights[slot]);
            self.weight_buckets[old_bucket].fetch_sub(1, Ordering::Relaxed);
        }

        // Zero state/policy (content doesn't matter for weight tests).
        let s_start = slot * STATE_STRIDE;
        for v in &mut self.states[s_start..s_start + STATE_STRIDE] { *v = 0; }
        let p_start = slot * POLICY_STRIDE;
        for v in &mut self.policies[p_start..p_start + POLICY_STRIDE] { *v = 0.0; }
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

    /// Push positions from short (5 moves), medium (15), and long (40) games,
    /// sample 10,000 times, and verify short-game positions appear ~0.15x as
    /// often as long-game positions.
    #[test]
    fn test_weighted_sampling_distribution() {
        let default_w = f16::from_f32(1.0).to_bits();
        let mut buf = RustReplayBuffer {
            capacity: 300,
            size: 0,
            head: 0,
            states:   vec![0u16; 300 * STATE_STRIDE],
            policies: vec![0.0f32; 300 * POLICY_STRIDE],
            outcomes: vec![0.0f32; 300],
            game_ids: vec![-1i64; 300],
            weights:  vec![default_w; 300],
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
}

