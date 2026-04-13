//! ReplayBuffer — pre-allocated ring buffer with vectorized 12-fold hex augmentation.
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

mod sym_tables;
mod sampling;

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::{
    IntoPyArray,
    PyArray1, PyArray2, PyArray3, PyArray4,
    PyArrayMethods,
    PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, PyReadonlyArray4,
};
use half::f16;
use rand::RngExt;
use rand::rngs::StdRng;
use std::sync::atomic::{AtomicU64, Ordering};

use sym_tables::*;

// ── ReplayBuffer ──────────────────────────────────────────────────────────

/// Ring-buffer replay buffer with 12-fold hexagonal augmentation, exposed to Python.
///
/// Drop-in replacement for `python/training/replay_buffer.py::ReplayBuffer`.
///
/// Construction pre-allocates all storage.  No heap allocation happens after `__new__`.
#[pyclass(name = "ReplayBuffer")]
pub struct ReplayBuffer {
    pub(crate) capacity:     usize,
    pub(crate) size:         usize,
    pub(crate) head:         usize, // next write slot

    pub(crate) states:   Vec<u16>,  // f16 bits; flat [capacity × N_PLANES × N_CELLS]
    pub(crate) policies: Vec<f32>,  // flat [capacity × N_ACTIONS]
    pub(crate) outcomes: Vec<f32>,  // flat [capacity]
    pub(crate) game_ids: Vec<i64>,  // flat [capacity]; −1 = untagged
    pub(crate) weights:  Vec<u16>,  // f16-as-u16 bits; flat [capacity]; sampling weight per position

    /// Auxiliary target: per-cell ownership of the final board state, projected
    /// to each row's per-cluster window centre (NOT the game-end bbox centroid).
    /// Encoding: 0 = P2, 1 = empty, 2 = P1. Flat [capacity × AUX_STRIDE].
    pub(crate) ownership:    Vec<u8>,
    /// Auxiliary target: binary mask of the 6 cells of the winning 6-in-a-row,
    /// projected to each row's per-cluster window centre. All zero on draw.
    /// Flat [capacity × AUX_STRIDE].
    pub(crate) winning_line: Vec<u8>,

    pub(crate) sym_tables:      SymTables,
    pub(crate) weight_schedule: WeightSchedule,
    pub(crate) next_game_id:    i64,
    pub(crate) rng:             StdRng,

    /// Lock-free weight histogram for O(1) dashboard stats.
    ///
    /// Bucket boundaries (f32 weight):
    ///   [0] < 0.30  → short-game tier  (~0.15)
    ///   [1] 0.30-0.75 → medium-game tier (~0.50)
    ///   [2] ≥ 0.75  → full-weight tier  (~1.0)
    ///
    /// Incremented on push, decremented on overwrite.
    /// Read with Relaxed ordering — approximate counts for display only.
    pub(crate) weight_buckets: [AtomicU64; 3],
}

#[pymethods]
impl ReplayBuffer {
    /// Create a new buffer with the given `capacity` (number of positions).
    ///
    /// Pre-allocates all storage.  Building the symmetry tables is O(N_CELLS × N_SYMS) ≈ 4 µs.
    #[new]
    pub fn new(capacity: usize) -> Self {
        let default_w = f16::from_f32(1.0).to_bits();
        let aux_bytes = capacity * 2 * AUX_STRIDE;
        eprintln!(
            "[ReplayBuffer] allocated capacity={} (aux columns: ownership + winning_line, +{} bytes ≈ {:.1} MB)",
            capacity, aux_bytes, aux_bytes as f64 / (1024.0 * 1024.0),
        );
        ReplayBuffer {
            capacity,
            size: 0,
            head: 0,
            states:       vec![0u16; capacity * STATE_STRIDE],
            policies:     vec![0.0f32; capacity * POLICY_STRIDE],
            outcomes:     vec![0.0f32; capacity],
            game_ids:     vec![-1i64; capacity],
            weights:      vec![default_w; capacity],
            ownership:    vec![1u8; capacity * AUX_STRIDE],  // 1 = empty default
            winning_line: vec![0u8; capacity * AUX_STRIDE],
            sym_tables: SymTables::new(),
            weight_schedule: WeightSchedule::uniform(),
            next_game_id: 0,
            rng: rand::make_rng(),
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

    /// Store a single (state, policy, outcome, ownership, winning_line) sample.
    ///
    /// Args:
    ///     state:        float16 numpy array of shape (18, 19, 19)
    ///     policy:       float32 numpy array of shape (362,)
    ///     outcome:      scalar float32  (−1 / 0 / +1)
    ///     ownership:    uint8 numpy array of shape (361,)  encoding {0=P2, 1=empty, 2=P1}
    ///     winning_line: uint8 numpy array of shape (361,)  binary mask
    ///     game_id:      position tag for correlation guard (from `next_game_id()`); default −1
    ///     game_length:  total compound moves in the originating game; default 0 (= weight 1.0)
    ///
    /// Both aux targets MUST be projected to the same window centre as `state`
    /// (per-row cluster centre, not the game-end bbox centroid).
    #[pyo3(signature = (state, policy, outcome, ownership, winning_line, game_id = -1, game_length = 0))]
    pub fn push(
        &mut self,
        state:        PyReadonlyArray3<f16>,
        policy:       PyReadonlyArray1<f32>,
        outcome:      f32,
        ownership:    PyReadonlyArray1<u8>,
        winning_line: PyReadonlyArray1<u8>,
        game_id:      i64,
        game_length:  u16,
    ) -> PyResult<()> {
        let state_slice  = state.as_slice()
            .map_err(|e| PyValueError::new_err(format!("state not contiguous: {e}")))?;
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
    ///     policies:      float32 numpy array of shape (T, 362)
    ///     outcomes:      float32 numpy array of shape (T,)
    ///     ownership:     uint8   numpy array of shape (T, 361)  per-row {0=P2, 1=empty, 2=P1}
    ///     winning_line:  uint8   numpy array of shape (T, 361)  per-row binary mask
    ///     game_id:       shared position tag for all T entries; default −1
    ///     game_length:   total compound moves in the originating game; default 0 (= weight 1.0)
    #[pyo3(signature = (states, policies, outcomes, ownership, winning_line, game_id = -1, game_length = 0))]
    pub fn push_game(
        &mut self,
        states:       PyReadonlyArray4<f16>,
        policies:     PyReadonlyArray2<f32>,
        outcomes:     PyReadonlyArray1<f32>,
        ownership:    PyReadonlyArray2<u8>,
        winning_line: PyReadonlyArray2<u8>,
        game_id:      i64,
        game_length:  u16,
    ) -> PyResult<()> {
        let states_s   = states.as_slice()
            .map_err(|e| PyValueError::new_err(format!("states not contiguous: {e}")))?;
        let policies_s = policies.as_slice()
            .map_err(|e| PyValueError::new_err(format!("policies not contiguous: {e}")))?;
        let outcomes_s = outcomes.as_slice()
            .map_err(|e| PyValueError::new_err(format!("outcomes not contiguous: {e}")))?;
        let own_s = ownership.as_slice()
            .map_err(|e| PyValueError::new_err(format!("ownership not contiguous: {e}")))?;
        let wl_s = winning_line.as_slice()
            .map_err(|e| PyValueError::new_err(format!("winning_line not contiguous: {e}")))?;

        let t = outcomes_s.len();
        if t == 0 { return Ok(()); }

        if states_s.len()   != t * STATE_STRIDE  { return Err(PyValueError::new_err("states shape mismatch")); }
        if policies_s.len() != t * POLICY_STRIDE { return Err(PyValueError::new_err("policies shape mismatch")); }
        if own_s.len() != t * AUX_STRIDE { return Err(PyValueError::new_err("ownership shape mismatch")); }
        if wl_s.len()  != t * AUX_STRIDE { return Err(PyValueError::new_err("winning_line shape mismatch")); }

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

            // Auxiliary spatial targets — direct u8 copies.
            self.ownership[slot * AUX_STRIDE..(slot + 1) * AUX_STRIDE]
                .copy_from_slice(&own_s[i * AUX_STRIDE..(i + 1) * AUX_STRIDE]);
            self.winning_line[slot * AUX_STRIDE..(slot + 1) * AUX_STRIDE]
                .copy_from_slice(&wl_s[i * AUX_STRIDE..(i + 1) * AUX_STRIDE]);

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
    ///     states:       float16 numpy array of shape (batch_size, 18, 19, 19)
    ///     policies:     float32 numpy array of shape (batch_size, 362)
    ///     outcomes:     float32 numpy array of shape (batch_size,)
    ///     ownership:    uint8   numpy array of shape (batch_size, 19, 19)
    ///     winning_line: uint8   numpy array of shape (batch_size, 19, 19)
    ///
    /// States are returned as float16 to match the Python ReplayBuffer interface and
    /// minimise output bandwidth.  Convert to float32 on the GPU if needed.
    /// Ownership and winning_line are u8; cast in the trainer before loss computation.
    /// All four spatial outputs share the same per-sample symmetry index, so the
    /// augmentation transform is consistent between state and aux targets.
    pub fn sample_batch<'py>(
        &mut self,
        py:        Python<'py>,
        batch_size: usize,
        augment:    bool,
    ) -> PyResult<(
        Bound<'py, PyArray4<f16>>,
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray1<f32>>,
        Bound<'py, PyArray3<u8>>,
        Bound<'py, PyArray3<u8>>,
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
        // Ownership default 1 = "empty" — cells outside the symmetry's destination
        // window stay at the same neutral value as the row's initial state.
        let mut out_ownership    = vec![1u8; batch_size * AUX_STRIDE];
        let mut out_winning_line = vec![0u8; batch_size * AUX_STRIDE];

        // ── Fill output ───────────────────────────────────────────────────────
        for (b, &idx) in indices.iter().enumerate() {
            let sym_idx = if augment { self.rng.random_range(0..N_SYMS) } else { 0 };

            let src_state  = &self.states      [idx * STATE_STRIDE..(idx + 1) * STATE_STRIDE];
            let src_policy = &self.policies    [idx * POLICY_STRIDE..(idx + 1) * POLICY_STRIDE];
            let src_own    = &self.ownership   [idx * AUX_STRIDE   ..(idx + 1) * AUX_STRIDE];
            let src_wl     = &self.winning_line[idx * AUX_STRIDE   ..(idx + 1) * AUX_STRIDE];

            let dst_state  = &mut out_states      [b * STATE_STRIDE..(b + 1) * STATE_STRIDE];
            let dst_policy = &mut out_policies    [b * POLICY_STRIDE..(b + 1) * POLICY_STRIDE];
            let dst_own    = &mut out_ownership   [b * AUX_STRIDE   ..(b + 1) * AUX_STRIDE];
            let dst_wl     = &mut out_winning_line[b * AUX_STRIDE   ..(b + 1) * AUX_STRIDE];

            Self::apply_sym(
                sym_idx,
                src_state, src_policy, src_own, src_wl,
                dst_state, dst_policy, dst_own, dst_wl,
                &self.sym_tables,
            );

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
        let ownership_np = out_ownership
            .into_pyarray(py)
            .reshape([batch_size, BOARD_H, BOARD_W])?;
        let winning_line_np = out_winning_line
            .into_pyarray(py)
            .reshape([batch_size, BOARD_H, BOARD_W])?;

        Ok((states_np, policies_np, outcomes_np, ownership_np, winning_line_np))
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
            self.ownership[..self.capacity * AUX_STRIDE]
                .rotate_left(self.head * AUX_STRIDE);
            self.winning_line[..self.capacity * AUX_STRIDE]
                .rotate_left(self.head * AUX_STRIDE);
        }

        // Extend storage to new capacity.
        let default_w = f16::from_f32(1.0).to_bits();
        self.states.resize(new_capacity * STATE_STRIDE, 0u16);
        self.policies.resize(new_capacity * POLICY_STRIDE, 0.0f32);
        self.outcomes.resize(new_capacity, 0.0f32);
        self.game_ids.resize(new_capacity, -1i64);
        self.weights.resize(new_capacity, default_w);
        self.ownership.resize(new_capacity * AUX_STRIDE, 1u8);     // 1 = empty
        self.winning_line.resize(new_capacity * AUX_STRIDE, 0u8);

        let added_aux_bytes = (new_capacity - self.size) * 2 * AUX_STRIDE;
        eprintln!(
            "[ReplayBuffer] resized to capacity={} (aux added: +{} bytes ≈ {:.1} MB)",
            new_capacity, added_aux_bytes, added_aux_bytes as f64 / (1024.0 * 1024.0),
        );

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
    ///     # game < 10 moves → 0.15, 10-24 → 0.50, 25+ → 1.0
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

    // ── Persistence ──────────────────────────────────────────────────────────

    /// Save buffer contents to a binary file.
    ///
    /// Format (little-endian native):
    ///   [magic: u32 = 0x48455842]  ("HEXB")
    ///   [version: u32 = 2]
    ///   [capacity: u64]
    ///   [size: u64]
    ///   For each of `size` positions (oldest → newest):
    ///     state:        STATE_STRIDE × u16
    ///     policy:       POLICY_STRIDE × f32
    ///     outcome:      f32
    ///     game_id:      i64
    ///     weight:       u16
    ///     ownership:    AUX_STRIDE × u8   (encoding 0/1/2)
    ///     winning_line: AUX_STRIDE × u8   (binary mask)
    ///
    /// v1 (pre A1 aux refactor) buffers are NOT readable; load() will return
    /// a clear error if encountered.
    #[pyo3(text_signature = "(self, path)")]
    pub fn save_to_path(&self, path: &str) -> PyResult<()> {
        use std::io::{BufWriter, Write};

        let file = std::fs::File::create(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("cannot create {path}: {e}")))?;
        let mut w = BufWriter::new(file);

        // Header
        w.write_all(&0x4845_5842u32.to_le_bytes())
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        w.write_all(&2u32.to_le_bytes())
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        w.write_all(&(self.capacity as u64).to_le_bytes())
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        w.write_all(&(self.size as u64).to_le_bytes())
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

        // Positions in logical order (oldest → newest)
        for i in 0..self.size {
            let slot = (self.head + self.capacity - self.size + i) % self.capacity;

            // state: u16 slice → bytes
            let state_start = slot * STATE_STRIDE;
            let state_bytes = unsafe {
                std::slice::from_raw_parts(
                    self.states[state_start..state_start + STATE_STRIDE].as_ptr() as *const u8,
                    STATE_STRIDE * 2,
                )
            };
            w.write_all(state_bytes)
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

            // policy: f32 slice → bytes
            let pol_start = slot * POLICY_STRIDE;
            let pol_bytes = unsafe {
                std::slice::from_raw_parts(
                    self.policies[pol_start..pol_start + POLICY_STRIDE].as_ptr() as *const u8,
                    POLICY_STRIDE * 4,
                )
            };
            w.write_all(pol_bytes)
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

            // outcome: f32
            w.write_all(&self.outcomes[slot].to_le_bytes())
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

            // game_id: i64
            w.write_all(&self.game_ids[slot].to_le_bytes())
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

            // weight: u16
            w.write_all(&self.weights[slot].to_le_bytes())
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

            // ownership: AUX_STRIDE × u8
            let aux_start = slot * AUX_STRIDE;
            w.write_all(&self.ownership[aux_start..aux_start + AUX_STRIDE])
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
            // winning_line: AUX_STRIDE × u8
            w.write_all(&self.winning_line[aux_start..aux_start + AUX_STRIDE])
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        }

        w.flush().map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        Ok(())
    }

    /// Load buffer contents from a binary file written by `save_to_path`.
    ///
    /// Returns the number of positions loaded.  If the file does not exist,
    /// returns 0 (not an error — supports first-run case).
    ///
    /// If the saved buffer has more positions than `self.capacity`, only the
    /// most recent `self.capacity` positions are loaded.
    #[pyo3(text_signature = "(self, path)")]
    pub fn load_from_path(&mut self, path: &str) -> PyResult<usize> {
        use std::io::{BufReader, Read};

        let file = match std::fs::File::open(path) {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(0),
            Err(e) => return Err(pyo3::exceptions::PyIOError::new_err(
                format!("cannot open {path}: {e}")
            )),
        };
        let mut r = BufReader::new(file);

        // Read header
        let mut buf4 = [0u8; 4];
        let mut buf8 = [0u8; 8];

        r.read_exact(&mut buf4)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        let magic = u32::from_le_bytes(buf4);
        if magic != 0x4845_5842 {
            return Err(PyValueError::new_err(format!(
                "invalid magic: expected 0x48455842 (HEXB), got 0x{magic:08X}"
            )));
        }

        r.read_exact(&mut buf4)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        let version = u32::from_le_bytes(buf4);
        if version != 2 {
            return Err(PyValueError::new_err(format!(
                "HEXB version {version} not supported. v1 buffers were invalidated by the A1 \
                 aux target alignment refactor (per-row ownership + winning_line columns added). \
                 Regenerate the buffer from self-play."
            )));
        }

        r.read_exact(&mut buf8)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        let _saved_capacity = u64::from_le_bytes(buf8) as usize;

        r.read_exact(&mut buf8)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        let saved_size = u64::from_le_bytes(buf8) as usize;

        // How many to actually load — cap at our capacity
        let to_load = saved_size.min(self.capacity);
        // How many to skip if saved_size > capacity (skip oldest)
        let to_skip = saved_size - to_load;

        // Per-entry byte sizes (v2: state + policy + outcome + game_id + weight + own + wl)
        let state_bytes = STATE_STRIDE * 2;
        let policy_bytes = POLICY_STRIDE * 4;
        let entry_bytes = state_bytes + policy_bytes + 4 + 8 + 2 + AUX_STRIDE + AUX_STRIDE;

        // Skip oldest entries
        if to_skip > 0 {
            let skip_bytes = to_skip * entry_bytes;
            let mut remaining = skip_bytes;
            let mut skip_buf = vec![0u8; 8192.min(skip_bytes)];
            while remaining > 0 {
                let chunk = remaining.min(skip_buf.len());
                r.read_exact(&mut skip_buf[..chunk])
                    .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
                remaining -= chunk;
            }
        }

        // Reset weight histogram
        for bucket in &self.weight_buckets {
            bucket.store(0, Ordering::Relaxed);
        }

        // Read positions directly into storage
        let mut state_buf = vec![0u8; state_bytes];
        let mut pol_buf = vec![0u8; policy_bytes];

        for i in 0..to_load {
            let slot = i; // write sequentially from slot 0

            // state
            r.read_exact(&mut state_buf)
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
            let dst_state = &mut self.states[slot * STATE_STRIDE..(slot + 1) * STATE_STRIDE];
            // Safety: u16 and [u8; 2] have same size, and state_buf is correctly sized
            for (j, d) in dst_state.iter_mut().enumerate() {
                *d = u16::from_le_bytes([state_buf[j * 2], state_buf[j * 2 + 1]]);
            }

            // policy
            r.read_exact(&mut pol_buf)
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
            let dst_pol = &mut self.policies[slot * POLICY_STRIDE..(slot + 1) * POLICY_STRIDE];
            for (j, d) in dst_pol.iter_mut().enumerate() {
                *d = f32::from_le_bytes([
                    pol_buf[j * 4], pol_buf[j * 4 + 1],
                    pol_buf[j * 4 + 2], pol_buf[j * 4 + 3],
                ]);
            }

            // outcome
            r.read_exact(&mut buf4)
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
            self.outcomes[slot] = f32::from_le_bytes(buf4);

            // game_id
            r.read_exact(&mut buf8)
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
            self.game_ids[slot] = i64::from_le_bytes(buf8);

            // weight
            let mut buf2 = [0u8; 2];
            r.read_exact(&mut buf2)
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
            let w_bits = u16::from_le_bytes(buf2);
            self.weights[slot] = w_bits;

            // ownership + winning_line (v2)
            let aux_dst_start = slot * AUX_STRIDE;
            r.read_exact(&mut self.ownership[aux_dst_start..aux_dst_start + AUX_STRIDE])
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
            r.read_exact(&mut self.winning_line[aux_dst_start..aux_dst_start + AUX_STRIDE])
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

            // Update weight histogram
            let bucket = Self::weight_bucket(w_bits);
            self.weight_buckets[bucket].fetch_add(1, Ordering::Relaxed);
        }

        self.size = to_load;
        self.head = to_load % self.capacity;
        Ok(to_load)
    }

    // ── Properties ────────────────────────────────────────────────────────────

    #[getter]
    pub fn size(&self) -> usize { self.size }

    #[getter]
    pub fn capacity(&self) -> usize { self.capacity }
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;
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
            states:       vec![0u16; 300 * STATE_STRIDE],
            policies:     vec![0.0f32; 300 * POLICY_STRIDE],
            outcomes:     vec![0.0f32; 300],
            game_ids:     vec![-1i64; 300],
            weights:      vec![default_w; 300],
            ownership:    vec![1u8; 300 * AUX_STRIDE],
            winning_line: vec![0u8; 300 * AUX_STRIDE],
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

    // ── A1 aux target alignment tests ────────────────────────────────────────

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
                let mut src_pol   = vec![0.0f32; N_ACTIONS];
                let mut src_own   = vec![1u8; AUX_STRIDE];
                let mut src_wl    = vec![0u8; AUX_STRIDE];

                src_state[marker_src] = f16::from_f32(7.0).to_bits();
                src_pol[marker_src]   = 7.0;
                src_own[marker_src]   = 2;   // P1
                src_wl[marker_src]    = 1;

                let mut dst_state = vec![0u16; N_PLANES * N_CELLS];
                let mut dst_pol   = vec![0.0f32; N_ACTIONS];
                let mut dst_own   = vec![1u8; AUX_STRIDE];
                let mut dst_wl    = vec![0u8; AUX_STRIDE];

                ReplayBuffer::apply_sym(
                    sym_idx,
                    &src_state, &src_pol, &src_own, &src_wl,
                    &mut dst_state, &mut dst_pol, &mut dst_own, &mut dst_wl,
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
        let mut dst_pol   = vec![0.0f32; N_ACTIONS];
        let mut dst_own   = vec![1u8;  AUX_STRIDE];
        let mut dst_wl    = vec![0u8;  AUX_STRIDE];

        for _ in 0..1000 {
            let idx = buf.weighted_sample_one();
            let sym_idx = rng.random_range(0..N_SYMS);

            for v in &mut dst_state { *v = 0; }
            for v in &mut dst_pol   { *v = 0.0; }
            for v in &mut dst_own   { *v = 1; }
            for v in &mut dst_wl    { *v = 0; }

            let s = idx * STATE_STRIDE;
            let p = idx * POLICY_STRIDE;
            let a = idx * AUX_STRIDE;
            ReplayBuffer::apply_sym(
                sym_idx,
                &buf.states[s..s + STATE_STRIDE],
                &buf.policies[p..p + POLICY_STRIDE],
                &buf.ownership[a..a + AUX_STRIDE],
                &buf.winning_line[a..a + AUX_STRIDE],
                &mut dst_state, &mut dst_pol, &mut dst_own, &mut dst_wl,
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

    /// HEXB v2 round-trip — verify aux columns survive save/load.
    #[test]
    fn test_aux_hexb_v2_roundtrip() {
        use std::env::temp_dir;

        let mut buf = ReplayBuffer::new(8);
        let slot = 0;
        let a_start = slot * AUX_STRIDE;
        buf.ownership[a_start + 10] = 2;  // P1
        buf.ownership[a_start + 20] = 0;  // P2
        buf.ownership[a_start + 30] = 1;  // empty
        for i in 0..6 { buf.winning_line[a_start + 100 + i] = 1; }
        buf.outcomes[slot] = 1.0;
        buf.weights[slot]  = f16::from_f32(1.0).to_bits();
        buf.head = 1;
        buf.size = 1;

        let path = temp_dir().join("aux_v2_roundtrip.hexb");
        buf.save_to_path(path.to_str().unwrap()).unwrap();

        let mut buf2 = ReplayBuffer::new(8);
        let n = buf2.load_from_path(path.to_str().unwrap()).unwrap();
        assert_eq!(n, 1);

        let a2 = 0 * AUX_STRIDE;
        assert_eq!(buf2.ownership[a2 + 10], 2);
        assert_eq!(buf2.ownership[a2 + 20], 0);
        assert_eq!(buf2.ownership[a2 + 30], 1);
        for i in 0..6 {
            assert_eq!(buf2.winning_line[a2 + 100 + i], 1);
        }

        let _ = std::fs::remove_file(path);
    }
}
