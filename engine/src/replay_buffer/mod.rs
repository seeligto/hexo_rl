//! ReplayBuffer — pre-allocated ring buffer with vectorized 12-fold hex augmentation.
//!
//! Stores (state, policy, outcome, ownership, winning_line) tuples from self-play.
//! On `sample_batch` the data is scatter-augmented in Rust and returned as owned
//! numpy arrays (zero extra copy).
//!
//! ## Module layout
//!   mod.rs        — `ReplayBuffer` struct + `#[pymethods]` facade (thin delegates)
//!   storage.rs    — resize, dashboard stats, weight schedule, monotonic id
//!   push.rs       — single-position `push`, batched `push_game`, test-only `push_raw`
//!   sample.rs     — `sample_batch` entry + weighted-sample + 12-fold apply_sym kernel
//!   persist.rs    — HEXB v7 save / load (on-disk format; current §174)
//!   sym_tables.rs — 12-fold permutation tables, axis-plane remap, constants
//!
//! ## Memory layout (flat, row-major)
//!   states       : Vec<u16> — f16 bits, logical shape
//!                            [capacity, encoding.n_planes, encoding.trunk_size²]
//!                            (encoding=v6: 8 buffer planes per KEPT_PLANE_INDICES,
//!                            trunk 19; chain planes stored separately)
//!   chain_planes : Vec<u16> — f16 bits, logical shape [capacity, 6, 361]
//!                            (Q13 chain-length planes: 3 axes × 2 players, normalized /6)
//!   policies     : Vec<f32> — logical shape [capacity, 362]
//!   outcomes     : Vec<f32> — logical shape [capacity]
//!   game_ids     : Vec<i64> — logical shape [capacity]; -1 = untagged
//!   weights      : Vec<u16> — f16 bits; one sample weight per position
//!   ownership    : Vec<u8>  — logical shape [capacity, 361] (0=P2, 1=empty, 2=P1)
//!   winning_line : Vec<u8>  — logical shape [capacity, 361] binary mask
//!
//! ## Symmetry tables
//! 12 symmetries = 6 rotations × 2 (with/without prior reflection).
//! Each symmetry is stored as a Vec<(u16, u16)> of valid (src_cell, dst_cell) scatter pairs
//! (cells that map out of the 19×19 window are dropped, matching the Python implementation).

pub mod sym_tables;
mod storage;
mod push;
pub mod push_config;
pub mod sample;
mod persist;

use half::f16;
use numpy::{
    PyArray1, PyArray2, PyArray3, PyArray4,
    PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, PyReadonlyArray4,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::rngs::StdRng;
use std::sync::atomic::AtomicU64;

use crate::encoding::RegistrySpec;
use sym_tables::{SymTables, WeightSchedule, sym_tables_for};

/// Return tuple of `sample_batch` (8-tuple): `(states, chain, policies,
/// outcomes, ownership, winning_line, is_full_search, value_target_valid)`.
/// DRAW-MASK (Phase 6) appended `value_target_valid` (per-row u8; 1 = supervise
/// value, 0 = masked) as the trailing field.
pub(crate) type SampleBatchOut<'py> = (
    Bound<'py, PyArray4<f16>>,
    Bound<'py, PyArray4<f16>>,
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray3<u8>>,
    Bound<'py, PyArray3<u8>>,
    Bound<'py, PyArray1<u8>>,
    Bound<'py, PyArray1<u8>>,
);

/// §S181-AUDIT Wave 4 4B-impl-1 — extended form, adds per-row
/// `position_indices` for the ply-to-end aux head. Emitted by `sample_batch_with_pos`.
/// DRAW-MASK (Phase 6) appended a trailing `value_target_valid` (per-row u8) so the
/// trainer can mask ply-capped rows out of the value loss.
pub(crate) type SampleBatchWithPosOut<'py> = (
    Bound<'py, PyArray4<f16>>,
    Bound<'py, PyArray4<f16>>,
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray3<u8>>,
    Bound<'py, PyArray3<u8>>,
    Bound<'py, PyArray1<u8>>,
    Bound<'py, PyArray1<u16>>,
    Bound<'py, PyArray1<u8>>,
);

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

    /// Encoding spec — drives all stride / cell-count calculations.
    /// Looked up from registry at construction time via `encoding` kwarg (default "v6").
    /// §173 A4: replaces scattered const references in push/sample/storage/persist.
    pub(crate) encoding: &'static RegistrySpec,

    pub(crate) states:       Vec<u16>,  // f16 bits; flat [capacity × encoding.state_stride()]
    /// Q13 chain-length planes stored separately from state.
    /// Flat [capacity × encoding.chain_stride()]. Scattered with axis-plane remap on augmentation.
    pub(crate) chain_planes: Vec<u16>,  // f16 bits; flat [capacity × encoding.chain_stride()]
    pub(crate) policies: Vec<f32>,  // flat [capacity × encoding.policy_stride()]
    pub(crate) outcomes: Vec<f32>,  // flat [capacity]
    pub(crate) game_ids: Vec<i64>,  // flat [capacity]; −1 = untagged
    pub(crate) weights:  Vec<u16>,  // f16-as-u16 bits; flat [capacity]; sampling weight per position

    /// Auxiliary target: per-cell ownership of the final board state, projected
    /// to each row's per-cluster window centre (NOT the game-end bbox centroid).
    /// Encoding: 0 = P2, 1 = empty, 2 = P1. Flat [capacity × encoding.aux_stride()].
    pub(crate) ownership:    Vec<u8>,
    /// Auxiliary target: binary mask of the 6 cells of the winning 6-in-a-row,
    /// projected to each row's per-cluster window centre. All zero on draw.
    /// Flat [capacity × encoding.aux_stride()].
    pub(crate) winning_line: Vec<u8>,

    /// Move-level playout cap flag: 1 = full-search (policy loss applies),
    /// 0 = quick-search (value/chain/aux losses only). Flat [capacity].
    /// Defaults to 1 so corpus and legacy positions always contribute to policy.
    pub(crate) is_full_search: Vec<u8>,

    /// DRAW-MASK (Phase 6): per-row value-supervision flag. 1 = supervise the
    /// value head, 0 = ply-capped game (horizon truncation, fabricated value
    /// label) → mask this row out of the value loss; policy target unaffected.
    /// Flat [capacity]. Defaults to 1 so corpus / bot / legacy positions always
    /// contribute to value (only self-play ply-capped rows are masked).
    pub(crate) value_target_valid: Vec<u8>,

    /// §S181-AUDIT Wave 4 4B-impl-1 — per-position ply index within its game.
    /// 0-based; flat [capacity]. Drives the ply-to-end auxiliary head
    /// (target = (game_length - position_index) / 100 trained via Huber).
    /// Defaults to 0 for corpus rows that lack per-position metadata
    /// (auxiliary loss masks pretrain rows so dummy values don't contribute).
    pub(crate) position_indices: Vec<u16>,

    /// Static sym tables for this buffer's encoding. Shared across all buffers
    /// of the same encoding via `sym_tables_for()`. §173 A4.
    pub(crate) sym_tables:      &'static SymTables,
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

// ── Python-facing facade ──────────────────────────────────────────────────
//
// Each method here is a thin delegate to a `*_impl` on `impl ReplayBuffer`
// defined in one of the sibling modules above.  PyO3 requires all
// `#[pymethods]` to live in a single `impl` block; keeping the facade here
// and the logic next to peers lets each sibling file own one concept.

#[pymethods]
impl ReplayBuffer {
    /// Create a new buffer with the given `capacity` (number of positions).
    ///
    /// Pre-allocates all storage.  Building the symmetry tables is O(N_CELLS × N_SYMS) ≈ 4 µs.
    ///
    /// `encoding` — encoding name from `engine/src/encoding/registry.toml`.
    /// Defaults to `"v6"` for backward compatibility with legacy callers that
    /// do not pass this kwarg. A deprecation warning will be added in A6 once
    /// all callsites are migrated. §173 A4.
    #[new]
    #[pyo3(signature = (capacity, encoding = "v6"))]
    pub fn new(capacity: usize, encoding: &str) -> Self {
        let spec = crate::encoding::registry::lookup_or_panic(encoding);
        let default_w = f16::from_f32(1.0).to_bits();
        ReplayBuffer {
            capacity,
            size: 0,
            head: 0,
            encoding: spec,
            states:         vec![0u16; capacity * spec.state_stride()],
            chain_planes:   vec![0u16; capacity * spec.chain_stride()],
            policies:       vec![0.0f32; capacity * spec.policy_stride()],
            outcomes:       vec![0.0f32; capacity],
            game_ids:       vec![-1i64; capacity],
            weights:        vec![default_w; capacity],
            ownership:      vec![1u8; capacity * spec.aux_stride()],  // 1 = empty default
            winning_line:   vec![0u8; capacity * spec.aux_stride()],
            is_full_search: vec![1u8; capacity],  // 1 = full-search default (legacy compat)
            value_target_valid: vec![1u8; capacity],  // 1 = supervise value default (DRAW-MASK)
            position_indices: vec![0u16; capacity],  // §S181 Wave 4 4B-impl-1
            sym_tables: sym_tables_for(spec),
            weight_schedule: WeightSchedule::uniform(),
            next_game_id: 0,
            rng: rand::make_rng(),
            weight_buckets: [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)],
        }
    }

    /// Return (size, capacity, weight_histogram) for dashboard display.
    pub fn get_buffer_stats(&self) -> (usize, usize, Vec<u64>) {
        self.get_buffer_stats_impl()
    }

    /// Return a fresh position ID and advance the internal counter.
    pub fn next_game_id(&mut self) -> i64 {
        self.next_game_id_impl()
    }

    /// Store a single (state, chain_planes, policy, outcome, ownership, winning_line) sample.
    // KEEP — PyO3 kwarg surface mirrors 9 user-tunable params (INV20). §S181 Wave 4
    // 4B-impl-1 added `position_index` as an additive kwarg (default 0) — old callers
    // continue to work without modification.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (state, chain_planes, policy, outcome, ownership, winning_line, game_id = -1, game_length = 0, is_full_search = true, position_index = 0, value_target_valid = true))]
    pub fn push(
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
        position_index: u16,
        value_target_valid: bool,
    ) -> PyResult<()> {
        self.push_impl(push_config::PushSingleConfig {
            state, chain_planes, policy, outcome, ownership, winning_line,
            game_id, game_length, is_full_search, position_index, value_target_valid,
        })
    }

    /// Store all positions from a completed game efficiently.
    // KEEP — PyO3 kwarg surface mirrors 9 user-tunable params (INV20). §S181 Wave 4
    // 4B-impl-1 added `position_indices` as an additive Option kwarg (default
    // None → fills 0; PIPE-1 2026-06-02) — old callers continue to work.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (states, chain_planes, policies, outcomes, ownership, winning_line, game_id = -1, game_length = 0, is_full_search = None, position_indices = None, value_target_valid = None))]
    pub fn push_game(
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
        position_indices: Option<PyReadonlyArray1<u16>>,
        value_target_valid: Option<PyReadonlyArray1<u8>>,
    ) -> PyResult<()> {
        self.push_game_impl(push_config::PushGameConfig {
            states, chain_planes, policies, outcomes, ownership, winning_line,
            game_id, game_length, is_full_search, position_indices, value_target_valid,
        })
    }

    /// Store N positions with per-row game_length and is_full_search in one PyO3 call.
    ///
    /// Replaces the per-row `push` loop in `WorkerPool._stats_loop`, avoiding N
    /// PyO3 method-dispatch + PyRefMut acquire/release cycles. All rows are
    /// treated as untagged (game_id = -1); use `push_game` if rows need a
    /// shared game_id.
    ///
    /// Per-row `game_lengths` must be pre-computed compound-move counts
    /// (i.e. `(plies + 1) / 2`); value 0 → default weight 1.0.
    // KEEP — PyO3 kwarg surface mirrors 8 user-tunable params (INV20). §S181 Wave 4
    // 4B-impl-1 added `position_indices` as an additive Option kwarg — default
    // None preserves legacy callers (fills zeros).
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (states, chain_planes, policies, outcomes, ownership, winning_line, game_lengths, is_full_search, position_indices = None, value_target_valid = None))]
    pub fn push_many(
        &mut self,
        states:         PyReadonlyArray4<f16>,
        chain_planes:   PyReadonlyArray4<f16>,
        policies:       PyReadonlyArray2<f32>,
        outcomes:       PyReadonlyArray1<f32>,
        ownership:      PyReadonlyArray2<u8>,
        winning_line:   PyReadonlyArray2<u8>,
        game_lengths:   PyReadonlyArray1<u16>,
        is_full_search: PyReadonlyArray1<u8>,
        position_indices: Option<PyReadonlyArray1<u16>>,
        value_target_valid: Option<PyReadonlyArray1<u8>>,
    ) -> PyResult<()> {
        self.push_many_impl(push_config::PushManyConfig {
            states, chain_planes, policies, outcomes, ownership, winning_line,
            game_lengths, is_full_search, position_indices, value_target_valid,
        })
    }

    /// Sample `batch_size` entries, optionally with 12-fold hex augmentation.
    ///
    /// Returns (shapes shown for encoding=v6; generally
    /// `(batch_size, encoding.n_planes, encoding.trunk_size, encoding.trunk_size)`):
    ///     states:          float16 numpy array of shape (batch_size, 8, 19, 19)
    ///     chain_planes:    float16 numpy array of shape (batch_size, 6, 19, 19)
    ///     policies:        float32 numpy array of shape (batch_size, 362)
    ///     outcomes:        float32 numpy array of shape (batch_size,)
    ///     ownership:       uint8   numpy array of shape (batch_size, 19, 19)
    ///     winning_line:    uint8   numpy array of shape (batch_size, 19, 19)
    ///     is_full_search:  uint8   numpy array of shape (batch_size,)
    ///     value_target_valid: uint8 numpy array of shape (batch_size,) — DRAW-MASK
    ///                          (Phase 6): 1 = supervise value, 0 = ply-capped → masked.
    pub fn sample_batch<'py>(
        &mut self,
        py:        Python<'py>,
        batch_size: usize,
        augment:    bool,
    ) -> PyResult<SampleBatchOut<'py>> {
        self.sample_batch_impl(py, batch_size, augment)
    }

    /// §S181-AUDIT Wave 4 4B-impl-1 — extended `sample_batch` returning 8-tuple
    /// with per-row `position_indices` (uint16, shape (batch_size,)) appended.
    /// Used by the ply-to-end auxiliary head training path. Existing callers
    /// continue to use the 7-tuple `sample_batch` (byte-identical to pre-Wave-4).
    pub fn sample_batch_with_pos<'py>(
        &mut self,
        py:        Python<'py>,
        batch_size: usize,
        augment:    bool,
    ) -> PyResult<SampleBatchWithPosOut<'py>> {
        self.sample_batch_with_pos_impl(py, batch_size, augment)
    }

    /// Grow the buffer to `new_capacity` positions, preserving all existing data.
    pub fn resize(&mut self, new_capacity: usize) -> PyResult<()> {
        self.resize_impl(new_capacity)
    }

    /// Phase B' Class-3 buffer composition probe — return the number of
    /// rows in `[..self.size]` whose outcome (value target) lies in
    /// the half-open interval `[lo, hi)`.
    ///
    /// `draw_target_fraction = outcome_in_range_count(-0.6, -0.4) / size`
    /// when the production `draw_value=-0.5` config is active. Pure read
    /// path; safe to call concurrently with non-write operations.
    pub fn outcome_in_range_count(&self, lo: f32, hi: f32) -> usize {
        self.outcome_in_range_count_impl(lo, hi)
    }

    /// Set the game-length weight schedule from Python config.
    pub fn set_weight_schedule(
        &mut self,
        thresholds:     Vec<u16>,
        weights:        Vec<f32>,
        default_weight: f32,
    ) -> PyResult<()> {
        self.set_weight_schedule_impl(thresholds, weights, default_weight)
    }

    /// Save buffer contents to a binary file (HEXB v7 on-disk format; current §174).
    #[pyo3(text_signature = "(self, path)")]
    pub fn save_to_path(&self, path: &str) -> PyResult<()> {
        self.save_to_path_impl(path)
    }

    /// Load buffer contents from a binary file written by `save_to_path`.
    ///
    /// Returns the number of positions loaded. Missing file → returns 0.
    #[pyo3(text_signature = "(self, path)")]
    pub fn load_from_path(&mut self, path: &str) -> PyResult<usize> {
        self.load_from_path_impl(path)
            .map_err(PyValueError::new_err)
    }

    #[getter]
    pub fn size(&self) -> usize { self.size }

    #[getter]
    pub fn capacity(&self) -> usize { self.capacity }

    /// Return the encoding spec driving this buffer's geometry.
    /// §174: enables Python consumers (benchmark.py, dashboard) to query
    /// trunk_size / n_planes / policy_logit_count without hardcoding.
    #[getter]
    pub fn encoding(&self) -> crate::PyRegistrySpec {
        crate::PyRegistrySpec::from_static(self.encoding)
    }
}

/// Non-PyO3 helpers used by Rust integration tests in `engine/tests/`.
impl ReplayBuffer {
    /// Push a zero-filled position with the given outcome, game_length, and
    /// is_full_search flag. Mirrors the push_raw logic but exposed `pub` so
    /// integration tests (which cannot reach `pub(crate)` items) can populate
    /// the buffer with known is_full_search values for save/load round-trip checks.
    pub fn push_for_test(&mut self, outcome: f32, game_length: u16, is_full_search: bool) {
        use std::sync::atomic::Ordering;
        let state_stride  = self.encoding.state_stride();
        let chain_stride  = self.encoding.chain_stride();
        let policy_stride = self.encoding.policy_stride();
        let aux_stride    = self.encoding.aux_stride();
        let slot = self.head;
        if self.size == self.capacity {
            let old_bucket = Self::weight_bucket(self.weights[slot]);
            self.weight_buckets[old_bucket].fetch_sub(1, Ordering::Relaxed);
        }
        let s = slot * state_stride;
        self.states[s..s + state_stride].fill(0);
        let c = slot * chain_stride;
        self.chain_planes[c..c + chain_stride].fill(0);
        let p = slot * policy_stride;
        self.policies[p..p + policy_stride].fill(0.0);
        let a = slot * aux_stride;
        self.ownership[a..a + aux_stride].fill(1);
        self.winning_line[a..a + aux_stride].fill(0);
        self.is_full_search[slot] = is_full_search as u8;
        self.position_indices[slot] = 0;
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

    /// Return the raw `is_full_search` byte at the given buffer slot.
    pub fn is_full_search_at(&self, slot: usize) -> u8 {
        self.is_full_search[slot]
    }

    /// Return the raw `value_target_valid` byte at the given buffer slot.
    /// 1 = supervise value head, 0 = ply-capped row masked from value loss.
    pub fn value_target_valid_at(&self, slot: usize) -> u8 {
        self.value_target_valid[slot]
    }

    /// Return the sampling weight at the given buffer slot as f32.
    pub fn weight_at_f32(&self, slot: usize) -> f32 {
        f16::from_bits(self.weights[slot]).to_f32()
    }
}
