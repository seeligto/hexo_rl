//! HEXG — graph-position replay ring for the GNN training-data path (WP-5a / C8).
//!
//! A PARALLEL ring beside the dense `ReplayBuffer` (contract §2.6 option-(c),
//! `docs/designs/gnn_training_path_design.md` §2). It stores a COMPACT
//! whole-board position record — sorted stone list + sparse coord-keyed MCTS
//! visit target + §178 outcome/value_valid placeholders + per-game scalars — and
//! rebuilds the axis graph + aligns the policy target AT SAMPLE TIME on the C1
//! native builder (`hexo_graph::build_axis_graph`). NO dense planes, NO aux, NO
//! K-cluster loop (the GNN is whole-board).
//!
//! ## Why a separate module (dense ring UNTOUCHED)
//! The dense HEXB stride math (`state_stride()` etc.) has no graph meaning; a
//! graph record has no `state_stride`. Reusing the audited HEXB stride-ring
//! MECHANICS (fixed-slot `rotate_left`+`resize`, `head`-overwrite,
//! weight-bucket histogram) on graph-shaped SoA Vecs keeps the surgery minimal
//! and the dense CNN 10-metric bench gate byte-identical (this file is NOT on
//! the dense push/sample path).
//!
//! ## Fixed-max-slot layout (contract §2.6 realization (ii))
//! Each record is padded to `MAX_STONES` + `MAX_VISITS`, with per-record
//! `n_stones`/`n_visits` counts — a byte-for-byte parallel of the HEXB
//! stride-ring. ~2.3 KB/record → ~1.15 GB @ 500k capacity (design §2.1). The
//! true-CSR memory-optimal realization is the documented RAM-bound fallback.
//!
//! ## Sample = rebuild-at-native-builder (F7)
//! `sample_graph_batch` weighted-samples record indices, D6-rotates the stored
//! stone coords AND the visit-map keys by one uniform per-sample element
//! (`sym_tables::rotate_axial` — the single source of the 12 elements shared
//! with the CNN cell-scatter), rebuilds via `build_axis_graph` (which stamps
//! `builder_impl = 1` by construction), aligns the rotated visit-keys to the
//! built legal nodes → the per-legal-node policy target, and block-diagonal
//! fuses via `GraphWire::from_axis_graphs` (the SAME fuser the inference seam
//! uses). One call emits graph + target together (F1 single-source), so a
//! graph/target desync is structurally impossible (the ADV-7 defense).

mod persist;
mod push;
mod sample;
mod storage;
#[cfg(test)]
mod tests;

use std::sync::atomic::AtomicU64;

use half::f16;
use pyo3::prelude::*;
use rand::rngs::StdRng;

use crate::encoding::RegistrySpec;
use super::sym_tables::WeightSchedule;

// ── fixed-slot geometry (contract §2.1) ──────────────────────────────────────

/// Max stones per record slot. A HTTT game caps at `max_moves` (~150) stones;
/// 256 = `max_moves + headroom` (design §2.1). Over-cap push is a LOUD error.
pub(crate) const MAX_STONES: usize = 256;
/// Max sparse visit entries per record slot. Visits are sparse (the deploy
/// Gumbel regime is `m=16`); 128 = `m + safety` (design §2.1). A record with
/// more nonzero visit cells keeps its top-`MAX_VISITS` by mass (built in
/// `record_position_graph`); an over-cap PUSH is a LOUD error.
pub(crate) const MAX_VISITS: usize = 128;

/// HEXG on-disk magic — "HEXG" little-endian (distinct from HEXB `0x48455842`).
pub(crate) const HEXG_MAGIC: u32 = 0x4845_5847;
/// HEXG on-disk version. v1.
pub(crate) const HEXG_VERSION: u32 = 1;

/// Weight-bucket boundaries mirror `ReplayBuffer::weight_bucket` (dashboard
/// histogram parity — sampling weight is game-length-driven, representation-
/// agnostic).
#[inline]
pub(crate) fn weight_bucket(w_bits: u16) -> usize {
    let w = f16::from_bits(w_bits).to_f32();
    if w < 0.30 {
        0
    } else if w < 0.75 {
        1
    } else {
        2
    }
}

/// The single compact graph-position record (contract §2.6 option-(c) / design
/// §1.2). Produced by `record_position_graph` (self-play worker) or the
/// corpus replay-and-rebuild export; consumed by `HexgBuffer::push_graph_position`.
/// Coords are `i16` (board coords are small); the visit target is the sparse
/// coord→prob MCTS distribution over the FULL legal set (in- AND off-window —
/// the `records.rs:62` off-window skip is NOT inherited, design §6.1).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GraphRecord {
    /// Sorted (order irrelevant — the builder re-sorts) stone list `(q, r, ±1)`.
    pub stones: Vec<(i16, i16, i8)>,
    /// Sparse coord-keyed visit target `(q, r, prob)` over legal moves.
    pub visits: Vec<(i16, i16, f32)>,
    /// Side to move (+1 / −1) — rebuild `BuildParams`.
    pub current_player: i8,
    /// Moves remaining this turn (0..=255) — rebuild `BuildParams`.
    pub moves_remaining: u8,
    /// 0-based ply of this decision — weight/diagnostics only (no ply-aux head).
    pub ply_index: u16,
    /// Move-level playout-cap flag — policy-loss gate (kept).
    pub is_full_search: bool,
    /// §178 outcome z (placeholder at record time → filled at `finalize`).
    pub outcome: f32,
    /// DRAW-MASK (§178): 1 = supervise value, 0 = ply-capped row masked.
    pub value_valid: bool,
    /// Completed-game length (compound moves) — sampling weight (kept).
    pub game_length: u16,
}

// ── HexgBuffer ───────────────────────────────────────────────────────────────

/// Graph-position replay ring (parallel to `ReplayBuffer`). Fixed-slot SoA Vecs;
/// ring overwrite by `head`; weighted rejection sampler + game-length weight
/// schedule lifted verbatim from HEXB.
#[pyclass(name = "HexgBuffer")]
pub struct HexgBuffer {
    pub(crate) capacity: usize,
    pub(crate) size: usize,
    pub(crate) head: usize,

    /// Encoding spec — a `representation == Graph` spec. Drives the rebuild
    /// `BuildParams` (win_length / radius / trunk_size) + `contract_version`.
    pub(crate) encoding: &'static RegistrySpec,
    pub(crate) win_length: u8,
    pub(crate) radius: u16,
    pub(crate) trunk_size: i32,
    pub(crate) contract_version: u32,

    // ── fixed-slot record storage (SoA) ──
    pub(crate) stones_qr: Vec<i16>,     // flat [cap * MAX_STONES * 2]
    pub(crate) stone_players: Vec<i8>,  // flat [cap * MAX_STONES]
    pub(crate) n_stones: Vec<u16>,      // [cap]
    pub(crate) visit_qr: Vec<i16>,      // flat [cap * MAX_VISITS * 2]
    pub(crate) visit_probs: Vec<f32>,   // flat [cap * MAX_VISITS]
    pub(crate) n_visits: Vec<u16>,      // [cap]
    pub(crate) current_player: Vec<i8>, // [cap]
    pub(crate) moves_remaining: Vec<u8>,// [cap]
    pub(crate) ply_index: Vec<u16>,     // [cap]
    pub(crate) is_full_search: Vec<u8>, // [cap]
    pub(crate) outcomes: Vec<f32>,      // [cap]
    pub(crate) value_valid: Vec<u8>,    // [cap]
    pub(crate) game_length: Vec<u16>,   // [cap]
    pub(crate) game_ids: Vec<i64>,      // [cap]; −1 = untagged
    pub(crate) weights: Vec<u16>,       // f16 bits; [cap]

    pub(crate) weight_schedule: WeightSchedule,
    pub(crate) next_game_id: i64,
    pub(crate) rng: StdRng,
    pub(crate) weight_buckets: [AtomicU64; 3],
}

#[pymethods]
impl HexgBuffer {
    /// Create a graph-position ring with `capacity` records.
    ///
    /// `encoding` MUST be a `representation == "graph"` spec (default
    /// `"gnn_axis_v1"`) — the rebuild `BuildParams` come from its schema-v4
    /// graph fields (no scattered literals). A grid encoding is a LOUD error.
    #[new]
    #[pyo3(signature = (capacity, encoding = "gnn_axis_v1"))]
    pub fn new(capacity: usize, encoding: &str) -> PyResult<Self> {
        use pyo3::exceptions::PyValueError;
        let spec = crate::encoding::registry::lookup_or_panic(encoding);
        if !spec.is_graph() {
            return Err(PyValueError::new_err(format!(
                "HexgBuffer requires a graph encoding; '{encoding}' is representation=grid \
                 (use ReplayBuffer for dense encodings)"
            )));
        }
        let win_length = spec
            .win_length
            .expect("validate guarantees win_length for a graph spec") as u8;
        let radius = spec
            .graph_radius
            .expect("validate guarantees graph_radius for a graph spec") as u16;
        let contract_version = spec
            .contract_version
            .expect("validate guarantees contract_version for a graph spec");
        let default_w = f16::from_f32(1.0).to_bits();
        Ok(HexgBuffer {
            capacity,
            size: 0,
            head: 0,
            encoding: spec,
            win_length,
            radius,
            trunk_size: spec.trunk_size as i32,
            contract_version,
            stones_qr: vec![0i16; capacity * MAX_STONES * 2],
            stone_players: vec![0i8; capacity * MAX_STONES],
            n_stones: vec![0u16; capacity],
            visit_qr: vec![0i16; capacity * MAX_VISITS * 2],
            visit_probs: vec![0.0f32; capacity * MAX_VISITS],
            n_visits: vec![0u16; capacity],
            current_player: vec![1i8; capacity],
            moves_remaining: vec![2u8; capacity],
            ply_index: vec![0u16; capacity],
            is_full_search: vec![1u8; capacity],
            outcomes: vec![0.0f32; capacity],
            value_valid: vec![1u8; capacity],
            game_length: vec![0u16; capacity],
            game_ids: vec![-1i64; capacity],
            weights: vec![default_w; capacity],
            weight_schedule: WeightSchedule::uniform(),
            next_game_id: 0,
            rng: rand::make_rng(),
            weight_buckets: [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)],
        })
    }

    /// Store one compact graph-position record. `stones`/`visits` are lists of
    /// `(q, r, player)` / `(q, r, prob)` tuples. LOUD error if a record exceeds
    /// `MAX_STONES` / `MAX_VISITS` (die loud — never silently truncate on push;
    /// `record_position_graph` top-k-truncates the visit map before it reaches
    /// here so a live self-play record never over-caps).
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (stones, visits, current_player, moves_remaining, ply_index,
                        is_full_search, outcome, value_valid, game_length, game_id = -1))]
    pub fn push_graph_position(
        &mut self,
        stones: Vec<(i16, i16, i8)>,
        visits: Vec<(i16, i16, f32)>,
        current_player: i8,
        moves_remaining: u8,
        ply_index: u16,
        is_full_search: bool,
        outcome: f32,
        value_valid: bool,
        game_length: u16,
        game_id: i64,
    ) -> PyResult<()> {
        let rec = GraphRecord {
            stones,
            visits,
            current_player,
            moves_remaining,
            ply_index,
            is_full_search,
            outcome,
            value_valid,
            game_length,
        };
        self.push_record_impl(&rec, game_id)
    }

    /// Sample `batch_size` records, REBUILD each graph via the native builder,
    /// align the per-legal-node policy target, and block-diagonal fuse.
    ///
    /// Returns `(GraphWire, GraphTargets)`: the wire is the SAME payload the
    /// inference seam emits (fed to `collate_graph_batch`); `GraphTargets`
    /// carries the aligned per-legal-node policy target + dist65 outcome /
    /// value_valid / is_full_search + per-graph target-argmax cells (the
    /// AugRoundTrip runtime canary). `augment` draws one uniform D6 element per
    /// sampled record and coord-rotates stones + visit-keys before rebuild
    /// (design §5, realization (ii)). `recent_frac` (WP-5b §4, default `0.0`)
    /// draws that fraction of the batch from the newest ring slots instead of
    /// the full-ring weighted sample; `0.0` is byte-identical to pre-commit-B.
    #[pyo3(signature = (batch_size, augment = false, recent_frac = 0.0))]
    pub fn sample_graph_batch(
        &mut self,
        batch_size: usize,
        augment: bool,
        recent_frac: f32,
    ) -> PyResult<(crate::inference_bridge::GraphWire, GraphTargets)> {
        self.sample_graph_batch_impl(batch_size, augment, recent_frac)
    }

    /// Grow to `new_capacity`, preserving all records (linearise + extend).
    pub fn resize(&mut self, new_capacity: usize) -> PyResult<()> {
        self.resize_impl(new_capacity)
    }

    /// Set the game-length weight schedule (identical semantics to `ReplayBuffer`).
    pub fn set_weight_schedule(
        &mut self,
        thresholds: Vec<u16>,
        weights: Vec<f32>,
        default_weight: f32,
    ) -> PyResult<()> {
        self.set_weight_schedule_impl(thresholds, weights, default_weight)
    }

    /// `(size, capacity, weight_histogram)` for dashboard display (representation
    /// -agnostic — `pool.py` reads it blind).
    pub fn get_buffer_stats(&self) -> (usize, usize, Vec<u64>) {
        self.get_buffer_stats_impl()
    }

    /// Fresh monotonic game id.
    pub fn next_game_id(&mut self) -> i64 {
        let id = self.next_game_id;
        self.next_game_id += 1;
        id
    }

    /// Save records to a binary file (HEXG v1 format, magic `0x48455847`).
    pub fn save_to_path(&self, path: &str) -> PyResult<()> {
        self.save_to_path_impl(path)
    }

    /// Load records written by `save_to_path`. Returns the number loaded;
    /// missing file → 0. LOUD-FAILs on a HEXB (dense) file (magic mismatch), a
    /// wrong version, or a slot-geometry mismatch.
    pub fn load_from_path(&mut self, path: &str) -> PyResult<usize> {
        use pyo3::exceptions::PyValueError;
        self.load_from_path_impl(path).map_err(PyValueError::new_err)
    }

    #[getter]
    pub fn size(&self) -> usize {
        self.size
    }

    #[getter]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    #[getter]
    pub fn encoding_name(&self) -> &'static str {
        self.encoding.name
    }
}

/// Aligned training targets emitted alongside the `GraphWire` by
/// `sample_graph_batch` (design §4.1 single-call graph+target emission).
///
/// * `policy_target` — flat `[Lg]` per-legal-node CE target (graphs
///   concatenated, in `legal_node_gather` order); each graph's segment sums to
///   ~1 over its legal set (the stored MCTS visit distribution, no off-window
///   drop).
/// * `outcomes` / `value_valid` — `[B]` dist65 value target + draw-mask.
/// * `is_full_search` — `[B]` policy-loss gate (quick-search rows contribute
///   value only).
/// * `target_argmax_cells` (getter) — per-graph `(q, r)` of the max-mass legal
///   node in the ROTATED frame, or `None` when the segment is all-zero; fed to
///   `collate_graph_batch(target_argmax_cells=...)` as the AugRoundTrip runtime
///   canary.
#[pyclass(name = "GraphTargets")]
pub struct GraphTargets {
    pub(crate) policy_target: Vec<f32>,
    pub(crate) outcomes: Vec<f32>,
    pub(crate) value_valid: Vec<u8>,
    pub(crate) is_full_search: Vec<u8>,
    pub(crate) argmax_q: Vec<i32>,
    pub(crate) argmax_r: Vec<i32>,
    pub(crate) argmax_valid: Vec<u8>,
}

#[pymethods]
impl GraphTargets {
    #[getter]
    fn policy_target<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArray1<f32>> {
        numpy::PyArray1::from_slice(py, &self.policy_target)
    }
    #[getter]
    fn outcomes<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArray1<f32>> {
        numpy::PyArray1::from_slice(py, &self.outcomes)
    }
    #[getter]
    fn value_valid<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArray1<u8>> {
        numpy::PyArray1::from_slice(py, &self.value_valid)
    }
    #[getter]
    fn is_full_search<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArray1<u8>> {
        numpy::PyArray1::from_slice(py, &self.is_full_search)
    }
    /// `[B]` list of `Optional[(q, r)]` — the collate `target_argmax_cells` arg.
    #[getter]
    fn target_argmax_cells(&self) -> Vec<Option<(i32, i32)>> {
        (0..self.argmax_valid.len())
            .map(|i| {
                if self.argmax_valid[i] != 0 {
                    Some((self.argmax_q[i], self.argmax_r[i]))
                } else {
                    None
                }
            })
            .collect()
    }
}
