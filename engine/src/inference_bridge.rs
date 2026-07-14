use std::collections::{VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use dashmap::DashMap;
use fxhash::FxBuildHasher;
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::encoding::Representation;
use crate::game_runner::records::{assemble_ls_from_gnn_probs, LegalSetPolicy};
use hexo_graph::{build_axis_graph, AxisGraph, BuildParams, StoneList, BUILDER_IMPL_NATIVE};

#[derive(Clone)]
struct PendingRequest {
    id: u64,
    features: Vec<f32>,
}

// ── GNN-integration WP-3 (C3): dormant graph seam ────────────────────────────
// A PARALLEL graph queue lives beside the dense queue (seam design §3.3). The
// dense `Vec<f32>` hot path is byte-identical — no branch, no allocation change
// — so the 10-metric CNN bench gate cannot regress. NO producer enqueues onto
// this queue yet (no call site in `worker_loop`); it is exercised only from the
// graph pymethods below + their tests. The graph waiter payload is the ragged
// `(LegalSetPolicy, value)` (assembled Rust-side from per-legal-node probs),
// vs the dense `(Arc<Vec<f32>>, Range, value)`.

/// One queued graph inference request (the once-per-leaf `AxisGraph` payload).
struct PendingGraphRequest {
    id: u64,
    graph: AxisGraph,
}

/// Assemble metadata retained per in-flight graph id: `next_graph_batch` moves
/// the graph's wire arrays into numpy but keeps the slot map + legal coords so
/// `submit_graph_inference_results` can build the `LegalSetPolicy` Rust-side
/// (`policy_dst_slot` is consumed here, never as a Python dense scatter —
/// contract §6.2 / seam design §3.4).
struct InFlightGraph {
    policy_dst_slot: Vec<i32>,
    legal_coords: Vec<(i32, i32)>,
}

/// Graph waiter payload — the ragged `(LegalSetPolicy, value)` (vs dense §P74).
type GraphWaiterPayload = Result<(LegalSetPolicy, f32), String>;

#[derive(Default)]
struct GraphWaiter {
    result: Mutex<Option<GraphWaiterPayload>>,
    cv: Condvar,
}

/// Build one axis graph from a self-play/test request, enforcing the WP-3
/// red-team seam obligations BEFORE any narrowing cast
/// (`reports/probes/gnn_integration/WP1_redteam.md` Attack-2 / Attack-4 +
/// `WP1_review.md` coord/current_player/moves_remaining notes):
///   - `current_player ∈ {-1, +1}` (range-validate before the `i8` cast);
///   - `moves_remaining ∈ [0, 255]` (before the `u8` cast — a negative or >255
///     value would wrap to a bogus `moves_feat`, Attack-4);
///   - each stone `|q|,|r|` bounded well below `i32::MAX - radius` (Attack-2:
///     beyond that the builder's release-mode coordinate math silently wraps —
///     the only guard today is the debug-build overflow check);
///   - each stone player ∈ {-1, +1}.
///
/// The builder's own always-on `verify_contract` + `builder_impl=1` stamp close
/// the rest; the handshake is re-asserted on the returned graph (§7 die-loud).
fn build_graph_from_request(
    stones: &[(i64, i64, i64)],
    current_player: i64,
    moves_remaining: i64,
    win_length: u8,
    radius: u16,
    trunk_size: i32,
) -> PyResult<AxisGraph> {
    if current_player != 1 && current_player != -1 {
        return Err(PyValueError::new_err(format!(
            "graph request: current_player {current_player} out of range (expected +1 / -1)"
        )));
    }
    if !(0..=i64::from(u8::MAX)).contains(&moves_remaining) {
        return Err(PyValueError::new_err(format!(
            "graph request: moves_remaining {moves_remaining} out of range 0..=255 \
             (narrowing-cast guard, WP1 Attack-4)"
        )));
    }
    let bound = i64::from(i32::MAX) - i64::from(radius) - 1;
    let mut typed: Vec<(i32, i32, i8)> = Vec::with_capacity(stones.len());
    for &(q, r, p) in stones {
        if q.abs() > bound || r.abs() > bound {
            return Err(PyValueError::new_err(format!(
                "graph request: stone coord ({q},{r}) exceeds |coord| < i32::MAX-radius \
                 (WP1 Attack-2 silent-wrap guard)"
            )));
        }
        if p != 1 && p != -1 {
            return Err(PyValueError::new_err(format!(
                "graph request: stone player {p} out of range (expected +1 / -1)"
            )));
        }
        typed.push((q as i32, r as i32, p as i8));
    }
    let params = BuildParams {
        win_length,
        radius,
        current_player: current_player as i8,
        moves_remaining: moves_remaining as u8,
        trunk_size,
    };
    let graph = build_axis_graph(&StoneList { stones: typed }, &params);
    if graph.builder_impl != BUILDER_IMPL_NATIVE {
        return Err(PyValueError::new_err(
            "graph request: non-native builder_impl (NonNativeSampleBuilder handshake)",
        ));
    }
    Ok(graph)
}

/// Waiter result payload (§P74): the policy buffer is delivered as
/// `(Arc<Vec<f32>>, Range<usize>, f32)` so a single bulk `to_vec()` at the
/// submitter side replaces N per-row `to_vec()` allocations. Consumers
/// materialise the owned `Vec<f32>` (preserving the public
/// `(Vec<f32>, f32)` contract) only at pull-time via `arc[range].to_vec()`.
///
/// Precedent: `TTEntry.policy = Arc<Vec<f32>>` in `engine/src/mcts/backup.rs`
/// (Wave 4 P7). Same Arc-share + late-materialise pattern.
type WaiterPayload = Result<(Arc<Vec<f32>>, std::ops::Range<usize>, f32), String>;

#[derive(Default)]
struct Waiter {
    result: Mutex<Option<WaiterPayload>>,
    cv: Condvar,
}

struct Inner {
    queue: Mutex<VecDeque<PendingRequest>>,
    queue_cv: Condvar,
    waiters: DashMap<u64, Arc<Waiter>, FxBuildHasher>,
    next_id: AtomicU64,
    closed: AtomicBool,
    completed_mock_games: AtomicUsize,
    /// Phase B' Class-1 instrumentation: monotonic counter incremented on
    /// every weight swap by `InferenceServer.load_state_dict_safe()`. Workers
    /// snapshot this once per move to bound the model-version range each
    /// game crosses. Read with Relaxed; precision is per-move, not
    /// per-leaf-eval (fewer atomic touches; same statistic to two sig figs).
    model_version: AtomicU64,

    // ── WP-3 dormant graph seam (parallel queue; dense fields untouched) ──
    graph_queue: Mutex<VecDeque<PendingGraphRequest>>,
    graph_queue_cv: Condvar,
    graph_waiters: DashMap<u64, Arc<GraphWaiter>, FxBuildHasher>,
    in_flight_graphs: DashMap<u64, InFlightGraph, FxBuildHasher>,
    completed_graph_games: AtomicUsize,
}

impl Inner {
    fn new() -> Self {
        Self {
            queue: Mutex::new(VecDeque::new()),
            queue_cv: Condvar::new(),
            waiters: DashMap::with_hasher(FxBuildHasher::default()),
            next_id: AtomicU64::new(1),
            closed: AtomicBool::new(false),
            completed_mock_games: AtomicUsize::new(0),
            model_version: AtomicU64::new(0),
            graph_queue: Mutex::new(VecDeque::new()),
            graph_queue_cv: Condvar::new(),
            graph_waiters: DashMap::with_hasher(FxBuildHasher::default()),
            in_flight_graphs: DashMap::with_hasher(FxBuildHasher::default()),
            completed_graph_games: AtomicUsize::new(0),
        }
    }

    /// Graph counterpart of `pop_batch_blocking` — same saturation threshold /
    /// timeout, on the parallel graph queue.
    fn pop_graph_batch_blocking(
        &self,
        batch_size: usize,
        max_wait_ms: u64,
    ) -> Vec<PendingGraphRequest> {
        let deadline = Instant::now() + Duration::from_millis(max_wait_ms);
        let mut queue = self.graph_queue.lock().expect("graph queue lock poisoned");
        let threshold = batch_size / 2;
        while queue.len() < threshold && !self.closed.load(Ordering::SeqCst) {
            let now = Instant::now();
            if now >= deadline {
                break;
            }
            let remaining = deadline.saturating_duration_since(now);
            let (q, _) = self
                .graph_queue_cv
                .wait_timeout(queue, remaining)
                .expect("graph queue condvar poisoned");
            queue = q;
        }
        if queue.is_empty() {
            return Vec::new();
        }
        let take = batch_size.min(queue.len());
        let mut out = Vec::with_capacity(take);
        for _ in 0..take {
            if let Some(req) = queue.pop_front() {
                out.push(req);
            }
        }
        out
    }

    fn submit_and_wait(
        &self,
        features: Vec<f32>,
        expected_feature_len: usize,
        expected_policy_len: usize,
    ) -> PyResult<(Vec<f32>, f32)> {
        if features.len() != expected_feature_len {
            return Err(PyValueError::new_err(format!(
                "feature length mismatch: got {}, expected {}",
                features.len(),
                expected_feature_len
            )));
        }

        if self.closed.load(Ordering::SeqCst) {
            return Err(PyValueError::new_err("batcher is closed"));
        }

        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let waiter = Arc::new(Waiter::default());

        self.waiters.insert(id, waiter.clone());

        {
            let mut queue = self.queue.lock().expect("queue lock poisoned");
            queue.push_back(PendingRequest { id, features });
            self.queue_cv.notify_all();
        }

        let mut guard = waiter.result.lock().expect("waiter lock poisoned");
        loop {
            if let Some(res) = guard.take() {
                // §P74: result is `(Arc<Vec<f32>>, Range<usize>, f32)`.
                // Materialise the owned Vec at consumer pull-time so the
                // external `(Vec<f32>, f32)` contract is preserved. Range
                // length is the validated policy length from the submitter.
                match res {
                    Ok((policy_arc, range, value)) => {
                        let policy_len = range.len();
                        if policy_len != expected_policy_len {
                            return Err(PyValueError::new_err(format!(
                                "policy length mismatch for request {id}: got {policy_len}, expected {expected_policy_len}"
                            )));
                        }
                        let policy = policy_arc[range].to_vec();
                        return Ok((policy, value));
                    }
                    Err(e) => return Err(PyValueError::new_err(format!("inference failed: {e}"))),
                }
            }

            if self.closed.load(Ordering::SeqCst) {
                return Err(PyValueError::new_err("batcher closed while request was waiting"));
            }

            guard = waiter.cv.wait(guard).expect("waiter condvar poisoned");
        }
    }

    fn pop_batch_blocking(
        &self,
        batch_size: usize,
        max_wait_ms: u64,
    ) -> Vec<PendingRequest> {
        let deadline = Instant::now() + Duration::from_millis(max_wait_ms);
        let mut queue = self.queue.lock().expect("queue lock poisoned");
        
        // Wait until we have enough requests OR the timeout expires.
        // We target at least 50% saturation (32/64).
        let threshold = batch_size / 2;
        while queue.len() < threshold && !self.closed.load(Ordering::SeqCst) {
            let now = Instant::now();
            if now >= deadline {
                break;
            }
            let remaining = deadline.saturating_duration_since(now);
            let (q, _) = self
                .queue_cv
                .wait_timeout(queue, remaining)
                .expect("queue condvar poisoned");
            queue = q;
        }

        if queue.is_empty() {
            return Vec::new();
        }

        let take = batch_size.min(queue.len());
        let mut out = Vec::with_capacity(take);
        for _ in 0..take {
            if let Some(req) = queue.pop_front() {
                out.push(req);
            }
        }
        out
    }
}

/// Rust-owned blocking inference queue for fused Python model inference.
///
/// Worker threads call `submit_batch_and_wait_rust`, which enqueues features
/// and blocks. Python calls `next_inference_batch` to fetch a fused batch
/// tensor, runs the model, then calls `submit_inference_results` to wake
/// waiters.
#[pyclass(name = "InferenceBatcher", skip_from_py_object)]
#[derive(Clone)]
pub struct InferenceBatcher {
    inner: Arc<Inner>,
    feature_len: usize,
    policy_len: usize,
    pool_sender: flume::Sender<Vec<f32>>,
    pool_receiver: flume::Receiver<Vec<f32>>,
    // ── WP-3 graph seam (Grid for every dense batcher; the graph fields are
    // inert unless constructed from a `representation="graph"` spec) ──
    representation: Representation,
    graph_win_length: u8,
    graph_radius: u16,
    graph_trunk_size: i32,
    graph_contract_version: u32,
}

impl InferenceBatcher {
    pub(crate) fn submit_batch_and_wait_rust(
        &self,
        batch_features: Vec<Vec<f32>>,
    ) -> Result<Vec<(Vec<f32>, f32)>, ()> {
        if self.inner.closed.load(Ordering::SeqCst) {
            return Err(());
        }

        let n = batch_features.len();
        let mut ids = Vec::with_capacity(n);
        let mut waiters = Vec::with_capacity(n);

        {
            let mut queue = self.inner.queue.lock().expect("queue lock poisoned");

            for features in batch_features {
                if features.len() != self.feature_len {
                    return Err(());
                }
                let id = self.inner.next_id.fetch_add(1, Ordering::SeqCst);
                let waiter = Arc::new(Waiter::default());
                self.inner.waiters.insert(id, waiter.clone());
                queue.push_back(PendingRequest { id, features });
                ids.push(id);
                waiters.push(waiter);
            }
            self.inner.queue_cv.notify_all();
        }

        let mut results = Vec::with_capacity(n);
        for waiter in waiters {
            let mut guard = waiter.result.lock().expect("waiter lock poisoned");
            loop {
                if let Some(res) = guard.take() {
                    // §P74: result is `(Arc<Vec<f32>>, Range<usize>, f32)`.
                    // Materialise the owned Vec here so `worker_loop.rs`'s
                    // `for (mut p, v) in results` (in-place rotation path)
                    // keeps working. The Arc bulk-share collapses N submitter
                    // allocations to 1; the per-row Vec materialise on the
                    // consumer side is required only because downstream
                    // mutates `p` for symmetry rotation.
                    match res {
                        Ok((policy_arc, range, value)) => {
                            let policy = policy_arc[range].to_vec();
                            results.push((policy, value));
                            break;
                        }
                        Err(_) => return Err(()),
                    }
                }
                if self.inner.closed.load(Ordering::SeqCst) {
                    return Err(());
                }
                guard = waiter.cv.wait(guard).expect("waiter condvar poisoned");
            }
        }

        Ok(results)
    }

    pub(crate) fn close_rust(&self) {
        self.inner.closed.store(true, Ordering::SeqCst);
        self.inner.queue_cv.notify_all();
        self.inner.graph_queue_cv.notify_all();

        for r in &self.inner.waiters {
            r.value().cv.notify_all();
        }
        for r in &self.inner.graph_waiters {
            r.value().cv.notify_all();
        }
    }

    /// Graph counterpart of `submit_batch_and_wait_rust` (seam design §3.3) —
    /// the worker-facing blocking submit. Enqueues each pre-built `AxisGraph`
    /// on the parallel graph queue, blocks on its graph waiter, and returns the
    /// assembled `(LegalSetPolicy, value)` per leaf. NOT wired to `worker_loop`
    /// yet (WP-3 step 6); exercised by `spawn_mock_graph_games` + tests. The
    /// builder_impl handshake is asserted per graph (contract §7 die-loud).
    pub(crate) fn submit_batch_and_wait_graph_rust(
        &self,
        graphs: Vec<AxisGraph>,
    ) -> Result<Vec<(LegalSetPolicy, f32)>, ()> {
        if self.inner.closed.load(Ordering::SeqCst) {
            return Err(());
        }
        let n = graphs.len();
        // N5: run the builder_impl + contract_version handshake (§7) as a
        // PRE-PASS before touching the queue / waiter maps, so a bad tag can
        // never leave half the batch enqueued with orphaned waiters. The ONLY
        // builder is native hexo-graph; a non-native tag never reaches the queue.
        if self.graph_contract_version != 1
            || graphs.iter().any(|g| g.builder_impl != BUILDER_IMPL_NATIVE)
        {
            return Err(());
        }
        let mut waiters = Vec::with_capacity(n);
        {
            let mut queue = self.inner.graph_queue.lock().expect("graph queue lock poisoned");
            for graph in graphs {
                let id = self.inner.next_id.fetch_add(1, Ordering::SeqCst);
                let waiter = Arc::new(GraphWaiter::default());
                self.inner.graph_waiters.insert(id, waiter.clone());
                queue.push_back(PendingGraphRequest { id, graph });
                waiters.push(waiter);
            }
            self.inner.graph_queue_cv.notify_all();
        }

        let mut results = Vec::with_capacity(n);
        for waiter in waiters {
            let mut guard = waiter.result.lock().expect("graph waiter lock poisoned");
            loop {
                if let Some(res) = guard.take() {
                    match res {
                        Ok(payload) => {
                            results.push(payload);
                            break;
                        }
                        Err(_) => return Err(()),
                    }
                }
                if self.inner.closed.load(Ordering::SeqCst) {
                    return Err(());
                }
                guard = waiter.cv.wait(guard).expect("graph waiter condvar poisoned");
            }
        }
        Ok(results)
    }

    pub fn feature_len(&self) -> usize {
        self.feature_len
    }

    /// §172 A10 T8b — Rust-only mirror of the Python-visible `policy_len_py`
    /// getter, kept for symmetry with `feature_len()` (cargo-test consumers).
    /// Python callers use `batcher.policy_len_py` (the `#[getter]` below).
    pub fn policy_len(&self) -> usize {
        self.policy_len
    }

    pub fn get_feature_buffer(&self) -> Vec<f32> {
        self.pool_receiver.try_recv().unwrap_or_else(|_| {
            vec![0.0f32; self.feature_len]
        })
    }

    pub fn return_feature_buffer(&self, mut buf: Vec<f32>) {
        if buf.capacity() >= self.feature_len {
            buf.clear();
            buf.resize(self.feature_len, 0.0);
            let _ = self.pool_sender.try_send(buf);
        }
    }

    /// Snapshot the current model version. Phase B' Class-1 probe.
    pub fn current_model_version(&self) -> u64 {
        self.inner.model_version.load(Ordering::Relaxed)
    }

    /// Whether this batcher was constructed from a `representation="graph"`
    /// spec. The worker loop reads this ONCE per leaf-batch to dispatch to
    /// `infer_and_expand_graph`; `false` for every grid batcher (the dense hot
    /// path after the dispatch branch is byte-identical).
    pub(crate) fn is_graph(&self) -> bool {
        matches!(self.representation, Representation::Graph)
    }

    /// The graph builder's slot-window trunk (= `spec.trunk_size`). Single
    /// source for the expand-time frame trunk (`expand_and_backup_ls_at`) so it
    /// can never drift from the `policy_dst_slot` trunk the builder baked.
    pub(crate) fn graph_trunk_size(&self) -> i32 {
        self.graph_trunk_size
    }

    /// Build one leaf's axis graph from its stones, running the WP-1 red-team
    /// seam guards (`build_graph_from_request`). Returns `None` on any guard
    /// violation (unreachable for a valid self-play board — coords are bounded,
    /// player/moves in range — so the worker degrades like the dense
    /// inference-failure path: skip the batch). Uses the batcher's own graph
    /// `BuildParams` fields (no scattered literals — seam design §3.2).
    pub(crate) fn build_leaf_graph(
        &self,
        stones: &[(i64, i64, i64)],
        current_player: i64,
        moves_remaining: i64,
    ) -> Option<AxisGraph> {
        build_graph_from_request(
            stones,
            current_player,
            moves_remaining,
            self.graph_win_length,
            self.graph_radius,
            self.graph_trunk_size,
        )
        .ok()
    }

    /// N5 helper: wake + drop every still-pending graph waiter in `ids` with
    /// `err_msg` so a mid-loop error return in `submit_graph_inference_results`
    /// never orphans a blocked worker (a waiter whose result is already set is
    /// left untouched — tolerant, idempotent with `submit_graph_inference_failure`).
    fn fail_remaining_graph_ids(&self, ids: &[u64], err_msg: &str) {
        for &fid in ids {
            self.inner.in_flight_graphs.remove(&fid);
            if let Some((_, waiter)) = self.inner.graph_waiters.remove(&fid) {
                let mut g = waiter.result.lock().expect("graph waiter lock poisoned");
                if g.is_none() {
                    *g = Some(Err(err_msg.to_string()));
                }
                waiter.cv.notify_all();
            }
        }
    }

    /// Guard: a graph seam method requires a `representation="graph"` batcher.
    /// A grid batcher (every dense construction) raises `RepresentationMismatch`
    /// (seam design §5-below / D-EVALGATE loud-fail precedent).
    fn require_graph(&self) -> PyResult<()> {
        if !matches!(self.representation, Representation::Graph) {
            return Err(PyValueError::new_err(
                "RepresentationMismatch: graph seam method called on a grid InferenceBatcher \
                 (construct with a representation=\"graph\" encoding spec)",
            ));
        }
        Ok(())
    }
}

#[pymethods]
impl InferenceBatcher {
    #[new]
    #[pyo3(signature = (encoding_spec = None, feature_len = None, policy_len = None, pool_size = None))]
    pub fn new(
        encoding_spec: Option<crate::PyRegistrySpec>,
        feature_len: Option<usize>,
        policy_len: Option<usize>,
        pool_size: Option<usize>,
    ) -> PyResult<Self> {
        // §172 A10 T8b / cycle 3 Wave 8 Batch C FF.10 — derive feature_len /
        // policy_len from `encoding_spec` when explicit kwargs are omitted.
        // The 3 legacy `audit: legacy-v6-fallback` arms (none-and-no-spec)
        // retire to `PyValueError` so a v8 caller who omits everything no
        // longer silently inherits v6 geometry (2888 / 362).
        //
        // Precedence: explicit kwargs > encoding_spec derivation > error.
        let spec_static = encoding_spec.as_ref().map(super::pyo3::encoding::PyRegistrySpec::inner);
        let (feature_len, policy_len) = match (feature_len, policy_len, spec_static) {
            (Some(f), Some(p), _) => (f, p),
            (None, None, Some(spec)) => (spec.state_stride(), spec.policy_stride()),
            (Some(f), None, Some(spec)) => (f, spec.policy_stride()),
            (None, Some(p), Some(spec)) => (spec.state_stride(), p),
            (None, _, None) | (_, None, None) => {
                return Err(PyValueError::new_err(
                    "InferenceBatcher: encoding_spec required when feature_len/policy_len omitted \
                     (cycle 3 Wave 8 Batch C FF.10 retired the legacy v6 fallback arms)",
                ));
            }
        };

        // ── WP-3 graph seam: representation-aware construction (seam design
        // §5-below). A `representation="graph"` spec constructs a graph-capable
        // batcher (the parallel graph queue is used; the dense feature-buffer
        // pool is NOT prefilled — a graph has no fixed `feature_len`). Every
        // other construction path (explicit lens, grid spec, no spec) stays
        // Grid → the dense hot path is byte-identical. A graph spec's geometry
        // is read here so `BuildParams` carries no scattered literals.
        let representation = spec_static.map_or(Representation::Grid, |s| s.representation);
        let (graph_win_length, graph_radius, graph_trunk_size, graph_contract_version) =
            if let (Representation::Graph, Some(spec)) = (representation, spec_static) {
                // N1: `validate()` guarantees these `Some` for a graph spec, so a
                // literal `6`/`1` fallback is dead code that would mask a
                // registry desync — die loud instead (design §5 no-literals rule).
                (
                    spec.win_length.expect("validate guarantees win_length for a graph spec") as u8,
                    spec.graph_radius.expect("validate guarantees graph_radius for a graph spec") as u16,
                    spec.trunk_size as i32,
                    spec.contract_version.expect("validate guarantees contract_version for a graph spec"),
                )
            } else {
                (0, 0, 0, 1)
            };
        let is_graph = matches!(representation, Representation::Graph);

        // §P55: pool_size = None preserves cycle 1 fixed 512 pre-send + 1024 channel.
        // When caller opts in (e.g. v6w25 16-worker mid-game K_avg≈6 → 768 working set
        // exceeds 512), channel grows to max(pool_size * 2, 1024) so try_send pre-fill
        // doesn't silently drop into a full channel.
        let prefill = pool_size.unwrap_or(512);
        let channel_cap = pool_size.map_or(1024, |n| (n * 2).max(1024));
        let (pool_sender, pool_receiver) = flume::bounded(channel_cap);
        // Graph batcher: no dense feature-buffer pool (the ragged union Vec is
        // moved into numpy, never recycled — seam design §4.2). Prefilling a
        // `feature_len`-sized pool would be meaningless (a graph has no
        // feature_len), so skip it. Dense path unchanged.
        if !is_graph {
            for _ in 0..prefill {
                let _ = pool_sender.send(vec![0.0f32; feature_len]);
            }
        }

        Ok(Self {
            inner: Arc::new(Inner::new()),
            feature_len,
            policy_len,
            pool_sender,
            pool_receiver,
            representation,
            graph_win_length,
            graph_radius,
            graph_trunk_size,
            graph_contract_version,
        })
    }


    /// Spawn N mock game requests on native threads (test utility).
    pub fn spawn_mock_games(&self, n_games: usize) {
        let inner = self.inner.clone();
        let feature_len = self.feature_len;
        let policy_len = self.policy_len;

        for _ in 0..n_games {
            let inner_clone = inner.clone();
            std::thread::spawn(move || {
                let features = vec![0.0_f32; feature_len];
                let res = inner_clone.submit_and_wait(features, feature_len, policy_len);
                if res.is_ok() {
                    inner_clone.completed_mock_games.fetch_add(1, Ordering::SeqCst);
                }
            });
        }
    }

    /// Number of completed mock games (for test assertions).
    pub fn completed_mock_games(&self) -> usize {
        self.inner.completed_mock_games.load(Ordering::SeqCst)
    }

    /// Return whether at least one request is currently queued.
    pub fn has_pending_requests(&self) -> bool {
        let queue = self.inner.queue.lock().expect("queue lock poisoned");
        !queue.is_empty()
    }

    /// Block until at least one request is available or timeout expires.
    ///
    /// Returns:
    ///   (request_ids, fused_features) where fused_features has shape
    ///   [batch, feature_len]. Returns empty arrays on timeout.
    #[pyo3(signature = (batch_size, max_wait_ms = 10))]
    pub fn next_inference_batch<'py>(
        &self,
        py: Python<'py>,
        batch_size: usize,
        max_wait_ms: u64,
    ) -> PyResult<(Vec<u64>, Bound<'py, PyArray2<f32>>)> {
        if batch_size == 0 {
            return Err(PyValueError::new_err("batch_size must be > 0"));
        }

        let pulled = py.detach(|| self.inner.pop_batch_blocking(batch_size, max_wait_ms));

        if pulled.is_empty() {
            // Return an explicit 0xfeature_len tensor for timeout/no-work polls.
            // Using from_vec2 on an empty Vec can raise and kill the Python
            // inference thread, which then deadlocks submit_batch_and_wait_rust callers.
            let arr = PyArray2::<f32>::zeros(py, [0, self.feature_len], false);
            return Ok((Vec::new(), arr));
        }

        let n = pulled.len();
        let mut ids = Vec::with_capacity(n);
        // Optimize: use a single flat vector to avoid Vec<Vec<f32>> overhead
        let mut flat_features = Vec::with_capacity(n * self.feature_len);
        for req in pulled {
            ids.push(req.id);
            flat_features.extend(&req.features);
            self.return_feature_buffer(req.features);
        }

        let arr = PyArray1::from_vec(py, flat_features)
            .reshape([n, self.feature_len])?;
        Ok((ids, arr))
    }

    /// Submit inference outputs and wake corresponding waiting requests.
    pub fn submit_inference_results(
        &self,
        request_ids: Vec<u64>,
        policies: Bound<'_, PyArray2<f32>>,
        values: Bound<'_, PyArray1<f32>>,
    ) -> PyResult<()> {
        let n = request_ids.len();
        if policies.shape()[0] != n || values.len() != n {
            return Err(PyValueError::new_err(format!(
                "length mismatch ids/policies/values: {}/{}/{}",
                n,
                policies.shape()[0],
                values.len()
            )));
        }

        if policies.shape()[1] != self.policy_len {
            return Err(PyValueError::new_err(format!(
                "policy length mismatch: expected {}, got {}",
                self.policy_len,
                policies.shape()[1]
            )));
        }

        // Access numpy arrays via read-only views for performance
        let policies_view = policies.readonly();
        let policies_slice = policies_view.as_slice()?;
        let values_view = values.readonly();
        let values_slice = values_view.as_slice()?;

        // §P74: wrap the whole policies buffer in `Arc<Vec<f32>>` ONCE.
        // The bulk copy from numpy-owned memory into a Rust-owned Vec
        // happens here exactly once per submit_inference_results call.
        // Each waiter receives an `Arc::clone(&shared_policies)` + its
        // own (start..end) range, replacing N per-row
        // `policies_slice[start..end].to_vec()` heap allocations with
        // N refcount bumps. Total bytes copied are unchanged; alloc
        // count drops from N → 1 (plus N Arc bumps).
        //
        // GIL-safety (§F risk 2 of Wave 5a PREP): `Arc::clone` happens
        // OUTSIDE any waiter mutex. The pattern is: clone → enter
        // waiter.result.lock() → store → drop guard → notify.
        // Mirrors the Wave 4 P7 TTEntry.policy = Arc<Vec<f32>> pattern
        // (engine/src/mcts/backup.rs:~298 / ~318).
        let shared_policies: Arc<Vec<f32>> = Arc::new(policies_slice.to_vec());

        for i in 0..n {
            let id = request_ids[i];
            if let Some((_, waiter)) = self.inner.waiters.remove(&id) {
                let start = i * self.policy_len;
                let end = start + self.policy_len;
                let policy_arc = Arc::clone(&shared_policies);
                let value = values_slice[i];

                let mut result_guard = waiter.result.lock().expect("waiter lock poisoned");
                *result_guard = Some(Ok((policy_arc, start..end, value)));
                waiter.cv.notify_all();
            }
        }
        Ok(())
    }

    /// Signal failure for a batch of requests.
    pub fn submit_inference_failure(
        &self,
        request_ids: Vec<u64>,
        error_msg: String,
    ) -> PyResult<()> {
        for id in request_ids {
            if let Some((_, waiter)) = self.inner.waiters.remove(&id) {
                let mut result_guard = waiter.result.lock().expect("waiter lock poisoned");
                *result_guard = Some(Err(error_msg.clone()));
                waiter.cv.notify_all();
            }
        }
        Ok(())
    }

    /// Close the queue and wake all blocked waiters.
    pub fn close(&self) {
        self.close_rust();
    }

    /// Increment the monotonic model version. Called by Python's
    /// `InferenceServer.load_state_dict_safe()` on every weight swap.
    /// Phase B' Class-1 instrumentation; cheap (relaxed atomic add).
    pub fn bump_model_version(&self) -> u64 {
        self.inner.model_version.fetch_add(1, Ordering::Relaxed) + 1
    }

    /// Read the current model version (snapshot). Useful from Python for
    /// dashboard / event payloads.
    #[getter]
    pub fn model_version(&self) -> u64 {
        self.inner.model_version.load(Ordering::Relaxed)
    }

    #[getter]
    pub fn feature_len_py(&self) -> usize {
        self.feature_len
    }

    #[getter]
    pub fn policy_len_py(&self) -> usize {
        self.policy_len
    }

    /// Wire `representation` ("grid" | "graph").
    #[getter]
    pub fn representation_py(&self) -> &'static str {
        self.representation.as_str()
    }

    // ── WP-3 dormant graph seam pymethods ────────────────────────────────
    // All guard `require_graph()` — a grid batcher raises RepresentationMismatch.

    /// Whether at least one graph request is queued.
    pub fn has_pending_graph_requests(&self) -> bool {
        let q = self.inner.graph_queue.lock().expect("graph queue lock poisoned");
        !q.is_empty()
    }

    /// Number of completed mock graph games (test assertions).
    pub fn completed_graph_games(&self) -> usize {
        self.inner.completed_graph_games.load(Ordering::SeqCst)
    }

    /// Seam obligations only: build a graph from the request params (running the
    /// coord / current_player / moves_remaining range guards) and discard it.
    /// Raises `ValueError` on any violation. Test/CI surface for the WP-1
    /// red-team seam obligations (no queue interaction).
    pub fn check_graph_request(
        &self,
        stones: Vec<(i64, i64, i64)>,
        current_player: i64,
        moves_remaining: i64,
    ) -> PyResult<()> {
        self.require_graph()?;
        let _ = build_graph_from_request(
            &stones,
            current_player,
            moves_remaining,
            self.graph_win_length,
            self.graph_radius,
            self.graph_trunk_size,
        )?;
        Ok(())
    }

    /// Spawn N mock graph games on native threads (test utility; mirrors
    /// `spawn_mock_games`). Each builds a FIXED mixed spread board (two far
    /// clusters → in-window + off-window legal nodes) and blocks on
    /// `submit_batch_and_wait_graph_rust`, so the caller can drive
    /// `next_graph_batch` → `submit_graph_inference_results` and observe
    /// completion via `completed_graph_games`.
    pub fn spawn_mock_graph_games(&self, n_games: usize) -> PyResult<()> {
        self.require_graph()?;
        for _ in 0..n_games {
            let b = self.clone();
            std::thread::spawn(move || {
                let mut stones: Vec<(i64, i64, i64)> = Vec::new();
                for q in 0..5i64 {
                    stones.push((q, 0, 1));
                }
                for q in 30..35i64 {
                    stones.push((q, 0, -1));
                }
                if let Ok(graph) = build_graph_from_request(
                    &stones,
                    1,
                    2,
                    b.graph_win_length,
                    b.graph_radius,
                    b.graph_trunk_size,
                ) {
                    if b.submit_batch_and_wait_graph_rust(vec![graph]).is_ok() {
                        b.inner.completed_graph_games.fetch_add(1, Ordering::SeqCst);
                    }
                }
            });
        }
        Ok(())
    }

    /// Pop up to `batch_size` graph requests and fuse them PyG block-diagonal
    /// (NO padding — contract §2.1/§2.2). Returns `(request_ids, GraphWire)`.
    /// `edge_index`/`legal_node_gather`/offsets are emitted ALREADY globally
    /// offset (i64) so Python does no index arithmetic. Retains per-id assemble
    /// metadata (`policy_dst_slot` + legal coords) for
    /// `submit_graph_inference_results`.
    #[pyo3(signature = (batch_size, max_wait_ms = 10))]
    pub fn next_graph_batch(
        &self,
        py: Python<'_>,
        batch_size: usize,
        max_wait_ms: u64,
    ) -> PyResult<(Vec<u64>, GraphWire)> {
        self.require_graph()?;
        if batch_size == 0 {
            return Err(PyValueError::new_err("batch_size must be > 0"));
        }

        let pulled = py.detach(|| self.inner.pop_graph_batch_blocking(batch_size, max_wait_ms));

        let mut ids: Vec<u64> = Vec::with_capacity(pulled.len());
        let mut node_feat: Vec<f32> = Vec::new();
        let mut node_coords: Vec<i32> = Vec::new();
        let mut edge_attr: Vec<f32> = Vec::new();
        let mut edge_src: Vec<i64> = Vec::new();
        let mut edge_dst: Vec<i64> = Vec::new();
        let mut legal_node_gather: Vec<i64> = Vec::new();
        let mut policy_dst_slot: Vec<i32> = Vec::new();
        let mut node_offsets: Vec<i64> = vec![0];
        let mut edge_offsets: Vec<i64> = vec![0];
        let mut legal_offsets: Vec<i64> = vec![0];
        let mut n_nodes_checksum: Vec<u32> = Vec::with_capacity(pulled.len());
        let mut n_stones: Vec<u16> = Vec::with_capacity(pulled.len());
        let mut window_center: Vec<i32> = Vec::with_capacity(pulled.len() * 2);
        let mut current_player: Vec<i8> = Vec::with_capacity(pulled.len());

        let mut node_off: i64 = 0;
        let mut edge_off: i64 = 0;
        let mut legal_off: i64 = 0;

        for req in pulled {
            let id = req.id;
            let g = req.graph;
            // builder_impl handshake (defense-in-depth; the build path already
            // asserted it) — a non-native tag must never reach the wire.
            if g.builder_impl != BUILDER_IMPL_NATIVE {
                return Err(PyValueError::new_err(
                    "next_graph_batch: non-native builder_impl on a queued graph",
                ));
            }
            ids.push(id);
            let n_g = g.num_nodes() as i64;
            let e_g = g.num_edges() as i64;
            let lg_g = g.legal_node_gather.len() as i64;

            // legal coords (before node_coords is moved into the wire).
            let legal_coords: Vec<(i32, i32)> = g
                .legal_node_gather
                .iter()
                .map(|&row| {
                    (
                        g.node_coords[row as usize * 2],
                        g.node_coords[row as usize * 2 + 1],
                    )
                })
                .collect();

            node_feat.extend_from_slice(&g.node_feat.0);
            edge_attr.extend_from_slice(&g.edge_attr.0);
            for &s in &g.edge_index.src {
                edge_src.push(node_off + i64::from(s));
            }
            for &d in &g.edge_index.dst {
                edge_dst.push(node_off + i64::from(d));
            }
            for &row in &g.legal_node_gather {
                legal_node_gather.push(node_off + i64::from(row));
            }
            policy_dst_slot.extend_from_slice(&g.policy_scatter_index.0);
            n_nodes_checksum.push(g.n_nodes_checksum);
            n_stones.push(g.n_stones);
            window_center.push(g.window_center.0);
            window_center.push(g.window_center.1);
            current_player.push(g.current_player);
            node_coords.extend_from_slice(&g.node_coords);

            self.inner.in_flight_graphs.insert(
                id,
                InFlightGraph {
                    policy_dst_slot: g.policy_scatter_index.0.clone(),
                    legal_coords,
                },
            );

            node_off += n_g;
            edge_off += e_g;
            legal_off += lg_g;
            node_offsets.push(node_off);
            edge_offsets.push(edge_off);
            legal_offsets.push(legal_off);
        }

        // edge_index = [src_global (E) | dst_global (E)] → Python reshapes (2,E).
        let mut edge_index = edge_src;
        edge_index.extend(edge_dst);

        let wire = GraphWire {
            contract_version: self.graph_contract_version,
            builder_impl: BUILDER_IMPL_NATIVE,
            n_graphs: ids.len(),
            node_feat,
            node_coords,
            edge_index,
            edge_attr,
            node_offsets,
            edge_offsets,
            legal_offsets,
            legal_node_gather,
            policy_dst_slot,
            n_nodes_checksum,
            n_stones,
            window_center,
            current_player,
        };
        Ok((ids, wire))
    }

    /// Ragged OUTPUT: wake each graph waiter with its assembled
    /// `(LegalSetPolicy, value)` (seam design §3.3/§3.4). `legal_probs_flat` is
    /// the concatenated per-legal-node probs (segment-softmaxed per graph);
    /// `legal_offsets` segments it per id. The per-leaf `LegalSetPolicy` is
    /// built Rust-side from the retained `policy_dst_slot` + coords — never a
    /// Python dense scatter.
    pub fn submit_graph_inference_results(
        &self,
        request_ids: Vec<u64>,
        legal_probs_flat: Bound<'_, PyArray1<f32>>,
        legal_offsets: Bound<'_, PyArray1<i64>>,
        values: Bound<'_, PyArray1<f32>>,
    ) -> PyResult<()> {
        self.require_graph()?;
        let n = request_ids.len();
        if values.len() != n {
            return Err(PyValueError::new_err(format!(
                "length mismatch ids/values: {}/{}",
                n,
                values.len()
            )));
        }
        let lo_view = legal_offsets.readonly();
        let lo = lo_view.as_slice()?;
        if lo.len() != n + 1 {
            return Err(PyValueError::new_err(format!(
                "legal_offsets length {} != n+1 ({})",
                lo.len(),
                n + 1
            )));
        }
        let probs_view = legal_probs_flat.readonly();
        let probs = probs_view.as_slice()?;
        let vals_view = values.readonly();
        let vals = vals_view.as_slice()?;
        if lo[0] != 0 || (lo[n] as usize) != probs.len() {
            return Err(PyValueError::new_err(format!(
                "legal_offsets endpoints [{},{}] inconsistent with legal_probs_flat len {}",
                lo[0],
                lo[n],
                probs.len()
            )));
        }

        for i in 0..n {
            let id = request_ids[i];
            let start = lo[i];
            let end = lo[i + 1];
            if start < 0 || end < start || (end as usize) > probs.len() {
                // N5: fail the whole remaining batch so no waiter is orphaned on
                // a mid-loop error return (the segment slice itself is invalid).
                let msg = format!("legal_offsets segment [{start},{end}] out of range for id {id}");
                self.fail_remaining_graph_ids(&request_ids[i..], &msg);
                return Err(PyValueError::new_err(msg));
            }
            let leaf_probs = &probs[start as usize..end as usize];

            let Some((_, meta)) = self.inner.in_flight_graphs.remove(&id) else {
                // Unknown id (already consumed / never emitted). Skip — the dense
                // path has the same tolerant remove semantics.
                continue;
            };
            if meta.policy_dst_slot.len() != leaf_probs.len() {
                // Segmentation desync — die loud (contract §7). N5: wake THIS
                // waiter + all remaining ids so none is orphaned.
                let msg = format!(
                    "submit_graph_inference_results: segment len {} != n_legal {} for id {id}",
                    leaf_probs.len(),
                    meta.policy_dst_slot.len()
                );
                if let Some((_, waiter)) = self.inner.graph_waiters.remove(&id) {
                    let mut g = waiter.result.lock().expect("graph waiter lock poisoned");
                    *g = Some(Err(msg.clone()));
                    waiter.cv.notify_all();
                }
                self.fail_remaining_graph_ids(&request_ids[i + 1..], &msg);
                return Err(PyValueError::new_err(msg));
            }
            let ls = match assemble_ls_from_gnn_probs(
                self.policy_len,
                leaf_probs,
                &meta.policy_dst_slot,
                &meta.legal_coords,
            ) {
                Ok(ls) => ls,
                Err(e) => {
                    // N4: graceful die-loud (release profile = panic=abort) — wake
                    // this waiter + N5 the remaining batch.
                    if let Some((_, waiter)) = self.inner.graph_waiters.remove(&id) {
                        let mut g = waiter.result.lock().expect("graph waiter lock poisoned");
                        *g = Some(Err(e.clone()));
                        waiter.cv.notify_all();
                    }
                    self.fail_remaining_graph_ids(&request_ids[i + 1..], &e);
                    return Err(PyValueError::new_err(e));
                }
            };
            if let Some((_, waiter)) = self.inner.graph_waiters.remove(&id) {
                let mut g = waiter.result.lock().expect("graph waiter lock poisoned");
                *g = Some(Ok((ls, vals[i])));
                waiter.cv.notify_all();
            }
        }
        Ok(())
    }

    /// Signal failure for a batch of graph requests (loud worker death path,
    /// seam design §7). Wakes each waiter with `Err` and drops in-flight state.
    pub fn submit_graph_inference_failure(
        &self,
        request_ids: Vec<u64>,
        error_msg: String,
    ) -> PyResult<()> {
        self.require_graph()?;
        for id in request_ids {
            self.inner.in_flight_graphs.remove(&id);
            if let Some((_, waiter)) = self.inner.graph_waiters.remove(&id) {
                let mut g = waiter.result.lock().expect("graph waiter lock poisoned");
                *g = Some(Err(error_msg.clone()));
                waiter.cv.notify_all();
            }
        }
        Ok(())
    }

    /// Blocking graph-inference driver for the OQ-7 step-0 smoke (seam design
    /// §7 step 7) + eval subprocess round-trip: builds one axis graph per
    /// `(stones, current_player, moves_remaining)` position (running the WP-1
    /// seam guards), submits the whole batch through the graph queue, releases
    /// the GIL, and blocks until the InferenceServer graph loop assembles each
    /// leaf's `LegalSetPolicy` (the SAME `submit_batch_and_wait_graph_rust` +
    /// `assemble_ls_from_gnn_probs` path the worker's `infer_and_expand_graph`
    /// rides). Returns per-position `(dense[policy_len], overflow[(q,r)->prob],
    /// value)` so Python can inspect the priors the engine actually received —
    /// the round-trip the design's step-7 parity gate demands. NOT a self-play
    /// hot-path method (workers never cross PyO3 per leaf); it exposes the
    /// assembled ragged policy to Python for the smoke.
    #[allow(clippy::type_complexity)]
    pub fn submit_graphs_and_wait(
        &self,
        py: Python<'_>,
        positions: Vec<(Vec<(i64, i64, i64)>, i64, i64)>,
    ) -> PyResult<Vec<(Vec<f32>, Vec<((i32, i32), f32)>, f32)>> {
        self.require_graph()?;
        let mut graphs = Vec::with_capacity(positions.len());
        for (stones, current_player, moves_remaining) in &positions {
            let g = build_graph_from_request(
                stones,
                *current_player,
                *moves_remaining,
                self.graph_win_length,
                self.graph_radius,
                self.graph_trunk_size,
            )?;
            graphs.push(g);
        }
        let results = py
            .detach(|| self.submit_batch_and_wait_graph_rust(graphs))
            .map_err(|()| {
                PyValueError::new_err(
                    "submit_graphs_and_wait: batcher closed or graph inference failed",
                )
            })?;
        let out = results
            .into_iter()
            .map(|(ls, v)| {
                let overflow: Vec<((i32, i32), f32)> =
                    ls.overflow.iter().map(|(&k, &p)| (k, p)).collect();
                (ls.dense, overflow, v)
            })
            .collect();
        Ok(out)
    }
}

/// Block-diagonal ragged graph wire (contract §2.1) — the fuse-out of
/// `next_graph_batch`, the SINGLE payload the Python `collate_graph_batch`
/// resolver reads. Owned flat Vecs; each getter copies into a fresh numpy
/// array (the resolver reads each field once). `edge_index` is `[2*E]`
/// (`[src_global | dst_global]`, Python reshapes to `(2, E)`); all index arrays
/// are i64 and ALREADY globally offset.
#[pyclass(name = "GraphWire")]
pub struct GraphWire {
    contract_version: u32,
    builder_impl: u8,
    n_graphs: usize,
    node_feat: Vec<f32>,
    node_coords: Vec<i32>,
    edge_index: Vec<i64>,
    edge_attr: Vec<f32>,
    node_offsets: Vec<i64>,
    edge_offsets: Vec<i64>,
    legal_offsets: Vec<i64>,
    legal_node_gather: Vec<i64>,
    policy_dst_slot: Vec<i32>,
    n_nodes_checksum: Vec<u32>,
    n_stones: Vec<u16>,
    window_center: Vec<i32>,
    current_player: Vec<i8>,
}

#[pymethods]
impl GraphWire {
    #[getter]
    fn contract_version(&self) -> u32 {
        self.contract_version
    }
    #[getter]
    fn builder_impl(&self) -> u8 {
        self.builder_impl
    }
    #[getter]
    fn n_graphs(&self) -> usize {
        self.n_graphs
    }
    #[getter]
    fn node_feat<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        PyArray1::from_slice(py, &self.node_feat)
    }
    #[getter]
    fn node_coords<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i32>> {
        PyArray1::from_slice(py, &self.node_coords)
    }
    #[getter]
    fn edge_index<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        PyArray1::from_slice(py, &self.edge_index)
    }
    #[getter]
    fn edge_attr<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        PyArray1::from_slice(py, &self.edge_attr)
    }
    #[getter]
    fn node_offsets<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        PyArray1::from_slice(py, &self.node_offsets)
    }
    #[getter]
    fn edge_offsets<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        PyArray1::from_slice(py, &self.edge_offsets)
    }
    #[getter]
    fn legal_offsets<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        PyArray1::from_slice(py, &self.legal_offsets)
    }
    #[getter]
    fn legal_node_gather<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        PyArray1::from_slice(py, &self.legal_node_gather)
    }
    #[getter]
    fn policy_dst_slot<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i32>> {
        PyArray1::from_slice(py, &self.policy_dst_slot)
    }
    #[getter]
    fn n_nodes_checksum<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        PyArray1::from_slice(py, &self.n_nodes_checksum)
    }
    #[getter]
    fn n_stones<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u16>> {
        PyArray1::from_slice(py, &self.n_stones)
    }
    #[getter]
    fn window_center<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i32>> {
        PyArray1::from_slice(py, &self.window_center)
    }
    #[getter]
    fn current_player<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i8>> {
        PyArray1::from_slice(py, &self.current_player)
    }
}

#[cfg(test)]
mod tests {
    //! P69 inline-test scaffold (cycle 3 Wave 10 Batch B). Covers the
    //! two early-return paths in `submit_batch_and_wait_rust`. Future
    //! waves may extend coverage to `infer_and_expand`'s partial-batch
    //! handler in `worker_loop/inner.rs` (deferred per U9 = Option B.2.c).
    use super::*;

    const FEATURE_LEN: usize = 8 * 19 * 19; // v6 default 2888
    const POLICY_LEN: usize = 19 * 19 + 1; // 362

    fn new_batcher() -> InferenceBatcher {
        // PyO3-free ctor arm at inference_bridge.rs:297
        // (Some(f), Some(p), _) => (f, p) — skips encoding_spec deref.
        InferenceBatcher::new(None, Some(FEATURE_LEN), Some(POLICY_LEN), None)
            .expect("explicit feature_len/policy_len must construct")
    }

    #[test]
    fn submit_batch_and_wait_rust_returns_err_when_closed() {
        let batcher = new_batcher();
        batcher.close_rust();
        let res = batcher.submit_batch_and_wait_rust(vec![vec![0.0_f32; FEATURE_LEN]]);
        assert!(res.is_err(), "expected Err(()) on closed batcher");
    }

    #[test]
    fn submit_batch_and_wait_rust_returns_err_on_length_mismatch() {
        let batcher = new_batcher();
        let bad = vec![vec![0.0_f32; FEATURE_LEN + 1]];
        let res = batcher.submit_batch_and_wait_rust(bad);
        assert!(res.is_err(), "expected Err(()) on feature length mismatch");
    }

    // ── WP-3 graph seam: seam obligations + construction (no Python needed) ──

    fn graph_batcher() -> InferenceBatcher {
        let spec = crate::encoding::lookup_or_panic("gnn_axis_v1");
        let py_spec = crate::pyo3::encoding::PyRegistrySpec::from_static(spec);
        InferenceBatcher::new(Some(py_spec), None, None, None)
            .expect("graph batcher constructs from gnn_axis_v1 spec")
    }

    #[test]
    fn grid_batcher_is_grid_representation() {
        let b = new_batcher();
        assert!(matches!(b.representation, Representation::Grid));
        // dense hot path intact: feature buffer pool is prefilled.
        assert!(b.get_feature_buffer().capacity() >= FEATURE_LEN);
    }

    #[test]
    fn graph_batcher_construction_from_spec() {
        let b = graph_batcher();
        assert!(matches!(b.representation, Representation::Graph));
        assert_eq!(b.policy_len, 362, "graph action space unchanged");
        assert_eq!(b.graph_win_length, 6);
        assert_eq!(b.graph_radius, 6);
        assert_eq!(b.graph_trunk_size, 19);
        assert_eq!(b.graph_contract_version, 1);
        // no dense feature buffers prefilled for a graph batcher.
        assert!(b.pool_receiver.try_recv().is_err(), "graph batcher must not prefill the dense pool");
    }

    #[test]
    fn build_graph_from_request_valid_stamps_native() {
        let stones = vec![(0i64, 0i64, 1i64), (1, 0, -1), (0, 1, 1)];
        let g = build_graph_from_request(&stones, 1, 2, 6, 6, 19).expect("valid request builds");
        assert_eq!(g.builder_impl, BUILDER_IMPL_NATIVE);
        assert_eq!(g.n_stones, 3);
    }

    #[test]
    fn build_graph_from_request_rejects_bad_current_player() {
        let stones = vec![(0i64, 0i64, 1i64)];
        assert!(build_graph_from_request(&stones, 2, 2, 6, 6, 19).is_err());
        assert!(build_graph_from_request(&stones, 0, 2, 6, 6, 19).is_err());
    }

    #[test]
    fn build_graph_from_request_rejects_bad_moves_remaining() {
        let stones = vec![(0i64, 0i64, 1i64)];
        assert!(build_graph_from_request(&stones, 1, -1, 6, 6, 19).is_err());
        assert!(build_graph_from_request(&stones, 1, 256, 6, 6, 19).is_err());
    }

    #[test]
    fn build_graph_from_request_rejects_coord_overflow() {
        // WP1 Attack-2: a coord near i32::MAX would wrap silently in release.
        let stones = vec![(i64::from(i32::MAX), 0i64, 1i64)];
        assert!(build_graph_from_request(&stones, 1, 2, 6, 6, 19).is_err());
    }

    #[test]
    fn build_graph_from_request_rejects_bad_player() {
        let stones = vec![(0i64, 0i64, 5i64)];
        assert!(build_graph_from_request(&stones, 1, 2, 6, 6, 19).is_err());
    }

    #[test]
    fn graph_queue_round_trip_assembles_ls() {
        // Rust-side full round-trip WITHOUT Python: a background thread submits a
        // graph and blocks; the main thread pops it, assembles a uniform LS from
        // the retained slot map, wakes the waiter, and the thread returns the LS.
        use std::sync::atomic::AtomicBool;
        let b = graph_batcher();
        let stones = vec![(0i64, 0i64, 1i64), (30, 0, -1), (31, 0, -1)];
        let graph = build_graph_from_request(&stones, 1, 2, 6, 6, 19).unwrap();
        let n_legal = graph.legal_node_gather.len();
        assert!(n_legal > 0);

        let bt = b.clone();
        let done = std::sync::Arc::new(AtomicBool::new(false));
        let done2 = done.clone();
        let handle = std::thread::spawn(move || {
            let res = bt.submit_batch_and_wait_graph_rust(vec![graph]);
            done2.store(true, Ordering::SeqCst);
            res
        });

        // main thread: pop + assemble + wake via the internal maps.
        let pulled = loop {
            let p = b.inner.pop_graph_batch_blocking(1, 200);
            if !p.is_empty() {
                break p;
            }
            assert!(!done.load(Ordering::SeqCst), "thread finished before pop");
        };
        assert_eq!(pulled.len(), 1);
        let id = pulled[0].id;
        let g = &pulled[0].graph;
        let coords: Vec<(i32, i32)> = g
            .legal_node_gather
            .iter()
            .map(|&row| (g.node_coords[row as usize * 2], g.node_coords[row as usize * 2 + 1]))
            .collect();
        let probs = vec![1.0f32 / n_legal as f32; n_legal];
        let ls = assemble_ls_from_gnn_probs(b.policy_len, &probs, &g.policy_scatter_index.0, &coords)
            .expect("assemble ok");
        // wake the waiter directly (mirrors submit_graph_inference_results).
        let (_, waiter) = b.inner.graph_waiters.remove(&id).expect("waiter registered");
        *waiter.result.lock().unwrap() = Some(Ok((ls, 0.5)));
        waiter.cv.notify_all();

        let out = handle.join().unwrap().expect("graph round-trip ok");
        assert_eq!(out.len(), 1);
        let (ls_out, v_out) = &out[0];
        assert!((v_out - 0.5).abs() < 1e-6);
        let total: f32 = ls_out.dense.iter().sum::<f32>() + ls_out.overflow.values().sum::<f32>();
        assert!((total - 1.0).abs() < 1e-4, "assembled LS is a distribution, sum={total}");
    }

    #[test]
    fn worker_graph_glue_builds_from_board_cells_and_expands() {
        // Mirrors `infer_and_expand_graph` (worker_loop/inner.rs) END-TO-END on a
        // real Board + MCTSTree WITHOUT the InferenceServer: extract stones the
        // worker way (`cells_iter` + `Cell as i64`), build the leaf graph, assemble
        // an LS, and run `expand_and_backup_ls_at` with the BUILDER's window_center
        // (S2). Closes the runtime gap on the worker glue that the Python smoke
        // (explicit-stones `submit_graphs_and_wait`) does not cover.
        use crate::board::Board;
        use crate::mcts::MCTSTree;

        let spec = crate::encoding::lookup_or_panic("gnn_axis_v1");
        let mut board = Board::with_registry_spec(spec);
        // A few LEGAL moves (adjacency-constrained) → a small in-window board.
        for &(q, r) in &[(0, 0), (1, 0), (0, 1), (1, 1), (2, 0)] {
            board.apply_move(q, r).expect("legal move");
        }

        // Worker-path stone extraction: cells_iter + `Cell as i64` (±1).
        let mut stones: Vec<(i64, i64, i64)> = Vec::new();
        for (&(q, r), &cell) in board.cells_iter() {
            stones.push((i64::from(q), i64::from(r), cell as i64));
        }
        assert_eq!(stones.len(), 5, "5 stones extracted from the board cells");
        let current_player = board.current_player as i64;
        let moves_remaining = i64::from(board.moves_remaining);

        let g = build_graph_from_request(&stones, current_player, moves_remaining, 6, 6, 19)
            .expect("worker builds the leaf graph from board cells");
        // Builder window_center must equal Board::window_center (the S2 coincidence).
        assert_eq!(
            g.window_center,
            board.window_center(),
            "builder window_center == Board::window_center (S2 frame identity)"
        );

        let n_legal = g.legal_node_gather.len();
        assert!(n_legal > 0);
        let coords: Vec<(i32, i32)> = g
            .legal_node_gather
            .iter()
            .map(|&row| (g.node_coords[row as usize * 2], g.node_coords[row as usize * 2 + 1]))
            .collect();
        let probs = vec![1.0f32 / n_legal as f32; n_legal];
        let ls = assemble_ls_from_gnn_probs(362, &probs, &g.policy_scatter_index.0, &coords)
            .expect("assemble ok");

        // Real tree: new_game + select_leaves(1) populates pending, then expand at
        // the builder frame (the worker's `expand_and_backup_ls_at` call).
        let mut tree = MCTSTree::new(1.5);
        tree.new_game(board.clone());
        let leaves = tree.select_leaves(1);
        assert_eq!(leaves.len(), 1, "root leaf selected");
        tree.expand_and_backup_ls_at(&[ls], &[0.5], &[g.window_center], 19);

        // The root expanded with children priored from the assembled LS.
        assert!(tree.pool[0].is_expanded(), "root expanded via expand_and_backup_ls_at");
        assert!(tree.pool[0].n_children > 0, "root has priored children");
    }
}
