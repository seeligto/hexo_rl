use std::collections::{VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use dashmap::DashMap;
use fxhash::FxBuildHasher;
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[derive(Clone)]
struct PendingRequest {
    id: u64,
    features: Vec<f32>,
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
        }
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
}

#[allow(dead_code)]
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

        for r in &self.inner.waiters {
            r.value().cv.notify_all();
        }
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

        // §P55: pool_size = None preserves cycle 1 fixed 512 pre-send + 1024 channel.
        // When caller opts in (e.g. v6w25 16-worker mid-game K_avg≈6 → 768 working set
        // exceeds 512), channel grows to max(pool_size * 2, 1024) so try_send pre-fill
        // doesn't silently drop into a full channel.
        let prefill = pool_size.unwrap_or(512);
        let channel_cap = pool_size.map_or(1024, |n| (n * 2).max(1024));
        let (pool_sender, pool_receiver) = flume::bounded(channel_cap);
        for _ in 0..prefill {
            let _ = pool_sender.send(vec![0.0f32; feature_len]);
        }

        Ok(Self {
            inner: Arc::new(Inner::new()),
            feature_len,
            policy_len,
            pool_sender,
            pool_receiver,
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
}
