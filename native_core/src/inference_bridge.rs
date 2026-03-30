use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use numpy::PyArray2;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[derive(Clone)]
struct PendingRequest {
    id: u64,
    features: Vec<f32>,
}

#[derive(Default)]
struct Waiter {
    result: Mutex<Option<(Vec<f32>, f32)>>,
    cv: Condvar,
}

struct Inner {
    queue: Mutex<VecDeque<PendingRequest>>,
    queue_cv: Condvar,
    waiters: Mutex<HashMap<u64, Arc<Waiter>>>,
    next_id: AtomicU64,
    closed: AtomicBool,
    completed_mock_games: AtomicUsize,
}

impl Inner {
    fn new() -> Self {
        Self {
            queue: Mutex::new(VecDeque::new()),
            queue_cv: Condvar::new(),
            waiters: Mutex::new(HashMap::new()),
            next_id: AtomicU64::new(1),
            closed: AtomicBool::new(false),
            completed_mock_games: AtomicUsize::new(0),
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

        {
            let mut waiters = self.waiters.lock().expect("waiters lock poisoned");
            waiters.insert(id, waiter.clone());
        }

        {
            let mut queue = self.queue.lock().expect("queue lock poisoned");
            queue.push_back(PendingRequest { id, features });
            self.queue_cv.notify_all();
        }

        let mut guard = waiter.result.lock().expect("waiter lock poisoned");
        loop {
            if let Some((policy, value)) = guard.take() {
                if policy.len() != expected_policy_len {
                    return Err(PyValueError::new_err(format!(
                        "policy length mismatch for request {id}: got {}, expected {}",
                        policy.len(),
                        expected_policy_len
                    )));
                }
                return Ok((policy, value));
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
        while queue.is_empty() && !self.closed.load(Ordering::SeqCst) {
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
/// Worker threads call `submit_request_and_wait`, which enqueues features and
/// blocks. Python calls `next_inference_batch` to fetch a fused batch tensor,
/// runs the model, then calls `submit_inference_results` to wake waiters.
#[pyclass(name = "RustInferenceBatcher")]
#[derive(Clone)]
pub struct RustInferenceBatcher {
    inner: Arc<Inner>,
    feature_len: usize,
    policy_len: usize,
}

impl RustInferenceBatcher {
    pub(crate) fn submit_request_and_wait_rust(
        &self,
        features: Vec<f32>,
    ) -> Result<(Vec<f32>, f32), ()> {
        if features.len() != self.feature_len {
            return Err(());
        }
        if self.inner.closed.load(Ordering::SeqCst) {
            return Err(());
        }

        let id = self.inner.next_id.fetch_add(1, Ordering::SeqCst);
        let waiter = Arc::new(Waiter::default());

        {
            let mut waiters = self.inner.waiters.lock().expect("waiters lock poisoned");
            waiters.insert(id, waiter.clone());
        }

        {
            let mut queue = self.inner.queue.lock().expect("queue lock poisoned");
            queue.push_back(PendingRequest { id, features });
            self.inner.queue_cv.notify_all();
        }

        let mut guard = waiter.result.lock().expect("waiter lock poisoned");
        loop {
            if let Some((policy, value)) = guard.take() {
                if policy.len() != self.policy_len {
                    return Err(());
                }
                return Ok((policy, value));
            }

            if self.inner.closed.load(Ordering::SeqCst) {
                return Err(());
            }

            guard = waiter.cv.wait(guard).expect("waiter condvar poisoned");
        }
    }

    pub(crate) fn close_rust(&self) {
        self.inner.closed.store(true, Ordering::SeqCst);
        self.inner.queue_cv.notify_all();

        let waiters = self.inner.waiters.lock().expect("waiters lock poisoned");
        for waiter in waiters.values() {
            waiter.cv.notify_all();
        }
    }

    pub(crate) fn feature_len(&self) -> usize {
        self.feature_len
    }
}

#[pymethods]
impl RustInferenceBatcher {
    #[new]
    #[pyo3(signature = (feature_len = 18 * 19 * 19, policy_len = 19 * 19 + 1))]
    pub fn new(feature_len: usize, policy_len: usize) -> Self {
        Self {
            inner: Arc::new(Inner::new()),
            feature_len,
            policy_len,
        }
    }

    /// Submit one request and block until Python returns its policy/value.
    pub fn submit_request_and_wait(
        &self,
        py: Python<'_>,
        features: Vec<f32>,
    ) -> PyResult<(Vec<f32>, f32)> {
        py.allow_threads(|| {
            self.inner
                .submit_and_wait(features, self.feature_len, self.policy_len)
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

        let pulled = py.allow_threads(|| self.inner.pop_batch_blocking(batch_size, max_wait_ms));

        if pulled.is_empty() {
            // Return an explicit 0xfeature_len tensor for timeout/no-work polls.
            // Using from_vec2 on an empty Vec can raise and kill the Python
            // inference thread, which then deadlocks submit_request_and_wait callers.
            let arr = PyArray2::<f32>::zeros(py, [0, self.feature_len], false);
            return Ok((Vec::new(), arr));
        }

        let mut ids = Vec::with_capacity(pulled.len());
        let mut rows = Vec::with_capacity(pulled.len());
        for req in pulled {
            ids.push(req.id);
            rows.push(req.features);
        }

        let arr = PyArray2::from_vec2(py, &rows)?;
        Ok((ids, arr))
    }

    /// Submit inference outputs and wake corresponding waiting requests.
    pub fn submit_inference_results(
        &self,
        request_ids: Vec<u64>,
        policies: Vec<Vec<f32>>,
        values: Vec<f32>,
    ) -> PyResult<()> {
        if request_ids.len() != policies.len() || request_ids.len() != values.len() {
            return Err(PyValueError::new_err(format!(
                "length mismatch ids/policies/values: {}/{}/{}",
                request_ids.len(),
                policies.len(),
                values.len()
            )));
        }

        let mut waiters = self.inner.waiters.lock().expect("waiters lock poisoned");
        for i in 0..request_ids.len() {
            let id = request_ids[i];
            if let Some(waiter) = waiters.remove(&id) {
                if policies[i].len() != self.policy_len {
                    return Err(PyValueError::new_err(format!(
                        "policy length mismatch for request {id}: got {}, expected {}",
                        policies[i].len(),
                        self.policy_len
                    )));
                }

                let mut result_guard = waiter.result.lock().expect("waiter lock poisoned");
                *result_guard = Some((policies[i].clone(), values[i]));
                waiter.cv.notify_all();
            }
        }
        Ok(())
    }

    /// Close the queue and wake all blocked waiters.
    pub fn close(&self) {
        self.close_rust();
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
