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

#[derive(Default)]
struct Waiter {
    result: Mutex<Option<Result<(Vec<f32>, f32), String>>>,
    cv: Condvar,
}

struct Inner {
    queue: Mutex<VecDeque<PendingRequest>>,
    queue_cv: Condvar,
    waiters: DashMap<u64, Arc<Waiter>, FxBuildHasher>,
    next_id: AtomicU64,
    closed: AtomicBool,
    completed_mock_games: AtomicUsize,
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
                match res {
                    Ok((policy, value)) => {
                        if policy.len() != expected_policy_len {
                            return Err(PyValueError::new_err(format!(
                                "policy length mismatch for request {id}: got {}, expected {}",
                                policy.len(),
                                expected_policy_len
                            )));
                        }
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
/// Worker threads call `submit_request_and_wait`, which enqueues features and
/// blocks. Python calls `next_inference_batch` to fetch a fused batch tensor,
/// runs the model, then calls `submit_inference_results` to wake waiters.
#[pyclass(name = "RustInferenceBatcher")]
#[derive(Clone)]
pub struct RustInferenceBatcher {
    inner: Arc<Inner>,
    feature_len: usize,
    policy_len: usize,
    pool_sender: flume::Sender<Vec<f32>>,
    pool_receiver: flume::Receiver<Vec<f32>>,
}

#[allow(dead_code)]
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

        self.inner.waiters.insert(id, waiter.clone());

        {
            let mut queue = self.inner.queue.lock().expect("queue lock poisoned");
            queue.push_back(PendingRequest { id, features });
            self.inner.queue_cv.notify_all();
        }

        let mut guard = waiter.result.lock().expect("waiter lock poisoned");
        loop {
            if let Some(res) = guard.take() {
                match res {
                    Ok((policy, value)) => {
                        if policy.len() != self.policy_len {
                            return Err(());
                        }
                        return Ok((policy, value));
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
                    match res {
                        Ok((policy, value)) => {
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

        for r in self.inner.waiters.iter() {
            r.value().cv.notify_all();
        }
    }

    pub(crate) fn feature_len(&self) -> usize {
        self.feature_len
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
}

#[pymethods]
impl RustInferenceBatcher {
    #[new]
    #[pyo3(signature = (feature_len = 18 * 19 * 19, policy_len = 19 * 19 + 1))]
    pub fn new(feature_len: usize, policy_len: usize) -> Self {
        let (pool_sender, pool_receiver) = flume::bounded(1024);
        for _ in 0..512 {
            let _ = pool_sender.send(vec![0.0f32; feature_len]);
        }

        Self {
            inner: Arc::new(Inner::new()),
            feature_len,
            policy_len,
            pool_sender,
            pool_receiver,
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

        for i in 0..n {
            let id = request_ids[i];
            if let Some((_, waiter)) = self.inner.waiters.remove(&id) {
                let start = i * self.policy_len;
                let end = start + self.policy_len;
                let policy_vec = policies_slice[start..end].to_vec();
                let value = values_slice[i];

                let mut result_guard = waiter.result.lock().expect("waiter lock poisoned");
                *result_guard = Some(Ok((policy_vec, value)));
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

    #[getter]
    pub fn feature_len_py(&self) -> usize {
        self.feature_len
    }

    #[getter]
    pub fn policy_len_py(&self) -> usize {
        self.policy_len
    }
}
