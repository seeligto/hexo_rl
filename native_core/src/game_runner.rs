use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};

use pyo3::prelude::*;

use crate::inference_bridge::RustInferenceBatcher;

#[pyclass(name = "RustSelfPlayRunner")]
pub struct RustSelfPlayRunner {
    batcher: RustInferenceBatcher,
    n_workers: usize,
    max_moves_per_game: usize,
    running: Arc<AtomicBool>,
    games_completed: Arc<AtomicUsize>,
    positions_generated: Arc<AtomicUsize>,
    handles: Arc<Mutex<Vec<JoinHandle<()>>>>,
}

#[pymethods]
impl RustSelfPlayRunner {
    #[new]
    #[pyo3(signature = (n_workers = 4, max_moves_per_game = 128, feature_len = 18 * 19 * 19, policy_len = 19 * 19 + 1))]
    pub fn new(
        n_workers: usize,
        max_moves_per_game: usize,
        feature_len: usize,
        policy_len: usize,
    ) -> Self {
        Self {
            batcher: RustInferenceBatcher::new(feature_len, policy_len),
            n_workers,
            max_moves_per_game,
            running: Arc::new(AtomicBool::new(false)),
            games_completed: Arc::new(AtomicUsize::new(0)),
            positions_generated: Arc::new(AtomicUsize::new(0)),
            handles: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn start(&self) {
        if self.running.swap(true, Ordering::SeqCst) {
            return;
        }

        let mut handles = self.handles.lock().expect("runner handles lock poisoned");
        for _ in 0..self.n_workers {
            let running = self.running.clone();
            let games_completed = self.games_completed.clone();
            let positions_generated = self.positions_generated.clone();
            let batcher = self.batcher.clone();
            let max_moves = self.max_moves_per_game;
            let feature_len = self.batcher.feature_len();

            let handle = thread::spawn(move || {
                while running.load(Ordering::SeqCst) {
                    for _ in 0..max_moves {
                        if !running.load(Ordering::SeqCst) {
                            break;
                        }

                        let features = vec![0.0_f32; feature_len];
                        if batcher.submit_request_and_wait_rust(features).is_err() {
                            if !running.load(Ordering::SeqCst) {
                                break;
                            }
                            continue;
                        }

                        positions_generated.fetch_add(1, Ordering::SeqCst);
                    }

                    games_completed.fetch_add(1, Ordering::SeqCst);
                }
            });
            handles.push(handle);
        }
    }

    pub fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
        self.batcher.close_rust();

        let mut handles = self.handles.lock().expect("runner handles lock poisoned");
        while let Some(handle) = handles.pop() {
            let _ = handle.join();
        }
    }

    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    #[getter]
    pub fn batcher(&self) -> RustInferenceBatcher {
        self.batcher.clone()
    }

    #[getter]
    pub fn games_completed(&self) -> usize {
        self.games_completed.load(Ordering::SeqCst)
    }

    #[getter]
    pub fn positions_generated(&self) -> usize {
        self.positions_generated.load(Ordering::SeqCst)
    }
}

impl Drop for RustSelfPlayRunner {
    fn drop(&mut self) {
        self.stop();
    }
}
