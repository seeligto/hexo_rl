//! `WorkerAtomics` — per-worker control-flag bundle (cycle 3 Wave 10 Batch A).
//!
//! Extracted verbatim from the pre-split `worker_loop.rs` (§P52 capture-bundle
//! prototype build at L320-323). 2 live tunables: `running` (kill switch),
//! `radius_override` (§174 curriculum). Cloned once per worker spawn via
//! `#[derive(Clone)]`; destructured at `inner::run_worker_thread` entry.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicI32};

#[derive(Clone)]
pub(super) struct WorkerAtomics {
    pub(super) running: Arc<AtomicBool>,
    pub(super) radius_override: Arc<AtomicI32>,
}
