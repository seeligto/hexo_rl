//! `WorkerChannels` — per-worker channel/queue bundle (cycle 3 Wave 10 Batch A).
//!
//! Extracted verbatim from the pre-split `worker_loop.rs` (§P52 capture-bundle
//! prototype build at L324-328). Shared queues + InferenceBatcher handle.
//! Cloned once per worker spawn via `#[derive(Clone)]`; destructured at
//! `inner::run_worker_thread` entry.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use crate::inference_bridge::InferenceBatcher;
use crate::replay_buffer::hexg::GraphRecord;

use super::super::{GameResultRow, WorkerResultRow};

#[derive(Clone)]
pub(super) struct WorkerChannels {
    pub(super) batcher: InferenceBatcher,
    pub(super) results_queue: Arc<Mutex<VecDeque<WorkerResultRow>>>,
    pub(super) recent_game_results: Arc<Mutex<VecDeque<GameResultRow>>>,
    /// GNN-integration WP-5b commit A (R6): parallel graph-position results
    /// queue. Constructed unconditionally (idle `Mutex<VecDeque>`, not on the
    /// dense hot path); only ever locked on the `is_graph` finalize branch.
    pub(super) graph_results_queue: Arc<Mutex<VecDeque<GraphRecord>>>,
}
