//! `WorkerStats` — per-worker accumulator bundle (cycle 3 Wave 10 Batch A).
//!
//! Extracted verbatim from the pre-split `worker_loop.rs` (§P52 capture-bundle
//! prototype build at L305-319). 13 `Arc<AtomicU*>` fields cloned once per
//! worker spawn (cheap `Arc::clone`-per-field via `#[derive(Clone)]`).
//! Destructured at `inner::run_worker_thread` entry into the local hot-loop
//! identifiers.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize};

#[derive(Clone)]
pub(super) struct WorkerStats {
    pub(super) games_completed: Arc<AtomicUsize>,
    pub(super) positions_generated: Arc<AtomicUsize>,
    pub(super) x_wins: Arc<AtomicU64>,
    pub(super) o_wins: Arc<AtomicU64>,
    pub(super) draws: Arc<AtomicU64>,
    pub(super) positions_dropped: Arc<AtomicU64>,
    pub(super) mcts_depth_accum: Arc<AtomicU64>,
    pub(super) mcts_conc_accum: Arc<AtomicU64>,
    pub(super) mcts_stat_count: Arc<AtomicU64>,
    pub(super) mcts_quiescence_fires: Arc<AtomicU64>,
    pub(super) cluster_value_std_accum: Arc<AtomicU64>,
    pub(super) cluster_policy_disagreement_accum: Arc<AtomicU64>,
    pub(super) cluster_variance_samples: Arc<AtomicU64>,
}
