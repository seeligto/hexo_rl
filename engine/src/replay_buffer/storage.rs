//! Storage layout, ring capacity, and weight-schedule configuration for `ReplayBuffer`.
//!
//! The ring-buffer backing Vecs are allocated here on `new_impl`, grown by
//! `resize_impl`, and read by dashboard stats via `get_buffer_stats_impl`.
//! Push/sample/persist logic lives in sibling modules.

use half::f16;
use std::sync::atomic::Ordering;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::sym_tables::{WeightBracket, WeightSchedule};
use super::ReplayBuffer;

impl ReplayBuffer {
    /// Return `(size, capacity, weight_histogram)` for dashboard display.
    ///
    /// `weight_histogram` is a length-3 list: counts of positions in each
    /// weight tier (low / medium / full).  Reads lock-free atomic counters — O(1),
    /// never blocks push() or sample_batch().
    pub(crate) fn get_buffer_stats_impl(&self) -> (usize, usize, Vec<u64>) {
        let histogram = vec![
            self.weight_buckets[0].load(Ordering::Relaxed),
            self.weight_buckets[1].load(Ordering::Relaxed),
            self.weight_buckets[2].load(Ordering::Relaxed),
        ];
        (self.size, self.capacity, histogram)
    }

    /// Return a fresh position ID and advance the internal counter.
    ///
    /// Call once per board position (not once per cluster).  Pass the returned
    /// ID to every cluster's `push()` so the sampler can enforce the
    /// Multi-Window correlation guard.
    pub(crate) fn next_game_id_impl(&mut self) -> i64 {
        let id = self.next_game_id;
        self.next_game_id += 1;
        id
    }

    /// Grow the buffer to `new_capacity` positions, preserving all existing data.
    ///
    /// The ring buffer is linearised in-place (oldest entry → slot 0) before
    /// extending.  Raises `ValueError` if `new_capacity <= self.capacity`.
    pub(crate) fn resize_impl(&mut self, new_capacity: usize) -> PyResult<()> {
        if new_capacity <= self.capacity {
            return Err(PyValueError::new_err(format!(
                "resize: new_capacity ({}) must be greater than current capacity ({})",
                new_capacity, self.capacity,
            )));
        }

        let state_stride  = self.encoding.state_stride();
        let chain_stride  = self.encoding.chain_stride();
        let policy_stride = self.encoding.policy_stride();
        let aux_stride    = self.encoding.aux_stride();

        // Linearise the ring buffer when it has wrapped around.
        if self.size == self.capacity && self.head != 0 {
            self.states[..self.capacity * state_stride]
                .rotate_left(self.head * state_stride);
            self.chain_planes[..self.capacity * chain_stride]
                .rotate_left(self.head * chain_stride);
            self.policies[..self.capacity * policy_stride]
                .rotate_left(self.head * policy_stride);
            self.outcomes[..self.capacity]
                .rotate_left(self.head);
            self.game_ids[..self.capacity]
                .rotate_left(self.head);
            self.weights[..self.capacity]
                .rotate_left(self.head);
            self.ownership[..self.capacity * aux_stride]
                .rotate_left(self.head * aux_stride);
            self.winning_line[..self.capacity * aux_stride]
                .rotate_left(self.head * aux_stride);
            self.is_full_search[..self.capacity]
                .rotate_left(self.head);
            self.position_indices[..self.capacity]
                .rotate_left(self.head);
        }

        // Extend storage to new capacity.
        let default_w = f16::from_f32(1.0).to_bits();
        self.states.resize(new_capacity * state_stride, 0u16);
        self.chain_planes.resize(new_capacity * chain_stride, 0u16);
        self.policies.resize(new_capacity * policy_stride, 0.0f32);
        self.outcomes.resize(new_capacity, 0.0f32);
        self.game_ids.resize(new_capacity, -1i64);
        self.weights.resize(new_capacity, default_w);
        self.ownership.resize(new_capacity * aux_stride, 1u8);     // 1 = empty
        self.winning_line.resize(new_capacity * aux_stride, 0u8);
        self.is_full_search.resize(new_capacity, 1u8);  // 1 = full-search default
        self.position_indices.resize(new_capacity, 0u16);

        self.head = self.size;
        self.capacity = new_capacity;
        Ok(())
    }

    /// Phase B' Class-3 (buffer composition) — count valid outcomes in the
    /// half-open interval `[lo, hi)`. Used to track `draw_target_fraction`
    /// (positions whose value target landed in [-0.6, -0.4]).
    ///
    /// Reads only the live prefix (`self.size` slots), so capacity-padding
    /// zeros never falsely count as draws on a partly-filled buffer.
    pub(crate) fn outcome_in_range_count_impl(&self, lo: f32, hi: f32) -> usize {
        self.outcomes[..self.size]
            .iter()
            .filter(|&&v| v >= lo && v < hi)
            .count()
    }

    /// Set the game-length weight schedule from Python config.
    ///
    /// Args:
    ///     thresholds: list of exclusive upper bounds (must be sorted ascending)
    ///     weights:    list of f32 weights, same length as thresholds
    ///     default_weight: weight for games >= all thresholds (typically 1.0)
    ///
    /// Example (from training.yaml):
    ///     buf.set_weight_schedule([10, 25], [0.15, 0.50], 1.0)
    ///     # game < 10 moves → 0.15, 10-24 → 0.50, 25+ → 1.0
    pub(crate) fn set_weight_schedule_impl(
        &mut self,
        thresholds:     Vec<u16>,
        weights:        Vec<f32>,
        default_weight: f32,
    ) -> PyResult<()> {
        if thresholds.len() != weights.len() {
            return Err(PyValueError::new_err(
                "thresholds and weights must have the same length"
            ));
        }
        let brackets: Vec<WeightBracket> = thresholds.iter().zip(weights.iter())
            .map(|(&t, &w)| WeightBracket {
                max_moves: t,
                weight: f16::from_f32(w).to_bits(),
            })
            .collect();
        self.weight_schedule = WeightSchedule {
            brackets,
            default_weight: f16::from_f32(default_weight).to_bits(),
        };
        Ok(())
    }
}
