//! HEXG sample path — rebuild-at-sample via the C1 native builder (design §4).
//!
//! Weighted-sample record indices (HEXB rejection sampler lifted over — weights
//! are game-length, representation-agnostic), then per sampled record: draw a
//! uniform D6 element, coord-rotate the stored stones AND the visit-map keys by
//! it (design §5 realization (ii)), rebuild via `hexo_graph::build_axis_graph`
//! (which stamps `builder_impl = 1` by construction — F7), and align the rotated
//! visit-keys to the built legal nodes by coord → the per-legal-node policy
//! target. Block-diagonal fuse via the SHARED `GraphWire::from_axis_graphs`.
//! ONE call emits graph + target together (F1 single-source): the target is read
//! off the SAME rebuilt `legal_node_gather` order the wire carries, so a
//! graph/target orientation desync is structurally unconstructable.

use std::collections::HashSet;

use fxhash::FxHashMap;
use hexo_graph::{build_axis_graph, BuildParams, StoneList};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::RngExt;

use super::super::sym_tables::{rotate_axial, N_SYMS};
use super::{GraphTargets, HexgBuffer};
use crate::inference_bridge::GraphWire;

/// B2 fix (WP-5a red-team, SILENT target under-weighting) — pure check, no
/// PyO3 involved (so it is directly unit-testable without a GIL/interpreter:
/// `cargo test` runs without `pyo3/auto-initialize`, and `PyErr`'s `Display`
/// impl requires an attached interpreter, so the labeled message is built
/// here as a plain `String` and only wrapped into `PyValueError` at the call
/// site). `push_graph_position` accepts visit coords with no legality check;
/// the sample-align loop only visits the rebuilt legal-node set, so mass on
/// an illegal/off-window cell is silently dropped — the argmax canary alone
/// cannot catch this (the reported argmax is always a legal cell). Compare
/// aligned mass against the mass stored at push time; LOUD-fail, naming the
/// record's `game_id`/`ply`, when they diverge beyond tolerance.
pub(crate) fn mass_drop_check(
    game_id: i64,
    ply_idx: u16,
    stored_mass: f32,
    aligned_mass: f32,
) -> Result<(), String> {
    const REL_TOL: f32 = 1e-4;
    const ABS_FLOOR: f32 = 1e-6;
    let dropped = stored_mass - aligned_mass;
    let tripped = if stored_mass.abs() > ABS_FLOOR {
        (dropped.abs() / stored_mass.abs()) > REL_TOL
    } else {
        aligned_mass.abs() > ABS_FLOOR
    };
    if tripped {
        Err(format!(
            "HEXG sample: visit mass dropped at sample-align (illegal/off-window \
             visit coord not in the rebuilt legal-node set) for game_id={game_id} \
             ply={ply_idx}: stored={stored_mass:.6} aligned={aligned_mass:.6} \
             dropped={dropped:.6} (tolerance rel={REL_TOL})"
        ))
    } else {
        Ok(())
    }
}

impl HexgBuffer {
    /// Sample a single index by weighted rejection (32-attempt cap, then
    /// unconditional accept). Identical to `ReplayBuffer::weighted_sample_one`.
    #[inline]
    fn weighted_sample_one(&mut self) -> usize {
        const MAX_REJECT: usize = 32;
        for _ in 0..MAX_REJECT {
            let idx = self.rng.random_range(0..self.size);
            let w = half::f16::from_bits(self.weights[idx]).to_f32();
            if w >= 1.0 || self.rng.random::<f32>() < w {
                return idx;
            }
        }
        self.rng.random_range(0..self.size)
    }

    /// Sample `batch_size` slot indices, deduping by `game_id` (the Multi-Window
    /// correlation guard; untagged −1 slots skip the guard). Mirrors
    /// `ReplayBuffer::sample_indices`.
    fn sample_indices(&mut self, batch_size: usize) -> Vec<usize> {
        const MAX_RETRIES: usize = 8;
        let mut indices: Vec<usize> = (0..batch_size).map(|_| self.weighted_sample_one()).collect();
        let mut seen: HashSet<i64> = HashSet::with_capacity(batch_size);
        for _ in 0..MAX_RETRIES {
            seen.clear();
            let mut all_unique = true;
            for idx in &mut indices {
                let gid = self.game_ids[*idx];
                if gid == -1 || seen.insert(gid) {
                    continue;
                }
                all_unique = false;
                let mut candidate = self.weighted_sample_one();
                for _ in 0..16 {
                    let cgid = self.game_ids[candidate];
                    if cgid == -1 || !seen.contains(&cgid) {
                        break;
                    }
                    candidate = self.weighted_sample_one();
                }
                *idx = candidate;
                let cgid = self.game_ids[candidate];
                if cgid != -1 {
                    seen.insert(cgid);
                }
            }
            if all_unique {
                break;
            }
        }
        indices
    }

    /// Rebuild + align + fuse `batch_size` sampled records. See module docs.
    pub(crate) fn sample_graph_batch_impl(
        &mut self,
        batch_size: usize,
        augment: bool,
    ) -> PyResult<(GraphWire, GraphTargets)> {
        if self.size == 0 {
            return Err(PyValueError::new_err(
                "Cannot sample from an empty HEXG buffer",
            ));
        }
        let indices = self.sample_indices(batch_size);

        let mut graphs = Vec::with_capacity(batch_size);
        let mut policy_target: Vec<f32> = Vec::new();
        let mut outcomes: Vec<f32> = Vec::with_capacity(batch_size);
        let mut value_valid: Vec<u8> = Vec::with_capacity(batch_size);
        let mut is_full_search: Vec<u8> = Vec::with_capacity(batch_size);
        let mut argmax_q: Vec<i32> = Vec::with_capacity(batch_size);
        let mut argmax_r: Vec<i32> = Vec::with_capacity(batch_size);
        let mut argmax_valid: Vec<u8> = Vec::with_capacity(batch_size);

        let params_base = BuildParams {
            win_length: self.win_length,
            radius: self.radius,
            current_player: 1, // overwritten per record
            moves_remaining: 2,
            trunk_size: self.trunk_size,
        };

        for &idx in &indices {
            // D6 element for this sample (uniform per-sample; matches run2's
            // dense `augment` semantics — one random sym per row, NOT 12-fold
            // enumeration). sym 0 = identity when augmentation is OFF.
            let sym = if augment { self.rng.random_range(0..N_SYMS) } else { 0 };

            let rec = self.record_at(idx);
            let game_id = self.game_ids[idx];
            let ply_idx = rec.ply_index;

            // Rotate stones by the element (axial lattice automorphism).
            let mut stones: Vec<(i32, i32, i8)> = Vec::with_capacity(rec.stones.len());
            for &(q, r, p) in &rec.stones {
                let (rq, rr) = rotate_axial(i32::from(q), i32::from(r), sym);
                stones.push((rq, rr, p));
            }
            let params = BuildParams {
                current_player: rec.current_player,
                moves_remaining: rec.moves_remaining,
                ..params_base
            };
            // Rebuild — the builder emits the correctly re-indexed graph (rotated
            // -coord sort → new edge_index), recomputed policy_dst_slot, and
            // stamps builder_impl=1. edge_index is NEVER cached across aug (F4).
            let g = build_axis_graph(&StoneList { stones }, &params);

            // Rotate the visit-map KEYS by the SAME element, so the policy target
            // follows each cell to its new location (rotating stones alone would
            // desync the label — F1/F4).
            // Visits are sparse (≤ MAX_VISITS = 128) — no pre-sizing needed.
            let mut vmap: FxHashMap<(i32, i32), f32> = FxHashMap::default();
            let stored_mass: f32 = rec.visits.iter().map(|&(_, _, prob)| prob).sum();
            for &(q, r, prob) in &rec.visits {
                let (rq, rr) = rotate_axial(i32::from(q), i32::from(r), sym);
                vmap.insert((rq, rr), prob);
            }

            // Align to the built legal nodes (gather order == the wire's segment
            // order): target[i] = rotated_visit_map[coord_of_legal_node_i] or 0.
            // No off-window drop — every legal node gets its coord's mass (the
            // records.rs:62 skip is NOT inherited, design §6.1).
            let mut best_prob = f32::NEG_INFINITY;
            let mut best_coord: Option<(i32, i32)> = None;
            let mut aligned_mass = 0.0f32;
            for &row in &g.legal_node_gather {
                let cq = g.node_coords[row as usize * 2];
                let cr = g.node_coords[row as usize * 2 + 1];
                let prob = vmap.get(&(cq, cr)).copied().unwrap_or(0.0);
                aligned_mass += prob;
                policy_target.push(prob);
                if prob > best_prob {
                    best_prob = prob;
                    best_coord = Some((cq, cr));
                }
            }

            // B2 fix (WP-5a red-team, SILENT target under-weighting): a visit
            // coord that is not a legal node of the rebuilt graph (occupied
            // stone, off-radius, ...) is dropped by the align loop above — the
            // argmax canary alone is blind to this (the reported argmax is
            // always a legal cell, so it always passes). ALWAYS-ON contract
            // check: compare aligned mass against the mass actually stored at
            // push time and LOUD-fail, naming the record, when they diverge
            // beyond tolerance. `rotate_axial` is exact integer-lattice math (no
            // float drift), so a legit producer's mass survives the align
            // bit-for-bit and never trips this.
            if let Err(msg) = mass_drop_check(game_id, ply_idx, stored_mass, aligned_mass) {
                return Err(PyValueError::new_err(msg));
            }

            match best_coord {
                Some((q, r)) if best_prob > 0.0 => {
                    argmax_q.push(q);
                    argmax_r.push(r);
                    argmax_valid.push(1);
                }
                // all-zero target (value-only / quick-search row): no argmax cell,
                // collate skips the AugRoundTrip canary for this graph.
                _ => {
                    argmax_q.push(0);
                    argmax_r.push(0);
                    argmax_valid.push(0);
                }
            }

            outcomes.push(rec.outcome);
            value_valid.push(u8::from(rec.value_valid));
            is_full_search.push(u8::from(rec.is_full_search));
            graphs.push(g);
        }

        let wire = GraphWire::from_axis_graphs(&graphs, self.contract_version);
        let targets = GraphTargets {
            policy_target,
            outcomes,
            value_valid,
            is_full_search,
            argmax_q,
            argmax_r,
            argmax_valid,
        };
        Ok((wire, targets))
    }
}
