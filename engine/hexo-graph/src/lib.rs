//! `hexo-graph` — axis-graph builder skeleton for the HeXO GNN encoding.
//!
//! ONE Rust source compiled to TWO targets (standing ruling, R4/WP-D
//! mission): native (`cargo check -p hexo-graph`) and wasm32
//! (`cargo check -p hexo-graph --no-default-features --features wasm
//! --target wasm32-unknown-unknown`). This file proves the crate/type
//! boundary compiles clean on both BEFORE the real builder (C1) has logic —
//! catching std-only deps, accidental threading, or PyO3 leakage into the
//! wasm feature set early, while the fix is a one-line revert instead of a
//! redesign.
//!
//! Payload field names/shapes here track `docs/designs/gnn_integration_scope.md`
//! (C1/C3/C4) and are provisional pending the WP-B ragged-contract doc
//! (`docs/designs/gnn_ragged_contract_v1.md`, in flight concurrently — this
//! crate does not read or depend on that doc, ratify against it later).
//! Every function body is `todo!()`; only the TYPES are load-bearing here.
#![cfg_attr(not(feature = "native"), allow(dead_code))]

/// Per-node feature width. C4 (`gnn_integration_scope.md`) names
/// `node_feat_dim=11` for the LEGACY axis-graph schema (`x (N,11)` in C3);
/// carried here as a placeholder constant, not re-derived.
pub const NODE_FEAT_DIM: usize = 11;

/// Per-edge feature width: axis one-hot(3) + signed_dist + src_player, per
/// C1's GINE `edge_attr (E,5)` (`gnn_integration_scope.md` line ~45).
pub const EDGE_FEAT_DIM: usize = 5;

/// Flat, row-major node feature matrix, shape `(N, NODE_FEAT_DIM)` — one
/// `AxisGraph`'s worth, or the disjoint-union of a batch's worth in
/// `AxisGraphBatch`. Flat `Vec<f32>` (not `Vec<[f32; NODE_FEAT_DIM]>`) so it
/// crosses the PyO3 boundary / an eventual wasm-bindgen boundary as one
/// contiguous buffer, matching how `inference_bridge.rs` already ships
/// dense CNN tensors.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct NodeFeat(pub Vec<f32>);

/// COO edge index as two parallel arrays (PyTorch-Geometric convention:
/// `edge_index[0]` = src, `edge_index[1]` = dst) rather than one
/// `Vec<(u32, u32)>` — cheaper to hand to a GNN forward pass as two flat
/// buffers. Layout is provisional; WP-B contract doc is authoritative once
/// it lands.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct EdgeIndex {
    pub src: Vec<u32>,
    pub dst: Vec<u32>,
}

/// Flat, row-major edge feature matrix, shape `(E, EDGE_FEAT_DIM)`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct EdgeAttr(pub Vec<f32>);

/// CSR-style prefix-sum node-range boundaries for a batched disjoint-union
/// graph: graph `i`'s nodes are `node_feat[graph_offsets[i]..graph_offsets[i+1]]`.
/// Length = `n_graphs + 1`. Rust-side equivalent of `_collate_gnn` in
/// `train_bc.py` (C3) — the thing that replaces `InferenceBatcher`'s
/// `[batch, feature_len]` fuse for a ragged payload.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GraphOffsets(pub Vec<u32>);

/// Per-legal-node index into the fixed 362-slot dense action space
/// (`policy_logit_count` stays 362 per C4 — the board's action space is
/// unchanged by a graph encoding). Lets a GNN's variable-N policy head be
/// scattered back into the dense buffer the rest of the stack (deploy bot,
/// MCTS priors) already expects — `strix_v1_bot.py::get_move` does this
/// re-projection today in Python; this is its future Rust-side home.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PolicyScatterIndex(pub Vec<u32>);

/// One built axis-graph: everything the C3 ragged inference contract will
/// ship per-leaf before batching. Field set mirrors
/// `docs/designs/gnn_integration_scope.md` line ~154: `x`, `edge_index`,
/// `edge_attr`, `legal_mask`, `stone_mask`, plus the policy scatter index
/// this crate owns for the re-projection step.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AxisGraph {
    pub node_feat: NodeFeat,
    pub edge_index: EdgeIndex,
    pub edge_attr: EdgeAttr,
    pub legal_mask: Vec<bool>,
    pub stone_mask: Vec<bool>,
    pub policy_scatter_index: PolicyScatterIndex,
}

/// Disjoint-union batch of `n_graphs = graph_offsets.len() - 1` `AxisGraph`s
/// — the wire shape a ragged inference batch actually ships as (C3: "the
/// Rust equivalent of `_collate_gnn` in `train_bc.py`"). `policy_scatter_index`
/// here is the concatenation of each graph's scatter index, offset into the
/// batch's shared 362-wide-per-graph output block by the caller.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AxisGraphBatch {
    pub node_feat: NodeFeat,
    pub edge_index: EdgeIndex,
    pub edge_attr: EdgeAttr,
    pub graph_offsets: GraphOffsets,
    pub policy_scatter_index: PolicyScatterIndex,
}

/// Minimal position input for the builder — a stone list, nothing else.
/// Mirrors the inputs `strix_v1_graph.py::node_threat_features` /
/// `legal_moves_from_stones` need (that file is the pure-Python reference
/// port of the external `hexo-rs/hexo-mcts/src/axis_graph.rs` builder this
/// crate will eventually replace/absorb — NOT vendored into this repo).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct StoneList {
    /// `(q, r, player)` triples; player is `+1`/`-1` (the Python port's
    /// convention, `strix_v1_graph.py` module docstring).
    pub stones: Vec<(i32, i32, i8)>,
}

/// Board/window parameters the builder needs beyond the stone list.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BuildParams {
    pub win_length: u8,
    pub radius: u16,
    pub to_move: i8,
}

/// Placeholder for the real axis-graph builder (C1, `gnn_integration_scope.md`).
/// Body is `todo!()` on purpose — WP-D's mission is proving this SIGNATURE
/// and its return TYPE compile on both native and wasm32 targets before any
/// real logic exists, so a dependency-poison problem (std-only crate, a
/// rayon pull-in, a threading assumption) surfaces now, not after the
/// builder is written. Do not call; will panic.
pub fn build_axis_graph(_stones: &StoneList, _params: &BuildParams) -> AxisGraph {
    todo!("C1 axis-graph builder — see docs/designs/gnn_integration_scope.md")
}

// ── native-only surface ─────────────────────────────────────────────────
//
// Nothing in the skeleton actually spawns a thread yet — this function
// exists purely to establish the cfg-gating PATTERN the real C1 builder
// must follow once it adds rayon/std::thread for per-leaf parallel graph
// construction. `not(target_arch = "wasm32")` is belt-and-suspenders on top
// of the `native` feature gate: even if someone builds with
// `--features native --target wasm32-unknown-unknown` (a misconfiguration,
// not the documented wasm build command), this still won't be compiled in.
#[cfg(all(feature = "native", not(target_arch = "wasm32")))]
pub fn parallelism_hint() -> usize {
    std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .unwrap_or(1)
}

// ── python-only surface ─────────────────────────────────────────────────
//
// Placeholder proving the `python` feature wires pyo3 without pulling it
// into `native` or `wasm` builds. Not registered as a pymodule anywhere —
// `engine` does not depend on this crate yet (mission constraint: zero
// build-risk to `make build`). Real glue lands in `engine/src/pyo3/` per
// C1's file list, consuming this crate's types directly rather than through
// a pymodule defined here.
#[cfg(feature = "python")]
mod python_glue {
    use pyo3::prelude::*;

    /// Existence-only marker function so `cargo check -p hexo-graph
    /// --features python` has something to type-check against pyo3's
    /// macros. Unused outside that check — allowed dead code.
    #[allow(dead_code)]
    #[pyfunction]
    fn _hexo_graph_skeleton_marker() -> PyResult<bool> {
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The one thing this skeleton commits to: default-constructing every
    /// payload type must work with no real data, on whichever target this
    /// test runs on (native only — wasm32-unknown-unknown has no test
    /// harness runner here, `cargo check` is the wasm gate per WP-D task 3).
    #[test]
    fn payload_types_default_construct() {
        let g = AxisGraph::default();
        assert_eq!(g.node_feat.0.len(), 0);
        let b = AxisGraphBatch::default();
        assert_eq!(b.graph_offsets.0.len(), 0);
    }

    #[test]
    #[should_panic(expected = "C1 axis-graph builder")]
    fn build_axis_graph_is_unimplemented_by_design() {
        let stones = StoneList::default();
        let params = BuildParams { win_length: 6, radius: 4, to_move: 1 };
        let _ = build_axis_graph(&stones, &params);
    }
}
