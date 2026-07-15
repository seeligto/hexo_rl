//! `verify_edge_geometry` — compiled re-derivation of the ragged graph-wire
//! contract's `EdgeAttrGeometryMismatch` semantic check (contract §2.5,
//! check 14 of 18). S4 PERF item #1 (`reports/probes/gnn_perf/PREREG.md`
//! HOTSPOT #1): replaces the O(E) numpy re-derivation in
//! `hexo_rl/selfplay/graph_collate.py::_check_semantic` (lines ~493-530)
//! with a single Rust pass over the SAME post-marshal fused-wire arrays,
//! read as zero-copy readonly views (`numpy::PyReadonlyArray1`). Mirrors the
//! per-edge geometry re-derivation the Rust producer already runs in
//! `hexo_graph::verify_contract` (`engine/hexo-graph/src/lib.rs` ~779-841),
//! adapted to the FUSED multi-graph wire: global edge ids, per-graph
//! `current_player`/dummy-node identity looked up via a `node_offsets`-
//! derived ownership map instead of a single graph's fields.
//!
//! Every sub-assertion of the Python check moves verbatim: dummy-edge
//! all-zero attrs, clean axis one-hot, integral + bounded `signed_dist`,
//! `src_player` == stone own/opp identity × `current_player`. None sampled,
//! skipped, or debug-gated. On mismatch this raises a generic `ValueError`;
//! the Python call site (`graph_collate.py::_check_semantic`) catches it and
//! re-raises the NAMED `EdgeAttrGeometryMismatch` so every existing
//! die-loud catcher/test is unaffected (same exception type, same
//! post-marshal catching boundary — the ADV-8 test injects its corruption
//! into the Python payload *before* this call, so it still fires).
//!
//! Deliberately NOT in the `hexo-graph` crate (must stay wasm32-clean and
//! Python-optional) — this is `engine`-crate PyO3 glue only, matching the
//! `apply_symmetries_batch` pattern in `engine/src/pyo3/utils.rs`.
//!
//! Never panics on malformed input (defensive bounds/shape checks return an
//! `Err` instead of indexing out of range): the workspace release profile
//! sets `panic = "abort"`, so an unguarded panic here would abort the whole
//! self-play/training process rather than raise a catchable Python
//! exception — the opposite of "die loud but recoverable."
//!
//! Logic lives in the plain-slice `verify_edge_geometry_impl` (no PyO3
//! types) so `cargo test` can drive it directly without an attached Python
//! interpreter — same split as `HexgBuffer::push_record_impl` /
//! `sample_graph_batch_impl` (`engine/src/replay_buffer/hexg/`). The
//! `#[pyfunction]` is a thin shim: extract zero-copy slices, delegate, map
//! `Err(String)` to `PyValueError`.

use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use hexo_graph::WIN_AXES;

/// Recompute + verify every real (non-dummy) edge's `edge_attr` row against
/// the geometry implied by its endpoints' `node_coords` and the owning
/// graph's `current_player`. Pure Rust over plain slices — no PyO3 types.
///
/// Arrays are the exact post-marshal flat layout `_check_semantic` already
/// holds (contract §2.1): `node_feat` row-major `(N, node_feat_dim)` (col 0
/// = own, col 1 = opp), `node_coords` flat `(N, 2)` axial `(q, r)`,
/// `edge_index` flat `(2, E)`, `edge_attr` row-major `(E, edge_feat_dim)`
/// with columns `[onehot0, onehot1, onehot2, signed_dist, src_player]`,
/// `node_offsets` CSR `(B+1,)` (last row of each graph's range is its dummy
/// node), `current_player` per-graph `(B,)` in `{+1, -1}`.
///
/// # Errors
/// Returns `Err(message)` on the first geometry violation found, or on a
/// malformed-shape input. The structural layer (`_check_structural`) always
/// runs before this and already guarantees consistent shapes/dtypes, but
/// this function never trusts that blindly — see the module doc's panic
/// note: every index is range-checked before use.
// `float_cmp` allowed: the compared floats are EXACT constants the builder
// itself wrote (one-hot 0.0/1.0, integral signed_dist, ±1.0 src_player) —
// approximate comparison would WEAKEN the check. Same justification as
// `hexo_graph::verify_contract`'s `#[allow(clippy::float_cmp)]` (lib.rs),
// which this function mirrors for the fused wire.
#[allow(
    clippy::too_many_arguments,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::float_cmp
)]
fn verify_edge_geometry_impl(
    node_feat: &[f32],
    node_coords: &[i32],
    edge_index: &[i64],
    edge_attr: &[f32],
    node_offsets: &[i64],
    current_player: &[i8],
    node_feat_dim: usize,
    edge_feat_dim: usize,
    win_length: i64,
) -> Result<(), String> {
    // --- shape guards (defensive; structural layer already enforces these
    // upstream, but this fn must never index out of range on a corrupt
    // input — panic = "abort" in the release profile). ---
    if node_feat_dim == 0 || edge_feat_dim < 5 {
        return Err(format!(
            "verify_edge_geometry: degenerate dims node_feat_dim={node_feat_dim} edge_feat_dim={edge_feat_dim}"
        ));
    }
    if !node_feat.len().is_multiple_of(node_feat_dim) {
        return Err(format!(
            "verify_edge_geometry: len(node_feat)={} not divisible by node_feat_dim={node_feat_dim}",
            node_feat.len()
        ));
    }
    let n = node_feat.len() / node_feat_dim;
    if node_coords.len() != 2 * n {
        return Err(format!(
            "verify_edge_geometry: len(node_coords)={} != 2N={}",
            node_coords.len(),
            2 * n
        ));
    }
    if !edge_attr.len().is_multiple_of(edge_feat_dim) {
        return Err(format!(
            "verify_edge_geometry: len(edge_attr)={} not divisible by edge_feat_dim={edge_feat_dim}",
            edge_attr.len()
        ));
    }
    let e = edge_attr.len() / edge_feat_dim;
    if edge_index.len() != 2 * e {
        return Err(format!(
            "verify_edge_geometry: len(edge_index)={} != 2E={}",
            edge_index.len(),
            2 * e
        ));
    }
    if e == 0 {
        return Ok(()); // mirrors the Python `if E > 0:` guard — nothing to check
    }
    if node_offsets.is_empty() {
        return Err("verify_edge_geometry: node_offsets is empty".to_string());
    }
    let b = node_offsets.len() - 1;
    if current_player.len() != b {
        return Err(format!(
            "verify_edge_geometry: len(current_player)={} != B={b}",
            current_player.len()
        ));
    }

    // node_is_dummy[i] / node_graph[i]: single O(N) pass over node_offsets,
    // shared prep for every edge (mirrors the Python function's shared
    // `node_is_dummy` / `node_graph` setup).
    let mut node_is_dummy = vec![false; n];
    let mut node_graph = vec![0u32; n];
    for g in 0..b {
        let start = node_offsets[g];
        let end = node_offsets[g + 1];
        if start < 0 || end < start || (end as usize) > n {
            return Err(format!(
                "verify_edge_geometry: node_offsets[{g}..{}] = [{start},{end}] invalid for N={n}",
                g + 1
            ));
        }
        node_graph[(start as usize)..(end as usize)].fill(g as u32);
        if end > start {
            let dummy_idx = (end - 1) as usize;
            node_is_dummy[dummy_idx] = true;
        }
    }

    let win_max = win_length - 1;

    for edge in 0..e {
        let s = edge_index[edge];
        let d = edge_index[e + edge];
        if s < 0 || d < 0 || (s as usize) >= n || (d as usize) >= n {
            return Err(format!(
                "verify_edge_geometry: edge {edge} endpoints ({s},{d}) out of [0,{n})"
            ));
        }
        let (s, d) = (s as usize, d as usize);
        let a = &edge_attr[edge * edge_feat_dim..edge * edge_feat_dim + edge_feat_dim];

        if node_is_dummy[s] || node_is_dummy[d] {
            // dummy edges must be all-zero.
            if a.iter().any(|&x| x.abs() > 1e-6) {
                return Err(format!(
                    "a dummy edge has non-zero attrs (edge {edge}): {a:?}"
                ));
            }
            continue;
        }

        // exactly one of the 3 axis one-hots is 1.0, rest 0.0 (first-max
        // tie-break, matching np.argmax).
        let onehot = [a[0], a[1], a[2]];
        let mut axis = 0usize;
        for k in 1..3 {
            if onehot[k] > onehot[axis] {
                axis = k;
            }
        }
        let onehot_ok = (0..3).all(|k| onehot[k] == if k == axis { 1.0 } else { 0.0 });
        if !onehot_ok {
            return Err(format!(
                "edge axis one-hot is not a clean one-hot (edge {edge}): {onehot:?}"
            ));
        }

        // signed_dist must be integral (compare the original f32 against its
        // round-tripped value — any fractional part fails this regardless of
        // rounding convention, since a genuinely fractional value can never
        // equal an integer once rounded).
        let dist = a[3]; // plane-literal-ok: edge_attr col 3 = signed_dist (contract §2.1, not a state plane)
        let di = dist.round() as i64;
        if dist != di as f32 {
            return Err(format!("signed_dist is non-integral (edge {edge}): {dist}"));
        }

        let (aq, ar) = WIN_AXES[axis];
        let sq = i64::from(node_coords[s * 2]);
        let sr = i64::from(node_coords[s * 2 + 1]);
        let dq = i64::from(node_coords[d * 2]);
        let dr = i64::from(node_coords[d * 2 + 1]);
        let delta_q = dq - sq;
        let delta_r = dr - sr;
        let expect_q = di * i64::from(aq);
        let expect_r = di * i64::from(ar);
        if delta_q != expect_q || delta_r != expect_r || di == 0 || di.abs() > win_max {
            return Err(format!(
                "edge delta != signed_dist * axis_vec (rows misaligned/scrambled) (edge {edge}): \
                 delta=({delta_q},{delta_r}) expected=({expect_q},{expect_r}) di={di} win_max={win_max}"
            ));
        }

        // src_player: relative own/opp cols × current_player[graph], 0 for empty.
        let g_of = node_graph[s] as usize;
        let own = node_feat[s * node_feat_dim];
        let opp = node_feat[s * node_feat_dim + 1];
        let cp = f32::from(current_player[g_of]);
        let expect_sp = (own - opp) * cp;
        if (a[4] - expect_sp).abs() > 1e-6 {
            // plane-literal-ok: edge_attr col 4 = src_player (contract §2.1)
            return Err(format!(
                "edge src_player != node stone identity (edge {edge}): got {} expected {expect_sp}",
                a[4]
            ));
        }
    }

    Ok(())
}

/// PyO3 shim: extract zero-copy readonly slices, delegate to
/// `verify_edge_geometry_impl`, map `Err(String)` to `PyValueError`. See the
/// module doc for the Python-side re-raise into `EdgeAttrGeometryMismatch`.
///
/// # Errors
/// `PyValueError` on any geometry violation or malformed-shape input; also
/// propagates `AsSliceError` (via `?`) if a caller passes a non-contiguous
/// numpy view (never happens on the real wire — the arrays are always
/// contiguous flat vectors — but not assumed).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn verify_edge_geometry(
    node_feat: PyReadonlyArray1<'_, f32>,
    node_coords: PyReadonlyArray1<'_, i32>,
    edge_index: PyReadonlyArray1<'_, i64>,
    edge_attr: PyReadonlyArray1<'_, f32>,
    node_offsets: PyReadonlyArray1<'_, i64>,
    current_player: PyReadonlyArray1<'_, i8>,
    node_feat_dim: usize,
    edge_feat_dim: usize,
    win_length: i64,
) -> PyResult<()> {
    verify_edge_geometry_impl(
        node_feat.as_slice()?,
        node_coords.as_slice()?,
        edge_index.as_slice()?,
        edge_attr.as_slice()?,
        node_offsets.as_slice()?,
        current_player.as_slice()?,
        node_feat_dim,
        edge_feat_dim,
        win_length,
    )
    .map_err(PyValueError::new_err)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(verify_edge_geometry, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::verify_edge_geometry_impl;

    const NODE_FEAT_DIM: usize = 11;
    const EDGE_FEAT_DIM: usize = 5;
    const WIN_LENGTH: i64 = 6;

    /// One graph: 2 stones (own=+1 at (0,0), opp at (1,0)) + 1 legal node
    /// (2,0) + 1 dummy node (row 3). current_player = +1. One clean axis-0
    /// edge from stone 0 (0,0) to legal node 2 (2,0): signed_dist=2,
    /// src_player = (own-opp)*cp = (1-0)*1 = 1.
    fn clean_fixture() -> (Vec<f32>, Vec<i32>, Vec<i64>, Vec<f32>, Vec<i64>, Vec<i8>) {
        let mut node_feat = vec![0.0f32; 4 * NODE_FEAT_DIM];
        node_feat[0] = 1.0; // stone 0: own=1, opp=0
        node_feat[NODE_FEAT_DIM + 1] = 1.0; // stone 1: own=0, opp=1
        let node_coords: Vec<i32> = vec![0, 0, 1, 0, 2, 0, 0, 0]; // stone0,stone1,legal,dummy
        let edge_index: Vec<i64> = vec![0, 2]; // src=[0], dst=[2] (flat (2,E))
        let mut edge_attr = vec![0.0f32; EDGE_FEAT_DIM];
        edge_attr[0] = 1.0; // axis 0 one-hot
        edge_attr[3] = 2.0; // signed_dist
        edge_attr[4] = 1.0; // src_player = (1-0)*cp(+1) = 1
        let node_offsets: Vec<i64> = vec![0, 4]; // B=1, N=4
        let current_player: Vec<i8> = vec![1];
        (
            node_feat,
            node_coords,
            edge_index,
            edge_attr,
            node_offsets,
            current_player,
        )
    }

    #[test]
    fn clean_edge_passes() {
        let (nf, nc, ei, ea, no, cp) = clean_fixture();
        assert!(verify_edge_geometry_impl(
            &nf, &nc, &ei, &ea, &no, &cp, NODE_FEAT_DIM, EDGE_FEAT_DIM, WIN_LENGTH
        )
        .is_ok());
    }

    #[test]
    fn empty_edge_set_is_ok() {
        let (nf, nc, _ei, _ea, no, cp) = clean_fixture();
        assert!(verify_edge_geometry_impl(
            &nf, &nc, &[], &[], &no, &cp, NODE_FEAT_DIM, EDGE_FEAT_DIM, WIN_LENGTH
        )
        .is_ok());
    }

    /// ADV-8-equivalent: flip the signed_dist column (`edge_attr[3]`) sign —
    /// the same corruption `tests/selfplay/test_graph_collate.py::
    /// test_adv_8_edge_attr_permuted` injects into the post-marshal payload
    /// (`p.edge_attr[3] = -p.edge_attr[3]`). Must raise: the flipped delta no
    /// longer matches `coords[d] - coords[s]`.
    #[test]
    fn flipped_signed_dist_raises() {
        let (nf, nc, ei, mut ea, no, cp) = clean_fixture();
        ea[3] = -ea[3];
        let err = verify_edge_geometry_impl(
            &nf, &nc, &ei, &ea, &no, &cp, NODE_FEAT_DIM, EDGE_FEAT_DIM, WIN_LENGTH,
        )
        .unwrap_err();
        assert!(
            err.contains("edge delta"),
            "expected a geometry-delta mismatch message, got: {err}"
        );
    }

    #[test]
    fn dirty_onehot_raises() {
        let (nf, nc, ei, mut ea, no, cp) = clean_fixture();
        ea[1] = 1.0; // now two axes set — not a clean one-hot
        let err = verify_edge_geometry_impl(
            &nf, &nc, &ei, &ea, &no, &cp, NODE_FEAT_DIM, EDGE_FEAT_DIM, WIN_LENGTH,
        )
        .unwrap_err();
        assert!(err.contains("one-hot"), "got: {err}");
    }

    #[test]
    fn non_integral_signed_dist_raises() {
        let (nf, nc, ei, mut ea, no, cp) = clean_fixture();
        ea[3] = 2.5;
        let err = verify_edge_geometry_impl(
            &nf, &nc, &ei, &ea, &no, &cp, NODE_FEAT_DIM, EDGE_FEAT_DIM, WIN_LENGTH,
        )
        .unwrap_err();
        assert!(err.contains("non-integral"), "got: {err}");
    }

    #[test]
    fn wrong_src_player_raises() {
        let (nf, nc, ei, mut ea, no, cp) = clean_fixture();
        ea[4] = -1.0; // should be +1
        let err = verify_edge_geometry_impl(
            &nf, &nc, &ei, &ea, &no, &cp, NODE_FEAT_DIM, EDGE_FEAT_DIM, WIN_LENGTH,
        )
        .unwrap_err();
        assert!(err.contains("src_player"), "got: {err}");
    }

    #[test]
    fn nonzero_dummy_edge_raises() {
        let (nf, nc, _ei, _ea, no, cp) = clean_fixture();
        let edge_index: Vec<i64> = vec![0, 3]; // src=stone0, dst=dummy(row 3)
        let mut edge_attr = vec![0.0f32; EDGE_FEAT_DIM];
        edge_attr[3] = 1.0; // non-zero on a dummy edge
        let err = verify_edge_geometry_impl(
            &nf,
            &nc,
            &edge_index,
            &edge_attr,
            &no,
            &cp,
            NODE_FEAT_DIM,
            EDGE_FEAT_DIM,
            WIN_LENGTH,
        )
        .unwrap_err();
        assert!(err.contains("dummy edge"), "got: {err}");
    }

    #[test]
    fn clean_dummy_edge_is_ok() {
        let (nf, nc, _ei, _ea, no, cp) = clean_fixture();
        let edge_index: Vec<i64> = vec![0, 3]; // src=stone0, dst=dummy(row 3)
        let edge_attr = vec![0.0f32; EDGE_FEAT_DIM]; // all-zero
        assert!(verify_edge_geometry_impl(
            &nf,
            &nc,
            &edge_index,
            &edge_attr,
            &no,
            &cp,
            NODE_FEAT_DIM,
            EDGE_FEAT_DIM,
            WIN_LENGTH
        )
        .is_ok());
    }

    #[test]
    fn out_of_range_edge_endpoint_raises_not_panics() {
        let (nf, nc, _ei, ea, no, cp) = clean_fixture();
        let edge_index: Vec<i64> = vec![0, 99]; // dst way out of [0, N)
        let err = verify_edge_geometry_impl(
            &nf, &nc, &edge_index, &ea, &no, &cp, NODE_FEAT_DIM, EDGE_FEAT_DIM, WIN_LENGTH,
        )
        .unwrap_err();
        assert!(err.contains("out of"), "got: {err}");
    }

    #[test]
    fn negative_edge_endpoint_raises_not_panics() {
        let (nf, nc, _ei, ea, no, cp) = clean_fixture();
        let edge_index: Vec<i64> = vec![-1, 2];
        let err = verify_edge_geometry_impl(
            &nf, &nc, &edge_index, &ea, &no, &cp, NODE_FEAT_DIM, EDGE_FEAT_DIM, WIN_LENGTH,
        )
        .unwrap_err();
        assert!(err.contains("out of"), "got: {err}");
    }

    #[test]
    fn malformed_node_offsets_raises_not_panics() {
        let (nf, nc, ei, ea, _no, cp) = clean_fixture();
        let node_offsets: Vec<i64> = vec![0, 999]; // end > N
        let err = verify_edge_geometry_impl(
            &nf, &nc, &ei, &ea, &node_offsets, &cp, NODE_FEAT_DIM, EDGE_FEAT_DIM, WIN_LENGTH,
        )
        .unwrap_err();
        assert!(err.contains("node_offsets"), "got: {err}");
    }

    #[test]
    fn current_player_length_mismatch_raises() {
        let (nf, nc, ei, ea, no, _cp) = clean_fixture();
        let current_player: Vec<i8> = vec![1, -1]; // B=1 expected, got 2
        let err = verify_edge_geometry_impl(
            &nf, &nc, &ei, &ea, &no, &current_player, NODE_FEAT_DIM, EDGE_FEAT_DIM, WIN_LENGTH,
        )
        .unwrap_err();
        assert!(err.contains("current_player"), "got: {err}");
    }

    /// Distance out of [1, win_length-1] must raise even when the one-hot
    /// and delta arithmetic are internally consistent (di=6 > win_max=5).
    #[test]
    fn out_of_window_distance_raises() {
        let node_feat = vec![0.0f32; 4 * NODE_FEAT_DIM];
        let node_coords: Vec<i32> = vec![0, 0, 6, 0, 2, 0, 0, 0];
        let edge_index: Vec<i64> = vec![0, 1];
        let mut edge_attr = vec![0.0f32; EDGE_FEAT_DIM];
        edge_attr[0] = 1.0;
        edge_attr[3] = 6.0; // di=6 > win_max=5, but delta = (6,0) = di*axis0 — consistent
        let node_offsets: Vec<i64> = vec![0, 4];
        let current_player: Vec<i8> = vec![1];
        let err = verify_edge_geometry_impl(
            &node_feat,
            &node_coords,
            &edge_index,
            &edge_attr,
            &node_offsets,
            &current_player,
            NODE_FEAT_DIM,
            EDGE_FEAT_DIM,
            WIN_LENGTH,
        )
        .unwrap_err();
        assert!(err.contains("edge delta"), "got: {err}");
    }
}
