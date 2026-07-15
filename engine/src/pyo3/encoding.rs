//! Python-visible RegistrySpec — wraps `&'static crate::encoding::RegistrySpec`.
//!
//! Extracted from `engine/src/lib.rs` at §178 Wave 5b Commit 1. Byte-identical
//! move (struct + #[pymethods] + inherent impl); only the surrounding `use`
//! lines, the file-level doc comment, and the `register()` registration helper
//! are new.

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

use crate::encoding::RegistrySpec as RustRegistrySpec;

/// Python-visible RegistrySpec — wraps `&'static crate::encoding::RegistrySpec`.
/// Returned by `RegistrySpec.from_registry(name)`. Carries derived shape
/// accessors (`state_stride()`, `policy_stride()`) so PyO3 callers
/// constructing `SelfPlayRunner` / `InferenceBatcher` can derive
/// `feature_len` / `policy_len` from the canonical registry instead of
/// duplicating the per-encoding shape table.
///
/// Read-only — clone is `Copy` (just the &'static pointer).
//
// TODO(post-§172): pyo3 0.28 deprecated automatic `FromPyObject` derivation
// for `#[pyclass]` types implementing `Clone`. Build emits one warning at
// pyclass macro expansion; the warning will become a hard error on pyo3
// 1.0. Migration is mechanical: add `#[pyclass(from_py_object)]` to opt-in
// OR `#[pyclass(skip_from_py_object)]` to skip. No timeline — gated by pyo3
// upgrade prioritization.
#[pyclass(name = "RegistrySpec", from_py_object)]
#[derive(Clone, Copy)]
pub struct PyRegistrySpec {
    inner: &'static RustRegistrySpec,
}

#[pymethods]
impl PyRegistrySpec {
    #[getter] pub fn name(&self) -> &'static str { self.inner.name }
    #[getter] pub fn board_size(&self) -> usize { self.inner.board_size }
    #[getter] pub fn trunk_size(&self) -> usize { self.inner.trunk_size }
    #[getter] pub fn cluster_window_size(&self) -> Option<usize> { self.inner.cluster_window_size }
    #[getter] pub fn cluster_threshold(&self) -> Option<usize> { self.inner.cluster_threshold }
    #[getter] pub fn legal_move_radius(&self) -> usize { self.inner.legal_move_radius }
    #[getter] pub fn n_planes(&self) -> usize { self.inner.n_planes }
    #[getter] pub fn plane_layout(&self) -> Vec<&'static str> { self.inner.plane_layout.to_vec() }
    #[getter] pub fn policy_logit_count(&self) -> usize { self.inner.policy_logit_count }
    #[getter] pub fn has_pass_slot(&self) -> bool { self.inner.has_pass_slot }
    #[getter] pub fn is_multi_window(&self) -> bool { self.inner.is_multi_window }
    /// §P3.2 — wire-format pool enums exposed as strings (matches Python `Literal` shape
    /// returned by the @dataclass `value_pool` / `policy_pool` fields).
    #[getter] pub fn value_pool(&self) -> &'static str {
        match self.inner.value_pool {
            crate::encoding::ValuePool::None => "none",
            crate::encoding::ValuePool::Min => "min",
            crate::encoding::ValuePool::Max => "max",
            crate::encoding::ValuePool::Mean => "mean",
        }
    }
    #[getter] pub fn policy_pool(&self) -> &'static str {
        match self.inner.policy_pool {
            crate::encoding::PolicyPool::None => "none",
            crate::encoding::PolicyPool::ScatterMax => "scatter_max",
            crate::encoding::PolicyPool::ScatterMean => "scatter_mean",
            crate::encoding::PolicyPool::LegalSetScatterMax => "legal_set_scatter_max",
        }
    }
    #[getter] pub fn sym_table_id(&self) -> &'static str { self.inner.sym_table_id }
    #[getter] pub fn schema_version(&self) -> u32 { self.inner.schema_version }
    #[getter] pub fn notes(&self) -> &'static str { self.inner.notes }
    /// §173 A3 — physical source-plane indices retained by wire format.
    #[getter] pub fn kept_plane_indices(&self) -> Vec<usize> {
        self.inner.kept_plane_indices.to_vec()
    }
    /// §173 A3 — source tensor plane count before kept_plane_indices slice.
    #[getter] pub fn n_source_planes(&self) -> usize { self.inner.n_source_planes }
    /// cycle 3 P55 / Wave 9 — multi-window cluster-count upper bound per
    /// position emitted by `Board::get_cluster_views()`. = 1 for
    /// single-window encodings. Read by `hexo_rl/selfplay/pool.py` when
    /// computing the `InferenceBatcher.pool_size` auto-derive heuristic.
    #[getter] pub fn k_max(&self) -> u32 { self.inner.k_max }

    // ── GNN-integration schema v4 — representation discriminant + graph geom.
    /// "grid" (dense CNN planes) | "graph" (axis-graph GNN). Read by the
    /// Python `build_net` dispatch + `graph_collate` resolver (seam design
    /// §5-below). Grid for every pre-v4 encoding.
    #[getter] pub fn representation(&self) -> &'static str { self.inner.representation.as_str() }
    /// Convenience mirror of `representation == "graph"`.
    #[getter] pub fn is_graph(&self) -> bool { self.inner.is_graph() }
    /// Per-node feature width (graph only; `None` for grid). = 11.
    #[getter] pub fn node_feat_dim(&self) -> Option<usize> { self.inner.node_feat_dim }
    /// Per-edge feature width (graph only). = 5.
    #[getter] pub fn edge_feat_dim(&self) -> Option<usize> { self.inner.edge_feat_dim }
    /// GNN win-length (graph only). = 6.
    #[getter] pub fn win_length(&self) -> Option<usize> { self.inner.win_length }
    /// GNN legal-move / axis-walk radius (graph only). = 6.
    #[getter] pub fn graph_radius(&self) -> Option<usize> { self.inner.graph_radius }
    /// Number of win axes (graph only). = 3.
    #[getter] pub fn win_axes(&self) -> Option<usize> { self.inner.win_axes }
    /// Ragged-payload contract version this encoding speaks (graph only). = 1.
    #[getter] pub fn contract_version(&self) -> Option<u32> { self.inner.contract_version }
    /// Required native builder tag the resolver asserts (graph only). = 1.
    #[getter] pub fn builder_impl_required(&self) -> Option<u8> { self.inner.builder_impl_required }

    /// Alias for `policy_logit_count` — matches the retired Python @dataclass
    /// `n_actions` @property (Wave 8 Batch A FF.2 parity).
    #[getter] pub fn n_actions(&self) -> usize { self.inner.policy_logit_count }
    /// Cells per trunk input tensor = trunk_size². §173 A3 semantic: trunk_size, not board_size.
    #[getter] pub fn n_cells(&self) -> usize { self.inner.n_cells() }
    /// State plane stride = n_planes × n_cells.
    #[getter] pub fn state_stride(&self) -> usize { self.inner.state_stride() }
    /// Chain plane stride = N_CHAIN_PLANES × n_cells.
    #[getter] pub fn chain_stride(&self) -> usize { self.inner.chain_stride() }
    /// Aux plane stride = n_cells (single aux plane).
    #[getter] pub fn aux_stride(&self) -> usize { self.inner.aux_stride() }
    /// Policy logit count = `policy_logit_count` (mirror of the field).
    #[getter] pub fn policy_stride(&self) -> usize { self.inner.policy_stride() }

    pub fn __repr__(&self) -> String {
        format!(
            "RegistrySpec(name={:?}, board_size={}, n_planes={}, policy_logit_count={}, is_multi_window={})",
            self.inner.name, self.inner.board_size, self.inner.n_planes,
            self.inner.policy_logit_count, self.inner.is_multi_window,
        )
    }

    /// §P3.1 — registry-backed lookup. Returns a `PyRegistrySpec` (full-schema
    /// record incl. policy_logit_count + n_planes). Supersedes the legacy
    /// `EncodingSpec.from_registry` classmethod (whose return type is also
    /// `PyRegistrySpec`); the legacy entry stays alive for one commit so
    /// callers can migrate, then is deleted in P3.2.
    #[classmethod]
    pub fn from_registry(_cls: &Bound<'_, pyo3::types::PyType>, name: &str) -> PyResult<Self> {
        if let Some(spec) = crate::encoding::lookup(name) {
            Ok(PyRegistrySpec { inner: spec })
        } else {
            let mut known: Vec<&str> =
                crate::encoding::all_specs().map(|s| s.name).collect();
            known.sort_unstable();
            Err(PyValueError::new_err(format!(
                "RegistrySpec.from_registry: unknown encoding {name:?}; registered: {known:?}"
            )))
        }
    }
}

impl PyRegistrySpec {
    /// Crate-internal accessor — used by `SelfPlayRunner::new` /
    /// `InferenceBatcher::new` to read the static pointer.
    pub(crate) fn inner(&self) -> &'static RustRegistrySpec {
        self.inner
    }

    /// §173 A5a — test helper: construct from a `&'static RegistrySpec`
    /// reference (e.g. returned by `lookup_or_panic`). Allows Rust integration
    /// tests to pass a `PyRegistrySpec` to `SelfPlayRunner::new` without
    /// going through the Python boundary.
    pub fn from_static(spec: &'static RustRegistrySpec) -> Self {
        PyRegistrySpec { inner: spec }
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRegistrySpec>()?;
    Ok(())
}
