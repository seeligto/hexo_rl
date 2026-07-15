//! `RegistrySpec::validate` — cross-field invariant checks.
//!
//! Extracted from `engine/src/encoding/spec.rs` at cycle 3 P68 Wave 7 Batch E
//! as a pure module split. The impl block lives here; the type definition
//! and accessor methods stay in `super` (`spec/mod.rs`). Rust's allowance
//! for splitting `impl T { ... }` blocks across files means callers reach
//! `spec.validate()` through the `RegistrySpec` type regardless of file.

use std::collections::BTreeSet;

use super::{PolicyPool, RegistrySpec, Representation, ValuePool};

// GNN-integration schema v4: the graph invariants are single-sourced against
// the axis-graph builder's own schema constants (engine → hexo-graph rlib dep,
// seam design §2.2) so `node_feat_dim`/`edge_feat_dim`/`win_axes`/
// `builder_impl_required` can never drift from what `build_axis_graph` emits.
use hexo_graph::{BUILDER_IMPL_NATIVE, EDGE_FEAT_DIM, NODE_FEAT_DIM, WIN_AXES};

/// Graph-only field must be present and equal the builder-schema constant.
/// Pushes a named error otherwise (missing OR mismatched). Single-sources the
/// registry's graph dims against `hexo_graph`'s own constants.
fn require_graph_eq(errs: &mut Vec<String>, key: &str, got: Option<usize>, want: usize) {
    match got {
        Some(v) if v == want => {}
        Some(v) => errs.push(format!(
            "{key}={v} must equal {want} (axis-graph builder schema)"
        )),
        None => errs.push(format!("representation=graph requires {key}")),
    }
}

impl RegistrySpec {
    /// Validate cross-field invariants. Collects ALL violations into a
    /// single multi-line message; never short-circuits on the first.
    ///
    /// Per design §3.3:
    ///   - len(plane_layout) == n_planes
    ///   - policy_logit_count == board_size² + (1 if has_pass_slot else 0)
    ///   - is_multi_window ⇔ cluster_window_size.is_some() ⇔ cluster_threshold.is_some()
    ///   - is_multi_window=true  ⇒ value_pool ∈ {Min,Max,Mean},
    ///                              policy_pool ∈ {ScatterMax,ScatterMean}
    ///   - is_multi_window=false ⇒ value_pool == None, policy_pool == None,
    ///                              cluster_window_size == None,
    ///                              cluster_threshold == None
    ///   - trunk_size == cluster_window_size if is_multi_window else board_size
    ///   - legal_move_radius > 0
    // cycle 3 P68: hosts cross-field invariant collection; `#[allow]` preserved
    // because the per-invariant block (9 checks + error-collection) runs >100
    // LOC by design (SD4 vs PREP §J).
    #[allow(clippy::too_many_lines)]
    pub fn validate(&self) -> Result<(), String> {
        let mut errs: Vec<String> = Vec::new();

        // Action space = board_size² + (pass?1:0). Load-bearing for BOTH
        // representations (a graph plays the identical 19×19+pass board — the
        // policy_logit_count STAYS 362, seam design §5-below / contract §C4).
        let expected_logits =
            self.board_size * self.board_size + usize::from(self.has_pass_slot);

        let cw_some = self.cluster_window_size.is_some();
        let ct_some = self.cluster_threshold.is_some();
        if cw_some != self.is_multi_window {
            errs.push(format!(
                "is_multi_window={} != cluster_window_size.is_some()={}",
                self.is_multi_window, cw_some
            ));
        }
        if ct_some != self.is_multi_window {
            errs.push(format!(
                "is_multi_window={} != cluster_threshold.is_some()={}",
                self.is_multi_window, ct_some
            ));
        }

        if self.is_multi_window {
            if !matches!(
                self.value_pool,
                ValuePool::Min | ValuePool::Max | ValuePool::Mean
            ) {
                errs.push(format!(
                    "is_multi_window=true requires value_pool ∈ {{Min,Max,Mean}}; got {:?}",
                    self.value_pool
                ));
            }
            if !matches!(
                self.policy_pool,
                PolicyPool::ScatterMax | PolicyPool::ScatterMean | PolicyPool::LegalSetScatterMax
            ) {
                errs.push(format!(
                    "is_multi_window=true requires policy_pool ∈ {{ScatterMax,ScatterMean,LegalSetScatterMax}}; got {:?}",
                    self.policy_pool
                ));
            }
            if let Some(cw) = self.cluster_window_size {
                if self.trunk_size != cw {
                    errs.push(format!(
                        "is_multi_window=true requires trunk_size==cluster_window_size; \
                         got trunk_size={}, cluster_window_size={}",
                        self.trunk_size, cw
                    ));
                }
            }
        } else {
            if !matches!(self.value_pool, ValuePool::None) {
                errs.push(format!(
                    "is_multi_window=false requires value_pool=None; got {:?}",
                    self.value_pool
                ));
            }
            if !matches!(self.policy_pool, PolicyPool::None) {
                errs.push(format!(
                    "is_multi_window=false requires policy_pool=None; got {:?}",
                    self.policy_pool
                ));
            }
            if self.cluster_window_size.is_some() {
                errs.push(format!(
                    "is_multi_window=false requires cluster_window_size=None; got {:?}",
                    self.cluster_window_size
                ));
            }
            if self.cluster_threshold.is_some() {
                errs.push(format!(
                    "is_multi_window=false requires cluster_threshold=None; got {:?}",
                    self.cluster_threshold
                ));
            }
            if self.trunk_size != self.board_size {
                errs.push(format!(
                    "is_multi_window=false requires trunk_size==board_size; \
                     got trunk_size={}, board_size={}",
                    self.trunk_size, self.board_size
                ));
            }
        }

        if self.legal_move_radius == 0 {
            errs.push("legal_move_radius must be > 0".to_string());
        }

        // cycle 3 P55 / Wave 9 — k_max must be >= 1. Single-window encodings
        // emit exactly 1 cluster view per leaf (the canonical board encode);
        // multi-window encodings cap at the registry-declared upper bound.
        // No cross-validation against is_multi_window — a future canvas-
        // augmented single-window arch may legitimately set k_max > 1.
        if self.k_max == 0 {
            errs.push("k_max must be >= 1".to_string());
        }

        // Plane layout sanity: no duplicates, no empty strings.
        let mut seen: BTreeSet<&str> = BTreeSet::new();
        for (idx, p) in self.plane_layout.iter().enumerate() {
            if p.is_empty() {
                errs.push(format!("plane_layout[{idx}] is empty"));
            }
            if !seen.insert(p) {
                errs.push(format!("plane_layout[{idx}] duplicate name {p:?}"));
            }
        }

        // ── representation-gated invariants (GNN-integration schema v4) ──────
        // Grid-only invariants (plane/kept-index/policy-formula) are gated on
        // Grid; the axis-graph geometry invariants are gated on Graph. The
        // multi-window / legal_move_radius / k_max checks above stay universal
        // (a graph is single-window, so it passes them unchanged). Seam design
        // §5-below.
        match self.representation {
            Representation::Grid => {
                if self.plane_layout.len() != self.n_planes {
                    errs.push(format!(
                        "len(plane_layout)={} != n_planes={}",
                        self.plane_layout.len(),
                        self.n_planes
                    ));
                }
                if self.policy_logit_count != expected_logits {
                    errs.push(format!(
                        "policy_logit_count={} != board_size²+(pass_slot?1:0)={} \
                         (board_size={}, has_pass_slot={})",
                        self.policy_logit_count,
                        expected_logits,
                        self.board_size,
                        self.has_pass_slot
                    ));
                }
                // §173 A3 — kept_plane_indices + n_source_planes validators.
                // 3.1: len(kept_plane_indices) == n_planes
                if self.kept_plane_indices.len() != self.n_planes {
                    errs.push(format!(
                        "len(kept_plane_indices)={} != n_planes={}",
                        self.kept_plane_indices.len(),
                        self.n_planes
                    ));
                }
                // 3.5: n_source_planes >= n_planes
                if self.n_source_planes < self.n_planes {
                    errs.push(format!(
                        "n_source_planes={} < n_planes={} (kept set must be a subset of source)",
                        self.n_source_planes, self.n_planes
                    ));
                }
                // 3.2: no duplicates in kept_plane_indices
                {
                    let mut seen_idx: BTreeSet<usize> = BTreeSet::new();
                    for &idx in self.kept_plane_indices {
                        if !seen_idx.insert(idx) {
                            errs.push(format!("kept_plane_indices: duplicate index {idx}"));
                        }
                    }
                }
                // 3.3: max(kept_plane_indices) < n_source_planes
                if let Some(&max_idx) = self.kept_plane_indices.iter().max() {
                    if max_idx >= self.n_source_planes {
                        errs.push(format!(
                            "kept_plane_indices: max index {} >= n_source_planes={}",
                            max_idx, self.n_source_planes
                        ));
                    }
                }
                // A grid encoding must not carry graph-only fields.
                for (present, key) in [
                    (self.node_feat_dim.is_some(), "node_feat_dim"),
                    (self.edge_feat_dim.is_some(), "edge_feat_dim"),
                    (self.win_length.is_some(), "win_length"),
                    (self.graph_radius.is_some(), "graph_radius"),
                    (self.win_axes.is_some(), "win_axes"),
                    (self.contract_version.is_some(), "contract_version"),
                    (self.builder_impl_required.is_some(), "builder_impl_required"),
                ] {
                    if present {
                        errs.push(format!(
                            "representation=grid must not set graph-only key {key:?}"
                        ));
                    }
                }
            }
            Representation::Graph => {
                // action space UNCHANGED (identical 19×19+pass board, §C4).
                if self.policy_logit_count != expected_logits {
                    errs.push(format!(
                        "policy_logit_count={} != board_size²+(pass_slot?1:0)={} \
                         (graph plays the identical board; action space unchanged)",
                        self.policy_logit_count, expected_logits
                    ));
                }
                if !self.has_pass_slot {
                    errs.push("representation=graph requires has_pass_slot=true".to_string());
                }
                // node/edge/axis dims single-sourced against the builder schema.
                require_graph_eq(&mut errs, "node_feat_dim", self.node_feat_dim, NODE_FEAT_DIM);
                require_graph_eq(&mut errs, "edge_feat_dim", self.edge_feat_dim, EDGE_FEAT_DIM);
                require_graph_eq(&mut errs, "win_axes", self.win_axes, WIN_AXES.len());
                // win_length / graph_radius: present + positive (feed BuildParams).
                match self.win_length {
                    Some(w) if w >= 1 => {}
                    Some(w) => errs.push(format!("win_length={w} must be >= 1")),
                    None => errs.push("representation=graph requires win_length".to_string()),
                }
                match self.graph_radius {
                    Some(r) if r >= 1 => {}
                    Some(r) => errs.push(format!("graph_radius={r} must be >= 1")),
                    None => errs.push("representation=graph requires graph_radius".to_string()),
                }
                // contract + builder handshake the resolver asserts (§7).
                if self.contract_version != Some(1) {
                    errs.push(format!(
                        "representation=graph requires contract_version=1; got {:?}",
                        self.contract_version
                    ));
                }
                if self.builder_impl_required != Some(BUILDER_IMPL_NATIVE) {
                    errs.push(format!(
                        "representation=graph requires builder_impl_required={} (native); got {:?}",
                        BUILDER_IMPL_NATIVE, self.builder_impl_required
                    ));
                }
                // no dense planes (whole-board graph; seam design §5-below).
                if self.n_planes != 0 {
                    errs.push(format!("representation=graph requires n_planes=0; got {}", self.n_planes));
                }
                if !self.plane_layout.is_empty() {
                    errs.push(format!(
                        "representation=graph requires empty plane_layout; got {} entries",
                        self.plane_layout.len()
                    ));
                }
                if !self.kept_plane_indices.is_empty() {
                    errs.push(format!(
                        "representation=graph requires empty kept_plane_indices; got {} entries",
                        self.kept_plane_indices.len()
                    ));
                }
                if self.n_source_planes != 0 {
                    errs.push(format!(
                        "representation=graph requires n_source_planes=0; got {}",
                        self.n_source_planes
                    ));
                }
            }
        }

        if errs.is_empty() {
            Ok(())
        } else {
            Err(format!(
                "RegistrySpec {:?} validation failed:\n  - {}",
                self.name,
                errs.join("\n  - ")
            ))
        }
    }
}
