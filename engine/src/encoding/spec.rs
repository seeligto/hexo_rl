//! Encoding registry spec — full-schema record per `registry.toml`.
//!
//! Authored §172 Phase A3 (2026-05-09). Distinct from the legacy 4-field
//! `crate::encoding::EncodingSpec` (which §171 plumbing on this branch still
//! uses). A4 is the migration commit — it switches `Board::with_encoding`,
//! the PyO3 boundary, and every consumer to read from `RegistrySpec`. Keep
//! both types alive in parallel until then.
//!
//! Schema source of truth: `docs/designs/encoding_registry_design.md` §3.

use std::collections::BTreeSet;

/// Value-head pooling mode (multi-window only). `None` for single-window.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ValuePool {
    None,
    Min,
    Max,
    Mean,
}

/// Policy-head pooling mode (multi-window only). `None` for single-window.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum PolicyPool {
    None,
    ScatterMax,
    ScatterMean,
}

impl ValuePool {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "none" => Ok(ValuePool::None),
            "min" => Ok(ValuePool::Min),
            "max" => Ok(ValuePool::Max),
            "mean" => Ok(ValuePool::Mean),
            other => Err(format!(
                "value_pool must be one of [none,min,max,mean]; got {other:?}"
            )),
        }
    }

    pub fn is_some(&self) -> bool {
        !matches!(self, ValuePool::None)
    }
}

impl PolicyPool {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "none" => Ok(PolicyPool::None),
            "scatter_max" => Ok(PolicyPool::ScatterMax),
            "scatter_mean" => Ok(PolicyPool::ScatterMean),
            other => Err(format!(
                "policy_pool must be one of [none,scatter_max,scatter_mean]; got {other:?}"
            )),
        }
    }

    pub fn is_some(&self) -> bool {
        !matches!(self, PolicyPool::None)
    }
}

/// Full encoding record parsed from `registry.toml`.
///
/// All `&'static str` / `&'static [..]` fields point at heap data leaked
/// at registry init time (`Box::leak`), so addresses are stable for the
/// process lifetime. Cheap to copy — pass by value or `&'static`.
#[derive(Copy, Clone, Debug)]
pub struct RegistrySpec {
    /// &'static after Box::leak in registry::load(); stable for process lifetime.
    pub name: &'static str,
    pub board_size: usize,
    pub trunk_size: usize,
    pub cluster_window_size: Option<usize>,
    pub cluster_threshold: Option<usize>,
    pub legal_move_radius: usize,
    pub n_planes: usize,
    /// &'static after Box::leak in registry::load(); stable for process lifetime.
    pub plane_layout: &'static [&'static str],
    pub policy_logit_count: usize,
    pub has_pass_slot: bool,
    pub is_multi_window: bool,
    pub value_pool: ValuePool,
    pub policy_pool: PolicyPool,
    /// &'static after Box::leak in registry::load(); stable for process lifetime.
    pub sym_table_id: &'static str,
    pub schema_version: u32,
    /// &'static after Box::leak in registry::load(); stable for process lifetime.
    pub notes: &'static str,

    // ── §173 A3 additions ────────────────────────────────────────────────
    /// Physical source-plane indices retained by this encoding's wire format.
    /// Length == `n_planes`. See `registry.toml` header for the canonical
    /// `[0..3, 8..11]` X+history / O+history block convention.
    /// v6 family: 8 indices from 18-plane source. v8 family: 11 indices
    /// from 21-plane source (adds indices 18, 19, 20).
    ///
    /// &'static after Box::leak in registry::load(); stable for process lifetime.
    pub kept_plane_indices: &'static [usize],
    /// Source tensor plane count *before* `kept_plane_indices` slice.
    /// v6 family = 18; v8 family = 21.
    /// Used by validator for the kept-indices upper bound. Producer-pipeline
    /// migration to read this from spec is deferred to §174.
    pub n_source_planes: usize,
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
    // cycle 3 P68: module split — extract per-invariant check helpers
    #[allow(clippy::too_many_lines)]
    pub fn validate(&self) -> Result<(), String> {
        let mut errs: Vec<String> = Vec::new();

        if self.plane_layout.len() != self.n_planes {
            errs.push(format!(
                "len(plane_layout)={} != n_planes={}",
                self.plane_layout.len(),
                self.n_planes
            ));
        }

        let expected_logits =
            self.board_size * self.board_size + usize::from(self.has_pass_slot);
        if self.policy_logit_count != expected_logits {
            errs.push(format!(
                "policy_logit_count={} != board_size²+(pass_slot?1:0)={} \
                 (board_size={}, has_pass_slot={})",
                self.policy_logit_count, expected_logits, self.board_size, self.has_pass_slot
            ));
        }

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
                PolicyPool::ScatterMax | PolicyPool::ScatterMean
            ) {
                errs.push(format!(
                    "is_multi_window=true requires policy_pool ∈ {{ScatterMax,ScatterMean}}; got {:?}",
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
                    errs.push(format!(
                        "kept_plane_indices: duplicate index {idx}"
                    ));
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

impl RegistrySpec {
    /// Total cells per trunk input tensor = `trunk_size²`.
    ///
    /// `board_size` is canvas geometry (logical grid extent); `trunk_size` is
    /// NN input geometry (= `cluster_window_size` for multi-window, = `board_size`
    /// for single-window). For all current single-window encodings the two are
    /// equal, but future canvas-larger-than-trunk encodings must use `trunk_size`.
    /// §173 A3 semantic fix: was `board_size²` (wrong for multi-window). Now
    /// `trunk_size²` — matches Rust intent and Python parity.
    #[inline]
    pub fn n_cells(&self) -> usize {
        self.trunk_size * self.trunk_size
    }

    /// (board_size − 1) / 2 — board half-extent for axial→canvas mapping.
    #[inline]
    pub fn half(&self) -> i32 {
        (self.board_size as i32 - 1) / 2
    }

    /// State plane stride = n_planes × n_cells.
    #[inline]
    pub fn state_stride(&self) -> usize {
        self.n_planes * self.n_cells()
    }

    /// Chain plane stride = N_CHAIN_PLANES × n_cells.
    #[inline]
    pub fn chain_stride(&self) -> usize {
        crate::replay_buffer::sym_tables::N_CHAIN_PLANES * self.n_cells()
    }

    /// Aux plane stride = n_cells (single aux plane).
    #[inline]
    pub fn aux_stride(&self) -> usize {
        self.n_cells()
    }

    /// Policy stride is `policy_logit_count` (already a struct field — provided as
    /// accessor for parity with the strides above).
    #[inline]
    pub fn policy_stride(&self) -> usize {
        self.policy_logit_count
    }

    /// Wire-format signature for cross-encoding compatibility checks (§P13).
    ///
    /// Two encodings are wire-identical when they produce byte-identical on-disk
    /// rows for the HEXB replay-buffer format. The buffer wire layout depends on
    /// `(n_planes, board_size, policy_logit_count, has_pass_slot, sym_table_id)`
    /// — every other registry field affects training semantics but not stored
    /// bytes. Aliases sharing this tuple auto-cross-load via
    /// `ReplayBuffer::load_from_path_impl`:
    ///   - v6 / v7full / v7 / v7e30 / v7mw → (8, 19, 362, true, "size_19")
    ///   - v8 / v8_canvas_realness        → (11, 25, 625, false, "size_25")
    ///   - v6w25 stays distinct           → (8,  25, 362, true, "size_25")
    ///
    /// Derived from existing fields — Registry-as-SSR (TOML) untouched.
    #[inline]
    pub fn wire_signature(&self) -> (usize, usize, usize, bool, &'static str) {
        (
            self.n_planes,
            self.board_size,
            self.policy_logit_count,
            self.has_pass_slot,
            self.sym_table_id,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ok_v6() -> RegistrySpec {
        RegistrySpec {
            name: "v6",
            board_size: 19,
            trunk_size: 19,
            cluster_window_size: None,
            cluster_threshold: None,
            legal_move_radius: 5,
            n_planes: 8,
            plane_layout: &[
                "current_player_t0",
                "current_player_t-1",
                "current_player_t-2",
                "current_player_t-3",
                "opponent_t0",
                "opponent_t-1",
                "opponent_t-2",
                "opponent_t-3",
            ],
            policy_logit_count: 362,
            has_pass_slot: true,
            is_multi_window: false,
            value_pool: ValuePool::None,
            policy_pool: PolicyPool::None,
            sym_table_id: "size_19",
            schema_version: 2,
            notes: "test",
            kept_plane_indices: &[0, 1, 2, 3, 8, 9, 10, 11],
            n_source_planes: 18,
        }
    }

    #[test]
    fn validate_ok_single_window() {
        ok_v6().validate().unwrap();
    }

    #[test]
    fn validate_rejects_plane_layout_mismatch() {
        let mut s = ok_v6();
        s.n_planes = 9;
        let err = s.validate().unwrap_err();
        assert!(err.contains("len(plane_layout)"));
    }

    #[test]
    fn validate_rejects_policy_logit_count_mismatch_no_pass() {
        let mut s = ok_v6();
        s.has_pass_slot = false;
        // policy_logit_count still 362, but expected = 361
        let err = s.validate().unwrap_err();
        assert!(err.contains("policy_logit_count"));
    }

    #[test]
    fn value_pool_parse_known() {
        assert_eq!(ValuePool::parse("min").unwrap(), ValuePool::Min);
        assert_eq!(ValuePool::parse("max").unwrap(), ValuePool::Max);
        assert_eq!(ValuePool::parse("mean").unwrap(), ValuePool::Mean);
        assert_eq!(ValuePool::parse("none").unwrap(), ValuePool::None);
        assert!(ValuePool::parse("bogus").is_err());
    }

    #[test]
    fn policy_pool_parse_known() {
        assert_eq!(
            PolicyPool::parse("scatter_max").unwrap(),
            PolicyPool::ScatterMax
        );
        assert_eq!(
            PolicyPool::parse("scatter_mean").unwrap(),
            PolicyPool::ScatterMean
        );
        assert_eq!(PolicyPool::parse("none").unwrap(), PolicyPool::None);
        assert!(PolicyPool::parse("bogus").is_err());
    }

    #[test]
    fn test_v8_accessors() {
        let s = crate::encoding::registry::lookup_or_panic("v8");
        assert_eq!(s.n_cells(), 625);
        assert_eq!(s.half(), 12);
        assert_eq!(s.state_stride(), 11 * 625);
        assert_eq!(s.chain_stride(), 6 * 625);
        assert_eq!(s.aux_stride(), 625);
        assert_eq!(s.policy_stride(), 625);
    }

    #[test]
    fn test_v6_accessors() {
        let s = crate::encoding::registry::lookup_or_panic("v6");
        assert_eq!(s.n_cells(), 361);
        assert_eq!(s.half(), 9);
        assert_eq!(s.state_stride(), 8 * 361);
        assert_eq!(s.policy_stride(), 362);
    }

    // §173 A3 — kept_plane_indices validator tests.

    #[test]
    fn validate_rejects_kept_plane_indices_len_mismatch() {
        let mut s = ok_v6();
        s.kept_plane_indices = &[0, 1, 2, 3, 8, 9, 10]; // len=7, n_planes=8
        let err = s.validate().unwrap_err();
        assert!(
            err.contains("len(kept_plane_indices)"),
            "expected len mismatch error, got: {}",
            err
        );
    }

    #[test]
    fn validate_rejects_kept_plane_indices_duplicate() {
        let mut s = ok_v6();
        s.kept_plane_indices = &[0, 0, 2, 3, 8, 9, 10, 11]; // dup index 0
        let err = s.validate().unwrap_err();
        assert!(
            err.contains("duplicate"),
            "expected duplicate error, got: {}",
            err
        );
    }

    #[test]
    fn validate_rejects_kept_plane_indices_out_of_range() {
        let mut s = ok_v6();
        s.kept_plane_indices = &[0, 1, 2, 3, 8, 9, 10, 99]; // 99 >= n_source_planes=18
        let err = s.validate().unwrap_err();
        assert!(
            err.contains("n_source_planes"),
            "expected n_source_planes error, got: {}",
            err
        );
    }

    #[test]
    fn validate_accepts_kept_plane_indices_unsorted() {
        let mut s = ok_v6();
        s.kept_plane_indices = &[3, 1, 2, 0, 11, 10, 9, 8]; // valid unsorted
        s.validate().unwrap(); // no panic
    }

    #[test]
    fn test_n_cells_uses_trunk_size_squared() {
        // Single-window: trunk_size == board_size — no difference.
        let v6 = crate::encoding::registry::lookup_or_panic("v6");
        assert_eq!(v6.n_cells(), v6.trunk_size * v6.trunk_size);
        // Multi-window: trunk_size == cluster_window_size == 25 (board_size also 25 here).
        let v6w25 = crate::encoding::registry::lookup_or_panic("v6w25");
        assert_eq!(v6w25.n_cells(), v6w25.trunk_size * v6w25.trunk_size);
        assert_eq!(v6w25.n_cells(), 625);
    }

    #[test]
    fn test_kept_plane_indices_v6_and_v8() {
        let v6 = crate::encoding::registry::lookup_or_panic("v6");
        assert_eq!(v6.kept_plane_indices, &[0usize, 1, 2, 3, 8, 9, 10, 11]);
        assert_eq!(v6.n_source_planes, 18);

        let v8 = crate::encoding::registry::lookup_or_panic("v8");
        assert_eq!(v8.kept_plane_indices, &[0usize, 1, 2, 3, 8, 9, 10, 11, 18, 19, 20]);
        assert_eq!(v8.n_source_planes, 21);
    }
}
