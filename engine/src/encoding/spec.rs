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
                "value_pool must be one of [none,min,max,mean]; got {:?}",
                other
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
                "policy_pool must be one of [none,scatter_max,scatter_mean]; got {:?}",
                other
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
    pub name: &'static str,
    pub board_size: usize,
    pub trunk_size: usize,
    pub cluster_window_size: Option<usize>,
    pub cluster_threshold: Option<usize>,
    pub legal_move_radius: usize,
    pub n_planes: usize,
    pub plane_layout: &'static [&'static str],
    pub policy_logit_count: usize,
    pub has_pass_slot: bool,
    pub is_multi_window: bool,
    pub value_pool: ValuePool,
    pub policy_pool: PolicyPool,
    pub sym_table_id: &'static str,
    pub schema_version: u32,
    pub notes: &'static str,
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
            self.board_size * self.board_size + (if self.has_pass_slot { 1 } else { 0 });
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
                errs.push(format!("plane_layout[{}] is empty", idx));
            }
            if !seen.insert(p) {
                errs.push(format!("plane_layout[{}] duplicate name {:?}", idx, p));
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
    /// Total cells = board_size².
    pub fn n_cells(&self) -> usize {
        self.board_size * self.board_size
    }

    /// (board_size − 1) / 2 — board half-extent for axial→canvas mapping.
    pub fn half(&self) -> i32 {
        (self.board_size as i32 - 1) / 2
    }

    /// State plane stride = n_planes × n_cells.
    pub fn state_stride(&self) -> usize {
        self.n_planes * self.n_cells()
    }

    /// Chain plane stride = N_CHAIN_PLANES × n_cells.
    pub fn chain_stride(&self) -> usize {
        crate::replay_buffer::sym_tables::N_CHAIN_PLANES * self.n_cells()
    }

    /// Aux plane stride = n_cells (single aux plane).
    pub fn aux_stride(&self) -> usize {
        self.n_cells()
    }

    /// Policy stride is `policy_logit_count` (already a struct field — provided as
    /// accessor for parity with the strides above).
    pub fn policy_stride(&self) -> usize {
        self.policy_logit_count
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
            schema_version: 1,
            notes: "test",
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
}
