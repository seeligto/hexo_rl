//! Encoding registry spec — full-schema record per `registry.toml`.
//!
//! Authored §172 Phase A3 (2026-05-09). The sole encoding record type;
//! the legacy 4-field `EncodingSpec` retired in Wave 8 Batch B
//! (cycle 3 FF.3 residue). Per-Board construction goes through
//! `Board::with_registry_spec`.
//!
//! Schema source of truth: `docs/designs/encoding_registry_design.md` §3.

mod validate;

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

    // ── cycle 3 P55 / Wave 9 (schema v3) additions ───────────────────────
    /// Multi-window cluster-count upper bound per position emitted by
    /// `Board::get_cluster_views()`. Single-window encodings emit exactly
    /// 1 view per leaf, so `k_max = 1` is the canonical degenerate case;
    /// multi-window encodings (v6w25, v7mw) use a conservative cap above
    /// the observed mid/late-game K_avg from the α design doc.
    /// Used by `SelfPlayRunner::new` to auto-derive the
    /// `InferenceBatcher.pool_size` default when the caller omits an
    /// explicit kwarg. NOT a hard runtime cap on `get_cluster_views()`.
    pub k_max: u32,
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

    /// Number of chain-length planes (= 6 across all current encodings:
    /// 3 hex axes × 2 players). Equivalent to `chain_stride() / n_cells()`.
    /// §P19 — decouples consumers (e.g. `replay_buffer::sample`) from a
    /// hardcoded reach-through into `sym_tables::N_CHAIN_PLANES`.
    #[inline]
    pub fn n_chain_planes(&self) -> usize {
        debug_assert_eq!(
            self.chain_stride(),
            6 * self.n_cells(),
            "n_chain_planes invariant: chain_stride must equal 6 × n_cells",
        );
        6
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

    /// Kept-slot index of a source plane within `kept_plane_indices`.
    /// Panics if the source plane is not retained by this encoding.
    #[inline]
    fn kept_slot_of(&self, src_plane: usize) -> usize {
        self.kept_plane_indices
            .iter()
            .position(|&p| p == src_plane)
            .unwrap_or_else(|| {
                panic!(
                    "encoding {:?} does not keep source plane {} (kept={:?})",
                    self.name, src_plane, self.kept_plane_indices
                )
            })
    }

    /// Slice index of the current-player t0 stone plane (source plane 0).
    ///
    /// §P5-CT de-hardcoding sweep — Rust mirror of `resolve_arch().cur_stone_slot`.
    /// Always 0 today (every kept set leads with source plane 0) but derived
    /// from the registry so a plane-reorder cannot silently shift it.
    #[inline]
    pub fn cur_stone_slot(&self) -> usize {
        self.kept_slot_of(0)
    }

    /// Slice index of the opponent t0 stone plane (source plane 8).
    ///
    /// §P5-CT — Rust mirror of `resolve_arch().opp_stone_slot`. Slot 4 for the
    /// v6 family (kept [0,1,2,3,8,...]), slot 1 for v6_live2 (kept [0,8,16,17]).
    #[inline]
    pub fn opp_stone_slot(&self) -> usize {
        self.kept_slot_of(8)
    }

    /// Kept-slot indices of the history planes (source 1,2,3 / 9,10,11) the
    /// encoding retains. Empty for v6_live2 (history dropped — the H-PLANE fix).
    pub fn history_planes(&self) -> Vec<usize> {
        const HISTORY_SRC: [usize; 6] = [1, 2, 3, 9, 10, 11];
        self.kept_plane_indices
            .iter()
            .enumerate()
            .filter(|(_, &p)| HISTORY_SRC.contains(&p))
            .map(|(slot, _)| slot)
            .collect()
    }

    /// Kept-slot indices of the turn-phase planes (source 16,17) the encoding
    /// retains. Non-empty only for v6tp / v6_live2 (CF-2 signal, §P5-CT).
    pub fn turn_phase_planes(&self) -> Vec<usize> {
        const TURN_PHASE_SRC: [usize; 2] = [16, 17];
        self.kept_plane_indices
            .iter()
            .enumerate()
            .filter(|(_, &p)| TURN_PHASE_SRC.contains(&p))
            .map(|(slot, _)| slot)
            .collect()
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
    ///   - v6w25 stays distinct           → (8,  25, 626, true, "size_25")
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
            k_max: 1,
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

    // §P5-CT de-hardcoding sweep — Rust mirror of the Python `resolve_arch`
    // derived slot accessors. Init-time only (no MCTS hot path); parity with
    // hexo_rl/encoding/resolvers.py so a future Rust consumer reads the same
    // registry-derived slots instead of a v6 literal.
    #[test]
    fn test_arch_slot_accessors() {
        let look = crate::encoding::registry::lookup_or_panic;

        // v6: kept [0,1,2,3,8,9,10,11] → cur slot 0, opp slot 4,
        // history slots [1,2,3,5,6,7], no turn-phase.
        let v6 = look("v6");
        assert_eq!(v6.cur_stone_slot(), 0);
        assert_eq!(v6.opp_stone_slot(), 4);
        assert_eq!(v6.history_planes(), vec![1, 2, 3, 5, 6, 7]);
        assert_eq!(v6.turn_phase_planes(), Vec::<usize>::new());

        // v6tp: kept [...,16,17] → turn-phase at slots [8,9], opp still slot 4.
        let v6tp = look("v6tp");
        assert_eq!(v6tp.opp_stone_slot(), 4);
        assert_eq!(v6tp.turn_phase_planes(), vec![8, 9]);

        // v6_live2: kept [0,8,16,17] → opp slot 1, no history, turn-phase [2,3].
        let live2 = look("v6_live2");
        assert_eq!(live2.cur_stone_slot(), 0);
        assert_eq!(live2.opp_stone_slot(), 1);
        assert_eq!(live2.history_planes(), Vec::<usize>::new());
        assert_eq!(live2.turn_phase_planes(), vec![2, 3]);

        // v8: aux planes 18/19/20 are neither history nor turn-phase.
        let v8 = look("v8");
        assert_eq!(v8.cur_stone_slot(), 0);
        assert_eq!(v8.opp_stone_slot(), 4);
        assert_eq!(v8.turn_phase_planes(), Vec::<usize>::new());
    }
}
