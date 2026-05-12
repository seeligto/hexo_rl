//! Symmetry tables, weight schedule, and buffer constants.

use half::f16;

// ── Constants ─────────────────────────────────────────────────────────────────

/// Number of state planes stored in the replay buffer (HEXB v6, post-§131).
///
/// Reduced from 18 → 8 per D17 ablation verdict (`reports/audits/d17_ablation_20260429.md`):
/// only planes 0/8 are LOAD-BEARING, plane 2 is MARGINAL, all others REDUNDANT.
/// The kept set retains both ply-0 (LOAD-BEARING) and ply-1..3 history pairs
/// for both players (see `KEPT_PLANE_INDICES`). Scalar metadata
/// (moves_remaining, turn_parity) and older history (ply-4..7) are dropped.
pub const N_PLANES:  usize = 8;
/// Alias for N_PLANES — all 8 buffer planes scatter via coordinate permutation only.
pub const N_HISTORY_PLANES: usize = N_PLANES;
/// Number of Q13 chain-length planes: 3 hex axes × 2 players.
/// Stored separately from the state buffer; scatter via coordinate permutation
/// AND axis-plane remap (see `apply_chain_symmetry`).
pub const N_CHAIN_PLANES: usize = 6;
/// Offset used when building the chain_src_lookup (chains are their own buffer).
pub const CHAIN_PLANE_OFFSET: usize = 0;
pub const BOARD_H:   usize = 19;
pub const BOARD_W:   usize = 19;
pub const N_CELLS:   usize = BOARD_H * BOARD_W; // 361
/// **v6-specific** policy length (361 cells + 1 pass slot).
///
/// §172 A10 T8b — DO NOT use this const on v8 / v6w25 / canvas-realness paths.
/// v8 has a 25×25 board with NO pass slot (625 logits), v6w25 has 626. Using
/// `N_ACTIONS` outside the legacy v6 replay-buffer codepaths produces silent
/// shape divergence (v8 caller writes 362-wide policy buffer into a 625-wide
/// scatter destination → either truncation or out-of-bounds depending on the
/// codepath).
///
/// All in-tree callers as of T8b are v6-bound (`replay_buffer::push`,
/// `replay_buffer::sample`, `replay_buffer::mod`). v8 callers MUST derive
/// from `crate::encoding::RegistrySpec::policy_logit_count` /
/// `policy_stride()` instead.
pub const N_ACTIONS: usize = N_CELLS + 1;       // 362 (pass move at index 361; v6 only)
pub const N_SYMS:    usize = 12;

// State stride per buffer slot (f16 bits) — 8 planes × 361 cells = 2888.
pub const STATE_STRIDE:  usize = N_PLANES * N_CELLS;

/// Indices into the legacy 18-plane game-state tensor that survive the
/// channel-drop (D17 verdict, Set A). Used by the slice-on-push path in
/// `engine/src/game_runner/worker_loop.rs` to pack the 8 kept planes
/// contiguously into the buffer wire format.
///
/// Layout after slice (output plane → source plane):
/// ```text
///   out 0 ← src 0   (cur ply-0; LOAD-BEARING)
///   out 1 ← src 1   (cur ply-1; REDUNDANT but kept)
///   out 2 ← src 2   (cur ply-2; MARGINAL — D14 anchor)
///   out 3 ← src 3   (cur ply-3; REDUNDANT but kept)
///   out 4 ← src 8   (opp ply-0; LOAD-BEARING)
///   out 5 ← src 9   (opp ply-1; REDUNDANT but kept)
///   out 6 ← src 10  (opp ply-2; D14 anchor pair)
///   out 7 ← src 11  (opp ply-3; REDUNDANT but kept)
/// ```
pub const KEPT_PLANE_INDICES: [usize; N_PLANES] = [0, 1, 2, 3, 8, 9, 10, 11];
/// Chain-plane stride per buffer slot (f16 bits) — 6 planes × 361 cells = 2166
pub const CHAIN_STRIDE:  usize = N_CHAIN_PLANES * N_CELLS;
pub const POLICY_STRIDE: usize = N_ACTIONS;
/// Auxiliary spatial target stride per buffer slot (single 19×19 plane, u8 lanes).
/// Used by ownership and winning_line targets — both share the same shape and
/// scatter table as a single state plane.
pub const AUX_STRIDE:    usize = N_CELLS;

// ── v6w25 constants (§168 Gate 3 — K-cluster encoding at matched R=8 ───
//    perception). Wire format = v6 (8 KEPT planes + pass slot) but
//    cluster window grows 19 → 25, cluster threshold grows 5 → 8, and
//    legal-move radius grows 5 → 8. Used as the matched-perception A/B
//    baseline against v8 single-bbox (T3 §168 verdict pending).
//
// DEPRECATED (§173 A4): these constants are superseded by `sym_tables_for(spec)`
// and `spec.*_stride()` / `spec.n_cells()`. Use the spec-based accessors in all
// new code. These consts will be removed in a follow-up commit once A5a/A5b/A6
// have migrated all remaining callers.

// These constants are expressed as plain values to avoid triggering deprecation
// warnings within this module itself (Rust deprecation warnings fire on use,
// including at peer-constant definition sites). The deprecation doc comments
// and #[deprecated] annotations communicate the migration path to callers.

/// v6w25 cluster window side length = 25. DEPRECATED — use `spec.trunk_size`. §173 A4.
#[deprecated(since = "0.173.4", note = "use spec.trunk_size (via sym_tables_for / RegistrySpec::n_cells); §173 A4")]
#[allow(deprecated)]
pub const BOARD_H_V6W25: usize = 25;
/// DEPRECATED — use `spec.trunk_size`. §173 A4.
#[deprecated(since = "0.173.4", note = "use spec.trunk_size; §173 A4")]
#[allow(deprecated)]
pub const BOARD_W_V6W25: usize = 25; // = BOARD_H_V6W25
/// v6w25 total cells = 625. DEPRECATED — use `spec.n_cells()`. §173 A4.
#[deprecated(since = "0.173.4", note = "use spec.n_cells(); §173 A4")]
#[allow(deprecated)]
pub const N_CELLS_V6W25: usize = 625; // = BOARD_H_V6W25 * BOARD_W_V6W25
/// v6w25 plane count = 8. DEPRECATED — use `spec.n_planes`. §173 A4.
#[deprecated(since = "0.173.4", note = "use spec.n_planes; §173 A4")]
#[allow(deprecated)]
pub const N_PLANES_V6W25: usize = 8; // = N_PLANES
/// v6w25 action space = 626. DEPRECATED — use `spec.policy_stride()`. §173 A4.
#[deprecated(since = "0.173.4", note = "use spec.policy_stride(); §173 A4")]
#[allow(deprecated)]
pub const N_ACTIONS_V6W25: usize = 626; // = N_CELLS_V6W25 + 1
/// v6w25 cluster threshold = 8. DEPRECATED — use `spec.cluster_threshold.unwrap()`. §173 A4.
#[deprecated(since = "0.173.4", note = "use spec.cluster_threshold.unwrap(); §173 A4")]
#[allow(deprecated)]
pub const CLUSTER_THRESHOLD_V6W25: i32 = 8;
/// v6w25 legal-move radius = 8. DEPRECATED — use `spec.legal_move_radius`. §173 A4.
#[deprecated(since = "0.173.4", note = "use spec.legal_move_radius; §173 A4")]
#[allow(deprecated)]
pub const LEGAL_MOVE_RADIUS_V6W25: i32 = 8;
/// v6w25 state stride = 5000. DEPRECATED — use `spec.state_stride()`. §173 A4.
#[deprecated(since = "0.173.4", note = "use spec.state_stride(); §173 A4")]
#[allow(deprecated)]
pub const STATE_STRIDE_V6W25: usize = 5000; // = N_PLANES_V6W25 * N_CELLS_V6W25
/// v6w25 chain stride = 3750. DEPRECATED — use `spec.chain_stride()`. §173 A4.
#[deprecated(since = "0.173.4", note = "use spec.chain_stride(); §173 A4")]
#[allow(deprecated)]
pub const CHAIN_STRIDE_V6W25: usize = 3750; // = N_CHAIN_PLANES * N_CELLS_V6W25
/// v6w25 policy stride = 626. DEPRECATED — use `spec.policy_stride()`. §173 A4.
#[deprecated(since = "0.173.4", note = "use spec.policy_stride(); §173 A4")]
#[allow(deprecated)]
pub const POLICY_STRIDE_V6W25: usize = 626; // = N_ACTIONS_V6W25
/// v6w25 auxiliary stride = 625. DEPRECATED — use `spec.aux_stride()`. §173 A4.
#[deprecated(since = "0.173.4", note = "use spec.aux_stride(); §173 A4")]
#[allow(deprecated)]
pub const AUX_STRIDE_V6W25: usize = 625; // = N_CELLS_V6W25

// ── Weight schedule ──────────────────────────────────────────────────────────

/// A single threshold bracket: games with length < `max_moves` get `weight`.
/// Brackets are evaluated in order; the first match wins.
#[derive(Clone, Debug)]
pub(crate) struct WeightBracket {
    pub(crate) max_moves: u16,   // exclusive upper bound (game_length < max_moves)
    pub(crate) weight:    u16,   // f16-as-u16 bits
}

/// Config-driven weight schedule for game-length-based sampling.
/// Default: all positions have weight 1.0 (uniform sampling).
#[derive(Clone, Debug)]
pub(crate) struct WeightSchedule {
    pub(crate) brackets: Vec<WeightBracket>,
    pub(crate) default_weight: u16, // f16 bits for weight when no bracket matches
}

impl WeightSchedule {
    pub(crate) fn uniform() -> Self {
        WeightSchedule {
            brackets: Vec::new(),
            default_weight: f16::from_f32(1.0).to_bits(),
        }
    }

    /// Look up the weight (as f16 bits) for a given game length.
    #[inline]
    pub(crate) fn weight_for(&self, game_length: u16) -> u16 {
        for b in &self.brackets {
            if game_length < b.max_moves {
                return b.weight;
            }
        }
        self.default_weight
    }
}

// ── SymTables ─────────────────────────────────────────────────────────────────

/// Canonical hex-basis directions mirroring `engine/src/board/state.rs:48-52`
/// `HEX_AXES`. Used to derive the axis-plane permutation under each symmetry.
pub(crate) const HEX_BASIS: [(i32, i32); 3] = [
    (1, 0),  // axis 0 — E/W
    (0, 1),  // axis 1 — NE/SW
    (1, -1), // axis 2 — SE/NW
];

/// Apply the canonical `(q, r) → (−r, q + r)` 60° rotation, `n_rot` times.
#[inline]
fn rotate_n(mut q: i32, mut r: i32, n_rot: usize) -> (i32, i32) {
    for _ in 0..n_rot {
        let nq = -r;
        let nr = q + r;
        q = nq;
        r = nr;
    }
    (q, r)
}

/// Compare two axial vectors in a direction-unsigned sense (axis identity).
#[inline]
fn same_axis(a: (i32, i32), b: (i32, i32)) -> bool {
    a == b || (-a.0, -a.1) == b
}

/// Precomputed scatter tables for all 12 hexagonal symmetries.
///
/// `scatter[s]` is the list of `(src_cell, dst_cell)` pairs for symmetry `s`.
/// Cells that fall outside the 19×19 window after transformation are omitted —
/// the corresponding output cells remain zero (matching the Python behaviour).
///
/// `axis_perm[s]` is the per-symmetry axis-plane remap for the Q13 chain-length
/// planes (chain sub-buffer, planes 0..5). `axis_perm[s][dst_j] = src_i` means:
/// "under symmetry s, the destination plane for axis j holds values scattered
/// from the source plane for axis i". The remap is direction-unsigned (hex runs
/// are bi-directional) and composes reflection-then-rotation to match the same
/// order used in the coordinate-scatter construction below.
///
/// The 2-plane-per-axis (current/opponent) layout means the real scatter loop
/// iterates `(axis_perm[s][dst_j], player_off)` pairs for chain planes 0..5.
pub struct SymTables {
    /// Board side length for which scatter tables were built (square boards
    /// only). v6: 19. v8: 25.
    pub board_size: usize,
    /// Total cells = `board_size * board_size`. v6: 361. v8: 625.
    pub n_cells: usize,
    /// State plane count for which `src_plane_lookup` is sized. v6: 8. v8: 11.
    pub n_planes: usize,
    pub scatter:   [Vec<(u16, u16)>; N_SYMS],
    /// Per-symmetry axis-plane remap for Q13 chain-length planes.
    /// `axis_perm[s][dst_j] = src_i` means destination plane for axis j reads
    /// from source plane for axis i under symmetry s. Board-size invariant
    /// (depends only on hex axes).
    pub axis_perm: [[usize; 3]; N_SYMS],
    /// Fused per-symmetry source-plane lookup for the state planes.
    /// State planes are pure coordinate scatter (identity plane mapping), so
    /// `src_plane_lookup[s][p] == p` for all s and p. `apply_symmetry_state`
    /// no longer consumes this field (it's now plane-count-generic and uses
    /// implicit identity mapping); retained as P4's aug-table substrate for any
    /// future per-plane permutation use case.
    /// Outer length: N_SYMS=12. Inner length: `n_planes` (runtime).
    pub src_plane_lookup: Vec<Vec<usize>>,
    /// Fused per-symmetry source-plane lookup for the 6 chain-length planes.
    /// `chain_src_lookup[s][dst_p] = src_p`: coordinate + axis-plane remap.
    /// Inner length is the universal `N_CHAIN_PLANES = 6`.
    pub chain_src_lookup: [[usize; N_CHAIN_PLANES]; N_SYMS],
}

impl SymTables {
    /// Build sym tables at the v6 default shape (`board_size=19, n_planes=8`).
    /// v6 byte-exact: identical scatter LUTs and axis_perm to pre-§166 master.
    pub fn new() -> Self {
        Self::with_shape(BOARD_H, N_PLANES)
    }

    /// Build sym tables at an arbitrary square board shape and plane count.
    ///
    /// Used by v8 callers (e.g. `SymTables::with_shape(25, 11)`) and by tests.
    /// The 12-fold scatter LUTs depend on `board_size` (hex window dimensions);
    /// axis_perm is board-size invariant (purely a function of the 3 hex axes);
    /// chain_src_lookup is plane-count invariant (always 6 chain planes
    /// regardless of state plane count).
    ///
    /// Panics if `board_size` is even (sym scatter assumes a centred odd window).
    pub fn with_shape(board_size: usize, n_planes: usize) -> Self {
        assert!(
            board_size % 2 == 1,
            "SymTables: board_size must be odd, got {}", board_size
        );
        let n_cells = board_size * board_size;
        let half = (board_size as i32 - 1) / 2;

        // Axial → flat index.  Returns None if the result is out of the window.
        let to_flat = |q: i32, r: i32| -> Option<u16> {
            let qi = q + half;
            let ri = r + half;
            if qi >= 0 && qi < board_size as i32 && ri >= 0 && ri < board_size as i32 {
                Some((qi as usize * board_size + ri as usize) as u16)
            } else {
                None
            }
        };

        // Flat index → axial coordinates.
        let from_flat = |flat: usize| -> (i32, i32) {
            ((flat / board_size) as i32 - half, (flat % board_size) as i32 - half)
        };

        // Each symmetry gets its own Vec.  We use a const-size array with a dummy
        // initialiser then overwrite each element.
        const EMPTY: Vec<(u16, u16)> = Vec::new();
        let mut scatter = [EMPTY; N_SYMS];
        let mut axis_perm = [[0usize; 3]; N_SYMS];

        for sym_idx in 0..N_SYMS {
            let reflect = sym_idx >= 6;
            let n_rot   = sym_idx % 6;
            let mut pairs: Vec<(u16, u16)> = Vec::with_capacity(n_cells);

            for src in 0..n_cells {
                let (mut q, mut r) = from_flat(src);

                // Optional reflection first (swap axes).
                if reflect {
                    (q, r) = (r, q);
                }

                // Apply n_rot × 60° rotations: (q, r) → (−r, q+r).
                for _ in 0..n_rot {
                    (q, r) = (-r, q + r);
                }

                if let Some(dst) = to_flat(q, r) {
                    pairs.push((src as u16, dst));
                }
            }

            scatter[sym_idx] = pairs;

            // Derive the axis-plane permutation for this symmetry by applying
            // the SAME transform to each hex basis vector and matching the
            // result to one of the three canonical axes (direction-unsigned).
            //
            // For each source axis i, find the destination axis j such that
            // the transformed basis[i] matches ±basis[j]. The resulting
            // mapping i→j is inverted into `axis_perm[sym_idx][j] = i`, i.e.
            // "destination plane j reads from source plane i".
            let mut perm = [usize::MAX; 3];
            for src_i in 0..3 {
                let (mut q, mut r) = HEX_BASIS[src_i];
                if reflect {
                    (q, r) = (r, q);
                }
                let (tq, tr) = rotate_n(q, r, n_rot);
                let mut matched = false;
                for dst_j in 0..3 {
                    if same_axis((tq, tr), HEX_BASIS[dst_j]) {
                        perm[dst_j] = src_i;
                        matched = true;
                        break;
                    }
                }
                debug_assert!(
                    matched,
                    "transformed basis axis {:?} did not match any canonical axis \
                     (sym_idx={}, reflect={}, n_rot={})",
                    (tq, tr), sym_idx, reflect, n_rot
                );
            }
            debug_assert!(
                perm.iter().all(|&i| i < 3),
                "axis_perm[{}] has an unset slot: {:?}", sym_idx, perm
            );
            // Sanity check: perm must be a bijection on {0,1,2}.
            let mut seen = [false; 3];
            for &i in &perm {
                debug_assert!(!seen[i], "axis_perm[{}] is not a bijection: {:?}", sym_idx, perm);
                seen[i] = true;
            }
            axis_perm[sym_idx] = perm;
        }

        // Build fused src_plane_lookup for the state planes.
        // All state planes are pure coordinate scatter — identity plane mapping.
        let mut src_plane_lookup: Vec<Vec<usize>> = Vec::with_capacity(N_SYMS);
        for _ in 0..N_SYMS {
            src_plane_lookup.push((0..n_planes).collect());
        }

        // Build chain_src_lookup for the 6 chain-length planes.
        // Each destination axis j reads from source axis i (axis-perm remap),
        // with current/opponent interleaved: plane index = 2*axis + player_off.
        let mut chain_src_lookup = [[0usize; N_CHAIN_PLANES]; N_SYMS];
        for s in 0..N_SYMS {
            for dst_axis in 0..3 {
                let src_axis = axis_perm[s][dst_axis];
                for player_off in 0..2 {
                    chain_src_lookup[s][2 * dst_axis + player_off] =
                        CHAIN_PLANE_OFFSET + 2 * src_axis + player_off;
                }
            }
        }

        SymTables {
            board_size,
            n_cells,
            n_planes,
            scatter,
            axis_perm,
            src_plane_lookup,
            chain_src_lookup,
        }
    }
}

// ── sym_tables_for() — per-spec lazy constructor ──────────────────────────────

use once_cell::sync::Lazy;

/// Return the pre-built `SymTables` for the given encoding spec, keyed on
/// `spec.sym_table_id`.
///
/// Lazily initialises one static `SymTables` per distinct `(sym_table_id,
/// n_planes)` combination and returns a `&'static` reference. Building is
/// O(N_CELLS × N_SYMS) ≈ 4 µs for v6 / ≈ 7 µs for v6w25; subsequent calls
/// are free (pointer return only).
///
/// Supported `sym_table_id` values (from `registry.toml`):
///   `"size_19"` → v6/v7full sym tables (board 19×19, n_planes as given).
///   `"size_25"` → v6w25 / v8 / v8_canvas_realness sym tables (25×25).
///
/// Panics on an unknown `sym_table_id` — only valid registry entries should
/// ever be passed here.
///
/// §173 A4: used by A5a to replace the unconditional `SymTables::new()` call
/// in `worker_loop.rs:151` (H1-α hazard: SymTables v6 unconditional for all
/// encodings). A4 places the function here so A5a can call it without owning
/// the `sym_tables` module.
pub fn sym_tables_for(spec: &'static crate::encoding::RegistrySpec) -> &'static SymTables {
    // One static per distinct (sym_table_id, n_planes) pair. All current
    // encodings collapse to three combinations; the match on sym_table_id +
    // n_planes is explicit so future encodings panic loud rather than silently
    // reusing a wrong table.
    //
    // "size_19" (v6, v7full): board 19×19, n_planes=8 — shared singleton.
    static SIZE19_8:  Lazy<SymTables> = Lazy::new(|| SymTables::with_shape(19, 8));
    // "size_25" n_planes=8 (v6w25): board 25×25, 8-plane wire format.
    static SIZE25_8:  Lazy<SymTables> = Lazy::new(|| SymTables::with_shape(25, 8));
    // "size_25" n_planes=11 (v8, v8_canvas_realness): board 25×25, 11-plane wire format.
    static SIZE25_11: Lazy<SymTables> = Lazy::new(|| SymTables::with_shape(25, 11));

    match (spec.sym_table_id, spec.n_planes) {
        ("size_19", _)  => &*SIZE19_8,
        ("size_25", 8)  => &*SIZE25_8,
        ("size_25", 11) => &*SIZE25_11,
        (id, np) => panic!(
            "sym_tables_for: no sym table for encoding {:?} (sym_table_id={:?}, n_planes={}). \
             Add a Lazy<SymTables> entry in sym_tables::sym_tables_for().",
            spec.name, id, np
        ),
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Apply the sym transform `(reflect, n_rot)` to an axial coordinate and
    /// return the result. Matches the coordinate-scatter convention in
    /// `SymTables::new()` exactly.
    fn apply_sym_coord(q: i32, r: i32, reflect: bool, n_rot: usize) -> (i32, i32) {
        let (mut q, mut r) = (q, r);
        if reflect {
            (q, r) = (r, q);
        }
        rotate_n(q, r, n_rot)
    }

    #[test]
    fn identity_axis_perm_is_identity() {
        let tables = SymTables::new();
        assert_eq!(tables.axis_perm[0], [0, 1, 2]);
    }

    #[test]
    fn rot60_axis_perm_cycles_2_0_1() {
        // sym_idx = 1: reflect=false, n_rot=1. Hand-derived:
        //   basis[0]=(1,0) → (0,1)=basis[1]       → i=0 → j=1
        //   basis[1]=(0,1) → (-1,1)=-basis[2]     → i=1 → j=2
        //   basis[2]=(1,-1) → (1,0)=basis[0]      → i=2 → j=0
        // axis_perm[1][j] = i such that j is destination:
        //   j=0 ← i=2; j=1 ← i=0; j=2 ← i=1 → [2, 0, 1]
        let tables = SymTables::new();
        assert_eq!(tables.axis_perm[1], [2, 0, 1]);
    }

    #[test]
    fn rot120_axis_perm_cycles_1_2_0() {
        // Applying rot60 twice: [2,0,1] composed with itself.
        //   j=0 ← i=2; rot60 of src=2 → j=0 (already rotated)... let's just hand-derive.
        //   n_rot=2 on basis[0]=(1,0): step1→(0,1), step2→(-1,1)=-basis[2] → j=2
        //   basis[1]=(0,1): step1→(-1,1), step2→(-1,0)=-basis[0] → j=0
        //   basis[2]=(1,-1): step1→(1,0), step2→(0,1)=basis[1] → j=1
        // axis_perm[2]: j=0 ← i=1; j=1 ← i=2; j=2 ← i=0 → [1, 2, 0]
        let tables = SymTables::new();
        assert_eq!(tables.axis_perm[2], [1, 2, 0]);
    }

    #[test]
    fn rot180_axis_perm_is_identity() {
        // n_rot=3 = 180°: (q,r) → (-q,-r) up to the 60° rotation formula iterated 3x.
        // Direction-unsigned, all axes map to themselves.
        let tables = SymTables::new();
        assert_eq!(tables.axis_perm[3], [0, 1, 2]);
    }

    #[test]
    fn rot240_matches_rot60() {
        // Axis permutation has period 3 (180° is identity), so rot240 == rot60.
        let tables = SymTables::new();
        assert_eq!(tables.axis_perm[4], tables.axis_perm[1]);
    }

    #[test]
    fn rot300_matches_rot120() {
        let tables = SymTables::new();
        assert_eq!(tables.axis_perm[5], tables.axis_perm[2]);
    }

    #[test]
    fn reflection_only_swaps_axis0_and_axis1() {
        // sym_idx = 6: reflect=true, n_rot=0.
        //   basis[0]=(1,0) → reflect → (0,1) = basis[1] → i=0 → j=1
        //   basis[1]=(0,1) → reflect → (1,0) = basis[0] → i=1 → j=0
        //   basis[2]=(1,-1) → reflect → (-1,1) = -basis[2] → i=2 → j=2
        // axis_perm[6]: j=0 ← i=1; j=1 ← i=0; j=2 ← i=2 → [1, 0, 2]
        let tables = SymTables::new();
        assert_eq!(tables.axis_perm[6], [1, 0, 2]);
    }

    #[test]
    fn axis_perm_derived_from_coord_transform() {
        // Strong consistency check: for every symmetry and every source axis i,
        // applying the sym to basis[i] (direction-unsigned) must produce the
        // basis vector at axis_perm_inverse[i]. This ties the plane permutation
        // to the cell scatter from exactly the same derivation path, catching
        // any drift between the two.
        let tables = SymTables::new();
        for sym_idx in 0..N_SYMS {
            let reflect = sym_idx >= 6;
            let n_rot = sym_idx % 6;

            // Build inverse: perm_inv[i] = j s.t. perm[j] = i.
            let perm = tables.axis_perm[sym_idx];
            let mut perm_inv = [usize::MAX; 3];
            for (j, &i) in perm.iter().enumerate() {
                perm_inv[i] = j;
            }
            for &v in &perm_inv {
                assert!(v < 3, "axis_perm[{}] inverse has unset slot", sym_idx);
            }

            for src_i in 0..3 {
                let (bq, br) = HEX_BASIS[src_i];
                let (tq, tr) = apply_sym_coord(bq, br, reflect, n_rot);
                let expected_dst_j = perm_inv[src_i];
                assert!(
                    same_axis((tq, tr), HEX_BASIS[expected_dst_j]),
                    "sym_idx={} src_i={}: transformed basis {:?} does not match \
                     expected axis {} ({:?})",
                    sym_idx, src_i, (tq, tr), expected_dst_j, HEX_BASIS[expected_dst_j]
                );
            }
        }
    }

    #[test]
    fn axis_perm_is_bijection_for_all_syms() {
        let tables = SymTables::new();
        for sym_idx in 0..N_SYMS {
            let perm = tables.axis_perm[sym_idx];
            let mut seen = [false; 3];
            for &i in &perm {
                assert!(i < 3, "axis_perm[{}] has out-of-range index: {:?}", sym_idx, perm);
                assert!(!seen[i], "axis_perm[{}] is not a bijection: {:?}", sym_idx, perm);
                seen[i] = true;
            }
        }
    }

    #[test]
    fn reflection_composed_with_rot60() {
        // sym_idx = 7: reflect=true, n_rot=1.
        // Apply reflect first, then 1 rotation.
        //   basis[0]=(1,0) → reflect →(0,1) → rot60 →(-1,1)=-basis[2] → i=0 → j=2
        //   basis[1]=(0,1) → reflect →(1,0) → rot60 →(0,1)=basis[1]   → i=1 → j=1
        //   basis[2]=(1,-1) → reflect →(-1,1) → rot60 →(-1,0)=-basis[0] → i=2 → j=0
        // axis_perm[7]: j=0 ← i=2; j=1 ← i=1; j=2 ← i=0 → [2, 1, 0]
        let tables = SymTables::new();
        assert_eq!(tables.axis_perm[7], [2, 1, 0]);
    }

    // ── v8 SymTables tests (§166 Bucket A) ──────────────────────────────────

    #[test]
    fn v8_with_shape_records_dims() {
        let s = crate::encoding::registry::lookup_or_panic("v8");
        let tables = SymTables::with_shape(s.board_size, s.n_planes);
        assert_eq!(tables.board_size, 25);
        assert_eq!(tables.n_cells, 625);
        assert_eq!(tables.n_planes, 11);
    }

    #[test]
    fn v8_axis_perm_matches_v6() {
        // axis_perm depends only on the 3 hex axes, not on board size — v8
        // and v6 must produce identical permutations.
        let spec = crate::encoding::registry::lookup_or_panic("v8");
        let v6 = SymTables::new();
        let v8 = SymTables::with_shape(spec.board_size, spec.n_planes);
        for s in 0..N_SYMS {
            assert_eq!(
                v6.axis_perm[s], v8.axis_perm[s],
                "axis_perm[{}] must be board-size invariant", s
            );
        }
    }

    #[test]
    fn v8_chain_src_lookup_matches_v6() {
        // chain_src_lookup is plane-count invariant (always 6 chain planes)
        // and only depends on axis_perm — so v8 and v6 must agree.
        let spec = crate::encoding::registry::lookup_or_panic("v8");
        let v6 = SymTables::new();
        let v8 = SymTables::with_shape(spec.board_size, spec.n_planes);
        for s in 0..N_SYMS {
            assert_eq!(v6.chain_src_lookup[s], v8.chain_src_lookup[s],
                "chain_src_lookup[{}] must be board-size invariant", s);
        }
    }

    #[test]
    fn v8_identity_sym_is_identity_scatter() {
        // sym=0 (reflect=false, n_rot=0): every cell maps to itself.
        let spec = crate::encoding::registry::lookup_or_panic("v8");
        let tables = SymTables::with_shape(spec.board_size, spec.n_planes);
        let scatter = &tables.scatter[0];
        assert_eq!(scatter.len(), 625, "v8 identity must produce 625 pairs");
        for &(src, dst) in scatter {
            assert_eq!(src, dst,
                "identity sym must map every cell to itself; got {} → {}", src, dst);
        }
    }

    #[test]
    fn v8_scatter_indices_in_bounds() {
        let spec = crate::encoding::registry::lookup_or_panic("v8");
        let tables = SymTables::with_shape(spec.board_size, spec.n_planes);
        for s in 0..N_SYMS {
            for &(src, dst) in &tables.scatter[s] {
                assert!((src as usize) < 625, "v8 src out of bounds: sym {} src {}", s, src);
                assert!((dst as usize) < 625, "v8 dst out of bounds: sym {} dst {}", s, dst);
            }
        }
    }

    #[test]
    fn v8_scatter_pairs_have_unique_src_and_dst() {
        // Square hex windows drop out-of-window destinations under non-trivial
        // sym (matches v6 behavior — see comment at SymTables.scatter). The
        // surviving pairs must still have unique src and unique dst (no two
        // cells map to the same destination, no destination is reached from
        // two sources).
        let spec = crate::encoding::registry::lookup_or_panic("v8");
        let tables = SymTables::with_shape(spec.board_size, spec.n_planes);
        for s in 0..N_SYMS {
            let scatter = &tables.scatter[s];
            assert!(scatter.len() <= 625, "sym {} produced too many pairs: {}", s, scatter.len());
            let mut src_seen = vec![false; 625];
            let mut dst_seen = vec![false; 625];
            for &(src, dst) in scatter {
                assert!(!src_seen[src as usize], "sym {} duplicate src {}", s, src);
                assert!(!dst_seen[dst as usize], "sym {} duplicate dst {}", s, dst);
                src_seen[src as usize] = true;
                dst_seen[dst as usize] = true;
            }
        }
    }

    #[test]
    fn v8_identity_and_rot180_preserve_all_cells() {
        // sym 0 (identity) and sym 3 (rot180: (q,r)→(-q,-r)) are guaranteed
        // bijections on a square window centred at origin — every cell's
        // rotated image stays inside the window.
        let spec = crate::encoding::registry::lookup_or_panic("v8");
        let tables = SymTables::with_shape(spec.board_size, spec.n_planes);
        assert_eq!(tables.scatter[0].len(), 625, "v8 identity must keep all 625 cells");
        assert_eq!(tables.scatter[3].len(), 625, "v8 rot180 must keep all 625 cells");
    }

    #[test]
    fn v6_v8_rot180_preserve_all_cells() {
        // Same property applies at v6 dimensions — regression guard that the
        // refactor didn't change v6 behaviour on the bijective syms.
        let v6 = SymTables::new();
        assert_eq!(v6.scatter[0].len(), 361, "v6 identity must keep all 361 cells");
        assert_eq!(v6.scatter[3].len(), 361, "v6 rot180 must keep all 361 cells");
    }

    #[test]
    fn v8_src_plane_lookup_is_identity_at_n_planes_11() {
        let spec = crate::encoding::registry::lookup_or_panic("v8");
        let tables = SymTables::with_shape(spec.board_size, spec.n_planes);
        for s in 0..N_SYMS {
            assert_eq!(tables.src_plane_lookup[s].len(), 11);
            for p in 0..11 {
                assert_eq!(tables.src_plane_lookup[s][p], p,
                    "v8 src_plane_lookup must be identity; got [{}][{}] = {}",
                    s, p, tables.src_plane_lookup[s][p]);
            }
        }
    }

    #[test]
    fn v6_default_byte_exact() {
        // SymTables::new() must produce v6-shape output identical to before
        // §166. This is the v6 byte-exact regression guard.
        let tables = SymTables::new();
        assert_eq!(tables.board_size, 19);
        assert_eq!(tables.n_cells, 361);
        assert_eq!(tables.n_planes, 8);
        // Identity sym scatter must be 361 (1, 1) → … → (361, 361) pairs.
        assert_eq!(tables.scatter[0].len(), 361);
        for (i, &(src, dst)) in tables.scatter[0].iter().enumerate() {
            assert_eq!(src as usize, i);
            assert_eq!(dst as usize, i);
        }
        // src_plane_lookup must still be 8-wide identity.
        for s in 0..N_SYMS {
            assert_eq!(tables.src_plane_lookup[s].len(), 8);
            for p in 0..8 {
                assert_eq!(tables.src_plane_lookup[s][p], p);
            }
        }
    }

    /// §172 A10 T8b HIGH-3 — pin v8 path uses spec.policy_stride() (625),
    /// NOT the v6-specific `N_ACTIONS` const (362). Regression: any
    /// out-of-replay-buffer path that imports `N_ACTIONS` for v8 silently
    /// truncates the 25×25 policy → 19×19+1.
    #[test]
    fn test_v8_policy_stride_not_n_actions() {
        let s = crate::encoding::registry::lookup_or_panic("v8");
        assert_eq!(s.policy_stride(), 625);
        assert_ne!(
            s.policy_stride(),
            N_ACTIONS,
            "v8 must not silently use v6 N_ACTIONS=362"
        );
    }

    #[test]
    fn test_v6_policy_stride_matches_n_actions() {
        let s = crate::encoding::registry::lookup_or_panic("v6");
        assert_eq!(s.policy_stride(), N_ACTIONS);
    }

    // ── sym_tables_for() tests (§173 A4) ────────────────────────────────────

    #[test]
    fn sym_tables_for_v6_matches_new() {
        let spec = crate::encoding::registry::lookup_or_panic("v6");
        let via_fn = crate::replay_buffer::sym_tables::sym_tables_for(spec);
        let via_new = SymTables::new();
        // Same board geometry.
        assert_eq!(via_fn.board_size, via_new.board_size);
        assert_eq!(via_fn.n_cells,    via_new.n_cells);
        assert_eq!(via_fn.n_planes,   via_new.n_planes);
        // Same scatter tables for all 12 symmetries.
        for s in 0..N_SYMS {
            assert_eq!(via_fn.scatter[s], via_new.scatter[s],
                "scatter[{s}] mismatch between sym_tables_for(v6) and SymTables::new()");
            assert_eq!(via_fn.axis_perm[s], via_new.axis_perm[s],
                "axis_perm[{s}] mismatch");
            assert_eq!(via_fn.chain_src_lookup[s], via_new.chain_src_lookup[s],
                "chain_src_lookup[{s}] mismatch");
        }
    }

    #[test]
    fn sym_tables_for_v7full_matches_v6_shape() {
        // v7full uses sym_table_id="size_19" and n_planes=8 — same table as v6.
        let spec = crate::encoding::registry::lookup_or_panic("v7full");
        let tables = crate::replay_buffer::sym_tables::sym_tables_for(spec);
        assert_eq!(tables.board_size, 19);
        assert_eq!(tables.n_cells, 361);
        assert_eq!(tables.n_planes, 8);
    }

    #[test]
    fn sym_tables_for_v6w25_has_size_25() {
        let spec = crate::encoding::registry::lookup_or_panic("v6w25");
        let tables = crate::replay_buffer::sym_tables::sym_tables_for(spec);
        assert_eq!(tables.board_size, 25);
        assert_eq!(tables.n_cells, 625);
        assert_eq!(tables.n_planes, 8);
        // Identity scatter must cover all 625 cells.
        assert_eq!(tables.scatter[0].len(), 625);
    }

    #[test]
    fn sym_tables_for_v8_has_size_25_n11() {
        let spec = crate::encoding::registry::lookup_or_panic("v8");
        let tables = crate::replay_buffer::sym_tables::sym_tables_for(spec);
        assert_eq!(tables.board_size, 25);
        assert_eq!(tables.n_cells, 625);
        assert_eq!(tables.n_planes, 11);
    }

    #[test]
    fn sym_tables_for_returns_stable_ref() {
        // Calling twice returns the same static address (Lazy singleton).
        let spec = crate::encoding::registry::lookup_or_panic("v6");
        let t1 = crate::replay_buffer::sym_tables::sym_tables_for(spec) as *const _;
        let t2 = crate::replay_buffer::sym_tables::sym_tables_for(spec) as *const _;
        assert_eq!(t1, t2, "sym_tables_for must return the same static singleton");
    }
}
