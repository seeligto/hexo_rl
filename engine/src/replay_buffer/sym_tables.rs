//! Symmetry tables, weight schedule, and buffer constants.

use half::f16;

// ── Constants ─────────────────────────────────────────────────────────────────

/// Number of input state planes: 8 cur-history + 8 opp-history + 2 scalar = 18.
/// Chain-length planes moved to a separate sub-buffer (output-only auxiliary).
pub const N_PLANES:  usize = 18;
/// Alias for N_PLANES — all 18 planes scatter via coordinate permutation only.
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
pub const N_ACTIONS: usize = N_CELLS + 1;       // 362 (pass move at index 361)
pub const N_SYMS:    usize = 12;
const HALF: i32 = 9; // (BOARD_H - 1) / 2

// State stride per buffer slot (f16 bits) — 18 planes × 361 cells = 6498
pub const STATE_STRIDE:  usize = N_PLANES * N_CELLS;
/// Chain-plane stride per buffer slot (f16 bits) — 6 planes × 361 cells = 2166
pub const CHAIN_STRIDE:  usize = N_CHAIN_PLANES * N_CELLS;
pub const POLICY_STRIDE: usize = N_ACTIONS;
/// Auxiliary spatial target stride per buffer slot (single 19×19 plane, u8 lanes).
/// Used by ownership and winning_line targets — both share the same shape and
/// scatter table as a single state plane.
pub const AUX_STRIDE:    usize = N_CELLS;

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
/// planes (24-plane layout, planes 18..23). `axis_perm[s][dst_j] = src_i` means:
/// "under symmetry s, the destination plane for axis j holds values scattered
/// from the source plane for axis i". The remap is direction-unsigned (hex runs
/// are bi-directional) and composes reflection-then-rotation to match the same
/// order used in the coordinate-scatter construction below.
///
/// The 2-plane-per-axis (current/opponent) layout means the real scatter loop
/// iterates `(axis_perm[s][dst_j], player_off)` pairs for planes 18..23.
pub struct SymTables {
    pub scatter:   [Vec<(u16, u16)>; N_SYMS],
    /// Per-symmetry axis-plane remap for Q13 chain-length planes.
    /// `axis_perm[s][dst_j] = src_i` means destination plane for axis j reads
    /// from source plane for axis i under symmetry s.
    pub axis_perm: [[usize; 3]; N_SYMS],
    /// Fused per-symmetry source-plane lookup for the 18 state planes.
    /// All 18 planes are pure coordinate scatter (identity plane mapping),
    /// so `src_plane_lookup[s][dst_p] == dst_p` for all s and p.
    /// Kept as a lookup array so `apply_symmetry_state` can share the same
    /// loop structure as the former 24-plane kernel.
    pub src_plane_lookup: [[usize; N_PLANES]; N_SYMS],
    /// Fused per-symmetry source-plane lookup for the 6 chain-length planes.
    /// `chain_src_lookup[s][dst_p] = src_p`: coordinate + axis-plane remap.
    pub chain_src_lookup: [[usize; N_CHAIN_PLANES]; N_SYMS],
}

impl SymTables {
    pub fn new() -> Self {
        // Axial → flat index.  Returns None if the result is out of the 19×19 window.
        let to_flat = |q: i32, r: i32| -> Option<u16> {
            let qi = q + HALF;
            let ri = r + HALF;
            if qi >= 0 && qi < BOARD_H as i32 && ri >= 0 && ri < BOARD_W as i32 {
                Some((qi as usize * BOARD_W + ri as usize) as u16)
            } else {
                None
            }
        };

        // Flat index → axial coordinates.
        let from_flat = |flat: usize| -> (i32, i32) {
            ((flat / BOARD_W) as i32 - HALF, (flat % BOARD_W) as i32 - HALF)
        };

        // Each symmetry gets its own Vec.  We use a const-size array with a dummy
        // initialiser then overwrite each element.
        const EMPTY: Vec<(u16, u16)> = Vec::new();
        let mut scatter = [EMPTY; N_SYMS];
        let mut axis_perm = [[0usize; 3]; N_SYMS];

        for sym_idx in 0..N_SYMS {
            let reflect = sym_idx >= 6;
            let n_rot   = sym_idx % 6;
            let mut pairs: Vec<(u16, u16)> = Vec::with_capacity(N_CELLS);

            for src in 0..N_CELLS {
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

        // Build fused src_plane_lookup for the 18 state planes.
        // All state planes are pure coordinate scatter — identity plane mapping.
        let mut src_plane_lookup = [[0usize; N_PLANES]; N_SYMS];
        for s in 0..N_SYMS {
            for p in 0..N_PLANES {
                src_plane_lookup[s][p] = p;
            }
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

        SymTables { scatter, axis_perm, src_plane_lookup, chain_src_lookup }
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
}
