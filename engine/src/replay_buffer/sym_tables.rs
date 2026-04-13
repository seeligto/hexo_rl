//! Symmetry tables, weight schedule, and buffer constants.

use half::f16;

// ── Constants ─────────────────────────────────────────────────────────────────

pub(crate) const N_PLANES:  usize = 18;
pub(crate) const BOARD_H:   usize = 19;
pub(crate) const BOARD_W:   usize = 19;
pub(crate) const N_CELLS:   usize = BOARD_H * BOARD_W; // 361
pub(crate) const N_ACTIONS: usize = N_CELLS + 1;       // 362 (pass move at index 361)
pub(crate) const N_SYMS:    usize = 12;
const HALF: i32 = 9; // (BOARD_H - 1) / 2

// State stride per buffer slot (f16 bits)
pub(crate) const STATE_STRIDE:  usize = N_PLANES * N_CELLS;
pub(crate) const POLICY_STRIDE: usize = N_ACTIONS;
/// Auxiliary spatial target stride per buffer slot (single 19×19 plane, u8 lanes).
/// Used by ownership and winning_line targets — both share the same shape and
/// scatter table as a single state plane.
pub(crate) const AUX_STRIDE:    usize = N_CELLS;

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

/// Precomputed scatter tables for all 12 hexagonal symmetries.
///
/// `scatter[s]` is the list of `(src_cell, dst_cell)` pairs for symmetry `s`.
/// Cells that fall outside the 19×19 window after transformation are omitted —
/// the corresponding output cells remain zero (matching the Python behaviour).
pub(crate) struct SymTables {
    pub(crate) scatter: [Vec<(u16, u16)>; N_SYMS],
}

impl SymTables {
    pub(crate) fn new() -> Self {
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
        }

        SymTables { scatter }
    }
}
