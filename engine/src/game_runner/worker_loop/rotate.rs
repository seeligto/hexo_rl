//! Pure-transform rotation helpers + temperature schedule
//! (cycle 3 Wave 10 Batch A).
//!
//! Extracted verbatim from the pre-split `worker_loop.rs`:
//!   - `compute_move_temperature` (pre-split L11-31)
//!   - `inv_sym_idx` (pre-split L69-77)
//!   - `rotate_state_inplace` (pre-split L79-93)
//!   - `rotate_chain_inplace` (pre-split L95-105)
//!   - `rotate_policy_inplace` (pre-split L107-123)
//!   - `rotate_aux_inplace` (pre-split L125-140)
//!
//! Co-located by "pure transform / no captured state" theme. All 5
//! `#[inline]` attributes migrate verbatim — Wave 10 Batch A perf-gate hinges
//! on cross-module `#[inline]` preservation (LLVM cross-module inlining on
//! `--release` builds; pre-registered fragile claim L.4 in Wave 10 PREP §L).
//!
//! `compute_move_temperature` is `pub` (NOT `pub(super)`) so the
//! `pub use worker_loop::compute_move_temperature` re-export at
//! `engine/src/game_runner/mod.rs:19` keeps working through `mod.rs` of this
//! sub-module.

use crate::replay_buffer::sample::{apply_chain_symmetry, apply_symmetry_state};
use crate::replay_buffer::sym_tables::SymTables;

/// Quarter-cosine temperature schedule used by the self-play worker loop.
///
/// Returns 1.0 at compound_move=0, decays via cos(π/2·progress) toward
/// temp_min, then clamps at temp_min for compound_move ≥ temp_threshold.
///
/// # Arguments
/// * `compound_move`  — zero-indexed compound move number in the current game
/// * `temp_threshold` — compound move at which the floor kicks in. Config-driven;
///   default `0` = schedule OFF (returns a constant `temp_min` at every move).
/// * `temp_min`       — minimum temperature floor. Config-driven; default `0.5`.
pub fn compute_move_temperature(
    compound_move: usize,
    temp_threshold: usize,
    temp_min: f32,
) -> f32 {
    if compound_move < temp_threshold {
        let progress = compound_move as f32 / temp_threshold as f32;
        f32::max(temp_min, (std::f32::consts::FRAC_PI_2 * progress).cos())
    } else {
        temp_min
    }
}

/// Inverse of dihedral element `s` parameterized as reflect-then-rotate^n.
///
/// Pure rotations (`s ∈ 0..6`): inverse is rotation by `(6 - s) % 6`.
/// Reflective elements (`s ∈ 6..12`): self-inverse — `F·R^n` is an involution
/// since `F·R^n·F·R^n = R^n·F·F·R^n = R^n·R^-n = e` (using `F·R^n·F = R^-n`).
#[inline]
pub(super) fn inv_sym_idx(s: usize) -> usize {
    if s < 6 { (6 - s) % 6 } else { s }
}

/// Forward-scatter a state buffer in place under `sym_idx`.
///
/// Plane-count-generic — `apply_symmetry_state` deduces `n_planes` from
/// `buf.len() / N_CELLS`, so this helper handles both the 18-plane legacy
/// inference tensor (model still consumes 18 planes pre-P3 migration) and
/// the 8-plane HEXB v6 buffer-bound tensor.
///
/// Allocates a temporary buffer; cheap relative to inference. `sym_idx == 0` is
/// the identity scatter (the caller may short-circuit if it owns the path).
#[inline]
pub(super) fn rotate_state_inplace(buf: &mut Vec<f32>, sym_idx: usize, tables: &SymTables) {
    let mut tmp = vec![0.0f32; buf.len()];
    apply_symmetry_state::<f32>(buf, &mut tmp, sym_idx, tables);
    std::mem::swap(buf, &mut tmp);
}

/// Pack the 8 kept planes (HEXB v6 wire format) from an 18-plane game-state
/// tensor into a freshly-allocated 8-plane buffer.
///
/// Forward-scatter a 6-plane chain buffer in place under `sym_idx`.
/// Includes the axis-plane remap (chain planes encode hex-axis-specific data).
#[inline]
pub(super) fn rotate_chain_inplace(buf: &mut Vec<f32>, sym_idx: usize, tables: &SymTables) {
    let mut tmp = vec![0.0f32; buf.len()];
    apply_chain_symmetry::<f32>(buf, &mut tmp, sym_idx, tables);
    std::mem::swap(buf, &mut tmp);
}

/// Forward-scatter a single policy buffer in place. The pass-action slot
/// (at index `n_cells`) is a global identity — it stays at the same index.
///
/// §173 A5a (H2-α): `n_cells` replaces the hardcoded `SYM_N_CELLS = 361`
/// constant so the pass-slot guard works correctly for v6w25 (n_cells=625).
#[inline]
pub(super) fn rotate_policy_inplace(
    buf: &mut Vec<f32>,
    sym_idx: usize,
    tables: &SymTables,
    n_cells: usize,
) {
    let mut tmp = vec![0.0f32; buf.len()];
    let scatter = &tables.scatter[sym_idx];
    for &(sc, dc) in scatter {
        tmp[dc as usize] = buf[sc as usize];
    }
    if buf.len() > n_cells {
        tmp[n_cells] = buf[n_cells];
    }
    std::mem::swap(buf, &mut tmp);
}

/// Forward-scatter the combined aux_u8 buffer (ownership ‖ winning_line) in place.
/// Ownership default is 1 (empty); winning_line default is 0 (no win mask).
///
/// §173 A5a (H3-α): `n_cells` replaces hardcoded `TOTAL_CELLS = 361` so the
/// ownership/winning-line split point is correct for v6w25 (625 cells per half).
#[inline]
pub(super) fn rotate_aux_inplace(
    buf: &mut Vec<u8>,
    sym_idx: usize,
    tables: &SymTables,
    n_cells: usize,
) {
    let mut tmp = vec![0u8; buf.len()];
    tmp[..n_cells].fill(1); // ownership default = empty
    let scatter = &tables.scatter[sym_idx];
    for &(sc, dc) in scatter {
        tmp[dc as usize] = buf[sc as usize];
        tmp[n_cells + dc as usize] = buf[n_cells + sc as usize];
    }
    std::mem::swap(buf, &mut tmp);
}
