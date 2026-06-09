//! Per-game record helpers for the self-play worker loop.
//!
//! Contains policy aggregation between cluster-local and global action
//! frames, visit-count policy sampling, and the game-end reprojection of
//! final ownership + winning-line targets into each row's per-cluster
//! window centre. All functions are pure (no worker state) and free so the
//! worker loop can call them without holding `self`.

use crate::board::{Board, Cell};
use fxhash::FxHashMap;
use rand::{rng, RngExt};

/// Fuse K cluster-local policies into one global policy in the main board's
/// MCTS action frame.
///
/// §173 A5b: encoding geometry is passed as pre-extracted integers so the
/// caller's hot loop avoids copying the full `RegistrySpec` struct (~174 B)
/// on every call. Caller derives these once from spec before entering the
/// per-sim loop:
///   - `n_actions`     = `spec.policy_stride()` (362 for v6/v7full, 626 for v6w25, 625 for v8)
///   - `has_pass_slot` = `spec.has_pass_slot` (true for v6/v6w25/v7full; false for v8/v8_canvas_realness)
///   - `trunk_sz`      = `spec.trunk_size as i32` (19 for v6, 25 for v6w25/v8)
///
/// §P2: `has_pass_slot` gates the pass-slot skip + zero-write at the tail of
/// the policy array. Pre-P2, both writes assumed `policy[n_actions-1]` is
/// the pass slot, which CORRUPTED v8's bottom-right corner cell at flat
/// index 624 (audit FD.3): the legal-cell scatter contribution was skipped
/// AND the slot was zeroed unconditionally — structurally killing that cell
/// in v8 selfplay.
///
/// Scatter-max semantics UNCHANGED (α design §3.1): for each legal move m,
/// locate all clusters k that cover m, take max probability across them,
/// assign to `global_policy[mcts_idx(m)]`.
///
/// For each legal move, takes the max probability across clusters that cover
/// it. When `has_pass_slot=true`, the pass move (index n_actions - 1) is set
/// to 0.0; when false, every index from 0 to n_actions-1 is a legal cell.
/// The result is renormalised to sum to 1; if every entry is ~0 we fall back
/// to a uniform distribution to avoid division by zero downstream.
#[inline]
pub fn aggregate_policy(
    n_actions: usize,
    has_pass_slot: bool,
    trunk_sz: i32,
    board: &Board,
    centers: &[(i32, i32)],
    cluster_policies: &[Vec<f32>],
) -> Vec<f32> {
    let half = (trunk_sz - 1) / 2;
    // §173 A8'': MCTS-frame index map must use spec-derived trunk_sz, not the
    // BOARD_SIZE=19 default baked into `Board::window_flat_idx_at`. Use the
    // inline `window_flat_idx_at_geom` kernel with the same (trunk_sz, half)
    // already pre-extracted by the worker_loop boundary.
    let (bcq, bcr) = board.window_center();

    let mut global_policy = vec![0.0; n_actions];
    let legal = board.legal_moves();

    for &(q, r) in &legal {
        let mcts_idx = Board::window_flat_idx_at_geom(q, r, bcq, bcr, trunk_sz, half);
        // §P2: only skip the tail index when it really IS a pass slot. v8 has
        // no pass slot — the tail is a real legal cell.
        if has_pass_slot && mcts_idx >= n_actions - 1 { continue; }

        let mut max_prob = 0.0;
        for (k, &(cq, cr)) in centers.iter().enumerate() {
            let wq = q - cq + half;
            let wr = r - cr + half;
            if wq >= 0 && wq < trunk_sz && wr >= 0 && wr < trunk_sz {
                let local_idx = wq as usize * trunk_sz as usize + wr as usize;
                if cluster_policies[k][local_idx] > max_prob {
                    max_prob = cluster_policies[k][local_idx];
                }
            }
        }
        global_policy[mcts_idx] = max_prob;
    }

    // Pass slot — unreachable in HTTT (no pass move). Constant 0.0 makes
    // dead-ness explicit; prior [0] read looked like a boundary K-pick.
    // §P2: skip this write under has_pass_slot=false (v8 family) — the tail
    // cell is a real legal index, not a pass slot.
    if has_pass_slot {
        global_policy[n_actions - 1] = 0.0;
    }

    let sum: f32 = global_policy.iter().sum();
    if sum > 1e-9 {
        for p in &mut global_policy { *p /= sum; }
    } else {
        let uniform = 1.0 / n_actions as f32;
        global_policy.fill(uniform);
    }
    global_policy
}

/// Project a global policy into the cluster-local window centred on `center`.
///
/// §173 A5b: encoding geometry is passed as pre-extracted integers so the
/// caller's hot loop avoids copying the full `RegistrySpec` struct (~174 B)
/// on every call. Caller derives these once from spec before entering the
/// per-move loop:
///   - `n_actions`     = `spec.policy_stride()` (362 for v6/v7full, 626 for v6w25, 625 for v8)
///   - `has_pass_slot` = `spec.has_pass_slot` (true for v6/v6w25/v7full; false for v8/v8_canvas_realness)
///   - `trunk_sz`      = `spec.trunk_size as i32` (19 for v6, 25 for v6w25/v8)
///
/// §P2: `has_pass_slot` gates the pass-slot copy at the tail of the policy
/// array. Pre-P2, the copy ran unconditionally — under v8 it would clobber
/// `local_policy[624]` (a real legal cell) with the corresponding (also
/// corrupted by the sister bug in `aggregate_policy`) `global_policy[624]`.
///
/// Per-window local-frame projection UNCHANGED: for each legal move m, project
/// from the global MCTS index to local window coordinates via
/// `(q - cq + half, r - cr + half)`. When `has_pass_slot=true`, the pass move
/// is copied verbatim from global; when false, the legal-cell scatter loop
/// owns every index from 0 to n_actions-1.
///
/// Inverse of `aggregate_policy` for a single cluster: used at record time
/// to stash a cluster-local policy next to the row's 2-plane state snapshot
/// so that each row is self-consistent under later symmetry augmentation.
/// The result is renormalised, with a uniform fallback for the zero-mass case.
///
/// §P11 (Wave 4): the legal-moves slice is supplied by the caller (hoisted
/// once at the record-emission boundary in worker_loop.rs) so K cluster
/// scatters share one `board.legal_moves()` call instead of K separate
/// allocations.
#[inline]
pub fn aggregate_policy_to_local(
    n_actions: usize,
    has_pass_slot: bool,
    trunk_sz: i32,
    board: &Board,
    center: &(i32, i32),
    global_policy: &[f32],
    legal_moves: &[(i32, i32)],
) -> Vec<f32> {
    let (cq, cr) = *center;
    let half = (trunk_sz - 1) / 2;
    // §173 A8'': MCTS-frame index map must use spec-derived trunk_sz.
    // See aggregate_policy doc-comment.
    let (bcq, bcr) = board.window_center();

    let mut local_policy = vec![0.0; n_actions];

    for &(q, r) in legal_moves {
        let wq = q - cq + half;
        let wr = r - cr + half;
        if wq >= 0 && wq < trunk_sz && wr >= 0 && wr < trunk_sz {
            let local_idx = wq as usize * trunk_sz as usize + wr as usize;
            let mcts_idx = Board::window_flat_idx_at_geom(q, r, bcq, bcr, trunk_sz, half);
            if mcts_idx < global_policy.len() {
                local_policy[local_idx] = global_policy[mcts_idx];
            }
        }
    }

    // Pass move (the last element) copied from the global policy.
    // §P2: skip under has_pass_slot=false (v8 family) — the tail index is a
    // real legal cell already populated by the scatter loop above; copying
    // global_policy[n_actions-1] would silently overwrite it.
    if has_pass_slot && n_actions > 0 && global_policy.len() >= n_actions {
        local_policy[n_actions - 1] = global_policy[n_actions - 1];
    }

    let sum: f32 = local_policy.iter().sum();
    if sum > 1e-9 {
        for p in &mut local_policy { *p /= sum; }
    } else {
        let uniform = 1.0 / n_actions as f32;
        local_policy.fill(uniform);
    }
    local_policy
}

/// Sample a move from `legal_moves` proportional to the policy mass at each
/// move's global action index.
///
/// §173 A8'': accepts `trunk_sz` (spec-derived NN-input frame side length)
/// so the MCTS-frame index map matches the policy array's geometry under
/// v6w25 + future multi-window encodings. Caller pre-extracts trunk_sz
/// at the worker_loop boundary (`agg_trunk_sz as i32`).
///
/// Returns `None` when the total probability mass over the legal set is
/// ~zero (caller falls back to uniform choice). Uses a thread-local RNG.
pub(crate) fn sample_policy(
    policy: &[f32],
    legal_moves: &[(i32, i32)],
    board: &Board,
    trunk_sz: i32,
) -> Option<(i32, i32)> {
    let half = (trunk_sz - 1) / 2;
    let (bcq, bcr) = board.window_center();

    let mut probs = Vec::with_capacity(legal_moves.len());
    let mut sum = 0.0;
    for &(q, r) in legal_moves {
        let idx = Board::window_flat_idx_at_geom(q, r, bcq, bcr, trunk_sz, half);
        let p = if idx < policy.len() { policy[idx] } else { 0.0 };
        probs.push(p);
        sum += p;
    }

    if sum < 1e-9 {
        return None;
    }

    let mut rng = rng();
    let mut r: f32 = rng.random();
    r *= sum;

    let mut current = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        current += p;
        if r <= current {
            return Some(legal_moves[i]);
        }
    }
    Some(legal_moves[legal_moves.len() - 1])
}

/// Project the final board state and winning-line cell list into a single
/// per-row combined aux buffer, using `(cq, cr)` as the window centre.
///
/// The returned Vec has layout `[ownership u8 × n_cells || winning_line u8 × n_cells]`:
///   * ownership encoding: 0 = P2, 1 = empty, 2 = P1 (default 1 outside footprint)
///   * winning_line encoding: binary 0/1 mask over the same window
///
/// §173 A8'': `n_cells` is spec-derived (`spec.n_cells()` = trunk_size²).
/// Pre-A8'' this allocated `2 * TOTAL_CELLS = 722` bytes, hardcoding v6
/// geometry — under v6w25 the buffer was 722 B but `collect_data` sliced
/// `aux_u8[..625]` / `aux_u8[625..]` (only 97 B of winning-line, reshape
/// failed). Caller (worker_loop.rs) passes `n_cells` already pre-extracted
/// from the registry spec (§173 A5a `let n_cells = ...`).
///
/// Each row owns its own centre so that the aux targets align with the row's
/// state planes under later 12-fold hex augmentation in the replay buffer.
/// On draws, pass an empty `winning_cells` slice — the second half stays zero.
pub(crate) fn reproject_game_end_row(
    final_cells: &[((i32, i32), Cell)],
    winning_cells: &[(i32, i32)],
    cq: i32,
    cr: i32,
    n_cells: usize,
) -> Vec<u8> {
    // §173 A8'': derive trunk_sz from n_cells (= trunk_sz²). Both u8 (n_cells
    // ≤ 625 for current encodings, fits easily in i32). Per-game-end call
    // (not per-sim) — no perf concern, no #[inline] needed.
    let trunk_sz = (n_cells as f64).sqrt() as i32;
    debug_assert_eq!(
        (trunk_sz as usize) * (trunk_sz as usize), n_cells,
        "reproject_game_end_row: n_cells {n_cells} is not a perfect square — trunk_sz²"
    );
    let half = (trunk_sz - 1) / 2;

    let mut aux_u8 = vec![0u8; 2 * n_cells];
    aux_u8[..n_cells].fill(1);
    for &((q, r), cell) in final_cells {
        let flat = Board::window_flat_idx_at_geom(q, r, cq, cr, trunk_sz, half);
        if flat < n_cells {
            aux_u8[flat] = match cell {
                Cell::P1    => 2,
                Cell::P2    => 0,
                Cell::Empty => 1,
            };
        }
    }
    for &(q, r) in winning_cells {
        let flat = Board::window_flat_idx_at_geom(q, r, cq, cr, trunk_sz, half);
        if flat < n_cells {
            aux_u8[n_cells + flat] = 1;
        }
    }
    aux_u8
}

/// O1: blend a forced-win one-hot into a policy TARGET in place.
///
/// `weight == 1.0` → pure one-hot on `action` (the SootyOwl-validated default);
/// `0 < weight < 1` → convex blend toward it (`target = (1-w)·target + w·e_action`).
/// `target` is assumed a probability vector (sums to ~1); the result remains
/// one (`(1-w)·1 + w = 1`). No-op when `weight <= 0` or `action` is out of range,
/// so a disabled O1 (or an out-of-window forced-win cell) leaves the
/// improved-policy target byte-identical.
pub(crate) fn apply_forced_win_one_hot(target: &mut [f32], action: usize, weight: f32) {
    if weight <= 0.0 || action >= target.len() {
        return;
    }
    let w = weight.min(1.0);
    for p in target.iter_mut() {
        *p *= 1.0 - w;
    }
    target[action] += w;
}

// ===========================================================================
// §D-MULTICLUSTER-S0 — ragged legal-set policy (Rust-internal global
// intermediate). See docs/designs/dmulticluster_362_legalset_design.md §9.
// These functions are the legal_set (no-drop) counterparts of aggregate_policy
// / aggregate_policy_to_local / sample_policy / apply_forced_win_one_hot. They
// are selected (vs the dense scatter_max path) when
// `spec.policy_pool == LegalSetScatterMax`. The dense path is byte-identical
// for in-global-window cells (A/B byte-identity, §9.9.4); only off-global-window
// cells COVERED by some cluster are additionally retained (in `overflow`).
// ===========================================================================

/// Ragged legal-set policy: the in-global-window slots in `dense` (keyed by
/// `window_flat_idx`, fast array path), plus off-global-window cells COVERED by
/// some cluster in `overflow` (keyed by board coord). Never crosses PyO3 / the
/// buffer; re-projected per-cluster into dense-362 rows by
/// `aggregate_policy_to_local_ls`.
#[derive(Clone, Debug, Default)]
pub struct LegalSetPolicy {
    pub dense: Vec<f32>,
    pub overflow: FxHashMap<(i32, i32), f32>,
}

impl LegalSetPolicy {
    /// Read the prior/target mass for board coord `(q, r)`. In-global-window
    /// cells read `dense` (identical to the dense path); a covered off-window
    /// cell reads `overflow`; a cell outside ALL coverage (absent, off-window)
    /// reads `floor` (the no-coverage prior). `(bcq, bcr)` is the global window
    /// centre, `trunk_sz`/`half` the spec-derived geometry.
    #[inline]
    pub fn get(&self, q: i32, r: i32, bcq: i32, bcr: i32, trunk_sz: i32, half: i32, floor: f32) -> f32 {
        let flat = Board::window_flat_idx_at_geom(q, r, bcq, bcr, trunk_sz, half);
        if flat < self.dense.len() {
            self.dense[flat]
        } else {
            self.overflow.get(&(q, r)).copied().unwrap_or(floor)
        }
    }
}

/// §9.2a coverage predicate: is `(q, r)` inside ≥1 cluster window? Byte-identical
/// to `aggregate_policy`'s inner bound test and `aggregate_policy_to_local`'s
/// projection test (same `wq = q - cq + half ∈ [0, trunk_sz)`). The target/O1
/// producers (policy.rs / inner.rs) use this to scope the ragged set to the
/// union-of-cluster-windows ∩ legal, so no uncovered key leaks into `overflow`.
#[inline]
pub(crate) fn is_covered(q: i32, r: i32, centers: &[(i32, i32)], trunk_sz: i32, half: i32) -> bool {
    centers.iter().any(|&(cq, cr)| {
        let wq = q - cq + half;
        let wr = r - cr + half;
        wq >= 0 && wq < trunk_sz && wr >= 0 && wr < trunk_sz
    })
}

/// Legal-set counterpart of `aggregate_policy`: builds the ragged global MCTS
/// prior. The K-cluster scatter-max inner loop is FROZEN; the only change is the
/// OUTPUT keying — in-window cells write `dense[mcts_idx]` (byte-identical to the
/// dense path), off-window cells COVERED by a cluster write `overflow[(q,r)]`,
/// off-window NO-COVERAGE cells are dropped (read `no_coverage_floor` later).
/// Joint renorm over dense + overflow.
#[inline]
pub fn aggregate_policy_ls(
    n_actions: usize,
    has_pass_slot: bool,
    trunk_sz: i32,
    board: &Board,
    centers: &[(i32, i32)],
    cluster_policies: &[Vec<f32>],
) -> LegalSetPolicy {
    let half = (trunk_sz - 1) / 2;
    let (bcq, bcr) = board.window_center();

    let mut dense = vec![0.0; n_actions];
    let mut overflow: FxHashMap<(i32, i32), f32> = FxHashMap::default();
    let legal = board.legal_moves();

    for &(q, r) in &legal {
        let mcts_idx = Board::window_flat_idx_at_geom(q, r, bcq, bcr, trunk_sz, half);

        let mut max_prob = 0.0;
        let mut covered = false;
        for (k, &(cq, cr)) in centers.iter().enumerate() {
            let wq = q - cq + half;
            let wr = r - cr + half;
            if wq >= 0 && wq < trunk_sz && wr >= 0 && wr < trunk_sz {
                covered = true;
                let local_idx = wq as usize * trunk_sz as usize + wr as usize;
                if cluster_policies[k][local_idx] > max_prob {
                    max_prob = cluster_policies[k][local_idx];
                }
            }
        }

        if has_pass_slot && mcts_idx >= n_actions - 1 {
            // Off-global-window (mcts_idx == usize::MAX). RAGGED FIX: retain it in
            // overflow IFF a cluster covers it (§9.2a); a no-coverage cell is
            // dropped (it has no NN signal — reads no_coverage_floor at use).
            if covered {
                overflow.insert((q, r), max_prob);
            }
            continue;
        }
        dense[mcts_idx] = max_prob;
    }

    // Pass slot — dead-constant 0.0 (no pass move in HTTT), matching the dense path.
    if has_pass_slot {
        dense[n_actions - 1] = 0.0;
    }

    let sum: f32 = dense.iter().sum::<f32>() + overflow.values().sum::<f32>();
    if sum > 1e-9 {
        for p in &mut dense {
            *p /= sum;
        }
        for v in overflow.values_mut() {
            *v /= sum;
        }
    } else {
        // Zero-mass fallback: uniform over the dense window (matches the dense
        // path's uniform fallback); overflow stays empty.
        let uniform = 1.0 / n_actions as f32;
        dense.fill(uniform);
        overflow.clear();
    }

    LegalSetPolicy { dense, overflow }
}

/// Legal-set counterpart of `aggregate_policy_to_local`: projects the ragged
/// global `ls` into the dense-362 frame of the cluster centred on `center`. For
/// each legal move in this cluster's window it reads `ls` BY COORD — in-global-
/// window cells from `ls.dense`, covered off-window cells from `ls.overflow` —
/// so an off-GLOBAL-window cell covered by THIS cluster lands at its `local_idx`
/// (the whole fix). Each recorded row stays dense-362 (buffer UNCHANGED).
#[inline]
pub fn aggregate_policy_to_local_ls(
    n_actions: usize,
    has_pass_slot: bool,
    trunk_sz: i32,
    board: &Board,
    center: &(i32, i32),
    ls: &LegalSetPolicy,
    legal_moves: &[(i32, i32)],
) -> Vec<f32> {
    let (cq, cr) = *center;
    let half = (trunk_sz - 1) / 2;
    let (bcq, bcr) = board.window_center();

    let mut local_policy = vec![0.0; n_actions];

    for &(q, r) in legal_moves {
        let wq = q - cq + half;
        let wr = r - cr + half;
        if wq >= 0 && wq < trunk_sz && wr >= 0 && wr < trunk_sz {
            let local_idx = wq as usize * trunk_sz as usize + wr as usize;
            let mcts_idx = Board::window_flat_idx_at_geom(q, r, bcq, bcr, trunk_sz, half);
            // Read the ragged global by coord. A no-coverage target cell reads
            // 0.0 (not supervised) — distinct from the prior's floor.
            let val = if mcts_idx < ls.dense.len() {
                ls.dense[mcts_idx]
            } else {
                ls.overflow.get(&(q, r)).copied().unwrap_or(0.0)
            };
            local_policy[local_idx] = val;
        }
    }

    if has_pass_slot && n_actions > 0 && ls.dense.len() >= n_actions {
        local_policy[n_actions - 1] = ls.dense[n_actions - 1];
    }

    let sum: f32 = local_policy.iter().sum();
    if sum > 1e-9 {
        for p in &mut local_policy {
            *p /= sum;
        }
    } else {
        let uniform = 1.0 / n_actions as f32;
        local_policy.fill(uniform);
    }
    local_policy
}

/// Legal-set counterpart of `sample_policy`: samples a move from `legal_moves`
/// proportional to the ragged `ls` mass at each move's coord (off-window covered
/// moves are now sampleable). `floor` is the no-coverage prior.
pub(crate) fn sample_policy_ls(
    ls: &LegalSetPolicy,
    legal_moves: &[(i32, i32)],
    board: &Board,
    trunk_sz: i32,
    floor: f32,
) -> Option<(i32, i32)> {
    let half = (trunk_sz - 1) / 2;
    let (bcq, bcr) = board.window_center();

    let mut probs = Vec::with_capacity(legal_moves.len());
    let mut sum = 0.0;
    for &(q, r) in legal_moves {
        let p = ls.get(q, r, bcq, bcr, trunk_sz, half, floor);
        probs.push(p);
        sum += p;
    }

    if sum < 1e-9 {
        return None;
    }

    let mut rng = rng();
    let mut rv: f32 = rng.random();
    rv *= sum;

    let mut current = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        current += p;
        if rv <= current {
            return Some(legal_moves[i]);
        }
    }
    Some(legal_moves[legal_moves.len() - 1])
}

/// Legal-set counterpart of `apply_forced_win_one_hot` (§9.2a / §9.3). Applies
/// the forced-win one-hot to the ragged target BY COORD, but ONLY when the win
/// cell is covered by a cluster — an uncovered win would zero all global mass and
/// every per-cluster projection would hit the uniform fallback (the corruption
/// the Phase-1 review caught). `covered=false` → no-op (matching today's clean
/// off-window drop). Returns whether the one-hot fired (the caller's
/// `forced_win_fired`).
pub(crate) fn apply_forced_win_one_hot_ls(
    ls: &mut LegalSetPolicy,
    win: (i32, i32),
    weight: f32,
    covered: bool,
    bcq: i32,
    bcr: i32,
    trunk_sz: i32,
    half: i32,
) -> bool {
    if weight <= 0.0 || !covered {
        return false;
    }
    let w = weight.min(1.0);
    for p in ls.dense.iter_mut() {
        *p *= 1.0 - w;
    }
    for v in ls.overflow.values_mut() {
        *v *= 1.0 - w;
    }
    let flat = Board::window_flat_idx_at_geom(win.0, win.1, bcq, bcr, trunk_sz, half);
    if flat < ls.dense.len() {
        ls.dense[flat] += w;
    } else {
        // Covered off-window win: insert-if-absent then add the boost.
        *ls.overflow.entry(win).or_insert(0.0) += w;
    }
    true
}

#[cfg(test)]
mod o1_tests {
    use super::*;
    use crate::board::Player;

    #[test]
    fn test_apply_forced_win_one_hot_pure() {
        let mut target = vec![0.25_f32; 4];
        apply_forced_win_one_hot(&mut target, 2, 1.0);
        assert_eq!(target, vec![0.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_apply_forced_win_one_hot_blend_stays_distribution() {
        let mut target = vec![0.25_f32; 4];
        apply_forced_win_one_hot(&mut target, 2, 0.5);
        let sum: f32 = target.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "blend must stay a distribution, sum={sum}");
        assert!((target[2] - 0.625).abs() < 1e-6, "winning action boosted, got {}", target[2]);
        assert!(target[2] > target[0], "winning action must dominate");
    }

    #[test]
    fn test_apply_forced_win_one_hot_noop_when_disabled() {
        let mut target = vec![0.25_f32; 4];
        let before = target.clone();
        apply_forced_win_one_hot(&mut target, 2, 0.0);   // weight 0 ⇒ O1 off
        assert_eq!(target, before);
        apply_forced_win_one_hot(&mut target, 99, 1.0);  // out-of-range action
        assert_eq!(target, before);
    }

    #[test]
    fn test_one_hot_survives_aggregate_to_local() {
        // O1 sets the one-hot on the winning move's GLOBAL window index
        // (`Board::window_flat_idx`); `aggregate_policy_to_local` must carry it to
        // the local record as a one-hot. Pins the index-mapping consistency that
        // keeps the forced-win target from being silently dropped into the buffer.
        let mut board = Board::new();
        for q in 0..5i32 { board.cells.insert((q, 0), Cell::P1); }
        board.has_stones = true;
        board.min_q = -1; board.max_q = 5; board.min_r = 0; board.max_r = 0;
        board.cache_dirty.set(true);
        board.current_player = Player::One;
        board.moves_remaining = 2;

        let win = board.first_winning_move(Player::One).expect("winning move exists");
        let stride = 19 * 19 + 1;
        let action = board.window_flat_idx(win.0, win.1);
        assert!(action < stride, "winning move maps in-window");
        let mut global = vec![0.0_f32; stride];
        apply_forced_win_one_hot(&mut global, action, 1.0);

        let trunk = board.cluster_window_size() as i32;
        let center = board.window_center();
        let legal = board.legal_moves();
        let local =
            aggregate_policy_to_local(stride, true, trunk, &board, &center, &global, &legal);

        let nonzero: Vec<f32> = local.iter().copied().filter(|&p| p > 1e-6).collect();
        assert_eq!(nonzero.len(), 1, "aggregate must preserve a one-hot, got {} non-zero", nonzero.len());
        assert!((nonzero[0] - 1.0).abs() < 1e-6, "surviving mass ~1.0, got {}", nonzero[0]);
    }
}

#[cfg(test)]
mod ls_tests {
    //! §D-MULTICLUSTER-S0 ragged legal-set gate tests (design §9.9).
    use super::*;
    use crate::board::Player;

    /// Build a spread board with TWO far-apart stone clusters so the global
    /// window centre sits between them and a cell near cluster-2 is OFF the
    /// global 19×19 window. Returns the board; centres are supplied manually by
    /// the caller to control geometry exactly.
    fn spread_board() -> Board {
        let mut board = Board::new();
        for q in 0..5i32 {
            board.cells.insert((q, 0), Cell::P1);
        }
        for q in 30..35i32 {
            board.cells.insert((q, 0), Cell::P2);
        }
        board.has_stones = true;
        board.min_q = 0;
        board.max_q = 34;
        board.min_r = 0;
        board.max_r = 0;
        board.cache_dirty.set(true);
        board.current_player = Player::One;
        board.moves_remaining = 2;
        board
    }

    const TRUNK: i32 = 19;
    const HALF: i32 = 9;
    const NA: usize = 19 * 19 + 1; // 362

    #[test]
    fn test_ls_retains_off_window_covered_cell_round_trip() {
        // §9.9.1 — the P3-killer. Global centre = bbox midpoint (17,0); the global
        // window covers q∈[8,26]. (28,0) is OFF that window but COVERED by
        // cluster-2 (centre (32,0): q∈[23,41]). It must land in overflow and
        // project into cluster-2's local-362 — no usize::MAX index.
        let board = spread_board();
        assert_eq!(board.window_center(), (17, 0), "global centre between clusters");
        let legal = board.legal_moves();
        assert!(legal.contains(&(28, 0)), "(28,0) must be a legal move near cluster-2");

        let centers = vec![(2, 0), (32, 0)];
        // cluster-2's NN prob at (28,0)'s local index: wq=28-32+9=5, wr=9 → 104.
        let local_28 = (5usize) * TRUNK as usize + 9;
        let mut cp0 = vec![0.0f32; NA];
        let mut cp1 = vec![0.0f32; NA];
        cp1[local_28] = 1.0;
        // give cluster-1 some in-window mass so dense isn't all-zero
        let _ = &mut cp0;
        let cluster_policies = vec![cp0, cp1];

        let ls = aggregate_policy_ls(NA, true, TRUNK, &board, &centers, &cluster_policies);
        assert!(
            ls.overflow.contains_key(&(28, 0)),
            "off-window covered cell retained in overflow; got keys {:?}",
            ls.overflow.keys().collect::<Vec<_>>()
        );
        assert!((ls.overflow[&(28, 0)] - 1.0).abs() < 1e-6, "all mass on the one covered cell");

        // Project into cluster-2's local-362: (28,0) must land non-zero at local 104.
        let local = aggregate_policy_to_local_ls(NA, true, TRUNK, &board, &(32, 0), &ls, &legal);
        assert!((local[local_28] - 1.0).abs() < 1e-6, "off-window cell projected into covering cluster's local-362, got {}", local[local_28]);
    }

    #[test]
    fn test_ls_coverage_enforced_no_uncovered_overflow() {
        // §9.9.3 — every overflow key is covered by ≥1 cluster (true by the
        // aggregate_policy_ls construction: it only inserts when `covered`).
        let board = spread_board();
        let centers = vec![(2, 0), (32, 0)];
        let cp = vec![vec![0.1f32; NA], vec![0.1f32; NA]];
        let ls = aggregate_policy_ls(NA, true, TRUNK, &board, &centers, &cp);
        for &(q, r) in ls.overflow.keys() {
            assert!(
                is_covered(q, r, &centers, TRUNK, HALF),
                "overflow key ({q},{r}) must be covered by some cluster"
            );
        }
    }

    #[test]
    fn test_ls_o1_covered_win_fires_uncovered_noops() {
        // §9.9.2 + §9.9.6 — O1 on a COVERED off-window win fires + survives
        // projection; O1 on an UNCOVERED cell no-ops (no mass leak / no uniform-
        // fallback corruption — the Phase-1 review's exact counterexample).
        let board = spread_board();
        let centers = vec![(2, 0), (32, 0)];
        let mut cp = vec![vec![0.0f32; NA], vec![0.0f32; NA]];
        cp[1][(5usize) * TRUNK as usize + 9] = 0.4; // (28,0) covered, some prior
        let base = aggregate_policy_ls(NA, true, TRUNK, &board, &centers, &cp);

        // COVERED win (28,0): fires.
        let mut ls = base.clone();
        let covered = is_covered(28, 0, &centers, TRUNK, HALF);
        assert!(covered);
        let fired = apply_forced_win_one_hot_ls(&mut ls, (28, 0), 1.0, covered, 17, 0, TRUNK, HALF);
        assert!(fired, "covered off-window win must fire");
        assert!((ls.overflow[&(28, 0)] - 1.0).abs() < 1e-6, "pure one-hot on the covered win");
        let legal = board.legal_moves();
        let local = aggregate_policy_to_local_ls(NA, true, TRUNK, &board, &(32, 0), &ls, &legal);
        let local_28 = (5usize) * TRUNK as usize + 9;
        assert!(local[local_28] > 0.99, "covered win one-hot survives projection, got {}", local[local_28]);

        // UNCOVERED cell (60,0): no cluster window contains it → no-op.
        let mut ls2 = base.clone();
        let uncovered = is_covered(60, 0, &centers, TRUNK, HALF);
        assert!(!uncovered, "(60,0) is outside all cluster windows");
        let before = ls2.clone();
        let fired2 = apply_forced_win_one_hot_ls(&mut ls2, (60, 0), 1.0, uncovered, 17, 0, TRUNK, HALF);
        assert!(!fired2, "uncovered win must NO-OP (no leak)");
        assert_eq!(before.dense, ls2.dense, "uncovered O1 must not touch the target");
        assert!(!ls2.overflow.contains_key(&(60, 0)), "uncovered key never inserted");
    }

    #[test]
    fn test_ls_ab_byte_identity_when_no_off_window() {
        // §9.9.4 — with NO off-window legal cell (single cluster at the global
        // centre, all legal moves in-window), the legal_set dense row is
        // byte-identical to the dense scatter_max aggregate → clean A/B (the
        // change is inert when there is nothing off-window).
        let mut board = Board::new();
        for q in 0..5i32 {
            board.cells.insert((q, 0), Cell::P1);
        }
        board.has_stones = true;
        board.min_q = -1;
        board.max_q = 5;
        board.min_r = 0;
        board.max_r = 0;
        board.cache_dirty.set(true);
        board.current_player = Player::One;
        board.moves_remaining = 2;
        let center = board.window_center();
        let centers = vec![center];
        let mut cp = vec![0.0f32; NA];
        // put mass on a couple of in-window cells via cluster-local indices
        cp[10] = 0.3;
        cp[200] = 0.7;
        let cluster_policies = vec![cp];

        let dense = aggregate_policy(NA, true, TRUNK, &board, &centers, &cluster_policies);
        let ls = aggregate_policy_ls(NA, true, TRUNK, &board, &centers, &cluster_policies);
        assert!(ls.overflow.is_empty(), "single centre at global centre → no off-window cells");
        assert_eq!(dense, ls.dense, "legal_set dense row byte-identical to scatter_max when no off-window cell");
    }

    #[test]
    fn test_ls_sample_picks_off_window_mass() {
        // §9.9.5 — self-play sampling over the ragged set can pick an off-window
        // move (the dense sample_policy returns p=0.0 for it). All mass on the
        // covered off-window (28,0), floor=0.0 → deterministic pick.
        let board = spread_board();
        let centers = vec![(2, 0), (32, 0)];
        let mut cp = vec![vec![0.0f32; NA], vec![0.0f32; NA]];
        cp[1][(5usize) * TRUNK as usize + 9] = 1.0; // (28,0)
        let ls = aggregate_policy_ls(NA, true, TRUNK, &board, &centers, &cp);
        let legal = board.legal_moves();
        let picked = sample_policy_ls(&ls, &legal, &board, TRUNK, 0.0);
        assert_eq!(picked, Some((28, 0)), "sampler picks the only off-window mass");
    }
}
