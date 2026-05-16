//! Per-game record helpers for the self-play worker loop.
//!
//! Contains policy aggregation between cluster-local and global action
//! frames, visit-count policy sampling, and the game-end reprojection of
//! final ownership + winning-line targets into each row's per-cluster
//! window centre. All functions are pure (no worker state) and free so the
//! worker loop can call them without holding `self`.

use crate::board::{Board, Cell};
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
        for p in &mut global_policy { *p = uniform; }
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
        for p in &mut local_policy { *p = uniform; }
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
        "reproject_game_end_row: n_cells {} is not a perfect square — trunk_sz²", n_cells
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
