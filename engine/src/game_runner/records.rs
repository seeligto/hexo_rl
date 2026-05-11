//! Per-game record helpers for the self-play worker loop.
//!
//! Contains policy aggregation between cluster-local and global action
//! frames, visit-count policy sampling, and the game-end reprojection of
//! final ownership + winning-line targets into each row's per-cluster
//! window centre. All functions are pure (no worker state) and free so the
//! worker loop can call them without holding `self`.

use crate::board::{Board, Cell, TOTAL_CELLS};
use rand::{rng, RngExt};

/// Fuse K cluster-local policies into one global policy in the main board's
/// MCTS action frame.
///
/// §173 A5b: encoding geometry is passed as pre-extracted integers so the
/// caller's hot loop avoids copying the full `RegistrySpec` struct (~174 B)
/// on every call. Caller derives these once from spec before entering the
/// per-sim loop:
///   - `n_actions` = `spec.policy_stride()` (362 for v6/v7full, 626 for v6w25)
///   - `trunk_sz`  = `spec.trunk_size as i32` (19 for v6, 25 for v6w25)
///
/// Scatter-max semantics UNCHANGED (α design §3.1): for each legal move m,
/// locate all clusters k that cover m, take max probability across them,
/// assign to `global_policy[mcts_idx(m)]`.
///
/// For each legal move, takes the max probability across clusters that cover
/// it. The pass move (index n_actions - 1) is set to 0.0. The result is
/// renormalised to sum to 1; if every entry is ~0 we fall back to a uniform
/// distribution to avoid division by zero downstream.
#[inline]
pub fn aggregate_policy(
    n_actions: usize,
    trunk_sz: i32,
    board: &Board,
    centers: &[(i32, i32)],
    cluster_policies: &[Vec<f32>],
) -> Vec<f32> {
    let half = (trunk_sz - 1) / 2;

    let mut global_policy = vec![0.0; n_actions];
    let legal = board.legal_moves();

    for &(q, r) in &legal {
        let mcts_idx = board.window_flat_idx(q, r);
        if mcts_idx >= n_actions - 1 { continue; }

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
    global_policy[n_actions - 1] = 0.0;

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
///   - `n_actions` = `spec.policy_stride()` (362 for v6/v7full, 626 for v6w25)
///   - `trunk_sz`  = `spec.trunk_size as i32` (19 for v6, 25 for v6w25)
///
/// Per-window local-frame projection UNCHANGED: for each legal move m, project
/// from the global MCTS index to local window coordinates via
/// `(q - cq + half, r - cr + half)`. Pass move copied verbatim from global.
///
/// Inverse of `aggregate_policy` for a single cluster: used at record time
/// to stash a cluster-local policy next to the row's 2-plane state snapshot
/// so that each row is self-consistent under later symmetry augmentation.
/// The pass move is copied verbatim from the global policy; the result is
/// renormalised, with a uniform fallback for the zero-mass case.
#[inline]
pub fn aggregate_policy_to_local(
    n_actions: usize,
    trunk_sz: i32,
    board: &Board,
    center: &(i32, i32),
    global_policy: &[f32],
) -> Vec<f32> {
    let (cq, cr) = *center;
    let half = (trunk_sz - 1) / 2;

    let mut local_policy = vec![0.0; n_actions];
    let legal = board.legal_moves();

    for &(q, r) in &legal {
        let wq = q - cq + half;
        let wr = r - cr + half;
        if wq >= 0 && wq < trunk_sz && wr >= 0 && wr < trunk_sz {
            let local_idx = wq as usize * trunk_sz as usize + wr as usize;
            let mcts_idx = board.window_flat_idx(q, r);
            if mcts_idx < global_policy.len() {
                local_policy[local_idx] = global_policy[mcts_idx];
            }
        }
    }

    // Pass move (the last element) is always copied from the global policy
    if n_actions > 0 && global_policy.len() >= n_actions {
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
/// Returns `None` when the total probability mass over the legal set is
/// ~zero (caller falls back to uniform choice). Uses a thread-local RNG.
pub(crate) fn sample_policy(
    policy: &[f32],
    legal_moves: &[(i32, i32)],
    board: &Board,
) -> Option<(i32, i32)> {
    let mut probs = Vec::with_capacity(legal_moves.len());
    let mut sum = 0.0;
    for &(q, r) in legal_moves {
        let idx = board.window_flat_idx(q, r);
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
/// The returned Vec has layout `[ownership u8 × TOTAL_CELLS || winning_line u8 × TOTAL_CELLS]`:
///   * ownership encoding: 0 = P2, 1 = empty, 2 = P1 (default 1 outside footprint)
///   * winning_line encoding: binary 0/1 mask over the same window
///
/// Each row owns its own centre so that the aux targets align with the row's
/// state planes under later 12-fold hex augmentation in the replay buffer.
/// On draws, pass an empty `winning_cells` slice — the second half stays zero.
pub(crate) fn reproject_game_end_row(
    final_cells: &[((i32, i32), Cell)],
    winning_cells: &[(i32, i32)],
    cq: i32,
    cr: i32,
) -> Vec<u8> {
    let mut aux_u8 = vec![0u8; 2 * TOTAL_CELLS];
    aux_u8[..TOTAL_CELLS].fill(1);
    for &((q, r), cell) in final_cells {
        let flat = Board::window_flat_idx_at(q, r, cq, cr);
        if flat < TOTAL_CELLS {
            aux_u8[flat] = match cell {
                Cell::P1    => 2,
                Cell::P2    => 0,
                Cell::Empty => 1,
            };
        }
    }
    for &(q, r) in winning_cells {
        let flat = Board::window_flat_idx_at(q, r, cq, cr);
        if flat < TOTAL_CELLS {
            aux_u8[TOTAL_CELLS + flat] = 1;
        }
    }
    aux_u8
}
