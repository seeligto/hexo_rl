//! Per-game record helpers for the self-play worker loop.
//!
//! Contains policy aggregation between cluster-local and global action
//! frames, visit-count policy sampling, and the game-end reprojection of
//! final ownership + winning-line targets into each row's per-cluster
//! window centre. All functions are pure (no worker state) and free so the
//! worker loop can call them without holding `self`.

use crate::board::{Board, Cell, BOARD_SIZE, HALF, TOTAL_CELLS};
use rand::{rng, RngExt};

/// §153 T3 — per-move / per-turn rotation cadence helper. Returns true when
/// the worker loop should resample `sym_idx` BEFORE running MCTS / recording
/// for the move at the given `ply` and `moves_remaining`.
///
/// The initial sample at game start is the caller's responsibility — this
/// helper only governs WITHIN-game resampling.
///
/// Cadence codes:
///   * 0 (per_game): never resample within a game.
///   * 1 (per_move): resample for every recorded move.
///   * 2 (per_turn): resample on turn boundaries — i.e. when this is NOT
///     the opening single move (`ply >= 1`) and the player has just started
///     a new compound turn (`moves_remaining == 2`).
pub(crate) fn should_resample_sym(rotation_cadence: u8, ply: u32, moves_remaining: u8) -> bool {
    match rotation_cadence {
        1 => true,
        2 => ply >= 1 && moves_remaining == 2,
        _ => false,
    }
}

/// Fuse K cluster-local policies into one global policy in the main board's
/// MCTS action frame.
///
/// For each legal move, takes the max probability across clusters that cover
/// it. The pass move (index n_actions - 1) is copied from cluster 0. The
/// result is renormalised to sum to 1; if every entry is ~0 we fall back to
/// a uniform distribution to avoid division by zero downstream.
pub(crate) fn aggregate_policy(
    board: &Board,
    centers: &[(i32, i32)],
    cluster_policies: &[Vec<f32>],
) -> Vec<f32> {
    let n_actions = BOARD_SIZE * BOARD_SIZE + 1;
    let mut global_policy = vec![0.0; n_actions];
    let legal = board.legal_moves();

    for &(q, r) in &legal {
        let mcts_idx = board.window_flat_idx(q, r);
        if mcts_idx >= n_actions - 1 { continue; }

        let mut max_prob = 0.0;
        for (k, &(cq, cr)) in centers.iter().enumerate() {
            let wq = q - cq + HALF;
            let wr = r - cr + HALF;
            if wq >= 0 && wq < BOARD_SIZE as i32 && wr >= 0 && wr < BOARD_SIZE as i32 {
                let local_idx = wq as usize * BOARD_SIZE + wr as usize;
                if cluster_policies[k][local_idx] > max_prob {
                    max_prob = cluster_policies[k][local_idx];
                }
            }
        }
        global_policy[mcts_idx] = max_prob;
    }

    // Pass move is always copied from the first cluster (should be consistent)
    if !cluster_policies.is_empty() {
        global_policy[n_actions - 1] = cluster_policies[0][n_actions - 1];
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
/// Inverse of `aggregate_policy` for a single cluster: used at record time
/// to stash a cluster-local policy next to the row's 2-plane state snapshot
/// so that each row is self-consistent under later symmetry augmentation.
/// The pass move is copied verbatim from the global policy; the result is
/// renormalised, with a uniform fallback for the zero-mass case.
pub(crate) fn aggregate_policy_to_local(
    board: &Board,
    center: &(i32, i32),
    global_policy: &[f32],
) -> Vec<f32> {
    let (cq, cr) = *center;
    let n_actions = BOARD_SIZE * BOARD_SIZE + 1;
    let mut local_policy = vec![0.0; n_actions];
    let legal = board.legal_moves();

    for &(q, r) in &legal {
        let wq = q - cq + HALF;
        let wr = r - cr + HALF;
        if wq >= 0 && wq < BOARD_SIZE as i32 && wr >= 0 && wr < BOARD_SIZE as i32 {
            let local_idx = wq as usize * BOARD_SIZE + wr as usize;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn per_game_cadence_never_resamples_within_a_game() {
        // The initial sample is the caller's responsibility — should_resample_sym
        // governs only WITHIN-game resampling. per_game (code 0) must always
        // return false regardless of (ply, moves_remaining).
        for ply in 0u32..20 {
            for mr in [0u8, 1, 2] {
                assert!(!should_resample_sym(0, ply, mr),
                    "per_game must not resample at (ply={ply}, mr={mr})");
            }
        }
    }

    #[test]
    fn per_move_cadence_resamples_every_move() {
        for ply in 0u32..20 {
            for mr in [0u8, 1, 2] {
                assert!(should_resample_sym(1, ply, mr),
                    "per_move must resample at (ply={ply}, mr={mr})");
            }
        }
    }

    #[test]
    fn per_turn_cadence_resamples_only_on_turn_boundaries() {
        // Turn boundary semantics:
        //   * ply 0 (P1's first single move): caller's initial sample stands
        //     — should_resample_sym must return false.
        //   * ply 1 (P2's first compound move, mr == 2): boundary.
        //   * ply 2 (P2's second compound move, mr == 1): no boundary.
        //   * ply 3 (P1's first compound move, mr == 2): boundary.
        //   * ply 4 (P1's second compound move, mr == 1): no boundary.
        //   * ply 5 (P2's first compound move, mr == 2): boundary.
        let table: &[(u32, u8, bool)] = &[
            (0, 1, false),
            (1, 2, true),
            (2, 1, false),
            (3, 2, true),
            (4, 1, false),
            (5, 2, true),
        ];
        for &(ply, mr, expected) in table {
            assert_eq!(
                should_resample_sym(2, ply, mr),
                expected,
                "per_turn at (ply={ply}, mr={mr}) expected {expected}",
            );
        }
        // Defence-in-depth: at moves_remaining == 0 (mid-apply transient,
        // never observed in the worker loop because we read mr at start of
        // a move) the helper still returns false.
        assert!(!should_resample_sym(2, 5, 0));
    }

    #[test]
    fn unknown_cadence_codes_fall_back_to_per_game() {
        for code in [3u8, 4, 9, 255] {
            assert!(!should_resample_sym(code, 1, 2));
            assert!(!should_resample_sym(code, 0, 1));
        }
    }
}
