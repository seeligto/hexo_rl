//! Regression test: top-K leaf cap eliminates pool overflow at any board state.
//!
//! Background — `engine/src/mcts/backup.rs::expand_and_backup_single` previously
//! created one child per legal move. On a sparse board with 100+ stones the
//! legal-move set can balloon past 1k cells (radius-8 hex ball per stone), so
//! `n_simulations × leaf_batch × n_legal` overflowed `MAX_NODES`. The earlier
//! mitigation marked the leaf terminal with a fabricated value and silently
//! corrupted training targets.
//!
//! After the §-top-K change, leaf expansion creates at most
//! `MAX_CHILDREN_PER_NODE` children regardless of `legal_moves.len()`, so
//! `pool_overflow_count()` must stay at zero across a normal-sized pool, and
//! no node in the tree may exceed K children.
//!
//! These tests drive a pure-Rust self-play loop with uniform priors (no NN
//! server required), advancing 200 plies with `n_simulations=400`,
//! `leaf_batch=8` per move — well past the regime where the old code blew up.

use engine::board::{Board, BOARD_SIZE};
use engine::mcts::{MCTSTree, MAX_CHILDREN_PER_NODE, pool_overflow_count, take_pool_overflow_count};

const N_SIMS_PER_MOVE: usize = 400;
const LEAF_BATCH: usize = 8;
const TARGET_PLIES: u32 = 200;

/// Run `n_sims` MCTS sims on `tree` with uniform priors. No NN dependency —
/// every leaf gets the same flat policy and value=0.0.
fn run_uniform_search(tree: &mut MCTSTree, n_sims: usize, leaf_batch: usize) {
    let n_actions = BOARD_SIZE * BOARD_SIZE + 1;
    let uniform_prior = 1.0_f32 / n_actions as f32;
    let uniform_policy = vec![uniform_prior; n_actions];

    let mut completed = 0;
    while completed < n_sims {
        let take = leaf_batch.min(n_sims - completed);
        let boards = tree.select_leaves(take);
        if boards.is_empty() {
            break;
        }
        let policies: Vec<Vec<f32>> = (0..boards.len()).map(|_| uniform_policy.clone()).collect();
        let values: Vec<f32> = vec![0.0; boards.len()];
        tree.expand_and_backup(&policies, &values);
        completed += boards.len();
    }
}

/// Pick the most-visited root child and return its (q, r) action.
fn argmax_visit_action(tree: &MCTSTree) -> Option<(i32, i32)> {
    let root = &tree.pool[0];
    if !root.is_expanded() {
        return None;
    }
    let first = root.first_child as usize;
    let n = root.n_children as usize;
    let best = (first..first + n).max_by_key(|&i| tree.pool[i].n_visits)?;
    let val = tree.pool[best].action_idx;
    let q = (val >> 16) as i32 - 32768;
    let r = (val & 0xFFFF) as i32 - 32768;
    Some((q, r))
}

#[test]
fn topk_eliminates_pool_overflow_across_full_game() {
    let _ = take_pool_overflow_count();
    let before = pool_overflow_count();

    let mut board = Board::new();
    let mut tree = MCTSTree::new(1.5);
    let mut max_children_seen: u16 = 0;

    for ply in 0..TARGET_PLIES {
        if board.check_win() || board.legal_move_count() == 0 {
            break;
        }
        tree.new_game(board.clone());
        run_uniform_search(&mut tree, N_SIMS_PER_MOVE, LEAF_BATCH);

        for i in 0..tree.next_free_slot() as usize {
            let n_ch = tree.pool[i].n_children;
            if n_ch > max_children_seen {
                max_children_seen = n_ch;
            }
            assert!(n_ch as usize <= MAX_CHILDREN_PER_NODE,
                "ply {ply}: node {i} has {n_ch} children, exceeds K={MAX_CHILDREN_PER_NODE}");
        }

        let action = match argmax_visit_action(&tree) {
            Some(a) => a,
            None => break,
        };
        if board.apply_move(action.0, action.1).is_err() {
            break;
        }
    }

    let after = pool_overflow_count();
    assert_eq!(after, before,
        "pool overflow must remain zero with top-K cap (delta={})", after - before);
    assert!(max_children_seen as usize <= MAX_CHILDREN_PER_NODE,
        "max children observed = {max_children_seen}, K={MAX_CHILDREN_PER_NODE}");
}

#[test]
fn normal_sized_pool_does_not_overflow_on_empty_root() {
    let mut tree = MCTSTree::new(1.5);
    tree.new_game(Board::new());

    let n_actions = BOARD_SIZE * BOARD_SIZE + 1;
    let uniform = vec![1.0_f32 / n_actions as f32; n_actions];

    let _leaves = tree.select_leaves(1);
    tree.expand_and_backup(&[uniform], &[0.0]);

    let root = &tree.pool[0];
    assert!(!root.is_terminal, "default pool must expand root, not mark terminal");
    assert!(root.n_children > 0, "root must have children after first expansion");
    assert!(root.n_children as usize <= MAX_CHILDREN_PER_NODE);
}
