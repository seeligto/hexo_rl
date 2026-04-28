/// Regression test for A-004: pool overflow must mark node terminal, not silently backup.
///
/// Before fix, the overflow branch in expand_and_backup_single just called
/// self.backup(leaf_idx, value) and returned — leaving the node unexpanded with
/// is_terminal=false, causing every subsequent sim to retry and fail again silently.
///
/// After fix, the node is marked terminal so subsequent sims detect is_terminal and
/// skip expansion entirely.

use engine::board::Board;
use engine::mcts::{MCTSTree, pool_overflow_count, take_pool_overflow_count};

/// Build a tiny-pool tree (root=slot0, only 10 free slots) so the first
/// expansion attempt overflows (empty board has 25 legal moves → 25 children needed).
fn tiny_tree() -> MCTSTree {
    // Pool size 10: root=slot0, next_free=1, capacity=9 extra slots < 25 legal moves.
    MCTSTree::new_with_capacity(1.5, 10)
}

#[test]
fn pool_overflow_marks_root_terminal() {
    let mut tree = tiny_tree();
    tree.new_game(Board::new());

    let n_actions = 19 * 19 + 1;
    let uniform = vec![1.0_f32 / n_actions as f32; n_actions];

    // First sim: select leaf (root), then try to expand.
    // Pool has 9 free slots, board has 25 legal moves → overflow.
    let _leaves = tree.select_leaves(1);
    tree.expand_and_backup(&[uniform.clone()], &[0.0]);

    // Root node must now be marked terminal.
    let root = &tree.pool[0];
    assert!(root.is_terminal,
        "pool overflow must mark node terminal so subsequent sims skip it");
    assert!(root.terminal_value >= -1.0 && root.terminal_value <= 1.0,
        "terminal_value must be valid (quiescence-corrected), got {}", root.terminal_value);
}

#[test]
fn pool_overflow_subsequent_sim_skips_expansion() {
    let mut tree = tiny_tree();
    tree.new_game(Board::new());

    let n_actions = 19 * 19 + 1;
    let uniform = vec![1.0_f32 / n_actions as f32; n_actions];

    // First sim: triggers overflow, marks root terminal.
    let _leaves = tree.select_leaves(1);
    tree.expand_and_backup(&[uniform.clone()], &[0.0]);
    assert!(tree.pool[0].is_terminal, "pre-condition: root marked terminal after overflow");

    // Second sim: root is terminal — select_leaves returns 0 leaves (nothing to expand).
    let leaves2 = tree.select_leaves(1);
    assert_eq!(leaves2.len(), 0,
        "terminal root yields no leaves — subsequent sims skip expansion entirely");
    assert!(tree.pool[0].is_terminal, "terminal status must persist");
}

#[test]
fn pool_overflow_counter_increments() {
    // Counter is process-wide; other tests in this crate may concurrently
    // run overflows, so we assert >= rather than ==. The point is just
    // that triggering an overflow visibly bumps the counter.
    let before = pool_overflow_count();

    let mut tree = tiny_tree();
    tree.new_game(Board::new());
    let n_actions = 19 * 19 + 1;
    let uniform = vec![1.0_f32 / n_actions as f32; n_actions];
    let _leaves = tree.select_leaves(1);
    tree.expand_and_backup(&[uniform], &[0.0]);

    let after = pool_overflow_count();
    assert!(after >= before + 1,
        "overflow must increment counter (before={}, after={})", before, after);
}

#[test]
fn pool_overflow_take_returns_previous_and_resets_window() {
    // take_pool_overflow_count() returns the previous value and resets the
    // counter to 0. Bench uses this to bracket measurement windows. We can't
    // assert exact values because of test parallelism, but we can verify
    // that take returns >= the increment we just caused, and that an
    // immediate second take returns 0 IF no other test fires in between
    // (best-effort — timing-sensitive, so we only assert the take API
    // signature works without panicking).
    let mut tree = tiny_tree();
    tree.new_game(Board::new());
    let n_actions = 19 * 19 + 1;
    let uniform = vec![1.0_f32 / n_actions as f32; n_actions];
    let _leaves = tree.select_leaves(1);
    tree.expand_and_backup(&[uniform], &[0.0]);

    let taken = take_pool_overflow_count();
    assert!(taken >= 1,
        "take must return >= the overflow we just triggered (got {})", taken);
}

#[test]
fn normal_sized_pool_does_not_overflow() {
    // Sanity: default MAX_NODES pool should never overflow on a fresh board.
    let mut tree = MCTSTree::new(1.5);
    tree.new_game(Board::new());

    let n_actions = 19 * 19 + 1;
    let uniform = vec![1.0_f32 / n_actions as f32; n_actions];

    let _leaves = tree.select_leaves(1);
    tree.expand_and_backup(&[uniform.clone()], &[0.0]);

    let root = &tree.pool[0];
    assert!(!root.is_terminal || tree.root_board.check_win(),
        "default pool must expand root successfully (no overflow)");
    assert!(root.n_children > 0 || tree.root_board.check_win(),
        "root must have children after first expansion");
}
