//! Regression tests for the child-Q perspective flip in get_improved_policy,
//! GumbelSearchState::score, and get_top_visits.
//!
//! Each node stores w_value in its own player-to-move perspective (backup.rs
//! negamax). When root has moves_remaining==1, children belong to the opponent,
//! so their w_value must be negated before use as Q targets.
//! puct_score already handled this (selection.rs:21-22); three sibling sites
//! did not, inverting training targets at ~50% of positions.
//! Fixes A-001, A-002, A-010 (reports/master_review_2026-04-18/).

use engine::board::{Board, BOARD_SIZE};
use engine::game_runner::gumbel_search::GumbelSearchState;
use engine::mcts::MCTSTree;

/// Build a tree with one visited child (n_visits=1, w_value=child_value).
/// The board determines root.moves_remaining.
/// Returns (tree, flat_action_idx_of_visited_child).
fn build_tree_visit_one_child(board: Board, child_value: f32) -> (MCTSTree, usize) {
    let n_actions = BOARD_SIZE * BOARD_SIZE + 1;
    let uniform = vec![1.0_f32 / n_actions as f32; n_actions];

    let mut tree = MCTSTree::new(1.5);
    tree.new_game(board);

    // Sim 1: expand root (leaf = root itself; value 0.0 so root.w_value starts neutral)
    let _leaves = tree.select_leaves(1);
    tree.expand_and_backup(&[uniform.clone()], &[0.0]);

    // Sim 2: descend to one child, expand it, backup child_value
    let _leaves = tree.select_leaves(1);
    tree.expand_and_backup(&[uniform.clone()], &[child_value]);

    // find the visited child via get_top_visits
    let top = tree.get_top_visits(1);
    assert_eq!(top.len(), 1, "exactly one child should have visits");
    let (coord_str, visits, _prior, _q) = &top[0];
    assert_eq!(*visits, 1, "visited child should have n_visits=1");

    // parse "(q,r)" and compute flat index
    let inner = coord_str
        .trim_matches(|c| c == '(' || c == ')')
        .split(',')
        .map(|x| x.trim().parse::<i32>().unwrap())
        .collect::<Vec<_>>();
    let flat = tree.root_board.window_flat_idx(inner[0], inner[1]);
    assert!(flat < n_actions, "visited coord must be in window");

    (tree, flat)
}

mod perspective_parity {
    use super::*;

    /// Compare log(policy[flat]) between mr=2 and mr=1 trees.
    /// Both visited children have w_value=+0.8.
    /// After the fix: mr=2 assigns higher log-probability to its visited child
    /// than mr=1 does (because mr=1 must negate the opponent-perspective Q).
    #[test]
    fn test_improved_policy_flips_q_at_intermediate_ply() {
        // mr=2: apply one move to Board::new() (moves_remaining 1→2)
        let mut board_mr2 = Board::new();
        board_mr2.apply_move(0, 0).expect("(0,0) must be legal on fresh board");
        assert_eq!(board_mr2.moves_remaining, 2);

        let board_mr1 = Board::new(); // moves_remaining==1
        assert_eq!(board_mr1.moves_remaining, 1);

        let (tree_mr2, flat_mr2) = build_tree_visit_one_child(board_mr2, 0.8);
        let (tree_mr1, flat_mr1) = build_tree_visit_one_child(board_mr1, 0.8);

        let c_visit = 50.0_f32;
        let c_scale = 1.0_f32;

        let policy_mr2 = tree_mr2.get_improved_policy(BOARD_SIZE, c_visit, c_scale);
        let policy_mr1 = tree_mr1.get_improved_policy(BOARD_SIZE, c_visit, c_scale);

        // log(policy) ≈ logit - log(Z). Visited-child logit differs by ±σ*q.
        // After fix: diff >> 2*0.8*c_scale*max_n (actual ≈ 20 at c_visit=50).
        let log_p_mr2 = policy_mr2[flat_mr2].ln();
        let log_p_mr1 = policy_mr1[flat_mr1].ln();
        let diff = log_p_mr2 - log_p_mr1;

        let min_bound = 2.0 * 0.8 * c_scale * 1.0_f32; // max_n=1
        assert!(
            diff > min_bound,
            "log_policy(mr2)={log_p_mr2:.4} − log_policy(mr1)={log_p_mr1:.4} = {diff:.4}; expected > {min_bound}"
        );
    }

    /// GumbelSearchState::score must negate q_hat when root_mr==1.
    /// Directly constructs state (no tree) to isolate the score function.
    #[test]
    fn test_gumbel_score_flips_at_intermediate_ply() {
        let c_visit = 50.0_f32;
        let c_scale = 1.0_f32;

        // visited child at offset 0: n_visits=1, w_value=+0.8
        let make_state = |root_mr: u8| GumbelSearchState {
            gumbel_values: vec![0.0, 0.0],
            log_priors: vec![f32::ln(0.5), f32::ln(0.5)],
            candidates: vec![0, 1],
            num_phases: 1,
            c_visit,
            c_scale,
            first_child: 0,
            root_mr,
            cached_children: vec![(1, 0.8_f32), (0, 0.0_f32)],
        };

        let state_mr2 = make_state(2);
        let state_mr1 = make_state(1);

        let score_mr2 = state_mr2.score(0, 1);
        let score_mr1 = state_mr1.score(0, 1);

        // mr=2: sigma_contrib = (50+1)*1*0.8 = +40.8
        // mr=1: sigma_contrib = (50+1)*1*(-0.8) = -40.8
        assert!(
            score_mr2 > score_mr1,
            "Gumbel score at mr=2 ({score_mr2:.4}) should exceed score at mr=1 ({score_mr1:.4})"
        );
        let sigma = (c_visit + 1.0) * c_scale;
        let expected_diff = 2.0 * 0.8 * sigma;
        let actual_diff = score_mr2 - score_mr1;
        assert!(
            (actual_diff - expected_diff).abs() < 1e-3,
            "Gumbel score diff {actual_diff:.4} should equal 2*q*sigma={expected_diff:.4}"
        );
    }

    /// get_top_visits must return Q in root's perspective.
    /// root.moves_remaining==1, child w_value=+0.8 → returned q_value must be -0.8.
    #[test]
    fn test_get_top_visits_returns_root_perspective_q() {
        let board = Board::new(); // moves_remaining==1
        assert_eq!(board.moves_remaining, 1);

        let (tree, _flat) = build_tree_visit_one_child(board, 0.8);

        let top = tree.get_top_visits(1);
        assert_eq!(top.len(), 1);
        let (_coord, visits, _prior, q_returned) = top[0].clone();

        assert_eq!(visits, 1);
        assert!(
            (q_returned - (-0.8_f32)).abs() < 1e-5,
            "root-perspective q should be −0.8, got {q_returned}"
        );
    }
}
