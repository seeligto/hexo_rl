use super::*;

pub(super) fn setup_two_child_tree(c_puct: f32) -> (MCTSTree, u32, u32) {
    let mut tree = MCTSTree::new(c_puct);
    tree.pool[0].moves_remaining = 2;

    let first_child = 1u32;
    tree.next_free = 3;
    tree.pool[0].first_child = first_child;
    tree.pool[0].n_children  = 2;

    let action_a = ((0u32 + 32768) << 16) | (0u32 + 32768);
    let action_b = ((0u32 + 32768) << 16) | (1u32 + 32768);

    tree.pool[1] = Node {
        parent: 0, action_idx: action_a, n_visits: 0, w_value: 0.0,
        prior: 0.7, first_child: u32::MAX, n_children: 0,
        moves_remaining: 1, is_terminal: false, terminal_value: 0.0,
        virtual_loss_count: 0,
    };
    tree.pool[2] = Node {
        parent: 0, action_idx: action_b, n_visits: 0, w_value: 0.0,
        prior: 0.3, first_child: u32::MAX, n_children: 0,
        moves_remaining: 1, is_terminal: false, terminal_value: 0.0,
        virtual_loss_count: 0,
    };
    (tree, 1, 2)
}

#[test]
fn test_puct_prefers_higher_prior_when_unvisited() {
    let (mut tree, child_a, child_b) = setup_two_child_tree(1.5);
    tree.pool[0].n_visits = 1;
    // fpu_value=0.0: both children unvisited, Q=0 for both; prior drives selection.
    let score_a = tree.puct_score(child_a, 0, 1.0, 0.0);
    let score_b = tree.puct_score(child_b, 0, 1.0, 0.0);
    assert!(score_a > score_b,
        "child with prior 0.7 should score higher than 0.3: {score_a:.4} vs {score_b:.4}");
}

#[test]
fn test_puct_visits_reduce_exploration() {
    let (mut tree, child_a, child_b) = setup_two_child_tree(1.5);
    tree.pool[0].n_visits = 100;
    tree.pool[child_a as usize].n_visits = 99;
    tree.pool[child_a as usize].w_value  = 0.0;

    // child_a is visited (n_visits=99), child_b is unvisited → fpu_value applies.
    // With fpu_value=0.0 the unvisited child still looks like Q=0, but has higher U.
    let score_a = tree.puct_score(child_a, 0, 100.0, 0.0);
    let score_b = tree.puct_score(child_b, 0, 100.0, 0.0);
    assert!(score_b > score_a,
        "less-visited child_b should be preferred: {score_b:.4} vs {score_a:.4}");
}

#[test]
fn test_backup_single_value_reaches_root() {
    let mut tree = MCTSTree::new(1.5);
    tree.pool[0].moves_remaining = 1;
    tree.pool[1] = Node {
        parent: 0, action_idx: (32768u32 << 16) | 32768u32, n_visits: 0, w_value: 0.0,
        prior: 1.0, first_child: u32::MAX, n_children: 0,
        moves_remaining: 2, is_terminal: false, terminal_value: 0.0,
        virtual_loss_count: 0,
    };
    tree.pool[2] = Node {
        parent: 1, action_idx: (32768u32 << 16) | 32769u32, n_visits: 0, w_value: 0.0,
        prior: 1.0, first_child: u32::MAX, n_children: 0,
        moves_remaining: 1, is_terminal: false, terminal_value: 0.0,
        virtual_loss_count: 0,
    };
    tree.next_free = 3;

    tree.backup(2, 1.0);

    assert_eq!(tree.pool[2].w_value, 1.0);
    assert_eq!(tree.pool[2].n_visits, 1);
    assert_eq!(tree.pool[1].w_value, 1.0, "child should not flip (mr=2)");
    assert_eq!(tree.pool[1].n_visits, 1);
    assert_eq!(tree.pool[0].w_value, -1.0, "root should flip (mr=1)");
    assert_eq!(tree.pool[0].n_visits, 1);
}

#[test]
fn test_backup_negamax_player_change() {
    let mut tree = MCTSTree::new(1.5);
    tree.pool[0].moves_remaining = 1;
    tree.pool[1] = Node {
        parent: 0, action_idx: (32768u32 << 16) | 32768u32, n_visits: 0, w_value: 0.0,
        prior: 1.0, first_child: u32::MAX, n_children: 0,
        moves_remaining: 2, is_terminal: false, terminal_value: 0.0,
        virtual_loss_count: 0,
    };
    tree.next_free = 2;

    tree.backup(1, 0.6);

    assert!((tree.pool[1].w_value - 0.6).abs() < 1e-6, "child w = 0.6");
    assert!((tree.pool[0].w_value - (-0.6)).abs() < 1e-6,
        "root w should be -0.6, got {}", tree.pool[0].w_value);
}

#[test]
fn test_select_leaves_returns_root_when_empty() {
    let mut tree = MCTSTree::new(1.5);
    let board = Board::new();
    tree.new_game(board.clone());

    let leaves = tree.select_leaves(1);
    assert_eq!(leaves.len(), 1);
    assert_eq!(leaves[0].ply, board.ply);
    assert_eq!(leaves[0].moves_remaining, board.moves_remaining);
}

#[test]
fn test_expand_and_backup_creates_children() {
    let mut tree = MCTSTree::new(1.5);
    let board = Board::new();
    tree.new_game(board);

    let leaves = tree.select_leaves(1);
    let n_legal = leaves[0].legal_move_count();

    let policy = vec![1.0 / (BOARD_SIZE * BOARD_SIZE + 1) as f32; BOARD_SIZE * BOARD_SIZE + 1];
    tree.expand_and_backup(&[policy], &[0.0]);

    let root = &tree.pool[0];
    assert!(root.is_expanded());
    assert_eq!(root.n_children as usize, n_legal);
    assert_eq!(root.n_visits, 1);
}

#[test]
fn test_full_search_runs_n_simulations() {
    let mut tree = MCTSTree::new(1.5);
    let board = Board::new();
    tree.new_game(board);

    let n_sims = 20;
    let uniform = vec![1.0 / (BOARD_SIZE * BOARD_SIZE + 1) as f32; BOARD_SIZE * BOARD_SIZE + 1];

    for _ in 0..n_sims {
        let leaves = tree.select_leaves(1);
        let n = leaves.len();
        let policies: Vec<Vec<f32>> = (0..n).map(|_| uniform.clone()).collect();
        let values: Vec<f32> = (0..n).map(|_| 0.0).collect();
        tree.expand_and_backup(&policies, &values);
    }

    assert_eq!(tree.root_visits(), n_sims as u32);
}

#[test]
fn test_virtual_loss_applied_during_select() {
    let mut tree = MCTSTree::new(1.5);
    tree.new_game(Board::new());
    let _leaves = tree.select_leaves(1);
    assert_eq!(tree.pool[0].virtual_loss_count, 1);
}

#[test]
fn test_virtual_loss_reversed_after_backup() {
    let mut tree = MCTSTree::new(1.5);
    tree.new_game(Board::new());

    let leaves = tree.select_leaves(1);
    let n = leaves.len();
    let uniform = vec![1.0 / (BOARD_SIZE * BOARD_SIZE + 1) as f32; BOARD_SIZE * BOARD_SIZE + 1];
    let policies: Vec<Vec<f32>> = (0..n).map(|_| uniform.clone()).collect();
    tree.expand_and_backup(&policies, &vec![0.0; n]);

    for i in 0..tree.next_free as usize {
        assert_eq!(tree.pool[i].virtual_loss_count, 0,
            "node {i} should have virtual_loss_count=0 after backup");
    }
}

#[test]
fn test_virtual_loss_causes_path_divergence() {
    let (mut tree, child_a, child_b) = setup_two_child_tree(1.5);
    tree.pool[0].n_visits = 1;

    let batch = tree.select_leaves(2);
    assert_eq!(batch.len(), 2);

    let dummy = vec![0.5f32; BOARD_SIZE * BOARD_SIZE + 1];
    let policies: Vec<Vec<f32>> = (0..2).map(|_| dummy.clone()).collect();
    tree.expand_and_backup(&policies, &vec![0.0; 2]);

    assert_eq!(tree.pool[0].virtual_loss_count, 0);
    assert_eq!(tree.pool[child_a as usize].virtual_loss_count, 0);
    assert_eq!(tree.pool[child_b as usize].virtual_loss_count, 0);
}

#[test]
fn test_virtual_loss_q_adjustment() {
    let node = Node {
        parent: u32::MAX, action_idx: (32768u32 << 16) | 32768u32,
        n_visits: 4, w_value: 2.0,
        prior: 0.5, first_child: u32::MAX, n_children: 0,
        moves_remaining: 1, is_terminal: false, terminal_value: 0.0,
        virtual_loss_count: 2,
    };
    let q = node.q_value_vl(VIRTUAL_LOSS_PENALTY);
    assert!(q.abs() < 1e-6, "Q should be 0.0: got {q}");
}

#[test]
fn test_dynamic_fpu_reduces_unvisited_q() {
    // Dynamic FPU: unvisited children should receive parent_q - reduction*sqrt(mass).
    let (mut tree, child_a, child_b) = setup_two_child_tree(1.5);
    tree.fpu_reduction = 0.25;
    tree.pool[0].n_visits = 10;
    tree.pool[0].w_value  = 3.0; // parent Q = 0.3

    // child_a: visited (n_visits=1), Q from w_value/n_visits.
    tree.pool[child_a as usize].n_visits = 1;
    tree.pool[child_a as usize].w_value  = 0.2;

    // child_b: unvisited → should get fpu_value.
    // explored_mass = prior of child_a = 0.7  → reduction = 0.25*sqrt(0.7) ≈ 0.209
    // fpu_value = parent_q(0.3) - 0.209 ≈ 0.091
    let explored_mass: f32 = 0.7;
    let expected_fpu = (3.0f32 / 10.0) - 0.25 * explored_mass.sqrt();

    // child_b uses fpu_value; child_a (visited) uses its own Q.
    let score_b_fpu = tree.puct_score(child_b, 0, 10.0, expected_fpu);
    let score_b_zero_fpu = tree.puct_score(child_b, 0, 10.0, 0.0);
    // With parent_q > 0, FPU value < 0.3 but > 0.0 — so dynamic FPU raises score
    // compared to the legacy Q=0 baseline when parent_q is positive.
    assert!(
        score_b_fpu > score_b_zero_fpu || expected_fpu < 0.0,
        "dynamic FPU should raise unvisited score when parent_q > 0: \
         fpu_score={score_b_fpu:.4} vs zero_fpu={score_b_zero_fpu:.4} (fpu={expected_fpu:.4})"
    );
}

// ── Quiescence tests ──────────────────────────────────────────────────────


#[test]
fn test_quiescence_overrides_value_for_3_winning_moves() {
    // Board where P1 (current player to move next, but we'll evaluate from P1's perspective)
    // has ≥3 winning moves → value should be overridden to 1.0.
    let mut tree = MCTSTree::new(1.5);
    tree.quiescence_enabled = true;
    tree.quiescence_blend_2 = 0.3;

    // Build a board where P1 has exactly 3 winning cells:
    // Two from (0,0)..(4,0): q=-1 and q=5
    // One from (20,0)..(24,0): q=19 (west end blocked, east end free)
    let mut board = Board::new();
    for q in 0..5i32 {
        board.cells.insert((q, 0), crate::board::Cell::P1);
    }
    // Block west end of first threat so it has only 1 winning cell
    board.cells.insert((-1, 0), crate::board::Cell::P2);
    // Second threat (unblocked both ends → 2 winning cells: q=19 and q=25)
    for q in 20..25i32 {
        board.cells.insert((q, 0), crate::board::Cell::P1);
    }
    board.has_stones = true;
    board.cache_dirty.set(true);
    board.current_player = crate::board::Player::One;
    // ply must be ≥ 8 so the early-game ply gate does not short-circuit.
    board.ply = 20;

    // Current player is P1; P1 has 1 + 2 = 3 winning cells → forced win.
    let wins = board.count_winning_moves(crate::board::Player::One);
    assert!(wins >= 3, "expected ≥3 winning moves for P1, got {wins}");

    let corrected = tree.apply_quiescence(&board, 0.0);
    assert_eq!(corrected, 1.0,
        "quiescence should override to 1.0 for 3+ winning moves");
}

#[test]
fn test_quiescence_overrides_value_for_3_opponent_winning_moves() {
    let mut tree = MCTSTree::new(1.5);
    tree.quiescence_enabled = true;
    tree.quiescence_blend_2 = 0.3;

    // Current player is P1, but opponent (P2) has 3 winning moves → value = -1.0
    let mut board = Board::new();
    // P2 stones: (0,0)..(4,0) unblocked (2 cells) + (20,0)..(24,0) east-only blocked (1 cell)
    for q in 0..5i32 {
        board.cells.insert((q, 0), crate::board::Cell::P2);
    }
    board.cells.insert((-1, 0), crate::board::Cell::P1); // block west end
    for q in 20..25i32 {
        board.cells.insert((q, 0), crate::board::Cell::P2);
    }
    board.has_stones = true;
    board.cache_dirty.set(true);
    board.current_player = crate::board::Player::One;
    // ply must be ≥ 8 so the early-game ply gate does not short-circuit.
    board.ply = 20;

    let opp_wins = board.count_winning_moves(crate::board::Player::Two);
    assert!(opp_wins >= 3, "expected ≥3 winning moves for P2, got {opp_wins}");

    let corrected = tree.apply_quiescence(&board, 0.0);
    assert_eq!(corrected, -1.0,
        "quiescence should override to -1.0 when opponent has 3+ winning moves");
}

#[test]
fn test_quiescence_blend_for_2_winning_moves() {
    let mut tree = MCTSTree::new(1.5);
    tree.quiescence_enabled = true;
    tree.quiescence_blend_2 = 0.3;

    // P1 has exactly 2 winning moves (unblocked 5-in-a-row along E axis)
    let mut board = Board::new();
    for q in 0..5i32 {
        board.cells.insert((q, 0), crate::board::Cell::P1);
    }
    board.has_stones = true;
    board.cache_dirty.set(true);
    board.current_player = crate::board::Player::One;
    // ply must be ≥ 8 so the early-game ply gate does not short-circuit.
    board.ply = 10;

    let wins = board.count_winning_moves(crate::board::Player::One);
    assert_eq!(wins, 2, "unblocked 5-in-a-row should have exactly 2 winning moves");

    let nn_value = 0.5f32;
    let corrected = tree.apply_quiescence(&board, nn_value);
    let expected = (nn_value + 0.3).min(1.0);
    assert!((corrected - expected).abs() < 1e-6,
        "blend for 2 winning moves: expected {expected}, got {corrected}");
}

#[test]
fn test_quiescence_disabled_does_not_change_value() {
    let mut tree = MCTSTree::new(1.5);
    tree.quiescence_enabled = false;

    let mut board = Board::new();
    // Give P1 a huge number of winning moves
    for q in 0..5i32 {
        board.cells.insert((q, 0), crate::board::Cell::P1);
    }
    board.has_stones = true;
    board.cache_dirty.set(true);

    let nn_value = 0.42f32;
    let corrected = tree.apply_quiescence(&board, nn_value);
    assert_eq!(corrected, nn_value, "disabled quiescence must not change value");
}

#[test]
fn test_quiescence_fire_count_increments_and_resets() {
    let mut tree = MCTSTree::new_full(1.5, 0.0, 0.0);
    tree.quiescence_enabled = true;
    tree.quiescence_blend_2 = 0.3;

    // Board setup: P1 has ≥3 winning moves (same as test_quiescence_overrides_value_for_current_wins).
    let mut board = Board::new();
    for q in 0..5i32 {
        board.cells.insert((q, 0), crate::board::Cell::P1);
    }
    board.cells.insert((-1, 0), crate::board::Cell::P2); // block west end of first threat
    for q in 20..25i32 {
        board.cells.insert((q, 0), crate::board::Cell::P1);
    }
    board.has_stones = true;
    board.cache_dirty.set(true);
    board.current_player = crate::board::Player::One;
    board.ply = 20;

    let wins = board.count_winning_moves(crate::board::Player::One);
    assert!(wins >= 3, "expected ≥3 winning moves for P1, got {wins}");

    // Counter starts at 0.
    assert_eq!(tree.quiescence_fire_count.load(Ordering::Relaxed), 0);

    // First call fires → counter = 1.
    let result = tree.apply_quiescence(&board, 0.5);
    assert_eq!(result, 1.0, "forced win should override to 1.0");
    assert_eq!(tree.quiescence_fire_count.load(Ordering::Relaxed), 1,
        "counter should be 1 after one firing call");

    // Second call fires again → counter = 2.
    tree.apply_quiescence(&board, 0.5);
    assert_eq!(tree.quiescence_fire_count.load(Ordering::Relaxed), 2,
        "counter should accumulate across calls");

    // new_game() resets counter to 0.
    tree.new_game(Board::new());
    assert_eq!(tree.quiescence_fire_count.load(Ordering::Relaxed), 0,
        "counter should reset to 0 after new_game()");
}

#[test]
fn test_quiescence_no_override_in_early_game() {
    let mut tree = MCTSTree::new(1.5);
    tree.quiescence_enabled = true;
    tree.quiescence_blend_2 = 0.3;

    // Early game — no threatening formations.
    let board = Board::new();
    let nn_value = 0.123f32;
    let corrected = tree.apply_quiescence(&board, nn_value);
    assert_eq!(corrected, nn_value, "early game should not trigger quiescence");
}

#[test]
fn test_no_forced_win_short_circuit_in_expansion() {
    use crate::board::Player;

    let mut board = Board::new();
    for r in 0..4 {
        board.cells.insert((0, r), crate::board::Cell::P1);
    }
    board.ply = 7;
    board.current_player = Player::One;

    assert!(!board.check_win(), "test setup: board should not be a terminal win");

    let mut tree = MCTSTree::new(1.5);
    tree.new_game(board);

    let n_actions = BOARD_SIZE * BOARD_SIZE + 1;
    let uniform_policy = vec![1.0 / n_actions as f32; n_actions];
    let nn_value = 0.5;

    tree.expand_and_backup(&[uniform_policy], &[nn_value]);

    let root = &tree.pool[0];
    assert!(!root.is_terminal,
        "root must NOT be marked terminal for a forced-win formation");
}

// ── CF-1: compound-turn terminal sign ────────────────────────────────────
//
// `expand_and_backup_single` assigns the terminal value of a `check_win`
// leaf. The leaf's side-to-move (== `board.moves_remaining`) decides the
// sign, NOT a hardcoded -1.0:
//   * stone-2 / turn-final win → `apply_move` flips the player, leaf has
//     `mr==2` (opponent/loser to move) → terminal value -1.0 (correct).
//   * stone-1 win (mid-turn)   → `apply_move` keeps the player (mr 2→1, no
//     flip), leaf has `mr==1` (winner still to move) → terminal value +1.0.
// The pre-fix hardcode scored the stone-1 win as -1.0, dragging its parent's
// Q toward a loss → PUCT avoided completing on the first stone (CF-1).
//
// Placement: these exercise `expand_and_backup_single` and poke
// `Board::cells`/`last_move`, both `pub(crate)`. An external
// `engine/tests/inv*.rs` cannot reach either, so the discriminating cell
// lives crate-internal alongside `test_no_forced_win_short_circuit_*`.

/// Build a P1 6-in-a-row along the E/W axis with `last_move` on the line.
/// `mr`/`player` are set by the caller to model stone-1 vs stone-2 wins.
fn make_stone1_win_board(mr: u8, player: crate::board::Player) -> Board {
    let mut board = Board::new();
    for q in 0..6 {
        board.cells.insert((q, 0), crate::board::Cell::P1);
    }
    board.last_move = Some((5, 0));
    board.current_player = player;
    board.moves_remaining = mr;
    board.ply = 11;
    assert!(board.check_win(), "test setup: board must be a terminal win");
    board
}

/// Attach a single leaf child to the root and run its terminal backup.
/// Returns (leaf_terminal_value, parent_backed_up_w).
fn run_terminal_leaf(parent_mr: u8, leaf_mr: u8, board: &Board) -> (f32, f32) {
    let mut tree = MCTSTree::new(1.5);
    tree.pool[0].moves_remaining = parent_mr;
    tree.pool[0].first_child = 1;
    tree.pool[0].n_children = 1;
    tree.next_free = 2;
    tree.pool[1] = Node {
        parent: 0, action_idx: (32768u32 << 16) | 32773u32, n_visits: 0, w_value: 0.0,
        prior: 1.0, first_child: u32::MAX, n_children: 0,
        moves_remaining: leaf_mr, is_terminal: false, terminal_value: 0.0,
        virtual_loss_count: 0,
    };
    tree.expand_and_backup_single(1, board, &[], 0.0);
    (tree.pool[1].terminal_value, tree.pool[0].w_value)
}

/// Case A — stone-1 win: leaf `mr==1` (winner to move) from a `mr==2`
/// parent. Terminal value must be +1.0, and since the parent does NOT flip
/// (mr==2), the winning child drags the parent's Q toward +1 → PUCT prefers
/// completing on the first stone (the policy-target signal). FAILS on the
/// pre-fix hardcoded -1.0.
#[test]
fn test_cf1_stone1_win_scored_as_win() {
    let board = make_stone1_win_board(1, crate::board::Player::One);
    let (leaf_tv, parent_w) = run_terminal_leaf(2, 1, &board);
    assert_eq!(leaf_tv, 1.0,
        "stone-1 win leaf (mr==1, winner to move) must score +1.0, not -1.0");
    assert_eq!(parent_w, 1.0,
        "winning stone-1 child must back up +1.0 to its mr==2 parent \
         (policy target points at the winning move, not filler-first)");
}

/// Case B — stone-2 / turn-final win: leaf `mr==2` (opponent to move) from a
/// `mr==1` parent. Terminal value must stay -1.0; the negamax flip at the
/// mr==1 parent turns that into +1.0 for the mover. Proves the leaf-side
/// derivation does NOT regress the case the hardcode handled correctly.
#[test]
fn test_cf1_stone2_win_still_scored_as_loss_to_mover() {
    let board = make_stone1_win_board(2, crate::board::Player::Two);
    let (leaf_tv, parent_w) = run_terminal_leaf(1, 2, &board);
    assert_eq!(leaf_tv, -1.0,
        "turn-final win leaf (mr==2, loser to move) must stay -1.0");
    assert_eq!(parent_w, 1.0,
        "negamax flip at the mr==1 parent turns the -1.0 leaf into +1.0 \
         for the player who completed the line");
}

// ── CF-6: FPU sign consistency (pinning test, no production-logic change) ──
//
// Pins the verified no-bug invariant in `puct_score`:
//   * A VISITED child's stored Q is mr-negated: at an mr==1 parent the child is
//     the OTHER player (apply_move flips), so its own-perspective Q must be
//     flipped into the parent's frame (`-child.q_value_vl`); at mr==2 the child
//     is the SAME player (no flip, `+child.q_value_vl`).
//   * An UNVISITED child's `fpu_value` is supplied by the caller ALREADY in the
//     parent's to-move frame, so it is NEVER mr-negated — identical at mr==1 and
//     mr==2.
// Setting c_puct=0.0 makes the U term exactly 0, so `puct_score == q` and the
// q-part is read directly. Flipping either expected sign makes this FAIL.

#[test]
fn test_cf6_fpu_sign_consistent_with_visited_child_at_both_mr() {
    let (mut tree, c1, c2) = setup_two_child_tree(0.0);
    let sqrt_n = 2.0_f32.sqrt();
    const FPU: f32 = -0.3;

    // c1: visited child, known decisive own-frame Q. virtual_loss_count==0 from
    // setup, so q_value_vl == w_value / n_visits.
    tree.pool[c1 as usize].n_visits = 4;
    tree.pool[c1 as usize].w_value  = 2.0; // own-frame Q = 2.0 / 4 = 0.5
    let own_q = tree.pool[c1 as usize].q_value_vl(tree.virtual_loss);
    assert!((own_q - 0.5).abs() < 1e-6, "precondition: own_q = {own_q}");
    // c2 stays unvisited (n_visits == 0 from setup).
    assert_eq!(tree.pool[c2 as usize].n_visits, 0);

    // --- mr == 2 parent (children are the SAME player) ---
    assert_eq!(tree.pool[0].moves_remaining, 2);
    let q_visited_mr2 = tree.puct_score(c1, 0, sqrt_n, FPU);
    assert!((q_visited_mr2 - 0.5).abs() < 1e-6,
        "mr2 visited q = {q_visited_mr2}, want +0.5 (not negated)");
    let q_unvisited_mr2 = tree.puct_score(c2, 0, sqrt_n, FPU);
    assert!((q_unvisited_mr2 - FPU).abs() < 1e-6,
        "mr2 unvisited q = {q_unvisited_mr2}, want fpu_value {FPU}");

    // --- mr == 1 parent (children are the OTHER player) ---
    tree.pool[0].moves_remaining = 1;
    let q_visited_mr1 = tree.puct_score(c1, 0, sqrt_n, FPU);
    assert!((q_visited_mr1 - (-0.5)).abs() < 1e-6,
        "mr1 visited q = {q_visited_mr1}, want -0.5 (negated into parent frame)");
    let q_unvisited_mr1 = tree.puct_score(c2, 0, sqrt_n, FPU);
    assert!((q_unvisited_mr1 - FPU).abs() < 1e-6,
        "mr1 unvisited q = {q_unvisited_mr1}, want fpu_value {FPU} (UNCHANGED by mr)");
}

// ── Gumbel MCTS tests ────────────────────────────────────────────────────

pub(super) fn setup_expanded_root() -> MCTSTree {
    let mut tree = MCTSTree::new(1.5);
    let board = Board::new();
    tree.new_game(board);

    // Expand root with uniform priors.
    let _leaves = tree.select_leaves(1);
    let n_actions = BOARD_SIZE * BOARD_SIZE + 1;
    let policy = vec![1.0 / n_actions as f32; n_actions];
    tree.expand_and_backup(&[policy], &[0.0]);
    tree
}

#[test]
fn test_forced_root_child_selection() {
    let mut tree = setup_expanded_root();
    let root = &tree.pool[0];
    assert!(root.is_expanded());
    let first = root.first_child;
    let n_ch = root.n_children as usize;
    assert!(n_ch >= 2, "need at least 2 children");

    // Force selection to second child.
    let target_child = first + 1;
    tree.forced_root_child = Some(target_child);

    // Run several simulations — all should go through the forced child.
    let n_sims = 10;
    let n_actions = BOARD_SIZE * BOARD_SIZE + 1;
    let uniform = vec![1.0 / n_actions as f32; n_actions];
    for _ in 0..n_sims {
        let leaves = tree.select_leaves(1);
        let n = leaves.len();
        let policies: Vec<Vec<f32>> = (0..n).map(|_| uniform.clone()).collect();
        let values = vec![0.0f32; n];
        tree.expand_and_backup(&policies, &values);
    }

    // The forced child should have gotten all visits (minus root expansion).
    let forced_visits = tree.pool[target_child as usize].n_visits;
    assert!(forced_visits >= n_sims as u32 - 1,
        "forced child should have >= {} visits, got {}", n_sims - 1, forced_visits);

    // First child (not forced) should have 0 visits.
    let other_visits = tree.pool[first as usize].n_visits;
    assert_eq!(other_visits, 0,
        "non-forced child should have 0 visits, got {other_visits}");

    tree.forced_root_child = None;
}

#[test]
fn test_forced_root_none_uses_puct() {
    // With forced_root_child = None, PUCT selects normally.
    let mut tree = setup_expanded_root();
    tree.forced_root_child = None;

    let n_sims = 20;
    let n_actions = BOARD_SIZE * BOARD_SIZE + 1;
    let uniform = vec![1.0 / n_actions as f32; n_actions];
    for _ in 0..n_sims {
        let leaves = tree.select_leaves(1);
        let n = leaves.len();
        let policies: Vec<Vec<f32>> = (0..n).map(|_| uniform.clone()).collect();
        let values = vec![0.0f32; n];
        tree.expand_and_backup(&policies, &values);
    }

    // Multiple children should have visits (PUCT spreads them).
    let first = tree.pool[0].first_child as usize;
    let n_ch = tree.pool[0].n_children as usize;
    let visited_count = (first..first + n_ch)
        .filter(|&i| tree.pool[i].n_visits > 0)
        .count();
    assert!(visited_count >= 2,
        "PUCT should visit multiple children, only {visited_count} visited");
}

#[test]
fn test_gumbel_disabled_no_behavior_change() {
    // When forced_root_child is None, behavior is identical to pre-Gumbel code.
    // Verify by running search twice with same setup and checking same results.
    let run_search = || -> Vec<u32> {
        let mut tree = MCTSTree::new(1.5);
        let board = Board::new();
        tree.new_game(board);
        tree.forced_root_child = None; // explicitly None

        let n_actions = BOARD_SIZE * BOARD_SIZE + 1;
        let uniform = vec![1.0 / n_actions as f32; n_actions];
        for _ in 0..10 {
            let leaves = tree.select_leaves(1);
            let n = leaves.len();
            let policies: Vec<Vec<f32>> = (0..n).map(|_| uniform.clone()).collect();
            let values = vec![0.0f32; n];
            tree.expand_and_backup(&policies, &values);
        }

        // Extract visit counts for root children
        let first = tree.pool[0].first_child as usize;
        let n_ch = tree.pool[0].n_children as usize;
        (first..first + n_ch).map(|i| tree.pool[i].n_visits).collect()
    };

    let visits_a = run_search();
    let visits_b = run_search();
    // With deterministic input (uniform policy, value=0), results should match.
    assert_eq!(visits_a, visits_b,
        "search with forced_root_child=None should be deterministic");
}

#[test]
fn test_nonroot_uses_puct_when_root_forced() {
    // Verify that non-root selection still uses PUCT (spreads visits)
    // even when root selection is forced.
    let mut tree = setup_expanded_root();
    let first_child = tree.pool[0].first_child;
    tree.forced_root_child = Some(first_child);

    let n_actions = BOARD_SIZE * BOARD_SIZE + 1;
    let uniform = vec![1.0 / n_actions as f32; n_actions];

    // Run enough sims to expand the forced child and go deeper.
    for _ in 0..30 {
        let leaves = tree.select_leaves(1);
        let n = leaves.len();
        let policies: Vec<Vec<f32>> = (0..n).map(|_| uniform.clone()).collect();
        let values = vec![0.0f32; n];
        tree.expand_and_backup(&policies, &values);
    }

    // The forced child should now be expanded with multiple children.
    let fc = &tree.pool[first_child as usize];
    if fc.is_expanded() && fc.n_children > 1 {
        // Multiple grandchildren should have visits (PUCT below root).
        let gc_first = fc.first_child as usize;
        let gc_n = fc.n_children as usize;
        let gc_visited = (gc_first..gc_first + gc_n)
            .filter(|&i| tree.pool[i].n_visits > 0)
            .count();
        assert!(gc_visited >= 2,
            "PUCT at non-root should visit multiple grandchildren, got {gc_visited}");
    }
    // If forced child isn't expanded (e.g., terminal), the test still passes.
    tree.forced_root_child = None;
}

// ── D-QFIX-LAND A1: interior-selector tests ──────────────────────────────────

#[test]
fn test_interior_selector_default_is_puct() {
    // A1 invariant: a freshly constructed tree defaults to `Puct` so existing
    // runs are byte-identical (the selector must be opted into explicitly).
    let tree = MCTSTree::new(1.5);
    assert_eq!(tree.interior_selector, InteriorSelector::Puct);
    let tree_full = MCTSTree::new_full(1.5, crate::mcts::VIRTUAL_LOSS_PENALTY, 0.0);
    assert_eq!(tree_full.interior_selector, InteriorSelector::Puct);
}

#[test]
fn test_interior_selector_from_config_str_parses_known_variants() {
    // Both registry strings parse to their enum variant.
    assert_eq!(InteriorSelector::from_config_str("puct"), InteriorSelector::Puct);
    assert_eq!(
        InteriorSelector::from_config_str("gumbel_improved"),
        InteriorSelector::GumbelImproved
    );
}

#[test]
#[should_panic(expected = "unknown mcts.interior_selector")]
fn test_interior_selector_from_config_str_panics_on_unknown() {
    // A1 config is hard-read end-to-end: an unknown variant must panic rather
    // than silently falling back to a default.
    let _ = InteriorSelector::from_config_str("not_a_selector");
}

#[test]
fn test_interior_selector_puct_arm_matches_head_path() {
    // The `Puct` arm of `select_one_leaf` must select the same leaves as the
    // pre-A1 `pick_best_puct` path. We run an identical search under the
    // default tree (which dispatches through the `Puct` match arm) and an
    // explicitly-set `Puct` tree, and assert the per-child visit distributions
    // are byte-identical — locking the "Puct == HEAD behaviour" invariant.
    let run = |sel: InteriorSelector| -> Vec<u32> {
        let mut tree = setup_expanded_root();
        tree.forced_root_child = None;
        tree.interior_selector = sel;
        let n_actions = BOARD_SIZE * BOARD_SIZE + 1;
        let uniform = vec![1.0 / n_actions as f32; n_actions];
        for _ in 0..30 {
            let leaves = tree.select_leaves(1);
            let n = leaves.len();
            let policies: Vec<Vec<f32>> = (0..n).map(|_| uniform.clone()).collect();
            let values = vec![0.0f32; n];
            tree.expand_and_backup(&policies, &values);
        }
        let first = tree.pool[0].first_child as usize;
        let n_ch = tree.pool[0].n_children as usize;
        (first..first + n_ch).map(|i| tree.pool[i].n_visits).collect()
    };

    // Default tree path (no explicit set) vs explicit `Puct` — both dispatch to
    // `pick_best_puct`, so visit distributions must match exactly.
    let visits_default = run(InteriorSelector::Puct);
    let mut tree_default = setup_expanded_root();
    tree_default.forced_root_child = None;
    // (tree_default already defaults to Puct — verified by the default test.)
    let n_actions = BOARD_SIZE * BOARD_SIZE + 1;
    let uniform = vec![1.0 / n_actions as f32; n_actions];
    for _ in 0..30 {
        let leaves = tree_default.select_leaves(1);
        let n = leaves.len();
        let policies: Vec<Vec<f32>> = (0..n).map(|_| uniform.clone()).collect();
        let values = vec![0.0f32; n];
        tree_default.expand_and_backup(&policies, &values);
    }
    let first = tree_default.pool[0].first_child as usize;
    let n_ch = tree_default.pool[0].n_children as usize;
    let visits_default_tree: Vec<u32> =
        (first..first + n_ch).map(|i| tree_default.pool[i].n_visits).collect();

    assert_eq!(
        visits_default, visits_default_tree,
        "explicit Puct arm must match the default (HEAD) selection path"
    );

    // The placeholder `GumbelImproved` arm currently delegates to PUCT, so it
    // produces the same distribution (documents the delegation contract).
    let visits_gumbel = run(InteriorSelector::GumbelImproved);
    assert_eq!(
        visits_default, visits_gumbel,
        "GumbelImproved placeholder must delegate to PUCT (byte-identical for now)"
    );
}

#[test]
fn test_last_search_stats_bounds_after_sims() {
    let mut tree = MCTSTree::new(1.5);
    tree.new_game(Board::new());

    let n_sims = 10;
    let n_actions = BOARD_SIZE * BOARD_SIZE + 1;
    let uniform = vec![1.0 / n_actions as f32; n_actions];
    for _ in 0..n_sims {
        let leaves = tree.select_leaves(1);
        let n = leaves.len();
        let policies: Vec<Vec<f32>> = (0..n).map(|_| uniform.clone()).collect();
        let values = vec![0.0f32; n];
        tree.expand_and_backup(&policies, &values);
    }

    let (mean_depth, root_concentration) = tree.last_search_stats();
    assert!(mean_depth >= 0.0,
        "mean_depth must be >= 0.0, got {mean_depth}");
    assert!(root_concentration >= 0.0 && root_concentration <= 1.0,
        "root_concentration must be in [0.0, 1.0], got {root_concentration}");
}

// ── Top-K leaf cap tests ─────────────────────────────────────────────────

#[test]
fn test_topk_truncates_at_max_children() {
    use fxhash::FxHashSet;
    use crate::board::HALF;
    use super::backup::pick_topk_children;

    // 600 unique cells split between 200 in-window (high priors) and
    // 400 out-of-window (sort prior 0.0). Top K will be drawn from the
    // 200 in-window cells since out-of-window sinks under sort.
    let mut cells: FxHashSet<(i32, i32)> = FxHashSet::default();
    'iw: for q in -HALF..=HALF {
        for r in -HALF..=HALF {
            cells.insert((q, r));
            if cells.len() == 200 { break 'iw; }
        }
    }
    'ow: for q in 30..=60 {
        for r in 30..=60 {
            cells.insert((q, r));
            if cells.len() == 600 { break 'ow; }
        }
    }
    assert_eq!(cells.len(), 600, "test setup must produce 600 cells");

    // Strictly increasing prior with flat_idx → unique priors for
    // every in-window cell.
    let n_actions = BOARD_SIZE * BOARD_SIZE + 1;
    let policy: Vec<f32> =
        (0..n_actions).map(|i| (i + 1) as f32 / n_actions as f32).collect();

    let (chosen, sort_used) = pick_topk_children(&cells, 0, 0, &policy, BOARD_SIZE as i32, HALF);
    assert!(sort_used, "600 > K must take sort path");
    assert_eq!(chosen.len(), MAX_CHILDREN_PER_NODE,
        "chosen must equal K, got {}", chosen.len());

    // Top K should all be in-window since out-of-window sort_prior=0.0
    // and 200 in-window cells with policy > 0 dominate.
    for &((q, r), prior) in &chosen {
        let flat = Board::window_flat_idx_at(q, r, 0, 0);
        assert!(flat < n_actions,
            "top-K must be drawn from in-window cells, got flat={flat} for ({q},{r})");
        assert!(prior > 0.0,
            "in-window cell prior must be >0 for this fixture, got {prior}");
    }

    // With all-in-window selection and priors monotonic in flat, the
    // chosen Vec's priors must be non-increasing.
    for w in chosen.windows(2) {
        assert!(w[0].1 >= w[1].1,
            "priors must be non-increasing: {} then {}", w[0].1, w[1].1);
    }
}

#[test]
fn test_topk_tie_break_by_flat_idx() {
    use fxhash::FxHashSet;
    use crate::board::HALF;
    use super::backup::pick_topk_children;

    // K + 1 cells inside window with identical priors → exactly one is
    // dropped. Tie-break = flat_idx asc, so the cell with the largest
    // flat_idx is the one dropped.
    let target = MAX_CHILDREN_PER_NODE + 1;
    let mut cells: FxHashSet<(i32, i32)> = FxHashSet::default();
    let mut flats_inserted: Vec<usize> = Vec::new();
    'outer: for q in -HALF..=HALF {
        for r in -HALF..=HALF {
            let flat = Board::window_flat_idx_at(q, r, 0, 0);
            cells.insert((q, r));
            flats_inserted.push(flat);
            if cells.len() == target { break 'outer; }
        }
    }
    assert_eq!(cells.len(), target);

    let n_actions = BOARD_SIZE * BOARD_SIZE + 1;
    let uniform_high = vec![0.5_f32; n_actions];

    let (chosen, sort_used) = pick_topk_children(&cells, 0, 0, &uniform_high, BOARD_SIZE as i32, HALF);
    assert!(sort_used);
    assert_eq!(chosen.len(), MAX_CHILDREN_PER_NODE);

    let chosen_flats: std::collections::HashSet<usize> = chosen
        .iter()
        .map(|&((q, r), _)| Board::window_flat_idx_at(q, r, 0, 0))
        .collect();

    let max_flat = *flats_inserted.iter().max().unwrap();
    assert!(!chosen_flats.contains(&max_flat),
        "highest flat_idx must be the dropped cell under tie (max_flat={max_flat})");
    assert_eq!(chosen_flats.len(), MAX_CHILDREN_PER_NODE);
}

#[test]
fn test_topk_fast_path_keeps_all_when_under_cap() {
    use fxhash::FxHashSet;
    use crate::board::HALF;
    use super::backup::pick_topk_children;

    // 50 cells, K=192 → fast path; all cells must appear in the output
    // and `sort_used` is false.
    let mut cells: FxHashSet<(i32, i32)> = FxHashSet::default();
    'outer: for q in -3..=4 {
        for r in -3..=4 {
            cells.insert((q, r));
            if cells.len() == 50 { break 'outer; }
        }
    }
    assert_eq!(cells.len(), 50);

    let n_actions = BOARD_SIZE * BOARD_SIZE + 1;
    let policy = vec![1.0 / n_actions as f32; n_actions];

    let (chosen, sort_used) = pick_topk_children(&cells, 0, 0, &policy, BOARD_SIZE as i32, HALF);
    assert!(!sort_used, "fast path expected when n_legal <= K");
    assert_eq!(chosen.len(), 50);

    let chosen_set: std::collections::HashSet<(i32, i32)> =
        chosen.iter().map(|&(coord, _)| coord).collect();
    assert_eq!(chosen_set, cells.iter().copied().collect(),
        "fast path must include every legal move");
}

#[test]
fn test_topk_child_order_independent_of_hashset_capacity() {
    // §S182 regression guard. `pick_topk_children` must emit children in a
    // canonical order that does NOT depend on the `FxHashSet`'s capacity or
    // iteration order. §S182's `legal_moves_set` capacity-reserve changed the
    // hashbrown table layout; before the fix the `n_legal <= K` path collected
    // children in raw iteration order, leaking that layout into MCTS.
    use fxhash::FxHashSet;
    use crate::board::HALF;
    use super::backup::pick_topk_children;

    let coords: Vec<(i32, i32)> =
        (-3..=3).flat_map(|q| (-3..=3).map(move |r| (q, r))).collect();

    // Same elements, deliberately different table capacities → the two sets
    // iterate in different orders (exactly the §S182 scenario).
    let mut set_small: FxHashSet<(i32, i32)> = FxHashSet::default();
    for &c in &coords { set_small.insert(c); }
    let mut set_large: FxHashSet<(i32, i32)> = FxHashSet::default();
    set_large.reserve(4096);
    for &c in &coords { set_large.insert(c); }
    assert_eq!(set_small, set_large, "the two sets must hold identical moves");

    // Non-uniform policy so there is a real prior ordering to canonicalize.
    let n_actions = BOARD_SIZE * BOARD_SIZE + 1;
    let mut policy = vec![0.0f32; n_actions];
    for (i, p) in policy.iter_mut().enumerate() {
        *p = ((i % 17) as f32) * 0.013;
    }

    let (chosen_small, _) =
        pick_topk_children(&set_small, 0, 0, &policy, BOARD_SIZE as i32, HALF);
    let (chosen_large, _) =
        pick_topk_children(&set_large, 0, 0, &policy, BOARD_SIZE as i32, HALF);

    assert_eq!(
        chosen_small, chosen_large,
        "pick_topk_children child ORDER must be independent of FxHashSet \
         capacity / iteration order (§S182 regression — see backup.rs fn-doc)"
    );
}
