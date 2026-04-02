/// Sparse axial hex board with sliding 19×19 view window.
///
/// Coordinate system: axial (q, r).
///   E:  (+1,  0)   W:  (-1,  0)
///   NE: ( 0, +1)   SW: ( 0, -1)
///   NW: (-1, +1)   SE: (+1, -1)
///
/// Storage: FxHashMap<(q,r), Cell> — unbounded.
///
/// View window: fixed 19×19 tensor centred on the bounding-box centroid of all
/// placed stones.  On an empty board the window is centred at (0,0).
/// The window slides as play drifts; it never clips stones.
///
/// Legal moves: empty cells within bounding_box + 2 margin, clipped to the
/// current 19×19 window.  On an empty board all 361 window cells are legal.
///
/// Win condition: 6 stones of the same player in a row along one of the three
/// hex axes (E/W, NE/SW, NW/SE).
///
/// Turn structure:
///   ply 0 (first move ever): player 1 places exactly 1 stone.
///   ply 1+: each player places exactly 2 stones before the turn passes.

pub mod bitboard;
pub mod zobrist;
pub mod state;
mod moves;

pub use state::{
    Board, MoveDiff, Player, Cell,
    BOARD_SIZE, HALF, TOTAL_CELLS, HEX_AXES, HEX_DIRS,
    hex_distance,
};

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn recompute_zobrist(board: &Board) -> u128 {
        let mut hash = 0u128;
        for (&(q, r), &cell) in board.cells.iter() {
            let player_idx = match cell {
                Cell::P1 => 0,
                Cell::P2 => 1,
                Cell::Empty => continue,
            };
            hash ^= super::zobrist::ZobristTable::get_for_pos(q, r, player_idx);
        }
        hash
    }

    fn next_u64(seed: &mut u64) -> u64 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *seed
    }

    #[test]
    fn empty_board_no_win() {
        let b = Board::new();
        assert!(!b.check_win());
        assert!(b.winner().is_none());
    }

    #[test]
    fn first_move_is_single_for_player_one() {
        let mut b = Board::new();
        assert_eq!(b.moves_remaining, 1);
        assert_eq!(b.current_player, Player::One);
        b.apply_move(0, 0).unwrap();
        assert_eq!(b.current_player, Player::Two);
        assert_eq!(b.moves_remaining, 2);
    }

    #[test]
    fn subsequent_turns_have_two_moves() {
        let mut b = Board::new();
        b.apply_move(0, 0).unwrap(); // P1 ply 0
        b.apply_move(1, 0).unwrap(); // P2 first
        assert_eq!(b.current_player, Player::Two);
        assert_eq!(b.moves_remaining, 1);
        b.apply_move(2, 0).unwrap(); // P2 second — turn passes
        assert_eq!(b.current_player, Player::One);
        assert_eq!(b.moves_remaining, 2);
    }



    #[test]
    fn occupied_cell_rejected() {
        let mut b = Board::new();
        b.apply_move(0, 0).unwrap();
        assert!(b.apply_move(0, 0).is_err());
    }

    #[test]
    fn legal_moves_counts_empty_cells() {
        let mut b = Board::new();
        // Empty board: 5×5 init region = 25 cells (MCTS-optimised first-move).
        assert_eq!(b.legal_move_count(), 25);
        b.apply_move(0, 0).unwrap();
        // bbox+2 margin = [-2,2]×[-2,2] = 25 cells, minus 1 occupied = 24.
        assert_eq!(b.legal_move_count(), 24);
    }

    #[test]
    fn zobrist_changes_on_each_move() {
        let mut b = Board::new();
        let h0 = b.zobrist_hash;
        b.apply_move(0, 0).unwrap();
        let h1 = b.zobrist_hash;
        b.apply_move(1, 0).unwrap();
        let h2 = b.zobrist_hash;
        assert_ne!(h0, h1);
        assert_ne!(h1, h2);
        assert_ne!(h0, h2);
    }

    #[test]
    fn win_e_axis_player_one() {
        // P1: (0,0)…(5,0). P2 fillers on different rows.
        let mut b = Board::new();
        b.apply_move(0, 0).unwrap();
        b.apply_move(-9, 5).unwrap(); b.apply_move(-9, 6).unwrap();
        b.apply_move(1, 0).unwrap(); b.apply_move(2, 0).unwrap();
        b.apply_move(-9, 7).unwrap(); b.apply_move(-9, 8).unwrap();
        b.apply_move(3, 0).unwrap(); b.apply_move(4, 0).unwrap();
        b.apply_move(-9, -5).unwrap(); b.apply_move(-9, -6).unwrap();
        b.apply_move(5, 0).unwrap();
        assert!(b.player_wins(Player::One), "P1 should win along E axis");
        assert!(!b.player_wins(Player::Two), "P2 fillers must not win");
    }

    #[test]
    fn win_ne_axis_player_one() {
        // NE axis: (0,0),(0,1),(0,2),(0,3),(0,4),(0,5)
        let mut b = Board::new();
        b.apply_move(0, 0).unwrap();
        b.apply_move(-1, 0).unwrap(); b.apply_move(-2, 0).unwrap();
        b.apply_move(0, 1).unwrap(); b.apply_move(0, 2).unwrap();
        b.apply_move(-3, 0).unwrap(); b.apply_move(-4, 0).unwrap();
        b.apply_move(0, 3).unwrap(); b.apply_move(0, 4).unwrap();
        b.apply_move(-5, 0).unwrap(); b.apply_move(-6, 0).unwrap();
        b.apply_move(0, 5).unwrap();
        assert!(b.player_wins(Player::One), "P1 should win along NE axis");
    }

    #[test]
    fn win_nw_axis_player_one() {
        // NW axis: (0,0),(-1,1),(-2,2),(-3,3),(-4,4),(-5,5)
        let mut b = Board::new();
        b.apply_move(0, 0).unwrap();
        b.apply_move(1, 0).unwrap(); b.apply_move(2, 0).unwrap();
        b.apply_move(-1, 1).unwrap(); b.apply_move(-2, 2).unwrap();
        b.apply_move(3, 0).unwrap(); b.apply_move(4, 0).unwrap();
        b.apply_move(-3, 3).unwrap(); b.apply_move(-4, 4).unwrap();
        b.apply_move(5, 0).unwrap(); b.apply_move(6, 0).unwrap();
        b.apply_move(-5, 5).unwrap();
        assert!(b.player_wins(Player::One), "P1 should win along NW axis");
    }

    #[test]
    fn five_in_row_is_not_win() {
        let mut b = Board::new();
        b.apply_move(0, 0).unwrap();
        b.apply_move(-1, -1).unwrap(); b.apply_move(-2, -2).unwrap();
        b.apply_move(1, 0).unwrap(); b.apply_move(2, 0).unwrap();
        b.apply_move(-3, -3).unwrap(); b.apply_move(-4, -4).unwrap();
        b.apply_move(3, 0).unwrap(); b.apply_move(4, 0).unwrap();
        // P1 has (0,0),(1,0),(2,0),(3,0),(4,0) = 5 in a row — not a win
        assert!(!b.check_win(), "5 in a row should not be a win");
    }

    #[test]
    fn win_player_two() {
        // P2 builds 6 along E: (0,-1)..(5,-1).  P1 fillers at r=3..6.
        let mut b = Board::new();
        b.apply_move(0, 0).unwrap(); // P1 single first move
        b.apply_move(0, -1).unwrap(); b.apply_move(1, -1).unwrap();
        b.apply_move(0, 3).unwrap(); b.apply_move(0, 4).unwrap();
        b.apply_move(2, -1).unwrap(); b.apply_move(3, -1).unwrap();
        b.apply_move(0, 5).unwrap(); b.apply_move(0, 6).unwrap();
        b.apply_move(4, -1).unwrap(); b.apply_move(5, -1).unwrap();
        assert!(b.player_wins(Player::Two), "P2 should win along E axis");
        assert!(!b.player_wins(Player::One), "P1 fillers must not win");
    }

    #[test]
    fn win_at_board_edge() {
        // P2 builds 6 along NE (q=8, r=-2..3) near right edge of view window.
        // P1 fillers at q=-1,-2,-3,-4 (E axis, only 4 stones).
        let mut b = Board::new();
        b.apply_move(0, 0).unwrap(); // P1 single first move
        b.apply_move(8, -2).unwrap(); b.apply_move(8, -1).unwrap();
        b.apply_move(-1, 0).unwrap(); b.apply_move(-2, 0).unwrap();
        b.apply_move(8, 0).unwrap(); b.apply_move(8, 1).unwrap();
        b.apply_move(-3, 0).unwrap(); b.apply_move(-4, 0).unwrap();
        b.apply_move(8, 2).unwrap(); b.apply_move(8, 3).unwrap();
        assert!(b.player_wins(Player::Two), "P2 wins near right window edge at q=8");
        assert!(!b.player_wins(Player::One), "P1 fillers (4 in a row) must not win");
    }

    // ── New sliding-window tests ───────────────────────────────────────────────

    #[test]
    // Tests the Rust hot-path `to_planes()` method (single-window, 18-plane encoding).
    // This is NOT the Python-side split-responsibility path (get_cluster_views returns 2 planes).
    fn to_planes_empty_board_all_zeros() {
        let b = Board::new();
        let planes = b.to_planes();
        assert_eq!(planes.len(), 18 * TOTAL_CELLS);
        assert!(planes.iter().all(|x| *x == 0.0), "empty board planes must be all zero");
    }

    #[test]
    // Tests internal single-window helpers (window_center / in_window) used by
    // MCTS move generation and the Rust hot-path to_planes(). NOT the multi-cluster
    // get_cluster_views path used by Python GameState.
    fn single_window_center_slides_with_bbox() {
        // After P1@(0,0) and P2@(8,0) the window must slide right.
        // Both stones must remain visible; the left side must also be accessible.
        let mut b = Board::new();
        b.apply_move(0, 0).unwrap(); // P1
        b.apply_move(8, 0).unwrap(); // P2 — forces window right
        // Bounding box: [0,8]×[0,0]; centre = (4,0)
        assert_eq!(b.window_center(), (4, 0));
        // (0,0) and (8,0) must both be within the 19×19 window
        assert!(b.in_window(0, 0), "left stone must remain in window");
        assert!(b.in_window(8, 0), "right stone must remain in window");
        // Left edge of window is now 4-9 = -5; right edge is 4+9 = 13
        assert!(b.in_window(-5, 0), "left window edge must be reachable");
        assert!(!b.in_window(-6, 0), "one beyond left edge must be out-of-window");
    }

    #[test]
    fn legal_grows_with_bounding_box() {
        let mut b = Board::new();
        b.apply_move(0, 0).unwrap(); // P1 ply0: cluster bbox+2=[-2,2]×[-2,2] → 24 legal
        assert_eq!(b.legal_move_count(), 24);
        b.apply_move(5, 0).unwrap(); // P2: same cluster, bbox+2=[-2,7]×[-2,2]=50-2=48
        assert_eq!(b.legal_move_count(), 48);
    }

    #[test]
    fn test_action_anchors_tracking() {
        let mut board = Board::new();
        assert_eq!(board.action_anchors_count, 0);

        board.apply_move(0, 0).unwrap();
        assert_eq!(board.action_anchors_count, 1);
        assert_eq!(board.action_anchors[0], (0, 0));

        board.apply_move(1, 1).unwrap();
        board.apply_move(2, 2).unwrap();
        board.apply_move(3, 3).unwrap();
        assert_eq!(board.action_anchors_count, 4);
        assert_eq!(board.action_anchors[0], (0, 0));
        assert_eq!(board.action_anchors[3], (3, 3));

        board.apply_move(4, 4).unwrap();
        assert_eq!(board.action_anchors_count, 4);
        assert_eq!(board.action_anchors[0], (1, 1)); // (0,0) was evicted
        assert_eq!(board.action_anchors[3], (4, 4));
    }

    #[test]
    fn test_threat_anchors_identification() {
        let mut board = Board::new();
        // Place a 3-in-a-row for P1 along the E axis: (0,0), (1,0), (2,0)
        board.cells.insert((0, 0), Cell::P1);
        board.cells.insert((1, 0), Cell::P1);
        board.cells.insert((2, 0), Cell::P1);

        // It's open at both ends ((-1,0) and (3,0) are Empty)
        let anchors = board.get_threat_anchors();
        assert_eq!(anchors.len(), 1);
        assert_eq!(anchors[0], (1, 0)); // The center stone

        // Now place another 4-in-a-row for P2 along the NE axis: (5,5), (5,6), (5,7), (5,8)
        board.cells.insert((5, 5), Cell::P2);
        board.cells.insert((5, 6), Cell::P2);
        board.cells.insert((5, 7), Cell::P2);
        board.cells.insert((5, 8), Cell::P2);

        let anchors = board.get_threat_anchors();
        assert_eq!(anchors.len(), 2);
        // Centers for 4-in-a-row: count / 2 = 2. So (5, 5 + 2) = (5, 7).
        assert!(anchors.contains(&(1, 0)));
        assert!(anchors.contains(&(5, 7)));
    }

    #[test]
    fn test_apply_undo_symmetry() {
        let mut board = Board::new();
        let mut diffs = Vec::new();
        let mut seed = 0x5eed_1234_5678_90abu64;

        for _ in 0..10 {
            let legal = board.legal_moves();
            assert!(!legal.is_empty(), "expected at least one legal move");
            let idx = (next_u64(&mut seed) as usize) % legal.len();
            let (q, r) = legal[idx];

            let diff = board.apply_move_tracked(q, r).expect("move should be legal");
            diffs.push(diff);

            assert_eq!(board.zobrist_hash, recompute_zobrist(&board));
            assert!(board.has_stones);
            assert!(board.min_q <= board.max_q);
            assert!(board.min_r <= board.max_r);
            assert!(q >= board.min_q && q <= board.max_q);
            assert!(r >= board.min_r && r <= board.max_r);
        }

        while let Some(diff) = diffs.pop() {
            board.undo_move(diff);
        }

        let empty = Board::new();
        assert_eq!(board.cells.len(), empty.cells.len());
        assert_eq!(board.zobrist_hash, empty.zobrist_hash);
        assert_eq!(board.min_q, empty.min_q);
        assert_eq!(board.max_q, empty.max_q);
        assert_eq!(board.min_r, empty.min_r);
        assert_eq!(board.max_r, empty.max_r);
        assert_eq!(board.has_stones, empty.has_stones);
        assert_eq!(board.current_player, empty.current_player);
        assert_eq!(board.moves_remaining, empty.moves_remaining);
        assert_eq!(board.ply, empty.ply);
        assert_eq!(board.last_move, empty.last_move);
        assert_eq!(board.action_anchors_count, empty.action_anchors_count);
        assert_eq!(board.legal_moves_set(), empty.legal_moves_set(), "undo must restore legal moves to the initial 25-cell set");
    }

    #[test]
    fn cluster_views_returns_two_planes() {
        let mut b = Board::new();
        b.apply_move(0, 0).unwrap(); // P1 at origin; turn passes to P2
        let (views, centers) = b.get_cluster_views();
        assert_eq!(views.len(), 1, "one cluster expected");
        assert_eq!(centers.len(), 1);
        assert_eq!(
            views[0].len(),
            2 * TOTAL_CELLS,
            "get_cluster_views must return 2-plane views (2 * 361 = 722 floats)"
        );
        // Current player is P2. Plane 0 = P2 (current, no stones), Plane 1 = P1 (opponent).
        // P1 stone at origin → flat index = HALF * BOARD_SIZE + HALF = 9*19+9 = 180.
        let origin_flat = (HALF as usize) * BOARD_SIZE + (HALF as usize);
        assert_eq!(views[0][TOTAL_CELLS + origin_flat], 1.0,
            "P1 stone should be in opponent plane (offset TOTAL_CELLS)");
        assert_eq!(views[0][origin_flat], 0.0,
            "current player (P2) has no stones yet");
    }
}

// ── Property-based tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod prop_tests {
    use super::*;
    use proptest::prelude::*;

    // removed: superseded by splitmix128
    proptest! {
        /// apply_move_tracked + undo_move always restores the exact 128-bit Zobrist hash.
        ///
        /// Generates a random sequence of up to 20 moves, applies all of them, then
        /// undoes them all in reverse order. The final hash must equal the initial hash
        /// of an empty board, regardless of the move sequence.
        #[test]
        fn undo_restores_hash(indices in proptest::collection::vec(0usize..361, 1..=20usize)) {
            let mut board = Board::new();
            let initial_hash = board.zobrist_hash;
            let mut diffs = Vec::new();

            for idx in &indices {
                let legal = board.legal_moves();
                if legal.is_empty() { break; }
                let (q, r) = legal[idx % legal.len()];
                let diff = board.apply_move_tracked(q, r).expect("legal move must succeed");
                diffs.push(diff);
            }

            while let Some(diff) = diffs.pop() {
                board.undo_move(diff);
            }

            prop_assert_eq!(
                board.zobrist_hash,
                initial_hash,
                "undo sequence must restore the exact 128-bit Zobrist hash"
            );
            prop_assert_eq!(board.ply, 0, "undo must restore ply to 0");
            prop_assert_eq!(board.cells.len(), 0, "undo must restore empty board");
            prop_assert_eq!(board.has_stones, false, "has_stones must be false after full undo");
        }

        /// Partial undo/redo: undo k steps then replay the same k moves → same hash.
        ///
        /// This proves that undo_move is perfectly inverse to apply_move, not just for
        /// a full sequence but for any prefix of arbitrary length.
        #[test]
        fn partial_undo_redo_preserves_hash(
            indices in proptest::collection::vec(0usize..361, 2..=15usize),
            undo_fraction in 1usize..=8usize,
        ) {
            let mut board = Board::new();
            let mut applied: Vec<(i32, i32)> = Vec::new();
            let mut diffs: Vec<MoveDiff> = Vec::new();

            for idx in &indices {
                let legal = board.legal_moves();
                if legal.is_empty() { break; }
                let (q, r) = legal[idx % legal.len()];
                let diff = board.apply_move_tracked(q, r).expect("legal move must succeed");
                applied.push((q, r));
                diffs.push(diff);
            }

            if applied.is_empty() { return Ok(()); }

            let hash_after_all = board.zobrist_hash;
            let k = undo_fraction.min(applied.len());

            // Undo the last k moves
            let to_redo: Vec<(i32, i32)> = applied[applied.len() - k..].to_vec();
            for _ in 0..k {
                let diff = diffs.pop().unwrap();
                board.undo_move(diff);
            }

            // Replay the same k moves
            for (q, r) in &to_redo {
                board.apply_move(*q, *r).expect("replayed move must be legal");
            }

            prop_assert_eq!(
                board.zobrist_hash,
                hash_after_all,
                "partial undo then redo must restore the exact 128-bit hash"
            );
        }
    }
}
