/// Criterion benchmarks for board operations.
///
/// Benchmarks:
///   1. Win detection: Bitboard::has_six_in_row() vs Board::check_win() (HashMap scan)
///   2. Board::clone() cost at various game depths (bottleneck in reconstruct_board)
///   3. Zobrist hash incremental update (should be near-zero marginal cost)
///
/// Run:
///   cargo bench --bench board_bench
///   cargo bench --bench board_bench -- --output-format html
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use native_core::board::{bitboard::Bitboard, Board, BOARD_SIZE, HALF};

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Build a board with `n_stones` placed by always picking the lexicographic-minimum
/// legal move.  Guaranteed collision-free and deterministic.
fn board_with_n_stones(n_stones: usize) -> Board {
    let mut b = Board::new();
    for _ in 0..n_stones {
        let mv = *b.legal_moves_set()
            .iter()
            .min()
            .expect("no legal moves");
        b.apply_move(mv.0, mv.1).expect("apply failed");
    }
    b
}

/// Build a Bitboard for player 1 from a Board's stone layout.
///
/// Uses window-relative flat indices (same as the Bitboard's coordinate system).
fn bitboard_from_board_p1(board: &Board) -> Bitboard {
    use native_core::board::Cell;
    let mut bb = Bitboard::empty();
    let (cq, cr) = board.window_center();
    for (&(q, r), &cell) in board.cells_iter() {
        if cell == Cell::P1 {
            let wq = q - cq + HALF;
            let wr = r - cr + HALF;
            if wq >= 0 && wq < BOARD_SIZE as i32 && wr >= 0 && wr < BOARD_SIZE as i32 {
                let flat = wq as usize * BOARD_SIZE + wr as usize;
                bb.set(flat);
            }
        }
    }
    bb
}

// ── 1. Win detection: HashMap scan vs Bitboard ───────────────────────────────

fn bench_win_detection_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("win_detection");

    for &n_stones in &[10usize, 20, 40] {
        let board = board_with_n_stones(n_stones);
        let bb = bitboard_from_board_p1(&board);

        group.bench_with_input(
            BenchmarkId::new("hashmap_check_win", n_stones),
            &n_stones,
            |b, _| b.iter(|| black_box(board.check_win())),
        );

        group.bench_with_input(
            BenchmarkId::new("bitboard_has_six_in_row", n_stones),
            &n_stones,
            |b, _| b.iter(|| black_box(bb.has_six_in_row())),
        );
    }

    // Win case: 6-in-a-row on E axis for both methods.
    let mut win_board = Board::new();
    win_board.apply_move(0, 0).unwrap();
    win_board.apply_move(-9, 5).unwrap();
    win_board.apply_move(-9, 6).unwrap();
    win_board.apply_move(1, 0).unwrap();
    win_board.apply_move(2, 0).unwrap();
    win_board.apply_move(-9, 7).unwrap();
    win_board.apply_move(-9, 8).unwrap();
    win_board.apply_move(3, 0).unwrap();
    win_board.apply_move(4, 0).unwrap();
    win_board.apply_move(-9, -5).unwrap();
    win_board.apply_move(-9, -6).unwrap();
    win_board.apply_move(5, 0).unwrap();
    let win_bb = bitboard_from_board_p1(&win_board);

    group.bench_function("hashmap_check_win_TRUE", |b| {
        b.iter(|| black_box(win_board.check_win()))
    });
    group.bench_function("bitboard_has_six_in_row_TRUE", |b| {
        b.iter(|| black_box(win_bb.has_six_in_row()))
    });

    group.finish();
}

// ── 2. Board::clone() cost — the MCTS reconstruct_board bottleneck ────────────

fn bench_board_clone(c: &mut Criterion) {
    let mut group = c.benchmark_group("board_clone");

    for &n_stones in &[5usize, 15, 30, 60] {
        let board = board_with_n_stones(n_stones);

        group.bench_with_input(
            BenchmarkId::new("clone_n_stones", n_stones),
            &n_stones,
            |b, _| b.iter(|| black_box(board.clone())),
        );
    }

    group.finish();
}

// ── 3. Reconstruct-board path replay cost (simulates MCTS select_one_leaf) ───

/// Simulate what MCTSTree::reconstruct_board does: clone root + replay D moves.
fn simulate_reconstruct(root: &Board, path: &[(i32, i32)]) -> Board {
    let mut b = root.clone();
    for &(q, r) in path {
        b.apply_move(q, r).unwrap();
    }
    b
}

fn bench_reconstruct_path_replay(c: &mut Criterion) {
    let mut group = c.benchmark_group("reconstruct_board_path_replay");

    // Build a root board and a sequence of legal moves to replay.
    let root = Board::new();

    // Pre-compute a sequence of moves that can be applied from an empty board.
    // Interleave P1 and P2 moves to respect turn structure.
    // First move: P1 plays (0,0). Then pairs: P2 plays (-1,-1),(-2,-2); P1 plays (1,1),(2,2) ...
    let all_moves: Vec<(i32, i32)> = {
        let mut moves = Vec::new();
        moves.push((0, 0));         // P1 single first move
        // Depth 2-3: P2 turn
        moves.push((-1, -1));
        moves.push((-2, -2));
        // Depth 4-5: P1 turn
        moves.push((1, 1));
        moves.push((2, 2));
        // Depth 6-7: P2 turn
        moves.push((-3, -3));
        moves.push((-4, -4));
        // Depth 8-9: P1 turn
        moves.push((3, 3));
        moves.push((4, 4));
        // Depth 10-11: P2 turn
        moves.push((-5, -5));
        moves.push((-6, -6));
        // Depth 12-13: P1 turn
        moves.push((5, 0));
        moves.push((6, 0));
        // Depth 14-15: P2 turn
        moves.push((-7, 1));
        moves.push((-8, 1));
        // Depth 16-17: P1 turn
        moves.push((0, 5));
        moves.push((0, 6));
        // Depth 18-19: P2 turn
        moves.push((1, -5));
        moves.push((1, -6));
        moves
    };

    for &depth in &[5usize, 10, 15, 20] {
        let path = &all_moves[..depth.min(all_moves.len())];

        group.bench_with_input(
            BenchmarkId::new("clone_and_replay", depth),
            &depth,
            |b, _| b.iter(|| black_box(simulate_reconstruct(&root, path))),
        );
    }

    group.finish();
}

// ── 4. Zobrist: incremental XOR (confirms near-zero marginal cost) ────────────

fn bench_zobrist_incremental(c: &mut Criterion) {
    // apply_move already XORs the Zobrist key inline.
    // This bench measures the total apply_move cost as a proxy.
    let mut group = c.benchmark_group("zobrist_incremental");

    group.bench_function("apply_move_with_zobrist_update", |b| {
        b.iter(|| {
            let mut board = Board::new();
            black_box(board.apply_move(0, 0)).unwrap();
            black_box(board.zobrist_hash)
        })
    });

    group.finish();
}

// ── Registration ──────────────────────────────────────────────────────────────

criterion_group!(
    benches,
    bench_win_detection_comparison,
    bench_board_clone,
    bench_reconstruct_path_replay,
    bench_zobrist_incremental,
);
criterion_main!(benches);
