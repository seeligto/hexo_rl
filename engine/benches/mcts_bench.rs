/// Criterion micro-benchmarks for MCTS and win detection.
///
/// Run with:
///   cargo bench --bench mcts_bench
///   cargo bench --bench mcts_bench -- --output-format html

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use native_core::{board::Board, mcts::MCTSTree};

fn bench_win_detection(c: &mut Criterion) {
    // Place 5 stones in a row (no win) to make the check non-trivial.
    let mut board = Board::new();
    for i in 0..5i32 {
        board.apply_move(i - 2, 0).unwrap();
        // Move back to player 1 by playing on the other player's behalf.
        if i < 4 {
            board.apply_move(i - 2, -5).unwrap();
        }
    }
    c.bench_function("win_check_5_in_row_no_win", |b| {
        b.iter(|| board.check_win())
    });

    // Empty board baseline.
    let empty = Board::new();
    c.bench_function("win_check_empty_board", |b| {
        b.iter(|| empty.check_win())
    });
}

fn bench_mcts_simulations(c: &mut Criterion) {
    let mut group = c.benchmark_group("mcts_sims_cpu_only");
    for &n in &[100u64, 400, 800] {
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            let board = Board::new();
            let mut tree = MCTSTree::new(1.5);
            tree.new_game(board);
            b.iter(|| {
                tree.run_simulations_cpu_only(n as usize);
                tree.reset();
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_win_detection, bench_mcts_simulations);
criterion_main!(benches);
