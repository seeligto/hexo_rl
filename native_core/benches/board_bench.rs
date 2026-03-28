use criterion::{criterion_group, criterion_main, Criterion};
use native_core::board::{Board, Player};

fn bench_win_check(c: &mut Criterion) {
    let mut board = Board::new();
    board.apply_move(0, 0).unwrap();
    c.bench_function("win_check_empty", |b| b.iter(|| board.check_win()));
}

criterion_group!(benches, bench_win_check);
criterion_main!(benches);
