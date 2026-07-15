//! BUILD-HOT criterion bench for the once-per-leaf axis-graph builder.
//!
//! WP-A verdict BUILD-HOT (`reports/probes/gnn_integration/WPA_cuda_bench.md`):
//! the strix Rust-builder proxy is 0.539 ms/pos = 77-161% of the GNN forward,
//! so the build is a hot path and WP-1 carries a perf sub-package. This bench
//! deserializes the REAL WP-A self-play position set ONCE, then times build
//! only (no I/O, no parallelism — the caller parallelizes over leaves).
//!
//! Run (the RUSTFLAGS override is REQUIRED — the workspace `profile.release`
//! sets `panic = "abort"` for the MCTS hot path, which criterion's harness
//! cannot link against; `-C panic=unwind` restores unwinding for the bench
//! build only, and this crate's `cdylib` artifact makes the mismatch a hard
//! error otherwise):
//!
//!   RUSTFLAGS="-C panic=unwind" cargo bench -j4 -p hexo-graph --features harness
//!
//! Targets: beat the 0.539 ms/pos strix proxy; contract budget ≤1.5 ms/pos.
//! Median ns/pos is the headline number recorded in WP1_builder.md.

// Fixture-bounded JSON-parse casts; silence the pedantic cast lints.
#![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_possible_wrap, clippy::doc_markdown)]

use std::path::PathBuf;

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use hexo_graph::{build_axis_graph, BuildParams, StoneList};
use serde_json::Value;

fn load_positions() -> Vec<(StoneList, BuildParams)> {
    // Frozen WP-A self-play set, resolved relative to the workspace root.
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop(); // engine/
    p.pop(); // workspace root
    p.push("reports/probes/gnn_integration/wpa_positions.json");
    let raw = std::fs::read_to_string(&p)
        .unwrap_or_else(|e| panic!("read {}: {e}", p.display()));
    let root: Value = serde_json::from_str(&raw).expect("parse");
    let positions = root["positions"].as_array().expect("positions");
    positions
        .iter()
        .map(|pos| {
            let stones = pos["stones"]
                .as_array()
                .unwrap()
                .iter()
                .map(|s| {
                    let a = s.as_array().unwrap();
                    (
                        a[0].as_i64().unwrap() as i32,
                        a[1].as_i64().unwrap() as i32,
                        a[2].as_i64().unwrap() as i8,
                    )
                })
                .collect();
            let params = BuildParams {
                win_length: 6,
                radius: 6,
                current_player: pos["current_player"].as_i64().unwrap() as i8,
                moves_remaining: pos["moves_remaining"].as_i64().unwrap() as u8,
                trunk_size: 19,
            };
            (StoneList { stones }, params)
        })
        .collect()
}

fn bench_build(c: &mut Criterion) {
    let set = load_positions();
    let n = set.len();

    // Per-position throughput: one build per iteration, cycling the set so the
    // reported time is ns/pos over the real distribution (mean 490 nodes).
    let mut idx = 0usize;
    let mut group = c.benchmark_group("axis_graph_build");
    group.throughput(criterion::Throughput::Elements(1));
    group.bench_function("per_position", |b| {
        b.iter_batched(
            || {
                let cur = idx % n;
                idx += 1;
                &set[cur]
            },
            |(stones, params)| build_axis_graph(std::hint::black_box(stones), std::hint::black_box(params)),
            BatchSize::SmallInput,
        );
    });
    group.finish();

    // Whole-set sweep: build all N once, for a stable aggregate median.
    let mut g2 = c.benchmark_group("axis_graph_build_full_set");
    g2.throughput(criterion::Throughput::Elements(n as u64));
    g2.bench_function("all_positions", |b| {
        b.iter(|| {
            for (stones, params) in &set {
                std::hint::black_box(build_axis_graph(stones, params));
            }
        });
    });
    g2.finish();
}

criterion_group!(benches, bench_build);
criterion_main!(benches);
