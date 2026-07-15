//! Native JSON parity harness for `hexo-graph` (feature `harness` only).
//!
//! Bridges the Rust builder to the pytest oracle parity test WITHOUT touching
//! the `engine` crate or adding PyO3 (that is WP-3's seam). Reads a positions
//! JSON, builds each axis-graph, writes the payloads as JSON arrays that
//! `tests/test_hexo_graph_parity.py` compares against `build_axis_graph_raw`.
//!
//! Build:  cargo build --release -j4 -p hexo-graph --features harness
//! Run:    hexo_graph_harness <positions.json> [out.json]   (else stdout)
//!
//! Input schema (top-level object):
//!   { "win_length": 6, "radius": 6, "trunk_size": 19,   # optional defaults
//!     "positions": [ { "stones": [[q,r,p], ...],
//!                      "current_player": 1, "moves_remaining": 2,
//!                      ...optional per-position win_length/radius/trunk_size
//!                    }, ... ] }
//!
//! Parsing is STRICT / die-loud (red-team Attack-4 boundary note + review #9):
//! a present-but-wrong-typed field is an ERROR (never a silent default; only
//! an ABSENT optional field falls back), and every narrowing cast is
//! range-validated first — `moves_remaining` must fit u8, `current_player`
//! must fit i8, etc. — so a `-1` can never wrap to `255` on its way into
//! `BuildParams`.
//!
//! Output schema: `{ "harness_schema_version": 1, "graphs": [ ... ] }`. This
//! is TEST SCAFFOLDING, not the WP-3 wire contract — the version field exists
//! so it can't be silently vendored forward as-is (review #10); the real
//! contract's `contract_version`/`builder_impl` handshake lives in the WP-3
//! resolver.

// JSON-parse casts are range-validated by `int_in` before narrowing.
#![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_possible_wrap, clippy::doc_markdown)]

use std::io::Write;

use hexo_graph::{build_axis_graph, BuildParams, StoneList};
use serde_json::{json, Value};

/// Harness OUTPUT schema version (test scaffolding, not the wire contract).
const HARNESS_SCHEMA_VERSION: u32 = 1;

/// Read field `k`: REQUIRED — dies loud when missing OR wrong-typed.
fn req_i64(v: &Value, k: &str, ctx: &str) -> i64 {
    match v.get(k) {
        Some(x) => x
            .as_i64()
            .unwrap_or_else(|| panic!("{ctx}: field '{k}' present but not an integer: {x}")),
        None => panic!("{ctx}: required field '{k}' missing"),
    }
}

/// Read field `k`: OPTIONAL — defaults ONLY when ABSENT; a present-but-
/// wrong-typed field dies loud (review #9).
fn opt_i64(v: &Value, k: &str, dflt: i64, ctx: &str) -> i64 {
    match v.get(k) {
        Some(x) => x
            .as_i64()
            .unwrap_or_else(|| panic!("{ctx}: field '{k}' present but not an integer: {x}")),
        None => dflt,
    }
}

/// Range-validate BEFORE any narrowing cast (red-team Attack-4 note).
fn int_in(val: i64, lo: i64, hi: i64, what: &str, ctx: &str) -> i64 {
    assert!(
        (lo..=hi).contains(&val),
        "{ctx}: {what} = {val} outside [{lo}, {hi}] (refusing the narrowing cast)"
    );
    val
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: {} <positions.json> [out.json]", args[0]);
        std::process::exit(2);
    }
    let raw = std::fs::read_to_string(&args[1]).expect("read positions json");
    let root: Value = serde_json::from_str(&raw).expect("parse positions json");

    let d_win = opt_i64(&root, "win_length", 6, "top-level");
    let d_rad = opt_i64(&root, "radius", 6, "top-level");
    let d_trunk = opt_i64(&root, "trunk_size", 19, "top-level");

    let positions = root
        .get("positions")
        .and_then(Value::as_array)
        .expect("positions array");

    let mut out: Vec<Value> = Vec::with_capacity(positions.len());
    for (pi, pos) in positions.iter().enumerate() {
        let ctx = format!("position {pi}");
        let stones_v = pos
            .get("stones")
            .and_then(Value::as_array)
            .unwrap_or_else(|| panic!("{ctx}: 'stones' missing or not an array"));
        let mut stones: Vec<(i32, i32, i8)> = Vec::with_capacity(stones_v.len());
        for (si, s) in stones_v.iter().enumerate() {
            let a = s
                .as_array()
                .unwrap_or_else(|| panic!("{ctx}: stone {si} not an array"));
            assert!(a.len() == 3, "{ctx}: stone {si} is not a [q, r, player] triple");
            let read = |j: usize, what: &str| -> i64 {
                a[j].as_i64()
                    .unwrap_or_else(|| panic!("{ctx}: stone {si} {what} not an integer"))
            };
            stones.push((
                int_in(read(0, "q"), i64::from(i32::MIN), i64::from(i32::MAX), "stone q", &ctx) as i32,
                int_in(read(1, "r"), i64::from(i32::MIN), i64::from(i32::MAX), "stone r", &ctx) as i32,
                int_in(read(2, "player"), i64::from(i8::MIN), i64::from(i8::MAX), "stone player", &ctx) as i8,
            ));
        }
        let params = BuildParams {
            win_length: int_in(opt_i64(pos, "win_length", d_win, &ctx), 1, 32, "win_length", &ctx) as u8,
            radius: int_in(opt_i64(pos, "radius", d_rad, &ctx), 0, i64::from(u16::MAX), "radius", &ctx) as u16,
            current_player: int_in(
                req_i64(pos, "current_player", &ctx),
                i64::from(i8::MIN), i64::from(i8::MAX), "current_player", &ctx,
            ) as i8,
            moves_remaining: int_in(
                req_i64(pos, "moves_remaining", &ctx),
                0, i64::from(u8::MAX), "moves_remaining", &ctx,
            ) as u8,
            trunk_size: int_in(opt_i64(pos, "trunk_size", d_trunk, &ctx), 1, 1_000, "trunk_size", &ctx) as i32,
        };
        let g = build_axis_graph(&StoneList { stones }, &params);

        out.push(json!({
            "num_nodes": g.num_nodes(),
            "n_edges": g.num_edges(),
            "n_stones": g.n_stones,
            "n_nodes_checksum": g.n_nodes_checksum,
            "node_feat": g.node_feat.0,
            "edge_src": g.edge_index.src,
            "edge_dst": g.edge_index.dst,
            "edge_attr": g.edge_attr.0,
            "legal_mask": g.legal_mask.iter().map(|&b| u8::from(b)).collect::<Vec<u8>>(),
            "stone_mask": g.stone_mask.iter().map(|&b| u8::from(b)).collect::<Vec<u8>>(),
            "node_coords": g.node_coords,
            // i32 on the wire per the amended contract §2.1 (WP-3 option-(b)
            // ruling); OFF_WINDOW_SLOT = -1 serializes natively.
            "policy_dst_slot": g.policy_scatter_index.0,
            "legal_node_gather": g.legal_node_gather,
            "window_center": [g.window_center.0, g.window_center.1],
            "current_player": g.current_player,
            "builder_impl": g.builder_impl,
        }));
    }

    let payload = json!({
        "harness_schema_version": HARNESS_SCHEMA_VERSION,
        "graphs": Value::Array(out),
    });
    let text = serde_json::to_string(&payload).expect("serialize");
    if args.len() >= 3 {
        std::fs::write(&args[2], text).expect("write out json");
    } else {
        std::io::stdout().write_all(text.as_bytes()).expect("stdout");
    }
}
