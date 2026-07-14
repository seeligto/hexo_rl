//! Native JSON parity harness for `hexo-graph` (feature `harness` only).
//!
//! Bridges the Rust builder to the pytest oracle parity test WITHOUT touching
//! the `engine` crate or adding PyO3 (that is WP-3's seam). Reads a positions
//! JSON, builds each axis-graph, writes the full wire payload as JSON arrays
//! that `tests/test_hexo_graph_parity.py` compares against
//! `build_axis_graph_raw`.
//!
//! Build:  cargo build --release -j4 -p hexo-graph --features harness
//! Run:    hexo_graph_harness <positions.json> [out.json]   (else stdout)
//!
//! Input schema (top-level object):
//!   { "win_length": 6, "radius": 6, "trunk_size": 19,   # optional defaults
//!     "positions": [ { "stones": [[q,r,p], ...],
//!                      "current_player": 1, "moves_remaining": 2 }, ... ] }
//! Per-position win_length/radius/trunk_size override the top-level defaults.

// JSON-parse casts (`serde_json` i64 → the builder's i32/i8/u8/u16 inputs) are
// bounded by the fixture and intentional; silence the pedantic cast lints.
#![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_possible_wrap, clippy::doc_markdown)]

use std::io::Write;

use hexo_graph::{build_axis_graph, BuildParams, StoneList, OFF_WINDOW_SLOT};
use serde_json::{json, Value};

fn get_i64(v: &Value, k: &str, dflt: i64) -> i64 {
    v.get(k).and_then(Value::as_i64).unwrap_or(dflt)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: {} <positions.json> [out.json]", args[0]);
        std::process::exit(2);
    }
    let raw = std::fs::read_to_string(&args[1]).expect("read positions json");
    let root: Value = serde_json::from_str(&raw).expect("parse positions json");

    let d_win = get_i64(&root, "win_length", 6);
    let d_rad = get_i64(&root, "radius", 6);
    let d_trunk = get_i64(&root, "trunk_size", 19);

    let positions = root
        .get("positions")
        .and_then(Value::as_array)
        .expect("positions array");

    let mut out: Vec<Value> = Vec::with_capacity(positions.len());
    for pos in positions {
        let stones_v = pos.get("stones").and_then(Value::as_array).expect("stones");
        let mut stones: Vec<(i32, i32, i8)> = Vec::with_capacity(stones_v.len());
        for s in stones_v {
            let a = s.as_array().expect("stone triple");
            stones.push((
                a[0].as_i64().unwrap() as i32,
                a[1].as_i64().unwrap() as i32,
                a[2].as_i64().unwrap() as i8,
            ));
        }
        let params = BuildParams {
            win_length: get_i64(pos, "win_length", d_win) as u8,
            radius: get_i64(pos, "radius", d_rad) as u16,
            current_player: get_i64(pos, "current_player", 1) as i8,
            moves_remaining: get_i64(pos, "moves_remaining", 2) as u8,
            trunk_size: get_i64(pos, "trunk_size", d_trunk) as i32,
        };
        let g = build_axis_graph(&StoneList { stones }, &params);

        // OFF_WINDOW_SLOT -> -1 for a clean JSON int the test can special-case.
        let slots: Vec<i64> = g
            .policy_scatter_index
            .0
            .iter()
            .map(|&s| if s == OFF_WINDOW_SLOT { -1 } else { i64::from(s) })
            .collect();

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
            "policy_dst_slot": slots,
            "legal_node_gather": g.legal_node_gather,
            "window_center": [g.window_center.0, g.window_center.1],
            "current_player": g.current_player,
            "builder_impl": g.builder_impl,
        }));
    }

    let payload = Value::Array(out);
    let text = serde_json::to_string(&payload).expect("serialize");
    if args.len() >= 3 {
        std::fs::write(&args[2], text).expect("write out json");
    } else {
        std::io::stdout().write_all(text.as_bytes()).expect("stdout");
    }
}
