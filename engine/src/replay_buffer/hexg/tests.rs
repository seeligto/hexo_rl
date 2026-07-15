//! HEXG unit tests — ring push/wrap, persist round-trip + version/cross-format
//! reject, rebuild-at-sample parity vs the direct native builder, D6 aug
//! round-trip coherence, and the TRUE ADV-7 desync-detectability leg.

use super::*;
use hexo_graph::{build_axis_graph, BuildParams, StoneList};

const ENC: &str = "gnn_axis_v1";

fn sample_record() -> GraphRecord {
    GraphRecord {
        stones: vec![(0, 0, 1), (1, 0, -1), (0, 1, 1), (2, 1, -1)],
        // visit target over some legal cells (coords chosen adjacent to stones,
        // so they land in the radius-6 legal set).
        visits: vec![(2, 0, 0.5), (-1, 0, 0.3), (1, 1, 0.2)],
        current_player: -1,
        moves_remaining: 2,
        ply_index: 7,
        is_full_search: true,
        outcome: 1.0,
        value_valid: true,
        game_length: 30,
    }
}

fn unique_path(stem: &str) -> std::path::PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static C: AtomicU64 = AtomicU64::new(0);
    let n = C.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    std::env::temp_dir().join(format!("hexg_{stem}_{pid}_{nanos}_{n}.hexg"))
}

/// Group inverse of D6 element `s` under this parametrization (reflect-then-
/// rotate): reflections (s>=6) are involutions (inv=s); rotations invert to
/// `(6-n)%6`. Used to prove the aug round-trip recovers the identity.
fn inv_sym(s: usize) -> usize {
    if s >= 6 {
        s
    } else {
        (6 - s) % 6
    }
}

#[test]
fn push_read_roundtrip() {
    let mut buf = HexgBuffer::new(8, ENC).unwrap();
    let rec = sample_record();
    buf.push_record_impl(&rec, 42).unwrap();
    assert_eq!(buf.size, 1);
    assert_eq!(buf.record_at(0), rec, "record_at must invert push_record_impl");
    assert_eq!(buf.game_ids[0], 42);
}

#[test]
fn ring_wraps_and_caps_size() {
    let cap = 4;
    let mut buf = HexgBuffer::new(cap, ENC).unwrap();
    for i in 0..(cap + 3) {
        let mut rec = sample_record();
        rec.ply_index = i as u16; // tag each record distinctly
        buf.push_record_impl(&rec, i as i64).unwrap();
    }
    assert_eq!(buf.size, cap, "size caps at capacity");
    assert_eq!(buf.head, 3 % cap, "head wrapped to (7 % 4)");
    // The oldest 3 records were overwritten; the live set is ply 3..7.
    let live: std::collections::HashSet<u16> =
        (0..cap).map(|s| buf.record_at(s).ply_index).collect();
    for expect in 3u16..7 {
        assert!(live.contains(&expect), "expected ply {expect} live after wrap");
    }
}

#[test]
fn push_rejects_over_cap() {
    let mut buf = HexgBuffer::new(2, ENC).unwrap();
    let over = GraphRecord {
        stones: vec![(0, 0, 1); MAX_STONES + 1],
        ..sample_record()
    };
    assert!(buf.push_record_impl(&over, -1).is_err(), "over-MAX_STONES must die loud");
    let over_v = GraphRecord {
        visits: vec![(0, 0, 0.1); MAX_VISITS + 1],
        ..sample_record()
    };
    assert!(buf.push_record_impl(&over_v, -1).is_err(), "over-MAX_VISITS must die loud");
}

#[test]
fn persist_roundtrip_byte_identical() {
    let mut buf = HexgBuffer::new(16, ENC).unwrap();
    for i in 0..10 {
        let mut rec = sample_record();
        rec.ply_index = i;
        rec.outcome = if i % 2 == 0 { 1.0 } else { -1.0 };
        rec.value_valid = i != 3;
        buf.push_record_impl(&rec, i64::from(i)).unwrap();
    }
    let path = unique_path("roundtrip");
    buf.save_to_path_impl(path.to_str().unwrap()).unwrap();

    let mut buf2 = HexgBuffer::new(16, ENC).unwrap();
    let n = buf2.load_from_path_impl(path.to_str().unwrap()).unwrap();
    assert_eq!(n, 10);
    assert_eq!(buf2.size, 10);
    for slot in 0..10 {
        assert_eq!(
            buf.record_at(slot),
            buf2.record_at(slot),
            "record {slot} must survive save/load byte-identically"
        );
        assert_eq!(buf.game_ids[slot], buf2.game_ids[slot]);
        assert_eq!(buf.weights[slot], buf2.weights[slot], "weight must survive");
    }
    let _ = std::fs::remove_file(path);
}

#[test]
fn load_rejects_bad_version() {
    use std::io::Write;
    let path = unique_path("badver");
    {
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&HEXG_MAGIC.to_le_bytes()).unwrap();
        f.write_all(&999u32.to_le_bytes()).unwrap(); // bad version
    }
    let mut buf = HexgBuffer::new(4, ENC).unwrap();
    let err = buf.load_from_path_impl(path.to_str().unwrap()).unwrap_err();
    assert!(err.contains("not supported"), "bad version must LOUD-FAIL: {err}");
    let _ = std::fs::remove_file(path);
}

#[test]
fn load_rejects_dense_hexb_magic() {
    use std::io::Write;
    // A HEXB (dense) file handed to the HEXG loader must LOUD-FAIL on magic.
    let path = unique_path("hexb_magic");
    {
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&0x4845_5842u32.to_le_bytes()).unwrap(); // "HEXB"
        f.write_all(&9u32.to_le_bytes()).unwrap();
    }
    let mut buf = HexgBuffer::new(4, ENC).unwrap();
    let err = buf.load_from_path_impl(path.to_str().unwrap()).unwrap_err();
    assert!(err.contains("invalid magic"), "HEXB → HEXG load must reject on magic: {err}");
    let _ = std::fs::remove_file(path);
}

#[test]
fn dense_loader_rejects_hexg_file() {
    // The reverse direction: a HEXG file handed to the DENSE ReplayBuffer loader
    // must hit the HEXB magic LOUD-FAIL (cross-format safety, both directions).
    let mut buf = HexgBuffer::new(4, ENC).unwrap();
    buf.push_record_impl(&sample_record(), 0).unwrap();
    let path = unique_path("hexg_into_dense");
    buf.save_to_path_impl(path.to_str().unwrap()).unwrap();

    let mut dense = crate::replay_buffer::ReplayBuffer::new(4, "v6");
    let err = dense.load_from_path_impl(path.to_str().unwrap()).unwrap_err();
    assert!(
        err.contains("magic") || err.contains("Invalid") || err.contains("invalid"),
        "HEXG → dense HEXB load must reject on magic: {err}"
    );
    let _ = std::fs::remove_file(path);
}

#[test]
fn load_rejects_slot_geometry_mismatch() {
    use std::io::Write;
    let path = unique_path("slotgeo");
    {
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&HEXG_MAGIC.to_le_bytes()).unwrap();
        f.write_all(&HEXG_VERSION.to_le_bytes()).unwrap();
        f.write_all(&(MAX_STONES as u32 + 1).to_le_bytes()).unwrap(); // wrong max_stones
        f.write_all(&(MAX_VISITS as u32).to_le_bytes()).unwrap();
        f.write_all(&4u64.to_le_bytes()).unwrap();
        f.write_all(&0u64.to_le_bytes()).unwrap();
        f.write_all(&(ENC.len() as u32).to_le_bytes()).unwrap();
        f.write_all(ENC.as_bytes()).unwrap();
    }
    let mut buf = HexgBuffer::new(4, ENC).unwrap();
    let err = buf.load_from_path_impl(path.to_str().unwrap()).unwrap_err();
    assert!(err.contains("slot-geometry"), "slot-geometry mismatch must reject: {err}");
    let _ = std::fs::remove_file(path);
}

#[test]
fn grid_encoding_rejected_at_construction() {
    assert!(HexgBuffer::new(4, "v6").is_err(), "a grid encoding must be refused");
}

// ── rebuild-at-sample parity + builder_impl handshake ────────────────────────

#[test]
fn sample_wire_matches_direct_builder_unaugmented() {
    let mut buf = HexgBuffer::new(4, ENC).unwrap();
    let rec = sample_record();
    buf.push_record_impl(&rec, 0).unwrap();

    let (wire, targets) = buf.sample_graph_batch_impl(1, false).unwrap();
    assert_eq!(wire.t_n_graphs(), 1);
    assert_eq!(wire.t_builder_impl(), 1, "sampled wire must carry builder_impl=1 (F7)");
    assert_eq!(wire.t_contract_version(), 1);

    // Direct native build on the SAME stones (identity — augment off).
    let stones: Vec<(i32, i32, i8)> =
        rec.stones.iter().map(|&(q, r, p)| (i32::from(q), i32::from(r), p)).collect();
    let params = BuildParams {
        win_length: 6,
        radius: 6,
        current_player: rec.current_player,
        moves_remaining: rec.moves_remaining,
        trunk_size: 19,
    };
    let g = build_axis_graph(&StoneList { stones }, &params);

    // For a single graph node_off=0, so globals == locals.
    assert_eq!(wire.t_node_feat(), g.node_feat.0.as_slice(), "node_feat parity");
    assert_eq!(wire.t_node_coords(), g.node_coords.as_slice(), "node_coords parity");
    assert_eq!(wire.t_policy_dst_slot(), g.policy_scatter_index.0.as_slice(), "slot parity");
    assert_eq!(wire.t_n_stones(), &[g.n_stones], "n_stones parity");
    let mut edge_index = g.edge_index.src.iter().map(|&s| i64::from(s)).collect::<Vec<_>>();
    edge_index.extend(g.edge_index.dst.iter().map(|&d| i64::from(d)));
    assert_eq!(wire.t_edge_index(), edge_index.as_slice(), "edge_index parity");

    // Policy target: length == n_legal, each legal node gets its visit-map mass.
    let n_legal = g.legal_node_gather.len();
    assert_eq!(targets.policy_target.len(), n_legal);
    let mass: f32 = targets.policy_target.iter().sum();
    assert!((mass - 1.0).abs() < 1e-5, "target sums to ~1 (visit mass), got {mass}");
    // The stored argmax cell (2,0, 0.5) is the max — its coord must be the
    // argmax leg reported to the collate canary.
    assert_eq!(targets.argmax_valid, vec![1]);
    assert_eq!((targets.argmax_q[0], targets.argmax_r[0]), (2, 0));
}

// ── D6 aug round-trip coherence + ADV-7 TRUE desync detectability ────────────

#[test]
fn rotate_axial_roundtrips_under_inverse() {
    use super::super::sym_tables::rotate_axial;
    let coords = [(0, 0), (3, -2), (-4, 1), (5, 5), (1, -6)];
    for s in 0..12 {
        for &(q, r) in &coords {
            let (rq, rr) = rotate_axial(q, r, s);
            let (bq, br) = rotate_axial(rq, rr, inv_sym(s));
            assert_eq!((bq, br), (q, r), "s={s}: rotate then inverse must recover coord");
        }
    }
}

/// Replicate the collate argmax-canonical canary in Rust: the target-argmax
/// cell MUST be the coordinate of some legal node in the (possibly rotated)
/// graph. Returns whether the canary PASSES.
fn argmax_canary_passes(g: &hexo_graph::AxisGraph, argmax_cell: (i32, i32)) -> bool {
    g.legal_node_gather.iter().any(|&row| {
        let cq = g.node_coords[row as usize * 2];
        let cr = g.node_coords[row as usize * 2 + 1];
        (cq, cr) == argmax_cell
    })
}

#[test]
fn augmented_sample_target_is_coherent_every_element() {
    // POSITIVE (unconstructability): the REAL sample path reads the target off
    // the SAME rebuilt `legal_node_gather` order it fuses into the wire, so for
    // EVERY sampled record the target-argmax cell is a legal node of the emitted
    // graph — a graph/target desync cannot be constructed through the API (WP-3
    // S1). Drive `sample_graph_batch_impl(augment=true)` many times (each draws a
    // uniform D6 element) and assert coherence on the actual output.
    let mut buf = HexgBuffer::new(4, ENC).unwrap();
    buf.push_record_impl(&sample_record(), 0).unwrap();
    for _ in 0..48 {
        let (wire, targets) = buf.sample_graph_batch_impl(1, true).unwrap();
        if targets.argmax_valid[0] == 0 {
            continue; // all-zero target (no argmax leg)
        }
        let cell = (targets.argmax_q[0], targets.argmax_r[0]);
        // The argmax cell must be one of the wire's legal-node coords (the exact
        // collate AugRoundTrip argmax-leg check). Single graph → gather is local.
        let coords = wire.t_node_coords();
        let found = wire.t_legal_node_gather().iter().any(|&row| {
            (coords[row as usize * 2], coords[row as usize * 2 + 1]) == cell
        });
        assert!(found, "augmented sample target argmax {cell:?} must be a legal node (coherent)");
    }
}

#[test]
fn adv7_desync_is_caught_by_the_canary() {
    // NEGATIVE (ADV-7 TRUE): the single-call path forbids a graph/target desync;
    // this proves the AugRoundTrip argmax-leg canary DISCRIMINATES synced from
    // desynced. Build a graph, then POISON the target-argmax onto a cell that is
    // NOT a legal node (a STONE — occupied — cell, which the builder never lists
    // as legal). A target whose peak lands on a non-legal node is exactly the
    // graph/target-orientation corruption ADV-7 describes; the canary must fire.
    // (For a dense radius-6 legal ball, an unrotated in-ball cell can stay legal,
    // so the canary is a CANARY not a universal proof — the design's primary
    // defense is the single-call unconstructability proven in the positive test.)
    let rec = sample_record();
    let stones: Vec<(i32, i32, i8)> = rec
        .stones
        .iter()
        .map(|&(q, r, p)| (i32::from(q), i32::from(r), p))
        .collect();
    let params = BuildParams {
        win_length: 6,
        radius: 6,
        current_player: rec.current_player,
        moves_remaining: rec.moves_remaining,
        trunk_size: 19,
    };
    let g = build_axis_graph(&StoneList { stones }, &params);

    // A stone cell (occupied) — coherent target never peaks here.
    let stone_cell = (0i32, 0i32);
    assert!(
        !argmax_canary_passes(&g, stone_cell),
        "a stone (occupied) cell must NOT be a legal node — the canary fires on this desync"
    );
    // Sanity: a real legal-node coord passes (discriminator is meaningful).
    let a_legal = {
        let row = g.legal_node_gather[0] as usize;
        (g.node_coords[row * 2], g.node_coords[row * 2 + 1])
    };
    assert!(argmax_canary_passes(&g, a_legal), "a real legal cell must pass the canary");
}

// ── WP-5a red-team fix-pass regression tests (B1 / B2 / game_id-rebase) ──────

/// B1 (SILENT-CORRUPT): a load that dies mid-payload (truncated file) must
/// LOUD-fail AND leave `self` byte-for-byte identical to its pre-call state —
/// contents, size, head, histogram, and continued sampling all unaffected.
#[test]
fn failed_truncated_load_is_loud_and_leaves_buffer_untouched() {
    // Build a separate valid source file (distinct game_ids from the victim)
    // and truncate it mid-payload, matching the red-team's `probe_fuzz.py`
    // "partial-mutation on FAILED load" reproduction.
    let mut src = HexgBuffer::new(8, ENC).unwrap();
    for i in 0..3u16 {
        let mut rec = sample_record();
        rec.ply_index = i;
        src.push_record_impl(&rec, 900 + i as i64).unwrap();
    }
    let good_path = unique_path("b1_src");
    src.save_to_path_impl(good_path.to_str().unwrap()).unwrap();
    let raw = std::fs::read(&good_path).unwrap();
    assert!(raw.len() > 20, "fixture too small to truncate meaningfully");
    let trunc_path = unique_path("b1_trunc");
    std::fs::write(&trunc_path, &raw[..raw.len() - 20]).unwrap();

    // A pre-populated "victim" buffer, distinct contents from `src`.
    let mut victim = HexgBuffer::new(8, ENC).unwrap();
    for i in 0..5u16 {
        let mut rec = sample_record();
        rec.ply_index = i;
        victim.push_record_impl(&rec, 100 + i as i64).unwrap();
    }
    let size0 = victim.size;
    let head0 = victim.head;
    let hist0 = victim.get_buffer_stats_impl().2;
    let snapshot: Vec<(GraphRecord, i64, u16)> = (0..size0)
        .map(|s| (victim.record_at(s), victim.game_ids[s], victim.weights[s]))
        .collect();

    let err = victim.load_from_path_impl(trunc_path.to_str().unwrap());
    assert!(err.is_err(), "truncated mid-payload load must LOUD-FAIL");

    assert_eq!(victim.size, size0, "size must be unchanged after a failed load");
    assert_eq!(victim.head, head0, "head must be unchanged after a failed load");
    assert_eq!(
        victim.get_buffer_stats_impl().2,
        hist0,
        "weight-bucket histogram must be unchanged after a failed load (no partial repopulation)"
    );
    for (s, (rec, gid, w)) in snapshot.iter().enumerate() {
        assert_eq!(&victim.record_at(s), rec, "record {s} must be byte-identical after failed load");
        assert_eq!(victim.game_ids[s], *gid, "game_id {s} must be unchanged");
        assert_eq!(victim.weights[s], *w, "weight {s} must be unchanged");
    }
    // Sampling must still be coherent (size==sum(hist) invariant intact, no
    // foreign/corrupt slot content reachable).
    let (_, targets) = victim.sample_graph_batch_impl(4, false).unwrap();
    assert_eq!(targets.outcomes.len(), 4);

    let _ = std::fs::remove_file(&good_path);
    let _ = std::fs::remove_file(&trunc_path);
}

/// B2 (SILENT target under-weighting): a visit coord poisoned onto an occupied
/// (illegal) cell must raise a LOUD, labeled mass-drop error at sample time —
/// not silently under-weight the CE target while the argmax canary stays quiet.
///
/// The labeled-message content (game_id/ply/dropped-mass) is asserted directly
/// on the pure `mass_drop_check` helper below — `PyErr`'s `Display` impl needs
/// an attached Python interpreter (`Python::attach`), which `cargo test` does
/// NOT provide by default (no `pyo3/auto-initialize`; see `engine/Cargo.toml`'s
/// `test-with-python` feature comment) — asserting on `err.to_string()` here
/// would panic under plain `cargo test`. This test only checks the raise;
/// `mass_drop_check_message_names_game_id_ply_and_dropped_mass` below checks
/// the message.
#[test]
fn sample_rejects_illegal_cell_visit_mass_drop() {
    let mut buf = HexgBuffer::new(4, ENC).unwrap();
    // Stones occupy (0,0),(1,0),(0,1); 0.9 mass poisoned onto OCCUPIED (0,0),
    // 0.1 on a legal neighbor (2,0) — the red-team's `probe_canary.py` repro.
    let rec = GraphRecord {
        stones: vec![(0, 0, 1), (1, 0, -1), (0, 1, 1)],
        visits: vec![(0, 0, 0.9), (2, 0, 0.1)],
        current_player: 1,
        moves_remaining: 2,
        ply_index: 5,
        is_full_search: true,
        outcome: 1.0,
        value_valid: true,
        game_length: 30,
    };
    buf.push_record_impl(&rec, 7).unwrap();
    assert!(
        buf.sample_graph_batch_impl(1, false).is_err(),
        "illegal-cell visit mass drop must raise, not silently under-weight"
    );
}

/// Direct unit test of the B2 mass-drop message (pure Rust, no PyO3/GIL): the
/// error must name the record's `game_id`, `ply`, and report the dropped mass.
#[test]
fn mass_drop_check_message_names_game_id_ply_and_dropped_mass() {
    let msg = super::sample::mass_drop_check(7, 5, /* stored */ 1.0, /* aligned */ 0.1)
        .expect_err("a 0.9 mass drop (10x the 1e-4 relative tolerance) must trip the guard");
    assert!(msg.contains("game_id=7"), "error must name the record's game_id: {msg}");
    assert!(msg.contains("ply=5"), "error must name the record's ply: {msg}");
    assert!(msg.contains("dropped"), "error must report the dropped mass: {msg}");
}

/// Negative control on the pure helper: a within-tolerance float-noise gap
/// (well under the 1e-4 relative tolerance) must NOT trip the guard.
#[test]
fn mass_drop_check_tolerates_float_noise() {
    super::sample::mass_drop_check(1, 0, 1.0, 1.0 - 1e-6)
        .expect("float-summation-level noise must not trip the mass-drop guard");
    // The zero-mass / zero-mass case (e.g. a quick-search row with no visits)
    // must also not trip (absolute-floor branch).
    super::sample::mass_drop_check(1, 0, 0.0, 0.0)
        .expect("zero stored / zero aligned mass must not trip the guard");
}

/// B2 negative control: a legit push→sample round-trip (all visit mass on
/// legal cells, per the `record_position_graph` producer contract) must NEVER
/// trip the mass-drop guard, unaugmented or across many D6 augmented draws.
#[test]
fn legit_push_sample_roundtrip_does_not_trip_mass_drop_guard() {
    let mut buf = HexgBuffer::new(4, ENC).unwrap();
    buf.push_record_impl(&sample_record(), 0).unwrap();
    for aug in [false, true] {
        for _ in 0..24 {
            buf.sample_graph_batch_impl(1, aug)
                .expect("legit round-trip must not trip the B2 mass-drop guard");
        }
    }
}

/// game_id re-base fix: after a successful load, `next_game_id` must continue
/// past the max loaded game_id, not reset to 0 (which would let a fresh
/// self-play game collide with a just-loaded record and mis-fire the
/// Multi-Window correlation-guard dedup — red-team T1).
#[test]
fn load_rebases_next_game_id_past_loaded_max() {
    let mut src = HexgBuffer::new(8, ENC).unwrap();
    for i in 0..3u16 {
        let mut rec = sample_record();
        rec.ply_index = i;
        src.push_record_impl(&rec, 2 + i as i64).unwrap(); // game_ids 2, 3, 4
    }
    let path = unique_path("ngid_rebase");
    src.save_to_path_impl(path.to_str().unwrap()).unwrap();

    let mut dst = HexgBuffer::new(8, ENC).unwrap();
    let n = dst.load_from_path_impl(path.to_str().unwrap()).unwrap();
    assert_eq!(n, 3);
    assert_eq!(dst.next_game_id, 5, "next_game_id must continue past the loaded max game_id (4)");

    let fresh_id = dst.next_game_id();
    assert_eq!(fresh_id, 5, "a fresh game after load must get a non-colliding id");
    assert!(
        !(2..=4).contains(&fresh_id),
        "fresh id must not collide with any loaded game_id"
    );

    let _ = std::fs::remove_file(path);
}

/// Guard: loading a zero-entry file (empty ring) must not touch `next_game_id`
/// (the `parsed.iter().max()` over an empty Vec is `None` — no-op re-base).
#[test]
fn load_of_empty_file_does_not_touch_next_game_id() {
    let empty_src = HexgBuffer::new(4, ENC).unwrap(); // size 0
    let path = unique_path("ngid_guard_empty");
    empty_src.save_to_path_impl(path.to_str().unwrap()).unwrap();

    let mut dst = HexgBuffer::new(4, ENC).unwrap();
    let g0 = dst.next_game_id();
    assert_eq!(g0, 0);
    let n = dst.load_from_path_impl(path.to_str().unwrap()).unwrap();
    assert_eq!(n, 0);
    assert_eq!(
        dst.next_game_id, 1,
        "loading zero records must not reset/clobber next_game_id (empty-ring guard)"
    );

    let _ = std::fs::remove_file(path);
}

// ── WP-5a red-team RE-VERIFICATION regression tests (N1 / N2 / N3) ──────────

/// N1 (game_id re-base overflow): a loaded record with `game_id == i64::MAX`
/// must not panic (debug) or silently wrap-to-no-rebase (release) — the
/// re-base must use `saturating_add(1)` and land at `i64::MAX`, never below
/// the loaded max.
#[test]
fn load_with_i64_max_game_id_does_not_panic_and_saturates() {
    let mut src = HexgBuffer::new(4, ENC).unwrap();
    src.push_record_impl(&sample_record(), i64::MAX).unwrap();
    let path = unique_path("gid_i64max");
    src.save_to_path_impl(path.to_str().unwrap()).unwrap();

    let mut dst = HexgBuffer::new(4, ENC).unwrap();
    let n = dst.load_from_path_impl(path.to_str().unwrap()).unwrap();
    assert_eq!(n, 1);
    assert_eq!(
        dst.next_game_id,
        i64::MAX,
        "saturating_add(1) on a loaded game_id of i64::MAX must saturate at i64::MAX \
         (never panic, never silently wrap to a small/negative value)"
    );

    let _ = std::fs::remove_file(path);
}

/// N2 (NaN-blind mass guard): a NaN visit prob must be rejected LOUD at push
/// time, before it can ever reach a sampled `policy_target`.
#[test]
fn push_rejects_nan_visit_prob() {
    let mut buf = HexgBuffer::new(4, ENC).unwrap();
    let rec = GraphRecord {
        visits: vec![(2, 0, f32::NAN)],
        ..sample_record()
    };
    let err = buf
        .push_record_impl(&rec, 0)
        .expect_err("a NaN visit prob must be rejected at push, not silently pass the guard");
    // PyErr's Display needs an attached interpreter under plain `cargo test`
    // (no pyo3/auto-initialize) — assert on the pure helper's message instead,
    // matching the sample.rs `mass_drop_check` test convention.
    let msg = super::push::validate_visit_prob(2, 0, f32::NAN)
        .expect_err("pure helper must also reject NaN");
    assert!(msg.contains("(2, 0)"), "error must name the offending coord: {msg}");
    assert!(msg.contains("NaN"), "error must report the offending value: {msg}");
    let _ = err; // push_record_impl's PyErr construction exercised above.
    assert_eq!(buf.size, 0, "rejected push must not mutate the buffer");
}

/// N3 (sign-blind mass guard): a negative visit prob must be rejected LOUD at
/// push time, even when paired with a compensating positive entry that would
/// make the aligned mass sum to zero at sample time (guard-blind case).
#[test]
fn push_rejects_negative_visit_prob() {
    let mut buf = HexgBuffer::new(4, ENC).unwrap();
    let rec = GraphRecord {
        visits: vec![(2, 0, -0.5), (-1, 0, 0.5)],
        ..sample_record()
    };
    let err = buf.push_record_impl(&rec, 0).expect_err(
        "a negative visit prob must be rejected at push, not silently sign-flip the CE target",
    );
    let msg = super::push::validate_visit_prob(2, 0, -0.5)
        .expect_err("pure helper must also reject a negative prob");
    assert!(msg.contains("(2, 0)"), "error must name the offending coord: {msg}");
    assert!(msg.contains("-0.5"), "error must report the offending value: {msg}");
    let _ = err;
    assert_eq!(buf.size, 0, "rejected push must not mutate the buffer");
}

/// Negative control: a legit push (all visit mass finite, non-negative, on
/// legal cells) must be entirely unaffected by the N2/N3 push-time guard.
#[test]
fn legit_push_unaffected_by_prob_validation_guard() {
    let mut buf = HexgBuffer::new(4, ENC).unwrap();
    let rec = sample_record();
    buf.push_record_impl(&rec, 42)
        .expect("a legit record's finite non-negative visit probs must pass the guard");
    assert_eq!(buf.size, 1);
    assert_eq!(buf.record_at(0), rec, "legit push content must be unaffected by the guard");
    assert_eq!(buf.game_ids[0], 42);
}
