//! D-QFIX-LAND golden byte-identity harness for the completed-Q sites.
//!
//! Captures `f32::to_bits()` of every output float for:
//!   - S1 `get_improved_policy` (dense `Vec<f32>`)
//!   - S2 `get_improved_policy_ls` (dense + overflow, overflow sorted by key)
//!   - S3 `gumbel_search.rs::score` (per-candidate, seeded RNG → deterministic)
//! across the design's fixtures incl. 6 RED-TEAM adversarial ones.
//!
//! Goldens are baked in-source as `const [u32]` (design §3.6 permits in-test
//! constants). Capture flow: run `test_capture_goldens_print` (ignored) → paste
//! its stdout into the `GOLDEN_*` consts → the non-ignored `test_golden_*` asserts
//! live bits == frozen bits (Rust↔Rust BIT-EXACT — an FMA→mul regression is
//! sub-ULP but nonzero, so `==` on bits catches it; `abs()<1e-6` would NOT).
//!
//! S3 is a regression guard ONLY: A2a must not touch `gumbel_search.rs::score`.

use super::*;
use crate::board::Board;
use crate::mcts::node::Node;
use crate::game_runner::gumbel_search::GumbelSearchState;
use rand::SeedableRng;
use rand::rngs::StdRng;

// ── Fixture builders ──────────────────────────────────────────────────────────

/// Build a tree with root children placed at explicit axial coords.
/// Each child = (q, r, visits, w_value, prior). `mr` sets root moves_remaining
/// (the q_sign perspective flip toggles at mr==1).
fn build_tree(board: Board, mr: u8, children: &[(i32, i32, u32, f32, f32)]) -> MCTSTree {
    let mut tree = MCTSTree::new(1.5);
    tree.root_board = board;
    let n = children.len();
    tree.pool[0] = Node::uninit();
    tree.pool[0].first_child = 1;
    tree.pool[0].n_children = n as u16;
    tree.pool[0].moves_remaining = mr;

    let mut total_visits = 0u32;
    let mut total_w = 0.0f32;
    for (j, &(q, r, visits, w_value, prior)) in children.iter().enumerate() {
        let action_idx =
            ((q as u32).wrapping_add(32768) << 16) | (r as u32).wrapping_add(32768);
        tree.pool[1 + j] = Node {
            parent: 0,
            action_idx,
            n_visits: visits,
            w_value,
            prior,
            first_child: u32::MAX,
            n_children: 0,
            moves_remaining: if mr == 1 { 2 } else { 1 },
            is_terminal: false,
            terminal_value: 0.0,
            virtual_loss_count: 0,
        };
        total_visits += visits;
        total_w += w_value;
    }
    tree.pool[0].n_visits = total_visits;
    tree.pool[0].w_value = total_w;
    tree.next_free = 1 + n as u32;
    tree
}

/// Two-cluster board: stones near (0,0) and near (20,0) yield cluster centers
/// `[(0,0),(20,0)]` and bbox-centroid window center (10,0). Used by the S2
/// off-window fixture so one off-window child is COVERED (→overflow) and another
/// off-window child is UNCOVERED (→dropped).
fn two_cluster_board() -> Board {
    let mut b = Board::new();
    b.override_legal_move_radius(40);
    for &(q, r) in &[(0, 0), (1, 0), (20, 0), (21, 0)] {
        b.apply_move(q, r).expect("planted stone must be legal");
    }
    b
}

const N_ACTIONS: usize = crate::board::BOARD_SIZE * crate::board::BOARD_SIZE + 1;
const C_VISIT: f32 = 50.0;
const C_SCALE: f32 = 1.0;

/// The full fixture roster, S1+S2 share these (they only diverge in scatter).
/// Returns (name, tree). Coords on an EMPTY-board fixture all land in-window.
fn s1s2_fixtures() -> Vec<(&'static str, MCTSTree)> {
    let empty = Board::new();
    vec![
        // Nominal: 3 children mixed visits/Q (mirrors test_improved_policy_sums_to_one).
        ("nominal", build_tree(empty.clone(), 2, &[
            (0, 0, 10, 5.0, 0.5),
            (0, 1, 8, -2.0, 0.3),
            (0, 2, 2, 0.4, 0.2),
        ])),
        // RED-1: extreme q = +/-1 clamp rails (w/n = +1.5 and -1.5 → clamps to +/-1).
        ("red1_extreme_q", build_tree(empty.clone(), 2, &[
            (0, 0, 4, 6.0, 0.5),   // q = 1.5 → clamp +1
            (0, 1, 4, -6.0, 0.5),  // q = -1.5 → clamp -1
        ])),
        // RED-2: prior=0 floor (log_prior uses prior.max(1e-8)); NaN-prone none here
        // but the 0-prior child exercises the 1e-8 floor path.
        ("red2_zero_prior", build_tree(empty.clone(), 2, &[
            (0, 0, 5, 2.0, 0.0),
            (0, 1, 5, 1.0, 1.0),
        ])),
        // RED-3: all-unvisited (sum_n == 0) → prior-fallback path.
        ("red3_all_unvisited", build_tree(empty.clone(), 2, &[
            (0, 0, 0, 0.0, 0.6),
            (0, 1, 0, 0.0, 0.4),
        ])),
        // RED-4: single legal move.
        ("red4_single", build_tree(empty.clone(), 2, &[
            (0, 0, 7, 3.0, 1.0),
        ])),
        // RED-5: v_mix activation w/ partial visits (some unvisited children get v_mix).
        ("red5_vmix_partial", build_tree(empty.clone(), 2, &[
            (0, 0, 20, 9.0, 0.4),  // visited, q = 0.45
            (0, 1, 0, 0.0, 0.35),  // unvisited → v_mix
            (0, 2, 0, 0.0, 0.25),  // unvisited → v_mix
        ])),
        // RED-5b: visited_prior_sum <= 1e-8 else-branch — the ONLY visited child
        // has prior 0, so visited_prior_sum == 0 → v_mix falls back to v_hat.
        ("red5b_vps_zero", build_tree(empty.clone(), 2, &[
            (0, 0, 12, 4.0, 0.0),  // visited but prior 0 → vps == 0
            (0, 1, 0, 0.0, 0.5),   // unvisited → v_mix (== v_hat path)
            (0, 2, 0, 0.0, 0.5),
        ])),
        // RED-6: mr == 1 perspective flip (q_sign = -1).
        ("red6_mr1_flip", build_tree(empty.clone(), 1, &[
            (0, 0, 10, 5.0, 0.5),
            (0, 1, 8, -2.0, 0.3),
            (0, 2, 2, 0.4, 0.2),
        ])),
    ]
}

/// S2-only off-window fixture: requires the 2-cluster board so coverage geometry
/// splits children into dense / overflow / dropped.
///   - in-window child  q=10 (dense)
///   - off-window COVERED child q=25 (covered by center (20,0) → overflow)
///   - off-window UNCOVERED child q=30 (dropped)
fn s2_offwindow_fixture() -> MCTSTree {
    build_tree(two_cluster_board(), 2, &[
        (10, 0, 10, 5.0, 0.4),  // in-window
        (25, 0, 8, -3.0, 0.35), // off-window covered
        (30, 0, 6, 1.0, 0.25),  // off-window uncovered → dropped
    ])
}

// ── Bit extraction ────────────────────────────────────────────────────────────

fn s1_bits(tree: &MCTSTree) -> Vec<u32> {
    tree.get_improved_policy(N_ACTIONS, C_VISIT, C_SCALE)
        .iter()
        .map(|f| f.to_bits())
        .collect()
}

/// S2 dense bits followed by overflow entries sorted by (q,r): for each entry
/// push q (as u32 reinterpret), r (as u32 reinterpret), value.to_bits().
fn s2_bits(tree: &MCTSTree) -> Vec<u32> {
    let ls = tree.get_improved_policy_ls(N_ACTIONS, C_VISIT, C_SCALE);
    let mut out: Vec<u32> = ls.dense.iter().map(|f| f.to_bits()).collect();
    let mut entries: Vec<((i32, i32), f32)> =
        ls.overflow.iter().map(|(k, v)| (*k, *v)).collect();
    entries.sort_by_key(|&((q, r), _)| (q, r));
    for ((q, r), v) in entries {
        out.push(q as u32);
        out.push(r as u32);
        out.push(v.to_bits());
    }
    out
}

/// S3 score bits: build a seeded Gumbel state, score every candidate.
fn s3_bits() -> Vec<u32> {
    let mut tree = MCTSTree::new(1.5);
    let mut board = Board::new();
    board.apply_move(0, 0).expect("(0,0) legal");
    tree.new_game(board);
    let n_actions = N_ACTIONS;
    let policy = vec![1.0 / n_actions as f32; n_actions];
    let _ = tree.select_leaves(1);
    tree.expand_and_backup(&[policy], &[0.0]);

    // Give a few children deterministic visits/w_values so q_hat != 0 fires.
    let first = tree.pool[0].first_child as usize;
    let n_ch = tree.pool[0].n_children as usize;
    for j in 0..n_ch.min(6) {
        tree.pool[first + j].n_visits = (j as u32 + 1) * 3;
        tree.pool[first + j].w_value = (j as f32 - 2.5) * 1.7;
    }

    let mut rng = StdRng::seed_from_u64(0xD_F1_C0DE);
    let gs = GumbelSearchState::new(&tree, 16, C_VISIT, C_SCALE, &mut rng);
    let max_n = gs.max_n();
    let mut out = Vec::new();
    for off in 0..n_ch {
        out.push(gs.score(off, max_n).to_bits());
    }
    out
}

/// S4 PUCT Dirichlet root-noise bits: build a seeded tree, expand root, sample
/// symmetric Dirichlet on a FIXED seed (distinct from S3), apply to root, dump
/// every root child's post-noise `prior` via `to_bits()`.
///
/// This pins the PUCT root-noise sequence f32::to_bits-identical. The B1
/// gumbel-arm Dirichlet deletion touches only the `if gumbel_mcts` branch in
/// `worker_loop/inner.rs`; the PUCT `else` arm calls the SAME
/// `sample_dirichlet`+`apply_dirichlet_to_root` exercised here. S4 holding
/// byte-identical proves the deletion cannot perturb the PUCT path.
const S4_DIRICHLET_SEED: u64 = 0x5_4D_1_5EED;
const S4_DIRICHLET_ALPHA: f32 = 0.3;
const S4_DIRICHLET_EPSILON: f32 = 0.25;

fn s4_puct_dirichlet_bits() -> Vec<u32> {
    let mut tree = MCTSTree::new(1.5);
    let mut board = Board::new();
    board.apply_move(0, 0).expect("(0,0) legal");
    tree.new_game(board);
    let n_actions = N_ACTIONS;
    let policy = vec![1.0 / n_actions as f32; n_actions];
    let _ = tree.select_leaves(1);
    tree.expand_and_backup(&[policy], &[0.0]);

    let n_ch = tree.pool[0].n_children as usize;
    let mut rng = StdRng::seed_from_u64(S4_DIRICHLET_SEED);
    let noise = crate::mcts::dirichlet::sample_dirichlet(S4_DIRICHLET_ALPHA, n_ch, &mut rng);
    tree.apply_dirichlet_to_root(&noise, S4_DIRICHLET_EPSILON);

    let first = tree.pool[0].first_child as usize;
    (0..n_ch)
        .map(|j| tree.pool[first + j].prior.to_bits())
        .collect()
}

// ── Golden roster: (name, live_bits) computed from the live code ──────────────

/// All goldens as `(line-key, live u32 bits)`. The line-key must match the
/// prefix in `golden_bits.txt`. One source of truth for both capture + assert.
fn all_goldens() -> Vec<(String, Vec<u32>)> {
    let mut out = Vec::new();
    for (name, tree) in s1s2_fixtures() {
        let up = name.to_uppercase();
        out.push((format!("S1_{up}"), s1_bits(&tree)));
        out.push((format!("S2_{up}"), s2_bits(&tree)));
    }
    out.push(("S2_OFFWINDOW".to_string(), s2_bits(&s2_offwindow_fixture())));
    out.push(("S3_SCORE".to_string(), s3_bits()));
    out.push(("S4_PUCT_DIRICHLET".to_string(), s4_puct_dirichlet_bits()));
    out
}

/// Frozen golden bits, captured at canonical HEAD `e132e67` BEFORE any A2a/A1
/// edit. One line per fixture: `KEY v0 v1 v2 ...` (decimal u32 to_bits()).
const GOLDEN_BITS: &str = include_str!("../../tests/golden/completed_q/golden_bits.txt");

fn parse_golden(key: &str) -> Vec<u32> {
    for line in GOLDEN_BITS.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut it = line.split_whitespace();
        let k = it.next().expect("golden line has a key");
        if k == key {
            return it.map(|t| t.parse::<u32>().expect("u32 golden token")).collect();
        }
    }
    panic!("golden key {key:?} not found in golden_bits.txt");
}

// ── Capture (ignored) — regenerates golden_bits.txt from the live code ────────

#[test]
#[ignore]
fn test_capture_goldens_print() {
    use std::fmt::Write as _;
    let mut s = String::new();
    s.push_str("# D-QFIX-LAND completed-Q goldens (f32::to_bits, decimal u32).\n");
    s.push_str("# Captured at canonical HEAD e132e67 BEFORE any A2a/A1 edit.\n");
    s.push_str("# One line per fixture: KEY bit0 bit1 ...  (regen: golden_tests::test_capture_goldens_print --ignored)\n");
    for (key, bits) in all_goldens() {
        write!(s, "{key}").unwrap();
        for b in &bits {
            write!(s, " {b}").unwrap();
        }
        s.push('\n');
    }
    // Print to stdout — operator pipes into the golden file once at HEAD.
    print!("{s}");
}

// ── Byte-identity assertions (non-ignored) ────────────────────────────────────

#[test]
fn test_golden_s1_s2_byte_identical() {
    for (key, live) in all_goldens() {
        if key.starts_with("S3") || key.starts_with("S4") {
            continue;
        }
        let golden = parse_golden(&key);
        assert_eq!(
            live.len(),
            golden.len(),
            "{key}: length drift live={} golden={}",
            live.len(),
            golden.len()
        );
        assert_eq!(
            live, golden,
            "{key}: completed-Q output bits diverged from HEAD golden — \
             A2a refactor is NOT byte-pure on this fixture"
        );
    }
}

#[test]
fn test_golden_s3_score_unchanged() {
    let live = s3_bits();
    let golden = parse_golden("S3_SCORE");
    assert_eq!(
        live, golden,
        "S3_SCORE: gumbel_search.rs::score per-candidate bits changed — \
         A2a must NOT touch S3 (gumbel_search.rs untouched contract)"
    );
}

#[test]
fn test_golden_s4_puct_dirichlet_unchanged() {
    let live = s4_puct_dirichlet_bits();
    let golden = parse_golden("S4_PUCT_DIRICHLET");
    assert_eq!(
        live, golden,
        "S4_PUCT_DIRICHLET: PUCT root Dirichlet noise bits changed — the B1 \
         gumbel-arm Dirichlet deletion must NOT perturb the PUCT path \
         (sample_dirichlet + apply_dirichlet_to_root untouched contract)"
    );
}
