//! `hexo-graph` — axis-graph builder for the HeXO GNN encoding (C1 / WP-1).
//!
//! ONE Rust source compiled to TWO targets (standing ruling, R4/WP-D
//! mission): native (`cargo check -p hexo-graph`) and wasm32
//! (`cargo check -p hexo-graph --no-default-features --features wasm
//! --target wasm32-unknown-unknown`). The core builder is dep-free and
//! `std::thread`/`rayon`/PyO3-free so it crosses the wasm boundary clean; the
//! JSON parity harness (`src/bin/harness.rs`) and the criterion bench
//! (`benches/`) are feature-gated OUT of the wasm/core surface.
//!
//! This is a FAITHFUL port of the Python oracle
//! `hexo_rl/bots/strix_v1_graph.py::build_axis_graph_raw` for the LEGACY
//! schema (`relative_stones=true`, `threat_features=true`,
//! `prune_empty_edges=false`, `win_length=6`, `radius=6`) — the schema the
//! amended ragged-contract v1 fixes as `node_feat_dim=11`
//! (`docs/designs/gnn_ragged_contract_v1.md` §2.1). The Python builder stays
//! the TEST ORACLE (`tests/test_hexo_graph_parity.py`); it is never a
//! production path.
//!
//! Byte-exact on the integer outputs (node order, edge_index, n_stones,
//! scatter indices); float features to ≤1e-6 (features accumulate in f64 then
//! cast to f32, mirroring the oracle's Python-float → `torch.float32` path).
#![cfg_attr(not(feature = "native"), allow(dead_code))]
#![allow(clippy::many_single_char_names)]
// Deliberate, bounded integer work: axial coords pack into an i64 key, node
// ids are < ~1000 (WP-A max 897), slot math is the byte-parity port of engine
// `core.rs::window_flat_idx_at_geom`. The pedantic cast lints flag every one of
// these intentional, in-range conversions — silenced crate-wide with intent.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::doc_markdown,
    clippy::too_many_lines
)]

use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasherDefault, Hasher};

// ── constants (LEGACY relative+threat schema, contract v1 §2.1) ──────────────

/// Per-node feature width: relative-7 base + 4 threat = 11
/// (`gnn_ragged_contract_v1.md` §2.1; `strix_v1_graph.py:108-124`).
pub const NODE_FEAT_DIM: usize = 11;
/// Relative-schema base width (own, opp, empty, moves, norm_q, norm_r, inv_dist).
pub const BASE_DIM: usize = 7;
/// Per-edge feature width: axis one-hot(3) + signed_dist + src_player (GINE).
pub const EDGE_FEAT_DIM: usize = 5;
/// The 3 win axes in axial coords (`strix_v1_graph.py:16`).
pub const WIN_AXES: [(i32, i32); 3] = [(1, 0), (0, 1), (1, -1)];
/// Wire tag: 1 = native Rust builder (this crate); 2 = Python oracle.
/// The contract's `NonNativeSampleBuilder` handshake asserts this == 1 on any
/// training/self-play path (`gnn_ragged_contract_v1.md` §2.2 / F7).
pub const BUILDER_IMPL_NATIVE: u8 = 1;
/// Sentinel for a legal node whose cell falls OUTSIDE the trunk-sized policy
/// window (`window_flat_idx` returns `usize::MAX`, `core.rs:409`). Off-window
/// legal cells have NO dense action slot — see the ambiguity note in
/// `reports/probes/gnn_integration/WP1_builder.md`. The Python resolver maps
/// this to its own off-window handling; the builder refuses to invent a slot.
pub const OFF_WINDOW_SLOT: u32 = u32::MAX;

// ── fast, dep-free, wasm-clean coordinate hashing ────────────────────────────
//
// Coordinate keys are packed i64 (`(q<<32)|r`) integers; the default SipHash
// is overkill and slow for the ~30k point-lookups the threat + edge walks do
// per position. A tiny FNV-1a keeps Cargo.toml's core dependency list EMPTY
// (the crate's whole reason for existing as its own compilation unit) while
// buying the BUILD-HOT speedup. Not cryptographic — never fed untrusted input.
#[derive(Default)]
struct FnvHasher(u64);
impl Hasher for FnvHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let mut h = if self.0 == 0 { 0xcbf2_9ce4_8422_2325 } else { self.0 };
        for &b in bytes {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        self.0 = h;
    }
    #[inline]
    fn write_i64(&mut self, i: i64) {
        // Direct i64 path: the map keys are all packed coords, so avoid the
        // byte loop entirely on the hot lookups.
        let mut h = if self.0 == 0 { 0xcbf2_9ce4_8422_2325 } else { self.0 };
        h ^= i as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
        self.0 = h;
    }
    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.write_i64(i as i64);
    }
}
type FnvMap<K, V> = HashMap<K, V, BuildHasherDefault<FnvHasher>>;
type FnvSet<K> = HashSet<K, BuildHasherDefault<FnvHasher>>;

#[inline]
fn pack(q: i32, r: i32) -> i64 {
    (i64::from(q) << 32) | i64::from(r as u32)
}

// ── payload types (contract v1 §2.1 single-graph slice) ──────────────────────

/// Flat, row-major node feature matrix, shape `(N, NODE_FEAT_DIM)`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct NodeFeat(pub Vec<f32>);

/// COO edge index as two parallel LOCAL-id arrays (PyTorch-Geometric
/// convention: `edge_index[0]`=src, `[1]`=dst). Block-diagonal global
/// offsetting into an `i64` batch array is the collate resolver's job
/// (`gnn_ragged_contract_v1.md` §2.2) — this per-leaf builder emits local u32
/// ids (a single graph never exceeds ~900 nodes, WP-A distribution).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct EdgeIndex {
    pub src: Vec<u32>,
    pub dst: Vec<u32>,
}

/// Flat, row-major edge feature matrix, shape `(E, EDGE_FEAT_DIM)`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct EdgeAttr(pub Vec<f32>);

/// Per-legal-node destination action slot (0..361) in the fixed 362-slot dense
/// action space, computed via `window_flat_idx_at_geom` at the graph's
/// bbox-midpoint `window_center`. `OFF_WINDOW_SLOT` for off-window legal cells.
/// This is the contract's `policy_dst_slot` (`gnn_ragged_contract_v1.md`
/// §2.1/§2.4) — the input the Python collate scatters a per-node logit through.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PolicyScatterIndex(pub Vec<u32>);

/// One built axis-graph — the single-leaf slice of the ragged contract, plus
/// the semantic-layer geometry the amended contract carries on the wire
/// (`node_coords`, `n_stones`, `window_center`, `current_player`,
/// `builder_impl`, `n_nodes_checksum`) so the resolver's F1-F3 geometric
/// assertions can fire. Node rows are laid out `[stones | legal | dummy]`,
/// each block coordinate-sorted (contract-relevant ordering — the D6 rebuild
/// path depends on the deterministic sort, `gnn_ragged_contract_v1.md` Part 3).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AxisGraph {
    pub node_feat: NodeFeat,
    pub edge_index: EdgeIndex,
    pub edge_attr: EdgeAttr,
    pub legal_mask: Vec<bool>,
    pub stone_mask: Vec<bool>,
    pub policy_scatter_index: PolicyScatterIndex,
    /// Raw `(q, r)` per node, flat `[2 * N]` (dummy row = (0, 0)).
    pub node_coords: Vec<i32>,
    /// LOCAL node-row index of each legal node (into `node_feat`); the
    /// contract's `legal_node_gather` before block-diagonal offsetting.
    pub legal_node_gather: Vec<u32>,
    /// Number of stone nodes = the `[stones | legal]` split point.
    pub n_stones: u16,
    /// Declared node count (stones + legal + 1 dummy) = N; the off-by-one
    /// tripwire (`NodeCountChecksum`, ADV-1).
    pub n_nodes_checksum: u32,
    /// Bbox-midpoint window centre `(cq, cr)` (`core.rs::window_center`),
    /// the origin of the coord→action-slot map.
    pub window_center: (i32, i32),
    /// +1 / −1 side to move for this position.
    pub current_player: i8,
    /// = `BUILDER_IMPL_NATIVE` (1).
    pub builder_impl: u8,
}

impl AxisGraph {
    /// Total node count N (stones + legal + dummy).
    #[must_use]
    pub fn num_nodes(&self) -> usize {
        self.node_feat.0.len() / NODE_FEAT_DIM
    }
    /// Total directed edge count E.
    #[must_use]
    pub fn num_edges(&self) -> usize {
        self.edge_index.src.len()
    }
}

/// Minimal position input: a stone list `(q, r, player)`, player `+1`/`-1`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct StoneList {
    pub stones: Vec<(i32, i32, i8)>,
}

/// Board/window parameters. `current_player` is the side to move (+1/−1;
/// terminal → treat as −1, matching the oracle's `Some(P1)->1 else -1`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BuildParams {
    pub win_length: u8,
    pub radius: u16,
    pub current_player: i8,
    pub moves_remaining: u8,
    /// Trunk side length for the policy-slot window (19 for v6/v7full).
    pub trunk_size: i32,
}

impl Default for BuildParams {
    fn default() -> Self {
        BuildParams {
            win_length: 6,
            radius: 6,
            current_player: 1,
            moves_remaining: 2,
            trunk_size: 19,
        }
    }
}

// ── window geometry (byte-parity with engine `core.rs`) ──────────────────────

/// Bbox-midpoint window centre over stones — `core.rs::window_center`
/// (`(min+max)/2`, i32 truncate-toward-zero). `(0, 0)` when stoneless.
///
/// `manual_midpoint` is ALLOWED on purpose: `i32::midpoint` rounds toward
/// negative infinity, but the engine (and the anchor calibration of every v6/
/// v7 checkpoint) uses truncate-toward-zero `(min+max)/2`. Swapping to
/// `midpoint` would silently break byte-parity for negative-coordinate boards.
#[allow(clippy::manual_midpoint)]
#[inline]
fn window_center(stones: &[(i32, i32, i8)]) -> (i32, i32) {
    if stones.is_empty() {
        return (0, 0);
    }
    let (mut min_q, mut max_q) = (i32::MAX, i32::MIN);
    let (mut min_r, mut max_r) = (i32::MAX, i32::MIN);
    for &(q, r, _) in stones {
        min_q = min_q.min(q);
        max_q = max_q.max(q);
        min_r = min_r.min(r);
        max_r = max_r.max(r);
    }
    ((min_q + max_q) / 2, (min_r + max_r) / 2)
}

/// `core.rs::window_flat_idx_at_geom` — window-relative flat index, or
/// `usize::MAX` off-window. Returned as u32 slot (or `OFF_WINDOW_SLOT`).
#[inline]
fn window_flat_idx(q: i32, r: i32, cq: i32, cr: i32, trunk_sz: i32) -> u32 {
    let half = (trunk_sz - 1) / 2;
    let wq = q - cq + half;
    let wr = r - cr + half;
    if wq >= 0 && wq < trunk_sz && wr >= 0 && wr < trunk_sz {
        (wq as u32) * (trunk_sz as u32) + (wr as u32)
    } else {
        OFF_WINDOW_SLOT
    }
}

#[inline]
fn hex_distance(a: (i32, i32), b: (i32, i32)) -> i32 {
    let dq = (b.0 - a.0).abs();
    let dr = (b.1 - a.1).abs();
    let ds = (b.0 - a.0 + b.1 - a.1).abs();
    dq.max(dr).max(ds)
}

// ── node kind ────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    /// Stone owned by player (+1 / −1).
    Stone(i8),
    Empty,
}

impl Kind {
    /// `kind_player_feat` (`strix_v1_graph.py:215-218`): stone → ±1.0, empty → 0.0.
    #[inline]
    fn player_feat(self) -> f32 {
        match self {
            Kind::Stone(p) => {
                if p == 1 {
                    1.0
                } else {
                    -1.0
                }
            }
            Kind::Empty => 0.0,
        }
    }
}

// ── threat features (port of node_threat_features) ───────────────────────────

/// `strix_v1_graph.py::node_threat_features` — returns
/// `[own_max/wl, opp_max/wl, own_axes/3, opp_axes/3]` in f64. `stone_at`
/// returns 0 for empty (0 never equals ±1, so it never counts as own/opp).
#[inline]
fn node_threat_features(
    stone_map: &FnvMap<i64, i8>,
    coord: (i32, i32),
    to_move: i8,
    win_length: usize,
) -> [f64; 4] {
    let wl = win_length as i32;
    let opp = -to_move;
    let mut cells = [0i8; 64]; // 2*wl-1 <= 63 for wl<=32
    let n_cells = 2 * win_length - 1;
    let (mut own_max, mut opp_max, mut own_axes, mut opp_axes) = (0i32, 0i32, 0i32, 0i32);
    for &(dq, dr) in &WIN_AXES {
        // cells[k] for k in -(wl-1)..=(wl-1)
        for (slot, k) in ((-(wl - 1))..wl).enumerate() {
            let c = (coord.0 + k * dq, coord.1 + k * dr);
            cells[slot] = stone_map.get(&pack(c.0, c.1)).copied().unwrap_or(0);
        }
        let (mut axis_own, mut axis_opp) = (0i32, 0i32);
        for start in 0..win_length {
            let mut own_n = 0i32;
            let mut opp_n = 0i32;
            for &c in &cells[start..start + win_length] {
                if c == to_move {
                    own_n += 1;
                } else if c == opp {
                    opp_n += 1;
                }
            }
            if opp_n == 0 {
                axis_own = axis_own.max(own_n);
            }
            if own_n == 0 {
                axis_opp = axis_opp.max(opp_n);
            }
        }
        let _ = n_cells; // documents the 2*wl-1 window; cells sliced above
        own_max = own_max.max(axis_own);
        opp_max = opp_max.max(axis_opp);
        if axis_own >= wl - 2 {
            own_axes += 1;
        }
        if axis_opp >= wl - 2 {
            opp_axes += 1;
        }
    }
    let wlf = f64::from(wl);
    [
        f64::from(own_max) / wlf,
        f64::from(opp_max) / wlf,
        f64::from(own_axes) / 3.0,
        f64::from(opp_axes) / 3.0,
    ]
}

// ── legal moves ──────────────────────────────────────────────────────────────

/// `strix_v1_graph.py::legal_moves_from_stones` — empty cells within
/// hex-distance ≤ radius of any stone, sorted lexicographically `(q, r)`.
fn legal_moves_from_stones(
    stone_map: &FnvMap<i64, i8>,
    stones: &[(i32, i32, i8)],
    radius: i32,
) -> Vec<(i32, i32)> {
    // offsets: hex-ball of the given radius
    let mut offsets: Vec<(i32, i32)> = Vec::with_capacity((3 * radius * (radius + 1) + 1) as usize);
    for dq in -radius..=radius {
        for dr in -radius..=radius {
            if dq.abs().max(dr.abs()).max((dq + dr).abs()) <= radius {
                offsets.push((dq, dr));
            }
        }
    }
    let mut seen: FnvSet<i64> =
        FnvSet::with_capacity_and_hasher(stones.len() * offsets.len(), BuildHasherDefault::default());
    let mut legal: Vec<(i32, i32)> = Vec::new();
    for &(sq, sr, _) in stones {
        for &(dq, dr) in &offsets {
            let cq = sq + dq;
            let cr = sr + dr;
            let key = pack(cq, cr);
            if !stone_map.contains_key(&key) && seen.insert(key) {
                legal.push((cq, cr));
            }
        }
    }
    legal.sort_unstable();
    legal
}

// ── the builder ──────────────────────────────────────────────────────────────

/// Build one axis-graph — the once-per-evaluated-leaf construction. Faithful
/// port of `build_axis_graph_raw` for the LEGACY relative+threat schema.
/// Never call the search-time-incremental variant (there is none by design,
/// §S186): one payload per evaluated leaf, no parallelism inside (the caller
/// parallelizes over leaves).
#[must_use]
pub fn build_axis_graph(stones_in: &StoneList, params: &BuildParams) -> AxisGraph {
    let win_length = params.win_length as usize;
    let radius = i32::from(params.radius);
    let window = (params.win_length - 1) as usize; // axis-walk depth
    let cur = params.current_player;

    // stone_map (coord -> player) for membership + threat lookups.
    let mut stone_map: FnvMap<i64, i8> =
        FnvMap::with_capacity_and_hasher(stones_in.stones.len(), BuildHasherDefault::default());
    for &(q, r, p) in &stones_in.stones {
        stone_map.insert(pack(q, r), p);
    }

    // --- entry-point derivations (game_to_axis_graph_raw_lean) ---
    // stones sorted by coord (q, r) — carries the owning player. Derived from
    // the DEDUPED `stone_map` (not the raw input Vec) to match the oracle's
    // `sorted(stone_map.items())` exactly: the Python builder receives a dict,
    // so a duplicate coord collapses to one node (last-player-wins, the
    // HashMap-insert semantics above). A real board never has duplicates, but
    // matching the dict is what makes this a faithful port on the whole domain.
    let mut stones: Vec<(i32, i32, i8)> = stone_map
        .iter()
        .map(|(&k, &p)| ((k >> 32) as i32, k as u32 as i32, p))
        .collect();
    stones.sort_unstable_by_key(|&(q, r, _)| (q, r));
    let legal = legal_moves_from_stones(&stone_map, &stones, radius);

    let player_feat: f32 = if cur == 1 { 1.0 } else { -1.0 };
    let own_is_p1 = player_feat > 0.0;
    let moves_feat: f32 = (f64::from(params.moves_remaining) / 2.0) as f32;
    let to_move: i8 = if cur == 1 { 1 } else { -1 };

    let n_stones = stones.len();
    let n_legal = legal.len();
    let n_real = n_stones + n_legal;
    let n = n_real + 1;
    let dummy_idx = n_real as u32;
    let fdim = NODE_FEAT_DIM;

    // Node coords + coord->idx + kind. Layout: [stones | legal | dummy].
    let mut coords: Vec<i32> = Vec::with_capacity(n * 2);
    let mut node_kind: Vec<Kind> = Vec::with_capacity(n_real);
    let mut coord_to_idx: FnvMap<i64, u32> =
        FnvMap::with_capacity_and_hasher(n_real, BuildHasherDefault::default());
    for (i, &(q, r, p)) in stones.iter().enumerate() {
        coords.push(q);
        coords.push(r);
        coord_to_idx.insert(pack(q, r), i as u32);
        node_kind.push(Kind::Stone(p));
    }
    for (j, &(q, r)) in legal.iter().enumerate() {
        let idx = (n_stones + j) as u32;
        coords.push(q);
        coords.push(r);
        coord_to_idx.insert(pack(q, r), idx);
        node_kind.push(Kind::Empty);
    }
    coords.push(0); // dummy
    coords.push(0);

    // centroid + spread over stones (f64, oracle-faithful).
    let (cq, cr, spread): (f64, f64, f64) = if n_stones > 0 {
        let mut sq = 0f64;
        let mut sr = 0f64;
        for &(q, r, _) in &stones {
            sq += f64::from(q);
            sr += f64::from(r);
        }
        let cq = sq / n_stones as f64;
        let cr = sr / n_stones as f64;
        let mut max_dev = 0f64;
        for &(q, r, _) in &stones {
            max_dev = max_dev.max((f64::from(q) - cq).abs().max((f64::from(r) - cr).abs()));
        }
        (cq, cr, max_dev.max(1.0))
    } else {
        (0.0, 0.0, 1.0)
    };

    // Features (accumulate positions in f64, store f32).
    let mut features = vec![0f32; n * fdim];
    // relative layout: own=0 opp=1 empty=2 moves=3 norm_q=4 norm_r=5 inv=6
    let (l_own, l_opp, l_empty, l_moves, l_norm_q, l_norm_r, l_inv) = (0, 1, 2, 3, 4, 5, 6);
    let set_coords = |features: &mut [f32], base: usize, q: i32, r: i32| {
        features[base + l_norm_q] = ((f64::from(q) - cq) / spread) as f32;
        features[base + l_norm_r] = ((f64::from(r) - cr) / spread) as f32;
    };

    // Stone features.
    for (i, &(q, r, p)) in stones.iter().enumerate() {
        let base = i * fdim;
        let col = if (p == 1) == own_is_p1 { l_own } else { l_opp };
        features[base + col] = 1.0;
        features[base + l_moves] = moves_feat;
        set_coords(&mut features, base, q, r);
        // inv_dist stays 0 for stones
    }

    // Legal-move features.
    for (j, &(q, r)) in legal.iter().enumerate() {
        let idx = n_stones + j;
        let base = idx * fdim;
        features[base + l_empty] = 1.0;
        features[base + l_moves] = moves_feat;
        set_coords(&mut features, base, q, r);
        let min_d = if stones.is_empty() {
            1
        } else {
            let mut m = i32::MAX;
            for &(sq, sr, _) in &stones {
                m = m.min(hex_distance((q, r), (sq, sr)));
            }
            m
        };
        features[base + l_inv] = (1.0f64 / f64::from(min_d.max(1))) as f32;
    }

    // Dummy features.
    let dummy_base = (dummy_idx as usize) * fdim;
    features[dummy_base + l_moves] = moves_feat;

    // Masks.
    let mut legal_mask = vec![false; n];
    let mut stone_mask = vec![false; n];
    for m in stone_mask.iter_mut().take(n_stones) {
        *m = true;
    }
    for j in 0..n_legal {
        legal_mask[n_stones + j] = true;
    }

    // --- Axis-window edges ---
    // Upper bound: n_real nodes × 3 axes × 2 signs × window depth × 2 dirs.
    let cap = n_real * 3 * 2 * window * 2 + n_real * 2;
    let mut edge_src: Vec<u32> = Vec::with_capacity(cap);
    let mut edge_dst: Vec<u32> = Vec::with_capacity(cap);
    let mut edge_attr: Vec<f32> = Vec::with_capacity(cap * EDGE_FEAT_DIM);

    for i in 0..n_real {
        let iq = coords[i * 2];
        let ir = coords[i * 2 + 1];
        let i_kind = node_kind[i];
        let src_i = i_kind.player_feat();
        for (axis_idx, &(dq, dr)) in WIN_AXES.iter().enumerate() {
            for sign in [1i32, -1i32] {
                let sdq = dq * sign;
                let sdr = dr * sign;
                for d in 1..=(window as i32) {
                    let tq = iq + sdq * d;
                    let tr = ir + sdr * d;
                    let Some(&j) = coord_to_idx.get(&pack(tq, tr)) else {
                        break;
                    };
                    let j_kind = node_kind[j as usize];
                    let src_j = j_kind.player_feat();
                    let signed_dist = (d * sign) as f32;
                    // i -> j
                    edge_src.push(i as u32);
                    edge_dst.push(j);
                    push_attr(&mut edge_attr, axis_idx, signed_dist, src_i);
                    // j -> i
                    edge_src.push(j);
                    edge_dst.push(i as u32);
                    push_attr(&mut edge_attr, axis_idx, -signed_dist, src_j);
                    // walk stopping
                    let should_stop = match i_kind {
                        Kind::Stone(ip) => matches!(j_kind, Kind::Stone(jp) if jp != ip),
                        Kind::Empty => matches!(j_kind, Kind::Stone(_)),
                    };
                    if should_stop {
                        break;
                    }
                }
            }
        }
    }

    // --- dedup axis edges: key (src, dst, axis_idx), keep FIRST ---
    dedup_axis_edges(&mut edge_src, &mut edge_dst, &mut edge_attr);

    // --- legacy dummy edges: bidirectional to all real nodes, all-zero attr ---
    for i in 0..n_real as u32 {
        edge_src.push(dummy_idx);
        edge_dst.push(i);
        edge_attr.extend_from_slice(&[0.0; EDGE_FEAT_DIM]);
        edge_src.push(i);
        edge_dst.push(dummy_idx);
        edge_attr.extend_from_slice(&[0.0; EDGE_FEAT_DIM]);
    }

    // --- threat features (real nodes only) ---
    for idx in 0..n_real {
        let c = (coords[idx * 2], coords[idx * 2 + 1]);
        let tf = node_threat_features(&stone_map, c, to_move, win_length);
        let base = idx * fdim + BASE_DIM;
        for (k, &v) in tf.iter().enumerate() {
            features[base + k] = v as f32;
        }
    }

    // --- policy scatter (dense action slot per legal node) + gather rows ---
    let wc = window_center(&stones);
    let mut policy_scatter_index: Vec<u32> = Vec::with_capacity(n_legal);
    let mut legal_node_gather: Vec<u32> = Vec::with_capacity(n_legal);
    for (j, &(q, r)) in legal.iter().enumerate() {
        legal_node_gather.push((n_stones + j) as u32);
        policy_scatter_index.push(window_flat_idx(q, r, wc.0, wc.1, params.trunk_size));
    }

    let g = AxisGraph {
        node_feat: NodeFeat(features),
        edge_index: EdgeIndex { src: edge_src, dst: edge_dst },
        edge_attr: EdgeAttr(edge_attr),
        legal_mask,
        stone_mask,
        policy_scatter_index: PolicyScatterIndex(policy_scatter_index),
        node_coords: coords,
        legal_node_gather,
        n_stones: n_stones as u16,
        n_nodes_checksum: n as u32,
        window_center: wc,
        current_player: to_move,
        builder_impl: BUILDER_IMPL_NATIVE,
    };
    debug_assert_contract(&g, n_stones, n_legal, params.trunk_size);
    g
}

#[inline]
fn push_attr(edge_attr: &mut Vec<f32>, axis_idx: usize, signed_dist: f32, src_player: f32) {
    let mut a = [0.0f32; EDGE_FEAT_DIM];
    a[axis_idx] = 1.0;
    a[3] = signed_dist;
    a[4] = src_player;
    edge_attr.extend_from_slice(&a);
}

/// `axis_idx_of(a)` (`strix_v1_graph.py:265-270`): the one-hot axis of an attr.
#[inline]
fn axis_idx_of(a: &[f32]) -> u8 {
    if a[0] > 0.5 {
        0
    } else if a[1] > 0.5 {
        1
    } else {
        2
    }
}

/// Dedup by `(src, dst, axis_idx)` keeping the FIRST occurrence, preserving
/// insertion order (`strix_v1_graph.py:264-282`). Compacts all three arrays
/// IN PLACE (single pass, no reallocation) so `edge_attr[e]` stays bound to
/// `edge_index[:, e]`. Key packs into one u64 (src/dst < ~1000 nodes, axis in
/// 0..2) to skip tuple hashing on the hot dedup pass.
fn dedup_axis_edges(src: &mut Vec<u32>, dst: &mut Vec<u32>, attr: &mut Vec<f32>) {
    let e = src.len();
    let mut seen: FnvSet<u64> =
        FnvSet::with_capacity_and_hasher(e, BuildHasherDefault::default());
    let mut w = 0usize; // write cursor for the compacted arrays
    for rd in 0..e {
        let axis = axis_idx_of(&attr[rd * EDGE_FEAT_DIM..rd * EDGE_FEAT_DIM + EDGE_FEAT_DIM]);
        let key = (u64::from(src[rd]) << 34) | (u64::from(dst[rd]) << 2) | u64::from(axis);
        if seen.insert(key) {
            if w != rd {
                src[w] = src[rd];
                dst[w] = dst[rd];
                attr.copy_within(rd * EDGE_FEAT_DIM..rd * EDGE_FEAT_DIM + EDGE_FEAT_DIM, w * EDGE_FEAT_DIM);
            }
            w += 1;
        }
    }
    src.truncate(w);
    dst.truncate(w);
    attr.truncate(w * EDGE_FEAT_DIM);
}

/// Producer-side contract invariants (`gnn_ragged_contract_v1.md` §2.5). Cheap
/// checks so an oracle divergence / internal desync dies LOUD in debug + test
/// builds rather than surfacing as a silent parity mismatch.
#[inline]
fn debug_assert_contract(g: &AxisGraph, n_stones: usize, n_legal: usize, trunk_sz: i32) {
    if cfg!(debug_assertions) {
        let n = g.num_nodes();
        assert_eq!(g.n_nodes_checksum as usize, n, "n_nodes_checksum != N");
        assert_eq!(n, n_stones + n_legal + 1, "N != stones+legal+dummy");
        assert_eq!(g.node_coords.len(), 2 * n, "node_coords len != 2N");
        assert_eq!(g.legal_mask.len(), n);
        assert_eq!(g.stone_mask.len(), n);
        assert_eq!(g.edge_attr.0.len(), EDGE_FEAT_DIM * g.num_edges(), "edge_attr != 5E");
        assert_eq!(g.builder_impl, BUILDER_IMPL_NATIVE);
        // edge ids in-bounds
        for (&s, &d) in g.edge_index.src.iter().zip(&g.edge_index.dst) {
            assert!((s as usize) < n && (d as usize) < n, "edge id out of bounds");
        }
        // gather in the legal subrange [n_stones, n_stones+n_legal); slot canonical
        let half = (trunk_sz - 1) / 2;
        for (i, &row) in g.legal_node_gather.iter().enumerate() {
            assert!(
                (row as usize) >= n_stones && (row as usize) < n_stones + n_legal,
                "gather outside legal subrange"
            );
            let q = g.node_coords[row as usize * 2];
            let r = g.node_coords[row as usize * 2 + 1];
            let slot = g.policy_scatter_index.0[i];
            if slot != OFF_WINDOW_SLOT {
                let wq = q - g.window_center.0 + half;
                let wr = r - g.window_center.1 + half;
                assert!(wq >= 0 && wq < trunk_sz && wr >= 0 && wr < trunk_sz, "slot claims in-window but coord off-window");
                assert_eq!(slot, (wq as u32) * (trunk_sz as u32) + wr as u32, "non-canonical slot");
            }
        }
    }
}

// ── native-only surface (cfg-gating pattern for a future threaded caller) ────
#[cfg(all(feature = "native", not(target_arch = "wasm32")))]
#[must_use]
pub fn parallelism_hint() -> usize {
    std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .unwrap_or(1)
}

// ── python-only surface (unchanged skeleton marker; real glue is WP-3) ───────
#[cfg(feature = "python")]
mod python_glue {
    use pyo3::prelude::*;

    #[allow(dead_code)]
    #[pyfunction]
    fn _hexo_graph_skeleton_marker() -> PyResult<bool> {
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn payload_types_default_construct() {
        let g = AxisGraph::default();
        assert_eq!(g.num_nodes(), 0);
    }

    #[test]
    fn empty_board_builds_dummy_only() {
        // No stones → no legal nodes → 1 dummy node, 0 axis edges, 0 dummy edges.
        let g = build_axis_graph(&StoneList::default(), &BuildParams::default());
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.n_stones, 0);
        assert_eq!(g.num_edges(), 0);
        assert!(g.legal_node_gather.is_empty());
        assert_eq!(g.window_center, (0, 0));
    }

    #[test]
    fn single_stone_smoke() {
        let stones = StoneList { stones: vec![(0, 0, 1)] };
        let g = build_axis_graph(&stones, &BuildParams::default());
        assert_eq!(g.n_stones, 1);
        assert!(g.num_nodes() > 1); // 1 stone + legal ring + dummy
        assert!(g.stone_mask[0]);
        // legal nodes carry a canonical slot for cells inside the 19-window
        assert!(!g.policy_scatter_index.0.is_empty());
    }

    #[test]
    fn duplicate_coord_dedups_last_wins() {
        // The oracle takes a dict, so a repeated coord collapses to ONE node
        // with the LAST player. The builder must match on the whole domain.
        let g_dup = build_axis_graph(
            &StoneList { stones: vec![(0, 0, 1), (2, 0, -1), (0, 0, -1)] },
            &BuildParams::default(),
        );
        let g_dedup = build_axis_graph(
            &StoneList { stones: vec![(0, 0, -1), (2, 0, -1)] },
            &BuildParams::default(),
        );
        assert_eq!(g_dup.n_stones, 2);
        assert_eq!(g_dup.num_nodes(), g_dedup.num_nodes());
        assert_eq!(g_dup.node_coords, g_dedup.node_coords);
        assert_eq!(g_dup.edge_index.src, g_dedup.edge_index.src);
        assert_eq!(g_dup.node_feat, g_dedup.node_feat);
    }

    #[test]
    fn slot_and_gather_contract_invariants() {
        // gather rows land in the legal subrange; slots are canonical or the
        // documented off-window sentinel.
        let stones = StoneList { stones: vec![(0, 0, 1), (1, 0, -1), (0, 1, 1)] };
        let g = build_axis_graph(&stones, &BuildParams::default());
        let ns = g.n_stones as usize;
        for (i, &row) in g.legal_node_gather.iter().enumerate() {
            assert!((row as usize) >= ns && (row as usize) < g.num_nodes() - 1);
            let slot = g.policy_scatter_index.0[i];
            assert!(slot == OFF_WINDOW_SLOT || (slot as usize) < 19 * 19);
        }
        assert_eq!(g.builder_impl, BUILDER_IMPL_NATIVE);
        assert_eq!(g.n_nodes_checksum as usize, g.num_nodes());
    }
}
