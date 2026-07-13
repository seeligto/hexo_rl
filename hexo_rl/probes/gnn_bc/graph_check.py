"""Independent re-derivation of the strix axis-graph edge set + 5-dim edge
features — for the WP3 graph-builder numerical cross-check ONLY.

This is a SECOND, from-scratch implementation of the axis-window edge topology,
written deliberately WITHOUT importing ``strix_v1_graph`` so it can catch a
subtly-wrong builder (a subtly-wrong builder would fake an ARCH-NULL by feeding
the GNN a corrupted representation). The GNN-BC arm itself uses the
fidelity-gated ``hexo_rl.bots.strix_v1_graph.build_axis_graph_raw`` unchanged;
``cross_check.py`` compares that reference against THIS re-derivation on 10
positions. Exact match on both = both validated.

Scope: edge SET + edge_attr only (the part most prone to a silent off-by-one in
the axis walk / stop rule / dedup / dummy edges). Node features + threat features
are separately gated by ``tests/test_strix_v1_bot.py`` against strix's Rust unit
vectors; the cross-check re-derives node coords + the legal-node set here too so
the edge endpoints are index-comparable.

Ported semantics (independently, from the same public Rust source at SHA
c381ffbeb248313a1ec177eb650d9c3c2380caa8): axis-window edges over 3 WIN_AXES,
both-direction walk of length up to win_length-1, stop-at-opponent (for a stone
source) / stop-at-any-stone (for an empty source), per-(src,dst,axis) dedup, and
the bidirectional all-zero dummy-node edges.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

WIN_AXES = [(1, 0), (0, 1), (1, -1)]

Coord = Tuple[int, int]


def _hex_dist(a: Coord, b: Coord) -> int:
    dq = abs(b[0] - a[0])
    dr = abs(b[1] - a[1])
    ds = abs((b[0] - a[0]) + (b[1] - a[1]))
    return max(dq, dr, ds)


def _legal_nodes(stone_map: Dict[Coord, int], radius: int) -> List[Coord]:
    """Empty cells within hex-distance <= radius of any stone, sorted (q, r).

    Independent re-derivation (bounding-box scan + hex-distance filter), NOT
    the offset-set form used by the reference builder.
    """
    out = set()
    for (sq, sr) in stone_map:
        for dq in range(-radius, radius + 1):
            for dr in range(-radius, radius + 1):
                if max(abs(dq), abs(dr), abs(dq + dr)) <= radius:
                    cell = (sq + dq, sr + dr)
                    if cell not in stone_map:
                        out.add(cell)
    return sorted(out)


def node_index_layout(stone_map: Dict[Coord, int], radius: int):
    """Return (node_coords, coord_to_idx, kinds, dummy_idx).

    Node order = sorted stones, then sorted legal (empty) nodes, then one dummy.
    kinds[i] = ("stone", player) | ("empty", None) | ("dummy", None).
    """
    stones = sorted(stone_map.items(), key=lambda kv: kv[0])
    legal = _legal_nodes(stone_map, radius)
    coords: List[Coord] = []
    kinds: List[Tuple[str, object]] = []
    coord_to_idx: Dict[Coord, int] = {}
    for (q, r), p in stones:
        coord_to_idx[(q, r)] = len(coords)
        coords.append((q, r))
        kinds.append(("stone", p))
    for (q, r) in legal:
        coord_to_idx[(q, r)] = len(coords)
        coords.append((q, r))
        kinds.append(("empty", None))
    dummy_idx = len(coords)
    coords.append((0, 0))
    kinds.append(("dummy", None))
    return coords, coord_to_idx, kinds, dummy_idx


def _src_player_feat(kind) -> float:
    if kind[0] == "stone":
        return 1.0 if kind[1] == 1 else -1.0
    return 0.0


def axis_edge_set(stone_map: Dict[Coord, int], *, win_length: int = 6,
                  radius: int = 6, prune_empty_edges: bool = True):
    """Independently build the axis-window edge set + attrs + dummy edges.

    Returns a dict:
      ``axis_edges``: dict {(src, dst, axis_idx): (signed_dist, src_player)} —
                      the deduped axis edges (first-seen kept, matching the
                      reference's first-wins dedup).
      ``dummy_pairs``: set of (src, dst) for the all-zero dummy edges.
      ``coords``, ``kinds``, ``dummy_idx``: the node layout (for index compare).

    The 5-dim edge_attr of an axis edge is reconstructable as
    ``[axis one-hot(axis_idx), signed_dist, src_player]``; the dummy edge attr is
    all-zero by construction. Keyed by (src, dst, axis_idx) so a set-equality
    check against the reference is order-independent.
    """
    coords, coord_to_idx, kinds, dummy_idx = node_index_layout(stone_map, radius)
    n_real = dummy_idx  # real nodes are indices [0, dummy_idx)
    window = win_length - 1

    axis_edges: Dict[Tuple[int, int, int], Tuple[float, float]] = {}

    def add(src: int, dst: int, axis_idx: int, signed_dist: float, src_player: float):
        key = (src, dst, axis_idx)
        if key not in axis_edges:  # first-wins dedup (mirrors reference)
            axis_edges[key] = (signed_dist, src_player)

    for i in range(n_real):
        iq, ir = coords[i]
        i_kind = kinds[i]
        for axis_idx, (dq, dr) in enumerate(WIN_AXES):
            for sign in (1, -1):
                sdq, sdr = dq * sign, dr * sign
                for d in range(1, window + 1):
                    tq, tr = iq + sdq * d, ir + sdr * d
                    j = coord_to_idx.get((tq, tr))
                    if j is None:
                        break
                    j_kind = kinds[j]
                    both_empty = (i_kind[0] == "empty" and j_kind[0] == "empty")
                    if not (prune_empty_edges and both_empty):
                        signed_dist = float(d * sign)
                        add(i, j, axis_idx, signed_dist, _src_player_feat(i_kind))
                        add(j, i, axis_idx, -signed_dist, _src_player_feat(j_kind))
                    # walk-stop rule
                    if i_kind[0] == "stone":
                        should_stop = (j_kind[0] == "stone" and j_kind[1] != i_kind[1])
                    else:  # empty source
                        should_stop = (j_kind[0] == "stone")
                    if should_stop:
                        break

    dummy_pairs = set()
    for i in range(n_real):
        dummy_pairs.add((dummy_idx, i))
        dummy_pairs.add((i, dummy_idx))

    return {
        "axis_edges": axis_edges,
        "dummy_pairs": dummy_pairs,
        "coords": coords,
        "kinds": kinds,
        "dummy_idx": dummy_idx,
    }


def reference_edge_set(g: dict):
    """Project the REFERENCE builder's output (``build_axis_graph_raw`` dict)
    into the same (axis_edges, dummy_pairs) shape for comparison.

    An edge is a dummy edge iff its edge_attr is all-zero (axis one-hot all
    zero); everything else is an axis edge keyed by its argmax axis one-hot.
    """
    axis_edges: Dict[Tuple[int, int, int], Tuple[float, float]] = {}
    dummy_pairs = set()
    src, dst, attr = g["edge_src"], g["edge_dst"], g["edge_attr"]
    for e in range(len(src)):
        a = attr[e]
        onehot = a[0:3]
        if onehot[0] == 0.0 and onehot[1] == 0.0 and onehot[2] == 0.0:
            dummy_pairs.add((src[e], dst[e]))
            continue
        axis_idx = 0 if onehot[0] > 0.5 else (1 if onehot[1] > 0.5 else 2)
        key = (src[e], dst[e], axis_idx)
        if key not in axis_edges:  # reference already deduped; keep first
            axis_edges[key] = (a[3], a[4])
    return {"axis_edges": axis_edges, "dummy_pairs": dummy_pairs}
