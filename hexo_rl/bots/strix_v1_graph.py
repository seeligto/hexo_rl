"""Pure-Python reimplementation of strix's axis-graph builder.

Ported from hexo-rs/hexo-mcts/src/axis_graph.rs + hexo-mcts/src/graph.rs
(fill_threat_features) + hexo-engine/src/threat.rs (node_threat_features),
hexo-engine/src/hex.rs (hex_distance), hexo-engine/src/types.rs (WIN_AXES),
hexo-engine/src/legal_moves.rs (legal_moves) at SHA
c381ffbeb248313a1ec177eb650d9c3c2380caa8.

Node/edge/threat layout matches the Rust byte-exact builder for the
LEGACY schema (LeanOpts::default): axis graph, GINE edge_attr (E,5).

This is THE resolver — no routing through hexo_rl encoding/window code.
"""
from __future__ import annotations

WIN_AXES = [(1, 0), (0, 1), (1, -1)]


def hex_distance(a, b):
    dq = abs(b[0] - a[0])
    dr = abs(b[1] - a[1])
    ds = abs(b[0] - a[0] + b[1] - a[1])
    return max(dq, dr, ds)


def node_threat_features(stone_map, coord, to_move, win_length):
    """Port of hexo_engine::threat::node_threat_features.

    stone_map: dict {(q,r): player_int} where player_int 1=P1, -1=P2 (or
               any hashable player token; we compare equality).
    to_move:   the current player's token.
    Returns [own_max/wl, opp_max/wl, own_axes/3, opp_axes/3].
    """
    wl = int(win_length)
    n_cells = 2 * wl - 1
    opp = -to_move  # players are +1 / -1
    own_max = 0
    opp_max = 0
    own_axes = 0
    opp_axes = 0
    for (dq, dr) in WIN_AXES:
        cells = []
        for k in range(-(wl - 1), wl):  # -(wl-1) .. (wl-1) inclusive
            cells.append(stone_map.get((coord[0] + k * dq, coord[1] + k * dr)))
        axis_own = 0
        axis_opp = 0
        for start in range(wl):
            window = cells[start:start + wl]
            own_n = sum(1 for c in window if c == to_move)
            opp_n = sum(1 for c in window if c == opp)
            if opp_n == 0:
                axis_own = max(axis_own, own_n)
            if own_n == 0:
                axis_opp = max(axis_opp, opp_n)
        own_max = max(own_max, axis_own)
        opp_max = max(opp_max, axis_opp)
        if axis_own >= wl - 2:
            own_axes += 1
        if axis_opp >= wl - 2:
            opp_axes += 1
    return [own_max / wl, opp_max / wl, own_axes / 3.0, opp_axes / 3.0]


def legal_moves_from_stones(stone_map, radius):
    """Port of hexo_engine::legal_moves::legal_moves — empty cells within
    hex-distance <= radius of any stone, sorted lexicographically (q, r)."""
    offsets = []
    for dq in range(-radius, radius + 1):
        for dr in range(-radius, radius + 1):
            if max(abs(dq), abs(dr), abs(dq + dr)) <= radius:
                offsets.append((dq, dr))
    candidates = set()
    for (sq, sr) in stone_map.keys():
        for (dq, dr) in offsets:
            cell = (sq + dq, sr + dr)
            if cell not in stone_map:
                candidates.add(cell)
    return sorted(candidates)


def build_axis_graph_raw(
    stone_map,
    current_player,
    moves_remaining,
    *,
    win_length=6,
    radius=6,
    prune_empty_edges=False,
    threat_features=True,
    relative_stones=True,
):
    """Reproduce game_to_axis_graph_raw_opts(prune, threat, relative) for the
    LEGACY schema. Returns a dict with features (N,fdim), edge_index (2,E),
    edge_attr (E,5), legal_mask, stone_mask, coords, num_nodes, legal_coords.

    stone_map: dict {(q,r): player_int} (1=P1, -1=P2).
    current_player: +1 (P1) or -1 (P2); None -> terminal (treated as P2).
    moves_remaining: 1 or 2 (moves_remaining_this_turn).
    """
    # --- entry-point derivations (game_to_axis_graph_raw_lean) ---
    stones = sorted(stone_map.items(), key=lambda kv: kv[0])  # sort by coord
    legal = legal_moves_from_stones(stone_map, radius)
    player_feat = 1.0 if current_player == 1 else -1.0  # Some(P1)->1 else -1
    moves_feat = moves_remaining / 2.0
    own_is_p1 = player_feat > 0.0

    # --- NodeLayout (relative_stones + threat, legacy lean default) ---
    # relative base: [own, opp, empty, moves, norm_q, norm_r, inv_dist] = 7
    # absolute base: [p1, p2, empty, to_move, moves, norm_q, norm_r, inv_dist]=8
    if relative_stones:
        L_own, L_opp, L_empty = 0, 1, 2
        L_to_move = None
        L_moves = 3
        L_norm_q, L_norm_r = 4, 5
        L_inv = 6
        base_dim = 7
    else:
        L_own, L_opp, L_empty = 0, 1, 2
        L_to_move = 3
        L_moves = 4
        L_norm_q, L_norm_r = 5, 6
        L_inv = 7
        base_dim = 8
    fdim = base_dim + (4 if threat_features else 0)

    window = win_length - 1
    n_stones = len(stones)
    n_legal = len(legal)
    n_real = n_stones + n_legal
    n = n_real + 1
    dummy_idx = n_real

    coords = []
    coord_to_idx = {}
    node_kind = []  # ('stone', player) or ('empty', None)
    for i, ((q, r), player) in enumerate(stones):
        coords += [q, r]
        coord_to_idx[(q, r)] = i
        node_kind.append(("stone", player))
    for j, (q, r) in enumerate(legal):
        idx = n_stones + j
        coords += [q, r]
        coord_to_idx[(q, r)] = idx
        node_kind.append(("empty", None))
    coords += [0, 0]  # dummy

    # centroid + spread over stones
    if n_stones > 0:
        cq = sum(q for (q, r), _ in stones) / n_stones
        cr = sum(r for (q, r), _ in stones) / n_stones
        max_dev = 0.0
        for (q, r), _ in stones:
            max_dev = max(max_dev, max(abs(q - cq), abs(r - cr)))
        spread = max(max_dev, 1.0)
    else:
        cq, cr, spread = 0.0, 0.0, 1.0

    stone_coords = [c for c, _ in stones]

    features = [0.0] * (n * fdim)

    def set_coords(base, q, r):
        if L_norm_q is not None:
            features[base + L_norm_q] = float((q - cq) / spread)
        if L_norm_r is not None:
            features[base + L_norm_r] = float((r - cr) / spread)

    # Stone features
    for i, ((q, r), player) in enumerate(stones):
        base = i * fdim
        if relative_stones:
            col = L_opp if ((player == 1) != own_is_p1) else L_own
        else:
            col = L_own if player == 1 else L_opp
        features[base + col] = 1.0
        if L_to_move is not None:
            features[base + L_to_move] = player_feat
        features[base + L_moves] = moves_feat
        set_coords(base, q, r)
        # inv_dist stays 0 for stones

    # Legal move features
    for j in range(n_legal):
        idx = n_stones + j
        base = idx * fdim
        features[base + L_empty] = 1.0
        if L_to_move is not None:
            features[base + L_to_move] = player_feat
        features[base + L_moves] = moves_feat
        q, r = legal[j]
        set_coords(base, q, r)
        if stone_coords:
            min_d = min(hex_distance((q, r), sc) for sc in stone_coords)
        else:
            min_d = 1
        features[base + L_inv] = 1.0 / max(min_d, 1)

    # Dummy features
    dummy_base = dummy_idx * fdim
    if L_to_move is not None:
        features[dummy_base + L_to_move] = player_feat
    features[dummy_base + L_moves] = moves_feat

    # Masks
    legal_mask = [False] * n
    stone_mask = [False] * n
    for i in range(n_stones):
        stone_mask[i] = True
    for j in range(n_legal):
        legal_mask[n_stones + j] = True

    # --- Axis-window edges ---
    edge_src, edge_dst, edge_attr = [], [], []

    def kind_player_feat(kind):
        if kind[0] == "stone":
            return 1.0 if kind[1] == 1 else -1.0
        return 0.0

    for i in range(n_real):
        iq = coords[i * 2]
        ir = coords[i * 2 + 1]
        i_kind = node_kind[i]
        for axis_idx, (dq, dr) in enumerate(WIN_AXES):
            for sign in (1, -1):
                sdq = dq * sign
                sdr = dr * sign
                for d in range(1, window + 1):
                    tq = iq + sdq * d
                    tr = ir + sdr * d
                    j = coord_to_idx.get((tq, tr))
                    if j is None:
                        break
                    j_kind = node_kind[j]
                    both_empty = (i_kind[0] == "empty" and j_kind[0] == "empty")
                    if not (prune_empty_edges and both_empty):
                        src_i = kind_player_feat(i_kind)
                        src_j = kind_player_feat(j_kind)
                        signed_dist = float(d * sign)
                        # i -> j
                        edge_src.append(i)
                        edge_dst.append(j)
                        a = [0.0] * 5
                        a[axis_idx] = 1.0
                        a[3] = signed_dist
                        a[4] = src_i
                        edge_attr.append(a)
                        # j -> i
                        edge_src.append(j)
                        edge_dst.append(i)
                        a = [0.0] * 5
                        a[axis_idx] = 1.0
                        a[3] = -signed_dist
                        a[4] = src_j
                        edge_attr.append(a)
                    # walk stopping
                    if i_kind[0] == "stone":
                        should_stop = (j_kind[0] == "stone" and j_kind[1] != i_kind[1])
                    else:  # empty
                        should_stop = (j_kind[0] == "stone")
                    if should_stop:
                        break

    # --- dedup axis edges: key (src, dst, axis_idx), keep first ---
    def axis_idx_of(a):
        if a[0] > 0.5:
            return 0
        if a[1] > 0.5:
            return 1
        return 2

    seen = set()
    keep = []
    for e in range(len(edge_src)):
        key = (edge_src[e], edge_dst[e], axis_idx_of(edge_attr[e]))
        if key not in seen:
            seen.add(key)
            keep.append(e)
    if len(keep) < len(edge_src):
        edge_src = [edge_src[e] for e in keep]
        edge_dst = [edge_dst[e] for e in keep]
        edge_attr = [edge_attr[e] for e in keep]

    # --- legacy dummy edges: bidirectional to all real nodes, attr all-zero ---
    for i in range(n_real):
        edge_src.append(dummy_idx)
        edge_dst.append(i)
        edge_attr.append([0.0] * 5)
        edge_src.append(i)
        edge_dst.append(dummy_idx)
        edge_attr.append([0.0] * 5)

    # --- fill threat features (real nodes only; to_move from current player) ---
    if threat_features:
        # to_move: current_player (P1=1) else P2=-1 (matches
        # fill_threat_features which reads game.current_player()).
        to_move = 1 if current_player == 1 else -1
        for idx in range(n_real):
            c = (coords[idx * 2], coords[idx * 2 + 1])
            tf = node_threat_features(stone_map, c, to_move, win_length)
            base = idx * fdim
            for k in range(4):
                features[base + base_dim + k] = tf[k]

    return {
        "num_nodes": n,
        "fdim": fdim,
        "features": features,      # flat, row-major (n*fdim)
        "edge_src": edge_src,
        "edge_dst": edge_dst,
        "edge_attr": edge_attr,    # list of [5]
        "legal_mask": legal_mask,
        "stone_mask": stone_mask,
        "coords": coords,          # flat (n*2)
        "legal_coords": legal,     # sorted list of (q,r), aligned with policy order
        "base_dim": base_dim,
    }
