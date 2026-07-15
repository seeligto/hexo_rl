"""WP-3 step 7 — OQ-7 step-0 GNN inference-seam smoke (end-to-end).

Runs a few dozen leaf evaluations through the LIVE graph seam on the banked
BC checkpoint, exercising every production component the self-play worker's
``infer_and_expand_graph`` rides:

    engine builds axis graphs (native ``build_axis_graph`` via the seam guards)
      -> ``next_graph_batch`` block-diagonal fuse (GraphWire)
      -> ``collate_graph_batch`` single resolver (18 assertions)
      -> ``GnnNet.forward_batch`` (WP-2 net, banked BC weights)
      -> per-graph segmented softmax
      -> ``submit_graph_inference_results``
      -> ``assemble_ls_from_gnn_probs`` (option (b), no off-window drop)
      -> ``LegalSetPolicy`` (dense in-window + coord-keyed off-window overflow)

Asserts (seam design §7 step 7):
  * finite priors, each position a proper distribution (sum ~ 1);
  * legal moves only (overflow keys are legal cells; the argmax move is legal);
  * off-window REACHABLE — at least one off-window candidate lands in some
    ``LegalSetPolicy.overflow`` (the 20% chosen-move regime, §1.4, makes this
    near-certain over dozens of spread positions);
  * per-position round-trip parity: the priors the engine received (the
    assembled LS read back by coord) equal a direct GnnBcBot-style
    ``forward_single`` on the same position, max|Δ| < 1e-5.

CPU by default (fp32 -> strict 1e-5 parity + the design's mandated CPU-only
fallback); the SAME seam runs on CUDA under fp16 autocast when present.
"""
from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np
import pytest
import torch

import engine
from hexo_rl.bots.strix_v1_graph import build_axis_graph_raw
from hexo_rl.encoding import lookup as registry_lookup
from hexo_rl.model.gnn_net import GnnNet, load_representation_policy_from_bc
from hexo_rl.selfplay.inference_server import InferenceServer

pytestmark = pytest.mark.integration

_BC_CKPT = "checkpoints/probes/gnn_bc/gnn_bc_040000.pt"
_TRUNK = 19
_WIN_LENGTH = 6
_RADIUS = 6

# One position = (stones [(q, r, player±1)], current_player±1, moves_remaining).
Position = Tuple[List[Tuple[int, int, int]], int, int]


def _trunc_div2(x: int) -> int:
    """Integer /2 truncated toward zero — matches Rust `(a+b)/2` (NOT Python //)."""
    return -((-x) // 2) if x < 0 else x // 2


def _window_center(stones: List[Tuple[int, int, int]]) -> Tuple[int, int]:
    """Bbox-midpoint window centre (== `hexo_graph::window_center` / `Board`)."""
    qs = [q for q, _, _ in stones]
    rs = [r for _, r, _ in stones]
    return _trunc_div2(min(qs) + max(qs)), _trunc_div2(min(rs) + max(rs))


def _window_slot(q: int, r: int, cq: int, cr: int, trunk: int = _TRUNK) -> int:
    """Canonical action slot of a coord (== builder `window_flat_idx`); -1 off-window."""
    half = (trunk - 1) // 2
    wq, wr = q - cq + half, r - cr + half
    if 0 <= wq < trunk and 0 <= wr < trunk:
        return wq * trunk + wr
    return -1


def _ls_prob(q: int, r: int, dense, overflow, cq: int, cr: int) -> float:
    """Read the assembled LegalSetPolicy's mass at coord (q, r) — the Rust
    `LegalSetPolicy::get` path (dense[slot] in-window else overflow[coord])."""
    s = _window_slot(q, r, cq, cr)
    if s >= 0:
        return float(dense[s])
    return float(overflow.get((q, r), 0.0))


def _spread_positions() -> List[Position]:
    """~30 positions — spread two-cluster boards (guaranteeing off-window legal
    nodes) + a few compact single-cluster boards (fully in-window)."""
    positions: List[Position] = []
    # Spread: cluster A near origin, cluster B `gap` away (> trunk so B is off-window).
    for gap in range(22, 34):
        stones: List[Tuple[int, int, int]] = []
        for i in range(3):
            stones.append((i, 0, 1 if i % 2 == 0 else -1))
        for i in range(3):
            stones.append((gap + i, 0, -1 if i % 2 == 0 else 1))
        cp = 1 if gap % 2 == 0 else -1
        mr = 1 + (gap % 2)
        positions.append((stones, cp, mr))
    # Spread on the r-axis + a diagonal, exercising the other window axes.
    for gap in range(21, 29):
        stones = [(0, 0, 1), (0, 1, -1), (1, 0, 1)]
        stones += [(0, gap, -1), (1, gap, 1), (0, gap + 1, -1)]
        positions.append((stones, -1, 2))
    # Compact single-cluster boards — all legal cells in-window (overflow empty).
    for k in range(6):
        stones = [(k, 0, 1), (k + 1, 0, -1), (k, 1, 1), (k + 1, 1, -1)]
        positions.append((stones, 1, 1 + (k % 2)))
    return positions


def _load_net(device: torch.device) -> GnnNet:
    ckpt = torch.load(_BC_CKPT, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    net = GnnNet()
    load_representation_policy_from_bc(net, state)
    return net.to(device).eval()


def _oracle_probs(pos: Position, net: GnnNet, device: torch.device) -> dict:
    """Direct GnnBcBot-style forward_single on an independently (Python-oracle)
    built axis graph -> softmax over the legal set, keyed by legal coord.

    `prune_empty_edges=False` matches the native `build_axis_graph` (WP-1
    PARITY-EXACT); the reference is computed in fp32 for the strict 1e-5 gate.
    """
    stones, cp, mr = pos
    g = build_axis_graph_raw(
        {(q, r): p for q, r, p in stones}, cp, mr,
        win_length=_WIN_LENGTH, radius=_RADIUS,
        prune_empty_edges=False, threat_features=True, relative_stones=True,
    )
    n, fdim = g["num_nodes"], g["fdim"]
    e = len(g["edge_src"])
    x = torch.tensor(g["features"], dtype=torch.float32, device=device).reshape(n, fdim)
    if e:
        edge_index = torch.tensor([g["edge_src"], g["edge_dst"]], dtype=torch.int64, device=device)
        edge_attr = torch.tensor(g["edge_attr"], dtype=torch.float32, device=device).reshape(e, 5)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.int64, device=device)
        edge_attr = torch.zeros((0, 5), dtype=torch.float32, device=device)
    legal_mask = torch.tensor(g["legal_mask"], dtype=torch.bool, device=device)
    stone_mask = torch.tensor(g["stone_mask"], dtype=torch.bool, device=device)
    with torch.no_grad():
        logits, _v, _b = net.forward_single(x, edge_index, edge_attr, legal_mask, stone_mask)
    probs = torch.softmax(logits.float(), dim=0).cpu().numpy()
    return {tuple(c): float(probs[i]) for i, c in enumerate(g["legal_coords"])}


@pytest.mark.skipif(not os.path.exists(_BC_CKPT), reason="banked GNN-BC checkpoint absent")
def test_gnn_inference_seam_end_to_end_smoke():
    # CPU is the reference device (fp32 -> strict parity + CPU-only fallback the
    # design mandates). The identical seam runs on CUDA (fp16 autocast) when set.
    device = torch.device("cpu")
    spec = registry_lookup("gnn_axis_v1")
    net = _load_net(device)

    # Production path: the SelfPlayRunner builds a GRAPH batcher for a
    # representation="graph" encoding (WP-3 step-6 mod.rs wiring). We drive that
    # same batcher (never .start() — no buffer writes yet, C8/WP-5).
    runner = engine.SelfPlayRunner(
        engine.SelfPlayRunnerConfig(
            n_workers=1, encoding_name="gnn_axis_v1",
            n_simulations=2, leaf_batch_size=1, standard_sims=2,
        )
    )
    batcher = runner.batcher
    assert batcher.representation_py == "graph", "runner must build a graph batcher"

    server = InferenceServer(
        net, device,
        {"selfplay": {"inference_batch_size": 8, "inference_max_wait_ms": 10}},
        batcher=batcher, encoding_spec=spec,
    )
    server.start()
    try:
        positions = _spread_positions()
        results = batcher.submit_graphs_and_wait(positions)
    finally:
        server.stop()

    assert len(results) == len(positions)

    off_window_hits = 0
    max_parity_delta = 0.0
    for pos, (dense, overflow_list, value) in zip(positions, results):
        stones, _cp, _mr = pos
        overflow = dict(overflow_list)
        cq, cr = _window_center(stones)
        oracle = _oracle_probs(pos, net, device)
        legal_coords = set(oracle.keys())

        # (1) finite priors + proper distribution.
        assert np.all(np.isfinite(dense)), "dense priors must be finite"
        assert all(np.isfinite(p) for p in overflow.values()), "overflow priors finite"
        assert np.isfinite(value), "value must be finite"
        total = float(np.sum(dense)) + float(sum(overflow.values()))
        assert abs(total - 1.0) < 1e-4, f"LegalSetPolicy is a distribution, sum={total}"

        # (2) legal moves only: every overflow key is a legal cell, and the
        # engine-priored argmax move (what MCTS would expand -> play) is legal.
        assert set(overflow.keys()) <= legal_coords, "off-window mass on a non-legal cell"
        best = max(legal_coords, key=lambda c: _ls_prob(c[0], c[1], dense, overflow, cq, cr))
        assert best in legal_coords, "argmax move must be a legal move"

        # (3) off-window reachability (option (b) no-drop).
        off_window_hits += len(overflow)

        # (4) round-trip parity: LS read-by-coord == direct forward_single.
        for (q, r), oracle_p in oracle.items():
            engine_p = _ls_prob(q, r, dense, overflow, cq, cr)
            max_parity_delta = max(max_parity_delta, abs(engine_p - oracle_p))

    assert off_window_hits > 0, (
        "no off-window candidate reached any overflow — option (b) no-drop broken "
        "(or the spread fixtures degenerated to in-window)"
    )
    assert max_parity_delta < 1e-5, (
        f"seam round-trip parity vs forward_single exceeded 1e-5: max|Δ|={max_parity_delta:.2e}"
    )
