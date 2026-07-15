"""Tests for the production GNN net (WP-2 / C2, `docs/designs/gnn_integration_scope.md` §C2).

Covers: state-dict compatibility + landed-verify from the banked BC-prefit checkpoint
(`checkpoints/probes/gnn_bc/gnn_bc_040000.pt`), forward shape/finiteness on real self-play
positions, block-diagonal batch == per-graph forward parity, dist65 decode roundtrip sanity,
and the 6-in-a-row reachability test (the correctness risk named in the WP-2 charter: does
information from one end of a 6-line reach the empty completing cell through the GINE trunk).

All CPU, no GPU required.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from hexo_rl.bots.strix_v1_graph import build_axis_graph_raw
from hexo_rl.model.gnn_net import (
    BC_TRANSFER_PREFIXES,
    GnnDist65ValueHead,
    GnnNet,
    load_representation_policy_from_bc,
    segment_mean_with_fallback,
)
from hexo_rl.training.binned_value import N_VALUE_BINS, decode_binned_value

REPO = Path(__file__).resolve().parents[2]
BC_CHECKPOINT = REPO / "checkpoints" / "probes" / "gnn_bc" / "gnn_bc_040000.pt"
WPA_POSITIONS = REPO / "reports" / "probes" / "gnn_integration" / "wpa_positions.json"

WIN_LENGTH = 6
RADIUS = 6


def _build(stone_map, current_player, moves_remaining):
    return build_axis_graph_raw(
        stone_map, current_player, moves_remaining,
        win_length=WIN_LENGTH, radius=RADIUS,
        prune_empty_edges=True, threat_features=True, relative_stones=True,
    )


def _load_wpa_positions(n: int):
    data = json.loads(WPA_POSITIONS.read_text())
    out = []
    for rec in data["positions"][:n]:
        stone_map = {(q, r): p for (q, r, p) in rec["stones"]}
        out.append((stone_map, rec["current_player"], rec["moves_remaining"]))
    return out


def _tensors_from_graph(g):
    n, fdim = g["num_nodes"], g["fdim"]
    x = torch.tensor(g["features"], dtype=torch.float32).reshape(n, fdim)
    e = len(g["edge_src"])
    if e:
        ei = torch.tensor([g["edge_src"], g["edge_dst"]], dtype=torch.int64)
        ea = torch.tensor(g["edge_attr"], dtype=torch.float32).reshape(e, 5)
    else:
        ei = torch.zeros((2, 0), dtype=torch.int64)
        ea = torch.zeros((0, 5), dtype=torch.float32)
    lm = torch.tensor(g["legal_mask"], dtype=torch.bool)
    sm = torch.tensor(g["stone_mask"], dtype=torch.bool)
    return x, ei, ea, lm, sm


def _collate(graphs):
    """Block-diagonal disjoint-union collate — same convention as
    `scripts/research/gnn_infer_bench.py::collate_batch` / the WP-B contract
    shape, extended with `stone_mask` + `node_offsets`."""
    xs, eis, eas, lms, sms, offsets = [], [], [], [], [], [0]
    node_offset = 0
    for g in graphs:
        x, ei, ea, lm, sm = _tensors_from_graph(g)
        xs.append(x)
        eis.append(ei + node_offset)
        eas.append(ea)
        lms.append(lm)
        sms.append(sm)
        node_offset += x.shape[0]
        offsets.append(node_offset)
    return (
        torch.cat(xs, 0), torch.cat(eis, 1), torch.cat(eas, 0),
        torch.cat(lms, 0), torch.cat(sms, 0),
        torch.tensor(offsets, dtype=torch.long),
    )


# ── param count ────────────────────────────────────────────────────────────────

def test_param_count_probe_284k_class():
    net = GnnNet()
    # representation + policy_head params must equal the probe's (byte-identical
    # construction — GnnBcNet.num_params() == 283_970 total, including its own
    # unused scalar value head, `test_gnn_bc_probe.py::test_gnn_net_param_count_and_forward`).
    # Direct check against that total minus the probe's scalar value head
    # (in_dim=512, value_hidden=32 -> 512*32+32 + 32*1+1 = 16449 params).
    rep_policy = sum(
        p.numel() for name, p in net.named_parameters()
        if name.startswith(("representation.", "policy_head."))
    )
    probe_value_head_params = 512 * 32 + 32 + 32 * 1 + 1
    assert rep_policy == 283_970 - probe_value_head_params

    # The new dist65 value head is a SEPARATE architecture (bin-logit tail vs
    # scalar-tanh tail); report the net's own total honestly (not 283_970).
    total = net.num_params()
    value_head_params = sum(
        p.numel() for name, p in net.named_parameters() if name.startswith("value_head.")
    )
    assert total == rep_policy + value_head_params


# ── state-dict compatibility + landed-verify (mission item 1) ──────────────────

@pytest.mark.skipif(not BC_CHECKPOINT.exists(), reason="BC-prefit checkpoint not present")
def test_state_dict_loads_and_landed_verifies_from_bc_prefit():
    ckpt = torch.load(BC_CHECKPOINT, map_location="cpu", weights_only=False)
    bc_state = ckpt["model_state_dict"]

    net = GnnNet()
    result = load_representation_policy_from_bc(net, bc_state, verify_n=5)

    assert result["loaded_keys"], "no representation/policy_head keys loaded"
    # 5 sampled per prefix (or fewer if a prefix has < 5 tensors) -> >= 2 verified.
    assert result["verified_tensors"] >= 2 * min(
        5, min(
            sum(1 for k in bc_state if k.startswith(p)) for p in BC_TRANSFER_PREFIXES
        )
    )

    # Every representation.*/policy_head.* tensor in net now matches the source
    # exactly (not just the sampled subset) -- the strong form of the guard.
    net_sd = net.state_dict()
    for k in result["loaded_keys"]:
        assert torch.equal(net_sd[k], bc_state[k]), f"{k} did not load byte-exact"

    # value_head is untouched by the BC transfer (fresh init, E1 REVIVE).
    assert any(name.startswith("value_head.") for name, _ in net.named_parameters())


@pytest.mark.skipif(not BC_CHECKPOINT.exists(), reason="BC-prefit checkpoint not present")
def test_state_dict_mismatch_raises():
    ckpt = torch.load(BC_CHECKPOINT, map_location="cpu", weights_only=False)
    bc_state = dict(ckpt["model_state_dict"])
    # Drop one representation key -> must raise, not silently partial-load (F1 guard).
    dropped_key = next(k for k in bc_state if k.startswith("representation."))
    del bc_state[dropped_key]
    net = GnnNet()
    with pytest.raises(RuntimeError):
        load_representation_policy_from_bc(net, bc_state)


# ── forward shape / finiteness on real self-play positions ─────────────────────

@pytest.mark.skipif(not WPA_POSITIONS.exists(), reason="wpa_positions.json not present")
def test_forward_single_shape_and_finiteness_on_real_positions():
    net = GnnNet()
    net.eval()
    for stone_map, cp, mr in _load_wpa_positions(8):
        g = _build(stone_map, cp, mr)
        x, ei, ea, lm, sm = _tensors_from_graph(g)
        policy_logits, value, bin_logits = net.forward_single(x, ei, ea, lm, sm)
        n_legal = int(lm.sum().item())
        assert policy_logits.shape == (n_legal,)
        assert torch.isfinite(policy_logits).all()
        assert bin_logits.shape == (N_VALUE_BINS,)
        assert torch.isfinite(bin_logits).all()
        assert value.dim() == 0
        assert -1.0 <= float(value) <= 1.0


@pytest.mark.skipif(not WPA_POSITIONS.exists(), reason="wpa_positions.json not present")
def test_forward_batch_shape_and_finiteness_on_real_positions():
    net = GnnNet()
    net.eval()
    positions = _load_wpa_positions(6)
    graphs = [_build(sm, cp, mr) for sm, cp, mr in positions]
    x, ei, ea, lm, sm, offsets = _collate(graphs)
    with torch.no_grad():
        policy_logits, value, bin_logits = net.forward_batch(x, ei, ea, lm, sm, offsets)
    n_legal_total = int(lm.sum().item())
    b = len(graphs)
    assert policy_logits.shape == (n_legal_total,)
    assert value.shape == (b, 1)
    assert bin_logits.shape == (b, N_VALUE_BINS)
    assert torch.isfinite(policy_logits).all()
    assert torch.isfinite(value).all()
    assert torch.isfinite(bin_logits).all()
    assert (value >= -1.0).all() and (value <= 1.0).all()


# ── block-diagonal batch == per-graph forward parity ────────────────────────────

@pytest.mark.skipif(not WPA_POSITIONS.exists(), reason="wpa_positions.json not present")
def test_block_diagonal_batch_matches_per_graph_forward():
    torch.manual_seed(0)
    net = GnnNet()
    net.eval()
    positions = _load_wpa_positions(3)
    graphs = [_build(sm, cp, mr) for sm, cp, mr in positions]

    x, ei, ea, lm, sm, offsets = _collate(graphs)
    with torch.no_grad():
        batch_policy, batch_value, batch_bins = net.forward_batch(x, ei, ea, lm, sm, offsets)

    legal_cursor = 0
    for i, g in enumerate(graphs):
        xg, eig, eag, lmg, smg = _tensors_from_graph(g)
        pol, val, binlog = net.forward_single(xg, eig, eag, lmg, smg)
        n_legal = pol.shape[0]
        torch.testing.assert_close(
            batch_policy[legal_cursor:legal_cursor + n_legal], pol, atol=1e-5, rtol=1e-5
        )
        torch.testing.assert_close(batch_value[i], val.unsqueeze(0), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(batch_bins[i], binlog, atol=1e-5, rtol=1e-5)
        legal_cursor += n_legal
    assert legal_cursor == batch_policy.shape[0]


def test_segment_mean_with_fallback_matches_manual_per_graph_mean():
    torch.manual_seed(1)
    emb = torch.randn(9, 4)
    # graph0: nodes 0-3, stones {1,2}; graph1: nodes 4-8, NO stones -> fallback.
    mask = torch.tensor([False, True, True, False, False, False, False, False, False])
    batch_vec = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 1])
    pooled = segment_mean_with_fallback(emb, mask, batch_vec, num_graphs=2)
    torch.testing.assert_close(pooled[0], emb[1:3].mean(dim=0), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(pooled[1], emb[4:9].mean(dim=0), atol=1e-6, rtol=1e-6)  # fallback


# ── dist65 decode roundtrip sanity ──────────────────────────────────────────────

def test_dist65_head_decode_matches_shared_primitive():
    torch.manual_seed(2)
    head = GnnDist65ValueHead(in_dim=16, hidden=8)
    pooled = torch.randn(5, 16)
    value, bin_logits = head(pooled)
    assert bin_logits.shape == (5, N_VALUE_BINS)
    assert value.shape == (5, 1)
    torch.testing.assert_close(value, decode_binned_value(bin_logits), atol=1e-6, rtol=1e-6)
    assert (value >= -1.0).all() and (value <= 1.0).all()


def test_dist65_decode_extreme_bins_hit_support_endpoints():
    # All mass on bin 0 -> value -1; all mass on the last bin -> value +1.
    logits = torch.full((2, N_VALUE_BINS), -1e4)
    logits[0, 0] = 1e4
    logits[1, N_VALUE_BINS - 1] = 1e4
    value = decode_binned_value(logits)
    assert torch.isclose(value[0, 0], torch.tensor(-1.0), atol=1e-3)
    assert torch.isclose(value[1, 0], torch.tensor(1.0), atol=1e-3)


# ── 6-in-a-row reachability (THE correctness risk: under-reaching) ─────────────

def test_reachability_completing_cell_sees_far_end_stone_gradient():
    """5 own stones (0,0)..(4,0) [a 6-in-a-row minus the last cell] + the empty
    completing cell (5,0). Assert the completing cell's policy logit has NONZERO
    gradient w.r.t. the FAR-end stone's ((0,0), the opposite end of the line)
    input node features, through the full 4-layer GINE trunk + policy head.

    Honest finding (see WP2_net.md): the axis-window walk
    (`strix_v1_graph.py:220-262`) connects node i directly to every same-axis
    node within `window = win_length - 1 = 5` steps in ONE edge -- not a chain of
    adjacent-cell edges. A 5-stone line spans exactly `window` cells, so the
    builder emits a DIRECT edge from (0,0) to (5,0); this test proves that direct
    edge's information survives representation + policy MLP with nonzero
    gradient (a real risk: index_select/scatter bugs, dead ReLU, or a detached
    graph could all silently zero it), not that 4 message-passing hops are
    load-bearing for this exact span.
    """
    stone_map = {(0, 0): 1, (1, 0): 1, (2, 0): 1, (3, 0): 1, (4, 0): 1}
    g = _build(stone_map, current_player=1, moves_remaining=2)

    n_stones = int(sum(g["stone_mask"]))
    far_end_idx = 0  # sorted stone order: (0,0) is first (lexicographic sort).
    assert g["coords"][far_end_idx * 2:far_end_idx * 2 + 2] == [0, 0]

    completing_local = g["legal_coords"].index((5, 0))
    completing_idx = n_stones + completing_local
    assert g["legal_mask"][completing_idx] is True

    x, ei, ea, lm, sm = _tensors_from_graph(g)
    x.requires_grad_(True)

    net = GnnNet()
    net.train()
    emb = net.representation(x, ei, ea)
    legal_emb = emb[lm]
    policy_logits = net.policy_head.mlp(legal_emb).squeeze(-1)

    legal_positions = [i for i, v in enumerate(g["legal_mask"]) if v]
    slot = legal_positions.index(completing_idx)
    target_logit = policy_logits[slot]
    target_logit.backward()

    assert x.grad is not None
    far_end_grad = x.grad[far_end_idx]
    grad_norm = far_end_grad.abs().sum().item()
    assert grad_norm > 0.0, (
        "REACHABILITY FAIL: completing cell's policy logit has ZERO gradient "
        "w.r.t. far-end stone features -- the axis-window edge did not span, or "
        "message passing / policy MLP severed the gradient path."
    )


# ── zero-stone fallback, end-to-end (WP2 review SHOULD-FIX #6) ─────────────────


def test_zero_stone_fallback_both_paths_end_to_end():
    """Empty board (MCTS-root scenario before any stone lands in a fresh graph):
    stone_mask is all-False, so value pooling must take the all-nodes fallback in
    BOTH forward paths, without NaN/crash, and the two paths must agree."""
    g = _build({}, 1, 1)  # empty board -> dummy/legal nodes only, zero stones
    x, ei, ea, lm, sm = _tensors_from_graph(g)
    assert not sm.any(), "fixture must actually exercise the zero-stone branch"

    net = GnnNet()
    net.eval()

    pol_s, val_s, bins_s = net.forward_single(x, ei, ea, lm, sm)
    with torch.no_grad():
        pol_b, val_b, bins_b = net.forward_batch(
            x, ei, ea, lm, sm,
            node_offsets=torch.tensor([0, x.shape[0]], dtype=torch.long),
        )

    for t in (pol_s, val_s, bins_s, pol_b, val_b, bins_b):
        assert torch.isfinite(t).all()
    assert pol_s.shape[0] == int(lm.sum())
    assert torch.allclose(pol_s, pol_b, atol=1e-6)
    assert torch.allclose(val_s, val_b.squeeze(0).squeeze(-1), atol=1e-6)
    assert torch.allclose(bins_s, bins_b.squeeze(0), atol=1e-6)


def test_masks_must_be_bool():
    """uint8 masks ride torch's deprecated uint8-as-mask path today; when torch
    removes it they become integer indices (silent wrong-row gather). The module
    refuses non-bool masks outright (red-team GAP-1)."""
    g = _build({(0, 0): 1, (1, 0): 2}, 1, 2)
    x, ei, ea, lm, sm = _tensors_from_graph(g)
    net = GnnNet()
    net.eval()
    with pytest.raises(AssertionError, match="legal_mask must be bool"):
        net.forward_single(x, ei, ea, lm.to(torch.uint8), sm)
    with pytest.raises(AssertionError, match="stone_mask must be bool"):
        net.forward_batch(x, ei, ea, lm, sm.to(torch.uint8))
