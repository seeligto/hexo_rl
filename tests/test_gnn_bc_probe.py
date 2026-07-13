"""Tests for the GNN-BC probe (D-L WP3).

Covers the CPU verifications: graph-builder cross-check (independent re-derivation
== fidelity-gated reference on 10 positions), both nets instantiate at the target
param count + one finite forward pass, the GNN grad path flows through the trained
modules, and both raw-policy bots load a checkpoint + return a legal move.

The bot tests write/read a 1-step BC checkpoint to a tmp dir (no real training).
"""
import importlib.util
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]

from hexo_rl.bots.strix_v1_graph import build_axis_graph_raw
from hexo_rl.probes.gnn_bc.graph_check import axis_edge_set, reference_edge_set
from hexo_rl.probes.gnn_bc.gnn_bc_net import GnnBcNet
from hexo_rl.probes.gnn_bc.cnn_bc_net import build_cnn_bc_net, num_params
from hexo_rl.probes.gnn_bc.train_bc import (
    _collate_gnn, _collate_gnn_dict, _compact_from_graph, _gnn_loss,
    _gnn_loss_reference, parallel_ordered_map,
)


def _positions():
    # Load the strix fidelity fixtures by explicit path (the bare name `fixtures`
    # is shadowed by tests/fixtures/ when the suite runs together).
    fix_path = REPO / "reports/tourney/strix_fidelity/fixtures.py"
    spec = importlib.util.spec_from_file_location("_strix_fixtures", fix_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.POSITIONS[:10], mod.WIN_LENGTH, mod.RADIUS


# ── graph-builder cross-check ─────────────────────────────────────────────────

def test_graph_builder_cross_check_10_positions():
    positions, WL, RAD = _positions()
    for pos in positions:
        stone_map = {tuple(k): v for k, v in pos["stones"]}
        g = build_axis_graph_raw(
            stone_map, pos["current_player"], pos["moves_remaining"],
            win_length=WL, radius=RAD,
            prune_empty_edges=True, threat_features=True, relative_stones=True,
        )
        ref = reference_edge_set(g)
        ind = axis_edge_set(stone_map, win_length=WL, radius=RAD, prune_empty_edges=True)
        # edge SET
        assert set(ref["axis_edges"]) == set(ind["axis_edges"]), pos["name"]
        # 5-dim edge features (signed_dist, src_player) on every axis edge
        for k in ref["axis_edges"]:
            assert ref["axis_edges"][k] == ind["axis_edges"][k], (pos["name"], k)
        # dummy edges
        assert ref["dummy_pairs"] == ind["dummy_pairs"], pos["name"]


# ── param counts + forward ────────────────────────────────────────────────────

def test_gnn_net_param_count_and_forward():
    net = GnnBcNet()
    assert net.num_params() == 283_970
    g = build_axis_graph_raw({(0, 0): 1, (1, 0): -1, (2, 0): 1}, 1, 2,
                             win_length=6, radius=6, prune_empty_edges=True,
                             threat_features=True, relative_stones=True)
    n, fdim = g["num_nodes"], g["fdim"]
    x = torch.tensor(g["features"]).float().reshape(n, fdim)
    E = len(g["edge_src"])
    ei = torch.tensor([g["edge_src"], g["edge_dst"]])
    ea = torch.tensor(g["edge_attr"]).float().reshape(E, 5)
    lm = torch.tensor(g["legal_mask"])
    logits = net.forward_batch(x, ei, ea, lm)
    assert logits.shape[0] == int(lm.sum().item())
    assert torch.isfinite(logits).all()


def test_gnn_grad_flows_through_trained_path():
    net = GnnBcNet(); net.train()
    g = build_axis_graph_raw({(0, 0): 1, (1, 0): -1, (2, 0): 1}, 1, 2,
                             win_length=6, radius=6, prune_empty_edges=True,
                             threat_features=True, relative_stones=True)
    n, fdim = g["num_nodes"], g["fdim"]
    x = torch.tensor(g["features"]).float().reshape(n, fdim)
    E = len(g["edge_src"])
    ei = torch.tensor([g["edge_src"], g["edge_dst"]])
    ea = torch.tensor(g["edge_attr"]).float().reshape(E, 5)
    lm = torch.tensor(g["legal_mask"])
    net.forward_batch(x, ei, ea, lm).sum().backward()
    assert all(p.grad is not None for p in net.representation.parameters())
    assert all(p.grad is not None for p in net.policy_head.parameters())


def test_cnn_control_param_count_and_forward():
    net = build_cnn_bc_net()
    assert num_params(net) == 571_501
    net.eval()
    x = torch.zeros(2, net.in_channels, 19, 19)
    log_policy = net(x)[0]
    assert log_policy.shape == (2, 362)
    assert torch.isfinite(log_policy).all()


# ── bots load + legal move (1-step ckpt, no real training) ────────────────────

def _mini_gnn_ckpt(tmp_path):
    net = GnnBcNet()
    p = tmp_path / "gnn.pt"
    torch.save({"model_state_dict": net.state_dict(), "arm": "gnn"}, p)
    return str(p)


def _mini_cnn_ckpt(tmp_path):
    net = build_cnn_bc_net()
    p = tmp_path / "cnn.pt"
    torch.save({"model_state_dict": net.state_dict(), "arm": "cnn"}, p)
    return str(p)


def test_gnn_bot_plays_legal_move(tmp_path):
    from hexo_rl.eval.eval_board import make_eval_board
    from hexo_rl.env.game_state import GameState
    from hexo_rl.probes.gnn_bc.gnn_bc_bot import GnnBcBot
    b = make_eval_board("v6_live2_ls", 8); s = GameState.from_board(b)
    s = s.apply_move(b, 0, 0); s = s.apply_move(b, 1, 0)
    bot = GnnBcBot(_mini_gnn_ckpt(tmp_path), device="cpu")
    q, r = bot.get_move(s, b)
    assert (q, r) in b.legal_moves()
    assert bot.name() == "gnn-bc"


def test_cnn_bot_plays_legal_move(tmp_path):
    from hexo_rl.eval.eval_board import make_eval_board
    from hexo_rl.env.game_state import GameState
    from hexo_rl.probes.gnn_bc.cnn_bc_bot import CnnBcBot
    b = make_eval_board("v6_live2_ls", 8); s = GameState.from_board(b)
    s = s.apply_move(b, 0, 0); s = s.apply_move(b, 1, 0)
    bot = CnnBcBot(_mini_cnn_ckpt(tmp_path), device="cpu")
    q, r = bot.get_move(s, b)
    assert (q, r) in b.legal_moves()
    assert bot.name() == "cnn-bc"


# ── perf-rewrite equivalence: vectorized loss + parallel build ────────────────
# The vectorized _gnn_loss and the parallel graph build must be equivalent to the
# frozen serial path (D-L WP3 perf fix). _gnn_loss_reference is the oracle.

_EQUIV_POSITIONS = [
    ({(0, 0): 1, (1, 0): -1, (2, 0): 1}, 1, 2),
    ({(0, 0): 1, (0, 1): -1}, 1, 2),
    ({(3, 3): 1, (3, 4): -1, (4, 4): 1, (2, 2): -1}, 1, 2),
    ({(-1, 2): -1, (0, 0): 1}, 1, 2),
]


def _equiv_examples():
    """Same graphs as (dict_examples, compact_examples) — for collate equivalence."""
    dict_ex, compact_ex = [], []
    for i, (stones, cp, mr) in enumerate(_EQUIV_POSITIONS):
        g = build_axis_graph_raw(stones, cp, mr, win_length=6, radius=6,
                                 prune_empty_edges=True, threat_features=True,
                                 relative_stones=True)
        n_legal = int(sum(g["legal_mask"]))
        target_local = (i * 3) % n_legal
        w = float(1 + i)
        dict_ex.append((g, target_local, w))
        compact_ex.append(_compact_from_graph(g, target_local, w))
    return dict_ex, compact_ex


def _equiv_batch(device="cpu"):
    """A mixed batch of real axis-graphs, collated via the fast compact path."""
    _, compact_ex = _equiv_examples()
    return _collate_gnn(compact_ex, torch.device(device))


def test_compact_collate_matches_dict_reference():
    # The fast numpy collate must produce byte-identical tensors to the dict path.
    dict_ex, compact_ex = _equiv_examples()
    ref = _collate_gnn_dict(dict_ex, torch.device("cpu"))
    got = _collate_gnn(compact_ex, torch.device("cpu"))
    names = ["x", "edge_index", "edge_attr", "legal_mask",
             "target_idx", "legal_offsets", "w"]
    for name, a, b in zip(names, ref, got):
        assert a.shape == b.shape, (name, a.shape, b.shape)
        assert a.dtype == b.dtype, (name, a.dtype, b.dtype)
        torch.testing.assert_close(b, a, rtol=0, atol=0)  # exact — same values


def test_vectorized_gnn_loss_matches_reference():
    torch.manual_seed(0)
    net = GnnBcNet(); net.eval()
    batch = _equiv_batch()
    loss_ref, correct_ref, n_ref = _gnn_loss_reference(net, batch)
    loss_new, correct_new, n_new = _gnn_loss(net, batch)
    assert n_new == n_ref
    assert correct_new == correct_ref
    torch.testing.assert_close(loss_new, loss_ref, rtol=1e-4, atol=1e-5)


def test_vectorized_gnn_loss_grad_matches_reference():
    torch.manual_seed(0)
    net = GnnBcNet(); net.train()
    batch = _equiv_batch()
    loss_ref, _, _ = _gnn_loss_reference(net, batch)
    net.zero_grad(set_to_none=True); loss_ref.backward()
    g_ref = [p.grad.clone() for p in net.parameters() if p.grad is not None]
    loss_new, _, _ = _gnn_loss(net, batch)
    net.zero_grad(set_to_none=True); loss_new.backward()
    g_new = [p.grad.clone() for p in net.parameters() if p.grad is not None]
    assert len(g_ref) == len(g_new) and len(g_ref) > 0
    for a, b in zip(g_ref, g_new):
        torch.testing.assert_close(a, b, rtol=1e-3, atol=1e-5)


def test_parallel_ordered_map_preserves_order_and_values():
    # max_inflight < len exercises the streaming refill path; abs is picklable.
    data = list(range(-50, 50))
    got = list(parallel_ordered_map(abs, iter(data), workers=4, max_inflight=8))
    assert got == [abs(x) for x in data]


def test_parallel_ordered_map_input_smaller_than_inflight():
    got = list(parallel_ordered_map(abs, iter([-1, -2, -3]), workers=4, max_inflight=8))
    assert got == [1, 2, 3]
