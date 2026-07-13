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
