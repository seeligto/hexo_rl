"""GnnBcBot — raw-policy argmax BotProtocol wrapper for a GNN-BC checkpoint.

Deploy path is byte-identical to the shipped ``strix_v1_bot.StrixV1Bot`` (build
the strix axis-graph from the board's stones, forward, argmax the policy head
over legal nodes, filter through OUR board's legal set). Only the net differs —
a ``GnnBcNet`` trained by BC on our corpus, not strix's self-play checkpoint.

Raw-policy, temperature 0, CPU-friendly. No search. Used by the 640-game WP3
argmax eval.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import torch

from hexo_rl.bootstrap.bot_protocol import BotProtocol
from hexo_rl.env import GameState
from hexo_rl.bots.strix_v1_graph import build_axis_graph_raw
from hexo_rl.probes.gnn_bc.gnn_bc_net import GnnBcNet

_WIN_LENGTH = 6
_RADIUS = 6


class GnnBcBot(BotProtocol):
    def __init__(self, model_path: str, device: str = "cpu", label: str = "gnn-bc") -> None:
        ckpt = torch.load(str(model_path), map_location=device, weights_only=True)
        sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        self._model = GnnBcNet()
        self._model.load_state_dict(sd, strict=True)
        self._model.to(device).eval()
        self._device = device
        self._label = label

    def get_move(self, state: GameState, rust_board: object) -> tuple[int, int]:
        stone_map = {(q, r): p for (q, r, p) in rust_board.get_stones()}
        g = build_axis_graph_raw(
            stone_map, state.current_player, state.moves_remaining,
            win_length=_WIN_LENGTH, radius=_RADIUS,
            prune_empty_edges=True, threat_features=True, relative_stones=True,
        )
        if not g["legal_coords"]:
            legal = rust_board.legal_moves()
            if not legal:
                raise RuntimeError("no legal moves")
            return (0, 0) if (0, 0) in set(legal) else min(legal)

        n, fdim = g["num_nodes"], g["fdim"]
        x = torch.tensor(g["features"], dtype=torch.float32, device=self._device).reshape(n, fdim)
        E = len(g["edge_src"])
        if E:
            edge_index = torch.tensor([g["edge_src"], g["edge_dst"]], dtype=torch.int64, device=self._device)
            edge_attr = torch.tensor(g["edge_attr"], dtype=torch.float32, device=self._device).reshape(E, 5)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.int64, device=self._device)
            edge_attr = torch.zeros((0, 5), dtype=torch.float32, device=self._device)
        legal_mask = torch.tensor(g["legal_mask"], dtype=torch.bool, device=self._device)
        stone_mask = torch.tensor(g["stone_mask"], dtype=torch.bool, device=self._device)

        policy_logits, _value = self._model.policy_logits_for_graph(
            x, edge_index, edge_attr, legal_mask, stone_mask)

        legal_set = set(rust_board.legal_moves())
        strix_legal = g["legal_coords"]
        order = policy_logits.argsort(descending=True).tolist()
        for idx in order:
            cand = tuple(strix_legal[idx])
            if cand in legal_set:
                return cand
        legal = rust_board.legal_moves()
        return random.choice(legal)

    def reset(self) -> None:
        pass

    def name(self) -> str:
        return self._label
