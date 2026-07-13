"""Grad-capable GINE axis-graph net for the GNN-BC probe (D-L WP3).

REUSES strix's ``RepresentationNetwork`` / ``_GINEConv`` / ``PolicyHead`` /
``ValueHead`` modules from ``hexo_rl.bots.strix_v1_net`` (those classes are
grad-capable — only ``HeXONet.forward`` carries ``@torch.inference_mode()``,
which blocks backprop and so cannot be used for BC). This module wraps the SAME
modules with a grad-capable forward for supervised training + a raw-policy
argmax path for deploy.

Architecture is byte-identical to the shipped strix adapter's HeXONet
(hidden=128, num_layers=4, JK-cat 512-wide heads, per-node policy, stone-pooled
value) so a GNN-BC checkpoint can, if desired, be reloaded by the same code
path. Value head is present but the probe does NOT supervise it (policy probe).

Attribution: SootyOwl/hexo-strix @ c381ffbeb248313a1ec177eb650d9c3c2380caa8 (MIT).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from hexo_rl.bots.strix_v1_net import (
    RepresentationNetwork,
    PolicyHead,
    ValueHead,
)

EDGE_DIM = 5


class GnnBcNet(nn.Module):
    """GINE representation + per-node policy head (+ optional value head).

    forward_batch() consumes a block-diagonal batch of graphs (disjoint union;
    edge_index already offset into the concatenated node space) and returns
    per-legal-node policy logits, in the batch's legal-node order. This is the
    training path (grad-capable).

    policy_logits_for_graph() runs a single graph for deploy (raw-policy argmax).
    """

    def __init__(self, in_dim: int = 11, hidden: int = 128, num_layers: int = 4,
                 policy_hidden: int = 128, value_hidden: int = 32) -> None:
        super().__init__()
        self.representation = RepresentationNetwork(in_dim, hidden, num_layers, EDGE_DIM)
        head_in = self.representation.output_dim
        self.policy_head = PolicyHead(head_in, policy_hidden)
        self.value_head = ValueHead(head_in, value_hidden)  # present, unsupervised in the probe

    def node_embeddings(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        """(N, L*H) node embeddings for a (possibly batched/disjoint) graph."""
        return self.representation(x, edge_index, edge_attr)

    def forward_batch(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        legal_mask: Tensor,
    ) -> Tensor:
        """Grad-capable BC forward over a disjoint-union batch of graphs.

        Args:
            x:          (N_total, in_dim) node features (all graphs concatenated).
            edge_index: (2, E_total) with per-graph node offsets already applied.
            edge_attr:  (E_total, 5) edge features.
            legal_mask: (N_total,) bool — True on legal-move (empty) nodes.
        Returns:
            (num_legal_total,) policy logits over the concatenated legal nodes,
            in node order (== per-graph legal_coords order, graphs concatenated).
        """
        emb = self.representation(x, edge_index, edge_attr)
        legal_emb = emb[legal_mask]
        return self.policy_head.mlp(legal_emb).squeeze(-1)

    @torch.no_grad()
    def policy_logits_for_graph(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        legal_mask: Tensor,
        stone_mask: Tensor,
    ):
        """Deploy path: (policy_logits_over_legal_nodes, value_scalar) for ONE graph."""
        emb = self.representation(x, edge_index, edge_attr)
        legal_emb = emb[legal_mask]
        policy_logits = self.policy_head.mlp(legal_emb).squeeze(-1)
        if stone_mask.any():
            pooled = emb[stone_mask].mean(dim=0)
        else:
            pooled = emb.mean(dim=0)
        value = self.value_head.mlp(pooled).squeeze(-1)
        return policy_logits, value

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
