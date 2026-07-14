"""Production GNN net — GINE representation + policy head + dist65 pooled value head.

WP-2 (C2) of the GNN-integration program (R4 ratified b+, `docs/designs/run4_gnn_design.md`).
Net-scale ruling: **probe-284k class ships** (hidden=128, num_layers=4, JK-cat, EDGE_DIM=5,
in_dim=11 — the exact architecture that measured the +414 [+320,+560] BT-Elo signature,
`docs/07_PHASE4_SPRINT_LOG.md` D-M R-LADDER R1/R2). Any architecture change forfeits that
evidence base (WP-C Cost 1, `run4_gnn_design.md` §2).

**Trunk** (`representation` / `policy_head`): reuses `hexo_rl.bots.strix_v1_net`'s grad-capable
`RepresentationNetwork` / `PolicyHead` verbatim — byte-identical construction to the probe's
`hexo_rl.probes.gnn_bc.gnn_bc_net.GnnBcNet`, so `representation.*` / `policy_head.*` state-dict
keys match the banked BC-prefit checkpoint
(`checkpoints/probes/gnn_bc/gnn_bc_040000.pt`, 283,970 params, state under `model_state_dict`)
by construction. `load_representation_policy_from_bc()` below is the loader + F1 landed-verify
guard (E1 precedent: `hexo_rl/eval/checkpoint_loader.py:590-603` post-`strict=False`-load
`torch.allclose` verify — mirrored here onto representation + policy tensors, per the C7
red-team demand for representation+policy coverage, not value-only).

**dist65 pooled value head** (`GnnDist65ValueHead`): NEW — the probe's `ValueHead` (scalar,
tanh) is present but unsupervised in the probe and is NOT reused. This head pools the JK-cat
`num_layers * hidden` (= 512) node embeddings with the SAME stone-masked-mean-with-fallback
the probe's `GnnBcNet.policy_logits_for_graph` / strix's `HeXONet.forward` use
(`emb[stone_mask].mean(dim=0)`, else `emb.mean(dim=0)` when a graph has no stone nodes), then
an MLP tail to 65 bin logits, decoded via `hexo_rl.training.binned_value.decode_binned_value` —
the SAME primitive the CNN's dist65 head uses (`hexo_rl/model/network_min_max_head.py:98-105`).
Bins/decode/loss are NOT re-implemented (`scalar_to_two_hot` / `decode_binned_value` /
`binned_value_loss` are pool-agnostic per the C2 scoping-doc verification,
`docs/designs/gnn_integration_scope.md` §C2). Fresh-init is expected and fine (E1 REVIVE
finding, `reports/e1/` and memory `e1-cardone-integration-scoped`: dist65 warm-starts cleanly
from an absent value head).

**Forward signature = the contract.** `forward_batch()` consumes a block-diagonal disjoint
union of B graphs (concatenated nodes/edges, `edge_index` already globally offset) — this is
the WP-B ragged contract's in-memory shape (`docs/designs/gnn_ragged_contract_v1.md` §2.1
`node_feat` / `edge_index` / `edge_attr` / `node_offsets`), consumed here as already-collated
torch tensors (the Rust producer + `graph_collate.collate_graph_batch` resolver that emit these
tensors from the wire arrays are WP-1 / WP-3's job, not this module's). No PyG dependency
anywhere (pure torch `index_select` / `index_add_`, exactly like the probe / strix's `_GINEConv`).
`forward_single()` is the single-graph deploy path (mirrors `GnnBcNet.policy_logits_for_graph` /
`HeXONet.forward`, extended with the dist65 decode).

Attribution: representation/policy modules ported (via `strix_v1_net.py`) from
SootyOwl/hexo-strix @ c381ffbeb248313a1ec177eb650d9c3c2380caa8 (MIT).
"""
from __future__ import annotations

import random
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from hexo_rl.bots.strix_v1_net import PolicyHead, RepresentationNetwork
from hexo_rl.training.binned_value import N_VALUE_BINS, decode_binned_value

EDGE_DIM = 5
IN_DIM = 11

# The two state-dict prefixes that must load byte-compatibly from the BC-prefit
# checkpoint (`checkpoints/probes/gnn_bc/gnn_bc_040000.pt`, keys under
# `model_state_dict`). `value_head.*` is deliberately excluded — the probe's value
# head is unsupervised and this module's dist65 head has a different architecture
# (bin-logit tail vs scalar-tanh tail); it is always fresh-initialized (E1 REVIVE).
BC_TRANSFER_PREFIXES: Tuple[str, ...] = ("representation.", "policy_head.")


class GnnDist65ValueHead(nn.Module):
    """Stone-masked-pooled MLP -> 65 bin logits -> decoded scalar.

    Mirrors the CNN dist65 tail (`value_fc2_bins`, `network_min_max_head.py:98-105`)
    but consumes the GINE JK-cat pooled vector (``in_dim`` = ``num_layers * hidden``,
    e.g. 512) instead of the CNN's avg+max pooled 256-vector. Pooling itself
    (stone-mask mean, all-nodes fallback) happens in ``GnnNet`` / the free
    functions below, not in this module — this head is pool-agnostic, taking a
    ``(B, in_dim)`` (or ``(in_dim,)`` for a single graph) pooled vector.
    """

    def __init__(self, in_dim: int, hidden: int = 32, n_bins: int = N_VALUE_BINS) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.relu = nn.ReLU()
        self.fc2_bins = nn.Linear(hidden, n_bins)

    def forward(self, pooled: Tensor) -> Tuple[Tensor, Tensor]:
        """pooled: (..., in_dim) -> (value (..., 1) in [-1,1], bin_logits (..., n_bins))."""
        h = self.relu(self.fc1(pooled))
        bin_logits = self.fc2_bins(h)
        value = decode_binned_value(bin_logits)
        return value, bin_logits


def _node_offsets_to_batch_vec(node_offsets: Tensor) -> Tensor:
    """(B+1,) i64 ptr array -> (N,) i64 graph-id per node (torch.repeat_interleave)."""
    counts = node_offsets[1:] - node_offsets[:-1]
    return torch.repeat_interleave(
        torch.arange(node_offsets.shape[0] - 1, device=node_offsets.device, dtype=torch.long),
        counts,
    )


def segment_mean_with_fallback(emb: Tensor, mask: Tensor, batch_vec: Tensor, num_graphs: int) -> Tensor:
    """Per-graph mean over ``mask``-selected nodes; falls back to ALL nodes for any
    graph with zero masked nodes. Batched generalization of the probe/strix pooling
    (``emb[stone_mask].mean(dim=0)`` else ``emb.mean(dim=0)`` for a single graph).

    Args:
        emb:        (N, D) node embeddings (block-diagonal batch).
        mask:       (N,) bool — the preferred subset (stone nodes).
        batch_vec:  (N,) long — graph id per node, in [0, num_graphs).
        num_graphs: B.
    Returns:
        (num_graphs, D) pooled vectors.
    """
    d = emb.shape[1]
    device = emb.device
    dtype = emb.dtype
    mask_f = mask.to(dtype)

    masked_sums = torch.zeros(num_graphs, d, device=device, dtype=dtype)
    masked_sums.index_add_(0, batch_vec, emb * mask_f.unsqueeze(-1))
    masked_counts = torch.zeros(num_graphs, device=device, dtype=dtype)
    masked_counts.index_add_(0, batch_vec, mask_f)

    all_sums = torch.zeros(num_graphs, d, device=device, dtype=dtype)
    all_sums.index_add_(0, batch_vec, emb)
    all_counts = torch.zeros(num_graphs, device=device, dtype=dtype)
    all_counts.index_add_(0, batch_vec, torch.ones_like(mask_f))

    use_fallback = masked_counts == 0
    denom = torch.where(use_fallback, all_counts.clamp(min=1.0), masked_counts.clamp(min=1.0))
    numer = torch.where(use_fallback.unsqueeze(-1), all_sums, masked_sums)
    return numer / denom.unsqueeze(-1)


class GnnNet(nn.Module):
    """Production HeXONet-equivalent GNN: GINE representation + policy head + dist65
    pooled value head. Probe-284k class (`run4_gnn_design.md` §0 net-scale ruling)."""

    def __init__(
        self,
        in_dim: int = IN_DIM,
        hidden: int = 128,
        num_layers: int = 4,
        edge_dim: int = EDGE_DIM,
        policy_hidden: int = 128,
        value_hidden: int = 32,
        n_value_bins: int = N_VALUE_BINS,
    ) -> None:
        super().__init__()
        self.representation = RepresentationNetwork(in_dim, hidden, num_layers, edge_dim)
        head_in = self.representation.output_dim
        self.policy_head = PolicyHead(head_in, policy_hidden)
        self.value_head = GnnDist65ValueHead(head_in, value_hidden, n_value_bins)

    def node_embeddings(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        """(N, L*H) node embeddings for a (possibly batched/disjoint) graph."""
        return self.representation(x, edge_index, edge_attr)

    def forward_batch(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        legal_mask: Tensor,
        stone_mask: Tensor,
        node_offsets: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Grad-capable forward over a disjoint-union batch of graphs (the WP-B
        block-diagonal contract shape, already collated into tensors).

        Args:
            x:            (N_total, in_dim) node features (all graphs concatenated).
            edge_index:   (2, E_total) int64, per-graph node offsets already applied
                          (globally-offset, per WP-B `edge_index`).
            edge_attr:    (E_total, edge_dim) edge features.
            legal_mask:   (N_total,) bool — True on legal-move (empty) nodes.
            stone_mask:   (N_total,) bool — True on stone nodes (for value pooling).
            node_offsets: (B+1,) int64 non-decreasing ptr array, `[0]=0`, `[B]=N_total`
                          (WP-B `node_offsets` / PyG `ptr` convention). ``None`` ==
                          single graph (B=1): treated as ``[0, N_total]``.
        Returns:
            policy_logits: (num_legal_total,) per-legal-node logits, in node order
                          (== per-graph legal-node order, graphs concatenated) —
                          identical semantics to `GnnBcNet.forward_batch`.
            value:        (B, 1) decoded value per graph, in [-1, 1].
            bin_logits:   (B, n_value_bins) raw dist65 bin logits per graph.
        """
        # Full payload validation belongs to the WP-B resolver (18-assertion set);
        # the mask dtype alone is guarded here because torch's uint8-as-mask path
        # is deprecated — when removed, a uint8 mask silently becomes an integer
        # index (wrong-row gather, F1 class). Cheap defense-in-depth.
        assert legal_mask.dtype == torch.bool, f"legal_mask must be bool, got {legal_mask.dtype}"
        assert stone_mask.dtype == torch.bool, f"stone_mask must be bool, got {stone_mask.dtype}"
        n_total = x.shape[0]
        device = x.device
        if node_offsets is None:
            node_offsets = torch.tensor([0, n_total], dtype=torch.long, device=device)
        num_graphs = node_offsets.shape[0] - 1

        emb = self.representation(x, edge_index, edge_attr)
        legal_emb = emb[legal_mask]
        policy_logits = self.policy_head.mlp(legal_emb).squeeze(-1)

        batch_vec = _node_offsets_to_batch_vec(node_offsets)
        pooled = segment_mean_with_fallback(emb, stone_mask, batch_vec, num_graphs)
        value, bin_logits = self.value_head(pooled)
        return policy_logits, value, bin_logits

    @torch.no_grad()
    def forward_single(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        legal_mask: Tensor,
        stone_mask: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Deploy path: ONE graph in, no batch dimension on the outputs.

        Returns (policy_logits_over_legal_nodes (num_legal,), value (scalar Tensor),
        bin_logits (n_value_bins,)) — mirrors `GnnBcNet.policy_logits_for_graph` /
        strix `HeXONet.forward`, extended with the dist65 decode.

        Deliberately does NOT delegate to `forward_batch`: this path is bit-exact
        with the probe's deploy forward (WP2 red-team: 20/20 positions, max Δ=0.0),
        which is what carries the +414 BC evidence onto this module. Routing through
        the batched segment-pooling changes accumulation order (~5e-7 drift) and
        would forfeit that exactness. Keep in sync with forward_batch semantics.
        """
        assert legal_mask.dtype == torch.bool, f"legal_mask must be bool, got {legal_mask.dtype}"
        assert stone_mask.dtype == torch.bool, f"stone_mask must be bool, got {stone_mask.dtype}"
        emb = self.representation(x, edge_index, edge_attr)
        legal_emb = emb[legal_mask]
        policy_logits = self.policy_head.mlp(legal_emb).squeeze(-1)
        if stone_mask.any():
            pooled = emb[stone_mask].mean(dim=0)
        else:
            pooled = emb.mean(dim=0)
        # pooled is (in_dim,) -- no batch dim -- so value_head returns value (1,)
        # and bin_logits (n_value_bins,) already unbatched; only value needs the
        # squeeze to a true scalar.
        value, bin_logits = self.value_head(pooled)
        return policy_logits, value.squeeze(0), bin_logits

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def load_representation_policy_from_bc(
    net: GnnNet,
    bc_state_dict: dict,
    *,
    prefixes: Sequence[str] = BC_TRANSFER_PREFIXES,
    verify_n: Optional[int] = None,
    seed: int = 0,
) -> dict:
    """Load ONLY `representation.*` / `policy_head.*` tensors from a `GnnBcNet`
    checkpoint state dict (keys under `model_state_dict`, e.g.
    `checkpoints/probes/gnn_bc/gnn_bc_040000.pt`) onto ``net``. `value_head.*` is
    left fresh-initialized (E1 REVIVE: dist65 warm-starts fine from an absent value
    head; the probe's own value head is a different, unsupervised architecture).

    STRICT on the two transfer prefixes: every ``net`` key under ``prefixes`` must
    be present in ``bc_state_dict`` and vice versa, else raises. After load, a
    landed-verify pass (E1 `checkpoint_loader.py:590-603` pattern) checks ALL
    transferred tensors by default (``verify_n=None``; 46 tensors is cheap —
    WP2 review) — pass an int to sample instead — and asserts
    ``torch.allclose`` against the source — the F1 guard: a silent key-mismatch
    drop under `strict=False` is exactly the failure class that self-played the
    wrong representation for 272k+ steps undetected (`d-forensic-f1-lineage...`).

    Returns ``{"loaded_keys": [...], "verified_tensors": int}``.
    Raises ``RuntimeError`` on key mismatch or a failed landed-verify.
    """
    own_sd = net.state_dict()
    own_keys_for_prefixes = {k for k in own_sd if k.startswith(tuple(prefixes))}
    src = {k: v for k, v in bc_state_dict.items() if k.startswith(tuple(prefixes))}

    missing = own_keys_for_prefixes - src.keys()
    unexpected = src.keys() - own_keys_for_prefixes
    if missing or unexpected:
        raise RuntimeError(
            "load_representation_policy_from_bc: state-dict key mismatch for "
            f"prefixes={prefixes} — missing={sorted(missing)} unexpected={sorted(unexpected)}"
        )

    net.load_state_dict(src, strict=False)

    reloaded_sd = net.state_dict()
    rng = random.Random(seed)
    verified = 0
    for prefix in prefixes:
        keys = sorted(k for k in own_keys_for_prefixes if k.startswith(prefix))
        sample = keys if verify_n is None else rng.sample(keys, min(verify_n, len(keys)))
        for k in sample:
            loaded = reloaded_sd[k]
            source = src[k].to(device=loaded.device, dtype=loaded.dtype)
            if not torch.allclose(loaded, source):
                raise RuntimeError(
                    f"load_representation_policy_from_bc: landed-verify FAILED for {k!r} "
                    "(strict=False load did not land this tensor)."
                )
            verified += 1

    return {"loaded_keys": sorted(src.keys()), "verified_tensors": verified}
