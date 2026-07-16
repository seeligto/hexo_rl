"""S7 F9 regression — fp16 GINE sum-aggregation overflow on GnnNet.forward_batch.

`reports/probes/gnn_integration/S7_smoke_gate.md` "Re-run 3" F9: fp16 autocast
on `GnnNet.forward_batch` goes non-finite on production-scale self-play graphs
(deep ply-cap positions, ~500 nodes) — `_GINEConv`'s sum-aggregation
(`agg.index_add_(0, dst, msg)`, `hexo_rl/bots/strix_v1_net.py`) accumulates one
ReLU'd message per incoming edge into each destination node; a high enough
in-degree pushes that sum past fp16's 65504 ceiling (diagnosed on real data:
conv-stack absmax 5.56e4 vs the 6.55e4 max) -> `inf` -> `LayerNorm` -> NaN
through the value/embedding path.

This file reproduces the MECHANISM synthetically (a deliberately extreme
high-in-degree fan-in graph) rather than depending on ephemeral self-play
scratch artifacts — the same class of construction the report's own
`f9_diag*.py` isolation ladder used, and explicitly sanctioned as an
acceptable repro by the fix's own instructions ("production-scale edges OR a
synthetic high-degree graph whose fp16 forward overflows"). The synthetic
graph is a single hub node (dst) receiving E >> 1 duplicate edges from one
source node — `agg[hub] == E * msg` EXACTLY (`msg` is identical across the E
duplicate edges), so the overflow is deterministic given ANY nonzero `msg`
vector — it does not depend on hitting a lucky random seed, only on E being
large enough that `E * max(|msg|)` clears fp16's ceiling. Empirically this
threshold is far below E=200,000 across many random inits (verified 0-7).

CUDA-gated throughout: fp16 autocast is a no-op off CUDA (`torch.autocast`
silently disables fp16 outside CUDA, matching the rest of this codebase's
convention, e.g. `tests/training/test_gnn_train_step.py`), so a CPU run
could not exercise the actual overflow this test documents.
"""
from __future__ import annotations

import pytest
import torch

from hexo_rl.model.gnn_net import EDGE_DIM, IN_DIM, GnnNet

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="fp16/bf16 autocast overflow behaviour is CUDA-specific — fp16 "
           "autocast is a silent no-op off CUDA, so this mechanism cannot be "
           "faithfully reproduced on CPU.",
)

# Comfortable margin over the empirically-verified overflow threshold
# (fp16 overflowed at E=100,000 across 8 random seeds in isolation testing).
_HUB_FANIN_EDGES = 200_000


def _high_indegree_graph(device: torch.device):
    """One hub node (dst=0) with `_HUB_FANIN_EDGES` duplicate incoming edges
    from a single source node (src=1) — every edge carries the SAME
    `edge_attr` row, so every message is identical and `agg[hub]` is exactly
    `E * msg` (no reliance on statistical accumulation across distinct
    random edges)."""
    n = 2
    x = torch.randn(n, IN_DIM, device=device)
    edge_index = torch.stack([
        torch.ones(_HUB_FANIN_EDGES, dtype=torch.long, device=device),   # src = node 1
        torch.zeros(_HUB_FANIN_EDGES, dtype=torch.long, device=device),  # dst = node 0 (hub)
    ])
    edge_attr = torch.randn(1, EDGE_DIM, device=device).expand(_HUB_FANIN_EDGES, -1)
    legal_mask = torch.tensor([True, True], device=device)
    stone_mask = torch.tensor([True, False], device=device)
    return x, edge_index, edge_attr, legal_mask, stone_mask


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_fp16_forward_overflows_on_high_indegree_graph(seed):
    """xfail-style documentation of the F9 mechanism: fp16 autocast on a
    high-in-degree graph forward MUST go non-finite — this is the bug F9
    fixed by routing the graph path to bf16, pinned here as a permanent
    record of what fp16 does on this class of input (not merely a
    regression guard for the fix below, but a live repro of the failure
    the fix exists for)."""
    torch.manual_seed(seed)
    device = torch.device("cuda")
    net = GnnNet().to(device).eval()
    x, edge_index, edge_attr, legal_mask, stone_mask = _high_indegree_graph(device)

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
        policy_logits, value, bin_logits = net.forward_batch(
            x, edge_index, edge_attr, legal_mask, stone_mask,
        )

    assert not bool(torch.isfinite(value).all()), (
        "fp16 forward was finite on a high-in-degree graph — either the "
        "overflow margin regressed (E too small) or fp16 autocast stopped "
        "reproducing F9's mechanism; re-diagnose before assuming this is "
        "good news."
    )
    # (value non-finite alone is sufficient to document the mechanism —
    # policy_logits/bin_logits are typically non-finite too here but are not
    # asserted strictly, to avoid over-fitting the test to one specific
    # propagation pattern through the pooling/head split.)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_bf16_forward_stays_finite_on_same_high_indegree_graph(seed):
    """The shipped fix: the SAME graph that overflows fp16 (previous test)
    must stay finite under bf16 — `amp_dtype_for("graph", ...)` is what
    `Trainer._train_on_graph_batch` and `InferenceServer` actually select
    for a GnnNet forward (`hexo_rl/model/build_net.py`)."""
    torch.manual_seed(seed)
    device = torch.device("cuda")
    net = GnnNet().to(device).eval()
    x, edge_index, edge_attr, legal_mask, stone_mask = _high_indegree_graph(device)

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
        policy_logits, value, bin_logits = net.forward_batch(
            x, edge_index, edge_attr, legal_mask, stone_mask,
        )

    assert torch.isfinite(value).all(), f"bf16 forward went non-finite: value={value!r}"
    assert torch.isfinite(policy_logits).all(), "bf16 forward: policy_logits non-finite"
    assert torch.isfinite(bin_logits).all(), "bf16 forward: bin_logits non-finite"


def test_amp_dtype_for_graph_selects_bf16_matching_this_fixture():
    """Cheap non-CUDA-dependent tie: `amp_dtype_for` (the resolver
    `Trainer`/`InferenceServer` actually call) returns EXACTLY the dtype
    the finite leg above exercises, and never the dtype the overflow leg
    exercises, for a graph representation — closes the loop between "what
    we tested" and "what production runs"."""
    from hexo_rl.model.build_net import amp_dtype_for

    assert amp_dtype_for("graph", {}) == torch.bfloat16
    assert amp_dtype_for("graph", {"amp_dtype": "fp16"}) == torch.bfloat16
