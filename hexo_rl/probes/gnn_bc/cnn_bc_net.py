"""CNN control-arm factory for the GNN-BC probe (D-L WP3).

The CONTROL arm: OUR ResNet trunk on the v6_live2_ls encoding, fresh-init,
policy-only BC. Without it, "gnn-bc beats mantis-261k-raw" confounds
architecture with BC-vs-self-play training. The isolating comparison is
gnn-bc vs cnn-bc, matched corpus/protocol/steps.

Just a thin factory around ``hexo_rl.model.network.HexTacToeNet`` at a small,
param-matched config (see docs/designs/gnn_bc_probe_design.md §5). No new
architecture — the CNN is our own, unchanged.
"""
from __future__ import annotations

from hexo_rl.model.network import HexTacToeNet

# Small config used by the probe. The v6 policy head (Linear(2*361, 362)) alone
# is ~261k params, so the CNN cannot go much below ~0.55M; this config lands at
# 571,501 (VERIFIED). Full-size mantis (128/12, 4.25M) is deliberately NOT used
# — the probe is about representation class, not capacity.
PROBE_FILTERS = 24
PROBE_RES_BLOCKS = 3
PROBE_ENCODING = "v6_live2_ls"


def build_cnn_bc_net(
    filters: int = PROBE_FILTERS,
    res_blocks: int = PROBE_RES_BLOCKS,
    encoding: str = PROBE_ENCODING,
) -> HexTacToeNet:
    """Fresh-init CNN control net (policy-only BC target).

    value_head_type stays 'scalar' (the probe never supervises the value head).
    """
    return HexTacToeNet(
        encoding=encoding,
        filters=filters,
        res_blocks=res_blocks,
        value_head_type="scalar",
    )


def num_params(net: HexTacToeNet) -> int:
    return sum(p.numel() for p in net.parameters())
