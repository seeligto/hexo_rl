"""CnnBcBot — raw-policy argmax BotProtocol wrapper for a CNN-BC checkpoint.

The CONTROL arm at deploy. Decode is the v6_live2_ls K-cluster NO-DROP
scatter-max argmax: forward EVERY cluster window (K may exceed 1 once stones
spread beyond a single 19x19 window's centroid-relative reach — see
``Board::get_cluster_views`` "Massive Clusters" branch and ``engine/src/
encoding/registry.toml``'s ``v6_live2_ls`` entry: ``policy_pool =
"legal_set_scatter_max"``, ``is_multi_window = true``), then for each legal
move take the MAX per-window log-prob over whichever window(s) contain it and
argmax. This mirrors ``hexo_rl.eval.k_cluster_mcts_bot._aggregate_priors``
(the canonical no-drop aggregation used by the mantis-261k-raw / DeployHeadBot
eval path — see ``hexo_rl/eval/defender_dispatch.py``), applied to a single
raw-policy decode instead of MCTS priors.

R1 fix (D-M R-LADDER, reports/probes/gnn_bc/R1_adapter_sanity.md): the prior
window-0-only decode (mirroring ``V6ArgmaxBot``, itself correct for
``policy_pool="none"`` single-window encodings) silently dropped every legal
move outside cluster-0's frame to a score floor — exactly the drop bug
``defender_dispatch.py`` exists to prevent for ``legal_set_scatter_max``
encodings. Empirically this affected ~90% of cnn-bc's tournament decision
points (34% of its legal-move mass out-of-window on average, ~5.6% of
positions genuinely multi-cluster K>1). Never a fully degenerate tie (some
in-window legal candidate always existed), but a real structural handicap vs
the no-drop path mantis/strix use.

The net is a fresh-init ``HexTacToeNet(encoding='v6_live2_ls')`` (small config)
trained by BC on the same corpus/protocol/steps as the GNN arm.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from hexo_rl.bootstrap.bot_protocol import BotProtocol
from hexo_rl.encoding import lookup as _lookup_encoding
from hexo_rl.env import GameState
from hexo_rl.probes.gnn_bc.cnn_bc_net import build_cnn_bc_net


class CnnBcBot(BotProtocol):
    def __init__(self, model_path: str, device: str = "cpu", label: str = "cnn-bc",
                 filters: int = 24, res_blocks: int = 3) -> None:
        ckpt = torch.load(str(model_path), map_location=device, weights_only=True)
        sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        self._model = build_cnn_bc_net(filters=filters, res_blocks=res_blocks)
        self._model.load_state_dict(sd, strict=True)
        self._model.to(device).eval()
        self._device = torch.device(device)
        self._label = label
        # v6_live2_ls emits an 18-plane source tensor; the net takes the kept
        # planes [0,8,16,17]. Slice before forward (matches the corpus scatter).
        self._kept = list(_lookup_encoding("v6_live2_ls").kept_plane_indices)

    def reset(self) -> None:
        pass

    def name(self) -> str:
        return self._label

    @torch.no_grad()
    def get_move(self, state: GameState, rust_board: object) -> Tuple[int, int]:
        tensor, centers = state.to_tensor()   # (K, 18, S, S) source planes
        cluster_tensor = tensor[:, self._kept]   # (K, 4, S, S) slice to kept planes
        K, _, view_size, _ = cluster_tensor.shape
        view_half = (view_size - 1) // 2
        # Batched forward over ALL K cluster windows (one GPU/CPU call), not just
        # window 0 — see the class docstring (R1 no-drop fix).
        x = torch.from_numpy(np.ascontiguousarray(cluster_tensor)).float().to(self._device)
        log_policy, _value, _v_logit = self._model(x)
        log_p = log_policy.cpu().numpy()   # (K, S*S+1)

        legal_moves = rust_board.legal_moves()
        if not legal_moves:
            raise RuntimeError("CnnBcBot: no legal moves")
        scores = np.full(len(legal_moves), -1e30, dtype=np.float64)
        for i, (q, r) in enumerate(legal_moves):
            best = -1e30
            for k in range(K):
                cq, cr = centers[k]
                wq = q - cq + view_half
                wr = r - cr + view_half
                if 0 <= wq < view_size and 0 <= wr < view_size:
                    v = float(log_p[k, wq * view_size + wr])
                    if v > best:
                        best = v
            scores[i] = best
        return legal_moves[int(scores.argmax())]
