"""CnnBcBot — raw-policy argmax BotProtocol wrapper for a CNN-BC checkpoint.

The CONTROL arm at deploy. Decode is the v6_live2_ls K-cluster window-0 argmax
(same projection ``V6ArgmaxBot`` uses for v6): forward the cluster-0 window, then
score each legal move at its window-local cell and argmax. Raw-policy, temp 0,
no search.

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
        cluster_tensor = tensor[0][self._kept]   # slice to kept planes [0,8,16,17]
        cq, cr = centers[0]
        _, view_size, _ = cluster_tensor.shape
        view_half = (view_size - 1) // 2
        x = torch.from_numpy(np.ascontiguousarray(cluster_tensor)).unsqueeze(0).float().to(self._device)
        log_policy, _value, _v_logit = self._model(x)
        log_p = log_policy.squeeze(0).cpu().numpy()   # (S*S+1,)

        legal_moves = rust_board.legal_moves()
        if not legal_moves:
            raise RuntimeError("CnnBcBot: no legal moves")
        scores = np.full(len(legal_moves), -1e30, dtype=np.float64)
        for i, (q, r) in enumerate(legal_moves):
            wq = q - cq + view_half
            wr = r - cr + view_half
            if 0 <= wq < view_size and 0 <= wr < view_size:
                scores[i] = float(log_p[wq * view_size + wr])
        return legal_moves[int(scores.argmax())]
