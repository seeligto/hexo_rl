"""V6ArgmaxBot — policy-argmax player for v6 (HEXB) models (no MCTS).

Sister bot to `V8ArgmaxBot`. Used in §167 Gate 4 to capture an
**apples-to-apples cross-encoding baseline**: running v7full under the same
no-MCTS argmax-only eval that v8 variants get, so we can see whether v8's
0% SealBot WR is the v8 architecture or just an artefact of the eval
method (spoiler: argmax-only against SealBot's minimax is degenerate
regardless of encoding).

For each call:
  1. Get v6 cluster window 0 from the Rust Board.
  2. Slice the 18-plane HEXB tensor down to KEPT_PLANE_INDICES (8 planes).
  3. Forward through the v6 model → `log_policy: (1, 362)` (361 cells + pass).
  4. For each legal move `(q, r)`, project into v6's 19×19 cluster-0 window:
     `(wq, wr) = (q - cq + 9, r - cr + 9)`. Cells inside contribute
     `log_policy[wq*19 + wr]`.
  5. Pick the legal move with highest score.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch

from hexo_rl.bootstrap.bot_protocol import BotProtocol
from hexo_rl.encoding import lookup as _lookup_encoding
from hexo_rl.env.game_state import GameState
from hexo_rl.model.network import HexTacToeNet
from hexo_rl.utils.global_crop import compute_global_crop_from_board

_V6 = _lookup_encoding("v6")
BOARD_SIZE: int = _V6.board_size
BUFFER_CHANNELS: int = _V6.n_planes
KEPT_PLANE_INDICES: list[int] = list(_V6.kept_plane_indices)


_HALF: int = (BOARD_SIZE - 1) // 2  # 9


class V6ArgmaxBot(BotProtocol):
    """Policy-argmax player for v6 / v6w25 (HEXB 8-plane K-cluster) models. No MCTS.

    Both v6 (19×19 cluster window) and v6w25 (25×25 cluster window) share the
    same wire format: 8 KEPT planes + pass slot. The bot's tensor path reads
    spatial dims from the cluster tensor itself (line `view_h, view_w = ...`)
    so v6w25's wider window works without branching. §172 A4.4 widened the
    encoding guard to accept both labels.
    """

    def __init__(
        self,
        model: HexTacToeNet,
        device: torch.device,
        temperature: float = 0.0,
    ) -> None:
        _encoding = getattr(model, "encoding", "v6")
        if _encoding not in ("v6", "v6w25"):
            raise ValueError(
                f"V6ArgmaxBot requires a v6/v6w25 model; got encoding={_encoding!r}. "
                f"v6w25 shares the v6 wire format (8 planes, K-cluster); the bot's "
                f"tensor path reads spatial dims from the tensor itself."
            )
        self.model = model.eval()
        self.device = device
        self.temperature = float(temperature)
        self._encoding_label = _encoding

    def reset(self) -> None:
        # v6 model's history is reconstructed from board state each call;
        # nothing to reset.
        pass

    def name(self) -> str:
        # §172 A4.4 — distinguish v6 vs v6w25 in eval reports.
        return f"{self._encoding_label}_argmax"

    @torch.no_grad()
    def get_move(self, state: GameState, rust_board: object) -> Tuple[int, int]:
        tensor, centers = state.to_tensor()
        # Cluster 0 is the canonical view (matches pretrain.py:548-599 v6
        # validation path comment about K=0 being the aug fixture).
        cluster_tensor = tensor[0]
        cq, cr = centers[0]
        # Spatial dims read from the tensor itself — supports v6 (19×19) and
        # §168 v6w25 (25×25) without branching.
        _, view_h, view_w = cluster_tensor.shape
        assert view_h == view_w, "V6ArgmaxBot expects square cluster window"
        view_size = view_h
        view_half = (view_size - 1) // 2
        # Slice 18 → 8 planes (KEPT_PLANE_INDICES, HEXB v6 wire format) only
        # if the model expects 8 in_channels.
        if self.model.in_channels == BUFFER_CHANNELS:
            inp = cluster_tensor[KEPT_PLANE_INDICES]
        else:
            inp = cluster_tensor
        x = torch.from_numpy(inp).unsqueeze(0).float().to(self.device)
        # §169 A3 — pma_global needs a (1, 3, 32, 32) global summary crop
        # built from the live board's stones in the current-player frame.
        # §170 P3 — gpool_bias_active=True ALSO needs the global crop
        # threaded; the model raises ValueError without it. Other pool
        # types ignore the kwarg; we omit it to keep the v6 /
        # v6w25-with-pma path unchanged.
        fwd_kwargs: dict = {}
        needs_global_crop = (
            getattr(self.model, "pool_type", "min_max") == "pma_global"
            or getattr(self.model, "gpool_bias_active", False)
        )
        if needs_global_crop:
            gc_np = compute_global_crop_from_board(rust_board)
            fwd_kwargs["global_crop"] = (
                torch.from_numpy(gc_np).unsqueeze(0).float().to(self.device)
            )
        log_policy, _value, _v_logit = self.model(x, **fwd_kwargs)
        log_p = log_policy.squeeze(0).cpu().numpy()  # (S*S+1,) — last is pass

        legal_moves = rust_board.legal_moves()
        if not legal_moves:
            raise RuntimeError("V6ArgmaxBot: no legal moves on board")

        scores = np.full(len(legal_moves), -1e30, dtype=np.float64)
        for i, (q, r) in enumerate(legal_moves):
            wq = q - cq + view_half
            wr = r - cr + view_half
            if 0 <= wq < view_size and 0 <= wr < view_size:
                scores[i] = float(log_p[wq * view_size + wr])

        if self.temperature == 0.0:
            idx = int(scores.argmax())
        else:
            t = max(self.temperature, 1e-6)
            shifted = scores / t
            shifted = shifted - shifted.max()
            probs = np.exp(shifted)
            probs /= probs.sum()
            idx = int(np.random.choice(len(legal_moves), p=probs))
        return legal_moves[idx]
