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

from typing import Optional, Tuple

import numpy as np
import torch

from hexo_rl.bootstrap.bot_protocol import BotProtocol
from hexo_rl.env.game_state import GameState
from hexo_rl.model.network import HexTacToeNet
from hexo_rl.utils.constants import BOARD_SIZE, KEPT_PLANE_INDICES


_HALF: int = (BOARD_SIZE - 1) // 2  # 9


class V6ArgmaxBot(BotProtocol):
    """Policy-argmax player for v6 (HEXB 8-plane × 19×19) models. No MCTS."""

    def __init__(
        self,
        model: HexTacToeNet,
        device: torch.device,
        temperature: float = 0.0,
    ) -> None:
        if getattr(model, "encoding", "v6") != "v6":
            raise ValueError(
                f"V6ArgmaxBot requires a v6 model; got encoding={getattr(model, 'encoding', None)!r}"
            )
        self.model = model.eval()
        self.device = device
        self.temperature = float(temperature)

    def reset(self) -> None:
        # v6 model's history is reconstructed from board state each call;
        # nothing to reset.
        pass

    def name(self) -> str:
        return "v6_argmax"

    @torch.no_grad()
    def get_move(self, state: GameState, rust_board: object) -> Tuple[int, int]:
        tensor, centers = state.to_tensor()
        # Cluster 0 is the canonical view (matches pretrain.py:548-599 v6
        # validation path comment about K=0 being the aug fixture).
        cluster_tensor = tensor[0]
        cq, cr = centers[0]
        # Slice 18 → 8 planes (KEPT_PLANE_INDICES, HEXB v6 wire format) only
        # if the model expects 8 in_channels.
        if self.model.in_channels == 8:
            inp = cluster_tensor[KEPT_PLANE_INDICES]
        else:
            inp = cluster_tensor
        x = torch.from_numpy(inp).unsqueeze(0).float().to(self.device)
        log_policy, _value, _v_logit = self.model(x)
        log_p = log_policy.squeeze(0).cpu().numpy()  # (362,) — last is pass

        legal_moves = rust_board.legal_moves()
        if not legal_moves:
            raise RuntimeError("V6ArgmaxBot: no legal moves on board")

        scores = np.full(len(legal_moves), -1e30, dtype=np.float64)
        for i, (q, r) in enumerate(legal_moves):
            wq = q - cq + _HALF
            wr = r - cr + _HALF
            if 0 <= wq < BOARD_SIZE and 0 <= wr < BOARD_SIZE:
                scores[i] = float(log_p[wq * BOARD_SIZE + wr])

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


def load_v6_model_from_checkpoint(
    ckpt_path: str,
    device: torch.device,
) -> HexTacToeNet:
    """Reconstruct a v6 HexTacToeNet from a bootstrap inference checkpoint.

    v7full / v6 / v7 checkpoints store only state_dict (no architecture
    metadata); shape inferred from `trunk.input_conv.weight`.
    """
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    inp_w = state["trunk.input_conv.weight"]
    filters = int(inp_w.shape[0])
    in_channels = int(inp_w.shape[1])
    block_indices = sorted({
        int(k.split(".")[2]) for k in state.keys()
        if k.startswith("trunk.tower.") and len(k.split(".")) >= 4
    })
    res_blocks = max(block_indices) + 1 if block_indices else 12
    model = HexTacToeNet(
        board_size=BOARD_SIZE,
        in_channels=in_channels,
        filters=filters,
        res_blocks=res_blocks,
        encoding="v6",
    )
    model.load_state_dict(state, strict=False)
    model.eval().to(device)
    return model
