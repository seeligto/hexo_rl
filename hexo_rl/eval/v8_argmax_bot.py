"""V8ArgmaxBot — policy-argmax player for v8 models (no MCTS).

⚠️ **EVAL METHOD CAVEAT (§167 Gate 4)**: This bot does NOT use MCTS. It picks
the argmax legal move from the model's policy head. Against SealBot's
minimax-with-lookahead, argmax-only gets crushed (Phase B observed 0%
WR for v8 models, ALSO ~0% expected for v7full under the same setting).
The signal is **cross-arm comparable** — all variants get the same handicap —
but absolute WR is degenerate. Real MCTS-based eval is deferred to
§168 Phase D (engine's Rust MCTS hardcodes BOARD_SIZE=19 / N_ACTIONS=362
and would clip v8's 25×25 bbox edge moves through `board.to_flat`).

For each call:
  1. Encode the current Rust Board state to a v8 tensor via
     `dataset_v8.encode_position_v8` (matches Phase A wire format byte-exact).
  2. Forward through the model → `log_policy: (1, 625)`.
  3. For each legal move `(q, r)`, project into the bbox window centred on
     the v8 centroid: `(wq, wr) = (q - cq + 12, r - cr + 12)`. Cells inside
     the 25×25 window contribute `log_policy[wq*25 + wr]`; cells outside
     get score = `-inf` (rare under R=8 + 25×25, see corpus_export.log
     bbox_clip_fired telemetry).
  4. Pick the legal move with highest score.

Optional `temperature > 0`: sample legal moves from softmax-over-scores.
`temperature == 0` (default): argmax.
"""
from __future__ import annotations

from collections import deque
from typing import Deque, List, Optional, Tuple

import numpy as np
import torch

from hexo_rl.bootstrap.bot_protocol import BotProtocol
from hexo_rl.bootstrap.dataset_v8 import (
    BOARD_SIZE_V8,
    HALF_V8,
    HISTORY_LEN_V8,
    LEGAL_MOVE_RADIUS_V8,
    MAX_MOVES_V8,
    encode_position_v8,
)
from hexo_rl.env import GameState
from hexo_rl.model.network import HexTacToeNet


class V8ArgmaxBot(BotProtocol):
    """Policy-argmax player for v8 models. No MCTS.

    Args:
        model: Loaded HexTacToeNet with `encoding == "v8"`.
        device: torch.device for inference.
        temperature: 0.0 = argmax (default); >0 = softmax sampling.
        history_owner: Whose history this bot maintains internally. Each
            game requires a fresh bot instance (or call `.reset()`) so the
            history deque stays consistent across plies.
    """

    def __init__(
        self,
        model: HexTacToeNet,
        device: torch.device,
        temperature: float = 0.0,
    ) -> None:
        if getattr(model, "encoding", None) != "v8":
            raise ValueError(
                f"V8ArgmaxBot requires a v8 model; got encoding={getattr(model, 'encoding', None)!r}"
            )
        self.model = model.eval()
        self.device = device
        self.temperature = float(temperature)
        # History snapshots: pairs of (cur_stones_xy, opp_stones_xy) for the
        # last (HISTORY_LEN_V8 - 1) plies. Maintained per-game via reset().
        self._history: Deque[Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]] = deque(
            maxlen=HISTORY_LEN_V8 - 1
        )
        self._last_ply: Optional[int] = None

    def reset(self) -> None:
        self._history.clear()
        self._last_ply = None

    def name(self) -> str:
        return "v8_argmax"

    @torch.no_grad()
    def get_move(self, state: GameState, rust_board: object) -> Tuple[int, int]:
        # If the ply jumped (new game) the caller forgot to reset; clear history.
        if self._last_ply is not None and rust_board.ply <= self._last_ply:
            self.reset()

        board_stones: List[Tuple[int, int, int]] = list(rust_board.get_stones())
        cur_player: int = int(rust_board.current_player)
        ply: int = int(rust_board.ply)
        moves_rem: int = int(rust_board.moves_remaining)

        tensor, (cq, cr), _n_clipped = encode_position_v8(
            board_stones, cur_player, self._history, ply, moves_rem,
            canvas_realness=getattr(self.model, "canvas_realness", False),
        )
        x = torch.from_numpy(tensor).unsqueeze(0).float().to(self.device)
        log_policy, _value, _v_logit = self.model(x)
        log_p = log_policy.squeeze(0).cpu().numpy()  # (625,)

        legal_moves = rust_board.legal_moves()
        if not legal_moves:
            raise RuntimeError("V8ArgmaxBot: no legal moves on board")

        scores: List[float] = []
        for q, r in legal_moves:
            wq = q - cq + HALF_V8
            wr = r - cr + HALF_V8
            if 0 <= wq < BOARD_SIZE_V8 and 0 <= wr < BOARD_SIZE_V8:
                scores.append(float(log_p[wq * BOARD_SIZE_V8 + wr]))
            else:
                scores.append(-1e30)
        scores_np = np.asarray(scores, dtype=np.float64)

        if self.temperature == 0.0:
            idx = int(scores_np.argmax())
        else:
            t = max(self.temperature, 1e-6)
            shifted = scores_np / t
            shifted = shifted - shifted.max()
            probs = np.exp(shifted)
            probs /= probs.sum()
            idx = int(np.random.choice(len(legal_moves), p=probs))

        # Snapshot the pre-move stones for the next history entry. The encoder
        # treats history[-1] as "ply T-1 stones" so we append the snapshot
        # taken BEFORE applying this move.
        cur_stones_xy = [(q, r) for (q, r, p) in board_stones if p == cur_player]
        opp_stones_xy = [(q, r) for (q, r, p) in board_stones if p != cur_player]
        self._history.append((cur_stones_xy, opp_stones_xy))
        self._last_ply = ply

        return legal_moves[idx]


def load_v8_model_from_checkpoint(
    ckpt_path: str,
    device: torch.device,
) -> HexTacToeNet:
    """Reconstruct a v8 HexTacToeNet from a Phase B inference checkpoint.

    The pretrain pipeline writes only a state_dict to the inference file
    (no architecture metadata), so this loader infers shape/variant params
    from the state_dict keys. For full-checkpoint format (with config), use
    `_load_anchor_model` from eval_pipeline once it lands v8 awareness.
    """
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    # §169 A4 — under canvas_realness the trunk-entry conv is wrapped in
    # PartialConv2d, so the weight key shifts to
    # `trunk.input_conv.conv.weight`. Detection mirrors the canonical
    # loader in `eval.checkpoint_loader._build_v8_model`.
    canvas_realness = "trunk.input_conv.conv.weight" in state
    inp_w_key = (
        "trunk.input_conv.conv.weight" if canvas_realness
        else "trunk.input_conv.weight"
    )
    inp_w = state[inp_w_key]
    filters = int(inp_w.shape[0])
    in_channels = int(inp_w.shape[1])
    if in_channels != 11:
        raise ValueError(f"Expected v8 in_channels=11; got {in_channels}")

    # Count trunk blocks by scanning state-dict keys.
    block_indices = sorted({
        int(k.split(".")[2]) for k in state.keys()
        if k.startswith("trunk.tower.") and len(k.split(".")) >= 4
    })
    res_blocks = max(block_indices) + 1 if block_indices else 0

    # Identify gpool blocks by presence of `conv1.conv1g` weight.
    gpool_indices = sorted({
        i for i in block_indices
        if f"trunk.tower.{i}.conv1.conv1g.weight" in state
    })

    # Head G-branch presence (used by all variants except B0 control).
    head_use_gpool = "policy_head.conv1g.weight" in state

    model = HexTacToeNet(
        board_size=BOARD_SIZE_V8,
        in_channels=in_channels,
        filters=filters,
        res_blocks=res_blocks,
        encoding="v8",
        gpool_indices=gpool_indices if gpool_indices else None,
        head_use_gpool=head_use_gpool,
        canvas_realness=canvas_realness,
    )
    model.load_state_dict(state)
    model.eval().to(device)
    return model
