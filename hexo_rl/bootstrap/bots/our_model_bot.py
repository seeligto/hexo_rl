"""OurModelBot — wraps our own checkpoint + MCTS as a BotProtocol.

Used for self-evaluation and as a corpus source once the model is strong
enough to contribute useful training data.
"""

from __future__ import annotations

from typing import Any, Dict

import torch

from hexo_rl.bootstrap.bot_protocol import BotProtocol
from hexo_rl.encoding import lookup as _lookup_encoding
from hexo_rl.env import GameState
from hexo_rl.selfplay.worker import SelfPlayWorker
from hexo_rl.training.checkpoints import load_inference_model

_V6 = _lookup_encoding("v6")
BOARD_SIZE: int = _V6.board_size
BUFFER_CHANNELS: int = _V6.n_planes


class OurModelBot(BotProtocol):
    """Wraps a trained HexTacToeNet + MCTS as a BotProtocol.

    Args:
        checkpoint_path: Path to a saved model checkpoint (.pt file).
        config:          MCTS/search config dict (same schema as selfplay config).
        device:          Torch device to run inference on.
        temperature:     Sampling temperature (0 = greedy/argmax).
    """

    def __init__(
        self,
        checkpoint_path: str,
        config: Dict[str, Any],
        device: torch.device | None = None,
        temperature: float = 0.0,
    ) -> None:
        if device is None:
            from hexo_rl.utils.device import best_device
            device = best_device()

        # §176 P47: public load_inference_model handles state-dict
        # extraction, normalization, encoding detection, hparam inference,
        # and strict load. The previous in-line copy is retired.
        net, _spec, _label = load_inference_model(
            checkpoint_path, config, device=device,
        )

        self._worker = SelfPlayWorker(model=net, config=config, device=device)
        self._temperature = temperature
        self._checkpoint_path = checkpoint_path

    def get_move(self, state: GameState, rust_board: object) -> tuple[int, int]:
        policy = self._worker._run_mcts(
            rust_board,
            use_dirichlet=False,
            temperature=self._temperature if self._temperature > 0 else None,
        )
        legal = rust_board.legal_moves()
        if not legal:
            raise RuntimeError("OurModelBot.get_move called with no legal moves")
        q, r = self._worker._sample_action(policy, legal, rust_board)
        return q, r

    def name(self) -> str:
        import os
        base = os.path.basename(self._checkpoint_path)
        return f"our_model({base})"
