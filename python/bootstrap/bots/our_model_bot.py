"""OurModelBot — wraps our own checkpoint + MCTS as a BotProtocol.

Used for self-evaluation and as a corpus source once the model is strong
enough to contribute useful training data.
"""

from __future__ import annotations

from typing import Any, Dict

import torch

from python.bootstrap.bot_protocol import BotProtocol
from python.env import GameState
from python.model.network import HexTacToeNet
from python.selfplay.worker import SelfPlayWorker


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
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        net = HexTacToeNet(config)
        state_dict = torch.load(checkpoint_path, map_location=device)
        net.load_state_dict(state_dict)
        net.to(device)
        net.eval()

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
