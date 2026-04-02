"""RandomBot — selects a uniformly random legal move.

Useful as a baseline and for smoke-testing the BotProtocol pipeline.
"""

from __future__ import annotations

import random

from hexo_rl.bootstrap.bot_protocol import BotProtocol
from hexo_rl.env import GameState


class RandomBot(BotProtocol):
    """Places stones uniformly at random among legal moves."""

    def get_move(self, state: GameState, rust_board: object) -> tuple[int, int]:
        legal = rust_board.legal_moves()
        if not legal:
            raise RuntimeError("RandomBot.get_move called with no legal moves")
        return random.choice(legal)

    def name(self) -> str:
        return "random_bot"
