"""KrakenBotBot — wraps KrakenBot's pure-Python MinimaxBot (no build step required).

MinimaxBot uses iterative-deepening alpha-beta with learned pattern values loaded
from data/pattern_values.json.  No C++ compilation is needed; the only runtime
dependency is that the `krakenbot` submodule is present at vendor/bots/krakenbot.

Pair-move caching:
  MinimaxBot.get_move() returns both stones of a compound turn as a list of 2
  (q, r) tuples.  BotProtocol is single-move-per-call, so we cache the second
  stone and return it on the next call.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Optional

import structlog

from hexo_rl.bootstrap.bot_protocol import BotProtocol
from hexo_rl.env import GameState

log = structlog.get_logger()

_KRAKENBOT_ROOT = Path(__file__).parents[3] / "vendor" / "bots" / "krakenbot"
assert _KRAKENBOT_ROOT.exists(), f"KrakenBot not found at {_KRAKENBOT_ROOT}"
if str(_KRAKENBOT_ROOT) not in sys.path:
    sys.path.insert(0, str(_KRAKENBOT_ROOT))

from minimax_bot import MinimaxBot as _MinimaxBot  # type: ignore[import]
from game import Player as _KPlayer               # type: ignore[import]


class _MockGame:
    """Duck-typed game object matching KrakenBot's HexGame interface."""

    __slots__ = (
        "board", "current_player", "moves_left_in_turn",
        "move_count", "winner", "game_over",
    )

    def __init__(
        self,
        board: dict,
        current_player: _KPlayer,
        moves_left_in_turn: int,
        move_count: int,
    ) -> None:
        self.board = board
        self.current_player = current_player
        self.moves_left_in_turn = moves_left_in_turn
        self.move_count = move_count
        self.winner = _KPlayer.NONE
        self.game_over = False


class KrakenBotBot(BotProtocol):
    """BotProtocol wrapper for KrakenBot MinimaxBot (pure Python, learned patterns).

    Args:
        time_limit: Search budget in seconds per call to get_move (applied ×2
                    internally by MinimaxBot to allow iterative deepening to
                    complete the last full-depth search).
        pattern_path: Override path to pattern_values.json.  None uses the
                      default inside the submodule (data/pattern_values.json).
    """

    def __init__(
        self,
        time_limit: float = 0.05,
        pattern_path: Optional[str] = None,
    ) -> None:
        self._time_limit = time_limit
        self._bot = _MinimaxBot(time_limit=time_limit, pattern_path=pattern_path)
        self._pending_move: Optional[tuple[int, int]] = None

    def get_move(self, state: GameState, rust_board: object) -> tuple[int, int]:
        # Return cached second stone without re-searching.
        if self._pending_move is not None:
            move = self._pending_move
            self._pending_move = None
            return move

        # Reset pending cache at the start of a new full turn.
        if state.moves_remaining > 1:
            self._pending_move = None

        # Build board dict using KrakenBot's Player enum.
        board_dict: dict = {}
        for q, r, p in rust_board.get_stones():
            board_dict[(q, r)] = _KPlayer.A if p == 1 else _KPlayer.B

        kp = _KPlayer.A if state.current_player == 1 else _KPlayer.B
        game = _MockGame(
            board=board_dict,
            current_player=kp,
            moves_left_in_turn=state.moves_remaining,
            move_count=len(board_dict),
        )

        result = self._bot.get_move(game)

        if not result:
            log.warning("krakenbot_no_moves_found", ply=state.ply)
            legal = rust_board.legal_moves()
            if not legal:
                raise RuntimeError("No legal moves available on board")
            return random.choice(legal)

        # Cache second stone for the next call if this is a compound turn.
        if len(result) >= 2 and state.moves_remaining > 1:
            q2, r2 = result[1]
            if rust_board.get(q2, r2) == 0:
                self._pending_move = (q2, r2)
            else:
                log.warning("krakenbot_pending_move_illegal", q=q2, r=r2)

        q1, r1 = result[0]
        return (q1, r1)

    def name(self) -> str:
        return f"KrakenBot(t={self._time_limit})"
