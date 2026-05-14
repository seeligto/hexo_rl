"""KrakenBotRandomBot — wraps KrakenBot's pure-Python RandomBot.

Vendor `bot.py` defines `RandomBot` (distance-2 candidate sampling near
existing stones, falls back to (0, 0) on empty board). Returns a single
(q, r) tuple — NOT a compound list — so this wrapper does not need a
pair-move cache strictly speaking, but the BotProtocol single-stone
contract is preserved by mirroring `krakenbot_bot.py`'s structure for
consistency.

Build/runtime: zero compilation. Only python deps; pure-python upstream.

_MockGame is a private duck-type in `krakenbot_bot.py` (underscore prefix);
copy-pasted here (≤25 LOC) rather than imported — keeps wrapper standalone
and avoids cross-module private-API coupling. If `krakenbot_bot._MockGame`
ever changes shape, sync this copy.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import structlog

from hexo_rl.bootstrap.bot_protocol import BotProtocol
from hexo_rl.env import GameState

log = structlog.get_logger()

_KRAKENBOT_ROOT = Path(__file__).parents[2] / "vendor" / "bots" / "krakenbot"
assert _KRAKENBOT_ROOT.exists(), f"KrakenBot not found at {_KRAKENBOT_ROOT}"
if str(_KRAKENBOT_ROOT) not in sys.path:
    sys.path.insert(0, str(_KRAKENBOT_ROOT))

from bot import RandomBot as _KRandomBot  # type: ignore[import]
from game import Player as _KPlayer       # type: ignore[import]

from hexo_rl.bots.krakenbot_bot import _smart_legal_fallback


class _MockGame:
    """Duck-typed game object matching KrakenBot's HexGame interface.

    Copy of `hexo_rl.bots.krakenbot_bot._MockGame` (underscore-private upstream).
    """

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


class KrakenBotRandomBot(BotProtocol):
    """BotProtocol wrapper for KrakenBot's neighbour-biased RandomBot.

    Upstream RandomBot returns a single (q, r); we treat the result as a
    length-1 or length-2 sequence for symmetry with MinimaxBot, but the
    common case is length 1.
    """

    def __init__(self) -> None:
        self._bot = _KRandomBot()
        self._pending_move: tuple[int, int] | None = None

    def get_move(self, state: GameState, rust_board: object) -> tuple[int, int]:
        # Return cached second stone without re-sampling; revalidate first
        # (mirrors krakenbot_bot.py defensive fix).
        if self._pending_move is not None:
            move = self._pending_move
            self._pending_move = None
            q_p, r_p = move
            if rust_board.get(q_p, r_p) == 0:
                return move
            return _smart_legal_fallback(rust_board,
                                         "random_pending_now_illegal", state.ply)

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

        # Upstream RandomBot returns a single (q, r); normalise to list form.
        if isinstance(result, tuple) and len(result) == 2 and all(
            isinstance(x, int) for x in result
        ):
            moves = [result]
        elif isinstance(result, list):
            moves = result
        else:
            return _smart_legal_fallback(rust_board,
                                         "random_unexpected_shape", state.ply)

        if not moves:
            return _smart_legal_fallback(rust_board,
                                         "random_no_moves_found", state.ply)

        q1, r1 = moves[0]
        if rust_board.get(q1, r1) != 0:
            return _smart_legal_fallback(rust_board,
                                         "random_result0_illegal", state.ply)

        # Cache second stone for the next call if this is a compound turn.
        if len(moves) >= 2 and state.moves_remaining > 1:
            q2, r2 = moves[1]
            if rust_board.get(q2, r2) == 0:
                self._pending_move = (q2, r2)
            else:
                log.warning("krakenbot_random_pending_illegal", q=q2, r=r2)

        return (q1, r1)

    def reset(self) -> None:
        self._pending_move = None

    def name(self) -> str:
        return "KrakenBotRandom"
