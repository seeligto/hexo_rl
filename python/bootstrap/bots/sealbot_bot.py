"""SealBotBot — wraps the Ramora0/SealBot pybind11 minimax engine."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import structlog

from python.bootstrap.bot_protocol import BotProtocol
from python.env import GameState

log = structlog.get_logger()

# vendor/bots/sealbot/       — game.py (Player enum, imported by the C++ binding)
# vendor/bots/sealbot/best/  — minimax_cpp extension (.so)
_SEALBOT_ROOT = str(Path(__file__).parents[3] / "vendor" / "bots" / "sealbot")
_SEALBOT_BEST = str(Path(__file__).parents[3] / "vendor" / "bots" / "sealbot" / "best")

for _p in (_SEALBOT_ROOT, _SEALBOT_BEST):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from minimax_cpp import MinimaxBot as _MinimaxBot  # type: ignore[import]
from game import Player as SealPlayer              # type: ignore[import]


class SealBotBot(BotProtocol):
    def __init__(self, time_limit: float = 0.05) -> None:
        self._time_limit = time_limit
        self._bot = _MinimaxBot(time_limit=time_limit)
        self._pending_move: Optional[tuple[int, int]] = None
        self._colony_risk_count: int = 0

    def get_move(self, state: GameState, rust_board: object) -> tuple[int, int]:
        # Clear the cached second move at the start of each new turn.
        if state.moves_remaining > 1:
            self._pending_move = None

        # Return cached second move without re-searching.
        if self._pending_move is not None:
            move = self._pending_move
            self._pending_move = None
            return move

        # Colony bug guard: SealBot uses a flat 140×140 board — multiple distant
        # clusters may alias onto overlapping regions of that array, corrupting
        # pattern evaluations.  Log a warning once per occurrence; never skip.
        cluster_count = len(state.centers)
        if cluster_count > 4:
            self._colony_risk_count += 1
            log.warning(
                "sealbot_colony_bug_risk",
                clusters=cluster_count,
                ply=state.ply,
            )

        # Build a duck-typed game object matching what extract_game_state() reads:
        #   game.board              — dict {(q, r): Player}
        #   game.current_player     — Player enum instance
        #   game.moves_left_in_turn — int
        #   game.move_count         — int
        # The C++ binding imports game.Player and uses identity (is) comparison,
        # so we must use the actual SealPlayer enum values here.
        board_dict: dict = {}
        for q, r, p in rust_board.get_stones():
            board_dict[(q, r)] = SealPlayer.A if p == 1 else SealPlayer.B

        class _MockGame:
            board: dict
            current_player: SealPlayer
            moves_left_in_turn: int
            move_count: int

            def __init__(self, bd: dict, cp: int, ml: int, mc: int) -> None:
                self.board = bd
                self.current_player = SealPlayer.A if cp == 1 else SealPlayer.B
                self.moves_left_in_turn = ml
                self.move_count = mc

        game = _MockGame(board_dict, state.current_player, state.moves_remaining, len(board_dict))
        result = self._bot.get_move(game)

        if not result:
            log.warning("sealbot_bot_no_moves_found", ply=state.ply)
            legal_moves = rust_board.legal_moves()
            if not legal_moves:
                raise RuntimeError("No legal moves available on board")
            import random
            return random.choice(legal_moves)

        # result is a list of (q, r) tuples; length 1 on single-move turns, 2 otherwise.
        # Cache the second move for retrieval on the next call.
        if len(result) >= 2 and state.moves_remaining > 1:
            q2, r2 = result[1]
            if rust_board.get(q2, r2) == 0:
                self._pending_move = (q2, r2)
            else:
                log.warning("sealbot_bot_pending_move_illegal", q=q2, r=r2)

        q1, r1 = result[0]
        return (q1, r1)

    def name(self) -> str:
        return f"SealBot(t={self._time_limit})"
