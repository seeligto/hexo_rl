"""RamoraBot — wraps the Ramora0/HexTicTacToe minimax engine.

The bot lives at vendor/bots/ramora/ (git submodule). Its internal interface
expects a `HexGame` object, so this wrapper reconstructs that state from our
`GameState` + live `rust_board` on every call.

NOTE: The Ramora ai.py has `pair_moves = True` — it plans both moves of a
double turn together. We honour this by caching the second move of each pair
and returning it on the next call without re-running search.
"""

from __future__ import annotations

import sys
import os
from pathlib import Path
from typing import Optional

from python.bootstrap.bot_protocol import BotProtocol
from python.env import GameState

# Ensure vendor/bots/ramora is importable as a flat-namespace package.
_RAMORA_PATH = str(Path(__file__).parents[3] / "vendor" / "bots" / "ramora")
if _RAMORA_PATH not in sys.path:
    sys.path.insert(0, _RAMORA_PATH)

from game import HexGame, Player as RamoraPlayer  # type: ignore[import]
from ai import MinimaxBot  # type: ignore[import]  # noqa: E402


def _build_hex_game(state: GameState, rust_board: object) -> HexGame:
    """Reconstruct a Ramora HexGame from our GameState.

    Uses window_center() from rust_board to convert window-relative board
    indices back to absolute (q, r) axial coordinates.
    """
    cq, cr = rust_board.window_center()
    board_arr = state.board  # shape (19, 19), int8: 0/+1/-1

    game = HexGame()
    # Bypass reset so we can set fields directly.
    game.board = {}
    game.move_count = state.ply
    game.game_over = False
    game.winner = RamoraPlayer.NONE
    game.winning_cells = []

    # Populate stone positions.
    for wq_idx in range(19):
        for wr_idx in range(19):
            cell = board_arr[wq_idx, wr_idx]
            if cell == 0:
                continue
            q = wq_idx - 9 + cq
            r = wr_idx - 9 + cr
            game.board[(q, r)] = (
                RamoraPlayer.A if cell == 1 else RamoraPlayer.B
            )

    # Set turn state.
    game.current_player = (
        RamoraPlayer.A if state.current_player == 1 else RamoraPlayer.B
    )
    game.moves_left_in_turn = state.moves_remaining
    return game


class RamoraBot(BotProtocol):
    """Wraps Ramora0's MinimaxBot with our BotProtocol interface.

    Because MinimaxBot plans both moves of a double turn simultaneously
    (pair_moves=True), we cache the second move of each planned pair and
    return it on the next call without re-running search.

    Args:
        time_limit: Per-move time budget in seconds (passed to MinimaxBot).
    """

    def __init__(self, time_limit: float = 0.05) -> None:
        self._bot = MinimaxBot(time_limit=time_limit)
        self._pending_move: Optional[tuple[int, int]] = None

    def get_move(self, state: GameState, rust_board: object) -> tuple[int, int]:
        # Return cached second move if available.
        if self._pending_move is not None:
            move = self._pending_move
            self._pending_move = None
            return move

        game = _build_hex_game(state, rust_board)
        result = self._bot.get_move(game)

        # result is always a list of (q, r) tuples.
        if len(result) == 0:
            raise RuntimeError("RamoraBot returned empty move list")

        if len(result) >= 2:
            # Cache second move for the next call.
            self._pending_move = result[1]

        return result[0]

    def name(self) -> str:
        return f"ramora_bot(t={self._bot.time_limit:.3f})"
