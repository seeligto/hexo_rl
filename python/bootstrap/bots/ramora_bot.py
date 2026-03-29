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
import structlog

from python.bootstrap.bot_protocol import BotProtocol
from python.env import GameState

log = structlog.get_logger()

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

    # ── Monkey-patch is_valid_move ──
    # Force Ramora to respect our 19x19 sliding window bounds.
    legal_set = set(rust_board.legal_moves())
    original_is_valid = game.is_valid_move

    def patched_is_valid_move(q: int, r: int) -> bool:
        # Must be both valid by Ramora's rules AND within our current window.
        valid = original_is_valid(q, r) and (q, r) in legal_set
        if not valid and (q, r) in game.board:
            # Already occupied, Ramora handles this
            pass
        elif not valid:
            # This is likely an out-of-window move being pruned
            pass
        return valid

    game.is_valid_move = patched_is_valid_move

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
            # Check if still legal (window might have shifted)
            if rust_board.in_window(move[0], move[1]) and rust_board.get(move[0], move[1]) == 0:
                return move
            # Window shifted or occupied, fall through to re-run search.

        game = _build_hex_game(state, rust_board)
        try:
            result = self._bot.get_move(game)
        except ValueError as e:
            if "move out of window" in str(e):
                log.error("ramora_bot_internal_error_out_of_window", error=str(e))
                import random
                return random.choice(rust_board.legal_moves())
            raise e

        # result is always a list of (q, r) tuples.
        if len(result) == 0:
            # This should be rare with the patched is_valid_move, but if search
            # fails to find any valid move within the window, fallback.
            log.warning("ramora_bot_no_moves_found", ply=state.ply)
            legal_moves = rust_board.legal_moves()
            if not legal_moves:
                raise RuntimeError("No legal moves available on board")
            import random
            return random.choice(legal_moves)

        # ── Final safety check ──
        # Even with the patched is_valid_move, if Ramora's search somehow
        # returns an out-of-window move (e.g. from TT or a bug), we must
        # not return it to our Board.apply_move().
        safe_result = []
        for q, r in result:
            if rust_board.in_window(q, r) and rust_board.get(q, r) == 0:
                safe_result.append((q, r))
            else:
                log.warning("ramora_bot_returned_illegal_move", q=q, r=r, 
                            in_window=rust_board.in_window(q, r),
                            occupied=rust_board.get(q, r) != 0)

        if not safe_result:
            import random
            return random.choice(rust_board.legal_moves())

        if len(safe_result) >= 2:
            # Cache second move for the next call.
            self._pending_move = safe_result[1]

        return safe_result[0]

    def name(self) -> str:
        return f"ramora_bot(t={self._bot.time_limit:.3f})"
