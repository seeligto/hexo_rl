"""RamoraBot — wraps the Ramora0/HexTicTacToe minimax engine."""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Optional
import structlog

from python.bootstrap.bot_protocol import BotProtocol
from python.env import GameState

log = structlog.get_logger()

_RAMORA_PATH = str(Path(__file__).parents[3] / "vendor" / "bots" / "ramora")
if _RAMORA_PATH not in sys.path:
    sys.path.insert(0, _RAMORA_PATH)

from game import HexGame, Player as RamoraPlayer  # type: ignore[import]
from ai import MinimaxBot  # type: ignore[import]

def _build_hex_game(state: GameState, rust_board: object) -> HexGame:
    game = HexGame()
    game.board = {}
    game.move_count = state.ply
    game.game_over = False
    game.winner = RamoraPlayer.NONE
    game.winning_cells = []

    # Populate all stones directly from rust board
    stones = rust_board.get_stones()
    for q, r, p in stones:
        if p == 1:
            game.board[(q, r)] = RamoraPlayer.A
        elif p == -1:
            game.board[(q, r)] = RamoraPlayer.B

    game.current_player = (
        RamoraPlayer.A if state.current_player == 1 else RamoraPlayer.B
    )
    game.moves_left_in_turn = state.moves_remaining

    return game

class RamoraBot(BotProtocol):
    def __init__(self, time_limit: float = 0.05) -> None:
        self._bot = MinimaxBot(time_limit=time_limit)
        self._pending_move: Optional[tuple[int, int]] = None

    def get_move(self, state: GameState, rust_board: object) -> tuple[int, int]:
        if self._pending_move is not None:
            move = self._pending_move
            self._pending_move = None
            if rust_board.get(move[0], move[1]) == 0:
                return move

        game = _build_hex_game(state, rust_board)
        result = self._bot.get_move(game)

        if len(result) == 0:
            log.warning("ramora_bot_no_moves_found", ply=state.ply)
            legal_moves = rust_board.legal_moves()
            if not legal_moves:
                raise RuntimeError("No legal moves available on board")
            import random
            return random.choice(legal_moves)

        safe_result = []
        for q, r in result:
            if rust_board.get(q, r) == 0:
                safe_result.append((q, r))
            else:
                log.warning("ramora_bot_returned_illegal_move", q=q, r=r)

        if not safe_result:
            import random
            return random.choice(rust_board.legal_moves())

        if len(safe_result) >= 2:
            self._pending_move = safe_result[1]

        return safe_result[0]

    def name(self) -> str:
        return f"ramora_bot(t={self._bot.time_limit:.3f})"
