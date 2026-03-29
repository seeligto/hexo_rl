"""RamoraBot — wraps the Ramora0/HexTicTacToe minimax engine (C++ version)."""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Optional
import structlog

from python.bootstrap.bot_protocol import BotProtocol, ResignationException
from python.env import GameState

log = structlog.get_logger()

_RESIGN_THRESHOLD = -90_000_000 # Near -_WIN_SCORE (100,000,000)

_RAMORA_PATH = str(Path(__file__).parents[3] / "vendor" / "bots" / "ramora")
if _RAMORA_PATH not in sys.path:
    sys.path.insert(0, _RAMORA_PATH)

try:
    import ai_cpp
    log.info("ramora_bot_using_cpp_engine")
except ImportError:
    import ai as ai_cpp
    log.warning("ramora_bot_cpp_engine_not_found_using_python_fallback")

from game import Player as RamoraPlayer  # type: ignore[import]

class RamoraBot(BotProtocol):
    def __init__(self, time_limit: float = 0.05) -> None:
        self._bot = ai_cpp.MinimaxBot(time_limit=time_limit)
        self._pending_move: Optional[tuple[int, int]] = None

    def get_move(self, state: GameState, rust_board: object) -> tuple[int, int]:
        # Clear pending move if we are at the start of a turn (more than 1 move left)
        if state.moves_remaining > 1:
            self._pending_move = None

        if self._pending_move is not None:
            move = self._pending_move
            self._pending_move = None
            if rust_board.get(move[0], move[1]) == 0:
                return move
            else:
                log.warning("ramora_bot_pending_move_illegal", q=move[0], r=move[1])

        # C++ engine expects a dict of {(q, r): player_int} where player_int is 1 (A) or 2 (B)
        # and current_player as 1 (A) or 2 (B)
        # and moves_left_in_turn as 1 or 2
        
        board_dict = {}
        stones = rust_board.get_stones()
        for q, r, p in stones:
            board_dict[(q, r)] = 1 if p == 1 else 2

        current_player = 1 if state.current_player == 1 else 2
        moves_left = state.moves_remaining
        
        # We need to wrap the board_dict in an object that looks like HexGame for the Python side
        # but the C++ extension might take different arguments.
        # Looking at ai_cpp.cpp, it seems to expect a game object with .board, .current_player, .moves_left_in_turn
        
        class MockGame:
            def __init__(self, board, cp, ml):
                self.board = board
                self.current_player = RamoraPlayer.A if cp == 1 else RamoraPlayer.B
                self.moves_left_in_turn = ml
                self.move_count = len(board)
                self.winner = RamoraPlayer.NONE
                self.game_over = False

        game = MockGame(board_dict, current_player, moves_left)
        result = self._bot.get_move(game)

        # Check for resignation (forced loss) disabled to ensure full playouts
        # if hasattr(self._bot, "last_score"):
        #     score = self._bot.last_score
        #     if score <= _RESIGN_THRESHOLD:
        #         winner = -state.current_player
        #         log.info("ramora_bot_resigning", score=score, winner=winner, ply=state.ply)
        #         raise ResignationException(winner=winner)

        if not result:
            log.warning("ramora_bot_no_moves_found", ply=state.ply)
            legal_moves = rust_board.legal_moves()
            if not legal_moves:
                raise RuntimeError("No legal moves available on board")
            import random
            return random.choice(legal_moves)

        # C++ result might be a list of tuples or a single tuple
        if isinstance(result, tuple):
            result = [result]

        safe_result = []
        for q, r in result:
            if rust_board.get(q, r) == 0:
                safe_result.append((q, r))
            else:
                # Only log if it's not the first move on a fresh board (0,0)
                if not (q == 0 and r == 0 and state.ply > 0):
                    log.warning("ramora_bot_returned_illegal_move", q=q, r=r)

        if not safe_result:
            log.warning("ramora_bot_all_moves_illegal", ply=state.ply)
            legal_moves = rust_board.legal_moves()
            import random
            return random.choice(legal_moves)

        # Only store a pending move if we still have moves left in THIS turn
        if len(safe_result) >= 2 and state.moves_remaining > 1:
            self._pending_move = safe_result[1]

        return safe_result[0]

    def name(self) -> str:
        return f"ramora_bot_cpp(t={self._bot.time_limit:.3f})"
