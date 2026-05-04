"""ViewerEngine — game enrichment and play-against-model for /viewer.

Never imported by train.py, pool.py, or any training path.
"""

from __future__ import annotations

import re
from typing import Any

from engine import Board


def parse_axial(coord_str: str) -> tuple[int, int]:
    """Parse "(q,r)" string to (q, r) ints."""
    inner = coord_str.strip("()")
    parts = inner.split(",")
    return int(parts[0].strip()), int(parts[1].strip())


class ViewerEngine:
    """Enriches game records with threats and serves play-against-model."""

    def __init__(self, config: dict, checkpoint_path: str | None = None) -> None:
        self._config = config
        self._model_bot: Any = None
        if checkpoint_path:
            self._load_model(config, checkpoint_path)

    def _load_model(self, config: dict, checkpoint_path: str) -> None:
        """Lazy import to avoid pulling training deps at module level."""
        try:
            from hexo_rl.bootstrap.bots.our_model_bot import OurModelBot
            self._model_bot = OurModelBot(
                checkpoint_path=checkpoint_path,
                config=config,
                temperature=0.0,
            )
        except Exception:
            import traceback, sys
            traceback.print_exc(file=sys.stderr)
            self._model_bot = None

    def enrich_game(self, game_record: dict) -> dict:
        """Replay game move-by-move, calling board.get_threats() at each position.

        Returns game_record with added 'positions' field and a
        'data_capture_status' field describing which optional channels are
        populated (helps the frontend explain to users why MCTS heat / value
        sparkline may be empty — see spec §10 deferred items).
        """
        board = Board()
        positions: list[dict] = []
        moves = game_record.get("moves_list", [])
        value_trace = game_record.get("value_trace") or [None] * len(moves)
        moves_detail = game_record.get("moves_detail") or [None] * len(moves)

        has_value_trace = any(v is not None for v in value_trace)
        has_moves_detail = any(m is not None for m in moves_detail)

        for i, coord_str in enumerate(moves):
            q, r = parse_axial(coord_str)
            board.apply_move(q, r)
            threats = board.get_threats()
            positions.append({
                "move_index": i,
                "coord": coord_str,
                "value_est": value_trace[i] if i < len(value_trace) else None,
                "top_visits": moves_detail[i]["top_visits"] if (
                    i < len(moves_detail) and moves_detail[i]
                ) else None,
                "threats": [
                    {"q": t[0], "r": t[1], "level": t[2], "player": t[3]}
                    for t in threats
                ],
            })

        enriched = dict(game_record)
        enriched["positions"] = positions
        enriched["data_capture_status"] = {
            "threats": True,                  # always computed at enrich-time
            "value_trace": has_value_trace,   # spec §2.2 — captured by Rust if config.monitoring.capture_game_detail
            "moves_detail": has_moves_detail, # spec §2.2 — same
            "deferred_note": (
                "value_trace and moves_detail are spec §10 deferred items: "
                "they require Rust game_runner/worker_loop.rs to capture "
                "MCTSTree.root_value() and get_top_visits() per move before "
                "the tree is reset. Currently always None for self-play games."
            ),
        }
        return enriched

    def play_response(
        self, moves_so_far: list[str], human_moves: list[str]
    ) -> dict:
        """Reconstruct board, apply human moves, run model, return response."""
        if self._model_bot is None:
            return {"error": "no model loaded"}

        board = Board()

        # Replay existing game moves
        for coord_str in moves_so_far:
            q, r = parse_axial(coord_str)
            board.apply_move(q, r)

        # Apply human moves
        for coord_str in human_moves:
            q, r = parse_axial(coord_str)
            board.apply_move(q, r)

        # Build GameState for the model bot
        from hexo_rl.env.game_state import GameState
        state = GameState.from_board(board)

        # Model plays its turn (1 or 2 moves depending on moves_remaining)
        model_moves: list[str] = []
        moves_remaining = board.moves_remaining

        for _ in range(moves_remaining):
            q, r = self._model_bot.get_move(state, board)
            board.apply_move(q, r)
            model_moves.append(f"({q},{r})")
            state = GameState.from_board(board)

        # Get post-move analysis
        threats = board.get_threats()
        threat_list = [
            {"q": t[0], "r": t[1], "level": t[2], "player": t[3]}
            for t in threats
        ]

        # Get MCTS info from last search if available
        top_visits = None
        value_est = None
        try:
            tree = self._model_bot._worker.tree
            top_visits = [
                {"coord": c, "visits": int(v), "prior": float(p), "q_value": float(q)}
                for c, v, p, q in tree.get_top_visits(15)
            ]
            value_est = float(tree.root_value())
        except Exception:
            pass

        return {
            "moves": model_moves,
            "value_est": value_est,
            "top_visits": top_visits,
            "threats": threat_list,
            "winner": board.winner(),
        }
