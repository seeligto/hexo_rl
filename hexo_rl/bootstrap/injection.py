"""Human-seed → bot-continuation injection pipeline.

Replays a human game to move N, then hands off to two BotProtocol bots
to complete the game. Produces game records compatible with the existing
corpus format (same JSON schema as generate_corpus.py output).

This is a prototype for evaluation — not yet integrated into the main
corpus pipeline.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

import structlog

from engine import Board
from hexo_rl.bootstrap.bot_protocol import BotProtocol, ResignationException
from hexo_rl.env.game_state import GameState

log = structlog.get_logger()

MAX_CONTINUATION_MOVES = 500


def load_human_game(path: Path) -> dict:
    """Load a human game JSON and return the raw dict."""
    with open(path) as f:
        return json.load(f)


def replay_to_move(
    game_record: dict,
    n: int,
) -> tuple[Board, GameState, list[tuple[int, int]]]:
    """Replay a stored human game to move n and return a valid state.

    Args:
        game_record: Parsed JSON dict from a human game file.
        n: Number of moves to replay (0-indexed into the moves array).
            E.g., n=15 replays the first 15 moves.

    Returns:
        (board, state, replayed_moves) where board and state are in sync
        at ply n, and replayed_moves is the (q,r) sequence applied.

    Raises:
        ValueError: If the game has fewer than n moves, or the position
            is already terminal at move n.
    """
    moves_raw = game_record.get("moves", [])
    if n > len(moves_raw):
        raise ValueError(
            f"Game has only {len(moves_raw)} moves, cannot replay to move {n}"
        )

    board = Board()
    state = GameState.from_board(board)
    replayed: list[tuple[int, int]] = []

    for i in range(n):
        m = moves_raw[i]
        q, r = m["x"], m["y"]

        if board.check_win():
            raise ValueError(
                f"Position is already terminal at move {i} "
                f"(before reaching target move {n})"
            )

        if board.legal_move_count() == 0:
            raise ValueError(
                f"No legal moves at ply {i} (before reaching target move {n})"
            )

        state = state.apply_move(board, q, r)
        replayed.append((q, r))

    if board.check_win():
        raise ValueError(
            f"Position is terminal at move {n} — cannot continue with bots"
        )

    return board, state, replayed


def inject_bot_continuation(
    board: Board,
    state: GameState,
    human_moves: list[tuple[int, int]],
    bot_p1: BotProtocol,
    bot_p2: BotProtocol,
    max_moves: int = MAX_CONTINUATION_MOVES,
) -> dict | None:
    """Run bot-vs-bot play from a pre-built position to completion.

    Args:
        board: Live Rust Board at the injection point.
        state: GameState matching the board.
        human_moves: The human move prefix (for the full game record).
        bot_p1: Bot playing as player 1 (first player).
        bot_p2: Bot playing as player 2 (second player).
        max_moves: Maximum additional bot moves before capping.

    Returns:
        A game record dict compatible with generate_corpus.py output
        format, or None if the game ends without a winner (capped).
        The record includes an 'injection_point' field marking where
        human moves end and bot moves begin.
    """
    all_moves = list(human_moves)
    bot_moves_start = len(all_moves)
    bot_move_count = 0

    while (
        not board.check_win()
        and board.legal_move_count() > 0
        and bot_move_count < max_moves
    ):
        current_player = state.current_player
        bot = bot_p1 if current_player == 1 else bot_p2

        try:
            q, r = bot.get_move(state, board)
        except ResignationException as exc:
            # Bot resigned — record the winner
            winner = exc.winner
            return _build_record(
                all_moves, winner, bot_moves_start, bot_p1, bot_p2,
                reason="resignation",
            )
        except Exception as exc:
            log.warning(
                "injection_bot_error",
                ply=len(all_moves),
                bot=bot.name(),
                error=str(exc),
            )
            break

        state = state.apply_move(board, q, r)
        all_moves.append((q, r))
        bot_move_count += 1

    winner = board.winner()
    if winner is None:
        return None

    return _build_record(
        all_moves, int(winner), bot_moves_start, bot_p1, bot_p2,
    )


def _build_record(
    moves: list[tuple[int, int]],
    winner: int,
    injection_point: int,
    bot_p1: BotProtocol,
    bot_p2: BotProtocol,
    reason: str = "six-in-a-row",
) -> dict:
    """Build a corpus-compatible game record with injection metadata."""
    move_dicts = [{"x": q, "y": r} for q, r in moves]
    move_hash = hashlib.sha256(
        json.dumps(move_dicts, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()[:16]

    return {
        "moves": move_dicts,
        "winner": winner,
        "plies": len(moves),
        "bot_name": f"{bot_p1.name()}_vs_{bot_p2.name()}",
        "source": "human_seed_bot_continuation",
        "injection_point": injection_point,
        "human_moves": injection_point,
        "bot_moves": len(moves) - injection_point,
        "reason": reason,
        "hash": move_hash,
    }


def run_injections(
    game_path: Path,
    inject_at: int,
    n_games: int,
    bot_p1: BotProtocol,
    bot_p2: BotProtocol,
    output_dir: Path,
) -> list[Path]:
    """Run multiple bot continuations from a single human game.

    Each continuation starts from the same human prefix but may produce
    different games if the bots are non-deterministic (e.g., time-limited
    search producing different results each run).

    Returns list of paths to saved game files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    game_record = load_human_game(game_path)
    saved: list[Path] = []

    for i in range(n_games):
        try:
            board, state, human_moves = replay_to_move(game_record, inject_at)
        except ValueError as exc:
            log.error(
                "injection_replay_failed",
                game=game_path.name,
                inject_at=inject_at,
                error=str(exc),
            )
            break

        result = inject_bot_continuation(
            board, state, human_moves, bot_p1, bot_p2,
        )

        if result is None:
            log.info("injection_no_winner", game=game_path.name, attempt=i)
            continue

        filename = f"{result['hash']}.json"
        out_path = output_dir / filename

        if out_path.exists():
            log.debug("injection_duplicate", hash=result["hash"], attempt=i)
            continue

        with open(out_path, "w") as f:
            json.dump(result, f)
        saved.append(out_path)

        log.info(
            "injection_saved",
            game=game_path.name,
            attempt=i,
            plies=result["plies"],
            human=result["human_moves"],
            bot=result["bot_moves"],
            winner=result["winner"],
            output=str(out_path),
        )

    return saved
