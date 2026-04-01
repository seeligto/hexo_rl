"""Generate bot self-play corpus for bootstrap pretraining.

Provides:
  - ``load_cached_bot_games()``: returns cached bot games as move sequences
  - ``RAW_HUMAN_DIR``: path to scraped human game JSON files
  - CLI: ``python -m python.bootstrap.generate_corpus --bot sealbot --depth 4 --n-games 100``

Games are saved as JSON files to data/corpus/bot_games/<bot>_d<depth>/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import List, Tuple

import structlog

from native_core import Board
from python.bootstrap.bot_protocol import BotProtocol
from python.env.game_state import GameState

log = structlog.get_logger()

RAW_HUMAN_DIR = Path("data/corpus/raw_human")
BOT_GAMES_DIR = Path("data/corpus/bot_games")

MAX_MOVES_PER_GAME = 500


def _play_one_game(
    bot: BotProtocol,
    game_idx: int,
    rng_seed: int = 0,
) -> dict | None:
    """Play one self-play game using bot for both sides.

    Returns a dict with keys: moves, winner, plies, bot_name.
    Returns None if the game ends without a winner (capped).
    """
    import random

    board = Board()
    state = GameState.from_board(board)
    moves: list[tuple[int, int]] = []

    rng = random.Random(rng_seed + game_idx)

    while not board.check_win() and board.legal_move_count() > 0 and len(moves) < MAX_MOVES_PER_GAME:
        try:
            q, r = bot.get_move(state, board)
        except Exception as exc:
            log.warning("bot_move_error", game=game_idx, ply=len(moves), error=str(exc))
            break
        state = state.apply_move(board, q, r)
        moves.append((q, r))

    winner = board.winner()
    if winner is None:
        return None

    return {
        "moves": [{"x": q, "y": r} for q, r in moves],
        "winner": int(winner),
        "plies": len(moves),
        "bot_name": bot.name(),
    }


def _game_hash(moves: list[dict]) -> str:
    """SHA-256 of the move sequence, truncated to 16 hex chars."""
    key = json.dumps(moves, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def generate_bot_games(
    bot: BotProtocol,
    n_games: int,
    output_dir: Path,
    rng_seed: int = 42,
) -> int:
    """Generate n_games unique self-play games and save to output_dir.

    Games are named by a hash of their move sequence, so:
    - Re-running never overwrites existing games with different content
    - Duplicate games (identical move sequences) are detected and skipped

    Returns the number of new games saved (excludes duplicates and pre-existing).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    existing = set(p.stem for p in output_dir.glob("*.json"))
    saved = 0
    dupes = 0
    t0 = time.monotonic()

    for i in range(n_games):
        result = _play_one_game(bot, i, rng_seed=rng_seed)
        if result is None:
            log.info("game_no_winner", game=i, status="skipped")
            continue

        name = _game_hash(result["moves"])
        if name in existing:
            dupes += 1
            continue

        path = output_dir / f"{name}.json"
        with open(path, "w") as f:
            json.dump(result, f)
        existing.add(name)
        saved += 1

        if saved % 50 == 0:
            elapsed = time.monotonic() - t0
            rate = saved / elapsed if elapsed > 0 else 0
            log.info("corpus_progress", saved=saved, total=n_games,
                     dupes=dupes, rate_per_min=f"{rate * 60:.1f}")

    elapsed = time.monotonic() - t0
    log.info("corpus_generation_complete",
             saved=saved, dupes=dupes, attempted=n_games,
             total_on_disk=len(existing), elapsed_min=f"{elapsed / 60:.1f}")
    return saved


def load_cached_bot_games(bot_dir: Path | None = None) -> List[List[Tuple[int, int]]]:
    """Load all cached bot games from disk as move sequences.

    Args:
        bot_dir: Directory containing game JSON files.  Defaults to
                 data/corpus/bot_games/ (searches all subdirs).

    Returns:
        List of move sequences, each a list of (q, r) tuples.
    """
    if bot_dir is None:
        bot_dir = BOT_GAMES_DIR

    if not bot_dir.exists():
        log.info("no_bot_games_dir", path=str(bot_dir))
        return []

    games: list[list[tuple[int, int]]] = []
    json_files = sorted(bot_dir.rglob("*.json"))

    for p in json_files:
        try:
            with open(p) as f:
                data = json.load(f)
            moves = [(m["x"], m["y"]) for m in data["moves"]]
            games.append(moves)
        except Exception:
            continue

    log.info("loaded_cached_bot_games", count=len(games), dir=str(bot_dir))
    return games


def _make_bot(bot_name: str, depth: int | None, time_limit: float | None) -> BotProtocol:
    """Create a bot instance by name with optional depth/time overrides."""
    if bot_name == "sealbot":
        from python.bootstrap.bots.sealbot_bot import SealBotBot
        tl = time_limit if time_limit is not None else 1.0
        return SealBotBot(time_limit=tl, max_depth=depth)
    elif bot_name == "random":
        from python.bootstrap.bots.random_bot import RandomBot
        return RandomBot()
    else:
        raise ValueError(f"Unknown bot: {bot_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate bot self-play corpus")
    parser.add_argument("--bot", type=str, default="sealbot",
                        choices=["sealbot", "random"],
                        help="Bot to use for self-play")
    parser.add_argument("--depth", type=int, default=None,
                        help="Max search depth (SealBot only)")
    parser.add_argument("--time-limit", type=float, default=None,
                        help="Time limit per move in seconds")
    parser.add_argument("--n-games", type=int, default=100,
                        help="Number of games to generate")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (default: auto from bot+depth)")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed")
    args = parser.parse_args()

    bot = _make_bot(args.bot, args.depth, args.time_limit)
    log.info("bot_created", name=bot.name(), depth=args.depth,
             time_limit=args.time_limit)

    if args.output:
        output = Path(args.output)
    else:
        suffix = f"d{args.depth}" if args.depth else f"t{args.time_limit or 'default'}"
        output = BOT_GAMES_DIR / f"{args.bot}_{suffix}"

    generate_bot_games(bot, args.n_games, output, rng_seed=args.seed)


if __name__ == "__main__":
    main()
