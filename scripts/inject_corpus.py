#!/usr/bin/env python3
"""CLI: Generate bot-continuation games from human game seeds.

Usage:
    python scripts/inject_corpus.py \
        --human-game data/corpus/raw_human/some-game.json \
        --inject-at 15 \
        --games 5

    # Process all eligible games in a directory:
    python scripts/inject_corpus.py \
        --human-dir data/corpus/raw_human/ \
        --inject-at 15 \
        --games 3 \
        --min-moves 30

Output goes to data/corpus/injected/ by default.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import structlog

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hexo_rl.bootstrap.bot_protocol import BotProtocol
from hexo_rl.bootstrap.injection import (
    load_human_game,
    run_injections,
)

log = structlog.get_logger()

DEFAULT_OUTPUT_DIR = Path("data/corpus/injected")

# Preset time limits for named bot variants
_BOT_TIME_LIMITS = {
    "sealbot_fast": 0.1,
    "sealbot_strong": 0.5,
}


def _make_bot(name: str, time_limit: float) -> BotProtocol:
    """Create a bot by name."""
    if name in ("sealbot", "sealbot_fast", "sealbot_strong"):
        from hexo_rl.bootstrap.bots.sealbot_bot import SealBotBot
        effective_time = _BOT_TIME_LIMITS.get(name, time_limit)
        return SealBotBot(time_limit=effective_time)
    elif name == "random":
        from hexo_rl.bootstrap.bots.random_bot import RandomBot
        return RandomBot()
    else:
        raise ValueError(f"Unknown bot: {name}")


def _find_eligible_games(
    directory: Path,
    min_moves: int,
    inject_at: int,
) -> list[Path]:
    """Find human games with enough moves to inject at the given point."""
    eligible = []
    for p in sorted(directory.glob("*.json")):
        try:
            with open(p) as f:
                data = json.load(f)
            move_count = data.get("moveCount", len(data.get("moves", [])))
            # Need at least inject_at moves, plus some room for the game
            # to not already be terminal
            if move_count >= max(min_moves, inject_at + 5):
                eligible.append(p)
        except Exception:
            continue
    return eligible


def _process_one_game(
    game_path: str,
    inject_at: int,
    n_games: int,
    bot_name: str,
    time_limit: float,
    output_dir: str,
) -> int:
    """Worker function for parallel injection (runs in subprocess)."""
    bot_p1 = _make_bot(bot_name, time_limit)
    bot_p2 = _make_bot(bot_name, time_limit)
    saved = run_injections(
        game_path=Path(game_path),
        inject_at=inject_at,
        n_games=n_games,
        bot_p1=bot_p1,
        bot_p2=bot_p2,
        output_dir=Path(output_dir),
    )
    return len(saved)


def _run_parallel(
    games: list[Path],
    args: argparse.Namespace,
) -> tuple[int, int, int]:
    """Run injection across multiple worker processes."""
    total_saved = 0
    total_skipped = 0
    total_dupes = 0
    completed = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _process_one_game,
                str(game_path),
                args.inject_at,
                args.games,
                args.bot,
                args.time_limit,
                str(args.output),
            ): game_path
            for game_path in games
        }

        for future in as_completed(futures):
            completed += 1
            try:
                n_saved = future.result()
                total_saved += n_saved
                if n_saved == 0:
                    total_skipped += 1
            except Exception as exc:
                log.warning(
                    "injection_worker_error",
                    game=futures[future].name,
                    error=str(exc),
                )
                total_skipped += 1

            if completed % 50 == 0:
                log.info(
                    "injection_progress",
                    processed=completed,
                    total=len(games),
                    saved=total_saved,
                )

    return total_saved, total_skipped, total_dupes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate bot-continuation games from human seeds"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--human-game", type=Path,
        help="Path to a single human game JSON file",
    )
    group.add_argument(
        "--human-dir", type=Path,
        help="Directory of human game JSON files (process all eligible)",
    )
    parser.add_argument(
        "--inject-at", type=int, default=15,
        help="Move number to inject bots at (default: 15)",
    )
    parser.add_argument(
        "--games", "--games-per-seed", type=int, default=5,
        dest="games",
        help="Number of bot continuations per human game (default: 5)",
    )
    parser.add_argument(
        "--n-seeds", type=int, default=None,
        help="Limit number of seed games to process (default: all eligible)",
    )
    parser.add_argument(
        "--bot", type=str, default="sealbot",
        choices=["sealbot", "sealbot_fast", "sealbot_strong", "random"],
        help="Bot to use for continuation (default: sealbot)",
    )
    parser.add_argument(
        "--time-limit", type=float, default=0.1,
        help="Bot time limit per move in seconds (default: 0.1)",
    )
    parser.add_argument(
        "--output", "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        dest="output",
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--min-moves", type=int, default=30,
        help="Minimum moves in human game to be eligible (default: 30)",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of parallel worker processes (default: 1)",
    )
    args = parser.parse_args()

    log.info(
        "injection_config",
        bot=args.bot,
        time_limit=args.time_limit,
        inject_at=args.inject_at,
        games_per_seed=args.games,
        n_seeds=args.n_seeds,
        workers=args.workers,
        output=str(args.output),
    )

    if args.human_game:
        games = [args.human_game]
    else:
        games = _find_eligible_games(args.human_dir, args.min_moves, args.inject_at)
        log.info("eligible_games_found", count=len(games), dir=str(args.human_dir))

    if args.n_seeds is not None:
        games = games[: args.n_seeds]
        log.info("n_seeds_limit_applied", kept=len(games))

    total_saved = 0
    total_skipped = 0
    total_dupes = 0

    if args.workers > 1:
        # Parallel: each worker gets its own bot instances
        total_saved, total_skipped, total_dupes = _run_parallel(
            games, args,
        )
    else:
        bot_p1 = _make_bot(args.bot, args.time_limit)
        bot_p2 = _make_bot(args.bot, args.time_limit)
        for i, game_path in enumerate(games):
            saved = run_injections(
                game_path=game_path,
                inject_at=args.inject_at,
                n_games=args.games,
                bot_p1=bot_p1,
                bot_p2=bot_p2,
                output_dir=args.output,
            )
            total_saved += len(saved)
            if not saved:
                total_skipped += 1
            if (i + 1) % 50 == 0:
                log.info("injection_progress", processed=i + 1, total=len(games), saved=total_saved)

    log.info(
        "injection_complete",
        total_saved=total_saved,
        total_skipped=total_skipped,
        human_games_processed=len(games),
    )


if __name__ == "__main__":
    main()
