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


def _make_bot(name: str, time_limit: float) -> BotProtocol:
    """Create a bot by name."""
    if name == "sealbot":
        from hexo_rl.bootstrap.bots.sealbot_bot import SealBotBot
        return SealBotBot(time_limit=time_limit)
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
        "--games", type=int, default=5,
        help="Number of bot continuations per human game (default: 5)",
    )
    parser.add_argument(
        "--bot", type=str, default="sealbot",
        choices=["sealbot", "random"],
        help="Bot to use for continuation (default: sealbot)",
    )
    parser.add_argument(
        "--time-limit", type=float, default=0.1,
        help="Bot time limit per move in seconds (default: 0.1)",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--min-moves", type=int, default=30,
        help="Minimum moves in human game to be eligible (default: 30)",
    )
    args = parser.parse_args()

    bot_p1 = _make_bot(args.bot, args.time_limit)
    bot_p2 = _make_bot(args.bot, args.time_limit)

    log.info(
        "injection_config",
        bot=args.bot,
        time_limit=args.time_limit,
        inject_at=args.inject_at,
        games_per_seed=args.games,
        output=str(args.output),
    )

    if args.human_game:
        games = [args.human_game]
    else:
        games = _find_eligible_games(args.human_dir, args.min_moves, args.inject_at)
        log.info("eligible_games_found", count=len(games), dir=str(args.human_dir))

    total_saved = 0
    for game_path in games:
        saved = run_injections(
            game_path=game_path,
            inject_at=args.inject_at,
            n_games=args.games,
            bot_p1=bot_p1,
            bot_p2=bot_p2,
            output_dir=args.output,
        )
        total_saved += len(saved)

    log.info(
        "injection_complete",
        total_saved=total_saved,
        human_games_processed=len(games),
    )


if __name__ == "__main__":
    main()
