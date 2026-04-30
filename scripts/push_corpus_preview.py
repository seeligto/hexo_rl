#!/usr/bin/env python3
"""Push a selection of corpus games to the web dashboard for browsing.

Selection strategy (of N total games):
  - 40% longest games by ply count
  - 40% random sample
  - 20% most recent by timestamp

Writes /tmp/hexo_corpus_preview.jsonl (JSONL, one game object per line),
then optionally signals the running dashboard to reload.

Usage:
    python scripts/push_corpus_preview.py [--n 50] [--signal]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import List

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hexo_rl.monitoring.game_browser import (
    GameBrowser,
    GameSummary,
    SOURCE_SELF_PLAY,
)

OUTPUT_PATH = Path("/tmp/hexo_corpus_preview.jsonl")
CORPUS_DIR = "data/corpus"
REPLAY_DIR = "logs/replays"


def select_games(browser: GameBrowser, n: int) -> List[GameSummary]:
    """Select N corpus games using the 40/40/20 strategy.

    Returns deduplicated list (may be shorter than N if corpus is small).
    """
    n_longest = math.ceil(n * 0.4)
    n_random = math.ceil(n * 0.4)
    n_recent = n - n_longest - n_random  # remainder = 20%

    # Only corpus games — exclude self-play
    corpus_sources = ["human", "bot_fast", "bot_strong"]

    longest: List[GameSummary] = []
    for src in corpus_sources:
        longest.extend(browser.list_games(source=src, sort_by="length", limit=n_longest))
    longest.sort(key=lambda g: g.length, reverse=True)
    longest = longest[:n_longest]

    random_pool: List[GameSummary] = []
    for src in corpus_sources:
        random_pool.extend(browser.list_games(source=src, sort_by="random", limit=n_random * 3))
    # Deduplicate against longest
    seen = {g.game_id for g in longest}
    random_picks: List[GameSummary] = []
    for g in random_pool:
        if g.game_id not in seen:
            seen.add(g.game_id)
            random_picks.append(g)
            if len(random_picks) >= n_random:
                break

    recent: List[GameSummary] = []
    for src in corpus_sources:
        recent.extend(browser.list_games(source=src, sort_by="timestamp", limit=n_recent * 3))
    recent.sort(key=lambda g: g.timestamp, reverse=True)
    recent_picks: List[GameSummary] = []
    for g in recent:
        if g.game_id not in seen:
            seen.add(g.game_id)
            recent_picks.append(g)
            if len(recent_picks) >= n_recent:
                break

    return longest + random_picks + recent_picks


def _outcome_to_winner(outcome: str) -> str:
    """Map GameBrowser outcome to winner string."""
    return {"p1_win": "x_win", "p2_win": "o_win", "draw": "draw"}.get(outcome, "unknown")


def _source_label(source: str) -> str:
    """Map GameBrowser source to human-readable label."""
    return {
        "human": "human",
        "bot_fast": "sealbot-fast",
        "bot_strong": "sealbot-strong",
    }.get(source, source)


def write_preview(browser: GameBrowser, games: List[GameSummary]) -> int:
    """Write games to JSONL. Returns count written."""
    records = []
    for g in games:
        try:
            detail = browser.load_game(g.game_id)
        except (KeyError, Exception):
            continue
        moves = [list(m) for m in detail.moves]
        record = {
            "game_id": g.game_id,
            "game_length": g.length,
            "outcome": _outcome_to_winner(g.outcome),
            "timestamp": g.timestamp,
            "checkpoint_step": 0,
            "moves": moves,
            "source": _source_label(g.source),
        }
        records.append(record)

    OUTPUT_PATH.write_text(
        "\n".join(json.dumps(r) for r in records) + "\n",
        encoding="utf-8",
    )
    return len(records)


def signal_dashboard() -> None:
    """Send reload signal to running dashboard (silent on failure)."""
    try:
        import urllib.request
        req = urllib.request.Request(
            "http://localhost:5001/api/reload-corpus",
            method="POST",
            data=b"",
        )
        urllib.request.urlopen(req, timeout=2)
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Push corpus games to dashboard")
    parser.add_argument("--n", type=int, default=50, help="Number of games to push")
    parser.add_argument(
        "--signal", action="store_true", default=True,
        help="Signal running dashboard to reload (default: true)",
    )
    parser.add_argument("--no-signal", action="store_true", help="Skip dashboard signal")
    parser.add_argument("--corpus-dir", default=CORPUS_DIR)
    parser.add_argument("--replay-dir", default=REPLAY_DIR)
    args = parser.parse_args()

    browser = GameBrowser(corpus_dir=args.corpus_dir, replay_dir=args.replay_dir)
    selected = select_games(browser, args.n)

    if not selected:
        print("No corpus games found in data/corpus/")
        sys.exit(1)

    count = write_preview(browser, selected)

    if not args.no_signal:
        signal_dashboard()

    print(
        f"Pushed {count} corpus games to dashboard. "
        f"Open http://localhost:5001 to browse."
    )


if __name__ == "__main__":
    main()
