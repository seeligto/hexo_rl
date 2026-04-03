"""Read-only corpus and replay game browser for monitoring and diagnostics.

Index is built lazily on first access and cached in memory.
Re-scanned automatically when source files change (mtime check).

CLI usage:
    python -m hexo_rl.monitoring.game_browser --game-id <id>
    python -m hexo_rl.monitoring.game_browser --latest-selfplay 5
    python -m hexo_rl.monitoring.game_browser --longest 10
    python -m hexo_rl.monitoring.game_browser --shortest 10 --source self_play
"""
from __future__ import annotations

import json
import os
import random as _random
from collections import namedtuple
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

# ── Public types ───────────────────────────────────────────────────────────────

GameSummary = namedtuple(
    "GameSummary",
    ["game_id", "source", "length", "outcome", "quality_score", "timestamp"],
)
GameDetail = namedtuple(
    "GameDetail",
    [
        "game_id",
        "source",
        "length",
        "outcome",
        "quality_score",
        "timestamp",
        "moves",
        "value_estimates",
        "cluster_count_at_end",
    ],
)

# ── Source constants ───────────────────────────────────────────────────────────

SOURCE_HUMAN = "human"
SOURCE_BOT_FAST = "bot_fast"
SOURCE_BOT_STRONG = "bot_strong"
SOURCE_SELF_PLAY = "self_play"

# ── Internal index record ──────────────────────────────────────────────────────


@dataclass
class _IndexEntry:
    game_id: str
    source: str
    length: int
    outcome: str        # "p1_win" | "p2_win" | "draw" | "unknown"
    quality_score: float
    timestamp: str      # ISO 8601 or ""
    file_path: str      # absolute path to source file
    line_no: int        # >= 0 for JSONL line index; -1 for standalone JSON


# ── GameBrowser ────────────────────────────────────────────────────────────────


class GameBrowser:
    """Read-only index and loader for corpus and self-play replay games.

    Data sources:
        Human:      ``<corpus_dir>/raw_human/*.json``
        Bot fast:   ``<corpus_dir>/bot_games/sealbot_fast/*.json``
        Bot strong: ``<corpus_dir>/bot_games/sealbot_strong/*.json``
        Self-play:  ``<replay_dir>/games_*.jsonl``

    Quality scores are joined from ``<corpus_dir>/quality_scores.json``.
    """

    def __init__(self, corpus_dir: str, replay_dir: str) -> None:
        self._corpus_dir = Path(corpus_dir)
        self._replay_dir = Path(replay_dir)
        self._index: List[_IndexEntry] = []
        self._index_built = False
        self._scan_fingerprint: dict = {}  # {path_str: mtime_float}

    # ── Public ────────────────────────────────────────────────────────────────

    def list_games(
        self,
        source: str = "all",
        sort_by: str = "length",
        min_length: int = 0,
        max_length: int = 999,
        outcome: str = "all",
        limit: int = 50,
    ) -> List[GameSummary]:
        """Filter and sort available games.

        Args:
            source:     ``"all"`` | ``"human"`` | ``"bot_fast"`` | ``"bot_strong"`` |
                        ``"self_play"``
            sort_by:    ``"length"`` (desc) | ``"length_asc"`` | ``"quality"`` |
                        ``"timestamp"`` (most-recent-first) | ``"random"``
            min_length: Inclusive lower bound on game length.
            max_length: Inclusive upper bound on game length.
            outcome:    ``"all"`` | ``"p1_win"`` | ``"p2_win"`` | ``"draw"`` |
                        ``"unknown"``
            limit:      Maximum number of results to return.
        """
        self._ensure_index()

        results = [
            e for e in self._index
            if (source == "all" or e.source == source)
            and min_length <= e.length <= max_length
            and (outcome == "all" or e.outcome == outcome)
        ]

        if sort_by == "random":
            sample = _random.sample(results, min(limit, len(results)))
            return [_to_summary(e) for e in sample]

        if sort_by == "length":
            results.sort(key=lambda e: e.length, reverse=True)
        elif sort_by == "length_asc":
            results.sort(key=lambda e: e.length)
        elif sort_by == "quality":
            results.sort(key=lambda e: e.quality_score, reverse=True)
        elif sort_by == "timestamp":
            results.sort(key=lambda e: e.timestamp, reverse=True)

        return [_to_summary(e) for e in results[:limit]]

    def load_game(self, game_id: str) -> GameDetail:
        """Load the full move sequence and metadata for a specific game."""
        self._ensure_index()
        entry = next((e for e in self._index if e.game_id == game_id), None)
        if entry is None:
            raise KeyError(f"game {game_id!r} not found in index")

        moves, value_estimates = self._read_moves(entry)
        cluster_count = _compute_cluster_count(moves)

        return GameDetail(
            game_id=entry.game_id,
            source=entry.source,
            length=entry.length,
            outcome=entry.outcome,
            quality_score=entry.quality_score,
            timestamp=entry.timestamp,
            moves=moves,
            value_estimates=value_estimates,
            cluster_count_at_end=cluster_count,
        )

    # ── Index management ──────────────────────────────────────────────────────

    def _ensure_index(self) -> None:
        if not self._index_built or self._needs_rescan():
            self._build_index()

    def _needs_rescan(self) -> bool:
        return self._current_fingerprint() != self._scan_fingerprint

    def _current_fingerprint(self) -> dict:
        fp: dict = {}

        for subdir in ["raw_human", "bot_games/sealbot_fast", "bot_games/sealbot_strong"]:
            d = self._corpus_dir / subdir
            if d.exists():
                try:
                    for e in os.scandir(d):
                        if e.name.endswith(".json"):
                            fp[e.path] = e.stat().st_mtime
                except OSError:
                    pass

        qs = self._corpus_dir / "quality_scores.json"
        if qs.exists():
            try:
                fp[str(qs)] = qs.stat().st_mtime
            except OSError:
                pass

        if self._replay_dir.exists():
            try:
                for e in os.scandir(self._replay_dir):
                    if e.name.endswith(".jsonl"):
                        fp[e.path] = e.stat().st_mtime
            except OSError:
                pass

        return fp

    def _build_index(self) -> None:
        self._scan_fingerprint = self._current_fingerprint()
        quality_scores = self._load_quality_scores()

        entries: List[_IndexEntry] = []
        entries.extend(_index_human_games(self._corpus_dir, quality_scores))
        entries.extend(_index_bot_games(self._corpus_dir, "sealbot_fast", SOURCE_BOT_FAST, quality_scores))
        entries.extend(_index_bot_games(self._corpus_dir, "sealbot_strong", SOURCE_BOT_STRONG, quality_scores))
        entries.extend(_index_replay_games(self._replay_dir))

        self._index = entries
        self._index_built = True

    def _load_quality_scores(self) -> dict:
        qs_path = self._corpus_dir / "quality_scores.json"
        if not qs_path.exists():
            return {}
        try:
            with qs_path.open() as f:
                return json.load(f)
        except Exception:
            return {}

    # ── Move loading ──────────────────────────────────────────────────────────

    def _read_moves(
        self, entry: _IndexEntry
    ) -> Tuple[List[Tuple[int, int]], Optional[List[float]]]:
        if entry.line_no >= 0:
            return _read_replay_moves(entry.file_path, entry.line_no)
        elif entry.source == SOURCE_HUMAN:
            return _read_human_moves(entry.file_path), None
        else:
            return _read_bot_moves(entry.file_path), None


# ── Module-level indexing helpers ─────────────────────────────────────────────


def _index_human_games(corpus_dir: Path, quality_scores: dict) -> List[_IndexEntry]:
    human_dir = corpus_dir / "raw_human"
    if not human_dir.exists():
        return []

    entries = []
    for path in human_dir.glob("*.json"):
        try:
            data = json.loads(path.read_bytes())
        except Exception:
            continue

        moves_data = data.get("moves", [])
        if not moves_data:
            continue

        length = data.get("moveCount", len(moves_data))

        # P1 is the player who made the first move (move number 2 in notation;
        # move number 1 is a game-creation event absent from the JSON).
        p1_id = moves_data[0].get("playerId", "")
        winning_id = data.get("gameResult", {}).get("winningPlayerId", "")
        if p1_id and winning_id:
            outcome = "p1_win" if winning_id == p1_id else "p2_win"
        else:
            outcome = "unknown"

        started_at = data.get("startedAt")
        if started_at:
            ts = datetime.fromtimestamp(started_at / 1000.0, tz=timezone.utc).isoformat()
        else:
            ts = ""

        game_id = path.stem
        qs = quality_scores.get(game_id, {})
        quality_score = float(qs.get("quality_score", 0.0))

        entries.append(_IndexEntry(
            game_id=game_id,
            source=SOURCE_HUMAN,
            length=length,
            outcome=outcome,
            quality_score=quality_score,
            timestamp=ts,
            file_path=str(path),
            line_no=-1,
        ))
    return entries


def _index_bot_games(
    corpus_dir: Path, subdir: str, source: str, quality_scores: dict
) -> List[_IndexEntry]:
    bot_dir = corpus_dir / "bot_games" / subdir
    if not bot_dir.exists():
        return []

    entries = []
    for path in bot_dir.glob("*.json"):
        try:
            data = json.loads(path.read_bytes())
        except Exception:
            continue

        plies = data.get("plies", len(data.get("moves", [])))
        winner = data.get("winner", 0)
        if winner == 1:
            outcome = "p1_win"
        elif winner == 2:
            outcome = "p2_win"
        else:
            outcome = "unknown"

        try:
            ts = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
        except OSError:
            ts = ""

        game_id = path.stem
        qs = quality_scores.get(game_id, {})
        quality_score = float(qs.get("quality_score", 0.0))

        entries.append(_IndexEntry(
            game_id=game_id,
            source=source,
            length=plies,
            outcome=outcome,
            quality_score=quality_score,
            timestamp=ts,
            file_path=str(path),
            line_no=-1,
        ))
    return entries


def _index_replay_games(replay_dir: Path) -> List[_IndexEntry]:
    if not replay_dir.exists():
        return []

    entries = []
    for path in sorted(replay_dir.glob("games_*.jsonl")):
        try:
            with path.open(encoding="utf-8") as f:
                for line_no, raw in enumerate(f):
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        record = json.loads(raw)
                    except json.JSONDecodeError:
                        continue

                    outcome_map = {
                        "x_win": "p1_win",
                        "o_win": "p2_win",
                        "draw":  "draw",
                    }
                    outcome = outcome_map.get(record.get("outcome", ""), "unknown")
                    length = record.get("game_length", len(record.get("moves", [])))
                    ts = record.get("timestamp", "")
                    game_id = f"sp_{path.stem}_{line_no:04d}"

                    entries.append(_IndexEntry(
                        game_id=game_id,
                        source=SOURCE_SELF_PLAY,
                        length=length,
                        outcome=outcome,
                        quality_score=0.0,
                        timestamp=ts,
                        file_path=str(path),
                        line_no=line_no,
                    ))
        except OSError:
            continue
    return entries


# ── Game-file readers ─────────────────────────────────────────────────────────


def _read_human_moves(file_path: str) -> List[Tuple[int, int]]:
    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)
    return [(m["x"], m["y"]) for m in data.get("moves", [])]


def _read_bot_moves(file_path: str) -> List[Tuple[int, int]]:
    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)
    return [(m["x"], m["y"]) for m in data.get("moves", [])]


def _read_replay_moves(
    file_path: str, line_no: int
) -> Tuple[List[Tuple[int, int]], Optional[List[float]]]:
    with open(file_path, encoding="utf-8") as f:
        for i, raw in enumerate(f):
            if i == line_no:
                record = json.loads(raw.strip())
                moves = [tuple(m) for m in record.get("moves", [])]
                value_estimates = record.get("value_estimates")
                return moves, value_estimates  # type: ignore[return-value]
    return [], None


# ── Cluster count (requires Rust engine) ─────────────────────────────────────


def _compute_cluster_count(moves: List[Tuple[int, int]]) -> Optional[int]:
    """Return cluster count at end of game, or None if engine is unavailable."""
    try:
        from engine import Board  # type: ignore[import]
        from hexo_rl.env.game_state import GameState
    except ImportError:
        return None

    try:
        board = Board()
        state = GameState.from_board(board)
        for q, r in moves:
            state = state.apply_move(board, q, r)
        _, centers = state.to_tensor()
        return len(centers)
    except Exception:
        return None


# ── Namedtuple conversion ─────────────────────────────────────────────────────


def _to_summary(e: _IndexEntry) -> GameSummary:
    return GameSummary(
        game_id=e.game_id,
        source=e.source,
        length=e.length,
        outcome=e.outcome,
        quality_score=e.quality_score,
        timestamp=e.timestamp,
    )


# ── CLI ───────────────────────────────────────────────────────────────────────


def _print_game_detail(game: GameDetail) -> None:
    """Print a game's move sequence in human-readable form."""
    outcome_label = {
        "p1_win": "P1 wins",
        "p2_win": "P2 wins",
        "draw":   "draw",
    }.get(game.outcome, game.outcome)

    print(
        f"\nGame: {game.game_id}  |  "
        f"{game.source.replace('_', '-')}  |  "
        f"{game.length} moves  |  {outcome_label}"
    )

    if game.value_estimates:
        print(f"{'#':>3}  {'Move':>10}  {'Value':>7}")
        print("-" * 25)
        for i, (move, val) in enumerate(zip(game.moves, game.value_estimates), 1):
            print(f"{i:>3}  {str(move):>10}  {val:>7.4f}")
        # Any remaining moves without value estimates
        for i, move in enumerate(game.moves[len(game.value_estimates):],
                                  len(game.value_estimates) + 1):
            print(f"{i:>3}  {str(move):>10}  {'':>7}")
    else:
        print(f"{'#':>3}  {'Move':>10}")
        print("-" * 16)
        for i, move in enumerate(game.moves, 1):
            print(f"{i:>3}  {str(move):>10}")

    if game.cluster_count_at_end is not None:
        print(f"\nClusters at end: {game.cluster_count_at_end}")


def _print_game_list(games: List[GameSummary]) -> None:
    """Print a list of games as a table."""
    if not games:
        print("No games found.")
        return
    header = f"{'#':>3}  {'game_id':<40}  {'source':<10}  {'length':>6}  {'outcome':<8}  {'quality':>7}"
    print(header)
    print("-" * len(header))
    for i, g in enumerate(games, 1):
        print(
            f"{i:>3}  {g.game_id:<40}  {g.source:<10}  {g.length:>6}  "
            f"{g.outcome:<8}  {g.quality_score:>7.4f}"
        )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Corpus and replay game browser")
    parser.add_argument("--corpus-dir",  default="data/corpus",  help="Corpus directory")
    parser.add_argument("--replay-dir",  default="logs/replays", help="Self-play replay directory")
    parser.add_argument("--source",      default="all",
                        choices=["all", "human", "bot_fast", "bot_strong", "self_play"])
    parser.add_argument("--game-id",        help="Show full move sequence for a specific game")
    parser.add_argument("--latest-selfplay", type=int, metavar="N",
                        help="Show N most-recent self-play games")
    parser.add_argument("--longest",         type=int, metavar="N",
                        help="Show N longest games (by --source)")
    parser.add_argument("--shortest",        type=int, metavar="N",
                        help="Show N shortest games (by --source)")
    args = parser.parse_args()

    browser = GameBrowser(corpus_dir=args.corpus_dir, replay_dir=args.replay_dir)

    if args.game_id:
        try:
            game = browser.load_game(args.game_id)
        except KeyError as exc:
            print(f"Error: {exc}")
            return
        _print_game_detail(game)

    elif args.latest_selfplay:
        games = browser.list_games(
            source=SOURCE_SELF_PLAY, sort_by="timestamp", limit=args.latest_selfplay
        )
        _print_game_list(games)

    elif args.longest:
        games = browser.list_games(
            source=args.source, sort_by="length", limit=args.longest
        )
        _print_game_list(games)

    elif args.shortest:
        games = browser.list_games(
            source=args.source, sort_by="length_asc", min_length=1, limit=args.shortest
        )
        _print_game_list(games)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
