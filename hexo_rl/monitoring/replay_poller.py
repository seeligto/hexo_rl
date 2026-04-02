"""Background poller for self-play game replays.

Scans the replay directory for new JSONL game files on a low-priority
daemon thread. Parsed games are stored in a capped LRU cache. All file I/O
happens on the poller thread — never in the Flask request handler.

Game files are written by ``GameRecorder`` as daily-rotated JSONL:
    logs/replays/games_YYYY-MM-DD.jsonl

Each line is a JSON object with keys:
    moves, outcome, game_length, timestamp, checkpoint_step

Note: MCTS visit counts are not currently recorded. To enable them,
set ``game_replay.include_mcts_visits: true`` in configs/game_replay.yaml
(not part of this task — left for a separate change).
"""

from __future__ import annotations

import json
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class GameReplaySummary:
    """Minimal parsed game replay for display."""

    filename: str
    game_length: int
    winner: str  # "x_win", "o_win", "draw"
    timestamp: str  # ISO 8601
    checkpoint_step: int
    move_sequence: List[List[int]] = field(default_factory=list)


class GameReplayPoller:
    """Daemon thread that scans for new self-play game replays.

    Args:
        replay_dir: Directory containing ``games_*.jsonl`` files.
        poll_interval_s: Seconds between scans (default 30).
        cache_cap: Maximum number of games in the LRU cache (default 50).
    """

    def __init__(
        self,
        replay_dir: str | Path = "logs/replays",
        poll_interval_s: float = 30.0,
        cache_cap: int = 50,
    ) -> None:
        self._replay_dir = Path(replay_dir)
        self._poll_interval = poll_interval_s
        self._cache_cap = cache_cap

        # LRU cache: OrderedDict keyed by (filename, line_index)
        self._cache: OrderedDict[str, GameReplaySummary] = OrderedDict()
        self._lock = threading.Lock()

        # Track byte offsets per file to only read new lines
        self._file_offsets: Dict[str, int] = {}
        self._game_counter = 0

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start the background poller thread."""
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._poll_loop, daemon=True, name="replay-poller"
        )
        self._thread.start()

    def stop(self) -> None:
        """Signal the poller to stop."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def get_recent(self, n: int = 5) -> List[GameReplaySummary]:
        """Return the N most recent games (newest first). Thread-safe."""
        with self._lock:
            items = list(self._cache.values())
        return list(reversed(items[-n:]))

    def get_game(self, key: str) -> Optional[GameReplaySummary]:
        """Return a specific game by cache key. Thread-safe."""
        with self._lock:
            return self._cache.get(key)

    # ── Internal ─────────────────────────────────────────────────────────────

    def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._scan()
            except Exception:
                pass  # never crash the background thread
            self._stop_event.wait(timeout=self._poll_interval)

    def _scan(self) -> None:
        if not self._replay_dir.is_dir():
            return
        for path in sorted(self._replay_dir.glob("games_*.jsonl")):
            self._read_new_lines(path)

    def _read_new_lines(self, path: Path) -> None:
        key = str(path)
        offset = self._file_offsets.get(key, 0)
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                f.seek(offset)
                for raw_line in f:
                    offset += len(raw_line.encode("utf-8", errors="replace"))
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    self._ingest(record, path.name)
        except OSError:
            return
        self._file_offsets[key] = offset

    def _ingest(self, record: Dict[str, Any], filename: str) -> None:
        self._game_counter += 1
        cache_key = f"{filename}:{self._game_counter}"
        summary = GameReplaySummary(
            filename=filename,
            game_length=int(record.get("game_length", 0)),
            winner=str(record.get("outcome", "unknown")),
            timestamp=str(record.get("timestamp", "")),
            checkpoint_step=int(record.get("checkpoint_step", 0)),
            move_sequence=record.get("moves", []),
        )
        with self._lock:
            # LRU eviction
            if cache_key not in self._cache and len(self._cache) >= self._cache_cap:
                self._cache.popitem(last=False)  # evict oldest
            self._cache[cache_key] = summary
            self._cache.move_to_end(cache_key)
