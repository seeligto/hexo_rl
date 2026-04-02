"""CorpusMetrics — per-source counters and throughput logging for the corpus pipeline."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import structlog

log = structlog.get_logger()


@dataclass
class SourceMetrics:
    """Accumulated counters for a single corpus source."""

    games_processed: int = 0
    games_duplicated: int = 0
    positions_pushed: int = 0
    colony_bug_games: int = 0   # hybrid only: cluster_count > 4 at handoff
    value_flip_games: int = 0   # hybrid only: hybrid_winner != human_winner
    wall_time_start: float = field(default_factory=time.monotonic)

    def positions_per_hour(self) -> float:
        elapsed = time.monotonic() - self.wall_time_start
        if elapsed <= 0:
            return 0.0
        return self.positions_pushed / elapsed * 3600.0


class CorpusMetrics:
    """Accumulates per-source counters and flushes them via structlog.

    :meth:`record_game` should be called after each successfully pushed game.
    :meth:`record_duplicate` should be called when a game is skipped due to
    deduplication.

    Throughput events are emitted every ``flush_interval`` games (per source)
    and once at :meth:`flush` time (end of pipeline run).

    Args:
        flush_interval: Emit a ``corpus_throughput`` log event every this many
                        games per source. Default 100.
    """

    def __init__(self, flush_interval: int = 100) -> None:
        self._flush_interval = flush_interval
        self._sources: dict[str, SourceMetrics] = {}

    def _get(self, source: str) -> SourceMetrics:
        if source not in self._sources:
            self._sources[source] = SourceMetrics()
        return self._sources[source]

    def record_game(
        self,
        source: str,
        n_positions: int,
        *,
        colony_bug: bool = False,
        value_flip: bool = False,
    ) -> None:
        """Record a successfully pushed game."""
        m = self._get(source)
        m.games_processed += 1
        m.positions_pushed += n_positions
        if colony_bug:
            m.colony_bug_games += 1
        if value_flip:
            m.value_flip_games += 1

        if m.games_processed % self._flush_interval == 0:
            self._emit_throughput(source, m)

    def record_duplicate(self, source: str) -> None:
        """Record a game skipped due to deduplication."""
        self._get(source).games_duplicated += 1

    def flush(self, source: Optional[str] = None) -> None:
        """Emit final throughput log events.

        Args:
            source: If given, flush only that source. Otherwise flush all.
        """
        targets = [source] if source is not None else list(self._sources.keys())
        for s in targets:
            if s in self._sources:
                self._emit_throughput(s, self._sources[s])

    def summary(self) -> dict:
        """Return a dict summary of all source metrics (for tests)."""
        return {
            s: {
                "games_processed": m.games_processed,
                "games_duplicated": m.games_duplicated,
                "positions_pushed": m.positions_pushed,
                "colony_bug_games": m.colony_bug_games,
                "value_flip_games": m.value_flip_games,
            }
            for s, m in self._sources.items()
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _emit_throughput(self, source: str, m: SourceMetrics) -> None:
        log.info(
            "corpus_throughput",
            source=source,
            positions_per_hour=round(m.positions_per_hour()),
            positions_total=m.positions_pushed,
            games_processed=m.games_processed,
            games_duplicated=m.games_duplicated,
        )
