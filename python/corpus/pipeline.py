"""CorpusPipeline — dedup, convert, and push GameRecords to RustReplayBuffer."""

from __future__ import annotations

import hashlib
from typing import Optional

import structlog

from python.bootstrap.dataset import replay_game_to_triples
from python.corpus.sources.base import CorpusSource, GameRecord
from python.corpus.metrics import CorpusMetrics

log = structlog.get_logger()


class CorpusPipeline:
    """Consumes :class:`~python.corpus.sources.base.GameRecord` objects from one or
    more :class:`~python.corpus.sources.base.CorpusSource` instances, converts each
    to training arrays, and pushes them into a ``RustReplayBuffer``.

    Responsibilities:
    1. Per-source in-memory deduplication by ordered SHA-256 move hash (Q1, Q6).
    2. Conversion: replay move sequence → pre-allocated (states, policies, outcomes).
    3. game_id assignment: monotonic ``int64`` counter starting at 0.
    4. Push to buffer via ``buf.push_game(states, policies, outcomes, game_id)``.
    5. Emit metrics via :class:`~python.corpus.metrics.CorpusMetrics`.

    Sources are consumed in order (human first, then hybrid). Human positions
    enter the buffer first and are proportionally sampled more heavily in early
    training steps.

    Args:
        sources: Ordered list of corpus sources. Consumed left-to-right.
        buffer:  ``RustReplayBuffer`` instance to push positions into.
        metrics: Optional :class:`~python.corpus.metrics.CorpusMetrics` for
                 per-source counters and throughput logging. A default instance
                 is created if not provided.
    """

    def __init__(
        self,
        sources: list[CorpusSource],
        buffer: object,
        metrics: Optional[CorpusMetrics] = None,
    ) -> None:
        self._sources       = sources
        self._buffer        = buffer
        self._metrics       = metrics if metrics is not None else CorpusMetrics()
        self._next_game_id  = 0

    def run(self, max_games: Optional[int] = None) -> None:
        """Run the pipeline over all sources.

        Args:
            max_games: If set, stop after pushing this many games total
                       (across all sources).
        """
        total_pushed = 0

        for source in self._sources:
            seen_hashes: set[str] = set()

            for record in source:
                if max_games is not None and total_pushed >= max_games:
                    return

                h = _game_hash(record)
                if h in seen_hashes:
                    self._metrics.record_duplicate(source.name())
                    continue
                seen_hashes.add(h)

                states, policies, outcomes = replay_game_to_triples(
                    record.moves, record.winner
                )
                if len(states) == 0:
                    log.warning("corpus_empty_game",
                                source=source.name(), game_id=record.game_id_str)
                    continue

                gid = self._next_game_id
                self._next_game_id += 1

                try:
                    self._buffer.push_game(states, policies, outcomes, gid)
                except Exception as exc:
                    log.error("corpus_push_failed",
                              source=source.name(), game_id=record.game_id_str,
                              error=str(exc))
                    self._next_game_id -= 1  # roll back counter
                    continue

                colony_bug  = bool(record.metadata.get("colony_bug_at_handoff", False))
                value_flip  = _is_value_flip(record)

                self._metrics.record_game(
                    source.name(),
                    n_positions=len(states),
                    colony_bug=colony_bug,
                    value_flip=value_flip,
                )

                if colony_bug:
                    log.info(
                        "corpus_colony_bug_exposure",
                        source=source.name(),
                        game_id_str=record.game_id_str,
                        clusters_at_handoff=record.metadata.get("cluster_count_at_handoff"),
                        ply_at_handoff=record.metadata.get("handoff_ply"),
                    )
                if value_flip:
                    log.info(
                        "corpus_value_flip",
                        source=source.name(),
                        game_id_str=record.game_id_str,
                        human_winner=record.metadata.get("human_winner"),
                        hybrid_winner=record.winner,
                    )

                total_pushed += 1

        self._metrics.flush()


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _game_hash(record: GameRecord) -> str:
    """SHA-256 of the ordered move sequence.

    Uses "|".join(f"{q},{r}" for q, r in moves) so the hash is sensitive to
    move order (distinct games with the same unordered set of moves are NOT
    considered duplicates).
    """
    payload = "|".join(f"{q},{r}" for q, r in record.moves).encode()
    return hashlib.sha256(payload).hexdigest()


def _is_value_flip(record: GameRecord) -> bool:
    """Return True if the hybrid outcome differs from the original human winner."""
    human_winner = record.metadata.get("human_winner")
    if human_winner is None:
        return False
    return int(human_winner) != int(record.winner)
