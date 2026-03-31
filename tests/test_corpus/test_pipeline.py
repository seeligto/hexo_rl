"""Tests for CorpusPipeline — uses mock source and real RustReplayBuffer."""

from typing import Iterator

import pytest
from native_core import RustReplayBuffer
from python.corpus.sources.base import CorpusSource, GameRecord
from python.corpus.pipeline import CorpusPipeline, _game_hash
from python.corpus.metrics import CorpusMetrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_source(records: list[GameRecord], name: str = "mock") -> CorpusSource:
    class _MockSource(CorpusSource):
        def name(self):
            return name

        def __iter__(self) -> Iterator[GameRecord]:
            return iter(records)

        def __len__(self):
            return len(records)

    return _MockSource()


def _real_game_record(game_id: str = "g0", winner: int = 1, offset: int = 0) -> GameRecord:
    """Build a short but valid game record that replay_game_to_triples can process.

    Uses the 1-2-2 turn structure starting at offset coordinates, giving 5 moves and
    up to 5 training positions. Each distinct offset produces a unique move sequence
    and therefore a unique dedup hash.
    """
    q = offset
    moves = [(q, 0), (q + 1, 0), (q + 1, 1), (q + 2, 0), (q + 2, 1)]
    return GameRecord(game_id_str=game_id, moves=moves, winner=winner, source="mock")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGameHash:
    def test_same_moves_same_hash(self):
        r1 = _real_game_record("a")
        r2 = _real_game_record("b")
        assert _game_hash(r1) == _game_hash(r2)

    def test_different_moves_different_hash(self):
        r1 = GameRecord("a", [(0, 0), (1, 0)], 1, "mock")
        r2 = GameRecord("b", [(1, 0), (0, 0)], 1, "mock")  # same moves, different order
        assert _game_hash(r1) != _game_hash(r2)

    def test_empty_moves(self):
        r = GameRecord("empty", [], 1, "mock")
        h = _game_hash(r)
        assert isinstance(h, str) and len(h) == 64  # SHA-256 hex


class TestCorpusPipelineBasic:
    def test_positions_pushed_to_buffer(self):
        src = _make_source([_real_game_record()])
        buf = RustReplayBuffer(capacity=1000)
        pipe = CorpusPipeline([src], buf)
        pipe.run()
        assert buf.size > 0

    def test_next_game_id_increments(self):
        src = _make_source([
            _real_game_record("g0", offset=0),
            _real_game_record("g1", offset=5),
            _real_game_record("g2", offset=10),
        ])
        buf = RustReplayBuffer(capacity=1000)
        pipe = CorpusPipeline([src], buf)
        pipe.run()
        assert pipe._next_game_id == 3

    def test_max_games_stops_early(self):
        src = _make_source([
            _real_game_record("g0", offset=0),
            _real_game_record("g1", offset=5),
            _real_game_record("g2", offset=10),
        ])
        buf = RustReplayBuffer(capacity=1000)
        pipe = CorpusPipeline([src], buf)
        pipe.run(max_games=1)
        assert pipe._next_game_id == 1

    def test_empty_source_does_nothing(self):
        src = _make_source([])
        buf = RustReplayBuffer(capacity=1000)
        pipe = CorpusPipeline([src], buf)
        pipe.run()
        assert buf.size == 0
        assert pipe._next_game_id == 0


class TestCorpusPipelineDedup:
    def test_duplicate_game_is_skipped(self):
        rec = _real_game_record("g0")
        # Same record yielded twice from the same source.
        src = _make_source([rec, rec])
        buf = RustReplayBuffer(capacity=1000)
        metrics = CorpusMetrics()
        pipe = CorpusPipeline([src], buf, metrics=metrics)
        pipe.run()
        # Only one game should be pushed.
        assert pipe._next_game_id == 1
        assert metrics.summary()["mock"]["games_duplicated"] == 1

    def test_dedup_is_per_source(self):
        rec = _real_game_record("g0")
        src_a = _make_source([rec], name="src_a")
        src_b = _make_source([rec], name="src_b")
        buf = RustReplayBuffer(capacity=1000)
        pipe = CorpusPipeline([src_a, src_b], buf)
        pipe.run()
        # Cross-source dedup is NOT applied (Q1) — both should be pushed.
        assert pipe._next_game_id == 2


class TestCorpusPipelineMetrics:
    def test_positions_counted(self):
        src = _make_source([_real_game_record("g0", offset=0), _real_game_record("g1", offset=5)])
        buf = RustReplayBuffer(capacity=1000)
        metrics = CorpusMetrics()
        pipe = CorpusPipeline([src], buf, metrics=metrics)
        pipe.run()
        summary = metrics.summary()
        assert summary["mock"]["games_processed"] == 2
        assert summary["mock"]["positions_pushed"] > 0

    def test_colony_bug_counted(self):
        rec = _real_game_record("g0")
        rec.metadata["colony_bug_at_handoff"] = True
        src = _make_source([rec])
        buf = RustReplayBuffer(capacity=1000)
        metrics = CorpusMetrics()
        pipe = CorpusPipeline([src], buf, metrics=metrics)
        pipe.run()
        assert metrics.summary()["mock"]["colony_bug_games"] == 1

    def test_value_flip_counted(self):
        rec = _real_game_record("g0", winner=1)
        rec.metadata["human_winner"] = -1  # flip: human says P2 won, hybrid says P1
        src = _make_source([rec])
        buf = RustReplayBuffer(capacity=1000)
        metrics = CorpusMetrics()
        pipe = CorpusPipeline([src], buf, metrics=metrics)
        pipe.run()
        assert metrics.summary()["mock"]["value_flip_games"] == 1
