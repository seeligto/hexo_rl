"""Tests for CorpusMetrics and SourceMetrics."""

import time
import unittest.mock as mock

import pytest
from python.corpus.metrics import CorpusMetrics, SourceMetrics


class TestSourceMetrics:
    def test_defaults_are_zero(self):
        m = SourceMetrics()
        assert m.games_processed == 0
        assert m.games_duplicated == 0
        assert m.positions_pushed == 0
        assert m.colony_bug_games == 0
        assert m.value_flip_games == 0

    def test_positions_per_hour_zero_when_no_positions(self):
        m = SourceMetrics()
        # Zero positions → rate is 0 regardless of elapsed time.
        assert m.positions_per_hour() == 0.0

    def test_positions_per_hour_positive_when_positions_pushed(self):
        m = SourceMetrics()
        m.positions_pushed = 3600
        # wall_time_start was set at construction; if < 1s elapsed, rate is very high.
        # Just verify it's positive and finite.
        pph = m.positions_per_hour()
        assert pph > 0


class TestCorpusMetrics:
    def test_record_game_increments_counters(self):
        cm = CorpusMetrics()
        cm.record_game("human", n_positions=50)
        summary = cm.summary()
        assert summary["human"]["games_processed"] == 1
        assert summary["human"]["positions_pushed"] == 50

    def test_record_game_multiple_sources(self):
        cm = CorpusMetrics()
        cm.record_game("human", n_positions=30)
        cm.record_game("hybrid", n_positions=80, colony_bug=True)
        summary = cm.summary()
        assert summary["human"]["games_processed"] == 1
        assert summary["hybrid"]["colony_bug_games"] == 1

    def test_record_duplicate(self):
        cm = CorpusMetrics()
        cm.record_duplicate("human")
        cm.record_duplicate("human")
        summary = cm.summary()
        assert summary["human"]["games_duplicated"] == 2
        assert summary["human"]["games_processed"] == 0  # duplicates don't count as processed

    def test_colony_bug_counter(self):
        cm = CorpusMetrics()
        cm.record_game("hybrid", n_positions=10, colony_bug=True)
        cm.record_game("hybrid", n_positions=10, colony_bug=False)
        cm.record_game("hybrid", n_positions=10, colony_bug=True)
        summary = cm.summary()
        assert summary["hybrid"]["colony_bug_games"] == 2

    def test_value_flip_counter(self):
        cm = CorpusMetrics()
        cm.record_game("hybrid", n_positions=10, value_flip=True)
        cm.record_game("hybrid", n_positions=10, value_flip=False)
        summary = cm.summary()
        assert summary["hybrid"]["value_flip_games"] == 1

    def test_colony_bug_only_for_hybrid_not_human(self):
        cm = CorpusMetrics()
        cm.record_game("human", n_positions=40, colony_bug=False)
        summary = cm.summary()
        assert summary["human"]["colony_bug_games"] == 0

    def test_flush_interval_triggers_log(self):
        cm = CorpusMetrics(flush_interval=5)
        with mock.patch("python.corpus.metrics.log") as mock_log:
            for _ in range(5):
                cm.record_game("human", n_positions=10)
            mock_log.info.assert_called_once()
            call_kwargs = mock_log.info.call_args
            assert "corpus_throughput" in call_kwargs.args

    def test_flush_emits_log_for_all_sources(self):
        cm = CorpusMetrics(flush_interval=1000)  # won't auto-flush
        cm.record_game("human", n_positions=10)
        cm.record_game("hybrid", n_positions=20)
        with mock.patch("python.corpus.metrics.log") as mock_log:
            cm.flush()
            # Should emit one event per source.
            assert mock_log.info.call_count == 2

    def test_flush_single_source(self):
        cm = CorpusMetrics(flush_interval=1000)
        cm.record_game("human", n_positions=10)
        cm.record_game("hybrid", n_positions=20)
        with mock.patch("python.corpus.metrics.log") as mock_log:
            cm.flush(source="human")
            assert mock_log.info.call_count == 1

    def test_summary_returns_all_sources(self):
        cm = CorpusMetrics()
        cm.record_game("human", n_positions=5)
        cm.record_game("hybrid", n_positions=8)
        summary = cm.summary()
        assert set(summary.keys()) == {"human", "hybrid"}
