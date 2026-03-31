"""Tests for python/corpus/sources/base.py — GameRecord and CorpusSource."""

import pytest
from python.corpus.sources.base import GameRecord, CorpusSource


class TestGameRecord:
    def test_required_fields(self):
        rec = GameRecord(
            game_id_str="abc-123",
            moves=[(0, 0), (1, 0)],
            winner=1,
            source="human",
        )
        assert rec.game_id_str == "abc-123"
        assert rec.moves == [(0, 0), (1, 0)]
        assert rec.winner == 1
        assert rec.source == "human"

    def test_metadata_defaults_to_empty_dict(self):
        rec = GameRecord(game_id_str="x", moves=[], winner=-1, source="human")
        assert rec.metadata == {}

    def test_metadata_independent_between_instances(self):
        r1 = GameRecord(game_id_str="a", moves=[], winner=1, source="human")
        r2 = GameRecord(game_id_str="b", moves=[], winner=-1, source="human")
        r1.metadata["key"] = "val"
        assert "key" not in r2.metadata

    def test_winner_negative_one(self):
        rec = GameRecord(game_id_str="x", moves=[], winner=-1, source="hybrid")
        assert rec.winner == -1

    def test_hybrid_game_id_format(self):
        rec = GameRecord(
            game_id_str="uuid-seed:c2",
            moves=[],
            winner=1,
            source="hybrid",
            metadata={"continuation_idx": 2},
        )
        assert rec.game_id_str.endswith(":c2")
        assert rec.metadata["continuation_idx"] == 2


class TestCorpusSourceABC:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            CorpusSource()  # type: ignore[abstract]

    def test_concrete_subclass_requires_name_and_iter(self):
        class BadSource(CorpusSource):
            pass  # missing name() and __iter__

        with pytest.raises(TypeError):
            BadSource()

    def test_concrete_subclass_works(self):
        class GoodSource(CorpusSource):
            def name(self):
                return "good"

            def __iter__(self):
                yield GameRecord(game_id_str="g1", moves=[], winner=1, source="good")

        src = GoodSource()
        assert src.name() == "good"
        assert list(src) == [
            GameRecord(game_id_str="g1", moves=[], winner=1, source="good")
        ]

    def test_len_defaults_to_none(self):
        class MinimalSource(CorpusSource):
            def name(self):
                return "m"

            def __iter__(self):
                return iter([])

        # Call __len__ directly — builtin len() requires an integer return,
        # but CorpusSource.__len__ signals "unknown length" with None.
        assert MinimalSource().__len__() is None
