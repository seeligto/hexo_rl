"""Tests for HybridGameSource — uses RandomBot to avoid C++ SealBot dependency."""

import json
import tempfile
from pathlib import Path
from typing import Iterator

import pytest
from python.corpus.sources.base import CorpusSource, GameRecord
from python.corpus.sources.hybrid_game_source import HybridGameSource
from python.bootstrap.bots.random_bot import RandomBot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_seed_source(moves_list: list[list[tuple[int, int]]], winner: int = 1) -> CorpusSource:
    """Build a CorpusSource that yields GameRecords with the given move sequences."""
    records = [
        GameRecord(
            game_id_str=f"seed-{i:04d}",
            moves=moves,
            winner=winner,
            source="human",
        )
        for i, moves in enumerate(moves_list)
    ]

    class _StaticSource(CorpusSource):
        def name(self):
            return "static"

        def __iter__(self) -> Iterator[GameRecord]:
            return iter(records)

        def __len__(self):
            return len(records)

    return _StaticSource()


def _long_game_moves(n: int = 60) -> list[tuple[int, int]]:
    """Generate a legal move sequence long enough to survive the turn-boundary snap
    and the min_bot_plies filter when RandomBot is used.

    Uses the 1-2-2 turn structure: P1 opens with 1 move, then both players play 2
    per turn, spiralling outwards from (0,0).
    """
    from native_core import Board
    board = Board()
    moves = []
    coords = [(q, r) for q in range(-5, 6) for r in range(-5, 6) if (q, r) != (0, 0)]
    all_coords = [(0, 0)] + coords
    for coord in all_coords:
        q, r = coord
        if board.check_win() or board.legal_move_count() == 0:
            break
        if board.get(q, r) == 0:
            try:
                board.apply_move(q, r)
                moves.append((q, r))
            except Exception:
                pass
        if len(moves) >= n:
            break
    return moves


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestHybridGameSourceBasic:
    def test_yields_game_records(self):
        moves = _long_game_moves(60)
        src = _make_seed_source([moves])
        hyb = HybridGameSource(src, RandomBot(), games_per_seed=1, min_bot_plies=2)
        records = list(hyb)
        assert len(records) >= 1
        assert all(isinstance(r, GameRecord) for r in records)

    def test_source_label_is_hybrid(self):
        moves = _long_game_moves(60)
        src = _make_seed_source([moves])
        hyb = HybridGameSource(src, RandomBot(), games_per_seed=1, min_bot_plies=2)
        for r in hyb:
            assert r.source == "hybrid"

    def test_winner_is_valid(self):
        moves = _long_game_moves(60)
        src = _make_seed_source([moves])
        hyb = HybridGameSource(src, RandomBot(), games_per_seed=1, min_bot_plies=2)
        for r in hyb:
            assert r.winner in (1, -1, 0)  # 0 = draw (game capped)

    def test_name(self):
        src = _make_seed_source([])
        assert HybridGameSource(src, RandomBot()).name() == "hybrid"

    def test_len_when_seed_source_has_known_len(self):
        moves = _long_game_moves(60)
        src = _make_seed_source([moves, moves])  # 2 seeds
        hyb = HybridGameSource(src, RandomBot(), games_per_seed=3)
        assert len(hyb) == 6  # 2 seeds × 3 continuations

    def test_len_none_when_seed_len_unknown(self):
        class _UnknownLen(CorpusSource):
            def name(self):
                return "x"
            def __iter__(self):
                return iter([])
            def __len__(self):
                return None

        hyb = HybridGameSource(_UnknownLen(), RandomBot())
        # Call __len__ directly — builtin len() requires an integer return.
        assert hyb.__len__() is None


class TestHybridHandoffPlayer:
    """Verify that even continuations give bot=P2 and odd give bot=P1."""

    def _get_records(self, games_per_seed: int = 3) -> list[GameRecord]:
        moves = _long_game_moves(60)
        src = _make_seed_source([moves])
        hyb = HybridGameSource(
            src, RandomBot(), games_per_seed=games_per_seed, min_bot_plies=2
        )
        return list(hyb)

    def test_continuation_0_bot_plays_as_p2(self):
        records = self._get_records()
        c0 = next((r for r in records if r.game_id_str.endswith(":c0")), None)
        assert c0 is not None
        assert c0.metadata["bot_plays_as"] == "P2"

    def test_continuation_1_bot_plays_as_p1(self):
        records = self._get_records()
        c1 = next((r for r in records if r.game_id_str.endswith(":c1")), None)
        assert c1 is not None
        assert c1.metadata["bot_plays_as"] == "P1"

    def test_continuation_2_bot_plays_as_p2(self):
        records = self._get_records()
        c2 = next((r for r in records if r.game_id_str.endswith(":c2")), None)
        assert c2 is not None
        assert c2.metadata["bot_plays_as"] == "P2"


class TestHybridEntropyInjection:
    """Verify that continuations from the same seed produce different move sequences."""

    def test_continuations_differ(self):
        moves = _long_game_moves(60)
        src = _make_seed_source([moves])
        hyb = HybridGameSource(
            src, RandomBot(), games_per_seed=3, min_bot_plies=2, rng_seed=0
        )
        records = list(hyb)
        # At least two continuations should exist and have different move sequences.
        assert len(records) >= 2
        move_sets = [tuple(r.moves) for r in records]
        assert len(set(move_sets)) > 1, "all continuations produced identical move sequences"


class TestHybridMinBotPlies:
    """Verify the min_bot_plies filter discards short games."""

    def test_very_high_threshold_discards_all(self):
        moves = _long_game_moves(60)
        src = _make_seed_source([moves])
        # A threshold of 10_000 is impossible to meet — all games should be discarded.
        hyb = HybridGameSource(
            src, RandomBot(), games_per_seed=2, min_bot_plies=10_000
        )
        assert list(hyb) == []

    def test_zero_threshold_passes_all(self):
        moves = _long_game_moves(60)
        src = _make_seed_source([moves])
        hyb = HybridGameSource(
            src, RandomBot(), games_per_seed=2, min_bot_plies=0
        )
        assert len(list(hyb)) == 2


class TestHybridMetadataKeys:
    def test_metadata_contains_required_keys(self):
        moves = _long_game_moves(60)
        src = _make_seed_source([moves], winner=1)
        hyb = HybridGameSource(src, RandomBot(), games_per_seed=1, min_bot_plies=2)
        records = list(hyb)
        assert len(records) >= 1
        meta = records[0].metadata
        required = {
            "seed_game_id", "continuation_idx", "human_winner",
            "handoff_ply", "bot_plays_as",
            "cluster_count_at_handoff", "colony_bug_at_handoff",
        }
        assert required <= meta.keys()

    def test_human_winner_preserved_in_metadata(self):
        moves = _long_game_moves(60)
        src = _make_seed_source([moves], winner=-1)
        hyb = HybridGameSource(src, RandomBot(), games_per_seed=1, min_bot_plies=2)
        for r in hyb:
            assert r.metadata["human_winner"] == -1


class TestHybridSeedTooShort:
    def test_short_seed_produces_no_records(self):
        # A 3-move game cannot reach the N=8 handoff.
        short_moves = [(0, 0), (1, 0), (1, 1)]
        src = _make_seed_source([short_moves])
        hyb = HybridGameSource(src, RandomBot(), n_opening_moves=8, games_per_seed=2)
        assert list(hyb) == []
