"""Tests for python.eval.results_db — SQLite results store."""

import tempfile
from pathlib import Path

import pytest

from python.eval.results_db import ResultsDB


@pytest.fixture
def db(tmp_path: Path) -> ResultsDB:
    return ResultsDB(tmp_path / "test_results.db")


def test_create_and_query_player(db: ResultsDB) -> None:
    pid = db.get_or_create_player("checkpoint_100", "checkpoint", {"step": 100})
    assert isinstance(pid, int)
    assert db.get_player_name(pid) == "checkpoint_100"


def test_get_or_create_is_idempotent(db: ResultsDB) -> None:
    pid1 = db.get_or_create_player("random_bot", "random")
    pid2 = db.get_or_create_player("random_bot", "random")
    assert pid1 == pid2


def test_insert_and_query_match(db: ResultsDB) -> None:
    a = db.get_or_create_player("ckpt_100", "checkpoint")
    b = db.get_or_create_player("random_bot", "random")
    db.insert_match(
        eval_step=100, player_a_id=a, player_b_id=b,
        wins_a=18, wins_b=2, draws=0, n_games=20,
        win_rate_a=0.9, ci_lower=0.7, ci_upper=0.98,
    )
    pairs = db.get_all_pairwise()
    assert len(pairs) == 1
    assert pairs[0] == (a, b, 18, 2)


def test_pairwise_aggregation(db: ResultsDB) -> None:
    a = db.get_or_create_player("ckpt_100", "checkpoint")
    b = db.get_or_create_player("SealBot", "sealbot")
    db.insert_match(100, a, b, 10, 10, 0, 20, 0.5, 0.3, 0.7)
    db.insert_match(200, a, b, 15, 5, 0, 20, 0.75, 0.5, 0.9)
    pairs = db.get_all_pairwise()
    assert len(pairs) == 1
    assert pairs[0] == (a, b, 25, 15)


def test_insert_and_query_ratings(db: ResultsDB) -> None:
    a = db.get_or_create_player("ckpt_100", "checkpoint")
    b = db.get_or_create_player("random_bot", "random")
    db.insert_ratings(100, {a: (50.0, 20.0, 80.0), b: (-50.0, -80.0, -20.0)})

    history = db.get_ratings_history()
    assert len(history) == 2
    names = {h["player_name"] for h in history}
    assert names == {"ckpt_100", "random_bot"}


def test_ratings_history_ordering(db: ResultsDB) -> None:
    a = db.get_or_create_player("ckpt_100", "checkpoint")
    db.insert_ratings(100, {a: (10.0, 5.0, 15.0)})
    db.insert_ratings(200, {a: (20.0, 15.0, 25.0)})

    history = db.get_ratings_history()
    steps = [h["eval_step"] for h in history]
    assert steps == [100, 200]


def test_unknown_player_raises(db: ResultsDB) -> None:
    with pytest.raises(KeyError):
        db.get_player_name(99999)


def test_unique_match_constraint_replaces(db: ResultsDB) -> None:
    a = db.get_or_create_player("ckpt_100", "checkpoint")
    b = db.get_or_create_player("random_bot", "random")
    db.insert_match(100, a, b, 10, 10, 0, 20, 0.5, 0.3, 0.7)
    # Insert again with different results (OR REPLACE)
    db.insert_match(100, a, b, 15, 5, 0, 20, 0.75, 0.5, 0.9)
    pairs = db.get_all_pairwise()
    assert len(pairs) == 1
    assert pairs[0] == (a, b, 15, 5)


def test_all_player_ids(db: ResultsDB) -> None:
    a = db.get_or_create_player("a", "checkpoint")
    b = db.get_or_create_player("b", "sealbot")
    ids = db.get_all_player_ids()
    assert a in ids and b in ids
