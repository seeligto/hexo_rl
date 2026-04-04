"""Tests for python.eval.results_db — SQLite results store."""

import tempfile
from pathlib import Path

import pytest

from hexo_rl.eval.results_db import ResultsDB


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


def test_results_db_run_id_scoping(db: ResultsDB) -> None:
    run1 = "run1"
    run2 = "run2"

    # Reference opponent (no run_id / empty run_id)
    ref_id = db.get_or_create_player("SealBot", "sealbot", run_id="")

    # Players for run1
    p1_run1 = db.get_or_create_player("ckpt_100", "checkpoint", run_id=run1)

    # Players for run2
    p1_run2 = db.get_or_create_player("ckpt_100", "checkpoint", run_id=run2)

    # Matches for run1
    db.insert_match(100, p1_run1, ref_id, 10, 0, 0, 10, 1.0, 0.9, 1.0, run_id=run1)

    # Matches for run2
    db.insert_match(100, p1_run2, ref_id, 5, 5, 0, 10, 0.5, 0.3, 0.7, run_id=run2)

    # Scoping for run1
    pairwise1 = db.get_all_pairwise(run_id=run1)
    # Should see p1_run1 vs ref, NOT p1_run2 vs ref
    assert len(pairwise1) == 1
    assert pairwise1[0][0] == p1_run1
    assert pairwise1[0][2] == 10  # wins_a

    # Scoping for run2
    pairwise2 = db.get_all_pairwise(run_id=run2)
    # Should see p1_run2 vs ref, NOT p1_run1 vs ref
    assert len(pairwise2) == 1
    assert pairwise2[0][0] == p1_run2
    assert pairwise2[0][2] == 5  # wins_a

