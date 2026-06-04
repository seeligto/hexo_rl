"""Tests for scripts/export_corpus_jsonl.py — the encoding-free JSONL exporter."""

from __future__ import annotations

import json
from pathlib import Path

import importlib.util

# Load the script as a module (scripts/ is not a package).
_SPEC = importlib.util.spec_from_file_location(
    "export_corpus_jsonl",
    Path(__file__).resolve().parent.parent / "scripts" / "export_corpus_jsonl.py",
)
assert _SPEC is not None and _SPEC.loader is not None
exporter = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(exporter)


_DEFAULT_MOVES = [(0, 0), (2, -1), (1, 0)]


def _human_game(p1, p2, winner_id, *, rated=True, reason="six-in-a-row", move_count=20, moves=None):
    """Build a minimal scrubbed-schema human game dict. p1 plays the first move.

    ``moves`` is a list of ``(x, y)`` tuples; only the first move's player is
    consulted by the parser, so we tag move[0] as p1 and the rest as p2.
    """
    moves = _DEFAULT_MOVES if moves is None else moves
    moves_data = [
        {"anon_player": p1 if i == 0 else p2, "x": x, "y": y}
        for i, (x, y) in enumerate(moves)
    ]
    return {
        "gameOptions": {"rated": rated},
        "moveCount": move_count,
        "gameResult": {"reason": reason, "anon_winner": winner_id},
        "players": [
            {"anon_profile_id": p1, "elo": 1100},
            {"anon_profile_id": p2, "elo": 1200},
        ],
        "moves": moves_data,
    }


def _write(d: Path, name: str, obj: dict) -> None:
    (d / f"{name}.json").write_text(json.dumps(obj))


def test_export_winner_mapping_and_schema(tmp_path: Path):
    src = tmp_path / "raw"
    src.mkdir()
    # game A: first mover (p1) wins -> winner 1
    _write(src, "aaa", _human_game("p1", "p2", winner_id="p1"))
    # game B: second mover (q2) wins -> winner -1 (distinct moves -> distinct hash)
    _write(src, "bbb", _human_game("q1", "q2", winner_id="q2", moves=[(0, 0), (3, -1), (1, 1)]))

    out = tmp_path / "out"
    meta = exporter.export([src], out, include_meta=True)

    assert meta["n_games"] == 2
    assert meta["encoding"].startswith("none")
    lines = (out / "hexo_human_corpus.jsonl").read_text().splitlines()
    assert len(lines) == 2

    recs = [json.loads(l) for l in lines]
    by_winner = {r["winner"]: r for r in recs}
    assert set(by_winner) == {1, -1}
    a = by_winner[1]
    assert "game_id" not in a  # replaced by game_hash
    assert a["moves"][0] == [0, 0]  # forced opener
    assert a["moves"] == [[0, 0], [2, -1], [1, 0]]
    assert a["source"] == "human"
    assert a["elo"] == [1100, 1200]
    # game_hash matches the canonical scheme, 16 hex chars
    assert a["game_hash"] == exporter.game_hash([(0, 0), (2, -1), (1, 0)])
    assert len(a["game_hash"]) == 16
    # sidecar docs written
    assert (out / "dataset_metadata.json").exists()
    assert (out / "SCHEMA.md").exists()
    assert (out / "README.md").exists()
    # sha256 in metadata matches the file
    assert meta["sha256"] == exporter._sha256_of(out / "hexo_human_corpus.jsonl")


def test_ingestion_filter_drops_invalid(tmp_path: Path):
    src = tmp_path / "raw"
    src.mkdir()
    _write(src, "ok", _human_game("p1", "p2", winner_id="p1"))
    _write(src, "unrated", _human_game("p1", "p2", winner_id="p1", rated=False))
    _write(src, "short", _human_game("p1", "p2", winner_id="p1", move_count=10))
    _write(src, "timeout", _human_game("p1", "p2", winner_id="p1", reason="timeout"))

    out = tmp_path / "out"
    meta = exporter.export([src], out, include_meta=True)
    assert meta["n_games"] == 1  # only the valid game survives


def test_no_meta_drops_elo(tmp_path: Path):
    src = tmp_path / "raw"
    src.mkdir()
    _write(src, "aaa", _human_game("p1", "p2", winner_id="p1"))
    out = tmp_path / "out"
    exporter.export([src], out, include_meta=False)
    rec = json.loads((out / "hexo_human_corpus.jsonl").read_text().splitlines()[0])
    assert "elo" not in rec


def test_limit(tmp_path: Path):
    src = tmp_path / "raw"
    src.mkdir()
    for i in range(5):
        _write(src, f"g{i}", _human_game(f"a{i}", f"b{i}", winner_id=f"a{i}",
                                         moves=[(0, 0), (i + 1, -1), (1, 0)]))
    out = tmp_path / "out"
    meta = exporter.export([src], out, limit=3)
    assert meta["n_games"] == 3


def test_duplicate_games_dropped(tmp_path: Path):
    src = tmp_path / "raw"
    src.mkdir()
    # two files, different ids, identical move sequence -> one survives
    _write(src, "first", _human_game("p1", "p2", winner_id="p1"))
    _write(src, "dup", _human_game("p1", "p2", winner_id="p1"))
    out = tmp_path / "out"
    meta = exporter.export([src], out)
    assert meta["n_games"] == 1
    assert meta["n_duplicates_dropped"] == 1


def test_game_hash_matches_canonical():
    # byte-for-byte identical to generate_corpus._game_hash on the dict form
    from hexo_rl.bootstrap.generate_corpus import _game_hash

    moves = [(0, 0), (2, -1), (1, 0)]
    canonical = _game_hash([{"x": x, "y": y} for (x, y) in moves])
    assert exporter.game_hash(moves) == canonical


def test_multiple_input_dirs(tmp_path: Path):
    d1, d2 = tmp_path / "d1", tmp_path / "d2"
    d1.mkdir()
    d2.mkdir()
    _write(d1, "a", _human_game("p1", "p2", winner_id="p1"))
    _write(d2, "b", _human_game("q1", "q2", winner_id="q1", moves=[(0, 0), (4, -2), (2, 1)]))
    out = tmp_path / "out"
    meta = exporter.export([d1, d2], out)
    assert meta["n_games"] == 2
