"""Tests for the --from-jsonl source path of scripts/export_corpus_npz.py.

Covers the JSONL → record parsing in isolation (no engine replay). The full
JSONL → NPZ build (replay + plane slicing + shapes) is exercised end-to-end by
the /tmp probe install in the verification step.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "export_corpus_npz",
    Path(__file__).resolve().parent.parent / "scripts" / "export_corpus_npz.py",
)
assert _SPEC is not None and _SPEC.loader is not None
npz_exporter = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(npz_exporter)


def _line(moves, winner=1, elo=(1100, 1200)):
    return json.dumps({"game_hash": "x", "moves": moves, "winner": winner,
                       "source": "human", "elo": list(elo)})


def _write_jsonl(tmp_path: Path, lines) -> Path:
    p = tmp_path / "c.jsonl"
    p.write_text("\n".join(lines) + "\n")
    return p


def test_parses_valid_games(tmp_path: Path):
    moves = [[0, 0]] + [[i, -1] for i in range(1, 30)]  # 30 plies
    p = _write_jsonl(tmp_path, [_line(moves, winner=1), _line(moves, winner=-1)])
    recs = npz_exporter._scan_jsonl_games(p)
    assert len(recs) == 2
    r = recs[0]
    assert r["moves"][0] == (0, 0)
    assert all(isinstance(m, tuple) and len(m) == 2 for m in r["moves"])
    assert r["winner"] in (1, -1)
    assert r["game_len"] == 30
    assert r["elo_band"] == "1000_1200"  # avg(1100,1200)=1150


def test_drops_short_and_drawn(tmp_path: Path):
    long_moves = [[0, 0]] + [[i, -1] for i in range(1, 30)]
    short = [[0, 0], [1, -1], [2, -1]]  # < MIN_GAME_LENGTH (15)
    p = _write_jsonl(tmp_path, [
        _line(long_moves, winner=1),
        _line(short, winner=1),       # too short -> dropped
        _line(long_moves, winner=0),  # draw -> dropped
    ])
    recs = npz_exporter._scan_jsonl_games(p)
    assert len(recs) == 1


def test_elo_null_is_unrated(tmp_path: Path):
    moves = [[0, 0]] + [[i, -1] for i in range(1, 30)]
    p = _write_jsonl(tmp_path, [_line(moves, elo=(None, None))])
    recs = npz_exporter._scan_jsonl_games(p)
    assert recs[0]["elo_band"] == "unrated"


def test_pretrain_vs_prefill_weighting(tmp_path: Path):
    moves = [[0, 0]] + [[i, -1] for i in range(1, 30)]
    p = _write_jsonl(tmp_path, [_line(moves, elo=(1100, 1200))])  # band 1000_1200
    prefill = npz_exporter._scan_jsonl_games(p, pretrain=False)[0]["weight"]
    pretrain = npz_exporter._scan_jsonl_games(p, pretrain=True)[0]["weight"]
    # buffer-prefill band weight 0.7 vs pretrain 1.0 for 1000_1200
    assert prefill == npz_exporter.SOURCE_WEIGHTS["human"] * 0.7
    assert pretrain == npz_exporter.SOURCE_WEIGHTS["human"] * 1.0


def test_skips_blank_and_malformed_lines(tmp_path: Path):
    moves = [[0, 0]] + [[i, -1] for i in range(1, 30)]
    p = tmp_path / "c.jsonl"
    p.write_text("\n".join(["", _line(moves), "{not json", "   "]) + "\n")
    recs = npz_exporter._scan_jsonl_games(p)
    assert len(recs) == 1
