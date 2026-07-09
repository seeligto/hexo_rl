"""T-WORKERS: 4-pair result byte-identical at workers=1 (twice) and workers=1 vs workers=4."""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import pytest

PYTHON = ".venv/bin/python"
CKPT = "scripts/arena/weights/run2_175k.pt"
REPO = "/home/timmy/Work/Hexo/hexo_rl"


def _run_eval(out_dir: str, workers: int, n_pairs: int = 4) -> tuple[str, str]:
    """Run run_eval.py for n_pairs and return (games_jsonl_content, result_json_content)."""
    cmd = [
        PYTHON, "-m", "scripts.evalfair.run_eval",
        "--ckpt", CKPT,
        "--out", out_dir,
        "--n-pairs", str(n_pairs),
        "--workers", str(workers),
        "--expect-encoding", "v6_live2_ls",
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=REPO, timeout=300
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"run_eval failed (workers={workers}):\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    games = (Path(out_dir) / "games.jsonl").read_text()
    res = (Path(out_dir) / "result.json").read_text()
    return games, res


_TIMING_KEYS = frozenset([
    "head_move_wall_s", "sealbot_search_wall_s", "sealbot_search_max_s",
])


def _normalize_games(games_text: str) -> str:
    """Sort games.jsonl by (arm, opening_idx, head_as_p1), strip timing for determinism comparison."""
    records = [json.loads(l) for l in games_text.splitlines() if l.strip()]
    records.sort(key=lambda g: (g.get("arm",""), g.get("opening_idx", 0), g.get("head_as_p1", True)))
    clean = [{k: v for k, v in r.items() if k not in _TIMING_KEYS} for r in records]
    return "\n".join(json.dumps(r, sort_keys=True) for r in clean)


def _normalize_result(res_text: str) -> dict:
    """Parse result.json, drop wall_sec (timing varies), return comparable dict."""
    r = json.loads(res_text)
    r.pop("wall_sec", None)
    r.pop("wall_per_move_head_s", None)
    r.pop("wall_per_move_sealbot_s", None)
    r.pop("sealbot_search_max_s", None)
    return r


@pytest.mark.slow
def test_workers_1_twice_byte_identical(tmp_path):
    """workers=1 run twice produces byte-identical games.jsonl (sorted) and result.json (minus timing)."""
    out1 = str(tmp_path / "run1")
    out2 = str(tmp_path / "run2")
    Path(out1).mkdir()
    Path(out2).mkdir()

    g1, r1 = _run_eval(out1, workers=1, n_pairs=4)
    g2, r2 = _run_eval(out2, workers=1, n_pairs=4)

    assert _normalize_games(g1) == _normalize_games(g2), "workers=1 games.jsonl not identical across two runs"
    assert _normalize_result(r1) == _normalize_result(r2), "workers=1 result.json not identical across two runs"


@pytest.mark.slow
def test_workers_1_vs_4_byte_identical(tmp_path):
    """workers=1 and workers=4 produce byte-identical games.jsonl (sorted) and result.json (minus timing)."""
    out1 = str(tmp_path / "w1")
    out4 = str(tmp_path / "w4")
    Path(out1).mkdir()
    Path(out4).mkdir()

    g1, r1 = _run_eval(out1, workers=1, n_pairs=4)
    g4, r4 = _run_eval(out4, workers=4, n_pairs=4)

    assert _normalize_games(g1) == _normalize_games(g4), "workers=1 vs workers=4 games.jsonl differ"
    assert _normalize_result(r1) == _normalize_result(r4), "workers=1 vs workers=4 result.json differ"
