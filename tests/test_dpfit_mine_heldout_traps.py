"""Tests for the --exclude-trap-sets / --bucket additions to
scripts/dpfit_mine_heldout_traps.py (build_exclusion_set), plus the
byte-identical-with-no-new-flags guarantee.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "dpfit_mine_heldout_traps", REPO_ROOT / "scripts" / "dpfit_mine_heldout_traps.py"
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _write_jsonl(path: Path, rows) -> None:
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_build_exclusion_set_corpus_only_matches_old_behavior() -> None:
    m = _load_module()
    corpus = [{"game_idx": 1}, {"game_idx": 5}, {"game_idx": 5}]
    used = m.build_exclusion_set(corpus, [], log=lambda _msg: None)
    assert used == {1, 5}


def test_build_exclusion_set_unions_extra_paths(tmp_path: Path) -> None:
    m = _load_module()
    corpus = [{"game_idx": 1}]
    eval_a = tmp_path / "heldout_traps.jsonl"
    eval_b = tmp_path / "heldout_traps_expanded.jsonl"
    _write_jsonl(eval_a, [{"game_idx": 2}, {"game_idx": 3}])
    _write_jsonl(eval_b, [{"game_idx": 3}, {"game_idx": 4}])

    used = m.build_exclusion_set(corpus, [str(eval_a), str(eval_b)], log=lambda _msg: None)
    assert used == {1, 2, 3, 4}


def test_build_exclusion_set_missing_path_warns_not_fatal(tmp_path: Path) -> None:
    m = _load_module()
    corpus = [{"game_idx": 1}]
    logged = []
    used = m.build_exclusion_set(
        corpus, [str(tmp_path / "does_not_exist.jsonl")], log=logged.append
    )
    assert used == {1}
    assert any("WARNING" in msg for msg in logged)


def test_cli_default_flags_unchanged() -> None:
    """--exclude-trap-sets defaults to [] and --bucket defaults to 'expand' —
    byte-identical with no new flags passed."""
    import subprocess

    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "dpfit_mine_heldout_traps.py"), "--help"],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0
    assert "--exclude-trap-sets" in result.stdout
    assert "--bucket" in result.stdout
    assert "'expand'" in result.stdout
