"""E1 promotion-gate CUDA isolation — subprocess worker unit tests.

Unit-level (no GPU, no real training): the sidecar bridge round-trips an EvalRoundResult, and every
failure mode (absent / malformed / wrong-event / non-zero exit) → None so the parent routes to its
LOUD _eval_broken path. The full boundary-crossing smoke is test_promotion_gate_isolation_smoke.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from hexo_rl.eval.promotion_gate_worker import (
    RESULT_EVENT,
    read_result,
    run_promotion_gate_subprocess,
)


def test_read_result_roundtrips_eval_round_result(tmp_path):
    f = tmp_path / "r.jsonl"
    payload = {"event": RESULT_EVENT, "step": 250_000, "promoted": True, "eval_games": 4}
    f.write_text(json.dumps(payload) + "\n")
    got = read_result(str(f))
    assert got == {"step": 250_000, "promoted": True, "eval_games": 4}
    assert "event" not in got  # the event tag is stripped, the EvalRoundResult remains


def test_read_result_absent_returns_none(tmp_path):
    assert read_result(str(tmp_path / "missing.jsonl")) is None


def test_read_result_malformed_returns_none(tmp_path):
    f = tmp_path / "bad.jsonl"
    f.write_text("{not json\n")
    assert read_result(str(f)) is None


def test_read_result_wrong_event_returns_none(tmp_path):
    f = tmp_path / "w.jsonl"
    f.write_text(json.dumps({"event": "something_else", "promoted": True}) + "\n")
    assert read_result(str(f)) is None


def test_read_result_empty_file_returns_none(tmp_path):
    f = tmp_path / "empty.jsonl"
    f.write_text("")
    assert read_result(str(f)) is None


def test_subprocess_nonzero_exit_returns_none(tmp_path):
    # A worker that crashes (a bogus checkpoint path) → non-zero exit → None (broken eval).
    got = run_promotion_gate_subprocess(
        candidate_ckpt=str(tmp_path / "does_not_exist.pt"),
        best_ckpt=None,
        full_config={"eval_pipeline": {"enabled": True}},
        step=1, best_step=None, radius=None,
        work_dir=str(tmp_path / "work"),
        timeout_sec=120,
        python_exe=sys.executable,
    )
    assert got is None
    # A worker log was written for forensics (bridge is a file, never a pipe).
    logs = list((tmp_path / "work").glob("promotion_gate_worker_*.log"))
    assert logs, "worker log file not written"


def test_subprocess_result_bridge_is_a_file_not_stderr(tmp_path):
    # The config + result + log are all files in work_dir (the sidecar pattern) — no pipe the
    # parent blocks on. Verify the config file is written before the worker runs.
    work = tmp_path / "work"
    run_promotion_gate_subprocess(
        candidate_ckpt=str(tmp_path / "nope.pt"), best_ckpt=None,
        full_config={"eval_pipeline": {"enabled": True}, "marker": 123},
        step=7, best_step=None, radius=None, work_dir=str(work),
        timeout_sec=120, python_exe=sys.executable,
    )
    cfg = work / "eval_config_7.json"
    assert cfg.exists()
    assert json.loads(cfg.read_text())["marker"] == 123
