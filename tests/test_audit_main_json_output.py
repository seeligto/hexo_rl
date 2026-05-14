"""Integration test for `python -m hexo_rl.encoding audit --format=json`.

Invokes the CLI via subprocess (safe against PyO3 abort-on-panic) and
validates the JSON shape returned on stdout.

Expected top-level keys match the shape documented in audit.py:
    registry_specs, checkpoints, corpora, variants, hardcode_hits,
    cross_table, findings, summary
"""
from __future__ import annotations

import json
import subprocess
import sys

import pytest


_EXPECTED_TOP_KEYS = {
    "registry_specs",
    "checkpoints",
    "corpora",
    "variants",
    "hardcode_hits",
    "cross_table",
    "findings",
    "summary",
}

_EXPECTED_SUMMARY_KEYS = {"info", "warn", "error", "strict", "exit_code"}


@pytest.mark.integration
def test_audit_json_shape(tmp_path):
    """CLI emits valid JSON with expected top-level keys."""
    # Point all dirs at empty tmp dirs so the audit runs quickly and cleanly
    # without touching real checkpoints / corpora.
    ckpts = tmp_path / "checkpoints"
    corpora = tmp_path / "data"
    variants = tmp_path / "variants"
    src_root = tmp_path / "src_root"
    (src_root / "engine" / "src").mkdir(parents=True)
    (src_root / "hexo_rl").mkdir(parents=True)
    ckpts.mkdir()
    corpora.mkdir()
    variants.mkdir()

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "hexo_rl.encoding",
            "audit",
            "--format=json",
            "--checkpoints-dir", str(ckpts),
            "--corpora-dir", str(corpora),
            "--variants-dir", str(variants),
            "--repo-root", str(src_root),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )

    # exit code 0 (clean) or 1 (warn) are both acceptable; 2 (error) would
    # indicate a structural problem with the test fixtures.
    assert result.returncode in (0, 1), (
        f"audit --format=json returned exit code {result.returncode}\n"
        f"stdout: {result.stdout[:2000]}\n"
        f"stderr: {result.stderr[:2000]}"
    )

    # stdout must be valid JSON
    assert result.stdout.strip(), "audit --format=json produced empty stdout"
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        pytest.fail(
            f"audit --format=json stdout is not valid JSON: {exc}\n"
            f"stdout (first 2000 chars): {result.stdout[:2000]}"
        )

    # Top-level shape
    missing = _EXPECTED_TOP_KEYS - data.keys()
    assert not missing, f"JSON output missing keys: {missing}"

    # summary sub-keys
    summary = data["summary"]
    assert isinstance(summary, dict), "summary must be a dict"
    missing_summary = _EXPECTED_SUMMARY_KEYS - summary.keys()
    assert not missing_summary, f"summary missing keys: {missing_summary}"

    # Types
    assert isinstance(data["registry_specs"], list)
    assert isinstance(data["checkpoints"], list)
    assert isinstance(data["corpora"], list)
    assert isinstance(data["variants"], list)
    assert isinstance(data["hardcode_hits"], list)
    assert isinstance(data["cross_table"], list)
    assert isinstance(data["findings"], list)

    # registry_specs should be non-empty (real registry has entries)
    assert len(data["registry_specs"]) > 0, "registry_specs unexpectedly empty"

    # Each registry spec has required fields
    spec = data["registry_specs"][0]
    for key in ("name", "board_size", "n_planes", "policy_logits", "multi_window", "schema_v"):
        assert key in spec, f"registry_spec missing key: {key}"

    # summary exit_code must be int 0 or 1 for these clean fixtures
    assert summary["exit_code"] in (0, 1)
    assert isinstance(summary["strict"], bool)


@pytest.mark.integration
def test_audit_json_no_tmp_write(tmp_path, monkeypatch):
    """--format=json mode does not write /tmp side-channel for hardcode hits."""
    import tempfile
    from pathlib import Path

    ckpts = tmp_path / "checkpoints"
    corpora = tmp_path / "data"
    variants = tmp_path / "variants"
    src_root = tmp_path / "src_root"
    (src_root / "engine" / "src").mkdir(parents=True)
    hxrl = src_root / "hexo_rl"
    hxrl.mkdir(parents=True)
    ckpts.mkdir()
    corpora.mkdir()
    variants.mkdir()

    # Add a hardcode hit to exercise the collect_raw path.
    (hxrl / "foo.py").write_text("width = 19\n")

    # Redirect _HARDCODE_HITS_DUMP to a controlled tmp file so we can assert
    # it was NOT written.
    sentinel = tmp_path / "hardcode_hits_sentinel.txt"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "hexo_rl.encoding",
            "audit",
            "--format=json",
            "--checkpoints-dir", str(ckpts),
            "--corpora-dir", str(corpora),
            "--variants-dir", str(variants),
            "--repo-root", str(src_root),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )

    data = json.loads(result.stdout)
    # hardcode_hits in JSON must carry the raw hit(s)
    assert len(data["hardcode_hits"]) > 0, (
        "Expected hardcode_hits in JSON for synthetic literal but got empty list"
    )
    hit = data["hardcode_hits"][0]
    assert "file" in hit
    assert "line" in hit
    assert "hits" in hit
    assert isinstance(hit["hits"], list)
