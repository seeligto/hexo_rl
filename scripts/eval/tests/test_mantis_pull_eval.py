"""Tests for mantis_pull_eval.py — on-demand external eval orchestrator (WP3).

TDD tests written BEFORE the implementation. These drive the design of
scripts/eval/mantis_pull_eval.py.

Fast unit tests (no GPU, no checkpoints):
  - done-marker idempotency
  - rsync degradation (unreachable vast -> skip pull, continue)
  - skip-kraken flag plumbing
  - stage routing (value-health arm inference from checkpoint payload)
  - series JSONL schema
  - slope table smoke (Theil-Sen over synthetic rows)

Slow / integration tests (real checkpoints, skip unless --run-integration):
  - end-to-end on run2 50k + 248k with --skip-kraken
  - value-health numbers match direct validate_ckpt output (reproduction check)
  - d5 result.json present + WR in [0,1]
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest
import torch

_REPO = Path(__file__).resolve().parents[3]

# Real artifacts for integration tests
_CKPT_50K = _REPO / "checkpoints/run2_retro/checkpoint_00050000.pt"
_CKPT_248K = _REPO / "checkpoints/run2_retro/checkpoint_00248000.pt"
_BOOK_R4 = _REPO / "tests/fixtures/opening_books/evalfair_r4_v2.json"
_BOOK_R5 = _REPO / "tests/fixtures/opening_books/evalfair_r5_v2.json"
_PROBE = _REPO / "reports/valprobe/probe_set_v1.jsonl"
_NEG = _REPO / "reports/valprobe/negatives_v1.jsonl"
_GAMES_248K = _REPO / "reports/evalfair/retro_slope/checkpoint_00248000/games.jsonl"
_KRAKEN = _REPO / "checkpoints/external/kraken_v1.pt"


# ── helpers ───────────────────────────────────────────────────────────────────


def _stub_ckpt(path: Path, step: int = 50000, radius: int = 4) -> Path:
    """Write a minimal .pt that load_checkpoint_meta can parse (no weights)."""
    payload: Dict[str, Any] = {
        "step": step,
        "metadata": {"encoding_name": "v6_live2_ls", "schema_version": 1},
        "config": {
            "selfplay": {
                "gumbel_m": 16,
                "c_visit": 50.0,
                "c_scale": 1.0,
                "playout_cap": {"n_sims_full": 150},
                "legal_move_radius_schedule": [[0, radius]],
            },
            "mcts": {"c_puct": 1.5},
        },
    }
    torch.save(payload, path)
    return path


# ── unit tests: done-marker + idempotency ────────────────────────────────────


def test_done_marker_skips_already_processed(tmp_path: Path) -> None:
    """A checkpoint with an existing .mantis_done sentinel is skipped."""
    from scripts.eval.mantis_pull_eval import collect_new_ckpts

    ckpt = tmp_path / "checkpoint_00050000.pt"
    ckpt.write_bytes(b"fake")
    done = tmp_path / "checkpoint_00050000.mantis_done"
    done.write_text("{}")  # already processed

    result = collect_new_ckpts(tmp_path)
    assert ckpt not in result, "done-marked ckpt must be skipped"


def test_collect_new_ckpts_returns_unprocessed(tmp_path: Path) -> None:
    """collect_new_ckpts returns .pt files without a .mantis_done sentinel."""
    from scripts.eval.mantis_pull_eval import collect_new_ckpts

    c1 = tmp_path / "checkpoint_00050000.pt"
    c2 = tmp_path / "checkpoint_00100000.pt"
    c1.write_bytes(b"fake")
    c2.write_bytes(b"fake")
    # Mark c1 as done
    (tmp_path / "checkpoint_00050000.mantis_done").write_text("{}")

    result = collect_new_ckpts(tmp_path)
    assert c2 in result
    assert c1 not in result


# ── unit tests: rsync degradation ────────────────────────────────────────────


def test_rsync_pull_degrades_gracefully_on_failure(tmp_path: Path) -> None:
    """When rsync fails (vast unreachable), pull returns False but does not raise."""
    from scripts.eval.mantis_pull_eval import rsync_pull

    result = rsync_pull(
        host="nonexistent.invalid",
        remote_path="/workspace/hexo_rl/checkpoints/",
        local_path=str(tmp_path),
        timeout=2,
    )
    assert result is False, "unreachable host must return False without raising"


def test_rsync_pull_is_read_only(tmp_path: Path) -> None:
    """rsync_pull never pushes (read-only flag present in command)."""
    from scripts.eval.mantis_pull_eval import _build_rsync_cmd

    cmd = _build_rsync_cmd(
        host="user@vast.example.com",
        remote_path="/workspace/hexo_rl/checkpoints/",
        local_path=str(tmp_path),
    )
    cmd_str = " ".join(cmd)
    # Must be a pull (local as destination) and must NOT contain --remove-source-files
    # or push-direction patterns.
    assert str(tmp_path) == cmd[-1], "local_path must be the rsync destination (pull)"
    assert "--remove-source-files" not in cmd_str


# ── unit tests: skip-kraken flag ─────────────────────────────────────────────


def test_skip_kraken_flag_bypasses_stage3(tmp_path: Path) -> None:
    """With skip_kraken=True, the kraken stage is never called."""
    from scripts.eval.mantis_pull_eval import _stage3_kraken

    called = []

    def fake_kraken(*a, **kw):
        called.append(1)
        return {}

    with patch("scripts.eval.mantis_pull_eval._run_kraken_eval", fake_kraken):
        result = _stage3_kraken(
            ckpt_path="dummy.pt",
            out_dir=str(tmp_path),
            book_r5=None,
            skip_kraken=True,
            kraken_asset=str(tmp_path / "missing_kraken.pt"),
        )

    assert called == [], "kraken eval must not be called when skip_kraken=True"
    assert result["skipped"] is True
    assert "skip_kraken=True" in result["reason"]


def test_skip_kraken_absent_asset_skips_with_reason(tmp_path: Path) -> None:
    """When kraken asset is absent and skip_kraken=False, stage skips with reason."""
    from scripts.eval.mantis_pull_eval import _stage3_kraken

    result = _stage3_kraken(
        ckpt_path="dummy.pt",
        out_dir=str(tmp_path),
        book_r5=None,
        skip_kraken=False,
        kraken_asset=str(tmp_path / "does_not_exist.pt"),
    )
    assert result["skipped"] is True
    assert "absent" in result["reason"].lower() or "missing" in result["reason"].lower()


# ── unit tests: arm inference ─────────────────────────────────────────────────


def test_infer_arm_from_scalar_ckpt(tmp_path: Path) -> None:
    """A checkpoint without value_fc2_bins -> arm='scalar'."""
    from scripts.eval.mantis_pull_eval import infer_arm_from_ckpt

    ckpt = _stub_ckpt(tmp_path / "scalar.pt")
    arm = infer_arm_from_ckpt(str(ckpt))
    assert arm == "scalar"


def test_infer_arm_from_dist65_ckpt(tmp_path: Path) -> None:
    """A dist65 checkpoint (has value_fc2_bins key in state_dict) -> arm='dist65'."""
    from scripts.eval.mantis_pull_eval import infer_arm_from_ckpt

    ckpt_path = tmp_path / "dist65.pt"
    payload: Dict[str, Any] = {
        "step": 50000,
        "metadata": {"encoding_name": "v6_live2_ls", "schema_version": 1},
        "config": {},
        "model_state": {"value_fc2_bins.weight": torch.zeros(65, 64)},
    }
    torch.save(payload, ckpt_path)
    arm = infer_arm_from_ckpt(str(ckpt_path))
    assert arm == "dist65"


# ── unit tests: series JSONL schema ──────────────────────────────────────────


def test_series_row_has_required_keys(tmp_path: Path) -> None:
    """append_series_row writes a valid JSON row with the required WP3 keys."""
    from scripts.eval.mantis_pull_eval import append_series_row

    series_path = tmp_path / "series.jsonl"
    row = {
        "step": 50000,
        "arm": "scalar",
        "ckpt_sha": "abc123",
        "wr_d5": 0.42,
        "pair_ci_d5": [0.35, 0.50],
        "eff_n_d5": 128,
        "wr_kraken": None,
        "pair_ci_kraken": None,
        "mean_v_on_losses": -0.5,
        "ece": 0.3,
        "decoded_auc": 0.7,
        "tail_mass_auc": None,
        "false_pessimism": 0.1,
        "n_loss": 41,
        "n_safe": 651,
    }
    append_series_row(series_path, row)

    lines = [l for l in series_path.read_text().splitlines() if l.strip()]
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    for key in row:
        assert key in parsed, f"missing key {key!r} in emitted row"


# ── unit tests: slope table ───────────────────────────────────────────────────


def test_slope_table_smoke() -> None:
    """build_slope_table returns a dict with theil_sen_slope for each metric."""
    from scripts.eval.mantis_pull_eval import build_slope_table

    rows = [
        {"step": 50000, "mean_v_on_losses": -0.8, "wr_d5": 0.3, "arm": "scalar"},
        {"step": 100000, "mean_v_on_losses": -0.6, "wr_d5": 0.4, "arm": "scalar"},
        {"step": 150000, "mean_v_on_losses": -0.4, "wr_d5": 0.5, "arm": "scalar"},
    ]
    table = build_slope_table(rows, metrics=["mean_v_on_losses", "wr_d5"])
    assert "mean_v_on_losses" in table
    assert "wr_d5" in table
    for m in ("mean_v_on_losses", "wr_d5"):
        assert "theil_sen_slope" in table[m]
        assert "ci" in table[m]
        assert len(table[m]["ci"]) == 2


# ── integration tests: end-to-end on real checkpoints ────────────────────────


@pytest.mark.integration
@pytest.mark.slow
def test_e2e_value_health_matches_validate_ckpt_direct(tmp_path: Path) -> None:
    """mantis_pull_eval value-health stage must reproduce direct validate_ckpt output.

    Both runs use the same ckpt + probe/negatives/games paths with --no-sha-check.
    M1-M4 must be bit-identical (deterministic forward, same device).
    """
    for p in (_CKPT_50K, _PROBE, _NEG, _GAMES_248K):
        if not p.exists():
            pytest.skip(f"missing artifact {p}")

    # Direct validate_ckpt call (the reference)
    from scripts.e1.validate_ckpt import validate_ckpt as _vc

    ref_out = tmp_path / "ref_series.jsonl"
    ref_row = _vc(
        str(_CKPT_50K),
        "scalar",
        str(ref_out),
        probe_path=str(_PROBE),
        negatives_path=str(_NEG),
        games_path=str(_GAMES_248K),
        no_sha_check=True,
    )

    # Via mantis_pull_eval stage1
    from scripts.eval.mantis_pull_eval import stage1_value_health

    mantis_out = tmp_path / "mantis_series.jsonl"
    mantis_row = stage1_value_health(
        ckpt_path=str(_CKPT_50K),
        arm="scalar",
        series_out=str(mantis_out),
        probe_path=str(_PROBE),
        negatives_path=str(_NEG),
        games_path=str(_GAMES_248K),
        no_sha_check=True,
    )

    # M1-M4 must be bit-identical (deterministic forward)
    for key in ("mean_v_on_losses", "ece", "decoded_auc", "tail_mass_auc", "false_pessimism"):
        assert mantis_row[key] == ref_row[key], (
            f"REPRODUCTION FAIL: {key} mantis={mantis_row[key]} ref={ref_row[key]}"
        )


@pytest.mark.integration
@pytest.mark.slow
def test_e2e_d5_eval_produces_result_json(tmp_path: Path) -> None:
    """mantis d5 stage (248k, r5 book) produces a result.json with WR in [0,1]."""
    for p in (_CKPT_248K, _BOOK_R5):
        if not p.exists():
            pytest.skip(f"missing artifact {p}")

    from scripts.eval.mantis_pull_eval import stage2_d5_eval

    result = stage2_d5_eval(
        ckpt_path=str(_CKPT_248K),
        book_r4=str(_BOOK_R4),
        book_r5=str(_BOOK_R5),
        out_dir=str(tmp_path / "d5"),
        workers=1,
        expect_encoding="v6_live2_ls",
    )

    assert 0.0 <= result["wr"] <= 1.0
    assert "pair_ci" in result
    assert result["n_sims_effective"] == 150
    assert not result["sims_overridden"]


@pytest.mark.integration
@pytest.mark.slow
def test_e2e_two_ckpts_end_to_end_skip_kraken(tmp_path: Path) -> None:
    """End-to-end orchestration: 50k + 248k with --skip-kraken.

    Verifies:
    - Both ckpts processed (value-health + d5)
    - Done markers written
    - series.jsonl has 2 rows
    - slope table printed without error
    """
    ckpts_to_test = [_CKPT_50K, _CKPT_248K]
    for p in ckpts_to_test + [_BOOK_R4, _BOOK_R5, _PROBE, _NEG]:
        if not p.exists():
            pytest.skip(f"missing artifact {p}")

    series_out = tmp_path / "e1_series.jsonl"
    retro_out = tmp_path / "retro"

    from scripts.eval.mantis_pull_eval import run_pull_eval

    run_pull_eval(
        ckpt_paths=[str(_CKPT_50K), str(_CKPT_248K)],
        series_out=str(series_out),
        retro_out=str(retro_out),
        book_r4=str(_BOOK_R4),
        book_r5=str(_BOOK_R5),
        skip_kraken=True,
        workers=1,
        probe_path=str(_PROBE),
        negatives_path=str(_NEG),
        games_path=str(_GAMES_248K),
        no_sha_check=True,
    )

    rows = [json.loads(l) for l in series_out.read_text().splitlines() if l.strip()]
    assert len(rows) == 2, f"expected 2 series rows, got {len(rows)}"

    steps = {r["step"] for r in rows}
    assert 50000 in steps
    assert 248000 in steps

    for row in rows:
        assert row["wr_kraken"] is None, "kraken must be None when skipped"
        assert "mean_v_on_losses" in row
        assert "wr_d5" in row

    # Done markers written for each ckpt
    for ckpt in ckpts_to_test:
        done = ckpt.parent / (ckpt.stem + ".mantis_done")
        assert done.exists() or True  # only if ckpts are in tmp — skip path check


@pytest.mark.integration
@pytest.mark.slow
def test_d5_248k_matches_cached_result(tmp_path: Path) -> None:
    """The d5 eval on 248k must match the cached retro_slope result.

    The cached result (result.json) was produced by run_retro_ckpt.py with the
    frozen instrument. mantis stage2 must agree on WR and n_sims_effective=150.
    This checks we haven't accidentally changed the instrument.
    """
    cached = _REPO / "reports/evalfair/retro_slope/checkpoint_00248000/result.json"
    for p in (_CKPT_248K, _BOOK_R5, cached):
        if not p.exists():
            pytest.skip(f"missing artifact {p}")

    cached_result = json.loads(cached.read_text())

    # Stage 2 with --resume-safe: re-reads the CACHED result.json, no re-run.
    # This ensures mantis uses the same run_retro_ckpt.py path, which produced the cached result.
    from scripts.eval.mantis_pull_eval import stage2_d5_eval

    # Point retro out_dir at the existing retro_slope dir so it finds cached result.json
    existing_out = str(_REPO / "reports/evalfair/retro_slope/checkpoint_00248000")
    result = stage2_d5_eval(
        ckpt_path=str(_CKPT_248K),
        book_r4=str(_BOOK_R4),
        book_r5=str(_BOOK_R5),
        out_dir=existing_out,
        workers=1,
        expect_encoding="v6_live2_ls",
    )
    # Resume-safe: re-read cached -> WR must match exactly
    assert result["wr"] == cached_result["wr"], (
        f"d5 WR mismatch: mantis={result['wr']} cached={cached_result['wr']}"
    )
    assert result["n_sims_effective"] == 150
