"""Tests for checkpoint pruning policy: keep_last + anchor_every_steps + keep_all."""
from __future__ import annotations

from pathlib import Path

import pytest

from hexo_rl.training.checkpoints import prune_checkpoints


def _make_ckpts(tmp_path: Path, steps: list[int]) -> set[int]:
    """Write dummy checkpoint files; return the step set for assertion."""
    for s in steps:
        (tmp_path / f"checkpoint_{s:08d}.pt").write_bytes(b"x")
    return set(steps)


def _extant_steps(tmp_path: Path) -> set[int]:
    return {
        int(p.stem.split("_")[1])
        for p in tmp_path.glob("checkpoint_*.pt")
    }


def test_keep_all_skips_pruning(tmp_path: Path):
    """keep_all=True preserves everything regardless of max_kept."""
    _make_ckpts(tmp_path, list(range(500, 10500, 500)))  # 20 checkpoints
    prune_checkpoints(tmp_path, max_kept=5, keep_all=True)
    assert len(list(tmp_path.glob("checkpoint_*.pt"))) == 20


def test_keep_last_n_only(tmp_path: Path):
    """No anchors: only the N most recent survive."""
    steps = list(range(500, 6000, 500))  # 500,1000,...,5500 — 11 checkpoints
    _make_ckpts(tmp_path, steps)
    prune_checkpoints(tmp_path, max_kept=5)
    remaining = _extant_steps(tmp_path)
    assert remaining == {3500, 4000, 4500, 5000, 5500}


def test_anchor_every_steps_preserved(tmp_path: Path):
    """Steps at anchor_every_steps multiples survive beyond max_kept."""
    # Steps 500..10000 at 500-step intervals = 20 checkpoints.
    # anchor_every_steps=2000 → anchors at 2000,4000,6000,8000,10000 (5 anchors).
    # keep_last=3 on non-anchor steps.
    steps = list(range(500, 10500, 500))
    _make_ckpts(tmp_path, steps)
    prune_checkpoints(tmp_path, max_kept=3, anchor_every_steps=2000)
    remaining = _extant_steps(tmp_path)

    anchors = {s for s in steps if s % 2000 == 0}
    assert anchors.issubset(remaining), f"Anchors missing: {anchors - remaining}"

    # Rolling (non-anchor) survivors: last 3 of {500,1000,1500,2500,...,9500}
    non_anchor = sorted(s for s in steps if s % 2000 != 0)
    expected_rolling = set(non_anchor[-3:])
    assert expected_rolling.issubset(remaining), f"Rolling missing: {expected_rolling - remaining}"

    # Nothing extra survived.
    assert remaining == anchors | expected_rolling


def test_preserve_predicate_and_anchor_combined(tmp_path: Path):
    """preserve_predicate (eval steps) and anchor_every_steps are both respected."""
    steps = list(range(500, 8500, 500))  # 500..8000, 16 checkpoints
    _make_ckpts(tmp_path, steps)

    # eval_interval=4000 → preserve steps 4000, 8000
    # anchor_every_steps=3000 → preserve steps 3000, 6000
    def _eval_pred(s: int) -> bool:
        return s > 0 and s % 4000 == 0

    prune_checkpoints(
        tmp_path,
        max_kept=2,
        preserve_predicate=_eval_pred,
        anchor_every_steps=3000,
    )
    remaining = _extant_steps(tmp_path)

    # Preserved by predicate.
    assert 4000 in remaining
    assert 8000 in remaining
    # Preserved by anchor.
    assert 3000 in remaining
    assert 6000 in remaining

    # At most 2 rolling survivors (non-preserved steps).
    non_preserved = [s for s in steps if not _eval_pred(s) and s % 3000 != 0]
    rolling_survivors = [s for s in remaining if s not in {3000, 4000, 6000, 8000}]
    assert len(rolling_survivors) <= 2
    # Rolling survivors are the most recent.
    assert set(rolling_survivors) == set(sorted(non_preserved)[-2:])
