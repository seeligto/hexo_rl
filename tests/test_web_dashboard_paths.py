"""Regression: /analyze path-traversal guard uses `is_relative_to`, not startswith.

Previously the guard did `abs_path.startswith(str(Path.cwd()))`, which is a
classic prefix-mimic bug — a sibling directory sharing the cwd prefix
(e.g. `<cwd>-evil/steal.pt`) passed the check. D-006.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from hexo_rl.monitoring.analyze_api import _get_model, analyze_bp


def test_path_traversal_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Stand up a dedicated checkpoints subtree plus a sibling that mimics its
    # prefix. The prefix-mimic is the actual bug (was bypassing guard); the
    # escaping path is the classic case.
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    inside = ckpt_dir / "fake.pt"
    inside.write_bytes(b"")

    evil_sibling = tmp_path / "checkpoints_evil"
    evil_sibling.mkdir()
    mimic = evil_sibling / "steal.pt"
    mimic.write_bytes(b"")

    # Real escape target.
    outside = tmp_path / "etc_passwd.txt"
    outside.write_bytes(b"")

    # Escape via symlink placed inside the allowed tree.
    escape_link = ckpt_dir / "escape.pt"
    escape_link.symlink_to(outside)

    monkeypatch.setattr(analyze_bp, "checkpoint_dir", ckpt_dir)

    # Inside — OK (guard passes; fails later on load, which is fine here).
    with pytest.raises(Exception) as ok_exc:
        _get_model(str(inside))
    assert "outside allowed root" not in str(ok_exc.value)

    # Prefix-mimic sibling — must reject.
    with pytest.raises(ValueError, match="outside allowed root"):
        _get_model(str(mimic))

    # `..`-style traversal — must reject.
    with pytest.raises(ValueError, match="outside allowed root"):
        _get_model(str(ckpt_dir / ".." / "etc_passwd.txt"))

    # Symlink escape — resolve() follows, is_relative_to() rejects.
    with pytest.raises(ValueError, match="outside allowed root"):
        _get_model(str(escape_link))
