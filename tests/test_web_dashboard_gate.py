"""Regression tests for H-002: viewer reload must use best_model.pt, not latest numbered ckpt."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch


def _make_config(best_model_path: str) -> dict:
    return {
        "monitoring": {"web_port": 5099, "web_host": "127.0.0.1"},
        "eval_pipeline": {
            "gating": {"best_model_path": best_model_path},
        },
    }


def _make_dashboard(config: dict):
    """Construct WebDashboard without starting Flask/SocketIO threads."""
    from hexo_rl.monitoring.web_dashboard import WebDashboard

    with patch("flask_socketio.SocketIO.__init__", return_value=None), \
         patch("flask_socketio.SocketIO.on", return_value=lambda f: f):
        wd = WebDashboard.__new__(WebDashboard)
        WebDashboard.__init__(wd, config)
    return wd


class TestGatePassedReloadsFromBestModelPath:
    """gate_passed=True must reload viewer from best_model.pt, not _find_latest_checkpoint."""

    def test_gate_passed_reloads_from_best_model_path(self, tmp_path):
        best = tmp_path / "best_model.pt"
        best.write_bytes(b"")

        numbered = tmp_path / "checkpoint_9999.pt"
        numbered.write_bytes(b"")

        cfg = _make_config(str(best))

        wd = _make_dashboard(cfg)

        captured_paths = []

        class FakeViewerEngine:
            def __init__(self, config, checkpoint_path=None):
                captured_paths.append(checkpoint_path)

        wd._viewer_engine = object()  # non-None sentinel, avoids FakeViewerEngine __init__

        with patch("hexo_rl.viewer.engine.ViewerEngine", FakeViewerEngine):
            wd.on_event({"event": "eval_complete", "anchor_promoted": True})

        assert len(captured_paths) == 1
        assert captured_paths[0] == str(best), (
            f"Expected best_model.pt path {best!r}, got {captured_paths[0]!r}"
        )
        assert "9999" not in (captured_paths[0] or ""), (
            "Must not load numbered checkpoint on gate pass"
        )

    def test_gate_passed_no_reload_when_best_model_absent(self, tmp_path):
        cfg = _make_config(str(tmp_path / "best_model.pt"))  # file does not exist
        wd = _make_dashboard(cfg)

        captured_paths = []

        class FakeViewerEngine:
            def __init__(self, config, checkpoint_path=None):
                captured_paths.append(checkpoint_path)

        wd._viewer_engine = object()  # non-None sentinel

        with patch("hexo_rl.viewer.engine.ViewerEngine", FakeViewerEngine):
            wd.on_event({"event": "eval_complete", "anchor_promoted": True})

        assert captured_paths == [], "No reload when best_model.pt absent"


class TestColdStartViewerUsesBestModelPath:
    """_init_viewer must load best_model.pt, not latest numbered ckpt."""

    def test_cold_start_viewer_uses_best_model_path(self, tmp_path):
        best = tmp_path / "best_model.pt"
        best.write_bytes(b"")

        numbered = tmp_path / "checkpoint_5000.pt"
        numbered.write_bytes(b"")

        cfg = _make_config(str(best))
        wd = _make_dashboard(cfg)

        captured_paths = []

        class FakeViewerEngine:
            def __init__(self, config, checkpoint_path=None):
                captured_paths.append(checkpoint_path)

        with patch("hexo_rl.viewer.engine.ViewerEngine", FakeViewerEngine):
            wd._init_viewer()

        assert len(captured_paths) == 1
        assert captured_paths[0] == str(best), (
            f"Expected best_model.pt path {best!r}, got {captured_paths[0]!r}"
        )

    def test_cold_start_no_model_when_best_model_absent(self, tmp_path):
        cfg = _make_config(str(tmp_path / "best_model.pt"))
        wd = _make_dashboard(cfg)

        captured_paths = []

        class FakeViewerEngine:
            def __init__(self, config, checkpoint_path=None):
                captured_paths.append(checkpoint_path)

        with patch("hexo_rl.viewer.engine.ViewerEngine", FakeViewerEngine):
            wd._init_viewer()

        assert len(captured_paths) == 1
        assert captured_paths[0] is None, "checkpoint_path=None when best_model.pt absent"
