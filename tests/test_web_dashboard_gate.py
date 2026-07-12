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
    """gate_passed=True must reload viewer from best_model.pt, not a numbered checkpoint."""

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


class TestNoWebDashboardFlagGate:
    """--no-web-dashboard suppresses ONLY the Flask-SocketIO web dashboard,
    keeping the terminal dashboard. Prolonged-run hygiene: the web-socket
    teardown raised a benign exit-134 (SIGABRT) after the final checkpoint
    saved, masking the real exit code on the 200-300k run."""

    @staticmethod
    def _build(no_web_dashboard: bool, tmp_path):
        import argparse

        import torch

        from hexo_rl.training import lifecycle

        config = {
            "monitoring": {
                "enabled": True,
                "web_dashboard": True,
            }
        }
        args = argparse.Namespace(
            no_dashboard=False,
            no_web_dashboard=no_web_dashboard,
            checkpoint_dir=str(tmp_path),
            log_dir=str(tmp_path),
        )
        # Mock every side-effecting subsystem so only the dashboard gate runs.
        with patch.object(lifecycle, "GPUMonitor", MagicMock()), \
             patch.object(lifecycle, "DiskGuard", MagicMock()), \
             patch.object(lifecycle, "EarlyGameProbe", MagicMock()), \
             patch.object(lifecycle, "register_renderer", MagicMock()), \
             patch.object(lifecycle, "register_jsonl_sink", MagicMock()), \
             patch("hexo_rl.monitoring.metrics_writer.MetricsWriter", MagicMock()), \
             patch("hexo_rl.monitoring.web_dashboard.WebDashboard") as WD:
            lifecycle.build_subsystems(args, config, torch.device("cpu"), "run_test")
        return WD

    def test_web_dashboard_suppressed_when_flag_set(self, tmp_path):
        # D-J DASH WP3: terminal_dashboard (A2) retired — only the web gate remains.
        WD = self._build(no_web_dashboard=True, tmp_path=tmp_path)
        assert WD.call_count == 0, "--no-web-dashboard must suppress the web dashboard"

    def test_web_dashboard_built_when_flag_absent(self, tmp_path):
        WD = self._build(no_web_dashboard=False, tmp_path=tmp_path)
        assert WD.call_count == 1, "web dashboard built when flag not set (config web_dashboard: true)"


def test_train_argparser_exposes_no_web_dashboard():
    """The --no-web-dashboard flag exists and defaults False (web dashboard on)."""
    from scripts.train import build_argparser

    parser = build_argparser()  # full parser; peek_only=True omits run-time flags
    assert parser.parse_args([]).no_web_dashboard is False
    assert parser.parse_args(["--no-web-dashboard"]).no_web_dashboard is True
