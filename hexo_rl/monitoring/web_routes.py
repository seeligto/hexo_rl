"""Flask route handlers for WebDashboard.

D-J DASH WP3.2: SocketIO routes REMOVED. New polling endpoints:
  GET /api/events.jsonl?since=<offset>    — emit_event JSONL tail
  GET /api/structlog.jsonl?since=<offset> — structlog JSONL tail
  GET /api/series/<name>                  — file-sourced series (value_health, external_bars)

Viewer/analyze routes PRESERVED (out-of-scope per WP3 operator ruling §8.2).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import structlog
from flask import Flask, jsonify, request, send_from_directory

from hexo_rl.encoding import resolve_from_config
from hexo_rl.monitoring.series_reader import (
    tail_jsonl,
    read_value_health_series,
    read_external_bars,
    compute_external_slope,
)

if TYPE_CHECKING:
    from hexo_rl.monitoring.web_dashboard import WebDashboard

log = structlog.get_logger()


def _find_structlog_jsonl(log_dir: Path, run_name: str | None = None) -> Path | None:
    """Return the structlog log file for this run.

    Priority:
    1. logs/<run_name>.jsonl if run_name supplied and file exists
    2. Most-recent *.jsonl in log_dir that is NOT events_*.jsonl
    """
    if run_name:
        p = log_dir / f"{run_name}.jsonl"
        if p.exists():
            return p

    candidates = [
        f for f in log_dir.glob("*.jsonl")
        if not f.name.startswith("events_")
    ]
    if not candidates:
        return None
    try:
        return max(candidates, key=lambda p: p.stat().st_mtime)
    except OSError:
        return None


def _find_events_jsonl(log_dir: Path) -> Path | None:
    """Return the most-recent events_*.jsonl in log_dir."""
    candidates = list(log_dir.glob("events_*.jsonl"))
    if not candidates:
        return None
    try:
        return max(candidates, key=lambda p: p.stat().st_mtime)
    except OSError:
        return None


def register_routes(app: Flask, dashboard: "WebDashboard") -> None:
    """Register every HTTP endpoint on app against dashboard."""

    # ── log dir resolution (resolved once at request time via config) ─────────

    def _log_dir() -> Path:
        cfg = dashboard._config
        ld = cfg.get("log_dir") or cfg.get("monitoring", {}).get("log_dir") or "logs"
        return Path(ld)

    def _valprobe_dir() -> Path:
        cfg = dashboard._config
        return Path(cfg.get("valprobe_dir") or cfg.get("monitoring", {}).get("valprobe_dir") or "reports/valprobe")

    def _evalfair_dir() -> Path:
        cfg = dashboard._config
        return Path(cfg.get("evalfair_dir") or cfg.get("monitoring", {}).get("evalfair_dir") or "reports/evalfair")

    # ── Static index ──────────────────────────────────────────────────────────

    @app.route("/")
    def index():
        return send_from_directory("static", "index.html")

    # ── API: monitoring config (kept for backwards compat) ────────────────────

    @app.route("/api/monitoring-config")
    def monitoring_config():
        mon = dashboard._config.get("monitoring", dashboard._config)
        mc = dashboard._mon_cfg
        return jsonify({
            "training_step_history": int(mc.training_step_history),
            "game_history": int(mc.game_history),
            "num_actions_for_entropy_norm": int(
                mon.get("num_actions_for_entropy_norm",
                        resolve_from_config(dashboard._config).policy_logit_count)
            ),
            "alert_entropy_min": float(mc.alert_entropy_min),
            "alert_entropy_warn": float(mc.alert_entropy_warn),
            "collapse_threshold_nats": float(mc.collapse_threshold_nats),
            "alert_grad_norm_max": float(mc.alert_grad_norm_max),
            "ema_alpha": float(mc.ema_alpha),
            "p0_win_rate_target_low": float(mc.p0_win_rate_target_low),
            "p0_win_rate_target_high": float(mc.p0_win_rate_target_high),
        })

    # ── API: dual-channel JSONL tails ────────────────────────────────────────

    @app.route("/api/events.jsonl")
    def api_events_jsonl():
        """Tail emit_event JSONL (logs/events_*.jsonl).

        ?since=<byte_offset>  (default 0 = first-load bounded tail)
        Returns: {"lines": [...], "next_offset": int, "truncated": bool}
        """
        since = int(request.args.get("since", 0))
        ld = _log_dir()
        p = _find_events_jsonl(ld)
        if p is None:
            return jsonify({"lines": [], "next_offset": 0, "truncated": False, "file": None})
        result = tail_jsonl(p, since_offset=since)
        result["file"] = str(p)
        return jsonify(result)

    @app.route("/api/structlog.jsonl")
    def api_structlog_jsonl():
        """Tail structlog JSONL (logs/<run_name>.jsonl).

        ?since=<byte_offset>  (default 0 = first-load bounded tail)
        Required for promotion, value_bce, fp16, forced_win, startup/config.
        Returns: {"lines": [...], "next_offset": int, "truncated": bool}
        """
        since = int(request.args.get("since", 0))
        ld = _log_dir()
        run_name = dashboard._run_id if dashboard._run_id != "default" else None
        p = _find_structlog_jsonl(ld, run_name)
        if p is None:
            return jsonify({"lines": [], "next_offset": 0, "truncated": False, "file": None})
        result = tail_jsonl(p, since_offset=since)
        result["file"] = str(p)
        return jsonify(result)

    @app.route("/api/series/<name>")
    def api_series(name: str):
        """File-sourced series (value_health, external_bars).

        value_health  → valprobe JSONL from _valprobe_dir()
        external_bars → evalfair JSONL from _evalfair_dir()

        Returns: {"records": [...], "slope": {...}}
        """
        if name == "value_health":
            vdir = _valprobe_dir()
            # Merge recognition_lag + ece outputs
            records: list[dict] = []
            for fname in ("recognition_lag.jsonl", "value_health.jsonl", "valprobe.jsonl"):
                p = vdir / fname
                if p.exists():
                    records.extend(read_value_health_series(p))
            if not records:
                # Fallback: any jsonl in valprobe dir
                for jf in sorted(vdir.glob("*.jsonl")):
                    records.extend(read_value_health_series(jf))
            return jsonify({"records": records})

        elif name == "external_bars":
            edir = _evalfair_dir()
            records = read_external_bars(edir)
            slope_info = compute_external_slope(records)
            return jsonify({"records": records, "slope": slope_info})

        else:
            return jsonify({"error": f"unknown series: {name}"}), 404

    # ── Analyze route (PRESERVED, out-of-scope) ───────────────────────────────

    @app.route("/analyze")
    def analyze_page():
        try:
            return send_from_directory("static", "analyze.html")
        except Exception as exc:
            return f"analyze.html not found: {exc}", 404

    # ── Viewer routes (PRESERVED, out-of-scope) ───────────────────────────────

    @app.route("/viewer")
    def viewer_page():
        try:
            return send_from_directory("static", "viewer.html")
        except Exception as exc:
            return f"viewer.html not found: {exc}", 404

    @app.route("/viewer/recent")
    def viewer_recent():
        try:
            n = min(int(request.args.get("n", 20)), 100)
            with dashboard._history_lock:
                refs = list(dashboard._game_index)
            return jsonify([
                {
                    "game_id": r["game_id"],
                    "winner": r["winner"],
                    "moves": r["moves"],
                    "ts": r["ts"],
                }
                for r in reversed(refs[-n:])
            ])
        except Exception as exc:
            log.warning("viewer_recent_error", error=str(exc))
            return jsonify([])

    @app.route("/viewer/game/<game_id>")
    def viewer_game(game_id: str):
        try:
            path_str: str | None = None
            with dashboard._history_lock:
                for ref in dashboard._game_index:
                    if ref["game_id"] == game_id:
                        path_str = ref["path"]
                        break

            if path_str is None:
                candidates = list(
                    dashboard._games_base_dir.glob(f"*/games/{game_id}.json")
                )
                if candidates:
                    path_str = str(candidates[0])

            if path_str is None:
                return jsonify({"error": "game not found"}), 404

            try:
                record = json.loads(Path(path_str).read_text(encoding="utf-8"))
            except Exception as exc:
                log.warning("game_load_failed", error=str(exc), game_id=game_id)
                return jsonify({"error": "game not found"}), 404

            if dashboard._viewer_engine is None:
                return jsonify(record)
            return jsonify(dashboard._viewer_engine.enrich_game(record))
        except Exception as exc:
            log.warning("viewer_game_error", error=str(exc), game_id=game_id)
            return jsonify({"error": str(exc)}), 500

    @app.route("/viewer/play", methods=["POST"])
    def viewer_play():
        if dashboard._viewer_engine is None or dashboard._viewer_engine._model_bot is None:
            return jsonify({"error": "no model loaded"}), 503
        try:
            data = request.get_json()
            result = dashboard._viewer_engine.play_response(
                data.get("moves_so_far", []),
                data.get("human_moves", []),
            )
            return jsonify(result)
        except Exception as exc:
            log.warning("viewer_play_error", error=str(exc))
            return jsonify({"error": str(exc)}), 500
