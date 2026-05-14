"""Flask + SocketIO route handlers for WebDashboard.

Extracted from web_dashboard.py (§176 P50). Route handler bodies copied
verbatim — response codes, URL paths, and JSON shapes preserved.

``register_routes(app, dashboard)`` wires every HTTP + SocketIO endpoint
against the supplied ``WebDashboard`` instance (captured via closure).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import structlog
from flask import Flask, jsonify, request, send_from_directory
from flask_socketio import emit

from hexo_rl.encoding import resolve_from_config

if TYPE_CHECKING:
    from hexo_rl.monitoring.web_dashboard import WebDashboard

log = structlog.get_logger()


def register_routes(app: Flask, dashboard: "WebDashboard") -> None:
    """Register every HTTP + SocketIO endpoint on ``app`` against ``dashboard``."""
    socketio = dashboard._socketio

    @app.route("/")
    def index():
        return send_from_directory("static", "index.html")

    @app.route("/api/monitoring-config")
    def monitoring_config():
        mon = dashboard._config.get("monitoring", dashboard._config)
        mc = dashboard._mon_cfg
        return jsonify({
            "training_step_history": int(mc.training_step_history),
            "game_history": int(mc.game_history),
            "num_actions_for_entropy_norm": int(mon.get("num_actions_for_entropy_norm", resolve_from_config(dashboard._config).policy_logit_count)),
            "alert_entropy_min": float(mc.alert_entropy_min),
            "alert_entropy_warn": float(mc.alert_entropy_warn),
            "collapse_threshold_nats": float(mc.collapse_threshold_nats),
            "alert_grad_norm_max": float(mc.alert_grad_norm_max),
            "ema_alpha": float(mc.ema_alpha),
            "p0_win_rate_target_low": float(mc.p0_win_rate_target_low),
            "p0_win_rate_target_high": float(mc.p0_win_rate_target_high),
        })

    @socketio.on("connect")
    def on_connect():
        dashboard._connected_sids.add(request.sid)
        with dashboard._history_lock:
            # Merge training_step history with other events, sorted by ts so
            # the client replays them in chronological order.
            history = sorted(
                list(dashboard._event_history) + list(dashboard._training_step_history),
                key=lambda e: e.get("ts", 0),
            )
        emit("replay_history", history)

    @socketio.on("disconnect")
    def on_disconnect():
        dashboard._connected_sids.discard(request.sid)

    # ── Analyze route ─────────────────────────────────────────────────

    @app.route("/analyze")
    def analyze_page():
        try:
            return send_from_directory("static", "analyze.html")
        except Exception as exc:
            return f"analyze.html not found: {exc}", 404

    # ── Viewer routes ─────────────────────────────────────────────────

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
            # Look up path from in-memory index first
            path_str: str | None = None
            with dashboard._history_lock:
                for ref in dashboard._game_index:
                    if ref["game_id"] == game_id:
                        path_str = ref["path"]
                        break

            # Fallback: search on disk for older games not in the index
            if path_str is None:
                candidates = list(
                    dashboard._games_base_dir.glob(f"*/games/{game_id}.json")
                )
                if candidates:
                    path_str = str(candidates[0])

            if path_str is None:
                return jsonify({"error": "game not found"}), 404

            try:
                record = json.loads(
                    Path(path_str).read_text(encoding="utf-8")
                )
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
