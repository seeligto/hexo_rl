"""Flask + SocketIO web dashboard for HeXO training monitoring.

Passive observer — consumes events from emit_event(), forwards them to
connected browsers via SocketIO. Never blocks the training loop.
"""

from __future__ import annotations

import collections
import glob
import json
import re
import threading
from pathlib import Path
from typing import Any

import logging

import structlog
from flask import Flask, jsonify, request, send_from_directory
from flask_socketio import SocketIO, emit

log = structlog.get_logger()

# Suppress Flask/Werkzeug startup banners — they corrupt Rich Live's
# in-place terminal rendering by flushing stdout mid-escape-sequence.
logging.getLogger("werkzeug").setLevel(logging.ERROR)


def _find_latest_checkpoint(config: dict) -> str | None:
    """Find the latest checkpoint_*.pt file in the checkpoint directory."""
    ckpt_dir = Path(config.get("checkpoint_dir", "checkpoints"))
    if not ckpt_dir.exists():
        return None
    pattern = re.compile(r"^checkpoint_(\d+)\.pt$")
    candidates: list[tuple[int, Path]] = []
    for p in ckpt_dir.iterdir():
        m = pattern.match(p.name)
        if m:
            candidates.append((int(m.group(1)), p))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return str(candidates[-1][1])


class WebDashboard:
    """Web dashboard server that forwards events to browsers via SocketIO."""

    def __init__(self, config: dict) -> None:
        mon = config.get("monitoring", config)
        self._port = int(mon.get("web_port", 5001))
        maxlen = int(mon.get("event_log_maxlen", 500))
        self._config = config

        self._event_history: collections.deque = collections.deque(maxlen=maxlen)
        self._history_lock = threading.Lock()
        self._thread: threading.Thread | None = None

        # Game index — lightweight refs only; full records written to disk
        max_games = int(mon.get("viewer_max_memory_games", 50))
        self._game_index: collections.deque = collections.deque(maxlen=max_games)
        self._games_base_dir = Path(mon.get("viewer_games_dir", "runs"))
        self._run_id: str = "default"

        # Viewer engine — lazy init at start()
        self._viewer_engine: Any = None

        # Build Flask app
        self._app = Flask(__name__, static_folder="static")
        self._app.config["SECRET_KEY"] = "hexo-training-dashboard"
        self._socketio = SocketIO(
            self._app, cors_allowed_origins="*", async_mode="threading"
        )

        # Routes
        self._register_routes()

    def _init_viewer(self) -> None:
        """Initialize ViewerEngine with latest checkpoint if available."""
        try:
            from hexo_rl.viewer.engine import ViewerEngine
            ckpt = _find_latest_checkpoint(self._config)
            self._viewer_engine = ViewerEngine(self._config, checkpoint_path=ckpt)
            if ckpt:
                log.info("viewer_engine_loaded", checkpoint=ckpt)
            else:
                log.info("viewer_engine_loaded", checkpoint="none")
        except Exception as exc:
            log.warning("viewer_engine_init_failed", error=str(exc))
            # Create engine without model so enrich_game still works
            try:
                from hexo_rl.viewer.engine import ViewerEngine
                self._viewer_engine = ViewerEngine(self._config)
            except Exception:
                self._viewer_engine = None

    def _register_routes(self) -> None:
        app = self._app
        socketio = self._socketio
        dashboard = self  # capture for closures

        @app.route("/")
        def index():
            return send_from_directory("static", "index.html")

        @socketio.on("connect")
        def on_connect():
            with dashboard._history_lock:
                history = list(dashboard._event_history)
            emit("replay_history", history)

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

    def start(self) -> None:
        """Start Flask server in a daemon thread."""
        self._init_viewer()

        port = self._port
        socketio = self._socketio
        app = self._app

        def _run():
            # Suppress Flask/Werkzeug startup banner — it writes to stdout
            # via click.echo and corrupts Rich Live's in-place rendering.
            import flask.cli
            flask.cli.show_server_banner = lambda *a, **kw: None
            socketio.run(
                app,
                host="127.0.0.1",
                port=port,
                allow_unsafe_werkzeug=True,
                log_output=False,
                use_reloader=False,
            )

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()
        log.info("web_dashboard", url=f"http://localhost:{port}")

    def stop(self) -> None:
        """Stop the server (best-effort — daemon thread will die with process)."""
        self._thread = None

    def _persist_game(self, payload: dict) -> None:
        """Write full game record to disk and add a lightweight ref to the index."""
        game_id = payload.get("game_id", "unknown")
        run_dir = self._games_base_dir / self._run_id / "games"
        path = run_dir / f"{game_id}.json"
        try:
            run_dir.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload), encoding="utf-8")
        except Exception as exc:
            log.warning("game_persist_failed", error=str(exc), game_id=game_id)
        ref = {
            "game_id": game_id,
            "path": str(path),
            "winner": payload.get("winner"),
            "moves": payload.get("moves"),
            "worker_id": payload.get("worker_id"),
            "ts": payload.get("ts"),
        }
        with self._history_lock:
            self._game_index.append(ref)

    def on_event(self, payload: dict) -> None:
        """Receive an event, store it, and forward to connected browsers."""
        event_name = payload.get("event", "unknown")

        if event_name == "run_start":
            self._run_id = payload.get("run_id", "default")

        if event_name == "game_complete":
            # Persist full record to disk; keep only lightweight ref in memory.
            self._persist_game(payload)
            # Store a stripped copy in event_history so SocketIO replay_history
            # doesn't carry heavy per-move data.
            _STRIP_KEYS = {"moves_list", "moves_detail", "value_trace"}
            payload = {k: v for k, v in payload.items() if k not in _STRIP_KEYS}

        with self._history_lock:
            self._event_history.append(payload)

        # Reload viewer model on successful eval gate pass
        if event_name == "eval_complete" and payload.get("gate_passed"):
            try:
                ckpt = _find_latest_checkpoint(self._config)
                if ckpt and self._viewer_engine is not None:
                    from hexo_rl.viewer.engine import ViewerEngine
                    self._viewer_engine = ViewerEngine(
                        self._config, checkpoint_path=ckpt
                    )
                    log.info("viewer_engine_reloaded", checkpoint=ckpt)
            except Exception as exc:
                log.warning("viewer_engine_reload_failed", error=str(exc))

        try:
            self._socketio.emit(event_name, payload)
        except Exception:
            pass  # Never propagate to training loop
