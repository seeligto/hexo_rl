"""Flask + SocketIO web dashboard for HeXO training monitoring.

Passive observer — consumes events from emit_event(), forwards them to
connected browsers via SocketIO. Never blocks the training loop.
"""

from __future__ import annotations

import collections
import threading
from typing import Any

import structlog
from flask import Flask, send_from_directory
from flask_socketio import SocketIO, emit

log = structlog.get_logger()


class WebDashboard:
    """Web dashboard server that forwards events to browsers via SocketIO."""

    def __init__(self, config: dict) -> None:
        mon = config.get("monitoring", config)
        self._port = int(mon.get("web_port", 5001))
        maxlen = int(mon.get("event_log_maxlen", 500))

        self._event_history: collections.deque = collections.deque(maxlen=maxlen)
        self._history_lock = threading.Lock()
        self._thread: threading.Thread | None = None

        # Build Flask app
        self._app = Flask(__name__, static_folder="static")
        self._app.config["SECRET_KEY"] = "hexo-training-dashboard"
        self._socketio = SocketIO(
            self._app, cors_allowed_origins="*", async_mode="threading"
        )

        # Routes
        self._register_routes()

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

    def start(self) -> None:
        """Start Flask server in a daemon thread."""
        port = self._port
        socketio = self._socketio
        app = self._app

        def _run():
            socketio.run(
                app,
                host="127.0.0.1",
                port=port,
                allow_unsafe_werkzeug=True,
                log_output=False,
            )

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()
        log.info("web_dashboard", url=f"http://localhost:{port}")

    def stop(self) -> None:
        """Stop the server (best-effort — daemon thread will die with process)."""
        self._thread = None

    def on_event(self, payload: dict) -> None:
        """Receive an event, store it, and forward to connected browsers."""
        with self._history_lock:
            self._event_history.append(payload)

        event_name = payload.get("event", "unknown")
        try:
            self._socketio.emit(event_name, payload)
        except Exception:
            pass  # Never propagate to training loop
