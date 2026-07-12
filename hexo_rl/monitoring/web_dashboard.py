"""Flask web dashboard for HeXO training monitoring.

D-J DASH WP3.2: SocketIO/gevent REMOVED. Static page + JSON polling.
Non-display duties preserved: game-persistence, game-index, viewer reload.
"""

from __future__ import annotations

import collections
import json
import os
import secrets
import threading
from pathlib import Path
from typing import Any

import logging

import structlog
from flask import Flask

from hexo_rl.monitoring.config import MonitoringConfig
from hexo_rl.monitoring.web_routes import register_routes

log = structlog.get_logger()

# Suppress Flask/Werkzeug startup banners
logging.getLogger("werkzeug").setLevel(logging.ERROR)


class WebDashboard:
    """Web dashboard server — static page + JSON-poll endpoints."""

    def __init__(self, config: dict) -> None:
        self._mon_cfg = MonitoringConfig.from_dict(config)
        self._port = int(self._mon_cfg.web_port)
        self._config = config

        # Shared history for non-training_step events
        maxlen = int(self._mon_cfg.event_log_maxlen)
        training_step_maxlen = int(self._mon_cfg.training_step_history)
        self._event_history: collections.deque = collections.deque(maxlen=maxlen)
        self._training_step_history: collections.deque = collections.deque(maxlen=training_step_maxlen)
        self._history_lock = threading.Lock()

        # Game index — lightweight refs only; full records written to disk
        max_games = int(self._mon_cfg.viewer_max_memory_games)
        self._game_index: collections.deque = collections.deque(maxlen=max_games)
        self._max_disk_games = int(self._mon_cfg.viewer_max_disk_games)
        self._games_base_dir = Path(self._mon_cfg.viewer_games_dir)
        self._run_id: str = "default"

        self._host = str(self._mon_cfg.web_host)
        self._thread = None

        # Viewer engine — lazy init at start()
        self._viewer_engine: Any = None

        # Build Flask app
        self._app = Flask(__name__, static_folder="static")
        self._app.config["SECRET_KEY"] = os.environ.get(
            "HEXO_DASHBOARD_SECRET_KEY", secrets.token_hex(32)
        )

        # Routes
        register_routes(self._app, self)

        # Analyze blueprint (policy viewer API)
        from hexo_rl.monitoring.analyze_api import analyze_bp
        self._app.register_blueprint(analyze_bp)

    def _best_model_path(self) -> str | None:
        path_str = (
            self._config.get("eval_pipeline", {})
            .get("gating", {})
            .get("best_model_path", "checkpoints/best_model.pt")
        )
        p = Path(path_str)
        return str(p) if p.exists() else None

    def _init_viewer(self) -> None:
        try:
            from hexo_rl.viewer.engine import ViewerEngine
            ckpt = self._best_model_path()
            self._viewer_engine = ViewerEngine(self._config, checkpoint_path=ckpt)
            if ckpt:
                log.info("viewer_engine_loaded", checkpoint=ckpt)
            else:
                log.info("viewer_engine_loaded", checkpoint="none")
        except Exception as exc:
            log.warning("viewer_engine_init_failed", error=str(exc))
            try:
                from hexo_rl.viewer.engine import ViewerEngine
                self._viewer_engine = ViewerEngine(self._config)
            except Exception:
                self._viewer_engine = None

    def _load_existing_games(self, run_dir: Path) -> None:
        """Scan run_dir/*/games/*.json and pre-populate _game_index."""
        if not run_dir.exists():
            log.info("load_existing_games_skip", reason="run_dir_missing", path=str(run_dir))
            return

        matched = list(run_dir.glob("*/games/*.json"))
        if not matched:
            log.info("load_existing_games_skip", reason="no_files", path=str(run_dir))
            return

        max_games = self._game_index.maxlen or 50
        try:
            matched.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        except OSError:
            pass
        matched = matched[:max_games]
        matched.reverse()

        loaded = 0
        for file_path in matched:
            try:
                with open(file_path, encoding="utf-8") as fh:
                    data = json.load(fh)
                game_id = data.get("game_id", file_path.stem)
                winner = data.get("winner")
                moves_raw = data.get("moves", 0)
                move_count = moves_raw if isinstance(moves_raw, int) else len(moves_raw) if hasattr(moves_raw, "__len__") else 0
                ts = data.get("ts") or file_path.stat().st_mtime
                ref = {
                    "game_id": game_id,
                    "path": str(file_path),
                    "winner": winner,
                    "moves": move_count,
                    "worker_id": data.get("worker_id"),
                    "ts": ts,
                }
                with self._history_lock:
                    self._game_index.append(ref)
                loaded += 1
            except (json.JSONDecodeError, OSError) as exc:
                log.warning("load_existing_game_failed", path=str(file_path), error=str(exc))

        log.info("load_existing_games_done", loaded=loaded, path=str(run_dir))

    def start(self) -> None:
        """Start Flask dev server as a daemon thread."""
        self._init_viewer()
        self._load_existing_games(self._games_base_dir)

        port = self._port
        host = self._host

        # Pre-flight bind probe — fail fast on EADDRINUSE
        import socket as _socket
        _probe = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
        try:
            _probe.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
            _probe.bind((host, port))
        except OSError as exc:
            _probe.close()
            raise OSError(
                f"dashboard cannot bind {host}:{port} ({exc.strerror or exc}). "
                "Another dashboard is likely already running — `make dashboard.stop` "
                "or set DASHBOARD_PORT=<other> for this one."
            ) from exc
        finally:
            _probe.close()

        app = self._app

        def _run():
            import flask.cli
            flask.cli.show_server_banner = lambda *a, **kw: None
            app.run(host=host, port=port, use_reloader=False, threaded=True)

        self._thread = threading.Thread(target=_run, daemon=True, name="flask-dashboard")
        self._thread.start()
        log.info("web_dashboard", url=f"http://localhost:{port}")

    def stop(self) -> None:
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
        try:
            all_files = sorted(run_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
            excess = len(all_files) - self._max_disk_games
            for old_path in all_files[:excess]:
                try:
                    old_path.unlink()
                except Exception as del_exc:
                    log.warning("game_disk_rotate_delete_failed", error=str(del_exc), path=str(old_path))
        except Exception as exc:
            log.warning("game_disk_rotate_failed", error=str(exc), game_id=game_id)
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
        """Receive an event from the JSONL tailer and store it."""
        event_name = payload.get("event", "unknown")

        if event_name == "run_start":
            self._run_id = payload.get("run_id", "default")

        if event_name == "game_complete":
            self._persist_game(payload)
            _STRIP_KEYS = {"moves_list", "moves_detail", "value_trace"}
            payload = {k: v for k, v in payload.items() if k not in _STRIP_KEYS}

        with self._history_lock:
            if event_name == "training_step":
                self._training_step_history.append(payload)
            else:
                self._event_history.append(payload)

        if event_name == "eval_complete" and payload.get("anchor_promoted"):
            try:
                ckpt = self._best_model_path()
                if ckpt and self._viewer_engine is not None:
                    from hexo_rl.viewer.engine import ViewerEngine
                    self._viewer_engine = ViewerEngine(self._config, checkpoint_path=ckpt)
                    log.info("viewer_engine_reloaded", checkpoint=ckpt)
            except Exception as exc:
                log.warning("viewer_engine_reload_failed", error=str(exc))
