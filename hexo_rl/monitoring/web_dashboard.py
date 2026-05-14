"""Flask + SocketIO web dashboard for HeXO training monitoring.

Passive observer — consumes events from emit_event(), forwards them to
connected browsers via SocketIO. Never blocks the training loop.
"""

from __future__ import annotations

import collections
import glob
import json
import os
import queue
import secrets
import threading
from pathlib import Path
from typing import Any

import logging

import structlog
from flask import Flask
from flask_socketio import SocketIO

from hexo_rl.monitoring.config import MonitoringConfig
from hexo_rl.monitoring.web_routes import register_routes

log = structlog.get_logger()

# Suppress Flask/Werkzeug startup banners — they corrupt Rich Live's
# in-place terminal rendering by flushing stdout mid-escape-sequence.
logging.getLogger("werkzeug").setLevel(logging.ERROR)

# engineio/socketio raise `KeyError('Session is disconnected')` from
# `engineio.base_server._get_socket` when a browser tab closes mid-stream
# while a writer/handler thread is still emitting. The training loop is
# unaffected, but the unhandled traceback pollutes stdout/stderr and
# corrupts Rich Live rendering. We install a one-shot threading.excepthook
# that swallows that specific exception (KeyError from the engineio module)
# and delegates everything else to the previous handler.

_NOISY_ENGINEIO_MODULES: tuple[str, ...] = (
    "engineio.base_server",
    "engineio.server",
    "engineio.socket",
    "socketio.server",
    "socketio.manager",
)

_NOISY_KEYERROR_MESSAGES: tuple[str, ...] = (
    "Session is disconnected",
    "Session not found",
)


def _is_engineio_disconnect_noise(exc_type: type, exc_value: BaseException, tb) -> bool:
    """Return True if the exception is a known-benign engineio disconnect race."""
    if exc_type is not KeyError:
        return False
    msg = str(exc_value).strip("'\"")
    if not any(noise in msg for noise in _NOISY_KEYERROR_MESSAGES):
        return False
    # Walk the traceback looking for an engineio/socketio frame.
    cur = tb
    while cur is not None:
        module = cur.tb_frame.f_globals.get("__name__", "")
        if any(module.startswith(m) for m in _NOISY_ENGINEIO_MODULES):
            return True
        cur = cur.tb_next
    return False


_excepthook_installed = False


def _install_engineio_excepthook() -> None:
    """Install a threading.excepthook that drops engineio disconnect KeyErrors."""
    global _excepthook_installed
    if _excepthook_installed:
        return
    _excepthook_installed = True

    previous = threading.excepthook

    def _hook(args: threading.ExceptHookArgs) -> None:
        if _is_engineio_disconnect_noise(args.exc_type, args.exc_value, args.exc_traceback):
            return  # swallow silently — known benign race during tab close
        previous(args)

    threading.excepthook = _hook


class WebDashboard:
    """Web dashboard server that forwards events to browsers via SocketIO."""

    def __init__(self, config: dict) -> None:
        self._mon_cfg = MonitoringConfig.from_dict(config)
        self._port = int(self._mon_cfg.web_port)
        maxlen = int(self._mon_cfg.event_log_maxlen)
        training_step_maxlen = int(self._mon_cfg.training_step_history)
        self._config = config

        # Shared history for all non-training_step events (run_start, game_complete, etc.)
        self._event_history: collections.deque = collections.deque(maxlen=maxlen)
        # Dedicated history for training_step events — must not be evicted by game_complete flood.
        self._training_step_history: collections.deque = collections.deque(maxlen=training_step_maxlen)
        self._history_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._connected_sids: set = set()

        # Bounded send queue: background thread drains into socketio.emit().
        # Training loop puts with put_nowait() — if full, newest event is dropped.
        _queue_maxsize = int(self._mon_cfg.emit_queue_maxsize)
        self._emit_queue: queue.Queue = queue.Queue(maxsize=_queue_maxsize)
        self._drain_thread: threading.Thread | None = None

        # Game index — lightweight refs only; full records written to disk
        max_games = int(self._mon_cfg.viewer_max_memory_games)
        self._game_index: collections.deque = collections.deque(maxlen=max_games)
        self._max_disk_games = int(self._mon_cfg.viewer_max_disk_games)
        self._games_base_dir = Path(self._mon_cfg.viewer_games_dir)
        self._run_id: str = "default"

        self._host = str(self._mon_cfg.web_host)

        # async_mode: "threading" (werkzeug dev server, in-process default) or
        # "gevent" (production WSGI, used by scripts/serve_dashboard.py).
        # Gevent avoids the "Session is disconnected" KeyError storms that
        # werkzeug's threaded mode produces under backpressure.
        self._async_mode = str(self._mon_cfg.socketio_async_mode)

        # Viewer engine — lazy init at start()
        self._viewer_engine: Any = None

        # Build Flask app
        self._app = Flask(__name__, static_folder="static")
        self._app.config["SECRET_KEY"] = os.environ.get(
            "HEXO_DASHBOARD_SECRET_KEY", secrets.token_hex(32)
        )
        self._socketio = SocketIO(
            self._app, cors_allowed_origins="*", async_mode=self._async_mode
        )

        # Routes
        register_routes(self._app, self)

        # Analyze blueprint (policy viewer API)
        from hexo_rl.monitoring.analyze_api import analyze_bp
        self._app.register_blueprint(analyze_bp)

    def _safe_emit(self, event: str, data: dict) -> None:
        """Enqueue event for sending; never blocks, never propagates exceptions."""
        if not self._connected_sids:
            return
        try:
            self._emit_queue.put_nowait((event, data))
        except queue.Full:
            # Queue full — drop newest event (stale data is worse than gaps)
            pass

    def _drain_emit(self) -> None:
        """Background thread: drain _emit_queue into socketio.emit()."""
        while True:
            try:
                event, data = self._emit_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if self._connected_sids:
                try:
                    self._socketio.emit(event, data)
                except KeyError as exc:
                    # Known race: client tab closed mid-emit. Drop silently.
                    if not any(noise in str(exc) for noise in _NOISY_KEYERROR_MESSAGES):
                        log.warning("socketio_emit_keyerror", event=event, error=str(exc))
                except Exception as exc:
                    log.warning("socketio_emit_failed", event=event, error=str(exc))
            self._emit_queue.task_done()

    def _best_model_path(self) -> str | None:
        """Return best_model.pt path from config; None if file absent."""
        path_str = (
            self._config.get("eval_pipeline", {})
            .get("gating", {})
            .get("best_model_path", "checkpoints/best_model.pt")
        )
        p = Path(path_str)
        return str(p) if p.exists() else None

    def _init_viewer(self) -> None:
        """Initialize ViewerEngine with best_model.pt if available."""
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
            # Create engine without model so enrich_game still works
            try:
                from hexo_rl.viewer.engine import ViewerEngine
                self._viewer_engine = ViewerEngine(self._config)
            except Exception:
                self._viewer_engine = None

    def _load_existing_games(self, run_dir: Path) -> None:
        """Scan run_dir/*/games/*.json and pre-populate _game_index.

        Called once from start() before Flask begins serving.
        Caps to self._game_index.maxlen most-recent by mtime.
        """
        if not run_dir.exists():
            log.info("load_existing_games_skip", reason="run_dir_missing", path=str(run_dir))
            return

        matched = list(run_dir.glob("*/games/*.json"))
        if not matched:
            log.info("load_existing_games_skip", reason="no_files", path=str(run_dir))
            return

        max_games = self._game_index.maxlen or 50
        # Sort descending by mtime — most-recent first — then cap.
        try:
            matched.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        except OSError:
            pass
        matched = matched[:max_games]
        # Re-sort ascending so the deque ordering matches live-training (oldest→newest).
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
        """Start Flask server and drain thread as daemon threads.

        Raises OSError if (host, port) is already in use so the CLI can
        exit cleanly instead of leaving a half-alive process where the
        socketio thread crashed silently but the rest kept running.
        """
        _install_engineio_excepthook()
        self._init_viewer()
        self._load_existing_games(self._games_base_dir)

        port = self._port
        host = self._host
        async_mode = self._async_mode

        # Pre-flight bind probe — fail fast on EADDRINUSE before launching
        # the daemon thread (where exceptions would die silently).
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

        self._drain_thread = threading.Thread(
            target=self._drain_emit, daemon=True, name="socketio-drain"
        )
        self._drain_thread.start()

        socketio = self._socketio
        app = self._app

        def _run():
            # Suppress Flask/Werkzeug startup banner — it writes to stdout
            # via click.echo and corrupts Rich Live's in-place rendering.
            import flask.cli
            flask.cli.show_server_banner = lambda *a, **kw: None
            run_kwargs: dict = {
                "host": host,
                "port": port,
                "log_output": False,
                "use_reloader": False,
            }
            if async_mode == "threading":
                # Werkzeug dev server needs this opt-in flag; gevent/eventlet
                # provide their own WSGI servers and reject it.
                run_kwargs["allow_unsafe_werkzeug"] = True
            socketio.run(app, **run_kwargs)

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
        # Rotate: delete oldest files if disk cap exceeded (checked after write)
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
            if event_name == "training_step":
                self._training_step_history.append(payload)
            else:
                self._event_history.append(payload)

        # Reload viewer model on successful eval gate pass
        if event_name == "eval_complete" and payload.get("anchor_promoted"):
            try:
                ckpt = self._best_model_path()
                if ckpt and self._viewer_engine is not None:
                    from hexo_rl.viewer.engine import ViewerEngine
                    self._viewer_engine = ViewerEngine(
                        self._config, checkpoint_path=ckpt
                    )
                    log.info("viewer_engine_reloaded", checkpoint=ckpt)
            except Exception as exc:
                log.warning("viewer_engine_reload_failed", error=str(exc))

        self._safe_emit(event_name, payload)
