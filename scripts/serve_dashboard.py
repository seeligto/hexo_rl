#!/usr/bin/env python3
"""Serve HeXO web dashboard without active training process.

Usage:
    python scripts/serve_dashboard.py [options]

Options:
    --run-dir PATH        Root of runs/ tree (default: runs/)
    --checkpoint-dir PATH Checkpoint dir for analyze/play (default: checkpoints/)
    --port INT            Web server port (default: 5001)
    --host STR            Bind host (default: 127.0.0.1)
    --async-mode STR      SocketIO async mode: gevent|threading (default: gevent)
"""
from __future__ import annotations

# gevent.monkey.patch_all() MUST run before any stdlib network/threading imports
# so flask-socketio picks up the greened versions. Done unconditionally here
# because this script is the dashboard entry point — no PyTorch/Rust ext code
# runs in this process, so monkey-patching is safe.
import sys as _sys

if "gevent" not in _sys.modules:
    try:
        from gevent import monkey as _gevent_monkey
        _gevent_monkey.patch_all()
    except ImportError:
        # Falls back to threading mode below; user gets a clear message.
        pass

import argparse
import sys
import threading
from pathlib import Path

# Ensure project root is importable when invoked directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Serve HeXO web dashboard without active training process."
    )
    p.add_argument("--run-dir", default="runs/", help="Root of runs/ tree (default: runs/)")
    p.add_argument(
        "--checkpoint-dir",
        default="checkpoints/",
        help="Checkpoint dir for analyze/play (default: checkpoints/)",
    )
    p.add_argument("--port", type=int, default=5001, help="Web server port (default: 5001)")
    p.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    p.add_argument(
        "--async-mode",
        default="gevent",
        choices=["gevent", "threading"],
        help="SocketIO async server (default: gevent — production-grade, "
             "avoids the 'Session is disconnected' traceback storms that "
             "werkzeug's threading mode produces under backpressure).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    host = args.host
    port = args.port
    run_dir = args.run_dir
    checkpoint_dir = args.checkpoint_dir
    async_mode = args.async_mode

    if async_mode == "gevent" and "gevent" not in _sys.modules:
        print(
            "ERROR: --async-mode=gevent requested but gevent is not installed.\n"
            "       Run: .venv/bin/pip install gevent gevent-websocket\n"
            "       Or re-run with --async-mode=threading (not recommended for "
            "long-lived sessions).",
            file=sys.stderr,
        )
        sys.exit(2)

    # Minimal config — monitoring section only.
    # Does NOT load full configs/ yaml; uses safe defaults for all keys.
    config: dict = {
        "monitoring": {
            "web_port": port,
            "web_host": host,
            "viewer_games_dir": run_dir,
            "viewer_max_disk_games": 1000,
            "viewer_max_memory_games": 50,
            "event_log_maxlen": 500,
            "training_step_history": 2000,
            "emit_queue_maxsize": 200,
            "socketio_async_mode": async_mode,
        },
        "checkpoint_dir": checkpoint_dir,
    }

    from hexo_rl.monitoring.web_dashboard import WebDashboard
    from hexo_rl.monitoring.analyze_api import analyze_bp

    analyze_bp.checkpoint_dir = Path(checkpoint_dir)

    print(f"HeXO dashboard: http://{host}:{port}   (async_mode={async_mode})")
    print(f"  Viewer:      http://{host}:{port}/viewer")
    print(f"  Analyze:     http://{host}:{port}/analyze")
    print(f"  Run dir:     {run_dir}")
    print(f"  Checkpoints: {checkpoint_dir}")

    dashboard = WebDashboard(config)
    dashboard.start()

    # Block main thread; daemon threads keep Flask/SocketIO running.
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
