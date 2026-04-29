#!/usr/bin/env python3
"""Serve HeXO web dashboard without active training process.

Live events are picked up by tailing ``logs/events_*.jsonl`` (written by
hexo_rl.monitoring.events.JSONLSink in scripts/train.py). The newest file
matching the pattern is followed; if a newer run starts, the tailer
switches automatically.

Usage:
    python scripts/serve_dashboard.py [options]

Options:
    --run-dir PATH        Root of runs/ tree (default: runs/)
    --checkpoint-dir PATH Checkpoint dir for analyze/play (default: checkpoints/)
    --log-dir PATH        Where events_*.jsonl files live (default: logs/)
    --port INT            Web server port (default: 5001)
    --host STR            Bind host (default: 127.0.0.1)
    --async-mode STR      SocketIO async mode: gevent|threading (default: gevent)
    --no-tail             Disable JSONL tailer (replay-only mode)
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
    p.add_argument(
        "--log-dir",
        default="logs/",
        help="Directory containing events_*.jsonl event streams (default: logs/)",
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
    p.add_argument(
        "--no-tail",
        action="store_true",
        help="Disable events_*.jsonl tailer (replay/analyze only — no live training panel).",
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

    log_dir = args.log_dir
    enable_tail = not args.no_tail

    print(f"HeXO dashboard: http://{host}:{port}   (async_mode={async_mode})")
    print(f"  Viewer:      http://{host}:{port}/viewer")
    print(f"  Analyze:     http://{host}:{port}/analyze")
    print(f"  Run dir:     {run_dir}")
    print(f"  Checkpoints: {checkpoint_dir}")
    print(f"  Log dir:     {log_dir}   (tail={'on' if enable_tail else 'off'})")

    dashboard = WebDashboard(config)
    dashboard.start()

    tailer = None
    if enable_tail:
        from hexo_rl.monitoring.events_tail import EventsTailer

        tailer = EventsTailer(
            log_dir=log_dir,
            callback=dashboard.on_event,
        )
        tailer.start()

    # Block main thread; daemon threads keep Flask/SocketIO running.
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        if tailer is not None:
            tailer.stop()


if __name__ == "__main__":
    main()
