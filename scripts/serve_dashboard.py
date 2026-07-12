#!/usr/bin/env python3
"""Serve HeXO web dashboard without active training process.

D-J DASH WP3.2: SocketIO/gevent removed. Static page + JSON polling.
Live events picked up by tailing both JSONL channels:
  - logs/events_*.jsonl         (emit_event channel)
  - logs/<run_name>.jsonl       (structlog channel)

Usage:
    python scripts/serve_dashboard.py [options]

Options:
    --run-dir PATH        Root of runs/ tree (default: runs/)
    --checkpoint-dir PATH Checkpoint dir for analyze/play (default: checkpoints/)
    --log-dir PATH        Where JSONL logs live (default: logs/)
    --valprobe-dir PATH   Where valprobe JSONL outputs live (default: reports/valprobe)
    --evalfair-dir PATH   Where evalfair JSONL outputs live (default: reports/evalfair)
    --port INT            Web server port (default: 5001)
    --host STR            Bind host (default: 127.0.0.1)
    --no-tail             Disable JSONL tailer (replay-only mode)
"""
from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Serve HeXO web dashboard without active training process."
    )
    p.add_argument("--run-dir", default="runs/", help="Root of runs/ tree")
    p.add_argument("--checkpoint-dir", default="checkpoints/", help="Checkpoint dir")
    p.add_argument("--log-dir", default="logs/", help="JSONL log directory")
    p.add_argument("--valprobe-dir", default="reports/valprobe", help="valprobe JSONL dir")
    p.add_argument("--evalfair-dir", default="reports/evalfair", help="evalfair JSONL dir")
    p.add_argument("--port", type=int, default=5001)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--no-tail", action="store_true", help="Disable JSONL tailer")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    config: dict = {
        "monitoring": {
            "web_port": args.port,
            "web_host": args.host,
            "viewer_games_dir": args.run_dir,
            "viewer_max_disk_games": 1000,
            "viewer_max_memory_games": 50,
            "event_log_maxlen": 500,
            "training_step_history": 2000,
        },
        "checkpoint_dir": args.checkpoint_dir,
        "log_dir": args.log_dir,
        "valprobe_dir": args.valprobe_dir,
        "evalfair_dir": args.evalfair_dir,
    }

    from hexo_rl.monitoring.web_dashboard import WebDashboard
    from hexo_rl.monitoring.analyze_api import analyze_bp

    analyze_bp.checkpoint_dir = Path(args.checkpoint_dir)

    print(f"HeXO dashboard: http://{args.host}:{args.port}")
    print(f"  Viewer:      http://{args.host}:{args.port}/viewer")
    print(f"  Analyze:     http://{args.host}:{args.port}/analyze")
    print(f"  Run dir:     {args.run_dir}")
    print(f"  Log dir:     {args.log_dir}   (tail={'on' if not args.no_tail else 'off'})")

    dashboard = WebDashboard(config)
    try:
        dashboard.start()
    except OSError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)

    tailer = None
    if not args.no_tail:
        from hexo_rl.monitoring.events_tail import EventsTailer
        tailer = EventsTailer(
            log_dir=args.log_dir,
            callback=dashboard.on_event,
        )
        tailer.start()

    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        if tailer is not None:
            tailer.stop()


if __name__ == "__main__":
    main()
