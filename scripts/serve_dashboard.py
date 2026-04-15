#!/usr/bin/env python3
"""Serve HeXO web dashboard without active training process.

Usage:
    python scripts/serve_dashboard.py [options]

Options:
    --run-dir PATH        Root of runs/ tree (default: runs/)
    --checkpoint-dir PATH Checkpoint dir for analyze/play (default: checkpoints/)
    --port INT            Web server port (default: 5001)
    --host STR            Bind host (default: 127.0.0.1)
"""
from __future__ import annotations

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
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    host = args.host
    port = args.port
    run_dir = args.run_dir
    checkpoint_dir = args.checkpoint_dir

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
        },
        "checkpoint_dir": checkpoint_dir,
    }

    from hexo_rl.monitoring.web_dashboard import WebDashboard
    from hexo_rl.monitoring.analyze_api import analyze_bp

    analyze_bp.checkpoint_dir = Path(checkpoint_dir)

    print(f"HeXO dashboard: http://{host}:{port}")
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
