#!/usr/bin/env python3
"""Plot Ramora evaluation JSONL output from scripts/eval_vs_ramora.py.

Usage:
  .venv/bin/python scripts/plot_ramora_eval.py --latest
  .venv/bin/python scripts/plot_ramora_eval.py --eval-file logs/ramora_eval_20260330_024500.jsonl
  .venv/bin/python scripts/plot_ramora_eval.py --all --out plots/ramora_eval_trend.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime, timezone


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot Ramora evaluation trend")
    p.add_argument("--eval-file", default=None, help="Specific eval JSONL file")
    p.add_argument("--latest", action="store_true", help="Use latest ramora_eval_*.jsonl")
    p.add_argument("--all", action="store_true", help="Aggregate all ramora_eval_*.jsonl files")
    p.add_argument(
        "--train-log",
        default=None,
        help="Restrict eval points to this training log window (startup -> session_end)",
    )
    p.add_argument(
        "--latest-train-run",
        action="store_true",
        help="Restrict eval points to the latest training log window",
    )
    p.add_argument(
        "--min-games",
        type=int,
        default=0,
        help="Only include eval points with n_games >= this value",
    )
    p.add_argument("--out", default=None, help="Output PNG path")
    return p.parse_args()


def _discover_latest_train_log() -> Path:
    logs_dir = Path("logs")
    if not logs_dir.exists():
        raise FileNotFoundError("logs/ directory not found")

    candidates: list[tuple[datetime, Path]] = []
    for path in sorted(logs_dir.glob("*.jsonl")):
        try:
            with path.open() as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if rec.get("event") == "startup":
                        ts = _parse_ts(rec.get("timestamp"))
                        if ts is not None:
                            candidates.append((ts, path))
                        break
        except OSError:
            continue

    if not candidates:
        raise FileNotFoundError("No training logs with startup event found in logs/")
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def _read_training_window(train_log: Path) -> tuple[datetime, datetime | None]:
    startup_ts: datetime | None = None
    session_end_ts: datetime | None = None

    with train_log.open() as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            ts = _parse_ts(rec.get("timestamp"))

            event = rec.get("event")
            if event == "startup" and ts is not None and startup_ts is None:
                startup_ts = ts
            if event == "session_end" and ts is not None:
                session_end_ts = ts

    if startup_ts is None:
        raise ValueError(f"No startup event with timestamp found in training log: {train_log}")

    # If the run is still active or session_end wasn't logged, keep the window open-ended.
    if session_end_ts is None:
        return startup_ts, None
    return startup_ts, session_end_ts


def discover_files(args: argparse.Namespace) -> list[Path]:
    if args.eval_file:
        p = Path(args.eval_file)
        if not p.exists():
            raise FileNotFoundError(f"Eval file not found: {p}")
        return [p]

    logs_dir = Path("logs")
    if not logs_dir.exists():
        raise FileNotFoundError("logs/ directory not found")

    files = sorted(logs_dir.glob("ramora_eval_*.jsonl"))
    if not files:
        raise FileNotFoundError("No ramora_eval_*.jsonl files found in logs/")

    if args.all:
        return files

    if args.latest or not args.eval_file:
        return [files[-1]]

    return [files[-1]]


def load_records(files: list[Path]) -> list[dict]:
    records: list[dict] = []
    for path in files:
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("event") != "eval_vs_ramora":
                    continue
                rec["_source"] = str(path)
                records.append(rec)
    records.sort(key=lambda r: (int(r.get("step", -1)), str(r.get("timestamp", ""))))
    return records


def filter_records(
    records: list[dict],
    *,
    min_games: int,
    start_ts: datetime | None,
    end_ts: datetime | None,
) -> list[dict]:
    filtered: list[dict] = []
    for rec in records:
        if int(rec.get("n_games", 0)) < min_games:
            continue
        if start_ts is not None:
            ts = _parse_ts(rec.get("timestamp"))
            if ts is None:
                continue
            if ts < start_ts:
                continue
            if end_ts is not None and ts > end_ts:
                continue
        filtered.append(rec)
    return filtered


def main() -> None:
    args = parse_args()
    files = discover_files(args)
    records = load_records(files)

    train_log: Path | None = None
    start_ts: datetime | None = None
    end_ts: datetime | None = None
    if args.train_log and args.latest_train_run:
        raise ValueError("Use only one of --train-log or --latest-train-run")
    if args.train_log:
        train_log = Path(args.train_log)
        if not train_log.exists():
            raise FileNotFoundError(f"Training log not found: {train_log}")
        start_ts, end_ts = _read_training_window(train_log)
    elif args.latest_train_run:
        train_log = _discover_latest_train_log()
        start_ts, end_ts = _read_training_window(train_log)

    records = filter_records(
        records,
        min_games=max(0, int(args.min_games)),
        start_ts=start_ts,
        end_ts=end_ts,
    )
    if not records:
        print("No eval_vs_ramora events found after filters.")
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed. Install with: .venv/bin/pip install matplotlib")
        return

    steps = [int(r.get("step", -1)) for r in records]
    winrates = [float(r.get("winrate", 0.0)) for r in records]
    n_games = [int(r.get("n_games", 0)) for r in records]

    out_path = Path(args.out) if args.out else Path("plots") / "ramora_eval_trend.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(steps, winrates, marker="o", linewidth=1.5, label="Winrate vs Ramora")

    # Encode confidence intuition by marker size (more games => larger marker).
    sizes = [max(20, min(220, ng * 2)) for ng in n_games]
    ax.scatter(steps, winrates, s=sizes, alpha=0.35)

    for s, w, ng in zip(steps, winrates, n_games):
        ax.annotate(f"n={ng}", (s, w), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)

    ax.set_title("Model vs Ramora Winrate Trend")
    ax.set_xlabel("Checkpoint step")
    ax.set_ylabel("Winrate")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")

    subtitle = "latest" if len(files) == 1 else f"{len(files)} files"
    if train_log is not None:
        subtitle += f", run={train_log.name}"
    if args.min_games > 0:
        subtitle += f", min_games={args.min_games}"
    fig.suptitle(f"Ramora Eval ({subtitle})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    main()
