"""Supply/demand preflight analyzer for §107 smoke JSONL logs.

Reads a smoke log at ``<run>.jsonl`` and emits the three metrics the
preflight decision rule needs:

  * trainer idle fraction — 5-second-throttled warmup / waiting_for_games
    log events are counted; each is attributed ~5 s of blocked time (cf.
    ``hexo_rl/training/loop.py`` throttle).  idle_frac = idle_sec /
    wall_sec, measured from the first train_step past step 100 to exclude
    model / corpus warm-up.

  * supply/demand ratio — positions produced (sum of plies from
    ``game_complete`` events in window) / positions consumed
    (train_step count × batch_size) over the post-warm-up window.
    1.0 = balanced; <1 = trainer over-samples the buffer.

  * policy loss slope — OLS slope on ``train_step.policy_loss`` over the
    first 1000 steps.

Usage:
    .venv/bin/python scripts/analyze_supply_demand_smoke.py \\
        logs/smoke_tsp_2_0.jsonl logs/smoke_tsp_1_5.jsonl
"""
from __future__ import annotations

import datetime as _dt
import json
import statistics
import sys
from pathlib import Path
from typing import Any


_WARMUP_IDLE_EVENTS = {"warmup", "waiting_for_games"}
_IDLE_EVENT_SECONDS = 5.0          # throttle interval in loop.py
_WARMUP_CUTOFF_STEP = 100           # exclude initial model/corpus warm-up
_SLOPE_WINDOW = 1000


def _load(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in path.read_text().splitlines():
        try:
            rows.append(json.loads(raw))
        except Exception:
            continue
    return rows


def _ts(r: dict[str, Any]) -> float:
    try:
        return _dt.datetime.fromisoformat(r["timestamp"].replace("Z", "+00:00")).timestamp()
    except Exception:
        return 0.0


def _slope(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    return num / den if den else float("nan")


def analyze(path: Path) -> dict[str, Any]:
    rows = _load(path)
    cfg = next((r["config"] for r in rows if "config" in r), {})
    tsp = float(cfg.get("training_steps_per_game", float("nan")))
    batch = int(cfg.get("batch_size", 256))

    train_rows = [r for r in rows if r.get("event") == "train_step" and "step" in r]
    idle_rows = [r for r in rows if r.get("event") in _WARMUP_IDLE_EVENTS]
    game_rows = [r for r in rows if r.get("event") == "game_complete"]

    # Post-warm-up window: from first train_step >= cutoff to end.
    post_train = [r for r in train_rows if int(r.get("step", 0)) >= _WARMUP_CUTOFF_STEP]
    if len(post_train) < 2:
        return dict(
            path=str(path), tsp=tsp, batch=batch,
            n_train_steps=len(train_rows), n_games=len(game_rows),
            idle_frac=float("nan"), supply_demand_ratio=float("nan"),
            policy_loss_slope=float("nan"),
            policy_loss_first=float("nan"), policy_loss_last=float("nan"),
            avg_plies_per_game=float("nan"),
            note="insufficient post-cutoff train_step rows",
        )

    t0 = _ts(post_train[0])
    t1 = _ts(post_train[-1])
    wall_sec = max(t1 - t0, 1e-6)
    post_idle = [r for r in idle_rows if _ts(r) >= t0]
    idle_sec = len(post_idle) * _IDLE_EVENT_SECONDS
    idle_frac = min(idle_sec / wall_sec, 1.0)

    post_games = [r for r in game_rows if _ts(r) >= t0]
    plies_produced = sum(int(r.get("plies", 0)) for r in post_games)
    positions_consumed = len(post_train) * batch
    ratio = (plies_produced / positions_consumed) if positions_consumed else float("nan")

    p_win = [r for r in train_rows if int(r.get("step", 0)) <= _SLOPE_WINDOW]
    xs = [float(r["step"]) for r in p_win]
    ys = [float(r["policy_loss"]) for r in p_win]
    slope = _slope(xs, ys)

    avg_plies = (statistics.mean(int(r["plies"]) for r in game_rows if r.get("plies"))
                 if game_rows else float("nan"))

    return dict(
        path=str(path), tsp=tsp, batch=batch,
        n_train_steps=len(train_rows), n_games=len(game_rows),
        wall_sec_post=round(wall_sec, 1), idle_sec_post=round(idle_sec, 1),
        idle_frac=round(idle_frac, 4),
        supply_demand_ratio=round(ratio, 4),
        policy_loss_slope=slope,
        policy_loss_first=ys[0] if ys else float("nan"),
        policy_loss_last=ys[-1] if ys else float("nan"),
        avg_plies_per_game=round(avg_plies, 2) if game_rows else float("nan"),
    )


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def main() -> None:
    paths = [Path(p) for p in sys.argv[1:]] or [
        Path("logs/smoke_tsp_2_0.jsonl"), Path("logs/smoke_tsp_1_5.jsonl")
    ]
    results = [analyze(p) for p in paths if p.exists()]
    if not results:
        print("no logs found", file=sys.stderr)
        sys.exit(2)
    keys = list(results[0].keys())
    # Print header
    col_widths = [max(len(k), max(len(_fmt(r.get(k, ""))) for r in results)) for k in keys]
    fmt_row = lambda vs: " | ".join(v.ljust(w) for v, w in zip(vs, col_widths))
    print(fmt_row(keys))
    print(fmt_row(["-" * w for w in col_widths]))
    for r in results:
        print(fmt_row([_fmt(r.get(k, "")) for k in keys]))


if __name__ == "__main__":
    main()
