#!/usr/bin/env python3
"""Analyze a Phase C1 diagnostic-run JSONL and print per-metric summaries.

Usage:
  scripts/perf/analyze_C1.py logs/diag_C1_main.jsonl [reports/perf/diag_C1_.../dmon.log]

Reads perf-probe events (train_step_timing, inference_batch_timing, vram_probe,
cuda_stream_audit, buffer_sample_timing), reports p50/p95/mean/stdev for each
time-valued field. Optional second arg parses dmon.log for GPU util/mem/clock
summary.

No external deps. Outputs to stdout.
"""
from __future__ import annotations

import json
import math
import statistics as S
import sys
from collections import defaultdict, Counter
from pathlib import Path


def pct(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    return xs[int(p * (len(xs) - 1))]


def summarize(xs, fmt="{:.1f}"):
    if not xs:
        return "n=0"
    return (f"n={len(xs):>5d}  p50=" + fmt.format(pct(xs, 0.5))
            + "  p95=" + fmt.format(pct(xs, 0.95))
            + "  mean=" + fmt.format(S.mean(xs))
            + "  stdev=" + fmt.format(S.pstdev(xs) if len(xs) > 1 else 0.0))


def load_events(path: Path) -> list[dict]:
    events: list[dict] = []
    with path.open() as f:
        for line in f:
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def analyze(events: list[dict]) -> None:
    by_event: defaultdict[str, list[dict]] = defaultdict(list)
    for e in events:
        by_event[e.get("event", "?")].append(e)

    print("\n=== Event counts ===")
    for ev, xs in sorted(by_event.items(), key=lambda kv: -len(kv[1]))[:30]:
        print(f"  {len(xs):>6d}  {ev}")

    # ── CUDA stream audit ────────────────────────────────────────────────────
    print("\n=== CUDA stream audit (B4) ===")
    for e in by_event.get("cuda_stream_audit", []):
        print(f"  {e.get('context')}: current_ptr={e.get('current_stream_ptr')} "
              f"default_ptr={e.get('default_stream_ptr')} "
              f"on_default={e.get('on_default_stream')}")
    if len({e.get("current_stream_ptr") for e in by_event.get("cuda_stream_audit", [])}) == 1:
        print("  → VERDICT: both contexts share the same CUDA stream — no copy/compute overlap possible.")

    # ── InferenceServer batch timing (B2) ────────────────────────────────────
    ib = by_event.get("inference_batch_timing", [])
    print(f"\n=== InferenceServer batch timing (B2, n={len(ib)}) ===")
    if ib:
        # Skip first N batches as cudnn autotune warmup.
        warmup = min(50, len(ib) // 10)
        ib_steady = ib[warmup:]
        print(f"  (skipping first {warmup} as cudnn autotune warmup; steady n={len(ib_steady)})")
        for field, fmt in [("batch_n", "{:.1f}"),
                            ("fetch_wait_us", "{:.0f}"),
                            ("h2d_us", "{:.0f}"),
                            ("forward_us", "{:.0f}"),
                            ("d2h_scatter_us", "{:.0f}")]:
            xs = [e[field] for e in ib_steady if field in e]
            print(f"  {field:>18s}: {summarize(xs, fmt)}")
        # batch_n distribution
        bn_counter = Counter(e.get("batch_n", 0) for e in ib_steady)
        top = bn_counter.most_common(5)
        total = sum(bn_counter.values())
        print(f"  batch_n top-5 (of {len(bn_counter)} distinct):")
        for bn, c in top:
            print(f"    batch_n={bn:>3d}: n={c:>6d}  ({100*c/total:.1f}%)")

    # ── Trainer step timing (B1) ─────────────────────────────────────────────
    ts = by_event.get("train_step_timing", [])
    print(f"\n=== Trainer step timing (B1, n={len(ts)}) ===")
    for field, fmt in [("h2d_us", "{:.0f}"),
                        ("fwd_loss_us", "{:.0f}"),
                        ("bwd_opt_us", "{:.0f}"),
                        ("total_us", "{:.0f}")]:
        xs = [e[field] for e in ts if field in e]
        print(f"  {field:>14s}: {summarize(xs, fmt)}")

    # ── Buffer sample timing ─────────────────────────────────────────────────
    bs = by_event.get("buffer_sample_timing", [])
    print(f"\n=== Buffer sample timing (n={len(bs)}) ===")
    if bs:
        xs = [e["sample_us"] for e in bs if "sample_us" in e]
        print(f"  sample_us: {summarize(xs, '{:.0f}')}")

    # ── VRAM probe (B1) ──────────────────────────────────────────────────────
    vr = by_event.get("vram_probe", [])
    print(f"\n=== VRAM probe (n={len(vr)}) ===")
    for field in ("vram_peak_gb", "vram_allocated_gb", "vram_reserved_gb", "vram_frag_gb"):
        xs = [e[field] for e in vr if field in e]
        if xs:
            print(f"  {field:>20s}: min={min(xs):.3f}  max={max(xs):.3f}  mean={S.mean(xs):.3f} GB")
    oom = max((e.get("num_ooms", 0) for e in vr), default=0)
    print(f"  num_ooms (max): {oom}")

    # ── Train step throughput ────────────────────────────────────────────────
    trs = by_event.get("train_step", [])
    if trs and len(trs) >= 2:
        try:
            from datetime import datetime
            t0 = datetime.fromisoformat(trs[0]["timestamp"].replace("Z", "+00:00"))
            t1 = datetime.fromisoformat(trs[-1]["timestamp"].replace("Z", "+00:00"))
            dur = (t1 - t0).total_seconds()
            steps = trs[-1]["step"] - trs[0]["step"]
            print(f"\n=== Train step throughput ===")
            print(f"  steps={steps}  wall={dur:.1f}s  {steps/max(dur,1e-9)*3600:.0f} steps/hr  {steps/max(dur,1e-9):.2f} steps/s")
        except Exception as exc:  # noqa: BLE001
            print(f"  throughput parse failed: {exc}")

    # ── game_complete: pos/hr ────────────────────────────────────────────────
    gc = by_event.get("game_complete", [])
    if gc and len(gc) >= 2:
        try:
            from datetime import datetime
            t0 = datetime.fromisoformat(gc[0]["timestamp"].replace("Z", "+00:00"))
            t1 = datetime.fromisoformat(gc[-1]["timestamp"].replace("Z", "+00:00"))
            dur = (t1 - t0).total_seconds()
            positions = sum(e.get("num_moves", 0) or e.get("moves", 0) or e.get("plies", 0)
                             for e in gc)
            print(f"\n=== Game production ===")
            print(f"  games={len(gc)}  wall={dur:.1f}s  {len(gc)/max(dur,1e-9)*3600:.0f} games/hr")
            if positions:
                print(f"  positions={positions}  {positions/max(dur,1e-9)*3600:.0f} pos/hr")
        except Exception as exc:  # noqa: BLE001
            print(f"  game throughput parse failed: {exc}")


def analyze_dmon(path: Path) -> None:
    if not path.exists():
        print(f"\n=== nvidia-smi dmon ===")
        print(f"  file {path} not found")
        return
    rows: list[dict] = []
    header_cols: list[str] | None = None
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            cols = line.lstrip("#").split()
            if header_cols is None and "gpu" in cols and ("sm" in cols or "pwr" in cols):
                header_cols = cols
            continue
        if header_cols is None:
            continue
        parts = line.split()
        if len(parts) != len(header_cols):
            continue
        row = {}
        for k, v in zip(header_cols, parts):
            try:
                row[k] = float(v)
            except ValueError:
                row[k] = v
        rows.append(row)
    print(f"\n=== nvidia-smi dmon (n_rows={len(rows)}, 1-sec cadence) ===")
    if not rows:
        print("  no rows parsed")
        return
    for key in ("sm", "mem", "pwr", "gtemp", "mtemp", "mclk", "gclk"):
        xs = [r.get(key) for r in rows if isinstance(r.get(key), (int, float))]
        if xs:
            print(f"  {key:>5s}: {summarize(xs, '{:.1f}')}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    events = load_events(Path(sys.argv[1]))
    analyze(events)
    if len(sys.argv) >= 3:
        analyze_dmon(Path(sys.argv[2]))
