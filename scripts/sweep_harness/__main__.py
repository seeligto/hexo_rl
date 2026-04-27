"""CLI entrypoint: ``python -m scripts.sweep_harness <subcommand>``.

Subcommands:
* ``detect`` — write ``reports/sweeps/detected_host.json`` (CPU/GPU/VRAM).
* ``run`` — execute the sweep against the knob registry.

Most users invoke this via ``scripts/sweep.sh`` (a 30-line shell wrapper).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from .knobs import KNOB_ORDER, KNOBS
from .reporting import write_report
from .runner import (
    DEFAULT_N_RUNS,
    DEFAULT_POOL_DURATION,
    DEFAULT_WARMUP,
    SweepConfig,
    detect_host,
    run_sweep,
)


def _parse_fix(items: list[str]) -> dict[str, object]:
    out: dict[str, object] = {}
    for it in items:
        if "=" not in it:
            raise SystemExit(f"--fix expects KEY=VALUE, got {it!r}")
        k, v = it.split("=", 1)
        if k not in KNOBS:
            raise SystemExit(f"--fix: unknown knob {k!r}")
        try:
            out[k] = int(v)
        except ValueError:
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
    return out


def _cmd_detect(args: argparse.Namespace) -> int:
    host = detect_host()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "detected_host.json"
    path.write_text(json.dumps(host, indent=2))
    print(json.dumps(host, indent=2))
    print(f"\n[wrote] {path}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    host = detect_host()
    knobs = list(args.knobs) if args.knobs else list(KNOB_ORDER)
    fixed = _parse_fix(args.fix or [])

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    out_dir = Path(args.out_dir) / f"{host['host_id']}_{timestamp}"

    cfg = SweepConfig(
        knobs=knobs,
        fixed=fixed,
        pool_duration=args.pool_duration,
        warmup=args.warmup,
        n_runs=args.n_runs,
        max_minutes=args.max_minutes,
        out_dir=out_dir,
        dry_run=args.dry_run,
    )
    print(f"[sweep] host={host['host_id']} knobs={knobs} fixed={fixed} dry_run={args.dry_run}",
          flush=True)
    print(f"[sweep] out_dir={out_dir}", flush=True)

    try:
        result = run_sweep(cfg, host)
    except KeyboardInterrupt:
        print("\n[sweep] interrupted before any knob completed — nothing to save", flush=True)
        return 1
    report_path, config_path = write_report(result)
    print(f"\n[report] {report_path}")
    print(f"[config] {config_path}")
    print(f"[csv]    {result['cells_csv']}")
    if result.get("interrupted"):
        print(f"\nPartial winners (interrupted): {result['winners']}")
        print(f"Wall: {result['wall_minutes']:.1f} min")
        return 1
    print(f"\nFinal winners: {result['winners']}")
    print(f"Wall: {result['wall_minutes']:.1f} min")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="sweep_harness")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_det = sub.add_parser("detect", help="Detect host CPU/GPU/VRAM and write detected_host.json")
    p_det.add_argument("--out-dir", default="reports/sweeps")
    p_det.set_defaults(func=_cmd_detect)

    p_run = sub.add_parser("run", help="Run the knob-registry sweep")
    p_run.add_argument("--knobs", nargs="*", default=None,
                       help=f"Subset of knobs to search (default: all). "
                            f"Available: {', '.join(KNOB_ORDER)}")
    p_run.add_argument("--fix", nargs="*", default=None,
                       help="Lock a knob to a value, e.g. --fix n_workers=24")
    p_run.add_argument("--max-minutes", type=int, default=90,
                       help="Wall-time budget (default 90). Estimate is computed from the "
                            "knob plan; sweep aborts pre-flight if estimate > budget.")
    p_run.add_argument("--pool-duration", type=int, default=DEFAULT_POOL_DURATION,
                       help=f"Bench pool seconds per run (default {DEFAULT_POOL_DURATION})")
    p_run.add_argument("--warmup", type=float, default=DEFAULT_WARMUP,
                       help=f"Worker warmup seconds (default {DEFAULT_WARMUP})")
    p_run.add_argument("--n-runs", type=int, default=DEFAULT_N_RUNS,
                       help=f"Bench reps per cell (default {DEFAULT_N_RUNS})")
    p_run.add_argument("--out-dir", default="reports/sweeps",
                       help="Parent dir; per-sweep subdir is appended (host_id + timestamp)")
    p_run.add_argument("--dry-run", action="store_true",
                       help="Use synthetic eval (no bench subprocess); for harness validation")
    p_run.set_defaults(func=_cmd_run)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
