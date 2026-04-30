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
from pathlib import Path as _Path


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


def _resolve_variant_template(variant: str | None) -> Path:
    """Return the Path for --variant NAME (configs/variants/<name>.yaml), or the default template."""
    if variant is None:
        from .runner import SWEEP_TEMPLATE
        return SWEEP_TEMPLATE
    p = Path("configs") / "variants" / f"{variant}.yaml"
    if not p.exists():
        raise SystemExit(f"--variant: file not found: {p}")
    return p


def _parse_knob_overrides(coarse_items: list[str] | None,
                          bounds_items: list[str] | None,
                          values_items: list[str] | None = None) -> dict[str, dict]:
    """Parse --coarse and --bounds into a knob_overrides dict.

    --coarse inference_batch_size=256,384,512
        → {"inference_batch_size": {"coarse": [256, 384, 512]}}
        Also works for grid knobs (overrides "values" key).

    --bounds n_workers=24:96
        → {"n_workers": {"bounds": (24, 96)}}
        Also works for bisect knobs.
    """
    out: dict[str, dict] = {}

    for it in coarse_items or []:
        if "=" not in it:
            raise SystemExit(f"--coarse expects KNOB=v1,v2,..., got {it!r}")
        k, vs = it.split("=", 1)
        if k not in KNOBS:
            raise SystemExit(f"--coarse: unknown knob {k!r}")
        strategy = KNOBS[k]["strategy"]
        try:
            parsed = [int(v) if "." not in v else float(v) for v in vs.split(",")]
        except ValueError:
            raise SystemExit(f"--coarse: non-numeric values in {it!r}")
        field = "values" if strategy == "grid" else "coarse"
        out.setdefault(k, {})[field] = parsed

    for it in bounds_items or []:
        if "=" not in it or ":" not in it:
            raise SystemExit(f"--bounds expects KNOB=lo:hi, got {it!r}")
        k, rng = it.split("=", 1)
        if k not in KNOBS:
            raise SystemExit(f"--bounds: unknown knob {k!r}")
        lo_s, hi_s = rng.split(":", 1)
        try:
            lo, hi = int(lo_s), int(hi_s)
        except ValueError:
            raise SystemExit(f"--bounds: expected integers in {it!r}")
        out.setdefault(k, {})["bounds"] = (lo, hi)

    # --values always writes to "values" key (explicit alias for grid knobs).
    for it in values_items or []:
        if "=" not in it:
            raise SystemExit(f"--values expects KNOB=v1,v2,..., got {it!r}")
        k, vs = it.split("=", 1)
        if k not in KNOBS:
            raise SystemExit(f"--values: unknown knob {k!r}")
        try:
            parsed = [int(v) if "." not in v else float(v) for v in vs.split(",")]
        except ValueError:
            raise SystemExit(f"--values: non-numeric values in {it!r}")
        out.setdefault(k, {})["values"] = parsed

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
    knob_overrides = _parse_knob_overrides(args.coarse or [], args.bounds or [], args.values or [])
    template = _resolve_variant_template(getattr(args, "variant", None))

    resume_csv = _Path(args.resume) if args.resume else None
    if resume_csv is not None and not resume_csv.exists():
        raise SystemExit(f"--resume: file not found: {resume_csv}")

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
        template=template,
        dry_run=args.dry_run,
        knob_overrides=knob_overrides,
        resume_csv=resume_csv,
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
    p_run.add_argument("--variant", default=None, metavar="NAME",
                       help="Base variant YAML to layer under knob overrides "
                            "(configs/variants/<NAME>.yaml). Defaults to _sweep_template.yaml.")
    p_run.add_argument("--coarse", nargs="*", default=None, metavar="KNOB=v1,v2,...",
                       help="Override coarse/values list for a grid or grid_coarse_refine knob. "
                            "Example: --coarse inference_batch_size=256,384,512")
    p_run.add_argument("--values", nargs="*", default=None, metavar="KNOB=v1,v2,...",
                       help="Override values list for a grid knob (explicit alias for --coarse). "
                            "Example: --values inference_max_wait_ms=1.0,2.0,4.0,8.0")
    p_run.add_argument("--bounds", nargs="*", default=None, metavar="KNOB=lo:hi",
                       help="Override search bounds for a ternary or bisect knob. "
                            "Example: --bounds n_workers=32:128")
    p_run.add_argument("--resume", default=None, metavar="CELLS_CSV",
                       help="Path to a prior sweep's cells.csv. Already-evaluated "
                            "(knob, value) pairs are loaded from that file and skip bench. "
                            "Example: --resume reports/sweeps/ryzen-16t-rtx3070_2026-04-28_20-00/cells.csv")
    p_run.set_defaults(func=_cmd_run)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
