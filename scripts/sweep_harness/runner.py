"""Sweep runner — orchestrates per-knob search via subprocess bench calls.

Each knob in ``KNOB_ORDER`` runs its declared strategy. Earlier-knob winners
are passed forward via ``fixed`` so downstream evals see the right config.

Subprocess design (matches ``scripts/sweep_epyc4080.py``):
* Fresh ``python scripts/benchmark.py`` per cell — the alternative would be
  to import the bench harness in-process, but each bench creates and tears
  down a CUDA context plus a worker pool. A leaked context from a prior cell
  has historically produced bimodal startup-race results; fresh process
  guarantees clean state.
* ``--no-compile`` is always passed (§124 methodology pin: production
  training has ``torch_compile=false``; bench gate must mirror it).
* Override YAML layers ``configs/variants/_sweep_template.yaml`` + per-cell
  knob values, written to a tempfile and unlinked after the cell exits.

Recoverability: every cell appends a row to ``cells.csv`` immediately after
it finishes. A killed sweep can be resumed by a future ``--resume`` flag
without re-running completed cells. (Resume is not wired in this revision —
the CSV is the durable record; pulling rows out is a one-line filter.)
"""

from __future__ import annotations

import csv
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

try:
    import yaml  # PyYAML is already a dep (configs/variants/*.yaml)
except ImportError as exc:  # pragma: no cover - dep is in pyproject
    raise SystemExit("sweep_harness needs PyYAML — pip install pyyaml") from exc

from .compare import CellResult, bimodal_from_raw, compare_iqr, upper_mode_filter
from .knobs import (
    KNOB_ORDER,
    KNOBS,
    merge_dicts,
    param_path_to_yaml,
    resolve_auto_bounds,
    resolve_constraint,
)
from .strategies import (
    bisect_search,
    grid_coarse_refine,
    grid_search,
    ternary_search_int,
)

ROOT = Path(__file__).resolve().parent.parent.parent
PY = str(ROOT / ".venv" / "bin" / "python")
BENCH = str(ROOT / "scripts" / "benchmark.py")
SWEEP_TEMPLATE = ROOT / "configs" / "variants" / "_sweep_template.yaml"

# Default measurement settings.
# pool_duration=90 is the harness default — matches `make bench` warmup
# baseline and fits the 90-min wall budget for a typical multi-knob sweep.
# §124/§125's 180 s methodology is available via `make sweep.long` (or any
# of the `sweep.*.long` variants) and `--pool-duration 180` on the CLI.
DEFAULT_POOL_DURATION = 90
DEFAULT_WARMUP = 90.0
DEFAULT_N_RUNS = 5
RETRY_POOL_DURATION = 240
RETRY_N_RUNS = 8

# Bench stdout line: "Worker pool throughput pos/hr: median=388,426  IQR=+/-143,426  [266.9k-410.3k]  n=3"
_NUM = r"([\d,]+\.?\d*)"
_LINE_RE = re.compile(
    rf"^\s*Worker pool throughput pos/hr:\s*median={_NUM}\s+IQR=\+/-{_NUM}\s+\[([^\]]+)\]\s+n=(\d+)"
)
_GPU_RE = re.compile(rf"^\s*GPU utilisation util%:\s*median={_NUM}")
_BATCH_RE = re.compile(rf"^\s*Worker pool throughput batch%:\s*median={_NUM}")


def _to_float(s: str) -> float:
    s = s.strip()
    mult = 1.0
    if s.endswith("k"):
        mult, s = 1e3, s[:-1]
    elif s.endswith("M"):
        mult, s = 1e6, s[:-1]
    return float(s.replace(",", "")) * mult


def parse_bench_pos(text: str) -> tuple[CellResult | None, dict]:
    """Pull pos/hr CellResult + side metrics (gpu, batch fill) from stdout."""
    cell: CellResult | None = None
    side: dict[str, float] = {}
    for line in text.splitlines():
        m = _LINE_RE.match(line)
        if m:
            median = float(m.group(1).replace(",", ""))
            iqr = float(m.group(2).replace(",", ""))
            rng = m.group(3)
            try:
                lo_s, hi_s = rng.split("-", 1) if rng.count("-") == 1 else rng.rsplit("-", 1)
                lo, hi = _to_float(lo_s), _to_float(hi_s)
            except Exception:
                lo, hi = float("nan"), float("nan")
            n = int(m.group(4))
            cell = CellResult(median=median, iqr=iqr, min=lo, max=hi, n_runs=n, raw=())
        elif (g := _GPU_RE.match(line)) is not None:
            side["gpu_util"] = float(g.group(1).replace(",", ""))
        elif (b := _BATCH_RE.match(line)) is not None:
            side["batch_fill"] = float(b.group(1).replace(",", ""))
    return cell, side


# ── host detection ────────────────────────────────────────────────────────────


def detect_host() -> dict[str, Any]:
    """Detect CPU threads, GPU name, VRAM. Returns a JSON-serialisable dict."""
    cpu_threads = os.cpu_count() or 4
    gpu_name = "unknown"
    vram_gb = 0.0
    sm_arch = ""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap",
             "--format=csv,noheader,nounits"],
            text=True, timeout=5,
        ).strip().splitlines()[0]
        parts = [p.strip() for p in out.split(",")]
        if parts:
            gpu_name = parts[0]
            vram_gb = round(float(parts[1]) / 1024.0, 2) if len(parts) > 1 else 0.0
            sm_arch = parts[2] if len(parts) > 2 else ""
    except Exception:
        pass

    # Short host_id for filenames/dirs.
    cpu_short = "cpu"
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    mn = line.split(":", 1)[1].strip().lower()
                    for tag in ("epyc", "ryzen", "xeon", "core i9", "core i7", "threadripper"):
                        if tag in mn:
                            cpu_short = tag.replace(" ", "")
                            break
                    break
    except Exception:
        pass
    gpu_short = re.sub(r"[^a-z0-9]+", "", gpu_name.lower()) or "gpu"
    host_id = f"{cpu_short}-{cpu_threads}t-{gpu_short}"

    return {
        "host_id": host_id,
        "cpu_threads": cpu_threads,
        "gpu_name": gpu_name,
        "vram_gb": vram_gb,
        "sm_arch": sm_arch,
    }


# ── per-cell evaluation ───────────────────────────────────────────────────────


@dataclass
class SweepConfig:
    knobs: list[str]
    fixed: dict[str, Any] = field(default_factory=dict)
    pool_duration: int = DEFAULT_POOL_DURATION
    warmup: float = DEFAULT_WARMUP
    n_runs: int = DEFAULT_N_RUNS
    max_minutes: int = 90
    out_dir: Path = field(default_factory=lambda: Path("reports/sweeps"))
    template: Path = SWEEP_TEMPLATE
    dry_run: bool = False
    # Per-knob registry overrides applied at search time (not persisted to
    # knobs.py). Keys are knob names; values are dicts merged onto the
    # registry spec. Supported keys:
    #   "coarse": list[int|float]  — replaces spec["coarse"] or spec["values"]
    #   "bounds": tuple[int,int]   — replaces spec["bounds"] for ternary/bisect
    knob_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)


def _write_override(template: Path, fixed: dict[str, Any],
                    knob_value: dict[str, Any]) -> Path:
    """Layer template + fixed-knob params + this-cell params, write tempfile."""
    base = {}
    if template.exists():
        base = yaml.safe_load(template.read_text()) or {}
    overrides: list[dict] = []
    for k, v in fixed.items():
        spec = KNOBS[k]
        overrides.append(param_path_to_yaml(spec["param_path"], v))
    for k, v in knob_value.items():
        spec = KNOBS[k]
        overrides.append(param_path_to_yaml(spec["param_path"], v))
    merged = merge_dicts(base, *overrides)
    fd, path = tempfile.mkstemp(suffix=".yaml", prefix="sweep_cell_")
    os.close(fd)
    Path(path).write_text(yaml.safe_dump(merged, sort_keys=False))
    return Path(path)


def run_bench_subprocess(override: Path, n_workers: int, pool_duration: int,
                         n_runs: int, warmup: float) -> tuple[str, int, float]:
    """Spawn ``python scripts/benchmark.py``; stream stdout; return text."""
    cmd = [
        PY, BENCH,
        "--config", str(override),
        "--pool-workers", str(n_workers),
        "--pool-duration", str(pool_duration),
        "--n-runs", str(n_runs),
        "--pool-warmup", str(warmup),
        "--no-compile",
    ]
    print(f"[cmd]  {' '.join(cmd)}", flush=True)
    t0 = time.time()
    env = dict(os.environ, MALLOC_ARENA_MAX="2")
    proc = subprocess.Popen(
        cmd, env=env, cwd=str(ROOT),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        bufsize=1, universal_newlines=True,
    )
    captured: list[str] = []
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            captured.append(line)
        proc.wait()
    except KeyboardInterrupt:
        # subprocess is in the same process group and already received SIGINT;
        # terminate + force-kill as insurance (e.g. hung CUDA context).
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        raise
    return "".join(captured), proc.returncode, time.time() - t0


def make_eval_fn(knob_name: str, fixed: dict[str, Any], cfg: SweepConfig,
                 csv_path: Path, host: dict[str, Any]) -> Callable[[Any], CellResult]:
    """Build the ``eval_fn`` passed into a strategy.

    Resolves the worker count for this cell (from ``fixed`` or default 16),
    writes the override, runs bench, parses pos/hr, retries once on
    bimodal cells with a longer window.
    """

    def eval_fn(value: Any) -> CellResult:
        knob_value = {knob_name: value}
        # n_workers cell needs the value as worker count; for other knobs
        # use the resolved fixed worker (raise if not yet known).
        if knob_name == "n_workers":
            n_workers = int(value)
        else:
            n_workers = int(fixed.get("n_workers")
                            or KNOBS["n_workers"].get("default", 16))

        override = _write_override(cfg.template, fixed, knob_value)
        try:
            print(f"\n[cell] knob={knob_name} value={value} (workers={n_workers})",
                  flush=True)
            text, rc, elapsed = run_bench_subprocess(
                override, n_workers, cfg.pool_duration, cfg.n_runs, cfg.warmup,
            )
            cell, side = parse_bench_pos(text)
            retry = False
            if cell is not None and bimodal_from_raw(
                # No raw runs available from stdout; synthesize a 3-element
                # proxy [min, median, max] so bimodal_from_raw can apply
                # the same min/median rule. Test cases use real raws.
                (cell.min, cell.median, cell.max), cell.median
            ):
                print(f"[bimodal] retry with pool_duration={RETRY_POOL_DURATION}s "
                      f"n_runs={RETRY_N_RUNS}", flush=True)
                text2, rc2, elapsed2 = run_bench_subprocess(
                    override, n_workers, RETRY_POOL_DURATION, RETRY_N_RUNS, cfg.warmup,
                )
                cell2, side2 = parse_bench_pos(text2)
                if cell2 is not None:
                    cell = cell2
                    side = side2 or side
                rc = rc2
                elapsed += elapsed2
                retry = True
            if cell is None:
                # Failed to parse — return a sentinel zero result so the
                # strategy can still proceed (and so cells.csv records it).
                cell = CellResult(median=0.0, iqr=0.0, min=0.0, max=0.0, n_runs=0)

            bimodal = bimodal_from_raw(
                (cell.min, cell.median, cell.max), cell.median
            )
            if bimodal and retry:
                # After one retry, recompute IQR from upper-mode runs only.
                upper = upper_mode_filter([cell.min, cell.median, cell.max], cell.median)
                if upper:
                    cell = CellResult(
                        median=cell.median,
                        iqr=max(upper) - min(upper),
                        min=cell.min, max=cell.max, n_runs=cell.n_runs,
                        raw=cell.raw, bimodal=True,
                    )
                else:
                    cell = CellResult(
                        median=cell.median, iqr=cell.iqr, min=cell.min,
                        max=cell.max, n_runs=cell.n_runs, raw=cell.raw, bimodal=True,
                    )

            _append_cells_csv(csv_path, knob_name, value, fixed, cell, side,
                              elapsed, rc, host)
            return cell
        finally:
            try:
                override.unlink()
            except Exception:
                pass

    return eval_fn


def _append_cells_csv(path: Path, knob: str, value: Any, fixed: dict, cell: CellResult,
                      side: dict, wall_seconds: float, rc: int, host: dict) -> None:
    new = not path.exists()
    cols = [
        "timestamp", "host_id", "knob", "value",
        "n_workers", "inference_batch_size", "inference_max_wait_ms", "max_train_burst",
        "median_pos", "iqr_pos", "min_pos", "max_pos", "n_runs",
        "gpu_util_pct", "batch_fill_pct",
        "bimodal_flag", "wall_seconds", "exit_code",
    ]
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "host_id": host["host_id"],
        "knob": knob,
        "value": value,
        "n_workers": fixed.get("n_workers", value if knob == "n_workers" else ""),
        "inference_batch_size": fixed.get("inference_batch_size",
                                          value if knob == "inference_batch_size" else ""),
        "inference_max_wait_ms": fixed.get("inference_max_wait_ms",
                                            value if knob == "inference_max_wait_ms" else ""),
        "max_train_burst": fixed.get("max_train_burst",
                                      value if knob == "max_train_burst" else ""),
        "median_pos": cell.median,
        "iqr_pos": cell.iqr,
        "min_pos": cell.min,
        "max_pos": cell.max,
        "n_runs": cell.n_runs,
        "gpu_util_pct": side.get("gpu_util", ""),
        "batch_fill_pct": side.get("batch_fill", ""),
        "bimodal_flag": cell.bimodal,
        "wall_seconds": round(wall_seconds, 1),
        "exit_code": rc,
    }
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if new:
            w.writeheader()
        w.writerow(row)


# ── budget estimate ──────────────────────────────────────────────────────────


def estimate_budget(knobs: list[str], pool_duration: int, warmup: float, n_runs: int,
                    host: dict[str, Any],
                    knob_overrides: "dict[str, dict[str, Any]] | None" = None) -> tuple[int, int]:
    """Return (estimated_cells, estimated_minutes)."""
    overrides = knob_overrides or {}
    cells = 0
    for k in knobs:
        spec = {**KNOBS[k], **overrides.get(k, {})}
        s = spec["strategy"]
        if s == "ternary":
            cells += 2 * spec.get("iterations", 4) + 2  # iter pairs + endpoints (cache offsets some)
        elif s == "grid_coarse_refine":
            cells += len(spec["coarse"]) + 2 * spec.get("refine_window", 1)
        elif s == "grid":
            cells += len(spec["values"])
        elif s == "bisect":
            cells += 2 * spec.get("iterations", 3) + 2
        elif s == "fixed":
            cells += 0
    per_cell_s = pool_duration * n_runs + warmup + 60  # +60 for proc spawn / load
    return cells, int(cells * per_cell_s / 60)


# ── orchestration ────────────────────────────────────────────────────────────


def run_sweep(cfg: SweepConfig, host: dict[str, Any]) -> dict[str, Any]:
    """Run the sweep. Returns a dict suitable for ``reporting.write_report``.

    ``dry_run=True`` skips bench and uses a synthetic eval (used by tests
    and to verify the orchestration without paying GPU time).
    """
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    cells_csv = cfg.out_dir / "cells.csv"

    knob_list = [k for k in KNOB_ORDER if k in cfg.knobs] or cfg.knobs
    fixed: dict[str, Any] = dict(cfg.fixed)
    traces: dict[str, list[dict]] = {}
    winners: dict[str, Any] = {}
    final_results: dict[str, CellResult] = {}

    cells_est, mins_est = estimate_budget(knob_list, cfg.pool_duration,
                                          cfg.warmup, cfg.n_runs, host,
                                          cfg.knob_overrides)
    print(f"[budget] estimated cells={cells_est} wall≈{mins_est} min "
          f"(max={cfg.max_minutes} min)", flush=True)
    if not cfg.dry_run and mins_est > cfg.max_minutes:
        raise SystemExit(
            f"budget {mins_est} min > --max-minutes {cfg.max_minutes}. "
            f"Either raise --max-minutes, lower --pool-duration/--n-runs, "
            f"or reduce --knobs."
        )

    cmp_fn = compare_iqr
    t_start = time.time()
    interrupted = False

    try:
        for k in knob_list:
            if k in fixed:
                print(f"\n[skip] {k} fixed at {fixed[k]} (--fix override)", flush=True)
                traces[k] = [{"fixed": fixed[k], "skipped": "user --fix"}]
                continue
            spec = {**KNOBS[k], **cfg.knob_overrides.get(k, {})}
            s = spec["strategy"]

            if s == "fixed":
                fixed[k] = spec["value"]
                winners[k] = spec["value"]
                traces[k] = [{"fixed": spec["value"], "reason": spec.get("skip_reason", "")}]
                print(f"\n[knob {k}] fixed = {spec['value']}  ({spec.get('skip_reason', '')})",
                      flush=True)
                continue

            ovr_note = f" [overrides: {cfg.knob_overrides[k]}]" if k in cfg.knob_overrides else ""
            print(f"\n========== knob: {k} ({s}){ovr_note} ==========", flush=True)
            if cfg.dry_run:
                eval_fn = _dry_run_eval(k)
            else:
                eval_fn = make_eval_fn(k, fixed, cfg, cells_csv, host)

            if s == "ternary":
                low, high = (spec["bounds"] if isinstance(spec.get("bounds"), tuple)
                             else resolve_auto_bounds(k, host))
                best, trace = ternary_search_int(
                    eval_fn, low, high, spec["iterations"], spec["tolerance"], cmp_fn,
                )
            elif s == "grid_coarse_refine":
                constraint = resolve_constraint(spec.get("constraint"), fixed)
                best, trace = grid_coarse_refine(
                    eval_fn, list(spec["coarse"]),
                    spec.get("refine_window", 1), spec.get("refine_step", 32),
                    cmp_fn, constraint=constraint,
                )
            elif s == "grid":
                best, trace = grid_search(eval_fn, list(spec["values"]), cmp_fn)
            elif s == "bisect":
                low, high = (spec["bounds"] if isinstance(spec.get("bounds"), tuple)
                             else resolve_auto_bounds(k, host))
                best, trace = bisect_search(eval_fn, low, high, spec["iterations"], cmp_fn)
            else:
                raise ValueError(f"unknown strategy {s} for knob {k}")

            winners[k] = best
            fixed[k] = best
            traces[k] = trace
            # Record final cell result for the winner (last cached).
            try:
                final_results[k] = eval_fn(best)
            except Exception:  # eval cache misses if the strategy already evaluated best
                pass
            print(f"[knob {k}] WINNER = {best}", flush=True)

    except KeyboardInterrupt:
        interrupted = True
        print("\n\n[interrupt] Ctrl+C — saving partial results...", flush=True)

    wall_minutes = (time.time() - t_start) / 60.0
    return {
        "host": host,
        "winners": winners,
        "fixed": fixed,
        "traces": traces,
        "wall_minutes": round(wall_minutes, 1),
        "out_dir": cfg.out_dir,
        "cells_csv": cells_csv,
        "final_results": final_results,
        "interrupted": interrupted,
    }


def _dry_run_eval(knob_name: str) -> Callable[[Any], CellResult]:
    """Synthetic eval for ``--dry-run``. Uses a smooth unimodal function so
    the orchestration can be exercised without invoking bench."""

    def eval_fn(value: Any) -> CellResult:
        # Unimodal peak around 24 for n_workers, 192 for batch, 4 for wait, 16 for burst.
        peaks = {"n_workers": 24, "inference_batch_size": 192,
                 "inference_max_wait_ms": 4.0, "max_train_burst": 16}
        peak = peaks.get(knob_name, 16)
        v = float(value)
        median = max(1e3, 4e5 - 5e3 * (v - peak) ** 2)
        iqr = median * 0.04
        return CellResult(median=median, iqr=iqr, min=median * 0.9,
                          max=median * 1.05, n_runs=5, raw=())

    return eval_fn
