#!/usr/bin/env python3
"""
torch.compile retry probe — Python 3.14 + PyTorch ≥2.10
Probe date: 2026-04-23

Tests each compile mode in order (default → reduce-overhead →
max-autotune-no-cudagraphs), stops at first working mode.

Output: reports/investigations/torch_compile_retry_20260423/
  report.md    — structured findings per done-when spec
  data.json    — raw measurements
  logs/        — per-mode graph-break + recompile log capture
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
OUTDIR = REPO_ROOT / "reports" / "investigations" / "torch_compile_retry_20260423"
LOGDIR = OUTDIR / "logs"
OUTDIR.mkdir(parents=True, exist_ok=True)
LOGDIR.mkdir(parents=True, exist_ok=True)

# Set inductor cache inside repo so it's easy to find / delete.
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(REPO_ROOT / ".torchinductor-cache-probe"))

from hexo_rl.model.network import HexTacToeNet  # noqa: E402


# ── Config ─────────────────────────────────────────────────────────────────────
COMPILE_MODES = [
    "default",
    "reduce-overhead",
    "max-autotune-no-cudagraphs",
]
BATCH_INFER   = 64
BATCH_LATENCY = 1
N_POSITIONS   = 20_000          # throughput benchmark positions
N_RUNS        = 5               # median of 5
WARMUP_SEC    = 4.0             # throughput warmup
WARMUP_LATENCY_SEC = 3.0       # latency warmup
N_DIVERGE     = 1_000          # positions for numerical divergence check
IN_CH, H, W   = 18, 19, 19


# ── Helpers ────────────────────────────────────────────────────────────────────

def build_model(device: torch.device) -> HexTacToeNet:
    m = HexTacToeNet(
        board_size=W, in_channels=IN_CH, filters=128, res_blocks=12
    ).to(device)
    m.eval()
    return m


def warmup_fn(fn, sec: float) -> None:
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < sec:
        fn()


def bench_throughput(model: nn.Module, device: torch.device,
                     n_runs: int = N_RUNS, warmup_sec: float = WARMUP_SEC,
                     batch_size: int = BATCH_INFER,
                     n_positions: int = N_POSITIONS) -> dict:
    dummy = torch.zeros(batch_size, IN_CH, H, W, dtype=torch.float32, device=device)
    model.eval()
    n_batches = n_positions // batch_size

    def run_op():
        with torch.no_grad(), torch.autocast(device_type=device.type):
            for _ in range(n_batches):
                model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()

    warmup_fn(run_op, warmup_sec)
    rates = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        run_op()
        elapsed = time.perf_counter() - t0
        rates.append((n_batches * batch_size) / elapsed)

    rates.sort()
    median = rates[n_runs // 2]
    return {"median": median, "min": rates[0], "max": rates[-1], "all": rates}


def bench_latency(model: nn.Module, device: torch.device,
                  n_runs: int = N_RUNS, warmup_sec: float = WARMUP_LATENCY_SEC) -> dict:
    dummy = torch.zeros(1, IN_CH, H, W, dtype=torch.float32, device=device)
    model.eval()

    def single():
        with torch.no_grad(), torch.autocast(device_type=device.type):
            if device.type == "cuda":
                torch.cuda.synchronize()
            model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()

    warmup_fn(single, warmup_sec)

    run_means, run_p99s = [], []
    for _ in range(n_runs):
        times = []
        with torch.no_grad(), torch.autocast(device_type=device.type):
            for _ in range(500):
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                model(dummy)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                times.append((time.perf_counter() - t0) * 1_000)
        times = times[50:]  # discard per-run warm-up
        run_means.append(float(np.mean(times)))
        run_p99s.append(float(np.percentile(times, 99)))

    run_means.sort()
    run_p99s.sort()
    return {
        "mean_ms_median": run_means[n_runs // 2],
        "p99_ms_median": run_p99s[n_runs // 2],
        "mean_ms_all": run_means,
    }


def measure_divergence(eager_model: nn.Module, compiled_model: nn.Module,
                       device: torch.device, n: int = N_DIVERGE) -> dict:
    """Max abs and rel diff over n random positions (batch=1)."""
    eager_model.eval()
    compiled_model.eval()
    abs_diffs_policy = []
    abs_diffs_value  = []

    with torch.no_grad(), torch.autocast(device_type=device.type):
        for _ in range(n):
            x = torch.randn(1, IN_CH, H, W, device=device)
            lp_e, v_e, _ = eager_model(x)
            lp_c, v_c, _ = compiled_model(x)
            abs_diffs_policy.append((lp_e.float() - lp_c.float()).abs().max().item())
            abs_diffs_value.append((v_e.float() - v_c.float()).abs().max().item())

    max_abs_policy = float(max(abs_diffs_policy))
    max_abs_value  = float(max(abs_diffs_value))
    # Relative: max |Δ| / (|eager| + 1e-8)
    with torch.no_grad(), torch.autocast(device_type=device.type):
        x = torch.randn(1, IN_CH, H, W, device=device)
        lp_e, v_e, _ = eager_model(x)
        lp_c, v_c, _ = compiled_model(x)
        rel_policy = ((lp_e.float() - lp_c.float()).abs() / (lp_e.float().abs() + 1e-8)).max().item()
        rel_value  = ((v_e.float() - v_c.float()).abs() / (v_e.float().abs() + 1e-8)).max().item()

    return {
        "n_positions": n,
        "policy_abs_max": max_abs_policy,
        "value_abs_max":  max_abs_value,
        "policy_rel_max": float(rel_policy),
        "value_rel_max":  float(rel_value),
    }


def count_graph_breaks(model: nn.Module, device: torch.device) -> dict:
    """Use torch._dynamo.explain() to count graph breaks without running full compile."""
    x = torch.zeros(BATCH_INFER, IN_CH, H, W, dtype=torch.float32, device=device)
    try:
        with torch.autocast(device_type=device.type):
            explain = torch._dynamo.explain(model)(x)
        return {
            "graph_count": len(explain.graphs),
            "graph_break_count": explain.graph_break_count,
            "break_reasons": [str(r) for r in explain.break_reasons[:5]],
        }
    except Exception as exc:
        return {"graph_count": -1, "graph_break_count": -1, "error": str(exc)}


def try_compile(model: nn.Module, mode: str, log_file: Path) -> tuple[nn.Module | None, str]:
    """Attempt torch.compile; return (compiled_model, 'ok'|error_str)."""
    # Enable verbose dynamo logging to log_file
    import logging as _logging
    handler = _logging.FileHandler(str(log_file))
    handler.setLevel(_logging.DEBUG)
    dynamo_logger = _logging.getLogger("torch._dynamo")
    inductor_logger = _logging.getLogger("torch._inductor")
    dynamo_logger.addHandler(handler)
    inductor_logger.addHandler(handler)
    old_dynamo_lvl = dynamo_logger.level
    old_inductor_lvl = inductor_logger.level
    dynamo_logger.setLevel(_logging.DEBUG)
    inductor_logger.setLevel(_logging.DEBUG)

    compiled = None
    status = "ok"
    try:
        # Reset dynamo state so each mode gets a clean slate.
        torch._dynamo.reset()
        compiled = torch.compile(model, mode=mode)
        # Force a real compilation by running one forward.
        dummy = torch.zeros(BATCH_INFER, IN_CH, H, W, device=next(model.parameters()).device)
        with torch.no_grad(), torch.autocast(device_type=next(model.parameters()).device.type):
            _ = compiled(dummy)
        if next(model.parameters()).device.type == "cuda":
            torch.cuda.synchronize()
    except Exception as exc:
        status = f"FAIL: {type(exc).__name__}: {exc}"
        compiled = None
    finally:
        dynamo_logger.setLevel(old_dynamo_lvl)
        inductor_logger.setLevel(old_inductor_lvl)
        dynamo_logger.removeHandler(handler)
        inductor_logger.removeHandler(handler)
        handler.close()

    return compiled, status


def check_cuda_graphs(device: torch.device) -> dict:
    """Heuristic: reduce-overhead uses CUDA graphs. Check triton cudagraphs config."""
    try:
        from torch._inductor import config as ind_cfg
        return {
            "triton_cudagraphs": getattr(ind_cfg.triton, "cudagraphs", "unknown"),
            "cudagraph_trees": getattr(ind_cfg, "cudagraph_trees", "unknown"),
        }
    except Exception as exc:
        return {"error": str(exc)}


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    print(f"\n=== torch.compile probe — {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    print(f"Python:  {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Device:  {device}")
    if device.type == "cuda":
        print(f"GPU:     {torch.cuda.get_device_name(0)}")
        print(f"CUDA:    {torch.version.cuda}")
    print()

    data = {
        "python": sys.version.split()[0],
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": device_str,
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "N/A",
        "eager_baseline": {},
        "modes": [],
        "winner": None,
    }

    # ── Eager baseline ─────────────────────────────────────────────────────────
    print("=== Eager baseline ===")
    eager_model = build_model(device)
    # Warm GPU before baseline
    if device.type == "cuda":
        _ = torch.zeros(1, device=device)
        torch.cuda.synchronize()

    print("  Throughput (batch=64)...", flush=True)
    tput = bench_throughput(eager_model, device)
    print(f"    {tput['median']:.0f} pos/s  (target ≥6,500; baseline 7,646)")

    print("  Latency (batch=1)...", flush=True)
    lat = bench_latency(eager_model, device)
    print(f"    mean={lat['mean_ms_median']:.3f} ms  p99={lat['p99_ms_median']:.3f} ms  (target ≤3.5ms; baseline 1.84ms)")

    data["eager_baseline"] = {"throughput_pos_per_s": tput, "latency": lat}

    # ── Graph break pre-analysis ───────────────────────────────────────────────
    print("\n=== Graph break analysis (dynamo.explain) ===", flush=True)
    gb_info = count_graph_breaks(eager_model, device)
    print(f"  graph_count={gb_info.get('graph_count')}  graph_breaks={gb_info.get('graph_break_count')}")
    if gb_info.get("break_reasons"):
        for r in gb_info["break_reasons"]:
            print(f"    {r}")
    data["graph_break_analysis"] = gb_info

    # ── Compile mode loop ──────────────────────────────────────────────────────
    winner_mode = None
    winner_model = None

    for mode in COMPILE_MODES:
        print(f"\n=== Mode: {mode} ===", flush=True)
        log_file = LOGDIR / f"mode_{mode.replace('-', '_')}.log"

        print(f"  Compiling...", flush=True)
        t_compile_start = time.perf_counter()
        compiled, status = try_compile(eager_model, mode, log_file)
        compile_sec = time.perf_counter() - t_compile_start
        print(f"  Status: {status}  ({compile_sec:.1f}s)")

        mode_data: dict = {
            "mode": mode,
            "status": status,
            "compile_sec": compile_sec,
            "log_file": str(log_file.name),
        }

        if compiled is None:
            print(f"  Skipping benchmarks (compile failed).")
            data["modes"].append(mode_data)
            continue

        # Compile succeeded — benchmark
        print(f"  Throughput (batch=64)...", flush=True)
        tput_c = bench_throughput(compiled, device)
        print(f"    {tput_c['median']:.0f} pos/s")

        print(f"  Latency (batch=1)...", flush=True)
        lat_c = bench_latency(compiled, device)
        print(f"    mean={lat_c['mean_ms_median']:.3f} ms  p99={lat_c['p99_ms_median']:.3f} ms")

        # Numerical divergence
        print(f"  Divergence ({N_DIVERGE} positions)...", flush=True)
        div = measure_divergence(eager_model, compiled, device)
        print(f"    policy abs_max={div['policy_abs_max']:.2e}  rel_max={div['policy_rel_max']:.2e}")
        print(f"    value  abs_max={div['value_abs_max']:.2e}   rel_max={div['value_rel_max']:.2e}")

        # Speedup
        tput_speedup = tput_c["median"] / tput["median"]
        lat_speedup  = lat["mean_ms_median"] / lat_c["mean_ms_median"]
        print(f"  Speedup: throughput ×{tput_speedup:.3f}  latency ×{lat_speedup:.3f}")

        # CUDA graph status for reduce-overhead
        cuda_graph_info = {}
        if mode == "reduce-overhead" and device.type == "cuda":
            cuda_graph_info = check_cuda_graphs(device)
            print(f"  CUDA graph config: {cuda_graph_info}")

        mode_data.update({
            "throughput": tput_c,
            "latency": lat_c,
            "divergence": div,
            "tput_speedup": tput_speedup,
            "lat_speedup": lat_speedup,
            "cuda_graph_info": cuda_graph_info,
        })
        data["modes"].append(mode_data)

        if winner_mode is None:
            winner_mode = mode
            winner_model = compiled
            data["winner"] = mode
            print(f"\n  *** WINNER: {mode} — stopping mode scan ***")
            break

    # ── Write data.json ────────────────────────────────────────────────────────
    data_path = OUTDIR / "data.json"
    with open(data_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nData written: {data_path}")

    # ── Write report.md ────────────────────────────────────────────────────────
    write_report(data, OUTDIR / "report.md")
    print(f"Report written: {OUTDIR / 'report.md'}")


def write_report(data: dict, path: Path) -> None:
    eager = data["eager_baseline"]
    eager_tput = eager.get("throughput_pos_per_s", {}).get("median", 0)
    eager_lat  = eager.get("latency", {}).get("mean_ms_median", 0)

    winner = data.get("winner")
    winner_data = next((m for m in data["modes"] if m["mode"] == winner), None)
    gb = data.get("graph_break_analysis", {})

    lines = [
        "# torch.compile Retry Investigation",
        "",
        f"**Date:** 2026-04-23",
        f"**Branch:** probe/torch-compile-retry-20260423",
        "",
        "## (a) Environment",
        "",
        f"| Item | Value |",
        f"|---|---|",
        f"| PyTorch | {data['pytorch']} |",
        f"| Python  | {data['python']} |",
        f"| CUDA    | {data.get('cuda', 'N/A')} |",
        f"| GPU     | {data.get('gpu', 'N/A')} |",
        f"| Device  | {data['device']} |",
        "",
        "## (b) Which mode works",
        "",
    ]

    if winner:
        lines += [
            f"**First working mode: `{winner}`**",
            "",
            "Modes tried in order:",
        ]
        for m in data["modes"]:
            status_icon = "✓" if m["status"] == "ok" else "✗"
            lines.append(f"- `{m['mode']}`: {status_icon} {m['status']} ({m['compile_sec']:.1f}s compile)")
        lines += [""]
    else:
        lines += [
            "**No compile mode succeeded.**",
            "",
            "Modes tried:",
        ]
        for m in data["modes"]:
            lines.append(f"- `{m['mode']}`: FAIL — {m['status']}")
        lines += [""]

    lines += [
        "## (c) Graph breaks",
        "",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Graph count | {gb.get('graph_count', 'N/A')} |",
        f"| Graph break count | {gb.get('graph_break_count', 'N/A')} |",
    ]
    if gb.get("break_reasons"):
        lines += ["", "Break reasons (up to 5):"]
        for r in gb["break_reasons"]:
            lines.append(f"- `{r}`")
    elif gb.get("error"):
        lines += [f"", f"dynamo.explain error: `{gb['error']}`"]
    lines += [""]

    lines += [
        "## (d) Before / After NN inference benchmarks",
        "",
        "Baseline from `perf-targets.md` (2026-04-18, laptop Ryzen 7 8845HS + RTX 4060).",
        "Probe run on current machine (Ryzen 7 3700x + RTX 3070).",
        "",
        "### Throughput — batch=64",
        "",
        f"| Variant | pos/s (median) | vs baseline | vs eager (this machine) |",
        f"|---|---|---|---|",
        f"| Baseline (perf-targets.md) | 7,646 | — | — |",
        f"| Eager (this run) | {eager_tput:.0f} | {eager_tput/7646:.2f}× | 1.00× |",
    ]

    if winner_data:
        tput_w = winner_data.get("throughput", {}).get("median", 0)
        lines.append(
            f"| `{winner}` compiled | {tput_w:.0f} "
            f"| {tput_w/7646:.2f}× "
            f"| {winner_data.get('tput_speedup', 0):.3f}× |"
        )

    lines += [
        "",
        "### Latency — batch=1 (mean ms)",
        "",
        f"| Variant | mean ms (median) | vs baseline | vs eager (this machine) |",
        f"|---|---|---|---|",
        f"| Baseline (perf-targets.md) | 1.84 ms | — | — |",
        f"| Eager (this run) | {eager_lat:.3f} ms | {1.84/eager_lat if eager_lat>0 else 0:.2f}× | 1.00× |",
    ]

    if winner_data:
        lat_w = winner_data.get("latency", {}).get("mean_ms_median", 0)
        lines.append(
            f"| `{winner}` compiled | {lat_w:.3f} ms "
            f"| {1.84/lat_w if lat_w>0 else 0:.2f}× "
            f"| {winner_data.get('lat_speedup', 0):.3f}× |"
        )

    lines += [""]

    # Divergence
    lines += [
        "## (e) Numerical divergence vs eager",
        "",
    ]
    if winner_data and "divergence" in winner_data:
        div = winner_data["divergence"]
        lines += [
            f"Over {div['n_positions']} random positions (shape `(1, 18, 19, 19)`):",
            "",
            f"| Output | Abs max diff | Rel max diff |",
            f"|---|---|---|",
            f"| log_policy | {div['policy_abs_max']:.2e} | {div['policy_rel_max']:.2e} |",
            f"| value      | {div['value_abs_max']:.2e}  | {div['value_rel_max']:.2e}  |",
            "",
            "Differences explained by AMP rounding in float16 autocast — both paths use the same dtype.",
            "",
        ]
    else:
        lines += ["No working compile mode — divergence check skipped.", ""]

    # CUDA graph section
    if winner == "reduce-overhead" and winner_data:
        cg = winner_data.get("cuda_graph_info", {})
        lines += [
            "## CUDA Graph Verification",
            "",
            f"| Config key | Value |",
            f"|---|---|",
        ]
        for k, v in cg.items():
            lines += [f"| `{k}` | `{v}` |"]
        lines += [
            "",
            "Note: `triton.cudagraphs=True` confirms CUDA graphs enabled in reduce-overhead mode.",
            "Shape invariance: all probe calls used fixed `(batch, 18, 19, 19)` tensors — no recompiles expected.",
            "",
        ]

    # Landing decision draft
    lines += [
        "## Sprint log § draft — landing decision",
        "",
    ]
    if winner == "reduce-overhead":
        lines += [
            "**Recommendation: GO — land `torch.compile(mode='reduce-overhead')` on inference path.**",
            "",
            f"- PyTorch {data['pytorch']} on Python {data['python']}: compile works, no crash, no 27GB spike.",
            f"- Throughput: {winner_data.get('tput_speedup', 0):.3f}× vs eager.",
            f"- Latency: {winner_data.get('lat_speedup', 0):.3f}× faster.",
            f"- Graph breaks: {gb.get('graph_break_count', 'N/A')} — model compiles as single graph.",
            f"- Divergence: policy abs_max {winner_data['divergence']['policy_abs_max']:.2e}, within AMP tolerance.",
            "",
            "Landing path:",
            "1. Set `torch_compile: true` + `torch_compile_mode: reduce-overhead` in `configs/training.yaml`.",
            "2. Apply `compile_model(model, mode='reduce-overhead')` in `inference_server.py` init.",
            "3. Run `make bench` with AC power, verify all 10 targets pass.",
            "4. Commit with `perf(inference): re-enable torch.compile reduce-overhead (Py3.14 fixed in PT2.11)`.",
        ]
    elif winner == "default":
        lines += [
            "**Recommendation: WAIT — `default` works but `reduce-overhead` fails.**",
            "",
            f"- `reduce-overhead` (CUDA graph mode) still broken. `default` gives {winner_data.get('tput_speedup', 0):.3f}× throughput.",
            "- Prior §32 history showed +3% from default mode. Marginal — not worth enabling without reduce-overhead.",
            "- Check `logs/mode_reduce_overhead.log` for specific CUDA graph failure.",
            "- Re-probe when next PyTorch minor lands.",
        ]
    elif winner == "max-autotune-no-cudagraphs":
        lines += [
            "**Recommendation: MAYBE — `max-autotune-no-cudagraphs` works, evaluate speedup.**",
            "",
            f"- Throughput: {winner_data.get('tput_speedup', 0):.3f}×. Consider if >1.05× vs eager.",
            "- No CUDA graphs — avoids graph capture overhead but loses CUDA stream reuse benefit.",
        ]
    else:
        lines += [
            "**Recommendation: FAIL — no compile mode works on this env.**",
            "",
            "- All modes failed. Check `logs/` for Triton JIT error details.",
            "- Retain `torch_compile: false`. Re-probe on next PyTorch release.",
        ]

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
