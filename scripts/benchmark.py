#!/usr/bin/env python3
"""
Phase 3.5 / 4 benchmark harness.

Measures the metrics that gate Phase 4.5 and prints a pass/fail table.

Usage:
    .venv/bin/python scripts/benchmark.py
    .venv/bin/python scripts/benchmark.py --config configs/fast_debug.yaml
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*pynvml package is deprecated.*")

import argparse
import sys
import time
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np
import torch
import yaml
from rich.console import Console
from rich.table import Table

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if TYPE_CHECKING:
    from python.model.network import HexTacToeNet
    from native_core import RustReplayBuffer

console = Console()

# ── Individual benchmarks ─────────────────────────────────────────────────────


def benchmark_mcts(n_simulations: int = 50_000) -> Dict[str, Any]:
    """CPU-only MCTS throughput (no neural network)."""
    from native_core import Board, MCTSTree  # type: ignore[attr-defined]

    board = Board()
    tree  = MCTSTree(c_puct=1.5)
    tree.new_game(board)

    # Warm up
    tree.run_simulations_cpu_only(n=1_000)
    tree.reset()

    start = time.perf_counter()
    tree.run_simulations_cpu_only(n=n_simulations)
    elapsed = time.perf_counter() - start

    return {
        "name":          "MCTS (CPU only, no NN)",
        "sims":          n_simulations,
        "elapsed_sec":   elapsed,
        "sims_per_sec":  n_simulations / elapsed,
    }


def benchmark_inference(model: "HexTacToeNet", n_positions: int = 20_000, batch_size: int = 64) -> Dict[str, Any]:
    """NN throughput in batched evaluation mode."""
    device = next(model.parameters()).device
    dummy_local  = torch.zeros(batch_size, 18, 19, 19, dtype=torch.float32, device=device)
    model.eval()

    # Warm up: 15-20 iterations to fully stabilize torch.compile
    with torch.no_grad(), torch.autocast(device_type=device.type):
        for _ in range(20):
            model(dummy_local)

    if device.type == "cuda":
        torch.cuda.synchronize()

    n_batches = n_positions // batch_size
    total_positions = n_batches * batch_size

    start = time.perf_counter()
    with torch.no_grad(), torch.autocast(device_type=device.type):
        for _ in range(n_batches):
            model(dummy_local)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return {
        "name":              f"NN inference (batch={batch_size})",
        "positions":         total_positions,
        "elapsed_sec":       elapsed,
        "positions_per_sec": total_positions / elapsed,
        "latency_ms":        elapsed / n_batches * 1000,
    }


def benchmark_inference_latency(model: "HexTacToeNet") -> Dict[str, Any]:
    """Single-position latency (worst case for synchronous MCTS)."""
    device = next(model.parameters()).device
    dummy_local  = torch.zeros(1, 18, 19, 19, dtype=torch.float32, device=device)
    model.eval()
    times: List[float] = []

    with torch.no_grad(), torch.autocast(device_type=device.type):
        for _ in range(500):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(dummy_local)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

    times = times[50:]  # discard warm-up
    return {
        "name":    "NN latency (batch=1)",
        "mean_ms": float(np.mean(times)),
        "p50_ms":  float(np.percentile(times, 50)),
        "p99_ms":  float(np.percentile(times, 99)),
    }


def benchmark_replay_buffer(buffer: "RustReplayBuffer") -> Dict[str, Any]:
    """Replay buffer sample speed."""
    raw_iters = 2_000
    aug_iters = 200

    # 1. Raw sampling
    t0 = time.perf_counter()
    for _ in range(raw_iters):
        buffer.sample_batch(256, False)
    elapsed_raw = time.perf_counter() - t0

    # 2. Augmented sampling
    t1 = time.perf_counter()
    for _ in range(aug_iters):
        buffer.sample_batch(256, True)
    elapsed_aug = time.perf_counter() - t1
    
    return {
        "name":          "Replay buffer sample (batch=256)",
        "samples":       raw_iters,
        "elapsed_sec":   elapsed_raw,
        "us_per_sample": elapsed_raw / raw_iters * 1e6,
        "aug_ms":        elapsed_aug / aug_iters * 1000,
    }


def benchmark_gpu_utilisation(model: "HexTacToeNet") -> Dict[str, Any]:
    """Estimate GPU utilisation by measuring compute vs wall-clock time."""
    device = next(model.parameters()).device
    if device.type != "cuda":
        return {
            "name": "GPU utilisation",
            "gpu_util_pct": None,
            "vram_used_gb": None,
            "vram_total_gb": None,
        }

    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)

        # Run inference loop while sampling GPU util.
        model.eval()
        dummy_local = torch.zeros(64, 18, 19, 19, dtype=torch.float32, device=device)
        util_samples: List[float] = []

        t_end = time.monotonic() + 5.0  # 5-second sample window
        with torch.no_grad(), torch.autocast(device_type="cuda"):
            while time.monotonic() < t_end:
                model(dummy_local)
                util_samples.append(
                    float(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
                )
        torch.cuda.synchronize()

        mem        = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_util   = float(np.mean(util_samples)) if util_samples else 0.0
        vram_gb    = float(mem.used) / 1e9
        vram_total = float(mem.total) / 1e9
    except Exception as exc:
        console.print(f"[yellow]GPU util measurement failed: {exc}[/yellow]")
        gpu_util = 0.0
        vram_gb  = 0.0
        vram_total = 0.0

    return {
        "name":          "GPU utilisation",
        "gpu_util_pct":  gpu_util,
        "vram_used_gb":  vram_gb,
        "vram_total_gb": vram_total,
    }


def benchmark_worker_pool(
    model: "HexTacToeNet",
    config: Dict[str, Any],
    device: torch.device,
    duration_sec: int = 15,
    n_workers: int = 4,
    mcts_sims_override: Optional[int] = None,
    quick: bool = False,
) -> Dict[str, Any]:
    """Measure end-to-end self-play throughput in the multiprocess pool."""
    from python.selfplay.pool import WorkerPool
    from native_core import RustReplayBuffer

    if quick:
        duration_sec = min(duration_sec, 5)

    bench_cfg = {
        "mcts": {
            "n_simulations": mcts_sims_override if mcts_sims_override is not None else int(config.get("n_simulations", 30)),
            "c_puct": float(config.get("c_puct", 1.5)),
            "temperature_threshold_ply": int(config.get("temperature_threshold_ply", 30)),
        },
        "selfplay": {
            "n_workers": n_workers,
            "inference_batch_size": int(config.get("inference_batch_size", 32)),
            "inference_max_wait_ms": float(config.get("inference_max_wait_ms", 8.0)),
            "max_moves_per_game": int(config.get("max_moves_per_game", 128)),
        },
    }
    replay = RustReplayBuffer(capacity=25_000)
    try:
        pool = WorkerPool(model, bench_cfg, device, replay, n_workers=n_workers)
    except Exception as exc:
        return {
            "name": "Worker pool throughput",
            "games_completed": 0,
            "positions_pushed": 0,
            "elapsed_sec": 0.0,
            "games_per_hour": 0.0,
            "batch_saturation": 0.0,
            "skipped": 1.0,
            "error": str(exc),
        }

    pool.start()
    t0 = time.perf_counter()
    stop_error: str | None = None
    
    # Live reporting loop
    last_report = t0
    try:
        while time.perf_counter() - t0 < duration_sec:
            time.sleep(1.0)
            now = time.perf_counter()
            if now - last_report >= 5.0:
                elapsed = now - t0
                gph = (pool.games_completed / elapsed) * 3600.0
                console.print(f"    [dim]… {elapsed:.0f}s: {gph:,.1f} games/hour, {pool.positions_pushed} positions[/dim]")
                last_report = now
    finally:
        done = threading.Event()

        def _stop_pool() -> None:
            nonlocal stop_error
            try:
                pool.stop()
            except Exception as exc:  # pragma: no cover - defensive cleanup path
                stop_error = str(exc)
            finally:
                done.set()

        stop_thread = threading.Thread(target=_stop_pool, daemon=True)
        stop_thread.start()
        if not done.wait(10.0):
            stop_error = "pool.stop timeout"

    elapsed = max(time.perf_counter() - t0, 1e-6)
    games_per_hour = (pool.games_completed / elapsed) * 3600.0
    
    # Calculate batch saturation
    server = pool._inference_server
    if server.forward_count > 0:
        batch_saturation = server.total_requests / (server.forward_count * server._batch_size) * 100.0
    else:
        batch_saturation = 0.0

    result = {
        "name":             "Worker pool throughput",
        "games_completed":  pool.games_completed,
        "positions_pushed": pool.positions_pushed,
        "elapsed_sec":      elapsed,
        "games_per_hour":   games_per_hour,
        "batch_saturation": batch_saturation,
    }
    if stop_error is not None:
        result["stop_error"] = stop_error
    return result


# ── Report ────────────────────────────────────────────────────────────────────

# (name, metric_key, target, higher_is_better)
_CHECKS = [
    ("MCTS (CPU only, no NN)",          "sims_per_sec",     150_000,  True),
    ("NN inference (batch=64)",          "positions_per_sec",  8_000,  True),
    ("NN latency (batch=1)",             "mean_ms",                5,  False),
    ("Replay buffer sample (batch=256)", "us_per_sample",       1000,  False),
    ("GPU utilisation",                  "gpu_util_pct",          80,  True),
    ("GPU utilisation",                  "vram_used_gb",           0,  False), # 0 means dynamic target
    ("Worker pool throughput",           "games_per_hour",      1500,  True),
    ("Worker pool throughput",           "batch_saturation",      50,  True),
]


def print_benchmark_report(results: List[Dict[str, Any]]) -> bool:
    """Print a rich pass/fail table. Returns True if all checks pass."""
    table = Table(title="Phase 3.5 / 4 Benchmark Report", show_lines=True)
    table.add_column("Benchmark",    style="bold", min_width=36)
    table.add_column("Result",       justify="right")
    table.add_column("Target",       justify="right", style="dim")
    table.add_column("Status",       justify="center")

    by_name = {r["name"]: r for r in results}
    all_pass = True

    for name, key, target, higher in _CHECKS:
        r   = by_name.get(name, {})
        val = r.get(key)
        if val is None:
            table.add_row(name, "-", "SKIP", "[yellow]SKIP[/yellow]")
            continue

        # Dynamic VRAM Target
        if name == "GPU utilisation" and key == "vram_used_gb":
            total = r.get("vram_total_gb", 0.0)
            limit = total * 0.8
            ok = val <= limit
            status = "[green]PASS[/green]" if ok else "[red]FAIL[/red]"
            table.add_row(
                "VRAM usage",
                f"{val:.1f} / {total:.1f} GB",
                f"≤ {limit:.1f} GB (80%)",
                status
            )
            if not ok: all_pass = False
            continue

        ok = (val >= target) if higher else (val <= target)
        if not ok:
            all_pass = False
        status    = "[green]PASS[/green]" if ok else "[red]FAIL[/red]"
        op        = "≥" if higher else "≤"
        unit      = _unit(key)
        table.add_row(
            f"{name} [{key}]" if key == "batch_saturation" else name,
            f"{val:,.1f}{unit}",
            f"{op} {target:,}{unit}",
            status,
        )

    console.print()
    console.print(table)
    console.print()
    if all_pass:
        console.print("[bold green]All checks PASS — Phase 3.5 / 4 exit criteria met.[/bold green]")
    else:
        console.print("[bold red]Some checks FAILED — profile and optimise before Phase 4.5.[/bold red]")
    console.print()
    return all_pass


def _unit(key: str) -> str:
    if "per_sec" in key: return " units/s"
    if "ms" in key:      return " ms"
    if "us" in key:      return " μs"
    if "pct" in key:     return "%"
    if "saturation" in key: return "%"
    if "hour" in key:    return " /hr"
    return ""


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    import os
    default_workers = os.cpu_count() or 4
    p = argparse.ArgumentParser(description="Phase 3.5 / 4 benchmark harness")
    p.add_argument("--config", default="configs/fast_debug.yaml")
    p.add_argument("--no-compile", action="store_true",
                   help="Skip torch.compile (faster startup for quick checks)")
    p.add_argument("--mcts-sims",  type=int, default=50_000,
                   help="MCTS simulations for throughput benchmark")
    p.add_argument("--pool-workers", type=int, default=default_workers,
                   help="Worker count for self-play throughput benchmark")
    p.add_argument("--pool-duration", type=int, default=15,
                   help="Duration in seconds for worker pool benchmark")
    p.add_argument("--mcts-search-sims", type=int, default=None,
                   help="Override MCTS simulations per move (default from config)")
    p.add_argument("--quick", action="store_true", help="Run shorter benchmarks for fast verification")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Optimization: Enable TensorFloat32 (TF32) for better performance on Ampere+ GPUs
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')
        # Also enable CUDNN autotuner
        torch.backends.cudnn.benchmark = True

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"[bold]Benchmarking on {device}[/bold]")

    from python.model.network import HexTacToeNet, compile_model
    from native_core import RustReplayBuffer

    # ── Build model ──
    model = HexTacToeNet(
        board_size  = int(config.get("board_size",  19)),
        in_channels = 18,
        filters     = int(config.get("filters",     128)),
        res_blocks  = int(config.get("res_blocks",  10)),
    ).to(device)

    if not args.no_compile:
        model = compile_model(model)

    # ── Fill replay buffer with dummy data ──
    buffer = RustReplayBuffer(capacity=100_000)
    for _ in range(10_000):
        buffer.push(
            np.zeros((18, 19, 19), dtype=np.float16),
            np.ones(362, dtype=np.float32) / 362,
            0.0,
        )

    # ── Run benchmarks ──
    results: List[Dict[str, Any]] = []

    with console.status("[bold green]MCTS throughput…"):
        results.append(benchmark_mcts(n_simulations=args.mcts_sims if not args.quick else 5000))

    with console.status("[bold green]NN inference (batch=64)…"):
        results.append(benchmark_inference(model, batch_size=64, n_positions=20000 if not args.quick else 2000))

    with console.status("[bold green]NN latency (batch=1)…"):
        results.append(benchmark_inference_latency(model))

    with console.status("[bold green]Replay buffer…"):
        results.append(benchmark_replay_buffer(buffer))

    with console.status("[bold green]GPU utilisation (5 s)…"):
        results.append(benchmark_gpu_utilisation(model))

    with console.status("[bold green]Worker pool throughput…"):
        results.append(
            benchmark_worker_pool(
                model=model,
                config=config,
                device=device,
                duration_sec=args.pool_duration,
                n_workers=args.pool_workers,
                mcts_sims_override=args.mcts_search_sims,
                quick=args.quick,
            )
        )

    # ── Print individual results ──
    for r in results:
        console.print(f"  [dim]{r['name']}:[/dim]", end=" ")
        if "sims_per_sec" in r:
            console.print(f"{r['sims_per_sec']:,.0f} sim/s")
        elif "positions_per_sec" in r:
            console.print(f"{r['positions_per_sec']:,.0f} pos/s  "
                          f"latency={r['latency_ms']:.1f} ms")
        elif "mean_ms" in r:
            console.print(f"mean={r['mean_ms']:.2f} ms  "
                          f"p50={r['p50_ms']:.2f} ms  "
                          f"p99={r['p99_ms']:.2f} ms")
        elif "us_per_sample" in r:
            console.print(f"{r['us_per_sample']:.1f} μs/sample (raw), {r.get('aug_ms', 0):.2f} ms/sample (augmented)")
        elif "gpu_util_pct" in r:
            pct = r.get("gpu_util_pct")
            vram = r.get("vram_used_gb")
            console.print(
                f"{pct:.0f}%  VRAM={vram:.2f} GB"
                if pct is not None else "N/A"
            )
        elif "games_per_hour" in r:
            console.print(
                f"{r['games_per_hour']:.1f} games/hour  "
                f"games={r['games_completed']}  positions={r['positions_pushed']}",
                end=""
            )
            if "stop_error" in r:
                console.print(f"  [bold red](STOP ERROR: {r['stop_error']})[/bold red]")
            else:
                console.print()

    all_pass = print_benchmark_report(results)
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
