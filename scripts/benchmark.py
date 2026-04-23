#!/usr/bin/env python3
"""
Phase 3.5 / 4 benchmark harness.

Measures the metrics that gate Phase 4.5 and prints a pass/fail table.
Methodology: warm-up per metric, n=5 repeated runs, median +/- IQR summary.

Usage:
    .venv/bin/python scripts/benchmark.py
    .venv/bin/python scripts/benchmark.py --mcts-sims 50000 --pool-duration 60
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*pynvml package is deprecated.*")

import argparse
import json
import statistics
import sys
import time
import threading
from datetime import datetime
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
    from hexo_rl.model.network import HexTacToeNet
    from engine import ReplayBuffer

console = Console()

# ── Statistics ───────────────────────────────────────────────────────────────


def summarise(values: list[float]) -> dict[str, float]:
    """Compute median, IQR, min, max from a list of measurements."""
    arr = sorted(values)
    n = len(arr)
    return {
        "median": statistics.median(arr),
        "p25":    arr[n // 4],
        "p75":    arr[(3 * n) // 4],
        "iqr":    arr[(3 * n) // 4] - arr[n // 4],
        "min":    arr[0],
        "max":    arr[-1],
        "n":      n,
    }


# ── Warm-up helper ───────────────────────────────────────────────────────────


def warmup(operation, duration_sec: float):
    """Run operation repeatedly for duration_sec to stabilise caches/clocks."""
    deadline = time.monotonic() + duration_sec
    while time.monotonic() < deadline:
        operation()


# ── Individual benchmarks ─────────────────────────────────────────────────────


def benchmark_mcts(n_simulations: int = 50_000, sims_per_move: int = 800,
                   n_runs: int = 5, warmup_sec: float = 3.0) -> Dict[str, Any]:
    """CPU-only MCTS throughput (no neural network).

    Measures realistic per-move throughput by running many iterations of
    sims_per_move searches (matching default.yaml mcts.n_simulations) with
    tree reset between each, rather than a single monolithic search.  A single
    large tree exceeds L2 cache and underreports real self-play throughput.
    """
    from engine import Board, MCTSTree  # type: ignore[attr-defined]

    board = Board()
    tree = MCTSTree(c_puct=1.5)
    tree.new_game(board)

    n_iters = max(1, n_simulations // sims_per_move)

    def run_op():
        for _ in range(n_iters):
            tree.run_simulations_cpu_only(n=sims_per_move)
            tree.reset()

    warmup(run_op, warmup_sec)

    rates: list[float] = []
    for _ in range(n_runs):
        tree.reset()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            tree.run_simulations_cpu_only(n=sims_per_move)
            tree.reset()
        elapsed = time.perf_counter() - t0
        total_sims = sims_per_move * n_iters
        rates.append(total_sims / elapsed)

    stats = summarise(rates)
    return {
        "name": "MCTS (CPU only, no NN)",
        "key": "mcts_sim_per_s",
        "stats": stats,
        "value": stats["median"],
    }


def benchmark_inference(model: "HexTacToeNet", n_positions: int = 20_000,
                        batch_size: int = 64, n_runs: int = 5,
                        warmup_sec: float = 3.0) -> Dict[str, Any]:
    """NN throughput in batched evaluation mode."""
    device = next(model.parameters()).device
    dummy_local = torch.zeros(batch_size, 18, 19, 19, dtype=torch.float32, device=device)
    model.eval()
    n_batches = n_positions // batch_size
    total_positions = n_batches * batch_size

    def run_op():
        with torch.no_grad(), torch.autocast(device_type=device.type):
            for _ in range(n_batches):
                model(dummy_local)
        if device.type == "cuda":
            torch.cuda.synchronize()

    warmup(run_op, warmup_sec)

    rates: list[float] = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        run_op()
        elapsed = time.perf_counter() - t0
        rates.append(total_positions / elapsed)

    stats = summarise(rates)
    return {
        "name": f"NN inference (batch={batch_size})",
        "key": "nn_inference_pos_per_s",
        "stats": stats,
        "value": stats["median"],
    }


def benchmark_inference_latency(model: "HexTacToeNet", n_runs: int = 5,
                                warmup_sec: float = 2.0) -> Dict[str, Any]:
    """Single-position latency (worst case for synchronous MCTS)."""
    device = next(model.parameters()).device
    dummy_local = torch.zeros(1, 18, 19, 19, dtype=torch.float32, device=device)
    model.eval()

    def single_inference():
        with torch.no_grad(), torch.autocast(device_type=device.type):
            if device.type == "cuda":
                torch.cuda.synchronize()
            model(dummy_local)
            if device.type == "cuda":
                torch.cuda.synchronize()

    warmup(single_inference, warmup_sec)

    # Each "run" collects 450 latency samples (after discarding 50 warm-up)
    run_means: list[float] = []
    run_p99s: list[float] = []
    for _ in range(n_runs):
        times: list[float] = []
        with torch.no_grad(), torch.autocast(device_type=device.type):
            for _ in range(500):
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                model(dummy_local)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                times.append((time.perf_counter() - t0) * 1000)
        times = times[50:]  # discard per-run warm-up
        run_means.append(float(np.mean(times)))
        run_p99s.append(float(np.percentile(times, 99)))

    stats = summarise(run_means)
    p99_stats = summarise(run_p99s)
    return {
        "name": "NN latency (batch=1)",
        "key": "nn_latency_mean_ms",
        "stats": stats,
        "value": stats["median"],
        "p99_stats": p99_stats,
        "p99_value": p99_stats["median"],
    }


def benchmark_replay_buffer(buffer: "ReplayBuffer", n_runs: int = 5,
                            warmup_sec: float = 2.0) -> Dict[str, Any]:
    """Replay buffer push + sample speed."""
    BATCH = 256
    raw_iters = 2_000
    aug_iters = 500
    push_iters = 10_000

    dummy_state = np.zeros((18, 19, 19), dtype=np.float16)
    dummy_chain = np.zeros((6, 19, 19), dtype=np.float16)
    dummy_policy = np.ones(362, dtype=np.float32) / 362.0
    dummy_own = np.ones(361, dtype=np.uint8)
    dummy_wl  = np.zeros(361, dtype=np.uint8)

    # Warm-up push
    warmup(lambda: buffer.push(dummy_state, dummy_chain, dummy_policy, 0.0, dummy_own, dummy_wl), warmup_sec)

    # Warm-up sample
    warmup(lambda: buffer.sample_batch(BATCH, False), warmup_sec)

    push_rates: list[float] = []
    raw_times: list[float] = []
    aug_times: list[float] = []

    for _ in range(n_runs):
        # Push throughput
        t0 = time.perf_counter()
        for _ in range(push_iters):
            buffer.push(dummy_state, dummy_chain, dummy_policy, 0.0, dummy_own, dummy_wl)
        elapsed_push = time.perf_counter() - t0
        push_rates.append(push_iters / elapsed_push)

        # Raw sampling
        t0 = time.perf_counter()
        for _ in range(raw_iters):
            buffer.sample_batch(BATCH, False)
        elapsed_raw = time.perf_counter() - t0
        raw_times.append(elapsed_raw / raw_iters * 1e6)

        # Augmented sampling
        t0 = time.perf_counter()
        for _ in range(aug_iters):
            buffer.sample_batch(BATCH, True)
        elapsed_aug = time.perf_counter() - t0
        aug_times.append(elapsed_aug / aug_iters * 1e6)

    return {
        "name": "Replay buffer",
        "push": {"key": "buffer_push_per_s", "stats": summarise(push_rates), "value": summarise(push_rates)["median"]},
        "raw":  {"key": "buffer_sample_raw_us", "stats": summarise(raw_times), "value": summarise(raw_times)["median"]},
        "aug":  {"key": "buffer_sample_aug_us", "stats": summarise(aug_times), "value": summarise(aug_times)["median"]},
    }


def benchmark_gpu_utilisation(model: "HexTacToeNet", n_runs: int = 5) -> Dict[str, Any]:
    """Estimate GPU utilisation by measuring compute vs wall-clock time."""
    device = next(model.parameters()).device
    if device.type != "cuda":
        return {
            "name": "GPU utilisation",
            "gpu": {"key": "gpu_util_pct", "stats": None, "value": None},
            "vram": {"key": "vram_used_gb", "stats": None, "value": None},
            "vram_total": None,
        }

    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        model.eval()
        dummy_local = torch.zeros(64, 18, 19, 19, dtype=torch.float32, device=device)

        util_runs: list[float] = []
        vram_runs: list[float] = []

        for _ in range(n_runs):
            util_samples: list[float] = []
            t_end = time.monotonic() + 5.0
            with torch.no_grad(), torch.autocast(device_type="cuda"):
                while time.monotonic() < t_end:
                    model(dummy_local)
                    util_samples.append(
                        float(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
                    )
            torch.cuda.synchronize()
            util_runs.append(float(np.mean(util_samples)) if util_samples else 0.0)
            vram_runs.append(float(torch.cuda.max_memory_allocated()) / 1e9)

        vram_total = float(pynvml.nvmlDeviceGetMemoryInfo(handle).total) / 1e9
    except Exception as exc:
        console.print(f"[yellow]GPU util measurement failed: {exc}[/yellow]")
        return {
            "name": "GPU utilisation",
            "gpu": {"key": "gpu_util_pct", "stats": None, "value": None},
            "vram": {"key": "vram_used_gb", "stats": None, "value": None},
            "vram_total": None,
        }

    return {
        "name": "GPU utilisation",
        "gpu": {"key": "gpu_util_pct", "stats": summarise(util_runs), "value": summarise(util_runs)["median"]},
        "vram": {"key": "vram_used_gb", "stats": summarise(vram_runs), "value": summarise(vram_runs)["median"]},
        "vram_total": vram_total,
    }


def benchmark_worker_pool(
    model: "HexTacToeNet",
    config: Dict[str, Any],
    device: torch.device,
    duration_sec: int = 15,
    n_workers: int = 4,
    mcts_sims_override: Optional[int] = None,
    worker_sims: int = 200,
    bench_max_moves: int = 128,
    n_runs: int = 5,
    warmup_sec: float = 10.0,
) -> Dict[str, Any]:
    """Measure end-to-end self-play throughput in the multiprocess pool.

    ``worker_sims`` sets per-move simulation budget (default 200).
    ``bench_max_moves`` caps game length so games complete within the bench
    window on desktop hardware (default 100; production config uses 200+).
    Pass --worker-sims / --bench-max-moves to override at the CLI level.
    """
    from hexo_rl.selfplay.pool import WorkerPool
    from engine import ReplayBuffer

    _sp = config.get("selfplay", {})
    _mcts = config.get("mcts", {})
    # Determine per-move sim count: explicit CLI override > bench default.
    # Do NOT fall back to production config (400+ sims): games would not complete
    # within the measurement window on desktop hardware.
    effective_sims = mcts_sims_override if mcts_sims_override is not None else worker_sims
    bench_cfg = {
        "mcts": {
            "n_simulations": effective_sims,
            "c_puct": float(_mcts.get("c_puct", config.get("c_puct", 1.5))),
            "temperature_threshold_ply": int(_mcts.get("temperature_threshold_ply", config.get("temperature_threshold_ply", 30))),
            "fpu_reduction": float(_mcts.get("fpu_reduction", 0.25)),
            "quiescence_enabled": bool(_mcts.get("quiescence_enabled", True)),
            "quiescence_blend_2": float(_mcts.get("quiescence_blend_2", 0.3)),
            "dirichlet_alpha": float(_mcts.get("dirichlet_alpha", 0.3)),
            "epsilon": float(_mcts.get("epsilon", 0.25)),
            "dirichlet_enabled": bool(_mcts.get("dirichlet_enabled", True)),
        },
        "selfplay": {
            "n_workers": n_workers,
            "inference_batch_size": int(_sp.get("inference_batch_size", config.get("inference_batch_size", 32))),
            "inference_max_wait_ms": float(_sp.get("inference_max_wait_ms", config.get("inference_max_wait_ms", 8.0))),
            "max_moves_per_game": bench_max_moves,
            "leaf_batch_size": int(_sp.get("leaf_batch_size", 8)),
            # Forward Gumbel / completed-Q flags so the worker-pool bench
            # actually exercises the variant's root search path.
            "gumbel_mcts": bool(_sp.get("gumbel_mcts", False)),
            "gumbel_m": int(_sp.get("gumbel_m", 16)),
            "gumbel_explore_moves": int(_sp.get("gumbel_explore_moves", 10)),
            "completed_q_values": bool(_sp.get("completed_q_values", False)),
            "c_visit": float(_sp.get("c_visit", 50.0)),
            "c_scale": float(_sp.get("c_scale", 1.0)),
            # pool.py enforces playout_cap.fast_sims as a required key (no silent
            # defaults). The benchmark doesn't care about fast-game mixing — set
            # fast_prob=0.0 so fast_sims is never actually consumed, and pass
            # standard_sims=0 so SelfPlayRunner falls back to mcts.n_simulations
            # (the value this bench used before c915004 added the required-key
            # check). Preserves apples-to-apples comparison against the pre-c915004
            # worker-throughput baseline.
            "playout_cap": {
                "fast_prob": 0.0,
                "fast_sims": 64,
                "standard_sims": 0,
            },
        },
    }

    # CUDA warm-up: force PyTorch to JIT-compile autocast kernels before workers
    # start. Without this, the first inference call triggers 90-120s of kernel
    # compilation, causing "0 games" for the entire warm-up window. Same fix
    # applied in training/loop.py (commit 79fd415).
    if device.type == "cuda":
        _board_size = int(getattr(model, "board_size", 19))
        _in_ch = int(config.get("in_channels", config.get("model", {}).get("in_channels", 18)))
        with torch.no_grad():
            with torch.autocast(device_type="cuda"):
                _dummy = torch.zeros(1, _in_ch, _board_size, _board_size, device=device)
                model(_dummy)
        torch.cuda.synchronize()

    # Run one pool for the entire benchmark: warm-up + N measurement windows.
    # Previous approach created separate pools per run, so each run had its own
    # ~20-25s cold-start (no games completing), biasing the 60s window low.
    replay = ReplayBuffer(capacity=100_000)
    try:
        pool = WorkerPool(model, bench_cfg, device, replay, n_workers=n_workers)
    except Exception as exc:
        console.print(f"[yellow]Worker pool init failed: {exc}[/yellow]")
        return {
            "name": "Worker pool throughput",
            "gph":  {"key": "worker_games_per_hr", "stats": summarise([0.0]), "value": 0.0},
            "pph":  {"key": "worker_pos_per_hr", "stats": summarise([0.0]), "value": 0.0},
            "bat":  {"key": "worker_batch_fill_pct", "stats": summarise([0.0]), "value": 0.0},
        }

    pool.start()
    stop_error: str | None = None

    try:
        # Warm-up phase: let games reach steady state before measuring
        if warmup_sec > 0:
            console.print(f"    [dim]Worker pool warm-up ({warmup_sec:.0f}s)...[/dim]")
            t_warmup = time.perf_counter()
            last_report = t_warmup
            while time.perf_counter() - t_warmup < warmup_sec:
                time.sleep(1.0)
                now = time.perf_counter()
                if now - last_report >= 5.0:
                    elapsed = now - t_warmup
                    console.print(f"    [dim]... warm-up {elapsed:.0f}s: {pool.games_completed} games, {pool.positions_pushed} positions[/dim]")
                    last_report = now

        # Measurement runs: snapshot counters, wait, compute delta
        gph_runs: list[float] = []
        pph_runs: list[float] = []
        bat_runs: list[float] = []

        for i in range(n_runs):
            console.print(f"    [dim]Worker pool run {i+1}/{n_runs} ({duration_sec}s)...[/dim]")
            # Snapshot counters at start of measurement window
            games_start = pool.games_completed
            pos_start = pool.positions_pushed
            server = pool._inference_server
            fwd_start = server.forward_count
            req_start = server.total_requests

            t0 = time.perf_counter()
            last_report = t0
            while time.perf_counter() - t0 < duration_sec:
                time.sleep(1.0)
                now = time.perf_counter()
                if now - last_report >= 5.0:
                    elapsed = now - t0
                    delta_pos = pool.positions_pushed - pos_start
                    delta_games = pool.games_completed - games_start
                    gph = (delta_games / elapsed) * 3600.0
                    console.print(f"    [dim]... {elapsed:.0f}s: {gph:,.1f} games/hour, {delta_pos} positions[/dim]")
                    last_report = now

            elapsed = max(time.perf_counter() - t0, 1e-6)
            delta_games = pool.games_completed - games_start
            delta_pos = pool.positions_pushed - pos_start
            delta_fwd = server.forward_count - fwd_start
            delta_req = server.total_requests - req_start

            gph_runs.append((delta_games / elapsed) * 3600.0)
            pph_runs.append((delta_pos / elapsed) * 3600.0)

            if delta_fwd > 0:
                bat_runs.append(delta_req / (delta_fwd * server._batch_size) * 100.0)
            else:
                bat_runs.append(0.0)

    finally:
        done = threading.Event()

        def _stop_pool() -> None:
            nonlocal stop_error
            try:
                pool.stop()
            except Exception as exc:
                stop_error = str(exc)
            finally:
                done.set()

        stop_thread = threading.Thread(target=_stop_pool, daemon=True)
        stop_thread.start()
        if not done.wait(10.0):
            stop_error = "pool.stop timeout"

    return {
        "name": "Worker pool throughput",
        "gph":  {"key": "worker_games_per_hr", "stats": summarise(gph_runs), "value": summarise(gph_runs)["median"]},
        "pph":  {"key": "worker_pos_per_hr", "stats": summarise(pph_runs), "value": summarise(pph_runs)["median"]},
        "bat":  {"key": "worker_batch_fill_pct", "stats": summarise(bat_runs), "value": summarise(bat_runs)["median"]},
    }


# ── Report ────────────────────────────────────────────────────────────────────

# (row_label, result_name, sub_key, metric_key_in_stats_or_value, target, higher_is_better)
_CHECKS_CUDA: list[tuple[str, str, str | None, str, float, bool]] = [
    ("MCTS sim/s (CPU, no NN)",           "MCTS (CPU only, no NN)",  None,    "value",   26_000,   True),
    # §102 rebaseline 2026-04-17: 9.8k→7.7k sustained driver/boost-clock drift (same basket as §72).
    # Floor at observed × 0.85 so alarms fire on real regressions, not one-off drift.
    ("NN inference batch=64 pos/s",       "NN inference (batch=64)", None,    "value",    6_500,   True),
    ("NN latency batch=1 mean ms",        "NN latency (batch=1)",    None,    "value",      3.5,   False),
    # §102 rebaseline 2026-04-17: 762k→618k sustained drift; observed × 0.85 = 525k floor.
    ("Buffer push pos/s",                 "Replay buffer",           "push",  "value",  525_000,   True),
    # §113 2026-04-22: cda9dde always-on dedup adds ~33 µs (HashSet alloc + 256 game_id lookups);
    # correctness-required. push.rs transmute fix recovered push + aug regressions. 1_500→1_550.
    ("Buffer sample raw us/batch",        "Replay buffer",           "raw",   "value",    1_550,   False),
    # §98 rebaseline 2026-04-16: 18ch split scatter (state + chain) raises aug latency; 1400→1800
    # §102 2026-04-17: 1,241 µs observed (improved vs §98's 1,663); do not tighten on one run.
    ("Buffer sample augmented us/batch",  "Replay buffer",           "aug",   "value",    1_800,   False),
    ("GPU utilisation %",                 "GPU utilisation",         "gpu",   "value",       85,   True),
    ("VRAM usage GB",                     "GPU utilisation",         "vram",  "value",        0,   False),  # dynamic
    # §102 rebaseline 2026-04-17: 90s warmup fixed §98 warmup artifact (IQR 188% → 5.7%).
    # New floor = observed 167,755 × 0.85 = 142,592 → 142,000. PROVISIONAL until second stable run confirms.
    ("Worker throughput pos/hr",          "Worker pool throughput",  "pph",   "value",  142_000,   True),
    ("Worker batch fill %",              "Worker pool throughput",  "bat",   "value",       84,   True),
]

_CHECKS_MPS: list[tuple[str, str, str | None, str, float, bool]] = [
    ("MCTS sim/s (CPU, no NN)",           "MCTS (CPU only, no NN)",  None,    "value",   26_000,   True),
    ("NN inference batch=64 pos/s",       "NN inference (batch=64)", None,    "value",    3_000,   True),
    ("NN latency batch=1 mean ms",        "NN latency (batch=1)",    None,    "value",      8.0,   False),
    ("Buffer push pos/s",                 "Replay buffer",           "push",  "value",  630_000,   True),
    ("Buffer sample raw us/batch",        "Replay buffer",           "raw",   "value",    1_500,   False),
    ("Buffer sample augmented us/batch",  "Replay buffer",           "aug",   "value",    1_800,   False),
    # GPU util / VRAM omitted — pynvml not available on macOS; benchmark_gpu_utilisation returns None
    ("Worker throughput pos/hr",          "Worker pool throughput",  "pph",   "value",  200_000,   True),
    ("Worker batch fill %",              "Worker pool throughput",  "bat",   "value",       84,   True),
]

_CHECKS_CPU: list[tuple[str, str, str | None, str, float, bool]] = [
    ("MCTS sim/s (CPU, no NN)",           "MCTS (CPU only, no NN)",  None,    "value",   20_000,   True),
    ("NN inference batch=64 pos/s",       "NN inference (batch=64)", None,    "value",      800,   True),
    ("NN latency batch=1 mean ms",        "NN latency (batch=1)",    None,    "value",     20.0,   False),
    ("Buffer push pos/s",                 "Replay buffer",           "push",  "value",  630_000,   True),
    ("Buffer sample raw us/batch",        "Replay buffer",           "raw",   "value",    1_500,   False),
    ("Buffer sample augmented us/batch",  "Replay buffer",           "aug",   "value",    1_800,   False),
    # GPU util / VRAM omitted — no GPU on CPU-only systems
    ("Worker throughput pos/hr",          "Worker pool throughput",  "pph",   "value",   80_000,   True),
    ("Worker batch fill %",              "Worker pool throughput",  "bat",   "value",       84,   True),
]


def _build_checks(
    device: "torch.device",
) -> "list[tuple[str, str, str | None, str, float, bool]]":
    """Return the appropriate check list for the given device type."""
    import torch as _torch
    if device.type == "cuda":
        return _CHECKS_CUDA
    if device.type == "mps":
        return _CHECKS_MPS
    return _CHECKS_CPU


def _get_metric(by_name: dict, result_name: str, sub_key: str | None) -> dict | None:
    r = by_name.get(result_name)
    if r is None:
        return None
    if sub_key is not None:
        return r.get(sub_key)
    return r


def _fmt_range(stats: dict) -> str:
    """Format min-max range with K/M suffixes."""
    lo, hi = stats["min"], stats["max"]
    def _short(v: float) -> str:
        if abs(v) >= 1_000_000:
            return f"{v/1e6:.2f}M"
        if abs(v) >= 1_000:
            return f"{v/1e3:.1f}k"
        return f"{v:.1f}"
    return f"{_short(lo)}-{_short(hi)}"


def print_benchmark_report(results: List[Dict[str, Any]],
                           n_runs: int, warmup_note: str,
                           checks: "list[tuple] | None" = None) -> bool:
    """Print a rich pass/fail table. Returns True if all checks pass."""
    if checks is None:
        checks = _CHECKS_CUDA
    table = Table(title="Benchmark Report", show_lines=True)
    table.add_column("Metric", style="bold", min_width=36)
    table.add_column("Median", justify="right")
    table.add_column("IQR", justify="right")
    table.add_column("Range", justify="right")
    table.add_column("Target", justify="right", style="dim")
    table.add_column("", justify="center")

    by_name = {r["name"]: r for r in results}
    all_pass = True

    for row_label, result_name, sub_key, val_key, target, higher in checks:
        metric = _get_metric(by_name, result_name, sub_key)
        if metric is None or metric.get("value") is None:
            table.add_row(row_label, "-", "-", "-", "SKIP", "[yellow]SKIP[/yellow]")
            continue

        val = metric["value"]
        stats = metric.get("stats", {})

        # Dynamic VRAM target
        if row_label == "VRAM usage GB":
            gpu_result = by_name.get("GPU utilisation", {})
            total = gpu_result.get("vram_total", 0.0)
            if total is None or total == 0:
                table.add_row(row_label, "-", "-", "-", "SKIP", "[yellow]SKIP[/yellow]")
                continue
            limit = total * 0.8
            ok = val <= limit
            status = "[green]PASS[/green]" if ok else "[red]FAIL[/red]"
            if not ok:
                all_pass = False
            iqr_str = f"+/-{stats.get('iqr', 0):.2f}" if stats else "-"
            range_str = _fmt_range(stats) if stats else "-"
            table.add_row(row_label, f"{val:.2f}/{total:.1f}", iqr_str, range_str,
                          f"<= {limit:.1f} (80%)", status)
            continue

        ok = (val >= target) if higher else (val <= target)
        if not ok:
            all_pass = False
        status = "[green]PASS[/green]" if ok else "[red]FAIL[/red]"
        op = ">=" if higher else "<="

        # Format values
        def _fmt_val(v: float) -> str:
            if abs(v) >= 10_000:
                return f"{v:,.0f}"
            if abs(v) >= 100:
                return f"{v:,.1f}"
            return f"{v:,.2f}"

        iqr_str = f"+/-{_fmt_val(stats.get('iqr', 0))}" if stats else "-"
        range_str = _fmt_range(stats) if stats else "-"

        table.add_row(
            row_label,
            _fmt_val(val),
            iqr_str,
            range_str,
            f"{op} {target:,.0f}",
            status,
        )

    console.print()
    console.print(table)
    console.print()

    console.print(f"  Warm-up: {warmup_note} | Runs: n={n_runs} | Summary: median +/- IQR")
    console.print()

    if all_pass:
        console.print("[bold green]All checks PASS -- Phase 4.5 exit criteria met.[/bold green]")
    else:
        console.print("[bold red]Some checks FAILED -- profile and optimise before Phase 4.5.[/bold red]")
    console.print()
    return all_pass


def write_json_report(results: List[Dict[str, Any]], n_runs: int,
                      checks: "list[tuple] | None" = None) -> Path:
    """Write structured JSON report to reports/benchmarks/."""
    if checks is None:
        checks = _CHECKS_CUDA
    report_dir = ROOT / "reports" / "benchmarks"
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    report_path = report_dir / f"{timestamp}.json"

    by_name = {r["name"]: r for r in results}
    metrics: dict[str, Any] = {}
    targets_met: dict[str, bool] = {}

    for row_label, result_name, sub_key, val_key, target, higher in checks:
        metric = _get_metric(by_name, result_name, sub_key)
        if metric is None or metric.get("value") is None:
            continue
        key = metric.get("key", row_label)
        val = metric["value"]
        stats = metric.get("stats", {})

        entry = {k: v for k, v in stats.items()} if stats else {}
        entry["median"] = val
        metrics[key] = entry

        if row_label == "VRAM usage GB":
            gpu_result = by_name.get("GPU utilisation", {})
            total = gpu_result.get("vram_total", 0.0)
            limit = (total or 1) * 0.8
            targets_met[key] = val <= limit
        else:
            targets_met[key] = (val >= target) if higher else (val <= target)

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "n_runs": n_runs,
        "metrics": metrics,
        "targets_met": targets_met,
        "all_targets_met": all(targets_met.values()),
    }

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    console.print(f"  JSON report: {report_path}")
    return report_path


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    import os
    default_workers = os.cpu_count() or 4
    p = argparse.ArgumentParser(description="Phase 3.5 / 4 benchmark harness")
    p.add_argument("--config", default=None,
                   help="Optional override config applied on top of base configs")
    p.add_argument("--no-compile", action="store_true",
                   help="Skip torch.compile (faster startup for quick checks)")
    p.add_argument("--mcts-sims", type=int, default=50_000,
                   help="MCTS simulations for throughput benchmark")
    p.add_argument("--pool-workers", type=int, default=default_workers,
                   help="Worker count for self-play throughput benchmark")
    p.add_argument("--pool-duration", type=int, default=60,
                   help="Duration in seconds per measurement run for worker pool benchmark")
    p.add_argument("--mcts-search-sims", type=int, default=None,
                   help="Override MCTS simulations per move for MCTS CPU bench (default from config)")
    p.add_argument("--worker-sims", type=int, default=200,
                   help="Simulations per move for the worker-pool bench (default 200). "
                        "Keep low enough that games complete within the bench window on target hardware.")
    p.add_argument("--bench-max-moves", type=int, default=128,
                   help="Max plies per game for the worker-pool bench (default 128). "
                        "Production config uses 200; lower value ensures games finish within bench window.")
    p.add_argument("--n-runs", type=int, default=5,
                   help="Number of measurement repeats per metric (default 5)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    n_runs = int(args.n_runs)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    from hexo_rl.utils.config import load_config
    _BASE_CONFIGS = [
        "configs/model.yaml",
        "configs/training.yaml",
        "configs/selfplay.yaml",
    ]
    if args.config:
        config = load_config(*_BASE_CONFIGS, args.config)
    else:
        config = load_config(*_BASE_CONFIGS)

    # Per-host TF32 configuration (§117). Applies backend flags once.
    from hexo_rl.model.tf32 import resolve_and_apply as _tf32_resolve_and_apply
    _tf32_resolved = _tf32_resolve_and_apply(config)
    console.print(
        f"[dim]TF32: matmul={_tf32_resolved['tf32_matmul']} "
        f"cudnn={_tf32_resolved['tf32_cudnn']} "
        f"cap={_tf32_resolved.get('compute_capability')}[/dim]"
    )

    from hexo_rl.utils.device import best_device
    device = best_device()
    checks = _build_checks(device)
    console.print(f"[bold]Benchmarking on {device} | n={n_runs}[/bold]")

    from hexo_rl.model.network import HexTacToeNet, compile_model
    from engine import ReplayBuffer

    # Reset peak memory tracking before any GPU allocations so
    # benchmark_gpu_utilisation captures the full process footprint.
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # Build model — read from nested model section (new style) with flat fallback
    model_cfg = config.get("model", {})
    model = HexTacToeNet(
        board_size=int(model_cfg.get("board_size", config.get("board_size", 19))),
        in_channels=int(model_cfg.get("in_channels", config.get("in_channels", 18))),
        filters=int(model_cfg.get("filters", config.get("filters", 128))),
        res_blocks=int(model_cfg.get("res_blocks", config.get("res_blocks", 12))),
    ).to(device)

    if not args.no_compile:
        model = compile_model(model)

    # Fill replay buffer with dummy data
    buffer = ReplayBuffer(capacity=100_000)
    _bm_own = np.ones(361, dtype=np.uint8)
    _bm_wl  = np.zeros(361, dtype=np.uint8)
    _bm_chain = np.zeros((6, 19, 19), dtype=np.float16)
    for _ in range(10_000):
        buffer.push(
            np.zeros((18, 19, 19), dtype=np.float16),
            _bm_chain,
            np.ones(362, dtype=np.float32) / 362,
            0.0,
            _bm_own, _bm_wl,
        )

    # Warm-up durations
    warmup_mcts = 3.0
    warmup_nn = 3.0
    warmup_latency = 2.0
    warmup_buffer = 2.0
    warmup_worker = 90.0

    try:
        # Run benchmarks
        results: List[Dict[str, Any]] = []

        with console.status(f"[bold green]MCTS throughput (n={n_runs})..."):
            results.append(benchmark_mcts(
                n_simulations=args.mcts_sims,
                n_runs=n_runs, warmup_sec=warmup_mcts))

        with console.status(f"[bold green]NN inference batch=64 (n={n_runs})..."):
            results.append(benchmark_inference(
                model, batch_size=64,
                n_positions=20000,
                n_runs=n_runs, warmup_sec=warmup_nn))

        with console.status(f"[bold green]NN latency batch=1 (n={n_runs})..."):
            results.append(benchmark_inference_latency(
                model, n_runs=n_runs, warmup_sec=warmup_latency))

        with console.status(f"[bold green]Replay buffer (n={n_runs})..."):
            results.append(benchmark_replay_buffer(
                buffer, n_runs=n_runs, warmup_sec=warmup_buffer))

        with console.status(f"[bold green]GPU utilisation (n={n_runs})..."):
            results.append(benchmark_gpu_utilisation(model, n_runs=n_runs))

        with console.status(f"[bold green]Worker pool throughput (n={n_runs})..."):
            results.append(benchmark_worker_pool(
                model=model,
                config=config,
                device=device,
                duration_sec=args.pool_duration,
                n_workers=args.pool_workers,
                mcts_sims_override=args.mcts_search_sims,
                worker_sims=args.worker_sims,
                bench_max_moves=args.bench_max_moves,
                n_runs=n_runs,
                warmup_sec=warmup_worker,
            ))

        # Print per-metric summaries
        for r in results:
            name = r["name"]
            if "stats" in r:
                s = r["stats"]
                console.print(f"  [dim]{name}:[/dim] median={s['median']:,.1f}  IQR=+/-{s['iqr']:,.1f}  [{_fmt_range(s)}]  n={s['n']}")
            elif "push" in r:
                for sub_name, sub in [("push", r["push"]), ("raw", r["raw"]), ("aug", r["aug"])]:
                    s = sub["stats"]
                    console.print(f"  [dim]{name} {sub_name}:[/dim] median={s['median']:,.1f}  IQR=+/-{s['iqr']:,.1f}  [{_fmt_range(s)}]  n={s['n']}")
            elif "gpu" in r:
                for sub_name, sub in [("util%", r["gpu"]), ("vram", r["vram"])]:
                    if sub.get("stats"):
                        s = sub["stats"]
                        console.print(f"  [dim]{name} {sub_name}:[/dim] median={s['median']:,.1f}  IQR=+/-{s['iqr']:,.1f}  [{_fmt_range(s)}]  n={s['n']}")
            elif "pph" in r:
                for sub_name, sub in [("pos/hr", r["pph"]), ("games/hr", r["gph"]), ("batch%", r["bat"])]:
                    s = sub["stats"]
                    console.print(f"  [dim]{name} {sub_name}:[/dim] median={s['median']:,.1f}  IQR=+/-{s['iqr']:,.1f}  [{_fmt_range(s)}]  n={s['n']}")

        warmup_note = f"{warmup_mcts:.0f}s MCTS / {warmup_nn:.0f}s NN / {warmup_buffer:.0f}s buffer / {warmup_worker:.0f}s worker"
        all_pass = print_benchmark_report(results, n_runs, warmup_note, checks=checks)
        write_json_report(results, n_runs, checks=checks)
        sys.exit(0 if all_pass else 1)

    finally:
        pass


if __name__ == "__main__":
    main()
