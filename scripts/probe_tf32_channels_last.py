#!/usr/bin/env python3
"""
TF32 + channels_last probe — four-arm matrix on NN inference + train step.
Probe date: 2026-04-23

Arms:
    A baseline            — no TF32, default memory format
    B tf32                — TF32 matmul on
    C cl                  — channels_last (model + inputs)
    D both                — TF32 on + channels_last

Per arm metrics (median of N_RUNS):
    NN inference pos/s @ BATCH_INFER=64
    NN latency mean ms @ batch=1
    Train step ms @ BATCH_TRAIN=128 (synthetic fwd+bwd+opt)
    Max VRAM (process, GB)
    GPU util (mean %, pynvml poll at 50ms)
    Batch fill % (N/A for synthetic — fixed batches — noted explicitly)

Output: reports/investigations/tf32_channels_last_<date>/
    report.md    — 4×1 arm table (per-host) + verdict per arm
    data.json    — raw measurements
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
OUTDIR = REPO_ROOT / "reports" / "investigations" / "tf32_channels_last_20260423"
OUTDIR.mkdir(parents=True, exist_ok=True)

from hexo_rl.model.network import HexTacToeNet  # noqa: E402

# ── Config ─────────────────────────────────────────────────────────────────────
BATCH_INFER    = 64
BATCH_TRAIN    = 128
N_POSITIONS    = 20_000
N_TRAIN_STEPS  = 100
N_RUNS         = 5
WARMUP_SEC     = 4.0
WARMUP_LAT_SEC = 3.0
LAT_ITERS      = 500
LAT_WARMUP     = 50
IN_CH, H, W    = 18, 19, 19

ARMS = [
    ("A_baseline", {"tf32": False, "channels_last": False}),
    ("B_tf32",     {"tf32": True,  "channels_last": False}),
    ("C_cl",       {"tf32": False, "channels_last": True}),
    ("D_both",     {"tf32": True,  "channels_last": True}),
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def set_tf32(on: bool) -> None:
    torch.backends.cuda.matmul.allow_tf32 = on
    torch.backends.cudnn.allow_tf32 = on


def build_model(device: torch.device, channels_last: bool) -> HexTacToeNet:
    m = HexTacToeNet(board_size=W, in_channels=IN_CH, filters=128, res_blocks=12).to(device)
    m.eval()
    if channels_last:
        m = m.to(memory_format=torch.channels_last)
    return m


def make_input(device: torch.device, batch: int, channels_last: bool,
               dtype: torch.dtype = torch.float32) -> torch.Tensor:
    t = torch.zeros(batch, IN_CH, H, W, dtype=dtype, device=device)
    if channels_last:
        t = t.contiguous(memory_format=torch.channels_last)
    return t


def warmup_fn(fn, sec: float) -> None:
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < sec:
        fn()


class GpuUtilSampler:
    """Background pynvml sampler of GPU utilization at 50ms intervals."""

    def __init__(self, device_index: int = 0, interval_s: float = 0.05) -> None:
        import pynvml
        pynvml.nvmlInit()
        self.h = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        self.pynvml = pynvml
        self.interval = interval_s
        self._samples: list[int] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                u = self.pynvml.nvmlDeviceGetUtilizationRates(self.h)
                self._samples.append(int(u.gpu))
            except Exception:
                pass
            time.sleep(self.interval)

    def start(self) -> None:
        self._samples.clear()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> dict:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        if not self._samples:
            return {"mean": float("nan"), "p50": float("nan"), "p99": float("nan"), "n": 0}
        arr = np.asarray(self._samples)
        return {
            "mean": float(arr.mean()),
            "p50":  float(np.percentile(arr, 50)),
            "p99":  float(np.percentile(arr, 99)),
            "n":    int(arr.size),
        }


def bench_throughput(model: nn.Module, device: torch.device,
                     channels_last: bool,
                     sampler: GpuUtilSampler | None = None) -> dict:
    dummy = make_input(device, BATCH_INFER, channels_last)
    model.eval()
    n_batches = N_POSITIONS // BATCH_INFER

    def run_op():
        with torch.no_grad(), torch.autocast(device_type=device.type):
            for _ in range(n_batches):
                model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()

    warmup_fn(run_op, WARMUP_SEC)
    rates = []
    util_runs: list[dict] = []
    for _ in range(N_RUNS):
        if sampler is not None:
            sampler.start()
        t0 = time.perf_counter()
        run_op()
        elapsed = time.perf_counter() - t0
        if sampler is not None:
            util_runs.append(sampler.stop())
        rates.append((n_batches * BATCH_INFER) / elapsed)

    rates.sort()
    median = rates[N_RUNS // 2]
    util_mean_runs = [u["mean"] for u in util_runs] if util_runs else []
    util_mean_runs.sort()
    util_median = util_mean_runs[N_RUNS // 2] if util_mean_runs else float("nan")
    return {
        "median": median, "min": rates[0], "max": rates[-1], "all": rates,
        "gpu_util_mean_median": util_median,
        "gpu_util_runs": util_runs,
    }


def bench_latency(model: nn.Module, device: torch.device,
                  channels_last: bool) -> dict:
    dummy = make_input(device, 1, channels_last)
    model.eval()

    def single():
        with torch.no_grad(), torch.autocast(device_type=device.type):
            if device.type == "cuda":
                torch.cuda.synchronize()
            model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()

    warmup_fn(single, WARMUP_LAT_SEC)

    run_means, run_p99s = [], []
    for _ in range(N_RUNS):
        times = []
        with torch.no_grad(), torch.autocast(device_type=device.type):
            for _ in range(LAT_ITERS):
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                model(dummy)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                times.append((time.perf_counter() - t0) * 1_000)
        times = times[LAT_WARMUP:]
        run_means.append(float(np.mean(times)))
        run_p99s.append(float(np.percentile(times, 99)))

    run_means.sort()
    run_p99s.sort()
    return {
        "mean_ms_median": run_means[N_RUNS // 2],
        "p99_ms_median":  run_p99s[N_RUNS // 2],
        "mean_ms_all":    run_means,
    }


def bench_train_step(device: torch.device, channels_last: bool,
                     sampler: GpuUtilSampler | None = None) -> dict:
    """Synthetic train step: fwd (AMP) + bwd + optimizer step on fixed random inputs.

    Builds a fresh training-mode model + AdamW + GradScaler per call so we're
    not contaminated by the inference model (eval mode, grad off).
    """
    model = HexTacToeNet(board_size=W, in_channels=IN_CH, filters=128, res_blocks=12).to(device)
    model.train()
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scaler = torch.amp.GradScaler(device=device.type, enabled=True)

    states = make_input(device, BATCH_TRAIN, channels_last, dtype=torch.float32)
    # Synthetic targets.
    policies = torch.zeros(BATCH_TRAIN, H * W + 1, device=device)
    policies[:, 0] = 1.0
    outcomes = torch.zeros(BATCH_TRAIN, 1, device=device)

    def step_once() -> None:
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=True):
            log_p, value, v_logit = model(states)
            # Approximate the real loss composition without aux heads.
            p_loss = -(policies * log_p).sum(dim=1).mean()
            v_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                v_logit, (outcomes + 1) / 2
            )
            loss = p_loss + v_loss
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    # Warmup.
    for _ in range(20):
        step_once()
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Runs.
    ms_per_step_runs: list[float] = []
    util_runs: list[dict] = []
    for _ in range(N_RUNS):
        if sampler is not None:
            sampler.start()
        t0 = time.perf_counter()
        for _ in range(N_TRAIN_STEPS):
            step_once()
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        if sampler is not None:
            util_runs.append(sampler.stop())
        ms_per_step_runs.append(elapsed / N_TRAIN_STEPS * 1000.0)

    # VRAM peak after train path.
    vram_peak_gb = torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else 0.0

    # Release.
    del model, opt, scaler, states, policies, outcomes
    torch.cuda.empty_cache() if device.type == "cuda" else None

    ms_per_step_runs.sort()
    util_mean_runs = [u["mean"] for u in util_runs] if util_runs else []
    util_mean_runs.sort()
    return {
        "ms_per_step_median": ms_per_step_runs[N_RUNS // 2],
        "ms_per_step_all":    ms_per_step_runs,
        "vram_peak_gb":       vram_peak_gb,
        "gpu_util_mean_median": util_mean_runs[N_RUNS // 2] if util_mean_runs else float("nan"),
        "gpu_util_runs": util_runs,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def host_info(device: torch.device) -> dict:
    info = {
        "python":  sys.version.split()[0],
        "pytorch": torch.__version__,
        "cuda":    torch.version.cuda,
        "device":  device.type,
    }
    if device.type == "cuda":
        info["gpu"] = torch.cuda.get_device_name(0)
        major, minor = torch.cuda.get_device_capability(0)
        info["compute_cap"] = f"sm_{major}{minor}"
        info["driver"] = torch.cuda.driver_version() if hasattr(torch.cuda, "driver_version") else "n/a"
    return info


def run_arm(arm_name: str, cfg: dict, device: torch.device) -> dict:
    print(f"\n=== Arm {arm_name}  tf32={cfg['tf32']}  channels_last={cfg['channels_last']} ===", flush=True)
    set_tf32(cfg["tf32"])

    sampler = GpuUtilSampler() if device.type == "cuda" else None

    # Fresh inference model.
    infer_model = build_model(device, cfg["channels_last"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    print("  Throughput (batch=64)...", flush=True)
    tput = bench_throughput(infer_model, device, cfg["channels_last"], sampler)
    print(f"    {tput['median']:.0f} pos/s   util_median={tput['gpu_util_mean_median']:.1f}%")

    print("  Latency (batch=1)...", flush=True)
    lat = bench_latency(infer_model, device, cfg["channels_last"])
    print(f"    mean={lat['mean_ms_median']:.3f} ms  p99={lat['p99_ms_median']:.3f} ms")

    inf_vram_peak_gb = torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else 0.0
    del infer_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("  Train step (batch=128, 200 steps)...", flush=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    tr = bench_train_step(device, cfg["channels_last"], sampler)
    print(f"    {tr['ms_per_step_median']:.2f} ms/step  VRAM={tr['vram_peak_gb']:.2f} GB  util_median={tr['gpu_util_mean_median']:.1f}%")

    return {
        "arm":     arm_name,
        "config":  cfg,
        "throughput": tput,
        "latency":    lat,
        "train_step": tr,
        "inference_vram_peak_gb": inf_vram_peak_gb,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== TF32 + channels_last probe  {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    info = host_info(device)
    for k, v in info.items():
        print(f"  {k}: {v}")

    # Warm GPU before first arm (avoid first-arm measurement noise).
    if device.type == "cuda":
        _ = torch.zeros(1, device=device)
        torch.cuda.synchronize()

    results = []
    for arm_name, cfg in ARMS:
        results.append(run_arm(arm_name, cfg, device))

    data = {
        "probe":  "tf32_channels_last",
        "date":   time.strftime("%Y-%m-%d"),
        "host":   info,
        "params": {
            "batch_infer":   BATCH_INFER,
            "batch_train":   BATCH_TRAIN,
            "n_positions":   N_POSITIONS,
            "n_train_steps": N_TRAIN_STEPS,
            "n_runs":        N_RUNS,
        },
        "arms":   results,
    }
    host_tag = info.get("gpu", "cpu").replace(" ", "_").replace("/", "_")
    out_json = OUTDIR / f"data_{host_tag}.json"
    with open(out_json, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nData written: {out_json}")


if __name__ == "__main__":
    main()
