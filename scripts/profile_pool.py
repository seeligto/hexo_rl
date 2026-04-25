"""Minimal worker-pool driver for py-spy profiling and trace-fix smoke.

Skips MCTS/NN/buffer subtests so the profile contains only model load,
pool warmup, and steady-state self-play. Two uses:
  1. py-spy record --duration N -- .venv/bin/python scripts/profile_pool.py
     captures a speedscope of the inference-server thread for analysis.
  2. Plain run with TRACE_INFERENCE=0 / 1 to A/B the dispatcher fix without
     py-spy attached (py-spy at 200 Hz × 4 threads distorts absolute pos/hr).

Env knobs: TRACE_INFERENCE, WARMUP_SEC, STEADY_SEC, N_WORKERS, WORKER_SIMS,
BENCH_MAX_MOVES.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hexo_rl.utils.config import load_config
from hexo_rl.utils.device import best_device
from hexo_rl.model.network import HexTacToeNet
from hexo_rl.model.tf32 import resolve_and_apply as tf32_apply
from hexo_rl.selfplay.pool import WorkerPool
from engine import ReplayBuffer

import os
WARMUP_SEC = int(os.environ.get("WARMUP_SEC", 90))
STEADY_SEC = int(os.environ.get("STEADY_SEC", 180))
N_WORKERS = int(os.environ.get("N_WORKERS", 10))
WORKER_SIMS = int(os.environ.get("WORKER_SIMS", 200))
BENCH_MAX_MOVES = int(os.environ.get("BENCH_MAX_MOVES", 128))
TRACE = os.environ.get("TRACE_INFERENCE", "1") not in ("0", "false", "False")


def main() -> None:
    print(f"[driver] loading config", flush=True)
    cfg = load_config(
        "configs/model.yaml",
        "configs/training.yaml",
        "configs/selfplay.yaml",
        "configs/variants/gumbel_targets_desktop.yaml",
    )
    tf32_apply(cfg)

    device = best_device()
    print(f"[driver] device={device}", flush=True)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    mcfg = cfg.get("model", {})
    model = HexTacToeNet(
        board_size=int(mcfg.get("board_size", 19)),
        in_channels=int(mcfg.get("in_channels", 18)),
        filters=int(mcfg.get("filters", 128)),
        res_blocks=int(mcfg.get("res_blocks", 12)),
    ).to(device)

    # Force CUDA JIT before pool starts (else first inference call costs ~90s).
    if device.type == "cuda":
        with torch.no_grad(), torch.autocast(device_type="cuda"):
            dummy = torch.zeros(1, int(mcfg.get("in_channels", 18)),
                                int(mcfg.get("board_size", 19)),
                                int(mcfg.get("board_size", 19)), device=device)
            model(dummy)
        torch.cuda.synchronize()

    bench_cfg = dict(cfg)
    bench_cfg["mcts"] = {**cfg.get("mcts", {}), "n_simulations": WORKER_SIMS}
    sp = dict(cfg.get("selfplay", {}))
    sp.update({
        "n_workers": N_WORKERS,
        "max_game_moves": BENCH_MAX_MOVES,
        "max_moves_per_game": BENCH_MAX_MOVES,
        "playout_cap": {**sp.get("playout_cap", {}), "fast_prob": 0.0},
        "random_opening_plies": 0,
        "trace_inference": TRACE,
    })
    print(f"[driver] trace_inference={TRACE}", flush=True)
    bench_cfg["selfplay"] = sp

    replay = ReplayBuffer(capacity=100_000)
    pool = WorkerPool(model, bench_cfg, device, replay, n_workers=N_WORKERS)
    print(f"[driver] pool starting n_workers={N_WORKERS}", flush=True)
    pool.start()

    t0 = time.perf_counter()
    g0 = pool.games_completed
    p0 = pool.positions_pushed

    # Warmup
    while time.perf_counter() - t0 < WARMUP_SEC:
        time.sleep(2.0)
        elapsed = time.perf_counter() - t0
        gph = (pool.games_completed - g0) / max(elapsed, 1e-3) * 3600
        print(f"[driver] warmup {elapsed:5.1f}s gph={gph:7.1f} games={pool.games_completed} pos={pool.positions_pushed}", flush=True)

    # Snapshot at end of warmup, then measurement window
    g1 = pool.games_completed
    p1 = pool.positions_pushed
    server = pool._inference_server
    fwd_start = server.forward_count
    req_start = server.total_requests
    t1 = time.perf_counter()
    print(f"[driver] STEADY-STATE START at t={t1-t0:.1f}s", flush=True)

    while time.perf_counter() - t1 < STEADY_SEC:
        time.sleep(2.0)
        elapsed = time.perf_counter() - t1
        d_pos = pool.positions_pushed - p1
        d_games = pool.games_completed - g1
        d_fwd = server.forward_count - fwd_start
        d_req = server.total_requests - req_start
        pph = (d_pos / max(elapsed, 1e-3)) * 3600
        bat = (d_req / max(d_fwd * server._batch_size, 1)) * 100
        print(f"[driver] steady {elapsed:5.1f}s pph={pph:9.1f} bat={bat:5.1f}% fwd={d_fwd}", flush=True)

    print(f"[driver] STEADY-STATE END — stopping pool", flush=True)
    pool.stop()


if __name__ == "__main__":
    main()
