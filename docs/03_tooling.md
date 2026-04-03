# Tooling & Conventions — Logging, Benchmarking, Progress Display

---

## 1. Logging architecture

Two separate concerns that must not be conflated:

| Concern | Tool | Output |
|---|---|---|
| Structured event logs | `structlog` (Python) + `tracing` (Rust) | JSON file, queryable |
| Human-readable console | `rich` live display | Terminal only |
| Numeric metrics | TensorBoard / wandb | Plots over time |

### Python — structlog setup

```python
# hexo_rl/monitoring/configure.py
import structlog
import logging
from pathlib import Path
from datetime import datetime

def configure_logging(log_dir: str = "logs", run_name: str | None = None):
    run_name = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path(log_dir) / f"{run_name}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        logger_factory=structlog.WriteLoggerFactory(file=log_path.open("a")),
    )
    return structlog.get_logger()
```

### Usage throughout the codebase

```python
import structlog
log = structlog.get_logger()

# Training step
log.info("train_step",
    iteration=step,
    policy_loss=float(policy_loss),
    value_loss=float(value_loss),
    total_loss=float(loss),
    lr=optimizer.param_groups[0]["lr"],
    buffer_size=replay_buffer.size,
)

# Self-play game completed
log.info("game_complete",
    game_id=game_id,
    worker_id=worker_id,
    outcome=outcome,          # 1, -1, or 0
    plies=ply_count,
    duration_sec=elapsed,
    mcts_sims_per_move=avg_sims,
)

# Evaluation result
log.info("eval_result",
    iteration=step,
    new_elo=new_elo,
    delta_elo=new_elo - old_elo,
    win_rate_vs_prev=win_rate,
    promoted=is_new_best,
)
```

### Rust — tracing setup

```rust
// In engine/src/lib.rs, called once at init via PyO3
use tracing_subscriber::fmt;

pub fn init_tracing(level: &str) {
    let level = level.parse().unwrap_or(tracing::Level::INFO);
    fmt()
        .with_max_level(level)
        .with_target(false)
        .compact()
        .init();
}
```

```rust
// In MCTS hot paths — use sparingly (tracing has overhead if called per-node)
// Log at game level, not node level
tracing::info!(
    game_id = %id,
    simulations = n_sims,
    tree_depth = depth,
    "mcts_search_complete"
);
```

**Rule**: Never log inside the MCTS inner loop. Only at game boundaries. The inner loop runs millions of times — any logging there kills throughput.

---

## 2. Progress display — rich

### Monitoring system (event-driven fan-out)

Training emits structured events via `emit_event()` (`hexo_rl/monitoring/events.py`).
Registered renderers receive every event — they are passive observers and never
block the training loop. Two built-in renderers:

**Terminal dashboard** (`hexo_rl/monitoring/terminal_dashboard.py`):
- Rich Live panel, max 4Hz refresh
- Shows loss/throughput/buffer/system stats
- Alert line for: entropy collapse (< 1.0), grad spikes (norm > 10),
  consecutive loss increases, eval gate failures
- Responds to `training_step` and `iteration_complete` events

**Web dashboard** (`hexo_rl/monitoring/web_dashboard.py`):
- Flask+SocketIO on localhost:5001
- Chart.js loss curves, win rate bars, system stats
- Event history replay on browser reconnect (last 500 events)
- Routes: `/` (dashboard), `/viewer` (game viewer),
  `/viewer/game/<id>`, `/viewer/recent`, `/viewer/play` (POST)

**How to start:**

```bash
make train           # both dashboards enabled by default
make train.nodash    # training only, no dashboard
make train.bg        # background with PID tracking
make train.stop      # kill background run
make train.status    # check if running, show recent log
make dash.open       # open web dashboard in browser
```

**Event types** (see `docs/08_DASHBOARD_SPEC.md` for full schema):
- `run_start`, `run_end` — bookend events with run_id
- `training_step` — loss, entropy, grad_norm, lr every N steps
- `iteration_complete` — throughput, games/hr, buffer state
- `game_complete` — winner, move count, move sequence
- `eval_complete` — Elo, win rate vs SealBot, gate status
- `system_stats` — GPU util, VRAM, worker count

**Config:** `configs/monitoring.yaml` controls enable/disable, web port,
alert thresholds, and event log maxlen.

### Post-training results summary

```python
# hexo_rl/monitoring/results.py
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

def print_training_summary(history: list[dict]):
    console = Console()

    console.print()
    console.print(Panel.fit(
        "[bold green]Training complete[/bold green]",
        border_style="green"
    ))

    # Elo progression table
    elo_table = Table(title="Elo progression", box=box.SIMPLE)
    elo_table.add_column("Iteration", justify="right", style="dim")
    elo_table.add_column("Elo",       justify="right", style="bold")
    elo_table.add_column("Δ Elo",     justify="right")
    elo_table.add_column("Win rate vs prev", justify="right")

    prev_elo = None
    for row in history[::max(1, len(history)//10)]:  # show ~10 rows
        elo = row.get("elo", 0)
        delta = f"[green]+{elo - prev_elo:.0f}[/green]" if prev_elo and elo > prev_elo else (
                f"[red]{elo - prev_elo:.0f}[/red]" if prev_elo else "-")
        wr = f"{row.get('win_rate_vs_prev', 0)*100:.1f}%"
        elo_table.add_row(str(row["iteration"]), f"{elo:.0f}", delta, wr)
        prev_elo = elo

    console.print(elo_table)

    # Loss summary
    loss_table = Table(title="Final losses", box=box.SIMPLE)
    loss_table.add_column("Metric")
    loss_table.add_column("Value", justify="right")
    last = history[-1]
    loss_table.add_row("Policy loss",  f"{last.get('policy_loss', 0):.4f}")
    loss_table.add_row("Value loss",   f"{last.get('value_loss',  0):.4f}")
    loss_table.add_row("Total games",  f"{last.get('games_total', 0):,}")
    loss_table.add_row("Best Elo",     f"{max(r.get('elo',0) for r in history):.0f}")
    console.print(loss_table)
```

---

## 3. Benchmarking harness

### What to measure

See CLAUDE.md § "Benchmarks — must pass before Phase 4.5" for the canonical
benchmark table (2026-04-03 baseline, correct 12-block × 128-channel model).

| Benchmark | Unit | Target | Baseline (2026-04-03, 16w) |
|---|---|---|---|
| MCTS throughput | simulations/sec | ≥ 140,000 | 164,946 |
| Inference throughput | positions/sec | ≥ 8,500 | 10,201 |
| Inference latency (batch=1) | ms mean | ≤ 3.5 ms | 2.82 ms |
| Worker throughput | positions/hour | ≥ 450,000 | 530,526 |
| GPU utilization | % | ≥ 85% | 100.0% |
| VRAM peak | GB | ≤ 80% (6.9 GB) | 0.10 GB |
| Batch fill % | % | ≥ 80% | 95.2% |
| Replay buffer push | positions/sec | ≥ 640,000 | 755,880 |
| Replay buffer sample raw (batch=256) | µs/batch | ≤ 1,500 | 1,237.6 |
| Replay buffer sample augmented (batch=256) | µs/batch | ≤ 1,400 | 1,177 |

### Practical benchmark commands

Quick local sanity pass:

```bash
make bench.lite
```

Throughput-oriented pass:

```bash
make bench.full
```

Heavy stability and deep search stress test:

```bash
make bench.stress
```

Save a dated baseline JSON:

```bash
make bench.baseline
```

Corpus and pretrain targets:

```bash
make corpus.npz        # export corpus to data/bootstrap_corpus.npz
make pretrain.lite     # smoke test (100 steps)
make pretrain.full     # full pretrain (15 epochs)
```

If worker-pool throughput reports 0 positions/hour in short runs, increase `--pool-duration`.

### Focused validation for inference/pool changes

```bash
.venv/bin/python -m pytest -q tests/test_inference_server.py tests/test_worker_pool.py tests/test_benchmark_smoke.py
```

For Rust-runner handshake debugging during migration, run:

```bash
.venv/bin/python -m pytest -vv -s --maxfail=1 tests/test_inference_server.py tests/test_worker_pool.py
```

This pair is the fastest signal for deadlocks and queue handoff regressions.

### Python benchmark runner

> **Note:** The code below is a simplified illustration. The actual `scripts/benchmark.py`
> implements the full n=5 median + IQR methodology described above. See that file for the
> current implementation.

```python
# scripts/benchmark.py (simplified)
import time
import torch
import numpy as np
from rich.console import Console
from rich.table import Table
from engine import MCTSTree, Board

console = Console()

def benchmark_mcts(n_simulations=50_000) -> dict:
    board = Board(size=19)
    tree  = MCTSTree(c_puct=1.5)
    tree.new_game(board)

    # Warm up
    tree.run_simulations_cpu_only(n=1000)

    start = time.perf_counter()
    tree.run_simulations_cpu_only(n=n_simulations)
    elapsed = time.perf_counter() - start

    return {
        "name":        "MCTS (CPU only, no NN)",
        "sims":        n_simulations,
        "elapsed_sec": elapsed,
        "sims_per_sec": n_simulations / elapsed,
    }

def benchmark_inference(model, n_positions=10_000, batch_size=64) -> dict:
    model.eval()
    device = next(model.parameters()).device
    dummy  = torch.zeros(batch_size, 18, 19, 19, dtype=torch.float16, device=device)

    # Warm up
    with torch.no_grad(), torch.cuda.amp.autocast():
        for _ in range(10):
            model(dummy)
    torch.cuda.synchronize()

    n_batches = n_positions // batch_size
    start = time.perf_counter()
    with torch.no_grad(), torch.cuda.amp.autocast():
        for _ in range(n_batches):
            model(dummy)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    total_positions = n_batches * batch_size
    return {
        "name":             f"NN inference (batch={batch_size})",
        "positions":        total_positions,
        "elapsed_sec":      elapsed,
        "positions_per_sec": total_positions / elapsed,
        "latency_ms":       elapsed / n_batches * 1000,
    }

def benchmark_inference_latency(model) -> dict:
    """Single-position latency — worst case for synchronous inference."""
    device = next(model.parameters()).device
    dummy  = torch.zeros(1, 18, 19, 19, dtype=torch.float16, device=device)
    model.eval()
    times = []
    with torch.no_grad(), torch.cuda.amp.autocast():
        for _ in range(500):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(dummy)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
    times = times[50:]  # discard warm-up
    return {
        "name":       "NN latency (batch=1)",
        "mean_ms":    float(np.mean(times)),
        "p50_ms":     float(np.percentile(times, 50)),
        "p99_ms":     float(np.percentile(times, 99)),
    }

def benchmark_replay_buffer(buffer) -> dict:
    t0 = time.perf_counter()
    for _ in range(10_000):
        buffer.sample_batch(256, augment=True)
    elapsed = time.perf_counter() - t0
    return {
        "name":          "Replay buffer sample (augmented, batch=256)",
        "samples":       10_000,
        "elapsed_sec":   elapsed,
        "us_per_sample": elapsed / 10_000 * 1e6,
    }

def print_benchmark_report(results: list[dict]):
    table = Table(title="Benchmark report", show_lines=True)
    table.add_column("Benchmark", style="bold")
    table.add_column("Result",    justify="right")
    table.add_column("Target",    justify="right", style="dim")
    table.add_column("Status",    justify="center")

    checks = [
        ("MCTS (CPU only, no NN)",        "sims_per_sec",      10_000,  ">= 10,000 sim/s"),
        ("NN inference (batch=64)",        "positions_per_sec",  5_000,  ">= 5,000 pos/s"),
        ("NN latency (batch=1)",           "mean_ms",                5,  "<= 5 ms"),
        ("Replay buffer sample (batch=256)","us_per_sample",        500,  "<= 500 μs"),
    ]

    by_name = {r["name"]: r for r in results}
    for name, key, target, label in checks:
        r = by_name.get(name, {})
        val = r.get(key)
        if val is None:
            table.add_row(name, "-", label, "[yellow]SKIP[/yellow]")
            continue
        is_latency = "latency" in key or "us_per" in key or "ms" in key
        ok = val <= target if is_latency else val >= target
        status = "[green]PASS[/green]" if ok else "[red]FAIL[/red]"
        table.add_row(name, f"{val:,.1f}", label, status)

    console.print(table)

if __name__ == "__main__":
    from hexo_rl.model.network import HexTacToeNet
    from engine import ReplayBuffer

    model = HexTacToeNet().cuda().half()
    buffer = ReplayBuffer(capacity=100_000)
    # fill buffer with dummy data for benchmark
    for i in range(10_000):
        buffer.push(
            np.zeros((18, 19, 19), dtype=np.float16),
            np.ones(362, dtype=np.float32) / 362,
            0.0,
            i,  # game_id
        )

    results = [
        benchmark_mcts(),
        benchmark_inference(model, batch_size=64),
        benchmark_inference_latency(model),
        benchmark_replay_buffer(buffer),
    ]
    print_benchmark_report(results)
```

### Benchmark variance (historical)

Before 2026-04-01, benchmarks were single-run with no CPU frequency pinning and no
warm-up phase. LLVM codegen differences across rebuilds (function layout,
instruction-cache alignment) combined with AMD boost clock behaviour produced
+/-50% swings on buffer push/s and +/-49% on worker throughput. These were
measurement artifacts, not real performance changes.

The new methodology (warm-up per metric, n=5 median +/- IQR) reduces variance
to <10% IQR on all metrics. Key changes:

- **Warm-up phase**: Each metric runs its operation for 2-10 seconds before timing
  begins, evicting cold-cache effects and stabilising boost clocks.
- **Realistic MCTS workload**: MCTS throughput is measured using 800 sims/move
  (matching `selfplay.yaml` `mcts.n_simulations`) with tree reset between each move,
  rather than a single monolithic 50k-sim search.  A single oversized tree exceeds
  L2 cache and underreports real self-play throughput by ~15%.
- **Multiple runs**: n=5 (full) or n=10 (stress). Median is reported instead of
  mean to resist outliers from scheduler interruptions.
- **IQR (P75-P25)**: Reported alongside median as the spread metric. More robust
  than standard deviation for small n with potential outliers.

Benchmark modes:

| Mode | Runs | Use case |
|---|---|---|
| `make bench.lite` | n=3 | Quick local sanity check |
| `make bench.full` | n=5 | Standard regression gate |
| `make bench.stress` | n=10 | Pre-release confidence |

JSON reports are written to `reports/benchmarks/YYYY-MM-DD_HH-MM.json` for
historical comparison.

### Rust micro-benchmarks (criterion)

```toml
# engine/Cargo.toml
[dev-dependencies]
criterion = { version = "0.5", features = ["html_reports"] }

[[bench]]
name = "mcts_bench"
harness = false
```

```rust
// engine/benches/mcts_bench.rs
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use engine::{Board, MCTSTree};

fn bench_win_detection(c: &mut Criterion) {
    let mut board = Board::new(19);
    // Place some stones to make it non-trivial
    for i in 0..5 {
        board.apply_move(9, i, 1);
    }
    c.bench_function("win_check_no_win", |b| {
        b.iter(|| board.check_win())
    });
}

fn bench_mcts_simulations(c: &mut Criterion) {
    let mut group = c.benchmark_group("mcts_sims");
    for n in [100, 400, 800].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(n), n, |b, &n| {
            let board = Board::new(19);
            let mut tree = MCTSTree::new(1.5);
            tree.new_game(board);
            b.iter(|| {
                tree.run_simulations_cpu_only(n);
                tree.reset();
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_win_detection, bench_mcts_simulations);
criterion_main!(benches);
```

Run with: `cargo bench --bench mcts_bench -- --output-format html`

---

## 4. GPU monitoring

During training, poll `pynvml` every 5 seconds and log to structlog:

```python
# hexo_rl/monitoring/gpu_monitor.py
import pynvml
import threading
import time
import structlog

log = structlog.get_logger()

class GPUMonitor(threading.Thread):
    def __init__(self, interval_sec=5, device_index=0):
        super().__init__(daemon=True)
        self.interval = interval_sec
        self.device_index = device_index
        self._stop = threading.Event()
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)

    def run(self):
        while not self._stop.wait(self.interval):
            util  = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            mem   = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            temp  = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)
            log.info("gpu_stats",
                gpu_util_pct=util.gpu,
                mem_util_pct=util.memory,
                vram_used_gb=mem.used / 1e9,
                vram_total_gb=mem.total / 1e9,
                temp_c=temp,
            )

    def stop(self):
        self._stop.set()
```

---

## 5. TensorBoard / wandb integration

```python
# hexo_rl/monitoring/metrics_writer.py
from torch.utils.tensorboard import SummaryWriter

class MetricsWriter:
    def __init__(self, log_dir: str, use_wandb: bool = False):
        self.writer = SummaryWriter(log_dir)
        self.use_wandb = use_wandb
        if use_wandb:
            import wandb
            wandb.init(project="hex-tac-toe-az", dir=log_dir)

    def log_step(self, step: int, metrics: dict):
        for key, val in metrics.items():
            if isinstance(val, (int, float)):
                self.writer.add_scalar(key, val, step)
        if self.use_wandb:
            import wandb
            wandb.log(metrics, step=step)

    def log_game_record(self, step: int, game_pgn: str):
        """Log a game as text for manual review."""
        self.writer.add_text("eval_game", game_pgn, step)
```

---

## 6. Developer conventions

### Log levels

| Level | When to use |
|---|---|
| `DEBUG` | Per-node MCTS detail — disabled in production |
| `INFO` | Per-game and per-step events |
| `WARNING` | Unexpected states (empty buffer, slow GPU) |
| `ERROR` | Recoverable failures (worker crash, checkpoint missing) |

### Config management

All hyperparameters live in YAML. Never hardcode values in source files.

Configs are split by concern. `train.py` deep-merges them via `load_config()`:

```yaml
# configs/model.yaml          — network architecture
model:
  board_size: 19
  in_channels: 18
  res_blocks: 12
  filters: 128
  se_reduction_ratio: 4

# configs/training.yaml        — optimizer, scheduler, buffer
training:
  batch_size: 256
  lr: 0.002
  weight_decay: 0.0001
  lr_schedule: cosine

# configs/selfplay.yaml        — MCTS, workers
mcts:
  n_simulations: 800
  c_puct: 1.5
selfplay:
  n_workers: 16
  inference_batch_size: 64

# configs/monitoring.yaml      — dashboards, alerts
monitoring:
  enabled: true
  terminal_dashboard: true
  web_dashboard: true
  web_port: 5001
```

Load with:

```python
from hexo_rl.utils.config import load_config
config = load_config("configs/model.yaml", "configs/training.yaml",
                     "configs/selfplay.yaml", "configs/monitoring.yaml")
```

### Reproducibility

```python
import random, numpy as np, torch

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
```

Call at startup with seed from config. Log the seed so runs can be reproduced.
