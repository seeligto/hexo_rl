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
# python/logging/setup.py
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
// In native_core/src/lib.rs, called once at init via PyO3
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

### Training dashboard

```python
# python/logging/dashboard.py
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.layout import Layout
from rich.console import Console
import time

class TrainingDashboard:
    def __init__(self):
        self.console = Console()
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=30),
            TextColumn("{task.completed}/{task.total}"),
            TextColumn("•"),
            TimeElapsedColumn(),
        )
        self.train_task = self.progress.add_task("Training steps", total=None)
        self.game_task  = self.progress.add_task("Self-play games", total=None)

    def make_metrics_table(self, metrics: dict) -> Table:
        table = Table(show_header=True, header_style="bold cyan", box=None)
        table.add_column("Metric", style="dim", width=24)
        table.add_column("Value", justify="right")

        table.add_row("Iteration",        str(metrics.get("iteration", "-")))
        table.add_row("Policy loss",      f"{metrics.get('policy_loss', 0):.4f}")
        table.add_row("Value loss",       f"{metrics.get('value_loss', 0):.4f}")
        table.add_row("Elo (latest)",     str(metrics.get("elo", "-")))
        table.add_row("Buffer size",      f"{metrics.get('buffer_size', 0):,}")
        table.add_row("Positions/hour",    f"{metrics.get('positions_per_hour', 0):.0f}")
        table.add_row("Sims/sec",         f"{metrics.get('sims_per_sec', 0):,.0f}")
        table.add_row("GPU util",         f"{metrics.get('gpu_util', 0):.0f}%")
        table.add_row("VRAM used",        f"{metrics.get('vram_gb', 0):.1f} GB")
        return table

    def run(self, train_fn, total_steps: int):
        """Context manager — call update(metrics) inside train_fn."""
        layout = Layout()
        layout.split_column(
            Layout(name="progress", size=5),
            Layout(name="metrics", size=14),
        )
        with Live(layout, console=self.console, refresh_per_second=2):
            for step, metrics in train_fn():
                self.progress.update(self.train_task, completed=step, total=total_steps)
                self.progress.update(self.game_task,  completed=metrics.get("games_total", 0))
                layout["progress"].update(Panel(self.progress, title="Progress"))
                layout["metrics"].update(
                    Panel(self.make_metrics_table(metrics), title=f"[bold]Training — step {step}")
                )
```

### Post-training results summary

```python
# python/logging/results.py
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

| Benchmark | Unit | Target | Baseline (2026-03-31, 16w) |
|---|---|---|---|
| MCTS throughput | simulations/sec | ≥ 150,000 | 160,882 |
| Inference throughput | positions/sec | ≥ 8,000 | 11,479 |
| Inference latency (batch=1) | ms mean | ≤ 5 ms | 0.74 ms |
| Worker throughput | positions/hour | ≥ 500,000 | 1,734,003 |
| GPU utilization | % | ≥ 80% | 95.4% |
| VRAM peak | GB | ≤ 6.9 GB (80%) | 0.77 GB |
| Batch fill % | % | ≥ 50% | 99.8% |
| Replay buffer push | positions/sec | ≥ 50,000 | 219,444 |
| Replay buffer sample raw (batch=256) | µs/batch | ≤ 1,000 | 951 |
| Replay buffer sample augmented (batch=256) | µs/batch | ≤ 1,000 | 936 (3.66 µs/pos) |

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

```python
# scripts/benchmark.py
import time
import torch
import numpy as np
from rich.console import Console
from rich.table import Table
from native_core import MCTSTree, Board, GameBenchmarks

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
    from python.model.network import HexTacToeNet
    from native_core import RustReplayBuffer

    model = HexTacToeNet().cuda().half()
    buffer = RustReplayBuffer(capacity=100_000)
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

### Rust micro-benchmarks (criterion)

```toml
# native_core/Cargo.toml
[dev-dependencies]
criterion = { version = "0.5", features = ["html_reports"] }

[[bench]]
name = "mcts_bench"
harness = false
```

```rust
// native_core/benches/mcts_bench.rs
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use native_core::{Board, MCTSTree};

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
# python/logging/gpu_monitor.py
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
# python/logging/metrics_writer.py
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

```yaml
# configs/default.yaml
training:
  batch_size: 256
  lr: 0.002
  weight_decay: 0.0001
  lr_schedule: cosine
  eval_interval: 100
  checkpoint_interval: 50

mcts:
  n_simulations: 800
  c_puct: 1.5
  dirichlet_alpha: 0.3
  epsilon: 0.25
  temperature_threshold_ply: 30

model:
  board_size: 19
  in_channels: 18
  res_blocks: 10
  filters: 128

selfplay:
  n_workers: 6
  inference_batch_size: 64
  replay_buffer_capacity: 500000

rewards:
  shaped_decay_steps: 500
  shaped_4_in_a_row: 0.05
  shaped_5_in_a_row: 0.10
  shaped_block_4: 0.03
  shaped_block_5: 0.08
```

Load with:
```python
import yaml
from dataclasses import dataclass

with open("configs/default.yaml") as f:
    cfg = yaml.safe_load(f)
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
