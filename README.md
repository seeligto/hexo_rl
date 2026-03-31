# Hex Tac Toe AlphaZero

Welcome to the Hex Tac Toe AlphaZero project. This project implements an AlphaZero-style reinforcement learning agent to play Hex Tac Toe on an infinite hexagonal grid.

## Project Introduction

Hex Tac Toe is played on an infinite hexagonal grid. This repository contains the core Rust engine for high-performance board representation and Monte Carlo Tree Search (MCTS), bound to a Python environment where a PyTorch-based neural network is trained via self-play.

We employ a "Multi-Window Cluster-Based Approach" to handle the infinite board: the Rust core dynamically clusters active stones into distinct colonies and returns K distinct 2-plane (19×19) cluster snapshots per colony. Python's `GameState.to_tensor()` stacks these snapshots with `move_history` to assemble the full 18-plane temporal tensor. These are evaluated as a batch by a single sliding-window ResNet, resolving Attention Hijacking while maintaining high performance.

## Setup Instructions

1. **Python Virtual Environment:**
   Ensure you have Python 3.11+ installed. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

   Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. **Rust Core:**
   The high-performance core is written in Rust. Build the Python bindings using Maturin:

   ```bash
   pip install maturin
   maturin develop --release -m native_core/Cargo.toml
   ```

3. **Git Submodules:**
   Initialize and update required submodules (for baseline bots, Ramora engine, etc.):

   ```bash
   git submodule update --init --recursive
   ```

4. **Verify Core Build and Imports:**

   ```bash
   .venv/bin/maturin develop --release -m native_core/Cargo.toml
   .venv/bin/python -c "from native_core import Board, MCTSTree; print('native_core ok')"
   ```

## Roadmap

* **Phase 1: Foundation** - Basic board representation, game state, and neural network architecture. [DONE]
* **Phase 2: MCTS** - Monte Carlo Tree Search implementation in Rust with PyO3 bindings. [DONE]
* **Phase 3: Bootstrap** - Integration of baseline bots and community engines. [DONE]
* **Phase 3.5: Foveated Vision & Hybrid Windowing** - Dual-Resolution CNN and Attention-Anchored Windowing to prevent attention hijacking. [DONE]
* **Phase 3C: Supervised Pretraining** - Training model on expert corpus (Minimax/Human). [COMPLETE/VALIDATING]
* **Phase 4.0: Self-Play RL** - Initializing the AlphaZero self-play loop (CURRENT).
* **Phase 5: Evaluation** - Benchmarking against state-of-the-art bots.

## Training Workflow

The training process is divided into three main stages:

### 1. Corpus Generation (Bootstrap)

Generate a dataset of expert-level games using the compiled Ramora C++ engine and human games scraped from the community archive.

```bash
# Generate 500 bot games and 50 pages of human games
python python/bootstrap/generate_corpus.py --bot-games 500 --human-pages 50
```

### 2. Pre-training (Imitation Learning)

Train the model to imitate the expert play from the corpus. This gives the model a "head start" before self-play begins.

```bash
# Train for 15 epochs and save to checkpoints/bootstrap_model.pt
python -m python.bootstrap.pretrain --force-regenerate --epochs 15
```

### 3. Full Reinforcement Learning (Self-Play)

Start the AlphaZero-style loop where the bot improves by playing against itself.

```bash
# Start RL training from the bootstrap checkpoint
python scripts/train.py --config configs/default.yaml --checkpoint checkpoints/bootstrap_model.pt
```

## Useful Commands

### Focused Performance/Concurrency Tests

```bash
.venv/bin/python -m pytest -q tests/test_inference_server.py tests/test_worker_pool.py tests/test_benchmark_smoke.py
```

### Core Project Tests (without vendor bot test tree)

```bash
.venv/bin/python -m pytest -q tests
```

### Fast Benchmark Pass (quick local check)

```bash
.venv/bin/python scripts/benchmark.py --config configs/fast_debug.yaml --no-compile --mcts-sims 2000 --pool-workers 1 --pool-duration 10
```

### Full Benchmark Pass (higher confidence)

```bash
.venv/bin/python scripts/benchmark.py --config configs/default.yaml --mcts-sims 50000 --pool-workers 6 --pool-duration 30
```

### Short Debug Training Run

```bash
.venv/bin/python scripts/train.py --config configs/fast_debug.yaml --iterations 50 --no-dashboard --no-compile
```

Notes:

* The primary worker throughput metric is **positions/hour** (games/hour is also reported). Very short durations can show 0 positions/hour — increase `--pool-duration` for representative numbers.
* Warnings about pynvml deprecation are emitted by third-party dependencies and do not indicate a training or benchmark failure.

## Performance (March 2026 Baseline)

Measured 2026-03-31, Ryzen 7 3700x + RTX 3070, 16 workers, `make bench.full`. The Rust core uses a Transposition Table (FxHashMap + **128-bit Zobrist hashing**, splitmix128) — 128-bit keys eliminate collision risk at sustained >150k sim/s throughput.

| Metric | Measured |
|---|---|
| MCTS (CPU only) | 160,882 sim/s |
| NN inference (batch=64) | 11,479 pos/s |
| NN latency (batch=1, mean) | 0.74 ms (p99: 2.57 ms) |
| Replay buffer push | 219,444 pos/sec |
| Replay buffer sample (aug, batch=256) | 936 µs/batch (3.66 µs/pos) |
| Worker throughput | 4,134 games/hr / **1,734,003 pos/hr** |
| GPU utilization | 95.4% |
| Batch fill % | 99.8% |

This project uses **Batched MCTS Inferences**, providing a ~15x speedup on NVIDIA GPUs.
