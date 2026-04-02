# Hex Tac Toe AlphaZero

An AlphaZero-style self-learning AI for **Hex Tac Toe** — a community game on an infinite hexagonal grid where the goal is 6 stones in a row. Player 1 opens with 1 stone, then both players alternate placing 2 stones per turn.

The high-performance core is written in Rust (`engine/`) and exposed to Python via PyO3. A PyTorch neural network is trained via self-play using MCTS-guided game generation. See `CLAUDE.md` for full project context.

## Prerequisites

- Python 3.11+
- Rust 1.75+ (`rustup`)
- CUDA 12.x + an NVIDIA GPU (RTX 3070 or better recommended)
- `maturin` (installed automatically by `make install`)

## Quick start

```bash
git clone --recursive <url>
cd hexo_rl

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

make native.build        # build Rust engine extension
make test.all            # run Rust + Python test suites

make corpus.scrape       # fetch latest human games
make corpus.npz          # export corpus to data/bootstrap_corpus.npz
make pretrain.full       # supervised pretrain on corpus

python scripts/train.py --config configs/default.yaml \
    --checkpoint checkpoints/pretrain/pretrain_00000100.pt
```

## Make targets

```bash
make env.check           # verify .venv + engine import
make native.build        # build/install Rust extension (LTO + native CPU)
make clean               # remove Rust artifacts and Python caches
make rebuild             # full clean + rebuild

make test.rust           # Rust tests
make test.py             # Python tests
make test.all            # both

make bench.lite          # quick benchmark (n=3)
make bench.full          # standard benchmark gate (n=5)

make corpus.scrape       # scrape latest human games
make corpus.d4           # SealBot depth-4 corpus (2,000 games)
make corpus.d6           # SealBot depth-6 corpus (1,000 games)
make corpus.npz          # export to data/bootstrap_corpus.npz

make pretrain.lite       # pretrain smoke test (100 steps)
make pretrain.full       # full pretrain (15 epochs)

make train.full          # RL training from bootstrap checkpoint
make dashboard           # start web dashboard on port 5001
```

## Performance baseline (2026-04-02, Ryzen 7 3700x + RTX 3070)

| Metric | Value |
|---|---|
| MCTS (CPU only) | 189,656 sim/s |
| NN inference (batch=64) | 10,080 pos/s |
| NN latency (batch=1, mean) | 1.52 ms |
| Replay buffer push | 905,697 pos/s |
| Worker throughput | 1,177,745 pos/hr |
| GPU utilization | 100% |

## Architecture overview

- **Rust (`engine/`)**: MCTS tree, board logic, replay buffer, self-play runner, inference batcher
- **Python (`hexo_rl/`)**: neural network (ResNet-12 × 128ch + SE), training loop, evaluation, corpus pipeline
- **PyO3 bridge**: `from engine import Board, MCTSTree, ReplayBuffer, SelfPlayRunner, InferenceBatcher`

See `CLAUDE.md` for complete context, working rules, and session protocols.
