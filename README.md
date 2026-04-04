# Hex Tac Toe AlphaZero

An AlphaZero-style self-learning AI for **Hex Tac Toe** — a community game on an
infinite hexagonal grid where the goal is 6 stones in a row. Player 1 opens with
1 stone, then both players alternate placing 2 stones per turn.

The high-performance core is written in Rust (`engine/`) and exposed to Python via
PyO3. A PyTorch neural network is trained via self-play using MCTS-guided game
generation. See `CLAUDE.md` for full project context.

---

## Prerequisites

- Python 3.11+ (tested on 3.14)
- Rust 1.75+ (`rustup`)
- CUDA 12.x + an NVIDIA GPU (RTX 3070 or better recommended)
- `maturin` (installed automatically by `make install`)

---

## Quick start

```bash
git clone --recursive <url>
cd hexo_rl

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

make native.build        # build Rust engine extension

make test.all            # run Rust + Python test suites (86 + 576 tests)

# Generate corpus (SealBot games + human games)
make corpus.scrape       # fetch latest human games from hexo.did.science
make corpus.fast         # generate SealBot fast corpus (0.1s/move, 5,000 games)
make corpus.strong       # generate SealBot strong corpus (0.5s/move, 2,500 games)
make corpus.npz          # export combined corpus → data/bootstrap_corpus.npz

# Supervised pretraining on corpus
make pretrain.full       # 15 epochs (~2–4 hours depending on corpus size)

# Self-play RL
make train               # self-play RL with web + terminal dashboard
```

> **Note:** `data/bootstrap_corpus.npz` must exist before running `make train`.
> It is the mixed-data source for the pretrained buffer. If missing, training
> runs on self-play data only (slower cold-start convergence).

---

## Make targets

### Build & environment

```bash
make env.check           # verify .venv + engine import
make native.build        # build/install Rust extension (LTO + native CPU)
make clean               # remove Rust artifacts and Python caches
make rebuild             # full clean + rebuild
```

### Testing

```bash
make test.rust           # Rust unit tests
make test.py             # Python tests (tests/ only)
make test.all            # Rust + Python
make test.focus          # buffer / inference / pool smoke tests
make ci                  # full pre-push gate: all tests + quick benchmark
```

### Benchmarks

```bash
make bench.quick         # 30s sanity check
make bench.lite          # quick benchmark (n=3)
make bench.full          # standard gate (n=5, 3s warm-up) — run before Phase 4.5
make bench.stress        # heavy 5-min stability test
make bench.baseline      # save bench.full result to reports/benchmarks/
make bench.mcts          # Rust MCTS micro-benchmark only
```

### Corpus generation

Run these in order before pretraining:

```bash
make corpus.scrape       # scrape latest human games (hexo.did.science)
make corpus.fast         # SealBot fast corpus (0.1s think, 5,000 games)
make corpus.strong       # SealBot strong corpus (0.5s think, 2,500 games)
make corpus.all          # fast + strong + manifest update
make corpus.npz          # export to data/bootstrap_corpus.npz (~140 MB)
make corpus.analysis     # quality analysis on human + bot games
```

### Pretraining

```bash
make pretrain.lite       # smoke test — 100 steps only
make pretrain.full       # full supervised pretrain — 15 epochs
```

Checkpoint saved to `checkpoints/pretrain/pretrain_<step>.pt`.

### Self-play training

```bash
make train               # RL from bootstrap checkpoint + corpus (production default)
make train.nodash        # same, no dashboard (useful for tmux/ssh)
make train.bg            # background training (logs to logs/)
make train.stop          # stop background training
make train.status        # check if running, show recent log tail
make train.resume        # resume from latest checkpoint
make train.smoke         # 200-step smoke test — verifies end-to-end stack
make train.raw           # from random init, no pretrain (ablation only)
```

The dashboard is available at `http://localhost:5001` when running `make train`.

### Evaluation

```bash
make eval.sealbot.quick  # 10 games, 64 sims (fast sanity check)
make eval.sealbot.full   # 100 games, 128 sims (full gate)
make eval.sealbot.latest # eval latest checkpoint
```

### Plotting

```bash
make plot.train.latest   # plot latest training log
make plot.sealbot.latest # plot latest SealBot eval result
make plot.sealbot.all    # SealBot Elo trend over all evals
```

---

## Performance baseline

**Hardware:** Ryzen 7 3700x + RTX 3070, 16 workers, LTO + native CPU  
**Date:** 2026-04-04  
**Model:** 12 residual blocks × 128 channels, SE blocks, dual-pool value head  
**torch.compile:** DISABLED (Python 3.14 CUDA graph incompatibility — see below)

| Metric | Baseline (median, n=5) | Target |
|---|---|---|
| MCTS (CPU only, no NN) | 30,963 sim/s | ≥ 26,000 sim/s |
| NN inference (batch=64) | 10,993 pos/s | ≥ 8,500 pos/s |
| NN latency (batch=1, mean) | 2.83 ms | ≤ 3.5 ms |
| Replay buffer push | 839,289 pos/s | ≥ 640,000 pos/s |
| Replay buffer sample (batch=256) | 1,270.9 µs | ≤ 1,500 µs |
| Replay buffer sample augmented (batch=256) | 1,147.5 µs | ≤ 1,400 µs |
| GPU utilization | 100% | ≥ 85% |
| VRAM usage | 0.10 GB / 8.6 GB | ≤ 80% |
| Worker throughput | 758,748 pos/hr | ≥ 625,000 pos/hr |
| Batch fill % | 100% | ≥ 80% |

Run `make bench.full` to reproduce. All 10 targets pass.

> **Benchmark methodology:** median of n=5 runs, 3s warm-up per metric.
> MCTS workload: 800 sims/move × 62 iterations with tree reset between moves.
> Targets set at 85% of observed median.

---

## Architecture overview

- **Rust (`engine/`)** — MCTS tree, board logic (infinite sparse HashMap board,
  128-bit Zobrist hashing), replay buffer (f16-as-u16, 12-fold hex augmentation),
  self-play runner (Rust worker threads), inference batcher
- **Python (`hexo_rl/`)** — neural network (ResNet-12 × 128ch, SE blocks, dual-pool
  value head, BCE loss), training loop, evaluation pipeline, corpus pipeline,
  event-driven monitoring dashboard
- **PyO3 bridge** — `from engine import Board, MCTSTree, ReplayBuffer, SelfPlayRunner, InferenceBatcher`

### Network architecture

| Component | Setting |
|---|---|
| Residual blocks | 12 × 128 channels |
| Squeeze-and-excitation | Every block, reduction ratio 4 |
| Value head | Global avg + max pool → FC(256) → FC(1) → BCE on sigmoid |
| Policy head | Conv → flatten → log-softmax |
| Auxiliary head | Opponent reply prediction (weight 0.15, training only) |
| Entropy regularisation | 0.01 (prevents policy collapse) |

### Known issues / current status

**torch.compile disabled** — Python 3.14 has incompatibilities with PyTorch's
CUDA graph implementation that caused three consecutive blocking failures:
`mode="reduce-overhead"` TLS crash (§25), `mode="default"` first-pass crash (§30),
and a 27 GB RAM spike during Triton JIT compilation that blocked workers for 5+
minutes (§32). The benchmark delta was only +3% worker throughput. Compile support
is disabled via `torch_compile: false` in `configs/training.yaml` and will be
re-enabled when PyTorch stabilizes on Python 3.14.

**Board is infinite** — The NN sees a fixed 19×19 window centred on active stone
clusters (Hybrid Attention-Anchored Windowing). The Rust engine is genuinely
unbounded (`HashMap<(q,r), Player>`).

---

## Training workflow

```
corpus.scrape + corpus.fast/strong
        ↓
    corpus.npz
        ↓
  pretrain.full   ← supervised on corpus (~15 epochs)
        ↓
     train        ← self-play RL, resumes from pretrain checkpoint
                     mixes corpus + self-play (exponential decay over 1M steps)
```

The mixed-data schedule:
- Step 0: 80% corpus / 20% self-play  
- Step 1M: 10% corpus / 90% self-play (floor)

Buffer grows with training: 250K → 500K (step 500K) → 1M (step 1.5M).

---

See `CLAUDE.md` for complete context, working rules, and session protocols.
See `docs/02_roadmap.md` for phase exit criteria.
See `docs/07_PHASE4_SPRINT_LOG.md` for a full record of architectural decisions.
