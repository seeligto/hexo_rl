# Agent Context — Hex Tac Toe AlphaZero

> **Read this first.** This file orients any AI agent or developer to the project at a glance. Other files go deeper on specific areas.

---

## What this project is

An AlphaZero-style self-learning AI for **Hex Tac Toe** — a community-played game on a hexagonal grid where the goal is **6 stones in a row**. One player opens with a single move; after that both players alternate placing **2 stones per turn**. The board is theoretically infinite but practically bounded to ~19×19.

There is an active human community with existing openings and strategies. The intent is to build an AI that can:

1. Play at or beyond top human level through self-play
2. Serve as a research tool for the community to explore openings
3. Be trained on consumer hardware (AMD Ryzen 7 3700x, RTX 3070, 48GB RAM)

---

## Repository map

```txt
hexo_rl/
├── engine/             Rust crate — MCTS tree, board logic, replay buffer (hot paths)
├── hexo_rl/
│   ├── model/          PyTorch ResNet + dual heads (policy + value)
│   ├── selfplay/       Worker pool, batched GPU inference queue
│   ├── training/       Training loop, loss functions, checkpoints
│   ├── eval/           Tournament runner, Bradley-Terry Elo ladder
│   ├── bootstrap/      Bot integration, supervised pretraining
│   ├── monitoring/     Event-driven monitoring — events.py fan-out, terminal+web dashboards, GPU monitor, game recorder
│   ├── viewer/         Game viewer engine and play-against-model interface
│   └── corpus/         Corpus pipeline and metrics
├── configs/            YAML hyperparameter configs
├── scripts/            CLI entrypoints (train.py, benchmark.py, eval_vs_sealbot.py)
├── docs/               These markdown files
└── tests/
```

---

## Language boundary — critical to understand

| Layer | Language | Why |
|---|---|---|
| MCTS tree traversal | **Rust** | Sequential pointer-chasing loop — Python is 30-100× too slow |
| Board logic + win detection | **Rust** | Bitboard ops + 128-bit Zobrist hashing at call frequency of millions/sec |
| Replay buffer | **Rust** (ReplayBuffer) | f16-as-u16 ring buffer, 12-fold hex augmentation, zero-copy PyO3 transfer |
| Neural network | **Python + PyTorch CUDA** | Already native speed via CUDA kernels — never rewrite |
| Temporal tensor assembly | **Python + NumPy** | Stacks 2-plane cluster snapshots + `move_history` into `(18, 19, 19)` tensors |
| Orchestration, training loop | **Python** | Runs ~once per second — Python speed is irrelevant here |

Rust exposes its API to Python via **PyO3**. Import as: `from engine import MCTSTree, Board, ReplayBuffer, SelfPlayRunner, InferenceBatcher`.

---

## Key design decisions already made

- **Board representation**: axial (cube) hex coordinates internally; offset 2D array for tensor input
- **State tensor**: 18 channels — 2×8 history planes (current/opponent stones) + 2 meta planes (moves remaining, turn parity)
- **Network**: ResNet-12 with 128 filters, SE blocks on every residual block (reduction ratio 4); policy head → `board_size² + 1` logits; value head → dual-pooling (global avg+max → FC), BCE loss
- **Auxiliary loss**: opponent reply prediction head (weight 0.15)
- **MCTS**: batched leaf evaluation — leaves queued across N parallel games, single GPU forward pass per batch
- **Turn structure**: `moves_remaining` field in game state (1 or 2); encoded as feature plane so network learns the difference
- **Reward**: terminal only by default (+1/-1); optional shaped intermediate rewards that decay to zero over training
- **Bootstrap phase**: supervised pretraining from minimax-generated games before self-play begins (see `04_BOOTSTRAP_STRATEGY.md`)
- **Monitoring**: event-driven fan-out (events.py → passive renderers). Dashboards never block training. Terminal (Rich, 4Hz max) and web (Flask+SocketIO on :5001) renderers consume identical event stream. Viewer is isolated — `hexo_rl/viewer/` is never imported by training path.

---

## What "done" looks like (north star)

- Elo consistently above the strongest known human players in the community
- Opening book analysis mode: query what the model thinks of any position
- Training fully automated: run `python scripts/train.py` and walk away
- Reproducible: configs + seeds produce the same training run
- Benchmarks pass: MCTS ≥ 140,000 sim/s, GPU util ≥ 85% during training

---

## Pointer to other docs

| File | Read when you need to... |
|---|---|
| `01_architecture.md` | Understand the full technical design of each component |
| `02_roadmap.md` | Know which phase we're in and what the next milestone is |
| `03_tooling.md` | Set up logging, benchmarking, progress display, dev environment |
| `05_community_integration.md` | Community bot integration, API, notation, formations |
| `06_OPEN_QUESTIONS.md` | Active research questions and ablation plans |
| `08_DASHBOARD_SPEC.md` | Monitoring event schema and dashboard specification |
| `09_VIEWER_SPEC.md` | Game viewer, threat overlay, and play-against-model spec |
