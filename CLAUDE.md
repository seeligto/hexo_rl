# CLAUDE.md — Hex Tac Toe AlphaZero

This file is read automatically by Claude Code at the start of every session.
Read it fully before doing anything. Then read the docs it references.

---

## What this project is

An AlphaZero-style self-learning AI for Hex Tac Toe — hexagonal grid, 6-in-a-row to win,
player 1 opens with 1 move then both players alternate 2 moves per turn.
Target hardware: AMD Ryzen 7 3700x + RTX 3070 + 48GB RAM.

Full context is in `docs/`. Read them in order before starting any task:

- `docs/00_AGENT_CONTEXT.md` — orientation, language boundary, key decisions
- `docs/01_ARCHITECTURE.md` — full technical spec
- `docs/02_ROADMAP.md` — phases with entry/exit criteria (always check current phase)
- `docs/03_TOOLING.md` — logging, benchmarking, progress display conventions
- `docs/04_BOOTSTRAP_STRATEGY.md` — minimax corpus generation and pretraining
- `docs/05_COMMUNITY_INTEGRATION.md` — community bot, API, notation, formations

---

## Working rules

### One feature = one commit

After completing each discrete feature or task, commit immediately before moving on.
Use conventional commit format:

```
feat(env): add axial coordinate board with win detection
feat(mcts): implement PUCT node selection in Rust
feat(training): add FP16 replay buffer with NumPy ring arrays
fix(mcts): apply Ramora0 transposition table bug fix
test(env): add win detection tests for all 3 hex axes
chore(deps): add pyo3, maturin, structlog to dependencies
```

Never batch multiple features into one commit.
Never leave the repo in a broken state between commits.
After each commit, confirm tests still pass before starting the next task.

### Phase discipline

Always check `docs/02_ROADMAP.md` for the current phase before starting work.
Do not implement Phase 2 components while Phase 0 is incomplete.
Each phase has explicit exit criteria — do not advance until they are met.
If you are unsure what phase we are in, check git log for the most recent feat commits.

### Test as you go

Write tests alongside implementation, not after.
The test suite in `tests/` must pass before any commit.
Win detection tests are especially critical — a bug here corrupts all training data.
Run `cargo test` and `pytest` before every commit.

---

## Language and toolchain

| Layer | Language | Notes |
|---|---|---|
| MCTS tree, board logic, win detection | **Rust** | Build with `maturin develop --release` |
| Neural network, training, replay buffer | **Python + PyTorch** | CUDA, FP16, torch.compile |
| Array/batch operations | **NumPy** | Pre-allocated, never allocate during training |
| Orchestration, config, logging | **Python** | structlog (JSON) + rich (console) |

PyO3 exposes Rust to Python. Import as: `from native_core import Board, MCTSTree`

Build commands:
```bash
cd native_core && cargo build --release   # Rust only
maturin develop --release                 # builds + installs Python extension
pytest tests/                             # Python tests
cargo test                                # Rust tests
cargo bench                               # Rust micro-benchmarks
```

---

## Community resources — check live state before implementing

The following resources are **active and may have changed** since the docs were written.
Before implementing anything that touches them, clone/fetch and read current state:

### Ramora0 engine (bootstrap source)
```bash
# Clone the strongest public bot — used to generate training corpus
git clone https://github.com/Ramora0/HexTicTacToe vendor/ramora_engine
# Also clone the shared bot collection / tournament runner
git clone https://github.com/Ramora0/HexTacToeBots vendor/httt_bots
```
Read `vendor/ramora_engine/cpp/engine.h` to understand the interface.
Apply the line-1094 bug fix before using it (documented in `docs/05_COMMUNITY_INTEGRATION.md`).
Read the tournament runner in `vendor/httt_bots` to understand how to run matches.

### Bot API spec — DRAFT, not final
```bash
# Fetch current spec before implementing the API server
curl -L https://raw.githubusercontent.com/hex-tic-tac-toe/htttx-bot-api/main/definitions/bot-api-v1.yaml \
  -o docs/reference/bot-api-v1.yaml
```
Read the downloaded YAML. Do not assume our docs reflect the current spec.
The community explicitly noted this spec is still evolving.
Implement only what the current YAML actually requires.

### Notation standard — DRAFT, not final
```bash
# Fetch current notation spec
git clone https://github.com/hex-tic-tac-toe/hexagonal-tic-tac-toe-notation \
  docs/reference/notation
```
Read it before implementing the BKE parser.
Our docs describe the notation as we understood it — the repo is ground truth.

---

## Repository layout (target)

```
hex_tac_toe_az/
├── CLAUDE.md                        ← this file
├── docs/
│   ├── 00_AGENT_CONTEXT.md
│   ├── 01_ARCHITECTURE.md
│   ├── 02_ROADMAP.md
│   ├── 03_TOOLING.md
│   ├── 04_BOOTSTRAP_STRATEGY.md
│   ├── 05_COMMUNITY_INTEGRATION.md
│   └── reference/                   ← downloaded community specs (git-ignored)
│       ├── bot-api-v1.yaml
│       └── notation/
├── native_core/                     ← Rust crate
│   ├── src/
│   │   ├── board/                   ← bitboard, axial coords, win detection
│   │   ├── mcts/                    ← PUCT tree, node pool, virtual loss
│   │   ├── formations/              ← formation detection (incremental)
│   │   └── lib.rs                   ← PyO3 bindings
│   ├── benches/
│   └── Cargo.toml
├── python/
│   ├── model/                       ← ResNet + dual heads
│   ├── selfplay/                    ← worker pool, inference server
│   ├── training/                    ← trainer, replay buffer, loss
│   ├── bootstrap/                   ← Ramora wrapper, corpus gen, pretrain
│   ├── eval/                        ← tournament, Elo, SPRT
│   ├── opening_book/                ← BKE parser, named openings
│   ├── api/                         ← bot API HTTP server (FastAPI)
│   └── logging/                     ← structlog config, rich dashboard, metrics
├── configs/
│   ├── default.yaml
│   └── fast_debug.yaml              ← tiny model, board_size=9, 50 sims
├── scripts/
│   ├── train.py
│   ├── benchmark.py
│   ├── serve_bot.py
│   └── watch_game.py
├── tests/
│   ├── test_board.py
│   ├── test_mcts.py
│   ├── test_formations.py
│   └── test_bke_parser.py
├── vendor/                          ← git-ignored, populated by setup script
│   ├── ramora_engine/
│   └── httt_bots/
├── .gitignore
├── pyproject.toml
└── Cargo.toml                       ← workspace root
```

---

## Coding conventions

- Never hardcode hyperparameters in source files — everything goes in `configs/`
- Never log inside MCTS inner loops — only at game boundaries
- Pre-allocate NumPy arrays at init, never allocate during training
- All structured logs via `structlog` (JSON to file), all console output via `rich`
- Config loaded via `yaml.safe_load`, passed as dict through the call stack
- Seed everything: `random`, `numpy`, `torch`, `torch.cuda` — log the seed used
- Type hints on all Python function signatures
- Rust: prefer flat pre-allocated node pools over per-node heap allocation

---

## Benchmarks — must pass before Phase 3

Run `python scripts/benchmark.py` to check. Phase 2 does not complete until:

| Metric | Target |
|---|---|
| MCTS simulations/sec | ≥ 10,000 |
| NN inference (batch=64) | ≥ 5,000 pos/sec |
| GPU utilization during training | ≥ 80% |
| VRAM usage | ≤ 6 GB |
| Self-play games/hour | ≥ 500 |

---

## If you are unsure about anything

1. Check `docs/` first
2. Check the live community repos (clone/fetch as above)
3. Check git log to understand what has already been implemented
4. Ask before making architectural decisions that contradict the docs
