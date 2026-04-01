# CLAUDE.md — Hex Tac Toe AlphaZero

This file is read automatically by Claude Code at the start of every session.
Read it fully before doing anything. Then read the docs it references.

---

## What this project is

An AlphaZero-style self-learning AI for Hex Tac Toe — hexagonal grid, 6-in-a-row to win,
player 1 opens with 1 move then both players alternate 2 moves per turn.
The board is theoretically infinite — see "Board representation" below for how we handle this.
Target hardware: AMD Ryzen 7 3700x + RTX 3070 + 48GB RAM.

Full context is in `docs/`. Read them in order before starting any task:

- `docs/00_AGENT_CONTEXT.md` — orientation, language boundary, key decisions
- `docs/01_ARCHITECTURE.md` — full technical spec
- `docs/02_ROADMAP.md` — phases with entry/exit criteria (always check current phase)
- `docs/03_TOOLING.md` — logging, benchmarking, progress display conventions
- `docs/04_BOOTSTRAP_STRATEGY.md` — minimax corpus generation and pretraining
- `docs/05_COMMUNITY_INTEGRATION.md` — community bot, API, notation, formations

---

## Prime Directive

**Context first, benchmarking mandatory.** Never suggest an architectural change without first reading the relevant `docs/` and `native_core` source. Never commit a performance-sensitive change (MCTS, NN, Buffer) without running `make bench.full` and verifying no regressions against the baseline.

---

## Board representation — infinite board strategy

The board is infinite. The NN requires fixed-size tensors. We resolve this as follows:

**Internal storage (Rust):** `HashMap<(q,r), Player>` — sparse, genuinely unbounded.
No allocation for empty cells. No fixed grid size in the data structure.
**Transposition Table (TT):** Uses `FxHashMap` with **128-bit Zobrist hashing** (splitmix128) for O(1) state lookups, critical for MCTS efficiency. 128-bit keys eliminate collision risk at sustained >150k sim/s throughput.

**NN view window (Hybrid Attention-Anchored Windowing):** The board state is dynamically grouped into K distinct clusters (colonies) of stones. The Rust core returns K distinct **2-plane (19×19) cluster snapshots** (current player + opponent stones). Python's `GameState.to_tensor()` stacks these snapshots with `move_history` to assemble the full 18-plane temporal tensor. To prevent "Attention Hijacking" (where the model ignores distant but winning threats), we use **Attention-Anchored Windowing**: windows are centered on high-attention regions and critical formations, not just centroids.

**Value Aggregation (Min-Pooling):** When multiple windows are evaluated for a single board state, the scalar Value ($v$) is aggregated using **Min-Pooling** (from the perspective of the current player) to ensure that if any window contains a losing threat, the entire state is treated as high-risk.

**Legal moves:** All empty cells within a margin of existing stones, across all K clusters. The network outputs K policy distributions which are mapped back to global coordinates and unified via softmax.

---

## Working rules

### One feature = one commit

After completing each discrete feature or task, commit immediately before moving on.
Use conventional commit format:

```
feat(env): add axial coordinate board with win detection
feat(mcts): implement PUCT node selection in Rust
feat(mcts): integrate FxHashMap Transposition Table with Zobrist hashing
feat(training): add FP16 replay buffer with NumPy ring arrays
fix(mcts): apply SealBot transposition table bug fix
test(env): add win detection tests for all 3 hex axes
chore(deps): add pyo3, maturin, structlog to dependencies
```

Never batch multiple features into one commit.
Never leave the repo in a broken state between commits.
After each commit, confirm tests still pass before starting the next task.

### Phase discipline

Always check `docs/02_ROADMAP.md` for the current phase before starting work.

**Current Status:** Phase 4.0 Self-Play RL loop is **Active**.
Blockers cleared: pytest hang, action space verification, bot corpus.
Next milestone: Phase 4.5 (benchmark gate — see docs/02_ROADMAP.md).

Each phase has explicit exit criteria — do not advance until they are met.
If you are unsure what phase we are in, check git log for the most recent feat commits.

### Test as you go

Write tests alongside implementation, not after.
The test suite in `tests/` must pass before any commit.
Win detection tests are especially critical — a bug here corrupts all training data.
Prefer Make targets for consistency:

```bash
make test.rust
make test.py
```

Fallback (if Makefile is unavailable): run `cargo test` and `pytest` directly.

### Session start protocol

At the start of every session, in this order:

1. Read this file (CLAUDE.md)
2. Check the memory MCP for stored phase progress and notes from previous sessions
3. Run baseline checks with Make:

- `make env.check`
- `make test.rust`
- `make test.py`

4. Check `git log --oneline -20` to understand what was last committed
2. Only then begin work

### Session end protocol

Before ending any session or when asked to stop:

1. Finish the current atomic task and commit it
2. Run full test suite — confirm it passes
3. Write a memory note via the memory MCP containing:
   - Current phase and which checklist items are complete
   - Test counts (pytest N passing, cargo test N passing)
   - Any architectural decisions made this session
   - Exact next task to resume from
4. Summarise the above in chat before closing

---

## Language and toolchain

| Layer | Language | Notes |
|---|---|---|
| MCTS tree, board logic, win detection | **Rust** | Build with `maturin develop --release`. Concurrency via Rust-native Game-Level Parallelism (Phase 3.5). |
| Replay buffer | **Rust** (RustReplayBuffer) | f16-as-u16 ring buffer, 12-fold hex augmentation, zero-copy PyO3 transfer. |
| Neural network, training loop | **Python + PyTorch** | CUDA, FP16, TF32 enabled, torch.compile. InferenceServer bridges Rust worker threads. |
| Temporal tensor assembly | **Python + NumPy** | Stacks 2-plane cluster snapshots + `move_history` into `(18, 19, 19)` tensors. |
| Orchestration, config, logging | **Python** | structlog (JSON) + rich (console) |

PyO3 exposes Rust to Python. Import as: `from native_core import Board, MCTSTree, RustReplayBuffer`

Build commands:

```bash
cd native_core && cargo build --release   # Rust only
maturin develop --release                 # builds + installs Python extension
pytest tests/                             # Python tests
cargo test                                # Rust tests
cargo bench                               # Rust micro-benchmarks
```

Make commands (preferred):

```bash
make env.check        # verify .venv + native_core import
make native.build     # build/install Rust extension via maturin
make test.rust        # Rust tests
make test.py          # Python tests (tests/ only)
make test.all         # Rust + Python tests
make bench.lite       # quick benchmark pass
make bench.full       # higher-confidence benchmark pass
make bench.stress     # heavy 5-min stability test
```

---

## Community bots — use existing bots, never build your own minimax

**We do not implement our own minimax or bot heuristics.** The community already has
strong bots. We use them directly as git submodules, read their source to understand
the interface, and wrap them behind BotProtocol. This gives us:

- Stronger corpus data than any minimax we could write
- Diversity of playing styles across multiple bots
- The exact bots the community benchmarks against — so our Elo comparisons are meaningful

### Adding a bot as a submodule (correct way)

Always use `git submodule add` — never clone into a tracked path:

```bash
# Add a bot as a submodule under vendor/bots/
git submodule add https://github.com/Ramora0/SealBot vendor/bots/sealbot
git submodule add https://github.com/Ramora0/HexTacToeBots vendor/bots/httt_collection

# After cloning the repo fresh, initialise submodules:
git submodule update --init --recursive

# To update a submodule to latest upstream:
cd vendor/bots/sealbot && git pull origin main && cd -
git add vendor/bots/sealbot && git commit -m "chore(vendor): update sealbot to latest"
```

### When integrating a new bot

1. Add as submodule (above)
2. Read its source — understand the interface, build system, and move format
3. Check for known bugs (SealBot has a documented colony-bug risk — see docs/05_COMMUNITY_INTEGRATION.md)
4. Write a `BotProtocol` wrapper in `python/bootstrap/bots/`
5. Add a build step to `scripts/build_vendor.sh` if it needs compilation
6. Write a smoke test: bot returns a legal move on a fresh board
7. Commit: `feat(bootstrap): add <botname> wrapper`

### Current bot submodules

| Path | Bot | Notes |
|---|---|---|
| `vendor/bots/sealbot` | Ramora0/SealBot | Strongest public bot — pybind11 minimax engine and primary ELO benchmark target for Phase 4+ |
| `vendor/bots/httt_collection` | Ramora0/HexTacToeBots | Community collection + tournament runner |

When the community adds new bots, add them here as submodules. Check the
HexTacToeBots repo and the community Discord periodically for new entries.

### Bot compilation

SealBot uses pybind11 and is imported directly as a Python module — no separate
compilation step is needed. The wrapper at `python/bootstrap/bots/sealbot_bot.py`
adds `vendor/bots/sealbot` to `sys.path` and imports `minimax_cpp.MinimaxBot`.

The agent must read the actual README/build instructions in the submodule
before writing the build command — do not guess.

---

## Bot protocol — all bots are interchangeable

Every game source implements `BotProtocol` (python/bootstrap/bot_protocol.py).
This makes all bots swappable for corpus generation and evaluation.

```python
class BotProtocol(ABC):
    @abstractmethod
    def get_move(self, state: GameState) -> tuple[int, int]: ...
    @abstractmethod
    def name(self) -> str: ...

# Wrappers live in python/bootstrap/bots/:
#   sealbot_bot.py       — wraps SealBot pybind11 minimax engine
#   our_model_bot.py     — wraps our checkpoint + MCTS
#   random_bot.py        — uniform random (baseline)
#   community_api_bot.py — wraps any bot-api-v1 HTTP endpoint
```

`CommunityAPIBot` is the key one: any community bot at a known URL can be
plugged into corpus generation or evaluation with zero extra code.
Never hardcode which bots generate corpus games — drive from config.

---

## Community resources — check live state before implementing

### Human game archive (bootstrap data — 42k+ real games)

URL: <https://hexo.did.science/games>
Paginated listing of all community games. Filter: rated games, moves > 20.
Scraper: python/bootstrap/scraper.py — see docs/04_BOOTSTRAP_STRATEGY.md.
**Before implementing the scraper:** fetch one game page, inspect the actual HTML
structure, then implement. Do not guess selectors.

### Bot API spec — DRAFT, not final

Deployment target: <https://explore.htttx.io/>
Spec repo: <https://github.com/hex-tic-tac-toe/htttx-bot-api>

```bash
curl -L https://raw.githubusercontent.com/hex-tic-tac-toe/htttx-bot-api/main/definitions/bot-api-v1.yaml \
  -o docs/reference/bot-api-v1.yaml
```

Read the downloaded YAML before implementing anything. Do not assume our docs
reflect the current spec — the repo is ground truth.

### Notation standard — DRAFT, not final

```bash
git clone https://github.com/hex-tic-tac-toe/hexagonal-tic-tac-toe-notation \
  docs/reference/notation
```

Read before implementing the BKE parser.

---

## Repository layout (target)

```
hex_tac_toe_az/
├── CLAUDE.md
├── docs/
│   ├── 00_AGENT_CONTEXT.md
│   ├── 01_ARCHITECTURE.md
│   ├── 02_ROADMAP.md
│   ├── 03_TOOLING.md
│   ├── 04_BOOTSTRAP_STRATEGY.md
│   ├── 05_COMMUNITY_INTEGRATION.md
│   └── reference/                   ← downloaded specs (git-ignored)
├── native_core/
│   ├── src/
│   │   ├── board/                   ← sparse HashMap board, axial coords, win detection
│   │   ├── mcts/                    ← PUCT tree, node pool, virtual loss
│   │   ├── formations/              ← incremental formation detection
│   │   └── lib.rs
│   └── Cargo.toml
├── python/
│   ├── model/
│   ├── selfplay/
│   ├── training/
│   ├── bootstrap/
│   │   ├── bot_protocol.py          ← BotProtocol ABC
│   │   ├── bots/
│   │   │   ├── sealbot_bot.py       ← SealBotBot wrapper
│   │   │   ├── our_model_bot.py     ← OurModelBot wrapper
│   │   │   ├── random_bot.py        ← RandomBot
│   │   │   └── community_api_bot.py ← CommunityAPIBot (HTTP)
│   │   ├── scraper.py               ← hexo.did.science scraper
│   │   ├── generate_corpus.py       ← orchestrates all corpus sources
│   │   └── pretrain.py
│   ├── eval/
│   ├── opening_book/
│   ├── api/
│   └── logging/
├── configs/
├── scripts/
│   ├── build_vendor.sh              ← compiles C++ bots after submodule init
│   └── ...
├── tests/
├── vendor/
│   └── bots/                        ← git submodules
│       ├── sealbot/                 ← Ramora0/SealBot
│       └── httt_collection/         ← Ramora0/HexTacToeBots
└── .gitmodules                      ← submodule tracking (committed)
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
- All bot integrations go through `BotProtocol` — never call a bot binary directly

---

## Testing conventions

### Loss-convergence tests must disable augmentation

Any test that asserts on loss values decreasing over N training steps **must** pass
`augment=False` to `trainer.train_step()`. Example:

```python
loss1 = trainer.train_step(buf, augment=False)
loss2 = trainer.train_step(buf, augment=False)
assert loss2 < loss1
```

**Why:** 12-fold hex augmentation applies a random symmetry transform to each sampled
batch. With augmentation enabled, the effective training distribution varies per call,
introducing RNG-dependent variance that can flip the loss ordering over a short N-step
window. This produces flaky tests even when the optimizer is converging correctly.

**Scope:** This restriction applies only to short-window convergence assertions in unit
tests. Full training runs must always use `augment=True` (the default).

---

## Benchmarks — must pass before Phase 4.5

Run `make bench.full`. Latest baseline (2026-04-01, Ryzen 7 3700x + RTX 3070, 16 workers):

| Metric | Baseline | Target | Notes |
|---|---|---|---|
| MCTS (CPU only, no NN) | 218,385 sim/s | ≥ 180,000 sim/s | Raised from 150k after legal_cache optimisation |
| NN inference (batch=64) | 10,715 pos/s | ≥ 8,000 pos/s | Kept — GPU-bound, may drop under training load |
| NN latency (batch=1, mean) | 0.67 ms (p99: 2.37 ms) | ≤ 2 ms | Tightened from 5 ms — 0.7 ms is stable |
| Replay buffer push | 220,777 pos/sec | ≥ 150,000 pos/sec | Raised from 50k — headroom for larger buffers |
| Replay buffer sample raw (batch=256) | 1,011 µs/batch | ≤ 1,100 µs | Relaxed from 1,000 — codegen-sensitive, ~1,013 stable |
| Replay buffer sample augmented (batch=256) | 933 µs/batch (3.64 µs/pos) | ≤ 1,000 µs | Kept |
| GPU utilization | 90.7% | ≥ 85% | Raised from 80% |
| VRAM usage | 0.77 GB / 8.6 GB | ≤ 80% | Kept |
| Worker throughput | 3,350 games/hr / 1,486,031 pos/hr | ≥ 1,000,000 pos/hr | Raised from 500k — positions/hr is the training-critical metric |
| Batch fill % | 92.7% | ≥ 80% | Raised from 50% — below 80% wastes GPU on padding |

Targets set at worst-case floor across observed LLVM codegen variance
(±50% swing on buffer push is a measurement artifact — see docs/03_TOOLING.md#benchmark-variance).

## Phase 4.0 architecture baseline

Starting config for self-play RL (do not exceed without benchmarking):
- Network: 12 residual blocks × 128 channels, SE blocks on every block
- Value head: global avg + max pooling → FC → cross-entropy loss
- Auxiliary loss: opponent reply prediction (weight 0.15)
- Replay buffer: start at 250K samples, grow toward 1M as training stabilises
- ELO benchmark target: SealBot (replaces Ramora0 as external reference)

Resolved before Phase 4.0 launch:
- [x] Open Question 6: sequential vs compound action space
- [x] Open Question 5: supervised→self-play transition schedule
- [ ] Open Question 2: value aggregation strategy (min/mean/attention)

---

## Background tasks

### Scraping

Run the scrape script in the background at session start if it has not run
in the last 3 hours (check logs/scrape_history.log for the last timestamp).

```bash
nohup ./scripts/scrape_daily.sh > /dev/null 2>&1 &
```

Let it run silently — it takes ~30 seconds and handles its own logging.
After it finishes, commit ONLY data/corpus/manifest.json if the game count changed:

```bash
git add data/corpus/manifest.json
git commit -m "chore(data): update corpus manifest (N games)"
```

Do not commit raw game JSON files. Do not wait for the scrape to finish
before starting other work.

---

## If you are unsure about anything

1. Check `docs/` first
2. Check the live community repos and submodules
3. Check git log to understand what has already been implemented
4. Ask before making architectural decisions that contradict the docs

---

## MCP tools available

- **context7**: use when writing code that uses PyTorch, PyO3, maturin,
  structlog, rich, or any library where API details matter. Call
  resolve_library_id() first, then get_library_docs().

- **github**: use to fetch current versions of community specs and check
  for new bots in the community repos. Requires GITHUB_TOKEN env var.
  If unset, fall back to curl/git as shown above.

- **memory**: record completed phase checklist items, benchmark results, and
  architectural decisions so they persist across sessions. Follow the session
  start and end protocols above.
