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

## Board representation — infinite board strategy

The board is infinite. The NN requires fixed-size tensors. We resolve this as follows:

**Internal storage (Rust):** `HashMap<(q,r), Player>` — sparse, genuinely unbounded.
No allocation for empty cells. No fixed grid size in the data structure.

**NN view window (Multi-Window Clustering):** The board state is dynamically grouped into K distinct clusters (colonies) of stones. The Rust core generates K distinct 19×19 tensors, one centered on each cluster's centroid. These are evaluated as a batch by the neural network.

**Legal moves:** All empty cells within a margin of existing stones, across all K clusters. The network outputs K policy distributions which are mapped back to global coordinates and unified via softmax.

**Phase 0 note:** Phase 0 built a fixed 2D array board. This was migrated to a
sparse HashMap in Phase 1.5 before bootstrap corpus generation began.
If resuming and unsure which is current, check native_core/src/board/mod.rs.

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

### Session start protocol

At the start of every session, in this order:
1. Read this file (CLAUDE.md)
2. Check the memory MCP for stored phase progress and notes from previous sessions
3. Run `cargo test` and `pytest` to confirm clean baseline
4. Check `git log --oneline -20` to understand what was last committed
5. Only then begin work

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
git submodule add https://github.com/Ramora0/HexTicTacToe vendor/bots/ramora
git submodule add https://github.com/Ramora0/HexTacToeBots vendor/bots/httt_collection

# After cloning the repo fresh, initialise submodules:
git submodule update --init --recursive

# To update a submodule to latest upstream:
cd vendor/bots/ramora && git pull origin main && cd -
git add vendor/bots/ramora && git commit -m "chore(vendor): update ramora to latest"
```

### When integrating a new bot

1. Add as submodule (above)
2. Read its source — understand the interface, build system, and move format
3. Check for known bugs (Ramora0 has a documented line-1094 bug — see docs/05_COMMUNITY_INTEGRATION.md)
4. Write a `BotProtocol` wrapper in `python/bootstrap/bots/`
5. Add a build step to `scripts/build_vendor.sh` if it needs compilation
6. Write a smoke test: bot returns a legal move on a fresh board
7. Commit: `feat(bootstrap): add <botname> wrapper`

### Current bot submodules

| Path | Bot | Notes |
|---|---|---|
| `vendor/bots/ramora` | Ramora0/HexTicTacToe | Strongest public bot — apply line-1094 fix before use |
| `vendor/bots/httt_collection` | Ramora0/HexTacToeBots | Community collection + tournament runner |

When the community adds new bots, add them here as submodules. Check the
HexTacToeBots repo and the community Discord periodically for new entries.

### Bot compilation

Ramora0 is C++ and must be compiled before use:

```bash
# scripts/build_vendor.sh — run once after cloning or updating submodules
cd vendor/bots/ramora/cpp
# Apply the line-1094 bug fix first (see docs/05_COMMUNITY_INTEGRATION.md)
g++ -O3 -o engine engine.h   # or whatever the actual build command is
                               # read the repo's README first
```

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
#   ramora_bot.py        — wraps compiled Ramora0 binary at configurable depth
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
URL: https://hexo.did.science/games
Paginated listing of all community games. Filter: rated games, moves > 20.
Scraper: python/bootstrap/scraper.py — see docs/04_BOOTSTRAP_STRATEGY.md.
**Before implementing the scraper:** fetch one game page, inspect the actual HTML
structure, then implement. Do not guess selectors.

### Bot API spec — DRAFT, not final
Deployment target: https://explore.htttx.io/
Spec repo: https://github.com/hex-tic-tac-toe/htttx-bot-api
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
│   │   │   ├── ramora_bot.py        ← RamoraBot wrapper
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
│       ├── ramora/                  ← Ramora0/HexTicTacToe
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

## Benchmarks — must pass before Phase 3

Run `python scripts/benchmark.py`. Phase 2 does not complete until:

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
