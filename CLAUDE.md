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

- `docs/00_agent_context.md` — orientation, language boundary, key decisions
- `docs/01_architecture.md` — full technical spec
- `docs/02_roadmap.md` — phases with entry/exit criteria (always check current phase)
- `docs/03_tooling.md` — logging, benchmarking, progress display conventions
- `docs/04_bootstrap_strategy.md` — minimax corpus generation and pretraining
- `docs/05_community_integration.md` — community bot, API, notation, formations
- `docs/06_OPEN_QUESTIONS.md` — active research questions and ablation plans
- `docs/07_PHASE4_SPRINT_LOG.md` — Phase 4.0 sprint changelog (most current record)
- `docs/08_DASHBOARD_SPEC.md` — monitoring event schema and dashboard spec
- `docs/09_VIEWER_SPEC.md` — game viewer and threat overlay spec

---

## Prime Directive

**Context first, benchmarking mandatory.** Never suggest an architectural change without first reading the relevant `docs/` and `engine` source. Never commit a performance-sensitive change (MCTS, NN, Buffer) without running `make bench.full` and verifying no regressions against the baseline.

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

- Pretrain validated: policy loss 5.0 → 2.07, ≥95/100 wins vs RandomBot.
- First self-play run: 4,940 steps, found 5 issues, all fixed.
- Dashboard rebuilt: event-driven fan-out (terminal + web at :5001).
- Game viewer live at `/viewer` with threat overlay and replay controls.
- Threat detection implemented in Rust (`Board.get_threats()`).
- Benchmark rebaselined 2026-04-03 (correct 12-block model). All 10 metrics PASS.
- Ready for sustained 24-48hr training run (Phase 4.0 exit criterion).
Next milestone: Phase 4.5 (benchmark gate — see docs/02_roadmap.md).

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

### Config override discipline

Configs are split by concern: `configs/model.yaml`, `configs/training.yaml`,
`configs/selfplay.yaml`, `configs/monitoring.yaml` (plus `eval.yaml`, `corpus.yaml`
for those subsystems). `train.py` deep-merges all base configs — later files win
on overlapping keys. If a hyperparameter appears in multiple files, **all must be
updated**. Verify with:

```bash
grep -r 'key_name' configs/
```

Never assume a key in one config file is the effective value — `load_config()`
merges them and logs warnings on key overlap, but a stale value in any file can
silently override.

### Kill running processes before starting new ones

**Always kill any lingering training or benchmark processes before launching a new one.**
Running multiple training processes simultaneously will saturate the GPU and RAM,
freeze the machine, and corrupt checkpoint state.

```bash
# Kill before any make train.*, make bench.*, or direct scripts/train.py invocations:
# IMPORTANT: pkill returns exit code 1 if no process matched. Use || true to prevent
# this from propagating as an error and aborting subsequent commands in the same shell.
pkill -f "scripts/train.py" 2>/dev/null || true
pkill -f "scripts/benchmark.py" 2>/dev/null || true
sleep 1
pgrep -fl "train.py\|benchmark.py" || echo "clear"
```

Also use `make train.stop` before starting any new training run if a background
run may be active (check with `make train.status`).

### Session start protocol

At the start of every session, in this order:

1. Read this file (CLAUDE.md)
2. Check the memory MCP for stored phase progress and notes from previous sessions
3. Kill any lingering training/benchmark processes (see above)
4. Run baseline checks with Make:

- `make env.check`
- `make test.rust`
- `make test.py`

1. Check `git log --oneline -20` to understand what was last committed
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
| MCTS tree, board logic, win detection | **Rust** | Build with `maturin develop --release -m engine/Cargo.toml`. Concurrency via Rust-native Game-Level Parallelism (Phase 3.5). |
| Replay buffer | **Rust** (ReplayBuffer) | f16-as-u16 ring buffer, 12-fold hex augmentation, zero-copy PyO3 transfer. |
| Neural network, training loop | **Python + PyTorch** | CUDA, FP16, TF32 enabled. InferenceServer bridges Rust worker threads. |
| Temporal tensor assembly | **Python + NumPy** | Stacks 2-plane cluster snapshots + `move_history` into `(18, 19, 19)` tensors. |
| Orchestration, config, monitoring | **Python** | Event-driven fan-out (events.py → terminal/web renderers), structlog (JSON), rich (console) |

PyO3 exposes Rust to Python. Import as: `from engine import Board, MCTSTree, ReplayBuffer, SelfPlayRunner, InferenceBatcher`

**Always use the project virtual environment (`.venv`).** The `maturin develop` command
installs the Rust extension into `.venv`. Never copy the `.so` to the system site-packages
or install packages outside the venv. Run Python commands via `.venv/bin/python` or
activate the venv first (`source .venv/bin/activate`). The Makefile targets (`make test.py`,
`make train`, etc.) already use the venv python.

Build commands:

```bash
cd engine && cargo build --release               # Rust only
maturin develop --release -m engine/Cargo.toml  # builds + installs Python extension
pytest tests/                                    # Python tests
cargo test                                       # Rust tests
cargo bench                                      # Rust micro-benchmarks
```

Make commands (preferred):

```bash
# Core
make env.check        # verify .venv + engine import
make build            # build/install Rust extension via maturin
make clean            # remove Rust build artifacts and Python caches
make rebuild          # full clean + optimized rebuild

# Testing
make test.rust        # Rust tests
make test.py          # Python tests (excludes slow/integration)
make test.all         # Rust + Python tests
make test.focus       # buffer/inference/pool smoke tests
make test.integration # lifecycle integration test (~2-5 min, slow)
make ci               # full pre-push gate (all tests + quick benchmark)

# Benchmarks
make bench            # benchmark (n=5, warm-up; full Phase 4.5 gate)

# Training
make train            # train with web + terminal dashboard (default)
make train DASHBOARD=0  # train without dashboard
make train.bg         # background training (logs to logs/)
make train.stop       # stop background training
make train.status     # check if running, show recent log
make train.resume     # resume from latest checkpoint
make train.smoke      # 200-step smoke test
make dash.open        # open web dashboard in browser

# Eval
make eval.sealbot.quick   # quick eval (10 games, 0.1s, 64 sims)
make eval.sealbot.full    # full eval (100 games, 0.5s, 128 sims)
make eval.sealbot.latest  # eval latest checkpoint vs SealBot

# Plotting
make plot.train.latest    # plot latest training log
make plot.sealbot.latest  # plot latest SealBot eval
make plot.sealbot.all     # plot SealBot trend

# Corpus & pretrain
make corpus.fetch     # scrape human games + generate SealBot corpus
make corpus.export    # export corpus to data/bootstrap_corpus.npz
make pretrain         # full bootstrap pretrain (15 epochs)
```

Run `make help` for the complete list of targets.

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
4. Write a `BotProtocol` wrapper in `hexo_rl/bootstrap/bots/`
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
compilation step is needed. The wrapper at `hexo_rl/bootstrap/bots/sealbot_bot.py`
adds `vendor/bots/sealbot` to `sys.path` and imports `minimax_cpp.MinimaxBot`.

The agent must read the actual README/build instructions in the submodule
before writing the build command — do not guess.

---

## Bot protocol — all bots are interchangeable

Every game source implements `BotProtocol` (hexo_rl/bootstrap/bot_protocol.py).
This makes all bots swappable for corpus generation and evaluation.

```python
class BotProtocol(ABC):
    @abstractmethod
    def get_move(self, state: GameState) -> tuple[int, int]: ...
    @abstractmethod
    def name(self) -> str: ...

# Wrappers live in hexo_rl/bootstrap/bots/:
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
Scraper: hexo_rl/bootstrap/scraper.py — see docs/04_bootstrap_strategy.md.
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

## Repository layout (current)

```
hexo_rl/
├── CLAUDE.md
├── docs/
│   ├── 00_agent_context.md
│   ├── 01_architecture.md
│   ├── 02_roadmap.md
│   ├── 03_tooling.md
│   ├── 04_bootstrap_strategy.md
│   ├── 05_community_integration.md
│   ├── 06_OPEN_QUESTIONS.md
│   ├── 07_PHASE4_SPRINT_LOG.md
│   ├── 08_DASHBOARD_SPEC.md
│   ├── 09_VIEWER_SPEC.md
│   └── reference/                   ← downloaded specs (git-ignored)
├── engine/
│   ├── src/
│   │   ├── board/                   ← sparse HashMap board, axial coords, win detection
│   │   │   └── threats.rs           ← threat detection (3 hex axes, sliding window)
│   │   ├── mcts/                    ← PUCT tree, node pool, virtual loss
│   │   ├── formations/              ← incremental formation detection
│   │   ├── replay_buffer/           ← f16-as-u16 ring buffer, 12-fold augmentation
│   │   ├── game_runner.rs           ← SelfPlayRunner (Rust worker threads)
│   │   ├── inference_bridge.rs      ← InferenceBatcher (Rust→Python GPU queue)
│   │   └── lib.rs
│   └── Cargo.toml
├── hexo_rl/
│   ├── bootstrap/
│   │   ├── bot_protocol.py          ← BotProtocol ABC
│   │   ├── bots/
│   │   │   ├── sealbot_bot.py       ← SealBotBot wrapper
│   │   │   ├── our_model_bot.py     ← OurModelBot wrapper
│   │   │   ├── random_bot.py        ← RandomBot
│   │   │   └── community_api_bot.py ← CommunityAPIBot (HTTP)
│   │   ├── scraper.py               ← hexo.did.science scraper
│   │   ├── generate_corpus.py       ← orchestrates all corpus sources
│   │   ├── corpus_analysis.py       ← corpus quality analysis
│   │   ├── injection.py             ← human-seed bot-continuation injection
│   │   ├── opening_classifier.py    ← opening pattern classifier
│   │   ├── human_seeding.py         ← human game seeding for bot games
│   │   └── pretrain.py
│   ├── corpus/                      ← corpus pipeline and metrics
│   │   └── sources/                 ← pluggable corpus sources (human, hybrid)
│   ├── env/                         ← GameState, tensor assembly
│   ├── eval/                        ← Bradley-Terry, eval pipeline, results DB, colony detection
│   ├── model/                       ← HexTacToeNet (ResNet-12, SE blocks, dual-pool value head)
│   ├── monitoring/                  ← event-driven monitoring fan-out
│   │   ├── events.py                ← register_renderer(), emit_event() dispatcher
│   │   ├── terminal_dashboard.py    ← Rich Live renderer (4Hz max)
│   │   ├── web_dashboard.py         ← Flask+SocketIO server (:5001)
│   │   ├── gpu_monitor.py           ← GPU utilization daemon thread
│   │   ├── game_recorder.py         ← game persistence
│   │   ├── game_browser.py          ← game index/browser
│   │   ├── replay_poller.py         ← game replay analysis
│   │   ├── metrics_writer.py        ← metrics persistence
│   │   ├── configure.py             ← structlog/logging setup
│   │   └── static/                  ← index.html (dashboard SPA), viewer.html (game viewer SPA)
│   ├── opening_book/                ← game record parser
│   ├── selfplay/                    ← inference server, worker pool, policy projection
│   ├── training/                    ← Trainer, checkpoints, losses
│   ├── utils/                       ← config loader, constants
│   └── viewer/                      ← game viewer engine
│       └── engine.py                ← ViewerEngine (enrich_game, play_response)
├── configs/
│   ├── model.yaml                   ← network architecture (res_blocks, channels, SE)
│   ├── training.yaml                ← optimizer, scheduler, buffer, mixing
│   ├── selfplay.yaml                ← MCTS sims, workers, playout cap
│   ├── monitoring.yaml              ← dashboard enable/disable, alerts, web port
│   ├── eval.yaml                    ← eval pipeline, SealBot gate
│   └── corpus.yaml                  ← corpus generation settings
├── scripts/
│   ├── train.py
│   ├── benchmark.py
│   ├── eval_vs_sealbot.py
│   ├── scrape_daily.sh
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

> **Methodology:** median of n=5 runs, 3s warm-up per metric.
> MCTS uses realistic workload: 800 sims/move × 62 iterations with
> tree reset between moves (matches selfplay.yaml mcts.n_simulations).
> CPU frequency unpinned (cpupower unavailable on this system).
> Targets set at 85% of observed median unless otherwise noted.
> Run `make bench.full` to reproduce.
> Full results: reports/benchmarks/

Run `make bench.full`. Latest baseline (2026-04-06, Ryzen 7 8845HS + RTX 4060 Laptop, 16 workers, no CPU pin, LTO + native CPU):

| Metric | Baseline (median, n=5) | Target | Notes |
|---|---|---|---|
| MCTS (CPU only, no NN) | 55,478 sim/s | ≥ 26,000 sim/s | IQR ±400; **pre-Gumbel baseline — re-bench pending** (measured with `gumbel_mcts: false`) |
| NN inference (batch=64) | 9,810 pos/s | ≥ 8,250 pos/s | GPU-bound (IQR ±1); **target rebaselined 2026-04-09** — see §72 |
| NN latency (batch=1, mean) | 1.59 ms | ≤ 3.5 ms | IQR ±0.05 ms |
| Replay buffer push | 762,130 pos/sec | ≥ 630,000 pos/sec | IQR ±114,320 (15%) |
| Replay buffer sample raw (batch=256) | 1,037 µs/batch | ≤ 1,500 µs | IQR ±34 µs |
| Replay buffer sample augmented (batch=256) | 940 µs/batch | ≤ 1,400 µs | IQR ±62 µs |
| GPU utilization | 100.0% | ≥ 85% | Saturated during inference-only benchmark |
| VRAM usage (process) | 0.05 GB / 8.0 GB | ≤ 6.4 GB | torch.cuda.max_memory_allocated (process-specific, not pynvml global) |
| Worker throughput | 659,983 pos/hr | ≥ 500,000 pos/hr | IQR ±56,835 (8.6%); **target rebaselined 2026-04-09** — see §72 |
| Batch fill % | 100.0% | ≥ 80% | IQR ±0.0% |

Historical variance note: before the warm-up/n=5/pinning methodology, single-run
benchmarks showed ±50% swings due to LLVM codegen lottery and AMD boost clocks.
See `docs/03_TOOLING.md` § "Benchmark variance (historical)" for details.
2026-04-06: rebaseline on laptop (Ryzen 7 8845HS + RTX 4060). MCTS sim/s higher
than prior desktop baseline due to faster single-thread IPC. All 10 targets PASS.
2026-04-09: NN inference and worker throughput targets rebaselined after a sustained
NVIDIA driver/boost-clock shift (~14% GPU throughput reduction, persistent across
cold/hot/idle runs, not a code regression). See `archive/bench_investigation_2026-04-09/verdict.md` and §72.

## Phase 4.0 architecture baseline

Starting config for self-play RL (do not exceed without benchmarking):

- Network: 12 residual blocks × 128 channels, SE blocks on every block
- Value head: global avg + max pooling → FC → BCE loss (binary cross-entropy on sigmoid)
- Auxiliary loss: opponent reply prediction (weight 0.15)
- Auxiliary heads: ownership (spatial MSE, weight 0.1) + threat (BCE, weight 0.1) added alongside existing opponent reply head (weight 0.15)
- Temperature: cosine-annealed 1.0 → 0.05 (replaces hard step at move 30)
- ZOI: candidate moves restricted to hex-distance ≤ 5 of last 16 moves (fallback to full legal set if < 3 candidates)
- Checkpoint loading: strict=False required when resuming — new head weights not present in pre-§37 checkpoints
- torch.compile: DISABLED — Python 3.14 CUDA graph incompatibility (see sprint §25, §30, §32)
  - Re-enable when PyTorch + Python 3.14 CUDA graph support stabilizes
  - Weight sync: inf_model ← train_model after every checkpoint save and model promotion
- Replay buffer: start at 250K samples, grow toward 1M as training stabilises
- ELO benchmark target: SealBot (replaces Ramora0 as external reference)
- Gumbel MCTS (per-host override):
  - Gumbel-Top-k root sampling replaces PUCT exploration at root (Danihelka et al., ICLR 2022)
  - Sequential Halving budget allocation across halving phases
  - Non-root nodes: unchanged (PUCT + dynamic FPU)
  - Config: `gumbel_mcts`, `gumbel_m` (default 16), `gumbel_explore_moves` (default 10)
  - **Desktop (3070):** `gumbel_mcts: true` — intentional for Phase 4.0 sustained run (pre-§69 defaults, not yet swept on desktop hardware)
  - **Laptop (8845HS + 4060):** `gumbel_mcts: false`, P3 sweep winner as base config — `ratio=4, burst=16, game_moves=150, wait_ms=4, leaf_bs=8, inf_bs=64, workers=14` (see sprint log §69 for provenance)
  - Completed Q-values (`completed_q_values: true`) provides policy targets for training
  - **Known issue:** `completed_q_values` KL loss path was silently dead due to nested-dict lookup bug in `trainer.py` (see architecture review C1). Fix tracked separately; prior training used CE loss instead of KL.

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

**Note:** `scrape_daily.sh` runs two passes: (1) standard paginated scrape
of the 500-game public window, then (2) top-player profile scrape via
`/api/profiles/:id/games` which can surface games **outside** that window
(up to 10 per player). Per-game Elo is stored in the game JSON
(`player_black_elo`, `player_white_elo`) and the manifest includes an
`elo_bands` breakdown.

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
