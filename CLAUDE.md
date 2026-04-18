# Response style — apply from first word
Respond like smart caveman. Cut all filler, keep technical substance.
- Drop articles (a, an, the), filler (just, really, basically, actually).
- Drop pleasantries (sure, certainly, happy to).
- No hedging. Fragments fine. Short synonyms.
- Technical terms stay exact. Code blocks unchanged.
- Pattern: [thing] [action] [reason]. [next step].
- Surface tradeoffs — dont defer decisions.

---

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

**Context first, benchmarking mandatory.** Never suggest an architectural change without first reading the relevant `docs/` and `engine` source. Never commit a performance-sensitive change (MCTS, NN, Buffer) without running `make bench` and verifying no regressions against the baseline.

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

- Pretrain v3b validated: 18-plane trunk, final policy_loss ≈ 2.18, ≥95/100 wins vs RandomBot (§93).
- Dirichlet root noise ported to Rust training path (§73, commit `71d7e6e`). Resolves Q17 mode-collapse.
- Named variants live: `gumbel_full`, `gumbel_targets`, `baseline_puct`, `gumbel_targets_desktop` in `configs/variants/` (§67, §81, §96).
- Input layout: 18 planes. Chain-length planes (Q13) moved out of input to replay-buffer aux sub-buffer (§97).
- Normalization: GroupNorm(8) throughout trunk; BatchNorm removed (§99). Pre-§99 checkpoints refuse to load.
- Playout cap: move-level selective policy loss active (`full_search_prob: 0.25`, `n_sims_quick: 100`, `n_sims_full: 600`). Game-level `fast_prob` disabled; mutex enforced at pool init (§100).
- Graduation gate: `best_model` anchor; promotion requires `wr_best ≥ 0.55` AND `ci_lo > 0.5` over 400 games (§101, §101.a; raised 200→400 per calibration 2026-04-17).
- Dashboard rebuilt: event-driven fan-out (terminal + web at :5001). `loss_chain/ownership/threat` surfaced (§82, §93 C14).
- Game viewer live at `/viewer` with threat overlay; `/analyze` endpoint for interactive policy inspection (§78).
- Benchmark rebaselined 2026-04-16 post-18ch migration (§98). 8/10 metrics PASS; worker-throughput target recalibrated to ≥250K pos/hr with warmup-artifact caveat.
- In flight: laptop `gumbel_targets` exp D + desktop `gumbel_full` exp E (§96). Sustained 24-48hr run gated on graduation fire-rate confirmation.
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
make build            # build/install Rust extension via maturin
make clean            # remove Rust build artifacts and Python caches
make rebuild          # full clean + optimized rebuild

# Testing
make test.rust        # Rust tests
make test.py          # Python tests (excludes slow)
make test.slow        # slow/integration tests

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
make eval             # run eval pipeline (see configs/eval.yaml)
# Eval game diversity: eval_temperature, eval_random_opening_plies, eval_seed_base in configs/eval.yaml (§80)

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
│   │   │   ├── mod.rs               ← ReplayBuffer struct + #[pymethods] facade
│   │   │   ├── storage.rs           ← new, resize, weight schedule, dashboard stats
│   │   │   ├── push.rs              ← push, push_game, test-only push_raw
│   │   │   ├── sample.rs            ← sample_batch entry + weighted sample + apply_sym
│   │   │   ├── persist.rs           ← HEXB v5 save/load (v4 legacy read)
│   │   │   └── sym_tables.rs        ← 12-fold permutation tables + WeightSchedule
│   │   ├── game_runner/             ← SelfPlayRunner (Rust worker threads)
│   │   │   ├── mod.rs               ← SelfPlayRunner struct + #[pymethods] facade + Drop
│   │   │   ├── worker_loop.rs       ← start_impl — per-worker self-play main loop
│   │   │   ├── gumbel_search.rs     ← GumbelSearchState (Sequential Halving)
│   │   │   └── records.rs           ← policy aggregation + game-end aux reprojection
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
│   ├── training/                    ← Trainer, losses, checkpoints, and helpers
│   │   ├── trainer.py               ← Trainer: forward/backward/optim/scheduler/save/load
│   │   ├── losses.py                ← policy, value, aux, uncertainty loss functions
│   │   ├── checkpoints.py           ← save/load/prune checkpoint helpers
│   │   ├── recency_buffer.py        ← Python-side recent-positions ring buffer
│   │   ├── aux_decode.py            ← u8→fp32 ownership/winning_line decode + mask helper
│   │   ├── batch_assembly.py        ← pre-alloc buffers, corpus load, mixed-batch concat
│   │   └── loop.py                  ← run_training_loop: inf model, pool, dashboards, main loop
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
│   ├── train.py                     ← CLI + config merge + build core objects → run_training_loop
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

### Threat-logit probe — run at step 5k and before any checkpoint promotion

Run `make probe.latest` (1) at training step 5000 as the kill criterion for every
new sustained run, and (2) before promoting any checkpoint to "best". Exit code 0 =
PASS; code 1 = FAIL; code 2 = error.

Current criterion (§91, revised from §85/§89):

- C1: `contrast_mean ≥ max(0.38, 0.8 × bootstrap_contrast)`
- C2: `ext_in_top5_pct ≥ 25`
- C3: `ext_in_top10_pct ≥ 40`
- C4 (warning only, does not gate): `abs(ext_logit_mean − bootstrap_ext_logit_mean) < 5.0`

C1–C3 must all PASS. C4 prints a `WARNING` line; it is a BCE-drift canary
(Q19 monitoring hook) and never flips the exit code. Full rationale in
§91. Baseline JSON lives at `fixtures/threat_probe_baseline.json` (v4 post-§93).

---

## Benchmarks — must pass before Phase 4.5

> **Methodology:** median of n=5 runs. Per-metric warm-up: 3s MCTS / 3s NN /
> 2s buffer / **90s worker pool** (raised from 30s at §98 to eliminate the
> 0-position measurement windows — §102).
> MCTS uses realistic workload: 800 sims/move × 62 iterations with
> tree reset between moves (matches selfplay.yaml mcts.n_simulations).
> Worker pool: 200 sims/move × max_moves=128, pool_duration=120s (`make bench`).
> CPU frequency unpinned (cpupower unavailable on this system).
> Targets set at `min(observed_median × 0.85, prior_target)` — **conservative**;
> see §102 for target-setting rules.
> Run `make bench` to reproduce. Full results: reports/benchmarks/

Latest baseline **2026-04-17** post-§102 rebaseline (laptop Ryzen 7 8845HS
+ RTX 4060, 14 workers, no CPU pin, LTO + native CPU, 18-plane model,
GroupNorm(8) per §99). Run: `reports/benchmarks/bench_2026-04-17.json`:

| Metric | Baseline (median, n=5) | Target | Notes |
|---|---|---|---|
| MCTS (CPU only, no NN) | 56,404 sim/s | ≥ 26,000 sim/s | IQR ±178 (0.3%); stable vs 2026-04-16 (55.5k) |
| NN inference (batch=64) | 7,676.5 pos/s | ≥ 6,500 pos/s | IQR ±1.2 (0.02%); **down 22% vs §98 — §102 tracks as driver/boost-clock drift (see §72 precedent), NOT code regression** |
| NN latency (batch=1, mean) | 2.19 ms | ≤ 3.5 ms | IQR ±0.55 (25%); target passes, IQR flags launch/sync jitter |
| Replay buffer push | 618,552 pos/sec | ≥ 525,000 pos/sec | IQR ±5,868 (1%); down 19% vs 762k baseline — treat as driver/kernel-cache drift |
| Replay buffer sample raw (batch=256) | 1,379 µs/batch | ≤ 1,500 µs | IQR ±36 (2.6%) |
| Replay buffer sample augmented (batch=256) | 1,241 µs/batch | ≤ 1,800 µs | IQR ±22 (1.8%); **improved vs §98 (1,663 ±566) — do not tighten target on one run** |
| GPU utilization | 100.0% | ≥ 85% | IQR ±0.1; saturated during inference-only benchmark |
| VRAM usage (process) | 0.12 GB / 8.6 GB | ≤ 6.88 GB (80%) | torch.cuda.max_memory_allocated (process-specific, not pynvml global) |
| Worker throughput | 167,755 pos/hr | ≥ 142,000 pos/hr (**PROVISIONAL**) | IQR ±9,601 (5.7%) — 90s warmup fixed §98 variance (was 0-364k, IQR 188%). 3.52× production 47,650 pos/hr (see §102) |
| Batch fill % | 97.49% | ≥ 84% | IQR ±1.1 |

Historical variance note: before the warm-up/n=5/pinning methodology, single-run
benchmarks showed ±50% swings due to LLVM codegen lottery and AMD boost clocks.
See `docs/03_TOOLING.md` § "Benchmark variance (historical)" for details.
2026-04-06: rebaseline on laptop (Ryzen 7 8845HS + RTX 4060). MCTS sim/s higher
than prior desktop baseline due to faster single-thread IPC. All 10 targets PASS.
2026-04-09: NN inference and worker throughput targets rebaselined after a sustained
NVIDIA driver/boost-clock shift (~14% GPU throughput reduction, persistent across
cold/hot/idle runs, not a code regression). See `archive/bench_investigation_2026-04-09/verdict.md` and §72.
2026-04-16 (§98): buffer augmented and worker throughput flagged post-18ch migration. Worker warmup-design bug (30s insufficient → 0-pos windows) unresolved.
2026-04-17 (§102): worker warmup raised to 90s; worker IQR dropped from 188% to 5.7%. New conservative baseline set at `observed × 0.85`. NN inference and buffer push continue drifting downward across runs — tracked, not codified as regression. See `reports/bench_physical_check_2026-04-17.md`.

## Phase 4.0 architecture baseline

Starting config for self-play RL (do not exceed without benchmarking):

- Network: 12 residual blocks × 128 channels, GroupNorm(8), SE blocks on every block (§99).
- Input: 18 planes (§97). Chain-length planes (Q13) stored in ReplayBuffer
  `chain_planes` sub-buffer, not in the input tensor.
- Value head: global avg + max pooling → Linear(2C → 256) → ReLU → Linear(256 → 1) → tanh.
  Loss: BCE on the pre-tanh logit against `(z+1)/2`.
- Auxiliary heads (training only — never called from InferenceServer / evaluator / MCTS):
  - opp_reply — mirror of policy head, cross-entropy, weight 0.15.
  - ownership — Conv(1×1) → tanh → (19×19), spatial MSE, weight 0.1.
  - threat — Conv(1×1) → raw logit → (19×19), BCEWithLogitsLoss with
    `pos_weight = threat_pos_weight` (default 59.0, Q19), weight 0.1.
  - chain_head — Conv(1×1) → (6, 19, 19), smooth-L1 (Huber), weight
    `aux_chain_weight: 1.0` (§92; target comes from the replay-buffer
    chain sub-buffer post-§97, not from the input slice).
- Temperature: per-compound-move quarter-cosine schedule with hard
  `temp_min: 0.05` floor at compound_move ≥ 15 (Rust:
  `engine/src/game_runner/worker_loop.rs:510-515`). **Docs-vs-code drift
  vs §36 half-cosine-per-ply flagged in §70 C.1 — unresolved.**
- ZOI: candidate moves restricted to hex-distance ≤ 5 of last 16 moves
  (fallback to full legal set if < 3 candidates) — post-search move
  selection only; does not reduce MCTS tree branching (§77).
- Checkpoint loading: pre-§99 (BatchNorm) checkpoints refuse to load —
  `normalize_model_state_dict_keys` raises `RuntimeError` rather than
  silently corrupting trunk weights via `strict=False`. Retrain from
  `bootstrap_model.pt` when crossing §99.
- torch.compile: DISABLED — Python 3.14 CUDA graph incompatibility
  (sprint §25, §30, §32). Re-enable when PyTorch + Python 3.14 CUDA
  graph support stabilizes.
- Replay buffer: start at 250K samples, grow toward 1M as training
  stabilises (§79). HEXB v5 on-disk format (v4 legacy read).
- Graduation gate (§101, §101.a): self-play workers consume `inf_model`
  weights, which track the `best_model` anchor (not `trainer.model`).
  Sync fires only on graduation or on cold-start load. Gate is
  two-part: `wr_best ≥ promotion_winrate` (default 0.55 over 400 games;
  raised 200→400 per calibration 2026-04-17)
  AND `ci_lo > 0.5` (binomial 95% CI). CI guard cuts false-positive
  rate at n=400 from ~9% to <1% under null. Promotion copies from the
  `eval_model` snapshot (the one that was actually scored), not from
  drifted `trainer.model`. Eval cadence split via per-opponent
  `stride`: effective eval_interval is 5000 steps (`training.yaml`
  overrides `eval.yaml` per §101 H1); best_checkpoint every 5000
  (`stride: 1`), SealBot every 20000 (`stride: 4`), random every 5000
  (`stride: 1`).
- Selective policy loss (§100): per-move coin-flip chooses full-search
  (600 sims) vs quick-search (100 sims). Policy / opp_reply losses
  gated on `is_full_search=1`; value / chain / ownership / threat
  losses apply to all rows. Mutex with game-level `fast_prob` enforced
  at pool init (`fast_prob: 0.0` base; `full_search_prob: 0.25`).
- ELO benchmark target: SealBot (replaces Ramora0 as external reference).
- Gumbel MCTS (per-variant, not per-host). `configs/selfplay.yaml` base
  is `gumbel_mcts: false, completed_q_values: false`; enable via
  `--variant`:
  - `gumbel_full` — Gumbel root search + completed-Q targets. Desktop
    Phase 4.0 sustained run (`gumbel_full`, n_workers=10 per §81 D3).
  - `gumbel_targets` — PUCT search + completed-Q targets (P3 sweep
    winner per §69; `max_game_moves: 200` post-§76; `inf_bs=64,
    wait_ms=4` post-§90).
  - `baseline_puct` — PUCT + CE visit targets. Ablation baseline.
  Gumbel provides root noise by construction; Dirichlet is additionally
  applied post-§73 in both branches.

Resolved before / during Phase 4.0:

- [x] Open Question 6: sequential vs compound action space
- [x] Open Question 5: supervised→self-play transition schedule
- [x] Q13: chain-length planes (§92 landed as input, §97 moved to aux
      sub-buffer)
- [x] Q17: self-play mode collapse (Dirichlet port §73, commit `71d7e6e`)
- [x] Q19: threat-head BCE class imbalance (`threat_pos_weight = 59.0`;
      §92 landing)
- [x] Q25: 24-plane throughput variance (reverted by §97; the 24-plane
      payload no longer exists)
- [ ] Open Question 2: value aggregation strategy (min/mean/attention) — HIGH
- [ ] Q3, Q8, Q9, Q10, Q15, Q16, Q18, Q21 — see `docs/06_OPEN_QUESTIONS.md`

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
