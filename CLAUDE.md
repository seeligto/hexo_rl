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

## Board representation

See `docs/rules/board-representation.md` — infinite board, NN windowing, value aggregation.

---

## Workflow

See `docs/rules/workflow.md` — commits, phase discipline, tests, config overrides, process-kill, session start/end, coding + testing conventions, corpus/probe discipline.

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

URL: <https://[site-redacted]/games>
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
│   │   ├── scraper.py               ← [site-redacted] scraper
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

Latest baseline **2026-04-18** (laptop Ryzen 7 8845HS + RTX 4060,
14 workers, no CPU pin, LTO + native CPU, 18-plane model, GroupNorm(8)
per §99). Run: `reports/benchmarks/2026-04-18_18-36.json`. All 10
targets PASS:

| Metric | Baseline (median, n=5) | Target | Notes |
|---|---|---|---|
| MCTS (CPU only, no NN) | 69,680 sim/s | ≥ 26,000 sim/s | IQR ±273 (0.4%); **up 23.5% vs 2026-04-17** — treat as cooler thermals / boost-clock upswing, not a code win |
| NN inference (batch=64) | 7,646 pos/s | ≥ 6,500 pos/s | IQR ±0.73 (0.01%); flat vs 2026-04-17 (−0.4%) |
| NN latency (batch=1, mean) | 1.84 ms | ≤ 3.5 ms | IQR ±0.004 (0.2%); 16% faster than 2026-04-17 (2.19 ms) |
| Replay buffer push | 696,880 pos/sec | ≥ 525,000 pos/sec | IQR ±109,127 (15.7%); up 12.7% vs 2026-04-17 — IQR widened, watch |
| Replay buffer sample raw (batch=256) | 1,496 µs/batch | ≤ 1,550 µs | IQR ±11 (0.7%); up 8.5% vs 2026-04-17, still under target. **§113 2026-04-22:** target recalibrated 1,500→1,550 µs — `cda9dde` always-on dedup adds one HashSet alloc + 256 game_id lookups (correctness-required); residual +33 µs confirmed after push.rs transmute fix recovered all other regressions. |
| Replay buffer sample augmented (batch=256) | 1,654 µs/batch | ≤ 1,800 µs | IQR ±293 (17.7%); up 33% vs 2026-04-17 (1,241) — back inside §98 band, confirms §102 "do not tighten on one run" |
| GPU utilization | 99.9% | ≥ 85% | IQR ±0.1; saturated during inference-only benchmark |
| VRAM usage (process) | 0.08 GB / 8.6 GB | ≤ 6.88 GB (80%) | torch.cuda.max_memory_allocated (process-specific, not pynvml global) |
| Worker throughput | 164,052 pos/hr | ≥ 142,000 pos/hr (**PROVISIONAL**) | IQR ±30,138 (18.4%); flat vs 2026-04-17 (−2.2%); IQR widened from 5.7% — likely the single-run warm-up variance §102 warned about |
| Batch fill % | 99.58% | ≥ 84% | IQR ±0.28 |

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
2026-04-18: new reference baseline, supersedes 2026-04-17. MCTS +23.5% and NN latency −16% — read as cooler thermals / boost-clock upswing, not a code win (the targets were deliberately conservative per §102). Buffer-aug IQR widened 1.8% → 17.7% and worker IQR 5.7% → 18.4% — within historical bench variance, but flag if subsequent runs stay high. Do not tighten targets on this single run.

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
  `compute_move_temperature` in `engine/src/game_runner/worker_loop.rs:20-31`;
  call site at `worker_loop.rs:346-353`). §36 text reconciled against
  code (see §70 C.1 resolution in `docs/07_PHASE4_SPRINT_LOG.md`).
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
