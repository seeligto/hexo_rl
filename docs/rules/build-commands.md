# Build commands

## Language and toolchain

| Layer | Language | Notes |
|---|---|---|
| MCTS tree, board logic, win detection | **Rust** | Build with `maturin develop --release -m engine/Cargo.toml`. Concurrency via Rust-native Game-Level Parallelism (Phase 3.5). |
| Replay buffer | **Rust** (ReplayBuffer) | f16-as-u16 ring buffer, 12-fold hex augmentation, zero-copy PyO3 transfer. |
| Neural network, training loop | **Python + PyTorch** | CUDA, FP16, TF32 enabled. InferenceServer bridges Rust worker threads. |
| Temporal tensor assembly | **Python + NumPy** | Stacks 2-plane cluster snapshots + `move_history` into `(8, 19, 19)` tensors. |
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

# Throughput sweep (per-host tuning — knob registry, hardware-agnostic, §126)
make sweep.detect     # write reports/sweeps/detected_host.json (CPU/GPU/VRAM)
make sweep            # full registry sweep (90 s cells, ~70 min)
make sweep.long       # §124/§125 stable methodology (180 s cells, ~2x wall)
make sweep.workers    # n_workers ternary only (90 s cells)
make sweep.fast KNOB=inference_max_wait_ms MAX_MIN=15  # short-cell shakeout
make sweep.dryrun     # validate orchestration with synthetic eval (no GPU)
# Output: reports/sweeps/<host>_<date>/{report.md,cells.csv,config.yaml}
# Recipe for new knobs: docs/sweep_harness.md

# Training
make train            # train with web + terminal dashboard (default)
make train DASHBOARD=0  # train without dashboard
make train BOOTSTRAP=checkpoints/foo.pt VARIANT=vast  # §174 W3 — canonical bootstrap knob
make train.bg         # background training (logs to logs/)
make train.stop       # stop background training
make train.status     # check if running, show recent log
make train.resume     # resume from latest checkpoint
make train.smoke      # 200-step smoke test
make dash.open        # open web dashboard in browser

# Eval
make eval             # legacy: CKPT=latest, N_GAMES=100, SIMS=128
make eval.sealbot EVAL_CHECKPOINT=ckpt.pt EVAL_N=200  # §174 W3 — encoding auto-detect; EVAL_ENCODING= to override
# Eval game diversity: eval_temperature, eval_random_opening_plies, eval_seed_base in configs/eval.yaml (§80)

# Self-play smoke (§174 W3)
make selfplay.smoke SMOKE_CHECKPOINT=ckpt.pt SMOKE_N=20 SMOKE_MODE=both  # mcts | argmax | both

# Corpus & pretrain
make corpus.export    # export raw cache to data/bootstrap_corpus.npz
make pretrain                                          # 15 epochs, encoding from configs/model.yaml
make pretrain PRETRAIN_ENCODING=v6w25 PRETRAIN_EPOCHS=30 PRETRAIN_LR=2e-3  # cold-start a new encoding
make pretrain PRETRAIN_CHECKPOINT=ckpt.pt PRETRAIN_LR=5e-4  # fine-tune (encoding auto-detected from --resume ckpt)

# Transfer weights into a new encoding (§174 W3)
make transfer TRANSFER_SOURCE=checkpoints/bootstrap_model.pt \
              TRANSFER_OUTPUT=checkpoints/bootstrap_model_v6w25_transfer.pt
```

Run `make help` for the complete list of targets.

## Repository layout (current)

```
hexo_rl/
├── CLAUDE.md
├── docs/
│   ├── 00_agent_context.md
│   ├── 01_architecture.md
│   ├── 02_roadmap.md
│   ├── 03_tooling.md
│   ├── 05_community_integration.md
│   ├── 06_OPEN_QUESTIONS.md
│   ├── 07_PHASE4_SPRINT_LOG.md
│   ├── 08_DASHBOARD_SPEC.md
│   ├── 09_VIEWER_SPEC.md
│   ├── handoffs/                    ← PIN/handoff docs (relocated from reports/, D-REPOSTRUCT)
│   ├── sprint_archive/              ← archived sprint reports (relocated from reports/)
│   ├── archive/                     ← archived closed-sprint/investigation docs
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
│   │   │   ├── persist.rs           ← HEXB v6 save/load (v5/v4 hard-rejected)
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
│   │   ├── generate_corpus.py       ← orchestrates all corpus sources
│   │   ├── corpus_analysis.py       ← corpus quality analysis
│   │   ├── injection.py             ← human-seed bot-continuation injection
│   │   ├── opening_classifier.py    ← opening pattern classifier
│   │   ├── human_seeding.py         ← human game seeding for bot games
│   │   └── pretrain.py
│   ├── bots/                        ← bot wrappers (sealbot_bot, our_model_bot, krakenbot_*, nnue_bot, offwindow_*)
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
│   ├── diagnosis/                   ← diagnosis/investigation instruments + lifecycle ledger (README.md)
│   └── ...
├── tests/
│   └── fixtures/                    ← probe/value fixtures (absorbed root fixtures/, D-REPOSTRUCT)
├── vendor/
│   └── bots/                        ← git submodules
│       ├── sealbot/                 ← Ramora0/SealBot
│       └── krakenbot/                ← Ramora0/KrakenBot
└── .gitmodules                      ← submodule tracking (committed)
```
