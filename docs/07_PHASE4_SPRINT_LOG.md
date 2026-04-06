# docs/07_PHASE4_SPRINT_LOG.md
# HeXO Phase 4.0 Sprint Log — 2026-04-01

This document records every architectural decision, implementation, and finding
from the Phase 4.0 sprint. Read this alongside CLAUDE.md at the start of any
new session to avoid re-litigating resolved decisions.

---

## What was built this sprint (in order)

### 1. SE blocks + value head overhaul + auxiliary loss
**Files:** `hexo_rl/model/network.py`, `hexo_rl/training/trainer.py`,
`configs/model.yaml`, `configs/training.yaml`

- Added `SEBlock` (squeeze-and-excitation) to every residual block.
  Reduction ratio 4 (C → C/4 → C). Overhead: ~1% FLOPs.
- Value head replaced: spatial flatten → FC removed.
  New: global avg pool + global max pool → concat(2C) → FC(256) → FC(1) → tanh.
- Value loss replaced: MSE → binary cross-entropy on raw logit.
  `BCE(sigmoid(v_logit), (z+1)/2)` where z ∈ {-1, +1}.
- Added opponent reply auxiliary head. Weight: 0.15 (configs/training.yaml).
  Head active during training only — excluded from inference path.
- `forward()` always returns `(log_policy, value, value_logit)` 3-tuple.
  BCE needs the raw logit; atanh(tanh(x)) was numerically unstable (NaN).

**Why:** SE blocks are empirically validated in KataGo and Leela Chess Zero at
negligible cost. Cross-entropy value loss has sharper gradients than MSE for
binary outcomes. Global pooling value head is board-size-independent.

---

### 2. Growing replay buffer + mixed data streams + playout cap randomisation
**Files:** `engine/src/replay_buffer/mod.rs`, `engine/src/game_runner.rs`,
`hexo_rl/training/trainer.py`, `hexo_rl/selfplay/pool.py`, `scripts/train.py`,
`configs/default.yaml`, `configs/training.yaml`

- `ReplayBuffer.resize()` implemented in Rust: linearizes ring buffer
  in-place via rotate_left, extends backing vecs, updates head/capacity.
- Buffer growth schedule (configs):
  - Step 0:       250,000 samples
  - Step 500K:    500,000 samples
  - Step 1.5M:    1,000,000 samples
- Mixed pretrained + self-play streams with exponential decay:
  `pretrained_weight = max(0.1, 0.8 * exp(-step / 1_000_000))`
- KataGo playout cap randomisation:
  - 25% of games: 50 sims, τ=1 throughout (value targets only, policy masked)
  - 75% of games: 400 sims, τ=1 for first 15 compound moves then τ→0
- Policy loss masked on zero-policy rows (fast games, sum < 1e-6).

**Why:** Small buffer flushes rapidly in early training when data is most stale.
Playout cap decouples data volume from data quality (KataGo finding).

---

### 3. FP16 AMP + torch.compile + policy target pruning
**Files:** `hexo_rl/training/trainer.py`, `configs/training.yaml`, `scripts/train.py`

- `torch.cuda.amp.GradScaler` + `autocast` wrapping forward + loss.
  Config: `fp16: true`. Auto-disabled with warning on CPU.
- `torch.compile(mode="reduce-overhead", fullgraph=False)`.
  Config: `torch_compile: true`. Graceful fallback if compilation fails.
  Checkpoint saving uses `model._orig_mod.state_dict()` to unwrap compiled model.
- Policy target pruning: zero out entries < 2% of max visits, renormalise.
  Config: `policy_prune_frac: 0.02`. Applied before cross-entropy loss.
  Prevents policy head from fitting exploration noise on clearly bad moves.

**Why:** FP16 gives 1.3-1.8× throughput on Ampere (RTX 3070). torch.compile
adds 10-20% on top. Pruning reduces effective policy loss noise in early self-play.

---

### 4. Phase 4.0 evaluation pipeline
**Files:** `hexo_rl/eval/results_db.py`, `hexo_rl/eval/bradley_terry.py`,
`hexo_rl/eval/display.py`, `hexo_rl/eval/eval_pipeline.py`,
`scripts/train.py`, `configs/eval.yaml`

- Bradley-Terry MLE (not incremental Elo). scipy L-BFGS-B with analytical
  gradient + L2 regularisation (1e-6) to prevent divergence on perfect records.
  Ratings scaled to Elo-like units (θ × 400/ln10), anchor = Checkpoint_0 = 0.
- SQLite results store (WAL mode). Schema: players, matches, ratings tables.
  Full BT recomputation from all historical pairwise data after each eval round.
- Gating rule: new checkpoint replaces best if win_rate ≥ 0.55 over 200 games.
  Binomial CI logged alongside every win rate: `p ± 1.96 * sqrt(p(1-p)/n)`.
- Evaluation runs in a separate thread (non-blocking vs self-play workers).
  Model cloned (fresh HexTacToeNet with copied state_dict) to avoid sharing
  torch.compiled training model.
- Opponents: previous best checkpoint, SealBot (fixed external reference),
  random bot (sanity floor).
- Evaluation frequency: every 1,000 training steps (configs/eval.yaml).

**Why:** Self-play Elo inflates without a fixed external reference (SealBot).
Bradley-Terry is path-independent and requires no K-factor tuning.

---

### 5. Corpus generation pipeline
**Files:** `hexo_rl/bootstrap/generate_corpus.py`, `scripts/update_manifest.py`,
`hexo_rl/bootstrap/corpus_analysis.py` (--include-bot-games flag),
`hexo_rl/bootstrap/bots/sealbot_bot.py` (max_depth parameter), Makefile

- `generate_corpus.py` CLI: SealBot self-play with hash-based filenames
  (SHA-256 of move sequence) for deduplication and idempotent re-runs.
- Random opening injection: 3 random moves before SealBot takes over for d4
  (1 for d6+). Reduced dupe rate from 87% → 43% at d4.
- SealBot time cap: 1s per move (not depth limit). d8 was renamed d6 because
  with 1s cap the effective search depth reached is ~6 regardless of setting.
- Corpus targets: 2,000 games at d4, 1,000 games at d6.
- Unified manifest (`scripts/update_manifest.py`): atomic writes via rename.
  Shows human/bot/total breakdown. Safe under concurrent scraper + generator.
- Bot games and reports in .gitignore (generated data stays local).

**Human game constraint:** hexo.did.science API limit is 500 games per pull.
VPS cron job scrapes every 4 hours. As of 2026-04-01: 899 human games.
Strategy: supplement with SealBot self-play corpus until human corpus grows.
Label smoothing ε=0.05 during small-corpus phase (raise to 0.1 at 2k+ human games).

Makefile targets added: `corpus.d4`, `corpus.d6`, `corpus.all`,
`corpus.analysis`, `corpus.manifest`, `bench.baseline`.

---

### 6. Pytest hang fix + sequential action space verification
**Commits:** `21e3c0b`, `9b899e9`

- Pytest hang root cause: `HybridGameSource` running infinite games with
  RandomBot on an infinite board. `get_cluster_views()` grew unboundedly.
  Fix: `max_bot_plies=500` cap. Games hitting cap scored as draws.
- Sequential action space confirmed correct (no code changes needed):
  - 2 MCTS plies per 2-stone compound turn
  - Q-value sign flips only at turn boundaries, not at intermediate ply
  - Dirichlet noise skipped at intermediate ply
  - Plane 16 encodes `moves_remaining == 2`
  - 10 verification tests added.

---

### 7. Benchmark methodology overhaul
**Files:** `scripts/benchmark.py`, `Makefile`, `docs/03_tooling.md`

**Root cause of historical ±50% variance:**
Old benchmark ran 50,000 simulations in a single MCTS tree. At ~5,000 nodes
the tree exceeds L2 cache, dropping throughput by ~15%. The old 218k baseline
was a burst measurement averaging fast small-tree phases with boost clocks
at maximum. Not a valid production baseline.

**Fix:** MCTS benchmark now runs 800 sims/move × 62 iterations with tree
reset between each move — matching `default.yaml mcts.n_simulations = 800`.

**New methodology:**
- n=5 runs, median + IQR reported (not single-point mean)
- 2-10s warm-up per metric before timing begins
- CPU frequency pinning removed (sudo prompt was disruptive; n=5 median
  averaging provides sufficient variance control without it)
- `bench.lite` (n=3), `bench.full` (n=5), `bench.stress` (n=10, pin required)
- `bench.baseline` target: runs bench.full + saves dated JSON

---

## Final Phase 4.0 baseline (2026-04-01, all 10 metrics PASS)

| Metric | Baseline | Target | Status |
|---|---|---|---|
| MCTS sim/s (800 sims/move × 62 iters) | 176,963 | ≥ 160,000 | ✅ |
| NN inference batch=64 pos/s | 10,064 | ≥ 8,500 | ✅ |
| NN latency batch=1 mean ms | 1.50 | ≤ 2 ms | ✅ |
| Buffer push pos/s | 745,523 | ≥ 630,000 | ✅ |
| Buffer sample raw µs/batch | 1,040 | ≤ 1,200 | ✅ |
| Buffer sample augmented µs/batch | 1,001 | ≤ 1,200 | ✅ |
| GPU utilisation % | 100.0 | ≥ 85% | ✅ |
| VRAM usage GB | 0.77 | ≤ 80% | ✅ |
| Worker throughput pos/hr | 1,522,127 | ≥ 1,290,000 | ✅ |
| Batch fill % | 99.4 | ≥ 84% | ✅ |

**Phase 4.5 benchmark gate: CLEAR.**
Methodology: median n=5, 3s warm-up, realistic MCTS workload, CPU unpinned.

---

## Post-baseline changes and re-baseline (2026-04-02)

**Worker throughput regression analysis:**

After the Phase 4.0 baseline was set (1,522,127 pos/hr), two changes caused regression:

1. **SealBot mixed opponent schedule** (`b9b140b`) — Python daemon threads caused 3.3× GIL contention regression (1.52M → 464k). Reverted in `c9f39de`.

2. **Forced-win detection removal** (`fc9eb6f`) — `FormationDetector::has_forced_win()` was bypassing NN inference for near-win positions, making MCTS faster but hurting training quality (network didn't learn to evaluate these positions). Intentionally removed. This adds ~30% more NN calls per game (batch fill improved 99.4% → 99.82%), making each game take longer and reducing pos/hr by ~23%.

**Build optimisations added** (`perf(build)` commit):
- `.cargo/config.toml`: `target-cpu=native` — enables AVX2/FMA/BMI2
- `Cargo.toml` (workspace): `[profile.release]` with `lto=fat`, `codegen-units=1`, `panic=abort`, `strip=symbols`
- MCTS throughput improved 7% (176,963 → 189,656 sim/s)
- Compile time: ~12s (up from ~8s due to LTO)

**Re-baselined (2026-04-02, 16 workers, LTO + native CPU):**

| Metric | New Baseline | Target | Status |
|---|---|---|---|
| MCTS sim/s | 189,656 | ≥ 160,000 | ✅ |
| NN inference batch=64 pos/s | 10,080 | ≥ 8,500 | ✅ |
| NN latency batch=1 mean ms | 1.52 | ≤ 2 ms | ✅ |
| Buffer push pos/s | 905,697 | ≥ 630,000 | ✅ |
| Buffer sample raw µs/batch | 1,000 | ≤ 1,200 | ✅ |
| Buffer sample augmented µs/batch | 949 | ≤ 1,200 | ✅ |
| GPU utilisation % | 100.0 | ≥ 85% | ✅ |
| VRAM usage GB | 0.78 | ≤ 80% | ✅ |
| Worker throughput pos/hr | 1,177,745 | ≥ 1,000,000 | ✅ |
| Batch fill % | 99.82 | ≥ 84% | ✅ |

All 10 metrics pass. Test counts: 63 Rust + 294 Python.

---

## Test counts at sprint close

| Suite | Count | Status |
|---|---|---|
| Python (pytest) | 285 passing | ✅ 0 hangs |
| Rust (cargo test) | 59 passing | ✅ |

---

## Open questions status (see docs/06_OPEN_QUESTIONS.md for full detail)

| # | Question | Status |
|---|---|---|
| Q5 | Supervised→self-play transition schedule | ✅ Resolved — exponential decay 0.8→0.1 over 1M steps |
| Q6 | Sequential vs compound action space | ✅ Resolved — sequential confirmed correct |
| Q2 | Value aggregation: min vs mean vs attention | 🔴 Active — HIGH priority, blocks Phase 4.5 |
| Q3 | Optimal K (number of cluster windows) | 🟡 Active — MEDIUM priority |
| Q8 | First-player advantage in value training | 🟡 Active — MEDIUM priority (corpus shows 51.6% P1 overall, 57.1% in 1000-1200 Elo band) |
| Q1, Q4, Q7 | MCTS convergence rate, augmentation equivariance, Transformer encoder | 🔵 Deferred — Phase 5+ |

---

## Immediate next steps

In priority order:

1. **Q2 ablation: value aggregation strategy** — design and run the
   min vs mean vs attention experiment. This is the single highest-priority
   open question. Needs a baseline checkpoint from the first training run.

2. **First sustained self-play training run** — all infrastructure is
   in place. Run `python scripts/train.py` and monitor for 24-48 hours.
   Watch for: policy entropy collapse, value loss plateau, pretrained_weight
   decay curve, buffer growth transitions.

3. **Corpus completion** — wait for d4 (2,000 games) and d6 (1,000 games)
   to finish, then run `make corpus.analysis` for the final combined report.

4. **NN/bot development research** — a separate context window has analysed
   community discussions and identified potential architectural improvements.
   Derive prompts from that analysis after the first training run establishes
   a stable baseline to compare against.

---

## Key config values to know
```yaml
# configs/default.yaml / training.yaml
mcts:
  n_simulations: 800
  temp_threshold_compound_moves: 15
  fast_prob: 0.25
  fast_sims: 50
  standard_sims: 400

training:
  fp16: true
  torch_compile: true
  policy_prune_frac: 0.02
  aux_opp_reply_weight: 0.15
  decay_steps: 1_000_000   # pretrained_weight decay

buffer_schedule:
  - {step: 0,         capacity: 250_000}
  - {step: 500_000,   capacity: 500_000}
  - {step: 1_500_000, capacity: 1_000_000}

# configs/model.yaml
res_blocks: 12
channels: 128
se_reduction_ratio: 4

# configs/eval.yaml
eval_interval: 1000       # training steps
eval_n_games: 200
promotion_threshold: 0.55
```

---

## Post-sprint cleanup (2026-04-02)

### First training run findings (before cleanup)

Running `python scripts/train.py` revealed 5 issues:
1. `pretrained_weight` stuck at 0.0 — config key missing from training.yaml (only in default.yaml)
2. Checkpoint spam — saving every step instead of every N steps
3. GPU underutilisation during early training — batch fill low when buffer was cold
4. Broken stats logging — `positions_per_hour` always 0 in dashboard
5. `monitoring/configure.py` not called on startup — structlog outputting to stderr

All 5 fixed and verified in follow-up commits.

### Codebase cleanup

537 lines deleted, 11 files removed, 6 modules refactored across:

**Directory renames:**
- `engine/` — PyO3 module name updated to match: `from engine import ...`
- `hexo_rl/`
- `hexo_rl/monitoring/` — `setup.py` → `configure.py`

**Class renames — `Rust` prefix removed from all PyO3 exports:**
- `ReplayBuffer`
- `SelfPlayRunner`
- `InferenceBatcher`

**Files removed:** dead shims, duplicate helpers, unused bootstrap scripts

### Benchmark re-confirmation

All 10 Phase 4.5 gate metrics pass with no regressions after the cleanup.
Test counts: 63 Rust + 301 Python (all passing).

---

## Architecture summary (current state)
```
Input:  (18, 19, 19) tensor
        Planes 0-15: 8 history steps × 2 players (cluster snapshots)
        Planes 16-17: metadata (moves_remaining, turn parity)

Trunk:  12 × ResidualBlock(128ch, SE reduction=4)
        Pre-activation (BN → ReLU → Conv)

Heads:
  Policy:      Conv(128→2, 1×1) → BN → ReLU → FC → log_softmax
  Value:       GlobalAvgPool + GlobalMaxPool → concat(256) → FC(256) → FC(1) → tanh
               Loss: BCE(sigmoid(v_logit), (z+1)/2)
  Opp reply:   Mirror of policy head, training only, weight=0.15

Output: (log_policy, value, value_logit)  ← always 3-tuple
```

---

## Dashboard cleanup & event migration (2026-04-03)

### What was done

Replaced the legacy push-based dashboard system with a structured event emitter
following the spec in `docs/08_DASHBOARD_SPEC.md`.

**New file:** `hexo_rl/monitoring/events.py`
- `emit_event(payload)` — thread-safe fan-out to registered renderers
- `register_renderer(renderer)` — add a renderer at startup
- Never raises; renderer failures caught and printed to stderr
- Zero import side effects (no Flask, no rich)

**train.py changes:**
- Removed: `TrainingDashboard`, `DashboardClient`, `--web-dashboard`,
  `--web-dashboard-url` CLI args, all `web_dash.*` calls
- Added: `emit_event()` calls for `run_start`, `training_step`,
  `iteration_complete`, `eval_complete`, `run_end`
- Added: `run_id` (uuid4) for session tracking
- Added: rolling 60s window for `games_per_hour` computation

**pool.py fixes:**
- `sims_per_sec` bug: was initialised as `None` with `elapsed > 1.0` guard
  preventing first update. Now initialised to `0.0`, updates on every drain.
- Added `avg_game_length` tracking via rolling deque(maxlen=200)
- Emits `game_complete` events with `moves_list` in axial notation

**Files deleted (4,098 lines removed):**
- `dashboard.py` (root level Flask+SocketIO web dashboard)
- `hexo_rl/monitoring/dashboard.py` (TrainingDashboard + Phase40Dashboard)
- `hexo_rl/training/dashboard_utils.py` (DashboardClient HTTP bridge)
- `tests/test_dashboard_client.py`
- `tests/test_dashboard_completeness.py`
- `tests/test_dashboard_phase40.py`
- `tests/test_game_length_median.py`

**Tests:** 14 new tests in `tests/test_dashboard_events.py`.
Total: 356 Python tests passing.

### TODOs for Prompt 2 (renderer implementation)

- ~~`hexo_rl/monitoring/terminal_dashboard.py` — rich renderer (reads events)~~ ✅ Done (§12)
- ~~`hexo_rl/monitoring/web_dashboard.py` — Flask+SocketIO renderer~~ ✅ Done (§12)
- ~~`hexo_rl/monitoring/static/index.html` — single-file SPA~~ ✅ Done (§12)
- ~~Wire `GPUMonitor` to emit `system_stats` events via `emit_event()`~~ ✅ Done (§14)
- ~~Add `monitoring:` config block to `configs/default.yaml`~~ ✅ Done (§12)
- ~~Compute `value_accuracy` in `trainer.py` and include in loss_info~~ ✅ Done (§14)
- ~~Compute `grad_norm` in `trainer.py` (before clip) and include in loss_info~~ ✅ Done (§14)
- ~~Include `lr` from scheduler in loss_info~~ ✅ Done (§14)
- Add per-worker ID to `game_complete` events (requires Rust change)

### Notes

- `moves_list` is populated from Rust `drain_game_results()` which returns
  `Vec<(i32, i32)>` — formatted as `"(q,r)"` strings in Python.
- `worker_id` is hardcoded to 0 for now — Rust SelfPlayRunner doesn't expose
  per-game worker identification yet.
- `push_corpus_preview.py` still references the old dashboard HTTP API for
  signaling reloads. It will silently fail (fire-and-forget). Should be
  updated when the web dashboard renderer is built.

---

### 12. Dashboard renderer implementation (2026-04-03)
**Files:**
- `configs/monitoring.yaml` — new config file for dashboard/alert settings
- `hexo_rl/monitoring/terminal_dashboard.py` — rich Live terminal renderer
- `hexo_rl/monitoring/web_dashboard.py` — Flask+SocketIO web server renderer
- `hexo_rl/monitoring/static/index.html` — single-file browser SPA
- `scripts/train.py` — wired up renderer registration and shutdown
- `tests/test_dashboard_renderers.py` — 12 renderer tests

**Summary:**
Both dashboard renderers implemented per `docs/08_DASHBOARD_SPEC.md`.
Terminal dashboard shows stat panel with loss, throughput, buffer, and system
rows, plus configurable alert line (entropy collapse, grad norm spikes, loss
trends, eval gate failures). Web dashboard at `http://localhost:5001` with
Chart.js loss curves, win rate bars, game length chart, stat cards, buffer/corpus
display, ELO tracking with gate badges, and scrollable event log.

**Test results:** 368 Python tests passing (12 new renderer tests + 14 existing
emitter/schema tests).

**Deferred items:**
- ELO sparkline (needs 3+ eval_complete events — wired in SPA, no data yet)
- `view →` links in event log (greyed out, pending `/viewer` sprint)
- `moves_list` forwarded through SocketIO for future game viewer consumption

### 13. Fix warmup stdout leak (2026-04-03)
**File:** `scripts/train.py`

Warmup and waiting-for-games status messages were using `print()` directly,
bypassing structlog. This leaked `[warmup]` and `[waiting]` lines to stdout
instead of going through the structured logging pipeline. Replaced with
`structlog.get_logger().info(...)` calls.

The web dashboard SPA (`index.html`) was confirmed clean — no REST polling
or `/api/*` fetch calls exist. The 404s reported in server logs were from
a previous version or stale browser cache, not the current SPA code.

**Test results:** 368 Python tests passing, no regressions.

---

### 14. Wire grad_norm, value_accuracy, lr, and GPUMonitor (2026-04-03)
**Files:** `hexo_rl/training/losses.py`, `hexo_rl/training/trainer.py`,
`hexo_rl/monitoring/gpu_monitor.py`, `configs/training.yaml`,
`tests/test_trainer.py`

Closed the remaining TODOs from the dashboard migration:

- **grad_norm:** `fp16_backward_step()` now returns the pre-clip gradient norm
  from `clip_grad_norm_()`. Trainer captures it and includes it in `loss_info`.
  Clipping threshold configurable via `grad_clip` in training.yaml (default 1.0).
- **value_accuracy:** Computed in `_train_on_batch()` as the fraction of samples
  where `sign(v_logit) == sign(outcome)`. Included in `loss_info`.
- **lr:** Read from `optimizer.param_groups[0]["lr"]` after each step, so it
  reflects scheduler state. Included in `loss_info`.
- **GPUMonitor → system_stats:** GPUMonitor polling loop now calls `emit_event()`
  with `system_stats` payload on each poll cycle, matching the existing interval.

**Tests:** 5 new tests in `test_trainer.py` (grad_norm, value_accuracy, lr,
config grad_clip, lr changes with scheduler). Total: 373 Python tests passing.

---

### 15. Game viewer implementation (2026-04-03)
**Files:**
- `engine/src/board/threats.rs` — sliding-window threat detection (3 hex axes)
- `engine/src/board/mod.rs` — module declaration
- `engine/src/mcts/mod.rs` — `get_top_visits()`, `root_value()` exports
- `engine/src/lib.rs` — PyO3 exports for Board.get_threats(), MCTSTree.get_top_visits/root_value
- `hexo_rl/viewer/__init__.py`, `hexo_rl/viewer/engine.py` — ViewerEngine
- `hexo_rl/monitoring/web_dashboard.py` — 4 new routes (/viewer, /viewer/recent, /viewer/game, /viewer/play)
- `hexo_rl/monitoring/static/viewer.html` — single-file SPA
- `hexo_rl/selfplay/pool.py` — moves_detail/value_trace fields (None for now)
- `configs/monitoring.yaml` — capture_game_detail key
- `tests/test_viewer.py` — 10 tests

**Summary:**
Game viewer for replaying self-play games with threat detection overlay. Threats
are EMPTY cells within 6-cell windows where one player has N≥3 stones (levels:
5=critical, 4=forced, 3=warning). Algorithm validated: test_threats_highlight_empty_cells_not_stones
passes — threat cells never overlap with occupied cells.

Viewer features: hex board canvas (pointy-top), threat overlay with color coding,
MCTS visit heatmap (toggle), value sparkline with seek, scrubber with play/pause,
game list with auto-refresh, play-against-model mode (POST /viewer/play).

**Benchmark delta:** 0% on MCTS sim/s, buffer push, GPU util — viewer code is
completely isolated from the training path. get_threats() is viewer-only, never
called from MCTS or training.

**Deferred items:**
- Per-move MCTS detail (moves_detail, value_trace): requires Rust game_runner to
  store top_visits/root_value per move in drain_game_results(). Fields are None
  for now; SPA handles gracefully (hides heat toggle, shows "no value data").
- Per-worker ID in game_complete events (existing TODO).

**Viewer URL:** http://localhost:5001/viewer (during training)

**Test counts:** 71 Rust + 383 Python, all passing.

---

### 16. Remove CPU frequency pinning from benchmark (2026-04-03)
**Files:** `scripts/benchmark.py`, `docs/03_tooling.md`, `docs/07_PHASE4_SPRINT_LOG.md`

Removed `pin_cpu_frequency()`, `restore_cpu_frequency()`, `--no-pin`, `--require-pin`
CLI args, and all `[UNCONTROLLED]` labels from the benchmark harness. The sudo
prompt for `cpupower` was disruptive during development. The n=5 median + IQR
methodology already provides sufficient variance control without frequency pinning.

Change requested by user.

---

### 11. Config loader self-merge bug fix
**File:** `scripts/train.py`

`--config configs/training.yaml` caused `training.yaml` to be merged with itself
because it was already in `_BASE_CONFIGS`. Fixed by resolving paths and deduplicating
before calling `load_config()`.

---

### 17. Community Discord intelligence review (2026-04-03)

**Source:** Bot dev Discord, messages from 2026-03-31 and 2026-04-01.
**Participants referenced:** Vladdy Daddy, Phoenix, Kubuxu, imaseal, Charlie, Rise.

Reviewed community discussions for architectural risks and validation of our
current approach. Seven findings, two requiring immediate action.

**Findings:**

1. **MCTS sim count during self-play — OPEN RISK.**
   Vladdy Daddy runs 1,200 sims/move and sees consistent generation-over-generation
   improvement. Our benchmark config uses 800 sims (evaluation only). Self-play
   worker sim count requires explicit audit — if set below ~400, policy signal will
   be too noisy for reliable learning. See Prompt 1 audit output.

2. **Short-game buffer poisoning — KNOWN RISK, MITIGATED.**
   Phoenix and Vladdy both confirmed that <30-move games filling the buffer is a
   primary failure mode. Kubuxu's nuance: short games are acceptable if game lengths
   trend longer over generations. Our game-length weighted sampling directly addresses
   this. Aggressiveness of the weight function requires audit (see Prompt 2).

3. **Replay buffer size — CURRENT APPROACH VALIDATED.**
   Community consensus: buffer collapse from too-small buffers is worse than any
   cost from a large buffer. Vladdy recommends 100K+ minimum; KataGo started at
   250K. Our 250K → 1M growth schedule is well-grounded. No change.

4. **Augmentation implementation — CURRENT APPROACH VALIDATED.**
   Kubuxu explicitly: do not store 12 copies; pick one symmetry at sample load time.
   This is exactly what our Rust ReplayBuffer scatter tables do. No change needed.

5. **Light SealBot classical mix — CONCEPT VALIDATED, IMPLEMENTATION DEFERRED.**
   Vladdy confirmed that mixing classical bot games into self-play catches obvious
   mistakes the NN otherwise ignores. Recommends non-deterministic bot to prevent
   cheesing. Our SealBot wrapper supports this. Deferred until stable Phase 4.5
   baseline — the prior throughput regression was an implementation issue, not a
   conceptual one.

6. **KL-divergence weighted buffer writes — NEW OPEN QUESTION (Q9).**
   Kubuxu confirmed KataGo uses this (in repo, not paper). Adds significant training
   efficiency by concentrating writes on high-uncertainty positions. Filed in
   `docs/06_OPEN_QUESTIONS.md`. Prerequisite: Phase 4.5 baseline checkpoint.

7. **Torus board encoding — WATCH ITEM.**
   imaseal experimenting with torus/circular padding for full rotational symmetry.
   Incompatible with current attention-anchored windowing. Phoenix interested but
   noted wrap-around artifact risk. Filed as watch item in `docs/06_OPEN_QUESTIONS.md`
   pending imaseal's results.

**Immediate actions required (pre-overnight run):**
- [ ] Confirm self-play worker sim count (Prompt 1 output)
- [ ] Confirm game-length weight ratio is >3× between short and long games (Prompt 2 output)

---

### 18. Benchmark baseline corrected (2026-04-03)

Prior CLAUDE.md baseline was measured against an undersized model due to config
parsing bug in benchmark.py (`config.get('res_blocks')` reading top-level instead
of `model:` block). Fixed in commit 1217555. VRAM measurement also fixed from
pynvml global (`nvmlDeviceGetMemoryInfo.used`) to process-specific
`torch.cuda.max_memory_allocated()`.

New baseline reflects correct 12-block × 128-channel production architecture.
Key changes vs prior (undersized model) baseline:
- NN latency: 1.52 → 2.90 ms (larger model = slower single inference)
- Worker throughput: 1,177,745 → 530,526 pos/hr (cascading effect of slower inference)
- VRAM: 0.78 → 0.10 GB (process-only measurement, not global GPU)
- Buffer sample raw: 1,000 → 1,293 µs (higher system load from correct model)

Targets recalibrated to 85% of new observed medians. See `make bench.full` output
`reports/benchmarks/2026-04-03_19-08.json`.

bench.stress confirmed ±27% worker IQR was warm-up artifact.
bench.full (2026-04-03_19-22): worker 1,094,976 ±1.7% — clean.
Two target updates applied (NN latency ≤3.5ms, buffer raw ≤1,500µs)
— both were stale targets set against undersized model.
game_length_weights units clarified to compound moves in training.yaml.

### Benchmark methodology fix (2026-04-03)

Worker pool warm-up extended from 10s to 30s and restructured to keep a single
pool instance across warm-up + measurement runs. Cold-start bench.full was
showing ~530k pos/hr (IQR ±27%) because each measurement run created a separate
WorkerPool, so every 60-second window included ~20s of no-output cold-start.
bench.stress correctly showed ~1.09M pos/hr (5-min window amortizes startup).

Fix: keep one pool alive for the entire benchmark. Run 30s warm-up to let at
least one full game cycle complete on all workers (~20-25s at 400 sims/move ×
64 compound moves). Then take N measurement windows using counter snapshots
(delta-based) from the already-warm pool.

Post-fix baseline: 735,777 pos/hr (median, n=5), IQR ±11k (1.5%).
Target set to ≥625,000 pos/hr (85% of median).

Note: gap vs bench.stress (1.09M) is expected — 5-min windows better amortize
game-completion burstiness than 60s bench.full windows. Both are correct for
their methodology; bench.full measures conservative steady-state.

---

### 19. Pretrain OOM fix — mmap corpus loading (2026-04-03)
**Files:** `hexo_rl/bootstrap/pretrain.py`, `scripts/train.py`,
`configs/corpus.yaml`, `docs/07_PHASE4_SPRINT_LOG.md`

**Root cause:** `load_corpus()` in `pretrain.py` builds per-game lists and calls
`np.concatenate()` on 906k positions, transiently doubling the ~13 GB corpus in
RAM (~26 GB peak). Combined with system load, this exceeds the 47 GB + 4 GB zram
ceiling and triggers a system freeze during `make pretrain.full`.

**Fix (3 changes):**

1. **NPZ mmap load path in pretrain.py:** Before calling `load_corpus()`, check
   for `data/bootstrap_corpus.npz` (configurable via `corpus_npz_path` in
   `configs/corpus.yaml`). If it exists, load with `np.load(mmap_mode='r')`.
   Memory-mapped arrays are file-backed and page in on demand — no upfront
   13 GB allocation. The `load_corpus()` fallback is retained for when the NPZ
   hasn't been exported yet.

2. **DataLoader num_workers=0 in pretrain.py:** With mmap'd file-backed arrays,
   multi-process workers provide no throughput benefit (GPU is the bottleneck,
   not CPU data prep). `num_workers=0` runs `__getitem__` in the main process,
   eliminating fork() COW page dirtying that would defeat the mmap savings.

3. **NPZ mmap in train.py:** The pretrained corpus load at line ~273 changed
   from `np.load(path)` to `np.load(path, mmap_mode='r')`. Arrays are immediately
   consumed by `ReplayBuffer.push_game()` which copies into Rust-owned memory.
   `del data` added after the push loop to release the mmap handle promptly.

**Memory model:** mmap'd arrays are backed by the OS page cache. Only pages
actually accessed during training are paged in. The Rust ReplayBuffer copies
data out immediately during `push_game()`, so the mmap can be released after.

**Warning for future corpus growth:** if `bootstrap_corpus.npz` is regenerated
and the `load_corpus()` fallback is used (NPZ missing), the double-allocation
risk returns. Always run `make corpus.npz` before `make pretrain.full`.

---

### 20. Validation game count increased to 100 (2026-04-03)
**Files:** `hexo_rl/bootstrap/pretrain.py`, `CLAUDE.md`, `docs/02_roadmap.md`

Changed pretrain validation from 5 to 100 greedy games vs RandomBot. 5 games
provided no statistical confidence — a model winning 5/5 could still have a true
win rate as low as ~48% (binomial CI). 100 games gives meaningful signal:
95% CI width is ±~10% at p=0.5.

User-specified change.

---

### 21. Scraper upgrade with white-box API knowledge (2026-04-03)
**Files:** `hexo_rl/bootstrap/scraper.py`, `scripts/update_manifest.py`,
`hexo_rl/bootstrap/corpus_analysis.py`, `scripts/scrape_daily.sh`,
`configs/corpus.yaml`, `Makefile`, `tests/test_scraper.py`,
`docs/04_bootstrap_strategy.md`, `CLAUDE.md`

**Context:** Analysis of the WolverinDEV/infhex-tic-tac-toe repository
confirmed it is the source code for hexo.did.science — the site we already
scrape. This is not a new data source; the value is white-box API knowledge
that upgrades our existing scraper from black-box to informed operation.

**What was changed (scraper upgrade, not new integration):**

1. **Scraper enhancements:**
   - `--min-elo` flag: post-fetch filter skipping games where either player's
     Elo is below threshold. API does not support server-side Elo filtering.
   - `--top-players-only` + `--top-n`: calls `/api/leaderboard` to get top-N
     player profileIds, then filters to games with at least one top player.
   - `--req-delay` flag: enforces 1 req/s minimum between requests (courtesy
     to the community-run single-instance server).
   - Per-game Elo passthrough: `player_black_elo` and `player_white_elo`
     fields stored in saved game JSON (from `DatabaseGamePlayer.elo`).
   - Fixed `baseTimestamp` param name (was incorrectly `before`).
   - Updated `UNAUTHENTICATED_GAME_LIMIT` from 480 to 500 (confirmed from
     source: `apiQueryService.ts:68-70`).
   - All new flags configurable via `configs/corpus.yaml` `scraper:` section.

2. **Manifest Elo band breakdown:**
   Manifest now includes `elo_bands` section bucketing human games by
   `max(player_black_elo, player_white_elo)`:
   `sub_1000`, `1000_1200`, `1200_1400`, `1400_plus`, `unrated`.
   Existing games without Elo fields classified as `unrated` (no backfill).

3. **Corpus analysis Elo stratification:**
   `--include-human-games` flag adds per-band breakdown: game count,
   median compound move length, top 5 openings (first 3 compound moves).

4. **scrape_daily.sh dual-pass:**
   Standard incremental pull + `--top-players-only --top-n 20` pass.
   Log entries distinguish `[standard]` vs `[top-player]` passes.

5. **Makefile targets:** `corpus.human.top`, `corpus.human.rated`.

6. **22 new tests** covering Elo filter, top-player mode, Elo passthrough,
   and manifest band bucketing.

**White-box API findings that drove each decision:**

- Elo fields already in `DatabaseGamePlayer` → passthrough, not computation.
- `/api/leaderboard` exists and is unauthenticated → top-player mode.
- No rate-limit middleware in source → but single-instance server, so 1 req/s
  politeness policy.
- `baseTimestamp` is the correct param name (not `before`) → fixed.
- 500 is the exact public cap (page * pageSize >= 500 → 401) → updated limit.
- Coordinates are direct axial (q=x, r=y) → no translation needed (confirmed).

**Elo band manifest schema:**
```json
{
  "elo_bands": {
    "sub_1000": 42,
    "1000_1200": 156,
    "1200_1400": 89,
    "1400_plus": 23,
    "unrated": 589
  }
}
```

**Open questions:**
- Whether to backfill Elo on existing pre-upgrade game files. Currently
  classified as `unrated`. Would require re-fetching from API (games within
  the 500-game window) or accepting partial coverage. Deferred.

---

### 22. Fix flaky test_train_step_returns_grad_norm (2026-04-04)
**Files:** `tests/test_trainer.py`

**Root cause:** The test asserted `np.isfinite(grad_norm)`, but with FP16
training (GradScaler enabled), random test data can produce gradients that
overflow after `scaler.unscale_()`. In that case `clip_grad_norm_()` correctly
returns `inf` as the pre-clip norm, and GradScaler skips the optimizer step.
This is expected FP16 behaviour — not a bug in the training code.

**Fix:** Changed assertion from `np.isfinite()` to `not np.isnan()` and
`>= 0.0`, matching the sibling test `test_grad_norm_uses_config_grad_clip`
which already handled this correctly.

**Impact on actual training:** None. The `fp16_backward_step()` implementation
has correct ordering (unscale → clip → step → update) since §14. Grad clipping
worked correctly during the 4,940-step self-play run. Only the test assertion
was too strict.

---

### 23. Diagnostic findings — GPU utilisation and colony detection (2026-04-04)

GPU burst-idle pattern is structural (CPU MCTS + GPU inference interleaving),
not a bug. 12ms timeout fallback is working. 45% median GPU util is expected
at 12 workers with 400 sims/move. Checkpoint_500 100% colony win rate against
pretrained model is a colony detection miscategorisation artefact, not evidence
of colony strategy. Honest benchmark: 10% win rate vs SealBot at step 500
(5/50 games).

---

### 24. Entropy regularization + draw penalty (2026-04-04)

Source: `docs/10_COMMUNITY_BOT_ANALYSIS.md` §5.1C and §5.1D (KrakenBot practice).

**Entropy regularization (§5.1C)**
**Files:** `hexo_rl/training/losses.py`, `hexo_rl/training/trainer.py`, `configs/training.yaml`

- Added entropy bonus term to total loss: `L_total = L_policy + L_value + w_aux·L_aux - w_entropy·H(π)`
- Entropy `H(π) = -Σ π log π` computed inside `autocast` block (before backward) using `log_policy` from the network forward pass.
- Weight `entropy_reg_weight: 0.01` in `configs/training.yaml`.
- `compute_total_loss()` in `losses.py` gains two new optional parameters: `entropy_bonus` and `entropy_weight`.
- `policy_entropy` was already logged in `loss_info` (and fed to the entropy collapse alert at < 1.0). Now it reflects real data.
- Expected range at a uniform 362-action policy: ~5.9 nats. In practice ~3-6 nats is healthy; < 1.0 signals collapse.

**Draw penalty (§5.1D)**
**Files:** `engine/src/game_runner.rs`, `hexo_rl/selfplay/pool.py`, `configs/training.yaml`, `tests/test_worker_pool.py`

- Draw outcome value target changed from `+0.01` → `-0.1`.
- Was hardcoded in `game_runner.rs`; now configurable via `draw_reward` field on `SelfPlayRunner` (PyO3 parameter, default `-0.1`).
- `pool.py` reads `training.draw_reward` from merged config and passes it through.
- Rationale: with ~51.6% P1 win rate in the corpus, draws are suboptimal for the stronger player. Negative draw reward pushes the network to press for wins rather than settle.

**Why negative is correct:** The value head learns `z ∈ {-1, +1}` with draws at `-0.1`. This makes draws strictly worse than wins but also worse than a neutral 0.0, which actively discourages draw-seeking. It mirrors the KrakenBot tuning that produced stronger play in sustained self-play.

---

### 25. Re-enable torch.compile — split train/inf model instances (2026-04-04)
**Files:** `scripts/train.py`, `configs/training.yaml`, `hexo_rl/training/trainer.py`,
`tests/test_trainer.py`, `tests/test_board.py`

**Problem:** torch.compile was disabled (`torch_compile: false` in training.yaml) because
the same compiled model instance was shared between the Trainer (main thread) and
InferenceServer (its own thread). CUDA graphs captured by `reduce-overhead` mode are
bound to a specific CUDA stream and thread — sharing them across threads caused silent
corruption or crashes.

**Fix — split model instances:**

- `train_model` = `trainer.model`. Compiled in `Trainer.__init__` as before.
  Exclusively owned by the main process thread (gradient updates).
- `inf_model` = separate `HexTacToeNet` instance created in `train.py` after
  trainer setup. Compiled independently. Passed to `WorkerPool` (and therefore
  `InferenceServer`) — never touches the main thread after init.
- Both compiled with `mode="reduce-overhead", fullgraph=False`. Graceful fallback
  on exception (log warning, continue uncompiled).

**Weight synchronisation:**

```python
def _sync_weights_to_inf():
    train_base = getattr(trainer.model, "_orig_mod", trainer.model)
    inf_base   = getattr(inf_model,     "_orig_mod", inf_model)
    inf_base.load_state_dict(train_base.state_dict())
```

Sync points:
1. **Startup** — `inf_model.load_state_dict(train_base.state_dict())` immediately after
   creation (and after `Trainer.load_checkpoint` for resume runs).
2. **Checkpoint saves** — every `checkpoint_interval` steps (detected by
   `train_step % _ckpt_interval == 0` after `trainer.train_step()` returns).
3. **Model promotion** — when eval pipeline promotes current model to new best.

`load_state_dict` on `_orig_mod` updates parameters in-place. CUDA graph replay in
`reduce-overhead` mode uses the memory addresses recorded at capture time, so in-place
weight updates are picked up on the next graph replay without recompilation.

**Checkpoint save / load / export:** unchanged — `save_full_checkpoint` and
`save_inference_weights` already use `get_base_model()` (`_orig_mod` unwrap), so they
correctly save from `train_model` only.

**Other fixes in this commit:**

- `hexo_rl/training/trainer.py`: cast float16 buffer states to float32 when `fp16=False`
  to avoid type mismatch between buffer data and model weights in test contexts.
- `tests/test_trainer.py` `test_scheduler_steps_each_train_step`: add `fp16: False` so
  FP16 gradient overflow cannot non-deterministically skip `scheduler.step()`.
- `tests/test_board.py` `test_legal_move_count_decrements`: update expected counts (24→216,
  28→232) to match the new hex-ball-radius-8 legal move rule (correct official rule).

**bench.full delta (2026-04-04, torch.compile ENABLED vs prior disabled baseline):**

| Metric | Before (disabled) | After (split+compile) | Delta | Pass? |
|---|---|---|---|---|
| MCTS sim/s | 164,946† | 52,959 | −68% (board rule change†) | ✅ ≥45,000 |
| NN inference b=64 | 10,201 | 11,038 | +8.2% | ✅ |
| NN latency b=1 | 2.82 ms | 2.93 ms | +3.9% (within IQR) | ✅ |
| Buffer push | 755,880 | 802,585 | +6.2% | ✅ |
| Buffer sample raw | 1,237.6 µs | 1,253.4 µs | +1.3% | ✅ |
| Buffer sample aug | 1,177 µs | 1,144.1 µs | −2.8% | ✅ |
| GPU util | 100.0% | 100.0% | 0% | ✅ |
| VRAM | 0.10 GB | 0.10 GB | 0% | ✅ |
| Worker throughput | 735,777 | 758,226 | +3.0% | ✅ |
| Batch fill % | 95.2% | 98.0% | +2.8pp | ✅ |

†MCTS regression is from the concurrent `board/moves.rs` change switching legal moves
from bbox+2 margin (old, incorrect) to hex-ball radius 8 (official rule). That change
expands branching factor ~9× → proportionally reduces sim/s. Unrelated to torch.compile.
MCTS target rebaselined to ≥45,000 (85% of new 52,959 median). All 10 targets PASS.

**Test counts:** 71 Rust + 573 Python, all passing.

---

### 26. Legal move margin audit — corrected to hex-ball radius 8 (2026-04-04)
**Files:** `engine/src/board/moves.rs`, `engine/src/board/mod.rs`

**Audit result:** The legal move margin was **incorrect** (bbox+2, not 8). Fixed.

**Official rule:** hexo.did.science states "a new hex can be placed at most 8 cells apart
from any other hex." KrakenBot community bot analysis (§5.5) confirmed `candidate_distance=8`.

**What was wrong:** `legal_moves_set()` expanded each cluster's axis-aligned bounding box
by 2 in each direction. This produced a rectangle approximately 5×5 around a single stone
— nowhere near the correct radius of 8. Any empty cell within hex_distance ≤ 8 of any
existing stone must be legal; the bbox+2 approach only covered cells within ~2 steps.

**Fix:** Replaced the bbox+2 rectangle expansion with exact per-stone hex ball iteration
(radius `LEGAL_MOVE_RADIUS = 8`). For each stone at (sq, sr), all (sq+dq, sr+dr) satisfying
`|dq| ≤ 8, |dr| ≤ 8, |dq+dr| ≤ 8` (the standard axial hex ball) are added to the legal
set if unoccupied. This is 217 cells per stone, deduplicated via FxHashSet.

**Cluster distance vs legal move margin:**
- `get_clusters()` uses `hex_distance ≤ 8` to group stones into colonies for NN windowing.
- `legal_moves_set()` uses `hex_distance ≤ 8` (LEGAL_MOVE_RADIUS) to find legal moves.
- Same numeric threshold, but independent purposes: clustering determines which stones share
  a NN window; legal moves determine where a stone can be placed. Both happen to be 8 because
  that is the official game rule, and colony grouping by the same distance is a natural choice
  for windowing stones that can legally interact.

---

### 27. KataGo-style dynamic FPU (2026-04-04)
**Files:** `engine/src/mcts/mod.rs`, `engine/src/mcts/selection.rs`,
`engine/src/game_runner.rs`, `engine/src/lib.rs`,
`hexo_rl/selfplay/pool.py`, `configs/selfplay.yaml`

**Context:** KrakenBot community bot analysis (docs/10_COMMUNITY_BOT_ANALYSIS.md §5.1A)
identified KataGo-style dynamic FPU as a key component of KrakenBot's MCTS.

**What classical FPU does:** Unvisited children get a fixed Q estimate (0.0 in our
prior code). This means MCTS treats all unexplored moves as neutral, and the exploration
bonus (U term) drives initial branching entirely from policy priors.

**What dynamic FPU does:**

```
explored_mass = sum of prior(a) for all visited children a
fpu_reduction = fpu_base * sqrt(explored_mass)
fpu_value     = parent_q - fpu_reduction
```

As more children are visited, `explored_mass` grows toward 1.0 and `fpu_value` becomes
more pessimistic relative to `parent_q`. This shifts exploration toward refining
already-known-good branches rather than sampling every legal move uniformly. When
`parent_q` is positive (winning position) the FPU value is below the parent's Q but
still positive — less optimistic than a fresh branch, but not zero.

**Implementation:**
- `MCTSTree` gains `fpu_reduction: f32` field. New constructor `new_full()` sets it.
  `new()` and `new_with_vl()` default to `0.0` (no-op, backward-compatible).
- `puct_score()` gains a `fpu_value: f32` parameter. Used only when
  `child.n_visits == 0 && child.virtual_loss_count == 0`.
- `select_one_leaf()` computes `explored_mass` and `fpu_value` once per node visit,
  before the `max_by` over children. Overhead: one extra pass over N children, O(N).
- `SelfPlayRunner` and `PyMCTSTree` both accept `fpu_reduction` and thread it through.
- `pool.py` reads `mcts.fpu_reduction` from config (default 0.25 matches KrakenBot).

**Config:** `configs/selfplay.yaml` `mcts.fpu_reduction = 0.25`.

**Benchmark:** Deferred — will rebaseline after all Wave 2 prompts land.

**Test coverage:** New unit test `test_dynamic_fpu_reduces_unvisited_q` verifies that
the FPU value raised for an unvisited child when `parent_q > 0` compared to the legacy
`Q=0` baseline.

---

### 28. Quiescence value override for forced wins at MCTS leaves (2026-04-04)
**Files:** `engine/src/board/moves.rs`, `engine/src/mcts/backup.rs`,
`engine/src/mcts/mod.rs`, `engine/src/game_runner.rs`, `engine/src/lib.rs`,
`hexo_rl/selfplay/pool.py`, `configs/selfplay.yaml`

**Context:** KrakenBot analysis (docs/10_COMMUNITY_BOT_ANALYSIS.md §5.1B) identified
a quiescence check used to override the NN value when the position is provably decided.

**The game-specific theorem:**
Each turn places exactly 2 stones.  Therefore the opponent can block at most 2 winning
cells per response.  If the current player has ≥3 empty cells where placing a stone
would complete a 6-in-a-row, the opponent cannot block all of them — the win is
mathematically forced.  Similarly, if the opponent has ≥3 winning cells, the current
player cannot prevent the loss on the opponent's next turn.

**Critical distinction from the removed forced-win short-circuit:**
The earlier short-circuit (removed post-baseline, sprint §post-baseline) fired at
MCTS expansion and marked positions as terminal, preventing the NN from evaluating
them at all.  This new quiescence check is a VALUE OVERRIDE at leaf evaluation:

- The NN still receives the position and produces (policy, value).
- The POLICY is used unchanged for MCTS expansion → network learns fork patterns.
- Only the VALUE is overridden with the proven result (±1.0 or a blend for 2 threats).

This means the network continues to learn about forced-win positions and the moves
that create or prevent them — which is essential for learning to build multi-threats.

**What was implemented:**

`Board::count_winning_moves(player: Player) -> u32` (engine/src/board/moves.rs):
- Iterates the legal-move set; for each empty cell checks whether placing `player`'s
  stone there would create a 6-in-a-row along any hex axis.
- Uses `count_direction()` without placing a stone (correct: counts runs through
  the empty cell without mutating board state).
- O(legal_moves) — acceptable for leaf-only evaluation.
- Exposed via PyO3: `Board.count_winning_moves(player: int) -> int`.

`MCTSTree::apply_quiescence()` (engine/src/mcts/backup.rs):
- `quiescence_enabled: bool` and `quiescence_blend_2: f32` fields on `MCTSTree`.
- If `current_wins >= 3` → value = +1.0 (forced win).
- If `opponent_wins >= 3` → value = -1.0 (forced loss on opponent's next turn).
- If `current_wins == 2` → value = min(value + blend_2, 1.0) (strong, unproven).
- If `opponent_wins == 2` → value = max(value - blend_2, -1.0).
- Applied in `expand_and_backup_single` for both fresh expansions and TT-hit paths.
- Terminal nodes (check_win / no legal moves) bypass quiescence — exact value used.

**Config:** `configs/selfplay.yaml`:
```yaml
mcts:
  quiescence_enabled: true
  quiescence_blend_2: 0.3
```

**Python wiring:** `pool.py` reads `mcts.quiescence_enabled` and `mcts.quiescence_blend_2`
from config and passes them to `SelfPlayRunner`.  `PyMCTSTree` also accepts them as
constructor arguments with getters/setters.

**Benchmark:** Deferred — will rebaseline after all Wave 2 prompts land.
Note: `count_winning_moves` adds O(legal_moves) cost at every leaf evaluation.
If MCTS sim/s drops below the 45,000 target during rebaselining, optimise by
gating the full count behind a cheaper pre-check (e.g. any level-5 threat from
`get_threats`).

**Test coverage (82 Rust tests, all pass):**
- `board::moves::tests::test_count_winning_moves_empty_board` — returns 0 on empty board.
- `board::moves::tests::test_count_winning_moves_five_in_row` — 5-in-a-row → 2 winning cells.
- `board::moves::tests::test_count_winning_moves_five_blocked_one_end` — blocked end → 1 cell.
- `board::moves::tests::test_count_winning_moves_zero_when_early_game` — early game → 0.
- `board::moves::tests::test_count_winning_moves_three_independent_winning_cells` — ≥3 threats → ≥3.
- `mcts::tests::test_quiescence_overrides_value_for_3_winning_moves` — override to +1.0.
- `mcts::tests::test_quiescence_overrides_value_for_3_opponent_winning_moves` — override to -1.0.
- `mcts::tests::test_quiescence_blend_for_2_winning_moves` — blend applied correctly.
- `mcts::tests::test_quiescence_disabled_does_not_change_value` — disabled → passthrough.
- `mcts::tests::test_quiescence_no_override_in_early_game` — no threats → no override.

### 27. Add per-worker ID to game_complete events (2026-04-04)

- Marked `TODO: add per-worker ID when Rust exposes it` as resolved by exposing `worker_id` from Rust `drain_game_results` and passing it to Python events.

### 30. Quiescence gate — threat pre-check + benchmark rebaseline (2026-04-04)
**Files:** `engine/src/board/moves.rs`, `engine/src/mcts/backup.rs`,
`scripts/benchmark.py`, `CLAUDE.md`

**Problem:** `apply_quiescence` called `count_winning_moves` on every MCTS leaf.
`count_winning_moves` is O(legal_moves) — with hex-ball-8 rules the legal set is
hundreds of cells even in early game, causing measurable overhead.

**Root-cause analysis (benchmark):**
The §25 bench baseline (52,959 sim/s) predates §27 (FPU, fpu_reduction=0.25 default).
§27 alone reduced MCTS sim/s to ~31,000 due to behavioral tree-shape change:
with cpu-only 0-value benchmark all NN values = 0.0, so FPU makes unvisited children
look worse than visited ones → deeper/narrower tree → more apply_move calls per sim.
§28 (quiescence, no gate) added a further ~21% overhead → ~24,716 sim/s.

**Fix — two-tier gate in `apply_quiescence`:**

Tier 1 — ply gate (free, one comparison):
  P1 first reaches 5 stones at ply=8. With `board.ply < 8` no player can have
  5 consecutive stones → `count_winning_moves` returns 0 → skip entirely.
  Eliminates quiescence cost for early-game leaves and the cpu-only benchmark
  (which starts from empty board; leaves at ply 1-3).

Tier 2 — long-run pre-check (O(stones × 3 × avg_run)):
  `Board::has_player_long_run(player, 5)` checks if `player` has ≥5 consecutive
  stones without calling `count_winning_moves`. If neither player has such a run,
  skip entirely. Much cheaper than O(legal_moves) because stone count << legal_move
  count with hex-ball-8 rules.

**New `Board::has_player_long_run`** (board/moves.rs):
  Iterates all stones for `player`, checks run length via `count_direction`.
  Returns early once a run of ≥ min_len is found.

**Measurement:**
- fpu=0.0, quiescence=OFF (§25 baseline equivalent): 53,107 sim/s
- fpu=0.0, quiescence=ON (gated): 52,143 sim/s   ← gate adds only 1.8% overhead ✅
- fpu=0.25, quiescence=OFF (§27 FPU baseline): ~31,000 sim/s
- fpu=0.25, quiescence=ON (gated): ~31,000 sim/s  ← quiescence overhead fully recovered ✅

**bench.full delta (2026-04-04, after quiescence gate, fpu_reduction=0.25):**

| Metric | Before gate (§28) | After gate (§30) | Target | Pass? |
|---|---|---|---|---|
| MCTS sim/s | ~24,716 | 30,963 | ≥ 26,000 | ✅ |
| NN inference b=64 | 11,038 | 10,993 | ≥ 8,500 | ✅ |
| NN latency b=1 | 2.93 ms | 2.83 ms | ≤ 3.5 ms | ✅ |
| Buffer push | 802,585 | 839,289 | ≥ 640,000 | ✅ |
| Buffer sample raw | 1,253.4 µs | 1,270.9 µs | ≤ 1,500 µs | ✅ |
| Buffer sample aug | 1,144.1 µs | 1,147.5 µs | ≤ 1,400 µs | ✅ |
| GPU util | 100.0% | 100.0% | ≥ 85% | ✅ |
| VRAM | 0.10 GB | 0.10 GB | ≤ 80% | ✅ |
| Worker throughput | 758,226 | 758,748 | ≥ 625,000 | ✅ |
| Batch fill % | 98.0% | 100.0% | ≥ 80% | ✅ |

MCTS target rebaselined to ≥26,000 (85% of 30,963 with FPU enabled).
All 10 targets PASS.

**Note on FPU regression:** §27 (fpu_reduction=0.25 default) is the dominant
source of MCTS benchmark regression. Investigation: with all NN values = 0.0
(cpu-only benchmark), FPU makes unvisited children appear worse than visited ones
(q = parent_q − 0.25 × √explored_mass < 0 when parent_q=0), shifting selection
toward revisiting explored branches rather than breadth-first expansion, increasing
average simulation depth and apply_move_tracked calls per sim. This is a benchmark
artifact: in real self-play NN values are non-zero and FPU improves selection quality.

**Test counts (post-gate):** 86 Rust + (unchanged) Python, all passing.

---

### 29. Emit training events from pretrain for continuous loss history (2026-04-04)

- `hexo_rl/bootstrap/pretrain.py`: Added `emit_event` calls after each training step.
- Pretrain steps now use negative indices counting up toward 0 (e.g. step=-5000...step=0).
- `phase: "pretrain"` field added to `training_step` event schema.
- `hexo_rl/monitoring/terminal_dashboard.py`: Renders `[PRETRAIN]` marker in header if phase is pretrain.
- `hexo_rl/monitoring/static/index.html`: Loss chart now shades pretrain region and adds a vertical dashed line at step 0 to separate pretrain from RL.
- `scripts/train.py`: Replays up to 500 events from `logs/pretrain.jsonl` when loading a pretrained checkpoint to populate the dashboard history buffer on first load.
- Verified via `tests/test_pretrain_events.py`.

---

### 30. torch.compile mode=default + game length cap 200 + T_max fix (2026-04-04)

**Bug: torch.compile CUDA graph TLS crash (BLOCKING)**
- `hexo_rl/bootstrap/pretrain.py`: Changed `compile_model(model, mode="reduce-overhead")` →
  `compile_model(model, mode="default")`. Python 3.14 incompatibility with CUDA graph
  thread-local storage (TLS) setup causes `AssertionError: assert torch._C._is_key_in_tls(attr_name)`
  on the first forward pass when `reduce-overhead` is used.
- `hexo_rl/training/trainer.py` and `scripts/train.py` (inf_model) were already using
  `mode="default"` — no change needed there.
- `mode="default"` still applies operator fusion and kernel optimisation but does NOT
  use CUDA graphs, avoiding the TLS issue entirely.

**Bug: game length cap too low (mass draws)**
- Added `max_game_moves: 200` to `configs/selfplay.yaml` under `selfplay:`.
  Previous default was 128 compound moves, causing most games to end as draws
  and filling the replay buffer with uninformative draw-reward positions.
- Updated `hexo_rl/selfplay/pool.py` to read `max_game_moves` key (falls back to
  `max_moves_per_game` for backwards compatibility).
- 200 compound moves (400 plies) matches human game length distribution with ample
  margin for exploratory self-play. Games at the cap are genuine draws.

**Bug: CosineAnnealingLR T_max mismatch**
- `hexo_rl/bootstrap/pretrain.py`: `total_pretrain_steps` is now computed before
  `BootstrapTrainer` is created and injected into config as `pretrain_total_steps`.
  Previously the trainer always used the fallback value of 50,000 regardless of the
  actual run length, causing a small LR rebound at the end of longer pretrain runs.

**Commits:**
- `fix(training): switch torch.compile to mode=default (CUDA graph TLS fix)`
- `fix(selfplay): increase game length cap to 200 compound moves`
- `fix(pretrain): compute CosineAnnealingLR T_max dynamically`

---

### 31. Cold-start hang fix + buffer visibility + SIGINT force-exit (2026-04-04)

**Root cause of cold-start hang (training stuck at step 0)**

Two contributing factors:
1. **Large corpus blocking startup.** If `data/bootstrap_corpus.npz` exists and is large
   (e.g. 2M positions ≈ 28 GB estimated), `pretrained_buffer.push_game()` copies the
   full dataset into the Rust buffer before `pool.start()` is ever called. The process
   sits at 27+ GB RAM with no dashboard activity for several minutes. The mmap'd
   `np.load(mmap_mode='r')` only defers the OS page-ins; PyO3 `push_game` triggers a
   full copy. Existing `min_buffer_size: 256` in `configs/training.yaml` was already
   correct — the threshold was never the bug.

2. **No dashboard visibility during warmup.** The warmup loop emitted `structlog` JSON
   only (file), so the terminal/web dashboard showed "—" for the Buffer field throughout
   warmup. Users couldn't distinguish "waiting for 256 positions" from "hung."

**Fixes applied**

- `scripts/train.py` — corpus loading path: added `log.info("loading_corpus_npz")` before
  `np.load()` and a `log.warning("corpus_prefill_high_ram")` when estimated RAM > 2 GB,
  so the user knows why startup is slow. Changed `log.error` → `log.warning` for missing
  NPZ (not an error — buffer fills from self-play, training proceeds normally).

- `scripts/train.py` — warmup loop: now emits a `system_stats` event (with `buffer_size`
  and `buffer_capacity`) every 5 s while waiting for the buffer threshold. Reduced
  warmup sleep from 1.0 s → 0.5 s so Ctrl+C is handled faster.

- `hexo_rl/monitoring/terminal_dashboard.py` — `system_stats` handler now also merges
  `buffer_size` and `buffer_capacity`, so the Buffer field populates during cold-start
  warmup before any `iteration_complete` event fires.

- `scripts/train.py` — SIGINT/SIGTERM handler: added double-press force-exit. First
  Ctrl+C sets `_running[0] = False` and logs a message. Second Ctrl+C calls
  `sys.exit(1)` immediately, bypassing `pool.stop()` if it hangs.

**Commits:**
- `fix(training): make corpus prefill optional, lower buffer threshold`
- `feat(dashboard): show buffer fill progress during cold-start`
- `fix(training): handle SIGINT/SIGTERM for clean shutdown`

---

### 33. Recency-weighted replay sampling + value uncertainty head (2026-04-04)

**Files:** `hexo_rl/training/recency_buffer.py` (new), `hexo_rl/selfplay/pool.py`,
`hexo_rl/training/trainer.py`, `hexo_rl/training/losses.py`,
`hexo_rl/model/network.py`, `scripts/train.py`, `configs/training.yaml`

#### Recency-weighted replay sampling
The Rust `ReplayBuffer.sample_batch()` has no recency API. A lightweight Python-side
`RecentBuffer` ring (NumPy, thread-safe) mirrors the newest ~50% of buffer capacity.
The pool stats thread pushes each position into both buffers simultaneously as data
arrives. Training batches are split 75% recent / 25% full-Rust-buffer (augmented).
Config: `recency_weight: 0.75` in `configs/training.yaml`. Both the single-buffer
and pretrained+self-play mixed paths are updated. Falls back to full-buffer when
`recency_weight=0` or the recent buffer is empty.

#### Value uncertainty head (diagnostic only)
Fifth output head on `HexTacToeNet`: trunk → `AdaptiveAvgPool2d(1)` → `Linear` →
`Softplus` → σ² > 0. Activated only via `forward(uncertainty=True)` during training.
Gradient is stopped before reaching the value head (`value.detach()`) so σ² does not
influence value learning. The existing 3-tuple contract `(log_policy, value, v_logit)`
is preserved for all InferenceServer / evaluator / MCTS callers — verified by grep.
Loss: Gaussian NLL `0.5 * (log σ² + (z - v_detached)² / σ²)`, weight 0.05.
`avg_sigma` (mean √σ² over batch) appears in `loss_info` and `training_step` events
as a passive diagnostic. Dashboard alerts do not trigger on sigma values.
Config: `uncertainty_weight: 0.05` in `configs/training.yaml`.

**Key follow-up fix:** When resuming from a checkpoint that predates these features,
`config_overrides` in `train.py` now explicitly forwards `uncertainty_weight` and
`recency_weight` from the merged YAML so both features activate on resume.
`Trainer.load_checkpoint` also catches `ValueError` on optimizer param-group
mismatch (new head = new params) and restarts the optimizer from scratch with a
warning instead of crashing.

**Commits:**
- `feat(training): recency-weighted replay sampling (75/25 recent/uniform)`
- `feat(model): value uncertainty head (diagnostic σ², weight 0.05)`
- `fix(training): propagate uncertainty/recency weights + graceful optimizer mismatch`

---

### 34. Checkpoint hygiene and RL starting point (2026-04-04)

**Decision: always use `checkpoints/bootstrap_model.pt` as the RL entry point.**

Do not use full pretrain checkpoints (e.g. `pretrain_00053130.pt`) as the RL start.
Here is why, and what each checkpoint contains:

| File | step | scheduler last_epoch | Type | Use |
|---|---|---|---|---|
| `pretrain/pretrain_00000000.pt` | 0 | 53505 | full checkpoint | **Do not use for RL** — step=0 but scheduler is stale (53K epochs exhausted). LR would be near-zero at RL start. |
| `pretrain/pretrain_00053130.pt` | 53130 | 53130 | full checkpoint | Human-only pretrain result (bot_games.bak was active). Useful as reference only. |
| `checkpoints/bootstrap_model.pt` | N/A | N/A | **weights-only** | **Use this for RL.** No optimizer/scheduler state. `Trainer` initialises both fresh from config → step=0, LR at initial value. |

**Why `pretrain_00000000.pt` is broken as RL start:**
The pretrain script resets `self.step = 0` when saving the final "RL-ready" checkpoint
(so RL starts at step 0), but does NOT reset the cosine scheduler. The scheduler's
`last_epoch` carries over from the full pretrain run (~53K steps), meaning the
LR cosine cycle is already exhausted. RL would start with LR ≈ `eta_min` (1e-5)
and learn almost nothing. This is a known bug in the pretrain save logic — do not fix
it without also resetting `scheduler.last_epoch = 0`.

**Why `bootstrap_model.pt` is correct:**
It is a weights-only file. `Trainer.load_checkpoint` detects this (`is_full_ckpt=False`),
sets `trainer.step = 0`, and constructs a fresh scheduler from config. No pretrain
optimizer or scheduler state is inherited.

**`make train` already does the right thing:** `CHECKPOINT_BOOTSTRAP` in the Makefile
points to `checkpoints/bootstrap_model.pt`. Just run `make train`.

---

### 32. Disable torch.compile — Python 3.14 incompatibility cascade (2026-04-04)

**Background:** torch.compile has caused three consecutive blocking issues:

- **§25** — `mode="reduce-overhead"`: CUDA graph TLS crash on Python 3.14. Fixed by
  switching to `mode="default"`.
- **§30** — `mode="default"`: Still crashes on the first forward pass in some
  configurations.
- **Current** — `mode="default"`: Causes a 27 GB RAM spike during Triton JIT
  compilation and blocks all workers for 5+ minutes while the first forward pass
  compiles. Training appears hung during this window.

**Decision:** Disable torch.compile entirely. The benchmark delta was only +3%
worker throughput — not worth the instability.

**Changes:**
- `configs/training.yaml` — `torch_compile: false`
- `hexo_rl/training/trainer.py` — default changed to `False`; added comment
  referencing §25 and §30
- `scripts/train.py` — `_torch_compile_enabled` default changed to `False`;
  inf_model block comment updated; resource_tracker suppression added at exit
  to silence spurious "leaked semaphore" warning from Rust PyO3 OS primitives
- `scripts/train.py` — corpus load now logs `corpus_loaded` with position count
  and elapsed seconds so users can see how long the NPZ push takes

**Re-enablement criteria:** Deferred until PyTorch stabilizes CUDA graph support
on Python 3.14. Track upstream: https://github.com/pytorch/pytorch/issues

**Commit:**
- `fix(training): disable torch.compile (Python 3.14 compat issues)`

---

### 35. Optimized 50K-position uncompressed NPZ for buffer prefill (2026-04-04)

**Problem:** The full-corpus NPZ (912K positions) was 116 MB compressed but
decompressed to ~13 GB in RAM because `np.savez_compressed` defeats mmap.
This caused a 36 GB+ RAM spike at training startup and system instability.

**Solution:** Rewrote `scripts/export_corpus_npz.py` to produce a quality-filtered,
position-sampled, **uncompressed** NPZ. With `np.savez` (not `savez_compressed`),
`np.load(mmap_mode='r')` works correctly — the OS pages in data on demand and
RAM stays near-zero until positions are actually pushed to the buffer.

**Pipeline:**
1. Load all corpus sources: human (1,944), bot-fast (5,598), bot-strong (2,804), injected (5,567)
2. Filter: decisive games only, game length ≥ 15 plies
3. Weight human games by Elo band (unrated 0.5×, sub-1000 0.3×, 1000-1200 0.7×, 1200-1400 1.0×, 1400+ 1.5×)
4. Sample positions from plies 8-50 per game, weighted by `source_weight / game_length`
5. Weighted sample 50K positions without replacement (RNG seed 42)
6. Replay only the unique games needed (14,845 of 15,913), extract sampled positions
7. Save uncompressed

**Results:**
- Old: 116 MB compressed, ~13 GB RAM on load, 912K positions, took minutes to push
- New: 720 MB uncompressed, ~0.7 GB RAM when pushed to buffer, 49,878 positions, loaded in 0.9s
- Training startup RAM reduced from 36 GB+ → < 4 GB

**Changes:**
- `scripts/export_corpus_npz.py` — rewritten with sampling pipeline and `--max-positions`/`--no-compress` flags
- `scripts/train.py` — added `corpus_prefill` log (positions + file_mb); warning if >100K positions
- `Makefile` — `corpus.npz` target passes `--max-positions 50000 --no-compress`

**Commit:**
- `perf(corpus): optimized 50K-position uncompressed NPZ for buffer prefill`

---

### 36. Cosine-annealed temperature + ZOI lookback (2026-04-04)
**Files:** `engine/src/game_runner.rs`, `hexo_rl/selfplay/pool.py`,
`configs/selfplay.yaml`

**Cosine-annealed temperature**

Replaced the hard step at move 30 (`temperature = 1.0 if ply < 30 else temp_min`)
with a smooth cosine schedule:

```
temperature = temp_min + 0.5 * (1.0 - temp_min) * (1 + cos(π * ply / anneal_moves))
```

Temperature starts at 1.0 at ply=0 and reaches `temp_min=0.05` at `anneal_moves`
(defaults to 60 plies, matching the old hard step position). This removes the
discontinuity in exploration pressure mid-game, which was causing visible policy
entropy spikes at ply 30 in prior training runs.

Config keys in `configs/selfplay.yaml`:
- `mcts.temp_min: 0.05`
- `mcts.temp_anneal_moves: 60`

**ZOI (Zone of Interest) lookback**

Restricts candidate moves passed to MCTS to cells within hex-distance 5 of the
last 16 moves (the "zone of interest"). Falls back to the full legal set if fewer
than 3 ZOI candidates remain, preventing degenerate positions where the ZOI is
empty or trivially small.

Motivation: in a typical mid-game position the legal-move radius-8 set contains
hundreds of candidates. The vast majority are far from any recent activity and
carry near-zero policy weight. Restricting to ZOI reduces the effective branching
factor without changing which moves are ultimately legal, allowing MCTS to focus
sims where the action is.

Config keys:
- `mcts.zoi_enabled: true`
- `mcts.zoi_radius: 5`
- `mcts.zoi_history: 16`
- `mcts.zoi_min_candidates: 3`

**Impact on CPU-only benchmark:** Negligible. ZOI candidate reduction only fires
in real games with recent-move history; the empty-board MCTS benchmark starts
from an initial position where ZOI falls back to the full legal set immediately.

**Test counts:** 86 Rust + 603 Python, all passing.

---

### 37. Ownership + threat auxiliary heads (2026-04-04)
**Files:** `hexo_rl/model/network.py`, `engine/src/game_runner.rs`,
`hexo_rl/selfplay/pool.py`, `hexo_rl/training/trainer.py`,
`hexo_rl/training/losses.py`, `configs/training.yaml`,
`tests/test_trainer.py`, `tests/test_smoke.py`

**Ownership head**

`HexTacToeNet` gains an `ownership_head`: trunk → `Conv2d(1×1)` → `tanh` →
(19×19) spatial map, one scalar per cell ∈ [-1, +1] indicating final stone
ownership (+1 = current player, -1 = opponent, 0 = empty at game end).

Rust `game_runner.rs` computes the ownership map at game end: each occupied cell
is tagged with ±1 from the perspective of the player to move; unoccupied cells
remain 0. The WorkerPool ring buffer supplies ownership targets alongside
policy/value targets. Loss: spatial MSE, weight `ownership_weight: 0.1` in
`configs/training.yaml`.

**Threat head**

`HexTacToeNet` gains a `threat_head`: trunk → `Conv2d(1×1)` → `sigmoid` →
(19×19) binary map, marking cells that belong to a winning line at game end.
Rust computes the threat map using the existing `Board.get_threats()` path at
terminal nodes. Loss: binary cross-entropy, weight `threat_weight: 0.1`.

**Why both heads help:**

- Ownership teaches the network where stones will end up, giving the value head
  richer spatial grounding even from positions far from game end.
- Threat detection teaches the network which cells form winning patterns,
  directly supporting the quiescence reasoning added in §28.

**Smoke test (20 steps, 200-step game cap):**
```
ownership_loss=0.29, threat_loss=0.27, avg_sigma=0.88 — all finite, no panics
```

**Commit:**
- `feat(model): ownership + threat auxiliary heads`

**Test counts:** 86 Rust + 603 Python, all passing.

---

### 38. Fix action_idx u16→u32 (2026-04-04)
**Files:** `engine/src/mcts/mod.rs`, `engine/src/mcts/node.rs`

**Pre-existing silent bug:** The action index used to identify moves in the MCTS
node pool was stored as `u16`, capping at 65,535. With hex-ball-8 legal move
radius the candidate set can be hundreds of cells per position, and global axial
coordinate encoding produces action indices well above the u16 range. When an
index overflowed, the wrong child node was selected or created, corrupting tree
statistics silently — no panic, no assertion failure.

**Fix:** Widened `action_idx` from `u16` to `u32` throughout. This is not a
regression introduced in this session; the bug predated Phase 4.0 and was exposed
only after the legal move radius was corrected to 8 in §26.

**Commit:**
- `fix(mcts): widen action_idx u16→u32 to support infinite board coordinates`

**Test counts:** 86 Rust + 603 Python, all passing.

---

### 39. Benchmark rebaseline 2026-04-04_19-53 (2026-04-04)
**Files:** `reports/benchmarks/2026-04-04_19-53.json`, `CLAUDE.md`

Full `make bench.full` (n=5, 3s warm-up) after §36-§38 landed:

| Metric | Median | IQR | Target | Status |
|---|---|---|---|---|
| MCTS sim/s | 31,331 | ±103 | ≥ 26,000 | PASS |
| NN inference batch=64 pos/s | 11,062 | ±5 | ≥ 8,500 | PASS |
| NN latency batch=1 ms | 2.75 | ±0.02 | ≤ 3.5 | PASS |
| Buffer push pos/s | 789,923 | ±14,790 | ≥ 630,000 | PASS |
| Buffer sample raw µs/batch | 1,220.8 | ±1.7 | ≤ 1,500 | PASS |
| Buffer sample aug µs/batch | 1,113.1 | ±2.1 | ≤ 1,400 | PASS |
| GPU util % | 100.0 | ±0 | ≥ 85 | PASS |
| VRAM GB | 0.10/8.6 | ±0 | ≤ 6.9 | PASS |
| Worker throughput pos/hr | 723,036 | ±19,486 | ≥ 625,000 | PASS |
| Batch fill % | 100.0 | ±0 | ≥ 80 | PASS |

All 10 targets PASS. Report: `reports/benchmarks/2026-04-04_19-53.json`.

**Note on ZOI and MCTS sim/s:** ZOI had negligible impact on the CPU-only
benchmark (31,331 vs 30,963 prior baseline — within IQR). Expected: ZOI candidate
reduction only fires in real games with populated recent-move history. The
benchmark starts from an empty board and falls back to full legal set immediately,
so no reduction occurs.

**Test counts:** 86 Rust + 603 Python, all passing.

Immediate next steps:
1. Run `make train` (resume from latest checkpoint, `strict=False` on head load
   to accommodate new ownership/threat head weights not present in checkpoint)
2. Monitor entropy over first 5,000 steps — ZOI + cosine temperature expected
   to stabilise or recover the 1.85 amber reading; if entropy continues below
   1.5 after 5k steps, consider fresh bootstrap start
3. Watch `ownership_loss` and `threat_loss` alongside policy/value — both should
   decrease over the first 10k RL steps

---

### 40. Raise draw_reward -0.1 → -0.5 (draw collapse fix) (2026-04-05)
**Files:** `configs/training.yaml`, `docs/01_architecture.md`

**Observed failure:** First overnight self-play run produced 56.6% draws, with
nearly half of games hitting the 180-move cap. Both players scored ~20% wins.
This is early self-play's classic failure mode: two equally weak players can't
finish games, so everything runs to the cap and gets scored as a draw.

**Root cause:** At 56% draw frequency the expected value target per position is
`0.56 × draw_reward`. With `draw_reward = -0.1` that is only **-0.056** — far
too weak a signal against the ±1.0 win/loss targets. The value head was training
predominantly on draw data and learning "draws are what happens" rather than
learning to pursue wins.

**The §24 KrakenBot rationale (-0.1) assumed draws were the minority outcome.**
That assumption was correct for a trained network where draws occur occasionally.
At 56% frequency the incentive structure inverts: the model needs the draw penalty
to dominate the gradient to break the equilibrium.

**Fix:** `draw_reward: -0.1` → `draw_reward: -0.5`

At 56% draws the expected target becomes **-0.280**, a clear gradient away from
draw-seeking behaviour. Once draws become rare the penalty rarely fires and the
value distribution naturally re-centres on ±1.0 outcomes. -0.5 is well within
the [-1, +1] range and can be tuned down later if draws overcorrect to near zero.

**Why not -0.4:** At 56% frequency the difference between -0.4 and -0.5 is
0.056 per position — meaningful. -0.4 risks being too timid to break the mode
decisively; -0.5 is the cleaner choice to force the transition.

**Also applied in this session (§40b):** config reductions committed separately:
- `buffer_schedule`: 0→100K / 150K→250K / 500K→500K (was 250K/500K/1M)
- `playout_cap.standard_sims`: 400→200, `fast_sims`: 50→30
- `mixing.decay_steps`: 1_000_000→300_000

**Commit:**
- `config(training): raise draw_reward -0.1→-0.5 to break draw-collapse mode`

**Test counts:** 603 Python, all passing.

---

### 41. OOM resilience: checkpoint on SIGTERM/SIGINT + corpus page-cache cap (2026-04-05)
**Files:** `scripts/train.py`, `configs/training.yaml`

**Root cause of overnight OOM loss:** Two independent failure modes combined:

1. **No checkpoint saved on signal exit.** The only unconditional `trainer.save_checkpoint()`
   call was at line 841 of `train.py`, placed *after* the `try/finally` block that calls
   `pool.stop()`. If `pool.stop()` hangs (a known risk when worker threads are mid-inference),
   the process blocks indefinitely and the checkpoint is never written. The signal handler
   already set `_running[0] = False` and the periodic save inside `trainer.train_step()`
   was correct, but those saves happen every 500 steps; any progress since the last periodic
   save is lost on an unclean exit.

2. **Page cache growth from full corpus copy.** `push_game()` copies the entire NPZ corpus
   into the Rust `ReplayBuffer` (not mmap-resident — fully allocated). A large corpus
   (e.g. 500K positions × ~14.1 KB/pos ≈ 7 GB) fills RAM during startup and the mmap
   read that precedes it also warms the OS page cache. Over a long run, other page-cache
   pressure competes with GPU/model state and can trigger OOM.

**Fix 1 — Checkpoint on signal** (`scripts/train.py`):

Added `_shutdown_save = [False]` flag. The existing `_stop()` handler (registered for
both SIGTERM and SIGINT) now also sets this flag. At the top of `while _running[0]:` in
`_run_loop()`, a new check fires before any training work:

```python
if _shutdown_save[0]:
    log.info("shutdown_signal_checkpoint",
             msg="Shutdown signal received — saving checkpoint before exit", step=train_step)
    trainer.save_checkpoint(last_loss_info if last_loss_info else None)
    break
```

This saves the checkpoint *inside the loop*, before the finally block's `pool.stop()`.
The handler itself only sets the flag — no non-reentrant code called from signal context.
Double Ctrl+C (`_stop_count >= 2`) still calls `sys.exit(1)` immediately.

**Fix 2 — Corpus cap** (`scripts/train.py`, `configs/training.yaml`):

Added `mixing.pretrain_max_samples: 200_000` to `training.yaml`. After loading the mmap'd
NPZ and computing `T = len(pre_outcomes)`, the code now draws a sorted random subset before
`push_game()`:

```python
max_pre = int(mixing_cfg.get("pretrain_max_samples", 0))
if max_pre and T > max_pre:
    subset_idx = np.sort(np.random.default_rng(seed).choice(T, size=max_pre, replace=False))
    log.info("corpus_capped", original=T, kept=max_pre)
    pre_states, pre_policies, pre_outcomes = (pre_states[subset_idx], ...)
    T = max_pre
```

Sorted indices minimise random page faults during the initial mmap read. With 200K positions
the Rust buffer allocation and page cache footprint are bounded to ~2.5 GB
(200K × 18×19×19×2 bytes) regardless of corpus size. No-op when corpus ≤ 200K.

Config location: `pretrain_max_samples` lives under `mixing` in `training.yaml` (alongside
`pretrained_buffer_path`) rather than `corpus.yaml` (which governs corpus generation).

**Commits:**
- `fix(train): save checkpoint on SIGTERM/SIGINT before exit`
- `fix(pretrain): cap page cache footprint via pretrain_max_samples`

**Test counts:** 603 Python, all passing.

---

### 42. Viewer game replays written to disk, in-memory index capped (2026-04-05)
**Files:** `hexo_rl/monitoring/web_dashboard.py`, `configs/monitoring.yaml`

**Problem:** `WebDashboard._event_history` (deque maxlen=500) accumulated full
`game_complete` payloads — moves, moves_list, moves_detail, value_trace — consuming
1-3 GB of RAM over a long training run.

**Fix:** Full game records are now written to disk on arrival at
`runs/<run_id>/games/<game_id>.json`. Only a lightweight ref `{game_id, path,
winner, moves, worker_id, ts}` is kept in a separate `_game_index` deque (maxlen=50).
`_event_history` still receives a stripped copy (moves_list/moves_detail/value_trace
removed) for SocketIO `replay_history` replay on browser reconnect.

`/viewer/recent` reads `_game_index`. `/viewer/game/<id>` loads from disk via the
index path, with a glob fallback across `runs/*/games/` for games evicted from the
50-entry index. Disk write failures emit `log.warning("game_persist_failed")` and
never propagate.

`run_id` is captured from the `run_start` event (uuid set in `scripts/train.py`);
defaults to `"default"` if no run_start has arrived.

New config key: `monitoring.viewer_max_memory_games: 50`.

**Invariants unchanged:** events.py dispatch untouched; no training/selfplay files
modified; public viewer API routes unchanged.

**Commit:** `fix(viewer): write game replays to disk, cap in-memory index`

**Test counts:** 603 Python, all passing.

---

### 43. Revert fast_sims to 50 (2026-04-05)
**Files:** `configs/selfplay.yaml`

**Change:** `playout_cap.fast_sims` reverted from 30 → 50.

`standard_sims` (200) and `fast_sims` serve different purposes and don't
need to scale together. 50 sims produces meaningfully better policy signal
for fast-game value targets; the 30-sim reduction applied in §40b was not
justified.

**Commit:** `config(selfplay): revert fast_sims to 50`

**Test counts:** 603 Python, all passing.

---

### 44. Viewer disk rotation: cap at viewer_max_disk_games (2026-04-05)
**Files:** `hexo_rl/monitoring/web_dashboard.py`, `configs/monitoring.yaml`

**Problem:** `runs/<run_id>/games/` accumulated game JSON files without bound
over long training runs. Only the in-memory index was capped (§42); the disk
directory itself grew indefinitely.

**Fix:** After each successful game write in `_persist_game`, all `*.json`
files in the run's `games/` directory are listed sorted by mtime oldest-first.
If the count exceeds `viewer_max_disk_games`, the oldest files are deleted
until exactly `viewer_max_disk_games` remain. Individual deletion failures
emit `log.warning("game_disk_rotate_delete_failed")` and continue; outer
scan failures emit `log.warning("game_disk_rotate_failed")`. Neither ever
propagates to the training path (monitoring invariant).

Deletion happens **after** the new file is written, never before.

New config key: `monitoring.viewer_max_disk_games: 1000`.

The in-memory index cap (`viewer_max_memory_games: 50`) is unchanged and
applies independently.

**Commit:** `fix(viewer): rotate disk game files, cap at viewer_max_disk_games`

**Test counts:** 603 Python, all passing.

---

### 45. RSS memory tracking added to system monitoring (2026-04-05)
**Files:** `hexo_rl/monitoring/gpu_monitor.py`, `hexo_rl/monitoring/terminal_dashboard.py`,
`hexo_rl/monitoring/static/index.html`

**Problem:** Overnight run OOMed with no RSS history — impossible to tell whether
it was a slow leak, a spike, or which component was responsible.

**Fix:** Process RSS (`psutil.Process().memory_info().rss`) is now sampled on
every gpu_monitor poll cycle and included in the `system_stats` event as `rss_gb`.

- `_PROCESS = psutil.Process()` created once at module level (not per cycle).
- Wrapped in its own `try/except`; emits `rss_gb=0.0` and logs a warning on failure —
  never propagates (monitoring invariant preserved).
- Terminal dashboard system row: `rss  X.X GB` added between `ram` and `cpu`.
- Web dashboard system panel: `RSS` row added between `RAM` and `CPU`; populated
  by `onSystemStats` JS handler when `rss_gb` is present in the event.

No new config keys. No changes to events.py, web_dashboard.py, or any
training/selfplay files. Monitoring invariant fully preserved.

**Commit:** `feat(monitoring): add RSS memory tracking to system stats panel`

**Test counts:** 603 Python, all passing.

---

### 46. Fix three RSS memory leaks (2026-04-05)
**Files:** `scripts/train.py`, `hexo_rl/monitoring/web_dashboard.py`

**Problem:** Run `0944a606` grew to 14.8 GB RSS by step 590 (expected ~7-8 GB).
Three confirmed leak sources:

1. **mmap views held alive after push_game (~720 MB):** `del data` only released
   the `NpzFile` handle. The three `pre_*` array views kept the corpus mmap'd for
   the entire process lifetime. Fixed by adding
   `del pre_states, pre_policies, pre_outcomes` immediately after `push_game()`.

2. **SocketIO send-queue buildup (0.5-1.5 GB over multi-hour runs):**
   `self._socketio.emit()` was called unconditionally. When a browser disconnects
   without a clean WebSocket CLOSE the server-side socket enters CLOSING state and
   buffers messages. Fixed by tracking connected SIDs via `on_connect` /
   `on_disconnect` handlers and replacing the bare emit with `_safe_emit()`, which
   skips the call when `_connected_sids` is empty.

3. **Heap fragmentation from np.concatenate (~3-5 GB over 590 steps):**
   Five `np.concatenate` calls per training step each triggered a ~3.4 MB
   malloc/free directly via glibc, bypassing pymalloc. Repeated cycles of this
   size fragment the system heap; Python never returns fragmented arena memory to
   the OS. Fixed by pre-allocating `_states_buf / _policies_buf / _outcomes_buf`
   once (~3.6 MB total) before `_run_loop()` and filling them in-place with
   `np.copyto()`. Falls back to `np.concatenate` if `batch_size` changes at
   runtime (logs a warning after step 100).

**Commits:**
- `fix(train): release pretrain mmap references after push_game`
- `fix(monitoring): gate socketio emits on connected clients`
- `fix(train): pre-allocate mixed batch arrays to eliminate heap fragmentation`

**Test counts:** 86 Rust + 604 Python, all passing.

### 47. Guard FP16 0×-inf NaN in aux CE and entropy (2026-04-05)

**Problem:** Run `0944a606` produced NaN loss at step 590. Root cause: `0.0 × (-inf) = NaN`
under FP16 autocast in two locations, compounded by BatchNorm running stats being poisoned
before GradScaler could act.

**Root cause chain:**

1. **`compute_aux_loss` (primary source):** The opp_reply head's log-softmax produces
   `-inf` entries under FP16 when logit spread exceeds ~17.5 nats. After `policy_prune_frac`
   zeroes low-visit entries in the target distribution, those zero entries pair with `-inf`
   log-probs → `0.0 × -inf = NaN` in the cross-entropy sum.

2. **Entropy bonus/diagnostic:** Same pattern in the manual `-(p * log_p).sum()` computation.
   `log_policy.exp()` underflows to 0.0 for near-zero probabilities under FP16, then
   multiplies the corresponding `-inf` log entry.

3. **BatchNorm contamination:** BN running stats (`running_mean`, `running_var`) are updated
   during the forward pass, _before_ GradScaler can inspect or skip the step. One poisoned
   forward pass permanently NaN's all subsequent passes regardless of whether the optimizer
   step was skipped.

**Fixes (`fix(training): guard aux CE and entropy against FP16 0×-inf NaN`):**

- **`losses.py` — `compute_aux_loss`:** Clamp log-probs to `min=-100.0` before multiplying
  by target_policy. `exp(-100) ≈ 0` preserves all meaningful gradient information.

- **`trainer.py` — entropy (two sites):** Replace `-(p * log_p).sum()` with
  `torch.special.entr(p_fp32).sum()`, which defines `0·log(0) ≡ 0` and promotes to FP32
  before `exp()` to avoid FP16 underflow.

- **`trainer.py` — NaN guard:** After `compute_total_loss()`, before `fp16_backward_step()`:
  detect non-finite loss, reset any BN module with poisoned `running_mean`, call
  `scaler.update()` to let loss scale decay, and return early without touching weights.

**Why GradScaler alone wasn't enough:** GradScaler calls `torch.isinf()` on gradients, not
`torch.isnan()`. A NaN-producing `0 × -inf` operation slips past the inf check entirely.
BN stat corruption then propagates even if the optimizer step is skipped.

**Commits:**
- `fix(training): guard aux CE and entropy against FP16 0×-inf NaN`

**Test counts:** 86 Rust + 604 Python, all passing.

### 48. Pre-allocate stats loop receive buffers to reduce heap churn (2026-04-05)
**Files:** `hexo_rl/selfplay/pool.py`

**Problem:** `_stats_loop` called `collect_data()` at ~10 Hz. Each position returned
from Rust (as a Python list) was converted with `np.array(feat, dtype=np.float16)` and
`np.array(pol, dtype=np.float32)`, creating and immediately discarding two numpy array
objects per position. Python's arena allocator grows to service peak allocations but
never returns arena memory to the OS → estimated 100-300 MB RSS creep over multi-hour runs.

**Fix:** Stored `_feat_len` and `_pol_len` on `self` in `__init__`. Pre-allocate
`_feat_buf` (float16) and `_pol_buf` (float32) once before the while loop. Replace
`np.array(feat, ...)` / `np.array(pol, ...)` with in-place `_feat_buf[:] = feat` /
`_pol_buf[:] = pol`. `_feat_2d` is a persistent reshaped view — no array object created
or freed per iteration. Safe because `replay_buffer.push()` and `recent_buffer.push()`
copy data into Rust memory before returning.

**Commits:**
- `perf(selfplay): pre-allocate stats loop receive buffers to reduce heap churn`

**Test counts:** 86 Rust + 604 Python, all passing.

### 49. Guard uncertainty head against FP16 sigma2 underflow (2026-04-05)
**Files:** `hexo_rl/training/losses.py`

**Problem:** `compute_uncertainty_loss` ran under `torch.autocast(dtype=float16)`.
`sigma2 = Softplus(linear_output)` can approach the FP16 minimum (~6e-8). The
division `(z - v)² / sigma2` blows up to `+inf` if `sigma2` enters subnormal
territory → `0.5 * inf = inf` → NaN in subsequent ops. Same 0×-inf / overflow
family as the §47 aux CE / entropy bugs. Didn't trigger at 20 steps but would
manifest at longer runs as uncertainty head weights grow.

**Fix:** Promote `sigma2`, `z`, and `value_detached` to FP32 before all operations.
Clamp `sigma2_fp32` to `min=1e-6` before `log()` and division.
`log(1e-6) ≈ -13.8` — well within FP32 range; `σ ≈ 0.001` is a physically meaningful
minimum uncertainty. Result scalar is FP32 and combines with the total loss at full
precision, consistent with the §47 entropy fix pattern.

**Commits:**
- `fix(training): guard uncertainty head against FP16 sigma2 underflow`

**Test counts:** 86 Rust + 604 Python, all passing.

---

### 50. Batch buffer warm-up fallback + stale BCE docstring (2026-04-05)
**Files:** `scripts/train.py`, `hexo_rl/training/losses.py`

**Investigation:** The §49 session brief identified value BCE and threat BCE as
remaining `F.binary_cross_entropy(sigmoid(x), target)` uses. Code audit found both
were already fixed prior to §47:

- `compute_value_loss()` (`losses.py:46`) already uses
  `nn.functional.binary_cross_entropy_with_logits` — present since the value head
  overhaul in §1.
- Threat head (`network.py:204-205`) already returns raw logits (no sigmoid). Trainer
  (`trainer.py:308-310`) already uses `binary_cross_entropy_with_logits` for threat
  loss — introduced when the threat head was added in §37.

The module docstring (`losses.py` line 6) and function docstring (line 44) still
described the old `BCE(sigmoid(v_logit), ...)` form. Corrected to match the
implementation.

**Batch buffer warm-up edge case:**
The pre-allocated `np.copyto` path added in §46 assumed sampled arrays always have
exactly the requested number of rows. During warm-up (first ~100 K positions), the
Rust replay buffer may have fewer entries than `n_self_uniform` and can return fewer
rows than requested. This causes a shape mismatch crash in `np.copyto`.

**Fix:** Added `n_available = n_pre + len(s_r) + len(s_u)` (recency path) and
`n_available = n_pre + len(s_self)` (uniform path) checks after sampling. When
`n_available < batch_size` (warm-up period), fall back to `np.concatenate` — correct
but allocates. When the buffer fills to `batch_size`, flip `_warmup_fallback_active`
to `False` and log one `buffer_warmup_ended` event at info level. From that point
onward the in-place path is used permanently.

**Commits:**
- `fix(train): batch buffer warm-up fallback when buffer underfull`
- `docs(training): correct stale BCE docstring in losses.py + sprint §50`

**Test counts:** 86 Rust + 604 Python, all passing.

---

### 51. collect_data/drain_game_results return numpy arrays — eliminate pymalloc RSS growth (2026-04-05)
**Files:** `engine/src/game_runner.rs`, `hexo_rl/selfplay/pool.py`, `tests/test_worker_pool.py`

**Root cause confirmed:** `collect_data()` returned `Vec<(Vec<f32>, Vec<f32>, f32, usize)>`.
PyO3 converts each `Vec<f32>` to a Python list of Python float objects using pymalloc arenas.
At 7 positions/sec × ~164 KB pymalloc/position ≈ 1.1 MB/sec pymalloc growth; arenas are
never returned to the OS → ~0.15 GB/min RSS leak. Estimated 1.5-2.5 GB fragmentation over
a 40-minute run.

**Fix — `collect_data()`:** Changed return type from `Vec<(Vec<f32>, Vec<f32>, f32, usize)>`
to `(PyArray2<f32>, PyArray2<f32>, PyArray1<f32>, PyArray1<u64>)` (shapes: (N, feat_len),
(N, pol_len), (N,), (N,)). Implementation drains the results queue into flat Vecs
(`extend_from_slice` — no per-element Python overhead), then wraps via `into_pyarray + reshape`.
This is the same zero-copy pattern used in `replay_buffer/sampling.rs::sample_batch`.
`pol_len` stored as a new field in `SelfPlayRunner` (set from the `policy_len` constructor
param) to handle the n=0 empty-array case without GIL overhead.

**Fix — `drain_game_results()`:** Changed `ownership_flat` and `winning_line_flat` from
`Vec<f32>` to `Py<PyArray1<f32>>` in the return tuple. `into_pyarray(py).unbind()` transfers
Vec ownership to numpy (zero copy). Rust-internal helper `drain_game_results_raw()` added
to the non-pymethods impl block — returns raw Vecs, used by the `cargo test` suite (which
can't link Python in extension-module mode). The pymethods version delegates to the helper.

**pool.py _stats_loop update:**
- `collect_data()` result destructured as `feats_np, pols_np, vals_np, plies_np = ...`
- Loop iterates `for i in range(n)` over numpy rows
- `_feat_buf[:] = feats_np[i]` — numpy view (no copy), f32→f16 at C speed, no Python floats
- `drain_game_results()` ownership/winning_line now numpy — `if ownership_flat:` replaced with
  `if ownership_flat.size > 0:` (numpy array bool is ambiguous for multi-element arrays)
- `np.array(ownership_flat, ...).reshape(...)` replaced with `ownership_flat.reshape(...)`
  (already float32 numpy — no conversion needed)

**Performance:** `collect_data` is called at ~10Hz from the stats thread, not in the MCTS
hot path. No worker throughput regression expected or observed.

**Benchmark (post-fix, 2026-04-05_21-15, n=5):**

| Metric | §39 Baseline | Post-fix | Target | Status |
|---|---|---|---|---|
| MCTS sim/s | 31,331 | 30,808 | ≥ 26,000 | PASS |
| NN inference batch=64 | 11,062 | 10,849 | ≥ 8,500 | PASS |
| NN latency batch=1 | 2.75 ms | 2.93 ms | ≤ 3.5 ms | PASS |
| Buffer push pos/s | 789,923 | 806,789 | ≥ 630,000 | PASS |
| Buffer sample raw | 1,220.8 µs | 1,247.0 µs | ≤ 1,500 µs | PASS |
| Buffer sample aug | 1,113.1 µs | 1,137.5 µs | ≤ 1,400 µs | PASS |
| GPU utilization | 100% | 100% | ≥ 85% | PASS |
| VRAM | 0.10 GB | 0.10 GB | ≤ 6.9 GB | PASS |
| Worker throughput | 723,036 | 729,919 | ≥ 625,000 | PASS |
| Batch fill % | 100% | 100% | ≥ 80% | PASS |

All 10 targets PASS. Worker throughput 729,919 (+0.9% vs §39 baseline) — within IQR variance.

**Commits:**
- `fix(engine): collect_data returns PyArray instead of Vec to eliminate pymalloc churn`
  (also contains drain_game_results fix — single file, committed together)

**Test counts:** 86 Rust + 604 Python, all passing.

---

### §46. Buffer display fix + buffer persistence + inference autocast fix
**Files:** `hexo_rl/selfplay/pool.py`, `scripts/train.py`,
`engine/src/replay_buffer/mod.rs`, `configs/training.yaml`,
`tests/test_rust_replay_buffer.py`, `hexo_rl/selfplay/inference_server.py`

**Problem 1 — Dashboard buffer row goes stale:**
Between `iteration_complete` events (every `log_interval` steps), the buffer
display showed stale data. `gpu_monitor.py` emits `system_stats` every 5s but
never included `buffer_size`. The pool stats thread pushed positions but emitted
only `game_complete` events.

**Fix:** Pool `_stats_loop` now emits `system_stats` with `buffer_size` /
`buffer_capacity` every ~5s (throttled, not every 0.1s loop tick). Also emit
after corpus loading completes to give the dashboard an initial buffer state.
Dashboard merge logic (`terminal_dashboard.py`, `web_dashboard.py`) already
handles partial `system_stats` updates — no renderer changes needed.

**Problem 2 — Buffer lost on shutdown:**
The Rust ReplayBuffer contents were discarded on clean shutdown, causing a
20-65 minute cold-start penalty on `make train.resume`.

**Fix — Rust persistence API:**
Added `save_to_path(path)` and `load_from_path(path)` to ReplayBuffer.

Binary format (HEXB v1, little-endian native):
```
[magic: u32 = 0x48455842]  // "HEXB"
[version: u32 = 1]
[capacity: u64]
[size: u64]
// Per position (oldest → newest):
//   state:  6498 × u16   (12,996 bytes)
//   policy: 362 × f32    (1,448 bytes)
//   outcome: f32          (4 bytes)
//   game_id: i64          (8 bytes)
//   weight:  u16          (2 bytes)
// Total: 14,458 bytes/entry
```

- `save_to_path`: linearizes ring buffer (oldest→newest), writes via BufWriter.
- `load_from_path`: validates magic+version, loads min(saved_size, capacity)
  most recent positions (skips oldest if saved > capacity), rebuilds weight
  histogram. Missing file returns Ok(0) silently (first-run case).
- File size: ~1.4 GB for 100K positions, ~7.2 GB for 500K.

**Fix — Python wiring:**
- SIGINT/SIGTERM handler: saves buffer after checkpoint save.
- Resume path: restores buffer before pool.start(). **Skips corpus prefill**
  when buffer was restored — stale corpus is unnecessary on top of fresh
  self-play data.
- Gated on `mixing.buffer_persist: true` (default) in `configs/training.yaml`.

**RecentBuffer repopulation — deferred:**
The Rust API doesn't expose a position iterator. RecentBuffer fills naturally
from new self-play within ~30 minutes. Adding a `get_positions(start, count)`
method to expose a slice of the ring buffer for RecentBuffer seeding is a
clean follow-up if cold-start time matters.

**Problem 3 — Inference tests fail on CPU-only environments:**
`torch.autocast("cpu")` produces bfloat16 (vs float16 on CUDA). NumPy doesn't
support bfloat16, so `log_policy.exp().cpu().numpy()` threw
`TypeError: Got unsupported ScalarType BFloat16`.

**Fix:** Added `.float()` before `.numpy()` and re-normalize `exp(log_policy)`
to correct reduced-precision rounding drift (bfloat16 has only 7-bit mantissa,
so exp() over 362 actions can drift ~1.2% from sum=1.0). The `.float()` call
is a no-op when the tensor is already float32, so zero impact on CUDA path.

**Config changes (`configs/training.yaml`):**
```yaml
mixing:
  buffer_persist: true
  buffer_persist_path: "checkpoints/replay_buffer.bin"
```

**Tests added:**
- `test_buffer_save_load_roundtrip` — push 150 positions, save, load, verify
- `test_buffer_load_missing_file_ok` — nonexistent path returns 0
- `test_buffer_load_size_mismatch` — save cap=500, load into cap=250

**Commits:**
- `fix(dashboard): emit buffer_size during corpus load and from pool stats thread`
- `feat(engine): ReplayBuffer.save_to_path / load_from_path for buffer persistence`
- `feat(training): save and restore replay buffer on shutdown/resume`
- `fix(inference): cast autocast output to float32 before numpy conversion`

**Test counts:** 86 Rust + 596 Python passing (3 env-specific failures on
remote server: 2 psutil not installed, 1 missing game data files).

---

### 52. eval_interval 500 → 2000 (eval blocking self-play) (2026-04-06)
**Files:** `configs/training.yaml`

**Observed failure:** At `eval_interval: 500`, each eval round blocked
self-play for ~25 minutes (20 random games + 50 SealBot games + 200
best_arena games × ~8.8 s/game). With a cold-start buffer (first ~500
steps after resume), this was longer than the actual training time.

**Root cause:** Two conflicting `eval_interval` settings existed:
- `configs/eval.yaml`: `eval_interval: 1000` (read at line 423 for eval
  pipeline gating)
- `configs/training.yaml`: `eval_interval: 500` (read at line 507 for
  the actual loop trigger — **overwrites** the eval.yaml value)

The training.yaml value won, so evals fired every 500 steps.

**Fix:** `eval_interval: 500` → `eval_interval: 2000` in `configs/training.yaml`.
At ~600 steps/hr this gives one eval every ~3.3 hours — appropriate for
sustained RL runs where the buffer needs time to fill between evals.

**Commit:** `config(training): increase eval_interval 500→2000`

---

### 53. Memory leak investigation — glibc arenas + Python heap (2026-04-06)
**Files:** `Makefile`, `scripts/train.py`

#### Observed RSS trajectory

| Time | Step | RSS | Event |
|---|---|---|---|
| +10 min | ~700 | 6.34 GB | Buffer at 42k, demand-paging in progress |
| +25 min | ~950 | 17.2 GB | Buffer hit 100k cap — all slots demand-paged |
| +27 min | ~950 | 17.45 GB | Flat for 1 min — initial plateau |
| +65 min | ~1480 | 22.17 GB | Still growing |
| +75 min | ~1480 | 22.36 GB | +0.19 GB / 60s |
| +125 min | ~1980 | 24.40 GB | +2.04 GB since plateau |

Rate after plateau: **~+2 GB/hr**. At 48 GB total RAM, OOM risk within ~12 hrs.

#### Investigation method

`/proc/PID/smaps` segment-level snapshot taken at step ~1980, repeated 20
minutes later. Delta compared by address.

#### Culprit 1 — glibc malloc arenas (+~2.2 GB / 20 min)

**35 new 64 MB `[anon]` segments** appeared in 20 minutes. glibc creates
one arena per thread when existing arenas are contended. Sources:
- Rust's rayon/crossbeam thread pool (SelfPlayRunner worker threads)
- PyTorch OpenMP/MKL threads (vary with batch workload)

These arenas are never returned to the OS — glibc retains them indefinitely.
With the default `MALLOC_ARENA_MAX` (up to 128 on a 16-thread system), the
process accumulates hundreds of 64 MB arenas over a multi-hour run.

**Fix:** `MALLOC_ARENA_MAX=2` prepended to all training Makefile targets.
Caps arena count to 2 regardless of thread count. Standard fix for Python
+ native extension workloads (TF, Gunicorn, uWSGI all recommend it).
GPU tensor allocation is unaffected (uses CUDA's allocator, not glibc).

**Why not jemalloc:** jemalloc would be marginally better (actively returns
freed memory to OS, superior thread-local caching). However, it is not
installed on this system and adds a system dependency. `MALLOC_ARENA_MAX=2`
achieves 90%+ of the benefit with zero dependencies. jemalloc is the
preferred upgrade if the arena fix proves insufficient.

#### Culprit 2 — Python heap (+642 MB / 20 min)

The Python heap segment grew 642 MB in the same 20-minute window. Exact
source unknown. All bounded collections were audited and found clean:
- All `deque`s in monitoring have `maxlen` set ✓
- `GameRecorder` writes to disk via a queue, not memory ✓
- Pool rings (`_ownership_ring`, `_threat_ring`, `_game_lengths`) capped at 200 ✓
- `_iter_games_window` trimmed to 60s rolling window ✓
- `replay_poller._file_offsets` grows but is O(log files) — negligible ✓

Likely sources (unconfirmed): torch gradient buffers accumulating without
explicit `del`; structlog context dict fragmentation; eval model clone not
being GC'd due to reference cycles in the `_run_eval` closure.

**Fix:** `tracemalloc.start(nframe=3)` before `_run_loop()`, snapshot logged
every 500 steps as `tracemalloc_top10` event (file, size_mb, count).
`tracemalloc.stop()` in finally block. Will identify exact file+line within
~10 minutes of starting the next training run.

#### Segment breakdown at step ~1980

| Segment | Size | Identified as |
|---|---|---|
| 1377 MB [anon] | ~1.4 GB | Rust ReplayBuffer (100k × 13.7 KB ≈ 1.37 GB ✓) |
| 687 MB [anon] | ~0.7 GB | PyTorch train_model weights + optimizer state |
| 631 MB [heap] | ~0.6 GB | Python heap |
| 620 MB [anon] | ~0.6 GB | PyTorch inf_model + compiled graph buffers |
| 162 MB × 8+ | ~1.3 GB | glibc malloc arenas (one per active thread) |

**Commits:**
- `fix(train): cap glibc malloc arenas with MALLOC_ARENA_MAX=2 in Makefile train targets`
- `feat(monitoring): add tracemalloc top-10 heap logging every 500 steps`

**Test counts:** 86 Rust + 607 Python, all passing.

---

### 54. grad_norm and loss written to structured log unconditionally (2026-04-06)
**Files:** `hexo_rl/training/trainer.py`, `tests/test_trainer.py`

`grad_norm` was previously only emitted via `emit_event()` to the dashboard.
If the dashboard was not running, the metric was invisible and could not be
audited post-hoc from the JSONL log.

Added an unconditional `log.info("train_step", ...)` call at the end of
`_train_on_batch()`, immediately before `return result`. Fields logged:

| Field | Source |
|---|---|
| `step` | `self.step` |
| `grad_norm` | `result["grad_norm"]` |
| `total_loss` | `result["loss"]` |
| `policy_loss` | `result["policy_loss"]` |
| `value_loss` | `result["value_loss"]` |
| `aux_loss` | `result.get("opp_reply_loss")` (None if aux head off) |
| `lr` | `result["lr"]` |
| `fp16_scale` | `self.scaler.get_scale()` |

`emit_event()` and all training logic unchanged. Test added:
`test_train_step_logs_grad_norm` — uses `structlog.testing.capture_logs()`
to assert the event is present with the required fields.

**Commit:** `feat(monitoring): write grad_norm and loss to structured log independent of dashboard`

**Test counts:** 86 Rust + 608 Python, all passing.

---

### 55. Bounded SocketIO send queue + client-side auto-reconnect (2026-04-06)
**Files:** `hexo_rl/monitoring/web_dashboard.py`, `hexo_rl/monitoring/static/index.html`,
`tests/test_dashboard_renderers.py`

**Problem:** Under sustained training load, `socketio.emit()` in `_safe_emit()` could
block when the WebSocket send buffer backed up (inactive or disconnected browser tab).
This stalled the training loop — violating the "passive observer" invariant — and caused
the dashboard to show frozen metric values (grad_norm stuck at last-received value).

**Fix 1 — bounded emit queue (web_dashboard.py):**
- Replaced direct `socketio.emit()` in `_safe_emit()` with a `queue.Queue(maxsize=200)`.
- Training loop calls `put_nowait()` — returns immediately; raises `queue.Full` if
  backed up, which is caught and the event is silently dropped (newest-dropped policy:
  stale data is worse than gaps).
- A daemon thread (`socketio-drain`) started in `start()` drains the queue into
  `socketio.emit()`. If emit blocks or fails, only the drain thread is affected;
  the training loop is never touched.
- Queue ceiling is configurable via `monitoring.emit_queue_maxsize` (default 200).
  At ~1KB per event, worst-case footprint is ~200KB — no RAM leak risk.

**Fix 2 — client-side auto-reconnect (index.html):**
- `io()` → `io({ reconnection: true, reconnectionDelay: 2000, reconnectionAttempts: Infinity })`
- Status indicator shows "Reconnecting…" on disconnect/reconnect_attempt (was "disconnected").
- On successful reconnect the server's existing `on_connect` handler sends `replay_history`,
  which calls `resetState()` + replays all buffered events — dashboard repopulates
  immediately without any additional client-side request.

**Tests added:**
- `test_emit_does_not_block_when_no_client` — asserts `on_event()` returns in <10ms
  with no connected client.
- `test_queue_drops_when_full` — fills queue to maxsize, pushes one more via
  `_safe_emit()`, asserts no exception and queue size stays at maxsize.

**Commit:** `fix(dashboard): bound SocketIO send queue and add client-side auto-reconnect`

**Test counts:** 86 Rust + 610 Python, all passing.

---

### §56 — Buffer save reliability on shutdown + structlog JSONL file sink

**Files:** `scripts/train.py`, `hexo_rl/monitoring/configure.py`
**Tests:** `tests/test_buffer_shutdown.py`, `tests/test_structlog_sink.py`

**Fix 1 — buffer save on shutdown signal (train.py):**

The buffer save was in the post-loop cleanup block, executed after `pool.stop()`.
If the user sends a second Ctrl+C during teardown, `sys.exit(1)` is called inside
the signal handler and the cleanup block is skipped entirely — buffer lost.

Fix: added an early save in the `_shutdown_save[0]` branch of `_run_loop()`,
immediately after `trainer.save_checkpoint()` and before `break`. Guarded by the
same `buffer_persist` config check that exists in the post-loop block. Wrapped in
`try/except` so a failing save logs a `buffer_save_failed` warning and never
prevents the process from exiting. The post-loop save remains as a fallback for
the normal (non-signal) exit path — two save points total.

**Fix 2 — structlog JSONL file sink (configure.py + train.py):**

`configure_logging` creates a JSONL file via `WriteLoggerFactory(file=log_file)` but
the file handle was never explicitly closed, risking data loss on abrupt exit. The
filename used a raw timestamp or `--run-name`, not the training `run_id`.

Changes:
- `configure_logging` now returns `(log, log_file)` instead of just `log`. The
  caller is responsible for closing the handle.
- `run_id = uuid.uuid4().hex` moved to the top of `main()`, before
  `configure_logging` is called. Log filename is `logs/train_{run_id}.jsonl`
  (or `logs/{run_name}.jsonl` if `--run-name` is passed explicitly).
- `_log_fh.close()` added to the `finally` block in the training-run scope.

**Tests added:**
- `test_buffer_save_on_shutdown_signal` — replicate the shutdown-signal save block,
  assert `save_to_path` produces a non-empty file and logs `buffer_saved`.
- `test_buffer_save_on_shutdown_signal_exception_does_not_propagate` — bad path,
  assert `buffer_save_failed` warning emitted and exception not raised.
- `test_buffer_save_skipped_when_persist_disabled` — assert no save when
  `buffer_persist: false`.
- `test_structlog_file_sink_creates_jsonl` — assert JSONL file exists and every
  line is valid JSON with an `event` key.
- `test_structlog_file_sink_returns_file_handle` — assert return type is a tuple
  with a file-like second element.
- `test_structlog_file_sink_train_prefix` — assert `run_name='train_{run_id}'` →
  file `logs/train_{run_id}.jsonl`, with correct event content.

**Commits:**
- `fix(train): save replay buffer immediately on shutdown signal before pool teardown`
- `feat(monitoring): add structlog JSONL file sink to logs/train_{run_id}.jsonl`

**Test counts:** 86 Rust + 616 Python, all passing.

---

### §57 — Integration test: cold-start → prefill → run → SIGINT → resume lifecycle

**Problem:** The buffer persistence feature (§55-§56) had no end-to-end acceptance
test. Unit tests verified individual save/load paths in isolation but could not
catch lifecycle regressions (e.g., corpus being re-loaded on resume, buffer not
surviving a SIGINT, or the save happening after pool teardown instead of before).

**Changes:**
- `tests/test_train_lifecycle.py` — new slow/integration test that exercises the
  complete four-phase lifecycle:
  - **Phase A** (cold start): launches `train.py` via `Popen` with a 60 K synthetic
    corpus. Polls the structlog JSONL for `corpus_loaded` (≥ 50 K positions) and
    asserts it precedes `selfplay_pool_started`; asserts `corpus_prefill_skipped`
    is absent.
  - **Phase A continued**: polls for a `train_step` event with `step ≥ 50`, records
    buffer size N.
  - **Phase B** (SIGINT): sends `SIGINT`, polls for `replay_buffer.bin` within 30 s,
    asserts file size > 0.
  - **Phase C** (resume): relaunches with `--checkpoint <latest> --iterations 5`,
    asserts `buffer_restored` (positions ≈ N), `corpus_prefill_skipped`, and
    `selfplay_pool_started` in the log.
- `pytest.ini` — registered `slow` and `integration` markers.
- `Makefile` — `test.py` now passes `-m "not slow and not integration"` so the
  lifecycle test is excluded from the normal test loop; new `test.integration`
  target runs it explicitly.
- `CLAUDE.md` — added `make test.integration` to the make-commands table; updated
  test count (617 Python = 616 unit + 1 integration).

**Test counts:** 86 Rust + 617 Python (616 unit + 1 integration), all passing.

---

### §58 — Three resume bugs found in JSONL log (2026-04-06)

**Problem:** After resuming from a checkpoint at step 828, the JSONL showed:
1. `pretrained_weight: 0.0, selfplay_weight: 1.0` — corpus mixing disabled.
2. `buffer_restored` with only 1,770 positions still firing `corpus_prefill_skipped`.
3. `total_loss: 393-1545` at steps 835-843 while `policy_loss/value_loss/aux_loss` were all normal (~3-7 total).

**Root causes:**

**Bug 1 — pretrained stream not initialised on resume:**
`scripts/train.py` used an `if _buffer_restored: ... elif pretrained_path ...` structure that
prevented `pretrained_buffer` from ever being loaded when `_buffer_restored = True`.
Since both the `corpus_prefill_skipped` log AND the `pretrained_buffer = ReplayBuffer(...)`
init were inside the same conditional, any buffer restore (even trivially small) silently
disabled corpus batch mixing for the entire resumed run.

**Fix:** Decoupled the two concerns. `_buffer_restored` now only controls the
`corpus_prefill_skipped` log message; `pretrained_buffer` is always initialised
from the corpus NPZ when the file exists, regardless of `_buffer_restored`.
`compute_pretrained_weight(train_step)` already uses the restored step counter, so
the decay formula automatically starts from the correct position.

**Bug 2 — corpus prefill skip threshold was `n_loaded > 0`:**
Any non-zero buffer restore (even 1,770 positions, 1.7% of capacity) set
`_buffer_restored = True` and fired `corpus_prefill_skipped`. Combined with Bug 1
this left the buffer nearly empty with no pretrained mixing.

**Fix:** Added `_MIN_BUFFER_PREFILL_SKIP = 10_000`. `_buffer_restored` is now
`n_loaded >= 10_000`. When restored positions are below the threshold, a new
`corpus_prefill_running` event is logged with `restored_positions` and
`reason="buffer_too_small"` to make the decision auditable.

**Bug 3 — uncertainty/ownership/threat losses not logged → apparent total_loss spikes:**
`trainer.py`'s per-step `log.info("train_step", ...)` included `total_loss` but
omitted `uncertainty_loss`, `ownership_loss`, and `threat_loss`.  These terms ARE
computed and added to total_loss via `compute_total_loss()` (with weights 0.05, 0.1,
0.1 respectively) but were invisible in the JSONL.  The uncertainty head is randomly
initialised when resuming from a pretrain checkpoint (loaded with `strict=False`); in
the first few steps small sigma² values (Softplus output) can produce large Gaussian
NLL values that make total_loss appear to spike with no explainable source.

**Fix:** Added `uncertainty_loss`, `ownership_loss`, and `threat_loss` to the
trainer's `log.info("train_step", ...)` call.  Every term that feeds into total_loss
is now individually visible.  Residual delta of ~0.018 between `total_loss` and the
sum of the four main components is the entropy regularisation (`0.01 × policy_entropy`),
which is already logged as `policy_entropy`.

**Commits:**
- `e54622a fix(train): initialise pretrained data stream on resume using restored step count`
- `e54622a` also covers Bug 2 (threshold change in same code section)
- `19fda11 fix(train): identify and fix hidden loss term causing total_loss spikes`

**Verification (smoke run confirmed):**
- `pretrained_weight: 0.8` at step 10 of fresh run; on resume it correctly decays from
  restored step via `0.8 * exp(-step / 300_000)`.
- `corpus_prefill_running` fires when restored_positions < 10 K.
- `uncertainty_loss`, `ownership_loss`, `threat_loss` present in every train_step JSONL event.

**Test counts:** 86 Rust + 616 Python unit tests, all passing.

---

### §59 — TT memory leak, uncertainty loss divergence, Ctrl+C shutdown crash (2026-04-06)

Three bugs discovered during a sustained self-play run (step ~1,640, 800+ games, 500 min).

**Bug 1 — Transposition table memory leak (28 GB RSS)**

`MCTSTree::new_game()` reset the node pool (`next_free = 1`) but never called
`self.transposition_table.clear()`. Each `TTEntry` holds a heap-allocated
`Vec<f32>` (362-element policy). With 12 workers playing games indefinitely, the
TT accumulated every MCTS-visited position across all games without release.

Estimated scale: 12 workers × ~67 games each × ~100 moves × ~300 unique positions/move
× ~1.5 KB/entry ≈ tens of GB. Observed: 28 GB RSS after 500 min.

**Fix:** Added `self.transposition_table.clear();` to `new_game()` in
`engine/src/mcts/mod.rs`. The TT is per-game by nature — cross-game position reuse
is negligible and not worth the unbounded memory cost.

**Bug 2 — Gaussian NLL divergence (total_loss spikes to ~394)**

`uncertainty_weight: 0.05` was active. `compute_uncertainty_loss` uses
Gaussian NLL: `0.5 * (log σ² + (z−v)²/σ²)`. When σ² collapses toward the 1e-6
clamp floor, the `(z−v)²/σ²` term is unbounded. Observed: `uncertainty_loss: 7812`
recurring every ~50 steps → `0.05 × 7812 ≈ 390` added to total_loss, spiking it
to ~394 while policy/value/aux remained normal (~5.5 total).

The FP16 underflow guard added in §49 clamps σ² from below (prevents NaN) but
does not prevent the reciprocal explosion. The fix in §49 was necessary but not
sufficient for long runs where σ² can genuinely converge toward zero.

**Fix:** Set `uncertainty_weight: 0.0` in `configs/training.yaml`. The uncertainty
head is diagnostic-only and has never been used in MCTS. Disabled until a
proper σ² regularisation strategy (e.g. log-barrier or β-VAE-style KL term) is designed.

**Bug 3 — "I/O operation on closed file" crash on Ctrl+C**

The `finally` block in `scripts/train.py` closed `_log_fh` before the
post-finally session-end code ran (`trainer.save_checkpoint()` → `log.info()`).
One Ctrl+C triggered a clean shutdown that hit this ordering, raising
`ValueError: I/O operation on closed file` instead of saving gracefully.

**Fix:** Moved `_log_fh.close()` to after all session-end logging, at the very end
of `main()`.

**Commit:** `452397f fix(train): three shutdown/training bugs — TT leak, uncertainty spikes, closed-log crash`

**Test counts:** 86 Rust + 616 Python unit tests, all passing.
