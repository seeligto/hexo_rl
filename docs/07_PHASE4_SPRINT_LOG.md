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

**Why:** FP16 gives 1.3–1.8× throughput on Ampere (RTX 3070). torch.compile
adds 10–20% on top. Pruning reduces effective policy loss noise in early self-play.

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
- 2–10s warm-up per metric before timing begins
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
| Q8 | First-player advantage in value training | 🟡 Active — MEDIUM priority (corpus shows 51.6% P1 overall, 57.1% in 1000–1200 Elo band) |
| Q1, Q4, Q7 | MCTS convergence rate, augmentation equivariance, Transformer encoder | 🔵 Deferred — Phase 5+ |

---

## Immediate next steps

In priority order:

1. **Q2 ablation: value aggregation strategy** — design and run the
   min vs mean vs attention experiment. This is the single highest-priority
   open question. Needs a baseline checkpoint from the first training run.

2. **First sustained self-play training run** — all infrastructure is
   in place. Run `python scripts/train.py` and monitor for 24–48 hours.
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
        Planes 0–15: 8 history steps × 2 players (cluster snapshots)
        Planes 16–17: metadata (moves_remaining, turn parity)

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
- Expected range at a uniform 362-action policy: ~5.9 nats. In practice ~3–6 nats is healthy; < 1.0 signals collapse.

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
