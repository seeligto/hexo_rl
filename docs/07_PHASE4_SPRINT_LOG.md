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
