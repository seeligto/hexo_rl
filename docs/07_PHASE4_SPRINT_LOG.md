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
- CPU frequency pinning attempted via cpupower (graceful fallback —
  cpupower unavailable on omarchy Linux; results marked [UNCONTROLLED])
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

- `hexo_rl/monitoring/terminal_dashboard.py` — rich renderer (reads events)
- `hexo_rl/monitoring/web_dashboard.py` — Flask+SocketIO renderer
- `hexo_rl/monitoring/static/index.html` — single-file SPA
- Wire `GPUMonitor` to emit `system_stats` events via `emit_event()`
- Add `monitoring:` config block to `configs/default.yaml`
- Compute `value_accuracy` in `trainer.py` and include in loss_info
- Compute `grad_norm` in `trainer.py` (before clip) and include in loss_info
- Include `lr` from scheduler in loss_info
- Add per-worker ID to `game_complete` events (requires Rust change)

### Notes

- `moves_list` is populated from Rust `drain_game_results()` which returns
  `Vec<(i32, i32)>` — formatted as `"(q,r)"` strings in Python.
- `worker_id` is hardcoded to 0 for now — Rust SelfPlayRunner doesn't expose
  per-game worker identification yet.
- `push_corpus_preview.py` still references the old dashboard HTTP API for
  signaling reloads. It will silently fail (fire-and-forget). Should be
  updated when the web dashboard renderer is built.
