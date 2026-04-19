# Dashboard Specification — HeXO Training Monitor
# docs/08_DASHBOARD_SPEC.md

**Status:** Authoritative spec. Supersedes version dated 2026-04-03.
**Scope:** Training monitor. Game viewer is separate — see `docs/09_VIEWER_SPEC.md`.
**Last updated:** 2026-04-07

---

## 1. Architecture overview

Unchanged from prior spec. Two renderers, one event schema. The training loop
(`scripts/train.py`) emits structured events via `emit_event()`. Both renderers
consume the same events independently — they share no code other than the schema
defined in section 2.

```
scripts/train.py
    │
    ├── hexo_rl/monitoring/events.py   ← emit_event() — single call site
    │       │
    │       ├── hexo_rl/monitoring/terminal_dashboard.py   ← rich renderer
    │       └── hexo_rl/monitoring/web_dashboard.py        ← Flask+SocketIO renderer
    │               │
    │               └── hexo_rl/monitoring/static/         ← single HTML+JS page
```

**Key constraint:** Both renderers are **passive observers**. They must never
block the training loop, never raise exceptions that propagate to train.py, and
never write to the replay buffer or any training state. Failures are logged and
silently swallowed.

---

## 2. Event schema

All events are Python dicts with a mandatory `event` string key and `ts` float
(Unix timestamp, always set by `emit_event()`). Renderers pattern-match on `event`.

### 2.1 `training_step`

Emitted every `config.monitoring.log_interval` training steps (default: 10).
**§47 addition: `policy_target_entropy`.**

```python
{
    "event":                   "training_step",
    "ts":                      float,    # unix timestamp — set by emit_event()
    "step":                    int,      # global training step
    "loss_total":              float,
    "loss_policy":             float,
    "loss_value":              float,
    "loss_aux":                float,
    "policy_entropy":          float,    # mean entropy of policy distribution (nats)
    "policy_target_entropy":   float,    # NEW §47 — mean entropy of MCTS policy *target*
                                         # over the batch, post-pruning+renorm, nats.
                                         # Computed only over non-zero-policy rows.
                                         # 0.0 if unavailable. Must not be NaN.
    # §101 D-Gumbel / D-Zeroloss split — mean policy-target entropy and
    # KL(target || uniform) bucketed by is_full_search. NaN when the bucket
    # is empty; renderers must treat NaN as "no data this step" and never
    # propagate. y-axis range [0, log(num_actions) ≈ 5.89]. frac in [0, 1].
    "policy_target_entropy_fullsearch":    float,  # mean H over full-search rows (nats)
    "policy_target_entropy_fastsearch":    float,  # mean H over quick-search rows (nats)
    "policy_target_kl_uniform_fullsearch": float,  # mean KL(tgt || uniform), full (nats)
    "policy_target_kl_uniform_fastsearch": float,  # mean KL(tgt || uniform), fast (nats)
    "frac_fullsearch_in_batch":            float,  # n_full / B, in [0, 1]
    "n_rows_policy_loss":                  int,    # rows that contributed to policy gradient
    "n_rows_total":                        int,    # rows with valid policy target
    "value_accuracy":          float,    # fraction: value head correctly predicted winner
    "lr":                      float,    # current learning rate
    "grad_norm":               float,    # gradient norm before clipping (may be inf on
                                         # FP16 GradScaler overflow — never NaN)
    "phase":                   str,      # "pretrain" or "self_play"
}
```

**§101 usage.** `n_rows_policy_loss == 0` ⇒ the selective gate dropped every
row this step (zero-gradient on the policy head). `n_rows_policy_loss > 0`
with `loss_policy == 0.0` ⇒ genuine zero loss on surviving rows. This
disambiguates the "zero loss" case that §100 flagged as a known follow-up.
`H_fast(CQ) ≈ H_full(CQ)` is the §101 D-Gumbel signal that quick-search
completed-Q targets carry usable gradient.

### 2.2 `iteration_complete`

**§47 addition: `mcts_mean_depth`, `mcts_root_concentration`.**

```python
{
    "event":                    "iteration_complete",
    "ts":                       float,
    "step":                     int,
    "games_total":              int,
    "games_this_iter":          int,
    "games_per_hour":           float,
    "positions_per_hour":       float,
    "avg_game_length":          float,
    "win_rate_p0":              float,
    "win_rate_p1":              float,
    "draw_rate":                float,
    "sims_per_sec":             float,
    "buffer_size":              int,
    "buffer_capacity":          int,
    "corpus_selfplay_frac":     float,
    "batch_fill_pct":           float,   # avg inference batch fill % this iteration.
                                         # 0.0 if not available.
    "mcts_mean_depth":          float,   # NEW §47 — mean leaf depth per search since
                                         # SelfPlayRunner.start() (run-wide rolling mean,
                                         # not reset per iteration). 0.0 if unavailable.
    "mcts_root_concentration":  float,   # NEW §47 — mean of (max_child_visits/total)
                                         # at root since SelfPlayRunner.start() (run-wide).
                                         # Range [0.0, 1.0]. 0.0 if unavailable.
    "cluster_value_std_mean":   float,   # NEW §107 (I2) — lifetime mean per-position
                                         # std-dev of per-cluster values (before min-pool).
                                         # Only K≥2 positions contribute. 0.0 if unavailable.
    "cluster_policy_disagreement_mean": float,  # NEW §107 (I2) — lifetime mean of
                                         # (1 − top1-majority-count/K) across K≥2 positions.
                                         # Range [0.0, 1.0]. 0.0 = all windows agree.
    "cluster_variance_sample_count":    int,    # NEW §107 (I2) — count of K≥2 positions
                                         # scored; divisor for the two means above.
}
```

**How to compute:** `SelfPlayRunner` accumulates `depth_accum` and `conc_accum` via
`MCTSTree.last_search_stats()` called once per move (not inside the sim loop).
`mcts_mean_depth` = `depth_accum / sim_count`; `mcts_root_concentration` =
`conc_accum / sim_count`. Both exposed as Python properties on `SelfPlayRunner`.

### 2.3 `game_complete`

No schema changes.

```python
{
    "event":       "game_complete",
    "ts":          float,
    "game_id":     str,
    "winner":      int,       # 0 = P0, 1 = P1, -1 = draw
    "moves":       int,
    "moves_list":  list[str],
    "worker_id":   int,
    # §107 I1 — colony-extension detector fields.  Count/total of stones placed
    # at hex-distance > 6 from any opponent stone at game end.  Zero when
    # monitoring.log_investigation_metrics is disabled.
    "colony_extension_stone_count": int,
    "colony_extension_stone_total": int,
    "colony_extension_fraction":    float,   # count / total (0.0 if total == 0)
}
```

### 2.4 `eval_complete`

No schema changes.

```python
{
    "event":               "eval_complete",
    "ts":                  float,
    "step":                int,
    "elo_estimate":        float | None,
    "win_rate_vs_sealbot": float,
    "eval_games":          int,
    "gate_passed":         bool,
}
```

### 2.5 `system_stats`

**Changed: added `ram_used_gb`, `ram_total_gb`, `cpu_util_pct`, `batch_fill_pct`, `rss_gb`.**

Emitted every 5 seconds by `gpu_monitor.py`.

```python
{
    "event":          "system_stats",
    "ts":             float,
    "gpu_util_pct":   float,
    "vram_used_gb":   float,
    "vram_total_gb":  float,
    "workers_active": int,
    "workers_total":  int,
    "ram_used_gb":    float,           # psutil.virtual_memory().used / 1e9
    "ram_total_gb":   float,           # psutil.virtual_memory().total / 1e9
    "cpu_util_pct":   float,           # psutil.cpu_percent(interval=None), aggregate
    "rss_gb":         float,           # NEW — psutil.Process().memory_info().rss / 1e9
}
```

**Implementation notes for `gpu_monitor.py`:**
- `psutil` is already a transitive dependency. Import it at the top of the module.
- `_PROCESS = psutil.Process()` is created once at module level (not per poll cycle).
- `psutil.cpu_percent(interval=None)` returns the percent since the last call
  (non-blocking). Call it on every 5s poll cycle — do not use `interval=5`
  (that would block the monitor thread).
- RAM, CPU, and RSS fields must never raise. Wrap in try/except; emit 0.0 on failure.

### 2.6 `run_start` / `run_end`

No schema changes.

```python
{
    "event":          "run_start",  # or "run_end"
    "ts":             float,
    "step":           int,
    "run_id":         str,
    "config_summary": dict,
}
```

---

## 3. Emitter — `hexo_rl/monitoring/events.py`

No changes to `events.py` from prior spec. The module and its API are stable.

---

## 4. Terminal dashboard — `hexo_rl/monitoring/terminal_dashboard.py`

### 4.1 Layout

```
┌─────────────────────────────────────────────────────────────────────┐
│ HeXO training · phase 4.0 · run abc123 · step 53,260               │
├──────────┬──────────┬──────────┬──────────┬──────────┬─────────────┤
│ loss     │ policy   │ value    │ aux      │ entropy  │ lr          │
│ 2.7494   │ 1.38     │ 0.52     │ 0.11     │ 1.85 ▲  │ 1.87e-3    │
├──────────┴──────────┴──────────┴──────────┴──────────┴─────────────┤
│ games/hr  332  │  pos/hr  13K  │  sims/sec  7K  │  batch fill  98% │
│ avg len   48   │  P0 54.5%  P1 43.9%  draw 1.5%  │  grad  0.42    │
├─────────────────────────────────────────────────────────────────────┤
│ buffer  7,755 / 250,000 (3%)  │  sp 24%  pre 76%                   │
│ gpu 99%  │  vram 5.5/8.6 GB  │  ram 32.1/48.0 GB  │  rss  4.2 GB  │  cpu 87%      │
└─────────────────────────────────────────────────────────────────────┘
```

Changes from prior spec:
- Added `grad_norm` to the throughput row (abbreviated `grad`)
- Added `batch_fill_pct` to the throughput row (abbreviated `batch fill`)
- Added `ram`, `rss`, and `cpu` to the system row
- `▲` next to entropy when entropy < `alert_entropy_warn` (2.0); `!!` when < `alert_entropy_min` (1.0)
- **§47:** Value loss cell shows inline ratio: `0.6084 (×0.88)` where ratio = `loss_value / ln(2)`
- **§47:** Entropy cell shows inline % of max: `2.35 (40% max)` where pct = `entropy / log(num_actions) * 100`
- **§47:** Throughput row second line appends `│  MCTS depth  N.N  │  root concen  0.NN` before `grad`
- `—` for any field not yet received.

All other rules unchanged: no bars for open-ended metrics, `—` when not yet received,
4 Hz max refresh.

### 4.2 Alert line

Conditions unchanged. Thresholds now configurable — see section 7.

---

## 5. Web dashboard — `hexo_rl/monitoring/web_dashboard.py`

### 5.1 Server

No changes. Flask + Flask-SocketIO on `localhost:5001`.

### 5.2 Stat cards

**Changed from prior spec.** Six cards across the top:

| Card | Source field | Alert behaviour |
|---|---|---|
| `step` | `training_step.step` | — |
| `total loss` | `training_step.loss_total` | — |
| `policy entropy` | `training_step.policy_entropy` | amber when < 2.0; red when < 1.0 |
| `value accuracy` | `training_step.value_accuracy` | — |

**§47 subtitle lines:** Two stat cards gain a small dim subtitle beneath the main value:
- **Policy Entropy card:** `{pct:.0f}% of max` where `pct = entropy / log(num_actions) * 100`
- **Value Accuracy card:** `loss × {ratio:.2f} of random` where `ratio = loss_value / 0.6931`

Show `—` before any `training_step` received.
| `pos / hr` | `iteration_complete.positions_per_hour` | green tint |
| `games / hr` | `iteration_complete.games_per_hour` | — |

**Removed from top row (prior spec):** `gpu util` (now in system panel only),
`elo vs sealbot` (now in right-column panel, shown only when eval data exists).

**Entropy card alert states:**
- `> 2.0`: normal colour
- `1.0-2.0`: amber text + small amber badge reading "low"
- `< 1.0`: red text + red badge reading "collapse"

### 5.3 Main row layout (2/3 + 1/3 split)

**Left — Loss chart (updated):**

Line chart: policy, value, aux, total. Last 500 `training_step` events.

Two toggle buttons in the panel header:
- `raw` — show faint (20% opacity) raw loss lines
- `EMA` — show solid EMA-smoothed lines (α = `monitoring.ema_alpha`, default 0.06)

Both toggles active by default. Clicking a toggle shows/hides its dataset group.
EMA is computed client-side from the raw ring buffer on each update — do not
emit pre-computed EMA from the server.

Legend below chart: coloured swatches for total / policy / value / aux.

**§47 ratio strip:** A single dim text line below the chart legend, updated on every
`training_step` event:
```
value × 0.65 of random  │  entropy 58% of max  │  policy excess +0.32
```
- `value` ratio = `loss_value / 0.6931`
- `entropy` pct = `policy_entropy / log(num_actions) * 100`
- `policy excess` = `loss_policy − policy_target_entropy` (positive = loss > target)
- Show `—` for any missing field.

Pretrain region: if `training_step.phase == "pretrain"`, shade that x-axis region
in a faint background colour and draw a dashed vertical line at step 0 (the
pretrain→RL transition). This is carried over from the prior implementation.

**Right column — three stacked panels (unchanged positions, updated content):**

Panel 1 — Buffer:
- `{buffer_size:,} / {buffer_capacity:,}` as header numbers
- Horizontal fill bar (capacity %)
- Corpus mix: two percentage rows (self-play / pretrain) with a segmented bar

Panel 2 — ELO (conditional):
- Hidden entirely until at least one `eval_complete` has been received.
- When visible: latest `elo_estimate`, step annotation, gate badge (PASSED/FAILED).
- If 3+ eval_complete events received: small line chart of ELO history.
- When not visible: no placeholder, no "—" card. The buffer and system panels
  expand to fill the space.

Panel 3 — System stats:
```
GPU util      99%
VRAM          5.5 / 8.6 GB
RAM           32.1 / 48.0 GB      ← NEW (§5 revision)
RSS           4.2 GB               ← NEW (§45 — process RSS for OOM diagnosis)
CPU           87%                  ← NEW (§5 revision)
Workers       12
Sims/sec      7K
Batch fill    98%                  ← NEW (from iteration_complete.batch_fill_pct)
MCTS depth    14.2               ← NEW §47 (from iteration_complete.mcts_mean_depth)
Root concen   0.42               ← NEW §47 (from iteration_complete.mcts_root_concentration)
Grad norm     0.42
LR            1.87e-3
```

`Grad norm` and `LR` are sourced from `training_step` events, not `system_stats`.
The panel merges the latest values from both event types. All fields show `—`
until their source event is received.

### 5.4 Bottom row — four panels (changed from three)

**Panel 1 — P0 win rate (rolling)**

Replaces the prior stacked bar chart entirely.

- Line chart: P0 win rate (%) over time. One data point per `game_complete`
  event, computed as a rolling window of the last 200 games.
- Y-axis: 0-100%, but displayed range auto-zooms to data ± 15pp.
- Target band: filled region between `alert_p0_target_low` (54%) and
  `alert_p0_target_high` (58%) — faint teal fill, dashed boundary lines.
  This is the Q8 target range for P0 advantage correction.
- Below chart: inline summary `P0 54.5%  P1 43.9%  draw 1.5%`
- Source: `game_complete` events (individual games, not `iteration_complete`
  aggregates — this gives higher resolution and avoids batch-size aliasing).

**Why this replaces stacked bars:** The stacked bar chart showed per-iteration
P0/P1/draw fractions which are too granular to read trends from. The rolling
line makes drift visible (Q8 concern: P0 advantage persisting above target).

**Panel 2 — Game length histogram**

Replaces the prior rolling average line chart.

- Bar chart (histogram): game length distribution over the last 200 `game_complete`
  events. Bins: 0-20, 20-40, 40-60, 60-80, 80-100, 100-120, 120-140, 140-160,
  160-180, 180-200+. The 200+ bin captures draws (capped at `max_game_moves`).
- X-axis: bin labels. Y-axis: game count.
- Below chart: `median {N}  draws {D} / 200`
- Update: rebuild histogram on every `game_complete` event (cheap client-side).

**Why histogram over rolling average:** The average hides distribution shape.
A rising average that masks a bimodal distribution (many very short games +
many draws) would be a corpus quality problem invisible in a line chart.

**Panel 3 — Policy entropy trend**

New chart (previously just a number in the system panel).

- Line chart: `training_step.policy_entropy` over last 500 steps.
- Horizontal dashed red line at `alert_entropy_min` (1.0) labelled "collapse".
- Y-axis: 0-max(data)+0.5, minimum range 0-4.
- Colour: amber (#EF9F27).
- Below chart: `now {entropy:.2f}  collapse at {alert_entropy_min}`

**Panel 4 — Gradient norm trend**

New chart (previously just a number in the system panel).

- Line chart: `training_step.grad_norm` over last 500 steps.
- Horizontal dashed red line at `alert_grad_norm_max` (10.0) labelled "clip".
- Y-axis: 0-max(data)+1, minimum range 0-5.
- Colour: coral (#D85A30).
- `inf` values (FP16 GradScaler overflow skips) are rendered as gaps in the
  line — do not connect across `inf` points. Use Chart.js `spanGaps: false`.
- Below chart: `now {grad_norm:.2f}  clip at {alert_grad_norm_max}`

### 5.5 Event log

No layout change. Last 20 events, newest first. `view →` links unchanged.

### 5.6 Chart data retention

Updated:

| Chart | Ring buffer | Source event | Notes |
|---|---|---|---|
| Loss curves (all 4 lines) | last **2000** events | `training_step` | EMA computed client-side |
| Policy entropy trend | last **2000** events | `training_step` | shared ring buffer with loss |
| Grad norm trend | last **2000** events | `training_step` | shared ring buffer with loss |
| P0 win rate line | last **500** games | `game_complete` | rolling window |
| Game length histogram | last **500** games | `game_complete` | shared ring with win rate |
| System stats | latest values | `system_stats` + `training_step` | no history |

**§47 change:** `trainingStepHistory` capacity bumped from 500 → **2000**
(`monitoring.training_step_history`). Game ring bumped from 200 → **500**
(`monitoring.game_history`). Both values are server-injected via `/api/monitoring-config`
and read by `fetch()` on page load — not hardcoded in JS.

**Implementation note:** the 4 `training_step`-sourced charts (loss×4, entropy,
grad norm) all read from a single `trainingStepHistory` ring buffer. Do not
create separate ring buffers per chart — slice the same array.

### 5.7 SocketIO event names

No changes from prior spec. Server → client only.

---

## 6. Deferred items

No change from prior spec. Historical run comparison, Discord alerts, and
mobile layout remain deferred to Phase 5+.

---

## 7. Config keys

Updated. All monitoring config lives under `monitoring:` in `configs/monitoring.yaml`.

```yaml
monitoring:
  enabled: true
  terminal_dashboard: true
  web_dashboard: true
  web_port: 5001
  log_interval: 10               # emit training_step every N steps
  event_log_maxlen: 500          # in-memory event replay buffer (non-game events + stripped game refs)
  viewer_max_memory_games: 50    # max game refs held in _game_index; full records written to disk

  # Alert thresholds
  alert_entropy_min: 1.0         # RED — collapse imminent
  alert_entropy_warn: 2.0        # AMBER — entropy degrading (NEW)
  alert_grad_norm_max: 10.0      # RED — instability

  # Chart config
  ema_alpha: 0.06                # EMA smoothing factor for loss curves
  win_rate_window: 200           # rolling window size for P0 win rate line
  game_length_window: 200        # histogram window (same games as win rate)
  training_step_history: 2000    # §47 — bumped from 500; ring buffer for step-sourced charts
  game_history: 500              # §47 NEW — replaces win_rate_window/game_length_window in JS
  num_actions_for_entropy_norm: 362  # §47 NEW — board_size^2 + 1 for entropy % display

  # Target bands (used in P0 win rate chart)
  p0_win_rate_target_low: 54.0   # lower bound of target band (%)
  p0_win_rate_target_high: 58.0  # upper bound of target band (%)

  # §101 — gate for the 7 policy-target-quality metrics in `training_step`.
  # Default true. Disable via monitoring.log_policy_target_metrics: false for
  # benchmark harnesses where even the ~200 µs/call is unwanted.
  log_policy_target_metrics: true
```

**§47 notes:**
- `game_history` replaces the separate `win_rate_window` / `game_length_window` JS constants.
  Both now read `MAX_GAMES` from the `/api/monitoring-config` endpoint.
- `num_actions_for_entropy_norm` is passed through to the terminal renderer and web client
  for computing `% of max` annotations. Derive from `model.board_size^2 + 1` at startup
  if available rather than hardcoding.
- All config values are served to JS via the `/api/monitoring-config` endpoint
  added to `web_dashboard.py`.

---

## 8. Implementation files

No new files required. All changes are to existing files:

```
hexo_rl/monitoring/
├── events.py                  ← no changes
├── gpu_monitor.py             ← add ram_used_gb, ram_total_gb, cpu_util_pct, rss_gb
├── terminal_dashboard.py      ← add grad_norm, batch_fill_pct, ram, rss, cpu to layout
├── web_dashboard.py           ← no changes (event forwarding unchanged)
└── static/
    └── index.html             ← major rewrite — new layout per §5

hexo_rl/selfplay/
└── pool.py                    ← add batch_fill_pct to iteration_complete payload

configs/
└── monitoring.yaml            ← add new keys per §7
```

---

## 9. Testing requirements

All changes must have tests. Extend existing `tests/test_dashboard_events.py`
and `tests/test_dashboard_renderers.py`. Do not create additional test files
for this sprint — keep dashboard tests consolidated.

**New required test cases (add to existing files):**

Schema validation:
1. `system_stats` event contains `ram_used_gb`, `ram_total_gb`, `cpu_util_pct`, `rss_gb`
2. `iteration_complete` event contains `batch_fill_pct`
3. `system_stats.cpu_util_pct` is a float in [0.0, 100.0]
4. `system_stats.ram_used_gb` ≤ `system_stats.ram_total_gb`
5. `system_stats.rss_gb` is a positive float (> 0.0 when psutil is available)

gpu_monitor:
6. `GPUMonitor` emits `system_stats` with all new fields (incl. `rss_gb`) when psutil is available
7. `GPUMonitor` emits 0.0 for `rss_gb` (not exception) when `memory_info()` raises

terminal_dashboard:
7. Terminal renderer handles `system_stats` with new fields without error
8. Terminal renderer renders `—` for `grad_norm` and `batch_fill_pct` before
   any `training_step` / `iteration_complete` received
9. Entropy `▲` marker appears when entropy < `alert_entropy_warn` (2.0)
10. Entropy `!!` marker appears when entropy < `alert_entropy_min` (1.0)

pool.py:
11. `iteration_complete` payload from pool includes `batch_fill_pct` key
12. `batch_fill_pct` is 0.0 when batcher has no data (not omitted, not None)

**§47 new test cases:**
Schema (events):
13. `training_step` event contains `policy_target_entropy` (float, ≥ 0.0)
14. `iteration_complete` event contains `mcts_mean_depth` (float, ≥ 0.0)
15. `iteration_complete` event contains `mcts_root_concentration` (float, ∈ [0.0, 1.0])
16. All three new fields default to 0.0 (not None, not NaN, not omitted) when source unavailable

Trainer:
17. `policy_target_entropy` is computed only over non-zero-policy rows
18. `policy_target_entropy` is finite when target is one-hot (entropy = 0, not -inf)

Engine / MCTS:
19. `mcts_mean_depth` is > 0 after at least one MCTS search
20. `mcts_root_concentration` is in [0.0, 1.0] after at least one MCTS search
21. `last_search_stats()` mean_depth is not unreasonably large (guard against loop-level accumulation)

Renderer (terminal):
22. Terminal renderer displays `(×N.NN)` next to value loss when received; renders without error before events
23. Terminal renderer displays MCTS depth and root concentration values after `iteration_complete`
24. Terminal renderer entropy display shows `(NN% max)` annotation

**Existing tests that may break and need updating:**
- Any test that validates the exact field set of `system_stats` — update to
  include the 3 new fields
- Any test that validates the exact field set of `iteration_complete` — update
  to include `batch_fill_pct`
- Any snapshot/equality test on terminal dashboard output format

---

## 10. What is NOT changing

To be explicit about scope:

- `events.py` — no changes
- `web_dashboard.py` — no changes (Flask routes, SocketIO emit, event replay
  buffer all unchanged; the layout change is entirely in `index.html`)
- `train.py` — no changes (emit_event calls are already correct)
- `trainer.py` — no changes (grad_norm, lr, entropy already in training_step)
- `09_VIEWER_SPEC.md` and `viewer.html` — entirely out of scope

The web dashboard layout change (§5) is implemented entirely in `index.html`.
The Flask server is not aware of the layout — it just forwards events.

---

## 11. Changelog

| Date | Change |
|---|---|
| 2026-04-03 | Initial implementation — event fan-out, terminal + web renderer |
| 2026-04-04 | system_stats + 3 new fields; iteration_complete + batch_fill_pct; stat card redesign; loss chart EMA toggle; bottom row → 4 panels (P0 win rate line, game length histogram, entropy trend, grad norm trend); ELO panel made conditional; system panel expanded with RAM/CPU/batch-fill/grad/LR |
| 2026-04-05 | **§45** — add `rss_gb` (process RSS) to `system_stats` event and system panels (terminal + web). Needed for OOM post-mortem — overnight run OOMed with no RSS history |
| 2026-04-07 | **§47** — meaningful-ratios pass: `training_step` now emits `policy_target_entropy`; `iteration_complete` now emits `mcts_mean_depth` and `mcts_root_concentration`. Stat cards and loss chart show normalized ratios inline (no toggles). System panel adds MCTS depth and root concentration rows. Ring buffers bumped to 2000 steps / 500 games. Config-driven via `/api/monitoring-config`. 12 new tests added. |
| 2026-04-18 | **§101** — D-Gumbel / D-Zeroloss instrumentation. `training_step` now emits 7 new keys: `policy_target_entropy_{full,fast}search`, `policy_target_kl_uniform_{full,fast}search`, `frac_fullsearch_in_batch`, `n_rows_policy_loss`, `n_rows_total`. NaN is a first-class signal (empty subset). Terminal dashboard adds one policy-target line; web dashboard extends the loss ratio strip. Gated by `monitoring.log_policy_target_metrics` (default true, <0.2% step cost). New `tests/test_policy_target_metrics.py` exercises the math on synthetic batches. |
