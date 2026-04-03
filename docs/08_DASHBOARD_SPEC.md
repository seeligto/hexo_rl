# Dashboard Specification — HeXO Training Monitor
# docs/08_DASHBOARD_SPEC.md

**Status:** Authoritative spec. Implementation complete as of 2026-04-03.
**Scope:** Training monitor. Game viewer is implemented separately — see `docs/09_VIEWER_SPEC.md`.
**Last updated:** 2026-04-03

---

## 1. Architecture overview

Two renderers, one event schema. The training loop (`scripts/train.py`) emits
structured events. Both renderers consume the same events independently — they
share no code other than the schema defined in section 2.

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

**Previous files being replaced:**
- `dashboard.py` (root level) — delete entirely
- Any `phase40dashboard*` or `phase4_dashboard*` files in `hexo_rl/monitoring/`
- All `web_dash.*` or `WebDashboard` references in `train.py` and `pool.py`

---

## 2. Event schema

All events are Python dicts with a mandatory `event` string key and `ts` float
(Unix timestamp, always set by `emit_event()`). Renderers pattern-match on `event`.

### 2.1 `training_step`

Emitted every `config.monitoring.log_interval` training steps (default: 10).

```python
{
    "event":           "training_step",
    "ts":              float,          # unix timestamp — set by emit_event()
    "step":            int,            # global training step
    "loss_total":      float,
    "loss_policy":     float,
    "loss_value":      float,
    "loss_aux":        float,
    "policy_entropy":  float,          # mean entropy of policy distribution (nats)
    "value_accuracy":  float,          # fraction: value head correctly predicted winner at move-20
    "lr":              float,          # current learning rate (for LR schedule visibility)
    "grad_norm":       float,          # gradient norm before clipping (NaN if not computed)
}
```

**Why policy_entropy:** Collapsing entropy before loss plateau is an early
warning of mode collapse. Log it always; alert if it drops below 1.0.

**Why value_accuracy:** Raw value loss is hard to interpret. % of games where
the value head at move 20 correctly predicted the winner is a human-readable
proxy. Compute over the last N games in the buffer (N=200 or whatever fits).

**Why grad_norm:** Spikes indicate instability. Clip threshold is in config;
norm before clip is the useful diagnostic.

### 2.2 `iteration_complete`

Emitted once per training iteration (after each batch of self-play games is
collected and a training step runs). This is the primary throughput event.

```python
{
    "event":                  "iteration_complete",
    "ts":                     float,
    "step":                   int,
    "games_total":            int,     # cumulative games played since run start
    "games_this_iter":        int,     # games in this iteration
    "games_per_hour":         float,
    "positions_per_hour":     float,   # total positions generated/hr
    "avg_game_length":        float,   # mean move count this iteration
    "win_rate_p0":            float,   # fraction [0,1]
    "win_rate_p1":            float,
    "draw_rate":              float,
    "sims_per_sec":           float,   # aggregate across all workers
    "buffer_size":            int,
    "buffer_capacity":        int,
    "corpus_selfplay_frac":   float,   # fraction of buffer that is self-play [0,1]
}
```

### 2.3 `game_complete`

Emitted for every completed self-play game. The terminal renderer ignores this
(too high frequency). The web renderer samples 1-in-N for the event log
(N configurable, default 1 — show all, but log panel displays last 20).

```python
{
    "event":       "game_complete",
    "ts":          float,
    "game_id":     str,       # unique ID — use uuid4 hex
    "winner":      int,       # 0 = P0, 1 = P1, -1 = draw
    "moves":       int,       # total move count (plies)
    "moves_list":  list[str], # move sequence in axial notation e.g. ["(0,0)","(1,0)"…]
                              # KEEP THIS — /viewer will consume it later
    "worker_id":   int,
}
```

### 2.4 `eval_complete`

Emitted after a SealBot evaluation run finishes (asynchronous — may arrive
seconds or minutes after the training step that triggered it).

```python
{
    "event":               "eval_complete",
    "ts":                  float,
    "step":                int,           # step at which eval was triggered
    "elo_estimate":        float | None,  # Bradley-Terry derived ELO vs SealBot
    "win_rate_vs_sealbot": float,         # fraction [0,1] over eval_games games
    "eval_games":          int,           # number of games played
    "gate_passed":         bool,          # win_rate >= 0.55
}
```

### 2.5 `system_stats`

Emitted every 5 seconds by the GPU monitor thread (already exists in
`hexo_rl/monitoring/gpu_monitor.py` — wire it into emit_event).

```python
{
    "event":          "system_stats",
    "ts":             float,
    "gpu_util_pct":   float,
    "vram_used_gb":   float,
    "vram_total_gb":  float,
    "workers_active": int,
    "workers_total":  int,
}
```

### 2.6 `run_start` / `run_end`

Bookend events. Used by web dashboard to reset state on reconnect and by
the event log to show session boundaries.

```python
{
    "event":          "run_start",  # or "run_end"
    "ts":             float,
    "step":           int,          # starting step (from checkpoint or 0)
    "run_id":         str,          # uuid4 — stable for this run, use in log filenames
    "config_summary": dict,         # {"n_blocks": 12, "channels": 128, "n_sims": 800, …}
}
```

---

## 3. Emitter — `hexo_rl/monitoring/events.py`

Single module. All of train.py calls `emit_event(payload)` — never calls
renderers directly.

```python
# hexo_rl/monitoring/events.py

import time
import threading
from typing import Any

_renderers: list = []
_lock = threading.Lock()

def register_renderer(renderer) -> None:
    """Call once at startup for each active renderer."""
    with _lock:
        _renderers.append(renderer)

def emit_event(payload: dict[str, Any]) -> None:
    """
    Add ts, then dispatch to all registered renderers.
    Never raises — failures are caught and logged to stderr only.
    """
    payload = {"ts": time.time(), **payload}
    with _lock:
        targets = list(_renderers)
    for r in targets:
        try:
            r.on_event(payload)
        except Exception as exc:
            import sys
            print(f"[dashboard] renderer {r} failed: {exc}", file=sys.stderr)
```

Renderers implement one method: `on_event(self, payload: dict) -> None`.
The method must be non-blocking — if it needs to do I/O (e.g. SocketIO emit)
it must do so in a background thread or async queue.

---

## 4. Terminal dashboard — `hexo_rl/monitoring/terminal_dashboard.py`

### 4.1 Layout

A single `rich.live.Live` panel updated on `training_step` and
`iteration_complete` events. Does **not** respond to `game_complete` (too noisy).

```
┌─────────────────────────────────────────────────────────────┐
│ HeXO training · phase 4.0 · run abc123 · step 5,082        │
├──────────┬──────────┬──────────┬──────────┬──────────┬──────┤
│ loss     │ policy   │ value    │ aux      │ entropy  │ lr   │
│ 2.97     │ 2.13     │ 0.58     │ 0.26     │ 4.21     │3e-4  │
├──────────┴──────────┴──────────┴──────────┴──────────┴──────┤
│ games/hr  3,284  │  pos/hr  198K  │  sims/sec  189K        │
│ avg len   61     │  P0 51.4%  P1 48.5%  draw 0.1%          │
├─────────────────────────────────────────────────────────────┤
│ buffer  60,318 / 250,000  (24%)  │  sp 20%  pre 80%        │
│ ELO  —        │  gpu  89%  │  vram  0.8/8.6 GB            │
└─────────────────────────────────────────────────────────────┘
```

**Rules:**
- No progress bars with `/None` or `/0` totals — they are removed entirely.
  Step count is shown as a plain integer; game count is shown as a plain integer.
  Progress bars are only appropriate when there is a known finite target
  (e.g. a fixed pretrain epoch), never for open-ended self-play.
- No bars for corpus mix — show `sp 20% pre 80%` as plain text.
- No bars for GPU util — show `89%` as plain text.
- Numbers use `{:,}` formatting (thousands separators).
- Floats: loss to 4 dp, rates to 1 dp, percentages to 1 dp.
- `—` when a metric has never been received (never `None`, `null`, or `0`).
- Refresh rate: 4 Hz maximum (update on event, not on timer).

### 4.2 Alert line

A single line below the main panel, initially hidden. Shown in bold red when:
- `policy_entropy` < 1.0 (mode collapse warning)
- `grad_norm` > 10.0 (instability warning)
- `loss_total` increases for 3 consecutive `training_step` events
- `eval_complete.gate_passed` is False

Cleared automatically after 60 seconds or when the condition resolves.

---

## 5. Web dashboard — `hexo_rl/monitoring/web_dashboard.py`

### 5.1 Server

Flask + Flask-SocketIO. Runs on `localhost:5001` (configurable).
Non-blocking: started in a daemon thread from train.py, does not affect
training even if the browser is not open.

Static files served from `hexo_rl/monitoring/static/`:
- `index.html` — single-page app, all JS inline (no build step)
- No external CDN dependencies during training (offline-safe)

### 5.2 Panel layout (matches mockup)

Six stat cards across the top:
`step` | `total loss` | `games/hr` | `pos/hr` | `elo vs sealbot` | `gpu util`

Main row (2/3 + 1/3 split):
- Loss chart (policy, value, aux, total over last 500 steps)
- Buffer panel (size / capacity fill bar) + corpus mix (two numbers, no bar)

Secondary row (three equal columns):
- Win rates (P0 / P1 / draw) — rolling bar chart, last 100 games
- Avg game length — line chart, last 100 iterations
- System panel (gpu util %, vram GB, workers active, sims/sec, policy entropy, value accuracy)

Event log (full width):
- Last 20 events, newest first
- Shows: game_complete (winner, moves, worker), eval_complete, run_start/end
- Each game_complete row has a `view →` link to `/viewer/game/<game_id>`

### 5.3 Chart data retention

Client-side ring buffers (no server-side storage):

| Chart | Retention |
|---|---|
| Loss curves | last 500 `training_step` events |
| Win rate bar | last 100 games (from `game_complete`) |
| Game length | last 100 `iteration_complete` events |
| System stats | latest values only (no chart, just numbers) |

On browser reconnect, the server replays the last 500 events from an
in-memory deque (maxlen=500). This means a browser refresh restores the
last few minutes of charts without requiring persistence.

### 5.4 SocketIO event names

Server → client only. Client never sends events to server.

| SocketIO event | Source dashboard event | Notes |
|---|---|---|
| `training_step` | `training_step` | Forwarded as-is |
| `iteration_complete` | `iteration_complete` | Forwarded as-is |
| `game_complete` | `game_complete` | Forwarded as-is (moves_list included for future viewer) |
| `eval_complete` | `eval_complete` | Forwarded as-is |
| `system_stats` | `system_stats` | Forwarded as-is |
| `run_start` | `run_start` | Also triggers client state reset |
| `replay_history` | — | Sent on connect: last 500 events as array |

### 5.5 ELO panel behaviour

When no `eval_complete` has been received: show `—`, not `1000`, not `null`.
When received: show latest `elo_estimate` with step annotation.
ELO is plotted over time if 3+ eval_complete events have been received.
Gate status shown as a small badge: `PASSED` (green) or `FAILED` (red).

---

## 6. Deferred items

- Game replay viewer — **implemented** in separate sprint, see `docs/09_VIEWER_SPEC.md`
- Historical run comparison (separate tooling)
- Alerts via email/Discord webhook (post Phase-4.5)
- Mobile-responsive layout (desktop-only for now)

---

## 7. Config keys

All monitoring config lives under `monitoring:` in `configs/monitoring.yaml`.
Do not hardcode any of these values.

```yaml
monitoring:
  enabled: true
  terminal_dashboard: true
  web_dashboard: true
  web_port: 5001
  log_interval: 10           # emit training_step every N steps
  event_log_maxlen: 500      # in-memory event replay buffer
  alert_entropy_min: 1.0
  alert_grad_norm_max: 10.0
  alert_loss_increase_window: 3
```

---

## 8. Implementation files

```
hexo_rl/monitoring/
├── events.py                  ← emit_event, register_renderer
├── terminal_dashboard.py      ← Rich Live renderer
├── web_dashboard.py           ← Flask+SocketIO renderer
└── static/
    ├── index.html             ← dashboard SPA
    └── viewer.html            ← game viewer SPA (see 09_VIEWER_SPEC.md)
```

## 9. Completed cleanup (2026-04-03)

The following files/references were deleted as part of the dashboard rebuild:

| File | Action |
|---|---|
| `dashboard.py` (root) | Deleted |
| `phase40dashboard*`, `phase4_dashboard*` in `hexo_rl/monitoring/` | Deleted |
| Old `WebDashboard` class imports in `train.py` | Removed |
| Old `web_dash.*` calls in `train.py` and `pool.py` | Replaced with `emit_event()` calls |
| Progress bar code using `total=None` | Removed |

## 10. Testing requirements

All new monitoring code must have tests in `tests/test_dashboard.py`.

Required test cases:
1. `emit_event` dispatches to all registered renderers
2. `emit_event` does not raise when a renderer raises
3. `emit_event` adds `ts` key to every payload
4. Terminal dashboard `on_event` does not raise for any valid event type
5. Terminal dashboard `on_event` does not raise for unknown event types
6. Terminal dashboard renders without error when all optional fields are None/missing
7. Web dashboard `on_event` does not raise when no SocketIO client is connected
8. All 7 event types pass schema validation (required keys present, correct types)
9. Event replay buffer caps at `event_log_maxlen`

No test should require a running Flask server or a browser.
Use a mock SocketIO in web dashboard tests.
