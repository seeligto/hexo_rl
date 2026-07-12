# dash_rebuild_design.md — D-J DASH WP2 design

**Task:** D-J DASH — rebuild the monitor + dashboard design-first.
**Inputs:** WP1 teardown (`docs/audits/dash_teardown.md`) + operator rulings §8.
**GATE:** this doc + manifest approved before WP3 writes code.
**Companion artifact:** `hexo_rl/monitoring/dashboard_manifest.yaml` (the machine-readable
panel contract; `tests/test_event_contract.py` asserts every row has a live producer).

---

## 1. THE ONE LAW (enforced mechanically)

Every panel names its producer in `dashboard_manifest.yaml` (panel_id → channel → source →
field). The contract test extends to load the manifest and assert:
- `source_type: event` rows → the `event` is in the produced set for its **channel** (§1.1).
- `source_type: derived` rows → the `producer_path` resolver function exists on disk.
- `source_type: file` rows → the `producer_path` script that writes the series exists.
- `source_type: config` rows → the resolver key exists in the resolved_config schema.

Mutation check (WP3): a fake panel citing a nonexistent producer must make the test FAIL.
**A panel with no manifest row does not render.** This kills the silent-dead-instrument
class (root_concentration on Gumbel, the empty value_spread chart) at the UI layer.

### 1.1 TWO producer channels (load-bearing — WP1 missed this, discovered in WP2)

The monitor has TWO event streams with DIFFERENT schemas, and the two renderers read
DIFFERENT streams:

| Channel | Mechanism | JSONL file | Producer grep | Consumed by | Example names |
|---|---|---|---|---|---|
| `emit_event` | `emit_event({"event": "..."})` | `logs/events_<run_id>.jsonl` (JSONLSink) | `"event"\s*:\s*"NAME"` | **B (web)** | training_step, eval_complete, iteration_complete, game_complete |
| `structlog` | `log.info("NAME", **fields)` | `logs/<run_name>.jsonl` (configure.py) | `\.(info\|warning\|error)\(\s*"NAME"` | **A1 (TUI)** via run_feed_reader | train_step_summary, evaluation_round_complete, evaluation_games_complete, forced_win_trend |

**The current `test_event_contract._gather_produced()` scans ONLY the `emit_event` literal
pattern — A1's entire structlog channel is UNGUARDED today.** That is why the WP1 audit saw
A1 consuming `train_step_summary`/`forced_win_trend` while the produced set (emit_event only)
showed `training_step`/`eval_complete`. Both are real producers; different channels.

WP3's extended contract test gathers the produced set from **both** channels and checks each
manifest row against the produced set for its declared `channel`. This finally guards A1.
(No new instrumentation — both channels already emit at runtime; the test just learns to
see the second one.)

### 1.2 Contract-test rewrite (WP3 — do NOT just extend the old test)
The current `test_event_contract.py` derives the CONSUMED set from `terminal_dashboard.py`
(retired) + `index.html` (rebuilt) — so `test_consumed_set_size >= 8` **breaks the moment
A2/index.html are deleted** (NB-6). WP3 must port the CONSUMED side to the manifest:
consumed = the set of `event`s referenced across all panel bindings. Then:
- **Produced (structlog):** whole-file `re.finditer` (NOT line-based — `train_step_summary`
  is a multiline call, `events.py:336`; NB-4). Docstring/message false-positives inflate the
  produced *superset* harmlessly — the subset check still holds.
- **Mutation check:** the fake-panel test uses a NONSENSE producer name (e.g.
  `__no_such_event__`) so an accidental docstring match can't mask the failure (NB-4).
- The `_KNOWN_ORPHANS` mechanism carries over for file/derived rows that have no event.

---

## 2. Architecture — TWO renderers, ONE manifest (operator ruling §8.4)

Both surfaces read the same renderer-agnostic manifest; math is shared, not reimplemented.

| Renderer | Context | Data path | Reused math |
|---|---|---|---|
| **A1 — remote TUI** (`d1m_monitor.py`) | SSH monitoring of the vast run | `run_feed_reader` remote-grep of the **structlog** log `logs/<run_name>.jsonl` (KB, not 340MB) | `run_feed_reader`: gap-skip effective rate, `sealbot_slope` Theil-Sen CI (false-green guard), `depth_health`, step_at bisect, stale-log/live-tip override |
| **B — local web** (rebuilt) | Browser dashboard, laptop | STATIC page polling JSON; **tails BOTH jsonl files** (emit_event `events_*.jsonl` + structlog `<run_name>.jsonl`) — see §5 | evalfair `retro_slope` for external bars; same run_feed_reader parse module for series |

Both renderers read BOTH channels (§1.1): the load-bearing fields (promotion results,
value_bce, fp16, forced_win, startup/config) live ONLY on the structlog channel, so the
web dashboard must tail the structlog log too — it is not an emit_event-only consumer.

**Retired:** `terminal_dashboard.py` (A2, ruling §8.1 — rewire alerts headless first, §6),
`d1m_status.sh` (superseded by `--once`).
**Untouched (out-of-scope, ruling §8.2):** viewer/analyze cluster + web_dashboard
game-persistence.

---

## 3. Layout — 5 panel groups

### 3.1 RUN HEADER
step · steps/hr (gap-skip effective rate — excludes restart gaps + resume step-resets) ·
ETA (total from the **`startup`** structlog event's `config.total_steps` — `run_feed_reader`
already reads it :530) · GPU util · **watchdog status** (armed-state from `startup`'s
`config.selfplay_stall_timeout_sec`; last-progress = newest `game_complete` ts; stall-risk =
now−last_progress vs timeout, using the A1 stale-log/live-tip override to survive file-sync
lag — **no new instrumentation**, `startup` already emits the full config) · **resolved_config
provenance link** (the CONFRES `resolved_config` emit_event, payload key `knobs` — the
forensic single source of the run's actually-executing config).

### 3.2 TRAINING HEALTH strip
entropy (`policy_entropy_selfplay`, canonical — drop the `selfplay_model_entropy_batch`
alias; **band 2.1–2.9 drawn on the chart**, collapse floor <1.5 marked) · draw rate
(hard-abort line at the resolved `monitors.hard_abort_draw_rate` threshold; dispatcher
intent 0.55×3 — drawn where configured) · value-bce **sp vs corpus** gap · colony trace
(`colony_extension_fraction` rolling) · fp16/loss-scale (sharp-drop alert) ·
forced_win_conversion (+ off_window rate).
**Planner-guarded (two mechanisms — they differ, BF-3):**
- `mcts_root_concentration` — CONFRES S2 emits it **null** (emit_event) / **omits the key**
  (structlog) under Gumbel (`events.py:37-48,324`). Guard `gumbel_hidden`: render iff the
  field is present AND non-null. Hidden, not zeroed.
- `mcts_mean_depth` — is emitted **UNCONDITIONALLY** by design (interior-PUCT descent is
  Gumbel-valid, `events.py:284`). Null-guarding would NOT hide it. Guard `planner_puct`:
  render iff the resolved planner is PUCT, read from the `startup` event's
  `config.selfplay.gumbel_mcts` flag (`run_feed_reader.py:534` already parses it). This
  hides the cumulative-mean depth-cliff artifact (`game_runner:669`) on run3 (Gumbel).

### 3.3 PROMOTION panel
Bound to the **structlog `evaluation_round_complete`** event (BF-1: the emit_event
`eval_complete` payload lacks wr_best/ci_best/distinct-n — `eval_drain.py:45`; the full
results dict goes only to structlog `eval_pipeline.py:509`; web reaches it via the
structlog tail). wr_best + **pair-CI** per 25k gate · promote/hold markers (`promoted`) ·
**eff_n** = `deploy_strength_distinct_per_pair_min` (BF-4: the real §D-ARGMAX distinct-game
n; `ci_argmax_n` is a CI tuple, NOT a count) shown next to raw n, **deploy-round only**
(stride-4) · sealbot WR as **slope** (`sealbot_slope` Theil-Sen + measurement-error CI —
the false-green guard; per-point σ from `ci_sealbot`) with level as descriptive-tooltip
only. **DROP** the `sealbot FAILED` pass/fail badge (BIASED bar).

### 3.4 VALUE-HEALTH panel
recognition-lag + ECE series, read from the **laptop valprobe JSONL** (synced/uploaded —
read-only file consumption, zero coupling to the training host). Producers:
`scripts/valprobe/measure_recognition_lag.py`, `scripts/valprobe/value_health.py`.

### 3.5 EXTERNAL BARS panel
d5 (`scripts/evalfair/compute_slope_report.py`) + kraken-MCTS
(`scripts/evalfair/head_vs_krakenbot.py` — NB-3: compute_slope_report is d5-only) **fair-book**
series · **Theil-Sen slope + CI** (reuse `scripts/evalfair/retro_slope.py` — one-resolver
law, no reimplementation) · stage boundaries (R 4→5→6→8 curriculum) marked as
**discontinuities, never spliced**. Fair-book n is distinct-by-construction, so no separate
eff_n column here (state it in the panel note).

---

## 4. Principles (binding on WP3)

1. **Slope-based wr everywhere**; level = descriptive tooltip only.
2. **Every number that has a CI shows it**; **eff_n next to every n**.
3. **No panel without a manifest row** (the ONE LAW).
4. **Planner-guard** PUCT-only stats: render iff field non-null (hidden under Gumbel).
5. **Reuse, don't reimplement** slope/CI: run_feed_reader for the live sealbot guard,
   evalfair `retro_slope` for external bars. (One-resolver law extends to statistics.)
6. **Never splice across stage boundaries** — draw discontinuities.
7. **Silent truncation logs itself** — any top-N/sampling cap emits what it dropped.

### Known seam (flagged for WP2 review, NOT a WP3 merge task)
Two Theil-Sen impls exist: `run_feed_reader.sealbot_slope` (measurement-error bootstrap +
analytic-t CI over aggregated wr+per-point σ — the live monitor's false-green guard) and
evalfair `retro_slope` (pair-bootstrap CI over raw per-checkpoint pair scores, stage-aware).
They take **different inputs** and serve different panels. Recommendation: keep both,
document why; do NOT force-merge (merging would drop the conservative measurement-error
lower-bound that guards §D-WS3V3-class false greens). Each is already a single resolver in
its own file — the law is satisfied.

## 5. Stack — boring, fewer deps

**B rebuilt as a STATIC page + JSON polling.** Drop Flask-SocketIO + gevent monkey-patching
(the current push stack, `web_dashboard.py:23,127`). A ~15s-refresh monitor does not need
server push; the static page polls a tiny stdlib HTTP server (reuse the `events_tail.py`
JSONL-tail logic behind plain HTTP, no SocketIO). **Web tails BOTH channels** (§1.1, BF-2):
- `GET /api/events.jsonl?since=<offset>` — emit_event tail (`logs/events_*.jsonl`).
- `GET /api/structlog.jsonl?since=<offset>` — structlog tail (`logs/<run_name>.jsonl`) —
  REQUIRED for promotion, value_bce, fp16, forced_win, startup/config (structlog-only fields).
- `GET /api/series/{value_health,external_bars}` — file-sourced series (valprobe/evalfair JSON).

**First-load backfill (NB-7):** `?since=offset` handles increments, but the JSONLSink file
does NOT rotate (`monitoring/events.py:66`) so first-load history needs a bounded tail —
serve the last N MB / downsample to a max point budget, and **log what was truncated**
(no-silent-caps principle). Durable-file polling strictly improves on the current
`_training_step_history` deque replay (`web_dashboard.py:106`) — no capability lost.

**Dependency delta: NEGATIVE** — removes flask-socketio + gevent from the dashboard path;
adds nothing (vanilla `fetch`, stdlib server). JSONLSink stays the durable source. A1 keeps
its rich/plotext + run_feed_reader stack (already boring, low-bandwidth). Series parsing +
slope/CI are shared Python modules both renderers import — unit-testable without a server
(TDD target in WP3).

---

## 6. Alert rewire (must land WITH the A2 delete — WP3)

Move these from the display-only `evaluate_*_alerts` wrappers to the headless structlog
path in `training/events.py` so they keep firing after A2 is retired:
`check_entropy_collapse` (<1.0) · `check_selfplay_entropy_collapse` (<1.5) ·
`check_grad_norm_spike` (>10) · `check_loss_increase_window` (3 consec ↑).
Hard-aborts + stall-watchdog already fire headless from `step_coordinator.py` (untouched).

---

## 7. Manifest summary (full rows in dashboard_manifest.yaml)

Authoritative rows + per-renderer channel bindings live in `dashboard_manifest.yaml` (fable5
fixes applied there). Illustrative summary (channel per binding; web tails both files):

| group | panel_id | channel → source → field | notes |
|---|---|---|---|
| run_header | header.step | web emit training_step.step / a1 structlog train_step_summary.step | dual-cadence |
| run_header | header.steps_per_hr | derived run_feed_reader.parse_feed.rate_effective | gap-skip |
| run_header | header.eta | rate + structlog startup.config.total_steps | startup carries config |
| run_header | header.gpu_util | web emit system_stats.gpu_util_pct / a1 structlog train_step_summary.gpu_util | |
| run_header | header.watchdog | structlog startup.config.selfplay_stall_timeout_sec + emit game_complete.ts | no new instr |
| run_header | header.resolved_config | web emit resolved_config.knobs | CONFRES link |
| training_health | health.entropy | web emit training_step / a1 structlog train_step_summary → policy_entropy_selfplay | band 2.1–2.9 |
| training_health | health.draw_rate | web emit iteration_complete / a1 structlog train_step_summary → draw_rate | abort line |
| training_health | health.value_bce_gap | structlog train_step → value_bce_{selfplay,corpus} | structlog-only |
| training_health | health.colony | emit/structlog game_complete.colony_extension_fraction | rolling |
| training_health | health.fp16_scale | structlog train_step.fp16_scale | structlog-only |
| training_health | health.forced_win_conv | structlog forced_win_trend → forced_win_conversion, off_window_forced_win_rate | structlog-only |
| training_health | health.root_concentration | iteration_complete/train_step_summary.mcts_root_concentration | guard gumbel_hidden |
| training_health | health.mcts_depth | iteration_complete/train_step_summary.mcts_mean_depth | guard **planner_puct** (unconditional emit) |
| promotion | promo.wr_best | structlog evaluation_round_complete.wr_best | slope+level |
| promotion | promo.ci_best | structlog evaluation_round_complete.ci_best | pair-CI |
| promotion | promo.eff_n | structlog evaluation_round_complete.**deploy_strength_distinct_per_pair_min** | eff_n, deploy-round only |
| promotion | promo.marker | structlog evaluation_round_complete.promoted | |
| promotion | promo.sealbot_slope | derived run_feed_reader.sealbot_slope | false-green guard |
| value_health | vh.recognition_lag | file scripts/valprobe/measure_recognition_lag.py | laptop, read-only |
| value_health | vh.ece | file scripts/valprobe/value_health.py | laptop, read-only |
| external_bars | ext.d5_series | file scripts/evalfair/compute_slope_report.py | stage-marked |
| external_bars | ext.kraken_series | file scripts/evalfair/**head_vs_krakenbot.py** | NB-3: d5 report is d5-only |
| external_bars | ext.slope_ci | derived scripts/evalfair/retro_slope.py | one-resolver |

DROP (no manifest row → not rendered): `mcts_root_concentration` as a *health signal*
(kept only as PUCT-guarded diagnostic), `value_spread` t3/alt panels, `sealbot FAILED`
badge. Producers for the first two stay (demoted); the third's raw WR stays as slope.
