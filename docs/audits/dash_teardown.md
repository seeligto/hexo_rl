# dash_teardown.md — D-J DASH WP1 teardown audit

**Task:** D-J DASH — scrap the d1m monitor + web dashboard, rebuild design-first.
**WP1 deliverable.** KEEP / DROP / REWORK table for every panel, route, chart, and
non-display duty across the monitoring surface. Read-only audit; nothing deleted yet.
**GATE:** operator (or strategy layer) approves this doc before WP3 deletes anything.

Method: 3 parallel read-only sonnet audits (bucket A = d1m TUI + terminal_dashboard,
B = web dashboard + serving, C = non-display duties + producers), each cross-referenced
against `docs/audits/stat_audit_2026-06-23/` (148 stats: **WRONG 4 / BIASED 16 /
REDUNDANT 7 / CORRECT 121**) and the corrected notation (Gumbel≠PUCT, turn≠ply, wr
level→slope, cumulative-mean artifacts).

---

## 0. THE ONE LAW (carry into WP2/WP3)

Every rebuilt panel names its PRODUCER EVENT in a manifest (panel_id → event → field).
`tests/test_event_contract.py` extends to assert every manifest row has a live producer.
A panel that can't cite its producer does not get built. §7 seeds that manifest with the
consumed-event → producer map derived here.

---

## 1. Surface map — FOUR areas, not one

| Area | Files | Role | Verdict headline |
|---|---|---|---|
| **A1. d1m_monitor TUI** | `scripts/d1m_monitor.py` (969), `hexo_rl/monitoring/run_feed_reader.py` (1001), `scripts/d1m_status.sh`, `scripts/d1m_replay_analyzer.py` | Remote-log rich TUI over SSH for the vast run | Rebuild target. Most panels correct; 2 DROP, ~6 REWORK. `run_feed_reader` holds load-bearing parse/CI math. |
| **A2. terminal_dashboard** | `hexo_rl/monitoring/terminal_dashboard.py` (756) | In-process live Rich TUI during training | **Retire recommended** — redundant with web dashboard AND is the sole caller of headless-worthy alerts (rewire risk, §5). |
| **B. web dashboard** | `web_dashboard.py`, `web_routes.py`, `static/index.html` (1380), `serve_dashboard.py`, `events_tail.py` (+ viewer/analyze — separate, §6) | SocketIO live dashboard + JSONL-tail serving | Rebuild target. User dislikes it. Core panels salvageable; 1 DROP, ~7 REWORK. |
| **C. non-display duties** | `alert_rules.py`, `value_spread_canary.py`, `gpu_monitor.py`, `disk_guard.py`, `value_probe.py`, `early_game_probe.py`, `events.py`, `game_recorder.py`, `metrics_writer.py`, `configure.py`, stall watchdog in `step_coordinator.py` | Producers, guards, alerts, event sink | **Mostly LOAD-BEARING — must survive teardown.** §4. The teardown must MOVE, not lose, these. |

Two independent monitors exist (A1 remote-log vs A2 in-process) plus the web dashboard —
**three renderers of overlapping data.** The rebuild collapses to: keep **A1** (remote
vast monitor) + **B** (local web dashboard) rebuilt to the manifest; **retire A2**.

---

## 2. Disposition summary

| Disposition | Count (approx) | Meaning |
|---|---|---|
| KEEP | ~55 panels/duties | Correct stat, still meaningful under run3/Gumbel |
| REWORK | ~22 | Right idea, wrong stat (level→slope, cumulative→windowed, biased median, missing CI/eff_n, re-anchor) |
| DROP | 4 display + 1 wrapper class | WRONG/dead/PUCT-only/phantom OR display-only scaffolding for a load-bearing rule |
| OPERATOR-DECIDES | viewer/analyze cluster (§6) | Possibly superseded by hexo-ref-server React frontend |

**Hard DROPs (display):** `mcts_root_concentration` sparkline+chart (A1) and `sys-root-conc`
row (B) — SH-pinned flat under Gumbel, WRONG as health signal. `sealbot_gate_passed`
"FAILED" badge (A2/B) — BIASED, near-always-False. `value_spread` t3/alt panels (A1/A2) —
VOID instrument (§3). **Keep the raw `mcts_root_concentration` emit** planner-guarded per
stat-audit carve-out (PUCT ablation diagnostic) — DROP is of the *panel*, not the producer.

---

## 3. DROP rows (explicit — these get deleted in WP3)

| Surface | Panel | Consumed event → field | Why DROP | Producer disposition |
|---|---|---|---|---|
| A1 d1m_monitor:717 + :790 | root_concentration sparkline + trajectory chart | `train_step_summary.mcts_root_concentration` | WRONG — SH-pinned under Gumbel, banked flat 0.545–0.576; trend is noise | KEEP emit, planner-guarded (`gumbel_mcts`) for future PUCT A/B |
| B index.html `sys-root-conc` | Root concen row | `iteration_complete.mcts_root_concentration` | same | same |
| A1 :607 + :770 / A2 :633 | value_spread t3/alt panel + chart + ALERT block | `value_spread.{t3_spread,alt_spread}` | VOID: t3 band anchored to wrong synthetic (banked inverted −0.525); alt = NaN sentinel (enc=None, 227/227 skipped); empty all of run2 | Producer `fire_canary` → REWORK/retire (§4); keep JSONL/structlog headless path, DROP the display |
| A2 :744 / B elo-badge | `sealbot FAILED` badge | `eval_complete.sealbot_gate_passed` | BIASED — wr≥0.5 bar vs SealBot fair-WR ~18% → fires ~every eval; misleading | Raw `win_rate_vs_sealbot` KEEP as informational; drop the pass/fail badge |
| C alert_rules | `evaluate_training_step_alerts` / `evaluate_eval_complete_alerts` / `evaluate_value_spread_alerts` aggregators | — | Display-only wrappers, sole caller is terminal_dashboard | **DROP wrappers, KEEP individual `check_*` rules and REWIRE headless (§5) — do NOT lose the rules.** |

Also retire: `scripts/d1m_status.sh` (superseded by `d1m_monitor.py --once`; its naive
`rate(first,last)` gives wrong ETAs on resumed runs — no gap-skip).

---

## 4. Non-display duties — the MOVE-DON'T-LOSE list (§WP1 core ask)

A UI teardown must not kill headless machinery. Status of each:

| Duty | File | Headless today? | Teardown risk | Action |
|---|---|---|---|---|
| **Stall watchdog** (games_completed freeze → `os._exit(42)` + buffer save) | `step_coordinator.py:186,671,835` | YES — structlog only, zero display coupling | **NONE** — lives outside monitoring/ entirely | Leave untouched. (memory: run2 45h livelock fix) |
| **promotion_gate_subprocess_isolation** ROOT fix | `step_coordinator.py` (default OFF) | YES | NONE | Leave untouched; run3 arms it |
| **JSONLSink** (events → `logs/events_<run>.jsonl`) | `events.py:66`, registered `lifecycle.py:304` unconditionally | YES — file write independent of dashboards | NONE — the durable event artifact; new dashboard TAILS it | Keep; rebuild reads this file (WP2 principle: static page + JSONL fetch) |
| **structlog rotating log** | `configure.py:39` | YES | NONE | Keep — durable alert log for post-mortem grep |
| **DiskGuard** (SIGTERM < 5GB) | `disk_guard.py:18` | YES — SIGTERM is OS-level | NONE (display only renders the event) | Keep producer; new dashboard may render `disk_alert` |
| **GPUMonitor** (`system_stats` every 5s) | `gpu_monitor.py:44` | Producer independent; also direct-attr read in step_coordinator warmup | LOW | Keep producer; new dashboard reads the event |
| **value_probe_drift** (decisive/draw drift, gated `instrumentation.enabled`) | `value_probe.py`, `step_coordinator.py:1462` | Producer independent; default OFF | LOW | Keep; VALUE-HEALTH panel consumes it |
| **EarlyGameProbe** (entropy>4.5 warn, fields into `training_step`) | `early_game_probe.py:181` | YES — structlog warning | NONE | Keep |
| **GameRecorder** (samples replays; feeds `ForcedWinTrend`) | `game_recorder.py`, consumed `step_coordinator.py:622` | N/A — I/O, feeds eval signal not display | NONE | Keep |
| **MetricsWriter** (TensorBoard/wandb) | `metrics_writer.py` | YES — TB files | NONE | Keep |
| **Headless hard-aborts** (draw_rate, stride5_p90, robustness, strength) | `step_coordinator.py:1055/1077/1209/1248` set `shutdown.running=False` directly | YES — do NOT route through alert_rules aggregators | NONE for the abort path | Keep; see §5 for the display-coupled *warn* rules |
| **value_spread_canary producer** (`fire_canary`) | `value_spread_canary.py:385`, called `trainer.py:1348` | Structlog path headless; never raises | Display DROP (§3) is safe | REWORK-or-retire: DROP display, keep emit demoted-informational. Re-anchor is run3-watcher work (out of scope) |
| **run_feed_reader parse/CI** (remote grep, gap-skip effective rate, `sealbot_slope` Theil-Sen CI, `depth_health`, step_at bisect, stale-log/live-tip override, 30k binning) | `run_feed_reader.py` | pure parse for A1 | display-only BUT correctness-critical | **Reuse verbatim in rebuilt A1.** `sealbot_slope`/CI = the false-green guard (moved with explicit "load-bearing" note); **one-resolver law: reuse, do not reimplement** (also true for evalfair slope/CI in WP2). |

---

## 5. THE REWIRE HAZARD (biggest teardown risk — flag for operator)

The individual alert rules that SHOULD fire headless are today invoked **only** through
`terminal_dashboard.py` via the `evaluate_*_alerts` aggregators:

- `check_entropy_collapse` (policy_entropy < 1.0) — **CORRECT** (stat_audit i=200)
- `check_selfplay_entropy_collapse` (< 1.5) — **CORRECT** (i=201)
- `check_grad_norm_spike` (> 10) — standard sentinel
- `check_loss_increase_window` (3 consec ↑) — sentinel

Retiring A2 (terminal_dashboard) **silently kills these alerts** unless rewired to the
headless structlog path in `training/events.py`. **WP3 MUST rewire these before/with the
A2 delete** — this is the load-bearing thing that would die with the UI. The hard-aborts
and watchdog are already headless (safe); only these four warn-rules are display-trapped.

Separately: `strength_regression_abort` is headless-wired but fed by the **phantom**
`strength_aggregate` (no producer — dead since stat_audit i=183; gate falls through to
wr_best). Default-OFF, so harmless. Not a display concern; note only.

---

## 6. OPERATOR-DECIDES — viewer / analyze cluster (scope boundary)

Not run-monitoring; possibly superseded by the hexo-ref-server React frontend (memory:
user prefers it for HeXO viz/arena, dislikes this Flask dashboard).

| Component | Files / routes | Recommendation |
|---|---|---|
| Game viewer | `viewer.html` (623), `/viewer`, `/viewer/recent`, `/viewer/game/<id>` | OUT-OF-SCOPE for DASH rebuild; spec is `docs/09_VIEWER_SPEC.md`. Keep as thin passthrough or retire in favor of hexo-ref-server |
| Policy analyzer / play | `analyze.html` (1029), `analyze_api.py`, `/api/analyze*`, `POST /viewer/play` | OUT-OF-SCOPE; no hexo-ref-server equivalent yet. Keep in Flask untouched or migrate later |
| `hex_canvas.js` | shared by viewer+analyze only | Out iff both above are out |
| `game_browser.py` | offline CLI, no web route | OUT-OF-SCOPE entirely (not a dashboard consumer) |

**Recommendation:** DASH rebuild scope = A1 (remote monitor) + B core run-monitoring
panels only. Leave the viewer/analyze cluster in place, untouched, decoupled — do NOT
delete in WP3 unless operator rules OUT. `web_dashboard.py`'s game-persistence duties
(`_persist_game`, `_game_index`, `_load_existing_games`) stay if viewer stays.

---

## 7. Manifest seed — consumed events → producer (feeds WP2 ONE LAW)

Every event a rebuilt panel may consume, with its verified producer. Orphans flagged.

| Event | Producer (file) | Key fields | Notes |
|---|---|---|---|
| `training_step` | `training/events.py` | loss_{total,policy,value,aux,chain,ownership,threat}, policy_entropy{,_pretrain,_selfplay}, policy_target_*, grad_norm, lr, value_accuracy{,_masked}, value_bce_{selfplay,corpus}, early_game_* | `selfplay_model_entropy_batch` alias → migrate to `policy_entropy_selfplay`, drop alias |
| `iteration_complete` | `training/events.py` | games_per_hour, positions_per_hour, sims_per_sec, batch_fill_pct, buffer_{size,capacity}, corpus_selfplay_frac, mcts_mean_depth, mcts_root_concentration, cluster_{value_std,policy_disagreement}_mean | depth=cumulative artifact; root_conc=DROP display; cluster means=REWORK windowed |
| `game_complete` | `selfplay/pool.py` | winner, moves, colony_extension_stone_{count,total}, stride5_run_{max,p90}, row_max_density | |
| `eval_complete` / `evaluation_round_complete` | `eval_pipeline.py` | step, wr_best, ci_best, wr_sealbot, ci_sealbot, wr_bootstrap_anchor, wr_random, elo_estimate, promoted, value_fc2_weight_abs_max, g4_value_head_band_pass | wr level→slope (A1 already does Theil-Sen). `sealbot_gate_passed` badge=DROP |
| `evaluation_games_complete` | `eval_pipeline.py` | phase, winrate, n_games | per-phase live WR (A1) |
| `forced_win_trend` | `step_coordinator.py` (ForcedWinTrend) | forced_win_conversion, off_window_forced_win_rate, n | coherence/golong cluster — KEEP |
| `system_stats` | `gpu_monitor.py` | gpu_util_pct, vram_*, ram_*, rss_gb, cpu_util_pct, workers_* | KEEP |
| `value_probe_drift` | `step_coordinator.py:1462` (ValueProbe) | decisive_mean, draw_mean, step | gated instrumentation.enabled; VALUE-HEALTH |
| `buffer_composition` | `step_coordinator.py` | corpus_fraction, draw_target_fraction, {six,colony,cap}_terminal_fraction | gated; KEEP |
| `model_version_summary` | `pool.py` | median_range, p90_range, max_range, current_version, spearman_rho_range_vs_draw | median=REWORK ((n-1)//2 index) |
| `worker_draw_rate` | `pool.py` | per_worker | KEEP |
| `disk_free` / `disk_alert` | `disk_guard.py` | disk_free_gb | KEEP optional render |
| `run_start` / `run_end` | `training/loop.py` | run_id, step, worker_count | KEEP |
| `value_spread` | `value_spread_canary.py` | t3_spread, alt_spread | VOID → DROP display, keep demoted emit |
| `strength_regression_hard_abort` | `step_coordinator.py:1220` | — | headless abort event; **no display consumer** (fine) |

**Orphan producers (produced, no display consumer — fine, name them per WP4):**
`strength_regression_hard_abort`, `sealbot_wr_revert_abort`, `system_stats` (no abort
consumer), `disk_free` (display-only). **Phantom (no producer):** `strength_aggregate`,
`strength_cycle_density` — dead, do not build a panel on them.

---

## 8. OPEN DECISIONS — operator RULED 2026-07-12 (Tom)

1. **Retire terminal_dashboard (A2)?** → **RETIRE, rewire alerts first.** WP3 moves
   `check_entropy_collapse` / `check_selfplay_entropy_collapse` / `check_grad_norm_spike` /
   `check_loss_increase_window` to the headless structlog path in `training/events.py`
   BEFORE deleting A2 (§5).
2. **Viewer/analyze cluster scope (§6)?** → **OUT-OF-SCOPE, leave untouched.** WP3 does
   NOT touch `viewer.html`/`analyze.html`/`hex_canvas.js`/`analyze_api.py`/`game_browser.py`
   or the game-persistence duties in `web_dashboard.py` (`_persist_game`, `_game_index`,
   `_load_existing_games`).
3. **value_spread canary producer?** → **KEEP emit, demoted-informational; DROP display.**
   `fire_canary` keeps writing to JSONL/structlog; the t3/alt panels are deleted. Re-anchor
   is separate run3-watcher work.
4. **A1 vs B split?** → **KEEP BOTH, one shared manifest.** A1 (remote vast SSH TUI) + B
   (local web dashboard) both rebuilt to the same panel_id→event→field manifest.

WP1 GATE PASSED. WP2 (design doc + panel manifest) proceeds.
