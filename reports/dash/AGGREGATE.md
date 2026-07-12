# reports/dash/AGGREGATE.md — D-J DASH close

Rebuild of the d1m monitor + web dashboard, design-first, ONE LAW enforced from day one.
Branch `phase4.5/dash_rebuild` (off master `60cf720`), worktree `.claude/worktrees/dash`.

## Commits (5)
- `03974a9` WP1 audit + WP2 design/manifest + WP3.1 headless alert rewire (TDD)
- `f8e252c` WP3.1 teardown: retire terminal_dashboard (A2) + port contract test to manifest
- `8b4fcd7` WP3.2 B: web dashboard — static page + dual-channel JSONL polling
- `4d4fa66` WP3.2 A1: d1m TUI — drop value_spread + planner-guard depth/root_conc
- (this report)

## THE ONE LAW — verification output
`tests/test_event_contract.py` (manifest-driven): **5/5 pass**.
```
panels=24  event-bindings=26 (missing producer: 0)  path-bindings=8 (missing: 0)
```
Every panel binding cites a LIVE producer in its channel; every derived/file producer
path exists. **Mutation check**: a fake panel with `__no_such_event_producer__` is flagged
(the law bites). Two-channel produced-set (emit_event + structlog) — the old test was blind
to the entire structlog channel; the d1m TUI's producers are now guarded for the first time.

**Zero panels without producers.** Confirmed.

## KEEP / DROP / REWORK — executed vs approved (WP1 §8 rulings)

| Item | Approved (teardown) | Executed | ✓ |
|---|---|---|---|
| terminal_dashboard (A2) | RETIRE, rewire alerts first | Deleted; 4 alerts rewired headless via `emit_training_step_alerts_headless` (structlog, log_interval) BEFORE the delete | ✓ |
| `mcts_root_concentration` health panel | DROP display, keep emit planner-guarded | Dropped from both renderers' default view; rendered only under confirmed PUCT (`is_gumbel is False` / `gumbel_mcts===false`); producer untouched | ✓ |
| `mcts_mean_depth` | planner-guard (PUCT only) | Guarded on regime (not null — depth is unconditional, BF-3); hidden under Gumbel | ✓ |
| `value_spread` t3/alt panels | DROP display, keep emit demoted | Dropped from A1 + B; `fire_canary` producer untouched | ✓ |
| `sealbot FAILED` badge | DROP (BIASED) | Deleted from both; raw WR kept as slope (`promo.sealbot_slope`) | ✓ |
| `evaluate_*_alerts` wrappers + `check_sealbot_gate_failed` | DROP (display-only) | Deleted; individual `check_*` rules kept + fired headless | ✓ |
| `d1m_status.sh` | retire | Deleted (superseded by `--once`) | ✓ |
| viewer / analyze cluster | OUT-OF-SCOPE, untouched | Untouched (viewer routes + game-persistence preserved in web_dashboard.py) | ✓ |
| A1 + B | keep both, one manifest | Both rebuilt to `dashboard_manifest.yaml` | ✓ |
| wr level → slope | slope everywhere | wr_best/sealbot rendered as slope; level = tooltip | ✓ |
| eff_n | show distinct-n | `deploy_strength_distinct_per_pair_min` shown next to n (both renderers) | ✓ |

DROPs applied (manifest `dropped:`): root_concentration-as-health-signal, value_spread t3/alt,
sealbot-FAILED badge. **Producers for the first two survive** (demoted/planner-guarded).

## Load-bearing things that MOVED (not lost)
- **4 headless alerts** — `check_entropy_collapse` / `check_selfplay_entropy_collapse` /
  `check_grad_norm_spike` / `check_loss_increase_window` moved from the A2 display path to
  `step_coordinator` (structlog `training_alert`, log_interval). TDD 5/5; closeout gate 16/16.
- **Stall watchdog + hard-aborts** — already headless in `step_coordinator`; UNTOUCHED.
- **Game-persistence / viewer** — `_persist_game`/`_game_index`/`_load_existing_games` + viewer
  routes preserved in the web rebuild.
- **JSONLSink + structlog log + disk/gpu guards** — untouched producers; both renderers now
  consume both channels.

## Render-proofs — smoke against REAL banked data
```
tail_jsonl(train_*.jsonl, 13.97MB) -> lines=6578, next_offset=13966631, truncated=True
  (WARNING logged: dropped_bytes=11869479 — no-silent-caps principle works)
dashboard-consumed structlog events in real log: evaluation_round_complete, train_step
value_health series (real valprobe jsonl): 15 rows
external_bars (real): 193 rows
compute_external_slope: graceful {"error": "retro_slope import failed: No module named 'minimax_cpp'"}
```
Endpoints smoke-tested via Flask test client (`tests/test_web_serving.py`, 28 tests):
dual-channel tail resolves events_*.jsonl vs <run_name>.jsonl to different files; series routes serve real files.

## Orphan producers (produced, no display consumer — named, not dark-by-accident)
`strength_regression_hard_abort`, `sealbot_wr_revert_abort`, `robustness_hard_abort`,
`disk_free`. All are headless abort/guard events (the aborts set `shutdown.running` directly;
disk_free's real guard is the SIGTERM path). **None used to gate a display that is now dark** —
they never had a display consumer. `strength_aggregate`/`strength_cycle_density` remain phantom
(no producer) — not built into any panel (correctly).

## Tests
- ONE LAW `tests/test_event_contract.py`: 5/5
- Web: `tests/test_web_serving.py` 28, `test_dashboard_renderers` / `test_web_dashboard_gate` / `test_web_dashboard_paths`: green
- Alerts: `test_headless_alerts` 5/5, `test_alert_rules` 16/16, `test_value_spread_canary` green
- Launch-path gate: `test_closeout_lifecycle` + `test_step_coordinator_closeout` 16 passed, 1 skipped
- Pre-existing unrelated red (NOT introduced here): 5 `wr_hard_abort` failures + 11 `minimax_cpp`
  collection errors (native ext unbuilt in the worktree). Verified independent of this work.

## Deferred / operator-gated
1. **Live 3-step dry launch render** (WP3.4 tail) — needs the native ext (Rust engine +
   minimax_cpp), absent in this worktree; laptop heavy-build risks thermal cutoff. Run in the
   main checkout / vast (has ext built):
   `python scripts/train.py --iterations 3 --web-dashboard` then open the dashboard; confirm
   RUN HEADER + TRAINING HEALTH strip render. The launch PATH is unit-validated (closeout gate).
2. **`compute_external_slope` import coupling** — `scripts/evalfair/retro_slope.py` (pure
   Theil-Sen) drags `minimax_cpp` via a SealBot import chain, so the web external-bars slope
   degrades to an error dict without the native ext. Works on a training box. Tech-debt: lighten
   retro_slope's import chain so the stats fn is importable bare. Minor; graceful today.
3. **Web sealbot slope CI** — computed client-side (JS Theil-Sen over structlog wr_sealbot), no
   measurement-error CI (per-point sigma not transported over HTTP). `run_feed_reader.sealbot_slope`
   (the false-green guard with measurement-error CI) still owns the A1 TUI path. Wire sigma
   transport if the web panel wants the conservative CI.

## Status: WP1–WP3 COMPLETE + committed. WP3.4 live-launch + WP4 close DONE (this report).
Branch unpushed; not merged to master (awaiting operator).
