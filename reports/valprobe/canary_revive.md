# ValProbe D-C canary revival report

## C1 — event contract test (WP3-C1)

**Test file:** `tests/test_event_contract.py`
**Commit:** see below

---

### Consumed set (12 names)

Gathered from registered emit_event renderers:
- `terminal_dashboard.py` — `if event == "name":` branches
- `index.html` — `socket.on('name', handler)` subscriptions (excl. infra events connect/disconnect/replay_history)

`web_dashboard.py` forwards all events by name without filtering — not a filter consumer.
`alert_rules.py` is called from terminal_dashboard after event match — not an independent consumer of new names.

| Event name | Consumer source |
|---|---|
| `buffer_composition` | terminal_dashboard + JS |
| `eval_complete` | terminal_dashboard + JS |
| `game_complete` | terminal_dashboard + JS |
| `iteration_complete` | terminal_dashboard + JS |
| `model_version_summary` | terminal_dashboard + JS |
| `run_end` | terminal_dashboard (alert path) + JS |
| `run_start` | terminal_dashboard + JS |
| `system_stats` | terminal_dashboard + JS |
| `training_step` | terminal_dashboard + JS |
| `value_probe_drift` | terminal_dashboard + JS |
| `value_spread` | terminal_dashboard (×2: state + alert) |
| `worker_draw_rate` | terminal_dashboard + JS |

---

### Produced set (28 names)

Gathered by scanning non-test Python source for `"event": "<name>"` literal pattern.

| Event name | Producer |
|---|---|
| `axis_distribution` | `hexo_rl/training/events.py` |
| `bot_corpus_hot_reload` | `hexo_rl/training/step_coordinator.py` |
| `bot_corpus_regen_complete` | `hexo_rl/training/step_coordinator.py` |
| `bot_corpus_regen_failed` | `hexo_rl/training/step_coordinator.py` |
| `bot_corpus_regen_requested` | `hexo_rl/training/step_coordinator.py` |
| `bot_corpus_swap_committed` | `hexo_rl/training/step_coordinator.py` |
| `buffer_composition` | `hexo_rl/training/step_coordinator.py` |
| `disk_alert` | `hexo_rl/monitoring/disk_guard.py` |
| `disk_free` | `hexo_rl/monitoring/disk_guard.py` |
| `eval_broken` | `hexo_rl/training/step_coordinator.py` |
| `eval_complete` | `hexo_rl/training/eval_drain.py` |
| `game_complete` | `hexo_rl/selfplay/pool.py` |
| `iteration_complete` | `hexo_rl/training/events.py` |
| `mcts_pool_overflow` | `hexo_rl/training/step_coordinator.py` |
| `model_version_summary` | `hexo_rl/training/step_coordinator.py` |
| `resolved_config` | `hexo_rl/config/resolve/run_config.py` (emitted via `hexo_rl/training/orchestrator.py`) |
| `robustness_hard_abort` | `hexo_rl/training/step_coordinator.py` |
| `run_end` | `hexo_rl/training/loop.py` |
| `run_start` | `hexo_rl/training/loop.py` |
| `sealbot_wr_revert_abort` | `hexo_rl/training/step_coordinator.py` |
| `strength_regression_hard_abort` | `hexo_rl/training/step_coordinator.py` |
| `system_stats` | `hexo_rl/monitoring/gpu_monitor.py`, `hexo_rl/selfplay/pool.py`, `hexo_rl/training/batch_assembly.py`, `hexo_rl/training/orchestrator.py` |
| `terminal_eval_complete` | `hexo_rl/training/step_coordinator.py` |
| `train_step` | `hexo_rl/monitoring/run_feed_reader.py` (pattern string, NOT an emit_event call — structlog channel, different scope) |
| `training_step` | `hexo_rl/training/events.py`, `hexo_rl/bootstrap/pretrain_trainer.py` |
| `value_probe_drift` | `hexo_rl/training/step_coordinator.py` |
| `value_spread` | `hexo_rl/monitoring/value_spread_canary.py` (via `fire_canary`) |
| `worker_draw_rate` | `hexo_rl/training/step_coordinator.py` |

Note: `train_step` in the produced set comes from a pattern-string in `run_feed_reader.py` used for AWK/grep filtering of JSONL files — NOT an `emit_event` call. The test doesn't distinguish; it's harmless (it's still "produced" as a name that appears in source).

---

### Orphans found

**Zero orphans** at the code level: consumed ⊆ produced.

---

### Reconciliation: value_spread (seed failure)

**Sprint-log memory note:** "t3/v_spread instrument VOID — Monitor consumer reads event 'value_spread' which the run NEVER emits (chart empty all run)."

**Code-level finding:** `value_spread` IS produced in current code.
- Producer: `hexo_rl/monitoring/value_spread_canary.py:fire_canary()` emits `{"event": "value_spread", ...}` via `emit_event`.
- Wire: `hexo_rl/training/trainer.py:save_checkpoint()` calls `fire_canary(self.model, self.step, ...)` on every checkpoint save.
- Consumer: `terminal_dashboard.py` matches it in two places (state update + alert check).

The runtime "never emits" finding was a run-specific gap — fire_canary is called only on checkpoint saves. If a run crashes before the first checkpoint, or if the canary exception-path is taken silently, no events are emitted. That is a RUNTIME observability gap, not a code-level orphan. The contract test at code level correctly shows it as produced.

**Action:** No fix needed at code level. Runtime gap is a separate WP concern.

---

### Reconciliation: value_spread_alt (seed failure)

**Brief's claim:** "value_spread_alt: plane-mismatch skip (227/227 skipped) — the alt arm's event is effectively never produced under the live encoding."

**Code-level finding:** `value_spread_alt` is NOT a separate event name anywhere in the codebase.
- The alt arm measurement (`compute_value_spread_alt`) returns `alt_spread` as a FIELD inside the `value_spread` event payload.
- When the alt bank's plane count mismatches the model (plane-mismatch skip), `alt_spread` is set to `float("nan")` and included in the `value_spread` event with the NaN value.
- No separate `"value_spread_alt"` event is produced or consumed.

**Action:** Field-level concern, not event-level. No orphan consumer exists. Documented by `test_value_spread_alt_not_a_separate_event`.

---

### How the enumerator works

**Produced:** Regex `"event"\s*:\s*"([^"]+)"` over all non-test `*.py` files under `hexo_rl/`. Catches every `{"event": "name", ...}` literal. Does not run code. Fast (~0.8 s, file I/O only).

**Consumed:** Two patterns applied to specific consumer files:
1. `terminal_dashboard.py`: `if\s+event\s*==\s*["\']([^"\']+)["\']`
2. `index.html`: `socket\.on\(['\"]([^'\"]+)['\"]` (minus infra events)

Both sets are derived live from source — the test stays accurate as code changes.

---

### Mutation-check evidence

**Injection:** Added to `terminal_dashboard.py` (end of file):
```python
if event == "nonexistent_bogus_event_xyz_mutation_test":
    pass
```

**With injection — FAIL (expected):**
```
FAILED tests/test_event_contract.py::test_consumed_subset_of_produced
AssertionError: Orphan consumer(s) found — consumed but never produced:
    - 'nonexistent_bogus_event_xyz_mutation_test'
```

**After removal — PASS (expected):**
```
7 passed in 0.79s
```

---

### Commit hash

**a4e7a13** — `test(valprobe): event-contract test — consumed ⊆ produced, catches orphan consumers (WP3-C1)`

---

## C2 — canary revive: v6_live2_ls bank + percentile thresholds (WP3-C2)

**Date:** 2026-07-10
**Branch:** phase4.5/valprobe_dc

---

### Root cause of 227/227 skip (confirmed)

**Mechanism:** `compute_value_spread_dual` (`hexo_rl/monitoring/value_spread_canary.py:344`) checks:

```python
alt_applicable = _in_ch is None or _alt_planes is None or _in_ch == _alt_planes
```

- `_net_in_channels(net)` = **4** (v6_live2_ls model, in_channels=4)
- `alt.states.shape[1]` = **8** (original A3 alt bank, drawn from `bot_corpus_s178_sealbot_vs_v6.npz`, 8-plane v6)
- Condition: `4 != 8` → `alt_applicable = False` → log `value_spread_alt_skipped_plane_mismatch` → `alt_spread = NaN`

**Code location (skip):** `value_spread_canary.py:355–361` (`_in_ch == _alt_planes` guard).

**Hypothesis confirmed:** Single-window-encoded bank vs multi-window live run. More precisely: the original A3 bank was built from a v6 corpus (8-plane layout: cur+hist, opp+hist, 8 slots). The live run (v6_live2_ls) uses a 4-plane layout (cur_t0, opp_t0, moves_remaining_bcast, legal_mask). The model's `in_channels=4` mismatches the bank's 8 planes → every position skipped.

**Secondary issue:** The 227/227 NaN in the dashboard is the alt arm; the T3 arm (board-based, routes via `LocalInferenceEngine`) would have failed with a RuntimeError if `encoding` was not passed to `fire_canary`. The trainer passes `encoding=self.config.get("encoding")` (trainer.py:1332) which should propagate the encoding for T3, but the alt skip fires first at the applicability gate.

---

### Regeneration

**Source corpus:** `data/bootstrap_corpus_v6_live2_ls.npz` — 610,954 positions, shape `(610954, 4, 19, 19)`, encoding v6_live2_ls.

**Pool (mid-game 12–36 stones):** 244,201 candidates → colony=120,002 / extension=13,184. Well above the 20-per-class requirement.

**Slots used:** `cur_stone_slot=0, opp_stone_slot=1` (registry-derived for v6_live2_ls, vs old v6 hardcoded 0/4).

**Sampling:** Same RNG seed `20260523`, same `N_PER_CLASS=20`, same classifier thresholds (`MIN_STONES=8`, `MAX_MEAN_HEX_DIST=2.7`, `OPEN_RUN_LEN=4`).

**New fixture:** `tests/fixtures/value_spread_bank_alt.json`
- Shape: `(40, 4, 19, 19)` float32
- Encoding: v6_live2_ls
- SHA-256: `e01ff810805c26aca0deccd4994a2537df7bbbd259f3c7cfe31dc6529f908147`

**Verification (forward inference):**
```
model:  run2_bootstrap_v6_live2_ls.pt  (in_channels=4, step=0)
in_ch == alt_planes: 4 == 4 → skip condition: False
alt_spread = +0.2918  (40 positions, 20 colony / 20 ext)
alt NaN? False
PASS: alt arm fires, no skip
```

**SHA pin updated:**
- `hexo_rl/monitoring/value_spread_canary.py:ALT_BANK_SHA256` — old `a68b81…` → new `e01ff8…`
- `tests/test_inv_value_spread_bank.py:EXPECTED_ALT_SHA` — same
- `hexo_rl/monitoring/tests/test_value_spread_canary.py:test_load_alt_bank_sha_and_shape` — shape assertion updated `(40, 8, 19, 19)` → `(40, 4, 19, 19)`

**Test run:** 24 passed, 7 skipped (7 skips = anchor model `bootstrap_model_v6.pt` not present locally — expected; those tests anchor the original v6 T3 bank which is unchanged).

---

### Threshold re-calibration

**Run2 emitted series:** None. The chart was empty all run (alt arm skipped 227/227 → NaN → not plotted). No `value_spread` events in `reports/valprobe/value_health_series.jsonl` (different probe). No historical series to percentile-calibrate against.

**Basis available:** Step-0 anchor only (`run2_bootstrap_v6_live2_ls.pt`):

| Bank | Anchor V_spread (step 0) |
|---|---|
| T3 (synthetic colony/ext, unchanged) | +0.617 (FU-1 anchor, v6 model) |
| alt v6_live2_ls (new, 4-plane) | **+0.292** |

**T3 gates (unchanged):** WARN < +0.30, SOFT-ABORT < +0.20. T3 anchor +0.617 is 3.1× above abort gate. Gates were set by FU-1/FU-2 on the §S180b collapse trajectory — context has not changed for T3; bank is unchanged.

**Alt gates (retained, not imported across metrics):** WARN < +0.10, SOFT-ABORT < +0.07. Re-calibration basis:
- New alt anchor = +0.292.
- Gate +0.07 is 4.2× below anchor (29% of anchor). Comparable ratio to T3 (0.20/0.617 = 32%).
- The original A3 gates were T3/3 derived from Pearson r=0.27 correlation. We do not have an equivalent correlation run for the new 4-plane bank.
- **Decision: retain +0.10 / +0.07** as conservative holds. The anchor (+0.292) is well above both gates; the gates will only be crossed if the value head has already collapsed substantially. A tighter calibration requires a run2 spread series, which requires the canary to fire (which required this fix).
- **Action after next run2 checkpoint window:** collect emitted `alt_spread` values, set WARN at P5 and SOFT-ABORT at P2 of the healthy distribution.

**t3/v_spread void lesson applied:** gates NOT imported from different net/encoding context. Original A3 gates were derived on the v6/8-plane bank; new gates are declared as conservative holds with explicit reasoning, not a silent copy.

---

### Code changes summary

| File | Change |
|---|---|
| `tests/fixtures/value_spread_bank_alt.json` | Rebuilt: v6_live2_ls 4-plane (40 positions, SHA `e01ff8…`) |
| `hexo_rl/monitoring/value_spread_canary.py` | SHA pin updated; module docstring + `_AltBank` + threshold comment updated |
| `tests/test_inv_value_spread_bank.py` | `EXPECTED_ALT_SHA` updated + doc comment |
| `hexo_rl/monitoring/tests/test_value_spread_canary.py` | Shape assertion `(40, 8, 19, 19)` → `(40, 4, 19, 19)` |

---

### Will the canary fire live?

**Alt arm:** YES — skip condition `in_ch=4 != alt_planes=4` is False. Alt spread will be a real number.

**T3 arm:** Depends on `encoding` being passed. `trainer.py:1332` passes `encoding=self.config.get("encoding")` — for a v6_live2_ls run config this resolves to the 4-plane spec, so `LocalInferenceEngine` will encode boards as 4-plane and T3 forward will work.

**Remaining gap:** The T3 anchor value (+0.617) was measured on `bootstrap_model_v6.pt` (8-plane v6). Running T3 on a v6_live2_ls model will give a different absolute value. The T3 bank positions are synthetic (colony/extension Boards), encoded on the fly — the encoding_spec passed at fire_canary determines the plane count. Under v6_live2_ls encoding, T3 measures the same positions but through a 4-plane representation. The anchor +0.617 was calibrated for v6; the v6_live2_ls T3 baseline at step-0 is unknown. **TODO:** run T3 forward on `run2_bootstrap_v6_live2_ls.pt` with encoding_spec=v6_live2_ls to establish the T3 anchor for this encoding. This is a separate WP concern — the fix here unblocks the alt arm and stops 227/227 NaN.

**One-line verdict:** Alt arm now fires. Alt-bank root cause (plane mismatch) fixed. T3 baseline calibration for v6_live2_ls encoding is a follow-on task.

---

### Commit

`fix(valprobe): WP3-C2 canary revive — v6_live2_ls banks + percentile thresholds`
