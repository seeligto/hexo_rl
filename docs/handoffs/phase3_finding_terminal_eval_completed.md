# Phase-3 finding — `terminal_eval_complete` structlog line omits `completed`/`terminal`

**Severity:** BLOCKER for the Phase-3 pre-registered gate (stage A integration test FAILS),
but **NOT a lifecycle defect** — the terminal eval runs and completes correctly. Fix is XS
(telemetry only, no logic change, no hot path → no bench gate).

## What failed

`scripts/run_phase3_armc_smoke.sh` stage A runs `tests/test_closeout_lifecycle.py`
(`slow+integration` — **deselected by `make test`**, so it never ran in Phase 0 and Phase-1 B1
explicitly deferred it). It FAILED:

```
assert final.get("completed") is True
E   AssertionError: terminal eval did not complete:
E   {'step':8,'promoted':False,'wr_best':0.5,'event':'terminal_eval_complete','level':'info',...}
E   assert None is True   # event has no 'completed' key
1 failed in 1021.59s
```

## Root cause

`hexo_rl/training/step_coordinator.py` emits `terminal_eval_complete` to **two sinks** with
**different payloads**:

- **event_emitter** (L1547-1558): carries `completed`, `terminal`, `wr_best`, `wr_sealbot`,
  `offwindow_forced_win_rate`, `error`.
- **structlog** (L1559-1560) — the JSONL the integration test + the operator watch sheet read —
  carries only `step`, `promoted`, `wr_best`. **Missing `completed` and `terminal`.**

The eval itself completed (wr_best=0.5, promoted=False). The hard-cap-timeout path (L1518-1530)
emits a *different* event (`terminal_eval_timeout`), so on the clean path `completed` is simply
absent rather than false — but the test (and any monitor keying on `completed:true`) can't
confirm completion from the JSONL log.

## Fix (XS — mirror the event_emitter payload into the structlog line)

```diff
--- a/hexo_rl/training/step_coordinator.py
+++ b/hexo_rl/training/step_coordinator.py
@@ -1556,8 +1556,9 @@
             "error": bool(result and result.get("error")),
         })
-        self._logger.info("terminal_eval_complete", step=step, promoted=promoted,
-                          wr_best=(result.get("wr_best") if result else None))
+        self._logger.info("terminal_eval_complete", step=step, promoted=promoted,
+                          completed=bool(result is not None), terminal=True,
+                          wr_best=(result.get("wr_best") if result else None))
```

After this, the JSONL `terminal_eval_complete` line carries `completed:true` + `terminal:true`,
the integration test passes, and the operator watch sheet's `terminal_eval_complete` token
gains the completion bit (more robust monitoring — distinguishes clean close from the
`terminal_eval_timeout` hung-evaluator path in the same log).

## Verification path to GREEN

1. Apply the 2-kwarg patch above.
2. Re-run stage A only (cheap, no full GPU training): `.venv/bin/python -m pytest
   tests/test_closeout_lifecycle.py -v -m "slow and integration"` (~17 min — its 8-iter
   subprocess dominates).
3. Then re-run the full `scripts/run_phase3_armc_smoke.sh` for the 9-criteria gate.

## Why Phase 1 missed it (process note)

Phase-1 B1 read the code + the **unit** tests (`test_terminal_eval_runs_full_battery_ignoring_stride`
asserts `terminal=True` on the *event_emitter* events captured in-process) and PASSED — that path
*does* carry the fields. The divergence is sink-specific: the structlog line is the one the
integration test + operators read, and only the integration test exercises it. The meta-lesson
holds: **a green unit test on one sink does not prove the other sink's contract** — the
integration test was the right instrument, and the plan correctly front-loaded it as Phase 3.
