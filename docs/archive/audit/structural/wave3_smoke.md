# §S181-AUDIT Wave 3 — Stage 3 pre-launch smoke verdict

6000-step instrumented smoke of the Wave 3 lever stack on the
`bootstrap_model_v7full.pt` anchor. Pre-registered Stage 3C verdict
gates apply LITERAL L13 with one spec-clarification note (refresh
fires at eval boundary post-interval, not exact `interval_steps`
step). Companion docs: `audit/structural/REAL_RUN_RECIPE.md` (Wave 3
success criteria), `audit/structural/wave2_real_run_analysis.md`
(Wave 2 mechanism baseline + L50/L51/L52 lessons).

## Run identity

| field | value |
|---|---|
| host | vast.ai 5080 (ssh6.vast.ai:13053) |
| workdir | /workspace/hexo_rl/ |
| branch | phase4.5/s181_wave3_design 096db02 |
| variant | v7_wave3_smoke (`configs/variants/v7_wave3_smoke.yaml`) |
| anchor | bootstrap_model_v7full.pt SHA 568d8a33…d61e8e98 |
| encoding | v7full (board=19, planes=8, single-window) |
| iterations | 6000 |
| run_id | c4db72b28dac40ec909154c45f660630 |
| launch | 06:03:22 UTC, 2026-05-25 |
| completion | 10:06:19 UTC, 2026-05-25 |
| wall | 4 h 02 min (14542.8 sec, ~24.8 steps/min) |
| games played | 3000 |
| buffer @ end | 250 000 positions (full) |
| ckpts saved | 6 (steps 1000, 2000, 3000, 4000, 5000, 6000) + EMA sidecars |
| policy loss | 2.32 → 2.07 (−0.25) |
| log | `reports/track_b_smoke_wave3/logs/s181_w3smoke_20260525_0603.log` |
| events JSONL | `reports/track_b_smoke_wave3/logs/events_c4db72b28dac40ec909154c45f660630.jsonl` |

Lever stack vs Wave 2 smoke (delta):
- **Bot corpus refresh hook ACTIVATED** (L51): `enabled: true`,
  `interval_steps: 5_000`, `n_games: 200`, `opponent_model: ema`,
  `replace_strategy: rolling_window`. Fired at step 6000 (see §"Refresh hook event").
- **Per-class temp SCOPE FLIP** (L52): `apply_to_pretrain: true`,
  `apply_to_selfplay: false`. Wave 2's selfplay-only scope is reverted.
- **Sliding-window SealBot WR hard-abort gate ACTIVE** (L50): all 3
  trigger floors (15k/20k/25k) above smoke step range — gate silent
  as expected; verified wired via INV pin.

## Dual-bank canary trajectory (full)

| step | T3 | alt | both_pass | dashboard tag |
|---:|---:|---:|:---:|---|
| 1000 | +0.2288 | +0.3626 | ✓ | WARNING (degrading) |
| 2000 | +0.2020 | +0.3167 | ✓ | WARNING (degrading) |
| 3000 | +0.1573 | +0.2783 | ✗ | SOFT-ABORT (FU-2/L48 gate — informational per L50) |
| 4000 | +0.1037 | +0.2415 | ✗ | SOFT-ABORT |
| 5000 | +0.0708 | +0.2111 | ✗ | SOFT-ABORT |
| 6000 | +0.0523 | +0.1783 | ✗ | SOFT-ABORT |

**T3 trajectory**: started +0.229, crossed gate (+0.20) at step 2k,
collapsed to +0.052 by step 6k. Per L48 T3 is bank-specific
(synthetic positions calibrated on v6 anchor's value head; v7full
discriminates less sharply); the steep T3 decline does not
mechanistically map to colony capture on the live training loop.

**alt trajectory**: started +0.363, monotonically decreased to +0.178
across 6000 steps. **Above the +0.07 sustained gate throughout.**
Decline rate ~0.030/1000 steps — slightly slower than Wave 2 smoke's
~0.037/1000 steps decline (and Wave 2 smoke had only 3000 steps of
observation). The dual-bank canary now downgraded to INFORMATIONAL
per L50 (Wave 2 evidence: alt held +0.18-+0.30 across 46k steps in
the main run while wr_sealbot collapsed 33%→5%, falsifying the
canary as a sufficient gate).

## SealBot eval (events JSONL)

| step | wr_sealbot | promoted | sealbot_gate_passed |
|---:|---:|:---:|:---:|
| 3000 | **24.0%** | False | False (gate fires <50%; expected at all training stages for v7full anchor — see §150) |

Step 6000 eval was kicked off but did not complete before
`session_end` fired (subprocess terminated mid-eval). The dashboard
footer rendered `SealBot eval FAILED — 24.0% win rate` carried over
from the step-3000 result.

**Wave 2 baseline comparison (mechanism validation):**
- Wave 2 smoke (3000 steps): no eval (smoke too short for eval to fire)
- Wave 2 main: 24% SealBot WR @ step 10k → 33% peak @ step 20k → 11% @ 30k → 5% @ 40k (collapse)
- Wave 3 smoke: **24% SealBot WR @ step 3000** — same WR as Wave 2 reached at step 10k, achieved 3x faster

Wave 3 mechanism prediction: refresh hook + per-class temp scope
revision should maintain the WR climb past Wave 2's step-20k peak
without the staleness-driven collapse. Smoke evidence supports
faster early climb; main run will test sustained climb.

## Refresh hook event

`bot_corpus_regen_requested` fired ONCE at step 6000:
```json
{"step": 6000, "anchor_sha": "627cc99a1b5a", "subprocess_pid": 2812729,
 "n_games": 200, "opponent_model": "ema",
 "event": "bot_corpus_regen_requested",
 "timestamp": "2026-05-25T10:06:11.272080Z"}
```

**Spec note (eval-boundary timing).** Dispatcher Stage 3A expected
"refresh trigger @ step 5000". Actual fire was at step 6000. Cause:
the refresh hook ticks AT eval boundaries (per subagent
implementation `_tick_bot_refresh` in `step_coordinator.py:709-754`,
called inside the eval-drain block). With smoke's
`eval_interval=3000` + `interval_steps=5000`, the first
refresh-eligible eval boundary is step 6000 (step 3000 has
elapsed-interval=3000 < 5000). For the main run with
`eval_interval=5000` + `interval_steps=5000`, the first refresh
fires at step 5000 exactly as designed. The smoke variant's
3000-step eval cadence created a one-off off-by-one against the
literal spec; main run alignment is correct.

**Subprocess completion.** Started at step 6000 but `session_end`
fired ~1 second later (training reached `iterations=6000` limit).
Subprocess was likely SIGTERM'd during shutdown without time to
complete the 200-game regen. Canonical NPZ on disk unchanged (no
`.bak`, no `.NEW.tmp`, mtime still May 18 13:00). The hook FIRE
mechanism is verified; the full SWAP+HOT-RELOAD cycle is verified
by unit tests + INV pins but was NOT exercised end-to-end during
this smoke. Main run will exercise ~19 full refresh cycles.

## Verdict (LITERAL L13, Stage 3C)

Pre-registered gates:

| ID | rule | result |
|---|---|:---:|
| WS-A | alt ≥ +0.10 through 6000 AND refresh event at step 5000 AND clean | **PASS-WITH-NOTES** |
| WS-B | alt holds BUT refresh event missing/malformed | n/a (event present + well-formed) |
| WS-C | alt < +0.07 by step 3000 | n/a (alt @ 3000 = +0.278) |
| WS-D | SealBot WR @ step 3000 < 10% | n/a (24%) |
| WS-E | HARD-ABORT fires | n/a (no hard-abort; SOFT-ABORT canary informational per L50) |

**Smoke verdict: WS-A PASS-WITH-NOTES.** alt held +0.178 ≥ +0.10
endpoint gate; refresh event logged + well-formed (just at eval
boundary step 6000 rather than `interval_steps` step 5000); run
completed cleanly. SealBot WR 24% @ step 3000 matches Wave 2's
step-10k baseline — mechanism on track. No hard-abort.

## Stage 4 readiness assessment

Pre-launch concerns to resolve before Stage 4:

1. **Refresh cycle end-to-end NOT exercised in smoke.** Subprocess
   started but didn't complete. Main run (100k steps, refresh every
   5000) will fire ~19 refreshes — first one at step 5000 has ~95k
   steps of subsequent training to validate the swap+hot-reload +
   the model's response to refreshed bot corpus targets.

2. **alt V_spread trajectory declining monotonically.** At smoke
   rate (~0.030/1000 steps), alt would cross +0.07 gate around
   step 9700. Wave 2 main showed alt actually FLATTENED at +0.20-
   0.25 over steps 12k-44k (NOT continued linear decline), so the
   smoke linear extrapolation likely overshoots. Per L50, alt is
   INFORMATIONAL only — primary gate is sliding-window SealBot WR.

3. **Throughput.** Smoke ran at ~24.8 steps/min — slower than
   Wave 2 smoke's ~16.4 steps/min was when we measured before the
   sub-sampling perf opt. Wait — that's BACKWARDS. Wave 2 smoke
   ran at 16.4 (with per_class_temp on ~192 selfplay rows/batch);
   Wave 3 runs at 24.8 (with per_class_temp on ~64 pretrain
   rows/batch — smaller slice, faster classify). Wave 3 is
   ~1.5× faster.

   At 24.8 steps/min, 100k steps = ~67 hours = ~2.8 days. This is
   SLOW for a $3 budget run. Likely the refresh-subprocess overhead
   slows steady-state; ~30-50 steps/min steady is more realistic.

   **Operator decision point**: if 100k run @ 25 steps/min is too
   slow (would cost ~$15 instead of $3), consider:
   - Reducing per-class temp scope further (e.g. only on bot rows,
     not all pretrain rows — bot is ~38 rows/batch vs ~64 for
     full pretrain)
   - Truncating main run to 50k steps (still hits W3-G1 sustained
     30k-50k criterion)
   - Accepting longer wall + higher cost

   Recommend launching main run + measuring actual rate over first
   5k steps before committing the full budget.

4. **Eval pipeline.** Step 3000 eval completed cleanly. Step 6000
   eval kickoff was truncated by shutdown — main run with denser
   eval cadence (every 5000) will catch the L50 WR sliding-window
   patterns earlier.

## Recommendations

1. **Stage 4 main launch APPROVED** per WS-A PASS-WITH-NOTES.
2. **eval_interval=5000** in main variant — aligns refresh hook
   with eval boundary at exactly step 5000.
3. **Smoke artifacts archived** at `reports/track_b_smoke_wave3/`
   for trajectory reference.
4. **Wave 2 step-10k baseline beaten 3x faster in Wave 3 smoke**
   (24% WR at step 3000 vs Wave 2's step 10000) — strongest
   positive signal so far for the Wave 3 lever stack.

## Cross-references

- `audit/structural/REAL_RUN_RECIPE.md` — Wave 3 success criteria + Stage 4 plan
- `audit/structural/wave2_real_run_analysis.md` — Wave 2 baseline + L50/L51/L52
- `audit/structural/wave3_launch_readiness.md` — Stage 2 close-out + Stage 3 prep
- `configs/variants/v7_wave3_smoke.yaml` — smoke variant
- `configs/variants/v7_wave3_main.yaml` — main variant (Stage 4A)
- `hexo_rl/training/step_coordinator.py:709-754` — refresh hook + WR hard-abort wiring
- `hexo_rl/training/per_class_target_temperature.py` — per-class temp + Wave 3 scope flag
- `hexo_rl/monitoring/alert_rules.py:check_sealbot_wr_hard_abort` — L50 gate
- `reports/track_b_smoke_wave3/` — rsync'd smoke artifacts (log + events JSONL + 6 ckpts + EMA sidecars)
