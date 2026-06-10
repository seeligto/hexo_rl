# §S181-AUDIT Wave 2 — Stage 4 pre-launch smoke verdict

3000-step instrumented smoke of the V-B-A `uniform_self` lever stack on
the v7full anchor. Pre-registered Stage 4C verdict gates apply LITERAL
L13. Companion docs: `audit/structural/REAL_RUN_RECIPE.md` (success
criteria), `audit/structural/track_b/B_verdict_synthesis.md` (V-B
verdict + routing rationale), `audit/structural/track_d_pipeline_regression.md`
(smoking-gun ranking).

## Run identity

| field | value |
|---|---|
| host | vast.ai 5080 (REMOTE_HOST:REMOTE_PORT) |
| workdir | $REPO_ROOT/ |
| branch | phase4.5/s181_wave2_lever_vba_selfplay 5af2115 |
| variant | v7_real_run_smoke (`configs/variants/v7_real_run_smoke.yaml`) |
| anchor | bootstrap_model_v7full.pt SHA 568d8a33…d61e8e98 |
| encoding | v7full (board=19, planes=8, single-window) |
| iterations | 3000 |
| run_id | 6d4337f8b8d34c819be69054918a7642 |
| launch | 19:29 UTC, 2026-05-23 |
| completion | 22:32 UTC, 2026-05-23 |
| wall | 3 h 03 min (10931 sec, ~16.4 steps/min) |
| games played | 1500 |
| buffer @ end | 159 344 positions |
| ckpts saved | 6 (steps 500, 1000, 1500, 2000, 2500, 3000) |
| policy loss | 2.22 → 1.93 (-0.29) |
| log | reports/track_b_smoke/logs/s181_smoke_20260523_1929.log |
| events JSONL | reports/track_b_smoke/logs/events_6d4337f8…d54c.jsonl |

Lever stack enabled (delta vs §S180b 3-knob recipe):
- encoding migration v6 → v7full
- EMA of weights (decay 0.999, every 10 optimizer steps) on self-play / eval / promotion dispatch
- per-class target temperature on selfplay slice (T_colony=1.5, others 1.0,
  pretrain slice untouched)

## Canary trajectory (dual-bank V_spread)

| step | T3 | alt | T3 Δ | alt Δ | dual-canary both_pass | S-A gate (alt) |
|---:|---:|---:|---:|---:|:---:|:---:|
| 500  | +0.2227 | +0.3893 |    —  |    —  | PASS | PASS |
| 1000 | +0.1955 | +0.3577 | -0.027 | -0.032 | FAIL (T3 < 0.20) | PASS |
| 1500 | +0.1789 | +0.3341 | -0.017 | -0.024 | FAIL              | PASS |
| 2000 | +0.1545 | +0.3104 | -0.024 | -0.024 | FAIL              | PASS |
| 2500 | +0.1338 | +0.2961 | -0.021 | -0.014 | FAIL              | PASS |
| **3000** | **+0.1183** | **+0.2821** | -0.016 | -0.014 | FAIL | **PASS** |

Notes on `both_pass: FAIL` rows: T3 crosses the dual-canary +0.20
SOFT-ABORT gate at step 1000 and never recovers — but per L48 T3 is
**bank-specific** (synthetic positions calibrated on the v6 anchor's
value head; v7full discriminates them less sharply per C-LITE-1).
Dispatcher Stage 4C verdicts are explicitly **alt-only** because alt is
the corpus-grounded reference. The dual-canary's both_pass signal is a
dashboard alert, not a verdict trigger.

## Verdict (LITERAL L13, Stage 4C)

Pre-registered gates:

| ID | rule | result |
|---|---|:---:|
| S-A | alt ≥ +0.10 at step 3000 **AND** ≥ +0.07 sustained throughout | **PASS** |
| S-B | alt holds > +0.07 throughout, ends +0.07-0.10 | n/a |
| S-C | alt crashes below +0.07 by step 1500 | CLEARED (alt @ 1500 = +0.3341) |
| S-D | alt crashes below +0.07 by step 500 | CLEARED (alt @ 500 = +0.3893) |
| S-E | other (grad/policy collapse) | clean (grad_norm 1.2-1.7, policy loss falling) |

**Smoke verdict: S-A.** alt @ step 3000 = +0.2821 is ~3× the +0.10
endpoint gate; alt sustained throughout (lowest value = +0.2821 at
step 3000) is ~4× the +0.07 sustained gate. Decay rate decelerated
from -0.032 / 500 steps (early) to -0.014 / 500 steps (late) — alt
plateauing near +0.28, no late-run cliff.

## B4 baseline comparison (mechanism validation)

| step | B4 alt | smoke alt | Δ | B4 T3 | smoke T3 | Δ |
|---:|---:|---:|---:|---:|---:|---:|
| 500 | +0.245 | +0.3893 | +0.144 | -0.039 | +0.2227 | +0.262 |

B4 step 500 was the first canary fire on the §S180b 3-knob recipe
running under v6 encoding; it triggered SOFT-ABORT on T3 (already
crashed below the +0.20 gate) and stayed barely above the alt floor
(+0.245). The smoke under v7full encoding + EMA + per-class temp
delivers:

- **alt +0.14 better at first checkpoint** (~60 % bigger margin)
- **T3 +0.26 better at first checkpoint** (NOT collapsed)
- **alt sustained ~+0.30 across 3000 steps** vs B4 trajectory which
  would have crashed near zero by step 2-4k per FU-1.5 L44 (86 % of
  V_spread loss in 0→2k window)

The B4 → smoke delta isolates the Wave 2 lever stack (v7full +
EMA + per-class temp) as the load-bearing change. The dispatcher
Stage 4C smoke gate was designed to falsify the hypothesis; instead
the data strongly corroborates it.

## Stage 5 readiness assessment

S-A → proceed to Stage 5 main run per dispatcher routing.

### Pre-launch concerns to resolve before Stage 5

1. **Throughput regression**. Smoke ran at 16.4 steps/min vs B4's
   37 steps/min (~2.3× slower). The CPU-side per-class classify
   (`scripts/structural_diagnosis/track_a/position_classifier.classify_batch`)
   is a pure python loop over hex axes per state; for batch 256 with
   ~75% selfplay slice = 192 classifications/batch + a GPU→CPU
   sync per batch. GPU still ran at 94 % util — the bottleneck is
   the train-step's CPU side.

   100k steps at 16.4 steps/min = **~6100 min = ~102 h ≈ 4.3 days
   wall**. At vast hourly $0.21/h = **~$22**. Dispatcher original
   estimate was $3 / ~14 h based on B4 baseline rate.

   Mitigations (cheapest first):

   - **Sub-sample selfplay rows in per-class temp** (~5 LOC).
     Classify and apply temperature to ~10-25 % of selfplay rows per
     batch; rows outside the sample keep T=1.0. Statistical effect
     scales linearly with sample rate; perf overhead drops ~4-10×.
   - **Vectorize classify_batch** (~50-100 LOC). Replace the python
     per-state loop with batched numpy ops over (B, 8, 19, 19). Likely
     ~5-10× speedup. Higher complexity than sub-sampling.
   - **Increase update_every for per-class temp** to 2-4 (apply
     classification + temperature every N batches, not every batch).
     Cheap; effective rate × (1/N).
   - **Accept the slower rate** and budget ~$22-25 + 4 days for the
     full 100k run.

2. **EMA sidecar disk cost** at 50 checkpoints (every 2k steps, plus
   eval/anchor-every preservation). At 17 MB per ckpt × 2 (raw + EMA)
   = ~5 GB across the run. Vast disk-guard warn_gb=10.0 — well within
   margin; not a concern.

3. **Buffer hygiene**. Smoke launched with clean `replay_buffer.bin`
   (verified `buffer_size_before_corpus_load: 0`). Stage 5 must
   similarly clean before launch — both smoke ckpts AND smoke's
   `replay_buffer.bin` should be removed (already rsync'd to local
   `reports/track_b_smoke/checkpoints/` for audit retention).

### Recommendations

1. **Apply sub-sampling optimization to per-class temp** before Stage 5
   launch (≤5 LOC, low risk). Pick sample_rate=0.20 — 5× perf
   speedup, classify ~38 rows/batch instead of 192. Expected throughput
   recovery to ~50-60 steps/min (above B4 baseline since no
   per_source_grad_norm overhead).

2. **Stage 5 budget revision**: with the optimization, ~$3-5 / ~14-22h
   wall. Without optimization, ~$22 / ~100h wall — operator decides
   whether to optimize or accept.

3. **Surface to operator** for explicit Stage 5 go/no-go gate per
   dispatcher hard constraint ("Operator-mediated gates at every stage
   transition. NEVER auto-launch the next stage").

## Cross-references

- `audit/structural/REAL_RUN_RECIPE.md` — §4 success criteria, §5 stop conditions
- `audit/structural/track_b/B_verdict_synthesis.md` — V-B verdict synthesis
- `audit/structural/track_b/B_track_d_xref.md` — Track D candidate confrontation
- `hexo_rl/training/per_class_target_temperature.py` — lever implementation
- `hexo_rl/training/ema.py` — EMA infrastructure
- `configs/variants/v7_real_run_smoke.yaml` — smoke variant
- `reports/track_b_smoke/` — rsync'd artifacts (log + events JSONL + 6 ckpts)
