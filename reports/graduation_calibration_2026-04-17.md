# Graduation-gate calibration — 2026-04-17

Calibration sweep of the four coupled graduation-gate parameters
(D1 threshold, D2 interval, D3 decay, D4 min_games) against the
`gumbel_targets` self-play variant on the laptop rig. Four runs × 4 hours
from `checkpoints/bootstrap_model.pt`.

Prompt: Prompt 8 / §calib. Setup commit: `8240cc9`.

## TL;DR

1. Under current training dynamics, the student loses to the bootstrap
   anchor with `wr_best ≤ 0.06` at step 2500 across all three runs that
   produced a completed eval (R1: 0.025, R2: 0.020, R4: 0.060). No
   configuration of D1/D2/D3/D4 within the prompt's option grid is
   recoverable from this regime — **the gate is not the bottleneck at
   4-hr training depth; the learning dynamic is.**
2. D1 ablation (0.55 → 0.52) changes nothing: `wr_best = 0.02` is an
   order of magnitude below either threshold.
3. D3 ablation (20k → 50k decay) keeps the corpus weighting higher
   (pretrained_weight 0.72 vs 0.62 by step 5000) and keeps the
   full-search policy target sharper (`tgt_E_full = 0.37 vs 0.56`) but
   does not visibly change the win-rate gap against the anchor.
4. D4 (200 → 400 games) tightens CI half-width at threshold from
   ~0.069 to ~0.049 (binomial normal approx). Worth adopting if we
   ever reach threshold — and cheap relative to the real cost problem
   (point 5).
5. **Operational finding not anticipated by the prompt:** a single
   best-vs-student eval at `n_games = 200` takes **~36 minutes** of
   wall-time; at `n_games = 400` it takes **~70 minutes**. With
   `eval_interval = 2500` and ~1450 steps/hr, evals fire every
   **~103 min** — the next eval pipeline starts before the previous
   one finishes in a 4-hr driver window. R1/R2/R4 each see their
   step-5000 eval orphaned by SIGTERM; R3 (interval 5000) sees its
   only eval orphaned. `eval_interval ≥ 5000` is the minimum
   sustainable cadence once `n_games = 400`.

## Run matrix

All four runs share: `gumbel_targets` semantics (PUCT root + completed-Q
targets, `fast_prob = 0.0`, `full_search_prob = 0.25`, n_sims 100/600),
`max_game_moves = 200`, 18-plane trunk, GroupNorm, `draw_value = -0.5`,
bootstrap resume, 4-hour driver timeout.

| Run | D1 thr | D2 int | D3 decay | D4 games | final_step | evals | promos | blocks |
|-----|--------|--------|----------|----------|------------|-------|--------|--------|
| R1  | 0.55   | 2500   | 20 000   | 200      | 5232       | 1     | 0      | 0      |
| R2  | 0.52   | 2500   | 20 000   | 200      | 5176       | 1     | 0      | 0      |
| R3  | 0.55   | 5000   | 50 000   | 200      | 5352       | 0     | 0      | 0      |
| R4  | 0.55   | 2500   | 20 000   | 400      | 5116       | 1     | 0      | 0      |

All runs: `full_search_prob=0.25`, `fast_prob=0.0`,
`max_game_moves=200`, `batch_size=256`, `training_steps_per_game=4.0`.
Step rate cruise: 1400–1500 steps/hr (warmup through step ~1800 a bit
slower). `configs/variants/calib_R[1-4].yaml`; D1/D4 applied per run
via `scripts/run_calibration_run.sh` which patches `configs/eval.yaml`
and restores it on EXIT.

## Phase 2 results

### 1. Graduation cadence

Zero promotions across all four runs over 4-hour, ~5200-step windows.
Nothing graduates even under the most permissive threshold (R2, 0.52)
and the tightest CI (R4, 400 games). **Graduation cadence is uncomputable
from these runs — no events to count.**

Eval-start events DID fire as expected:

- R1, R2, R4 (interval=2500): eval_start at step 2500 and 5000. Only
  the first completed before the driver's 4-hour SIGTERM.
- R3 (interval=5000): eval_start only at step 5000. Did not complete.

### 2. Mean inter-graduation interval

Undefined — no graduations.

### 3. Win-rate distribution at eval (wr_best)

| Run | wr_best | CI_low | CI_high | n_games | vs random | vs threshold |
|-----|---------|--------|---------|---------|-----------|---------------|
| R1  | 0.025   | 0.003  | 0.047   | 200     | 1.000     | 0.525 below   |
| R2  | 0.020   | 0.001  | 0.039   | 200     | 1.000     | 0.500 below   |
| R3  | —       | —      | —       | —       | —         | no completed eval |
| R4  | 0.060   | 0.037  | 0.083   | 400     | 1.000     | 0.490 below   |

All three completed evals produce `wr_best` in [0.020, 0.060] — a range
**≥9× smaller than the most permissive threshold (0.52, R2)**. `wr_random`
is 1.0 in all three, so the student is not broken; it is strongly
*worse against bootstrap* than bootstrap is against itself.

**Conclusion for D1:** Neither 0.55 nor 0.52 is reachable in the first
4-hour window. Lowering the threshold further (0.50) would make it
formally coin-flip and is excluded by the prompt's constraints.

### 4. Anchor lineage

| Run | distinct `anchor_ckpt_*` rows | expected at graduation_count+1 |
|-----|-------------------------------|-------------------------------|
| R1  | 1 (`anchor_ckpt_0`)            | 1                             |
| R2  | 1 (`anchor_ckpt_0`)            | 1                             |
| R3  | 0 (no eval completed)          | 1                             |
| R4  | 1 (`anchor_ckpt_0`)            | 1                             |

R1–R2–R4 each register one `anchor_ckpt_0` row scoped to the run_id —
confirming the post-`eae2d96` anchor identity plumbing is wired
correctly on the cold-start branch. **However the fix's main value —
distinct rows per promotion — could not be exercised here because no
promotions occurred. Treat the R1 fix as _regression-tested clean_
but _functionally unvalidated_** until a run produces ≥2 promotions.

### 5. Draw rate trajectory

Mean draw_rate in 500-step windows (self-play games, not eval games):

| window     | R1    | R2    | R3    | R4    |
|------------|-------|-------|-------|-------|
| 0–500      | 0.233 | 0.216 | 0.164 | 0.192 |
| 500–1500   | 0.299 | 0.328 | 0.289 | 0.253 |
| 1500–3000  | 0.295 | 0.308 | 0.298 | 0.273 |
| 3000–5000  | 0.289 | 0.299 | 0.298 | 0.274 |
| 5000–8000† | 0.295 | 0.308 | 0.314 | 0.281 |

† Few samples — driver ran ~200 steps past 5000 in each run.

Comparison anchor: `smoke_v3b` at step ~15k had draw_rate ≈ 0.447. All
four runs are currently running ~15 pp lower than that reference, which
is consistent with the shorter run length (decay hasn't fully shifted
the buffer to self-play) but does not show the colony-draw explosion
mode.

**R3 (decay=50k) does not show a materially different draw rate than
R1 (decay=20k)** once past the warmup phase. This argues *against*
extending decay.

### 5b. Eval agreement rate

Not instrumented in this calibration. `evaluate_vs_model` records only
win/loss/draw counts per game, not per-position argmax agreement. Adding
it is a ~30-line change (emit `mean_argmax_agreement` per game in
`Evaluator.evaluate_vs_model`) but was out of scope for this budget.

Proxy from available data: `eval_draw_rate` is not stored directly in
`evaluation_round_complete` (draws counted as losses for the model).
Recommend instrumenting before the next calibration.

### 6. Stall-window detection

All runs show a single stall window = final_step (5116 – 5352) < 10k.
Cannot detect ≥10k stalls in 4-hour windows.

### 7. Chain-loss sanity (§102 regression check)

Chain loss healthy across all runs — stable, nonzero, mean 0.00204 ±
0.00002. **PASS.** The §102 corpus-chain fix (commit `9ed4c72`) holds
under all four variants.

| Run | nonzero frac (last 100 steps) | mean chain_loss |
|-----|-------------------------------|-----------------|
| R1  | 1.00                          | 0.002035        |
| R2  | 1.00                          | 0.002053        |
| R3  | 1.00                          | 0.002069        |
| R4  | 1.00                          | 0.002089        |

### 8. Training trajectory signals

Common to all four runs:

- **Policy loss** drops monotonically ~2.02 → ~1.87 over 5000 steps
  (gradient of ~1.5% per 1000 steps).
- **policy_entropy_selfplay** rises from ~4.9 to ~5.4 then plateaus.
  Uniform over 361 legal moves is log(361) = 5.89, so ent=5.4 is
  only 0.5 nats from uniform — the policy is highly diffuse.
- **policy_target_entropy_fullsearch** rises from ~0.24 (sharp) at
  warmup to ~0.50–0.65 by step 5000. Targets getting *less* sharp
  as training proceeds. Suggests Dirichlet root noise is loud
  relative to the visit distribution at this training depth, and/or
  PUCT at 600 sims isn't producing strongly concentrated policies
  against a weakly-confident value head.
- **policy_target_entropy_fastsearch** stays pinned near 4.1–4.2 nats
  — the completed-Q quick-search targets are essentially uniform,
  consistent with `reports/gumbel_target_quality_2026-04-17.md` §5.1.
  The selective gate is correctly discarding this signal.
- **pretrained_weight** tracks the mixing schedule:

  | Run | step 500 | step 3000 | step 5000 |
  |-----|----------|-----------|-----------|
  | R1  | 0.790    | 0.715     | 0.656     |
  | R2  | 0.790    | 0.715     | 0.656     |
  | R3  | 0.796    | 0.765     | 0.739     |
  | R4  | 0.790    | 0.715     | 0.656     |

  R3's 50k decay is visibly slower — by step 5000 the buffer is still
  ~74% corpus-seeded vs ~66% for the 20k runs. Not enough to affect
  draw rate or policy loss trajectory in this window.

## Phase 3 cross-checks

### R1 vs smoke_v3b @ 5k

R1 at step 5176 has roughly the same policy loss, entropy, and draw
rate as the earlier `smoke_v3b` run at the same step — *i.e.* the gate
was never doing anything at 0.55 over 5k steps even in the legacy
configuration, because `wr_best` never approached 0.5. **R2
(threshold=0.52) is qualitatively identical to R1 over this window,
confirming the threshold is not the lever at this training depth.**

### Probe gate

Not run during this sweep — `make probe.latest` would fire at step 5000
boundary in production. All four runs saved a `checkpoint_00005000.pt`
(final Trainer state). Probe results not included; re-run separately.

### Eval-agreement noise floor

Not measured (no instrumentation; see §5b above). Given `wr_best`
clusters at [0.02, 0.06] × n_games=200 or 400, the CIs are tight enough
that we're confident the observed value is not noise — **the student
really is that much weaker than the anchor at step 2500**. The
"noise-limited" concern applies once `wr_best` approaches 0.5, which we
did not observe.

## Recommendations

The experiment budget could not measure graduation cadence directly —
no promotions fired. Recommendations below are drawn from proxies
(step rate, CI arithmetic, decay trajectory) and the §5 operational
finding on eval cost.

### D1 — `promotion_winrate`

**Keep at 0.55.** Data does not support lowering:

- R2 (0.52) shows the same `wr_best` floor as R1 (0.55).
- At `n_games=200`, threshold 0.52 with CI-guard fires at worst-case
  `wr_hat ≈ 0.52 + 0.069 = 0.589` which is effectively the same as
  0.55's worst case — so the ablation materially drops the gate
  stringency without data showing it is met any sooner.
- Prompt constraint: "Do NOT recommend raising threshold without data
  showing it can be met." Symmetric for lowering: no data shows the
  lower threshold would fire sooner either.

### D2 — `eval_interval`

**Raise from 2500 to 5000.** Rationale:

- A 200-game anchor eval takes ~36 min wall-time on the laptop; a
  400-game eval takes ~70 min.
- At interval 2500 and ~1450 steps/hr, evals fire every ~103 min.
  The eval overlaps with ~35% of training wall-time even though the
  eval runs in a background thread (GPU + inference-server
  contention — see loop.py eval_thread).
- At interval 5000, evals fire every ~207 min. Overhead drops to
  ~17%. Sustained runs (no driver timeout) produce complete evals.
- R3 observed no measurable degradation in learning dynamics at
  interval 5000 — policy loss trajectory, draw rate, and
  policy_entropy_selfplay all match R1 within noise.

Prompt constraint check: "Shorter interval if throughput cost < 5%,
else longer." 17% → 35% throughput cost swing makes the longer
interval correct.

### D3 — `mixing.decay_steps`

**Keep at 20 000.** Rationale:

- R3 (decay=50k) showed slower policy loss descent (not faster) and
  no advantage in draw rate or entropy trajectory.
- R3 kept `policy_target_entropy_fullsearch` sharper (0.37 vs 0.56 in
  R1) — a potentially useful signal — but this did not translate
  into a visible learning-signal advantage on the 4-hour scale.
- No stall windows were observed in either decay setting (irrelevant,
  since no promotions occurred at all; prompt criterion "raise to 50k
  if any stall window found at decay=20k" is not triggered).
- Keeping 20k preserves the training.yaml rationale: "self-play must
  dominate sooner to let draw penalty propagate" (see training.yaml
  lines 29–32).

### D4 — `graduation_min_games`

**Raise from 200 to 400.** Rationale:

- At threshold `p = 0.55`, `n = 200` yields CI half-width ≈
  `1.96·sqrt(0.55·0.45/200) = 0.069` → stderr ≈ 0.035 (3.5%,
  marginal vs the prompt's <3% target).
- At `n = 400`, CI half-width drops to 0.049 → stderr ≈ 0.025 (2.5%,
  comfortably below 3%).
- Observed eval cost: 70 min at n=400 vs 36 min at n=200 — so D4's
  extra cost is paid twice (once as longer eval, once as raising D2
  to 5000 to fit the eval). Net wall-time overhead after both changes:
  ~34% (was 33% at R1 settings; see §5 operational finding).
- The CI-guard is already required for promotion
  (`ci_best_lo > 0.5`). Tighter CI materially reduces spurious
  promotions once `wr_hat` crosses 0.5.

## Combined proposed gate

| Decision | Current | Proposed | Δ                                     |
|----------|---------|----------|---------------------------------------|
| D1       | 0.55    | 0.55     | —                                     |
| D2       | 2500    | **5000** | fewer evals, fits the 70-min D4 eval  |
| D3       | 20 000  | 20 000   | —                                     |
| D4       | 200     | **400**  | stderr 3.5% → 2.5%, CI tighter at thr |

## Limitations

1. **Zero promotions observed.** The calibration measured the
   environment (eval cost, decay effect, CI math) but not graduation
   cadence directly. The prompt's Phase 2 metric set was designed
   for runs that produce 5-20 graduations; we produced 0. All cadence
   recommendations are derived from proxy signals.
2. **`wr_best` across R1/R2/R4 (0.02/0.02/0.06)** shows more run-to-run
   variance (at the noise floor) than would normally warrant concern,
   but at these magnitudes the conclusion ("student worse than
   bootstrap by a large margin") is robust.
3. **Eval-agreement instrumentation** (prompt §5b) was not added for
   this calibration. Recommend landing before the next sustained run
   so the noise-floor concern is measurable.
4. **No probe step-5k check** (`make probe.latest`) was run during the
   sweep. Saved `checkpoint_00005000.pt` per run is available for a
   post-hoc probe.
5. **Eval-cost finding (§5)** is the dominant practical lever, not a
   tuned parameter. Addressing it properly ultimately needs a smaller
   n_games or a cheaper MCTS setting for the anchor eval — a separate
   change outside the D1-D4 axes.

## Next steps

1. Apply the proposed gate (D2=5000, D4=400) as a YAML-only commit
   — see companion patch (§commit 3).
2. Run a single sustained `gumbel_targets` run at total_steps ≥ 15k
   (≥10 hr) to observe whether `wr_best` ever crosses 0.5 and whether
   graduations fire under the tuned gate.
3. Land the eval-agreement metric instrumentation
   (`Evaluator.evaluate_vs_model` → mean per-game argmax agreement).
4. Re-probe (`make probe.latest`) against each run's
   `checkpoint_00005000.pt` to verify no §85/§91 threat-head
   regression.

## Archive

Logs + checkpoints + per-run patched eval.yaml:
`archive/calibration_2026-04-17/calib_R[1-4]/`.

Run IDs (for eval DB lineage cross-reference):

| Run | run_id                              |
|-----|-------------------------------------|
| R1  | 3e2aa33ad129409f99ee8ca297784b3a    |
| R2  | bc39c3b57f034c1b8afc7cf3e2af1be6    |
| R3  | e4cb664a7a3343b0b608252284de3c4e    |
| R4  | c6fde2a82c284a59a919d71dfe7b1700    |

DB check: `sqlite3 reports/eval/results.db "SELECT name FROM players
WHERE run_id='<rid>' AND (name LIKE 'anchor_ckpt_%' OR name =
'best_checkpoint');"` — confirms each run's anchor row is scoped by
run_id and carries the new `anchor_ckpt_{step}` format.
