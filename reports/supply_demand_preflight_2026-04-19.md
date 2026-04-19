# Supply/demand preflight — 2026-04-19

Phase 1 of the §107 post-W1 sustained run prompt. Decide
`training_steps_per_game` for laptop `gumbel_targets` from measured
idle % and supply/demand ratio.

## Method

Three 500-step smokes from `bootstrap_model.pt`, `gumbel_targets`
variant, laptop (8845HS + RTX 4060, 14 workers). Config otherwise at
master post-§107 instrumentation (commits 77699f1 → 17ef5ee). No
`--no-dashboard` suppression; `monitoring.log_policy_target_metrics:
true` + `log_investigation_metrics: true`. The buffer was cleared
between runs that start from bootstrap; Smoke-B inherited Smoke-A's
97k-position buffer (see §Confound).

Analyzer: `scripts/analyze_supply_demand_smoke.py`. Metric
definitions:

- **idle_frac** — count of `waiting_for_games` + `warmup` structlog
  events times the 5 s throttle interval in `loop.py`, divided by wall
  seconds from first post-step-100 `train_step` to end. Lower bound on
  idle time (events can fire at <5 s intervals via the re-entry path).
- **supply_demand_ratio** — plies produced (sum over game_complete) /
  positions consumed (`n_train_steps × batch_size`). 1.0 = balanced;
  <1.0 = trainer over-samples the buffer.
- **policy_loss_slope** — OLS over `train_step.policy_loss` for steps
  ≤ 1000 (all 500 in scope).

## Results

| Smoke | tsp | wall_sec | n_games | idle_frac | ratio | slope | loss first→last | avg_plies |
|---|---|---|---|---|---|---|---|---|
| A | 2.0 | 1724 | 250 | **0.992** | 0.18 | -3.8e-4 | 1.89 → 1.97 (+0.08) | 88.8 |
| B | 1.5 | 1888 | 253 | 0.988 | 0.20 | -3.4e-4 | 2.17 → 1.91 (-0.26) | 98.4 |
| C | 1.0 | 3516 | 500 | 0.993 | 0.39 | -3.5e-4 | 1.90 → 1.64 (-0.26) | 101.3 |

All three are supply-bottlenecked. Idle fraction is ~99% regardless
of tsp — consistent with `0.145 games/sec × 14 workers` supply rate
being the dominant constraint, not trainer compute capacity. The
prompt's "idle < 15%" criterion is unreachable on this hardware at
this model size; no tsp value satisfies it.

## Confound

Smoke-A→Smoke-B ran back-to-back without wiping the persisted replay
buffer. Smoke-B started with Smoke-A's 97k positions in-buffer, which
lets the trainer run immediately without waiting on self-play fill.
That shifts Smoke-B's starting loss (2.17) vs. fresh-buffer Smoke-A /
Smoke-C (1.89 / 1.90). `checkpoints/replay_buffer.bin` was removed
before Smoke-C to restore fair comparison to Smoke-A. Direct A↔C
comparison is preferred; B is a bracketing datapoint.

## Decision

**Chosen: `training_steps_per_game = 1.5`.**

Justification (per prompt tie-break order, with idle-threshold rule
acknowledged unachievable):

1. **Smoke-A (tsp=2.0) fails the directional floor.** Fresh-buffer
   run showed policy loss *rising* 1.89 → 1.97 over 500 steps.
   Trainer is over-sampling stale buffer at 2 updates per new game
   and regressing the model. Rejecting.
2. **Smoke-C (tsp=1.0) matches Smoke-B's improvement at 2× wall
   time.** Both hit `Δpolicy_loss ≈ −0.26`, but Smoke-C consumed
   3516 s vs. 1888 s. For the 50k-step sustained-run target, tsp=1.0
   projects to ≈ 97 h wall-time, tsp=1.5 to ≈ 52 h, tsp=2.0 to ≈ 48 h
   (but harmful). Progress-per-hour: tsp=1.5 = −0.50 /h, tsp=1.0 =
   −0.26 /h, tsp=2.0 = +0.17 /h (regressing).
3. **tsp=1.5 is the Pareto point.** Best wall-clock progress per
   hour AND an above-floor supply ratio (0.20 vs. 0.18 at 2.0 —
   smaller margin but monotonic improvement as tsp drops). No regime
   evidence (fresh-buffer Smoke-A) would argue for staying at 2.0.

**Did not re-smoke at tsp=1.0 after prompt's "both-fail" fallback.**
The fallback exists to rescue a decision when neither 2.0 nor 1.5
produces a learning signal. Smoke-B at tsp=1.5 already produces
`Δloss = −0.26`, so the fallback condition (both failing
idle AND failing ratio AND failing to learn) does not apply — only
idle fails, and idle is supply-hardware-bound. Re-running at tsp=1.0
(Smoke-C) was done opportunistically and confirms the 1.5 choice
rather than unsetting it.

## Sustained-run scoping implication

At tsp=1.5 and ~953 steps/hr observed rate, a 50k-step run projects
to ≈ 52 GPU-hours. Prompt target is 35 GPU-hours — the hardware
doesn't hit that for tsp ≥ 1.0 at this model size. Options:

- Accept ≈ 52 h wall-time for 50k-step run at tsp=1.5 (preferred,
  continues to produce learning signal).
- Cut target to 30k steps → ≈ 31 h wall-time at tsp=1.5. Also fine if
  milestone cadence is already tight.

Recommend option 1 (50k @ tsp=1.5, ~52 h) unless the operator has a
hard ship-before deadline that breaks at 52 h — §107 prompt does not
cite one. Pick this up on the launch commit.

## Config change

`configs/variants/gumbel_targets.yaml`: `training_steps_per_game:
2.0 → 1.5`. Single-line change; `configs/variants/gumbel_targets_desktop.yaml`
and `configs/training.yaml` base left alone (desktop variants still
run 2.0 — desktop workers are 10, not 14, and the measurement above
does not generalise).
