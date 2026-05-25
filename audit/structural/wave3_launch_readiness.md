# §S181-AUDIT Wave 3 — launch readiness assessment

**Status.** Stage 2 design close. All 4 Wave 3 levers landed on
`phase4.5/s181_wave3_design`. REAL_RUN_RECIPE v2 in
`audit/structural/REAL_RUN_RECIPE.md`. Operator approval required to
proceed to Stage 3 (pre-launch smoke).

## Levers landed (Stage 2)

| Stage | lever | commit | files | LOC | tests |
|---|---|---|---|---|---|
| 2A | Bot corpus refresh hook (activation per L51 / Track D C4) | `b6983bf` | step_coordinator.py + batch_assembly.py + loop.py + 2 new test files | +1454 / -67 | 30 |
| 2B | Sliding-window SealBot WR hard-abort gate (L50) | `133d25a` | alert_rules.py + config.py + step_coordinator.py + 2 new test files | +481 | 23 + 4 INV pin |
| 2C | Per-class target temperature + Wave 3 scope (L52) | `63e8c5c` | per_class_target_temperature.py + trainer.py + training.yaml + 2 new test files | +748 | 20 + 4 INV pin |
| 2D | REAL_RUN_RECIPE v2 (refresh hook + rolling-mean PRIMARY) | `a53cbca` | REAL_RUN_RECIPE.md | +147 / -77 | n/a (doc) |

Wave 2 reference (Stage 1 close on master): EMA infrastructure
(`95624af`), b4_analysis chain (`e973d1f` + `6fab8c9`), Wave 2 audit
docs (`96562fa`), sprint log entry (`f840a28`).

Branch state: `phase4.5/s181_wave3_design` HEAD `133d25a`; 4 commits
ahead of master `f840a28`. All commits on this branch are pure Wave 3
scope; no merge conflict expected when FF-merging to master post-run.

## Test summary

- 30 refresh hook tests (TC8 unit + INV-S179c-1..4 + failure paths)
- 23 hard-abort gate behaviour tests + 4 INV pin tests
- 20 per-class temp behaviour tests (14 backward-compat + 6 Wave 3 scope) + 4 INV pin tests
- 110/110 broader training + monitoring tests pass
- Full `make test.py`: 1621 passed / 0 failed (subagent post-Stage-2A
  verification, excluding pre-existing unrelated `tests/test_analyze_api.py`
  failures from an untracked `bootstrap_model_v6_a2.pt` shape mismatch
  — environmental, not Wave-3-related)

## Pre-registered success criteria (REAL_RUN_RECIPE v2 §4)

### PRIMARY (gate)

| ID | criterion | window |
|---|---|---|
| W3-G1 | Rolling-mean SealBot WR ≥ 20% across 3 consecutive eval rounds | sustained 30k → 50k |
| W3-G2 | No L34 anchor↑/sealbot↓ divergence alarm fires | every eval round |
| W3-G3 | No HARD-ABORT trigger fires (run reaches 100k) | run completion |

### SECONDARY (informational)

| ID | criterion | window |
|---|---|---|
| W3-G4 | Peak SealBot WR ≥ §150 baseline (17.4%) | any single eval |
| W3-G5 | Refresh hook fired ~19 times | event count over 100k @ interval 5k |
| W3-G6 | alt V_spread ≥ +0.10 sustained | informational per L50 |
| W3-G7 | T3 V_spread sign track | informational per L48 |
| W3-G8 | colony_a < 50/100 in eval rounds | informational |

Promotion: ALL PRIMARY PASS at step 50k AND no abort 50k→100k AND no
L34 divergence in any eval.

## Pre-registered hard-abort triggers (REAL_RUN_RECIPE v2 §5)

Wired in `hexo_rl/monitoring/alert_rules.py:check_sealbot_wr_hard_abort`
+ `hexo_rl/training/step_coordinator.py` post-eval-drain:

- Trigger A: rolling-mean SealBot WR < 10% for 2 consecutive evals AFTER
  step 20k
- Trigger B: current WR < peak × 0.5 AND past step 25k (Wave-2-style
  collapse — would have caught Wave 2 at step 30k via 11% < 33% × 0.5 =
  16.5%)
- Trigger C: current WR < 5% past step 15k (§S180b-style early death)
- L34 anchor↑/sealbot↓ divergence (tightened to 3 consecutive instances
  vs Wave 2's 5)
- GPU NaN/Inf gradients (existing)
- Standard §S180b hard-aborts retained (`grad_norm > 10` ×5 consec,
  `loss_nan`, `colony_ext_frac > 0.40`, `stride5_p90 > 60`)

Operator override `--ignore-hard-abort` exists for debug only.

## Compute budget

| stage | cost | wall |
|---|---|---|
| Stage 2 dev | $0 | done |
| Stage 3 smoke (6000 steps; refresh fires at step 5000) | ~$1.50 | ~6 h |
| Stage 4 main (100k steps; refresh subprocess overhead ~$0.50) | ~$3.50 | ~14 h |
| Stage 5 analysis | $0 | 0.5 day |
| **TOTAL** | **~$5** | **~3 days incl. launch** |

Hard cap unless re-approved: $8.

## Pre-launch checklist (operator-mediated for Stage 3)

- [x] Stage 2 close: 4 commits land on `phase4.5/s181_wave3_design`
      (`63e8c5c`, `a53cbca`, `b6983bf`, `133d25a`)
- [x] Tests green: 49 focused + 110 broader + subagent full-suite 1621/0
- [x] REAL_RUN_RECIPE.md v2 landed (a53cbca)
- [ ] Smoke variant `configs/variants/v7_wave3_smoke.yaml` drafted
      (Stage 3A; inherits Wave 2 smoke `v7_real_run_smoke.yaml` from
      lever branch + Wave 3 deltas)
- [ ] Pre-registered WS-A..WS-E smoke verdict gates from dispatcher
      Stage 3C (LITERAL L13)
- [ ] Vast host `ssh6.vast.ai:13053` workspace updated to
      `phase4.5/s181_wave3_design` HEAD (post-push)
- [ ] Anchor `checkpoints/bootstrap_model_v7full.pt` present on vast
      (SHA `568d8a33…d61e8e98`)
- [ ] Wave 3 design branch pushed to origin
- [ ] Operator confirms Stage 3 budget ($1.50 smoke)

## Wave 3 deltas vs Wave 2 (for variant authoring at Stage 3A)

Wave 3 smoke variant `v7_wave3_smoke.yaml` (Stage 3A) — inherit from
Wave 2 smoke + 3 deltas:
1. `mixing.bot_corpus_refresh.enabled: true` + `interval_steps: 5000`
   + `n_games: 200` + `opponent_model: ema` +
   `replace_strategy: rolling_window` (Stage 2A)
2. `per_class_target_temperature.apply_to_pretrain: true` +
   `apply_to_selfplay: false` (Stage 2C / L52 scope)
3. `monitoring.wr_hard_abort_enabled: true` (Stage 2B default; ensure
   not overridden in variant) — optional explicit set for clarity

Wave 3 main variant `v7_wave3_main.yaml` (Stage 4A) — inherit smoke
+ adds:
- `iterations: 100000`
- `eval_interval: 5000` (denser than Wave 2's 10000 for faster L50
  catch speed; matches refresh hook interval)
- `checkpoint_interval: 2000`

## Wave 2 lever code preservation

The Wave 2 lever branch `phase4.5/s181_wave2_lever_vba_selfplay`
(`54bd9da` local; `3354016` origin) stays as historical reference for
the Wave 2 33% peak reproduction. Per-class temp file
`hexo_rl/training/per_class_target_temperature.py` from lever has been
brought to master in Stage 2C with the new `apply_to_selfplay` flag —
default `true` preserves Wave 2 behaviour, Wave 3 variant flips to
`false`.

Wave 2 canonical deliverable `reports/track_b_main/checkpoints/checkpoint_00020000.pt`
(33% SealBot peak) archived at
`reports/canonical_models/wave2_step20k_peak33pct.pt` (SHA
`2a09b7f3584d1de44f39940bacb831feee825e2b6dddc8e48e9ac49dbe5f0162`).

## Cross-references

- `audit/structural/REAL_RUN_RECIPE.md` — Wave 3 plan + success criteria
- `audit/structural/wave2_real_run_analysis.md` — Wave 2 mechanism
  analysis + L50/L51/L52
- `docs/designs/s179c_bot_refresh_hook.md` — refresh hook design
- `docs/07_PHASE4_SPRINT_LOG.md` §S181-AUDIT Wave 2 — close-out entry
- `hexo_rl/training/per_class_target_temperature.py` — per-class temp
  + Wave 3 apply_to_selfplay flag
- `hexo_rl/monitoring/alert_rules.py` — check_sealbot_wr_hard_abort
  pure function
- `hexo_rl/training/step_coordinator.py:170-185` (state) +
  `:709-754` (active refresh hook + WR hard-abort wiring)
