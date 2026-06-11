<!-- Forensic archive extracted from docs/07_PHASE4_SPRINT_LOG.md during compression sprint 2026-05-13. Compressed counterpart in docs/07_PHASE4_SPRINT_LOG.md. -->

## §174 — v6w25 sustained smoke: LR fix, e50 retrain, bootstrap investigation — 2026-05-13 (PAUSED)

**Status:** PAUSED for bootstrap investigation. Track 1 (vast) closing eval data gaps; Track 2 (local) preparing codebase for relaunch. Decision blocked on Track 1 results. Cross-ref: `reports/s174_bootstrap_investigation.md`.

### Timeline

1. **First 30K run — FP16 scaler overflow.** Launched against `bootstrap_model_v6w25.pt` with LR 2e-3 (carried over from v6 recipe). FP16 GradScaler overflowed early on the 25×25 geometry; scaler kept halving the scale factor until effective LR collapsed and training stalled. Aborted.
2. **LR fix — 2e-3 → 1e-3.** Per §173 A8 v3 ablation (`reports/sprint_173/a8_cold_smoke_final.md`), 1e-3 is the stable point on the larger window. Pushed to `configs/variants/vast.yaml:24`.
3. **Second 30K run — eval crash.** Restarted with LR 1e-3 from the same bootstrap. Selfplay stable, train losses converging, but the eval round crashed mid-pipeline. Investigation deferred (orthogonal to bootstrap question).
4. **e50 retrain — more epochs ≠ better selfplay.** Re-pretrained `bootstrap_model_v6w25.pt` with 50 epochs (was 30). Hypothesis: extra capacity-utilisation epochs would tighten the policy/value heads. Result inverted: selfplay quality regressed (see findings below).
5. **Radius ablation — falsified.** Ran R ∈ {1, 2, 3, 4, 5, 6, 7, 8} smokes off the e30 bootstrap. Median game length identical across all radii. Radius has zero effect on game quality at the bootstrap level (decision boundary lives elsewhere).
6. **G4 marginal fail on e50.** `value_fc2.weight.abs().max() = 0.489` vs the ±50% band `[0.154, 0.462]` around the v7full baseline 0.308. Value head grew during the extra 20 epochs — most likely overfitting to corpus value targets that the larger v6w25 window cannot match in selfplay.
7. **PAUSE.** Decision: hold the relaunch until Track 1 returns the 30-epoch MCTS-128 sealbot baseline + retraced anchor curves. Track 2 prepares the codebase in parallel (this entry).

### Key findings

#### LR 2e-3 → 1e-3
FP16 scaler overflow on the 25×25 geometry. The v6 recipe LR was tuned to a smaller window; the larger v6w25 activations push pre-scaler logits past fp16 range. §173 A8 v3 ablation already exposed this; §174 P1 confirmed in production. Action: locked `lr: 0.001` in `configs/variants/vast.yaml`; comment cross-refs the A8 report.

#### e50 retrain — more epochs = worse selfplay
| Epochs | Train loss | Selfplay quality (qualitative) | G4 |
|---|---|---|---|
| 30 (current canonical) | converged | baseline | PASS (within band) |
| 50 (ablation) | lower | regressed | MARGINAL FAIL 0.489 |

Plane usage analysis (run on the e50 checkpoint) confirmed value-head growth on the channels that carry corpus-mode signal. Interpretation: extra pretrain epochs over-fit the value head to the human/bot corpus distribution, which selfplay cannot reproduce — selfplay value targets disagree with the corpus prior and the head's confident corpus-mode predictions get washed out as noise during selfplay. Result: e30 bootstrap is the correct anchor; e50 is dominated.

#### Radius ablation — falsified
Eight smokes (R=1 through R=8) off the e30 bootstrap, otherwise identical recipe. Median game length stable across all eight. Radius does not move the bootstrap-quality needle — it changes legal-move structure but not policy/value head behaviour at the bootstrap level. Action: keep the `legal_move_radius_schedule` curriculum in `vast.yaml` (R=5 → R=8 over training) as a downstream-training lever, but stop treating radius as a bootstrap-time hyperparameter.

#### G4 marginal fail (e50)
`value_fc2.weight.abs().max() = 0.489` exceeds the upper band 0.462. e30 was inside band. Confirms the value-head growth diagnosis above and validates G4 as a discriminator for this failure mode. G4 is now wired into `eval_pipeline.run_evaluation` (this sprint, Track 2) — every eval round measures and logs the value, emits a structlog WARNING on out-of-band.

### Current state

- Training PAUSED.
- Bootstrap investigation in progress on Track 1 (vast). Missing data slice: 30-epoch MCTS-128 vs SealBot anchor curve at n=100, plus value-head |max| trace across e30 epochs.
- Track 2 (local, this entry): codebase prepared for relaunch — vast.yaml audited clean, G4 wired, sprint log + investigation pointer in place.

### Available checkpoints

| Name | Epochs | Encoding | G4 status | Use |
|---|---|---|---|---|
| `bootstrap_model_v6w25.pt` (e30) | 30 | v6w25 | PASS (within band) | canonical §174 anchor |
| `bootstrap_model_v6w25_e50.pt` | 50 | v6w25 | MARGINAL FAIL 0.489 | dominated, retain for analysis |
| `bootstrap_model_v7full.pt` | n/a | v7full | reference | §170 / §171 / §172 anchor, not used in §174 |

### Decision pending

Bootstrap choice blocked on Track 1 results. Decision matrix (to be filled in once Track 1 reports):

| Track 1 finding | §174 action |
|---|---|
| e30 v6w25 ≥ §150 v7full anchor on MCTS-128 sealbot n=100 | Launch sustained with e30 v6w25, recipe as in vast.yaml |
| e30 v6w25 < §150 anchor by > 5pp absolute | Re-evaluate: either retrain v6w25 with a different recipe or fall back to v7full encoding for §174 |
| e30 v6w25 within ±5pp of §150 anchor (within noise) | Launch sustained with e30 v6w25 — gap is in measurement noise and α multi-window + radius curriculum are net new levers |

### §174 locked parameters (re-confirmation, post-PAUSE)

Identical to the pre-PAUSE block above (LR 1e-3, eval_interval 10000, training_steps_per_game 2.0, selfplay.random_opening_plies 0, eval.eval_random_opening_plies 0 via configs/eval.yaml:88, mcts.n_simulations 400, hard_abort_grad_norm 10.0). vast.yaml audited 2026-05-13 against this list — all fields match. `eval_interval` raised 5000 → 10000 vs §173 P0 to halve eval wall-time on the 5080; preserves stride math (sealbot stride=4 still fires every 40k steps with the §155 H1 effective-interval fix).

### G-gate wiring status (Track 2 audit, 2026-05-13)

- **G3** — monotonic depth scaling. WIRED. `avg_game_length` emitted in `iteration_complete` (orchestrator.py:336), per-game `game_length` in structlog `game_complete` log (pool.py:593).
- **G4** — value-head |max| ±50% band [0.154, 0.462]. WIRED THIS SPRINT (Track 2). `_g4_value_head_band_check` runs at the start of every `run_evaluation` round; result persisted in `results["value_fc2_weight_abs_max"]` + `results["g4_value_head_band_pass"]`; structlog WARNING on violation. Constants are gate-internal — variants do not override.
- **G5** — per-cluster variance drift ≤30%. WIRED. `cluster_value_std_mean` + `cluster_policy_disagreement_mean` + `cluster_variance_sample_count` emitted in both `iteration_complete` (orchestrator.py:349-351) and `train_step_summary` (orchestrator.py:404-406). Drift detection (≤30% delta vs baseline) is a post-hoc operator computation against `cluster_variance_sample_count > 0` rounds; no inline gate fire.

### `random_opening_plies` propagation audit (Track 2, 2026-05-13)

Two distinct fields, two distinct config sources:
- **Selfplay path:** `selfplay.random_opening_plies` — `configs/selfplay.yaml:66` defaults to 1, `configs/variants/vast.yaml:54` overrides to 0. Consumed at `hexo_rl/selfplay/pool.py:288`.
- **Eval path:** `eval_pipeline.eval_random_opening_plies` — `configs/eval.yaml:88` defaults to 0 (was 4 pre-§174; the §174 commit cited "inflated WR by giving free positional advantage"). Consumed at `hexo_rl/eval/eval_pipeline.py:197` → `hexo_rl/eval/evaluator.py:133`. vast.yaml does not override.

Pipeline build path (`hexo_rl/eval/pipeline_setup.py:52`) loads `configs/eval.yaml` directly — separate from `scripts/train.py`'s base-config list — so the eval.yaml value reaches the evaluator. Deep-merge from any variant `eval_pipeline:` block is applied on top (none in vast.yaml).

Conclusion: the §168 → §174 sealbot WR drop (14.5% → 0%) is fully explained by `eval_random_opening_plies` 4 → 0. With 4 random plies the model got free positional diversity that masked policy/value weaknesses; with 0 the model plays a fully-deterministic MCTS opening and SealBot's preparation lands cleanly. No code fix required — the eval.yaml flip is already in place.

---

## §174 — v6w25 sustained: bootstrap investigation + escalation — CLOSED 2026-05-13

### Verdict: ESCALATE to v6 sustained (§175)

Three v6w25 bootstraps tested (30-epoch, e50, v6→v6w25 transfer FT). None
clears selfplay viability gate (6–9 plies at R=8 MCTS-128). Radius
compression hypothesis falsified (smokes were already at R=8).

| Bootstrap | SealBot MCTS-128 | Selfplay median plies |
|---|---|---|
| 30-epoch | 0% (0/200) | 6 |
| e50 | 10% (10/100, artifact-suspect) | 6 |
| transfer FT | 0% (0/200) | 8 |

Root cause (from `reports/s174_v6w25_investigation.md`): the loss surface
is normal — v6w25 30-ep achieves 3.96 nats improvement over uniform (best
of v6 / v7full / v6w25 by that measure) and matches v7full value-loss
trajectory. Opening-fraction starvation hypothesis (H1) is refuted: ply ≤
10 fraction 16.09% (v6w25) vs 17.15% (v6) — gap is 1.06pp, not the
multi-× gap predicted. The collapse is at the **argmax-degeneracy /
selfplay-interaction layer**, not at the corpus or loss layer. Transfer
recipe inherited the v6 trunk but lost opening knowledge in the
re-initialised policy FC.

### Infrastructure landed (Track 2)

- Encoding auto-detect across `make pretrain` / `make eval` / `make
  selfplay.smoke` / `make transfer` (W1 — checkpoint metadata is
  authoritative; CLI flag overrides when ambiguous).
- G4 value-head |max| band check wired into every `run_evaluation`
  round (`_g4_value_head_band_check`, structlog WARNING on out-of-band).
- v6 → v6w25 transfer script (`scripts/transfer_v6_to_v6w25.py`).
- vast.yaml audited clean: LR 1e-3, eval_interval 10000,
  random_opening_plies 0 on both selfplay and eval paths.
- Makefile encoding-override knobs (`PRETRAIN_ENCODING`, `EVAL_ENCODING`,
  `SMOKE_ENCODING`, `TRANSFER_SOURCE`, `TRANSFER_OUTPUT`).

### Reports

- `reports/s174_v6w25_investigation.md` — opening fractions, policy-head
  capacity ratio, loss decomposition vs v6 / v7full, entropy normalised
  against `log(K)` floor.
- `reports/s174_bootstrap_investigation.md` — vast eval matrix (e30,
  e50, transfer FT) vs SealBot at random_plies=0.
- `reports/s174_bootstrap_fix.md` — transfer recipe + Xavier-init policy
  FC + drop-and-restart fine-tune.

### Available checkpoints (post-§174)

| Name | Epochs | Encoding | Use |
|---|---|---|---|
| `bootstrap_model_v6w25.pt` (e30) | 30 | v6w25 | retained for analysis; not §175 anchor |
| `bootstrap_model_v6w25_e50.pt` | 50 | v6w25 | G4 marginal fail; dominated |
| `bootstrap_model_v6w25_transfer_ft.pt` | 15 ft | v6w25 | 0% MCTS-128; retained for analysis |
| `bootstrap_model_v7full.pt` | 30 | v7full | §150 anchor (17.4% n=500); not §175 anchor |
| `bootstrap_model.pt` (v6) | 30 | v6 | **§175 anchor** |

### Forward: §175 v6 sustained

Recipe: 100K steps, n=100 SealBot eval, matched cosine LR schedule
inherited from §174 vast.yaml. Selfplay encoding v6 (single-window 19×19,
existing path). v6w25 retained as a future re-entry target once a
selfplay-friendly bootstrap recipe is found — current evidence says the
fix is at the policy/value head and selfplay-interaction layer, not the
corpus layer. Tracking: see §175 sprint log entry.

