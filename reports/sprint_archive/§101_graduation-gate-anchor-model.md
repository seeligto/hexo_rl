<!-- Forensic archive extracted from docs/07_PHASE4_SPRINT_LOG.md during compression sprint 2026-05-13. Compressed counterpart in docs/07_PHASE4_SPRINT_LOG.md. -->

## §101 — Graduation gate with anchor model (2026-04-16)

**Motivation.** Self-play workers were consuming `inf_model` weights re-synced from `trainer.model` every `checkpoint_interval` (500 steps) — effectively the current-training model, warts and all. Transient optimizer regressions fed directly into the data stream. KrakenBot-style graduation: new model must beat the current anchor at a configurable win rate before replacing it; workers keep the anchor between promotions. Monotonic data quality.

**Gap analysis.** 90% of the infrastructure was already live (`EvalPipeline` vs `best_model` gate at `eval_pipeline.py:188-190`; `best_model.pt` saved on promotion; `ResultsDB` + Bradley-Terry logs matches). Missing piece: routing — `best_model` was never consumed by self-play.

**Changes (branch `feat/graduation-gate`):**

- `loop.py`: remove unconditional `_sync_weights_to_inf()` call on checkpoint interval (buffer save retained). On startup with `best_model.pt` loaded, `inf_model` re-syncs from `best_model` (not `trainer.model`). `best_model_promoted` log gains `graduated=True`, `wr_best`.
- `eval_pipeline.py`: per-opponent `stride` gating — skip when `(train_step // base_interval) % stride != 0`. `EvalPipeline.__init__` caches `self._base_interval`.
- `eval.yaml`: `eval_interval: 5000 → 2500`; `best_checkpoint.n_games: 50 → 200` (tighter gating CI); strides `best=1 / sealbot=4 / random=1`.

**Behavioural invariants.**

- Between graduations, `inf_model` weights are frozen.
- On graduation: `best_model ← eval_model` (the scored snapshot — see §101.a C1), `inf_model ← best_model`, persisted + logged.
- Cold start with no `best_model.pt`: anchor is cloned from initial `trainer.model`. Candidate vs clone ~50% → no spurious promotion.

**Threshold & cadence.** `promotion_winrate: 0.55` (vs KrakenBot's 0.76 — conservative; tune up once graduations fire regularly). `n_games: 200` (binomial 95% CI ±~7% at p=0.55). Anchor eval every 2500 steps; SealBot every 10000.

### §101.a — Review fixes (applied before merge)

| # | Issue | Fix |
|---|---|---|
| **C1** | **Promoted weights ≠ evaluated weights.** Eval runs in a background thread with an `eval_model` snapshot; old code copied *current* `trainer.model` into `best_model` on promotion. Trainer had advanced ~1 `eval_interval` of steps between eval start and drain → every promotion committed unvalidated weights. | `eval_model` allocated once in outer scope; promotion branch loads `best_model ← eval_model` (drain fires before the next eval overwrites). |
| H1 | Stride cadence computed against `eval.yaml` `eval_interval`, ignoring `training.yaml` override. At training.yaml=5000, sealbot stride=4 fired every 20k steps not 10k. | Pipeline reads `full_config.eval_interval`; falls back to `self._base_interval`. Documented in both config files. |
| M1 | **False-promotion rate.** At n=200, p_true=0.5, P(X≥110) ≈ 9% → ~3-4 false promotions per 100k steps from sampling noise. | `gating.require_ci_above_half` (default true): promotion needs `wr_best ≥ threshold` AND `ci_lo > 0.5`. Drops false-positive rate below 1%. Flag preserves old behaviour for tuning. |
| M2 | Resume when `trainer.step != best_model_step` compares arbitrary weights vs anchor from a different time; lucky 55% wipes anchor. | Log `resume_anchor_step_mismatch` warning before first eval. |
| M3 | `eval_complete` event shipped `eval_games=0` (key never written). | Sum per-opponent `n_games` actually played (accounts for stride skips) → `results["eval_games"]`. |
| M4 | `stride: 0` or non-int silently collapsed to "every round" under `int(s) <= 1`. | `EvalPipeline.__init__` raises on stride not int ≥ 1; disable via `enabled: false`. |
| L1 | `eval_model` reallocated per round (~30 MB activations). | Allocated once outside loop; `load_state_dict` per round. |
| L2 | Dashboard read `.get("wr_sealbot", 0.0)` → stride-skipped rounds rendered as "0% vs SealBot". | Use `None` in event payload; dashboard distinguishes skip vs loss. |
| L3 | `eval_interval` coupling between trigger and stride math undocumented. | Comments added to `eval.yaml` + `training.yaml`. |
| L4 | Redundant `result["step"] = _step` in `run_evaluation`. | Removed. |

**Side cleanup.** `_sync_weights_to_inf()` (wrong direction — syncs from trainer, not anchor) deleted; sync sites now explicitly copy from `best_model` or `eval_model`.

**Tests added.** `test_stride_zero_rejected_at_init` (M4); `test_ci_guard_{blocks_marginal,disabled_allows_marginal}_promotion` (M1); `test_eval_games_reflects_opponents_run` (M3); `test_effective_eval_interval_override` (H1); `test_stride_{skips,runs}_sealbot_{off,on}_cadence` (stride). `test_run_evaluation_stores_results` updated to 9/10 wins (clears both gates without disabling CI guard).

**Known follow-ups (not blocking):** `graduation` boolean column on `ResultsDB.matches`; optional `skip_first_eval` flag for the guaranteed-neutral cold-start round.

