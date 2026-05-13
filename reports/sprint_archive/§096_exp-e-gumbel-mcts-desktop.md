<!-- Forensic archive extracted from docs/07_PHASE4_SPRINT_LOG.md during compression sprint 2026-05-13. Compressed counterpart in docs/07_PHASE4_SPRINT_LOG.md. -->

## §96 — Exp E: Gumbel MCTS desktop (A/B vs laptop exp D PUCT+completedQ) (2026-04-16)

### Hypothesis

Gumbel top-m + completed-Q policy targets produce better move rankings per sim → faster
tactical convergence. Expected -5 to -15% sims/s vs PUCT but net-positive pos/hr via better
training signal per game.

**Setup.** Hardware: Ryzen 7 3700x + RTX 3070 (desktop). Variant: `gumbel_full`. Checkpoint: `bootstrap_model.pt` (v3b). Run label: `exp_E_gumbel_full_desktop`.

**Config diffs vs laptop exp D (PUCT+completedQ):** `gumbel_mcts: true`; `n_workers: 10` (Zen2 GIL ceiling from §81 D3); `inference_max_wait_ms: 5.0`. All other keys identical (`training_steps_per_game=4`, `burst=16`, `max_game_moves=200`, `draw_value=-0.5`, `decay_steps=20k`, `total_steps=200k`).

**Kill conditions (relaxed per exp D learnings):** `draw_rate > 70%` sustained; `policy_entropy_selfplay < 1.5` for 500+ steps; `grad_norm > 10` for 50+ steps; `pos_per_hr < 35k`; NaN / OOM / crash. Probe C2/C3 FAIL does NOT kill.

**Success metrics at step 20k:** draw-rate ≤ laptop exp D; `pos_per_hr` ≥ 80% of laptop; C2 ≥ 30%, C3 ≥ 45% at step 15k. Monitoring: `scripts/monitor_exp_E.sh`.

**Q26 [WATCH] — nested `training:` block in `gumbel_targets_desktop.yaml`.** Deep-merger treats the nested `training:` key as a new sub-dict; flat `training_steps_per_game` in `training.yaml` is never overridden → defaults to 1.0. Scope limited to `gumbel_targets_desktop.yaml`; `gumbel_full.yaml` uses flat keys correctly. Audit after exp D completes.

---

