<!-- Forensic archive extracted from docs/07_PHASE4_SPRINT_LOG.md during compression sprint 2026-05-13. Compressed counterpart in docs/07_PHASE4_SPRINT_LOG.md. -->

## §109 — Q33 selfplay entropy diagnostic — 2026-04-21

(Numbered §109 because §107 and §108 were already taken; the Q33 task prompt
asked for "§107". Kept chronological.)

Follow-up to the `reports/diag_20k_collapse_2026-04-21.md` §"Additional signal"
candidate — that report flagged `pe_self ≈ 5.35` in the `train_c51d245d…`
`gumbel_targets` run and asked whether the flatness is expected at bootstrap
strength or a signal-processing bug in the completed-Q target path
(`get_improved_policy` on PUCT trees post-§74.1). Three 25-min smokes from
`bootstrap_model.pt` on laptop (Ryzen 7 8845HS + RTX 4060, 14 workers),
harmonized knobs, only `gumbel_mcts` / `completed_q_values` differ.
Report: `reports/q33_selfplay_entropy_2026-04-21.md`. Extractor:
`/tmp/q33_extract.py`. Smoke override configs: `/tmp/q33_smoke_*.yaml`
(not in tree, per the "no config changes" scope).

**Verdict: EXPECTED / INVERSION.** Not a completed-Q bug. With
`policy_loss` interpreted as the upper bound on `H(target)` (`CE ≥ H(target)`),
the three variants produce: baseline_puct `pl = 5.52` (targets near uniform),
gumbel_targets `pl = 1.12` (targets sharp, `H(target) ≤ 1.12`), gumbel_full
`pl = 2.33` (targets moderately sharp). `completed_q_values=true`
**sharpens** targets on both PUCT and Gumbel SH backends at bootstrap
strength. The diag report's `pe_self ≈ 5.35` observation was
**model-output entropy** (`H(p_model)` on selfplay rows, per
`trainer.py:570-572`), not target entropy — the two share the event key
`policy_entropy_selfplay` but measure different things. In the smoke,
`gumbel_targets` reproduces the production pe_self drift (first-quartile
4.62 → last-quartile 5.54) in 220 steps while the targets simultaneously
sharpen (CE 1.50 → 0.98). The 20K collapse signature is the **trainer
failing to fit sharp selfplay targets**, not flat targets. Phase 4.5
bootstrap work remains the correct next step: a stronger start should let
the model fit completed-Q targets from step 0 instead of drifting uniform.

Secondary observation: `gumbel_full` emits short games (27-ply mean vs
131/139 on the other two) and 0 % draws in the smoke window; orthogonal
to the target-entropy finding but flagged for a separate investigation.
`timeout --signal=INT --kill-after=30s 1500s` failed to terminate the
gumbel_full smoke (ran ~74 min before manual kill) — orchestration
artifact, not a Q33 finding. Caveat: the smoke override files
(`/tmp/q33_smoke_*.yaml`) accidentally put `mixing:` under `training:`
while the base `configs/training.yaml` keeps it top-level, so the
pretrain corpus was not loaded and `w_pre = 0` throughout — batches are
**100 % selfplay rows**. For the Q33 question this is useful (isolates
selfplay target signal) but the trainer-fit dynamic will differ from
production mixed-batch behaviour at later steps.

Links: Q33 entry promoted in `docs/06_OPEN_QUESTIONS.md` (WATCH, not a
bug). Related Q17 (sprint §70, §73) resolution held — Dirichlet port is
unaffected. Related diag-20K entry (`reports/diag_20k_collapse_2026-04-21.md`)
§"Additional signal" is now superseded: the recommendation to audit a
completed-Q flattening bug can be closed.

### Commits

- `docs(sprint): §109 Q33 selfplay entropy diagnostic`
- `docs(q33): smoke report + Q33 entry in open questions`

---

