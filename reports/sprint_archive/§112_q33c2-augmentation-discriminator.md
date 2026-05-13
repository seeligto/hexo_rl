<!-- Forensic archive extracted from docs/07_PHASE4_SPRINT_LOG.md during compression sprint 2026-05-13. Compressed counterpart in docs/07_PHASE4_SPRINT_LOG.md. -->

## §112 — Q33-C2 augmentation discriminator (retry, E1 confirmed) — 2026-04-21

Unblocks §111. `feat(training): expose augment as training.augment config
knob` (commit `eb17389f6a7315fde42a17ac19066fd3d94a4c7d`) adds a tracked
config knob and plumbs it through `loop.py` → `trainer.train_step` and
`assemble_mixed_batch` (replacing 5 hard-True sites in
`batch_assembly.py:232,265,271,323,333`). Missing-key policy: hard
`ValueError` at loop entry (CLAUDE.md § Config discipline). Default
`true` preserves production behaviour. 6 new unit tests
(`tests/test_augment_plumbing.py`). Full test suite pass (847 python
+ 131 rust).

Ran the Q33-C2 smoke as specified in §111's recommended-scope section:
two 25-min runs from `checkpoint_00017000.pt` on laptop, isolated
`/tmp/q33c2_ckpts_*`, mixing-isolation preserved (`w_pre = 0`). Arm
configs `/tmp/q33c2_smoke_with_aug.yaml` (control, `augment: true`)
and `/tmp/q33c2_smoke_no_aug.yaml` (test, `augment: false`).
Report: `reports/q33c2_augmentation_discriminator_2026-04-21.md`.
Extractor: `/tmp/q33c2_extract.py`.

**Result:**

| Metric | Arm A (aug) | Arm B (no aug) | Δ (A − B) |
|---|---|---|---|
| pe_self overall | 5.167 | 5.382 | −0.215 |
| pe_self Q4 | 5.373 | 5.422 | **−0.049** |
| policy_loss Q4 | 0.914 | 0.813 | +0.101 |

**Verdict: E1 (healthy steady state).** `|Δpe_Q4| = 0.049 nat ≪ 0.5
nat` threshold — augmentation-off does NOT reduce `pe_self`. If
anything, pe_B is *slightly higher* than pe_A (sign opposite of E2's
prediction). The `pe_self ≈ 5.4 nat` fixed point documented in §110
is self-play-distribution behaviour, not a 12-fold augmentation
rotation bug. Arm A's `pl_Q4 = 0.914` matches Q33-B's 0.924 within
smoke noise — plumbing commit introduces no behavioural regression.
Arm B's `pl_Q4 = 0.813` is lower, consistent with the CLAUDE.md
Testing-conventions note that augmentation introduces per-batch RNG
variance on CE; orthogonal to the E1/E2 question.

**Effect on Phase 4.5 gating:** **unblocked** on the `pe_self` premise.
§110 had flagged the risk that bootstrap-strengthening work would be
wasted if `pe_self` stayed pinned regardless of improvement. This
smoke resolves that: the fixed point is the distribution's, not the
update path's. A stronger bootstrap that reshapes the frontier region
should move `pe_self` downward for the same reasons
baseline_puct/gumbel_targets/gumbel_full produce different pl values
in §109.

**Q33 / Q37 updates (`docs/06_OPEN_QUESTIONS.md`):**

- Q33: closed as WATCH → **RESOLVED (non-pathology)** with E1 verdict
  pointer to this report.
- Q37: closed as HIGH → **RESOLVED (non-pathology)**. The augmentation
  mask hypothesis is ruled out by direct empirical test; the remaining
  §110 candidates (full-search mask, weight-decay, LR schedule, mixing
  path) are weakly motivated given the distribution-shift reading now
  has direct support. If `pe_self` behaviour later regresses on a
  different checkpoint / distribution, reopen as a separate question
  with a fresh audit list.

Secondary notes kept for follow-up (not blocking):

- **Cosmetic rename.** `policy_entropy_selfplay` is H(p_model) on
  augmented current-batch self-play rows. Rename to
  `model_entropy_selfplay_batch` (or similar) and/or emit target
  entropy as a parallel key. Bundles with the Q35 candidate.
- **Confirmatory re-run.** A `w_pre > 0` production-mixing arm would
  independently confirm the E1 reading on the production path. Not
  required for Phase 4.5 launch; queue if a production discrepancy
  emerges.

### Commits

- `feat(training): expose augment as training.augment config knob` (`eb17389`)
- `docs(q33-c2): §112 E1 verdict, Q33/Q37 resolution, Phase 4.5 unblock`

---

