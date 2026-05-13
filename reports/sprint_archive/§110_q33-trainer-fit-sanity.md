<!-- Forensic archive extracted from docs/07_PHASE4_SPRINT_LOG.md during compression sprint 2026-05-13. Compressed counterpart in docs/07_PHASE4_SPRINT_LOG.md. -->

## §110 — Q33 follow-up: trainer-fit sanity check (Q33-B) — 2026-04-21

Follow-up to §109. Q33 left open: is the model drifting uniform because
bootstrap is weak (H_weak) or because the trainer update path drives it
uniform regardless (H_bug)? Phase 4.5 SealBot-injection pretrain is wasted
effort if the answer is H_bug.

Ran the Q33 `gumbel_targets` smoke verbatim except the starting checkpoint:
swapped `bootstrap_model.pt` → `checkpoint_00017000.pt` (sharpest available
post-§99 checkpoint; mean K=0 H(π) = 2.528 nats vs bootstrap 2.860, top-1
0.381 vs 0.334 on 300 positions from the 20K-collapse run). Same 14-worker
laptop config, same accidental `w_pre = 0` mixing-isolation, same
completed-Q targets, same 1500 s timeout, isolated `/tmp/q33b_ckpts/` so
tracked checkpoints untouched. Report:
`reports/q33b_trainer_fit_sanity_2026-04-21.md`. Extractor:
`/tmp/q33b_extract.py`.

**Result: Δpe_self = +0.004 nats over 180 training steps (Q1=5.360,
Q4=5.364). `pl_end = 0.924` — targets stay sharp.** The model does not
drift — it **sits at a fixed point of ~5.36 nats from step 17010 onward**.

Compared to Q33 bootstrap start (Q1=4.62 → Q4=5.54, Δ=+0.92), the
sharper-K=0 ckpt starts *higher* on `pe_self` (5.36) and stays flat. The
"drift to uniform" signature in Q33 was not a drift toward uniform — it
was convergence to a ~5.4 nat fixed point regardless of start.

**Verdict: H_bug (with partial H_weak signal).** Strict application of the
task's decision rules (H_bug: `pe_end ≥ 5.0`; H_weak: `Δpe < 0.5` AND
`pl_end ≤ Q33 pl_end`) fires **both** branches, so the
discriminator is not clean — the premise "sharper checkpoint yields
sharper `pe_self` at step 0" failed: K=0 sharpness on a fixed fixture
does not translate to lower `pe_self` on **the checkpoint's own self-play
rows**. The operative finding is: `pe_self ≈ 5.4` is a fixed point of the
trainer-update-path on the Rust self-play distribution, not a drift. Two
candidate explanations, not discriminated by this smoke:

1. **Self-play distribution shift.** A sharper model reaches harder
   positions where its own prior is diffuse by construction — the
   "frontier" sits near-uniform entropy. Healthy, not pathological.
2. **Trainer-update path error.** Augmentation-mask mis-alignment,
   full-search mask inversion, entropy-regularizer sign error, or mixing
   interference, any of which pins `pe_self` near uniform regardless of
   signal quality.

**Implication for Phase 4.5:** do NOT launch on the premise that stronger
bootstrap will move `pe_self` off ~5.4 — ckpt_17000 already has 17k
self-play steps of training baked in and sits at the same fixed point.
Phase 4.5 is still justified for value-quality / opening-coverage reasons,
but is not the fix for the `pe_self` symptom.

**Audit list (Q37 candidate, see open-questions file):** in priority
order — (1) `apply_sym` 12-fold augmentation mask alignment for policy
target vs input rotation (`engine/src/replay_buffer/sample.rs`,
`sym_tables.rs`); (2) `is_full_search=1` policy-loss mask alignment on
augmented rows (`hexo_rl/training/losses.py`); (3) entropy-regularizer
sign / magnitude (`entropy_reg_weight: 0.01`); (4) `weight_decay` /
optimizer step; (5) LR schedule; (6) re-run with production mixing
(`w_pre > 0`) to check mixing path.

Secondary incidental finding: the `--override-scheduler-horizon` flag does
not fully propagate — observed LR at step 17001 is 0.001534 (implying
scheduler T_max ≈ 50000, from the checkpoint's persisted state), not
0.002 (which T_max = 1000000 would give). Harmless for this diagnostic
(Q33 bootstrap ran at 0.002 and produced drift; LR at 77 % of peak is
well within the range where the same drift would appear). Flag as a
separate defect in `trainer.py:952-959` for later triage — not a Q33-B
finding.

Report caveats: picker measures K=0 softmax entropy on fixed fixture
positions (cross-run, not current self-play); trainer `pe_self` measures
on 12-fold augmented batch of current self-play rows. These are different
quantities — rank-order across checkpoints is interpretable, absolute
values are not directly comparable. The "sharper" criterion for the
discriminator was satisfied on K=0 fixture but failed on `pe_self` — the
next follow-up should instrument the trainer to emit `pe_self` on a
**fixed** cross-run fixture alongside the current-batch `pe_self`, to
separate policy-sharpness from distribution-shift.

### Commits

- `docs(sprint): §110 Q33-B trainer-fit sanity check`
- `docs(q33-b): trainer-fit sanity report + Q33 verdict update + Q37 candidate`

---

