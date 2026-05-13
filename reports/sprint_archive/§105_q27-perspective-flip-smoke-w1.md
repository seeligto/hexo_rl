<!-- Forensic archive extracted from docs/07_PHASE4_SPRINT_LOG.md during compression sprint 2026-05-13. Compressed counterpart in docs/07_PHASE4_SPRINT_LOG.md. -->

## §105 — Q27 perspective-flip smoke: W1 necessary, not sufficient (2026-04-18 → 2026-04-19)

**Motivation.** `e9ebbb9` ("fix(mcts): negate child Q at intermediate
ply in `get_improved_policy`, Gumbel score, `get_top_visits`") was
landed as W1 on correctness grounds — three call sites failed the
`parent.moves_remaining==1` negation that `puct_score` already had,
inverting training targets at ~50% of move steps. The open question
on landing was whether W1 *also* closes the Q27 attention-hijacking
symptom (threat head passes C1 easily; policy head pins at 20% on
C2/C3) at the 5K-step smoke horizon. One-shot smoke, two machines.

**Setup.** Two-machine split to save wall-clock. Both arms start from
`checkpoints/bootstrap_model.pt`, run 5000 steps no-dashboard, run
`make probe.latest` immediately after:

- **pre_fix** — laptop (8845HS / RTX 4060), commit `723615e` (parent
  of `e9ebbb9`), variant `gumbel_targets` (14 workers).
- **post_fix** — desktop (3700x / RTX 3070), commit `a7efa78` (HEAD),
  variant `gumbel_targets_desktop` (10 workers, +1ms inference wait).
  Selfplay semantics identical per the variant header; machine and
  worker-count differ.

Desktop arm required one restart after accidental window close
(run1 → step 450 partial; run2 restarted from bootstrap, no
contamination).

**Results.**

| Metric | pre_fix (723615e) | post_fix (HEAD) | Δ |
|---|---|---|---|
| C1 contrast_mean | +3.665 PASS | +3.891 PASS | +0.226 |
| C2 ext_in_top5_pct | **20% FAIL** | **20% FAIL** | 0 |
| C3 ext_in_top10_pct | **20% FAIL** | **20% FAIL** | 0 |
| C4 \|Δ ext_logit_mean\| | 0.078 ok | 0.505 ok | +0.427 |
| Exit | 1 (FAIL) | 1 (FAIL) | — |
| H(policy) @ 5K | 5.3733 | 5.6284 | +0.255 |
| Settled entropy band (500–5000) | 5.17 – 5.57 | 5.51 – 5.72 | ~+0.2 |
| Policy loss 0 → 5K (post only) | — | 1.9619 → 1.6544 | −0.308 |
| Games / buffer (post only) | — | 1253 / 250K saturated | — |

Uniform entropy on ~361 legal cells ≈ 5.88. Post-fix sits ~0.25 nats
closer to uniform than pre-fix.

**Verdict.** W1 is **necessary, not sufficient at 5K.**

- Both arms fail threat probe on C2/C3 with **identical 20% / 20%**
  scores. The policy head is not routing top-K to extension cells in
  either arm. Attention hijacking persists across the fix.
- C1 passes ~10× threshold in both arms — threat head carries
  extension-vs-control information; the failure is downstream in how
  the policy trunk uses that signal.
- Entropy delta sign is in the wrong direction for a "W1 alone fixes
  it" story: post-fix is *closer* to uniform, not more committed. Two
  plausible readings: healthier ongoing exploration, or machine/
  variant noise. Cannot discriminate with n=1 per arm.

W1 stands on correctness grounds regardless. Q27 remains open.

**Caveats.**

- Two machines, two variants. Selfplay target construction is
  identical; worker-count (14 vs 10) and inference wait (4 vs 5ms)
  differ. The +0.25-nat entropy delta is within plausible worker-count
  noise.
- n=1 per arm; no CIs. The identical 20%/20% C2/C3 is decisive enough
  on its own that the n=1 limit does not flip the conclusion, but the
  entropy reading is directional, not quantitative.
- Desktop run had a restart from bootstrap (not resume); reported
  trajectory is run2 only.
- Pre-fix arm's policy_loss trajectory not preserved post log rotation.

**Follow-ups (not actioned this sprint).**

Three candidate probes for where the attention-hijacking root cause
lives. Prioritise before committing to another 5K smoke:

1. **Value aggregation (Open Question 2).** Min-pool over cluster
   windows may silently discard extension-cell evidence — would
   reproduce the "threat scalar learns contrast, policy ignores it"
   signature seen here. Ablate mean-pool at fixed 5K budget.
2. **Threat head → policy gradient coupling.** BCE weight 0.1 may be
   drowned out by policy CE at shared trunk. `aux_threat_weight`
   sweep 0.1 → 0.5 at 5K.
3. **ZOI post-search mask (§77).** If extension cells in the probe
   positions fall outside hex-distance-5 of the last 16 moves, C2/C3
   are capped by construction. Log ZOI reachability of probe
   positions' extension cells.

A controlled same-machine n=3 rerun is premature until one of the
above has a concrete hypothesis attached.

**Files.**

- `reports/q27_perspective_flip_smoke_2026-04-18/verdict.md`
- `reports/q27_perspective_flip_smoke_2026-04-18/pre_fix/{summary,probe_output,entropy_trajectory,train_stdout}.*`
- `reports/q27_perspective_flip_smoke_2026-04-18/post_fix/summary.txt`
- `reports/probes/latest_20260418_223903.md` (pre_fix detail)
- `reports/probes/latest_20260419_011839.md` (post_fix detail)

**Resolves.** Nothing. **Leaves open.** Q27 (attention hijacking).

### Commits

- `docs(sprint): §105 Q27 perspective-flip smoke verdict`

**POSTSCRIPT 2026-04-19.** §106 supersedes the "attention hijacking
persists" framing above. Probe 1b regenerated the fixture from real
game positions; the 5K post-W1 checkpoint PASSES all three probe gates
(C1 +3.317, C2 50%, C3 65%). The C2/C3 failures logged here were a
synthetic-fixture artifact, not a training pathology. The correctness
argument for W1 (inverted Q targets at ~50% of move steps) is
unaffected. Original body above retained as the record of what was
believed at the time.

---

