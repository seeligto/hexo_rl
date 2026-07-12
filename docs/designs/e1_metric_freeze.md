# E1 metric freeze (A2 pre-registration)

**Status:** FROZEN. This document pre-registers the E1 read-out. It is authoritative for
the E1 verdict and must be committed at a fixed SHA **before any E1 training step runs**
(the 5k checkpoint read is invalid otherwise). `E1_READY = YES` only once this file is
committed.

**Experiment.** E1 = 65-bin distributional value head (`value_head_type=dist65`) vs the
scalar head (`value_head_type=scalar`), paired A/B. Identical seed + data order; the only
config difference is the value-head type (+ its warm-start head path). One variable = the
value head (INV-D1 / R5: value target = outcome z only, two-hot, `soft_z_lambda=0`; no
teacher/TD/distill/solver in gradient; sims 150). Both arms **warm-start** from converged
HEADSWAP heads on the run2 248k trunk (scalar ← arm-A, dist ← arm-B) — this is load-bearing
for the slope requirement in §5.

**Authority / supersession.** This doc's M1–M4 + verdicts are the sole E1 gate. It
supersedes the provisional `Δrecog`/`ΔECE` binding sketched in
`docs/designs/e1_trainer_integration_plan.md` T8 and `docs/designs/escalation_e1_spec.md`.
Recognition-lag stays a *reported diagnostic* (T7) but is **not** a verdict gate.

Reference implementations and constants below are pinned at commit **`60cf720`** (master).
Reviewed (fable5, 2026-07-12): control-set reproducibility + verdict-logic holes closed;
operator-ratified (full-651 control set; one-sided REVIVE CI; longitudinal M4; exhaustive
verdict; slope requirement kept).

---

## 1. Frozen inputs (SHA-pinned)

| Input | Path | SHA256 | Count |
|---|---|---|---|
| Loss set | `reports/valprobe/probe_set_v1.jsonl` | `7899fa136ac083f0a428f5f6fa4c89918f1ba82c85618e8c7369a19506a9adb6` | 234 (all `"set":"loss"`) |
| Win-control set | `reports/valprobe/negatives_v1.jsonl` | `8faa6af74a7640f869cc3b1c4cb058b62660a052c5381e8ab7ad740a38cafef3` | 651 (all `"set":"safe"`) |

- **Probe anchor:** run2 trunk step 248000, `ckpt_sha=312f85f632ee5046`, encoding
  `v6_live2_ls` (`reports/valprobe/negatives_v1_summary.json`). Both files were built on
  this trunk. E1 reads a **different** net each checkpoint — the probe *positions* are
  frozen, the values are re-inferred per E1 checkpoint.
- **Loss-position subset = the V-CONFIRM flagged class.** All 234 probe positions are the
  V-CONFIRM class: SealBot head-**lost proof at depth 6** (`probe_set_v1.md` method step 3;
  `scripts/valprobe/wp2_expand_probe_set.py:61 SEALBOT_DEPTHS=[6]`) **AND** `v_raw ≥ −0.5`
  **AND** `replay_match=True` — i.e. the trunk value is optimistic while the position is
  provably lost. `v_raw ∈ [−0.479, 1.0]` over the set (`probe_set_v1.md`).
- **Win-control subset = the full 651 SAFE negatives, "as in D-C" (WP1).** The 651
  `"set":"safe"` positions from won source games, SAFE-verified at depth 7
  (`scripts/headswap/build_negatives.py:18`). Used whole — SHA-pinned above.

### Why full-651 (not a matched subset) is sound for the gates
The gating metrics do **not** require a ply-band-matched control set:
- **M1** uses only the 234 losses — no controls.
- **M2** gate is a **gap** `ECE_scalar − ECE_dist` computed on the *same* set for both arms;
  any shared ply-composition skew cancels in the difference.
- **M4** gate is a **difference** `FP_dist − FP_scalar` on the *same* controls; shared skew
  cancels.
Only the **secondary, non-gating M3** (discrimination AUC) is confound-sensitive. So the
D-FULLSPEC turn-phase confound (win/loss separable on ply-phase alone, AUC ≈ 0.807) is
handled **by construction** for the gates, and flagged for M3 by the ply-canary below.
This deviates from D-C's inline *matched* AUC set (operator-ratified 2026-07-12): the
committed matched-156 count was a per-band feasibility count that materializes no auditable
set and is not reproducible from the pinned files; matching only ever bound the AUC metric,
which is non-gating here.

### Reproduction check (run once, before the 5k read)
1. `sha256sum` both files == the SHAs above. Halt E1 read if either differs.
2. Compute the **ply-confound canary once** (features frozen ⇒ constant across checkpoints):
   AUC of `ply`/`t` predicting loss-vs-safe over the 234∪651 set
   (`metrics.py:37 CANARY_MAX_AUC=0.55`). If AUC > 0.55, **M3 is read as continuity-only**
   (its raw AUC is ply-confounded); the gates M1/M2/M4 are **unaffected** (composition-
   invariant, above).

---

## 2. Dist-head read-outs (fixed mapping — HEADSWAP continuity)

65 bins, support `torch.linspace(-1.0, 1.0, 65)`; bin width `2/64 = 0.03125`; scalar→bin
`pos = (z+1.0)*32.0` (`hexo_rl/training/binned_value.py:13–21`).

- **level-v (decoded-v)** = `E[softmax·support]`, clamped `[-1,1]`
  (`binned_value.py:31–36 decode_binned_value`). Scalar arm: the head's scalar output
  directly (already `[-1,1]`). Both arms yield one comparable scalar per position.
- **tail-mass** = `P(v ≤ −0.5)` = sum of softmax mass over bins **0…16 inclusive**
  (`support[16] = −0.5` exact; `scripts/headswap/targets.py:32–33 LOSS_TAIL_THRESHOLD=-0.5,
  LOSS_TAIL_BIN=16`; `targets.py:89–98 loss_tail_mass` = `probs[:,:17].sum`; mirrored in
  `scripts/headswap/metrics.py:42 TAIL_BIN_DEFAULT=16`). Same threshold bin HEADSWAP
  registered. Dist arm only (scalar arm has no distribution).

---

## 3. Checkpoints read

**5k / 10k / 20k / 50k**, both arms (scalar, dist65). No other checkpoints gate the verdict.
Positive-at-5k is *encouraging, not sufficient* — read to 50k regardless (per E1 spec).

---

## 4. Metrics (frozen)

All per-arm, per-checkpoint. Decoded-v is head-perspective; loss outcome `y=−1`, win-control
outcome `y=+1`.

### M1 — PRIMARY (level)
`M1(arm,t)` = **mean decoded-v on the 234 probe LOST positions**, for that arm.
`gap_M1(t) = M1_scalar(t) − M1_dist(t)`. **Positive gap ⇒ dist recognizes losses better**
(dist reads losses more negative than scalar).

### M2 — CO-PRIMARY (calibration)
`M2(arm,t)` = **ECE on the full probe set** = the **234 losses ∪ 651 win-controls** (declared
operationalization: "full probe set" = loss set ∪ control set; ECE on the all-loss 234
alone is degenerate). 10 equal-width bins on `P_win=(v+1)/2`, per
`scripts/valprobe/value_health.py:121 compute_ece` (`ECE_N_BINS=10`, :43) @`60cf720`.
`gap_M2(t) = ECE_scalar(t) − ECE_dist(t)`. **Positive gap ⇒ dist better calibrated** (lower
ECE). Class composition (234 loss : 651 safe) is frozen and identical for both arms.

### M3 — secondary (HEADSWAP continuity; reported, NOT a gate)
Dist arm **tail-mass AUC** discriminating the 234 losses vs 651 controls. Reported alongside
the scalar arm's decoded-v AUC as the continuity comparator against the HEADSWAP frozen gap
`C.tail − D.v = +0.052 CI[+0.018,+0.087]` (`reports/headswap/VERDICT.md:17`,
`escalation_e1_spec.md`). If the §1 ply-canary fires (AUC > 0.55 on the full set — likely,
since the 651 safes skew early-ply), M3 is continuity-only. Informational; never gates.

### M4 — guard (false-pessimism), LONGITUDINAL
`FP(arm,t)` = **fraction of the 651 win-controls with decoded-v ≤ −0.5**
(`metrics.py:36 FALSE_PESS_THRESHOLD=-0.5`; `metrics.py:336 false_pessimism`).
**Guard: `FP_dist(t) ≤ FP_scalar(t) + 0.05` at ALL FOUR checkpoints**
(5pp; `metrics.py:39 FALSE_PESS_SLACK=0.05`). "Must stay ≤ scalar+5pp" is read
longitudinally — dist may not be doom-happy at any read. Prevents dist "winning" M1 merely
by global pessimism.

### CI + slope (frozen estimators)
- **CI:** position-level bootstrap, **10k resamples**, per checkpoint. Gaps are paired —
  resample positions once per replicate, recompute both arms on the resample, take the
  difference. M1 resamples the 234 losses; M2 resamples the 234∪651 union.
- **Slope:** **Theil-Sen** slope of `gap(t)` vs step over the 4 points {5k,10k,20k,50k}
  (median of the 6 pairwise slopes). "Positive slope" ⇒ Theil-Sen > 0. (Matches WP3's slope
  table estimator.)

---

## 5. VERDICTS (frozen, exhaustive)

**REVIVE** = ( `gap_M1` has **positive Theil-Sen slope** across the 4 points **AND** the
final-point (50k) `gap_M1` bootstrap CI has **lower bound > 0** ) **OR** ( the same for
`gap_M2` ) — with the *other* co-primary **not worse in point estimate** (its 50k gap ≥ 0)
**AND** the M4 guard held at **all four** checkpoints.

**CONFIRM-DEMOTE** = **any outcome that is not REVIVE.** In particular: both gaps
flat/negative through 50k; OR a REVIVE co-primary clause met but the cross-guard or M4 fails;
OR a genuinely flat-but-positive gap (Theil-Sen ≤ 0) — see caveat.

The verdict space is exhaustive and mutually exclusive: REVIVE requires a *one-sided
positive* CI (lower bound > 0), so a dist-significantly-**worse** run (50k CI below 0) can
never be REVIVE; everything else is CONFIRM-DEMOTE.

**Positive-at-5k** = encouraging, NOT sufficient. Read to 50k regardless.

### Recorded caveats (accepted, operator-ratified)
- **Slope > 0 is required, by design (I4).** Both arms warm-start from HEADSWAP heads, so a
  gap already present at 5k is an *inherited head-start*, not training-driven learning. The
  positive-slope requirement demands the dist head keep improving over training. A flat
  (Theil-Sen ≤ 0) but positive gap therefore lands in CONFIRM-DEMOTE — this is intentional,
  not a loophole. If a flat-significant-positive result appears, it is recorded as such
  (dist warm-start head is better but the *architecture* did not compound it in training).
- **Two co-primaries (M1 level, M2 calib) = mild multiplicity.** Accepted and recorded; the
  REVIVE OR-clause is deliberate (either level or calibration lift qualifies) with the
  cross-guard (other co-primary not worse) + M4 to bound false wins.
- **Position-level bootstrap ignores source-game clustering** (frozen choice, per
  dispatcher). Anti-conservative if losses cluster by game. Not re-opened for E1.

---

## 6. Provenance

Dispatcher D-I E1LAUNCH WP0 (2026-07-12). Artifacts verified at `60cf720`:
probe/negatives SHAs computed locally; `binned_value.py`, `value_health.py`,
`scripts/headswap/{targets,metrics}.py` constants read at that commit. fable5 review +
operator ratification 2026-07-12 (full-651 control set; one-sided REVIVE CI lower-bound > 0;
longitudinal M4; exhaustive verdict; slope requirement retained as design caveat).
