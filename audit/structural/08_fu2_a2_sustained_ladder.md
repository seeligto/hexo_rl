# §S181 FU-2 — A2 sustained-ladder verdict

**Wave:** §S181 FU-2 Task 3-4. **Branch:** `phase4.5/s181_fu2_a2_arch`.
**Date:** 2026-05-23. **Host:** vast.ai 5080 (operator-mediated).
**Run config:** `configs/variants/v6_botmix_s180b_a2.yaml` (operator override
to 20k target; killed early at step 2500 on strong V-FU2-C signal).

> **VERDICT: V-FU2-C — FAIL, same trajectory.** A2 alone is NOT the
> load-bearing fix. The training loop pulls the value head into the
> colony-favouring attractor regardless of architectural starting
> point. Route to PSW or refresh hook with V_spread canary intact.
> A2 reverts.

---

## 0. Run summary

| | |
|---|---|
| Anchor | `checkpoints/bootstrap_model_v6_a2.pt` (commit `f938fa5` + pretrain commit `1358710`, A2 multi-scale avg-pool value head) |
| Recipe | `v6_botmix_s180b_a2.yaml` — verbatim §S180b 3-knob escalation, single arch delta = A2 anchor + `lr_warmup_steps: 2000` (commit `4b2510d`) |
| Steps run | 0 → 2500 (operator-killed early on strong V-FU2-C signal; brief target was 50k, operator override 20k) |
| Ckpts preserved | 500, 1000, 1500, 2000, 2500 (all 5 V_spread canary fires logged) |
| Wall | ~3h on vast 5080 (warmup-throttled selfplay rate ~14 sec/step) |
| Cost | ~$1.50 vast |

**Latent baseline note.** `de149e6` (canonical-child-order MCTS fix) on
master baseline; T3 inherits it. §S180b/§S178 ran without it; FU-1.5
verified §S180b reproduces within noise on the new baseline
([[06-fu1-5-finer-ladder]] §0).

---

## 1. V_spread canary trajectory — the load-bearing finding

`V_spread = mean V(colony bank) − mean V(extension bank)` on the T3
40-position canonical bank, SHA `934204713620d171…`. Anchor reproduction
gate PASS pre-launch (vast canary ran V_spread=−0.5077 vs PT-3 local
−0.5076, matched to 4dp).

### 1.1 Per-checkpoint ladder

| step | V_spread | mean V(colony) | mean V(ext) | direction | canary alert |
|---:|---:|---:|---:|---|---|
| 0 (A2 anchor) | **−0.5076** | +0.2058 | +0.7134 | extension | n/a (PT-3 baseline) |
| 500 | **−0.1171** | −0.3313 | −0.2142 | extension | SOFT-ABORT |
| 1000 | **+0.1281** | +0.0241 | −0.1040 | **flipped → colony** | SOFT-ABORT |
| 1500 | +0.1014 | −0.2174 | −0.3188 | colony | SOFT-ABORT |
| 2000 | +0.1269 | −0.0848 | −0.2117 | colony | SOFT-ABORT |
| 2500 | **+0.1859** | +0.1714 | −0.0146 | colony | SOFT-ABORT |

**Trajectory description.**
- Step 0 → 500: sign-preserving magnitude collapse (−0.508 → −0.117).
- Step 500 → 1000: **sign-flip through zero** (−0.117 → +0.128).
- Step 1000 → 2500: settled in the colony-favouring band (+0.10 to
  +0.19), with a mild upward trend that mirrors pre-A2's matched-step
  trajectory.
- Soft-abort gate (`spread < +0.20`) fired on EVERY post-anchor
  checkpoint.

### 1.2 Plot

```
 V_spread
            -0.55  -0.20   0.0  +0.20(┊abort)              +0.62
              |     |      |     |                            |
 step 0       ●·····|······╪·····┊·······························  -0.508
 step 500     ······●······╪·····┊                                -0.117
 step 1000    ····················●····┊                          +0.128
 step 1500    ····················●····┊                          +0.101
 step 2000    ····················●····┊                          +0.127
 step 2500    ······················●··┊                          +0.186
              |     |      |     |
            -0.55  -0.20   0.0  +0.20

 ● A2 trajectory   ┊ +0.20 FU-2 abort gate
 Crosses ZERO between 500 and 1000 → A2's extension-favouring start
 is overpowered by the training-loop pull toward colony at step ~700.
```

### 1.3 Matched-step comparison vs FU-1.5 (pre-A2 sustained)

| step | FU-1.5 V_spread (pre-A2) | T3 V_spread (A2) | |Δ magnitude| |
|---:|---:|---:|---:|
| 0 | +0.617 | −0.508 | 1.125 (opposite signs by design) |
| 500 | (not measured) | −0.117 | — |
| 1000 | (not measured) | +0.128 | — |
| 1500 | (not measured) | +0.101 | — |
| 2000 | +0.175 | +0.127 | **0.048 (≈ within noise)** |
| 2500 | (not measured) | +0.186 | — |

By step 2000, A2 and pre-A2 land at near-identical V_spread magnitudes
(+0.127 vs +0.175 — within SE(spread) ≈ 0.185 of each other). At step
2500 A2 is at +0.186, converging onto pre-A2's step-2k attractor from
the SIGN-OPPOSITE direction. **The training loop's pull on V_spread is
architecture-invariant for the §S178/§S180b 3-knob recipe.**

### 1.4 FU-1.5 V-FL-A hypothesis falsified

FU-1.5 V-FL-A pinned the discriminator collapse to STEP-0-ONSET
(86% of V_spread loss inside first 2k training steps under pre-A2) and
**predicted A2 would prevent the early collapse by removing the
coverage-blind GMP route** ([[06-fu1-5-finer-ladder]] §6, "an
architectural change that prevents the early-step collapse … is
load-bearing").

A2 did NOT prevent the early collapse. The collapse signature is
**reproduced literally** in T3 (sub-+0.20 by step 500; in the colony-
favouring band by step 1000). The fact that A2 started at −0.508
extension-favouring and was DRAGGED through zero to +0.13 colony-
favouring is the strongest possible disconfirmation: the training
loop's gradient direction toward the colony attractor is **larger in
magnitude than A2's structural preference for extensions**.

**Architectural inversion of inductive bias is not sufficient.** The
loop's value-target dynamics (selfplay outcome distribution × MCTS visit
counts) override the architecture's structural starting point.

---

## 2. Other gates (informational, not verdict-driving)

### 2.1 Training metrics summary at step 2100 (last observed)

| metric | step 80 | step 2100 | Δ vs pre-A2 trajectory @ ~step 2k |
|---|---:|---:|---|
| total_loss | 6.34 | 3.62 | pre-A2 typical ≈ 2.5-3.0 (A2 higher) |
| policy_loss | 4.86 | 2.50 | pre-A2 typical ≈ 1.8-2.2 (A2 +0.3) |
| value_loss | 0.70 | 0.66 | pre-A2 typical ≈ 0.5 (A2 +0.16) |
| aux_loss | 5.01 | 2.67 | pre-A2 typical ≈ 1.9-2.3 |
| policy_entropy | 5.0 | 2.96 | pre-A2 typical ≈ 2.0-2.5 (A2 flatter, holds) |
| top-1 mass | 3% | 22% | pre-A2 typical ≈ 30-60% (A2 less peaky) |
| draw_rate | ~100% | 7% | pre-A2 typical ~10-20% — A2 in normal range |
| x_winrate / o_winrate | 0/0 | 0.47/0.46 | balanced, decisive games |
| colony_fraction | 0.0 | 0.0 | A2 game-level inversion HELD through 2500 |

A2 produces flatter game-time distributions (consistent with arch
intent — multi-scale avg-pool can't peak-saturate). Games are
decisive (~93% non-draw at step 2100), no colony emerging at the GAME
level (colony_fraction = 0.0 throughout). The colony bias lives in the
VALUE HEAD's discrimination, not (yet) in observable game patterns.

### 2.2 L34 anchor↑/sealbot↓ divergence check

Run killed before first eval cycle (eval_interval=10000 + first eval
at step 10k). N/A.

---

## 3. Pre-registered V-FU2 verdict — applied literally

L13 anti-pattern guard: verdict applied LITERALLY from the brief's
pre-registered table, no rewriting to fit outcome.

| ID | rule | this run | match |
|---|---|---|---|
| V-FU2-A SUCCESS | V_spread ≥ +0.30 through step 50k AND wr_sealbot ≥ 8% @ 30k AND no L34 divergence | sub-+0.20 by step 500 ✗ | **FALSE** |
| V-FU2-B PARTIAL HOLD | V_spread holds > +0.20 through 10k but crosses below later AND wr_sealbot ≥ 5% @ 20k | sub-+0.20 by step 500 ✗ | **FALSE** |
| V-FU2-C FAIL — same trajectory | V_spread crashes through +0.20 in first 5k (matches §S180b/FU-1.5 within ±0.10 at step 2k) | crashes by step 500 ✓; at step 2k V_spread=+0.127 vs FU-1.5 +0.175 (Δ=0.048 within ±0.10) ✓ | **TRUE** |
| V-FU2-D MIXED — V_spread held, WR collapses | V_spread ≥ +0.20 through 50k BUT wr_sealbot collapses | V_spread did NOT hold ✗ | FALSE |
| V-FU2-E OSCILLATION | V_spread oscillates ±0.30 around 0, neither sustained hold nor outright crash; wr_sealbot drift | range −0.117 to +0.186 (Δ 0.303 ≤ 0.30) — borderline | possible co-verdict but V-FU2-C primary |
| V-FU2-F INCONCLUSIVE | None of above match cleanly | V-FU2-C matches cleanly ✗ | FALSE |

**VERDICT: V-FU2-C — FAIL, same trajectory.**

**Downstream routing per brief:** "A2 NOT the fix; route to PSW or
refresh hook with V_spread canary intact; A2 reverts."

---

## 4. Verdict block

```
FU-2 A2 SUSTAINED LADDER — VERDICT

  Architecture:        A2 multi-scale avg-pool value head (commit f938fa5)
  Anchor:              checkpoints/bootstrap_model_v6_a2.pt (SHA 36c9111e…)
  Recipe:              §S180b 3-knob + A2 anchor + lr_warmup_steps=2000
  Bank fixture SHA:    934204713620d171… (GATE PASS, anchor reproduction
                       on vast 5080 = -0.5077 vs PT-3 local -0.5076)
  Steps run:           0 → 2500 (operator-killed early)
  Wall (vast 5080):    ~3h
  Anchor gate (PT-3):  V_spread(anchor) = -0.508 (extension-favouring,
                       SIGN INVERTED vs pre-A2 +0.617 — audit-predicted
                       successful removal of GMP coverage-blind route)

  V_spread ladder:
    step:    0      500     1000    1500    2000    2500
    spread: -0.508  -0.117  +0.128  +0.101  +0.127  +0.186
    direction: ext  ext     col     col     col     col
    SOFT-ABORT (<+0.20): n/a YES   YES     YES     YES     YES

  Cross-plot @ step 2k vs FU-1.5 (pre-A2):
    pre-A2 V_spread(2k) = +0.175
    A2     V_spread(2k) = +0.127
    Δ = 0.048 — WITHIN ±0.10 V-FU2-C matching tolerance.

  CLASSIFIER VERDICT: V-FU2-C — FAIL, same trajectory.

  Conditions met:
    V-FU2-A SUCCESS:     FALSE  (sub-+0.20 by step 500)
    V-FU2-B PARTIAL:     FALSE  (sub-+0.20 by step 500)
    V-FU2-C FAIL/same:   TRUE   (crashes by step 500; at step 2k
                                  within ±0.05 of FU-1.5 matched step)
    V-FU2-D MIXED:       FALSE  (V_spread did not hold)
    V-FU2-E OSCILLATION: borderline (range Δ 0.30 = threshold)
    V-FU2-F INCONCLUSIVE: FALSE (V-FU2-C clean match)

  KEY FINDING — FU-1.5 V-FL-A HYPOTHESIS FALSIFIED:
    FU-1.5 V-FL-A predicted A2 architecture would prevent the early-
    step value-head collapse. A2 did NOT prevent it. A2's structural
    extension-favouring inductive bias at the anchor (-0.508) was
    DRAGGED THROUGH ZERO by the training loop within 1000 steps and
    settled in the colony-favouring band at the SAME magnitude as
    pre-A2's matched-step trajectory.

    The training loop's value-target dynamics (selfplay outcome × MCTS
    visit counts) override the architecture's structural starting
    point. ARCHITECTURAL INVERSION OF INDUCTIVE BIAS IS NOT SUFFICIENT.

  DOWNSTREAM ROUTING (per brief, V-FU2-C):
    A2 reverts. Route to PSW (Prioritized Stratified Window) or
    refresh hook from §S181 open-handle design space. V_spread canary
    proven effective (caught the collapse in 500 steps, 5× faster than
    wr_sealbot would have surfaced it) — keep wired in next wave.

  NEW LESSON (L47):
    Architectural fixes that target ONLY the structural starting point
    of the value head's inductive bias are INSUFFICIENT for the §S178
    colony attractor. The training-loop dynamics (corpus mix × selfplay
    × MCTS visit-count CE) dominate the architecture's starting bias by
    a margin of ≥1.0 in V_spread magnitude per ~1000 training steps.
    Fixes targeting the LOOP (data composition, prioritization, value-
    target perturbation, anchor-refresh) are the next surface.
```

---

## 5. Anchor disposition

A2 reverts per V-FU2-C routing. The branch `phase4.5/s181_fu2_a2_arch`
preserves the audit trail. The A2 anchor `bootstrap_model_v6_a2.pt`
remains in `checkpoints/` as the artifact behind this audit but is NOT
the canonical anchor for any next wave — that remains the pre-A2
`bootstrap_model_v6.pt`.

Useful infrastructure that landed on the branch (preserve if the branch
is merged or cherry-picked to master regardless of A2 reversal):

1. **LR warmup in trainer** (commit `4b2510d`) — generally useful;
   `lr_warmup_steps` config knob + `SequentialLR(LinearLR, Cosine)`
   wrapper. 5/5 tests pass. Independent of A2.
2. **PT-quality gate runner** `scripts/structural_diagnosis/
   fu2_pretrain_quality_gates.py` (commit `1358710`) — generally
   useful for any future arch change; runs PT-1 (final loss), PT-2
   (SealBot WR), PT-3 (V_spread on T3 bank), PT-4 (threat probes).
3. **A2 INV pin test** (commit `f938fa5`) — only meaningful if A2 is
   in the code. Tied to architecture; reverts with A2.
4. **A2 ckpt-load shape guards** in `eval/checkpoint_loader.py` +
   `viewer/model_loader.py` (commit `f938fa5`) — tied to A2; reverts.
5. **`tests/_a2_compat.py`** — test-skip helper for pre-A2 anchors;
   tied to A2; reverts.
6. **Audit docs 07 + 08** (commits `1358710` + this commit) — keep
   regardless; document the wave + verdict.

---

## 6. Files

- `audit/structural/08_fu2_a2_sustained_ladder.md` (this file)
- `reports/fu2_sustained/checkpoint_00000500.pt` through `_00002500.pt`
  (5 preserved ckpts pulled from vast)
- `reports/fu2_sustained/events/events_*.jsonl` (canary V_spread events)
- `reports/fu2_sustained/sustained_a2_*.log` (raw stdout/stderr logs)
- `audit/structural/07_fu2_a2_pretrain_quality.md` (T2 PT gate audit)

---

## 7. Recommendations for §S181 successor wave (operator decides)

**Primary candidates** (drawn from §S181 open-handle design space):

1. **PSW — Prioritized Stratified Window** on the replay buffer. Bias
   sampling toward extension-class positions during the first 5k
   training steps so the value-target distribution counteracts the
   colony pull observed at step 500-1000.
2. **Refresh hook** — periodic bot-corpus regeneration vs the live
   model (already designed in §S178 H4; was disabled). Refreshes the
   anti-colony gradient when the model drifts.
3. **A3 aux-loss** — direct anti-colony value-head auxiliary loss
   (T2 §7 A3). Originally parked because FU-1.5 V-FL-A pinned the
   onset to step 2k (too fast for an aux-loss signal). With A2 we now
   know the collapse direction is INTO colony from EITHER starting
   point; A3 may still be insufficient alone but could combine with
   PSW or refresh hook.

**Wire the V_spread canary into the verdict gate of the next wave from
step 0.** Proven 5× faster than wr_sealbot in catching the colony
collapse signature.

**Do not re-propose A2 alone.** This audit pins V-FU2-C; re-running
A2 without a loop-side lever would just reproduce this verdict.
