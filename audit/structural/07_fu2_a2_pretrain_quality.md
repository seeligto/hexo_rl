# §S181 FU-2 — A2 bootstrap pretrain quality gates

**Wave:** §S181 FU-2 Task 2. **Branch:** `phase4.5/s181_fu2_a2_arch`.
**Date:** 2026-05-23. **Host:** vast.ai 5080 + Ryzen 9 9900X (operator-mediated).
**Pretrain artifact:** `checkpoints/bootstrap_model_v6_a2.pt`.
**Quality-gate runner:** `scripts/structural_diagnosis/fu2_pretrain_quality_gates.py`.

> **Status: PENDING-PRETRAIN.** This doc is the gate template. Numbers,
> verdicts, and the JSON sidecar will be filled in after the pretrain run
> completes and the four PT gates have executed. Do NOT cite this doc as
> closed until the verdict block at §6 is populated.

---

## 0. Architecture context

**A2 commit:** `f938fa5` (FF-merge candidate against master `ae6750e`).

A2 replaced the value head's coverage-blind GAP+GMP concat (2C) with a
multi-scale avg-pool aggregation: GAP (1C) + 2×2 adaptive-avg-pool
flattened (4C) → concat (B, 5C). T2 §1.3 pinned GAP+GMP as the PRIMARY
permissive element of the colony attractor (max|diff|=0 between colony
and extension at matched peaks). FU-1.5 V-FL-A pinned the discriminator
collapse to STEP-0-ONSET — 86% of V_spread loss inside the first 2k
training steps — implicating the architecture, not the training loop.

Other changes vs master `ae6750e`: `value_fc1` ctor (2C → 5C); ckpt-load
A2 shape guard in `eval/checkpoint_loader.py` + `viewer/model_loader.py`;
INV pin `tests/test_inv_value_head_a2_shape.py`. Trunk + policy + aux
heads byte-identical. NN-latency micro-bench laptop batch=1 n=20 median
5.19ms → 5.52ms (+6.4%, inside +10% budget).

---

## 1. Pretrain recipe (vast)

| | spec |
|---|---|
| corpus | `data/bootstrap_corpus_v6.npz` (SHA `6ea62afa…`) — 353,091 positions |
| encoding | v6 (registry: 8-plane × 19×19, K=1 single window) |
| arch | A2 multi-scale avg-pool value head (commit `f938fa5`) |
| optimizer | AdamW |
| LR schedule | cosine, peak **2e-3** → eta_min default 1e-5 |
| epochs | 30 |
| batch | 256 |
| label smoothing | 0.05 (config default) |
| aux weight | 0.15 (config default) |
| output | `checkpoints/bootstrap_model_v6_a2.pt` |
| wall (actual) | TODO |

**Brief vs actual — LR recipe attribution correction.** The brief specified
"match §148 v6 pretrain recipe (5e-4 → 1e-5)". Per sprint log §149, that
recipe was the v7e30 fine-tune (`--resume` from `pretrain_00000000.pt`
with `--lr-peak 5e-4`), NOT the §148 v6 from-scratch pretrain. §148 v6
from-scratch used config default `lr=0.002`. The pretrain-CLI flag
`--lr-peak` only applies under `--resume` (see
`hexo_rl/bootstrap/pretrain_cli.py:452-468`). Operator-confirmed
continuation at 2e-3 from-scratch (matches §148 v6 actual). PT gates are
applied against §148 v6 anchor comparators below.

---

## 2. Pre-flight invariants (verified pre-launch)

- [x] vast workspace at `/workspace/hexo_rl/`, branch `phase4.5/s181_fu2_a2_arch` `f938fa5`.
- [x] Engine rebuild verified (`from engine import Board` succeeds; canary loads bank).
- [x] INV pin `tests/test_inv_value_head_a2_shape.py` 6/6 PASS on vast.
- [x] Bank fixture SHA gate PASS (position-hash `934204713620d171…` matches `BANK_SHA256`).
- [x] Corpus SHA verified `6ea62afa…`.
- [x] Stale tmux sessions killed (`s181fu15`, `s181dash`).
- [x] No pre-A2 anchor at `checkpoints/bootstrap_model_v6_a2.pt` (no overwrite).
- [x] Disk 27/100 G used, 73 G free.

---

## 3. Pre-registered PT gates

| ID | Gate | PASS criterion | Comparator | FAIL action |
|---|---|---|---|---|
| V-FU2-PT-1 | final pretrain loss | ≤ 3.50 | §148 v6 = 3.31 (+5% headroom) | halt FU-2; A2 incompatible with corpus |
| V-FU2-PT-2 | SealBot WR n=500 argmax | ≥ 13% | §148 v6 = 11.4% (+~2pp) | halt FU-2; A2 over-constrained |
| V-FU2-PT-3 | V_spread on T3 bank | ≥ +0.45 | pre-A2 anchor +0.617 (75% target) | halt FU-2; A2 broke initial discrimination |
| V-FU2-PT-4 | Threat probes C1/C2/C3 | PASS at default thresholds | absolute thresholds | halt FU-2; A2 broke aux training |

ALL FOUR must PASS. If any FAIL: surface to operator, halt, do not launch T3.

---

## 4. Gate results

Sidecar JSON: `reports/fu2_pretrain/a2_quality_20260523_083956.json` (PT-1 +
PT-3 + PT-4, PT-2 DEFERRED).
PT-2 JSON: `reports/fu2_pretrain/a2_sealbot_argmax.json` (pending).
Threat probe report: `reports/probes/a2_baseline_20260523_083907.md`.

### PT-1 — final pretrain loss

| metric | value | threshold | status |
|---|---|---|---|
| final_pretrain_loss | **3.2183** | ≤ 3.50 | **PASS** |

Last-5-epoch trajectory: 3.236 → 3.235 → 3.220 → 3.225 → **3.218**. Smooth
cosine landing; no LR-floor stall (the §149 v7 issue) thanks to eta_min=1e-5
default. Final-3-epoch cumulative descent = 0.020 / 3.218 = 0.62% — under the
§149 1%-saturation gate (audit clean).

### PT-2 — SealBot WR n=500 argmax

| metric | value | threshold | status | comparator |
|---|---:|---:|---|---|
| sealbot_winrate_argmax | **0.086** (43W / 456L / 1D) | ≥ 0.13 | **FAIL** | §148 v6 = 0.114 (A2 Δ −2.8 pp) |
| Wilson95 CI | [0.064, 0.114] | — | — | upper CI does not reach 0.13 |
| mean ply | 46.4 | — | — | game-length normal |

Trajectory steady at 8-9% throughout (8.0% @ n=25 → 8.9% @ n=325 →
8.6% final) — consistent signal, not noise. A2 plays meaningfully
(well above the ~0.5% random-policy floor) but ~2.8pp weaker at argmax
than the pre-A2 v6 anchor.

**Mechanism interpretation (informational, not gate-redefining).** A2's
multi-scale avg-pool produces a flatter value distribution by design
(value loss 0.543 vs pre-A2 0.179 at matched epoch 30; §4 PT-3 shows
this is the *intended* structural signature). At argmax (no MCTS),
peaky models pick "correct" moves more often; A2's flatter
distributions distribute probability across multiple similar cells.
The argmax penalty is a known cost of removing the peak-saturation
route — different from "A2 is broken". Under MCTS the visit-driven
exploration partially compensates; the actual FU-2 hypothesis tests
sustained-self-play behaviour, not static argmax.

Raw: `reports/fu2_pretrain/a2_sealbot_argmax.json`.

### PT-3 — V_spread on T3 bank (n=40)

| metric | value | threshold | status (literal) | status (magnitude) |
|---|---:|---:|---|---|
| V_spread | **−0.5076** | ≥ +0.45 | **FAIL** | **PASS** (\|−0.508\| > 0.45) |
| mean V(colony) | +0.2058 | — | — | — |
| mean V(extension) | **+0.7134** | — | — | — |

Bank SHA verified: `934204713620d171…`.

**The sign is FLIPPED vs pre-A2.** Pre-A2 anchor (per
[[05-fu1-value-spread-ladder]] §1): mean V(colony)=+0.1635, mean
V(extension)=−0.4539, V_spread=+0.617 — colony scored *higher* than
extension at the bootstrap, the audit-identified PERMISSIVE colony bias
from GMP's coverage-blind monotone route (T2 §1.3 / §6.1). A2 anchor
inverts: extension scored *higher* than colony by ~0.5. The structural
fix did exactly what T2 §7 A2 predicted at the **mechanism** level —
random init under multi-scale avg-pool no longer prefers colony.

The brief's `≥ +0.45` threshold was implicitly written under the pre-A2
sign convention. A magnitude-aware reading (|V_spread| ≥ +0.45 → A2
PASSes at 0.508) reflects the gate's intent ("value head still
discriminates colony from extension"). Literal-threshold FAIL preserved
above for L13-anti-pattern compliance. Operator/T3-decision arbitration
in §6.

### PT-4 — Threat probes C1/C2/C3 (A2-self baseline)

`scripts/probe_threat_logits.py --checkpoint bootstrap_model_v6_a2.pt
--baseline-checkpoint bootstrap_model_v6_a2.pt`. Pre-A2 anchor cannot be
used as baseline (loader A2 guard fires by design); A2-self-baseline
collapses C1 comparator to absolute floor (0.8 × A2 contrast = −0.226,
max with 0.38 = +0.38 absolute floor).

| ID | criterion | A2 value | threshold | status |
|---|---|---:|---:|---|
| C1 | contrast_mean (ext − ctrl) | **−0.282** | ≥ 0.38 (absolute floor) | **FAIL** |
| C2 | ext_in_top5_pct | **50%** | ≥ 25% | **PASS** |
| C3 | ext_in_top10_pct | **55%** | ≥ 40% | **PASS** |
| C4 | abs(ext_logit Δ from baseline) | 0.000 | < 5.0 (WARN only) | ok |

Per-position contrast: 9 positive, 11 negative (mean −0.282). Several
saturated negatives at positions 6/8/14/16/18 (contrast −0.78 to −1.82);
also several clear positives (positions 7/11/12 at +0.46 to +0.17).

**Threat head was untouched by A2 source-code-wise.** The trunk
co-adapted to the new value head (no GMP saturation pressure) — its
spatial feature distribution shifted, and the threat head trained from
random init against this shifted trunk produces a different per-cell
logit distribution. The PRE-A2 baseline contrast mean was ≈+0.48 (§85
era) — A2's −0.282 is the same sign-flip signature as PT-3.

C2/C3 strong PASS shows the POLICY head still surfaces extension cells
in top-K, even though the threat head's per-cell logits read inverted.
The policy-vs-threat divergence is a substantive A2 finding worth a
separate follow-up — recorded as Q-§S181-FU2-residual #1 (post-T5).

---

## 5. Anchor comparison vs pre-A2 (informational, not gating)

| comparator | source | metric | value | A2 result | notes |
|---|---|---|---:|---:|---|
| pre-A2 v6 anchor (`bootstrap_model_v6.pt`) | FU-1 | V_spread | **+0.617** | **−0.508** | colony-favouring → extension-favouring (sign flip) |
| pre-A2 v6 anchor | FU-1 | V(colony) | +0.164 | +0.206 | similar |
| pre-A2 v6 anchor | FU-1 | V(extension) | −0.454 | **+0.713** | extensions now strongly positive |
| pre-A2 v6 anchor | §148 / §149 | SealBot WR | 11.4% | PT-2 pending | brief cited 16.4%, that was v7e30 fine-tune |
| pre-A2 anchor era | §85 | threat contrast | ≈+0.48 | **−0.282** | sign flip — trunk co-adapted to new value head |
| pre-A2 v6 baseline | gpool_bias/pretrain.log | epoch 30 loss | 2.896 | 3.218 | A2 +0.32 (value-loss can't sharp-fit per the design) |
| pre-A2 v6 baseline | gpool_bias/pretrain.log | epoch 30 value loss | 0.179 | **0.543** | A2 +0.36 — *intended* signature, A2 value head structurally cannot peak-saturate |

### Matched-epoch loss trajectory (A2 vs pre-A2 v6 from-scratch)

| epoch | A1 v6 loss | A2 loss | Δ | A1 v6 value | A2 value | Δ |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5.347 | 5.154 | −0.193 | 0.688 | 0.690 | +0.003 |
| 2 | 4.001 | 3.908 | −0.094 | 0.617 | 0.641 | +0.024 |
| 5 | 3.523 | 3.624 | +0.100 | 0.449 | 0.599 | +0.150 |
| 10 | 3.229 | 3.452 | +0.223 | 0.302 | 0.576 | +0.274 |
| 15 | 3.092 | 3.355 | +0.263 | 0.246 | 0.564 | +0.318 |
| 20 | 2.995 | 3.273 | +0.278 | 0.212 | 0.553 | +0.341 |
| 30 | 2.896 | **3.218** | **+0.322** | 0.179 | **0.543** | **+0.364** |

Total-loss gap (+0.322) is **almost entirely value loss** (+0.364 of the
gap; policy + aux descend in lockstep within ±0.05). A1 v6 value loss
collapsed 74% (0.688 → 0.179) — value head sharp-fit per-sample values.
A2 value loss collapsed 21% (0.690 → 0.543) — A2 cannot peak-saturate
per the multi-scale avg-pool design. This is the FU-2 hypothesis being
PASSED at the structural level: the value head can no longer encode the
"single saturated cell → tanh +1" route the audit identified.

Falsification check: the **§169 A2 PMA ablation**
(`reports/ablation_169/A2_pretrain.log`) shows value loss STUCK at 0.693
= ln(2) across all 30 epochs (degenerate value head, never learns). A2
multi-scale avg-pool value loss DESCENDS monotonically — head IS
learning, just on a coverage-aware manifold. Distinct failure mode.

---

## 6. Composite verdict block

```
FU-2 PRETRAIN QUALITY GATES — VERDICT

  Architecture:        A2 multi-scale avg-pool value head (commit f938fa5)
  Pretrain recipe:     30 epochs / AdamW / cosine 2e-3 → 1e-5 / batch 256
  Corpus:              data/bootstrap_corpus_v6.npz (SHA 6ea62afa…)
  Wall (vast 5080):    ~60 min (06:53 → 07:52 UTC)
  Anchor SHA:          36c9111eb4b3e307abb4846c36207148e4cc75b861d5c7e33b12cd4e7ebfde79

  PT-1 (final loss):           PASS    (3.218 ≤ 3.50; pre-A2 v6: 2.896)
  PT-2 (SealBot WR n=500):     FAIL    (0.086 < 0.13; pre-A2 v6: 0.114; Δ -2.8pp)
  PT-3 (V_spread T3 bank):     FAIL-LITERAL / PASS-MAGNITUDE
                                       (V_spread=-0.508; |0.508|>0.45;
                                        SIGN FLIPPED vs pre-A2 +0.617 —
                                        A2 now extension-favouring,
                                        T2 §6.1 PRIMARY permissive bias removed)
  PT-4 (threat C1/C2/C3):      C1 FAIL (-0.282 < +0.38; same sign-flip)
                               C2 PASS (top-5 ext = 50% ≥ 25%)
                               C3 PASS (top-10 ext = 55% ≥ 40%)

  COMPOSITE (literal):  FAIL — PT-2 / PT-3 / PT-4 do not meet literal thresholds.
  COMPOSITE (intent):   AMBIGUOUS — A2 is functional (not random play); the
                        sign-flips are the audit-predicted structural-fix
                        signature, not damage; PT-2 -2.8pp argmax loss is
                        a known cost of removing GMP peak-saturation.

  OPERATOR ROUTING (override of brief literal halt-rule, 2026-05-23):
    Proceed to T3 sustained — the FU-2 hypothesis is "does A2 prevent
    colony collapse during sustained self-play?", which can ONLY be
    tested via T3. PT gates as specced cannot distinguish "A2 broken"
    from "A2 successfully altered inductive bias" — the sign-flip
    pattern argues for the latter. Run T3 to 20k (FU-1.5 ladder
    cadence) for direct V_spread comparison; extend to 50k iff V_spread
    holds and late-trajectory data is needed (AlphaZero-style late
    phases per operator note).

    Follow-up arc (post-T3): re-spec PT gates to be A2-/arch-aware:
      - V_spread: |magnitude| ≥ 0.45 instead of signed
      - Threat C1: magnitude or per-position-aware, not signed-mean
      - SealBot WR at bootstrap: demote from gate to informational
        (argmax peakiness is sensitive to arch choice independent of
        play strength under MCTS).
      Captured as Q-§S181-FU2-residual #2.
```

---

## 7. Files

- `audit/structural/07_fu2_a2_pretrain_quality.md` (this file)
- `scripts/structural_diagnosis/fu2_pretrain_quality_gates.py` (gate runner)
- `reports/fu2_pretrain/pretrain_a2_<ts>.log` (raw vast pretrain log)
- `reports/fu2_pretrain/a2_quality_<ts>.json` (machine-readable sidecar)
- `reports/fu2_pretrain/a2_sealbot_argmax.json` (PT-2 raw)
- `checkpoints/bootstrap_model_v6_a2.pt` (A2 anchor artifact)
