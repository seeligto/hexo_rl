# §S181 Track A — A4 H-CE-STRENGTH: per-class CE gradient asymmetry

**Wave:** §S181-AUDIT Track A. **Subtask:** A4. **Branch:**
`phase4.5/s181_audit_track_a`. **Date:** 2026-05-23.

> **VERDICT: INCONCLUSIVE — literal.** Per-sample trunk-output gradient
> L2 ratio colony/extension = **1.21**. The L43 entropy asymmetry
> (colony H_frac 0.50 vs extension 0.80) is reproduced, but the
> downstream gradient strength asymmetry is +21% — well below the +50%
> (ratio > 1.5) ASYMMETRY-CONFIRMED threshold and also outside the
> NEUTRAL band [0.85, 1.15]. Colony positions do push the trunk harder
> than extension positions per sample, but only mildly.

---

## 0. Inputs

| | |
|---|---|
| Bank | `tests/fixtures/value_spread_bank.json` |
| Bank SHA-256 | `934204713620d171…dcc23991` (gate PASS) |
| Anchor model | `checkpoints/bootstrap_model_v6.pt` |
| MCTS settings | n_sims=400, c_puct=1.5, dir_α=0.05, ε=0.10, leaf_batch=8 |
| RNG seed | 20260523 |
| Wall (laptop CPU) | ~46 s |

**Method recap.** For each T3-bank position: realize the Board, run
production MCTS using the bootstrap anchor to generate a visit-count
target (sum-to-1 distribution over 362 actions). Compute the CE loss
between the model's `log_softmax(policy_logits)` and the target.
Take the gradient L2 of the loss with respect to the **trunk's output
tensor** (intermediate (1, 128, 19, 19) feature map), not the model
weights — cheaper, isolates per-position gradient contribution
independent of the full backward through the policy head's weights.

---

## 1. L43 entropy reproduction

| class | n | target H_frac (this audit) | L43 reference H_frac |
|---|---:|---:|---:|
| colony    | 20 | **0.501** | 0.484 |
| extension | 20 | **0.799** | 0.805 |

L43's entropy asymmetry reproduces within ~3% (small differences
attributable to RNG seed for the Dirichlet noise + tree-search
non-determinism in leaf order). Colony MCTS visit-count targets are
~62% as entropic as extension targets — sharper-and-more-concentrated.

---

## 2. CE loss and gradient L2 per class

| class | n | mean CE loss | mean trunk grad L2 | std grad L2 |
|---|---:|---:|---:|---:|
| colony    | 20 | **75.41** | **9.031** | 0.600 |
| extension | 20 | **53.70** | **7.463** | 0.321 |
| **ratio col/ext** | | 1.40 | **1.21** | — |

**Observations.**
- CE loss is +40% higher on colony positions. The anchor model is more
  misaligned with the MCTS-derived colony policy targets than with
  extension targets. Consistent with the anchor's known colony-favouring
  value bias (+0.617 V_spread on T3): the model values colony positions
  highly but its POLICY is not well-aligned with the sharp colony
  MCTS visit counts.
- Trunk-output gradient L2 is +21% higher on colony positions. Sharper
  targets + larger CE misalignment combine to push the trunk harder per
  colony sample.
- Std dev of grad L2 is ~2× larger on colony (0.60 vs 0.32) — colony
  positions exhibit more variance in gradient strength across the
  20-position colony subset.

---

## 3. Pre-registered verdict applied LITERALLY (L13 guard)

| condition | rule | this run | match |
|---|---|---|---|
| ASYMMETRY-CONFIRMED | ratio > 1.5 | 1.21 | **FALSE** |
| NEUTRAL | ratio in [0.85, 1.15] | 1.21 | **FALSE** |
| REVERSE-ASYMMETRY | ratio < 0.67 | 1.21 | **FALSE** |
| INCONCLUSIVE | otherwise | | **TRUE** |

**VERDICT: INCONCLUSIVE (literal).**

### Interpretation guidance for A6

- **Direction confirmed, magnitude under-threshold.** The L43 mechanism
  (colony targets sharper → larger gradient) does fire, but at
  ~+21%, not the >50% the pre-registered threshold called for.
- **H-CE-STRENGTH is a co-driver, not dominant.** At a 1.21× gradient
  asymmetry, each colony sample contributes ~21% more trunk pressure
  per gradient step than each extension sample. Compounded across
  thousands of training steps with corpus colony fraction ~26%
  (A1 result), this is a real but limited effect.
- **Upper bound on per-step V_spread pull (back-of-envelope).** The
  gradient strength ratio is multiplicative on the loop's effective
  per-class sample weight. If the buffer mix gave colony and extension
  equal sample counts, H-CE-STRENGTH alone would imprint a ~21%
  imbalance in trunk update direction toward colony per step. With
  corpus colony fraction at 26%, this remains a small contributor.

**A6 dominant-source impact.** H-CE-STRENGTH is real and quantified but
not load-bearing in isolation. Per the routing table, "H-CE-STRENGTH
confirmed dominant → per-class gradient rescaling OR target temperature
on colony positions" — that lever's expected upside is modest given the
1.21 ratio. Recommendation: keep the lever in the design space but
combine with another (PSW, refresh hook, or pretrain-class-weighted LR
from A5 if H-PRETRAIN fires).

---

## 4. Outputs

- This file
- `audit/structural/track_a/A4_h_ce_strength.json` (sidecar)
- `scripts/structural_diagnosis/track_a/a4_h_ce_strength.py` (this script)
