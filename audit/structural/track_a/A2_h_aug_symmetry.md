# §S181 Track A — A2 H-AUG: augmentation symmetry asymmetry

**Wave:** §S181-AUDIT Track A. **Subtask:** A2. **Branch:**
`phase4.5/s181_audit_track_a`. **Date:** 2026-05-23.

> **VERDICT: INCONCLUSIVE — literal — H-AUG (position-level)
> FALSIFIED.** Position-level unique-variants ratio is 1.00 (both
> colony and extension produce 12 distinct augmented variants under the
> 12-fold hex symmetry). The pre-registered ASYMMETRIC-CONFIRMED
> requires BOTH unique-variants AND feature-variance ratios ≤ 0.6;
> unique-variants is 1.00, failing the first condition. Feature-variance
> ratio IS asymmetric (0.54, below 0.85 neutral band) but reflects
> learned model representation, not an augmentation-rate asymmetry.
> Augmentation does NOT effectively oversample colony positions.

---

## 0. Inputs

| | |
|---|---|
| Bank | `tests/fixtures/value_spread_bank.json` |
| Bank SHA-256 | `934204713620d171…dcc23991` (gate PASS) |
| Anchor model | `checkpoints/bootstrap_model_v6.pt` (v6 trunk 128×12) |
| Symmetries | 12 (6 rotations × {identity, reflect}) — `hexo_rl.augment.luts` |
| Positions | 40 (20 colony + 20 extension) |

---

## 1. Unique-variants per position

For each bank position, apply each of 12 hex-symmetry transformations to
the v6 8-plane state tensor; canonicalize by raw byte-equality (after
the scatter permutation). Count distinct variants.

| class | n | mean unique-variants | std |
|---|---:|---:|---:|
| colony    | 20 | **12.0** | 0.0 |
| extension | 20 | **12.0** | 0.0 |

**Ratio colony / extension = 1.00.** Every position in both classes
produces 12 distinct variants under augmentation. The T3 bank's colony
positions are NOT 6-fold rotationally symmetric — they are spirals
built with alternating-player cell placement, so the symmetry actions
permute the player-colour pattern and produce 12 distinct states.

**Position-level H-AUG mechanism FALSIFIED.** Augmentation does not
collapse colony positions to fewer canonical variants.

---

## 2. Feature-space variance under augmentation

For each position, forward all 12 augmented variants through
`bootstrap_model_v6.pt`'s trunk to produce a (12, 128, 19, 19) feature
tensor. Compute per-cell variance across the 12-axis (axis=0), then
mean over (channel, H, W) cells to obtain a scalar per position.

| class | n | mean feat_var | mean feat_std |
|---|---:|---:|---:|
| colony    | 20 | **0.0369** | **0.0442** |
| extension | 20 | **0.0688** | **0.0828** |
| **ratio col/ext** | | **0.536** | **0.535** |

Colony positions yield ~54% the per-cell feature variance of extension
positions under the same 12 augmentations. The bootstrap model's trunk
responds **more uniformly** to symmetry-augmented colony positions —
they map to a tighter region of feature space.

**Interpretation.** This is a measure of the **model's learned
representation**, not of augmentation per se. Even though the 12
augmented board states are all distinct (§1), the trunk has learned to
collapse them onto a similar manifold for colony positions. Mechanism
hypothesis: the trunk gives less weight to colony-distinguishing
features OR colony positions inherently lie on a lower-dimensional
manifold in trunk-feature space.

Either way, this is NOT what H-AUG was testing. H-AUG was the
hypothesis that 12-fold augmentation **oversamples** rotation-invariant
positions by producing fewer unique training examples per position.
That requires unique-variants to be asymmetric — and it is not (1.00
ratio).

---

## 3. Pre-registered verdict applied LITERALLY (L13 guard)

| condition | rule | this run | match |
|---|---|---|---|
| ASYMMETRIC-CONFIRMED | colony_uniq ≤ 0.6 × ext_uniq AND colony_feat_var ≤ 0.6 × ext_feat_var | uniq 1.00 ✗ ; var 0.54 ✓ | **FALSE** |
| NEUTRAL | both ratios in [0.85, 1.15] | uniq 1.00 ✓ ; var 0.54 ✗ | **FALSE** |
| INCONCLUSIVE | otherwise | | **TRUE** |

**VERDICT: INCONCLUSIVE (literal).**

### Interpretation guidance for A6

- **H-AUG MECHANISM FALSIFIED at the position level.** Augmentation
  does not oversample colony positions — every position gets 12 distinct
  augmented variants. No "fewer effective samples per epoch" effect.
- The feature-variance asymmetry IS real (col/ext = 0.54) but is a
  **downstream signature**, not a source. It reflects what the
  bootstrap value head has already learned about colony vs extension
  positions: lower feature spread for colony → the trunk treats colony
  variants as more similar to each other than extension variants.
  This is **consistent with H6 / L43** (colony positions have lower
  visit-target entropy + colony-favouring value bias) — but it is a
  CONSEQUENCE, not a cause.

H-AUG removed from A6 dominant-source candidates. Augmentation
rebalancing (dropping near-symmetric aug variants) is NOT recommended
— there are no near-symmetric aug variants in the T3 bank to drop.

---

## 4. Outputs

- This file
- `audit/structural/track_a/A2_h_aug_symmetry.json` (sidecar)
- `scripts/structural_diagnosis/track_a/a2_h_aug.py` (this script)
