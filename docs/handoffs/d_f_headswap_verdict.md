# D-F HEADSWAP VERDICT — R1 tail-scored gate: PASS-JOINT (marginal)

> Operator narrative ruling goes ABOVE this line. Tables below are the
> mechanically-recomputed R1 metrics (tail-scored gate per operator ruling R1;
> the dispatcher pre-registered TAIL-MASS as the 65-bin arm's score, not the
> decoded-mean the original RECIPE gate used).
>
> Recompute script: `headswap_safe/confirm_r1_tail_gate.py`
> (pure-numpy, PYTHONHASHSEED-stable, reuses tested `scripts/headswap/metrics.py`
> primitives `auc` / `greedy_match` / `cluster_bootstrap_delta_auc` / `pool_arm_scores`).
> Fixes applied: seed-pooled (mean v, mean tail_mass) before matching; sorted
> position_ids + PYTHONHASHSEED pin for reproducibility; canary on RAW t not the
> tautological ply_band fallback.

## VERDICT LABEL: PASS-JOINT (marginal)

C.tail − D.v = **+0.052**, CI95 **[+0.018, +0.087]** → point ≥ +0.05 AND CI
excludes 0. The load-bearing tail-scored gate clears. Head-shape (65-bin) recovers
losing-tail signal the fair scalar comparate (decoded-v) missed, but ONLY the
last-block-unfrozen arm (C vs D) clears; the frozen-trunk arm (B vs A) does not.

## Matched set (sorted / PYTHONHASHSEED-pinned)

- EXACT match: **213 pairs = 426 matched** (gate n≥200: PASS) — gate uses EXACT.
- RELAXED (nearest-band): 234 pairs = 468 matched (PASS).
- Distinct source games: 151 (pos) / 140 (neg).
- Deterministic across PYTHONHASHSEED ∈ {0,1,2,3} (matched_n + point ΔAUC + CI byte-identical).

## Canary (phase confound on the matched set)

| axis | n | AUC | verdict |
|------|---|-----|---------|
| (a) ply_band (RECIPE match axis, full) | 426 | **0.500** | CLEAN (≤0.55) |
| (b) raw fine-t (covered subset only) | 213 pos / 28 neg | 0.581 | COVERAGE-LIMITED — see note |

RECIPE §Negatives matches 1:1 on `ply_band = t//10`; exact match ⇒ bands balance ⇒
ply_band canary = 0.500 by construction (the axis the negatives were built to satisfy).
The fine-t canary (red-team target 0.504) is **NOT reproducible locally**: the safe-side
`v_raw`/`t` for ~185 of 213 matched negatives lives in `negatives_v2_wp2.jsonl`, which is
MISSING from every local backup. On the 28 covered negatives the 0.581 is a class-imbalance
artifact (213 deep-t losses vs 28 shallow-t safes), NOT ply-confounding. Matched set is
NOT flagged NO-TEST — the RECIPE canary (a) is clean.

## Per-arm AUC on the matched set (loss vs safe)

| arm | score | matched AUC |
|-----|-------|-------------|
| A | decoded v (−v) | 0.885 |
| D | decoded v (−v) | 0.878 |
| B | tail-mass | 0.927 |
| C | tail-mass | 0.931 |
| B | decoded v (−v) | 0.867 |
| C | decoded v (−v) | 0.880 |

## THE GATE — cross-arm tail contrasts (cluster bootstrap by source game, 10k)

| contrast | hi | lo | ΔAUC | CI95 | SE | r | clears +0.05 & CI>0 |
|----------|----|----|------|------|----|----|---------------------|
| **C.tail − D.v** | C tail-mass | D −v | **+0.052** | **[+0.018, +0.087]** | 0.018 | 0.841 | **YES (load-bearing)** |
| B.tail − A.v | B tail-mass | A −v | +0.042 | [+0.007, +0.076] | 0.018 | 0.849 | no (point < +0.05) |

**Match-variant sensitivity (why "marginal"):** on the RECIPE-prescribed EXACT match
(n=426, gate uses EXACT since it passes n≥200) C.tail−D.v = +0.052 clears. On the RELAXED
nearest-band match (n=468) it drops to +0.036 (CI [+0.005,+0.068] still >0, but point <
+0.05 → does NOT clear). The pre-registered gate match is EXACT → PASS-JOINT stands, but the
effect sits right at the +0.05 threshold and the nearest-band relaxation pushes it under —
hence PASS-JOINT (marginal), not a clean PASS.

## Continuity — decoded-v contrasts on the matched set (original RECIPE gate)

| contrast | ΔAUC | CI95 | SE | r |
|----------|------|------|----|----|
| B.v − A.v | −0.018 | [−0.028, −0.008] | 0.005 | 0.957 |
| C.v − D.v | +0.001 | [−0.009, +0.011] | 0.005 | 0.957 |

Reproduces the frozen-verdict decoded-v numbers (−0.017, −0.001 → the original TRUNK-FORK
gate). The tail-scored re-ruling is what flips C-vs-D from ~0 to +0.052.

## Anchor — original 248k scalar head (v_raw) on the matched set

| metric | value | note |
|--------|-------|------|
| AUC(−v_raw) | 0.771 | on 213 loss / **28 safe** (COVERAGE-LIMITED — v2 negatives missing; red-team's 0.824 needed full safe coverage) |
| mean v_raw on losses | **+0.730** | 213 losses (100% covered) — the LEVEL defect (red-team +0.73, EXACT) |

Anchor confirms the trunk is **not rank-blind** (mean loss v_raw = +0.73, deeply optimistic
on positions that are actually lost) and the defect is a LEVEL defect, not a ranking defect —
motivating the distributional head. AUC understated here only by the safe-side coverage gap.

## Data-gap flag

`negatives_v2_wp2.jsonl` (safe-side `v_raw`/`t` for ~3546 of 4204 negatives) is absent from
all local backups. It affects ONLY: (i) the fine-t canary balance, (ii) the anchor AUC safe
side. It does NOT affect: the tail-scored gate (tail_mass fully present in the scores tables),
the decoded-v contrasts, or mean-v-on-losses (losses 100% covered). If the gate needs the
balanced fine-t canary + full anchor AUC, regenerate v2 negatives per RECIPE §Negatives.

## Discrepancies vs red-team scratch

| metric | this run | red-team | status |
|--------|----------|----------|--------|
| C.tail − D.v | +0.052 [+0.018,+0.087] | +0.052 [+0.018,+0.087] | EXACT MATCH |
| decoded B−A / C−D | −0.018 / +0.001 | ~−0.017 / ~−0.001 | MATCH |
| mean v_raw on losses | +0.730 | +0.73 | EXACT MATCH |
| fine-t canary | 0.581 (28 neg) | 0.504 (balanced) | NOT reproducible — v2 negatives missing |
| anchor AUC | 0.771 (28 safe) | 0.824 (full) | understated by coverage gap |
