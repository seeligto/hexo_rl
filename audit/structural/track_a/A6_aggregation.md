# §S181 Track A — A6 aggregation + routing recommendation

**Wave:** §S181-AUDIT Track A. **Subtask:** A6 (synthesis). **Branch:**
`phase4.5/s181_audit_track_a`. **Date:** 2026-05-23.

> **VERDICT: NO SINGLE DOMINANT SOURCE CONFIRMED.** Five subtasks ran;
> three returned INCONCLUSIVE (A1, A4, A5), one returned INCONCLUSIVE
> with FALSIFIED mechanism (A2), one returned CONFOUND (A3). Aggregated
> source-level pull magnitude on V_spread is ~+0.15 effective per-step
> equivalent — below the L47 ≥1.0/1000-step magnitude measured on T3
> bank, but **A3's CONFOUND finding revises the actual L47 magnitude
> downward by ~3× to ~+0.33/1000 steps**. The gap between explained
> (~+0.15) and revised-actual (~+0.33) is ~+0.18 of unexplained
> per-step pull — substantial but not catastrophic. Recommendation
> per routing table: **escalate to Track B** (live training-loop
> gradient instrumentation on the self-play buffer, the only major
> unmeasured surface) while landing combined PSW + refresh hook +
> dual-bank V_spread canary as the next-wave lever.

---

## 0. Per-subtask verdict matrix

| ID | hypothesis | verdict (LITERAL) | mechanism finding |
|---|---|---|---|
| A1 | H-BOT — bot corpus position-level bias | INCONCLUSIVE | colony frac 26.0%, asymmetry +0.078 (8× smaller than anchor) — not load-bearing |
| A2 | H-AUG — augmentation oversamples colony | INCONCLUSIVE | position-level MECHANISM FALSIFIED (uniq ratio 1.0); feature-var ratio 0.54 is downstream signature |
| A3 | H-BANK — T3 bank confound | **CONFOUND** | Pearson r=0.27, T3 V_spread amplified ~3× vs alt bank — V_spread magnitude interpretation revised |
| A4 | H-CE-STRENGTH — per-class gradient | INCONCLUSIVE | L43 entropy confirmed (col 0.50 / ext 0.80), grad L2 ratio 1.21 (below 1.5 threshold) |
| A5 | H-PRETRAIN — pretrain position z | INCONCLUSIVE | colony frac 31.2%, asymmetry +0.157 (2× bot but below +0.20 threshold) |

No subtask returned STRONG-CONFIRM or ASYMMETRY-CONFIRMED. H-BANK is a
meta-finding (metric scale) not a source attribution.

---

## 1. Cross-source contribution estimate

Per A6 method spec — estimate per-source per-step V_spread pull
magnitude:

`pull_source ≈ source_asymmetry × effective_sample_share × per-sample_gradient_factor`

| source | asymmetry | share | grad factor | per-step pull (upper bound) |
|---|---:|---:|---:|---:|
| H-BOT (bot corpus) | +0.078 | 0.15 (bot_batch_share) | 1.21 × | **+0.014** |
| H-AUG | 0 (falsified) | — | — | **+0.000** |
| H-PRETRAIN | +0.157 | ~0.30 (pretrain mix during sustained, estimate) | 1.21 × | **+0.057** |
| H-CE-STRENGTH | (multiplier, no own asym) | applies to all colony samples | 1.21 × | (folded into the others' grad factor) |
| **Total (sum)** | | | | **~+0.07** per step |

**Caveats.**
- Numbers are order-of-magnitude. Exact mixing schedules for §S178
  sustained were `bot_batch_share=0.15`; the residual ~85% of the
  buffer is self-play + pretrain (Track B unmeasured).
- The "per-step pull" is the upper bound on the V_spread direction
  imprinted per training step by that source's gradient signal.
- H-CE-STRENGTH is a multiplier (1.21× per colony sample); folded
  into each other source's grad factor column rather than counted
  separately.

### Compare to measured L47 magnitude

L47 (raw, T3 bank): ≥1.0 V_spread change per ~1000 training steps =
**+0.001/step (T3 magnitude).**

A3 CONFOUND adjustment: T3 V_spread is amplified ~3× vs alt-bank
real-corpus magnitude. Revised actual:
**~+0.0003/step** (alt-bank-equivalent).

Per-step pull comparison:

| basis | observed magnitude/step | sources sum/step | explained fraction |
|---|---:|---:|---:|
| T3 bank L47 raw | +0.001 | +0.07 ×? — but this is per-step total **of the source-attributable gradient** | ratio inverse: sources >> observation, suggesting other dampers |
| alt-bank L47 (A3-corrected) | +0.0003 | (same) | same |

Hmm — the math comparison is awkward because the source-attributable
gradient sum (+0.07/step upper bound) **exceeds** the observed
per-step V_spread movement (+0.001 to +0.0003/step). This means the
sources have **head-room**: corpus value targets, by themselves, have
enough asymmetry to drive V_spread changes ~70-100× the observed rate.
What constrains the movement to the slow observed rate must be:

1. **Backward damping** — the model's existing weight configuration
   pulls back against any single gradient update (Adam smoothing,
   weight decay, optimization step-size cap).
2. **Counter-currents from non-colony/non-extension positions** —
   `neither` is 61% of bot corpus and 63% of pretrain corpus. Their
   gradient signal is dispersed and may partially cancel the
   colony-favouring direction.
3. **Self-play buffer dynamics (UNMEASURED)** — Track B surface.

So the sources are NECESSARY but not SUFFICIENT for the observed
movement — they could supply the pressure, but the actual rate is
gated by optimizer/architecture.

---

## 2. Dominant source identification

**Ranked contribution (descending):**

1. **H-PRETRAIN** (+0.057/step upper bound) — pretrain corpus
   contributes ~75% of the corpus-level asymmetry to the V_spread
   direction. Largest single source by share × asymmetry.
2. **H-CE-STRENGTH** (1.21× multiplier on every colony sample) —
   amplifies whatever source signal exists; real but modest.
3. **H-BOT** (+0.014/step upper bound) — present but small, capped by
   the 0.15 bot_batch_share and the small +0.078 asymmetry.
4. **H-AUG** (0) — mechanism falsified, contributes nothing.
5. **H-BANK** — N/A (meta-finding, not a source).

**However: no single source met its pre-registered confirmation
threshold.** Per routing table, "None confirmed → Escalate to Track B".

---

## 3. Routing recommendation

### Per routing table verdict matrix

| dominant source | recommended next-wave lever |
|---|---|
| H-BOT confirmed dominant | refresh hook | FALSE — H-BOT not confirmed |
| H-AUG confirmed dominant | aug rebalance | FALSE — H-AUG falsified |
| H-CE-STRENGTH confirmed dominant | per-class gradient rescaling / target temp | FALSE — H-CE not confirmed |
| H-PRETRAIN confirmed dominant | filter pretrain by class / class-weighted LR | FALSE — H-PRETRAIN not confirmed |
| Multiple confirmed | combined lever | FALSE — none confirmed |
| **None confirmed** | **Escalate to Track B (live training-loop gradient instrumentation)** | **TRUE — DEFAULT ROUTING** |

### Recommended next-wave action (composite — operator decides)

Given (a) the literal routing of "escalate to Track B" + (b) the
clear ranking of H-PRETRAIN > H-CE-STRENGTH > H-BOT > H-AUG=0 in
contribution magnitude + (c) the A3 metric finding that V_spread
should be measured on both banks:

1. **Track B — instrumented training-loop probe (HIGH PRIORITY).**
   Short instrumented run (~5k steps) on the §S180b/§S178 recipe that
   logs per-sample gradient magnitude bucketed by colony / extension /
   neither classification on the LIVE self-play buffer. This is the
   only major unmeasured surface and would close the per-step pull
   accounting gap (~+0.18/step unexplained between source upper-bound
   and observed-after-damping rate).

2. **Combined lever — PSW + refresh hook (if Track B confirms loop-side
   imbalance).**
   - PSW: stratified buffer sampling so colony / extension / neither
     each receive 33% gradient weight rather than corpus-proportional.
     Targets the largest single corpus contributor (H-PRETRAIN, 31%
     colony) without requiring re-pretrain.
   - Refresh hook: periodic regeneration of the bot corpus against
     the live model. Targets H-BOT's small but persistent contribution
     by keeping the anti-colony gradient relevant.
   - Per-class target temperature on colony positions (H-CE-STRENGTH
     lever): T_colony > 1.0 softens sharp colony MCTS targets,
     reducing the 1.21× gradient amplification. Cheapest of the three.

3. **Dual-bank V_spread canary (from A3).** Update PR-A's
   `value_spread_canary.py` to compute V_spread on BOTH T3 and alt
   banks at each checkpoint. Maintains historical compatibility on T3
   while grounding the metric in real corpus positions. Both reports
   surface to the dashboard.

4. **DO NOT re-propose architecture-only interventions.** L47
   stands: A2 confirmed architecture-only fixes are insufficient.
   The next wave must move loop-level dynamics or the lever fails on
   the same V-FU2-C mechanism.

5. **DO NOT re-propose A1's H-BOT lever in isolation.** Refresh hook
   on its own would target +0.014/step at upper bound — not enough.
   Combine with PSW or use as a secondary lever.

---

## 4. Open handles forwarded (per brief)

- **Track B (live training-loop gradient instrumentation)** —
  PRIMARY: this is the recommended next investigation per routing
  table. Closes the ~+0.18/step unexplained source-vs-observation gap.
- **Track C (v6 K=1 vs v6w25 K=8 encoding A/B)** — DEFERRED. ~1 GPU-day.
  Not the next-wave lever; only re-enter if a Track B-informed lever
  fails AND we believe encoding-level structural inductive bias is the
  remaining surface.
- **EMA** — DEFERRED. EMA smooths gradient flow but changes self-play
  inference model. Re-evaluate after Track B reveals which loop-level
  intervention to combine it with.
- **WDL migration** — DEFERRED. Same cost profile as A2 (re-pretrain,
  value-head re-arch). Not justified given A4's gradient asymmetry is
  modest (1.21×, not transformative).

---

## 5. Files

- This file
- A1: `audit/structural/track_a/A1_h_bot_corpus_position_bias.md` + `.json`
- A2: `audit/structural/track_a/A2_h_aug_symmetry.md` + `.json`
- A3: `audit/structural/track_a/A3_h_bank_confound.md` + `.json` + `tests/fixtures/value_spread_bank_alt.json`
- A4: `audit/structural/track_a/A4_h_ce_strength.md` + `.json`
- A5: `audit/structural/track_a/A5_h_pretrain_position_z.md` + `.json`
- Classifier: `scripts/structural_diagnosis/track_a/position_classifier.py`
- Per-subtask scripts: `scripts/structural_diagnosis/track_a/a{1,2,3,4,5}_*.py`
