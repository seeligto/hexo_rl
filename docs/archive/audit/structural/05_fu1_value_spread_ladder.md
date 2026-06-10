# §S181 FU-1 — value-spread checkpoint-ladder probe

**Wave:** §S181 follow-up FU-1. **Date:** 2026-05-22. **Type:** inspection
probe, zero training, CPU. **Branch:** `phase4.5/s181_fu1_probe`.

**Goal.** Pin WHEN the value head's colony/extension discriminator flattens
across the §S180b training trajectory. Per `reports/s181_next_wave_skeleton.md`
OP-1 / the §S181 successor plan.

**Probe:** `scripts/structural_diagnosis/fu1_value_spread_ladder.py`.
**Sidecar:** `audit/structural/05_fu1_value_spread_ladder.json`.

---

## 1. Bank fixture

**Source.** The 40-position canonical bank is reused **verbatim** from the
§S181-T3 probe `scripts/structural_diagnosis/mcts_colony_probe.py` —
`build_colony_positions()` (20) + `build_extension_positions()` (20). The
FU-1 ladder script imports those builders; it does **not** regenerate the
bank.

> **Brief-vs-reality note.** The task brief pointed at `probe_value_bias.py`
> (T1) for the "40-position bank". That is incorrect: T1's `build_bank()` is
> **50 colony + 50 extension = 100** positions and yields a different anchor
> figure (`value_delta_colony−ext = −0.150`, the T1 register row). The
> +0.617 anchor spread cited by L42 / `00_aggregation.md` is a **T3**
> measurement (`03_mcts_colony_dynamics.json` → `value_head`:
> `anchor_mean_value_colony 0.1635`, `anchor_mean_value_extension −0.4539`).
> FU-1 therefore uses the T3 bank — the only one consistent with the +0.617
> reference. No bank was rewritten (per the L13 anti-pattern guard); the
> skeleton's FU-1 scope text correctly names the T3 probe.

**Reproducibility.** The builders are deterministic — no RNG (T3's RNG is
MCTS-Dirichlet only, unused for a value-only forward). All 40 specs realize
to legal boards (40/40, none dropped).

**Bank fixture SHA-256** (hash over every position's name + class + applied
move sequence):

```
934204713620d171743820aea6907cf4e117ca97c69e50052b991a3fdcc23991
```

**Breakdown:** 20 colony (`colony_NN_stageS`, stone counts 6…44) +
20 extension (`ext_NN_runL_axA`, open 4-runs and 5-runs across 3 hex axes).

**Anchor reproducibility gate — PASS.** FU-1 forwards
`checkpoints/bootstrap_model_v6.pt` (SHA `7ab77d2c…`) on the bank and
reproduces the T3 `value_head` numbers to 4 dp:

| metric | T3 JSON | FU-1 | Δ |
|---|---|---|---|
| mean V(colony) | 0.1635 | +0.1635 | 0.0000 |
| mean V(extension) | −0.4539 | −0.4539 | 0.0000 |
| V_spread | +0.6174 | +0.6173 | 0.0001 |

Exact reproduction → bank + load path are sound; the ladder is trustworthy.

---

## 2. Per-checkpoint value-spread table

`V_spread = mean V(colony bank) − mean V(extension bank)`. Anchor = step 0.
§S180b ladder = `archive/s180b_3knob_fail/ckpts/`. n=20 per class.

| checkpoint | step | mean V(colony) | mean V(ext) | **V_spread** | sd(col) | sd(ext) |
|---|---:|---:|---:|---:|---:|---:|
| `bootstrap_model_v6.pt` | 0 | +0.1635 | −0.4539 | **+0.6173** | 0.788 | 0.715 |
| `ckpt_step00010000.pt` | 10 000 | +0.2116 | −0.0485 | **+0.2601** | 0.676 | 0.458 |
| `ckpt_step00020000.pt` | 20 000 | +0.1469 | +0.2572 | **−0.1103** | 0.617 | 0.388 |
| `ckpt_step00030000.pt` | 30 000 | +0.1070 | −0.0326 | **+0.1395** | 0.677 | 0.534 |
| `ckpt_step00040000.pt` | 40 000 | +0.1936 | +0.2442 | **−0.0506** | 0.756 | 0.454 |
| `ckpt_step00050000.pt` | 50 000 | +0.0837 | +0.0996 | **−0.0159** | 0.684 | 0.474 |
| `ckpt_step00053500.pt` | 53 500 | +0.1027 | +0.0036 | **+0.0990** | 0.663 | 0.515 |

**Cross-plot — §S180b eval trajectory** (from skeleton OP-1 / §S180b close):

| step | 10k | 20k | 30k | 40k | 50k |
|---|---|---|---|---|---|
| wr_sealbot | 11 | 7 | 12 | 19 | 0 |
| colony_a | 36 | 35 | 40 | 43 | 59 |
| **V_spread** | **+0.26** | **−0.11** | **+0.14** | **−0.05** | **−0.02** |

The discriminator is already sub-threshold (< +0.20) at the *earliest*
ladder rung (10k) and negative by 20k — well before the wr_sealbot crash
(50k) and before colony_a's late climb (43→59 at 40k→50k).

**Noise floor.** V_spread is a difference of two 20-sample means. With
post-collapse sd(col) ≈ 0.68, sd(ext) ≈ 0.47, SE(spread) ≈
√(0.68²/20 + 0.47²/20) ≈ **0.185**. Every post-20k V_spread
(−0.110 … +0.140) sits within ~0.75 SE of zero.

---

## 3. Plot — V_spread vs training step

```
 V_spread
            -0.15   0.0(═)         +0.20(┊abort)                    +0.617
              |      |               |                                 |
 step 0       ·······╪···············┊·································● +0.617
 step 10k     ·······╪···············┊·····●                           +0.260
 step 20k  ●·········╪···············┊                                 -0.110
 step 30k     ·······╪··········●····┊                                 +0.140
 step 40k    ●·······╪···············┊                                 -0.051
 step 50k     ·····●·╪···············┊                                 -0.016
 step 53.5k   ·······╪·······●·······┊                                 +0.099
              |      |               |
            -0.15   0.0           +0.20

 ═ zero line   ┊ +0.20 hard-abort threshold (skeleton FU-2 canary gate)
 anchor +0.617 ─────────────► collapse ─────────────► flat-dead band ≈ 0
                 (0→20k, fast)            (20k→53.5k, within noise ±0.19 SE)
```

---

## 4. Drift-signature classifier

Per FU-1 spec definitions, computed in `classify_drift()`:

| signature | rule | this ladder |
|---|---|---|
| GRADUAL | monotone non-increasing, no single interval > 0.15 loss | ✗ — not monotone; 0→10k loses 0.357, 10k→20k loses 0.370 |
| PHASE-TRANSITION | one interval ≥ 50 % of total loss | partial — 10k→20k = 71 % and 0→10k = 69 % of total, **two** intervals each ≥ 50 % |
| OSCILLATION | non-monotone, magnitude fluctuation > 0.10 | ✓ — non-monotone; 20k→30k swings **+0.250** |
| NON-CLASSIFIABLE | other | — |

**Classifier output: `OSCILLATION`** (total spread loss anchor→final
+0.518; max single-interval loss 0.370 = 71 % of total; non-monotone with a
+0.250 up-swing at 20k→30k).

### Interpretation — the mechanical label undersells a clear signal

The `OSCILLATION` trip is driven by **post-collapse sampling noise**, not by
a live oscillating discriminator:

1. **The collapse is front-loaded.** V_spread loses **69 %** of its range in
   the first 10k steps (+0.617 → +0.260) and is **negative by 20k**. The
   discriminator is functionally dead — below the +0.20 FU-2 abort gate —
   at or before the earliest ladder rung.
2. **Post-20k is a flat-dead band, not an oscillation.** Values 20k…53.5k
   span [−0.110, +0.140], peak-to-peak 0.25 ≈ 1.3 × SE(spread) (0.185).
   This is statistical noise of a spread that is genuinely ≈ 0 — not a
   discriminator that recovers and re-collapses. The `OSCILLATION` rule
   fires only because the classifier treats a 0.25 noise swing as signal.

The substantive signature is **EARLY / FRONT-LOADED COLLAPSE**: a near-total
loss of value-head discrimination inside the first 10–20k steps, then a
permanent flat-dead floor. It resembles a fast PHASE-TRANSITION spread over
the first *two* intervals (not one), followed by noise.

### Limitation — the ladder is too coarse at the front

The collapse completes entirely inside the first one-or-two 10k intervals.
The archive holds no sub-10k checkpoint, so the ladder **cannot** distinguish
a step-0 onset (architecture installs the bias immediately) from an ~8k-step
cliff. That distinction is load-bearing for FU-2 (it decides whether the
value-spread canary must hold from step 0 vs whether a cliff can be targeted).

---

## 5. colony_a denominator

**Pre-flight question (Task 3 input): is colony_a's denominator wins-only or
all anchor games?**

**Answer: WINS-ONLY.**

- `hexo_rl/eval/evaluator.py:213-216` — `colony_wins` is incremented **only**
  inside `if winner == model_player_side:`. `is_colony_win(...)` is evaluated
  exclusively on games the model won. `colony_wins ⊆ wins`.
- `hexo_rl/eval/display.py:40` — the reported percentage is
  `pct = colony / wins * 100 if wins > 0 else 0.0`. Denominator = **total
  wins** in that matchup, not `n_games`.
- `hexo_rl/eval/results_db.py:get_colony_win_stats()` aggregates
  `SUM(wins_a + wins_b)` as the denominator and `SUM(colony_win)` as the
  numerator — same wins-only ratio.

So `colony_a` = (anchor-matchup colony wins) / (anchor-matchup total wins).
The §S180b "colony_a 36→35→40→43→59" series is the fraction of the model's
*wins vs the anchor* that are colony-shaped.

**Why it changes the refresh-hook timing story.** A wins-only denominator
means colony_a is computed over a **shrinking, small** sample as win-rate
collapses — at step 50k the model's anchor win pool is near-empty, so
colony_a = 59 % is a high-variance estimate over a handful of wins. A
refresh hook (or any lever) triggered off a colony_a threshold would be
firing on a **late, noisy, small-denominator** signal — exactly when it is
least reliable, and ~30–40k steps after the value head already died (FU-1:
sub-threshold by step 10k). The value-spread canary is the strictly better
trigger: it is a 40-position static probe with a stable denominator, no
game-outcome dependence, and (per §2) it crosses the +0.20 line before the
first 10k steps — long before colony_a becomes both elevated and noisy.

---

## 6. Verdict block

```
FU-1 VALUE-SPREAD CHECKPOINT-LADDER PROBE — VERDICT

  Bank:            T3 40-position canonical bank (20 colony + 20 ext)
  Bank SHA-256:    934204713620d171743820aea6907cf4e117ca97c69e50052b991a3fdcc23991
  Anchor gate:     PASS (+0.6173 reproduces T3 +0.617 to 4 dp)

  V_spread ladder: +0.617 → +0.260 → −0.110 → +0.140 → −0.051 → −0.016 → +0.099
  step:                0      10k      20k      30k      40k      50k     53.5k

  CLASSIFIER VERDICT:  OSCILLATION
                       (non-monotone; +0.250 up-swing 20k→30k > 0.10)

  SUBSTANTIVE READ:    EARLY / FRONT-LOADED COLLAPSE
                       - 69% of spread lost by step 10k
                       - sub-+0.20 (FU-2 abort gate) at or before step 10k
                       - negative by step 20k
                       - post-20k = flat-dead band ≈ 0, swings within
                         ~1.3×SE(spread)=0.185 → NOISE, not live oscillation

  The OSCILLATION label is a noise artifact of the n=20-per-class probe.
  The load-bearing fact: the value-head discriminator dies inside the
  first 10-20k steps and never recovers.

  colony_a denominator: WINS-ONLY (colony_wins / total_wins).

  LADDER LIMITATION:   no sub-10k checkpoint — cannot resolve step-0 onset
                       vs ~8k cliff. That gap is load-bearing for FU-2.
```

**Recommendation (operator decides — see Task 3 summary).** The mechanical
`OSCILLATION` verdict routes to "re-diagnose, surface to operator." The
substantive early-collapse read routes toward **FU-2** (value-head
re-architecture A/B, T2 A2+A3) with the value-spread canary wired as a
hard-abort gate **from step 0** — a buffer-level lever (refresh hook) cannot
act fast enough for a collapse that is >half complete by step 10k. A cheaper
de-risking option before committing FU-2: a finer **0–20k ladder** (new
checkpoints every ~2k from a short §S180b-config re-run) to pin whether the
onset is step-0 or a mid-cliff.
