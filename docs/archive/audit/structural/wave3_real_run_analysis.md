# §S181-AUDIT Wave 3 — Stage 5 real-run analysis (50k steps, collapse confirmed)

Stage 4 main real training run on the Wave 3 lever stack (v7full
anchor + EMA + per-class target temperature on **pretrain+bot slices
only** per L52 scope flip + bot corpus refresh hook per L51). Run
executed across 3 sessions:
- Session 1 (s181_w3main 2026-05-25): step 0 → 11 746 (SIGINT operator-mediated for cadence-fix resume)
- Session 2 (s181_w3main_resume 2026-05-25): step 11 746 → 40 000 (L50 Trigger B auto-fire @ step 40k)
- Session 3 (s181_w3main_r2 2026-05-26): step 40 000 → 50 000 (L50 Trigger C auto-fire @ step 50k)

Trajectory collapsed in Wave-2-style pattern with delayed onset.
Wave 3's refresh-hook + per-class-temp-scope-flip lever stack
plateaued the model 16-25% wr_sb across steps 10k-30k (vs Wave 2's
33%-peak-then-monotonic-decline) but ultimately failed to prevent
the collapse, with wr_sb crashing 23%→2% over steps 25k→45k.

Companion docs: `audit/structural/REAL_RUN_RECIPE.md` (v2 success
criteria), `audit/structural/wave3_smoke.md` (Stage 3 smoke
WS-A PASS-WITH-NOTES), `audit/structural/wave2_real_run_analysis.md`
(Wave 2 baseline for trajectory comparison).

## Run identity

| field | value |
|---|---|
| host | vast.ai 5080 (ssh6.vast.ai:13053) |
| workdir | /workspace/hexo_rl/ |
| branch | `phase4.5/s181_wave3_design` (commits f840a28..4435f4d) |
| variant | `v7_wave3_main` (with mid-run L50 widening @ commit 4435f4d) |
| anchor | bootstrap_model_v7full.pt SHA `568d8a33…d61e8e98` |
| encoding | v7full (board=19, planes=8, single-window) |
| iterations target | 100 000; **actual 50 000** (auto-aborted via L50 Trigger C) |
| sessions | 3 (interrupt-resume sequence for cadence-fix + threshold-widen) |
| run_ids | s1: 66f2397e… / s2: fe43b9d3… / s3: 1acd80df… |
| total wall | ~32 h (s1: 7.4 h / s2: 18.4 h / s3: 6.0 h) |
| total cost | ~$7-8 vast 5080 |
| total games | ~25 000 (s1: 5 873 / s2: 14 127 / s3: 5 000) |
| ckpts saved | 25 across 3 sessions (every 2000 steps) |
| canonical local archive | `reports/track_b_main_wave3/` (353 MB: ckpts 30k/40k/50k + EMA sidecars + logs + events JSONL) |

Wave 3 lever stack (delta vs Wave 2 main):
1. **Bot corpus refresh hook** (L51 / Track D C4): activated `enabled: true`,
   interval_steps initially 5000 (s1) then widened to 10000 (s2+s3),
   EMA snapshot opponent, async subprocess + atomic NPZ swap + hot-reload.
2. **Per-class target temperature SCOPE FLIP** (L52): `apply_to_pretrain: true`,
   `apply_to_selfplay: false`. Wave 2 selfplay-only scope was reverted.
3. **Sliding-window SealBot WR HARD-ABORT gate** (L50): triggers A/B/C wired
   via `check_sealbot_wr_hard_abort`. Trigger B widened from 0.5→0.25 mid-run
   (s3) after firing @ step 40k as designed.

## SealBot WR trajectory (the headline)

| step | wr_sealbot | CI95 | wr_anchor | wr_best | colony_sb | colony_anchor | promoted | elo |
|---:|---:|---|---:|---:|---:|---:|:---:|---:|
| 5 000 | 16.0% | [10.1, 24.4] | 60.0% | 52.0% | 2 | 48 | ✗ | 435 |
| **10 000** | **25.0%** | [17.5, 34.2] | 62.0% | 58.0% | 3 | 40 | ✗ | 417 |
| 15 000 | 18.0% | [11.5, 26.7] | 46.0% | 59.0% | 7 | 20 | ✗ | 428 |
| 20 000 | 16.0% | [10.1, 24.4] | 68.0% | 55.0% | 3 | 45 | ✗ | 407 |
| 25 000 | 23.0% | [15.9, 31.9] | 59.0% | 66.0% | 8 | 39 | **✓** | 377 |
| 30 000 | 20.0% | [13.3, 28.5] | 65.0% | 65.0% | 11 | 48 | **✓** | 313 |
| 35 000 | 10.0% | [5.5, 17.6] | 61.0% | 57.0% | 6 | 42 | ✗ | 212 |
| **45 000** | **2.0%** | [0.5, 7.7] | **70.0%** | **69.0%** | 2 | 60 | **✓** | 397 |

**Peak: step 10k 25% wr_sb.** (Wave 2 peak: step 20k 33% wr_sb.) Wave 3
peaked at lower magnitude AND earlier in training than Wave 2.

**Plateau-then-collapse pattern:**
- Steps 10k-30k: wr_sb oscillating 16-25% (5 evals in this band)
- Steps 30k→35k: first sharp drop (20→10%)
- Steps 35k→45k: catastrophic collapse (10→2%)

**Wave 2 vs Wave 3 trajectory shape:**

| step | Wave 2 wr_sb | Wave 3 wr_sb |
|---:|---:|---:|
| 10k | 24% | 25% |
| 20k | 33% (peak) | 16% |
| 30k | 11% | 20% |
| 40k | 5% (HARD-ABORT) | (10% @ 35k; collapsed 45k) |
| 45k | — | 2% (final) |

Wave 3 delayed the steep decline by ~10k steps but ended at a LOWER
final wr_sb than Wave 2 (2% vs 5%).

## L34 anchor↑/sealbot↓ divergence (full eval-by-eval)

| transition | anchor Δ | sealbot Δ | L34 fires? |
|---|---:|---:|:---:|
| 5k → 10k | +2 pp | +9 pp | INVERSE (healthy climb) |
| 10k → 15k | -16 pp | -7 pp | both down (model jitter) |
| 15k → 20k | +22 pp | -2 pp | **YES (1st)** |
| 20k → 25k | -9 pp | +7 pp | INVERSE (recovery) |
| 25k → 30k | +6 pp | -3 pp | **YES (2nd)** |
| 30k → 35k | -4 pp | -10 pp | both down (sharp sb drop) |
| 35k → 45k | +9 pp | -8 pp | **YES (3rd)** |

3 L34 fires across the trajectory. Step 45k shows the FINAL form:
- anchor 70% (peak for run)
- sealbot 2% (collapse floor)
- 68pp divergence (Wave 2 peak L34 was 65pp at step 40k = anchor 70% / sb 5%)

**Wave 3 hits the SAME end-state colony-attractor as Wave 2.** The lever
stack delayed the trajectory but the equilibrium point is identical
(model exploits anchor's colony bias, fails general tactics).

## Dual-bank V_spread canary trajectory (49 fires, full)

```
session 1 (s181_w3main):
  step= 2000 T3=+0.242 alt=+0.324 PASS (only canary above gate)
  step= 4000 T3=+0.177 alt=+0.299 fail
  step= 6000 T3=+0.118 alt=+0.308 fail
  step= 8000 T3=+0.089 alt=+0.333 fail
  step=10000 T3=+0.066 alt=+0.352 fail
  step=11746 T3=+0.032 alt=+0.378 fail   (session 1 stop)
session 2 (s181_w3main_resume):
  step=12000 T3=+0.069 alt=+0.458 fail   (alt jumped up post-refresh)
  step=14000 T3=+0.064 alt=+0.466 fail
  ...
  step=24000 T3=+0.022 alt=+0.513 fail   (alt PEAK)
  step=26000 T3=+0.012 alt=+0.467 fail
  step=28000 T3=+0.008 alt=+0.433 fail
  step=30000 T3=+0.036 alt=+0.412 fail
  ...
  step=40000 T3=+0.051 alt=+0.400 fail   (session 2 stop, L50-B fired)
session 3 (s181_w3main_r2):
  step=42000 T3=+0.185 alt=+0.338 fail   (alt declining)
  step=44000 T3=+0.184 alt=+0.290 fail
  step=46000 T3=+0.194 alt=+0.284 fail
  step=48000 T3=+0.195 alt=+0.274 fail
  step=50000 T3=+0.140 alt=+0.265 fail   (session 3 stop, L50-C fired)
```

**alt trajectory**: climbed early-session from +0.32 → +0.513 peak @ step
24k, then monotonically declined to +0.265 by step 50k. **Never fell below
+0.07 sustained gate** — the alt canary remained "PASS" by the L48 gate
criterion throughout, EXACTLY reproducing Wave 2's L50 lesson: alt-bank
V_spread is necessary but not sufficient for sustained eval quality.

**T3 trajectory**: oscillated +0.008 to +0.242, never went deeply negative
like Wave 2 (which collapsed to -0.26). T3 stayed broadly positive
throughout — also consistent with L48 (T3 is bank-specific and decouples
from real selfplay quality).

## Refresh hook validation (4 full cycles)

| cycle | requested step | swap step | n_positions_after | session |
|---:|---:|---:|---:|---|
| 1 | 5 000 | 10 000 | 6 724 | s1 |
| 2 | 15 000 | 20 000 | 7 453 | s2 |
| 3 | 30 000 | 35 000 | 8 006 | s2 |
| 4 | 45 000 | 50 000 | 9 932 | s3 |

**End-to-end mechanism validated.** Each cycle: subprocess fires at eval
boundary → poll at next eval boundary → atomic NPZ swap (write-rename) →
hot-reload Rust ReplayBuffer → bot_buffer replaced with fresh 6.7k-10k
SealBot-vs-EMA position dataset. Reload_sec ≤ 6 sec each (within design
budget). Bot corpus position count grew across cycles (6724 → 9932 — the
subprocess captured more positions as the EMA model became stronger and
sustained longer games).

**Refresh hook is NOT defective.** The mechanism works exactly as
designed. The Wave 3 verdict is that the mechanism is INSUFFICIENT to
prevent the colony-attractor capture.

## L50 hard-abort trigger fires (mechanism validated end-to-end)

**Fire #1 — Trigger B (peak-collapse) @ step 40 000 (session 2):**
```
HARD-ABORT (L50/Wave3-B): SealBot WR 10.0% < peak 23.0% × 50% past step 25,000 — Wave-2-style collapse
wr_history: [[15000, 0.18], [20000, 0.16], [25000, 0.23], [30000, 0.2], [35000, 0.1]]
```
Session 2 terminated cleanly via `self.shutdown.running = False`.
Operator widened `wr_collapse_from_peak_ratio: 0.5 → 0.25` and resumed
session 3 from ckpt_00040000.

**Fire #2 — Trigger C (early-death) @ step 50 000 (session 3):**
```
HARD-ABORT (L50/Wave3-C): SealBot WR 2.0% < 5% past step 15,000 — §S180b-style early death
wr_history: [[45000, 0.02]]
```
Session 3 terminated cleanly. ckpt_00050000.pt + buffer.bin saved.

**Both triggers fired AS DESIGNED.** Stage 2B's L50 gate is validated:
catches Wave-2-style collapse (Trigger B) AND catastrophic floor breach
(Trigger C). The gate's sliding-window logic + per-step poll-at-eval-
boundary mechanism worked correctly across both fires.

## Mechanism diagnosis

Wave 3 lever stack produced a **plateau-then-collapse** trajectory
distinct from Wave 2's **peak-then-monotonic-decline**:
- Wave 2: 24% (10k) → 33% (20k peak) → 11% (30k) → 5% (40k)
- Wave 3: 25% (10k peak) → plateau 16-25% (10k-30k, 5 evals) → 10% (35k) → 2% (45k)

Same end-state attractor, different approach trajectory.

### M1 — Refresh hook insufficient (high confidence)

The refresh hook DID maintain anti-colony pressure across 4 cycles
(replacing 21,899 static → 9,932 fresh positions on the 4th refresh).
SealBot vs EMA generates representative games for the current model's
distribution. BUT the per-batch share is only 30% (76 positions out of
256), and the model's selfplay distribution (70% of batch) overpowers
the fresh corpus signal.

**Evidence**: refresh cycles complete at steps 10k/20k/35k/50k, but
wr_sb declined monotonically AFTER step 25k DESPITE the step 35k refresh
firing. The fresh corpus targets weren't enough to redirect the selfplay-
dominated gradient signal.

### M2 — Per-class temp SCOPE FLIP insufficient (high confidence)

The L52 fix (apply_to_pretrain: true / apply_to_selfplay: false) was
supposed to preserve selfplay tactical CE sharpness. Wave 3 ran with
this scope for the entire 50k steps.

**Evidence**: Wave 3 STILL collapsed to 2% wr_sb. The "preserving
selfplay tactical CE" hypothesis is necessary-not-sufficient. The
attractor pulls the policy toward colony shapes via a mechanism that
operates ABOVE the per-class CE softening level.

### M3 — best_model rotation amplifies the attractor (moderate confidence)

Wave 3 had 3 best_model promotions (steps 25k, 30k, 45k). At step 45k:
- wr_best = 69% — model strongly beats prior best_model
- wr_anchor = 70% — model strongly beats original anchor
- wr_sb = 2% — model catastrophically loses to SealBot

The PROMOTED snapshot at step 45k IS the colony-attractor at its peak.
The eval pipeline promotes models that are "better" by internal metrics
(wr_best, wr_anchor), but those metrics DON'T penalize colony exploitation
of the anchor's blind spots — leading to a positive feedback loop where
the best_model bar rises in the attractor direction.

**Mechanism: M1 + M2 + M3 compound.** No single lever in the Wave 3 stack
addresses the structural attractor dynamics. Refresh hook keeps the bot
slot fresh (M1 mitigated) but selfplay dominates the gradient. Per-class
temp scope flip preserves selfplay sharpness (M2 mitigated) but the
sharp policies move toward colony shapes via MCTS+value-head feedback.
best_model rotation reinforces the wrong direction (M3 active).

## What Wave 3 PROVED (positives — keep for Wave 4+)

1. **Refresh hook end-to-end mechanism works.** 4 cycles complete clean
   under realistic GPU contention conditions. Atomic swap + hot-reload
   pattern is production-ready.
2. **Per-class temp SCOPE FLIP runs cleanly.** apply_to_selfplay=false
   path validated; no crashes or silent corruption. Code surface is
   reusable for future scope experiments.
3. **L50 sliding-window WR hard-abort gate validated.** Both Trigger B
   AND Trigger C fired as designed at appropriate threshold crossings.
   Operator-widening workflow validated (config change + resume from
   ckpt + buffer-preserved). Wave 4+ should keep this gate.
4. **EMA infrastructure stable.** Maintained anti-colony pressure across
   32h wall + 3 session interrupts + 4 refresh cycles. No EMA-specific
   issues (M3 in wave2 analysis was already ruled out; Wave 3 confirms).

## What Wave 3 FAILED to deliver (negatives — Wave 4+ must address)

1. **PRIMARY W3-G1 FAIL**: rolling-mean SealBot WR ≥ 20% sustained 30k-50k.
   Rolling mean over [30k=20, 35k=10, 45k=2] = 10.7% (well below 20% gate).
2. **PRIMARY W3-G2 FAIL**: L34 divergence — 3 fires across trajectory.
3. **PRIMARY W3-G3 FAIL**: HARD-ABORT triggered (Trigger B @ 40k AND C @ 50k).
4. **SECONDARY W3-G4 PARTIAL**: peak SealBot WR = 25% > §150 baseline 17.4% ✓
   but below Wave 2's 33% peak. Lever stack PEAK is lower than the
   baseline Wave 2 collapsed-from peak.
5. **SECONDARY W3-G5 PASS**: 4 refresh fires (of ~10 expected over 50k at
   interval=10000 + s1's interval=5000 cadence).
6. **SECONDARY W3-G6 PASS**: alt V_spread held above +0.07 throughout.
7. **SECONDARY W3-G8 OVERSHOOT**: colony_wins_sealbot ranged 2-11 (vs Wave 2's
   3-6 baseline). Higher in Wave 3 mid-trajectory (step 25-30k = 8-11).

Net: 0 of 3 PRIMARY criteria PASS. Wave 3 lever stack DECISIVELY FAILED
its pre-registered success bar.

## L53 — Refresh hook + per-class temp scope flip insufficient against the colony attractor

**Rule.** A 30%-batch-share dynamic-refreshed bot corpus + per-class CE
softening scoped to static (pretrain+bot) rows is INSUFFICIENT to
prevent the colony-attractor capture in v7full sustained training.
The attractor operates above the per-class CE level and the selfplay
gradient signal (70% of batch) overpowers the 30% fresh bot signal.

**Why.** Wave 3 ran with both levers active for 50k steps. wr_sb peaked
at 25% (step 10k), oscillated 16-25% across steps 10k-30k, then
collapsed to 2% by step 45k. Same end-state attractor as Wave 2 (anchor
~70% / sealbot ~5% / 65pp divergence) reached via a different
trajectory shape (plateau-then-crash vs Wave 2's peak-then-decline).
Refresh hook mechanism worked (4 cycles end-to-end clean) but the
fresh corpus's anti-colony signal was overpowered. Per-class temp
scope flip preserved selfplay CE sharpness but the sharpness was
directed toward colony shapes.

**How to apply.** Wave 4+ must employ STRUCTURAL interventions, not
RATIO interventions. Candidates per dispatcher's deferred items:
- 2-stone opponent-reply aux head (forces trunk to discriminate
  on-policy responses, addresses M3 best_model rotation reinforcement)
- WDL value-head migration (decouples value-head training from binary
  outcome to value-distribution — addresses the value-head feedback
  channel)
- KL-weighted buffer writes (penalizes high-KL transitions to keep
  selfplay buffer distribution closer to corpus)
- Policy surprise weighting (PSW — design at s179b, parked)
- Class-weighted gradient scaling (instead of CE softening)
The refresh hook + per-class temp scope flip should REMAIN as a
defensive substrate (they delayed collapse + run clean) but a
fundamental mechanism change is needed atop them.

## L54 — Trigger B's "peak × 0.5" threshold caught Wave 3 collapse 5-10k steps later than ideal

**Rule.** L50 Trigger B (`current < peak × 0.5 past step 25k`) fires
when the trajectory has already dropped to half its peak. For a peak
of 25%, threshold is 12.5% — but a model dropping from 25% to 12.5%
is already deep into collapse. Trigger C (`current < 5% past step 15k`)
catches catastrophic floor breach but only after deep collapse.

**Why.** Wave 3 fired Trigger B at step 40k drain (wr_sb 10%, peak 23%
in 5-eval ring, threshold 11.5%). By then the trajectory had already
crashed from 23% (step 25k) to 10% (step 35k) — Trigger B fired 15k
steps into the collapse. Trigger C fired at step 50k drain after
wr_sb hit 2% — even later. Wave 3 lost ~$3 worth of compute training
through a collapse that was identifiable from step 30→35k onward.

**How to apply.** Future runs should add a sliding-MEAN derivative
trigger:
- Trigger D: rolling-mean WR drops ≥ Xpp across 2 consecutive evals
  past step 20k (e.g. X=5pp would have fired Wave 3 at step 35k
  when rolling-mean dropped 21.5% → 15% = 6.5pp)
Or tighten Trigger B's peak ratio earlier (e.g. ratio 0.7 + min_step
20k catches earlier decline).

## L55 — Wave 3 plateau-then-collapse is L34-attractor with delay, not new mechanism

**Rule.** Wave 3's plateau (steps 10k-30k stable 16-25%) is NOT a
"different attractor" or "weaker attractor pull" — it's the SAME
colony-attractor with the same end-state, just with the refresh hook +
per-class temp scope flip adding ~10-15k steps of inertia before the
collapse manifests in wr_sb metrics.

**Why.** Both Wave 2 and Wave 3 end at the same L34 signature (anchor
~70% / sealbot ≤5% / 65pp divergence). The end-state attractor is
the same. Wave 3 lever stack delays the wr_sb-measurable manifestation
but doesn't break the attractor. This means future "stack more levers"
strategies should expect SIMILAR end-state with possibly different
trajectory shapes — the structural change is what matters.

**How to apply.** Wave 4 design should NOT chase a different
end-state shape (longer plateau ≠ better outcome if it ends at 2%
collapse). Design for a different ATTRACTOR — i.e. a different value-
head / policy-head / value-target structure that doesn't have a
colony-exploit-of-anchor-bias attractor at all. WDL value-head, 2-stone
aux head, or PSW are candidates because they CHANGE WHAT THE MODEL
OPTIMIZES, not just the gradient-share of the existing optimization.

## Falsified Hypotheses Register additions

- **Refresh hook + per-class temp scope flip is sufficient to prevent
  colony-attractor capture in v7full sustained training.** FALSIFIED
  by Wave 3 — wr_sb collapsed to 2% by step 45k despite both levers
  active across the full trajectory. Mechanism works but is overpowered
  by the selfplay-dominated gradient channel.

- **Plateau (long sustained mid-WR phase) is a positive sign of attractor
  break.** FALSIFIED by Wave 3 — model plateaued 16-25% wr_sb across
  steps 10k-30k (5 consecutive evals) before catastrophically
  collapsing to 2%. Plateau is consistent with SAME attractor + lever-
  stack-induced inertia; NOT evidence of attractor escape.

- **best_model promotion is a reliable signal of model improvement
  toward Phase 4.5 readiness.** FALSIFIED by Wave 3 — best_model was
  promoted at step 25k AND step 45k. Step 45k promotion happened
  WHILE wr_sb crashed to 2%. The promotion criteria (wr_best ≥ 0.55 +
  bootstrap_floor 0.45) reinforces the colony attractor by promoting
  models that beat internal baselines via colony exploitation.

## Wave 3 canonical deliverable preservation

Wave 3 has NO project-record "peak" deliverable analogous to Wave 2's
33% snapshot. The closest candidates:

- **`checkpoints/checkpoint_00010000.pt`** (Wave 3 25% peak; archived
  pre-stop as session 1 snapshot). Highest wr_sb but only 100 games
  (Wilson CI overlaps Wave 2 baseline).
- **`checkpoints/checkpoint_00025000.pt`** (Wave 3 23%, PROMOTED).
  Second-highest wr_sb; first promotion; reasonable mid-trajectory.
- **`checkpoints/checkpoint_00045000.pt`** (the colony attractor end-state;
  2% wr_sb / 70% wr_anc; PROMOTED). Useful as the COLONY-CAPTURE
  reference for Wave 4 mechanism analysis.

Archived locally at `reports/track_b_main_wave3/checkpoints/` (steps
20k/30k/40k/50k + EMA sidecars). Operator may copy ckpt_00045000.pt to
`reports/canonical_models/wave3_step45k_collapse.pt` as a colony-
capture reference for future ablation work.

Wave 2's `wave2_step20k_peak33pct.pt` remains the project-record peak
SealBot WR snapshot.

## Stage 5 routing decision (per dispatcher §5C)

Per the dispatcher routing table:

> | All PRIMARY FAIL (run collapses early) | Mechanism wrong. Re-aggregate evidence. Possible Wave 4 with WDL or other levers from research doc. |

Wave 3 hit this branch. PRIMARY criteria all failed; the lever stack
is decisively insufficient.

**Phase 4.5 remains BLOCKED.** Wave 4 mechanism design required before
Phase 4.5 features (Gumbel CQV, KrakenBot wrapper, etc.) ship.

**Recommended Wave 4 escalation path** (operator decides priority):
1. **2-stone opponent-reply aux head** (V-B-D conditional, never tested
   in v7full sustained context). Forces trunk to discriminate
   on-policy reply patterns, may break the colony-exploits-anchor-bias
   attractor.
2. **WDL value-head migration** (parked since §S178 — A2 falsified
   arch-only fix, BUT Wave 3 evidence is that LOOP-side levers are
   ALSO insufficient → arch + loop combined hypothesis worth testing).
3. **PSW (Policy Surprise Weighting)** — design `s179b` parked. Penalizes
   high-KL transitions in selfplay buffer writes.
4. **Class-weighted gradient scaling** (vs CE softening) — different
   mechanism class than the per-class temp lever Wave 3 already tested.

L53/L54/L55 banking + this analysis doc is the Wave 4 design starting
point.

## Cost summary

| stage | spent | running total |
|---|---|---|
| Stage 3 smoke (s181_w3smoke) | ~$1.50 | $1.50 |
| Stage 4 main session 1 (cadence-fix stop) | ~$1.50 | $3.00 |
| Stage 4 main session 2 (L50-B fire) | ~$4.00 | $7.00 |
| Stage 4 main session 3 (L50-C fire) | ~$1.30 | **~$8.30** |

Wave 3 total ~$8.30 (within hard cap $8 + small overrun from operator
"continue to 50k regardless" override).

## Cross-references

- `audit/structural/REAL_RUN_RECIPE.md` — Wave 3 success criteria + plan
- `audit/structural/wave2_real_run_analysis.md` — Wave 2 baseline (L50/L51/L52)
- `audit/structural/wave3_smoke.md` — Stage 3 smoke WS-A PASS-WITH-NOTES
- `audit/structural/wave3_launch_readiness.md` — Stage 2 close + Stage 3 prep
- `docs/designs/s179c_bot_refresh_hook.md` — refresh hook design (validated)
- `docs/designs/s179b_policy_surprise_weighting.md` — PSW design (parked, Wave 4 candidate)
- `configs/variants/v7_wave3_main.yaml` — main variant (initial + widening commit 4435f4d)
- `hexo_rl/training/per_class_target_temperature.py` — scope flip implementation
- `hexo_rl/monitoring/alert_rules.py:check_sealbot_wr_hard_abort` — L50 gate (validated)
- `hexo_rl/training/step_coordinator.py:709-754` — refresh hook + WR hard-abort wiring (validated)
- `reports/track_b_main_wave3/` — local archive (ckpts + logs + events JSONL, 353 MB)
- `reports/canonical_models/wave2_step20k_peak33pct.pt` — Wave 2 project-record peak (for reference)
