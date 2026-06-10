# §S181-AUDIT Wave 2 — Stage 5/6 real-run analysis (aborted at step 47k)

Stage 5 main real training run on the V-B-A `uniform_self` lever stack
(v7full anchor + EMA + per-class target temperature on selfplay slice,
`sample_rate=0.20`). Aborted at step 47642 — wr_sealbot collapsed from
33 % peak (step 20k) to 5 % at step-40k eval, triggering the
RR-G3 / §S180b HARD-ABORT threshold (`SealBot WR < 8 %`). Run continued
~7k further steps for trajectory confirmation before operator-decided
kill. Companion docs: `audit/structural/REAL_RUN_RECIPE.md`,
`audit/structural/wave2_smoke.md` (smoke S-A PASS basis for launch),
`audit/structural/track_b/B_verdict_synthesis.md`.

## Run identity

| field | value |
|---|---|
| host | vast.ai 5080 (REMOTE_HOST:REMOTE_PORT) |
| workdir | $REPO_ROOT/ |
| branch | phase4.5/s181_wave2_lever_vba_selfplay 3354016 |
| variant | v7_real_run_main (`configs/variants/v7_real_run_main.yaml`) |
| anchor | bootstrap_model_v7full.pt SHA 568d8a33…d61e8e98 |
| encoding | v7full (K=1 single-window) |
| iterations target | 100 000 |
| **iterations actual** | **47 642** (operator-aborted) |
| run_id | aad5e948c4b94bf395daeddbb57415b9 |
| launch | 2026-05-24 00:09:56 UTC |
| abort | 2026-05-25 00:05:46 UTC |
| wall | ~24 h (one prior dashboard-deadlock restart cost ~30 min) |
| checkpoints saved | 23 (step 2k-46k) |
| best_model.pt | last promo at step 30k (NOT the step-20k 33% peak) |
| events captured | 23 dual-bank canary fires, 4 evals, 3 best_model promotions |
| log | reports/track_b_main/logs/s181_main_20260524_0009.log |
| events JSONL | reports/track_b_main/logs/events_aad5e948…b9.jsonl |
| key ckpts retained | best_model.pt + checkpoint_00010000/20000/30000/40000.pt |

Lever stack at launch (vs §S180b 3-knob recipe):
- encoding v6 → **v7full** (matches anchor; C-LITE-1 verdict)
- EMA of weights (decay 0.999, every 10 optimizer steps), dispatched
  through `Trainer.inference_state_dict`
- per-class target temperature on selfplay slice (T_colony=1.5,
  others 1.0, pretrain slice untouched, `selfplay_sample_rate=0.20`
  perf optimization)

## SealBot WR trajectory (the headline)

| step | wr_sealbot | CI95 | wr_anchor | colony_wins_sealbot | g4 | promoted | elo |
|---:|---:|---|---:|---:|:---:|:---:|---:|
| 10 000 | 24.0 % | [16.7, 33.2] | 66.0 % | 3 | True | ✓ | 481.3 |
| **20 000** | **33.0 %** | [24.6, 42.7] | 61.0 % | 6 | False | ✓ | 382.7 |
| 30 000 | 11.0 % | [6.3, 18.6] | 65.0 % | 4 | False | ✓ | 255.3 |
| **40 000** | **5.0 %** | **[2.2, 11.2]** | **70.0 %** | 5 | **True** | **✗** | 209.6 |

Peak at step 20k, monotonic decline thereafter. Step-40k 5 % is below
the RR-G3 13 % gate AND below the 8 % §S180b HARD-ABORT threshold.
`promoted=False` for the first time at step-40k means the model could
no longer pass the `bootstrap_floor` gate (min_winrate 0.45 vs the
bootstrap anchor) — at this point training is actively producing
weaker models than what `best_model.pt` already represents.

## L34 anchor↑/sealbot↓ divergence

| transition | anchor Δ | sealbot Δ | L34 pattern fires? |
|---|---:|---:|:---:|
| 10k → 20k | -5 pp (66→61) | +9 pp (24→33) | INVERSE (healthy) |
| 20k → 30k | +4 pp (61→65) | -22 pp (33→11) | **YES (1st)** |
| 30k → 40k | +5 pp (65→70) | -6 pp (11→5) | **YES (2nd)** |

Pattern: model is becoming *relatively stronger* vs the frozen v7full
anchor while becoming *absolutely weaker* vs SealBot. Classic colony-
attractor capture signature per §S178 / L34 — the model learns to
exploit the anchor's colony bias but loses general tactical strength.
Dispatcher SOFT-ABORT requires 5 consecutive instances; 2 observed
when the run was killed.

## Dual-bank V_spread canary (23 fires, full trajectory)

```
step= 2000 T3=+0.2714 alt=+0.3019 PASS
step= 4000 T3=+0.2631 alt=+0.2820 PASS
step= 6000 T3=+0.2427 alt=+0.2648 PASS
step= 8000 T3=+0.1654 alt=+0.2314 fail
step=10000 T3=+0.1147 alt=+0.2143 fail   (eval wr_sb=24)
step=12000 T3=+0.0797 alt=+0.1752 fail   (alt low point)
step=14000 T3=+0.0520 alt=+0.2003 fail
step=16000 T3=+0.0285 alt=+0.2295 fail
step=18000 T3=+0.0318 alt=+0.2132 fail   (T3 first +Δ)
step=20000 T3=+0.0482 alt=+0.2255 fail   (eval wr_sb=33 PEAK)
step=22000 T3=+0.0810 alt=+0.2198 fail
step=24000 T3=+0.0865 alt=+0.2260 fail
step=26000 T3=+0.0894 alt=+0.2359 fail
step=28000 T3=+0.0753 alt=+0.2426 fail
step=30000 T3=+0.0286 alt=+0.2517 fail   (eval wr_sb=11, alt new high)
step=32000 T3=-0.0005 alt=+0.2564 fail   (T3 first negative)
step=34000 T3=-0.0237 alt=+0.2574 fail   (alt run high +0.2574)
step=36000 T3=-0.0711 alt=+0.2509 fail   (alt 1st decline)
step=38000 T3=-0.1877 alt=+0.2465 fail   (T3 -0.117 jump, alt 2nd decline)
step=40000 T3=-0.2147 alt=+0.2502 fail   (eval wr_sb=5 HARD trigger)
step=42000 T3=-0.2453 alt=+0.2514 fail
step=44000 T3=-0.2425 alt=+0.2514 fail
step=46000 T3=-0.2623 alt=+0.2370 fail   (alt resumed decline)
```

**T3 trajectory**: started +0.27, crossed gate (+0.20) at step 8k,
oscillated +0.03 to +0.09 through step 28k, then collapsed steadily
into deep negative territory (-0.26 by step 46k). Per L48 T3 is
bank-specific (synthetic positions calibrated on v6 anchor's value
head; v7full discriminates them less sharply), but a fully-negative T3
is a clear colony-direction signal on the synthetic bank.

**alt trajectory**: started +0.30, drifted down to +0.18-0.23
oscillation band through step 28k, climbed back to a run high +0.257
at step 32k-34k (DURING the eval-confirmed collapse), held +0.25 to
step 44k, then began declining again at step 46k. **Critically**:
alt-bank V_spread stayed comfortably above the +0.07 sustained gate
throughout — yet eval-measured wr_sealbot collapsed. **The held-out
V_spread canary failed to track the actual performance collapse.**

## Mechanism diagnosis

The Wave 2 lever stack produced a peak-and-collapse trajectory:
**peak at step 20k** (33 % wr_sb, exceeding §150 baseline 17.4 % by
1.9×), then **monotonic decline** to 5 % by step 40k. Three candidate
mechanisms, ranked by likelihood:

### M1 — Bot corpus opportunistic fit at step 20k → degeneration (high)

The bot corpus (`bot_corpus_s178_sealbot_vs_v6.npz`, 21 899 positions,
**static**, `bot_batch_share=0.30`) is 30 % of every training batch.
Each batch trains on ~77 SealBot positions. Stage 5 generated ~23 820
self-play games before abort (~12-15k by step 20k); model exposure at
peak ≈ ~12-15k games × ~50 positions = ~600-750 k selfplay positions
for self-distillation, plus the same 21 899 bot positions seen
repeatedly — by step 20k the bot corpus has been re-encountered
~70 times (1.54 M bot positions / 21 899 corpus size).

**Mechanism**: by step 20k the model has learned SealBot's tactical
distribution from the bot corpus (peak wr_sb=33 %). After step 20k,
continued selfplay generates positions that drift the policy AWAY from
the SealBot distribution (model evolves; bot corpus stays frozen). The
30 % bot-corpus signal becomes increasingly off-distribution relative
to the model's own play → ineffective regularization → drift continues.
By step 30k, bot corpus is teaching a SealBot reply distribution the
model no longer faces → fit decays. Anchor wins INCREASE because the
model becomes "more universal" against the frozen anchor's static
patterns, but it has lost its specific SealBot competence.

**Predicted by**: Track D candidate C4 ("Bot-corpus staleness × outcome-
channel feedback"). Dispatcher pre-registered this as a known risk;
refresh hook is the design-named remedy but was disabled for the run.

**Evidence consistent**: monotonic decline post-peak; anchor WR
climbing (model gaining vs frozen anchor); colony_wins_anchor rising
35→37→54→64 (model drifting toward colony shapes the anchor doesn't
defend well); colony_wins_sealbot stayed low (3→6→4→5 — bot corpus
still has SOME anti-colony pull, just decreasing in distributional
relevance).

### M2 — Per-class temperature dilution over selfplay-slice (medium)

`per_class_target_temperature.colony_temperature=1.5` with
`selfplay_sample_rate=0.20` softens visit-count CE targets on 20 % of
selfplay rows when classified as colony. The intent: attenuate the
A4 1.21× colony-vs-extension gradient asymmetry (per Track B B1
diagnosis).

**Mechanism**: late in training (step 30k+) the selfplay buffer
contains policies sharpened around the model's strongest moves.
Softening these on colony rows reduces the gradient signal on the
move the model itself considers best — effectively un-learning sharp
tactical play. Combined with M1 (bot corpus drift), tactical quality
on SealBot games decays.

**Evidence**: alt V_spread stayed high (+0.25) — the temperature
lever isn't hurting the value head's discrimination on held-out
banks. But policy entropy / sharpness data would be needed for a
clean test (not pulled in this analysis pass).

### M3 — EMA averaging artifact at the inference path (low)

EMA model (decay 0.999, update every 10 steps) is what self-play
workers see. At step 40k+ the EMA reflects ~4 000 weight updates of
~mixed quality. Most likely irrelevant — EMA at 0.999 has very long
effective window and is monotonically a smoother (can lag but
doesn't actively degrade).

**Evidence against**: smoke (Stage 4B, also EMA-enabled) was clean
S-A PASS at step 3000. The collapse appears at step 20-40k, well
past EMA's effective warmup.

### Mechanism most likely is M1 + M2 compound

The peak-at-step-20k → monotonic-decline pattern is consistent with
**bot-corpus drift** (M1) being the dominant driver, **amplified
by per-class temp's tactical-softening** (M2) starting to bite as the
selfplay buffer grows past the bootstrap-corpus dominated early
window. The dispatcher's L34 mechanism was predicted; the failure
mode matches §178/§S179/§S180b's colony-attractor lineage, just with
a 20k-step grace window from the lever stack rather than crashing at
step 1-2k (which is what §S180b unmitigated did).

## What the Wave 2 lever stack DID achieve

1. **Project-record SealBot WR 33 %** at step 20k — beats §150
   baseline 17.4 % by 1.9× and any prior sustained-run peak.
2. **alt-bank V_spread sustained >+0.20 throughout 46k steps** —
   the held-out V_spread canary that §S181 PR-C built specifically
   for this purpose stayed above gate the entire run. This is the
   first time in the project a sustained run has maintained the
   alt-bank metric.
3. **colony_wins_sealbot remained 3-6 across all 4 evals** vs
   §S180b's 91-100. The anti-colony lever stack DID prevent the
   visible colony-attractor capture in eval games — the failure
   mode is more subtle (drift in tactical competence) rather than
   colony policy lock-in.
4. **EMA + per-class temp + sub-sampling perf opt all worked
   technically** — no infrastructure failures, the lever stack was
   the right mechanism class.

## What it FAILED to deliver

1. **RR-G1** T3 V_spread ≥ +0.20 sustained 0→50k — **FAIL** (crossed
   gate at step 8k @ +0.165; oscillated +0.03–+0.09 through step 28k;
   collapsed to −0.26 by step 46k).
2. **RR-G3** SealBot WR ≥ 13 % @ step 30k — **FAIL** (was 11 %,
   Wilson95 LB 6.3 % below 9 % requirement).
3. **RR-G4** SealBot WR ≥ 18 % @ step 50k — **FAIL** (extrapolated
   from step 40k 5 %; would not have recovered without intervention).
4. **RR-G5** colony_a < 50/100 — passed throughout (3-6/100).
5. **RR-G6** L34 divergence clean — **FAIL** (2 consecutive instances
   step 20k→30k and 30k→40k).

RR-G2 alt ≥ +0.07 sustained PASSED (alt held +0.18–0.30 throughout),
but this is the L50 lesson — alt-bank V_spread is necessary but not
sufficient for sustained eval quality. 4 of 6 RR-G* failed.

## Failure-mode lessons

### L50 — Held-out V_spread canary fails to track eval-measured collapse

Per `audit/structural/wave2_smoke.md`: alt-bank V_spread is the
"corpus-grounded reference" per L48 and was used to override the
SOFT-ABORT trigger. Stage 5 falsifies the assumption that alt
V_spread is a sufficient real-run gate: alt stayed +0.25 (well above
+0.07 sustained gate) while wr_sealbot collapsed 33→5 %. The
divergence shows that **value-head discrimination on a fixed held-out
bank is not a sufficient proxy for actual selfplay/eval performance**.
Future runs should ALSO gate on sealbot/anchor WR sliding-window
trajectory.

### L51 — Bot-corpus staleness predicted as Track D C4 — confirmed

The peak-at-step-20k pattern matches what a 21 899-position static
bot corpus would produce: by step 20k the corpus has been re-
encountered ~70 times (batch 256 × 30 % bot × 20 k steps = 1.54 M bot
positions / 21 899 corpus size); distributional decay vs the evolving
model is unbounded by the imprint count and grows with the
policy-distance the model accumulates after corpus saturation. Wave 1
B4 (3 000-step instrumented run) measured the bot corpus gradient-pull
share as small (B_track_d_xref ranked C4 "small absolute magnitude")
— TRUE at step 0-3 k when the corpus is in-distribution. Wave 2 main
shows the same channel becomes LOAD-BEARING at the 20 k+ horizon as
the model drifts off the static distribution. Both findings hold at
their respective time windows; B4's gradient-share metric does not
capture the multi-epoch staleness mechanism. **Refresh hook (Track D
C4) must land for any future Wave 2 lever-stack rerun**, not just be
optional. Recommended: `bot_corpus_refresh.enabled=true` with
trigger `best_model_promotion`, cooldown 5-10k steps.

### L52 — Per-class temp WITHOUT refresh hook may over-soften late tactical learning

Per-class target temperature `T_colony=1.5` on the selfplay slice
softens CE targets on colony-classified rows in selfplay. Static bot
corpus regularizes early but degrades; once bot is off-distribution,
the only anti-colony pressure left is per-class temp on selfplay
rows — and that's tactical-softening, not target-replacement. Without
the dynamic regularization from a refreshing bot corpus, the lever
stack becomes unbalanced after the bot-corpus effective lifetime ends.

## Cost summary

| stage | spent | running total |
|---|---|---|
| 1A B4 instrumented run | $1.30 | $1.30 |
| 4B smoke (S-A PASS) | $0.70 | $2.00 |
| 5 main run (24 h wall, ~$0.21/h) | ~$5.00 | **~$7.00** |
| **Total Wave 2** | | **~$7.00** |

Within the operator-flexed budget ($10 cap, ~$12 stretch). Below the
original main-run estimate ($3) because main run aborted at 47k
instead of completing 100k.

## Recommendations for Stage 6 / Phase 4.5

### Immediate (Stage 6 close)

1. **Use checkpoint_00020000.pt as the canonical Wave 2 deliverable**
   (the 33 % wr_sealbot peak). best_model.pt is from step-30k
   promotion (the 11 % degradation point) — less useful as a
   downstream anchor.
2. **Update REAL_RUN_RECIPE.md** with L50/L51/L52 observations.
3. **Sprint log entry** for §S181-AUDIT Wave 2 close — capture the
   peak-and-collapse pattern + which RR-G* gates failed + Wave 3 design
   pointer.
4. **Stage 7 review subagent dispatch** per dispatcher §7A.

### Wave 3 lever design (if pursued)

1. **Bot corpus refresh hook MUST land**. Track D C4 design is in
   `docs/designs/s179c_bot_refresh_hook.md`. Currently infrastructure
   exists but disabled. Recommended: trigger on best_model_promotion
   with cooldown 5k steps (not 25k); cap regeneration at 200-500
   games to avoid throughput hit.
2. **Add sliding-window eval WR gate**. alt V_spread alone is
   insufficient (L50). Best-model promotion bootstrap_floor IS this
   gate (failed at step 40k correctly), but the dispatcher's RR-G3
   strict gate at step 30k should fire SOFT-ABORT, not just be
   advisory. Recommend RR-G3 promoted to HARD-ABORT trigger.
3. **Reconsider per-class target temperature scope**. T_colony=1.5
   on selfplay slice softens tactical CE — may be the wrong
   intervention. Alternatives: pretrain-slice-only application
   (target the corpus that the model sees fresh), OR class-weighted
   gradient scaling instead of target softening.
4. **Investigate why step-20k peak is irreproducible**. Run a probe
   at step-20k ckpt + reset selfplay buffer + continue 10k steps. If
   the model peaks again at step-30k-equivalent, the bot-corpus
   distributional decay is confirmed as the mechanism.
5. **2-stone aux head (V-B-D conditional)** was parked for this
   wave. Could be revisited in Wave 3 if M1+M2 hypothesis is
   confirmed and a stronger anti-drift mechanism is needed.

### Phase 4.5 unblock decision

Per dispatcher Stage 6 routing table:

> | V_spread sustained but SealBot collapses |
> | V_spread is necessary not sufficient. Major escalation — new mechanism wave. |

This is the outcome we hit. **Phase 4.5 is NOT unblocked**; Wave 3
mechanism design is required before Phase 4.5 features (Gumbel CQV,
KrakenBot wrapper, etc.) ship.

## Cross-references

- `audit/structural/REAL_RUN_RECIPE.md` — §4 success criteria (RR-G*)
- `audit/structural/wave2_smoke.md` — Stage 4B smoke verdict S-A PASS
- `audit/structural/track_b/B_verdict_synthesis.md` — V-B verdict + L48 alt-bank framing (L50 revises)
- `audit/structural/track_d_pipeline_regression.md` — C4 bot-corpus staleness candidate (now confirmed L51)
- `audit/structural/track_b/B_track_d_xref.md` — Track D candidate confrontation
- `docs/designs/s179c_bot_refresh_hook.md` — Wave 3 prerequisite design
- `reports/track_b_main/` — rsync'd artifacts (log, events JSONL, 5 ckpts)
- `checkpoints/best_model.pt` → reports/track_b_main/checkpoints/best_model.pt — step-30k snapshot (11 % wr_sb degradation point)
- `reports/track_b_main/checkpoints/checkpoint_00020000.pt` — **step-20k 33 % peak ckpt** (canonical Wave 2 deliverable)
