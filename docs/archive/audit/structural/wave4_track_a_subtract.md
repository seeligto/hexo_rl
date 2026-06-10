# §S181-AUDIT Wave 4 Track 4A — subtract-the-variable trajectory analysis

Operator-mediated SIGINT at step ~12k after step-10k eval round_complete
confirmed monotonic decline. Hypothesis "bot mix is the load-bearing
failure variable" tentatively FALSIFIED: removing bot mix produced an
earlier and steeper decline than Wave 3 (which kept bot mix).

## Run identity

| field | value |
|---|---|
| host | vast.ai 5080 (REMOTE_HOST:REMOTE_PORT) |
| branch | `phase4.5/s181_wave4_subtract` (commit `5b7f85e`) |
| variant | `v7full_baseline_minus_bot.yaml` |
| anchor | `bootstrap_model_v7full.pt` SHA `568d8a33…d61e8e98` |
| encoding | v7full (board=19, planes=8, single-window) |
| iterations target | 60 000; **actual ~12 000 (SIGINTed)** |
| run_id | `1b8c649a21e7403abf8c2d5caee087fc` |
| start | 2026-05-27T04:24:42Z |
| SIGINT | 2026-05-27T11:25Z (~7h wall, ~$2.50-3 vast) |
| sessions | 1 (clean) |

## Three DELTAs from Wave 3 main

1. `mixing.bot_batch_share: 0.30 → 0.0`
2. `mixing.bot_corpus_refresh.enabled: true → false`
3. `per_class_target_temperature.enabled: true → false`

Wave 2-3 hygiene preserved: EMA, entropy_reg 0.005 (PR-B), eta_min 5e-4
(PR-B), build_param_groups no-decay split, dual-bank V_spread canary,
hard SealBot WR gate (STRICT thresholds).

## SealBot WR trajectory (the headline)

| step | wr_sealbot | CI95 | wr_anchor | wr_best | colony_sb | colony_anchor | colony_best | promoted | elo |
|---:|---:|---|---:|---:|---:|---:|---:|:---:|---:|
| 5 000 | **19.0%** | [12.5, 27.8] | 57.0% | 59.0% | 2 | 40 | 42 | ✗ | 448 |
| 10 000 | **11.0%** | [6.3, 18.6] | 53.0% | 68.0% | 2 | 40 | 38 | **✓** | **382** |
| Δ | **-8 pp** | (CI overlap) | -4 pp | +9 pp | 0 | 0 | -4 | — | -66 |

**Peak: step 5k 19% wr_sb.** Wave 3 peak was step 10k 25%. Track 4A
peaked EARLIER and LOWER, then declined faster.

## Comparison with Wave 3 main (same step points)

| step | Wave 3 wr_sb | Track 4A wr_sb | Δ | Wave 3 wr_anchor | Track 4A wr_anchor | Δ |
|---:|---:|---:|---:|---:|---:|---:|
| 5k | 16.0% | 19.0% | **+3 pp** | 60.0% | 57.0% | -3 pp |
| 10k | 25.0% | 11.0% | **-14 pp** | 62.0% | 53.0% | -9 pp |

Track 4A started +3pp ahead of Wave 3 at step 5k but ended -14pp
behind at step 10k. The "no bot mix" recipe is WORSE on the tactical
SealBot axis at step 10k. **Hypothesis "bot mix is load-bearing"
falsified or at least non-monotonic in the bad direction.**

## L34 anchor↑/sealbot↓ check

| transition | anchor Δ | sealbot Δ | L34 fires? |
|---|---:|---:|:---:|
| 5k → 10k | -4 pp | -8 pp | NO (both down) |

**L34 does NOT fire strictly** — both anchor and sealbot declined. This
is broader value-head degradation, not the colony attractor signature.
But colony_wins_anchor share is rising: 40/57 = 70.2% at step 5k →
40/53 = 75.5% at step 10k. The model wins against anchor MORE often
via colony patterns, even as overall wr_anchor drops.

## Dual-bank V_spread canary
Not analyzed at SIGINT time. Sparse data (~2 fires at step 2k, 5k).
EMA-of-weights active; canary fires logged but not extracted.

## Verdict (LITERAL L13 per variant header)

Pre-registered table:

| ID | rule | downstream |
|---|---|---|
| W4A-A | rolling-mean SealBot WR ≥ 15% sustained 20k-50k | bot mix = load-bearing → §S150 recipe + Wave 2-3 hygiene = the answer |
| W4A-B | peak ≥ 15% but late decline | bot mix not load-bearing alone → multi-aux (Track 4B) |
| W4A-C | §S180b-style colony attractor (WR < 5% by step 40k) | bot mix NOT the variable; colony lives in trainer/loop fundamentals → Track 4B |
| W4A-D | §S175-style early crash (WR < 5% by step 20k) | severe; major mechanism re-aggregation |
| W4A-E | other anomaly | STOP, debug |

**Verdict: W4A-B (peak ≥ 15% but decline).** Peak 19% at step 5k > 15%
threshold. Decline confirmed by step 10k (11%, well below threshold).
"Late decline" is loose here — the decline started immediately after
step 5k, suggesting "no bot mix" produces a faster collapse than Wave 3.
SIGINT applied because rolling-mean criterion for budget override
failed (declining trajectory).

W4A-C (colony attractor end-state WR<5% by step 40k) cannot be applied
literally — the run didn't reach step 40k. Trajectory shape is
consistent with eventual W4A-C if continued. The dispatcher's W4A-C
routing remark applies: "bot mix NOT the variable — colony lives in
trainer/loop fundamentals → Proceed to Track 4B as fix attempt."

## Cost

- Wall: 04:25 → 11:25 UTC ≈ 7 h
- Spent: ~$2.50-3 vast 5080
- Saved vs full 60k: ~$8-10

## Lessons banked

**L56 (new):** Bot mix is NOT the load-bearing failure variable in the
colony-attractor mechanism. Removing it produces a FASTER decline than
keeping it. The hypothesis from §S178+ that "bot mix introduces the
colony" is at least partially inverted — bot mix may actually delay or
attenuate the colony pattern; removing it accelerates it. The Wave-2/3
attractor mechanism lives downstream of bot mix, in the trainer or
loss structure.

**L57 (new):** Track 4A peaked at step 5k (19%) not step 10k like Wave 3
(25%). Suggests recipe sensitivity to selfplay supply rate. Without
bot mix, the trainer's `training_steps_per_game=2.0` rate forces
heavy reuse of early selfplay buffer, possibly causing premature
exposure to bootstrap distribution drift.

**L34 refinement (Wave 4 instance):** Track 4A produced anchor↓ +
sealbot↓ pattern (both declining), not the classic anchor↑/sealbot↓.
But colony_wins fraction vs anchor rose 70.2% → 75.5%, indicating
colony-pattern composition strengthening even as raw anchor WR falls.
The "colony attractor" signature can manifest without strict L34 fire.

## Track 4B handoff

Following dispatcher W4A-C routing per the broader colony-attractor
interpretation. Track 4B sustained launched 2026-05-27T11:44Z on
v7full_wave4_multiaux_w4ac variant (v7_wave3_main parent + multi-aux
density + ply_index head). Run_id 8e4568c66c80467cb764cd3aa00a8754.
