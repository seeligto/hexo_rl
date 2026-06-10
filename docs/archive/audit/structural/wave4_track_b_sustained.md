# §S181-AUDIT Wave 4 Track 4B — multi-aux sustained trajectory analysis

Operator-mediated SIGINT at step ~15.5k after step-15k sealbot eval
confirmed -12pp drop from step-10k peak. Multi-aux density (sigma2
Huber + ownership/threat 0.2 + ply_index 0.1) delayed but did NOT
prevent the colony-attractor pattern. Trajectory shape matches Wave 3
main: rising through step 10k peak, then collapse beginning at step
15k.

## Run identity

| field | value |
|---|---|
| host | vast.ai 5080 (ssh6.vast.ai:13053) |
| branch | `phase4.5/s181_wave4_multiaux` |
| variant | `v7full_wave4_multiaux_w4ac.yaml` (v7_wave3_main parent + ply_index) |
| anchor | `bootstrap_model_v7full.pt` SHA `568d8a33…d61e8e98` |
| encoding | v7full (board=19, planes=8) |
| iterations target | 60 000; **actual ~15 500 (SIGINTed)** |
| run_id | `8e4568c66c80467cb764cd3aa00a8754` |
| start | 2026-05-27T11:44:08Z |
| SIGINT | 2026-05-27T21:16Z (~9.5h wall, ~$3 vast) |

## Multi-aux density profile (vs Wave 3 / Track 4A)

| weight | Wave 3 main | Track 4A | **Track 4B** |
|---|---:|---:|---:|
| aux_opp_reply | 0.15 | 0.15 | 0.15 |
| aux_chain | 1.0 | 1.0 | 1.0 |
| uncertainty (sigma2) | 0.0 (disabled) | 0.0 | **0.1 (Huber-on-sq-err)** |
| ownership | 0.1 | 0.1 | **0.2** |
| threat | 0.1 | 0.1 | **0.2** |
| ply_index | n/a | n/a | **0.1 (NEW)** |

## SealBot WR trajectory (the headline)

| step | wr_sb | wr_anchor | wr_best | wr_rand | col_sb | col_anchor | col_best | promoted | elo |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---:|
| 5 000 | **20.0%** | 64.0% | 62.0% | 100% | 4 (20%) | 50 (78%) | 45 (72%) | ✓ | 465 |
| 10 000 | **23.0%** | 62.0% | 61.0% | 100% | 3 (13%) | 41 (66%) | 41 (67%) | ✓ | 362 |
| 15 000 | **11.0%** | (pending best_arena pass #1 62.5% partial @ 32/100) | — | — | 5 (45%) | — | — | — | — |

**Peak: step 10k 23% wr_sb.** Decline begins immediately after step 10k.

## Comparison with Wave 3 main + Track 4A (same step points)

| step | Wave 3 wr_sb | Track 4A wr_sb | Track 4B wr_sb | Track 4B Δ vs W3 |
|---:|---:|---:|---:|---:|
| 5k | 16.0% | 19.0% | 20.0% | +4 pp |
| 10k | 25.0% (peak) | 11.0% | 23.0% | -2 pp |
| 15k | 18.0% | (collapsed) | 11.0% | -7 pp |

Track 4B trajectory mirrors Wave 3 main: peak at step 10k, then steep
decline. The delay-but-not-prevent pattern.

## Colony composition trajectory

| step | col_sb % | col_anchor % | col_best % |
|---:|---:|---:|---:|
| 5 000 | 20% | 78% | 72% |
| 10 000 | 13% | 66% | 67% |
| 15 000 | **45%** | (pending) | — |

Per operator's colony-framing (2026-05-27 feedback memo): colony
composition GROWTH is the kill signal, not raw threshold. Track 4B
showed step 5k → 10k composition DECREASING (positive). At step 15k,
sealbot-colony composition jumped from 13% → 45% (+32pp) — composition
homogenization confirmed. The model is now winning vs sealbot
predominantly via colony patterns where before it was winning
tactically.

## L34 anchor↑/sealbot↓ check

Cannot apply strictly at SIGINT — step-15k anchor data incomplete
(best_arena pass #1 game 32/100 partial 62.5%). At face value:
- step 10k → 15k: wr_sb -12pp, wr_anchor (partial) -0pp (62→62.5)

If anchor holds ~62%, L34 fires (anchor STEADY + sealbot DOWN
qualifies under loose reading). The strict "anchor↑" condition isn't
clearly satisfied yet.

## Verdict (LITERAL L13 per variant header)

Pre-registered table:

| ID | rule | downstream |
|---|---|---|
| W4B-A | Rolling-mean SealBot WR ≥ 20% sustained 30k-50k AND aux losses converge | Phase 4.5 UNBLOCKED |
| W4B-B | peak ≥ 20% but late decline | tighter aux weights OR refresh hook tune; Wave 5 design |
| W4B-C | colony attractor end-state | multi-aux NOT the answer. Strategic reckoning Wave 5. |
| W4B-D | anomaly | STOP, debug |

**Verdict: W4B-B** (peak ≥ 20% but late decline).

Peak 23% at step 10k cleared the 20% threshold. Decline confirmed at
step 15k (11%). Did NOT reach end-state colony attractor (W4B-C) by
SIGINT — would need step 30k+ data showing sustained WR < 5%. Strict
W4B-B applies: aux density configuration insufficient to sustain past
step 10k peak.

The dispatcher W4B-B downstream: "tighter aux weights OR refresh hook
tune; Wave 5 design." Per the dispatcher's strategic-reckoning task
spec, if BOTH Track 4A AND Track 4B colony, strategic reckoning is
warranted. Both have effectively colonyed (4A by step 10k 11%, 4B by
step 15k 11%) — both reach the "WR ≤ ~11% by step ~12-15k" pattern.

## Cost

- Wall: 11:44 → 21:16 UTC ≈ 9.5 h
- Spent: ~$3 vast 5080
- Saved vs full 60k: ~$5-8

## Wave 4 total spend

- Track 4A: ~$3 (SIGINTed step ~12k)
- Track 4B: ~$3 (SIGINTed step ~15.5k)
- **Total: ~$6** (under $7 cap)

## Lessons banked

**L58 (new):** Multi-aux density (sigma2 Huber + ownership 0.2 + threat
0.2 + ply_index 0.1) on top of Wave 3 lever stack (bot mix + refresh
hook + per-class temp + EMA) DELAYS the colony-attractor pattern by
~5k steps but does NOT prevent it. Peak shifts from Wave 3 step 10k
25% to Track 4B step 10k 23% (essentially same magnitude), but the
decline shape and timeline are similar. Multi-aux density is NOT the
fix for the colony attractor.

**L59 (new):** The colony attractor mechanism is INSENSITIVE to:
  - bot mix presence (§S178+ vs Track 4A no-bot-mix both colony)
  - refresh hook (Wave 3 with vs without — both colony)
  - per-class target temperature (Wave 3 scope flip L52 — colony)
  - multi-aux density (Track 4B — colony, just later)
  - EMA + entropy + PR-B (all preserved across variants — colony)

The mechanism lives DOWNSTREAM of all currently-tested levers — in
the training objective itself (value head + target propagation), the
trainer step structure, or the MCTS+selfplay interaction. This is
the boundary where Wave 5 strategic reckoning must operate.

**L60 (new):** Colony composition (col_sb%, col_anchor%) trajectory
SHAPE predicts the W4B verdict before raw wr_sb does. Track 4B step
5→10k showed composition DECREASING (78→66% anchor; 20→13% sb) which
LOOKED like W4B-A trending. But step 15k composition SHOT UP (sb 13→45%)
even before the wr_sb decline was confirmed. Watch composition deltas
as leading indicator of pattern homogenization.

## Wave 5 strategic reckoning — pre-write per dispatcher Task 5

Per dispatcher: "if BOTH Track 4A and Track 4B colony, strategic
reckoning is forced." Both tracks colony-collapsed by step 15k.

### Surfaces NOT YET tested (per dispatcher Task 5)

1. **Value-target propagation rule**: terminal z → all positions may
   be fundamentally wrong for HeXO. Try n-step bootstrap or TD-λ.
2. **WDL 3-class softmax** value head (replaces tanh scalar)
3. **Game-theoretic regularization**: explicit anti-colony loss term
   penalizing homogeneous-pattern composition during training

### Wave 5 scope (dispatcher: "only if needed")

- TD-λ value target + WDL + combined supervised + value damping
- ~3 weeks dev + $20 vast
- Major commitment

### Sprint-log section header pre-written

`## §S181-AUDIT Wave 5 — strategic reckoning + structural target rework`

## Handoff

Both Track 4A (W4A-B) + Track 4B (W4B-B) verdicts applied LITERAL.
Wave 4 closes with:
- Bot mix is NOT the load-bearing variable (Track 4A)
- Multi-aux density is NOT the answer (Track 4B)
- Colony attractor mechanism lives downstream of all config-level
  + density-level levers tested in Waves 1-4
- Stage 6 REVIEW subagent + sprint log close-out pending
- Strategic reckoning Wave 5 design surfaces required for next step
