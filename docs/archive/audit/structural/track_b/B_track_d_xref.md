# §S181-AUDIT Wave 1 — Track B B4 ↔ Track D cross-reference

Companion to `B_aggregation.md` (V-B verdict) +
`audit/structural/track_d_pipeline_regression.md` (5 smoking-gun
candidates). Maps each Track D candidate against the B4 instrumented
findings.

## B4 headline numbers

| measure | value | source |
|---|---|---|
| per-source gradient mean share `uniform_self` (steps 500-2000) | **0.563** (near-miss V-B-A 0.60 gate) | B1 |
| per-source gradient mean share `recent` | 0.345 | B1 |
| per-source gradient mean share `pretrain` | 0.092 | B1 |
| max single-step source share (any source, any step) | 0.858 | B1 |
| buffer colony_frac trajectory step 500→3000 | 0.094 → 0.098 (stable, NO drift) | B2 |
| buffer extension mean value target step 500→3000 | +0.061 → -0.027 (trending negative) | B2 |
| buffer colony mean value target step 500→3000 | +0.051 → +0.008 (small positive throughout) | B2 |
| trunk inter_centroid_dist anchor / step-1000 / step-3000 | 23.23 / 19.46 / 22.13 (84% of anchor at step 1000) | B3 |
| dual canary at step 500 | T3=-0.039 (SOFT-ABORT), alt=+0.245 (PASS) | structlog |

## Candidate-by-candidate confrontation

### C1 — Bot-corpus value-target imprint at small share

**Track D prediction.** `bot_batch_share=0.30` imprints a colony-favouring
direction. Static corpus goes stale; this is the L34 channel.

**B4 evidence.**
- Bot corpus IS in the `pretrain` slice (mixing pulls from `bootstrap_corpus_v6.npz`
  + `bot_corpus_s178_sealbot_vs_v6.npz`). Wait — that conflates them.
  Re-check: B1 attribution buckets pretrain / recent / uniform_self;
  bot corpus rows land in the same slice as bootstrap-corpus rows
  per the trainer's batch assembly (see `hexo_rl/training/mixing.py`).
- `pretrain` mean share **0.092** — by far the smallest contributor.
  Even if 100% of pretrain-slice pull came from bot corpus (it doesn't —
  bot is one of two pretrain sources), it is too small to be the
  load-bearing channel under B1.
- Buffer colony-fraction stable at ~9 % throughout the run — the bot
  corpus is NOT seeding colony positions into the self-play buffer.

**Confrontation.** Track D ranked C1 #1 on likelihood × impact. B1
demotes it sharply: pretrain-slice pull is the smallest of the three.
Static-staleness mechanism still plausible (the corpus is unchanging
while the model drifts) but the magnitude is bounded by the 9 % gradient
share. **C1 is NOT the dominant lever.**

### C2 — Pretrain corpus colony pull × `recency_weight=0.75` cross

**Track D prediction.** Pretrain corpus asymmetry +0.157 (A5) × the
recency-weighted ring buffer compounding gives the largest single
contribution to V_spread direction.

**B4 evidence.**
- Pretrain slice pull is 0.092 mean — small.
- `recent` slice pull is 0.345 mean — moderate.
- **Combined `recent + pretrain` = 0.437**, still less than uniform_self
  alone (0.563).
- The hypothesised cross does exist (recent ring receives pretrain bias
  + selfplay bias), but B1 isolates the gradient pull per slice and the
  pretrain pull does not dominate.

**Confrontation.** Track D ranked C2 #2 on likelihood × impact. B1
shows it is real but the recent + pretrain combined pull is dominated
by `uniform_self`. **C2 contributes but is not the load-bearing channel.**

### C3 — `ply_cap_value=0.0` × `playout_cap.full_search_prob=0.5` cross

**Track D prediction.** Truncation positions carry value target 0,
combined with longer full-search games → more truncations → value head
learns flat targets → V_spread collapse.

**B4 evidence.**
- B2 buffer `neither` class dominates at 79 % throughout. The "neither"
  class is the heterogeneous remainder — would include ply-truncated
  positions. Mean v_target on `neither` hovers +0.02 → +0.04 (positive,
  small), NOT zero. So either ply-truncations are a minority of `neither`
  positions, or `neither` includes wins that pull the mean off zero.
- The B4 run did NOT log raw value-target distribution at truncation
  positions specifically. The canary's T3 collapse at step 500 (already
  -0.039) is consistent with very-fast-onset training-loop pressure
  rather than gradual ply-cap accumulation (truncations would build up
  over many games).

**Confrontation.** Track D ranked C3 #3. B4 data is inconclusive on C3
specifically — would require a `value_pred_at_ply_cap` event ladder
(the variant config has it enabled, but the structlog truncation by
default may have suppressed it). **C3 plausible but unmeasured by B1/B2/B3.**

### C4 — Bot-corpus staleness × outcome-channel feedback

**Track D prediction.** Static bot corpus pulls in the wrong direction
as the model drifts. Refresh hook needed.

**B4 evidence.**
- The variant has `bot_corpus_refresh.enabled: false` (deliberately —
  static for B4 instrumentation). So this channel is active.
- B1 pretrain slice pull is 0.092 — bot's share of this is bounded;
  the static-corpus drift channel is small in absolute terms.

**Confrontation.** Track D ranked C4 #4. B4 confirms the channel is
small. **C4 is not the dominant lever; refresh hook would target a
small fraction of total pull.**

### C5 — `recency_weight=0.75` × selfplay-buffer compounding

**Track D prediction.** 75 % of the batch from recent ring; once selfplay
is colony-biased, ring is dominated by colony positions.

**B4 evidence.**
- B2 shows buffer is NOT becoming colony-biased — colony_frac stable at
  ~9 % throughout the 3000 steps. The feedback-loop premise of C5
  (selfplay shifts colony → ring colonises) does NOT manifest in B4.
- B1 shows `recent + uniform_self` jointly carry ~91 % of pull. The
  selfplay channel is dominant by share but the FEEDBACK MECHANISM
  Track D names (ring colonisation) is NOT firing.

**Confrontation.** Track D classified C5 INCONCLUSIVE. B4 confirms
INCONCLUSIVE on the feedback mechanism — but reveals the selfplay
channel is the dominant gradient pull regardless. **C5 mechanism not
confirmed; selfplay channel dominance IS confirmed by B1.**

## Summary

B4 partially overturns Track D's ranking:

| candidate | Track D rank | B4 verdict |
|---|---:|---|
| C1 bot-corpus imprint | #1 | DEMOTED — pretrain slice pull 0.092 (smallest) |
| C2 pretrain × recency_weight cross | #2 | PARTIAL — present but bounded |
| C3 ply_cap × playout_cap cross | #3 | INCONCLUSIVE — B4 didn't probe truncation positions specifically |
| C4 bot-corpus staleness | #4 | DEMOTED — small absolute magnitude |
| C5 recency_weight cross | #5 (INCONCLUSIVE) | MECHANISM NOT CONFIRMED — buffer not colonising; **but selfplay channel IS the dominant gradient pull** |

**The B4 reframe: the dominant force on the value head is the live
selfplay gradient signal (`uniform_self` + `recent` share = 90.8 %),
which by literal V-B-A threshold (≥60 % single source) misses because
the pull is split across two selfplay slices. If V-B-A were stated as
"selfplay-family share ≥60 %" rather than "any single source ≥60 %",
the verdict would be V-B-A → selfplay-targeted lever.**

This favours operator-judged routing: PSW (Prioritized Stratified
Window on the selfplay portion of the buffer) + per-class target
temperature on colony positions inside the selfplay slice — Track D
C5 / REAL_RUN_RECIPE V-B-A `uniform_self` variant routing — over the
literal V-B-E escalation. **Operator decides.**
