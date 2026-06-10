# §S181-AUDIT Wave 1 — Track B / B4 verdict synthesis

Companion to `B_aggregation.md` (auto-generated, LITERAL L13) and
`B_track_d_xref.md` (Track D candidate confrontation). This document
surfaces the **operator-decision** picture combining all three outputs.

## Literal verdict (LITERAL L13, no judgement)

**V-B-E** — escalate; no real-run launch.

Reasoning:
1. V-B-A LITERAL **NO**: max single-source mean share = 0.563 (uniform_self)
   < 0.60 threshold across steps 500-2000
2. V-B-B LITERAL **NO**: pretrain mean 0.092 outside the [0.25, 0.45] band
3. V-B-C LITERAL **NO**: max colony_frac = 0.0984 (well below 0.50 by step 2000)
4. V-B-D LITERAL **NO**: step-1000 trunk inter-centroid ratio = 0.838 (well above 0.50)
5. Decision tree first-match wins → V-B-E (no clean match)

Per dispatcher hard constraint: **STOP. Escalate. No real-run launch.**

## Operator-information layer (NEAR-MISS + mechanism context)

### Near-miss V-B-A: selfplay channel dominance

The literal V-B-A threshold (`≥ 60 % single-source mean share`) was
calibrated against a typical "one source dominates" scenario. B4 reveals
that the *selfplay family* (uniform_self + recent) carries
**~90.8 % of total gradient pull** across the 500-2000 window, with
`uniform_self` alone at **0.563** — 3.7 percentage points below the
literal V-B-A threshold.

If the dispatcher's V-B-A rule were read as "any **source group** ≥ 60 %"
rather than "any **single source** ≥ 60 %", V-B-A would fire on the
selfplay group and route to selfplay-targeted lever (PSW + per-class
target temperature on the selfplay slice) per REAL_RUN_RECIPE §3.

### Buffer composition (B2) tells against feedback-loop hypothesis

Track D candidates C4 (bot staleness × feedback) and C5 (recency_weight
× selfplay compounding) both predicted the buffer would become
colony-heavy as the model drifts. **B2 falsifies this:** buffer
colony_frac is stable at 9 - 10 % throughout the 3000 steps. Self-play
games are NOT generating disproportionately colony-shaped positions.

### Trunk discrimination (B3) preserved

V-B-D (trunk co-adaptation) was the safety net. **B3 falsifies it for
this run:** inter-centroid distance at step 1000 is 84 % of the anchor
(threshold was ≤ 50 %). Trunk-level discrimination is intact across the
3000-step window.

### Bot corpus channel demoted

Track D ranked C1 (bot-corpus value-target imprint) #1 on likelihood ×
impact. B1 shows the pretrain slice (which includes the bot corpus
rows) carries only 0.092 mean share — the smallest of the three slices.
Even with full bot-corpus attribution, the absolute pull is bounded by
this share. **C1 is not the dominant lever.**

### Value-target signs on the selfplay buffer

Mean value targets on the selfplay-buffer samples (B2):
- colony positions: +0.005 to +0.056 (small positive — model says
  colony positions tend to win for the side that played them)
- extension positions: -0.027 to -0.118 (trending **negative** —
  extensions tend to lose for the side that played them)
- neither: +0.020 to +0.046 (small positive)

This is the **colony-attractor reinforcement signal in the raw value
targets**. The training loop's gradient pull (B1) plus the
sign-reversed value targets in the live selfplay buffer (B2)
mechanistically produce the V_spread collapse the canary fired at
step 500.

## What B4 mechanistically demonstrates

The colony-attractor mechanism observed in §S180b reproduces in B4 at
*the same step cadence and direction*:
- Step 500 dual canary: T3 V_spread = -0.039 (SOFT-ABORT signal,
  already crashed by the first checkpoint) — matches FU-1.5 L44
  observation that 86 % of trajectory loss is in the 0→2k window.
- alt-bank V_spread at step 500 = +0.245 — still above the alt-bank
  +0.07 SOFT-ABORT gate, but well below the v6 anchor +0.212 baseline
  (NB: alt was the corpus-grounded reference; in this run it has
  decayed less aggressively than T3, which matches L48).

The mechanism — selfplay buffer carries pro-colony value targets;
training loop's gradient is dominated by the selfplay slice
(~91 % combined) — is now mechanistically pinned by B4 even though
the literal V-B-A threshold misses on the single-source partition.

## Routing options for operator

### Option A — apply LITERAL L13: STOP

End the session. Escalate. No real-run launch. Defer until the
verdict tree is revised OR a different experimental probe is designed.

This is the dispatcher's hard-constraint path. Honest by the
pre-registered rules.

### Option B — apply spirit-of-V-B-A: selfplay-targeted lever

Treat the selfplay-family share dominance (~91 %) as the V-B-A
trigger. Route to REAL_RUN_RECIPE §3 `uniform_self` lever:

  > PSW (Prioritized Stratified Window on the replay buffer): under-sample
  > colony class on selfplay slice; OR per-class target temperature on
  > colony positions.

Pair with EMA (already landed Wave 2). Proceed to Stage 4 pre-launch
smoke with the lever stack.

This deviates from L13 literal but respects the mechanism the data
demonstrates. The literal threshold was authored without B4 data in
hand; operator may judge the spirit of the verdict survives the
near-miss.

### Option C — refine the verdict tree before re-running

Update REAL_RUN_RECIPE §3 V-B-A rule to read "source *or source-group*
share ≥ 60 %" and re-aggregate. Aggregator would then return V-B-A on
the selfplay group. Document the rule change in the sprint log
(needs an L## entry — likely L50: "V-B-A threshold partition by
source-group, not strict single-source, surfaces the selfplay-family
dominance B4 measured").

### Option D — partial smoke before full real-run commit

Run Stage 4 pre-launch smoke (3000 steps, ~$1.50 / ~6h) with Option B
lever stack but do NOT auto-launch Stage 5. If smoke dual-canary
V_spread (alt) holds ≥ +0.07 throughout, evidence-based GO/NOGO for
Stage 5 100k. If smoke S-C / S-D (alt crashes early), revert to
Option A escalation.

Cheap evidence path that respects both the literal V-B-E *and* the
near-miss mechanism interpretation.

## Recommendation (not binding; operator decides)

**Option D — partial smoke before full real-run commit.** B4 reframes
Track D rank order: selfplay channel is dominant, bot/pretrain channels
demoted. A $1.50 / 6h smoke with the selfplay-targeted lever stack
provides falsifiable evidence either way. If smoke fails, escalate
honestly per Option A; if smoke passes, the full real-run launch has
empirical support beyond the literal verdict gate.

## Cross-references

- `B_aggregation.md` — literal verdict + decision-tree detail
- `B_track_d_xref.md` — Track D candidate confrontation
- `B1_results.md` — gradient attribution detail
- `B2_results.md` — buffer composition detail
- `B3_trunk_drift.json` — trunk drift ladder (no markdown; JSON has full data)
- `audit/structural/REAL_RUN_RECIPE.md` — §3 lever stack + success criteria
- `docs/07_PHASE4_SPRINT_LOG.md` §S181-AUDIT — L47 / L48 / L49 context
