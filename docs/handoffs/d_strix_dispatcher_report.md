# D-STRIX dispatcher report — comparative audit vs hexo-strix — 2026-07-02

Read-only landscape audit vs `github.com/SootyOwl/hexo-strix` (SootyOwl, MIT) @ `1b8ae4d`.
Ran S1 ∥ S4 → S2 → S3 → REVIEW → RED-TEAM. Each deliverable carries its own appended
`## REVIEW` and `## RED-TEAM` sections; read those before trusting any body claim.

Deliverables:
- `d_strix_s1_economics_diff.md` — recipe economics diff table (PASS-WITH-CORRECTIONS)
- `d_strix_s2_portability.md` — 5 portability verdicts + radius-curriculum spec + tiny-net
  probe spec + axis-graph banked card (PASS-WITH-CORRECTIONS; Verdict 5 corrected inline)
- `d_strix_s4_play_ui_spec.md` — hexo-play spec + hexo.did.science verdict (PASS)
- Sprint log §D-STRIX + falsified-register row (§D-STRIX S3, custom kernel)

## Headline

**The "days-from-0" premise evaporated as a verified claim.** Their repo is a code+config-only
export: zero committed run logs, eval results, checkpoints, or throughput numbers; their own
config comments record collapses and regressions; the sole positive ("beats sealbot 5s") is an
unquantified code comment. Days-to-parity is formally incomparable regardless (time-limited
SealBot gate vs our fixed-depth-5 probe). The iteration-efficiency gap is **config-implied,
unconfirmed in either direction**. Landscape panic unwarranted.

**CORRECTION 2026-07-03:** beats-claim VOID — the line above (and the S1/RED-TEAM discussion of
"operator's net beats their bot at 512 sims") previously implied the operator holds undocumented-n
wins over hexo-strix's bot. No head-to-head vs hexo-strix (or its bot) exists at all, in either
direction — not an undocumented-n result, a nonexistent one. Strength ordering vs hexo-strix =
UNKNOWN in both directions; no head-to-head exists.

## Aggregation table (finding → action)

| Finding | Verdict after review+red-team | Action |
|---|---|---|
| Radius curriculum | TESTABLE-CHEAP (downgraded from PORTABLE — no verified benefit in OUR vast.yaml lineage either; §167 shows WR rising with radius) | Spec in S2 doc. Candidate single-variable arm for NEXT run (post-v3); re-derive endpoints for 19×19; jitter mandatory; draw-collapse + colony canaries; never bundled with solver/seeding |
| Tiny net (19.1× measured param gap; production = 4.25M confirmed) | TESTABLE-CHEAP, survives | 2×2 width×aux-heads economics probe spec in S2 doc; F1 anchor-cleanliness precondition; GPU decision = operator's |
| Axis-graph line-topology input | NOTE-ONLY | Banked representation card (S2 doc); unlock only with an independently-justified restart |
| No-window simplicity | Confirmation only | None (F1/D-DECODE already actioned it) |
| Train==deploy search consistency | NOTE-ONLY | §D-GUMBELSIMS stands at GUMBEL-SIMS-NULL/affordability-PARITY; open lever = matched-total-sim (m,n) sweep; already live on D-WS3V3 arms |
| Custom CUDA kernel | REJECT | Banked to falsified register — GNN ragged-batching problem we don't have (K-cluster varies batch count, not shape) |
| Play-UI | NOT zero-build (hexo.did.science multiplayer-only, no plug-in hook) | Operator picks: (a) `/viewer` already plays vs model locally — zero build; (b) `hexo-play/` separate repo per S4 spec, ~1 day core / 1.5–2 days with persistence+deploy scope |
| Iteration-efficiency gap | UNCONFIRMED either direction | Cheapest re-test: ask author to run their own `hexo-a0 eval-sealbot` and share output |

## Pre-registration scorecard

- "curriculum = big portable win" — **WEAKENED**: mechanism plausible, zero verified outcome on
  either side; downgraded to a testable arm, not a default.
- "tiny-net worth one cheap probe" — **HELD**.
- "axis-graph banked" — **HELD**.
- "kernel REJECT" — **HELD**, red-team found no hole.
- "landscape panic unwarranted" — **HELD, strengthened** (their verified record thinner than
  briefed).
- Surprise not pre-registered: S1 surfaced our PUCT-train/Gumbel-deploy asymmetry as driver #5;
  resolved NOTE-ONLY against the §D-GUMBELSIMS parity result.
- Correction to the brief itself: "operator's net (2.9M)" — production net measures 4,254,283
  params (aux heads NOT the delta; 2.9M matches an older res10/f112 trunk config).

## Invariants held

GNN not re-proposed anywhere; their repo untouched + nothing vendored (clone lived in session
scratchpad only); no HeXO source changes — docs/handoffs + sprint log only; same-units
discipline enforced (comparison refused where no shared anchor exists, MEASURED/CLAIMED flagged
per row).
