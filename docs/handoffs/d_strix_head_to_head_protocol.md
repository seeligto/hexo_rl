# D-EVALGATE G3 — Tyto/hexo-strix head-to-head protocol (pre-registered, ops doc)

Docs-only. Nothing built, nothing runs from this file. Operator executes by hand
(message send + play games on `hexo.did.science`). Venue check (§D-STRIX S4): that
site is human-multiplayer only, no bot-upload API — a "head-to-head" there means
**two operators manually entering moves from their own local bot process**, not the
site running either checkpoint. Compute is NOT site-enforced. Read before playing.

## 1. Artifact ask — message template to Tyto

> Before we play a head-to-head, want to trade run artifacts so both sides can read
> the result honestly — training curves, eval records (WR vs your own baselines,
> n and CI if you have them), and checkpoint lineage (step count, params trained,
> any curriculum stages). We'll send ours back. No obligation to share anything you
> consider unfinished/private — even a partial trade beats none.

3 sentences, mutual framing, no pressure. Send as-is or trim.

## 2. Match protocol — PIN ALL OF THIS BEFORE FIRST MOVE

**Compute budget.** Site cannot enforce equal compute (no bot-upload hook, see venue
note above) — state this to Tyto explicitly, don't paper over it. Each side runs
whatever sims/depth/time its own process uses locally; record the actual number
used (sims count or wall-clock/move) in the post-hoc writeup. Ask Tyto to state
theirs before play, we state ours before play — a stated-but-unenforced budget is
still worth more than an unstated one.

**n games: 40** (20 per starting color/side, see alternation below). Power
justification: at n=40, worst-case-variance (p≈0.5) 95% CI half-width ≈ ±15-16pp.
This resolves a *dominant* result (WR ≥75% or ≤25%) but NOT a close edge (55-60%
range) — pre-register that a close split is read as **indeterminate**, not "parity
confirmed" (absence of a resolved gap ≠ evidence of equality at this n).

**Opening diversity — mandatory.** Per CLAUDE.md §D-ARGMAX effective-n corollary: a
deterministic/argmax regime collapses to ~2 distinct games per pairing (openings
repeat, engine plays the same continuation from the same state) — a raw game count
of 40 could carry an effective n near 2 if diversity isn't forced. Requirement:
each of the 40 games starts from a distinct human-chosen opening (vary first 1-2
placed stones across the set; no repeated opening square/pair). If either bot runs
temp-0/argmax internally, the opening injection is the ONLY source of game
diversity — do not skip it.

**Alternation / color balance.** 20 games each bot moves first (P1 opens with the
single first stone), strict alternation (not random) so a first-move advantage
can't confound the tally.

**What counts as a game.** Completed game (win/loss/draw by board rule) counts.
Disconnect/timeout/abandon before completion: excluded from n, logged separately
(count + which side dropped) — do not backfill with a coin-flip or resume-from-
same-state re-attempt (that would silently violate the fixed-opening pre-registration
above).

**CI method.** Dedupe byte-identical game records first (guards against the
argmax-collapse failure mode even after diversity injection — verify it worked,
don't assume it). Bootstrap the win-rate CI over the DISTINCT deduped games, not a
raw-count Wilson interval — a Wilson CI over the nominal 40 overstates confidence
if effective distinct games < 40 (§D-ARGMAX: this exact overstatement mechanism
manufactured a false "CI-resolved" gap elsewhere in this project).

## 3. Claim-scope — pre-register BEFORE play, not after seeing the result

Compute/hardware are uncontrolled by the site (§1). Two readings, both legal
outputs of this match, pick the right one per the actual recorded sims/budget gap:

- **If sims/hardware end up roughly matched** (both sides disclose comparable
  budgets): result speaks to relative net/policy strength under that budget.
- **If sims/hardware are unequal or unknown** (default expectation given no site
  enforcement): the match measures **"X's deployment beats Y's deployment"** —
  a budget+net bundle — NOT the unlicensed stronger claim **"X's net is stronger."**
  Do not let a win under an unknown/mismatched budget get reported as a net-strength
  claim either direction.

Outcome buckets, fixed now:
- **Win** (≥75% WR, CI excludes 50%, diversity check passed): report per the
  correct reading above (budget-bundle unless matched).
- **Loss** (≤25% WR, same conditions): same, reversed.
- **Indeterminate**: WR in 25-75%, OR diversity/dedup check fails (effective n too
  low to resolve anything) — report as indeterminate, do not round to parity.

## 4. Kill / outcome line

Either this produces the first real cross-bot datum (a played, recorded,
pre-registered-conditions match — first one this project has), or it dies on
uncontrolled compute (budgets never disclosed/matched, diversity injection
skipped, or Tyto declines). Record which outcome occurred either way — a declined
or aborted attempt is itself a result worth one line in the sprint log, not silence.
