# RUN3 CONVENE RULING — AMENDMENT 1 (2026-07-13, commit alongside the ruling)
# Strategy layer. Amends RUN3_CONVENE_RULING.md. The original mapping is superseded ONLY
# as stated below; everything else stands.

## RECORDED VERDICT 2 (WP3, computed)
MIXED per the frozen rule, verbatim: Δ(gnn-bc − cnn-bc) = +418 BT Elo, CI [+318, +580],
excl 0 (✓, 4× the +100 bar); gnn-bc (−96) < mantis-261k-raw (+53) (✗). Head-to-head
59-0-5. gnn-bc: 284k params, 40k BC steps, top1 0.418. cnn-bc: 571k params, same
protocol, top1 0.426.

## REGISTERED MISS (recorded, not rescored)
The ARCH-DOMINANT conjunction was mis-specified by the strategy layer: the
"gnn-bc ≥ mantis-261k-raw" clause compares 40k supervised steps to 260k+ RL steps —
it tests whether BC can replace RL, not whether the architecture is better. The
architecture-isolating clause (Δ vs matched control) fired unambiguously. The verdict
stays MIXED on the books; this paragraph is the miss record. Lesson to register: verdict
conjunctions must not mix questions — one clause per question, each with its own
comparison class.

## SUPERSESSION (open, operator-visible — not a silent swerve)
The MIXED→"run3 GO CNN same-day" branch is SUSPENDED. Grounds: two independent
measurements now identify representation as the binding constraint (tournament raw gap
~230 Elo at production scale; matched-BC gap +418 at probe scale, direction robust to the
control-arm caveat). Auto-launching a CNN GPU-week against that evidence honors the
letter of the mapping and violates its purpose (route the week to the highest-EV run).

## NEW PRE-REGISTERED PATH (frozen now)
  R1. cnn-bc adapter sanity check (0/128 vs solvers is anomalous; the WP3 agent flagged
      it). If handicapped: re-run the 64-game gnn-vs-cnn cell only; Δ magnitude updates,
      decision logic below unchanged unless direction flips (it will not be assumed).
  R2. BC-SCALING RUNG (idle 5080, hours): extend/retrain gnn-bc to 150–200k BC steps
      (or strix-tier config from configs/axis/), same corpus, same eval field + book.
      FROZEN: BC-scaled gnn ≥ mantis-261k-raw (CI incl.) → ARCHITECTURE CASE CLOSED;
      recommendation hardens to run3 = GNN + dist-tail. Below mantis but improved vs 40k
      → case remains strong (tournament covers production scale); proceed to R4 decision
      with that stated. Flat vs 40k → BC-saturated; decision rests on tournament evidence
      alone, stated as such.
  R3. PARALLEL (sanctioned under MIXED mapping as run4 design work): GNN integration
      scoping doc — graph builder in our loop, MCTS/PyO3 interface, CONFRES seams,
      dist65-on-GNN head, encoding gates, eval-instrument compatibility — with
      per-component estimates from measured scopes (files touched, seams named), not
      vibes. This document is required input to R4 regardless of which run3 launches.
  R4. THE DECISION (operator, one message, after R1–R3): 
      (a) run3 = GNN + dist-tail as Mantis run #1 — integration lead time from R3
          accepted; comparison baseline = bar ladder + strix-raw ceiling (the run2-slope
          one-variable comparison is explicitly given up, stated).
      (b) run3 = CNN + dist65 interim now, GNN = run4 — legitimate only with the
          opportunity cost stated (a week of box + attention on a dominated
          representation).
      The CNN launch package stays prepped and launchable under both branches.

## TRANSFERS (unchanged by any branch)
E1 REVIVE + dist65 head + 234-probe + recognition-lag harness + EVALFAIR instrument +
promotion-gate isolation + external-eval doctrine are trunk-agnostic and ride onto
whichever run3 launches. Strix ships a scalar MSE value head — GNN + dist-tail is a
configuration no bot in the field has.
