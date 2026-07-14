# RUN3 CONVENE RULING — 2026-07-13

**Strategy layer. Supersedes chat-only branch mappings. Binding until amended in writing.**

> **AMENDED 2026-07-13** by `run3_convene_ruling_amendment_1.md`: WP3 verdict recorded
> (MIXED); the MIXED→"run3 GO CNN same-day" branch of the outcome mapping below is
> SUSPENDED. Read the amendment before consuming the mapping. Everything else stands.

This document is the in-repo canonical instance of the run3 convene ruling. It exists
because a frozen gate references it and, per the PROCESS RULE below, a chat-only ruling
is not citable by dispatchers. See `docs/handoffs/gnn_bc_probe_runbook.md` (WP3) and the
D-L STRIXPROBE state.

---

## RECORDED VERDICT 1 (WP1, computed)

**GO.** Paired dist−scalar vs sealbot-d5 = **−0.043, 95% CI [−0.133, +0.043], spans 0**
(scalar 0.586 [0.516, 0.652] · dist65 0.543 [0.473, 0.605] · eff_n 128 both, 0 bad pairs).

Run3 value head = **dist65, LOCKED**. Language discipline: report as "no significant
strength cost," not "free" — the CI leans negative; the gate accepts this by design
(the compounding mechanism is invisible in a 50k fine-tune; that is why the gate is not a
point-estimate gate).

Verdict doc: `reports/e1/dl_sealbot/WP1_VERDICT.md`;
data: `reports/e1/dl_sealbot/{scalar_298k,dist65_298k}/`.

---

## SEQUENCING RULING: B+ (probe-first, prep-parallel). C REJECTED.

Rationale: the convene rule requires BOTH verdicts before launch/pivot; WP3 is hours and
zero-blockers; launching run3 ahead of a result built to redirect it trades ~1 day of
latency for sunk-cost exposure on a GPU-week. Rejected explicitly so it stays rejected.

Actions:

1. Launch WP3 BC training on the idle 5080 per `docs/handoffs/gnn_bc_probe_runbook.md`.
   4 runs → 640-game raw RR → BT + pair-bootstrap. Report the verdict verbatim.
2. Parallel: prep run3 CNN+dist launch package (variant yaml from the frozen spec:
   fresh init `v6_live2_ls`, dist65 head, sims 150, promotion gate 25k / 64-pair /
   CI>0.5 subprocess-isolated, in-loop 100k strength eval disabled, external via
   `mantis_pull_eval`; fresh seeded books per stage). **Prep ≠ launch.**
3. No run3 launch and no pivot until the WP3 verdict is quoted against the mapping below.

---

## PRE-REGISTERED OUTCOME MAPPING (the "strategy ruling," now in-repo)

- **ARCH-NULL** (gnn-bc ≈ cnn-bc, CI spans 0):
  → run3 GO same-day: CNN + dist65 as specced. GNN register entry stays
    re-opened-unproven; probe banked as run4 evidence.

- **MIXED** (gnn-bc > cnn-bc but < +100 BT Elo):
  → run3 GO same-day: CNN + dist65 as specced. GNN = run4 pre-registered card #1 with
    this probe as its evidence base; run4 design work may start during run3.

- **ARCH-DOMINANT** (gnn-bc ≥ cnn-bc +100, CI excl 0, AND gnn-bc ≥ mantis-261k-raw):
  → run3 launch HELD. Operator decision within one message, costed honestly:
    - (i) pivot run3 to GNN + dist-tail — restart-class; requires integration lead time
      (graph builder + MCTS interface + CONFRES seams + encoding gates) estimated in
      DAYS-TO-WEEKS, estimate to be produced before the decision, from measured
      component scopes, not vibes;
    - (ii) launch CNN+dist run3 now as the interim run while GNN integrates for run4 —
      legitimate ONLY as an explicit choice with the opportunity cost stated.

    Neither branch is pre-decided; ARCH-DOMINANT escalates, it does not auto-pivot.

---

## PROCESS RULE (new, binding — the honesty-flag lesson)

Any strategy-layer ruling that a frozen gate references MUST be committed to the repo
(`docs/handoffs/` or `docs/designs/`) before the gate can consume it. Chat-only rulings
are not citable by dispatchers. This document is the corrective instance.

---

## STANDING (unchanged)

E1 REVIVE per pre-registration. WP1 GO recorded above. Idle-5080 spend is authorized for
WP3 only; stop the box after the probe artifacts are rsync'd and verified.
