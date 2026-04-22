---
name: investigation-probe-smoke-verdict
description: >
  Run when an investigation needs to discriminate between two competing hypotheses
  using a probe or smoke-test protocol before any full training run. Trigger phrases
  include "probe failed", "smoke to discriminate", "E1 vs E2", "need a smoke run",
  "fixed point at", "regression against baseline", "which hypothesis holds",
  "canary inverted", "contrast went negative", "discriminator experiment".
  Apply for any canary-style measurement where the outcome decides whether a deeper
  investigation proceeds or halts. Use when a training metric looks pathological,
  when a corpus/bootstrap anomaly is suspected, when a fixture looks wrong, or
  when you need to decide between two hypotheses before spending days on a full
  sustained run.
---

# Investigation — probe → smoke → verdict

A discipline for separating real trainer/architecture pathology from fixture artifacts,
RNG variance, or lurking variables. Skip the discipline and you burn GPU-days chasing
a bug that the probe fixture itself introduced.

## When to use

- Threat/ownership/value probe returned a result that contradicts a prior baseline.
- Training metric looks pathological (e.g. loss inverted, policy entropy collapsed).
- Suspected corpus / bootstrap anomaly — position counts, Elo weighting, plane layout.
- Need to decide between two competing causes (trainer bug vs. fixture bug,
  augmentation bug vs. augmentation correctness) before committing to a long run.
- Before promoting any checkpoint to `best_model`.

## Steps (7-stage pipeline)

1. **Fixture audit.** Dump shapes, value ranges, and a few human-readable samples
   from the fixture before measuring anything. Confirm the fixture represents the
   condition you think it does (real late-game positions vs. synthetic, aug-on vs.
   aug-off, player-perspective correct). Most "shock" probe results are fixture
   artifacts — audit first.

2. **Two-arm hypothesis design.** Name E1 and E2 with explicit expected outcomes.
   A single-arm design cannot discriminate; if only E1 is defined, stop and add E2.
   Example: E1 = "perspective flip in trainer is wrong", E2 = "perspective flip is
   correct and low contrast is fixture drift".

3. **Threshold selection.** Pre-register a numeric pass/fail gate for each arm
   BEFORE running. Post-hoc threshold moves are the single biggest source of false
   positives. Write the thresholds into the investigation doc.

4. **Smoke run.** Small N, short horizon. Target 5–30 minutes, not hours. Capture
   enough signal to cross the threshold; no more. Use existing make targets
   (`make probe.*`, `make train.smoke`) where possible.

5. **Verdict.** One-line PASS/FAIL per arm, with the measured value next to the
   pre-registered threshold. No qualifiers; decisions are binary here.

6. **Report.** Short doc (prefer `reports/investigations/<topic>_<date>.md`) with:
   fixture provenance, E1/E2, thresholds, smoke results, verdict, next action.
   Reviewer-friendly — a second agent should be able to audit in under 5 minutes.

7. **Sprint-log § append.** Add a new § to `docs/07_PHASE4_SPRINT_LOG.md` with the
   verdict and a link to the investigation doc. This is how learning memory persists.

## Anti-patterns

- **Post-hoc threshold moves.** Threshold set after seeing the number is not a
  discriminator; it's storytelling.
- **Untested fixture.** Using a fixture without the Step 1 audit — Q27 §106 landed
  on exactly this trap (C2/C3 failure was a synthetic-fixture artifact).
- **Single-arm design.** "If the contrast is low, the trainer is broken" — no, it
  could be the fixture, or the probe, or RNG. Always define the alternative.
- **Skipping to full training.** Running a 24-hr sustained run before the smoke has
  decided is how you burn a weekend on a fixture bug.
- **Rewriting the report to fit the verdict.** The report captures the smoke as run;
  if you change hypotheses mid-investigation, log that separately.

## Reference investigations

- **Q27 §106** (Probe 1b): C2/C3 failure was a fixture artifact, not a trainer
  pathology. The fixture-audit step would have caught it on day 1.
- **Q33-B §110** (trainer-fit sanity check): fixed point at `pe_self ≈ 5.36`;
  two-arm design separated "trainer converges correctly" from "training signal
  is pathological".
- **Q33-C2 §112** (augmentation discriminator retry): E1 confirmed after the
  initial §111 run was halted; disciplined retry with pre-registered thresholds
  produced a clean verdict and resolved Q37, unblocking Phase 4.5.
