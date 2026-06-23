# D-TEMPDECAY — final report (per-game within-game temperature decay)

Investigation: is the within-game cosine temperature schedule broken, can it be made
clean + working, and does re-enabling it (at a higher floor than the §143-toxic 0.05)
reduce late z-label noise without re-triggering the L9 draw-collapse?

Pre-registration (numbers fixed before any probe): `reports/investigations/tempdecay_phase0_2026-06-12.md`.
Branch `phase4.5/tempdecay` (pushed). Host: vast RTX 4080 (`ssh9.vast.ai:31743`, $0.21/hr).

## Verdict table

| Criterion | Verdict | Evidence |
|---|---|---|
| Is the cosine "broken"? | **NO — CORRECT-BUT-TOXIC-LEVER** | 11-agent audit + adversarial re-review: math + wiring correct; the §156/L9 collapse was floor=0.05, not a code bug. All 6 load-bearing claims held, 0 refutations. |
| Clean + working? | **YES — landed** | Commit `711919d`: one quarter-cosine, OFF=constant-0.5 default (footgun de-armed), eval helper unified, dead key/field removed, cross-language parity test. 2006 Python + engine + parity green. |
| L9-transfer to floor 0.20–0.45 (static regime)? | **DOES NOT TRANSFER** | Phase-2 probe (golong@50k static, N=400/arm): all 3 schedule arms PASS draw-safety. Aggressive floors draw *less* than control. |
| Does the schedule reduce late z-label noise (training)? | **NO — TEMP-NEGATIVE** | Smoke (control vs a20, 10k steps each): clean shared-data `value_accuracy_corpus` FLAT (Δ=−0.001, CI straddles 0). The pre-reg `value_accuracy_masked` "+0.016" is a confound artifact. Not toxic (gate never fired), but a20 mildly elevates training draws. |

## Phase 2 — draw-safety probe (golong@50k static, N=400/arm)

| arm | floor | draw rate | 95% CI (deduped) | median len (plies) | opening div (ply 1/3/5) | verdict |
|---|---|---|---|---|---|---|
| control | 0.50 (OFF) | 0.0475 | [0.028, 0.070] | 92.7 | 0.063 / 0.928 / 0.993 | baseline |
| a45 | 0.45 | 0.1225 | [0.093, 0.155] | 110.4 | 0.063 / 0.988 / 1.000 | **PROBE-PASS** |
| a30 | 0.30 | 0.0375 | [0.020, 0.058] | 97.3 | 0.063 / 0.990 / 1.000 | **PROBE-PASS** |
| a20 | 0.20 | 0.0200 | [0.008, 0.035] | 92.9 | 0.063 / 0.998 / 1.000 | **PROBE-PASS** |

`family_verdict=PASS`, `smoke_recommend=a20` (pre-registered tie-break: lowest draw-safe floor).
All 400/arm distinct games (no byte-identical duplicates; effective-n honest).

**Mechanism read:** the FLOOR is the load-bearing draw knob — lower floor → sharper late
play → more decisive, *fewer* ply-cap draws (a20/a30 < control). a45's early τ=1.0 window
*with* a high floor lengthens games (110 plies) → more draws, but still well inside both
gates. The cosine→draw-collapse fear (§152) is a **training** dynamic, absent in static
generation — exactly the pre-registered hypothesis. **τ-fired confirmed**: opening diversity
(ply-3) lifts control 0.928 → schedule arms 0.988–0.998, and draw/length differ per arm —
the schedule demonstrably changed generation.

**N reduced 2000→400** (logged, not silent): the pre-registration justified N by *CI
resolution*; with control ≈0.048, N=400 resolves the 0.10 gate at >10σ. Gate thresholds
unchanged. Throughput on the 4080 was GPU-forward-bound (~8 games/min, ~88-ply games).

### Pre-registered caveat (verbatim)
> This reads the STATIC regime only. L9's collapse was a TRAINING dynamic (feedback through
> generations). Probe-PASS ≠ collapse-safe; it de-risks only the immediate generation-side
> effect. A "~0" here from the wrong regime is NOT evidence of absence. The Phase-3 training
> smoke carries its own in-run draw gate (recent draw ≥ 0.30 / 3-consec) for the dynamic regime.

## Phase 3 — training smoke (control vs a20, golong config, 10k steps each) — DONE

Continues golong@50k PEAK; per-arm buffer isolation; mixing 0.8/0.1; eval disabled (verdict
is training-metric based); in-run draw gate 0.30/3-consec. `value_accuracy_*` logging confirmed
live (§D-VALCEIL Q3). Sanity (30 steps) green: resume + corpus mixing + per-source value
accuracy all working.

**Pre-registered directional verdict (banked, to be filled):**
- **TEMP-POSITIVE:** selfplay `value_accuracy_masked` improves vs control at honest CI
  (≥ +0.02 or ≥1 CI-sep) AND draw gate never fired AND finishing not degraded → CANDIDATE
  for next recipe; re-confirm on the winning encoding before fold-in.
- **TEMP-NEGATIVE:** no label-noise improvement at honest CI → lever parked, mechanism note banked.
- **TEMP-TOXIC:** draw gate fired in TRAINING despite probe-pass → L9 generalizes to within-game
  decay; floor 0.5 → falsified-register hard-constraint candidate.

**Results** (both arms 10k steps from golong@50k; exit 0, no abort/crash; last-2k = the diverged regime):

| metric | control | a20 | Δ (a20−control), 95% CI | read |
|---|---|---|---|---|
| **`value_accuracy_corpus`** (shared data — CLEAN) | 0.7512 | 0.7501 | **−0.0010 [−0.0032, +0.0011]** | **FLAT** |
| `value_bce_corpus` (↓ better, shared) | 0.4616 | 0.4630 | +0.0015 [−0.0008, +0.0038] | **FLAT** |
| `value_accuracy_masked` (pre-reg; incl. selfplay) | 0.6986 | 0.7148 | +0.0162 [+0.0144, +0.0181] | **confound** |
| `value_accuracy_selfplay` (each arm's own games) | 0.5808 | 0.6036 | +0.023 | confound |
| draw_rate (training, full / early→late) | 0.065 / 0.068→0.063 | 0.103 / 0.096→0.106 | +0.04, rising | a20 elevated, sub-gate |

**Verdict: TEMP-NEGATIVE.** On the apples-to-apples shared-corpus metric the value head is **not** better calibrated (Δ≈0, CI straddles 0). The pre-registered `value_accuracy_masked` gain (+0.016) is **a confound, not a real effect**: it comes entirely from the `selfplay` portion (a20's sharper, more-decisive games are easier to label on a20's *own* data; the corpus subset, scored on identical positions, is flat). This metric choice was pre-committed before the run, not a post-hoc swap.

**Not TEMP-TOXIC:** the 0.30 draw gate never fired; no collapse. **But a cost surfaced:** a20 draws *more* than control in *training* (0.10 vs 0.065) — the **opposite** of its static-probe behavior (0.020 < 0.048) — and creeps upward (0.096→0.106) over 10k steps. This is the pre-registered training-vs-static distinction made concrete: floor 0.20 carries a mild, slowly-rising draw tax under training feedback (a weak echo of the §152 direction), without the hoped-for calibration benefit.

**Routing:** lever **PARKED**. The CEIL-HEADROOM ceiling is, per the dispatcher's hypothesis, **more likely loop-downstream** — within-game temperature does not move it. Falsified-register note: re-enabling the within-game cosine at floor 0.20 buys a draw tax (training) for no shared-data value-calibration gain; the clean-mechanism cleanup stands on its own merits regardless.

**Meta-lesson (banked):** the natural cross-arm value metric (`value_accuracy_selfplay` / `_masked`) is **confounded** when arms generate different self-play distributions — each arm is scored on its own (easier/harder) games. Use the **shared-corpus** subset (`value_accuracy_corpus`) for any cross-arm value comparison. Caught here only because the metric was pre-committed before the run.

### Transfer caveat (verbatim)
> Any TEMP-POSITIVE rides on the golong config + v6_live2 encoding. Per the standing rule,
> context transfer must be re-validated on the winning encoding before folding into the
> canonical recipe; fold-in is queued BEHIND the Arm-C re-run verdict.

## Cost actuals vs estimate
- Probe: ~3.5h gen + ~15min provision ≈ **$0.8** (est. $1–2). ✓ under.
- Smoke: ~10.3h (control 23:12→04:09, a20 04:09→09:28) × $0.21/hr ≈ **$2.2** (est. $10–12 — estimate assumed a pricier host). ✓ well under.
- **Total run (provision + probe + smoke) ≈ 14h ≈ $2.9.** Host idle now — safe to destroy.

## Routing
- Clean mechanism: **landed** (`711919d`), pushed.
- Probe: **PASS** — re-enabling the within-game cosine at floor 0.20–0.45 is draw-safe in
  generation; the family is not killed.
- Smoke verdict → TEMP-POSITIVE folds in behind the Arm-C re-run; NEGATIVE/TOXIC → register updates.
