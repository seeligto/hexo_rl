<!-- Forensic archive extracted from docs/07_PHASE4_SPRINT_LOG.md during compression sprint 2026-05-13. Compressed counterpart in docs/07_PHASE4_SPRINT_LOG.md. -->

## §106 — Q27 Probe 1b: C2/C3 failure was fixture artifact (2026-04-19)

**Setup.** Probe 1 (synthetic ply=7 fixture, N=20) reported 0/20
extensions outside ZOI and bootstrap C2/C3 at 20%/20%, with the
load-bearing caveat that the fixture could not exercise §77's
truncation failure modes (ply > `zoi_lookback=16`, disjoint-cluster
threats). Probe 1b regenerated `fixtures/threat_probe_positions.npz`
from real mid/late positions sampled from
`runs/10cc8d56e4394a9ca542740c4bcee069` (500-game self-play, median
ply 169), with per-phase quotas early=7 / mid=7 / late=6. Ply span
9 → 150. Schema unchanged. Full report:
`reports/q27_zoi_reachability_realpositions_2026-04-19.md`.

**Baseline shift on `bootstrap_model.pt`.**

| metric | v5 synthetic early | v6 real mid/late |
|---|---:|---:|
| ext_logit_mean   |  0.080 |  0.015 |
| ext_logit_std    |  0.093 |  0.399 |
| ctrl_logit_mean  |  0.028 |  0.061 |
| ctrl_logit_std   |  0.012 |  0.030 |
| contrast_mean    | +0.052 | −0.046 |
| contrast_std     |  0.097 |  0.396 |
| ext_in_top5_pct  | 20.0%  | **60.0%** |
| ext_in_top10_pct | 20.0%  | **65.0%** |

The same `bootstrap_model.pt` jumps C2 20% → 60% and C3 20% → 65% on
the real fixture. The synthetic early-phase 3-in-a-row-with-far-stones
configuration was distributionally out-of-sample for the trained
policy (does not occur in real self-play at ply=7), so the probe was
asking the model to rank threats in a geometric configuration it had
never been trained to recognise. Baseline file bumped v5 → v6
(`BASELINE_SCHEMA_VERSION` in `scripts/probe_threat_logits.py:78`);
prior synthetic baseline preserved as
`fixtures/threat_probe_baseline_v5_synthetic.json.bak` (not committed).

**5K post-W1 checkpoint re-probe.** `checkpoint_00005000.pt` from the
run that ended 2026-04-18 22:38, re-probed against v6:

```
PASS  [C1] contrast=+3.317 (≥+0.380) OK
      [C2] top5=50% (≥25%) OK
      [C3] top10=65% (≥40%) OK
[C4] |Δ ext_logit_mean|=0.420 (<5.0) ok
```

All three gates PASS with margin — C1 contrast +3.317 sits ~9× above
the 0.38 floor. This is the inverse of the FAIL verdict recorded in
`reports/probes/latest_20260418_223903.md` against the v5 fixture.

**Supersedes §105's verdict.** §105 concluded "W1 necessary, not
sufficient" on the basis of identical 20%/20% C2/C3 in both arms of
the two-machine smoke. That conclusion was downstream of the v5
synthetic fixture. On the real-position v6 fixture the post-W1 5K
checkpoint PASSES all gates. The corrected framing:

> **W1 correctness fix lands clean; the apparent C2/C3 symptom was a
> fixture artifact.**

The correctness argument for W1 (three call sites inverted training
targets at ~50% of move steps) is independent of this and unchanged.

**§77 truncation failure mode — 1/20 instance.** Probe 1b found a
single position (late, ply=91, cluster center (37, 5), extension
(32, −1)) where the extension cell sits at `ext_d_zoi = 11` — outside
the live ZOI mask. Stones within hex-distance 3 of the extension are
all placed before the lookback window cut-in; the last 16 moves are
scattered across remote disjoint colonies. Concrete instance of §77's
disjoint-cluster prediction, but at 1/20 it cannot carry a
population-level C2/C3 miss. Kept as a note; not a blocker. Fix (raise
`zoi_lookback` or make ZOI colony-aware rather than recency-based) is
Phase 4.5+ if late-game disjoint-cluster failures surface in sustained
training.

**Q27 status.** Remains OPEN but reframed — no active root-cause
probe. Probes 2 (threat-weight sweep) and 3 (value-aggregation
ablation) shelved pending post-5K evidence of actual training-trajectory
regression. Next evidence point: sustained training smoke from
`bootstrap_model.pt`. Reopen if C2/C3 regress on the real-fixture
probe after 5K.

**C1 contrast flipped negative on bootstrap against real fixture.**
`ctrl_logit_mean` (0.061) > `ext_logit_mean` (0.015) on the v6 fixture
— the scalar threat head fires *more* on an empty far cell than on the
extension, yet the policy ranking still routes 60%/65% of extensions
into top-5 / top-10. Threat-scalar magnitude and policy-ranking
signals are decoupled on bootstrap. Not a bug; an unexplained
observation. Filed as **Q32** in `docs/06_OPEN_QUESTIONS.md` (WATCH
priority, threat-scalar magnitude vs policy ranking decoupling).

**Files touched in this cleanup pass.**

- `scripts/generate_threat_probe_fixtures.py` — `--n-per-phase` flag,
  compound_move phase thresholds, strict quota enforcement.
- `scripts/probe_threat_logits.py` — `BASELINE_SCHEMA_VERSION` 5 → 6.
- `fixtures/threat_probe_positions.npz` — regenerated from real run
  (7 early / 7 mid / 6 late).
- `fixtures/threat_probe_baseline.json` — v6 baseline committed in
  c5bce9c.
- `fixtures/threat_probe_baseline.CHANGELOG.md` — seeded (v2 → v6
  history).
- `docs/07_PHASE4_SPRINT_LOG.md` — §105 postscript + this entry.
- `docs/06_OPEN_QUESTIONS.md` — Q27 reframe, Q32 added.
- `reports/q27_perspective_flip_smoke_2026-04-18/verdict.md` —
  superseded banner pointing to Probe 1b report.
- `reports/q27_zoi_reachability_2026-04-19.md` — superseded banner
  pointing to real-fixture report.

**Resolves.** Nothing. **Reframes.** Q27 (no longer "attention
hijacking persists — root cause unknown"; now "reframed, no active
C2/C3 regression"). **Opens.** Q32 (threat-scalar vs policy-ranking
decoupling, WATCH).

### Commits

- `docs(sprint): §106 Q27 Probe 1b inverted verdict — fixture artifact`

---

