<!-- Forensic archive extracted from docs/07_PHASE4_SPRINT_LOG.md during compression sprint 2026-05-13. Compressed counterpart in docs/07_PHASE4_SPRINT_LOG.md. -->

## §89 — Threat-logit probe committed as step-5k kill criterion (2026-04-13, corrected §90, REVISED §91)

**What.** Two scripts + one test module committed to make the 20-position threat-logit
probe reproducible as a formal gate for every future sustained run.

### Files added / updated

```
scripts/probe_threat_logits.py          — CLI + importable probe functions
scripts/generate_threat_probe_fixtures.py — generate fixtures/threat_probe_positions.npz
tests/test_probe_threat_logits.py       — shape/dtype/determinism/pass-logic tests
fixtures/threat_probe_positions.npz     — 20 curated positions (generated on first run)
fixtures/threat_probe_baseline.json     — canonical baseline (written by make probe.bootstrap)
```

### Kill criterion (REVISED §91 — see that section for full rationale)

At training step **5000**, run `make probe.latest`. PASS requires **all of C1-C3**.
C4 is a warning only and never causes FAIL.

| # | condition | threshold |
|---|-----------|-----------|
| 1 | contrast_mean (ext − ctrl) | ≥ max(0.38, 0.8 × bootstrap_contrast) |
| 2 | extension cell in policy top-5 | ≥ 40% |
| 3 | extension cell in policy top-10 | ≥ 60% |
| 4 (warn) | abs(ext_logit_mean − bootstrap_ext_logit_mean) < 5.0 | warning only |

**Original criterion history.** §85 first draft: "logit > 0" — wrong because
bootstrap itself measures around −0.34 to −0.60. §85/§89 correction: replaced
with `ext_logit_mean ≥ baseline − 1.0`, designed to catch ckpt_19500's
absolute-magnitude collapse (−0.14 → −3.25). §91 revision: that criterion
incorrectly FAILed ckpt_00014344, which has IMPROVED position-conditional
sharpness (contrast +3.94, top-10 70%) but a global bias shift in the threat
head (ext_logit_mean −6.21). Old C1 was a BCE scale-drift detector dressed up
as a colony-spam detector. It is replaced by direct colony-spam tests on the
policy head (C2 top-5 + C3 top-10); the bias-shift signal is preserved as
warning-only C4. See §91 for the full diagnosis and decision trail.

The canonical baseline numbers live in `fixtures/threat_probe_baseline.json`, written
once by `make probe.bootstrap` (which passes `--write-baseline` to the script).
If that file is absent, `probe.latest` prints FAIL with
"no baseline recorded — run make probe.bootstrap first".

### Bootstrap baseline (§85 empirical, bootstrap_model.pt)

| metric | bootstrap_model.pt | ckpt_19500 (pre-A1, bad) |
|--------|-------------------|--------------------------|
| threat logit @ extension cell | −0.14 ± 0.74 | −3.25 ± 0.46 |
| threat logit @ control cell | −0.52 ± 0.39 | −5.11 ± 1.40 |
| contrast (extension − control) | **+0.38** | +1.86 (shortcut) |

ckpt_19500 contrast was *higher* than bootstrap — the head learned a marginal-class
shortcut against stale mis-aligned labels. The +0.38 bootstrap contrast is the floor.

### Determinism

Probe forces FP32 (no autocast) and sets `torch.manual_seed(42)` +
`torch.use_deterministic_algorithms(True)` at startup. Two consecutive
`make probe.bootstrap` runs must produce byte-identical ext_logit_mean.

### Fixture schema (fixtures/threat_probe_positions.npz)

```
states:           (20, 18, 19, 19) float16 — K=0 cluster window tensors
side_to_move:     (20,) int8              — 1=P1, -1=P2 (current player)
ext_cell_idx:     (20,) int32             — flat index [0, 361) of open extension cell
control_cell_idx: (20,) int32             — flat index of empty cell far from stones
game_phase:       (20,) U8 string         — "early" / "mid" / "late"
```

Cell indices are loaded verbatim from NPZ — never regenerated at load time.

To regenerate from game records: `make probe.fixtures`
To regenerate synthetically (no game records required): see script `--synthetic` flag.

### Makefile targets

| target | action |
|--------|--------|
| `make probe.bootstrap` | Probe bootstrap_model.pt; write `fixtures/threat_probe_baseline.json` + `reports/probes/bootstrap_<ts>.md` |
| `make probe.latest` | Probe latest checkpoint; three-condition PASS/FAIL against saved baseline |
| `make probe.fixtures` | Regenerate fixture NPZ from available game records |

### Exit codes for probe_threat_logits.py

| code | meaning |
|------|---------|
| 0 | PASS — both thresholds met |
| 1 | FAIL — at least one threshold missed |
| 2 | Error — checkpoint load failed, shape mismatch, missing file |

**Commit:** `feat(eval): commit threat-logit probe as step-5k kill criterion`

---

