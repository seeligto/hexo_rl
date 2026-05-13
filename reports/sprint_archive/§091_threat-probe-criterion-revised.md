<!-- Forensic archive extracted from docs/07_PHASE4_SPRINT_LOG.md during compression sprint 2026-05-13. Compressed counterpart in docs/07_PHASE4_SPRINT_LOG.md. -->

## §91 — Threat-probe criterion revised: target colony-spam, not BCE drift (2026-04-14)

Replaces §85/§89 step-5k probe C1 (`ext_logit_mean ≥ baseline − 1.0`) with a contrast-floor + top-N pair targeting policy-head colony-spam directly. Old C1 was a scale-drift detector misfiring on healthy runs (ckpt_00014344: contrast grew 10× while abs logits drifted globally negative — opposite of the ckpt_19500 marginal-class collapse the old C1 targeted).

**Locked criterion (enforced by `scripts/probe_threat_logits.py`):**

| # | condition | threshold | why |
|---|-----------|-----------|-----|
| C1 | `contrast_mean ≥ max(0.38, 0.8 × bootstrap_contrast)` | floor 0.40 (bootstrap=0.502) | direct contrast-floor; scales with bootstrap, preserves §85 absolute floor |
| C2 | `ext_in_top5_pct ≥ 40` | unchanged | direct colony-spam test on policy head |
| C3 | `ext_in_top10_pct ≥ 60` | NEW | catches partial sharpness — rank 6-10 fine |
| C4 | `abs(ext_logit_mean − bootstrap_ext_logit_mean) < 5.0` | warning only | catastrophic decode/mapping canary; never gates |

C1–C3 must all PASS for `make probe.latest` exit 0. C4 is warning-only (BCE-drift / Q19 monitoring hook).

**Baseline.** `fixtures/threat_probe_baseline.json` bumped to v2 (adds `ext_in_top10_pct`); regenerated from `bootstrap_model.pt`: contrast 0.502, top5 50%, top10 65%.

**Q19 [WATCH]** logged in `docs/06_OPEN_QUESTIONS.md` — `BCEWithLogitsLoss` at ~1.6% positive labels drives logits globally negative; proposed `pos_weight ≈ 59` on next bootstrap-from-scratch. C4 is the monitoring hook; escalate WATCH→HIGH if drift > 8 nats or aux loss > 4.0.

**Replaces:** §85 (original C1 floor), §89 (C1 retune). Forward: §92 lands Q19 `pos_weight=59` atomically with chain-plane bootstrap.

**Commits:**
- `fix(eval): revise threat-probe criterion to target colony-spam directly`
- `fix(monitoring): swallow engineio disconnect KeyError in web dashboard` (orthogonal: `threading.excepthook` filter for engineio `KeyError('Session is disconnected')` from tab-close races).

---

