# threat_probe_baseline — version history

Tracks `BASELINE_SCHEMA_VERSION` in `scripts/probe_threat_logits.py`
and the paired `fixtures/threat_probe_baseline.json`. One line per
bump: version, date, cause, sprint-log anchor.

| ver | date       | cause                                                                                                     | anchor |
|----:|------------|-----------------------------------------------------------------------------------------------------------|--------|
|  v2 | 2026-04-14 | §91 revision — dropped `ext_logit_mean ≥ baseline − 1.0` for contrast-floor + top-5 + top-10 + C4 warning. Added `version`, `ext_in_top10_frac`, explicit `*_pct` fields. Regenerated from `bootstrap_model.pt` (18-plane era). | §91    |
|  v3 | 2026-04-14 | §92 landing — 24-plane break (chain planes added as input). Regenerated against 24-plane bootstrap. Contrast −0.9366 (untrained head). Broken by pretrain F1 aug bug, superseded by v4. | §92    |
|  v4 | 2026-04-15 | §93 pretrain v3b — F1 aug fix (Rust kernel, axis_perm remap). Regenerated against v3b 24-plane bootstrap. | §93    |
|  v5 | 2026-04-17 | §100.d — post-§99 GroupNorm migration. Older v4 anchored to a stale 24-plane `bootstrap_model.pt`. NPZ sliced 24→18 planes in place. Regenerated against live 18-plane GroupNorm bootstrap. Absolute 0.38 contrast floor binds. | §100.d |
|  v6 | 2026-04-19 | §106 (Probe 1b) — fixture regenerated from real mid/late self-play positions (ply span 9–150, early=7/mid=7/late=6). Replaces synthetic ply=7 construction. Baseline C2/C3 jumps 20%/20% → 60%/65% on same `bootstrap_model.pt`; `contrast_mean` flips −0.046 (see Q32). Prior v5 preserved at `threat_probe_baseline_v5_synthetic.json.bak` (not committed). | §106   |

**Re-bump discipline.** Any change to the NPZ fixture, the bootstrap
weights, or the trunk input layout requires a new baseline and a line
here. `make probe.bootstrap` rewrites the JSON; the NPZ is regenerated
by `scripts/generate_threat_probe_fixtures.py`.
