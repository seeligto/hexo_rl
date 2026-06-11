# scripts/diagnosis/ — Instrument Lifecycle Ledger

This directory is the sole home for ALL diagnosis/investigation instruments, active and closed.
Instruments are TRACKED while their arc is ACTIVE; demoted (git rm --cached, local copy preserved per D3) on arc close.

## Package note
scripts/diagnosis/ is a REGULAR Python package (has __init__.py). Import as:
  from scripts.diagnosis.track_a.position_classifier import classify_batch

## ACTIVE arcs

| Instrument(s) | Arc | Sprint log |
|---|---|---|
| coherence_decomposition.py, coherence_inwindow_policy.py, coherence_overspread.py, globalconc_probe.py | §D-GLOBALCONC (H1/H2 discriminator; §D-MULTICLUSTER OPEN) | 5132-5220 |
| track_a/ (position_classifier.py, _validate_classifier.py, a1-a5_h_*.py, a2_h_aug.py) | §D-GLOBALCONC Track A; position_classifier is PRODUCTION dependency | 5132-5220 |
| determinism_audit.py | §D-RECONVERGE Phase-1b instrument | 5245 |
| turn_wins.py | re-export shim (PRODUCTION dependency from forced_win_detector) | — |
| new_probes.py, offwindow_placement_lift.py | §D-RECONVERGE/§D-EXTLINK→§D-MULTICLUSTER | 5222-5383 |
| finishing_sims_sweep.py | §D-EXPLOIT | 5383+ |
| value_calibration_ladder.py | §D-VALPROBE — fixture-agnostic value-calibration ladder (corpus/selfplay; E1-E2 + G1/G3 verdicts; kernels tested in tests/test_value_calibration_metrics.py) | — |

## CLOSED arcs (instruments retained for reference, no longer deployed)

| Instrument(s) | Arc | Closed | Citation |
|---|---|---|---|
| wallcausation_*.{py,sh} (5 files) | §D-WALLCAUSATION | 2026-06-05 | sprint log 4829-4882 |
| fragility_value_discrim.py, hplane_activation_dump.py | §D-FRAGILITY | 2026-06-07 | sprint log 4927-4981 |
| overspread_d*.py, overspread_*.py (12 files) | §D-OVERSPREAD | 2026-06-08 | sprint log 5050-5130 |
| prelong_centering_oracle.py, prelong_triage_probe.py | §PRELONG-2A FALLBACK | 2026-06-04 | prelong_2a_handoff |
| pathstrength_probe.py | §D-PATHSTRENGTH (superseded by §D-RECONVERGE LIFT) | 2026-06-08 | sprint log 5222-5383 |
| track_b/ (b1_grad_attribution_analysis.py, b2_buffer_composition_analysis.py, trunk_feature_drift.py) | §S181-AUDIT Wave 1 | 2026-05-23 | sprint log 3067-3143 |
| track_c/c_lite_1_v7full_reference.py | §S181-AUDIT Wave 1 | 2026-05-23 | sprint log 3067-3143 |
| determinism_audit.py (§S181 leg), export_value_spread_bank.py, fu1_value_spread_ladder.py, mcts_colony_probe.py, probe_architecture.py, probe_value_bias.py | §S181-AUDIT Wave 1 | 2026-05-23 | sprint log 3067-3143 |

## Untracked local scratch (ignored, on-disk only — D-REPOSTRUCT-EXEC R3, 2026-06-11)

depth_bisect_probe.py, game_bisect_probe.py (engine-bisect one-shots, no arc claim),
prelong_triage_analyze.py, sealbot_selfplay_t1.py (§PRELONG closed 2026-06-04 FALLBACK).
Ignored by explicit .gitignore rules; local copies preserved per D3.

## Demote protocol (D2)
1. Arc closes: record close date + sprint-log citation here.
2. `git rm --cached scripts/diagnosis/<instrument>.py` (local copy preserved per D3).
3. Update DISPATCH dispatcher doc or docs/handoffs/ entry if referenced.
4. If instrument is PRODUCTION imported, coordinate with trainer/eval teams before demotion.
