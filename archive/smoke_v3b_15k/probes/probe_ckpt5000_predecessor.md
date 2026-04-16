# Threat-Logit Probe: checkpoint_00005000.pt

Verdict: **FAIL**

## Pass conditions (§85 / §89 revised §91 kill criterion)

| # | condition | threshold | value | result |
|---|-----------|-----------|-------|--------|
| 1 | contrast_mean (ext − ctrl) | ≥ max(0.38, 0.8 × -0.937) = +0.380 | +5.055 | **PASS** |
| 2 | ext cell in policy top-5  | ≥ 40% | 20% | **FAIL** |
| 3 | ext cell in policy top-10 | ≥ 60% | 35% | **FAIL** |

**C4 (warning, not gated):** |Δ ext_logit_mean| = |-1.593 − +0.217| = 1.810 (threshold 5.0) → **ok**

| metric | checkpoint_00005000.pt | bootstrap_model.pt |
|--------|------------|-------------|
| threat logit @ extension cell | -1.59 ± 0.94 | +0.22 ± 0.26 |
| threat logit @ control cell | -6.65 ± 0.19 | +1.15 ± 0.04 |
| contrast (extension − control) | +5.06 | -0.94 |
| extension cell in policy top-5 | 20% | 20% |
| extension cell in policy top-10 | 35% | 20% |

## Per-position detail

| # | phase | ext_logit | ctrl_logit | contrast | ext∈top5 | ext∈top10 |
|---|-------|-----------|------------|----------|----------|-----------|
| 1 | early | +0.390 | -6.392 | +6.782 | yes | yes |
| 2 | early | -3.320 | -6.814 | +3.494 | no | no |
| 3 | early | -1.088 | -6.703 | +5.615 | no | no |
| 4 | early | -0.405 | -6.332 | +5.927 | no | no |
| 5 | early | -1.978 | -6.418 | +4.440 | no | yes |
| 6 | early | -0.992 | -6.836 | +5.844 | no | no |
| 7 | early | -2.101 | -6.720 | +4.619 | no | no |
| 8 | early | -2.368 | -6.865 | +4.497 | no | no |
| 9 | early | -0.568 | -6.786 | +6.218 | yes | yes |
| 10 | early | -2.180 | -6.531 | +4.350 | no | yes |
| 11 | early | -1.723 | -6.851 | +5.127 | no | no |
| 12 | early | -1.329 | -6.391 | +5.062 | yes | yes |
| 13 | early | -2.984 | -6.836 | +3.853 | no | no |
| 14 | early | -0.455 | -6.213 | +5.757 | no | yes |
| 15 | early | -0.711 | -6.694 | +5.983 | yes | yes |
| 16 | early | -2.967 | -6.798 | +3.831 | no | no |
| 17 | early | -2.223 | -6.729 | +4.506 | no | no |
| 18 | early | -1.607 | -6.661 | +5.053 | no | no |
| 19 | early | -1.590 | -6.707 | +5.117 | no | no |
| 20 | early | -1.659 | -6.684 | +5.025 | no | no |

*Positions: 20  Seed: 42  Baseline: threat_probe_baseline.json*