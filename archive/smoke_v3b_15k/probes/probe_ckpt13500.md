# Threat-Logit Probe: checkpoint_00013500.pt

Verdict: **FAIL**

## Pass conditions (§85 / §89 revised §91 kill criterion)

| # | condition | threshold | value | result |
|---|-----------|-----------|-------|--------|
| 1 | contrast_mean (ext − ctrl) | ≥ max(0.38, 0.8 × -0.937) = +0.380 | +4.418 | **PASS** |
| 2 | ext cell in policy top-5  | ≥ 25% | 20% | **FAIL** |
| 3 | ext cell in policy top-10 | ≥ 40% | 25% | **FAIL** |

**C4 (warning, not gated):** |Δ ext_logit_mean| = |-1.003 − +0.217| = 1.220 (threshold 5.0) → **ok**

| metric | checkpoint_00013500.pt | bootstrap_model.pt |
|--------|------------|-------------|
| threat logit @ extension cell | -1.00 ± 0.85 | +0.22 ± 0.26 |
| threat logit @ control cell | -5.42 ± 0.22 | +1.15 ± 0.04 |
| contrast (extension − control) | +4.42 | -0.94 |
| extension cell in policy top-5 | 20% | 20% |
| extension cell in policy top-10 | 25% | 20% |

## Per-position detail

| # | phase | ext_logit | ctrl_logit | contrast | ext∈top5 | ext∈top10 |
|---|-------|-----------|------------|----------|----------|-----------|
| 1 | early | +0.046 | -5.275 | +5.321 | yes | yes |
| 2 | early | -1.710 | -5.611 | +3.901 | no | no |
| 3 | early | -0.795 | -5.805 | +5.011 | no | no |
| 4 | early | -0.833 | -5.227 | +4.394 | no | no |
| 5 | early | -2.256 | -5.260 | +3.003 | no | no |
| 6 | early | -0.801 | -5.176 | +4.376 | no | no |
| 7 | early | -0.810 | -5.718 | +4.908 | no | no |
| 8 | early | -0.767 | -5.697 | +4.930 | no | no |
| 9 | early | -0.368 | -5.550 | +5.182 | yes | yes |
| 10 | early | -1.905 | -5.180 | +3.275 | no | yes |
| 11 | early | -1.706 | -5.176 | +3.470 | no | no |
| 12 | early | -0.877 | -5.228 | +4.351 | yes | yes |
| 13 | early | -2.676 | -5.176 | +2.500 | no | no |
| 14 | early | -1.082 | -5.181 | +4.099 | no | no |
| 15 | early | +0.422 | -5.556 | +5.978 | yes | yes |
| 16 | early | -2.554 | -5.569 | +3.015 | no | no |
| 17 | early | -0.424 | -5.725 | +5.301 | no | no |
| 18 | early | +0.160 | -5.247 | +5.408 | no | no |
| 19 | early | -0.506 | -5.500 | +4.994 | no | no |
| 20 | early | -0.607 | -5.545 | +4.938 | no | no |

*Positions: 20  Seed: 42  Baseline: threat_probe_baseline.json*