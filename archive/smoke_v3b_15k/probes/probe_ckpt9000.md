# Threat-Logit Probe: checkpoint_00009000.pt

Verdict: **FAIL**

## Pass conditions (§85 / §89 revised §91 kill criterion)

| # | condition | threshold | value | result |
|---|-----------|-----------|-------|--------|
| 1 | contrast_mean (ext − ctrl) | ≥ max(0.38, 0.8 × -0.937) = +0.380 | +6.184 | **PASS** |
| 2 | ext cell in policy top-5  | ≥ 25% | 20% | **FAIL** |
| 3 | ext cell in policy top-10 | ≥ 40% | 20% | **FAIL** |

**C4 (warning, not gated):** |Δ ext_logit_mean| = |-0.713 − +0.217| = 0.930 (threshold 5.0) → **ok**

| metric | checkpoint_00009000.pt | bootstrap_model.pt |
|--------|------------|-------------|
| threat logit @ extension cell | -0.71 ± 0.88 | +0.22 ± 0.26 |
| threat logit @ control cell | -6.90 ± 0.25 | +1.15 ± 0.04 |
| contrast (extension − control) | +6.18 | -0.94 |
| extension cell in policy top-5 | 20% | 20% |
| extension cell in policy top-10 | 20% | 20% |

## Per-position detail

| # | phase | ext_logit | ctrl_logit | contrast | ext∈top5 | ext∈top10 |
|---|-------|-----------|------------|----------|----------|-----------|
| 1 | early | +0.351 | -6.704 | +7.055 | yes | yes |
| 2 | early | -1.672 | -7.019 | +5.347 | no | no |
| 3 | early | +1.127 | -6.744 | +7.871 | no | no |
| 4 | early | +0.109 | -6.708 | +6.817 | no | no |
| 5 | early | -1.779 | -6.613 | +4.834 | no | no |
| 6 | early | -0.364 | -6.731 | +6.367 | no | no |
| 7 | early | -0.875 | -6.705 | +5.829 | no | no |
| 8 | early | -0.645 | -7.170 | +6.526 | no | no |
| 9 | early | +0.236 | -7.441 | +7.677 | yes | yes |
| 10 | early | -1.436 | -6.559 | +5.123 | no | no |
| 11 | early | -0.935 | -6.652 | +5.717 | no | no |
| 12 | early | -0.926 | -6.826 | +5.900 | yes | yes |
| 13 | early | -2.251 | -6.731 | +4.480 | no | no |
| 14 | early | -0.203 | -6.889 | +6.686 | no | no |
| 15 | early | +0.451 | -7.212 | +7.663 | yes | yes |
| 16 | early | -2.235 | -6.942 | +4.707 | no | no |
| 17 | early | -0.508 | -6.709 | +6.200 | no | no |
| 18 | early | -0.569 | -7.032 | +6.463 | no | no |
| 19 | early | -1.099 | -7.301 | +6.201 | no | no |
| 20 | early | -1.028 | -7.242 | +6.214 | no | no |

*Positions: 20  Seed: 42  Baseline: threat_probe_baseline.json*