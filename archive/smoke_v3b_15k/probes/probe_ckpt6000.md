# Threat-Logit Probe: checkpoint_00006000.pt

Verdict: **FAIL**

## Pass conditions (§85 / §89 revised §91 kill criterion)

| # | condition | threshold | value | result |
|---|-----------|-----------|-------|--------|
| 1 | contrast_mean (ext − ctrl) | ≥ max(0.38, 0.8 × -0.937) = +0.380 | +4.517 | **PASS** |
| 2 | ext cell in policy top-5  | ≥ 25% | 20% | **FAIL** |
| 3 | ext cell in policy top-10 | ≥ 40% | 25% | **FAIL** |

**C4 (warning, not gated):** |Δ ext_logit_mean| = |-0.897 − +0.217| = 1.114 (threshold 5.0) → **ok**

| metric | checkpoint_00006000.pt | bootstrap_model.pt |
|--------|------------|-------------|
| threat logit @ extension cell | -0.90 ± 0.86 | +0.22 ± 0.26 |
| threat logit @ control cell | -5.41 ± 0.26 | +1.15 ± 0.04 |
| contrast (extension − control) | +4.52 | -0.94 |
| extension cell in policy top-5 | 20% | 20% |
| extension cell in policy top-10 | 25% | 20% |

## Per-position detail

| # | phase | ext_logit | ctrl_logit | contrast | ext∈top5 | ext∈top10 |
|---|-------|-----------|------------|----------|----------|-----------|
| 1 | early | +0.941 | -5.176 | +6.117 | yes | yes |
| 2 | early | -0.999 | -5.432 | +4.433 | no | no |
| 3 | early | +0.342 | -5.363 | +5.705 | no | no |
| 4 | early | -0.051 | -5.151 | +5.100 | no | no |
| 5 | early | -1.916 | -5.203 | +3.287 | no | no |
| 6 | early | -0.261 | -5.275 | +5.015 | no | no |
| 7 | early | -1.570 | -5.345 | +3.775 | no | no |
| 8 | early | -1.366 | -5.339 | +3.974 | no | no |
| 9 | early | -0.277 | -5.827 | +5.550 | yes | yes |
| 10 | early | -1.495 | -5.149 | +3.654 | no | yes |
| 11 | early | -1.010 | -5.261 | +4.251 | no | no |
| 12 | early | -1.638 | -5.197 | +3.559 | yes | yes |
| 13 | early | -2.020 | -5.275 | +3.256 | no | no |
| 14 | early | -0.108 | -5.133 | +5.025 | no | no |
| 15 | early | +0.442 | -5.857 | +6.298 | yes | yes |
| 16 | early | -2.113 | -5.429 | +3.316 | no | no |
| 17 | early | -1.448 | -5.351 | +3.903 | no | no |
| 18 | early | -1.023 | -5.743 | +4.720 | no | no |
| 19 | early | -1.250 | -5.904 | +4.654 | no | no |
| 20 | early | -1.112 | -5.866 | +4.754 | no | no |

*Positions: 20  Seed: 42  Baseline: threat_probe_baseline.json*