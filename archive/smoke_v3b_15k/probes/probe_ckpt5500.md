# Threat-Logit Probe: checkpoint_00005500.pt

Verdict: **FAIL**

## Pass conditions (§85 / §89 revised §91 kill criterion)

| # | condition | threshold | value | result |
|---|-----------|-----------|-------|--------|
| 1 | contrast_mean (ext − ctrl) | ≥ max(0.38, 0.8 × -0.937) = +0.380 | +6.157 | **PASS** |
| 2 | ext cell in policy top-5  | ≥ 25% | 20% | **FAIL** |
| 3 | ext cell in policy top-10 | ≥ 40% | 20% | **FAIL** |

**C4 (warning, not gated):** |Δ ext_logit_mean| = |-1.044 − +0.217| = 1.262 (threshold 5.0) → **ok**

| metric | checkpoint_00005500.pt | bootstrap_model.pt |
|--------|------------|-------------|
| threat logit @ extension cell | -1.04 ± 1.17 | +0.22 ± 0.26 |
| threat logit @ control cell | -7.20 ± 0.25 | +1.15 ± 0.04 |
| contrast (extension − control) | +6.16 | -0.94 |
| extension cell in policy top-5 | 20% | 20% |
| extension cell in policy top-10 | 20% | 20% |

## Per-position detail

| # | phase | ext_logit | ctrl_logit | contrast | ext∈top5 | ext∈top10 |
|---|-------|-----------|------------|----------|----------|-----------|
| 1 | early | +1.591 | -7.283 | +8.874 | yes | yes |
| 2 | early | -1.559 | -7.083 | +5.523 | no | no |
| 3 | early | +0.498 | -6.616 | +7.114 | no | no |
| 4 | early | +0.470 | -7.296 | +7.766 | no | no |
| 5 | early | -1.920 | -7.254 | +5.335 | no | no |
| 6 | early | +0.032 | -7.351 | +7.383 | no | no |
| 7 | early | -2.483 | -6.688 | +4.206 | no | no |
| 8 | early | -2.258 | -6.861 | +4.603 | no | no |
| 9 | early | -0.568 | -7.410 | +6.842 | yes | yes |
| 10 | early | -1.585 | -7.228 | +5.643 | no | no |
| 11 | early | -1.902 | -7.308 | +5.406 | no | no |
| 12 | early | -0.176 | -7.288 | +7.112 | yes | yes |
| 13 | early | -2.467 | -7.351 | +4.885 | no | no |
| 14 | early | +0.206 | -7.233 | +7.439 | no | no |
| 15 | early | -0.015 | -7.475 | +7.460 | yes | yes |
| 16 | early | -2.591 | -7.299 | +4.708 | no | no |
| 17 | early | -1.349 | -6.782 | +5.433 | no | no |
| 18 | early | -1.458 | -7.393 | +5.934 | no | no |
| 19 | early | -1.772 | -7.401 | +5.629 | no | no |
| 20 | early | -1.581 | -7.424 | +5.843 | no | no |

*Positions: 20  Seed: 42  Baseline: threat_probe_baseline.json*