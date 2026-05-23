# §S181-AUDIT Wave 1 — Track B / B1 — per-source gradient attribution

Source log: `s181_b_20260523_1638.log` (3000 per_source_grad_norm events).

## Decision-relevant windows (per V-B aggregation L13 guard)

| window | n_steps | mean share pretrain | mean share recent | mean share uniform_self | max single-source share |
|---|---:|---:|---:|---:|---:|
| steps_0_500 | 500 | 0.098 | 0.342 | 0.560 | 0.858 |
| steps_500_1000 | 501 | 0.095 | 0.345 | 0.561 | 0.848 |
| steps_500_2000 | 1501 | 0.092 | 0.345 | 0.563 | 0.848 |
| steps_1000_3000 | 2001 | 0.090 | 0.342 | 0.568 | 0.845 |
| all | 3000 | 0.092 | 0.342 | 0.566 | 0.858 |

## Per-group share within decision window (mean steps 500-2000)

| group | pretrain | recent | uniform_self |
|---|---:|---:|---:|
| trunk | 0.115 | 0.332 | 0.554 |
| value | 0.184 | 0.313 | 0.502 |
| policy | 0.077 | 0.354 | 0.568 |

## Checkpoint snapshots

| ckpt step | actual | shares pretrain | recent | uniform_self | total pull |
|---:|---:|---:|---:|---:|---:|
| 500 | 500 | 0.094 | 0.381 | 0.525 | 31.8024 |
| 1000 | 1000 | 0.060 | 0.274 | 0.666 | 30.1336 |
| 1500 | 1500 | 0.106 | 0.353 | 0.540 | 19.7623 |
| 2000 | 2000 | 0.126 | 0.411 | 0.463 | 22.6084 |
| 2500 | 2500 | 0.045 | 0.139 | 0.816 | 47.8025 |
| 3000 | 3000 | 0.082 | 0.343 | 0.575 | 26.7997 |

## V-B-A discrimination guard

Routing per `audit/structural/REAL_RUN_RECIPE.md` §3 + `B_launch_and_analysis_spec.md` §Aggregation:

- V-B-A if any source share ≥ 60% across steps 500-2000
- V-B-B if all three sources land in 25-45% across steps 500-2000

- max source share over window: **0.563**
- min source share over window: **0.092**
- V-B-A literal trigger (≥0.60): **NO**
- V-B-B literal trigger (25-45 band): **NO**
