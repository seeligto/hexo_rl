# T1 — v7full training-mode knob isolation

**Date:** 2026-05-05
**Bootstrap:** v7full (`bootstrap_model_v7full.pt`)
**Sims:** per-variant (see knobs col)  **max_plies:** 150
**Draw reward:** -0.5

## Per-variant aggregate

| Variant | n | draws | draw_rate (95% CI) | mean_ply | stride5 P50/P90 | rmax P50/P90 | colony_wins | wall |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| R11 | 20 | 17 | 85.0% [64.0%, 94.8%] | 132 | 79 / 84 | 97 / 107 | 1 | 307s |
| R12 | 20 | 0 | 0.0% [0.0%, 16.1%] | 50 | 2 / 3 | 9 / 12 | 10 | 101s |
| R13 | 20 | 18 | 90.0% [69.9%, 97.2%] | 143 | 81 / 97 | 100 / 111 | 2 | 440s |
| R14 | 20 | 19 | 95.0% [76.4%, 99.1%] | 143 | 130 / 133 | 130 / 136 | 0 | 575s |

## Terminal reason breakdown

| Variant | six_in_a_row | colony | ply_cap | other_draw |
|---|---:|---:|---:|---:|
| R11 | 3 | 0 | 17 | 0 |
| R12 | 20 | 0 | 0 | 0 |
| R13 | 2 | 0 | 18 | 0 |
| R14 | 1 | 0 | 19 | 0 |

## Verdict

**Proximate cause: R11** — first variant to cross draw_rate ≥ 50%.

The knob(s) added by R11 (vs the next-lower variant in the chain) are the proximate cause of the smoke v6 step-0 draw explosion under frozen v7full weights.

Next step: implement the structural fix that suppresses or counter-balances this knob's effect on the stride-5 fixed point.  Update `configs/variants/w4c_smoke_v7_5080.yaml` to reflect the fix and re-run a 1k-step laptop smoke before any 5080 sustained launch.
