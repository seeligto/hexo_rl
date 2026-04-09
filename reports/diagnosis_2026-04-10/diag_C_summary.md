# Diagnostic C.2 — per-move entropy from the training trace

Records parsed: 30 (from `diag_A_trace_training.jsonl`, cap 30 in game_runner).

## Summary statistics

| Metric | mean | median | p10 | p90 | min | max |
|---|---|---|---|---|---|---|
| H(pi_prior) | 1.340 | 1.437 | 1.213 | 1.438 | 1.213 | 1.585 |
| H(pi_visits) | 1.213 | 1.207 | 1.199 | 1.250 | 1.169 | 1.379 |
| delta (prior - visits) | 0.127 | 0.178 | 0.014 | 0.230 | -0.055 | 0.333 |
| top-1 visit fraction | 0.526 | 0.509 | 0.399 | 0.649 | 0.395 | 0.649 |
| effective support (exp H) | 3.366 | 3.345 | 3.316 | 3.490 | 3.217 | 3.972 |

## Verdict

H(pi_visits) is approximately equal to H(pi_prior) on average -- MCTS is a no-op that rubber-stamps the prior. This is the worst signal because it means every simulation budget we spend on self-play produces training targets that are identical to the current network's output, so training converges to a fixed point.

## First 5 records (illustrative)

```
g= 0 w= 5 cm= 0 ply= 0 H_prior=1.437 H_visits=1.207 delta=+0.230 top1=0.649 eff_support=3.34
g= 0 w= 0 cm= 0 ply= 0 H_prior=1.437 H_visits=1.207 delta=+0.230 top1=0.649 eff_support=3.34
g= 0 w= 1 cm= 0 ply= 0 H_prior=1.437 H_visits=1.207 delta=+0.230 top1=0.649 eff_support=3.34
g= 0 w= 9 cm= 0 ply= 0 H_prior=1.437 H_visits=1.207 delta=+0.230 top1=0.649 eff_support=3.34
g= 0 w= 4 cm= 0 ply= 0 H_prior=1.437 H_visits=1.207 delta=+0.230 top1=0.649 eff_support=3.34
```
---

## Temperature sampling check (τ=1.0)

Model A: `checkpoint_00015000.pt`  Model B: `checkpoint_00015000.pt`  N=20

Move-count uniqueness: 13 distinct lengths across 20 games

**VERDICT: GAMES DIVERGE** — temperature sampling works. Collapse is purely due to missing Dirichlet noise on the training path.
