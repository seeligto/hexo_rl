# Diagnostic C.1 — temperature schedule audit

The Rust training path uses a compound-move cosine schedule with a
hard floor at `temperature_threshold_compound_moves`. Sprint log
§36 describes a different formula (per-ply cosine with
`temp_anneal_moves=60`). **Both are reproduced below and differ.**
This is a live docs-vs-code drift. It is not the cause of the mode
collapse on its own (no root noise is), but it belongs in §70 as a
separate bullet so it is greppable later.


Config values: `temperature_min = 0.05`, `temperature_threshold_compound_moves = 15`


## Rust code formula (`engine/src/game_runner.rs:510-515`)

```
tau(cm) = temp_min                                 if cm >= threshold
        = max(temp_min, cos(pi/2 * cm / threshold)) otherwise
```

| compound_move | tau |
|---|---|
| 0 | 1.0000 |
| 5 | 0.8660 |
| 10 | 0.5000 |
| 14 | 0.1045 |
| 15 | 0.0500 |
| 16 | 0.0500 |
| 20 | 0.0500 |
| 30 | 0.0500 |

## Sprint log §36 formula (not implemented)

```
tau(ply) = temp_min + 0.5 * (1 - temp_min) * (1 + cos(pi * ply / anneal_moves))
           with anneal_moves = 60, per-ply (not per compound_move)
```

| ply | tau (sprint log §36) |
|---|---|
| 0 | 1.0000 |
| 5 | 0.9838 |
| 10 | 0.9364 |
| 14 | 0.8780 |
| 15 | 0.8609 |
| 16 | 0.8428 |
| 20 | 0.7625 |
| 30 | 0.5250 |
| 60 | 0.0500 |
| 120 | 0.0500 |

## Conclusion

- The code drops to the `temp_min = 0.05` floor at compound_move 15
  (ply 29 or 30 depending on player-1's solo opener). Between cm=0
  and cm=15 the schedule is a quarter-cosine falling from 1.0 to 0.0,
  clamped at temp_min. After cm=15 there is zero further annealing.
- Sprint log §36 would keep tau above the floor until ply 60, with a
  symmetric half-cosine shape. Under that schedule a game has roughly
  four times as many moves with meaningfully stochastic sampling.
- Since the root policy on the collapsed ckpt_15000 is already 54%
  concentrated on a single move at cm=0 (see diag_A_trace_summary.md),
  even a temperature of 1.0 cannot produce meaningful variation --
  pi[top] / sum(pi^(1/1)) is still the top draw with >50% probability.
  The docs-vs-code drift therefore does not explain the collapse, but
  it does mean the implemented schedule gives the network even less
  room to escape a sharpened prior than the documented one did.
