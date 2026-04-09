# Diagnostic B — policy sharpness across checkpoints

## K=0 caveat (must be read first)

> Entropy values were measured on the K=0 (centroid) cluster window
> only, not the full min-pool aggregation used by the training path.
> Absolute comparison against the sprint log §1 heuristic ("expected
> 3-6 nats; < 1.0 signals collapse") is **indicative, not strict** --
> that heuristic was derived on the full min-pool path and may differ
> by several tenths of a nat from K=0 numbers. The **primary signal is
> the progression across checkpoints on the same positions**, which is
> unaffected by the K choice because every checkpoint is evaluated on
> identical inputs. Do not argue about whether "ckpt_15000 at 0.2
> nats" is really collapsed or really at 0.8 nats under min-pool --
> argue about whether the gap between `best_checkpoint.pt` and
> ckpt_15000 is catastrophic.


Positions evaluated: 500 (same 500 positions per checkpoint)

Source: recent self-play games from the latest run directory.

Device: cuda


## Per-checkpoint summary

| Checkpoint | H(π) mean | median | p10 | p90 | top-1 mean | eff support mean |
|---|---|---|---|---|---|---|
| bootstrap_model.pt | 2.665 | 2.688 | 1.330 | 3.889 | 0.379 | 21.48 |
| checkpoint_00013000.pt | 1.666 | 1.643 | 0.620 | 2.681 | 0.497 | 9.72 |
| checkpoint_00014000.pt | 1.581 | 1.547 | 0.556 | 2.622 | 0.520 | 7.00 |
| checkpoint_00015000.pt | 1.532 | 1.601 | 0.569 | 2.336 | 0.524 | 5.79 |
| checkpoint_00016000.pt | 1.649 | 1.650 | 0.521 | 2.572 | 0.504 | 7.05 |
| checkpoint_00017000.pt | 1.486 | 1.446 | 0.477 | 2.353 | 0.540 | 6.68 |
| checkpoint_00017428.pt | 1.698 | 1.644 | 0.531 | 2.755 | 0.505 | 9.35 |
| best_model.pt | 2.665 | 2.688 | 1.330 | 3.889 | 0.379 | 21.48 |

## Phase split — H(π) per checkpoint per phase bucket

| Checkpoint | Phase | n | mean | median | p10 | p90 |
|---|---|---|---|---|---|---|
| bootstrap_model.pt | early (cm<10) | 291 | 2.430 | 2.476 | 1.268 | 3.559 |
| bootstrap_model.pt | mid (10<=cm<25) | 118 | 2.665 | 2.780 | 1.242 | 3.966 |
| bootstrap_model.pt | late (cm>=25) | 91 | 3.418 | 3.592 | 2.012 | 4.353 |
| checkpoint_00013000.pt | early (cm<10) | 291 | 1.622 | 1.643 | 0.735 | 2.534 |
| checkpoint_00013000.pt | mid (10<=cm<25) | 118 | 1.466 | 1.463 | 0.179 | 2.652 |
| checkpoint_00013000.pt | late (cm>=25) | 91 | 2.070 | 2.001 | 0.842 | 3.245 |
| checkpoint_00014000.pt | early (cm<10) | 291 | 1.499 | 1.473 | 0.839 | 2.164 |
| checkpoint_00014000.pt | mid (10<=cm<25) | 118 | 1.443 | 1.351 | 0.191 | 2.757 |
| checkpoint_00014000.pt | late (cm>=25) | 91 | 2.021 | 2.072 | 0.495 | 3.282 |
| checkpoint_00015000.pt | early (cm<10) | 291 | 1.591 | 1.641 | 0.813 | 2.313 |
| checkpoint_00015000.pt | mid (10<=cm<25) | 118 | 1.317 | 1.298 | 0.081 | 2.311 |
| checkpoint_00015000.pt | late (cm>=25) | 91 | 1.621 | 1.596 | 0.692 | 2.652 |
| checkpoint_00016000.pt | early (cm<10) | 291 | 1.634 | 1.633 | 0.931 | 2.416 |
| checkpoint_00016000.pt | mid (10<=cm<25) | 118 | 1.465 | 1.523 | 0.294 | 2.697 |
| checkpoint_00016000.pt | late (cm>=25) | 91 | 1.935 | 1.963 | 0.690 | 3.123 |
| checkpoint_00017000.pt | early (cm<10) | 291 | 1.387 | 1.418 | 0.649 | 2.074 |
| checkpoint_00017000.pt | mid (10<=cm<25) | 118 | 1.419 | 1.468 | 0.132 | 2.567 |
| checkpoint_00017000.pt | late (cm>=25) | 91 | 1.887 | 1.749 | 0.527 | 3.288 |
| checkpoint_00017428.pt | early (cm<10) | 291 | 1.623 | 1.608 | 0.856 | 2.361 |
| checkpoint_00017428.pt | mid (10<=cm<25) | 118 | 1.620 | 1.545 | 0.136 | 2.927 |
| checkpoint_00017428.pt | late (cm>=25) | 91 | 2.037 | 2.087 | 0.660 | 3.584 |
| best_model.pt | early (cm<10) | 291 | 2.430 | 2.476 | 1.268 | 3.559 |
| best_model.pt | mid (10<=cm<25) | 118 | 2.665 | 2.780 | 1.242 | 3.966 |
| best_model.pt | late (cm>=25) | 91 | 3.418 | 3.592 | 2.012 | 4.353 |

## Histograms (10 bins over 0.0 - 4.0 nats)

```
bin edge:   0.0  0.4  0.8  1.2  1.6  2.0  2.4  2.8  3.2  3.6
bootstrap_model.pt                 2    7   25   48   48   57   89   61   69   61
checkpoint_00013000.pt            34   60   56   67  132   63   51   14   12    4
checkpoint_00014000.pt            39   34   85  110  102   66   31   19    6    3
checkpoint_00015000.pt            39   42   53  113  123   88   29   10    3    0
checkpoint_00016000.pt            34   39   63  101   98   79   57   19    4    6
checkpoint_00017000.pt            43   47   81  125  100   59   20    8    8    5
checkpoint_00017428.pt            37   38   73   89  102   63   53   20    6    8
best_model.pt                      2    7   25   48   48   57   89   61   69   61
```

## Restart candidate heuristic

`checkpoint_00017428.pt` is the latest checkpoint with mean raw-policy
entropy >= 1.5 nats across the sampled positions. Restart
candidates for the Phase 4.0 fix session are this checkpoint
or earlier.
