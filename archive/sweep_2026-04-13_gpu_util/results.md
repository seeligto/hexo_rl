# Phase 2 GPU Util Sweep — 2026-04-13

3-run narrowed sweep testing **H1: raise inference batch coalescing**. All runs 20 min from `bootstrap_model.pt`, `gumbel_targets` variant, laptop (Ryzen 7 8845HS + RTX 4060). Measurement window = last 15 min. See prompt in conversation log and `/tmp/gpu_util_phase1.md` for Phase 1 context.

## Config (held constant)

- variant: `gumbel_targets` (gumbel_mcts=false, completed_q_values=true)
- `standard_sims: 200`, `training_steps_per_game: 4`, `max_train_burst: 16`
- `n_workers: 14`, `leaf_batch_size: 8`, `fast_prob: 0.0`
- `dirichlet_enabled: true`
- fresh `bootstrap_model.pt`, `mixing.buffer_persist: false`

## Config (varied)

| run | inference_batch_size | inference_max_wait_ms |
|-----|---------------------:|----------------------:|
| A   | 64                   | 4.0                   |
| B   | 128                  | 8.0                   |
| C   | 128                  | 4.0                   |

## Results

Metrics computed over the **last 15 min** of each 20-min window (first 5 min discarded as warm-up, beyond the CUDA warmup already done at worker-pool start).

| metric | Run A | Run B | Run C | B vs A | C vs A |
|---|---:|---:|---:|---:|---:|
| games/hr | 545 | 381 | 372 | -30.0% | -31.8% |
| pos/hr (buffer delta) | 215527 | 200530 | 217535 | -7.0% | +0.9% |
| nn_forwards/sec | 88.2 | 54.0 | 53.4 | -38.7% | -39.4% |
| nn_mean_batch_size | 60.14 | 84.76 | 85.84 | +40.9% | +42.7% |
| nn_pos/sec (fwd × batch) | 5304 | 4579 | 4585 | -13.7% | -13.6% |
| batch_fill_pct (mean) | 91.4 | 63.4 | 67.4 | -30.6% | -26.2% |
| gpu_util_mean (dmon) | 83.7 | 83.2 | 83.1 | -0.6% | -0.8% |
| gpu_util_p10 (dmon) | 79 | 77 | 77 | -2.5% | -2.5% |
| gpu_util_p90 (dmon) | 91 | 90 | 89 | -1.1% | -2.2% |
| policy_entropy_selfplay (final) | 5.180 | 5.143 | 5.466 | -0.7% | +5.5% |
| policy_entropy_selfplay (min) | 4.848 | 4.982 | 5.097 | +2.8% | +5.1% |
| policy_entropy combined (final) | 2.686 | 2.842 | 2.883 | +5.8% | +7.4% |
| steps in window | 540 | 380 | 340 | -29.6% | -37.0% |
| games in window | 135 | 95 | 85 | -29.6% | -37.0% |

## Kill criterion check

Per Phase 1 correction (`/tmp/gpu_util_phase1.md`): `policy_entropy_selfplay` must remain ≥ 4.0 nats throughout the window; combined entropy is not a collapse signal at this training stage.

- Run A: min(policy_entropy_selfplay) = 4.848 — PASS
- Run B: min(policy_entropy_selfplay) = 4.982 — PASS
- Run C: min(policy_entropy_selfplay) = 5.097 — PASS

## Winner

- Run B pos/hr delta vs A: -7.0% (entropy safe: True)
- Run C pos/hr delta vs A: +0.9% (entropy safe: True)
- **No winner beats Run A by ≥5% pos/hr with entropy safe.** Config is already near-optimal on the inf_bs/wait_ms axis.

## Key findings

1. **H1 is falsified — raising inf_bs to 128 does not help.** Mean batch *does* grow 60 → 85
   (+42%) but `nn_forwards/sec` collapses 88.2 → 53.4 (−39%), so the product
   `nn_pos/sec` actually **drops 14%** (5304 → 4585). The larger batches cost more
   per-forward latency than they save in amortization, because the workers cannot
   supply 128 leaves in the same wall-clock window that they supply 64. Batch
   fill drops 91% → 63–67%, confirming starved-batcher, not starved-GPU.
2. **GPU util is invariant across all three runs (~83%).** Confirms the Phase 1
   finding: GPU is busy but inefficient. The levers in this sweep cannot move it.
3. **Run C is not meaningfully different from Run B.** Isolating wait_ms=4 from
   wait_ms=8 changes fill% from 63.4 → 67.4 and entropy selfplay min is slightly
   better, but `nn_forwards/sec` and `nn_pos/sec` are nearly identical. wait_ms
   is not the dominant lever at this batch size.
4. **pos/hr survives in Run C only because games_len_median doubles** (37 → 74).
   The larger batch slows per-move latency enough that `games/hr` collapses 545 →
   372 (−32%), but each game produces ~2× more positions (likely: longer
   per-move budget → fewer blunders → games run closer to the 200-ply cap).
   Net pos/hr is a coincidental wash, not a throughput gain. Training-step
   throughput (`steps/hr` = steps_in_window × 4) correspondingly drops
   540 → 340 (−37%), which is a real *learning signal* regression even if
   pos/hr is flat.
5. **Entropy kill criterion passes everywhere.** Run C actually shows the
   highest selfplay entropy (5.47 / 5.10 min), consistent with fewer games
   completed and longer per-move budget (more exploration in the window).

## Next lever — architectural, not config

The remaining 12.5 ms live NN forward latency vs 1.6 ms bench latency (§89,
`/tmp/gpu_util_phase1.md`) is the next ceiling. No `inference_batch_size` /
`inference_max_wait_ms` combination on this sweep moves it, and larger batches
actively make per-forward latency *worse* because GPU kernels run longer.

Candidate interventions (all architectural, all outside the prompt scope):

- **CUDA stream separation** — inference and training gradient steps on
  separate streams so training kernels don't pollute the inference kernel
  cache / evict autocast state.
- **Process split** — run `scripts/train.py`'s training loop in a second
  process, with the inference server and worker pool in the primary. Trades
  IPC + duplicate weight hosting for zero cross-contamination.
- **torch.compile re-enable** — blocked on Python 3.14 + CUDA graphs
  compatibility (sprint §25, §30, §32). When unblocked, compiled forward
  should cut per-forward dispatch overhead substantially.

Flag for Phase 4.5 followup. Not a Phase 4.0 blocker.

