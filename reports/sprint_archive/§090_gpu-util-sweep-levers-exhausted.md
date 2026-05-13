<!-- Forensic archive extracted from docs/07_PHASE4_SPRINT_LOG.md during compression sprint 2026-05-13. Compressed counterpart in docs/07_PHASE4_SPRINT_LOG.md. -->

## §90 — GPU util sweep: inf_bs / wait_ms levers are exhausted (2026-04-13)

**Verdict.** `inference_batch_size=64, inference_max_wait_ms=4.0` stays as laptop live config. H1 (raise inf_bs to lift GPU util) falsified — `gpu_util` invariant at ~83% across all swept cells; bottleneck is NN forward latency (12.5 ms live vs 1.6 ms bench), not batcher config. Phase 1 reframed Tom's "28% GPU util" report as throughput-vs-bench ratio; actual util is 84%.

**Sweep:** A=(64, 4.0), B=(128, 8.0), C=(128, 4.0). Laptop 4060, gumbel_targets, fresh bootstrap, 14 workers, 20-min windows last-15-min measured. Entropy kill-gate (≥ 4.0 nats `policy_entropy_selfplay`) passed all runs.

### Results (last 15 min)

| metric | Run A | Run B | Run C | B vs A | C vs A |
|---|---:|---:|---:|---:|---:|
| games/hr | 545 | 381 | 372 | −30.0% | −31.8% |
| pos/hr | 215,527 | 200,530 | 217,535 | −7.0% | +0.9% |
| nn_forwards/sec | 88.2 | 54.0 | 53.4 | −38.7% | −39.4% |
| nn_mean_batch_size | 60.1 | 84.8 | 85.8 | +40.9% | +42.7% |
| nn_pos/sec | 5,304 | 4,579 | 4,585 | **−13.7%** | **−13.6%** |
| batch_fill_pct | 91.4 | 63.4 | 67.4 | −30.6% | −26.2% |
| gpu_util_mean | 83.7 | 83.2 | 83.1 | −0.6% | −0.8% |
| steps in window | 540 | 380 | 340 | −29.6% | **−37.0%** |
| game_len_median | 37 | 62 | 74 | +68% | +100% |

Raising `inf_bs` to 128 grows mean batch 60 → 85 (+42%) but forwards/sec collapses 88 → 53 (−39%) — workers can't supply 128 leaves in the same wall-clock window. Run C's flat pos/hr masks −37% steps/hr because `game_len_median` doubled; **pos/hr is not a sufficient summary statistic when game length shifts**. Future sweeps must report steps/hr.

**No config commit.** Artifacts: `archive/sweep_2026-04-13_gpu_util/`. Architectural levers (CUDA stream separation, process split, `torch.compile` re-enable) deferred to Phase 4.5 as **Q18** in `docs/06_OPEN_QUESTIONS.md`. Desktop 3070 not validated here — single-run confirmation worth doing before committing if 3070 ever runs gumbel_targets sustained.

---

