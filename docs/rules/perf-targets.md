# Benchmarks — must pass before Phase 4.5

> **Methodology:** median of n=5 runs. Per-metric warm-up: 3s MCTS / 3s NN /
> 2s buffer / **90s worker pool** (raised from 30s at §98 to eliminate the
> 0-position measurement windows — §102).
> MCTS uses realistic workload: 800 sims/move × 62 iterations with
> tree reset between moves (matches selfplay.yaml mcts.n_simulations).
> Worker pool: 200 sims/move × max_moves=128, pool_duration=120s (`make bench`).
> CPU frequency unpinned (cpupower unavailable on this system).
> Targets set at `min(observed_median × 0.85, prior_target)` — **conservative**;
> see §102 for target-setting rules.
> Run `make bench` to reproduce. Full results: reports/benchmarks/

Latest baseline **2026-04-18** (laptop Ryzen 7 8845HS + RTX 4060,
14 workers, no CPU pin, LTO + native CPU, 18-plane model, GroupNorm(8)
per §99). Run: `reports/benchmarks/2026-04-18_18-36.json`. All 10
targets PASS:

| Metric | Baseline (median, n=5) | Target | Notes |
|---|---|---|---|
| MCTS (CPU only, no NN) | 69,680 sim/s | ≥ 26,000 sim/s | IQR ±273 (0.4%); **up 23.5% vs 2026-04-17** — treat as cooler thermals / boost-clock upswing, not a code win |
| NN inference (batch=64) | 7,646 pos/s | ≥ 6,500 pos/s | IQR ±0.73 (0.01%); flat vs 2026-04-17 (−0.4%) |
| NN latency (batch=1, mean) | 1.84 ms | ≤ 3.5 ms | IQR ±0.004 (0.2%); 16% faster than 2026-04-17 (2.19 ms) |
| Replay buffer push | 696,880 pos/sec | ≥ 525,000 pos/sec | IQR ±109,127 (15.7%); up 12.7% vs 2026-04-17 — IQR widened, watch |
| Replay buffer sample raw (batch=256) | 1,496 µs/batch | ≤ 1,550 µs | IQR ±11 (0.7%); up 8.5% vs 2026-04-17, still under target. **§113 2026-04-22:** target recalibrated 1,500→1,550 µs — `cda9dde` always-on dedup adds one HashSet alloc + 256 game_id lookups (correctness-required); residual +33 µs confirmed after push.rs transmute fix recovered all other regressions. |
| Replay buffer sample augmented (batch=256) | 1,654 µs/batch | ≤ 1,800 µs | IQR ±293 (17.7%); up 33% vs 2026-04-17 (1,241) — back inside §98 band, confirms §102 "do not tighten on one run" |
| GPU utilization | 99.9% | ≥ 85% | IQR ±0.1; saturated during inference-only benchmark |
| VRAM usage (process) | 0.08 GB / 8.6 GB | ≤ 6.88 GB (80%) | torch.cuda.max_memory_allocated (process-specific, not pynvml global) |
| Worker throughput | 164,052 pos/hr | ≥ 142,000 pos/hr (**PROVISIONAL**) | IQR ±30,138 (18.4%); flat vs 2026-04-17 (−2.2%); IQR widened from 5.7% — likely the single-run warm-up variance §102 warned about |
| Batch fill % | 99.58% | ≥ 84% | IQR ±0.28 |

Variance history (pre-warm-up methodology, 2026-04-06 laptop rebaseline,
2026-04-09 driver shift, 2026-04-16 §98 18-channel migration, 2026-04-17
§102 warmup fix): see `docs/07_PHASE4_SPRINT_LOG.md` §98 and §102 for
full narrative; `docs/03_TOOLING.md` § "Benchmark variance (historical)"
for the pre-methodology era.
