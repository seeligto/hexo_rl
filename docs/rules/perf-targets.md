# Benchmarks — must pass before Phase 4.5

> **Methodology:** median of n=5 runs. Per-metric warm-up: 3s MCTS / 3s NN /
> 2s buffer / **90s worker pool** (raised from 30s at §98 to eliminate the
> 0-position measurement windows — §102).
> MCTS uses realistic workload: 800 sims/move × 62 iterations with
> tree reset between moves (matches selfplay.yaml mcts.n_simulations).
> Worker pool: 200 sims/move × max_moves=128, pool_duration=120s (`make bench`).
> CPU frequency unpinned (cpupower unavailable on omarchy).
> **Compile OFF, trace ON** (per §124, 2026-04-25 — matches production
> training; `gumbel_targets_epyc4080.yaml` and other selfplay variants set
> `torch_compile: false`). The InferenceServer trace fix (gated by
> `selfplay.trace_inference`, default true) collapses ~100 `_call_impl`
> calls per forward into one ScriptModule — same dispatch-overhead win
> as compile, no Dynamo guard cost. Trace and compile are mutually
> exclusive (`jit.trace` cannot introspect FX/dynamo-wrapped functions).
> Targets set at `min(observed_median × 0.85, prior_target)` — **conservative**;
> see §102 for target-setting rules.
> Run `make bench` to reproduce. Use `make bench.compile` for the
> engineering datum (compile-on, NN-isolated peak — selfplay path is
> dispatch-bound there, not production-relevant). Full results: reports/benchmarks/

**Reference hardware:** laptop AMD Ryzen 7 8845HS (16 thread, no CPU
pin) + RTX 4060 Laptop GPU (8 GB VRAM), 30 GB RAM, AC power. Bench
numbers shift materially across hardware (e.g. desktop Ryzen 7 3700x
+ RTX 3070, rental EPYC 7702 + RTX 4080 Super); the targets below are
calibrated to the laptop reference. Re-bench on a different host owns
its own target table — do not mix.

**§128 metric change (2026-04-28):** `worker_pos_per_hr` counter switched
from `positions_pushed` (K cluster views × plies, bursts at game-end) to
`positions_generated` (1 per ply, continuous). K_avg ≈ 7 on typical mid-game
boards → all worker targets ÷ 7. Old 177,799 pos_pushed/hr ≡ ~25,400
pos_gen/hr. **Desktop RTX 3070 n=5 confirms:** 27,835 median, IQR ±2,398
(8.6%), range [24.6k–30.0k], PASS against 20k target. Bimodal artifact
eliminated — all 5 runs unimodal. Target 20k floor confirmed.
**Laptop reference re-bench still pending** — expect ~25k gen/hr (177,799/7).

Latest baseline **2026-04-25** (compile OFF, trace ON, post-§124).
Run: `reports/benchmarks/2026-04-25_20-45.json`. 9/10 targets PASS;
NN inference target lowered to track the compile-off methodology shift.

| Metric | Baseline (median, n=5) | Target | Notes |
|---|---|---|---|
| MCTS (CPU only, no NN) | 66,926 sim/s | ≥ 26,000 sim/s | IQR ±2,090 (3.1%); flat vs §123 (CPU-only path, trace fix has no effect) |
| NN inference (batch=64) | 4,859 pos/s | ≥ 4,000 pos/s (**lowered §124**) | IQR ±0.4 (0.0%); was 7,784 with compile on. Drop reflects loss of Inductor kernel fusion — selfplay benefits from trace instead, see Worker row |
| NN latency (batch=1, mean) | 2.56 ms | ≤ 3.5 ms | IQR ±0.08 (3%); up from 1.89 ms compile-on; same root cause as NN inference |
| Replay buffer push | 615,183 pos/sec | ≥ 525,000 pos/sec | IQR ±12,309 (2.0%); flat vs §123. (One spurious run during the §124 compile-on bench dropped to 400k → ±33% IQR; resolved on cool restart, environmental.) |
| Replay buffer sample raw (batch=256) | 1,400 µs/batch | ≤ 1,550 µs | IQR ±7 (0.5%); flat |
| Replay buffer sample augmented (batch=256) | 1,362 µs/batch | ≤ 1,800 µs | IQR ±38 (2.8%); flat |
| GPU utilization | 100.0% | ≥ 85% | saturated; NN-isolated benchmark |
| VRAM usage (process) | 0.11 GB / 8.0 GB | ≤ 6.4 GB (80%) | torch.cuda.max_memory_allocated |
| Worker throughput | 27,835 pos_gen/hr (desktop n=5, §128) | ≥ 20,000 pos_gen/hr | §128: metric switched to positions_generated (continuous). Desktop RTX 3070 n=5: IQR ±2,398 (8.6%), range [24.6k–30.0k]. 20k floor confirmed (27,835 × 0.85 = 23,659). Laptop baseline ~25,400 gen/hr (177,799 pushed ÷ K_avg 7); re-bench pending. |
| Batch fill % | 99.2% | ≥ 84% | IQR ±0.32 |

### Compile-on engineering datum (`make bench.compile`, 2026-04-25)

For reference only — **not** the production gate. Same hardware,
same commit. Trace falls back (jit.trace can't introspect FX/dynamo
output); the dispatcher runs the compiled model.

| Metric | compile-on | compile-off+trace | Δ |
|---|---|---|---|
| MCTS sim/s | 68,832 | 66,926 | -2.8% (run-to-run noise) |
| NN inference pos/s | 7,784 | 4,859 | -37.6% (Inductor kernel fusion) |
| NN latency ms | 1.89 | 2.56 | +35% (same root cause) |
| Worker pos/hr | 186,832 | 177,799 | -4.8% (within IQR) |

Conclusion on this hardware: **compile and trace deliver the same
selfplay throughput within noise.** Trace wins on simplicity (no
Dynamo guard cost, no cudagraph TLS thread issue, no Triton 27 GB
spike). On dispatch-bound hardware (EPYC 4080S, 60% GPU-util lock per
`feedback_compile_selfplay_dispatch_bound.md`) trace is expected to
lift selfplay materially; pending the §124 narrow validation sweep.

Variance history (pre-warm-up methodology, 2026-04-06 laptop rebaseline,
2026-04-09 driver shift, 2026-04-16 §98 18-channel migration, 2026-04-17
§102 warmup fix, 2026-04-25 §123 compile-on re-enabled, 2026-04-25 §124
compile-off + trace switch): see `docs/07_PHASE4_SPRINT_LOG.md` §98,
§102, §123, §124 for full narrative; `docs/03_TOOLING.md` § "Benchmark
variance (historical)" for the pre-methodology era.
