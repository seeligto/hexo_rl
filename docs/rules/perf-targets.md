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
**Laptop Q44 refloor (2026-04-30, 8-plane):** 33,174 median, IQR ±5.3%,
range [29.1k–36.3k]. +19% vs desktop; driven by 8-plane smaller tensor +
RTX 4060 Max-Q Ada Lovelace (sm_89). Old estimate ~25k was wrong.

Latest baseline **2026-04-30** (compile OFF, trace ON, 8-plane v6, Q44).
Run: `reports/benchmarks/2026-04-30_12-48.json`. 9/10 targets PASS;
batch_fill 78.6% < 84% = dispatch-GIL bound (Q35, Phase 4.5 item).

| Metric | Baseline (median, n=5) | Target | Notes |
|---|---|---|---|
| MCTS (CPU only, no NN) | 44,073 sim/s | ≥ 26,000 sim/s | IQR ±163.9 (0.4%); was 66,926 at §124 — drop likely from MCTS code changes §124→§131; CPU-only path, plane count irrelevant; target ≥26k still passes easily |
| NN inference (batch=64) | 4,800 pos/s | ≥ 4,000 pos/s (**lowered §124**) | IQR ±5.4 (0.1%); flat vs §124 (4,859). 8-plane model slightly lighter |
| NN latency (batch=1, mean) | 2.56 ms | ≤ 3.5 ms | IQR ±0.02 (0.8%); unchanged vs §124 |
| Replay buffer push | 633,284 pos/sec | ≥ 525,000 pos/sec | IQR ±13,164 (2.1%); up from 615,183 — 8-plane state tensor 56% smaller (6498→2888 bytes) |
| Replay buffer sample raw (batch=256) | 1,124 µs/batch | ≤ 1,300 µs | IQR ±43 (3.8%); was 1,400 µs — 8-plane smaller read |
| Replay buffer sample augmented (batch=256) | 1,256 µs/batch | ≤ 1,500 µs | IQR ±13 (1.0%); was 1,362 µs — same cause |
| GPU utilization | 100.0% | ≥ 85% | saturated; NN-isolated benchmark |
| VRAM usage (process) | 0.10 GB / 8.6 GB | ≤ 6.9 GB (80%) | down from 0.11/8.0 — 8-plane model; VRAM budget matches RTX 4060 Max-Q 8.6 GB |
| Worker throughput | 33,174 pos_gen/hr (laptop n=5, Q44) | ≥ 20,000 pos_gen/hr | IQR ±5.3%, range [29.1k–36.3k]. Desktop RTX 3070 18-plane (§128): 27,835. Laptop now primary reference (Q44 2026-04-30) |
| Batch fill % | 78.6% | ≥ 84% | IQR ±2.3%; **FAIL** — dispatch-GIL bound (Q35); was 99.2% with 18-plane/older dispatch path |

### Compile-on engineering datum (`make bench.compile`, 2026-04-25, 18-plane)

Historical reference only — **not** the production gate, **not** current
model. Measured on 18-plane v5 at §124. Worker numbers are pre-§128
`positions_pushed` metric (not `positions_generated`).

| Metric | compile-on | compile-off+trace | Δ |
|---|---|---|---|
| MCTS sim/s | 68,832 | 66,926 | -2.8% (run-to-run noise) |
| NN inference pos/s | 7,784 | 4,859 | -37.6% (Inductor kernel fusion) |
| NN latency ms | 1.89 | 2.56 | +35% (same root cause) |
| Worker pos_pushed/hr | 186,832 | 177,799 | -4.8% (within IQR) |

Conclusion: **compile and trace deliver the same selfplay throughput
within noise.** Trace wins on simplicity (no Dynamo guard cost, no
cudagraph TLS thread issue, no Triton 27 GB spike). On dispatch-bound
hardware (EPYC 4080S, 60% GPU-util lock per
`feedback_compile_selfplay_dispatch_bound.md`) trace is expected to
lift selfplay materially; pending Q35.

Variance history (pre-warm-up methodology, 2026-04-06 laptop rebaseline,
2026-04-09 driver shift, 2026-04-16 §98 18-channel migration, 2026-04-17
§102 warmup fix, 2026-04-25 §123 compile-on re-enabled, 2026-04-25 §124
compile-off + trace switch): see `docs/07_PHASE4_SPRINT_LOG.md` §98,
§102, §123, §124 for full narrative; `docs/03_TOOLING.md` § "Benchmark
variance (historical)" for the pre-methodology era.
