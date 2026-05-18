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

Latest baseline **2026-05-18** (§180 cycle 3 close at HEAD `5e0c09d` — Wave 11 Batch B `run_worker_thread` 8-sub-fn extraction, compile OFF, trace ON, 8-plane v6).
**10/10 targets PASS on laptop n=4 × 10 internal (40 samples) + vast.ai mirror n=2 × 10 internal (20 samples).**

### Laptop (Ryzen 7 8845HS + RTX 4060 Laptop GPU) — PRIMARY reference

| Metric | Baseline (median, n=4×10) | Target | Notes |
|---|---|---|---|
| MCTS (CPU only, no NN) | 70,520 sim/s | ≥ 26,000 sim/s | range 67k-72k, bimodal CUDA state across runs (runs 1+2 low / 3+4 high — §180 SD7 variance envelope) |
| NN inference (batch=64) | 4,500 pos/s | ≥ 4,000 pos/s | range 4.1k-4.8k, same bimodal pattern; −7.6% vs §156 4,800 within SD7 envelope |
| NN latency (batch=1, mean) | 2.66 ms | ≤ 3.5 ms | range 2.45-2.88 ms; flat vs §156 |
| Replay buffer push | 840,734 pos/sec | ≥ 525,000 pos/sec | range 768k-873k; +27.4% vs Wave 10 ref 660k (P79/P68 cycle 3 cumulative effect — buffer push path untouched in cycle 3, gain is from CUDA/dispatch state) |
| Replay buffer sample raw (batch=256) | 949 µs/batch | ≤ 1,300 µs | range 914-1072 µs; −24.3% vs Wave 10 ref 1,253 |
| Replay buffer sample augmented (batch=256) | 978 µs/batch | ≤ 1,500 µs | range 963-1127 µs; −30.4% vs Wave 10 ref 1,404 |
| GPU utilization | 100.0% | ≥ 85% | saturated; NN-isolated benchmark |
| VRAM usage (process) | 0.10 GB / 8.6 GB | ≤ 6.9 GB (80%) | unchanged — 8-plane model; VRAM budget matches RTX 4060 Max-Q 8.6 GB |
| Worker throughput | 29,118 pos_gen/hr | ≥ 20,000 pos_gen/hr | range 26k-30k, −9.9% vs Wave 10 ref 32,334 — at SD7 bidirectional variance edge; bimodal CUDA across runs; §180 close documents SD7 promotion datapoint |
| Batch fill % | 98.82% | ≥ 84% | range 98.27-99.77%; flat vs §156 |

### Vast.ai mirror (Ryzen 9 9900X + RTX 5080, CUDA 12.8) — n=2×10

Cycle close mirror per master directive. Vast.ai is the production training host; mirror confirms cycle 3 work doesn't regress on the canonical-rental hardware.

| Metric | Baseline (median, n=2×10) | Target | Notes |
|---|---|---|---|
| MCTS (CPU only, no NN) | 86,780 sim/s | ≥ 26,000 sim/s | range 85k-87k; 1.23× laptop (CPU-only, 12c/24t 9900X vs 8c/16t 8845HS) |
| NN inference (batch=64) | 14,301 pos/s | ≥ 4,000 pos/s | range 14.2k-14.4k; 3.18× laptop (Blackwell sm_120 vs Ada Lovelace sm_89) |
| NN latency (batch=1, mean) | 1.53 ms | ≤ 3.5 ms | range 1.50-1.55 ms; 0.58× laptop |
| Replay buffer push | 1,027,088 pos/sec | ≥ 525,000 pos/sec | range 999k-1.06M; 1.22× laptop |
| Replay buffer sample raw (batch=256) | 723 µs/batch | ≤ 1,300 µs | range 700-746 µs; 0.76× laptop |
| Replay buffer sample augmented (batch=256) | 731 µs/batch | ≤ 1,500 µs | range 702-759 µs; 0.75× laptop |
| GPU utilization | 94.0% | ≥ 85% | range 80-94%; not saturated (5080 finishes batches faster than 14 workers can refill — expected) |
| VRAM usage (process) | 0.10 GB / 17.1 GB | ≤ 13.7 GB (80%) | 5080 has 17.1 GB VRAM |
| Worker throughput | 83,692 pos_gen/hr | ≥ 20,000 pos_gen/hr | range 80k-87k; 2.87× laptop. v3 retake on `/root/hexo_rl` (non-canonical workspace) saw batch-fill 56% anomaly run 2; v4 on `/workspace/hexo_rl` did NOT reproduce (both runs 99%+); anomaly attributed to vast-host noisy-neighbor on shared infrastructure |
| Batch fill % | 99.51% | ≥ 84% | range 99.20-99.83% on v4 retake |

**Cross-host floor convention:** laptop targets are conservative and remain the canonical gate (Phase 4.5 exit criteria). Vast.ai serves as production-host mirror; same target floors apply because the gate is throughput-relative-to-floor, not relative-to-host. Vast medians being higher than laptop is expected and documented per cycle 3 §180.

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
