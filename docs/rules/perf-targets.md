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

Latest baseline **2026-05-22** (post-§S183 perf wave, master `1995873` — §S182 `legal_moves_set` capacity-reserve + §S183 MCTS micro-opts merged; §S184/§S186 perf strategies aborted, no engine change; compile OFF, trace ON, 8-plane v6).
**10/10 targets PASS — `make bench` n=5, laptop `n_workers=14` + vast.ai `n_workers=22`.**

> **MCTS floor refloored (2026-05-22):** the `MCTS (CPU only, no NN)` target
> was ≥26,000 sim/s against an ~87k median — 70% slack, caught no regression.
> Refloored to **≥73,000** (0.85× the lower host median, clear of both hosts'
> observed minimum). The other nine floors are unchanged by design: laptop
> buffer-push has an outlier-low run (709k vs 862k median) and worker
> throughput is documented-bimodal (IQR ±26%) — a blanket 0.85×-median floor
> would false-fail those. NOTE: the §S182 `mcts_sims_cpu_only` *criterion*
> micro-bench gain (+66.4%) does **not** appear in this realistic `make bench`
> MCTS metric — vast is flat (86.8k → 86.9k); the laptop's apparent +24.8% is
> the old §180 baseline having been bimodal-depressed ("range 67k-72k
> bimodal"), not a fresh gain. Different benchmarks, different workloads.

### Laptop (Ryzen 7 8845HS + RTX 4060 Laptop GPU) — PRIMARY reference

| Metric | Baseline (median, n=5) | Target | Notes |
|---|---|---|---|
| MCTS (CPU only, no NN) | 88,006 sim/s | ≥ 73,000 sim/s | range 87.7k-90.1k, IQR ±904 — tight, no bimodality. Old §180 baseline 70,520 was bimodal-depressed; floor refloored ≥26k→≥73k. |
| NN inference (batch=64) | 4,871 pos/s | ≥ 4,000 pos/s | range 4.87k-4.88k, IQR ±0.2; +8.2% vs §180 4,500 |
| NN latency (batch=1, mean) | 2.68 ms | ≤ 3.5 ms | range 2.60-2.73 ms; flat vs §180 2.66 |
| Replay buffer push | 862,037 pos/sec | ≥ 525,000 pos/sec | range 709k-866k; +2.5% vs §180 840,734 — one outlier-low run (709k), floor kept conservative |
| Replay buffer sample raw (batch=256) | 992 µs/batch | ≤ 1,300 µs | range 965-1015 µs; +4.5% vs §180 949 (within noise) |
| Replay buffer sample augmented (batch=256) | 966 µs/batch | ≤ 1,500 µs | range 926-1155 µs; −1.2% vs §180 978 |
| GPU utilization | 100.0% | ≥ 85% | range 97.4-100%; saturated, NN-isolated benchmark |
| VRAM usage (process) | 0.10 GB / 8.6 GB | ≤ 6.9 GB (80%) | unchanged — 8-plane model; VRAM budget matches RTX 4060 Max-Q 8.6 GB |
| Worker throughput | 33,565 pos_gen/hr | ≥ 20,000 pos_gen/hr | range 27.0k-35.8k, IQR ±8,579 — bimodal; +15.3% vs §180 29,118; floor kept loose for the bimodal low mode |
| Batch fill % | 98.75% | ≥ 84% | range 97.0-99.9%; flat vs §180 98.82 |

### Vast.ai mirror (Ryzen 9 9900X + RTX 5080, CUDA 12.8) — n=5

Vast.ai is the production training host; the mirror confirms the §S182/§S183 perf merges do not regress on the canonical-rental hardware. `make bench` `n_workers=22`.

| Metric | Baseline (median, n=5) | Target | Notes |
|---|---|---|---|
| MCTS (CPU only, no NN) | 86,890 sim/s | ≥ 73,000 sim/s | range 86.8k-86.9k, IQR ±19 — rock-stable; +0.1% vs §180 86,780 — flat. §S182/§S183 criterion-bench gains do not appear in this realistic make-bench MCTS workload. |
| NN inference (batch=64) | 14,206 pos/s | ≥ 4,000 pos/s | range 14.2k, IQR ±0.2; −0.7% vs §180 14,301; 2.92× laptop (Blackwell sm_120 vs Ada sm_89) |
| NN latency (batch=1, mean) | 1.54 ms | ≤ 3.5 ms | range 1.54 ms; flat vs §180 1.53; 0.57× laptop |
| Replay buffer push | 1,007,042 pos/sec | ≥ 525,000 pos/sec | range 994k-1.01M; −2.0% vs §180 1,027,088; 1.17× laptop |
| Replay buffer sample raw (batch=256) | 735 µs/batch | ≤ 1,300 µs | range 733-738 µs; +1.6% vs §180 723; 0.74× laptop |
| Replay buffer sample augmented (batch=256) | 743 µs/batch | ≤ 1,500 µs | range 740-748 µs; +1.6% vs §180 731; 0.77× laptop |
| GPU utilization | 94.0% | ≥ 85% | range 80.6-94%; not saturated (5080 outpaces 22 workers — expected) |
| VRAM usage (process) | 0.10 GB / 17.1 GB | ≤ 13.7 GB (80%) | 5080 has 17.1 GB VRAM |
| Worker throughput | 91,871 pos_gen/hr | ≥ 20,000 pos_gen/hr | range 89.0k-92.4k, IQR ±900; +9.8% vs §180 83,692; 2.74× laptop |
| Batch fill % | 99.998% | ≥ 84% | range 99.98-99.999% |

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
