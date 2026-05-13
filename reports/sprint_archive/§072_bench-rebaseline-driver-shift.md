<!-- Forensic archive extracted from docs/07_PHASE4_SPRINT_LOG.md during compression sprint 2026-05-13. Compressed counterpart in docs/07_PHASE4_SPRINT_LOG.md. -->

### §72 — Bench Baseline Rebaseline — 2026-04-09 Driver-State Shift

Three `bench.full` runs on 2026-04-09/10 failed the same two §66 targets (NN inference ~8,370 vs 8,500; worker throughput ~541k vs 625k). Cold/hot/idle investigation ruled out thermals (GPU stayed at 49°C). Root cause: NVIDIA laptop driver's `DynamicPowerManagement=3` settled the GPU into a lower boost-clock bin overnight — NN latency 1.59 ms → 1.77–1.80 ms (~14% clock reduction); worker throughput failures downstream.

**Rebaselined targets:** NN inference ≥ 8,250 pos/s (was 8,500); worker throughput ≥ 500,000 pos/hr (was 625,000). Baseline column retains 2026-04-06 peak for hardware capability reference; targets reflect sustained operating floor. Artifacts: `archive/bench_investigation_2026-04-09/`.

---

