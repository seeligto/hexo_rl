<!-- Forensic archive extracted from docs/07_PHASE4_SPRINT_LOG.md during compression sprint 2026-05-13. Compressed counterpart in docs/07_PHASE4_SPRINT_LOG.md. -->

## §135 — Bench gate: W4 8-plane migration, no regressions — 2026-04-30

**Date:** 2026-04-30  
**Hardware:** Desktop AMD Ryzen 7 3700x + RTX 3070, AC power.  
**Run:** `reports/benchmarks/2026-04-30_07-17.json`  
**Report:** `reports/benches/v6_8plane_baseline_20260429.md`

Bench gate after §131 P1–P3 + §134 bootstrap. Model confirmed 8-plane (`trunk.input_conv.weight` shape `[128, 8, 3, 3]`).

### Bugs fixed in benchmark.py (same commit `e9a4d72`)

`benchmark_inference`, `benchmark_inference_latency`, and `benchmark_gpu_utilisation` all hardcoded `18` in dummy tensor shapes — crashed against the P3 8-plane model. Fixed: `getattr(model, "in_channels", 18)`.

`_CHECKS_CUDA` NN inference target was 6,500 (pre-§124 compile-on value, never updated when §124 lowered the target to 4,000 in perf-targets.md). Fixed to 4,000.

### Result (n=5, vs pre-W4 desktop §128 baseline `2026-04-28_19-52`)

| Metric | Pre-W4 18-plane | 8-plane | Δ | Status |
|---|---|---|---|---|
| MCTS sim/s | 44,254 | 44,233 | −0.05% | ✓ flat |
| NN inference pos/s | 4,380 | 4,828 | +10.2% | ✓ improved (smaller H2D) |
| NN latency ms | 2.6 | 2.66 | +2.3% | ✓ within noise |
| Buffer push pos/s | 423,068 | 708,508 | +67.5% | ✓ improved (56% smaller state) |
| Buffer raw µs | 1,742 | 1,051 | −39.7% | ✓ improved |
| Buffer aug µs | 1,841 | 1,050 | −43.0% | ✓ improved |
| GPU util % | 100.0 | 100.0 | flat | ✓ |
| Worker pos/hr | 27,835 | 31,764 | +14.1% | ✓ improved |
| Batch fill % | 100.0 | 99.78 | −0.2pp | ✓ flat |
| Pool overflows | 0 | 0 | — | ✓ |

All 9 gated metrics PASS against perf-targets.md CUDA floors. No regressions > 10%.

**MCTS flat** — no MCTS code in §131. **NN +10%** — 56% smaller H2D tensor (2,888 vs 6,498 f16 elements per leaf). **Buffer push +68%** — state memcpy 56% smaller; spec predicted ~2×, actual 1.67× (overhead + lock floor the asymptote). **Buffer raw/aug −40%/−43%** — scatter reads 8-plane rows; 8/18 = 44% theoretical, observed ~40% consistent. **Worker +14%** — NN speedup cascades; IQR ±3.9%, no bimodal artifact.

No perf target updates — desktop evidence does not update laptop-calibrated floors. Laptop re-bench needed for buffer push/sample floors before tightening.

---

