# WP-A — COND-3: CUDA bench + build-vs-forward (GNN-integration program)

**Date:** 2026-07-14 · **Status:** VERDICT RECORDED · **Program:** GNN-integration (R4 ratified b+)

**Question (pre-registered):** does the CPU-derived cost story (ORT best; forward 91–96%,
build 4–8%) hold under CUDA, and what is graph-build cost relative to the dense CNN forward?

## Instrument change (operator, 2026-07-14)

Run3-CNN launches soon → **vast box off-limits to this program** (supersedes the dispatcher's
"GPU window" authorization). All cells measured on the **laptop RTX 4060 Max-Q** (Ryzen 7 8845HS),
torch 2.11.0+cu130, ORT 1.27.0. The pre-registered verdict buckets are ratio-based and the
scoping-doc transfer rule applies (GNN-vs-CNN RATIO transfers across GPUs; absolute ms do not;
the 5080's dense-GEMM advantage likely widens CNN-vs-GNN slightly). **RIDER: re-run
`scripts/research/gnn_infer_bench.py` unchanged on the 5080 when the box frees (post-run3 /
run4 handoff) before treating absolute throughput projections as final.**

## Method

`scripts/research/gnn_infer_bench.py` — 320 real self-play positions (games_2026-07-10/12/14
replay JSONLs, 1380-position pool, seed 42; **not** synthetic, **not** human-corpus), axis-graphs
via `build_axis_graph_raw`, block-diagonal batching, n=10 runs + 5 warmup per cell, median+IQR,
IQR-gated, cuda-synchronized, 1 Hz GPU-util sampling. ONNX opset 18, dynamic axes;
**parity gate PASS** (probe max|Δ|=5.5e-06, prod 6.6e-07 over 24 real graphs).
Raw tables: `gnn_infer_bench_results.md` (+ `gapfill_ort_cuda/` for 3 re-run cells at 6.5 GiB arena).

**Graph-size distribution (self-play ≠ human corpus):** mean 490 nodes / 2932 edges,
p90 729/4796, max 897/5690 — **1.7× the scoping doc's human-corpus prior** (290/1294 mean).
All per-position costs below carry this heavier, correct distribution.

## Headline table (bs=64, ms/position, laptop 4060)

| cell | ms/pos | pos/s | vs CNN |
|---|---|---|---|
| CNN 4.27M CUDA fp16 (today's deploy) | 0.218 | 4595 | 1.0× |
| GNN probe 284k torch-CUDA fp16 | 0.335 | 2982 | 1.54× |
| GNN probe torch-CUDA fp32 | 0.387 | 2582 | 1.78× |
| GNN prod 1.1M torch-CUDA fp16 | 0.696 | 1436 | 3.20× |
| GNN prod torch-CUDA fp32 | 0.879 | 1138 | 4.03× |
| GNN probe ORT-CUDA fp32 | 0.629 | 1590 | 2.89× |
| GNN prod ORT-CUDA fp32 | 1.738 | 576 | 7.98× |
| GNN probe torch-CPU fp32 | 5.32 | 188 | 24× |
| GNN probe ORT-CPU T=8 (best ORT-CPU) | 13.95 | 72 | 64× |

Build-vs-forward:

| timer | ms/pos | note |
|---|---|---|
| Python builder (`build_axis_graph_raw`) | **14.00** | self-play dist; was 5.94 on human corpus — Python path stays disqualified |
| Rust builder proxy (strix `axis_graph.rs`) | **0.539** | 55-pos origin-compat subsample (biased, directional); strix's schema ≈ ours |
| Dense plane-encode (Python `to_tensor`) | 0.030 | Rust `<50 µs` figure cited, not measured |

## Verdicts (pre-registered buckets)

**1. CUDA-WINS.** ORT-CUDA batched at bs=64 beats the best CPU cell by **8.5×** (40.2 ms vs
340.7 ms torch-CPU; threshold was ≥1.3×). GPU util 97–100% from bs=32 up. The CPU-HOLDS
option is dead at self-play scale: best CPU cell = 188 pos/s vs 2982 pos/s torch-CUDA fp16.
Self-play inference targets the GPU. **Literature prior REFUTED:** "small-graph GPU starvation
≤6% util" does not apply to block-diagonal batched inference at bs≥16 (measured 45–100%,
typically ~99%).

**2. BUILD-HOT.** Rust-builder proxy 0.539 ms/pos = **161% of the probe-GNN forward**
(0.335 ms/pos) and **77% of the prod-GNN forward** (0.696 ms/pos) — far past the ≥50%
threshold. → **WP-1 gets a perf sub-package** (capacity reserves first, §S182 lesson;
still NEVER search-time deltas — optimize the once-per-leaf build itself).

**3. TORCH-BEATS-ORT (new, beyond buckets — reshapes design-of-record inference plan).**
ORT loses to torch **on both devices** for this net: CUDA 1.6–2.0× slower (probe 40.2 vs
24.8 ms; prod 111.2 vs 56.2 ms at bs=64), CPU 2.6–2.8× slower (T=8 vs torch's default
threading). Worse, the exported graph's GINE message step (`Expand`+`Slice` decomposition)
**materializes E×hidden intermediates**: prod bs=256 requests ~1.5 GiB in one node and OOMs
the 8 GiB card even with a 6.5 GiB arena, while torch runs the identical batch in 213 ms.
→ **Self-play/training-loop inference should ride torch-CUDA** (also reuses the existing
InferenceServer batching seam); **ORT remains the browser/WASM path only** (onnxruntime-web),
where torch is not an option. Feeds WP-B (contract's hot-path consumer = torch tensors) and
WP-D (wasm op-set: avoid Expand-materializing patterns; memory blow-up scales with E×H).
Caveat: ORT-CUDA was fp32-only (no fp16 export attempted); torch's fp16 gain was ~1.25×, so
fp16 ORT would still lose. A scatter-op-native export could close some gap — un-benched, noted
as a lever, not assumed.

**4. Cost-story answer.** The CPU-derived story ("forward 91–96%, build 4–8%") does NOT
transfer: with a Rust builder and GPU forward, build is **~44% of total per-leaf cost at prod
scale** (0.539 of 1.235 ms) and **~62% at probe scale** (0.539 of 0.874 ms). Forward no longer
dominates once the builder is native and the forward is on-GPU.

## Throughput projection (revised; 4060-derived ratios, 5080 rider applies)

Per-leaf ≈ builder + forward: probe 0.539+0.335 = **0.874 ms/pos** (3.5× CNN's ~0.248);
prod 0.539+0.696 = **1.235 ms/pos** (5.0× CNN). Against run2's sustained 4.4k steps/hr →
**~0.9k steps/hr (prod-scale) / ~1.25k steps/hr (probe-scale)** if inference-bound — **below
the scoping doc's 2.0–3.0k/h projection** (its 1.7× forward assumption was optimistic;
measured 3.2× at prod fp16). Run4 design doc must set the throughput floor from THESE numbers
(then re-set from the 5080 rider run): the realistic options are probe-scale-ish net,
builder+forward perf work (BUILD-HOT package), or accepting ~4–5× fewer steps/GPU-week than run2.

## Caveats

- 4060 Max-Q instrument; ratio-transfer rule + explicit 5080 rider above.
- Rust-builder number is a *proxy* (strix's implementation, biased 55-pos subsample —
  only positions with a stone at (0,0) reconstruct through their public API). WP-1's own
  builder gets first-party benches under the BUILD-HOT package.
- prod-scale proxy here is hidden=256/L=4 GnnBcNet = 1.13M params, NOT strix's 4.25M
  (theirs is wider+deeper with heavier heads); the 3.2× ratio is therefore a LOWER bound
  on a true 4.25M-class forward.
- ORT cells fp32-only; CNN comparator ran autocast fp16 (deploy-realistic for the CNN).
- 3 OOM'd cells re-run in `gapfill_ort_cuda/` at 6.5 GiB arena (probe bs=128 gap-fill read
  111.6 ms unstable vs main run's 83.1 ms — GPU shared with desktop; main-run value used).

## Files

- `scripts/research/gnn_infer_bench.py` (the instrument; runnable unchanged on the 5080)
- `reports/probes/gnn_integration/gnn_infer_bench_results.{json,md}` (raw, incremental-flushed)
- `reports/probes/gnn_integration/gapfill_ort_cuda/` (3 re-run cells)
- `reports/probes/gnn_integration/wpa_positions.json` (frozen position set for the 5080 rider)
