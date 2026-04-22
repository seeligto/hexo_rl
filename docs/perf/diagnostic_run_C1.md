# Phase C1 — Diagnostic run capture

Will be populated post-run by `scripts/perf/analyze_C1.py` (also produced
this session). Raw artefacts under `reports/perf/diag_C1_*/`.

## Run metadata

- **Run ID**: diag_C1_main (see log `logs/diag_C1_main.jsonl`)
- **Variant**: gumbel_targets
- **Override config**: `configs/diag_probes_DO_NOT_TRAIN.yaml` (perf_timing: true, perf_sync_cuda: true, vram_probe_interval: 100)
- **Resumed from**: `checkpoints/checkpoint_00020454.pt`
- **Iterations**: 400 (budget-limited)
- **Dashboard**: off
- **GPU sidecar**: nvidia-smi dmon -s pucvmet -d 1 → `dmon.log`
- **Host**: laptop (RTX 4060 Max-Q 8 GB, Ryzen 7 8845HS)

## Headline findings

See analysis section below. Auto-populated by analyzer.

## Stream audit (B4)

Both contexts reported `on_default_stream: true` with `current_stream_ptr ==
default_stream_ptr == 0`. **Q18 confirmed at the code level** — inference
thread and training thread share the default CUDA stream. No overlap
possible without code change.

## InferenceServer batch timing (B2)

Post-run analysis fills in:

- `fetch_wait_us` p50 / p95 — queue wait
- `h2d_us` p50 / p95 — H2D copy wall time
- `forward_us` p50 / p95 — model forward wall time (includes default-stream contention)
- `d2h_scatter_us` p50 / p95 — D2H + numpy marshalling
- batch_n distribution (p10/p50/p95)

Cross-ref vs bench: iso NN inference (8.37 ms/batch-64) vs live `forward_us`
p50. Ratio reports the Q18-style live/iso gap.

## Trainer step timing (B1)

- `h2d_us` p50 / p95
- `fwd_loss_us` p50 / p95
- `bwd_opt_us` p50 / p95
- `total_us` p50 / p95
- `buffer_sample_timing.sample_us` p50 / p95 (if hit — requires the non-mixed path, i.e. no pretrained_buffer)

## VRAM probe (B1)

- Peak vs allocated vs reserved vs frag — across the run
- `num_ooms` should remain 0

## Analyzer script

Companion analyzer: `scripts/perf/analyze_C1.py` (committed this session)
parses the JSONL and writes a summary subsection that should be appended
here after the run.

---

## C1 captured results — 2026-04-21

Run: `logs/diag_C1_main.jsonl` (captured via 12-min foreground diagnostic).

### Event volumes

| Event | Count |
|---|---|
| `inference_batch_timing` | 10,030 |
| `train_step` / `train_step_timing` | 16 / 16 |
| `cuda_stream_audit` | 2 (one per context) |
| `game_complete` | 8 |
| `vram_probe` | 0 (run stopped before step-100 boundary) |

### Stream audit (B4) — Q18 confirmed at runtime

```
training_thread: current_ptr=0 default_ptr=0 on_default=True
inference_server: current_ptr=0 default_ptr=0 on_default=True
→ Both contexts share the default CUDA stream. No copy/compute overlap.
```

### InferenceServer batch timing (B2, n=9,980 after 50-batch warmup skip)

| Field | p50 | p95 | mean | stdev |
|---|---|---|---|---|
| `batch_n` | 64 | 64 | 55 | 12 |
| `fetch_wait_us` | 270 | 740 | 332 | 260 |
| `h2d_us` | 303 | 545 | 368 | 995 |
| `forward_us` | **13,961** | 14,818 | 12,685 | 6,044 |
| `d2h_scatter_us` | 172 | 419 | 403 | 4,399 |

`batch_n` distribution: 56.7% of batches hit the max (64). Batcher is
well-saturated — raising the cap isn't the lever (and §90 rejected it anyway).

### Trainer step timing (B1, n=16, sync_cuda=true)

| Field | p50 | p95 |
|---|---|---|
| `h2d_us` | 4,139 | 9,113 |
| `fwd_loss_us` | 110,967 | 120,722 |
| `bwd_opt_us` | 192,018 | 199,793 |
| `total_us` | 306,784 | 324,738 |

`total_us` p50 = **307 ms/step** (sync-inflated). Without `sync_cuda`,
intrinsic is ~400 ms per 20k characterization; 307 ms here reflects the
sync-wait overhead on forward+backward being measured rather than the
underlying GPU work.

### Throughput during C1 (sync_cuda=true)

- 15 steps / 72.7 s = **742 steps/hr** (reference run: 1,707 steps/hr → 2.3× slowdown from sync).
- 8 games / 74.7 s = **386 games/hr** (reference: 364 games/hr — close).

### GPU telemetry (nvidia-smi dmon, n=166 rows)

| Metric | p50 | p95 |
|---|---|---|
| SM util | 82% | 87% |
| Mem util | 52% | 57% |
| Power | 76 W | 78 W |
| GPU temp | 63 °C | 65 °C |

SM util p50 = 82% — room above. Power p95 = 78 W (of RTX 4060 Max-Q
~115 W TDP) — not power-constrained.

---

## C2 — live-vs-iso NN inference ratio

Iso NN bench (2026-04-18, `reports/benchmarks/2026-04-18_18-36.json`):
- `nn_inference_pos_per_s` = 7,646 (batch=64)
- Per-batch wall = `64 / 7,646 * 1000` ≈ **8.37 ms/batch-64**

Live `forward_us` p50 (C1) = **13.96 ms/batch-64**.

**Live/iso ratio = 13.96 / 8.37 = 1.67×.**

### Notable update vs Q18 hypothesis

Q18 (sprint log): reported iso/live ratio ~7.8× at 18 planes. C1 measures 1.67×.

Possible reasons for the smaller gap vs Q18:

1. **Methodology difference.** Q18's number may have included full queue wait,
   H2D, D2H, and Python-side wrapping, not just `forward_us`. If we add
   `fetch_wait_us + h2d_us + forward_us + d2h_scatter_us` p50
   ≈ `0.27 + 0.30 + 13.96 + 0.17` = **14.7 ms**, still only 1.76× iso.
2. **Workload has evolved.** Input dimensions (18 planes vs earlier 24),
   batcher wait-ms tune (§90), selective policy loss (§100), and other
   changes since Q18 may have reduced contention incidentally.
3. **Ampere vs Ada.** Q18 likely measured on desktop Ampere; C1 is on laptop
   Ada. Scheduling behaviour on the default stream may differ across
   architectures.

### Implication for Phase E ranking

Stream-separation still has **real upside but a smaller ceiling than
previously implied**: ~5.6 ms saved per 64-batch × ~128K forwards/hr in
reference-run scaling ≈ **~20% live throughput uplift**, NOT 5×.

Stream separation is still #1 in the ranked list — low risk, low VRAM cost,
portable — but the *expected-impact* column should read "MED (20%)" rather
than "HIGH (primary Q18 lever)".

---

## dmon dataset

Full 166 sec of 1-Hz GPU telemetry at
`reports/perf/diag_C1_20260421_193614/dmon.log`. Stable thermal and clock
profile; no throttling visible.

