<!-- Forensic archive extracted from docs/07_PHASE4_SPRINT_LOG.md during compression sprint 2026-05-13. Compressed counterpart in docs/07_PHASE4_SPRINT_LOG.md. -->

## §125 — EPYC 4080S validate sweep verdict + profiling methodology shift

**Date:** 2026-04-25
**Branch:** master (a85f895 + config update in same session)

### Validate sweep result (6-cell, n=5, 180s/cell, trace ON)

`MODE=validate bash scripts/sweep_epyc4080.sh` on EPYC 7702 + RTX 4080 Super.
Full data: `~/Downloads/hexo_sweep/sweep_{workers,batch_wait,leaf_burst}_2026-04-25_*.csv`

| Config | pos/hr median | GPU util | batch fill |
|---|---|---|---|
| n_workers=16, b=128, wait=4 | 234,526 | 65 % | 83.4 % |
| n_workers=20, b=128, wait=4 | 196,491 | 65 % | 98.8 % |
| n_workers=24, b=128, wait=4 | 375,240 | 65 % | 99.5 % |
| n_workers=24, b=128, wait=4 (stage 2) | 207,027 | 66 % | 100 % |
| n_workers=24, b=192, wait=4 | 364,645 | 65 % | 91.5 % |
| **n_workers=24, b=192, wait=4, leaf=8, burst=16** | **376,793** | **66 %** | **96.6 %** |

Pre-fix best (19-cell full sweep, trace OFF): **388,426** at n_workers=16.

**Verdict: trace neutral on EPYC 4080S.** GPU util locked at 65% pre- and
post-trace — dispatch elimination did not unblock the GPU bottleneck on this
hardware. Best post-fix median (377k) is within noise of the prior observed
best (~370k in the variant config comment). The dispatch hypothesis from
§124/py-spy is correct for 3070 (GPU compute > dispatch) but incomplete for
EPYC 4080S (removing dispatch reveals a different binding constraint).

**Why n_workers=16 regressed 388k→234k with trace:** trace accelerates NN
dispatch, so the GPU drains each batch faster. With only 16 workers, the
workers can't refill the 192-position batch before the GPU is idle → fill
drops 97%→83% → throughput collapses. n_workers=24 compensates by keeping
the batch filled despite faster turnover. Optimal worker count shifted up
due to the trace acceleration.

### Config update

`configs/variants/gumbel_targets_epyc4080.yaml`:
- `n_workers`: 20 → **24** (validate sweep winner)
- `inference_batch_size`: 192 (confirmed)
- `inference_max_wait_ms`: 4.0 (validate ran at 4.0; batch fill 96-99% → no
  benefit from longer wait; 8.0 was calibrated to a compile-path with ~84% fill)
- `max_train_burst`: 32 → **16** (winning cell value)
- Best benchmark comment updated to ~377k.

### Profiling methodology shift: py-spy → built-in perf_timing

`py-spy 0.4.2` is the latest published release and does not support
Python 3.14 (`No python processes found` error — version cannot parse
3.14 memory layout). Waiting for py-spy maintainers is not actionable.

**Replacement:** `diagnostics.perf_timing: true` in config enables
per-batch structured logging in `InferenceServer._run`:

```
inference_batch_timing  fetch_wait_us=…  h2d_us=…  forward_us=…  d2h_scatter_us=…
```

`fetch_wait_us` = queue wait (workers starving?)  
`h2d_us` = host→device copy  
`forward_us` = traced graph execution (GPU compute if `perf_sync_cuda=true`)  
`d2h_scatter_us` = device→host + scatter to waiters

Profiling script: `scripts/profile_epyc_pyspy.sh` (gitignored). Runs the pool
with `perf_timing=true`, `perf_sync_cuda=true` (serialises CUDA stream →
~30-50% pph drop during profile, but gives accurate phase split), then
parses the log and prints a percentile table. Saves to `reports/profile/epyc_perf_*.{log,txt}`.

Key question for next profile run: with trace eliminating Python dispatch,
does `forward_us` now dominate (GPU-bound, expected) or does `fetch_wait_us`
dominate (workers starving)? If `forward_us` < 20% of total and GPU is at 65%,
the remaining wall time is Rust-side (MCTS lock contention, result queue
crossing) — not visible via any Python profiler; use `perf stat` IPC metric.

### Profiling result (2026-04-25, EPYC 4080S, n_workers=24, batch=192, trace ON)

`reports/profile/epyc_perf_20260425_2132_{log,summary}.txt` — 15,890 batches.

| Phase | p50 | p90 | p99 | share |
|---|---|---|---|---|
| fetch_wait | 1.630 ms | 1.941 ms | 4.204 ms | 11.0 % |
| H2D | 1.016 ms | 3.673 ms | 12.477 ms | 6.8 % |
| **forward** | **11.959 ms** | **12.176 ms** | **27.978 ms** | **80.4 %** |
| D2H+scatter | 0.277 ms | 0.328 ms | 0.542 ms | 1.9 % |
| **Total cycle** | **14.882 ms** | | | |

batch_n: p50=192 p10=132 p90=192 — batch nearly always full at n_workers=24.

**The dispatch hypothesis was wrong for 4080S.** Forward = 80.4 % of cycle.
This box is GPU-compute-bound, not dispatch-bound. The "3-4 ms GPU compute"
estimate extrapolated from 3070 py-spy was incorrect; actual FP16 forward
at batch=192 with 12 ResBlocks + GroupNorm + SE + 7 heads = ~12 ms (no
kernel fusion without compile). The 65 % GPU util in the sweep is:
inference GPU share (80 %) spread across a 14.9 ms cycle = 80 % inference
GPU busy; nvidia-smi measures ~65 % because training also consumes GPU on
the same card.

Math check: 1000 ms / 14.9 ms × 192 / 200 sims × 3600 s = **232 k pos/hr**
with perf_sync_cuda overhead. Corrected for sync overhead (~35 %): **357 k ≈ 377 k
benchmark** — consistent.

**Next lever: `torch.compile` on top of trace.**

compile gave +45 % NN throughput in §123 tests. With forward = 80 % of cycle:
+45 % on forward → +36 % overall → 377k × 1.36 ≈ **512 k pos/hr** theoretical.

Previous ruling (compile regresses selfplay) was measured WITHOUT trace.
At that time the bottleneck was Python dispatch; compile does not remove
dispatch overhead, so it couldn't help and may have added Dynamo guard cost.
Now trace eliminates dispatch, and forward is the binding term — compile
can fuse the CUDA kernels that trace cannot. The two are complementary:

- trace: eliminates `_call_impl` Python overhead per forward (done)
- compile: fuses conv/GN/SE kernels into fewer, faster CUDA launches

Stack path: `torch.compile(model, mode="reduce-overhead")` then
`torch.jit.trace(compiled._orig_mod, example)` — verified possible in §124
follow-up note, not yet implemented. Needs:
1. Benchmark to confirm +45 % NN speed survives trace unwrap
2. Confirm no regression in weight-swap path
3. Confirm compile mode is `reduce-overhead` (not `default`) to avoid Dynamo guard per-call

This is now the **highest-value open lever** for EPYC 4080S throughput.

---

