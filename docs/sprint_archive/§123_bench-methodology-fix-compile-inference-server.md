<!-- Forensic archive extracted from docs/07_PHASE4_SPRINT_LOG.md during compression sprint 2026-05-13. Compressed counterpart in docs/07_PHASE4_SPRINT_LOG.md. -->

## §123 — Bench methodology fix: torch.compile + InferenceServer threading

**Date:** 2026-04-25  
**Commits:** `654da65`, `c26b9b4`, `e88032b`

### Problem

`make bench` (via commit `c399d41`) had `--no-compile` added, meaning it no longer measured production config. Three bench metrics were failing as a result.

Separately: when `--no-compile` was removed and compile re-enabled, all worker pool games completed with `plies=0`.

### Root cause: cudagraph_trees TLS

`torch.compile(mode="reduce-overhead")` uses `cudagraph_trees` internally. It stores the CUDA graph tree manager in **C++ dynamic TLS** (`torch._C._set_obj_in_tls`). TLS is per-thread. The bench passes the compiled model to `InferenceServer`, which runs in a background thread. That thread's TLS is uninitialized, so every call hits `AssertionError` in `cudagraph_trees.get_obj` → silent exception caught by `InferenceServer`'s inner handler → Rust `submit_inference_failure` → game loop returns 0 → no moves applied → 0-ply games.

### Fix

**pool_model** (the model given to `InferenceServer`) is compiled with `mode="default"` instead of `reduce-overhead`. `default` applies inductor kernel fusion but no CUDA graph capture — thread-safe from any thread. The NN inference benchmark still uses `reduce-overhead` (main thread only), preserving the production throughput measurement.

### Second bug: JIT warmup isolation

`reduce-overhead` and `default` modes produce **different compiled artifacts** (different inductor cache keys). The JIT warmup paid for `model` (reduce-overhead) does not cover `pool_model` (default). Without an explicit warmup, pool_model's first InferenceServer call triggered ~90s of JIT compilation inside the 90s pool warmup window → 0 games during warmup → IQR ±126k.

Fix: `compile_warmup(pool_model, ...)` called from main thread after pool_model creation. Safe because `mode="default"` has no CUDA graph TLS constraint.

### Takeaway for any multi-threaded compiled model use

If a compiled model (`reduce-overhead`) is called from a background thread, compile it with `mode="default"` instead — or ensure the background thread is the *first* caller (never called from main thread before the thread starts). Pay each mode's JIT cost separately from the main thread before the background thread starts.

### Bench result (2026-04-25, all PASS)

| Metric | Result | Target |
|---|---|---|
| MCTS sim/s | 72,711 | ≥26,000 |
| NN inference pos/s | 7,931 | ≥6,500 |
| NN latency ms | 0.51 | ≤3.5 |
| Buffer push pos/s | 621,156 | ≥525,000 |
| Buffer raw us | 1,374 | ≤1,550 |
| Buffer aug us | 1,356 | ≤1,800 |
| Worker pos/hr | 171,241 | ≥142,000 |
| Worker batch fill | 99.4% | ≥84% |

---

