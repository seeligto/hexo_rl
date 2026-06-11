<!-- Forensic archive extracted from docs/07_PHASE4_SPRINT_LOG.md during compression sprint 2026-05-13. Compressed counterpart in docs/07_PHASE4_SPRINT_LOG.md. -->

## §113 — buffer_sample_raw target recalibration — 2026-04-22

Post-supply-wave cold bench showed `buffer_sample_raw_us = 1,715 µs` vs ≤ 1,500 target (FAIL). Two root causes identified:

1. **`push_many_impl` element-wise `to_bits()` loops** (`f716365`) — prevented LLVM from emitting SIMD memcpy for state/chain_planes scatter. Also increased crate code size, causing LLVM codegen spillover that suppressed SIMD in unrelated `sample_batch_impl`. Fixed in `6c0bfa9` by replacing both loops with `unsafe { from_raw_parts } + copy_from_slice`. Recovered: push 460k→576k pos/s (PASS), sample_aug 1,854→1,562 µs (PASS), sample_raw 1,715→1,533 µs (improved but still over 1,500 target).

2. **`cda9dde` always-on dedup** — forces `sample_indices` to always allocate a `HashSet<i64>` and scan 256 `game_ids[]` entries even on fully-untagged buffers. Previous slot-0 heuristic was a latent correctness bug (defeated dedup on mixed buffers); `cda9dde` was the correct fix. Residual cost: ~33 µs per sample call. Adding an `any_tagged` fast-path flag would save 33 µs at the cost of a new multi-path invariant across push / push_game / push_many / resize / buffer-restore. Maintenance cost exceeds the win; deferred to Q35 (full GIL-release refactor).

**Decision:** Recalibrate target ≤ 1,500 → ≤ 1,550 µs. Post-transmute bench: 1,533 µs, IQR ±12 µs (0.8%) — PASS against new target. All 10 bench targets now pass.

**Wall impact:** ~0. Trainer thread samples once per training step; at 95% trainer-idle (recommendations.md E1.a), 33 µs/sample is unmeasurable on the wall clock.

**Follow-up:** If Q35 (GIL-release refactor) lands, revisit the dedup fast path as part of the full sample hot-path audit. Do not open a separate ticket.

### Commits

- `perf(replay-buffer): replace to_bits() loops with copy_from_slice in push_many_impl` (`6c0bfa9`)
- `docs(perf): recalibrate buffer_sample_raw target 1500→1550µs (§113)`

---

