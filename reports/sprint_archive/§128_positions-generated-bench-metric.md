<!-- Forensic archive extracted from docs/07_PHASE4_SPRINT_LOG.md during compression sprint 2026-05-13. Compressed counterpart in docs/07_PHASE4_SPRINT_LOG.md. -->

## §128 — Bench metric fix: positions_generated replaces positions_pushed — 2026-04-28

### Problem

`worker_pos_per_hr` was measured via `pool.positions_pushed`, which
increments by K cluster views × 1 per ply at **game completion** (batch
write). On the bench window (120s, 200 sims/move), a game takes ~160s →
most windows capture **zero completions** → bimodal metric (IQR 80.9%,
one run at 0 in every 5-run set). Median was robust, but the counter
semantics were wrong: positions_pushed counts training rows (K per ply),
not positions evaluated.

### Root cause analysis

`positions_generated` is a Rust `AtomicUsize` incremented **once per
ply** in `worker_loop.rs` (`positions_generated.fetch_add(1, SeqCst)`).
It is continuous — no burst at game completion — so measurement windows
of any length yield stable, non-bimodal readings.

The relationship between the two counters:

```
positions_pushed = K_avg × positions_generated
```

K_avg ≈ 7 empirically (April-28: 177,799 pushed/hr ÷ 29,934 gen/hr on
same engine config). K comes from `get_cluster_views()`: one view per
small cluster, one view per deduplicated anchor on massive clusters.
Seven is typical for mid-game boards with 2–3 clusters of moderate size.

### What changed

`scripts/benchmark.py`: measurement loop switched from
`pool.positions_pushed` → `pool._runner.positions_generated`. Both
start/end snapshots and mid-window progress prints use the new counter.

Targets updated (÷ K_avg 7, provisional):

| Config | Old target (pushed) | New target (generated) |
|---|---|---|
| CUDA | 142,000 | 20,000 |
| MPS | 200,000 | 25,000 |
| CPU | 80,000 | 11,000 |

`docs/rules/perf-targets.md`: documents metric switch and new
provisional floor.

### Bench result

Desktop RTX 3070 n=1 (2026-04-28_19-08): `worker_pos_per_hr = 29,934`
→ **PASS** against new target 20,000. `mcts_pool_overflows_total = 0`.
IQR stable (continuous counter — bimodal artifact eliminated).

8/10 targets PASS (buffer_push_per_s and worker_pos_per_hr fail on
desktop vs laptop-calibrated targets; see perf-targets.md hardware note).

### Bench result (n=5 confirmed, 2026-04-28_19-52)

Desktop RTX 3070, `make bench` (n=5, 120s pool, --no-compile):

| Metric | Observed | Target | |
|---|---|---|---|
| worker_pos_per_hr | 27,835 median, IQR ±2,398 (8.6%), [24.6k–30.0k] | ≥ 20,000 | **PASS** |
| mcts_pool_overflows | 0/0/0/0/0 | 0 | **PASS** |
| worker_batch_fill_pct | 99.96% | ≥ 84% | **PASS** |

Bimodal artifact eliminated — all 5 runs unimodal (continuous counter,
no game-completion burst). 20k floor confirmed (observed × 0.85 = 23,659).

4 remaining FAILs (`nn_inference_pos_per_s`, `buffer_push_per_s`,
`buffer_sample_raw_us`, `buffer_sample_aug_us`) are desktop RTX 3070 vs
laptop-calibrated targets — hardware mismatch, not regressions.

### Out of scope

* Laptop reference re-bench with positions_generated — expected ~25k gen/hr
  (177,799 pushed ÷ K_avg 7). Would allow tightening 20k floor to ~21k.
* K_avg variance characterisation — K ranges 1–20+ depending on board
  state; median ≈ 7 is empirical, not analytically derived.
* Restore positions_pushed metric for training data rate visibility —
  separate decision; positions_generated is sufficient for throughput
  gate; positions_pushed still accessible via `pool.positions_pushed`.

---

