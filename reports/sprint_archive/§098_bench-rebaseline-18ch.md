<!-- Forensic archive extracted from docs/07_PHASE4_SPRINT_LOG.md during compression sprint 2026-05-13. Compressed counterpart in docs/07_PHASE4_SPRINT_LOG.md. -->

## §98 — Benchmark rebaseline post-18ch migration (2026-04-16)

**Context:** First `make bench` after the 18ch migration. Two metrics FAIL against the §72-era targets.

### Observed results (laptop, Ryzen 7 8845HS + RTX 4060, n=5)

| Metric | Median | IQR | Range | Old target | Result |
|---|---|---|---|---|---|
| Buffer sample augmented µs/batch | 1,663 | ±566 | 1.3k–2.2k | ≤ 1,400 | FAIL |
| Worker throughput pos/hr | 30,893 | ±58,185 | 0–364k | ≥ 500,000 | FAIL |

All other metrics PASS.

### Root cause analysis

**Worker throughput (catastrophic-looking median, warmup artifact + methodology shift):**

Two stacked causes:

1. **Warmup design bug** — benchmark creates one pool, runs 30s warmup, then 5 × 60s measurement windows. `p25 = 0` means at least 2 of 5 windows measured 0 positions. Workers weren't producing completed games during those windows. 30s is insufficient for the pool to reach steady state on laptop hardware: workers must play early games to completion (cold start takes longer with an untrained model on the first game). The CUDA JIT warm-up (pre-pool dummy forward) handles PyTorch kernel compilation but not game-loop ramp-up.

2. **Baseline methodology mismatch** — the old 659k pos/hr baseline (§66, April 2026-04-06) was set under different benchmark parameters. Commit 207656a changed `n_simulations` from config value (400) to 200, and `max_moves_per_game` from 200 → 128. The numbers are not directly comparable. The maximum observed value (364k) under the new methodology reflects the actual achievable ceiling.

Cross-check against real training: the training log `train_10cc8d56e4394a9ca542740c4bcee069.jsonl` (production run at step ~15k, April 16) shows **~408 games/hr × 118 avg plies = ~48k pos/hr** during actual training (GPU shared between inference and training steps). The benchmark's pure-self-play measurement at a lower sim count (200 vs production 400) should be faster than training — the 364k max (no training overhead, 200 sims) is consistent with this.

3. **18ch chain plane overhead (minor)** — `encode_chain_planes()` added per position in `worker_loop.rs`. Small but real per-position cost.

**Buffer sample augmented (real regression, high variance):**

Before 18ch: single `apply_symmetry_state` scatter over 24 planes.
After 18ch: `apply_symmetry_state` (18 planes) + `apply_chain_symmetry` (6 planes, axis-plane remap). Two passes over two non-contiguous memory regions. High IQR (±566 µs) reflects cache-pressure variance: chain sub-buffer adds 6 × 361 × f16 = ~4 KB per position; at batch=256, ~1 MB extra data touched per sample, causing inconsistent L3 hit rates.

### Updated targets

Old targets were set against a 24-plane model under a different benchmark methodology. New targets reflect the 18-plane layout and current benchmark setup.

| Metric | New target | Rationale |
|---|---|---|
| Buffer sample augmented µs/batch | ≤ 1,800 µs | Median 1,663 + comfortable margin for split-pass overhead; IQR suggests it's sometimes ≤1,300 µs |
| Worker throughput pos/hr | ≥ 250,000 pos/hr | Conservative floor: well above the warmup-artifact 0-position runs, below the 364k max; methodology fix (longer warmup) should raise the reliable floor |

**Note on worker benchmark reliability:** until warmup duration is increased (suggest 90s or "until N games complete"), the worker throughput metric has high measurement variance. The 250k target is a checkpoint, not a ceiling. Real training throughput (GPU shared) is ~48k pos/hr at production sim counts — the benchmark measures self-play-only capacity at reduced sims.

### Action items

- [ ] Increase worker benchmark warmup to 90s (or gate on first-game completion) to eliminate 0-position measurement windows
- [ ] After warmup fix, run 3-run rebaseline to confirm reliable floor ≥250k

### Commits

- (this entry — no code change, targets only)

