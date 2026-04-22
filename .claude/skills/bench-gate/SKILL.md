---
name: bench-gate
description: >
  Run before any perf-sensitive commit lands. Trigger phrases include "perf()",
  "feat(perf)", "MCTS edit", "buffer edit", "replay buffer change", "inference
  batch", "worker pool change", "hot path", "bench regression", "gate before
  commit", "torch.compile re-enable". Fires automatically for any commit touching
  engine/src/mcts/**, engine/src/replay_buffer/**, engine/src/game_runner/**,
  engine/src/inference_bridge.rs, or NN training hot paths. Ensures `make bench`
  baseline → change → post-change bench → diff table → gate decision before the
  commit is allowed to land.
---

# Bench gate

Discipline for landing perf-sensitive changes without silently regressing the
10-metric bench target matrix. Single source of truth for targets lives at
`docs/rules/perf-targets.md` — this skill is the procedure, not the table.

## When to use

- Any edit to `engine/src/mcts/**` (PUCT, node pool, virtual loss, Gumbel search).
- Any edit to `engine/src/replay_buffer/**` (push, sample, aug tables, persist).
- Any edit to `engine/src/game_runner/**` (worker loop, inference bridge wiring).
- Any edit to `engine/src/inference_bridge.rs` (GPU queue, batching).
- NN hot-path changes in `hexo_rl/model/` or `hexo_rl/training/` that affect
  fwd/bwd shape.
- Re-enabling `torch.compile` — currently disabled per Python 3.14 CUDA-graph
  incompatibility (sprint §25/§30/§32).

## Steps (5-stage gate)

1. **Capture baseline.** `make bench` on current HEAD. Archive the JSON output
   under `reports/benchmarks/<date>_pre.json`. If a recent run already exists
   (same HEAD, same host, under 24h old), reuse it.

2. **Apply change.** Make the code edit. Do not proceed to step 3 until `cargo
   build --release`, `cargo test`, and `make test.py` pass.

3. **Post-change bench.** `make bench` again. Archive to
   `reports/benchmarks/<date>_post.json`.

4. **Diff table.** Build a per-metric % delta table against the baseline. Call
   out which metrics crossed the target threshold (FAIL) and which regressed
   by >5% (WATCH).

5. **Gate decision.**
   - Any target FAIL → revert unless commit body explicitly justifies and
     flags for follow-up.
   - Any >5% regression on a passing metric → revert or commit with explicit
     "accepted regression" note in body + sprint-log § pointer.
   - Methodology note: single runs can swing ±10% on AMD boost-clock hosts.
     If a single post-run looks bad, re-run n=5 before reverting (see §102).

## Anti-patterns

- **Skipping baseline** ("it can't regress, it's a one-line change"). Every
  perf wave has one edit that surprises the bench — the baseline is free
  insurance.
- **Committing single-run bench data.** Bench methodology requires n=5 median
  with warmup (3s MCTS / 3s NN / 2s buffer / 90s worker pool). A single hot
  run is noise, not data.
- **Tightening a target on a single hot run.** §102 rule: do not tighten
  targets on one run. New targets use `min(observed_median × 0.85, prior)`
  and only after multiple confirmation runs.
- **Accepting an unflagged regression.** "It still passes the target" is not
  an excuse if the delta is >5% — regressions compound over a sprint.

## Target table

Single source of truth: `docs/rules/perf-targets.md`. Do NOT duplicate the
10-metric table in this skill body — link it instead. If targets drift, the
rule file is authoritative.

## Reference gate runs

- **§113 `push_many_impl` recalibration (2026-04-22)**: perf commit on the
  ReplayBuffer push path tripped a buffer_sample_raw regression. Diff table
  showed +33 µs residual after upstream fixes landed; gate surfaced the need
  to recalibrate the target from 1,500→1,550 µs (commit 3165fab) rather than
  revert the correctness-required dedup change (commit cda9dde). Worked
  example of the "accepted regression" path with explicit sprint-log
  justification.
