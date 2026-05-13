<!-- Forensic archive extracted from docs/07_PHASE4_SPRINT_LOG.md during compression sprint 2026-05-13. Compressed counterpart in docs/07_PHASE4_SPRINT_LOG.md. -->

## §104 — D-Gumbel / D-Zeroloss instrumentation (2026-04-17)

**Motivation.** Post-§100 the dashboard could not answer two questions without
guessing:

- **D-Gumbel** — `completed_q_values: true` produces a structurally valid
  policy target even at 100 sims (`engine/src/mcts/mod.rs:266-276` —
  `softmax(log_prior + sigma · completed_q)` over all legal actions). The
  §100 selective gate keys only on `is_full_search`, not on target type, and
  drops those quick-search CQ targets from the policy gradient. Whether that
  is leaving usable signal on the floor is an empirical question.
- **D-Zeroloss** — `trainer.py:518-522` logs `full_search_frac` but cannot
  distinguish `policy_loss == 0 because mask selected no rows` from
  `policy_loss == 0 because loss was numerically zero`. Known follow-up
  from §100 "Known follow-ups".

Both require per-step policy-target diagnostics that were not being emitted.

**Changes.** Monitoring-only. No behaviour change.

- `hexo_rl/training/trainer.py`:
  - New module-level `compute_policy_target_metrics(target_policy,
    policy_valid, full_search_mask)` returning 7 fields split by
    `is_full_search`: `policy_target_entropy_{full,fast}search`,
    `policy_target_kl_uniform_{full,fast}search`,
    `frac_fullsearch_in_batch`, `n_rows_policy_loss`, `n_rows_total`.
    All reductions stay on device; a single `.cpu().tolist()` over 7
    packed scalars replaces 7 `.item()` syncs — under 200 µs / call on
    CUDA at (B=256, A=362).
  - NaN is a first-class signal: when the full-/fast-subset has zero rows
    the mean comes back NaN, and renderers handle that explicitly. Keeps
    the decision rules "H_fast(CQ) ≥ some bound" readable even when a batch
    lands entirely in one bucket.
  - Gated via `monitoring.log_policy_target_metrics: true` (default on).
  - NaN-loss guard pre-populates the 7 keys.
- `hexo_rl/training/loop.py`: forwarded all 7 keys onto the `training_step`
  emit_event payload and onto the `log.info("train_step", ...)` structlog
  entry, so the same values land on the dashboard and in
  `logs/<run_name>.jsonl` for post-hoc analysis.
- `hexo_rl/monitoring/terminal_dashboard.py`: new `policy target` row below
  the entropy line — `H_full / H_fast │ KL_u_full / KL_u_fast │ n_full/total`.
- `hexo_rl/monitoring/static/index.html`: ring-buffer carries the 7 keys and
  the loss ratio strip gains compact `H_full / H_fast / KL_u_fast / n_full`
  segments. No new Chart.js panels — deliberately minimal web wiring.
- `docs/08_DASHBOARD_SPEC.md`: §2.1 schema updated with the 7 new keys +
  value ranges + NaN-as-signal note. §7 adds
  `monitoring.log_policy_target_metrics` config key. Changelog entry.
- `configs/monitoring.yaml`: default-true gate.

**Tests.** `tests/test_policy_target_metrics.py` — 5 synthetic-batch cases:

1. Uniform-vs-one-hot split — verifies the math: H_full ≈ log(362) ≈ 5.89,
   H_fast ≈ 0, KL_u_full ≈ 0, KL_u_fast ≈ log(362).
2. All full-search — fastsearch metrics must be NaN; emit does not raise.
3. All fast-search — symmetric.
4. Empty valid mask — all 4 means NaN, counts 0, every promised key present.
5. Cost budget — <200 µs/call on CUDA at (B=256, A=362) after the single-sync
   optimisation (CPU fallback: <1000 µs).

`tests/test_trainer.py` was updated to allow NaN on the two new fastsearch
keys when the batch carries no quick-search rows (default path).

**Bench check.** `reports/instrumentation_bench_check_2026-04-17.md`.
Instrumentation-reachable metrics all within ±5.5% of the 2026-04-17 09:34
baseline (MCTS −1.7%, NN inference −0.3%, NN latency −22% faster, buffer
push −5.5%, buffer sample ±5-12% within IQR). Worker-pool throughput
regressed ~36% but `benchmark.py` does not construct a `Trainer` — the
instrumentation is not in that call path, and worker-pool has a historical
±40% IQR on this harness (§98 caveat).

**Decision support.** `reports/gumbel_target_quality_2026-04-17.md` — two
smokes from `bootstrap_model.pt` (`baseline_puct`, `gumbel_targets`) and a
per-variant mean table with the Option A / B / inconclusive mapping from
the prompt brief.

**Verdict: Option A.** Quick-search completed-Q targets on `gumbel_targets`
drift toward uniform as training progresses (steady-state ΔH = H_fast −
H_full ≈ **+3.5 nats**, well above the +1.5 threshold; KL_u_fast falls
from 5.3 → 1.1 over steps 10–60). The §100 selective gate correctly
discards noisy quick-search CQ targets. When the `gumbel_full.yaml` mutex
bug (`reports/selective_policy_audit_2026-04-18.md` §4 B1) is unblocked,
the repair should follow the audit's Option A (drop legacy game-level
`fast_prob`, keep move-level `full_search_prob` from base).

**Caveats.** 20 metric events for baseline (full 200 steps); only 7 for
gumbel_targets (run stopped at ~step 83 — per-move 600-sim cost pushed a
full 200-step run past a reasonable wall-time budget). Steps 10–20 on
gumbel_targets are corpus-dominated warmup (ΔH small; excluded from the
call). `gumbel_full` not measured — mutex-blocked and `gumbel_targets`
shares the relevant CQ target construction path.

**Follow-up applied same day.** `configs/variants/gumbel_full.yaml` Option A
landed — `fast_prob: 0.25 → 0.0`, keeping the base's `full_search_prob: 0.25`
move-level cap. Mutex resolved at pool init; the desktop Exp E variant is
launchable again. `tests/test_variant_configs.py::test_gumbel_full_passes_playout_cap_mutex`
pins the resolved config so the next base-config drift cannot silently
reintroduce the bug.

**Resolves.** §100 "Known follow-ups" item 3 (distinguish empty-mask vs
genuine 0.0 policy loss — `n_rows_policy_loss == 0` vs `> 0` does it).
§101 gains a telemetry hook for future graduation-gate D-Gumbel validation.

### Commits

- `feat(monitoring): policy target entropy/KL split by is_full_search`
- `test(monitoring): synthetic batch assertions for new metrics`
- `docs(dashboard): add policy target metrics to emit schema`
- `docs(sprint): §104 Gumbel target quality instrumentation + decision support`

