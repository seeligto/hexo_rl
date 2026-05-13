<!-- Forensic archive extracted from docs/07_PHASE4_SPRINT_LOG.md during compression sprint 2026-05-13. Compressed counterpart in docs/07_PHASE4_SPRINT_LOG.md. -->

## §107 — Post-W1 sustained run launch + live investigation instrumentation (2026-04-19)

**Launched.** `gumbel_targets` (laptop) from `bootstrap_model.pt`, 50k steps, tsp=1.5 (rebaselined 2.0→1.5 per preflight Pareto pick; all three smokes supply-bottlenecked at 14 workers ≈0.145 games/s). §101.a graduation params held (D2=5000, D4=400). Projected wall ≈52 h vs prompt 35 GPU-h target — laptop hardware ceiling.

**Live instrumentation (residuals R1 colony-extension + Q2/Q27 attention-hijack).**

- **I1 colony-extension (Python).** `pool.py::_compute_colony_extension` walks `move_history` at `game_complete`, counts stones >6 hex-dist from any opponent stone. Adds `colony_extension_stone_count/total/fraction` to payload. <1 ms/game.
- **I2 per-cluster variance (Rust).** Three `AtomicU64` fixed-point accumulators on `SelfPlayRunner` updated in `infer_and_expand` when K≥2: population std of per-cluster values pre-min-pool + `1−(top1-majority/K)` policy disagreement. Lifetime means emitted on `iteration_complete`. Rust-side because cluster structure is consumed by batcher before Python.
- **Gating.** `monitoring.log_investigation_metrics` (default true; off on bench).
- **Dashboards.** Web "Live Investigation" card (3 rows) + terminal summary line; schema in §08.

**Commits.** 77699f1 (I1), 59c0964 (I2), 914518f (tests), 17ef5ee (dashboard+schema), ed5d3b5 (tsp 2.0→1.5).

**Abort gates.** wr_random<0.90 ×2; NaN >10 steps; `colony_extension_fraction>0.80` 500+ games; `policy_entropy_selfplay<1.0` 500+ steps; OOM/disk.

**Status at close.** Residuals R1 + Q2/Q27 OPEN pending live data. Forward: §108 (desktop `gumbel_full` companion launch + b35de20 JSONL mirror for I1/I2); §109 (Q33 selfplay-entropy diagnostic; pe_self≈5.36 fixed point investigation through §110 Q33-B / §111 Q33-C HALT).

---

