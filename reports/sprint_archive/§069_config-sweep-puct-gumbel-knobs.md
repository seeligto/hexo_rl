<!-- Forensic archive extracted from docs/07_PHASE4_SPRINT_LOG.md during compression sprint 2026-05-13. Compressed counterpart in docs/07_PHASE4_SPRINT_LOG.md. -->

### §69 — Config Sweep 2026-04-08 — PUCT/Gumbel Knob Ranking

> **Historical, superseded by §90 (2026-04-13).** §90 is now the authoritative
> laptop `gumbel_targets` throughput baseline at HEAD (post-refactor, post-A1,
> post-A3). §69's P3 winner config remains the live config but the surrounding
> measurements predate the current engine state.

15+1 runs on laptop (Ryzen 7 8845HS + RTX 4060), 20-min windows each, all starting fresh from `bootstrap_model.pt` with `completed_q_values=true`. The sweep varied `training_steps_per_game` (ratio), `max_train_burst`, `max_game_moves`, `inference_max_wait_ms`, `leaf_batch_size`, `inference_batch_size`, `n_workers`, and `gumbel_m` across PUCT and Gumbel arms to identify the highest-throughput config for the Phase 4.0 overnight run. Full methodology and per-run data in `archive/sweep_2026-04-08/`.

**PUCT top 3:**

| run_id | steps/hr | games/hr | draw% | gl_p50 |
|--------|----------|----------|-------|--------|
| P8b    | 2,976    | 497      | 55%   | 75     |
| P8     | 2,487    | 415      | 63%   | 75     |
| **P3** | **2,422**| **606**  |**39%**| **45** |

**Gumbel top 3:**

| run_id | steps/hr | games/hr | draw% | gl_p50 |
|--------|----------|----------|-------|--------|
| G5     | 1,721    | 431      | 44%   | 39     |
| **G3** | **1,417**| **710**  |**30%**| **20** |
| G2     | 1,211    | 607      | 23%   | 20     |

#### Winners

- **PUCT overnight (P3):** `training_steps_per_game=4`, `max_train_burst=16`, `max_game_moves=150`, `inference_max_wait_ms=4`, `leaf_batch_size=8`, `inference_batch_size=64`, `n_workers=14`.
- **Gumbel overnight (G3):** `gumbel_m=8`, `training_steps_per_game=2`, `max_train_burst=8`, `max_game_moves=150`, `inference_max_wait_ms=2`, `leaf_batch_size=8`, `n_workers=14`.

P3 chosen over P8b despite P8b's +23% steps/hr because P3 has +22% games/hr, 16pp lower draw rate, and median game length 45 vs 75. More unique decisive games produce better learning signal than fewer long draws.

#### Headline win

`gpu_train_frac` moved from 3.4% (P0 control) to 12.7% (P3) — a **3.7× increase** in the fraction of GPU time spent on gradient steps. This is the metric that validates the sweep.

#### Negative results worth remembering

- **leaf_bs=16 consistently hurts:** increases calls/move (opposite of theory), decreases games/hr by 19–30%, inflates draw rate by 25pp. Do not re-try without a `game_runner.rs` change. See Q16 in `docs/06_OPEN_QUESTIONS.md`.
- **Replay ratio > 4 correlates with draw-rate inflation** on 20-min windows (ratio=6 → 55–63% draws even with the best other knobs). Revisit after Phase 4.5 baseline.
- **inf_bs=64 → 32 fills batches to 99.4%** (threshold-reachable confirmed) but costs GPU util 84% → 78%. Mechanism validated, tradeoff unfavorable at this scale.
- **gumbel_m=16 → 8 doubled throughput** in the Gumbel arm (largest single knob effect in the sweep).

#### Open issue for overnight

- `policy_entropy_mean ≈ 0.25 nats` on EVERY run (framework expected 3–6 nats). Flat over 20 min across all configs. Probably a bootstrap-concentration artifact, but if the P3 overnight hasn't crossed ~1.0 nat by the 6-hour mark, pause and investigate before running the remaining 18 hours.

---

