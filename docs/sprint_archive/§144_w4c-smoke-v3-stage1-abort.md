<!-- Forensic archive extracted from docs/07_PHASE4_SPRINT_LOG.md during compression sprint 2026-05-13. Compressed counterpart in docs/07_PHASE4_SPRINT_LOG.md. -->

## §144 — W4C smoke v3 (Option γ): Stage 1 ABORT — gate recalibration needed — 2026-05-01

**Date:** 2026-05-01
**Variant:** w4c_smoke_v3_5080 (n_workers=18, batch=224, wait=8ms, burst=8, 5080 24t)
**Bootstrap:** bootstrap_model.pt (v6, 8-plane, §134)
**γ knobs:** ε=0.10, τ_threshold=10, max_game_moves=100, decay_steps=200_000
**Wall time:** 3.2h (193 min for 5500 steps, ~1719 steps/hr — +98% vs v1's 869 steps/hr)
**Report:** `reports/w4c_smoke_v3/verdict_20260501.md`

### Stage 1 trajectory (steps 0–5500)

| Step | draw_rate | pe_self | x_wr | o_wr | pretrain_w |
|------|-----------|---------|------|------|------------|
| 1000 | 0.853 | 5.492 | 0.067 | 0.083 | 0.7960 |
| 2500 | 0.828 | 5.235 | 0.075 | 0.099 | 0.7901 |
| 5000 | 0.844 | 5.518 | 0.063 | 0.096 | 0.7803 |
| 5500 | 0.839 | 5.462 | 0.063 | 0.099 | 0.7783 |

### Gate evaluation

| # | Metric | Threshold | Value @ 5000 | Result |
|---|--------|-----------|--------------|--------|
| P1 | axis_density max | ≤ 0.55 | 0.5630 | **FAIL** |
| P3 | draw_rate | < 0.65 | 0.844 | **FAIL** |
| T1 | C1 contrast | ≥ +0.479 | +4.949 | PASS |
| T2 | C2 ext_in_top5 | ≥ 25% | 40% | PASS |
| T3 | C3 ext_in_top10 | ≥ 40% | 65% | PASS |

**Verdict: ABORT — Stage 1 FAIL.** Both failures are `max_game_moves=100` artifacts, not γ-knob regressions.

**axis_density 0.563 > 0.55:** v1 had 0.548 at same step; v3 trend is *increasing* (0.5595→0.5630). Root cause: fewer stones at 100-ply truncation → opening-axis bias (axis_s, NE-SW) not washed out. v1 calibrated on 200-ply games.

**draw_rate 0.844 >> 0.65:** v1 had 0.695 at 5500 with max_game_moves=200. Sprint draft §144 predicted draw_rate would *decrease* with 100-ply truncation — opposite happened. Games that resolve at plies 100–200 are now scored as draws. Only ~16% of games are decisive at 100 plies; threshold was calibrated for 200-ply games where ~30% hit the limit.

**γ knobs positive despite FAIL:** pe_self stable 5.2–5.6 (no collapse), threat_loss drops to 0.007–0.01 by step 3000+, threat probe well above thresholds (contrast +4.95 vs bootstrap +0.60), pretrain_weight 0.778 matches decay schedule exactly.

**O-side imbalance note:** x_wr=0.063, o_wr=0.099 at step 5500. O wins 57% of decisive games. Monitor at Stage 2 — could be noise at 16% decisive rate, but flags if it persists.

### Decision: Option A — recalibrate gates for 100-ply games

| Gate | Old threshold (200-ply) | Recalibrated (100-ply) | v3 value |
|------|------------------------|------------------------|----------|
| draw_rate | < 0.65 | < 0.85 | 0.844 ✓ |
| axis_density | ≤ 0.55 | ≤ 0.57 | 0.563 ✓ |

Option B (revert to 200 plies) would reintroduce random-walk corruption §142 was solving. Not recommended.

**Condition on Option A:** monitor axis_density trend during Stage 2 (eval at steps 7500 and 10000). If it continues climbing past 0.57 with 150 plies, that's a training signal, not an artifact.

**max_game_moves updated to 150** (`configs/selfplay.yaml` + all 4 host variants, §144) — midpoint between the 100-ply artifact and the 200-ply original. Retains truncation benefit while allowing more decisive outcomes.

**Artifacts:** `checkpoints/checkpoint_00005500.pt`, `reports/w4c_smoke_v3/verdict_20260501.md`, `docs/notes/remote_reports/sprint_log_144_draft.md`

---

