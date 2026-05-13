<!-- Forensic archive extracted from docs/07_PHASE4_SPRINT_LOG.md during compression sprint 2026-05-13. Compressed counterpart in docs/07_PHASE4_SPRINT_LOG.md. -->

## §145 — Smoke v4 ABORT and fallback to Option α' (radius cap) — 2026-05-02 (BACKFILLED 2026-05-03)

**Date:** 2026-05-02 (entry written retroactively in §150)
**Trigger:** §144 closed with Option A (gate recalibration) and
`max_game_moves` raised 100 → 150. Smoke v4 (the recalibrated run) was
launched on the 5080 with the relaxed thresholds. v4 also ABORTED at
Stage 1: draw_rate stayed ≥ 0.84 even with the longer truncation
window, indicating the encoding-window fragmentation isolated in §142
was not bounded by either γ knobs or truncation slack alone.

**Decision:** fall back to **Option α'** from
`reports/w4c_diag/encoding_audit.md` — cap `LEGAL_MOVE_RADIUS` 8 → 5.
The audit recommended γ first (which §144 ran) with α as the fallback.
v4 ABORT closed the γ-only path; α' is the next-cheapest intervention
(single Rust constant, no retrain, no schema change, colony rules
preserved at cluster threshold 8).

**Smoke v4 artifacts:** transient draft in `/tmp/` was not landed; the
ABORT signal is preserved here. v4's specific gate values are not
reproduced — the conclusion (γ + truncation slack insufficient) is
what carried into §146.

**Outcome:** §146 implements Option α'. Backfilled here so the
`§145 / Option α'` cross-references in §146 (line 5857) and §148
(lines 5914, 6043) resolve cleanly.

---

