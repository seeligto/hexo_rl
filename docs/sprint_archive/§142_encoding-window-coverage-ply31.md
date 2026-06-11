<!-- Forensic archive extracted from docs/07_PHASE4_SPRINT_LOG.md during compression sprint 2026-05-13. Compressed counterpart in docs/07_PHASE4_SPRINT_LOG.md. -->

## §142 — Encoding-window coverage audit: ply-31 fragmentation pivot confirmed — 2026-05-01

**Date:** 2026-05-01
**Probe:** `scripts/diag_encoding_window_audit.py`, `scripts/diag_sealbot_window_capture.py`
**Inputs:** `docs/notes/remote_reports/games_2026-04-30.jsonl` (20 self-play games), `reports/w4c_diag/sealbot_5500_games.jsonl` (5 ckpt_5500 vs SealBot games)
**Report:** `reports/w4c_diag/encoding_audit.md`

**Hypothesis confirmed.** ckpt_5500 self-play crosses the 19×19 single-window boundary at **ply 31** (median pct_outside 0% → 21.9%, sharp). Any-cluster windowing delays onset but does not prevent it: 8/16 draws end with ≥80% of stones invisible to every cluster window. End-of-game single-window blindness median: 97.7% on draws.

**Pathology is distribution-endogenous.** Against SealBot opposition ckpt_5500 plays 0% outside throughout (5/5 games, max ply 29) — tactical pressure forces concentrated play. Fragmentation only emerges when two mutually permissive policies play each other.

**Axis structure:** fragmentation runs predominantly along the q-axis (NE-SW), consistent with §138 axis_density finding — self-play exploits the residual directional bias that rotation didn't fully wash out.

**Per-ply pivot table (median pct_outside_single, n=20):**

| threshold | single-window pivot | any-cluster pivot |
|----------:|--------------------:|------------------:|
| 5%        | **ply 31**          | ply 36            |
| 50%       | ply 33              | ply 65            |

**Recommendation:** Option γ (tighten self-play exploration) — cheapest mitigation that keeps the encoding mechanism intact and leverages §141 finding that policy head is already preserved. Option α (cap `LEGAL_MOVE_RADIUS`) falls back if γ-smoke fails. Option β (larger window) too expensive.

**Artifacts:** `reports/w4c_diag/encoding_audit.md`, `reports/w4c_diag/per_ply_coverage.csv`, `reports/w4c_diag/per_ply_coverage_sealbot.csv`

---

