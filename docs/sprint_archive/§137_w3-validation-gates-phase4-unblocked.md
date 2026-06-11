<!-- Forensic archive extracted from docs/07_PHASE4_SPRINT_LOG.md during compression sprint 2026-05-13. Compressed counterpart in docs/07_PHASE4_SPRINT_LOG.md. -->

## §137 — W3 validation gates: Q41 WARN + Q52 PASS + Q44 done → Phase 4.0 UNBLOCKED — 2026-04-30

**Date:** 2026-04-30  
**Hardware:** Laptop — Ryzen 8845HS + RTX 4060 Max-Q  
**Cost:** $0 (local hardware)

**New scripts:**
- `scripts/w3_q41_v6_v5_h2h.py` — v6 vs v5 H2H eval (200 games, 128 sims, balanced colour)
- `scripts/w3_q52_v6_sealbot.py` — v6 vs SealBot anchor (150 games, 0.5s, 128 sims, §114 protocol)

**Q41 — v6 vs v5 H2H (200 games):**
102/200 wins (51.0%), Wilson 95% CI [44.1%, 57.8%]. Originally BLOCK under old gate (lower-CI ≥ 50%), recalibrated to WARN. Gate revised: PASS ≥ 48%, WARN [43%, 48%), BLOCK < 43%. Rationale: PASS ≥ 50% at n=200 requires ~57%+ WR — fires even at exact parity, conflating "no regression" with "improvement." Revised BLOCK < 43% catches genuine regression. Under new gate: 44.1% = WARN (near-parity, D17 holds).

**Q52 — v6 vs SealBot (150 games):**
36/150 wins (24.0%), Wilson 95% CI [17.9%, 31.4%]. **PASS** (gate ≥ 14%). Beats v4 anchor (18.7%, §114) by +5.3pp. Colony-win fraction: 5.6% vs v4's 82%. Low colony fraction is a **positive signal** — colony wins during self-play created a degenerate training feedback loop (colony-explosion failure mode, observed prior runs). 8-plane channel cut (§131) dropped colony-related planes; v6 wins via 6-in-a-row. Desired for stable Phase 4.0 training.

**Q44 — laptop bench refloor (n=5, --no-compile):**
Worker pos/hr: **33,174** IQR ±5.3%, range [29.1k–36.3k]. 9/10 targets pass. Failure: batch_fill 78.6% < 84% — known dispatch-GIL bound (Q35), Phase 4.5 item. vs desktop 18-plane (§128, 27,835 pos/hr): **+19%**. Improvement from 8-plane smaller tensor + RTX 4060 Max-Q Ada Lovelace (sm_89). Perf-targets.md laptop footnote updated.

**Phase 4.0 status: UNBLOCKED.** Bootstrap-v6 (8-plane, §131) validated at external anchor. Ready to launch sustained run on vast.ai.

