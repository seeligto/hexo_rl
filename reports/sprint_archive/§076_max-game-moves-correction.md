<!-- Forensic archive extracted from docs/07_PHASE4_SPRINT_LOG.md during compression sprint 2026-05-13. Compressed counterpart in docs/07_PHASE4_SPRINT_LOG.md. -->

### §76 — max_game_moves correction for gumbel_targets (2026-04-10)

Phase A diagnostic confirmed `max_game_moves` counts plies not compound moves. `gumbel_targets` was alone at 150 plies (a §69 artifact for `fast_prob=0.25`); with §75's `fast_prob=0.0`, 57.6% of games hit the cap. Fix: 150 → 200 plies; yaml comment "compound moves" → "plies". Resumed from `ckpt_25008`.

---

