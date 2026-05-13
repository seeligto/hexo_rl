<!-- Forensic archive extracted from docs/07_PHASE4_SPRINT_LOG.md during compression sprint 2026-05-13. Compressed counterpart in docs/07_PHASE4_SPRINT_LOG.md. -->

## §77 — MCTS depth & ZOI scope investigation (2026-04-11)

**Reference:** `reports/mcts_depth_investigation_2026-04-11/`

### Motivation

Prior sessions assumed ZOI restricted MCTS tree branching. Depth probe and
code audit performed to verify actual behavior and measure search depth.

### Findings

**1. ZOI is post-search only — §36 corrected.**
Code audit of `game_runner.rs:626–643` confirmed ZOI filtering runs *after*
`expand_and_backup` completes, on the root visit-count vector used for move
selection. The MCTS tree itself expands with the full radius-8 legal set at
all depths. §36 description amended.

**2. Measured branching factor.**
Depth probe (200 sims, PUCT): 360 median root children created, 7 receiving
visits. B_eff = 6.1. FPU and policy concentration — not ZOI — drive the low
effective branching factor. Children past rank ~10 receive zero visits under
200 sims.

**3. Measured leaf depth.**
Mean leaf depth 2.92 plies (PUCT, 200 sims), max depth 6–8. Top-5 visit
share 0.97 — search is appropriately concentrated given the compute budget.

**4. Depth projections.**
Gumbel m=16 gains approximately +0.6 plies vs PUCT at 200 sims; desktop
training logs at ~18k steps confirm ~3.5 mean depth (consistent with projection).
ZOI-at-expansion would add only +0.16 plies over Gumbel — below measurement noise.

### Decision: Option A (do nothing)

ZOI-at-expansion rejected. Depth improves automatically as policy sharpens
(lower B_eff). The correct lever for deeper search is n_sims increase (Option B),
not tree pruning. Revisit Option B at 200K+ steps if B_eff remains above 10.

No code changes. No config changes.

