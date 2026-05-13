<!-- Forensic archive extracted from docs/07_PHASE4_SPRINT_LOG.md during compression sprint 2026-05-13. Compressed counterpart in docs/07_PHASE4_SPRINT_LOG.md. -->

## §114 — bootstrap-v4: full-corpus retrain + eval — 2026-04-22

**Status:** RESOLVED. Superseded by v5 (§118), v6 (§134), v7 (§148), v7full (§150).

### Root cause: two silent corpus bugs

1. **POSITION_END=50 truncation** in `scripts/export_corpus_npz.py` — hard-coded cap discarded every position at ply ≥ 50, removing ~40% of all positions (entire late-game) from every pretrain corpus. Bootstrap was endgame-blind; value head never saw ply > 50. Retcons Q17 as two-cause: Dirichlet (§73) was necessary but not sufficient — corpus completeness was the structural fix.
2. **Broken Elo read** in `scripts/update_manifest.py` — read `player_black_elo` / `player_white_elo` (old scraper format), missing `players[].elo`. 5,694 / 5,706 games treated as unrated; Elo-weighted sampling effectively off.

### Fix sequence

- `aa16624` — Elo field fix: fall back to `players[].elo`. All 5,706 games rated.
- `ddd408f` — drop POSITION_END cap entirely. 305,410 positions (was 193,972).
- `8b446c5` — set POSITION_END=150 (P95.5) to trim time-scramble noise. 285,762 exported → ~3.4M with 12× aug (was ~2.3M).

### Eval results (bootstrap-v4)

| Probe | v4 | v3c |
|---|---|---|
| C1 contrast_mean | **+0.360** (margin 0.020 to 0.380 floor) | −0.046 |
| H2H WR (100g, 64 sims) | **67% ± 9.2%** | 33% |
| SealBot WR (150g, 128 sims, 0.5s) | **18.7% ± 6.2%** | — |

C1 delta +0.406 absolute from pure corpus fix. ctrl logits flipped +0.062 → −0.152 (threat head now suppresses far empties). Colony-win fraction 82% expected pre-RL.

### Rule established: corpus discipline

**Before diagnosing trainer pathology (hyperparameters, augmentation, architecture), verify the corpus is complete.** Pretrain loss curves were plausible throughout — only the C1 probe on real-game positions exposed the truncation. Upstream data quality gates downstream model quality more strongly than any downstream training improvement. Probe FAIL ⇒ check corpus before tuner.

### Forward pointers

- v5 retrain §118 · v6 §134 · v7 §148 · v7full §150 (canonical §150 anchor 17.4% n=500 vs SealBot)
- Also fixed in this session: `eval_vs_sealbot.py` `resolve_checkpoints` glob-before-early-return + `EvalResult` → `float()` crash.

