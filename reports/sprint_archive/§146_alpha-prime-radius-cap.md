<!-- Forensic archive extracted from docs/07_PHASE4_SPRINT_LOG.md during compression sprint 2026-05-13. Compressed counterpart in docs/07_PHASE4_SPRINT_LOG.md. -->

## §146 — Option α' implementation: cap LEGAL_MOVE_RADIUS 8→5 — 2026-05-02

**Date:** 2026-05-02
**Trigger:** §144 (smoke v3 ABORT) and the smoke v4 ABORT carried in /tmp draft (max_game_moves=150) both failed Stage-1 gates with draw_rate ≥ 0.84 under bootstrap-v6 self-play. γ knobs (ε=0.10, τ_threshold=10) and the truncation-midpoint move (100→150) did not bound the encoding-window fragmentation isolated in §142.

**Decision:** apply Option α from `reports/w4c_diag/encoding_audit.md` — cap the legal-move radius at 5 instead of the official rule's 8.

**Rationale:**
- §142 measured the fragmentation pivot at ply ~31, with stones beyond the 19×19 single-window encoding by ply 65 in 50% of self-play games.
- Real-game corpus (human + bot, including SealBot at the v6 anchor) never places a stone more than 5 cells from any existing stone — radius 5 is the empirical envelope of in-distribution play.
- A cap of 5 keeps colony wins reachable (cluster threshold remains 8), keeps the network architecture and 8-plane buffer schema unchanged, and is a single Rust constant edit (no config knobs, no retrain).

### Implementation

`engine/src/board/moves.rs:9` — `LEGAL_MOVE_RADIUS: i32 = 8 → 5`. Doc comments in the same file updated to cite §145 / Option α' instead of the official rule. Cluster threshold (`engine/src/board/moves.rs:267`, `hex_distance ≤ 8`) left untouched — it governs colony adjacency, not move legality.

Test updates in `engine/src/board/mod.rs`:
- `legal_moves_counts_empty_cells`: 216 → 90 (single-stone hex ball: 91-1).
- `legal_grows_with_bounding_box`: 216 → 90 single, 300 → 144 union of two radius-5 balls 5 apart.
- New test `legal_move_radius_capped_at_5`: verifies `(5,0)` and `(0,5)` are legal, `(6,0)`, `(0,6)`, `(8,0)` are not, every legal cell is within hex_distance 5 of `(0,0)`, and two stones at distance 5 still form one cluster.

`cargo test --workspace`: 174 tests pass (139 engine + 35 misc), 0 failures.

### Laptop smoke (bootstrap-v6, gumbel_full, 4 workers, 600 s) — **PRELIMINARY**

> ⚠ **TMP / placeholder.** Numbers below are from a 21-game laptop run, not a
> sustained remote run. Treat as a directional sanity check only — replace
> with the first vast.ai pull's pos/hr, draw_rate, and ply-distribution once
> available, and re-set the draw-rate gate against the post-α' baseline at
> that point.


| Metric | Pre-cap baseline | R=5 (this run) |
|---|---|---|
| games_completed | — | 21 |
| games_per_hour | — | 126 |
| draw_rate | 0.84 (smoke v4 step 5500) | 0.000 (n=21) |
| mean game length (plies) | ~110 (W4C self-play) | **16.0** |
| median game length (plies) | — | 16.0 |
| x / o / draws | — | 12 / 9 / 0 |

Recent length sample: 9, 14, 8, 7, 17, 13, 18, 25, 16, 27, 16, 14, 19, 21, 16, 10, 17, 27, 13, 19 (range 7–27 plies).

**Direction confirmed; magnitude exceeds prediction.** Pre-run estimate was 30–60 plies; observed 16. With R=5 the legal move ball collapses from 217 to 91 cells around the first stone, which forces compact play. Bootstrap-v6 exploits compact lines decisively (consistent with §141's "policy head intact"), driving every sampled game to a 6-in-a-row resolution before the 150-ply truncation gate engages. Zero draws across 21 games is below v6's ~20% baseline; with p=0.2 the chance of n=0 draws is 0.8^21 ≈ 0.9%, so the shift is real, not a sampling fluke at this n.

**No regressions:** colony wins remain reachable (cluster threshold 8), the buffer schema is unchanged, no config knob was touched, no checkpoint format changes — bootstrap-v6 loads as-is.

### Open follow-ups

- Vast.ai pull will pick up the constant change automatically from the next remote checkout. Confirm worker pool restart on the 5080 / 5090 hosts after pull.
- The next W4C smoke (post-α') should re-evaluate the draw-rate gate against the R=5 baseline — §144's < 0.85 calibration was built around max_game_moves=150 with the radius-8 fragmentation tail, and this run shows 0.0 at small n. Re-set after a longer remote run reports a stable draw rate.
- Threat-probe gate carries unchanged (C2 ≥ 25, C3 ≥ 40 vs bootstrap-v6).
- Game-length distribution at scale: 21 games is a directional signal, not a characterisation. Capture the full distribution from the next remote run.

**Artifacts:** `engine/src/board/moves.rs` (constant + doc), `engine/src/board/mod.rs` (tests), `/tmp/smoke_radius5.json` (laptop smoke; transient — replaced by remote results).

---

