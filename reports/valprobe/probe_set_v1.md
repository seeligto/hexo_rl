# D-C VALPROBE WP2 — card1 probe-set expansion (v1)

**Objective:** expand the run3 card#1 probe set from WP1's 41 → ≥200 distinct positions.
**Card#1 criterion:** SealBot head-lost proof AND raw v_t ≥ −0.5 AND replay_match (value optimistic while provably lost).
**Ckpt:** `run2_retro/checkpoint_00248000.pt` (step 248000, sha `312f85f632ee5046`), encoding `v6_live2_ls`, deploy head 150 sims vs SealBot d5, paired r5 fair openings.

## Result — SHORTFALL

| | |
|---|---|
| distinct start (WP1) | 41 |
| distinct final | **153** |
| target | 200 |
| **shortfall** | **47** |
| batches run | 3 (max) |

Ran all 3 authorized batches; did not reach 200. Stopped at the 3-batch cap per spec (did not grind past).

## Per-batch yield

| batch | book_id | seed | games | card1_raw | card1_added | distinct_total | wall (min) |
|---|---|---|---|---|---|---|---|
| 0 | evalfair_r5_wp2_b0 | 20260711 | 128 | 45 | 45 | 86 | 18.8 |
| 1 | evalfair_r5_wp2_b1 | 20261711 | 128 | 43 | 43 | 129 | 18.3 |
| 2 | evalfair_r5_wp2_b2 | 20262711 | 128 | 24 | 24 | 153 | 17.8 |

WP2 total added: **112** (mean 37.3/batch). **Zero cross-batch dedup collisions** (card1_raw == card1_added every batch) — fresh per-batch opening books (distinct seeds) produced fully distinct games as intended.

Batch 2 yield dropped to 24 (vs 45/43) — batch-to-batch loss-game/proof-rate variance, not a pipeline fault (criterion + provenance validated clean below).

## Cost to reach 200

Shortfall 47 positions.
- At mean WP2 yield (37.3/batch, 18.3 min/batch): **~1.3 more batches ≈ 24 min ≈ 0.4h**.
- At conservative batch-2 yield (24/batch): **~2 more batches ≈ 0.6h**.

Cheap to close if the operator authorizes 1-2 more batches (seeds 20263711+). Recommend 2 batches to clear 200 with margin.

## Method

Pipeline (`scripts/valprobe/wp2_expand_probe_set.py`, reuses WP1 `run_valprobe_sealbot.py` / `measure_recognition_lag.py` verbatim):
1. Fresh r5 paired fair-book per batch via `scripts/evalfair/core.build_book` — seed = 20260711 + batch·1000, 64 openings = 128 games/batch (color-swapped).
2. `evalfair` games: 248k deploy head (150 sims) vs SealBot d5, 8 workers.
3. SealBot backward-scan pipeline on loss games: v_t/q_t GPU phase + SealBot point-of-no-return solver (d6, window_half=9, 5s cap, colony≤4-cluster filter, 20 solver workers).
4. Extract card1: per head-turn-start snapshot where solver head_lost=True AND v_raw ≥ −0.5 AND replay_match=True.
5. **Dedup by (zobrist, side_to_move, moves_remaining)** per §4.6; merge into running set.

## Provenance & integrity (validated on final set)

- 153 rows, **153 distinct dedup keys** (perfect uniqueness).
- **0 criterion violations** — all rows satisfy head_lost AND v_raw ≥ −0.5 AND replay_match.
- v_raw range [−0.429, 1.0] (all ≥ −0.5).
- All rows: ckpt_step=248000, ckpt_sha=312f85f632ee5046, side_to_move=head, moves_remaining=2 (head-turn-start completing-stone granularity).
- WP1 41 + WP2 112 = 153. WP2 rows carry provenance {opening_idx, ply, turn, ckpt_step, ckpt_sha, book_id, seed(via book_id), zobrist, v_raw, T_provable_turn, head_as_p1, batch_idx, wp}.
- `T_provable_turn` null on 12 WP2 rows (+ all 41 WP1 rows, which predate the field): head_lost is per-position (solver probe_by_ply), independent of the game-level backward-scan T_provable, so these remain valid card1 positions where the game-level T_prov was censored/unset.

## Artifacts

- `reports/valprobe/probe_set_v1.jsonl` — 153 distinct card1 positions (merged 41 WP1 + 112 WP2).
- `reports/valprobe/wp2/batch{0,1,2}_card1.jsonl` — per-batch raw card1 rows.
- `reports/valprobe/wp2/evalfair_r5_wp2_b{0,1,2}.json` — per-batch opening books.
- `reports/valprobe/wp2/batch_log.json` — machine-readable batch summary.
- `reports/valprobe/wp2/valprobe_wp2.log` — full run log.

Box ephemeral; all outputs synced to laptop.
