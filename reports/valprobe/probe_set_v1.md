# D-C VALPROBE WP2 — card1 probe-set expansion (v1)

**Objective:** expand the run3 card#1 probe set from WP1's 41 → ≥200 distinct positions.
**Card#1 criterion:** SealBot head-lost proof AND raw v_t ≥ −0.5 AND replay_match (value optimistic while provably lost).
**Ckpt:** `run2_retro/checkpoint_00248000.pt` (step 248000, sha `312f85f632ee5046`), encoding `v6_live2_ls`, deploy head 150 sims vs SealBot d5, paired r5 fair openings.

## Result — TOP-UP COMPLETE (234 distinct)

| | |
|---|---|
| distinct start (WP1) | 41 |
| distinct final | **234** |
| target | 200 |
| shortfall | 0 (exceeded by 34) |
| batches run total | 5 (3 initial + 2 top-up) |

Initial 3 batches reached 153 (shortfall 47). Top-up (batches 3-4, seeds 20263711+20264711) closed the gap and exceeded target by 34.

## Per-batch yield

| batch | book_id | seed | games | card1_raw | card1_added | distinct_total | wall (min) |
|---|---|---|---|---|---|---|---|
| 0 | evalfair_r5_wp2_b0 | 20260711 | 128 | 45 | 45 | 86 | 18.8 |
| 1 | evalfair_r5_wp2_b1 | 20261711 | 128 | 43 | 43 | 129 | 18.3 |
| 2 | evalfair_r5_wp2_b2 | 20262711 | 128 | 24 | 24 | 153 | 17.8 |
| **3 (top-up)** | evalfair_r5_wp2_b3 | 20263711 | 128 | 38 | 38 | 191 | 19.4 |
| **4 (top-up)** | evalfair_r5_wp2_b4 | 20264711 | 128 | 43 | 43 | 234 | 19.6 |

WP2 total added: **193** (mean 38.6/batch over 5 batches). **Zero cross-batch dedup collisions** (card1_raw == card1_added every batch) — fresh per-batch opening books (distinct seeds) produced fully distinct games as intended.

## Method

Pipeline (`scripts/valprobe/wp2_expand_probe_set.py`, reuses WP1 `run_valprobe_sealbot.py` / `measure_recognition_lag.py` verbatim):
1. Fresh r5 paired fair-book per batch via `scripts/evalfair/core.build_book` — seed = 20260711 + batch·1000, 64 openings = 128 games/batch (color-swapped).
2. `evalfair` games: 248k deploy head (150 sims) vs SealBot d5, 8 workers.
3. SealBot backward-scan pipeline on loss games: v_t/q_t GPU phase + SealBot point-of-no-return solver (d6, window_half=9, 5s cap, colony≤4-cluster filter, 20 solver workers).
4. Extract card1: per head-turn-start snapshot where solver head_lost=True AND v_raw ≥ −0.5 AND replay_match=True.
5. **Dedup by (zobrist, side_to_move, moves_remaining)** per §4.6; merge into running set.

## Provenance & integrity (validated on final set, 234 rows)

- 234 rows, **234 distinct dedup keys** (perfect uniqueness).
- **0 criterion violations** — all rows satisfy head_lost AND v_raw ≥ −0.5 AND replay_match.
- v_raw range [−0.479, 1.0] (all ≥ −0.5).
- All rows: ckpt_step=248000, ckpt_sha=312f85f632ee5046, side_to_move=head, moves_remaining=2 (head-turn-start completing-stone granularity).
- WP1 41 + WP2 193 = 234. WP2 rows carry provenance {opening_idx, ply, turn, ckpt_step, ckpt_sha, book_id, seed(via book_id), zobrist, v_raw, T_provable_turn, head_as_p1, batch_idx, wp}.
- `T_provable_turn` null on some WP2 rows (+ all 41 WP1 rows, which predate the field): head_lost is per-position (solver probe_by_ply), independent of the game-level backward-scan T_provable, so these remain valid card1 positions where the game-level T_prov was censored/unset.

## Artifacts

- `reports/valprobe/probe_set_v1.jsonl` — **234 distinct card1 positions** (merged 41 WP1 + 193 WP2).
- `reports/valprobe/wp2/batch{0,1,2,3,4}_card1.jsonl` — per-batch raw card1 rows.
- `reports/valprobe/wp2/evalfair_r5_wp2_b{0,1,2,3,4}.json` — per-batch opening books.
- `reports/valprobe/wp2/batch_log.json` — machine-readable batch summary (top-up: 2 batches, batch_start_idx=3).
- `reports/valprobe/wp2/valprobe_wp2.log` — initial run log (batches 0-2).
- `reports/valprobe/wp2/valprobe_wp2_topup.log` — top-up run log (batches 3-4).

Box ephemeral; all outputs synced to laptop.
