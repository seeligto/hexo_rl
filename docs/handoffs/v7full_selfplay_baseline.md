# T2 — v7full vs v7full self-play baseline

**Date:** 2026-05-04
**Checkpoint:** `checkpoints/bootstrap_model_v7full.pt` (sha256 `29306533…`, §150)
**Hardware:** laptop 4060 Max-Q, sims=96, temp=0.5, 4-ply random opening, seed_base=42
**Wall:** 594 s (≈ 10 min)
**Script:** `scripts/v7full_selfplay_baseline.py` (commit `1aa73b7`)
**Outputs:** `games.jsonl` (200), `summary.json`

---

## Why this run exists

Phase B δ.c W4C 5080 smoke (§151) saw **draw_rate climb 0.92 → 0.945** across S1+S2
while colony fraction stayed safely under abort gate (mean 0.029, 17× under). δ.c
itself (cluster threshold 8 → 5) is acquitted on its own kill criterion.
Open question: is the high-draw equilibrium **structural** (intrinsic to v7full +
ply-cap=150 + draw_value=−0.5) or **drift** (the smoke's training loop pushed
ckpt_10k away from v7full toward draw-equilibrium)?

T2 settles it by playing v7full against itself under the same hyperparameters
the smoke uses (`max_game_moves=150`, sims=96, draw_value=−0.5 implicit).

---

## Aggregate

| Metric | Value |
|---|---:|
| n games | 200 |
| **draw_rate** | **3.0 %** (6 / 200) |
| Wilson 95 % CI | [1.4 %, 6.4 %] |
| x wins (P1) | 87 / 200 (43.5 %) |
| o wins (P2) | 107 / 200 (53.5 %) |
| draws | 6 / 200 (3.0 %) |
| x / (x + o) ex-draws | 87 / 194 = **44.8 %** |
| mean ply | 55.3 |
| median ply | 42 |
| ply range | 19 – 150 |
| P10 / P90 ply | 27 / 109 |

### Terminal-reason breakdown

| Reason | Count | % | Mean ply | Median ply | Range | x / o / draw |
|---|---:|---:|---:|---:|---|---|
| six_in_a_row | 97 | 48.5 % | 34.6 | 33 | 19 – 71 | 35 / 62 / 0 |
| colony | 97 | 48.5 % | 70.1 | 65 | 29 – 147 | 52 / 45 / 0 |
| ply_cap | 6 | 3.0 % | 150 | 150 | 150 – 150 | 0 / 0 / 6 |

Every draw is a ply-cap timeout — no on-board draw ever resolved by colony or
mutual-exhaustion. The 6 timeouts are the entire draw mass, contributing exactly
the 3.0 % draw_rate.

---

## Comparison vs Phase B smoke ckpt_10k self-play

| Metric | T2 — v7full vs v7full | Phase B smoke ckpt_10k (S2 self-play window) | Δ |
|---|---:|---:|---|
| draw_rate | 3.0 % | 94.5 % | **−91.5 pp** |
| mean ply (decisive games) | 52.4 (six+colony aggregate) | n/a (almost no decisive games) | — |
| mean ply (all games) | 55.3 | ≈ 150 (ply-cap dominant) | −95 |
| colony share of decisives | 50.0 % | n/a (decisive games rare) | — |

Smoke comparison row sourced from the §151 W4C 5080 smoke draw-rate climb
(0.92 → 0.945 across S1+S2) reported in the in-progress sprint draft.
Window-level decisive stats from the smoke are not available locally (the smoke
runs on a remote 5080 host) — the 94.5 % draw figure is the strict comparator.

For context: the §151 laptop sanity sample (v7full + δ.c, n=113, 600 s, gumbel_full)
recorded mean ply 36.6, draw_rate 0.000, colony-extension fraction 0.0030
(see `/tmp/sprint_log_151_phase_b_delta_c_draft.md`). T2's longer mean ply
(55.3 vs 36.6) reflects T2's `temperature=0.5` (eval-style) and 4-ply random
opening vs the smoke's training temperature schedule, plus T2 sampling the
full game-length distribution (no early terminations).

---

## Sample game records

5 representative entries from `games.jsonl`. Field schema:
`{ game_id, winner (+1=x / −1=o / 0=draw), moves, terminal_reason, moves_list }`.

| Game # | Winner | Plies | Reason | Notes |
|---|---|---:|---|---|
| 199 | o | 27 | six_in_a_row | shortest decisive (P10 region, 19-ply minimum) |
| 196 | x | 45 | six_in_a_row | typical six-in-a-row decisive (median region) |
| 198 | x | 37 | colony | short-ish colony decision |
| 197 | o | 47 | colony | mid-range colony decision |
| Any of 6 ply-caps | draw | 150 | ply_cap | every cap is a draw |

Full per-game JSONL committed alongside this report; pull any record by
`jq 'select(.game_id == N)' games.jsonl`.

---

## Verdict — **DRIFT**

`draw_rate = 3.0 %` is **far below the 50 % drift threshold** and **massively
below the 85 % structural threshold**. v7full self-play under the smoke's own
hyperparameters resolves decisively in **97 %** of games. The smoke's ckpt_10k
reaching 94.5 % draws is therefore a training-loop induced shift, **not**
intrinsic equilibrium of v7full + ply-cap=150 + draw_value=−0.5.

**Logic:**
- Structural threshold (≥ 85 %): would require v7full self-play to also draw
  most games. T2 = 3 % — falsified.
- Drift threshold (≤ 50 %): the smoke pushed the model so far that 91.5 pp of
  draw mass appeared between training step 0 (= v7full) and step ~10 k. T2
  satisfies the drift threshold by a 47 pp margin.

**Confidence:** high. n=200, Wilson 95 % CI [1.4 %, 6.4 %] excludes any
plausible structural-equilibrium value. Decisive-game balance (44.8 % P1
ex-draws) is within normal P1/P2 variance — no obvious sign of a corrupt
checkpoint.

---

## Phase B' priority order — UPDATE

T2 verdict is **DRIFT**, so the Phase B' priority order does **not** change to
"raise max_game_moves first / change draw_value second". The decisive next step
is a **training-loop investigation**:

1. **Reproduce the drift on the laptop**: pull a smoke ckpt at step ≈ 5 k (mid-S1,
   pre-collapse) and at step ≈ 10 k (post-collapse), run T2 against each.
   Locate the step interval where draw_rate crosses 0.50.
2. **Inspect the training-time signal at that interval**: aux loss, value-head
   accuracy on terminal positions, policy entropy, rolling x/o split during
   self-play. The smoke is logging all of these (§151 dashboards).
3. **Correlate with config knobs that touched between v7full → smoke**:
   `cluster_threshold` (δ.c), grad-norm override (10.0), the γ-knob block in
   `w4c_smoke_v6_5080.yaml`. δ.c is the most novel — first ablation candidate.

Re-anchoring `max_game_moves` is **not** indicated by T2 — at 150 plies under
v7full play, only 3 % of games hit the cap. The cap is not the binding
constraint; whatever the smoke trained the model into is.

The window-geometry investigation (separate prompt, scoped post-§151) remains
out of scope here.

---

## Artifacts

- `scripts/v7full_selfplay_baseline.py` (commit `1aa73b7`)
- `reports/phase_b/v7full_selfplay/baseline.md` (this file)
- `reports/phase_b/v7full_selfplay/games.jsonl` (200 records, ~250 KB)
- `reports/phase_b/v7full_selfplay/summary.json`
- Run log: `/tmp/t2_v7full_run.log` (tail captured in summary)

## Constraints honored

- 5080 smoke untouched (laptop run, separate host)
- v7full checkpoint not modified
- No edits to `docs/07_PHASE4_SPRINT_LOG.md` or
  `/tmp/sprint_log_151_phase_b_delta_c_draft.md`
- `scripts/w7_q41_v7_v6_h2h.py` left untouched (§148 audited tool)
- `.venv` used throughout
- Phase B' implementation **not** pre-empted — verdict only
