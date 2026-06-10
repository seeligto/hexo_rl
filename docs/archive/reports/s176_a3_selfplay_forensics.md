# §176 Wave A3 — §175 v6 sustained selfplay forensics

**Date:** 2026-05-14
**Branch:** `phase4.5/s176_phase_a_validation`
**Goal:** Characterize §175 v6 sustained colony attractor; verify operator
"one large diffuse cluster" observation; pick POC colony metric on
Cohen's d effect size (20K best-WR vs 50K worst-WR cohorts).

---

## (a) Provenance

| Field | Value |
|---|---|
| Vast host | `REMOTE_HOST:REMOTE_PORT` (per `memory/project_current_vast_host.md`) |
| Run dir | `$REPO_ROOT/runs/c7e74d2842404a82bdd9f62edf740ea2/` |
| Training start | 2026-05-13 15:52 UTC (per `memory/project_175_eval_fix.md`) |
| Training resumed from | `checkpoint_00015582.pt` |
| Latest ckpt at pull | `checkpoint_00060000.pt` (2026-05-14 15:09 UTC) |
| Source events file | `$REPO_ROOT/logs/events_c7e74d2842404a82bdd9f62edf740ea2.jsonl` (43 MB) |
| Source ckpt log | `$REPO_ROOT/checkpoints/checkpoint_log.json` (256 KB) |
| Total `game_complete` events | 22,318 |
| `training_step` events | 4,464 (steps 15,590 → 60,220) |
| Decisive games (after winner-map fix) | 21,371 (draws=835 excluded) |
| Bad move parses | 0 |
| Cohort coverage | 20K cohort n=4,483 · 50K cohort n=7,267 (each ≥ 40 floor satisfied) |

Pull commands (rsync via `rsync-vast` skill convention):

```bash
mkdir -p reports/s176_a3_games
rsync -avz -e 'ssh -i ~/.ssh/REMOTE_KEY -p REMOTE_PORT' \
  REMOTE_USER@REMOTE_HOST:$REPO_ROOT/logs/events_c7e74d2842404a82bdd9f62edf740ea2.jsonl \
  reports/s176_a3_games/events.jsonl
rsync -avz -e 'ssh -i ~/.ssh/REMOTE_KEY -p REMOTE_PORT' \
  REMOTE_USER@REMOTE_HOST:$REPO_ROOT/checkpoints/checkpoint_log.json \
  reports/s176_a3_games/checkpoint_log.json
rsync -avz -e 'ssh -i ~/.ssh/REMOTE_KEY -p REMOTE_PORT' \
  REMOTE_USER@REMOTE_HOST:$REPO_ROOT/logs/replays/games_2026-05-14.jsonl \
  reports/s176_a3_games/replays_05_14.jsonl  # cross-validation only
```

`reports/s176_a3_games/` falls under `reports/**` gitignore rule
(`$REPO_ROOT/.gitignore:35`) — no extra rule needed.

§175 training NOT interrupted; pulls are read-only.

---

## (b) Cohort × metric table

Step→cohort binning: game `ts` joined to nearest `training_step.ts` via
bisect; cohorts span ±5K around label.

| Cohort | Step range | n |
|---|---|---:|
| 10K | 5,000 – 15,000 | 0 (training resumed from 15,582 — no games before) |
| 20K | 15,000 – 25,000 | 4,483 |
| 30K | 25,000 – 35,000 | 4,801 |
| 40K | 35,000 – 45,000 | 4,820 |
| 50K | 45,000 – 60,000 | 7,267 |

Note: 10K cohort empty because §175 resumed from ckpt 15,582 (per
`memory/project_175_eval_fix.md`). Substituting 20K for "best-WR
baseline" is correct per operator narrative + eval timeline below.

| Metric | 20K (n=4483) | 30K (n=4801) | 40K (n=4820) | 50K (n=7267) |
|---|---|---|---|---|
| `mean_hex_dist` | 5.48 ± 1.49 | 5.43 ± 1.53 | 5.89 ± 1.32 | 6.24 ± 1.05 |
| `max_hex_dist` | 13.70 ± 3.85 | 13.57 ± 3.95 | 14.66 ± 3.37 | 15.48 ± 2.66 |
| `n_components` (all) | 9.29 ± 5.32 | 9.15 ± 5.32 | 11.70 ± 5.70 | 13.63 ± 5.26 |
| `n_components_nonorphan` | 4.04 ± 2.45 | 3.97 ± 2.48 | 4.68 ± 2.64 | 5.22 ± 2.67 |
| `terminal_threats` | 0.14 ± 0.39 | 0.14 ± 0.38 | 0.12 ± 0.36 | 0.10 ± 0.32 |
| `colony_extension_fraction` (in-trainer) | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| `game_length` (plies) | 65.80 ± 29.22 | 65.46 ± 29.32 | 69.64 ± 28.11 | 72.05 ± 26.00 |
| `won_via_threat` | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 |

`won_via_threat = 1.00` is structural — Hex Tac Toe wins by 6-in-a-row
necessarily complete a 5+1 line; this metric is degenerate and
discarded for POC selection.

`colony_extension_fraction` (in-trainer emit, `selfplay/pool.py` /
`engine` colony detector) is **flat zero** across the full 45K-step
window. The in-trainer signal does NOT capture the §175 colony attractor.
That justifies the new Python-side POC metric.

§175 eval timeline (cross-ref from `eval_complete` events):

| Step | SealBot WR (n=320) | Elo |
|---:|---:|---:|
| 20K | **18.0%** | 443.1 |
| 30K | 10.0% | 387.2 |
| 40K | 15.0% | 373.3 |
| 50K | **4.0%** | 257.9 |

Operator-stated trajectory (17.4% → 4%) directly observed in events
(18.0% → 4.0% at 50K). Operator's WR claim CONFIRMED. Now the
selfplay-distribution claim:

---

## (c) Trajectory (rolling 200-game mean, step-binned)

PNG: `reports/s176_a3_games/trajectory.png` (6-panel grid, scatter +
rolling mean).

ASCII trajectory (mean over each 5K-step bin):

| step_bin | n_components | n_comp_nonorph | mean_hex_dist | max_hex_dist | terminal_threats | game_length | n |
|---|---:|---:|---:|---:|---:|---:|---:|
| 15-20K | 9.33 | 4.06 | 5.48 | 13.71 | 0.14 | 65.52 | 2098 |
| 20-25K | 9.25 | 4.02 | 5.47 | 13.68 | 0.13 | 66.06 | 2385 |
| 25-30K | 9.10 | 3.95 | 5.41 | 13.54 | 0.13 | 65.16 | 2396 |
| 30-35K | 9.20 | 3.99 | 5.45 | 13.60 | 0.14 | 65.76 | 2405 |
| 35-40K | 9.61 | 4.08 | 5.53 | 13.84 | 0.14 | 66.60 | 2395 |
| **40-45K** | **13.77** | **5.27** | **6.24** | **15.48** | 0.11 | 72.64 | 2425 |
| 45-50K | 13.54 | 5.21 | 6.23 | 15.48 | 0.09 | 71.81 | 2411 |
| 50-55K | 13.67 | 5.19 | 6.23 | 15.43 | 0.10 | 72.00 | 2418 |
| 55-65K | 13.69 | 5.25 | 6.26 | 15.54 | 0.09 | 72.35 | 2438 |

**Step-change phenomenon**: between 35-40K and 40-45K bins:
- `n_components`: 9.61 → 13.77 (**+43%**)
- `mean_hex_dist`: 5.53 → 6.24 (**+13%**)
- `max_hex_dist`: 13.84 → 15.48 (**+12%**)
- `game_length`: 66.60 → 72.64 (**+9%** plies)
- `terminal_threats` (5+1 boards at term): 0.14 → 0.11 (**−21%**, opposite direction)

Pattern: at step ~40K, the model's selfplay distribution undergoes a
discrete shift — stones become MORE dispersed (more islands, longer
inter-pair distances) and games run LONGER without ever-forming
terminal-state threat density. Post-40K the metric plateaus through 60K.

This is the **inverse** of the operator's "one large diffuse cluster"
intuition: the attractor is fragmented-multi-island, not consolidated.

---

## (d) Single-cluster fraction (operator-observation verdict basis)

Operator stated qualitative observation: §175 selfplay terminal states
look like "one large diffuse cluster" rather than multi-island
formations. Pre-registered test: fraction of games with
`n_components == 1` per cohort.

Two flavours:

1. **Strict** (`n_components == 1`): every winner stone connected by
   hex-adjacency (distance ≤ 1).
2. **Non-orphan** (`n_components_nonorphan == 1`): substantial components
   only (size ≥ 2), matches `06_OPEN_QUESTIONS.md` Q11 resolution
   (`docs/06_OPEN_QUESTIONS.md:390`, 2026-04-28). Single orphan stones
   excluded from cluster count.

| Cohort | n | strict `==1` | strict % | non-orphan `==1` | non-orphan % | non-orphan 95% Wilson CI |
|---|---:|---:|---:|---:|---:|---|
| 20K | 4,483 | 286 | 6.4% | 810 | 18.1% | [17.0%, 19.2%] |
| 30K | 4,801 | 346 | 7.2% | 908 | 18.9% | [17.8%, 20.0%] |
| 40K | 4,820 | 150 | 3.1% | 535 | 11.1% | [10.2%, 12.0%] |
| 50K | 7,267 | 18 | 0.2% | 459 | 6.3% | [5.8%, 6.9%] |

**Direction is opposite to operator claim**: single-cluster fraction
DECREASES monotonically (20K → 50K), from 18.1% to 6.3% (non-orphan).
Strict measure crashes from 6.4% to 0.2%.

Verdict thresholds (pre-registered):
- ≥ 70% → CONFIRMED
- 50–70% (or monotonic decrease without reaching modal-single) → PARTIAL
- < 50% → REFUTED

At 50K cohort, non-orphan single-cluster = 6.3%. **REFUTED** under both
pre-registered criteria (well under 50%; trend is monotonic *down*,
opposite to operator intuition).

The §175 attractor is **multi-island, increasingly fragmented**, NOT
unified into a single diffuse colony.

---

## (e) POC metric choice + Cohen's d table

Cohen's d (20K vs 50K cohort), pooled SD. Negative d = metric grew
worse-WR-cohort vs best.

| Metric | Cohen's d (20K − 50K) | 20K mean | 50K mean | |d| rank |
|---|---:|---:|---:|---:|
| `n_components` (all) | **−0.822** | 9.29 | 13.63 | 1 |
| `mean_hex_dist` | −0.614 | 5.48 | 6.24 | 2 |
| `max_hex_dist` | −0.563 | 13.70 | 15.48 | 3 |
| `n_components_nonorphan` | −0.454 | 4.04 | 5.22 | 4 |
| `game_length` | −0.229 | 65.80 | 72.05 | 5 |
| `terminal_threats` | +0.123 | 0.14 | 0.10 | 6 |
| `colony_extension_fraction` | +0.016 | 0.00 | 0.00 | 7 |
| `won_via_threat` | 0.000 | 1.00 | 1.00 | 8 (degenerate) |

**POC pick: `n_components`** (raw, no orphan filter).

Cohen's d = **−0.822** (large effect; well above the 0.5 floor).

Rationale:
- Largest |d| of any candidate (0.822 vs next-best 0.614).
- Cheap online: BFS / union-find on winner's stone set at game-end (no
  pairwise distance loop). O(N) where N = winner stones ≤ ~30.
- Interpretable in one sentence: "average number of disconnected
  islands in the winner's final stone group."
- Captures the actual mechanism: §175 attractor produces more islands
  (model fails to consolidate winning formations).

The non-orphan variant has smaller |d| (0.454) because it filters
exactly the single-orphan tail where the §175 attractor visibly
deposits scattered exploratory stones. Use raw `n_components` for the
trainer-side POC; `_nonorphan` remains a useful diagnostic adjunct.

`mean_hex_dist` and `max_hex_dist` are second-best (|d| 0.6 and 0.56)
but cost O(N²) — both also redundant with `n_components` because more
islands → larger inter-stone spread.

`colony_extension_fraction` (the existing in-trainer signal, emitted
per-game via `selfplay/pool.py`) shows **zero discrimination** between
cohorts (|d| 0.016). It is a different phenomenon (forward-extension
during play). This justifies the §176 POC metric work.

---

## Verdict (pre-registered)

**REFUTED** — Operator's qualitative observation "one large diffuse
cluster" is not borne out by terminal-state forensics. Quantitatively:

- Strict single-cluster: 0.2% at 50K (well under 70% / 50% bounds)
- Non-orphan single-cluster: 6.3% at 50K
- Direction is *monotonic decreasing* (18.1% → 6.3%), opposite to claim
- The actual attractor is **multi-island fragmentation**: mean
  `n_components` grows from 9.29 to 13.63 (Cohen's d −0.822)

Mechanism hypothesis (for §176 Phase D synthesis): §175 model regression
vs SealBot (18% → 4%) coincides with a step-change at ~40K where the
selfplay distribution shifts to longer games (66 → 72 plies),
fewer terminal threats (0.14 → 0.10), and more island fragmentation
(9.29 → 13.63 components). The model is exploring further from
existing formations rather than consolidating winning runs. Model
improves vs self (per operator: 56→78%) by exploiting the SAME
distributional shift that handicaps it vs external bot SealBot —
classic non-stationary opponent overfit.

**POC metric for §176 trainer-side per-game emit:** `n_components`
(BFS on winner's final stones, hex-adjacency, all components incl
orphans). Cohen's d = **−0.822** (best of 8 candidates).

---

## Artifacts (local, gitignored under `reports/**`)

- `reports/s176_a3_games/events.jsonl` (43 MB, raw events from vast)
- `reports/s176_a3_games/checkpoint_log.json` (256 KB)
- `reports/s176_a3_games/replays_05_14.jsonl` (239 KB, cross-validation)
- `reports/s176_a3_games/per_game.csv` (21,371 rows, all per-game metrics + step)
- `reports/s176_a3_games/cohort_table.csv` (cohort × metric summary)
- `reports/s176_a3_games/trajectory.png` (6-panel scatter + rolling mean)
- `reports/s176_a3_games/analyze.py` (re-runnable analysis script)
- `reports/s176_a3_games/plot.py` (re-runnable plot script)

## Caveats

- `winner` field semantics in `events.jsonl` per `pool.py:657`:
  `{0=P1, 1=P2, -1=draw}` (NOT `{1=P1, -1=P2}`). First pass of analyze.py
  misread → caught and fixed before final results.
- 10K cohort empty (training resumed from step 15,582). Operator's
  "20K=best-WR" cohort is the natural reference; eval timeline confirms.
- `colony_extension_fraction` zero across whole window suggests either
  the metric's definition diverges from the §175 phenomenon, or detector
  thresholds need re-tune. Either way, in-trainer signal alone is
  insufficient — operator's call for new POC metric is correct.
- `won_via_threat` = 1.00 across all cohorts is structurally forced (HTT
  6-in-row wins always close a 5+1 line). Discarded from POC ranking.

## File:line citations

- `pool.py:657` — winner-code → winner_int mapping confirming `{0:-1, 1:0, 2:1}`
- `pool.py:593` — colony emission site (referenced in
  `docs/07_PHASE4_SPRINT_LOG.md:1431`)
- `engine/src/game_runner/worker_loop.rs:806-808` — terminal_reason u8
  enum (0=six_in_a_row / 1=colony / 2=ply_cap / 3=other_draw)
- `engine/src/game_runner/worker_loop.rs:336-338` — game-loop terminates
  on `check_win()` at top of iteration
- `engine/src/board/state.rs:111-115` — turn structure (1 move ply 0, then
  2 per turn)
- `engine/src/board/moves.rs:119-148` — `winner()` / `player_wins()`
- `hexo_rl/eval/colony_detection.py:31-52` — `_connected_components`
  (BFS reused for `n_components`)
- `hexo_rl/utils/coordinates.py:89-105` — `axial_distance` (hex Manhattan)
- `docs/06_OPEN_QUESTIONS.md:390` — Q11 orphan-exclusion rule
- `docs/07_PHASE4_SPRINT_LOG.md:1202` — §175 v6 sustained recipe pin
- `docs/07_PHASE4_SPRINT_LOG.md:1431` — G3 wiring (`avg_game_length`,
  per-game `game_length` in structlog `game_complete`)
- `memory/project_175_eval_fix.md` — §175 training start time + ckpt
  resume (`checkpoint_00015582.pt`)
- `memory/project_current_vast_host.md` — vast host coordinates
