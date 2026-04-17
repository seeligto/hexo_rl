# Elo DB ↔ Graduation Gate Audit — 2026-04-18

Scope: verify the graduation gate (commits `320e73e`, `1c4b878`) integrates
cleanly with the existing Bradley-Terry Elo DB. Four risks (R1–R4) evaluated
against current source and the live DB at `reports/eval/results.db`.

**Methodology:** read-only Phase 1–2. No training or eval runs. DB probed
via `sqlite3`.

---

## Schema summary

`hexo_rl/eval/results_db.py`:

- `players` — `UNIQUE(name, run_id)`. Identity is the string `name` scoped
  by `run_id`. `metadata` is an opaque JSON blob. No anchor lineage.
- `matches` — `UNIQUE(run_id, eval_step, player_a_id, player_b_id)`. One
  row per (run, eval_step, pair). On conflict `INSERT OR REPLACE`.
- `ratings` — `UNIQUE(eval_step, player_id)`. One rating per player per
  eval step. `run_id` not part of the key (see R2 sub-finding below).

Opponent identifier form:

| Opponent | Name | run_id | Source (file:line) |
|---|---|---|---|
| Student checkpoint | `checkpoint_{train_step}` | run-scoped | `eval_pipeline.py:143-147` |
| Anchor (graduation) | `best_checkpoint` (fixed) | run-scoped | `eval_pipeline.py:213-216` |
| SealBot | `SealBot(t={think_time})` | **global `""`** | `eval_pipeline.py:87-91` |
| Random bot | `random_bot` | **global `""`** | `eval_pipeline.py:92-94` |

Bradley-Terry (`bradley_terry.py:21-129`) takes the `pairwise` list returned
by `ResultsDB.get_all_pairwise(run_id=...)` (results pooled per pair via
`SUM(wins_a), SUM(wins_b)`, `GROUP BY player_a_id, player_b_id`) and fits
one rating per player, gauged at `anchor_player = "checkpoint_0"`.

---

## R1 — Anchor ID collision (the important one)

**Evidence:** `hexo_rl/eval/eval_pipeline.py:213-222`

```python
best_pid = self.db.get_or_create_player(
    "best_checkpoint", "checkpoint", {"role": "champion"},
    run_id=self.run_id,
)
self.db.insert_match(
    train_step, ckpt_pid, best_pid, ...)
```

The anchor is registered under a **fixed** name `"best_checkpoint"` per run.
When graduation fires (`loop.py:451-471`) the weights on disk at
`best_model.pt` are replaced; the player row keeps the same `id`. Every
future student-vs-anchor match is inserted against that same `best_pid`.

Bradley-Terry pools matches by `(player_a_id, player_b_id)` via
`SUM(wins_a), SUM(wins_b)`
(`results_db.py:212-227`). Result: one virtual "best_checkpoint" opponent
whose inferred strength is a weighted average over its lifetime — not the
strength of any specific anchor snapshot.

**Live DB state** (`reports/eval/results.db`): every run has exactly one
`best_checkpoint` player row. Several runs hold multiple anchor matches:

```
8579cbe5...  best_checkpoint  4 matches (steps 5k,10k,15k,20k)
3718c0e1...  best_checkpoint  3 matches
dcf8cbba...  best_checkpoint  3 matches
0e840e26...  best_checkpoint  2 matches
...
```

No graduation fired in any recorded run (wr_best stayed below the
promotion_winrate threshold), so R1 has not yet bitten in practice. But
the bug is latent: the first run that graduates AND evaluates again will
merge pre- and post-graduation anchor strengths into one rating.

**Verdict:** at risk. **Severity: HIGH** — corrupts Elo history the moment
the gate fires, which is the precondition for the sustained runs gated
against it.

---

## R2 — SealBot rating contamination

**Evidence:** `eval_pipeline.py:87-91`

SealBot is registered with **no `run_id` arg**, defaulting to `""`. Its
player row is therefore **global** across runs.

`get_all_pairwise(run_id=X)` filters with
`WHERE pa.run_id = ? OR pb.run_id = ?` (`results_db.py:218-227`). This
includes SealBot matches where the other side is in run X, correctly.
The run-scoped filter guarantees SealBot's rating within run X is computed
from run X's matches only; it **does not pool** SealBot matches across
runs.

However: the anchor's garbage rating (per R1) feeds into the same joint
MLE as every student in the run, and the student ratings in turn feed
SealBot's inferred rating. R1 therefore **smears** into SealBot's
within-run rating via joint-fit coupling. Magnitude: small (anchor
contributes << student-vs-SealBot information), but non-zero.

**Sub-finding (independent of R1):** `ratings` table uniqueness is
`(eval_step, player_id)` (`results_db.py:50-52`). SealBot's player_id is
global. If two runs evaluate at the same step, the second run's
`INSERT OR REPLACE` overwrites the first run's SealBot rating row at that
step. Cross-run overwrite race on fixed-reference rating rows. Not
graduation-gate related; predates this audit. Flag only.

**Verdict:** at risk (indirect). **Severity: MEDIUM** — SealBot's
published Elo within a run distorts with R1; no cross-run drift of the
fixed reference. The sub-finding is a separate low-severity concern.

---

## R3 — Self-play worker reload race

**Evidence:** `loop.py:437-502`, `eval_pipeline.py:96-300`

Eval games are played by `EvalPipeline.run_evaluation` in a background
thread (`loop.py:501-502`) against `best_model`, a loaded snapshot
reference. Self-play workers (Rust) push training tuples to the replay
buffer, not to the Elo DB. The eval DB is **only** written from the main
training-thread's post-eval drain (`loop.py:438-473`).

The anchor swap (`best_model.load_state_dict(eval_base.state_dict())` at
`loop.py:458` and `_inf_base.load_state_dict(...)` at `loop.py:463`) runs
sequentially after the eval thread has already completed and stored its
result. DB writes for the match rows happened inside
`run_evaluation(...)` at `eval_pipeline.py:184-189` BEFORE the swap. So
the match rows are always tagged with the student pid for the step the
eval was kicked off at, and the `best_pid` was the one registered by that
same `run_evaluation` call.

A worker-side race exists around `inf_model` weight corruption during
promotion (load_state_dict while workers forward), but that is a
weight-corruption concern, not a DB-tagging one. Out of scope for this
audit.

**Verdict:** safe (for DB integrity). **Severity: LOW** (worker weight
swap atomicity deferred).

---

## R4 — Elo banking semantics + anchor lineage

**Evidence:** `loop.py:451-471`

Graduation emits:
- structlog `best_model_promoted` (file log only)
- dashboard `eval_complete` event with `gate_passed = prev.get("promoted")`
  (emitted on the NEXT eval round when the result is drained — delayed by
  one `eval_interval`)
- Overwrite of `best_model.pt` (single file; previous anchor lost)

No DB row captures the graduation event itself. No anchor-snapshot
history. The weights that define anchor-at-step-N are unrecoverable after
the next graduation.

Dashboard invariant check (`CLAUDE.md` Phase 4.0 — "dashboard = passive
observer"): graduation does emit `eval_complete` with `gate_passed`, so
the dashboard is notified. Emission exists. Timing is delayed by
`eval_interval` due to the background-eval design — documented in
`loop.py:437-454` comments.

**Verdict:** design gap, no immediate bug. **Severity: LOW**. Flag for
`FUTURE_REFACTORS.md`.

---

## Summary

| Risk | Verdict | Severity |
|---|---|---|
| R1 — Anchor ID collision | **at risk, latent** | **HIGH** |
| R2 — SealBot contamination | at risk (indirect via R1 + cross-run row overwrite sub-finding) | MEDIUM |
| R3 — Reload race | safe for DB | LOW |
| R4 — Anchor lineage | gap, flag only | LOW |

**Recommended fix (one):** R1. Make the anchor identity include the
anchor checkpoint step so each graduated anchor is a distinct
`player` row. This also automatically narrows R2's coupling (each anchor
snapshot contributes information about its own lifetime only). Minimal
surface: thread `best_model_step` through `run_evaluation` and change the
registered name.

**Not fixing in this pass:**

- R2 sub-finding (ratings table ignores run_id) — out of graduation-gate
  scope, separate schema migration.
- R3 worker weight-swap atomicity — separate audit, not DB.
- R4 anchor lineage — FUTURE_REFACTORS entry.
