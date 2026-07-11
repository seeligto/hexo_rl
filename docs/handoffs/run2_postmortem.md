# run2 postmortem (stub) — facts only

**Run:** `run2_mw_fresh` · encoding `v6_live2_ls` · 4.25M net (params 4,249,675).
Launched 2026-07-05 (seeding-OFF; bot-mix retired/off — `bot_batch_share=0`).
**Stopped:** 2026-07-11 ~13:10Z by D-H CONSOLIDATE WP0 (graceful SIGTERM, no `close_out`
terminal-eval path → no livelock) to free the vast box for E1/run3. Instance stays rented.

## Final state
- **Last durable checkpoint: step 261500** (in-memory reached 261811; `run_end` fired at 261811
  but no `session_end` → the post-`finally` final-save at `loop.py:428` did not run, so 261500 is
  authoritative).
- **Bank verified** (checksums byte-identical to vast + gated load-test `label=v6_live2_ls`):
  - `checkpoints/run2_final/` — 261500 (final) + 260500/261000 + `replay_buffer.bin`
    (md5 `f24e56cd…`, consistent static copy) + `.recent.npz` + `inference_only.pt` + `checkpoint_log.json`
  - `checkpoints/run2_retro/` — 20k-stride grid 50k→255k
  - `reports/vast_run2_bank/` — reports + logs (5.1 GB)

## Self-play health at stop (final `iteration_complete`, step 261810)
- WR balanced: p0 0.4967 / p1 0.5033 · **draw_rate 0.0** (no draw collapse)
- avg_game_length 23.3 · games/h 3545 · sims/s 4494 · buffer full 250000/250000
- corpus_selfplay_frac 0.784 · `mcts_mean_depth` 2.49 — **cumulative-since-start artifact**, not a
  search regression (freezes/cliffs at restarts; see memory `mcts-depth-cumulative-metric-artifact`)

## Lineage / restart
This segment resumed **warm from `checkpoint_00250000`** after the livelock stall and ran
**250k→261811 CLEAN** — zero watchdog/stall/wedge events in-segment.

## Livelock incident (prior segment)
run2 wedged ~45h at an eval boundary near step 250k: eval-thread ⊥ self-play **concurrent GPU
forwards = CUDA livelock**. Recovered by the warm restart from 250k. Self-play stall watchdog
shipped (exit 42 after 1800s of frozen games; on master `e34fd80`+). ROOT fix (eval/self-play CUDA
isolation) was bench-gate-deferred → **now being built as WP1's promotion-gate subprocess
isolation**. Refs: `docs/designs/selfplay_stall_watchdog_design.md`, memory `run2-stall-watchdog`.

## r4→r5 radius boundary
Config-diff **ARTIFACT-CLEAN** (D-VETO V3). Depth cliffs at r-restart boundaries are the
cumulative-metric artifact (same weights), not regressions.

## Value-health final
Poor calibration; 200k radius cliff + 248k regression (D-C valprobe WP4, banked). Value-representation
weakness **confirmed** (V-CONFIRM) — the direct run3/E1 target.

## fair-slope final table
**Not emitted in-loop** — the EVALFAIR retro-slope driver (`d219334`) was not run against run2
during training. Producing the fair-slope series requires a separate evalfair pass over the banked
grid (`run2_retro` 50k→255k + `run2_final` 260500→261500). DEFERRED (out of stub scope).

## What run3 changes (one-liner)
E1 65-bin **distributional value head** (attacks the localized value-representation weakness) +
promotion-gate **CUDA subprocess isolation** (kills the eval-boundary livelock root the watchdog
only catches after the fact).
