# D-SOLVER Phase A — pre-registration + operator runbook

**Date:** 2026-06-27. **Branch:** `phase4.5/d-decide-track-b`. **Status:** A2 + A3 CLOSED (banked +
this session); **A1 instrument BUILT + soundness-verified; the full A1 tournament is operator-run on
vast (decisive gate).** Nothing committed — commits on ask.

Dispatcher: D-SOLVER (validate the tactical-solver lever moves WR, then build Phase B gated on A1=LIFT).
Premise accepted: deep value blind-spot = MCTS level-k tactical traps; net-rebuild + cheap-threat-add
both REJECTED; survivor lever = search-in-the-loop. Phase B native solver already fully designed:
`reports/d_tactical_2026-06-26/NATIVE_RUST_SOLVER_design.md`.

## Deploy net
`checkpoints/checkpoint_00272357.pt` — d1m run (`longrun_v6_live2_ls_gumbel_m16`), encoding
`v6_live2_ls`, deploy knobs `gumbel_m=16, n_sims_full=150, c_visit=50, c_scale=1, c_puct=1.5`. It is the
deployed ENDPOINT of the s150k/s175k/s200k lineage. (272357 does not reproduce the *corpus* `net_value`
— those came from the per-bucket nets — but A1 is a fresh self-contained baseline-vs-backup delta on the
deployed net, so the per-bucket provenance does not apply.)

Banked deploy-matched WR baseline (g=0, 150 sims, vs fixed-depth-5 SealBot, 100 distinct games/ckpt):
95k=45% 120k=45% 150k=51% 175k=39% 200k=48% → FLAT, ~even with SealBot.

---

## A2 — 2-stone-turn correctness — CLOSED: GRANULARITY-BINDING (single-stone), "NOT cheap" CONFIRMED
`reports/d_solver_A2/A2_verdict.md` + `scripts/dtactical/a2_turn_granularity.py` (this session; offline,
net-free, SealBot-free over `corpus.jsonl`).

- D-TACTICAL's cheap solver used SINGLE-STONE candidate gen (8% = 3/38 flipped). **28/35 = 80% of
  non-flipped proven-core mates carry a QUIET developmental winning-side stone** invisible to single-
  stone threat gen. Discriminator is clean: **flipped ⟺ all-forcing**; every determinable miss needs a
  quiet move.
- **compound_2stone = 0/35:** a within-line 2-stone win always makes the 5th stone first (→ win-in-1 →
  already threat-visible). So the gap is NOT compound forcing-turn generation but quiet moves needing
  **eval-guided quiescence**. Redoing at compound-turn granularity would NOT raise the cheap ceiling —
  confirms eval-guided quiescence (SealBot-grade) is required (consistent with `redteam_broad_search`
  0/38). The dispatcher's hoped-for "higher cheap ceiling at compound granularity" is falsified.

## A3 — solver-cost budget — CLOSED (banked, `NATIVE_RUST_SOLVER_design.md` TASK 4)
- Deploy ROOT-d6 backup affordable: ~0.37–0.65 s/move (optimized fork) / ~0.7 s median (vendored). Short
  band only. Per-LEAF d6 (×150) = 55–98 s/move = NOT affordable. d8 = training-time only.
- Training-z route DEMANDS the native Rust solver (GIL serialization across workers, colony-OOB on the
  unbounded board, time-cap nondeterminism, CANDIDATE_CAP=15 blindness) — not a SealBot pybind round-trip.
  This is exactly the SEARCH-VIABLE-BUT-NOT-CHEAP verdict.

---

## A1 — deploy solver-backup WR gate (THE decisive gate) — instrument BUILT + verified

**No Rust change.** The deploy play loop is Python; the override is a `BotProtocol` wrapper.

Files (this session, TDD, NOT committed):
- `hexo_rl/eval/solver_backup_bot.py` — `SolverBackupBot`. At a turn start, fixed-depth SealBot ROOT
  probe; override ONLY on a PROVEN mate (`|last_score| >= 99,999,000`, terminal mate within depth — never
  a heuristic). Proven WIN → play the proof move (cache the 2nd stone of a 2-stone turn, only if a
  distinct LEGAL cell, else lock the turn to the model). Proven LOSS → FLAG, do not override. Sign
  convention independently re-derived (`search.h:49` resets `_player` to side-to-move → positive =
  winning). Colony-OOB guard on BOTH cluster count (>4) and **coordinate magnitude (|q|,|r| > 60)** — the
  real OOB axis for SealBot's `[140][140]` array.
- `hexo_rl/eval/a1_stats.py` — paired bootstrap + soundness counting (pure, unit-tested).
- `scripts/eval/run_a1_solver_backup.py` — the PAIRED A1 driver.
- `tests/eval/test_solver_backup_bot.py` (13) + `tests/eval/test_a1_stats.py` (5) — **18/18 green**,
  incl. real-engine sign-convention + override tests.

Soundness verified on real games (n=6 smoke): **0 soundness violations** — every game that fired a proven
win was won. Colony guard confirmed relevant (boards drift to ≥5 clusters in deploy play). Paired pipeline
validated (n=6, backup_d6): baseline 0.500 → backup 0.833, **paired delta +0.333 driven by n_fired=2
diverging games** (the backup fired 19 proven wins but the baseline already won 4 of those games — the
paired design isolates exactly the 2 loss→win conversions; an unpaired design pays variance on all 6),
correctly gated INDETERMINATE_UNDERPOWERED at that n. Demonstrates the power gain pairing buys.

### Measurement design (PAIRED — the load-bearing fix)
Both arms share the SAME RNG-seeded openings + the same deterministic g=0 head + the same FRESH
fixed-depth SealBot opponent per game, so a per-seed game is byte-identical across arms until the
backup's first override. The statistic is the **paired bootstrap of the per-seed (backup − baseline)
delta**; non-firing games contribute exact 0 → the CI collapses onto the fired games. `n_fired` is the
power unit. (An unpaired design pays full opening variance on identical games → false-FLAT risk; fixed in
review.)

### PRE-REGISTERED criterion (per backup arm)
| verdict | condition |
|---|---|
| **LIFT_IN_WINDOW** | paired delta CI lower bound > 0 AND `n_fired >= --min-fired` (default 10) |
| **FLAT** | CI straddles/under 0 AND `n_fired >= --min-fired` |
| **INDETERMINATE_UNDERPOWERED** | `n_fired < --min-fired` (too few overrides to resolve) |

LIFT is expected PARTIAL (root-d6 catches the short band only; the mid band is the training-z route).
**LIFT_IN_WINDOW is NECESSARY-NOT-SUFFICIENT:** the override is sound, but SealBot is an in-window FLOOR
(38% in-corpus fail; colony-OOB guarded), so a fixed-bot WR lift CANNOT clear an off-window defect
(CLAUDE.md off-window false-clear gate). Phase B requires A1 LIFT_IN_WINDOW **AND** a separate
spread-uncapped / adversarial off-window probe — this verdict alone does not green-light the native build.

### MATCHED-DEPTH control (confound check)
`backup_d5` (== opponent depth) isolates trap-fix from a generic d6>d5 depth edge: if the lift persists at
matched depth, the net is missing mates WITHIN the opponent's own horizon → indefensible trap-fix.

### EXACT vast command (operator-run; ~half-day of games)
```bash
cd <repo> && git submodule update --init vendor/bots/sealbot   # ensure the SealBot .so is built (build.sh)
PYTHONPATH=. .venv/bin/python scripts/eval/run_a1_solver_backup.py \
    --checkpoint checkpoints/checkpoint_00272357.pt \
    --n-games 200 --sealbot-depth 5 --backup-depths 5,6 \
    --n-boot 2000 --min-fired 10 \
    --out reports/d_solver_A1/run1
```
Runs 3 arms (baseline, backup_d5 matched-control, backup_d6 decisive), paired. Reads
`reports/d_solver_A1/run1/A1_summary.json` (`deltas.backup_d6.verdict`). **Copy the report off the box —
`reports/` is gitignored.** Check `soundness_violations == 0` before trusting any verdict.

---

## Decision table (after the operator A1 run)
| backup_d6 | matched backup_d5 | path |
|---|---|---|
| LIFT_IN_WINDOW | also LIFT | Strong: trap-fix within the opponent's horizon. Run the off-window probe; if it also clears → **build Phase B** (native `engine::tactics` solver + `finalize_game` z-correction; deploy-backup ships as the short-band bonus). From-bootstrap if Probe-C chicken-egg favors (the current net can't guide search: 0/14). |
| LIFT_IN_WINDOW | FLAT | Lift is partly the d6>d5 depth edge — discount; still run the off-window probe before B. |
| FLAT | — | Short-band traps are not the dominant loss driver → re-scope (deeper backup test, or measure what fraction of SealBot losses ARE these traps) BEFORE any native build. Saves the weeks. |
| INDETERMINATE | — | Too few overrides fired — raise `--n-games`, or the trap class is rarer in deploy play than the corpus suggested. |

## Residual / caveats
- Soundness check is game-level (a false proof in a game won by other means is invisible). The coord-OOB
  guard + draw-inclusive loss check remove the real false-proof channels; full per-turn proof
  certification deferred (over-engineering for a WR gate).
- `reports/d_tactical_2026-06-26/` (corpus, design, red-team) and `reports/d_solver_*` are gitignored —
  unpinned. Copy to persist.
