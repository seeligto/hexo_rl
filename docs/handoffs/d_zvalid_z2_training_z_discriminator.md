# D-ZVALID Z2 — training-z efficacy discriminator (the GPU-week gate)

**Status:** PRE-REGISTRATION + RUNBOOK + SCAFFOLD. No fine-tune run yet (gated — see §6).
**Date:** 2026-06-29. **Anchor:** `checkpoints/checkpoint_00272357.pt` (v6_live2_ls, gumbel_m=16, n_sims_full=150; D-RECONFIRM R1–R3 lineage). **venv:** `.venv/bin/python`.

## 0. The question (one sentence)

The deploy-backup (Z1) beats SealBot by **+0.195** — but that is the *runtime solver* doing the
work (the crutch), capped at the solver's reach, net no stronger. **Training-z** is the distinct
claim that *policy-via-guided-search* teaches the net to play the trap moves **without the
solver**. Z2 is one **fractional-cost** fine-tune that decides — before the GPU-week — whether the
net actually internalises the lever (**TEACHES**) or stays backup-dependent (**DOESN'T-TEACH**).

This is DISTINCT from the value-distillation **D-FULLSPEC killed** (frozen-feature class-balanced
distill could not separate win/loss). Training-z surfaces the moves in self-play so the **policy**
gets a visit signal — a different mechanism. **Probe-C (0/14)** is the pessimistic prior: the
policy may be unable to pick up ~0-prior moves even when the solver surfaces them.

## 1. Pre-registered verdict (fix BEFORE the run — a post-hoc move is storytelling)

Success metric = **STANDALONE net strength, deploy-backup OFF** vs the eval ladder. **NEVER**
deploy-backup-vs-SealBot (that re-measures the crutch — Z3 discipline, §5). The discriminator
compares the **standalone fine-tuned candidate** against the **standalone 272k baseline** on three
axes, all backup OFF:

| axis | measure | TEACHES direction |
|---|---|---|
| trap-loss-rate | fraction of a **game-disjoint held-out** trap set the standalone net loses from vs fixed-depth SealBot | **drops** ≥ `teach_trap_drop` (default 0.10 abs) |
| SealBot WR | standalone net vs fixed-depth SealBot (d5), distinct-game-deduped | **rises** ≥ `teach_wr_rise` (default 0.03 abs — a bare >0 is noise-satisfiable) |
| self-play Elo | candidate vs baseline (both standalone), distinct-game cluster-bootstrap CI | candidate CI **lower bound > 0** |

**Verdicts:**
- **TEACHES** — trap-loss-rate drops by ≥ `teach_trap_drop` **AND** (SealBot WR rises by ≥
  `teach_wr_rise` **OR** self-play Elo CI-lo > 0), backup OFF → the policy internalised the moves →
  **full GPU-week training-z justified** (solver-in-loop self-play + z-correction + WIN-backup
  deploy; success = standalone eval ladder). The WR arm is gated by a min ABSOLUTE rise so a
  noise-level distinct-WR uptick cannot manufacture a TEACHES alongside a real trap drop.
- **DOESN'T-TEACH** — trap-loss-rate flat (within noise) **AND** neither strength axis rises **AND**
  the fine-tune HAD measured z-LOSS recall (`z_loss_coverage ≥ min_z_coverage`) → the policy cannot
  pick up the surfaced moves (Probe-C generalised) → the **deploy-backup is the PERMANENT mechanism**
  (beats SealBot, capped at solver-grade); **do NOT spend the GPU-week**. Reconsider: from-bootstrap
  solver-in-loop from step 0, or heavier policy injection, or accept the crutch.
- **INDETERMINATE_STARVED_RECALL** — flat-on-both BUT z-LOSS coverage was unmeasured or below
  `min_z_coverage` (default 0.5). Without the perf layer the budget-bound full-legal verify certifies
  ~0 not-in-check quiet traps within budget — **RECALL, not soundness, is the binding constraint on
  the A2 ~80%-quiet TARGET class** (§6) — so a starved fine-tune yields flat trap-loss BY
  CONSTRUCTION. That is **not** a dead lever; calling it DOESN'T-TEACH would wrongly kill the
  GPU-week. The discriminator withholds the verdict until z-coverage is measured and meets the floor.
- **INDETERMINATE** — too few distinct games per pair (< `min_distinct_per_pair`, §D-ARGMAX power
  floor), no trap set / unrun trap playout, or a mixed signal (one axis moves, the other flat).
  Re-run powered / wire the trap set; do not over-read a mixed result.

This binary is what one fractional-cost fine-tune **plus the held-out trap playout at adequate
z-LOSS coverage** resolves before the GPU-week (the fine-tune alone, with no trap playout, returns
INDETERMINATE — the trap corpus + playout are also load-bearing; see §9 open items).

## 2. The fine-tune (the candidate the discriminator evaluates)

Short solver-in-loop fine-tune from the 272k anchor, **~10–30k steps** (fractional vs a 200k–1M
run). LR continues its decay from the anchor (a NEW `total_steps` horizon, **not**
`--override-scheduler-horizon`).

- **Resume precedence (load-bearing):** `hexo_rl/training/orchestrator.py::build_resume_config_overrides`
  filters out `RESUME_CHECKPOINT_OWNED_KEYS` (line 233 — encoding/arch/scheduler pins win from the
  checkpoint); the **variant wins all other knobs**. The NEW solver knobs
  (`solver_enabled`, `solver_depth`, `solver_node_budget`, `solver_neighbor_dist`) must therefore
  live in the **variant config** and must **NOT** be added to the OWNED set, so resume keeps them.
  (See the resume-variant-override fix: the orchestrator precedence is correct only when the knob
  is outside the 18-key OWNED EXCLUDE set.)
- **Encoding pinned** by the checkpoint (`v6_live2_ls`) — do not override.
- The fine-tune produces `checkpoints/<finetuned_z>.pt`. That is the **candidate**.

## 3. The eval ladder we HAVE (no KrakenBot)

**KrakenBot is OUT** — we do not have its weights. The ladder (from `configs/eval.yaml` +
`scripts/eval/run_z2_standalone_ladder.py`):

- **SealBot, fixed-depth (d5)** — the reproducible external bar (`SealBotBot(max_depth=5)`, NOT a
  wall-clock `time_limit` — §D-LADDER reproducibility). This is the primary strength rung.
- **Self-play BT-Elo** — candidate vs baseline, both **standalone**, `aggregate_games` +
  `bootstrap_ratings_ci` over **DISTINCT** games (the §D-ARGMAX guard: the g=0 deterministic deploy
  regime collapses to ~2 games/pair without opening diversity; a raw-count CI is over-confident by
  √copies). Opening diversity via `--opening-plies 4`.
- **trap-loss-rate** — the internalisation signal; held-out, game-disjoint (§4).
- Existing rungs (`best_checkpoint`, `bootstrap_anchor` v7full) remain available for context but are
  NOT the discriminator. `nnue` (Hammerhead) exists but is OFF by default and is not part of this
  gate.

**Deploy-matched head:** `DeployHeadBot` — Gumbel-SH greedy winner, **g=0** (no root noise), **no
temperature**, deploy sims read from the checkpoint config (`gumbel_m`, `n_sims_full`). This is the
regime the net actually deploys (the §D-LADDER triple-miss fix). `--legal-set` routes through the
Track-2 multi-window decode — match whatever the fine-tune deployed under.

## 4. Trap-loss-rate (held-out, game-disjoint)

A trap = a board the net mis-evaluates and loses from (the §D-TACTICAL / D-LOCALIZE proven-loss
class — `net_value ≈ +0.6..1.0` at a d6 forced loss). Protocol: from each trap position, the
**standalone** net (to move) plays out vs fixed-depth SealBot; count terminal losses.

- **GAME-DISJOINT:** the held-out trap set MUST be drawn from games **not** in the solver-in-loop
  fine-tune corpus — else the drop measures memorisation, not generalisation (REVIEW concern). Split
  the D-TACTICAL proven-loss corpus by `source_game_id` before the run; reserve a held-out slice.
- **Format** (`load_trap_set` docstring in the scaffold): one JSONL row per trap —
  `{"stones": [[q,r,player],...], "current_player", "moves_remaining", "encoding", "source_game_id"}`,
  or (recommended) a **move-sequence** form replayed through the audited legal apply path (avoids
  bbox/turn-phase drift). **TODO(z2-corpus):** wire the exporter + the disjoint split; the scaffold
  loader/reconstruction is a documented stub until then.

## 5. Z3 — success-metric discipline (standing)

The eventual full training-z run's success is **standalone net strength vs the eval ladder, backup
OFF** — never deploy-backup-vs-SealBot. The scaffold (`run_z2_standalone_ladder.py`) **never wraps
the candidate in `SolverBackupBot`** — that is how backup-OFF is enforced structurally. Bank the
distinction so the run is not declared a win on the runtime tool.

## 6. The Rust solver-in-loop hook (the DEPENDENCY)

The fine-tune needs the solver running in self-play to surface moves. This hook is **greenfield
Rust** and is the gate on Z2 actually running:

- **Where:** `engine/src/game_runner/worker_loop/inner.rs` — the per-move select-and-apply loop
  (≈ L1042–1110) and `finalize_game` z-assignment (≈ L1253). Reference:
  `reports/d_tactical_2026-06-26/NATIVE_RUST_SOLVER_design.md` §2.4 + §3.6(b).
- **What:** per self-play move, run `engine::tactics::TacticalSolver` at a training budget (d8-class,
  amortised offline). On a proven **WIN**, realise the forcing line so the trap is converted and `z`
  is naturally correct AND the policy gets a visit signal on the winning move. On a proven **LOSS**,
  stamp corrected `z = -1` (`value_valid = 1`) on the recorded row.
- **Soundness gate — NOW MET for WIN; conditional for LOSS:** the **R3 LOSS-completeness guard
  landed** (Z0-A, `engine/src/tactics/search.rs`), so native z-LOSS labels are no longer
  emergent-unsound. WIN proofs are sound by construction (terminal backup). **But** not-in-check
  trap LOSS *recall* depends on Z0-B/Z0-C: the solver only **proves** a not-in-check LOSS when the
  candidate set covers the full legal set (`neighbor_dist` covering the radius → the guard's
  `moves_len >= legal_move_count()` branch). So set `solver_neighbor_dist` high enough for the trap
  band, or rely on WIN-line realisation (always sound) for the primary signal. **Do NOT** generate
  native z-LOSS labels through the threat-only solver (it cannot prove the quiet-trap LOSS class).
- **Bench:** the deploy/training hook is on a path `make bench` covers — re-gate (MCTS ≥ 73k sim/s)
  before committing the hook.

## 7. Vast operator-run commands

Heavy compute is operator-run on vast (5080 + 9900X). Mirror the D-SOLVER / D-RECONFIRM runbook
style. **Reports under `reports/` are gitignored — copy off the vast box before session cleanup**
(memory: audit artifacts must be durable).

```bash
# (a) the fractional fine-tune (after the Rust hook lands + bench-gates):
.venv/bin/python -m hexo_rl.training.run \
    --resume checkpoints/checkpoint_00272357.pt \
    --variant configs/variants/<z2_solver_in_loop>.yaml \
    --total-steps 30000        # ~10-30k; LR continues decay, NOT --override-scheduler-horizon

# (b) the STANDALONE discriminator (backup OFF). --trap-set DEGRADES GRACEFULLY (trap axis ->
#     None, verdict INDETERMINATE) until the held-out corpus + playout land — it does NOT crash.
#     Pass --z-loss-coverage (MEASURED from the fine-tune logs: fraction of held-out quiet traps
#     the solver certified a z-correction for within budget) so a flat result is not mis-read as
#     DOESN'T-TEACH (starved recall); --finetune-game-ids enables the fail-closed disjointness
#     assertion on the trap set.
.venv/bin/python scripts/eval/run_z2_standalone_ladder.py \
    --baseline-ckpt checkpoints/checkpoint_00272357.pt \
    --candidate-ckpt checkpoints/<finetuned_z>.pt \
    --encoding v6_live2_ls --sealbot-depth 5 --n-games 200 --opening-plies 4 \
    --trap-set reports/d_tactical_2026-06-26/heldout_traps.jsonl \
    --finetune-game-ids reports/d_zvalid_z2/finetune_game_ids.json \
    --z-loss-coverage <measured_fraction> \
    --out reports/d_zvalid_z2/run1

# copy results off-box:
rsync -avz vast:~/hexo_rl/reports/d_zvalid_z2/ reports/d_zvalid_z2/
```

The variant `<z2_solver_in_loop>.yaml` carries the new solver knobs
(`solver_enabled: true`, `solver_depth`, `solver_node_budget`, `solver_neighbor_dist`) — outside the
OWNED set so resume keeps them (§2).

## 8. Aggregation → path

| Z2 verdict | next |
|---|---|
| **TEACHES** | full GPU-week training-z (solver-in-loop self-play + z-correction + WIN-backup deploy); success = standalone eval ladder. The path to a net that beats SealBot WITHOUT the crutch. |
| **DOESN'T-TEACH** | deploy-backup is permanent (beats SealBot, solver-grade cap); pivot the net-strength question (from-bootstrap solver-in-loop, or accept the crutch). Saves the GPU-week. Only EARNED with measured adequate z-coverage. |
| **INDETERMINATE_STARVED_RECALL** | the fine-tune was recall-starved (z-coverage unmeasured / below floor) — the lever never got a fair test (the budget-bound verify certified ~0 quiet traps without the perf layer). Land the perf layer (α-β makes the verify affordable) OR raise the solver budget/`neighbor_dist`, then re-run. Do **NOT** kill the GPU-week. |
| **INDETERMINATE** | wire the held-out trap set / re-run powered (distinct-per-pair ≥ floor); do not over-read. |

## 9. Open items for the user / operator

1. **Held-out trap corpus** — needs an exporter + game-disjoint split (the D-TACTICAL proven-loss
   positions). Scaffold loader is a stub. Decide the corpus format (move-sequence replay recommended).
2. **`teach_trap_drop` threshold** — default 0.10 abs; confirm against the trap-set size / expected
   effect before the run (pre-reg discipline).
3. **The Rust hook** is gated on Z0-C completion (the quiet-move body). Until then Z2 cannot RUN —
   this doc + the scaffold are the design + tooling.
