# D-WS3 W1 — L1 solver-in-loop smoke (the GPU-day GENERALIZES/MEMORIZES gate)

**Status:** IMPLEMENTED + locally validated (Claude Code session 2026-06-30). The GPU
run is **operator-run on vast** (laptop is thermal-capped; LTO builds forbidden
locally). This runbook is the launch + discriminator + pre-registered verdict.

**The question (one sentence).** Does the L1 solver-in-loop fine-tune teach the
POLICY to play the saving move **standalone** (GENERALIZES, held-out trap-flip
≥25% → fund the GPU-week) or cap at the ~16% memorization floor (MEMORIZES →
deploy-backup is permanent for the class → save the GPU-week)?

Spec: `docs/designs/coupled_valuez_decode_design.md` §0/§1.3/§1.4/§5 (Stage-1A).
Pre-reg lineage: `docs/handoffs/d_zvalid_z2_training_z_discriminator.md`.

---

## 0. What was built this session (all committed-on-ask)

- **Rust hook** (`engine/src/game_runner/worker_loop/inner.rs::play_one_move`): per
  self-play move, the net-free `engine::tactics::TacticalSolver` proves the side-to-
  move's forced win and **SOFT-blends** visit mass onto the proving move (`line[0]`)
  into the POLICY target (the O1 `apply_forced_win_one_hot[_ls]` primitive at
  weight < 1). `window_half=None` surfaces off-window wins → the legal_set coverage
  gate routes them into the ragged multi-window target (D-DECODE action-space fix).
  WIN proofs are sound by construction → **no LOSS-derived z is written** (the R3
  guard governs the HELD L2 value-z path, not this smoke). 5 knobs, **default OFF**
  → the bench-gated hot path is byte-identical (INV19/25/26 green).
- **Knobs** (`solver_enabled/depth/node_budget/neighbor_dist/visit_weight`):
  `config.rs` get/set attrs → `pool.py` (`configs/selfplay.yaml`, default OFF) →
  the variant. NOT in `RESUME_CHECKPOINT_OWNED_KEYS` → a resume keeps them.
- **Variant** `configs/variants/z2_solver_in_loop.yaml` — single-variable vs the
  anchor (`longrun_v6_live2_ls_gumbel_m16`): only the `solver_*` block is new.
- **Held-out trap exporter** `scripts/dpfit_export_heldout_traps.py`.
- **The HEADLINE gate** `scripts/eval/run_l1_trapflip_smoke.py` (move-level deploy-
  flip, the dpfit-validated metric).
- **The OUTCOME cross-check** `scripts/eval/run_z2_standalone_ladder.py` (stubs
  wired: `_board_from_trap` move-seq replay, `trap_loss_rate` from-parent playout).

Local validation done: records `dws3_tests` 13/13; INV wiring 2/2; INV19/25/26
byte-identity; exporter → 31 clean held-out traps (25 in / 6 off-window); trap-flip
evaluator e2e (same-ckpt sanity: delta 0, KILL 0, flip 9.7%); Z2 reconstruction
parity; live solver-ON self-play (games complete, `prove()` runs in-worker, no
panic). The GPU-day run is the only thing left.

---

## 1. Preconditions (vast 5080 host)

```bash
# (a) REBUILD the binding WITH the new solver knobs (vast has no thermal limit → release/LTO ok):
make build            # = maturin develop --release -m engine/Cargo.toml
.venv/bin/python -c "import engine; c=engine.SelfPlayRunnerConfig(); assert hasattr(c,'solver_enabled'); print('solver knobs present')"

# (b) re-run the native tactics soundness sweeps on vast (the widened neighbor_dist path the
#     smoke uses — laptop could not run the #[ignore] full-width sweeps):
cd engine && cargo test --lib tactics:: -- --ignored && cd ..

# (c) anchor present + encoding pin. checkpoint_00200000 is v6_live2_ls (trained under
#     longrun_v6_live2_ls_gumbel_m16) but the .pt AUTO-DETECTS as v6_live2 (policy_pool none) —
#     ALWAYS pass --encoding v6_live2_ls to the eval scripts.
ls -la reports/d_decide_2026-06-24/checkpoints/checkpoint_00200000.pt   # the L1 warm-start
```

`make bench` is NOT required for this smoke: `solver_enabled=false` is the default
and the bench config does not set it → the hot path is byte-identical (INV-verified).
`make bench` becomes mandatory only when the deploy-root solver hook lands in
`mcts/mod.rs` (the SEPARATE deploy-backup workstream, not this).

---

## 2. The fine-tune (~5-10k steps ≈ 1 GPU-day)

WARM-START (resume) from the 200k anchor; the resume pins encoding/arch/scheduler
(LR continues its decay along the anchor's 1M cosine — do NOT set total_steps, do NOT
pass --override-scheduler-horizon); the variant supplies only the `solver_*` lever.

```bash
.venv/bin/python -m hexo_rl.training.run \
    --resume reports/d_decide_2026-06-24/checkpoints/checkpoint_00200000.pt \
    --variant configs/variants/z2_solver_in_loop.yaml \
    --iterations 8000
# → the fine-tuned candidate at checkpoints/<...>.pt  (rename to checkpoint_z2_l1.pt)
```

Watch: throughput will be LOWER than the anchor (the solver runs every move). If it
is intolerable, lower `solver_depth`/`solver_node_budget` or set `solver_neighbor_dist:
-1` (threat-only — but that drops recall to the 8% ceiling; prefer lowering the budget).
Also log the solver fire-rate (fraction of moves where a WIN was proven + injected) —
that is the z-coverage proxy; if ~0 the loop is starved (raise depth/budget/neighbor_dist).

---

## 3. THE HEADLINE GATE — move-level held-out trap-flip (PRE-REGISTERED)

```bash
# export the game-disjoint held-out corpus (CPU; default holdout_frac=1.0 = ALL 31
# held-out, correct for pure fresh-self-play — the corpus traps are frozen d_ladder
# games, disjoint from the fine-tune's fresh self-play by construction):
.venv/bin/python scripts/dpfit_export_heldout_traps.py

# the gate: baseline (200k anchor) vs candidate (fine-tuned), multi-window decode:
.venv/bin/python scripts/eval/run_l1_trapflip_smoke.py \
    --baseline-ckpt reports/d_decide_2026-06-24/checkpoints/checkpoint_00200000.pt \
    --candidate-ckpt checkpoints/checkpoint_z2_l1.pt \
    --encoding v6_live2_ls --legal-set \
    --out reports/d_zvalid_z2/l1_trapflip
```

**Pre-measured baseline (use as a WIRING sanity-check).** The 200k anchor's
multi-window deploy trap-flip on the 31 held-out traps is **≈ 9.7% (3/31), off-window
0/6** (measured locally 2026-06-30, debug build, g=0 deterministic — expect ~same on
vast). So: the static 16% floor is ABOVE the measured baseline (F4's "baseline already
>floor under multi-window" does NOT bite for this anchor); the candidate must clear
both the baseline (lift) and the 25% pass bar. If your baseline arm does NOT read ~9.7%,
the eval is mis-wired (wrong `--encoding`, wrong ckpt, or `--no-legal-set`) — fix before
trusting the candidate arm.

**PRE-REGISTERED VERDICT** (the script emits it; thresholds `--pass 0.25 --floor 0.16
--kill 0.16`):
- **GENERALIZES** — candidate held-out flip **≥ 0.25** AND > baseline flip AND the KILL
  gate clears → the net internalised the lever beyond the memorization floor → **the
  GPU-week (full training-z) is JUSTIFIED**.
- **MEMORIZES** — candidate held-out flip **≤ 0.16** → no lift over the floor → the
  decode-recovers-stranded-priors wager is FALSE → **deploy-backup is PERMANENT for the
  class**; the only remaining shot is from-bootstrap solver-in-loop from step 0 (a real
  gamble, separately gated). Saves the GPU-week.
- **KILL>16%** — normal-position deploy-disagree vs baseline **> 0.16** → the injection
  corrupted normal play → **soften `solver_visit_weight`** (e.g. 0.3 → 0.15) and
  re-smoke BEFORE any verdict. The soft/one-hot boundary is the knob. **KILL caveat:**
  the evaluator's deploy-disagree is a CHEAP COLLATERAL proxy on a quiet, early-
  midgame, solver-filtered normal set (positions with a forced win are excluded so a
  localized trap-lever's legitimate tactical improvement is NOT counted as corruption).
  It does NOT catch symmetric corruption (a candidate that shifts every move to a
  same-argmax-nearby cell scores 0). **The AUTHORITATIVE normal-play canary is
  threat-probe C1-C3** (§4c) — a KILL>16% reading should be confirmed against C1-C3 +
  the off-window co-gate before softening.
- **INDETERMINATE** — flip in (baseline, 0.25): a partial lift below the pass bar.
  Either thin power (31 traps) OR **STARVED RECALL** — the per-move solver found few
  forced wins to inject. **Before concluding the lever is weak, check the fine-tune's
  solver fire-rate / throughput** (§2): a low fire-rate (or a sharp throughput drop)
  means re-run with higher `solver_depth` / `solver_node_budget` / `solver_neighbor_dist`,
  NOT a MEMORIZES verdict. Otherwise expand the corpus (offline-mine more buckets) or
  run longer. (`INDETERMINATE_KILL_UNRUN` = the flip passed but the KILL set was empty
  → the co-gate could not clear; supply more traps / lower the normal-set floor.)

Also read the **off-window** breakdown the script prints: baseline vs candidate flip on
the 6 off-window traps — that is the multi-window-decode-recovered band working.

---

## 4. CO-GATES (all must clear for a GENERALIZES verdict)

```bash
# (a) off-window forced rate must HOLD 0.0 under no-drop (the D-DECODE floor). ABSOLUTE
#     rate is the load-bearing number (the exploit−control contrast is contaminated on
#     centroid-shifting bots; --adv-ref current is the correct effective-adversary build):
.venv/bin/python scripts/exploit_probe.py --checkpoint checkpoints/checkpoint_z2_l1.pt \
    --encoding v6_live2_ls --defender deploy --legal-set --adv-ref current

# (b) ModelPlayer false-clear cross-check (arm-aliasing-IMMUNE — the only contrast-
#     trustworthy instrument; catches a fixed-bot/deploy-head false-clear):
.venv/bin/python scripts/exploit_probe.py --checkpoint checkpoints/checkpoint_z2_l1.pt \
    --encoding v6_live2_ls --defender modelplayer --adv-ref current

# (c) threat-probe C1-C3 (the normal-play canary): must all PASS on the candidate.
.venv/bin/python scripts/probe_threat_logits.py --checkpoint checkpoints/checkpoint_z2_l1.pt
```

A GENERALIZES verdict requires: trap-flip ≥25% **AND** off-window forced rate held 0.0
**AND** C1-C3 PASS **AND** KILL ≤16%. Any co-gate regressing flips the verdict toward
KILL/INDETERMINATE — do not over-read a flip lift that came with an off-window or
normal-play regression.

---

## 5. OUTCOME cross-check (secondary, noisier — the Z2 standalone ladder)

The move-level flip (§3) is the clean headline. The outcome-level discriminator is the
strength confirmation (trap-loss-rate from the PARENT board + SealBot-WR + self-play
Elo), **backup OFF** (Z3 discipline — never re-measure the crutch). Heavier (full
playouts); run if §3 GENERALIZES and you want the strength corroboration:

```bash
.venv/bin/python scripts/eval/run_z2_standalone_ladder.py \
    --baseline-ckpt reports/d_decide_2026-06-24/checkpoints/checkpoint_00200000.pt \
    --candidate-ckpt checkpoints/checkpoint_z2_l1.pt \
    --encoding v6_live2_ls --legal-set --sealbot-depth 5 --n-games 200 --opening-plies 4 \
    --trap-set reports/d_tactical_2026-06-26/heldout_traps.jsonl \
    --finetune-game-ids reports/d_tactical_2026-06-26/finetune_game_ids.json \
    --z-loss-coverage <measured solver fire-rate from §2> \
    --out reports/d_zvalid_z2/run1
```

Note: `trap_loss_rate` plays from the PARENT board (where the saving choice exists) —
the POST board is a forced loss and carries no generalisation signal. The TEACHES arm
needs `base_trap − cand_trap ≥ 0.10`.

---

## 6. Aggregation → path (W1)

| verdict | next |
|---|---|
| **GENERALIZES** | full GPU-week training-z (soft injection + WIN-backup deploy). Success metric (W3) = STANDALONE net strength (backup OFF) gaining on the EVAL LADDER (SealBot-d5 fixed-depth + self-play BT-Elo over DISTINCT games), NOT the SealBot-WR soft number. End-state (W2) = net standalone on the ~69% + PERMANENT deploy-backup on the ~31% value-bound tail (L2 held at the D-FULLSPEC wall) — a net+tactical-backup agent, NOT a pure standalone net. Report the standalone/with-backup split explicitly; do NOT declare "standalone beats SealBot". |
| **MEMORIZES** | deploy-backup permanent for the whole class; standalone-strong net out of reach via this path → from-bootstrap solver-in-loop from step 0 is the only remaining bet (separately gated, a real gamble). Saves the GPU-week. |
| **KILL>16%** | soften `solver_visit_weight`, re-smoke before any verdict. |
| **INDETERMINATE** | expand corpus / run longer; do not over-read thin power. |

---

## 7. Durability + hygiene

- `reports/` is gitignored → **copy the run off the vast box before session cleanup**
  (`rsync -avz vast:~/hexo_rl/reports/d_zvalid_z2/ reports/d_zvalid_z2/`; same for any
  `heldout_traps.jsonl` regenerated on the box). Memory: audit artifacts must be durable.
- d1m / longrun runs are untouched — the smoke is a separate fine-tune from the 200k
  anchor.
- The exporter is deterministic (seeded); re-running it reproduces the same split.
