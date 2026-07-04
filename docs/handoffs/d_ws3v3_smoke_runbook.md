# D-WS3V3 — L1 solver-in-loop smoke DONE RIGHT (the GPU-day GENERALIZES/MEMORIZES gate, v3)

**Status:** IMPLEMENTED + locally validated (Claude Code session 2026-07-02, laptop
CPU-only — thermal-capped, no LTO/release builds locally). The GPU run is
**operator-run on vast**. This runbook is the launch + validity gate + discriminator
+ pre-registered verdict for the THREE-ARM v3 re-run.

**The question (unchanged from v2, one sentence).** Does the L1 solver-in-loop
fine-tune teach the POLICY to play the saving move **standalone** (GENERALIZES,
held-out trap-flip ≥25% → fund the GPU-week) or cap at the ~16% memorization floor
(MEMORIZES → deploy-backup is permanent for the class → save the GPU-week)? v3 adds
a second axis: does trap-corpus **start-position seeding** (density) move that
answer, distinguishing GENERALIZES from a fire-rate-starved false MEMORIZES read?

Spec: `docs/designs/coupled_valuez_decode_design.md` §0/§5 (Stage-1A). Predecessor:
`docs/handoffs/d_ws3_l1_smoke_runbook.md` (v2, SUPERSEDED — see its banner).

---

## 0. Why v3 (the v2 confound, with the exact numbers)

v2 (2026-07-01/02, `docs/handoffs/d_ws3_l1_smoke_runbook.md`) ran ONE candidate arm
(`z2_solver_in_loop.yaml`, solver ON, `solver_visit_weight=0.3`) resumed via
`--resume checkpoint_00200000.pt`. That checkpoint is a FULL checkpoint (carries
`config`/`optimizer_state`/`scaler_state`), so `trainer_ckpt_load.load_checkpoint`
classified the resume `is_full_ckpt=True` and the run inherited the ANCHOR's LR
scheduler STATE, not the variant's `lr: 2e-3` / `eta_min: 5e-4`:

- **LR confirmed from the run logs**: `reports/vast_ws3_v2/logs/ws3_z2_l1.jsonl`
  last `train_step` (step 200802) reads `lr=0.0018557751539875656`; the sibling
  control run (`ws3_ctrl.jsonl`, step 208000) reads `lr=0.0018456274188078784` —
  both ≈**1.85e-3**, matching the anchor's 1M-step cosine schedule evaluated at
  step ~200k (memory-recorded figure: 1.8569e-3), NOT the variant's intended 2e-3
  start / 5e-4 floor for a FRESH 8k-step decay.
- **Buffer cold-started**: `ws3_ctrl.jsonl`'s first `warmup` event reads
  `{"buffer": 0, "target": 256, "games": 0}` — the replay buffer had been rm'd
  before the run and `min_buffer_size` was the training.yaml default 256 (not a
  real prefill gate), so training began on ~0 real self-play positions.
- **ENCODING (FIX1, found 2026-07-02 building v3): the anchor checkpoint's
  `metadata['encoding_name']` is STALE `'v6_live2'` (single-window)**, and
  `trainer_ckpt_load.py` prefers ckpt metadata unconditionally +
  `orchestrator.py` back-propagates it into the combined config that
  `pool.py` resolves for self-play. v2 therefore ALSO self-played
  single-window `v6_live2` despite `z2_solver_in_loop.yaml` declaring
  `v6_live2_ls` (multi-window legal-set) — the z2 variant's own header claim
  ("HARD INVARIANT graft A ... Confirmed, no mismatch") was FALSE AT RUNTIME.
  This means v2's off-window injection routing (`window_half=None` →
  legal_set ragged target) was DEAD the whole time (the Dense/single-window
  path drops off-window wins outright). See `configs/variants/
  z2_solver_in_loop.yaml`'s correction note and `docs/07_PHASE4_SPRINT_LOG.md`
  §D-WS3 for the full retro-correction + an OPEN FORENSIC QUESTION about how
  far back this cascades.

A **solver-OFF CONTROL arm** (`z2_solver_in_loop_control.yaml`, same broken LR +
cold-buffer regime) was launched in parallel and reproduced the candidate's damage
almost exactly:

| metric | control (solver OFF) | candidate (solver ON, w=0.3) |
|---|---|---|
| trap-flip, registered-31 | 12.9% → **6.5%** | 12.9% → **6.5%** |
| trap-flip, combined-125 | 19.2% → 15.2% | 19.2% → 15.2% |
| deploy-disagree (KILL co-gate, n=55) | 49.1% (27/55) | 47.3% (26/55) |
| threat-probe C1 (contrast, vs bootstrap) | +5.393 PASS | +6.986 PASS |
| threat-probe C2 (top5, vs bootstrap) | **30% PASS** | **20% FAIL** |
| threat-probe C3 (top10, vs bootstrap) | 45% PASS | 40% PASS |
| off-window forced rate (deploy + modelplayer) | 0.0 | 0.0 |

Both arms crater the SAME direction on the SAME held-out corpus by the SAME
magnitude → the regime drift (LR + cold buffer), not the solver injection,
did the damage. **The injection is EXONERATED but the v2 read is INDETERMINATE**
(a broken-regime null result proves nothing about the lever). C2 is the one
metric that differs between arms (30% vs 20%) — real but too thin (n=20, single
metric) to carry a verdict alone, and it was gated against the WRONG baseline
(the bootstrap's `contrast_mean=+0.599`, not the anchor's own `+7.309` — the
anchor is 12× sharper than bootstrap on this probe, so gating a resumed
fine-tune against bootstrap is nearly a no-op floor).

**Also measured (v2, offline, no in-run logging existed then):** solver fire-rate
5.6% of moves overall / 16.4% late-game — a real fire-rate-starvation risk that a
fresh, unseeded self-play run may simply never reach a position the solver has
anything to prove at.

**v3 fixes, one lever each:**

| # | v2 hole | v3 fix |
|---|---|---|
| V0a | full-ckpt resume can't pin LR | weights-only warm-start (`scripts/make_ws3v3_warmstart.py`) → `lr: 1.0e-4` FLAT (`lr_schedule: none`) |
| V0b | buffer cold-started at 256 | `min_buffer_size: 25000` (POSITIONS) real prefill gate + per-arm buffer paths |
| V0c | no in-run fire-rate visibility | Rust counters (`solver_moves_eligible/injected/...`) + `training_step.solver_fire_rate` |
| V0d | C2 gated vs bootstrap, not the anchor | `--baseline-json tests/fixtures/threat_probe_baseline_anchor200k.json` |
| V1 | fire-rate starved on fresh self-play | trap-corpus START-POSITION SEEDING (`selfplay.seed_fraction`, KataGo `startPoses`) |
| V2 | weight 0.3 confounded (can't separate lever from regime) | re-run at `solver_visit_weight: 0.5` under the fixed regime |
| V3 | one arm, no control | THREE arms: CONTROL / INJECT / SEEDED |

---

## 0.5 ARM-SEEDED's mechanism, HONESTLY restated (FIX2, 2026-07-02)

It is DUAL, not a single `fire_rate_seeded ~1.0` knob:

1. **Trap-DECISION z-labels densify ALWAYS** — the KataGo `startPoses` value:
   every seeded game starts at (or near) a trap-corpus position, so the
   *outcome*-derived training signal near that decision densifies regardless
   of whether the native solver ever fires. This part of the lever needs no
   solver proof at all.
2. **Solver POLICY-injection densifies only on native-provable conversions.**
   Red-team measured (FIX2, 2026-07-02): seed landing positions at cuts
   k∈{0,2,4} — BEFORE the trap parent — prove NO forced win **by
   construction** (the parent is a "saving move exists" position; there is
   nothing yet for the solver to prove there). Even the POST-blunder position
   (where SealBot has proven the model lost) is often NOT provable by the
   native `engine::tactics::TacticalSolver` at a realistic self-play node
   budget (D-TACTICAL A2 — weak recall on quiet traps). The honest ceiling for
   how often solver policy-injection fires on POST-blunder seeds is
   `scripts/measure_native_provable_fraction.py`'s measured fraction (FIX2a,
   §3.5 below) — **NOT** ~1.0. `scripts/build_ws3v3_seed_corpus.py` (FIX2b)
   additionally emits POST-blunder seeds (`cut: -1`) ONLY for traps the native
   solver can actually prove at build time (computed fresh, NOT the miner's
   stored `native_loss_verified` flag — the DS1 stale-label lesson), so those
   specific seeds fire the solver injection near-immediately by construction;
   the k∈{0,2,4} parent-cut seeds do not.

**Practical read:** `solver_fire_rate_seeded` reads as a MIX of (near-0 fire
on the k∈{0,2,4} parent cuts) + (near-1.0 fire on the `_kpost` seeds, weighted
by how many of those exist in the corpus — per §3.5's MEASURED 0/40 [CI 0,
0.088] native-provable fraction, expect few-to-ZERO `_kpost` seeds, so this
term is likely absent) + (organic fire on the non-seeded tail of the game).
Do not expect it to read near the seed_fraction's raw density — see §3.5 for
the measured ceiling + its implication and §6's SEED-STARVED verdict row for
the pre-registered numeric check (SEED-STARVED is the DEFAULT expectation
given the measurement).

---

## 0.6 A1 INSTRUMENT RE-BASELINE (amendment, 2026-07-03) — VERDICT: MATCH

D-EVALGATE's "intended drift" (the gated eval loader resolves the strip's stamp
`v6_live2_ls` where the old shape/filename inference said `v6_live2`) forced a
pre-registered re-baseline BEFORE any v3 arm read: arm-vs-baseline must not
cross instruments.

**RED-TEAM provenance record (the original record didn't say — now it does):**
- The original 12.9% (4/31) / 19.2% (24/125) flip baselines were measured on the
  **RAW full checkpoint** `reports/d_decide_2026-06-24/checkpoints/checkpoint_00200000.pt`
  (stamp: stale `v6_live2`), NOT the strip — via
  `gumbel_greedy_bot._build_engine` (blind `torch.load`, **stamp never
  consulted**), decode forced by bare `--encoding v6_live2_ls` (the evaluator's
  default), `legal_set=True`, deploy knobs g=0/m=16/n_sims=150 (deterministic).
  Trap sets: `heldout_traps.jsonl` (31) / `heldout_traps_all.jsonl` (125).
  Host: vast 5080. Decode geometry was therefore ALREADY `v6_live2_ls` — the
  old-loader mislabel never touched the flip decode, only the label.
- Strip `ws3v3_warmstart_200k.pt` weights verified **byte-identical** to the raw
  200k `model_state` (2026-07-03, per-tensor `torch.equal` over the full dict).

**A1 re-measure (laptop 4060, gated loader, `--expect-encoding v6_live2_ls`,
strip as baseline ckpt, same sets/knobs/script):**

| set | original (vast, old instrument) | A1 (laptop, gated) | pre-reg band | verdict |
|---|---|---|---|---|
| registered-31 | 4/31 = 12.9% | **3/31 = 9.7%** | count ∈ [2,6] | **MATCH** |
| combined-125 | 24/125 = 19.2% | **22/125 = 17.6%** | count ∈ [19,29] | **MATCH** |

Attribution discriminator (ran both instruments on the SAME host): old
instrument (raw ckpt + bare `--encoding`) on the laptop = **byte-identical**
3/31, 22/125 to the gated read → the instrument correction is a **no-op** on
flip baselines; the residual −1/−2 flips vs vast are **host fp** (borderline
argmax flips, scattered both directions per-trap: 3/31 resp. 5/125 traps
changed class). Artifacts: `reports/d_ws3v3/a1_rebaseline_{31,125}/`,
`a1_oldinstrument_local_{31,125}/` (summary JSON now records
`encoding_decode`/`expect_encoding`/`loader`/`trap_set`).

**Probe C1–C3 anchor numbers:** reproduce **byte-exact** through the gated
loader — C1 +7.310 / C2 30% / C3 40%, C4 Δ=0.000 — PROVIDED the probe reads the
`v6_live2` position fixture. On `_ls`-stamped ckpts (the strip and every v3 arm
ckpt) the probe's implicit fixture fallback previously swapped SILENTLY to the
default v6 fixture and manufactured C1 +10.97 / C2 45% / C3 75% (different
position set, would have PASSed); the probe now hard-fails that fallback —
**always pass `--positions tests/fixtures/threat_probe_positions_v6_live2.npz`
on _ls-stamped ckpts** (valid: v6_live2/_ls are trunk- and shape-identical).

**Consequences for §3/§4 reads (all pre-registered here, none moved silently):**
1. Validity bands STAND as written ([2,6] / [19,29], C1≥5.848 / C2≥25% / C3≥40%).
2. All flip reads use `--expect-encoding v6_live2_ls` (never bare `--encoding`)
   and the **strip** as `--baseline-ckpt` (the raw 200k's stale stamp now
   correctly RAISES under the assertion). First gated read of a vast-side arm
   ckpt doubles as the loader's vast confirmation.
3. **HOST-MATCH rule** (new, from the discriminator): the baseline an arm is
   compared against must come from the SAME host as the arm read. Arm reads on
   vast → re-run the strip self-baseline once on vast first (expect ≈4/31,
   24/125; minutes). Arm reads on the laptop → the A1 numbers (3/31, 22/125)
   are the operative baseline and the bands re-center on them ([1,5] / [17,27]).

---

## 1. Vast preconditions

```bash
# (a) rebuild the binding WITH the seed-corpus + counter additions (vast: release/LTO OK):
make build
.venv/bin/python -c "
import engine
c = engine.SelfPlayRunnerConfig()
for attr in ('solver_enabled', 'solver_depth', 'solver_node_budget',
             'solver_neighbor_dist', 'solver_visit_weight', 'seed_corpus'):
    assert hasattr(c, attr), attr
print('solver + seed-corpus knobs present')
"
# counters are read off the pool/worker summary, not the config object — confirm via
# a short smoke run below (§2) that training_step events carry solver_fire_rate.

# (b) PARK (mv, DO NOT rm) the v2-contaminated shared buffer — it was overwritten
#     every 500 steps by BOTH v2 smoke arms and must not silently feed v3:
mv /workspace/hexo_rl/checkpoints/replay_buffer.bin \
   /workspace/hexo_rl/checkpoints/replay_buffer.bin.ws3v2 2>/dev/null || true

# (c) mine the seed corpus (CPU SealBot, ~1h for the full untapped population):
.venv/bin/python scripts/build_ws3v3_seed_corpus.py
# -> reports/d_ws3v3/seed_corpus.jsonl + reports/d_ws3v3/seed_corpus_stats.json

# (d) MEASURED disjointness — paste this output into the run log before trusting
#     ARM-SEEDED's flip numbers:
.venv/bin/python scripts/check_ws3v3_disjointness.py
# exit 0 = clean. Prints the leakage-distance distribution (max/p95/median) too.

# (e) strip the warm-start checkpoint + paste its key-assert output into the run log.
#     FIX1 (BLOCKER, 2026-07-02): the source checkpoint's metadata['encoding_name']
#     is STALE ('v6_live2', single-window) — trainer_ckpt_load.py prefers ckpt
#     metadata unconditionally and orchestrator.py back-propagates it into the
#     variant's combined config, so an unfixed strip would make all three arms
#     self-play single-window v6_live2 despite declaring v6_live2_ls. The script
#     now defaults --encoding-name to 'v6_live2_ls' and RE-STAMPS
#     metadata['encoding_name'] after asserting the override shares the
#     original's wire signature (n_planes/board_size/policy_logit_count/
#     has_pass_slot/sym_table_id — v6_live2 and v6_live2_ls share it exactly,
#     (4, 19, 362, True, "size_19"), so the weights still load no-reshape).
#     Prints the old->new stamp — paste it into the run log:
.venv/bin/python scripts/make_ws3v3_warmstart.py
# -> checkpoints/ws3v3_warmstart_200k.pt {model_state, metadata, step}; sha256 printed;
#    "encoding stamp: 'v6_live2' -> 'v6_live2_ls' (wire_signature (4, 19, 362, True, 'size_19') unchanged)"
#    LOCALLY REGENERATED + VERIFIED 2026-07-02 (see FIX1d evidence below §2).

# (f) mint the ANCHOR threat-probe baseline (once; laptop already minted a CPU copy —
#     re-run on vast if the GPU numbers should differ from the CPU-measured ones:
#     contrast_mean +7.3095, top5 30%, top10 40%, n=20).
#     A1 amendment 2026-07-03: VERIFIED byte-exact through the gated loader
#     (raw ckpt stamp v6_live2 -> v6_live2 fixture auto-selected; C4 delta 0.000)
#     — this mint command stands unchanged:
.venv/bin/python scripts/probe_threat_logits.py \
    --checkpoint reports/d_decide_2026-06-24/checkpoints/checkpoint_00200000.pt \
    --write-baseline-to tests/fixtures/threat_probe_baseline_anchor200k.json

# (g) make bench (game_runner touched this build — vast-only; do NOT run on laptop):
make bench

# (g2) ANCHOR INCUMBENT RE-STAMP (ADDED 2026-07-04 after a live launch crash — this
#      precondition is REQUIRED since the D-EVALGATE/A0 anchor gates, commits
#      9729e5b..a33f7df, which POST-DATE this runbook's §1 list). resolve_anchor
#      hard-refuses a best_model.pt whose stamp disagrees with the arm variants'
#      declared v6_live2_ls (anchor.py:249 "refusing to fall through" — observed:
#      the stale v6_live2 incumbent dc369bae…597228 crashed ARM-CONTROL 40s after
#      launch, BEFORE any training). Park the incumbent (mv, do not rm — it may be
#      a production best model) and install the _ls-stamped warmstart as the
#      smoke's incumbent (the natural arm-relative baseline: the starting net):
mv checkpoints/best_model.pt checkpoints/best_model.pt.pre_ws3v3_$(date +%Y%m%d)
cp checkpoints/ws3v3_warmstart_200k.pt checkpoints/best_model.pt
# (no expected_anchor_sha256 pin in the arm variants — no pin update needed)

# (h) optional: re-run the native tactics soundness sweeps (the widened
#     neighbor_dist path this smoke uses):
cd engine && cargo test --lib tactics:: -- --ignored && cd ..

# (i) OPTIONAL but recommended — the FULL-125 native-provable measurement
#     (FIX2a, §3.5). The laptop only ran a SAMPLED n=40 estimate (debug build,
#     ~60-80s/trap CPU); the release build is ~2.5x faster, making the full
#     125 ~1h — replaces the sampled CI with the exact population number:
.venv/bin/python scripts/measure_native_provable_fraction.py \
    --traps reports/d_tactical_2026-06-26/heldout_traps_all.jsonl \
    --out reports/d_ws3v3/native_provable_fraction.json
```

---

## 2. The three launches (SEQUENTIAL — single GPU, one arm at a time)

**LAUNCH-ORDERING STOP-GATE (pre-registered, not just a read-order suggestion):**
launch ARM-CONTROL alone first. Do NOT launch ARM-INJECT or ARM-SEEDED until
ARM-CONTROL has completed AND passed the §3 validity gate. ARM-INJECT/ARM-SEEDED
together are ~23 estimated GPU-hours (§2 wall-clock table below) — all of it
downstream of a gate that can fail and invalidate the whole batch, exactly the
mistake v2 made (a candidate read against a regime nobody had validated first).

```bash
# --variant takes the BARE NAME (configs/variants/{name}.yaml is constructed
# internally by scripts/train.py) — a yaml PATH here fails with
# FileNotFoundError (verified: configs/variants/{path}.yaml does not exist).
# --checkpoint-dir + --run-name are REQUIRED per arm: without them all three
# arms write the SAME checkpoints/checkpoint_00208000.pt (absolute-step
# naming — see the note below) and the SAME default logs/train_<run_id>.jsonl,
# clobbering each other.

# ── ARM-CONTROL — launch and read ALONE first ────────────────────────────
.venv/bin/python scripts/train.py \
    --checkpoint checkpoints/ws3v3_warmstart_200k.pt \
    --variant ws3v3_arm_control \
    --checkpoint-dir checkpoints/ws3v3_control \
    --run-name ws3v3_control \
    --iterations 8000
# -> checkpoints/ws3v3_control/checkpoint_00208000.pt (ABSOLUTE step: the
#    trainer's self.step starts at the warm-start's own step field (200000)
#    and increments per real training step — filename is
#    f"checkpoint_{self.step:08d}.pt", NOT relative to the run length. 8000
#    iterations from step 200000 -> 00208000, not 00008000. Verify against
#    the printed step in the checkpoint_saved log line regardless.)

# ── STOP — run §3 (the validity gate) here. Only on PASS continue below. ──

# ── ARM-INJECT ────────────────────────────────────────────────────────────
.venv/bin/python scripts/train.py \
    --checkpoint checkpoints/ws3v3_warmstart_200k.pt \
    --variant ws3v3_arm_inject \
    --checkpoint-dir checkpoints/ws3v3_inject \
    --run-name ws3v3_inject \
    --iterations 8000
# -> checkpoints/ws3v3_inject/checkpoint_00208000.pt

# ── ARM-SEEDED ─────────────────────────────────────────────────────────────
.venv/bin/python scripts/train.py \
    --checkpoint checkpoints/ws3v3_warmstart_200k.pt \
    --variant ws3v3_arm_seeded \
    --checkpoint-dir checkpoints/ws3v3_seeded \
    --run-name ws3v3_seeded \
    --iterations 8000
# -> checkpoints/ws3v3_seeded/checkpoint_00208000.pt
```

**Startup/config-echo caveat:** the config object the trainer PRINTS at startup is
the PRE-MERGE combined config (base + variant, before the checkpoint's owned keys
reconcile) — v2's own startup log echoed `"lr": "2e-3", "lr_schedule": "cosine",
"total_steps": 200000` verbatim even though the ACTUAL resume used the anchor's
1.85e-3 / cosine-to-1M state. **Do not trust the startup echo for the LR — verify
in-run** (below).

**Expected fresh-AMP-scaler warmup (NOT a kill signal):** an early `train_step`
event or two of a weights-only warm-start can log `grad_norm=inf` — the AMP
gradient scaler starts cold (no `scaler_state` in the stripped checkpoint) and
skips its first scale-adjustment step(s) correctly; the scheduler-step is also
correctly skipped (`lr_schedule: none`). This clears within a handful of steps.
LR pin itself was verified LOCALLY 2026-07-02 via a real 3-step launch on the
fixed encoding (`--variant ws3v3_arm_control --iterations 3
--min-buffer-size 64`, `logs/train_60cc566090434086a8218697040a54ed.jsonl`):
`train_step` `lr` read `0.0001` flat across all 3 steps (`grad_norm` was
`1.8132`, `inf`, `3.1786` — one of the three, not all, inf), step continuation
`200000 -> 200001 -> 200002 -> 200003` (absolute, matching the warm-start's
own step field, confirming the checkpoint-filename note above). `checkpoint_
encoding_resolved`/`train_encoding_resolved` both read `v6_live2_ls`
(`is_multi_window: true`) and no `checkpoint_encoding_overrides_variant`
warning fired (variant and ckpt-resolved encoding agree post-FIX1) — see FIX1
in this session's changelog for the full event dump.

**Acknowledged intentional diff vs the anchor recipe:** all three arm variants
set `selfplay.n_workers: 24` (not the anchor's `32`) — matches the vast 5080
host's actual `nproc` (32 oversubscribed the host under the per-move solver CPU
cost, load ~29 measured in v2); this is the SAME `n_workers` v2's runs actually
used, so it is regime-consistent, not a new variable.

**In-run LR verification (must read 1.0e-4 FLAT for the whole run):**

```bash
# last train_step event's lr field — grep the run's own JSONL, not the startup echo:
tail -c 2000000 logs/ws3v3_<arm>.jsonl | grep -o '"event": *"train_step".*"lr": *[0-9.e+-]*' | tail -1
# or, once the run has produced a handful of steps:
python3 -c "
import json
lrs = [json.loads(l)['lr'] for l in open('logs/ws3v3_<arm>.jsonl') if '\"event\": \"train_step\"' in l]
print('first', lrs[0], 'last', lrs[-1], 'min', min(lrs), 'max', max(lrs))
"
# PASS: first == last == 1.0e-4 (within fp rounding). Any drift means the strip
# (§1e) or the resume path silently reactivated full-ckpt semantics — STOP and
# re-check the warm-start checkpoint's key set before spending more GPU time.
```

**In-run fire-rate watch** (`training_step` event fields, per the interface
contract): `solver_eligible_per_step`, `solver_injected_per_step`,
`solver_fire_rate` (injected/eligible delta, null-safe), `solver_fire_rate_seeded`
(seeded-games-only slice). ARM-SEEDED's `solver_fire_rate_seeded` should sit near
the seeded-game density target (close to 1.0 on seeded games, since a seed prefix
is chosen to land near a proven position); if it reads near-0 even on seeded games,
the seed prefixes are landing on already-resolved boards — widen the cut set or
re-check `check_ws3v3_disjointness.py`'s terminal-skip count.

**Per-arm wall-clock estimate** (REAL numbers, computed from
`reports/vast_ws3_v2/logs/ws3_ctrl.jsonl` and `ws3_z2_l1.jsonl` train_step /
game_complete timestamps — v2's hardware is the same 5080 host, `n_workers: 24`
unchanged):

- **ARM-CONTROL** (solver OFF — throughput ≈ v2's control arm, `ws3_ctrl.jsonl`):
  8000 `train_step` events span 04:32:15→07:28:40 (2.940h) → **2721 steps/hour**.
  8335 `game_complete` events in that window, mean 51.2 plies/game → **~129,000
  positions/hour**. 8000-step run ≈ **~2.9h**. Prefill to `min_buffer_size: 25000`
  positions at that rate ≈ 25000/129000h ≈ **~12 minutes**.
- **ARM-INJECT / ARM-SEEDED** (solver ON — ESTIMATED, not measured, since v3
  hasn't run yet): v2's candidate arm (`ws3_z2_l1.jsonl`, solver ON at
  `solver_node_budget=50000`) measured **276 steps/hour** / **~15,979
  positions/hour** — a ~9.9× slowdown vs control, from the per-move solver CPU
  cost. v3 uses `solver_node_budget: 20000` (~2.5× faster per the
  `z2_solver_in_loop.yaml` node-budget comment, MEASURED 2026-07-01 on the
  registered-31/mined-94 trap population — no-win moves exhaust the budget at
  both depth 10 and 16, so budget not depth is the cost lever). Scaling v2's
  measured rate by 2.5×: **~690 steps/hour**, **~40,000 positions/hour**
  (ESTIMATE). 8000-step run ≈ **~11.6h**. Prefill to 25000 positions ≈
  25000/40000h ≈ **~38 minutes** (ESTIMATE — re-measure against the first hour
  of ARM-INJECT's own log and correct ARM-SEEDED's estimate before relying on it
  for scheduling).

---

## 3. VALIDITY GATE FIRST — read ARM-CONTROL before ANYTHING else

Pre-registered. If this gate fails, the regime is STILL broken (a v3-specific
confound) — fix it before reading ARM-INJECT or ARM-SEEDED at all; a candidate
lift measured against a broken control is exactly v2's mistake repeated.

```bash
# in-run LR check (§2) must read 1.0e-4 flat for the whole ARM-CONTROL run.

# HOST-MATCH first (§0.6 rule): once per host, the strip self-baseline —
# on vast expect ≈4/31 + 24/125 (paste actual counts into the run log; the §3
# bands below re-center on whatever this prints):
.venv/bin/python scripts/eval/run_l1_trapflip_smoke.py \
    --baseline-ckpt checkpoints/ws3v3_warmstart_200k.pt \
    --candidate-ckpt checkpoints/ws3v3_warmstart_200k.pt \
    --expect-encoding v6_live2_ls --legal-set \
    --out reports/d_ws3v3/hostbaseline_trapflip
.venv/bin/python scripts/eval/run_l1_trapflip_smoke.py \
    --baseline-ckpt checkpoints/ws3v3_warmstart_200k.pt \
    --candidate-ckpt checkpoints/ws3v3_warmstart_200k.pt \
    --expect-encoding v6_live2_ls --legal-set \
    --trap-set reports/d_tactical_2026-06-26/heldout_traps_all.jsonl \
    --out reports/d_ws3v3/hostbaseline_trapflip_combined

# trap-flip, registered-31 + combined-125 (A1 amendment: --expect-encoding, NEVER
# bare --encoding; baseline = the STRIP — the raw 200k's stale v6_live2 stamp
# correctly RAISES under the assertion; strip weights byte-identical, §0.6):
.venv/bin/python scripts/eval/run_l1_trapflip_smoke.py \
    --baseline-ckpt checkpoints/ws3v3_warmstart_200k.pt \
    --candidate-ckpt checkpoints/ws3v3_control/checkpoint_00208000.pt \
    --expect-encoding v6_live2_ls --legal-set \
    --out reports/d_ws3v3/control_trapflip
.venv/bin/python scripts/eval/run_l1_trapflip_smoke.py \
    --baseline-ckpt checkpoints/ws3v3_warmstart_200k.pt \
    --candidate-ckpt checkpoints/ws3v3_control/checkpoint_00208000.pt \
    --expect-encoding v6_live2_ls --legal-set \
    --trap-set reports/d_tactical_2026-06-26/heldout_traps_all.jsonl \
    --out reports/d_ws3v3/control_trapflip_combined

# threat-probe vs the ANCHOR baseline (NOT bootstrap — the v2 mistake).
# --positions is REQUIRED on _ls-stamped ckpts (§0.6: the implicit fixture
# fallback now hard-fails; the v6_live2 fixture is the valid shape-identical read):
.venv/bin/python scripts/probe_threat_logits.py \
    --checkpoint checkpoints/ws3v3_control/checkpoint_00208000.pt \
    --positions tests/fixtures/threat_probe_positions_v6_live2.npz \
    --baseline-json tests/fixtures/threat_probe_baseline_anchor200k.json
```

**PASS conditions (all three required, PRE-REGISTERED numerically — "flat" is
not a vibe check):**
- deploy-disagree (KILL gate) **≤ 0.16** — the same co-gate §4 uses for the ON
  arms; a solver-OFF arm regressing this much means something OTHER than the
  solver corrupted normal play (buffer/LR/scheduler drift).
- **registered-31 flip count within ±2 traps of the 4/31 baseline** (i.e. a
  raw flip COUNT in **[2, 6]**, not just "near 12.9%" — the baseline is 4/31 =
  12.9%, so ±2 traps is the operational noise band) **AND combined-125 flip
  count within ±5 of 24/125** (baseline 24/125 = 19.2%, so the operational band
  is **[19, 29]** flips). A flat-to-small-drop read is expected (a fresh
  fine-tune with no new signal should not IMPROVE flip); outside either band is
  a violation.
- threat-probe C1-C3, gated vs `tests/fixtures/threat_probe_baseline_anchor200k.json`,
  **all PASS**: C1 (contrast) **≥ 5.848**, C2 (top5) **≥ 25%**, C3 (top10)
  **≥ 40%** — the anchor's own values (contrast +7.3095, top5 30%, top10 40%)
  minus the probe's standard PASS margin; C3 sits EXACTLY at its 40% threshold
  at n=20, so report the raw top10 percentage alongside PASS/FAIL, not just the
  boolean.

**Any single violation above (deploy-disagree, EITHER flip-count band, OR any
of C1/C2/C3) means the regime is still broken** — fix before reading ANY
verdict; ARM-INJECT/ARM-SEEDED results are QUARANTINED (do not read them) until
a clean ARM-CONTROL re-run passes all conditions.

If ARM-CONTROL fails any of these, STOP — do not read ARM-INJECT/ARM-SEEDED.
Diagnose (LR drift? buffer path collision? scheduler reactivation?) and re-run
the control before spending more GPU-hours.

---

## 3.5 FIX2a — the honest native-provable ceiling (run BEFORE trusting ARM-SEEDED)

```bash
# SAMPLED local estimate (n=40, deterministic seed — what the laptop ran):
.venv/bin/python scripts/measure_native_provable_fraction.py \
    --traps reports/d_tactical_2026-06-26/heldout_traps_all.jsonl \
    --sample 40 --seed 20260702 \
    --out reports/d_ws3v3/native_provable_fraction_sample40.json
# FULL-125 exact number: §1(i) vast precondition (release build, ~1h).
```

Replays `post_move_seq` for each selected trap of the 125-trap combined
held-out corpus and asks `engine.TacticalSolver(window_half=None, cand_cap=40,
neighbor_dist=2)` to prove the POST position (defender/model to move) at depth
16, node_budget 20000 — the SAME solver config the v3 self-play arms run. A
proven LOSS for the defender at POST (`result == -1`) is a proven forced WIN
for the attacker; the reported fraction is over that frame. **MEASURED
2026-07-02 (laptop, debug build, CPU-bound, uniform n=40/125 sample,
seed=20260702, 3498s ≈ 87s/trap):**

> **0/40 proven — fraction 0.000, Wilson 95% CI [0.000, 0.088]**, n_skipped=0.
> ZERO proven in EVERY sampled mate_distance bucket (2t: 0/1, 3t: 0/14,
> 4t: 0/2, 5t: 0/10, 6t: 0/1, 7t: 0/11, 9t: 0/1) — not a deep-mate-only
> failure; the solver can't prove even the mate-in-2/3-turn POST positions of
> this class at depth 16 / budget 20000 / neighbor_dist 2. Artifacts:
> `reports/d_ws3v3/native_provable_fraction_sample40.json` + `_records.jsonl`.

**Implication (read BEFORE ARM-SEEDED):** the honest ceiling for solver
POLICY-injection on POST-blunder seeds is ≤8.8% (95% upper bound), point
estimate ~0. FIX2b's build-time gate will therefore emit few-to-zero `_kpost`
seeds (correct behavior — it refuses to seed positions the solver can't
convert), and `solver_fire_rate_seeded` is EXPECTED to read near the organic
rate: the §6 SEED-STARVED branch is the DEFAULT expectation for this corpus,
not an edge case. ARM-SEEDED's live mechanism is the z-LABEL densification
(§0.5 mechanism (i)); read its flip AGAINST ARM-INJECT for the z-label delta,
and treat a MEMORIZES read off ARM-SEEDED alone as invalid per the
SEED-STARVED row. Follow-up lever for the injection half stays native-solver
RECALL (D-TACTICAL), not more GPU. The §1(i) full-125 release-build run
replaces this sampled CI with the exact population number when the vast box
comes up.

---

## 4. HEADLINE GATES per ON-arm (ARM-INJECT, ARM-SEEDED)

```bash
# registered-31 (pre-reg headline) + combined-125 (powered corroboration) —
# run BOTH per arm, same pattern as §3 with each arm's own candidate ckpt:
# A1 amendment: baseline = the STRIP (raw 200k's stale stamp RAISES under the
# assertion; weights byte-identical), --expect-encoding NEVER bare --encoding:
for ARM in inject seeded; do
  .venv/bin/python scripts/eval/run_l1_trapflip_smoke.py \
      --baseline-ckpt checkpoints/ws3v3_warmstart_200k.pt \
      --candidate-ckpt checkpoints/ws3v3_${ARM}/checkpoint_00208000.pt \
      --expect-encoding v6_live2_ls --legal-set \
      --out reports/d_ws3v3/${ARM}_trapflip
  .venv/bin/python scripts/eval/run_l1_trapflip_smoke.py \
      --baseline-ckpt checkpoints/ws3v3_warmstart_200k.pt \
      --candidate-ckpt checkpoints/ws3v3_${ARM}/checkpoint_00208000.pt \
      --expect-encoding v6_live2_ls --legal-set \
      --trap-set reports/d_tactical_2026-06-26/heldout_traps_all.jsonl \
      --out reports/d_ws3v3/${ARM}_trapflip_combined
done
```

Report BOTH baselines (12.9% registered-31, 19.2% combined-125) alongside each
arm's candidate flip — the 125 set flips easier at baseline (population shift,
see `docs/handoffs/d_ws3_l1_smoke_runbook.md` §3), so read it baseline-relative,
not against the static 16%/25% bars directly (the evaluator's own `decide()`
already does this). **C3 anchor margin note:** the anchor sits EXACTLY at the 40%
C3 threshold at n=20 — report anchor-relative deltas (not just PASS/FAIL) so a
borderline regression is visible.

---

## 5. CO-GATES (all must clear for a GENERALIZES verdict, per ON-arm)

```bash
# (a) off-window forced rate must HOLD 0.0 (D-DECODE floor; absolute rate is
#     load-bearing, the exploit-control contrast is contaminated on centroid-
#     shifting bots — see memory exploit_probe arm-aliasing bug). --out is
#     REQUIRED (scripts/exploit_probe.py: `required=True`) — omitting it
#     fails argparse before anything runs.
.venv/bin/python scripts/exploit_probe.py --checkpoint checkpoints/ws3v3_${ARM}/checkpoint_00208000.pt \
    --expect-encoding v6_live2_ls --defender deploy --legal-set --adv-ref current \
    --out reports/d_ws3v3/${ARM}_exploit_deploy

# (b) ModelPlayer false-clear cross-check (arm-aliasing-immune):
.venv/bin/python scripts/exploit_probe.py --checkpoint checkpoints/ws3v3_${ARM}/checkpoint_00208000.pt \
    --expect-encoding v6_live2_ls --defender modelplayer --adv-ref current \
    --out reports/d_ws3v3/${ARM}_exploit_modelplayer

# (c) threat-probe C1-C3 vs the ANCHOR baseline (already run in §4-style loop).
#     --positions REQUIRED on _ls-stamped arm ckpts (§0.6):
.venv/bin/python scripts/probe_threat_logits.py --checkpoint checkpoints/ws3v3_${ARM}/checkpoint_00208000.pt \
    --positions tests/fixtures/threat_probe_positions_v6_live2.npz \
    --baseline-json tests/fixtures/threat_probe_baseline_anchor200k.json
```

v2 baseline for (a): off-window forced rate held **0.0** on BOTH the control and
candidate arm (deploy defender AND modelplayer defender, n=20/arm) — the floor
was never the risk in v2; re-confirm it holds under v3's regime too.

---

## 6. Verdict table

| verdict | condition | next |
|---|---|---|
| **GENERALIZES** | an ON-arm's held-out flip ≥25% AND control holds flat (§3) AND KILL ≤16% AND co-gates (§5) clear | fund the GPU-week: `z2_solver_in_loop_full.yaml` + seeding carried forward |
| **MEMORIZES** | ON-arm flip ≤16% held-out, AND (for ARM-SEEDED) `solver_fire_rate_seeded` confirmed HIGH (the density lever fired, it just didn't teach) | `D-BOOTSTRAP` fires — solver-in-loop from step 0 is the only remaining bet (separately gated, a real gamble) |
| **THIN-STILL** | fire-rate high (ARM-SEEDED's `solver_fire_rate_seeded` near target) but held-out flip UNMOVED vs ARM-INJECT | ONE escalation to `solver_visit_weight: 1.0` before declaring MEMORIZES — density alone wasn't the bottleneck, try a harder push |
| **KILL** | deploy-disagree > 0.16 on an ON-arm (control clean) | soften `solver_visit_weight` and re-smoke; the soft/one-hot boundary is the knob |
| **INDETERMINATE** | control itself drifts (§3 fails), or ambiguous mixed reads | fix the regime / re-run before concluding anything — do not repeat the v2 mistake of reading a candidate against a broken baseline |
| **SEED-STARVED** (pre-registered, FIX2c, checked BEFORE any of the rows above for ARM-SEEDED) | `solver_fire_rate_seeded` **< 2× the organic rate** (organic v2 measured 5.6%, so the SEED-STARVED threshold is **< 11.2%**) | the seeding lever did NOT densify solver POLICY-injection for this class — do NOT read MEMORIZES off ARM-SEEDED's flip alone (§0.5's dual mechanism means the z-label half of the lever may still show in the flip even when the solver half is starved); interpret ARM-SEEDED's flip AGAINST ARM-INJECT instead of standalone, and the follow-up lever is native-solver RECALL (D-TACTICAL), not more GPU |

Aggregation logic (read in this order): (1) §3 control validity gate — if it
fails, STOP, verdict is INDETERMINATE regardless of what the ON-arms show;
(2) **SEED-STARVED check for ARM-SEEDED** (`solver_fire_rate_seeded` vs the
11.2% floor, §3.5's measured native-provable fraction as the honest expected
ceiling for the injection half specifically) — if SEED-STARVED, do not read
ARM-SEEDED's flip standalone as MEMORIZES, route to the SEED-STARVED next-step
instead; (3) compare ARM-INJECT vs ARM-SEEDED flip — if SEEDED clears ≥25% and
INJECT doesn't, the density lever (seeding) is the one that mattered, note
this for the GPU-week's config; (4) apply the KILL/MEMORIZES/THIN-STILL/
GENERALIZES table above per arm.

---

## 7. Durability

`reports/` and `checkpoints*/` are gitignored — **rsync everything off the vast
box BEFORE stopping/destroying it**:

```bash
rsync -avz vast:/workspace/hexo_rl/reports/d_ws3v3/ reports/d_ws3v3/
rsync -avz vast:/workspace/hexo_rl/checkpoints/checkpoint_log.json reports/d_ws3v3/checkpoint_log.json
rsync -avz vast:/workspace/hexo_rl/logs/ws3v3_*.jsonl reports/d_ws3v3/logs/
# the seed corpus + its mining stats + the disjointness check output:
rsync -avz vast:/workspace/hexo_rl/reports/d_ws3v3/seed_corpus*.json* reports/d_ws3v3/
# the three per-arm checkpoint dirs (§2's --checkpoint-dir) — needed to re-run
# any eval offline after the box is gone:
rsync -avz vast:/workspace/hexo_rl/checkpoints/ws3v3_control/ checkpoints/ws3v3_control/
rsync -avz vast:/workspace/hexo_rl/checkpoints/ws3v3_inject/ checkpoints/ws3v3_inject/
rsync -avz vast:/workspace/hexo_rl/checkpoints/ws3v3_seeded/ checkpoints/ws3v3_seeded/
```

v2 artifacts are already pulled locally (`reports/d_zvalid_z2/`, `reports/vast_ws3_v2/`)
— do not re-pull, do not overwrite them with v3 output (different dir).

---

## 8. Billing

The box idles at ~$/h even between arm launches (SEQUENTIAL runs on a single GPU —
do not co-tenant). Stop/destroy the instance when pausing between §1 (mining, CPU-
only, could run on a cheaper CPU-only box if mining is split from training) and §2
(GPU launches), and again after §7's durability rsync completes — do not leave the
GPU box idling while writing up the verdict.
