# Phase B' v9 hex-native trunk — engineering handoff

**STATUS UPDATE 2026-05-05 evening — T4/T5/T6 COMPLETED.  v9 hex-trunk
turn is FALSIFIED.**  Full synthesis at
`reports/phase_b_prime/v9_smokes/synthesis.md`; sprint log §154 in
`docs/07_PHASE4_SPRINT_LOG.md`.  Headlines:

* Hex_kernel + corner_mask trained models win **0–1 game out of 200**
  vs SealBot (vs bootstraps' ~17%).
* Per-move rotation alone is null on intrinsic policy.
* Q2 jitter remains the only confirmed Class-4 lever.
* New Class-5 finding: under HexConv2d the eval gate promotes a
  colony-attractor at step 500.  v10 priority 1 = add
  `colony_wins_max_fraction` to gating before any future
  architecture-altering smoke.

The remainder of this document records the original engineering
handoff at commit-time and is preserved as the audit trail.

---

**(Original engineering handoff, 2026-05-05 morning):** T1, T2, T3 complete on
`phase_b_prime_v9_hex_native` branch (off `phase_b_prime_v8_plumbing`).
T4 / T5 / T6 require GPU + walltime and are operator-driven from here.

## What landed (commits in chronological order)

1. **T1 — HexConv2d + integration** (commit `1f23986`)
   - `hexo_rl/model/hex_conv.py:HexConv2d` — drop-in for `nn.Conv2d`,
     fixed binary mask zeroes the two long-diagonal positions
     (offsets `(-1,-1)` and `(+1,+1)`) of the 3×3 kernel.  Mask is
     enforced via `register_forward_pre_hook` and a parameter gradient
     hook so masked weights stay pinned at zero across optimiser steps.
   - `hex_mask` is `persistent=False` — checkpoints are
     architecture-agnostic; the trunk type is selected by
     `model.use_hex_kernel` in YAML, not by the saved buffer.
   - `HexTacToeNet`, `Trunk`, `ResidualBlock` accept
     `use_hex_kernel: bool = False`. When True, every `Conv2d` in the
     trunk (input + every `ResidualBlock.{conv1, conv2}`) is a
     `HexConv2d`. Knob threaded through trainer hparam resolver,
     `training/loop.py`, `bootstrap/pretrain.py`,
     `bootstrap/bots/our_model_bot.py`.
   - 11 unit tests in `tests/test_hex_conv.py`.

2. **T2 — corner-mask engine knob** (commit `3fd7ebd`)
   - `engine/src/board/state.rs` — global `CORNER_MASK_ENABLED` atomic
     gating both `encode_state_to_buffer` and
     `encode_state_to_buffer_channels`.  When on, stone planes 0 and 8
     are zeroed wherever `hex_dist > HALF` from window centre (90 of
     361 cells masked; 271 survive).  Mask is a `OnceLock`'d LUT.
   - `engine.set_corner_mask_enabled(bool)` and
     `engine.corner_mask_enabled()` exposed to Python.
   - `WorkerPool.__init__` flips the flag once at startup based on
     `model.corner_mask` in YAML; `bootstrap/pretrain.py` does the
     same so post-train validation tensors match the model's expected
     encoding.
   - 1 Rust + 4 Python tests.

3. **T3 — per-move / per-turn rotation cadence** (this commit)
   - `SelfPlayRunner` accepts `rotation_cadence: str ∈ {per_game,
     per_move, per_turn}` (default `per_game` — matches §130).
   - `engine/src/game_runner/records.rs:should_resample_sym` is the
     pure helper governing within-game `sym_idx` resampling; covered
     by 4 unit tests including the unknown-code fallback.
   - `worker_loop.rs` calls the helper before every move and stashes
     the per-move `sym_idx` into each row of `records_vec` so the
     game-end aux reprojection scatters into the same rotated frame
     this row's state was rotated to.
   - **Aux/state-frame consistency** is the load-bearing invariant:
     before T3 the game-level `sym_idx` was used to rotate aux at game
     end; T3 makes this a per-row value so per-move and per-turn
     cadences keep `(state, chain, policy, ownership, winning_line)`
     aligned in the same rotated frame.
   - Existing per-game rotation (`§130`) preserved bit-for-bit when
     cadence is unset (default).
   - 4 Python end-to-end tests in `tests/test_rotation_cadence.py`
     (per_move + per_turn drive `WorkerPool`, drain rows, verify
     aux/state agree on no-stone cells > 95%).

## Verification

- `cargo test --workspace`: **146 / 146 pass**
  (engine crate; +4 cadence tests + 1 corner-mask test vs §152 baseline)
- `make test.py`: **969 passed, 8 skipped, 5 xfailed, 1 xpassed** in
  ≈3 min, 0 failures (+15 new tests across hex_conv, corner_mask,
  rotation_cadence)

## Variant configs (ready, not yet launched)

The §153 T5 smoke matrix is wired up as four 5080 variants:

| Variant | Bootstrap | hex_kernel | corner_mask | Q2 jitter | rotation_cadence |
|---|---|:---:|:---:|:---:|---|
| `v9_s1_hex_only_5080` | `v8full` | true | true | false | `per_game` |
| `v9_s2_hex_q2_5080` | `v8full` | true | true | true | `per_game` |
| `v9_s3_per_move_no_hex_5080` | `v7full` | false | false | false | `per_move` |
| `v9_s4_full_combined_5080` | `v8full` | true | true | true | `per_move` |

Each is 2500 iterations, `eval_interval=500`, instrumentation on,
matches §152 + v8 smokes for direct comparability.

## T4 — bootstrap variant generation (operator-driven)

Two bootstraps to produce, both saved as `bootstrap_model_v8full.pt`
(only one — pick the better of the two on probe gates):

  **A. v7full warm-start with hex_kernel + corner_mask, 30 epochs.**
  - Override `model.use_hex_kernel: true`, `model.corner_mask: true`
    in `configs/model.yaml` (or a `pretrain_v8full.yaml` overlay).
  - `make pretrain` with `--resume checkpoints/bootstrap_model_v7full.pt
    --epochs 30 --eta-min 5e-5 --inference-out
    checkpoints/bootstrap_model_v8full_warm.pt`.
  - Note: corpus on disk encodes WITHOUT corner_mask (since v7full
    corpus pre-dates this knob). The post-train `validate()` loop
    flips the engine flag on, so validation tensors match the model
    expectation; the training data itself is not regenerated.

  **B. From-scratch hex_kernel + corner_mask, 30 epochs at the §150
  recipe.**
  - Same model overrides, no `--resume`.
  - `--inference-out checkpoints/bootstrap_model_v8full_scratch.pt`.

  **Pick:** evaluate both against SealBot at n=200 + threat probe
  C1/C2/C3. Promote the better as the canonical
  `bootstrap_model_v8full.pt`.  v7full canonical is **not** to be
  overwritten — that filename remains pinned to §150.

  **Abort criterion:** if the v7full warm-start drops > 5 pp on
  SealBot vs v7full, surface the regression — that implies the hex
  kernel introduces a learning pathology and T5 / T6 should not
  proceed without further investigation. Do NOT fall through to the
  scratch version silently.

## T5 — smoke matrix execution (operator-driven)

Run all four variants (above) on 5080 vast.ai. Parallel where the
host allows; otherwise serial.

```
python scripts/train.py --checkpoint checkpoints/bootstrap_model_v8full.pt \
  --variant v9_s1_hex_only_5080 \
  --checkpoint-dir checkpoints/v9_s1_hex_only --no-dashboard --iterations 2500
# …repeat for s2, s3, s4. s3 uses --checkpoint
# checkpoints/bootstrap_model_v7full.pt because it isolates per-move
# rotation on the v7full trunk.
```

Each smoke produces `events.jsonl`, `run.log`, `q*.jsonl` reports
under `reports/phase_b_prime/v9_smokes/{variant}/` (mirror the v8
layout in `reports/phase_b_prime/v8_smokes/`).

Gather: stride5_run_max P50/P90, row_max_density P90, draw_rate,
SealBot WR @ step 2000, Elo trajectory.

## T6 — synthesis report

Author `reports/phase_b_prime/v9_smokes/synthesis.md` mirroring the
v8 synthesis layout. Decompose Class-4 dampening contribution among
hex kernel, Q2 jitter, per-move rotation. Update
`/tmp/phase_b_prime_targets.md` to v10. List viewer / dashboard
items needing updates (per-move rotation rendering, per-row sym_idx
in game records — currently NOT exposed back to Python by design)
as a follow-up prompt.

## Constraints honoured

- `LEGAL_MOVE_RADIUS = 5` and `CLUSTER_THRESHOLD = 5` unchanged on
  master.
- `bootstrap_model_v7full.pt` not regenerated.
- Viewer / dashboard JS untouched.
- One commit per logical change (T1, T2, T3 are three separate
  commits on a feature branch).
- All knobs default OFF (use_hex_kernel=false, corner_mask=false,
  rotation_cadence=per_game) — pre-§153 callers see no behaviour
  change.

## Out of scope this prompt

- Bootstrap training, smoke launches, sustained-run promotion.
- Viewer rendering of per-move rotation IDs (per-row sym_idx is
  stored in the worker_loop's `records_vec` tuple but NOT round-
  tripped through the replay buffer's drain path or the recent_game
  ring-buffer; T6 should call out viewer changes that would be
  needed to surface per-move rotations on the game-replay panel).
- Corpus regeneration with corner_mask=True (v7full corpus is
  pre-corner-mask; T4 warm-start documents the workaround).
