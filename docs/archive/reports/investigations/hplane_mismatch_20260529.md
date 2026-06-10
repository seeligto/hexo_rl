# Investigation — H-PLANE-MISMATCH: pretrain↔selfplay history-plane shift (§P5-CT PROMPT 2)

**Date:** 2026-05-29. **Branch:** master (a352845 + uncommitted probe/ledger).
**Skill:** investigation-probe-smoke-verdict. **Scope:** code forensics + a
no-GPU activation dump. Companion deliverable: the encoding-width hardcode
ledger (`audit/structural/encoding_width_hardcode_ledger.md`).

Settle whether the v6-family wire format carries a **history-plane
distribution shift between pretrain and self-play** (planes 1-3 = my-stones
t-1/t-2/t-3, planes 9-11 = opp-stones t-1/t-2/t-3), and if so whether it is a
regression-class contributor worth fixing before Phase 6.

---

## 1. Fixture / claim audit (read from source)

- **Rust self-play encoder** — `engine/src/board/state/encode.rs`.
  `encode_state_to_buffer` (L32-58) writes ONLY planes 0 (my t0), 8 (opp t0),
  16 (moves_remaining==2 bcast), 17 (ply%2); planes 1-7 / 9-15 left untouched
  (caller zero-inits). `encode_state_to_buffer_channels` (L82-132) explicitly
  sets every non-{0,8,16,17} channel `<18` to `0.0` (L117-122, comment
  "History planes 1..7 / 9..15 are zero on the Rust self-play path"). Pinning
  test `channel_select_history_planes_are_zero` (L442).
- **Python corpus/pretrain encoder** — `hexo_rl/env/game_state.py:201`
  `to_tensor`. Planes 1-7 = my history t-1…t-7, planes 9-15 = opp history,
  populated from the `move_history` deque (L246-263). `replay_game_to_triples`
  (`hexo_rl/bootstrap/dataset.py:41-68`) accumulates the deque move-by-move
  (`apply_move` appends prior states, game_state.py:197-199), so the exported
  corpus carries live history.
- **Kept-plane slice.** v6 = `[0,1,2,3,8,9,10,11]`; v6tp =
  `[0,1,2,3,8,9,10,11,16,17]` (registry.toml). History planes 1,2,3,9,10,11
  ARE in the kept (wire) set for both — so the shift is on *kept* planes.

**Correction to the brief.** The brief cited `encode.rs:118-122` as zeroing and
implied an "un-zero" fix on planes 16/17. Planes 16/17 (turn-phase) are emitted
**live on BOTH** paths — the zeroing is **only** the history planes 1-3/9-11.
(Consistent with the prior CF-2 report correction, `compound_turn_cf1_cf2_…:31`.)

---

## 2. Two-arm hypothesis

- **E1 (CONFIRMED mismatch):** self-play mean-abs on planes 1-3/9-11 ≪ pretrain
  mean-abs → real distribution shift across the two phases.
- **E2 (NO mismatch):** history populated on both paths → 8/10-plane choice
  vindicated, close H-PLANE-MISMATCH.

---

## 3. Pre-registered verdict (from the brief, verbatim intent)

| Verdict | Criterion | Action |
|---|---|---|
| CONFIRMED | selfplay mean-abs(1-3/9-11) ≪ pretrain | register `v6_live2`, gate an **MCTS-matched** smoke |
| NO mismatch | history populated both paths | close H-PLANE-MISMATCH |
| INCONCLUSIVE | — | specify the single MCTS-matched run that resolves it |

---

## 4. Measurement (both arms, no GPU) — `scripts/structural_diagnosis/hplane_activation_dump.py`

**Part A — real pretrain corpus** (5000 rows sampled evenly across the file).

`data/bootstrap_corpus_v6.npz` (353091, 8, 19, 19):

| slot | plane | mean_abs | frac rows nonzero |
|---|---|---|---|
| 0 | my t0 | 0.0428 | 1.000 |
| 1,2,3 | my t-1/-2/-3 (HIST) | 0.0417 / 0.0406 / 0.0395 | 0.978 / 0.961 / 0.945 |
| 4 | opp t0 | 0.0440 | 1.000 |
| 5,6,7 | opp t-1/-2/-3 (HIST) | 0.0430 / 0.0419 / 0.0407 | 0.998 / 0.977 / 0.960 |

`data/bootstrap_corpus_v6tp.npz` (392251, 10, 19, 19): identical history profile;
turn-phase slots 8,9 mean_abs 0.4846 (nonzero 48.5%).

→ Corpus history planes carry **essentially the same activation mass as the
live t0 stone planes** (~0.04), present in 94-99.8% of rows.

**Part B — matched sample** (one 15-position compound-turn game through BOTH
encoders): Python corpus history planes nonzero (summed 0.0434); **Rust
self-play history planes EXACTLY 0.0**; t0 (0/8) and scalar (16/17) planes
agree between paths.

**Independent review (fresh-context agent):** re-dumped the Rust path with its
own code — history planes 0.000000, stones nonzero; cross-read encode.rs.
**PASS.** Also spot-checked 2 P0 + 1 P1 ledger entries — all accurate.

---

## 5. Verdict — **CONFIRMED**

The mismatch is real and large: **6 of 8 (v6) / 6 of 10 (v6tp) wire planes carry
~full activation mass in pretrain and exactly zero in self-play.** A model
pretrained to use history-plane features meets all-zero history planes the moment
it enters the RL self-play distribution.

---

## 6. Critical questioning — what this is NOT

- **The mismatch is INVARIANT across all v6-family runs** (baked into the wire
  format + the Rust split-responsibility rule). The §150 v7full anchor (17.4%,
  **no** colony collapse) had this exact mismatch; §175/§S178/§S181 (colony
  collapse) had the **same** mismatch. → It **cannot be the differentiator** that
  triggers the colony attractor. It is a **constant baseline handicap**, not the
  colony cause. The attractor is already suppressed by CF-1/CF-2 (§9 verdict).
- So H-PLANE-MISMATCH is a **regression-class encoding inconsistency**
  (pretrain→self-play transfer degradation), a *potential baseline lift* — NOT a
  colony fix. Scope it **against / before** Phase 6 league spend, not as a colony
  remedy.
- **Probe-gate limit (L2).** The "2 planes enough" prior is argmax-probe-level.
  Probe entropy cannot validate dynamic equivariance — any plane-drop
  recommendation needs an **MCTS-matched** eval, not a probe.

---

## 7. Recommendation — `v6_live2` (gated, MCTS-matched)

- New registry entry `v6_live2`: `kept_plane_indices = [0, 8, 16, 17]` (4 planes:
  my t0, opp t0, moves_remaining bcast, ply parity). Drops exactly the 6
  mismatched history planes; keeps the live turn-phase planes (CF-2). **Both
  paths then see the same 4 live planes — the shift is eliminated.** Fresh
  4-plane pretrain (not a v6/v7full transfer).
- **Populated-history alternative is INFEASIBLE** without Rust tensor history
  (violates split-responsibility) → not pursued (per brief).
- **Gate:** fresh `v6_live2` pretrain + an MCTS-matched smoke vs v6tp on the same
  harness. `v6_live2 ≥ v6tp` → adopt as the clean encoding (the history shift was
  a handicap). `v6_live2 < v6tp` → the pretrain head genuinely benefits from
  history context despite self-play zeroing it → keep v6tp, close H-PLANE-MISMATCH
  as "known asymmetry, net-neutral." This is a regression-fix candidate that
  competes with / precedes Phase 6.

---

## 8. Hardcode ledger (deliverable b — `audit/structural/encoding_width_hardcode_ledger.md`)

**P0 = 2, P1 = 6** (ledger ONLY — not fixed, to keep a clean bisect).
- **P0-1** `hexo_rl/training/orchestrator.py:286` — new-run model build
  `in_channels` falls back to literal **18** instead of the resolved spec
  `n_planes`; a variant YAML omitting `in_channels` builds an 18-channel trunk vs
  the 8/10-plane wire → first-forward crash.
- **P0-2** `scripts/generate_bot_corpus.py` — §S178 bot-corpus generator hardcoded
  to v6 end-to-end (no `--encoding`); emits 8-plane corpus regardless of run
  encoding (loud abort at the `batch_assembly` plane-count guard, not silent).
- P1×6 are diagnostic/probe/fixture surfaces that go dark or crash the probe (not
  run corruption): `early_game_probe`, `value_probe`/`build_value_probe_fixture`,
  `windowing_diagnostic`, `analyze_api`, `v6_argmax_bot`. Every reactively-fixed
  LIVE path (inference/eval/corpus-consume) verified clean.

---

## 9. Next action

1. Register `v6_live2` + MCTS-matched smoke vs v6tp (gated; input to PROMPT 3 —
   fix this regression-class finding **before** committing to a league).
2. Hardcode-ledger fix wave, P0s first (separate change with its own bench/INV);
   add v6tp to the test matrix as the non-8-plane regression canary.

**Artifacts:** `scripts/structural_diagnosis/hplane_activation_dump.py`,
`audit/structural/encoding_width_hardcode_ledger.md`.
