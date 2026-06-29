# D-ZVALID Z0 — native tactical body: status + perf-layer handoff

**Date:** 2026-06-29. **Branch:** `phase4.5/d-solver`. **Status:** sound quiet-move
MECHANISM landed + tested (13/13 `engine::tactics` tests green, debug). The
soundness-neutral PERF layer (scored α-β + 729-eval + aged-TT) is the documented
remainder — the genuinely multi-day part the repo's own design (`reports/d_tactical_2026-06-26/NATIVE_RUST_SOLVER_design.md`)
estimates at 4–7 days. **Nothing committed** (commits on ask).

Grounds on: the R3 coupling invariant (`docs/handoffs/d_reconfirm_status.md`),
the A2 ceiling finding (8% threat-only, 80% of mates start quiet), and the user's
two locked decisions: **full perf body** (Z0 scope) + **route the deploy-backup
through the native probe** (Z1, no interim SealBot spread-guard) + the confirmed
**reduced net-sharpened candidate set for search** (full-legal only to certify a
not-in-check LOSS).

---

## 1. LANDED THIS SESSION (tested, sound, uncommitted)

All in `engine/src/tactics/` + the `engine/src/pyo3/tactics.rs` signature.

### Z0-A — R3 LOSS-completeness guard (the HARD INVARIANT prereq)
`search.rs` `solve()` LOSS conclusion: emit `Outcome::Loss` only when
```
(in_check && moves_len < cand_cap) || moves_len >= legal_move_count()
```
holds (`in_check = count_winning_moves(opp) >= 1`); otherwise the candidate set is
incomplete and a LOSS would be unsound → `UNKNOWN` (conservative-safe). **LOSS is
now sound for ANY candidate set** — the body builds on a proven-safe gate. Tests:
`r3_guard_suppresses_truncated_false_loss` (RED→GREEN: a position that is truly a
P1 WIN but whose winning counter `cand_cap=1` truncates away → unguarded emits a
false LOSS; guarded → UNKNOWN). `soundness_fuzz_zero_false_loss` extended with a
NOT-IN-CHECK stream (39 constructed open-four doubles) + a `nic_checked > 0`
coverage assertion so the not-in-check surface is provably exercised.

### Z0-B — quiet-move candidate widening (`ordering.rs::candidates`)
New `TacticalConfig.neighbor_dist: Option<i32>` (default `None` = threat-only, the
fast deploy/A1 mode). `Some(d)` appends, at NOT-IN-CHECK nodes only, every empty
legal cell within cheb-distance `d` of a stone (sorted, after threats). In-check
nodes are NOT widened (threat-only is already complete there — a quiet move loses
to the standing threat). When `d` covers the legal radius the set becomes the full
legal set → the R3 guard's `moves_len >= legal_move_count()` branch fires and a
not-in-check LOSS is proven soundly. Tests:
`neighbor_dist_widens_not_in_check_candidates_to_full_legal`,
`neighbor_dist_does_not_widen_in_check_nodes`, `widened_solver_stays_sound`
(brute-confirmed, no false proofs). PyO3 `TacticalSolver(window_half, cand_cap,
neighbor_dist=None)` updated.

### Z0-C(correctness) — recall-preserving not-in-check LOSS verify
`search.rs` `solve()`: when the reduced set drove the search, all candidates lost,
and the cheap completeness branches did NOT certify, and `neighbor_dist.is_some()`
→ search the DROPPED legal moves (`legal_moves() \ searched`). Any WIN among them is
an escape the reduced set missed (return WIN with its line); any UNKNOWN leaves it
unresolved; only when EVERY dropped move also loses is the LOSS certified. This is
the **reduced-set-for-speed / full-legal-for-LOSS-certification** design the user
confirmed: the narrow set gives early WIN cutoffs, the verify gives sound LOSS
recall without ever trusting an incomplete set. Tests: `verify_recovers_truncated_win`
(verify finds the dropped winning counter → WIN where the bare guard said UNKNOWN),
`verify_certifies_truncated_loss` (dropped moves all lose → certified LOSS,
brute-confirmed), `verify_path_is_sound_over_grid` (cand_cap=1 forces every node
through the verify; no false WIN/LOSS across the compact grid, brute-checked).

**Soundness invariant preserved throughout:** proofs come ONLY from terminal CF-1
backup + the stone-count shortcuts; the value head is never read; the verify is a
pure search over the engine's own move generation. The `#[cfg(test)]` brute oracle
(`brute_solve`, full legal, no prune) cross-checks every LOSS claim.

### Bench note
The changes touch only `engine/src/tactics/**` + `engine/src/pyo3/tactics.rs` —
NONE of the bench-gated hot paths (`mcts/**`, `replay_buffer/**`, `game_runner/**`,
`inference_bridge.rs`). The solver is not yet called by MCTS, so there is no
regression surface. **`make bench` becomes mandatory when the deploy-root hook is
wired** (§3 below) — that call lands on the MCTS move path.

---

## 2. REMAINING — the perf layer (soundness-neutral, multi-day)

The mechanism above is CORRECT but, without α-β pruning + a static eval, the
quiet-move search is budget-bound: the not-in-check LOSS verify is a full-width
expansion, so deep traps exhaust the node budget → `UNKNOWN`. The perf layer makes
the same sound search reach deploy-root-d6 / training-d8 within budget. It does NOT
change any proof (every item below is ordering / pruning / leaf-heuristic only —
heuristic leaves report `UNKNOWN`, never a proof). Reference:
`reports/d_tactical_2026-06-26/NATIVE_RUST_SOLVER_design.md` §3–4.

Build order (each a TDD increment; bench-gate only once the deploy hook lands):

1. **Scored α-β core** (`search.rs`). Convert the 3-valued proof to a scored search
   where mate = `±(MATE_SCORE - ply)` and heuristic leaves = `eval.rs` score
   (bounded `|·| < WIN_THRESHOLD`). PROOF derived: `|score| >= WIN_THRESHOLD` ⟺
   proven; the R3 guard still gates a `<= -WIN_THRESHOLD` score at a not-in-check /
   truncated node (treat as heuristic, not a proven LOSS, unless verify-certified).
   Keep the existing 3-valued tests green (the scored version must reproduce every
   sound WIN/LOSS). α-β bounds are what make the not-in-check verify affordable.
2. **729 ternary static eval** (`eval.rs`, currently a `None` stub). Port SealBot's
   `3^6` 6-cell pattern table (`vendor/bots/sealbot/best/engine/board.h:185
   _move_delta`, `:63 _eval_score`) + an incremental accumulator maintained on
   make/undo. ORDERING tie-break + non-proof leaf scores ONLY. Soundness invariant:
   never returns a WIN/LOSS proof.
3. **2-slot aged TT** (`tt.rs`, currently a plain `FxHashMap` LOSS cache). Replace
   with the generation-aged 2-slot bucket (depth-preferred + always-replace), int16
   mate-distance-encoded score, EXACT/LOWER/UPPER flag, and a best-move field for
   ordering. Keep the "only game-theoretic proofs are trusted" property.
4. **PVS / LMR / aspiration + killers / history** (`search.rs` + `ordering.rs`).
   Standard. Soundness-neutral (ordering / re-search only).
5. **Net-policy candidate ordering** (`ordering.rs` step 4). Pass the deploy
   `improved_policy` as a `PolicyPrior`; order the REDUCED neighbor set by prior
   descending. This is the AlphaZero lever SealBot lacks (sharpen, NOT widen — the
   redteam "cheap broad search 0/38" warning). The threat enumeration still runs
   regardless of prior so a 0-prior refuter (67% per T0) is never skipped.

**The reduced set is the search frontier; full-legal is only the LOSS-certification
verify.** Do NOT widen the search frontier to full-legal (that re-introduces the
node explosion the redteam flagged). `neighbor_dist=Some(2)`, `cand_cap≈15` interior
/ `≈20` root is the SealBot-grade frontier; the verify already handles soundness.

---

## 3. DOWNSTREAM HOOKS (gated on §2)

- **Z1 deploy-root hook** — route `SolverBackupBot.solver_probe` through the native
  `PyTacticalSolver` (binding exists; Z1 fork wired the Python). WIN-only override =
  sound by construction, immune to SealBot spread-false-proofs. Bench-gate the MCTS
  move path when this lands in Rust (`mcts/mod.rs:267` root call).
- **Z2 self-play z-correction hook** — `game_runner/worker_loop/inner.rs`
  `finalize_game` + move-select loop: run the solver per-move at a training budget;
  realize the forcing WIN line (always sound) and/or stamp corrected `z=-1` on a
  verify-certified LOSS row (now sound — R3 guard landed). The Z2 discriminator
  (`docs/handoffs/d_zvalid_z2_training_z_discriminator.md`) is the GPU-week gate; it
  cannot RUN until this hook + §2 land. Success metric = STANDALONE net (backup OFF)
  vs the eval ladder (SealBot fixed-depth + self-play Elo; **no KrakenBot** — weights
  unavailable), NOT backup-vs-SealBot.

---

## 3.5 REVIEW + RED-TEAM verdict + fixes applied

Fresh review: **Z0 Rust = SOUND** ("the LOSS-proof path cannot emit a false
WIN/LOSS"); Z2 = CONCERNS (all fixed). Distinct adversarial red-team (incl. an
executable soundness attack): **native proof soundness HELD — could not construct
a false native WIN**; SealBot's `[140][140]+70` OOB phantom-mate mechanism is
structurally ABSENT (the native solver is HashMap/run-length based), so "immune to
spread-false-proofs by construction" verified. Fixes landed from the findings:

- **Off-window COMPLETING-stone guard** (MEDIUM, §D-COHERENCE). The override places
  BOTH turn stones (`line[0]` + the cached completing `line[1]`); the guard vetted
  only `line[0]`. Now vets both (`mod.rs::prove_in_place` `line.iter().take(2)`;
  Python `solver_backup_bot.py` vets `s2`). Test `off_window_completing_stone_suppressed`
  (a P2 block forces an in-window-setup / off-window-completion win — the case
  net-policy ordering will make common; the current lex order normally surfaces the
  outermost stone as `line[0]`, so today `line[0]` already catches it).
- **Spread/multi-cluster immunity now MEASURED** (LOW). `spread_multicluster_no_false_proof`
  cross-checks the solver vs the brute oracle at `|coord| > 63` (the SealBot OOB
  boundary) and across disjoint clusters — no false proof.
- **Z2 measurement-validity (HIGH) + falsifiability (MEDIUM)** — see the Z2 handoff:
  `INDETERMINATE_STARVED_RECALL` verdict (gated on a measured `z_loss_coverage`
  floor), a `teach_wr_rise` significance gate on the WR arm, graceful trap-loss
  degrade, and a fail-closed game-disjointness assertion.

**The binding constraint (HIGH, load-bearing for the GPU-week gate):** without the
§2 perf layer, the not-in-check quiet-trap LOSS routes through the budget-bound
full-legal verify → exhausts → UNKNOWN → z uncorrected → **z-LOSS RECALL on the
~80%-quiet-mate TARGET class is near-zero**. RECALL (not soundness) is the binding
constraint. The Z2 discriminator now refuses to return DOESN'T-TEACH unless measured
z-coverage met its floor (else `INDETERMINATE_STARVED_RECALL`), so a recall-starved
fine-tune cannot masquerade as "the lever doesn't teach". **Consequence: the §2 perf
layer is NOT optional for a valid Z2 verdict** — the discriminator must run with
enough solver reach (perf layer) to certify the quiet-trap LOSS class within budget.

**Documented LOW test gap:** no not-in-check-ROOT LOSS-certification fixture. Such
positions are deep (no zugzwang in 6-in-a-row) and the full-width verify exhausts the
debug node budget → UNKNOWN, so a cheap fixture does not exist. The verify's
LOSS-certification CODE is exercised by `verify_certifies_truncated_loss` /
`verify_path_is_sound_over_grid` (cand_cap=1 forces the verify); the not-in-check
case differs only in the `in_check` flag, which does not change the verify logic.
Close this when the perf layer makes a deeper fixture affordable.

---

## 4. TEST INVENTORY (`engine::tactics`, all green debug)

`test1_immediate_win_is_win`, `test2_quiet_position_not_loss`,
`test4_fork_is_proven_loss`, `test5_compact_loss_brute_confirmed`,
`soundness_fuzz_zero_false_loss` (now incl. not-in-check stream),
`in_window_guard_suppresses_offwindow_win`,
`r3_guard_suppresses_truncated_false_loss`,
`neighbor_dist_widens_not_in_check_candidates_to_full_legal`,
`neighbor_dist_does_not_widen_in_check_nodes`, `widened_solver_stays_sound`,
`verify_recovers_truncated_win`, `verify_certifies_truncated_loss`,
`verify_path_is_sound_over_grid`, `off_window_completing_stone_suppressed`
(red-team §D-COHERENCE fix), `spread_multicluster_no_false_proof` (red-team
immunity, measured), `redteam_flip_sign_two_stone_win_realized`,
`redteam_budget_exhaustion_no_false_proof`. Default run:
`cd engine && cargo test --lib tactics:: -j4` (17 tests, ~13 s debug,
dev-laptop thermal-safe, no release/LTO).

**On-demand exhaustive soundness sweeps** (`#[ignore]`, the not-in-check verify is
full-width so they run for minutes): `redteam_verify_grid_no_false_proof` (the
sweep that actually certifies a not-in-check ROOT LOSS via the verify — closes the
review's test gap) and `redteam_verify_random_compact_no_false_proof`. Run before
promotion / on the perf box: `cargo test --lib tactics:: -- --ignored`.

**Verified on the vast RTX 5080 (2026-06-29): 19/19 green including both exhaustive
sweeps — 0 false proofs** (isolated git worktree, branch `phase4.5/d-solver`). The
dev laptop thermal-throttles under the parallel test load, so the full suite incl.
sweeps is a vast/perf-box run; the laptop runs the fast 17 cleanly.
