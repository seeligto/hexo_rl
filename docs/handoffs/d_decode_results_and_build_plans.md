# D-DECODE — off-window DEFENSE: results + build plans

**Date:** 2026-06-27/28. **Branch:** `phase4.5/d-solver`. **Ckpt:** `checkpoints/checkpoint_00272357.pt`,
encoding `v6_live2_ls`. **Scope:** eval-only, no retrain, no live run touched (d1m/longrun on vast).
**Method:** 3 multi-agent workflows (reframe → gate-close → firm+plan), each with fresh review + adversarial red-team.

---

## Verdict

**Off-window defense is a cheap, no-retrain DECODING fix.** The single-window g=0 Gumbel-SH deploy head
cannot *represent* off-window saving moves (`board.to_flat()` = `usize::MAX` → no policy logit); the
**multi-window no-drop legal-set** action space the net already trains under fixes it. The net was never the
problem — the deploy action-space restriction was.

→ **The native tactical solver (Track 3) scopes to IN-WINDOW-OFFENSE-ONLY** — the hard off-window-capability
requirement is dropped, owned by the Track-2 decoding fix.

### How the framing changed
The probe was dispatched to test a pre-registered **CANDIDATE-DROP** hypothesis (saving move excluded from the
Gumbel m=16 root set by *low prior*) with a planned fix of *threat-aware root candidate injection*. That was
**FALSIFIED**:

| Regime | Share (n=20) | Mechanism |
|---|---|---|
| **Geometric exclusion** | 60% | saving move is *itself off-window* → no logit slot → unrepresentable (not low prior). Off-window block cell in single-window top-16 at rate **0.0**. |
| Already agrees | 25% | in-window save, deploy plays it |
| Value-blind selection | 15% | in-window save *is* a candidate but g=0 SH selects it out |

"Inject into the candidate set" is impossible when there is no slot. The real lever is the **action space**, not
the candidate selector.

---

## Evidence (firmed)

| Claim | Result | Strength |
|---|---|---|
| **Position-level block asymmetry** (firmed, n=**50** distinct, 6 axes, openings 4–9, disjoint seeds) | single-window blocks **0/50 at *both* 150 and 450 sims** (never plays off-window — structural, no logit); multi-window Gumbel-SH blocks **12/50** | **airtight** — carries the gate; effective-n=6 flag killed |
| Pre-check | saving move enters multi-window top-16 at **1.0** (vs 0.0 single-window); **no M16-redrop** | strong |
| Selfplay action space (G0) | **MULTI-WINDOW** — net *was* trained to search/value off-window (legal-set overflow priors → Gumbel candidates → improved-policy targets). Multi-window deploy **aligns to training**, doesn't surface untrained moves | strong, code-path verified |
| Full-game defense | multi-window off_window_forced_win_rate **0.0** (combined 0/260) vs single-window **0.335** | see caveat |
| WR vs SealBot-d5 | **0.50** (n=100 distinct) vs 0.47 baseline → **no in-window regression** | solid (neutral, not a proven gain) |
| Bot faithfulness (review) | true g=0 Gumbel-SH over multi-window, no PUCT/temp leak | confirmed |

### Red-team caveats (kept, not buried)
- The **game-level** "0/200 forced" was effective-n=**6** (deterministic g=0 × 3-axis × 2-side) **and** is
  prevention that matched-compute single-window@450 *also* achieves. So the action-space proof is the
  **position-level** 0/50-vs-12/50 (structural: no logit → no block regardless of compute), **not** the game probe.
- The offline reference bot uses the *weaker* Python KCluster engine (no virtual-loss/quiescence/dynamic-FPU) yet
  held WR → the action space isn't costing strength; a Rust port retains the engine behaviors and will be ≥ this.
- MW single-move block rate is 0.24 on diverse forcing positions (vs 0.6 on the earlier already-won set) — the
  gate rests on SW=0-structural + MW>0, both confirmed; MW's full-game defense is 0.0 (prevention + block).

### Bug fixed (committed with this work)
`exploit_probe` **arm-aliasing**: `OffWindowAdversaryBot._is_off()` classified against the *current* board
centroid (post model-move) while the probe used the `model_last_snapshot` centroid — a centroid-shifting defender
flipped off→in-window, collapsing exploit==control. Fix: `set_reference_board(model_last_snapshot)` threaded from
`play_game`; `_is_off` uses it (falls back to current board when unset). TDD: `tests/test_exploit_probe_arm_aliasing.py`.
The absolute forced-rate metric (used for all defense verdicts) was contrast-independent and unaffected.

Artifacts: `reports/d_decode/`, `scripts/d_decode/` (notably `multiwindow_gumbel_sh_bot.py` = the validated fix
reference, `firm_block_positions.py` = the effective-n=50 firming).

---

## Track 2 — Rust multi-window deploy port (READY-TO-BUILD)

**Productionize the fix:** make the deploy Gumbel-SH head use the multi-window legal-set action space.
**Verdict: SMALL-to-MEDIUM WIRING, ~1–2 days, ~90% reuse** — the Rust engine already has the no-drop machinery
(used in selfplay); the Python SH steering needs **zero** changes (coord-based, already off-window-capable).

**Net-new:** one PyO3 method `expand_and_backup_ls` on `PyMCTSTree` (~60 LOC mirroring
`worker_loop/inner.rs::infer_and_expand` L650-701) + a per-cluster inference helper (stop collapsing+dropping in
`inference.py::infer_batch`) + a `legal_set` flag through `run_gumbel_on_board` → `DeployHeadBot`.

**Reused verbatim:** `expand_and_backup_ls`/`pick_topk_children_ls` (backup.rs), `aggregate_policy_ls`/`LegalSetPolicy`
(records.rs), `GumbelSearchState` SH, `get_root_children_info`/`forced_root_child` getters, engine
quiescence/dynamic-FPU/virtual-loss.

- **Oracle:** position-level block asymmetry (0/50 vs 12/50), NOT the effective-n=6 game probe. Require
  off_window_forced → 0.0 and the offline 12/50 block as a **floor** (Rust adds quiescence/FPU/VL → not byte-identical).
- **Prereq:** the arm-aliasing fix (done) — don't measure with a broken instrument.
- **bench gate:** NOT needed *if* changes stay in `pyo3/mcts.rs` (outside the bench-gate globs; selfplay hot loop
  untouched) — flips to **required** if implementation edits `backup.rs`/`selection.rs`.
- **Thermal:** `maturin --release` LTO build → route to vast or throttle `-j4`/dev; validation is GPU-light (laptop OK).
- **Open:** does the head need `get_improved_policy_ls` (only if it later feeds promotion-gate policy logging — not
  for the off-window proof). Center-order contract (recompute centers in Rust). Pin value min-pool in Rust.

## Track 3 — native engine::tactics in-window-offense solver core (READY-TO-BUILD)

A1-validated (+0.165 LIFT_IN_WINDOW). **Now simplified** by D-DECODE: in-window-offense-only + `finalize_game`
z-correction — drop off-window capability.

**Effort: mixed.** Core (steps 1–5, the net-free pattern-guided alpha-beta) = **LARGE PORT ~4–7 days** — the
quiet-move alpha-beta body + threat-quiescence is the hard, load-bearing part (the Python `solver.py` is
threat-only = the measured **8% ceiling**). A1 swap (step 6) = small wiring (the `solver_probe` DI hook + the whole
PAIRED gate already exist). Deploy root hook (step 7) = **bench-mandatory**. Training-z (step 8) = large vast corpus.

- **Reuse:** all forcing-move primitives (`board/moves.rs`, V4-fuzzed) + bindings, O(1) `apply_move_tracked`/`undo_move`
  + incremental zobrist (zero-clone-per-node), `solver_backup_bot.py` override semantics + `solver_probe` DI hook,
  the PAIRED bootstrap gate (`run_a1_solver_backup.py` + `a1_stats.py`).
- **Soundness invariant:** net value head NEVER read inside search; proof = terminal backup or forced-win/double-threat
  shortcut only; brute cross-check fuzz at 0 false-LOSS before any z-correction corpus run.
- **Risk:** SealBot is an in-window FLOOR (+0.165 is a lower bound); the solver must EXCEED it, achievable only via
  the self-play bootstrap (current blind net 0/14 can't policy-guide). Deploy root hook alone is a bonus, not the win.
- **Recommended sequencing:** core → A1 swap → deploy hook (bank the in-window lift cheaply) before the training-z weeks.
- **Design:** `reports/d_tactical_2026-06-26/NATIVE_RUST_SOLVER_design.md` (this executes TASK 3 minus off-window).
- **Thermal:** dev-iterate the core `-j4`/debug on laptop; route release-LTO + `make bench` + the A1 tournament + any
  z-correction corpus run to vast.

---

## Sequencing note
Laptop (Ryzen 7 8845HS) hard-cuts power under sustained Rust LTO builds — do **not** run two parallel release builds
on it. Recommend: Track 2 first (smaller, ships the just-validated defense fix; deploy multi-window is greenlit despite
~3× forward cost), Track 3 core as the focused follow-on (its own large effort). Dev builds in worktree isolation;
release-LTO + bench + tournaments on vast. Track-2 ship must land before/with Track-3 deployment so the in-window
scope assumption holds.

## Memory
`d-decode-offwindow-decoding-reframe`, `exploit-probe-arm-aliasing-bug` (cross-linked from
`d-solver-offwindow-deploy-head-hole`, whose "why g=0 ≠ PUCT" open question this closes).
