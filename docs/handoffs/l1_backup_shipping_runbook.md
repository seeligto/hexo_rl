# L1 + Deploy-Backup Shipping Runbook (D-PFIT closeout → execution)

**Decision (2026-06-30):** D-PFIT resolved the training-z question to a COUPLED lever; user chose **HOLD L2 (value-restart), SHIP L1 (decode/policy) + deploy-backup.** Full design + gate results: `docs/designs/coupled_valuez_decode_design.md` (§0 split, §9 gate results). This runbook is the execution plan for the GO path. Nothing here is committed/merged yet — commit on ask.

## Why this path (one paragraph)
The policy head already represents the solver saving moves; the binding constraint is the blind value's completed-Q veto, but only on a ~31% residual. The cheap gates showed: L1's architecture is SOUND (a multi-window-aware fit reaches the deploy root prior, in-sample flip = oracle 59%), there is a FREE training-free decode lift, and the deploy-backup (+0.165) is already validated. L2 (value-restart) has NO cheap positive signal (Stage-0 delta=0 + mean_v canary fired) and isn't cheaply de-riskable, so it's shelved.

## Workstream 1 — multi-window no-drop decode (`legal_set=True`) — ✅ SHIPPED 2026-06-30 (commit d9cb08b)
**DONE:** WS1 investigation (`wcszxw4aq`) found the +6% lift was NOT live in the promotion-gate strength eval (`DeployHeadBot` defaulted `legal_set=False` → single-window handicap; in-loop `Evaluator`/`KClusterMCTSBot` was already multi-window). Fix landed: `DeployStrengthEvaluator` now derives `legal_set = needs_no_drop_bot(encoding_spec)` and threads it to both cand + best bots — multi-window for `legal_set_scatter_max` (v6_live2_ls), bitwise-unchanged for single-window encodings. Regression test `test_deploy_strength_legal_set_derived_from_encoding` green (14 passed). **REMAINING confirmation (not blocking, do with a real v6_live2_ls ckpt — fold into WS2/next promotion eval):** off-window forced-loss rate → 0.0 spread-uncapped via the native/no-drop probe (ModelPlayer, exploit_probe arm-aliasing immune); in-window trap-flip reproduces ~22% (dpfit_l1).

### (original WS1 notes, retained for context)
- **Finding:** raw-net deploy trap-flip 16%→22% just by decoding multi-window (`legal_set=True`) vs single-window (P1b used False). Source: `scripts/dpfit_l1_mwfit_probe.py`, `reports/d_tactical_2026-06-26/l1_mwfit_probe_report.md`.
- **Task:** confirm the PRODUCTION deploy/promotion path's decode regime. `legal_set` defaults to False in `hexo_rl/eval/deploy_strength_eval.py:123` + `gumbel_search_py.py:140`; the multi-window no-drop path is §D-DECODE Track 2 (`legal_set_scatter_max`, `defender_dispatch.py`). Determine whether the native deploy bot (the one that actually plays / gates promotion) already decodes multi-window inherently or whether the flag must be turned on. If off, turn it on for deploy + the promotion eval. Re-verify off-window forced rate stays 0.0 (spread-uncapped, ModelPlayer — exploit_probe arm-aliasing immune).
- Cost: laptop, eval-only.

## Workstream 2 — deploy-backup native (Z1 deploy-root hook) — bench-gated on vast
- **Status:** the native Rust perf body is COMPLETE (increments 1-5) in worktree `.claude/worktrees/wf_02f2bc67-5fa-1` (commits f1848f4, 206af59, b72cd8b, c79bd5b, d599e22; branch `worktree-wf_02f2bc67-5fa-1` off `phase4.5/d-solver`), UNMERGED. 45 tactics tests green + 2 `#[ignore]` exhaustive sweeps; two-tier oracle + randomized verdict-invariance fuzz (76 pos, 0 false proofs); proof-vs-ordering separation confirmed (net-policy ordering inert by default).
- **Tasks (in order):**
  1. Merge the perf body worktree to `phase4.5/d-solver` (on user ask).
  2. Re-run the 2 `#[ignore]` exhaustive verify sweeps + full suite on **vast** (RTX 5080) to re-validate the new PVS/LMR/TT layer soundness under the full-width grid: `cargo test --lib tactics:: -- --ignored` (laptop thermal cutoff forbids local LTO).
  3. Wire the Z1 deploy-root hook: route `SolverBackupBot.solver_probe` through the native `PyTacticalSolver` in Rust at `mcts/mod.rs:267`. **This is the first change on the MCTS move path → MANDATORY `make bench` gate on vast** (use the bench-gate skill). The deploy-backup is proven-mate-WIN-only override (R3 LOSS path stays deploy-unaffected).
  4. (Pure-perf, soundness-neutral, optional) `eval.rs` incremental `_eval_score` make/undo accumulator replacing the per-leaf full re-scan.
- The deploy-backup gives the validated +0.165/+0.195 in-window lift AND carries the deep value-bound tail as deploy-time computation (permanent co-component).

## Workstream 3 — L1 self-play smoke — vast GPU-day, the actual L1 test
- **What it tests:** the L1 architecture is confirmed; the OPEN question is training-dynamics generalization (the static LOO probe is a memorization-regime instrument and can't settle it — self-play's on-policy regen trains on the held-out distribution).
- **Build:** wire multi-window no-drop decode + solver **SOFT visit-injection on PROVEN traps** (NOT one-hot policy distill — the probe shows one-hot is collaterally destructive) into self-play (`worker_loop/inner.rs` finalize_game / move-select hook, gated behind the ~6-line R3 LOSS guard per D-RECONFIRM). Visit-injection only (not net-guidance — blind net 0/14). Warm-start `checkpoint_00200000` (clean, no input-plane change).
- **Smoke (5-10k steps, ~1 GPU-day, vast):** PASS = GAME-DISJOINT held-out in-window trap-flip **≥25%** (real margin over control 12% / single-window 16%) AND deploy off-window forced rate HOLDS 0.0 AND threat-probe C1-C3 PASS. KILL = held-out flip ≤16% (decode-recovers-stranded wager false → deploy-backup carries those traps) OR off-window rises OR C1-C3 regress.
- **Visit-floor decision:** ship the deploy-time solver visit-floor (= deploy-backup) REGARDLESS of L1's trained yield — L1 only reduces deploy-time solver reliance.

## HELD — L2 (value-restart) — do NOT build now
Shelved per the gate evidence (Stage-0 delta=0, mean_v −0.281, ~15% ceiling, D-FULLSPEC wall live). Revisit ONLY if L1+backup proves insufficient AND there's appetite: requires building the REAL solver-derived `forced_loss/win_within_2` planes (not the dead cheap proxy) + the mandatory §1.1 precision audit (FP≤5% on a won-bank — the mean_v warning makes this load-bearing) + re-running Stage-0 on the REAL planes with the expanded corpus (~140 new distinct-game pairs minable offline via `scripts/dpfit_stage0_rerank.py --expand-scan`), then the gated GPU-week. Necessary-condition metric = pairwise-rerank `P(value(saving)>value(blunder))` on game-disjoint held-out; success = STANDALONE strength (deploy-backup OFF), distinct-game bootstrap, ModelPlayer false-clear co-gate.

## Artifacts
- Spec: `docs/designs/coupled_valuez_decode_design.md`
- Cheap-gate scripts: `scripts/dpfit_{export,fit,search_mechanism,l1_mwfit_probe,stage0_rerank}.py`
- Reports (gitignored, durable on disk): `reports/d_tactical_2026-06-26/{dpfit_p1b_report,l1_mwfit_probe_report,stage0_rerank_report}.md`, `redteam_forced_prior_rows.json`
- Corpus: `reports/d_tactical_2026-06-26/corpus.jsonl` (38 proven-core); data `data/dpfit_traps.npz`
- Perf body: worktree `.claude/worktrees/wf_02f2bc67-5fa-1` (UNMERGED)
- Anchors: `checkpoint_00200000` (L1 warm-start), deploy-backup `solver_probe` DI hook (D-SOLVER A1)
