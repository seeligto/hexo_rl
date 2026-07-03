<!-- D-PFIT coupled value-z x multi-window-decode design. Generated 2026-06-30 via 3-approach design workflow + adversarial Opus judges. Pre-GPU spec per user directive. -->

Grounding complete. Writing the synthesized spec.

---

# COUPLED MECHANISM SPEC — Solver-in-Loop Search-Distilled Value-z × Multi-Window No-Drop Decode (STAGED-TAC+)

**Status:** design, pre-GPU. Spine = Approach C (STAGED-TAC, judge total 40) with grafts from B (REALIZE-the-line, orthogonal-head coupling, recall pre-gate, mean_v fail-fast) and A (pairwise within-sibling rerank metric, self-play-must-match-deploy-decode, solver visit-injection-not-reweight). Objective: close the §P3 value-blind forced-loss plateau by making the catchable subset (~15–33%) value-learnable while carrying the deep tail as deploy computation — without re-triggering the D-INJECT/D-FULLSPEC anti-correlation, and without spending a GPU-week on an un-gated representational gamble.

---

## 0. The structural decision the judges forced: SPLIT the cheap high-yield lever from the expensive low-yield restart

**B2/B3-reframe (2026-06-30, review-mandated): the `59–69%` is a FORCED-ORACLE-PRIOR ceiling, NOT a trained-prior outcome — do not conflate them.** The number comes from the D-PFIT RED-TEAM `redteam_forced_prior_rows.json` (frc 19/32=59%, frc_floor 22/32=69%) where the saving-move prior was *injected* at the deploy root — i.e. it is the reach of the **deploy-time solver visit-FLOOR = the deploy-backup** (D-SOLVER A1, already validated +0.165/+0.195 in-window). What a TRAINED net realizes via visit-injection (P1b B1 path: visit signal → policy target → boosted deploy-root prior) is bounded at **31% in-sample / 16% held-out ≈ 12% control** — and even on the 12/32 subset whose prior already reaches the root at 0.998, only 5/12 = 42% flip, because the blind value's completed-Q vetoes the save line. So the WR decomposes as:
- **Deploy-backup (oracle visit-floor):** reaches 59–69% in-window at DEPLOY time — ships NOW, already built, permanent co-component (carries the deep tail too).
- **L1 trained-policy lever (the WAGER):** internalize as much of that reach into the NET (so it needs no solver at deploy) via multi-window no-drop decode + visit-injection. **Realized yield is 16–42% held-out, value-capped** — the explicit, currently-UNPROVEN hypothesis is that multi-window decode recovers the **20/32 single-window-stranded priors** (deploy prior 0.139 → root), lifting held-out *above* the 16% single-window ceiling toward the 42% clean-transfer rate. That lift is the whole L1 bet; it is asserted, not shown.
- **~31% (irreducible)** VALUE-bound — needs corrected-z; D-PERCEPT bounds the *learnable* part to ~15% (short/mid), the rest stays deploy-time solver computation. (Note: the value head binds a HARD slice of the nominal "69%" — the 7/12 clean-transfer non-flips are value-vetoes, not decode-stranded.)

Therefore the plan is **two decoupled levers funded by two separate gates**, not one bundled restart:

| Lever | Mechanism | Realized yield (honest) | Cost | Gate |
|---|---|---|---|---|
| **L1 — DECODE/POLICY** | multi-window no-drop decode (kcluster `v6_live2_ls`) + solver visit-injection into policy target, warm-start from `checkpoint_00200000`, **no input-plane change** | **16–42% held-out, value-capped** (NOT 69%; the 69% is the deploy-backup oracle floor) | cheap (warm-start clean, no restart) | Stage-1A |
| **L2 — VALUE-z** | search-distilled soft corrected-z + 8-plane tac warm-restart + light top-block co-evolution | ~15% learnable of the 31% value-bound residual | expensive (restart, discards 272k progress) | **Stage-0 headline gate**, then Stage-1B |

L1 + deploy-backup ship first and bank the bulk of the *deployable* WR (the deploy-backup oracle floor regardless of how much L1 internalizes). L2 is gated behind the headline cheap falsifier. Judge-demand (c) — *is the decode lever obtainable WITHOUT the from-scratch restart?* — answer: **yes, L1 needs no input-plane change**; but its trained yield is the 16–42% wager above, not the oracle 69%.

---

## 1. Mechanism

### 1.1 Value-z target (L2) — search-distilled soft corrected-z, PROVEN-TERMINAL-ONLY, PAIRED, REALIZE-preferred

For each self-play position the native `engine::tactics` solver **proves** a forced loss for the side-to-move. **Soundness gate (B1-pinned 2026-06-30, CLAUDE.md measurement-unit rule):** the gate is **terminal mate** `|score| ≥ 99_999_000` with sign from `engine::moves::terminal_value_to_move` (CF-1), NEVER a stored `sealbot_score`, NEVER the value head. The solver SEARCH-DEPTH budget (in plies/stones ≈ 2× mate-distance-in-turns under the HTTT 1-then-2 structure) must be deep enough to *reach* the mate — that is the "depth ≥ 7" of the DS1 re-score rule: a SEARCH budget for minting NEW labels, **not** a mate-ply filter and **not** a `proven_depth` filter. On the existing clean `corpus.jsonl` all 38 proven-core rows already pass (reality = proven-d6/d7/d8); do **NOT** filter by `proven_depth` (= the SealBot search-depth band {6:34, 7:3, 8:1}, NOT half-moves — a ≥7 filter would wrongly drop 34/38).

**Monotone mate-distance loss target** (d = the corpus **`mate_distance` field, in TURNS** — range 2–9 — NOT `proven_depth`, NOT plies; `depth_band` buckets this: short ≤3, mid 5–8, deep 9):
```
z_loss(d) = -( 1.0 - (d - d_min)/(D_max - d_min) * (1.0 - z_floor) )
            with d_min = 2, D_max = 8, z_floor = 0.5
  → mate-in-2 = -1.00,  mate-in-5 = -0.75,  mate-in-8 = -0.50,  d>8 clamp -0.50
```
Monotone, NOT a constant: an ORDERED regression target the head can fit while leaving the win region intact (constant −1 has only ONE satisfying point → global value shift → the measured anti-correlation).

**PAIRED saving-sibling (load-bearing, grafted from A):** the solver returns the saving move `m`. Enqueue the sibling position `parent+m` as its OWN labeled record with `z_save = +(1.0 - (d'-d_min)/(D_max-d_min)*(1.0-z_floor))` from its own proven escape distance `d'` (default `+0.6` if escaping-but-unproven). The two records — blunder `(−z)` and sibling `(+z)` — are one move apart and enter the SAME minibatch. The head now fits a **local decision boundary between two adjacent boards**, not a region-wide depression. This is the in-distribution same-node contrast every D-FULLSPEC fixed-corpus distill structurally LACKED.

**REALIZE > STAMP (grafted from B):** two composition modes per record —
- **REALIZE (preferred):** when the losing side can still play the saving line, force self-play to TRAVERSE it so `z` is the TRUE terminal outcome — a genuine lost game state, not a counterfactual stamp overlaid on the win-set neighborhood. Fixes the label-provenance failure mode of D-INJECT's counterfactual −1.
- **STAMP (fallback):** only where the blunder is already committed, write `z_loss(d)` with `value_valid=1`.

Unproven / unknown positions keep their ORGANIC game-outcome `z`, untouched (D-PERCEPT: no TD bootstrap, ever).

**Feature-GATED correction (graft from C, with mandated precision audit):** corrected-z fires ONLY where the new `opp_forced_loss_within_k` tac-plane is non-zero at that row → a LOCAL conditional gradient, not the global push that was ENTANGLED_LT's only mechanism for raising KILL-A. *Mandatory before Stage-1B (judge-demand a):* tac-plane PRECISION audit on a won-position bank — if the fork detector false-fires on look-alike WONs in the value-blind neighborhood it will drag them down anyway. Require false-positive rate ≤ 5% on a 300-won-position bank or the gating decoupler is void.

### 1.2 Representational fix (L2) — 8-plane tac warm-restart, NOT light-trunk-only

D-FULLSPEC exhausted the STATIC cheap-lever space (E1 frozen 0.441, ENTANGLED_LT light-trunk 0.532, E2 threat-readout 0.155, Closeout any-readout 0.43/0.37, richer-target 0.045). The surviving named lever is **search-in-the-loop with a richer co-evolving representation**. Fix =
- **(i) Solver-derived 2-ply FORCING input planes** (not E2's 1-ply open-line counts, which were falsified as a readout): new registry encoding `v6_live2_ls_tac` = `v6_live2_ls` 4-plane multi-window bundle (`is_multi_window=true`, `value_pool=min`, `policy_pool=legal_set_scatter_max`, `k_max=8`) **+ 4 tac planes**: opp/self mate-in-1 maps, opp/self double-threat/fork maps (≥2 simultaneous open-4/open-5 — the QUIET-forcing structure E2 missed), opp/self `forced_loss/win_within_2` coarse maps from the native threat-space prover, multi-cluster spread plane (off-window geometry). Byte-verify vs `engine/src/board/threats.rs:check_window` (`max_abs_diff 0.0`), `python -m hexo_rl.encoding audit` for Rust/Python parity, bench-gate the hot-path extraction.
- **(ii) WARM-RESTART:** load `bootstrap_model_v6_live2.pt`, transfer trunk ResBlocks 2–12 weight-for-weight, **zero-init the 4 new conv1 input channels** (step-0 forward == bootstrap), fresh-init value_fc1/fc2 + the multi-window policy head, fresh AdamW/LR/replay. Top blocks co-evolve under the on-policy paired contrastive gradient.

**Why this clears the D-FULLSPEC wall where the same-looking config failed:** D-FULLSPEC failed on a STATIC class-bag distilled into a frozen/lightly-unfrozen head measuring ABSOLUTE separation on the OLD 4 planes. Here (a) the SIGNAL is on-policy PAIRED-sibling contrast (continually regenerated), (b) the planes are co-evolving INPUTS for circuit-building (KataGo aux mechanism), NOT a frozen readout, (c) the headline gate (§4) tests the PAIRWISE rerank the register NEVER ran. **Honest limit:** D-FULLSPEC's mechanism finding (unfreeze raises KILL-A only by depressing value) means co-evolution-separates is NOT cheaply provable; it is carried as a LIVE mean_v KILL (§2), not assumed. This is why L2 is gated behind the headline falsifier and L1 (which needs none of this) ships first.

### 1.3 Multi-window no-drop decode coupling (L1 — the high-yield lever)

D-DECODE: off-window defense is ACTION-SPACE. Single-window `board.to_flat() = usize::MAX` → 60% of saving moves have NO logit slot → structurally un-decodable; full-game single-window off-window forced rate **0.335**, multi-window **0.0**. D-PFIT: single-window strands 20/32 in-window priors (deploy prior 0.139); multi-window puts the saving move in top-16 at rate 1.0, flips 59–69% when the prior is boosted.

Coupling chain (shipped Rust `legal_set_scatter_max` head, DeployHeadBot `legal_set=True`):
```
solver surfaces saving move m (incl. off-window)
  → kcluster no-drop decode gives m a logit slot
  → solver visit-mass INJECTED into m's policy-target slot  (graft A: INJECT, not re-weight —
       67% of saving moves are ~0-prior, re-weighting structurally cannot reach them; D-TACTICAL T0)
  → policy prior on m climbs
  → at DEPLOY the same legal_set_scatter_max head carries the prior to the multi-window root
```
**Self-play decode MUST match deploy decode (graft A, hard orchestrator requirement):** if generation uses single-window decode, off-window saving moves are un-expressible → no z/visit regenerates → the ratchet silently dies. Self-play and deploy both run `v6_live2_ls(_tac)` kcluster no-drop. Verify the self-play decode path == deploy kcluster path before any sustained run. Deploy off-window forced rate must HOLD 0.0 (D-DECODE floor); any regression = KILL.

### 1.4 Solver-in-loop self-play integration + perf-body recall requirement

- **Hook:** `engine/src/game_runner/worker_loop/inner.rs` — `finalize_game` (line 1183, after winner/terminal_reason + the existing final_cells/winning_cells snapshot; overwrite `outcome` at 1253 under the existing `value_valid` mask at 1245). Walk BACKWARD over the decisive band only (surfaced by the existing WIN→LOSS-persist / decisive-blunder scan, amortized — NOT every ply). Fully native Rust, no Python/FFI.
- **Orthogonal heads compose on one record (graft B):** VISIT signal → improved-policy target via `legal_set_scatter_max` (trains policy → L1's internalized lever, 16–42% held-out); CORRECTED-Z → value head under `value_valid` (trains value → L2's ~15%). Same buffer row, same multi-window frame.
- **Perf-body recall is a PREREQ, not optional:** the one-primitive net-free threat search ceilings at **8%** (D-TACTICAL: 3/38, all mate-in-2; mid 0/16, deep 0/1) because mates START with QUIET developmental moves (28/35 = 80% quiet). The native perf body (P2 scored α-β + 729-eval **LANDED**; aged-TT / PVS / LMR / net-ordering **remaining**, the multi-day handoff) supplies RECALL on quiet mid/deep traps. Without it, corrected-z reproduces the 8% ceiling and the value signal is too sparse to move the head.
- **Solver is NET-FREE / enumeration-complete (graft B):** completeness from threat enumeration + terminal backup, recall independent of net quality; the net only ORDERS for acceleration, NEVER declares a proof. Net-policy ordering stays OFF early (D-TACTICAL Probe C: blind net 0/14, chicken-and-egg) → enabled as a speedup only after the net trains.
- **R3 LOSS guard (D-RECONFIRM):** the not-in-check LOSS proof is emergent-sound but UNENFORCED; the quiet-move body breaks it → gate ALL training-z corrections behind the ~6-line R3 guard. Deploy-backup (WIN-only) path is unaffected.

---

## 2. KILL-A ⊥ KILL-C avoidance — the defensible mechanism

D-INJECT/D-FULLSPEC anti-correlation: pushing a proven-loss CLASS toward −1 through the value head's global pool, where win/loss are feature-CONFLATED in the `net_value>0` blind-spot neighborhood (within-matched turn-phase AUC = 0.500; in-sample head fits only 46% of its OWN tp0 wins), can only lower losses by lowering the whole neighborhood → ~25% of wins flip; signature = mean_v +0.055 → −0.407 on 300 unrelated positions.

**The honest reduction (per all three judges): of the many listed "decouplers", only TWO add power beyond "does the representation separate":**

1. **PAIRED same-node sibling contrast** — the ONLY mechanism that gives the head a LOCAL boundary instead of a region. Two adjacent boards, opposite signs, same minibatch. This is the structural contrast the register's fixed corpora never had. *Risk the judges flagged:* adjacency makes the inputs nearly identical → it could DEMAND a sharper local discrimination the frozen-ish features can't express → reproduce the anti-correlation with a sharper ask. This is exactly what the headline gate (§4) measures — pairwise rerank IS the test of whether the (co-evolved) representation can make this local cut.
2. **DISTRIBUTION SHIFT (on-policy ratchet)** — once the policy ratchets onto saving moves, the trap positions STOP being generated; the win-set being preserved is the LIVE self-play distribution, not a static KILL-C bank being dragged. *Risk:* if the policy doesn't ratchet (Probe C 0/14), this reverts toward a fixed-corpus regime. Mitigated by solver visit-INJECTION (not net-guidance) which works on a blind net.

**Explicitly DOWNGRADED to contingent (do NOT count as independent breakers — re-validation honesty):**
- *Soft/monotone target* — D-FULLSPEC Closeout distilled a richer continuous target → KILL-C **0.045**, craters HARDER than ±1's 0.441. Softness alone is FALSIFIED as a collateral-reducer **on a frozen representation**. It helps ONLY conditioned on co-evolution + paired contrast. Kept as target-FORM hygiene, not as a dodge.
- *Feature-gating* — fires on the conflated look-alike WONs too unless the precision audit (§1.1) passes. Contingent on separation.
- *REALIZE-the-line* — fixes label PROVENANCE, not separability. Genuine improvement (real lost trajectories beat counterfactual stamps) but does NOT touch the binding constraint.

**The decisive detector (graft A+B):** `mean_v` on a FIXED 300 neutral-position probe, every 5k steps. KILL if it drifts > 0.2 below bootstrap (or absolute < −0.15) — the exact ENTANGLED_LT signature. This is the tripwire that fires if the anti-correlation re-emerges despite the two breakers. It is a KILL (burns remaining steps), not a preventer — which is WHY the headline gate must clear first.

---

## 3. Re-validation ledger (CLAUDE.md mandate: cite → context → transfer-test → keep/drop)

| Prior | Falsification CONTEXT | TRANSFER-TEST | Verdict |
|---|---|---|---|
| **D-INJECT** (constant −1 distill NO-GO; KILL-A⊥KILL-C across ALL weights) | STATIC one-sided class-push to the −1 RAIL on a fixed proven-loss corpus, no contrast partner, no distribution shift, counterfactual stamps | Target here is MONOTONE mate-distance (not constant), PAIRED with same-node +z sibling, REALIZED (real outcomes), on-policy regen. The anti-correlation MECHANISM (one-sided rail push drags shared-rep wins) does NOT transfer to a paired local re-ranking target. | **KEEP as a target-FORM ban** (never a constant class target); **DROP as a blanket veto.** Re-validated LIVE by the mean_v KILL (§2) — not assumed solved. |
| **D-FULLSPEC** (frozen + light-trunk + E2-readout + any-readout + richer-target ALL fail; KILL-C [0.32–0.46], Closeout richer-target 0.045; "restart NOT evidence-justified"; RULE: no GPU-week without a cheap KILL-C≥0.85 discriminator) | STATIC distill/readout on FROZEN/lightly-unfrozen OLD-4-plane representation measuring ABSOLUTE class separation; turn-phase shortcut AUC 0.807 | This design IS the search-in-the-loop survivor D-FULLSPEC explicitly named, with RICHER 2-ply forcing INPUT planes (not a readout) + PAIRED on-policy signal + PAIRWISE rerank metric. **Closeout-richer-target note (judge-demand b):** the 0.045 crater was on the FROZEN 272357 rep — does NOT transfer to a co-evolved rep, BUT it DOES falsify softness-as-standalone-dodge, so softness is downgraded (§2). The cheap-discriminator RULE is OBEYED literally: §4 supplies the missing turn-phase-matched discriminator. | **KEEP** the turn-phase-matched bar + the no-GPU-week-without-a-cheap-discriminator RULE (obeyed by §4) + "threat planes are accelerant not standalone separator". **DROP "restart never justified" ONLY conditional on §4 clearing.** |
| **D-PERCEPT** (67% of value-blind core DEEP; only ~15% short-lookahead-catchable → target must be OUTCOME/SEARCH-DISTILLED, not cheap TD) | depth characterization of the blind-spot | Transfers DIRECTLY and CONSTRAINS: z is solver-proven mate-distance + realized outcomes (search-distilled), TD/bootstrap value targets FORBIDDEN. L2 claims ONLY the ~15% catchable; the deep 67% is carried as deploy-time solver-backup computation (registered as bound + pre-reg prediction). | **KEEP as a hard constraint AND as the lever's reach-bound.** |
| **DS1-stale** (`is_proven_loss` labels stale; ~2.5% sign-flip; d6→d7 flips) | stored `sealbot_score` stale; soft heuristic labels | All solver labels re-proven LIVE on native **terminal mate** `|score|≥99_999_000` + engine CF-1 sign, never stored score; the "depth≥7" is the SEARCH-DEPTH budget for completeness (B1-pin), NOT a `proven_depth`/mate-ply filter. Cheap gate uses the clean reachable-replay corpus (all 38 proven-core, D-TACTICAL V2 net-parity 61/61). | **KEEP as a soundness invariant** (build requirement). |
| **D-TACTICAL** (blind net can't policy-guide solver 0/14; broadening regressed to 0%; exceed SealBot only via self-play bootstrap) | current blind net ORDERING the cheap solver; threat-only broadening | Solver kept NET-FREE / enumeration-complete; net-ordering gated to "after the net trains". Visit-INJECTION (not re-weight) reaches the 67% ~0-prior saving moves. Injection scoped to PROVEN traps only (broadening regressed → keep narrow), behind the R3 guard. | **KEEP, respected.** The chicken-and-egg is broken by closing the loop (§1.4), not by net-guidance. |

---

## 4. THE CHEAP PRE-GPU-WEEK GATE (headline — must PASS before any GPU-hour on L2)

> **Mirrors the D-PFIT cheap-probe philosophy that just overturned a false kill: the register only ever measured ABSOLUTE class separation; the decision-relevant question is PAIRWISE local re-ranking, which it NEVER ran.**

**Gate name:** `STAGE-0 PAIRWISE-RERANK SEPARATION DISCRIMINATOR` — eval-only, ≤ 1 GPU-day, NO self-play, NO training-week, laptop-feasible (-j4, the corpus already exists).

**Build:** on the clean `reports/d_tactical_2026-06-26/corpus.jsonl` (38 proven-core, game-disjoint). For each blunder position compute the native solver saving sibling → PAIRS `(blunder, z_loss(d)) ∪ (saving-sibling, +z_save)`. Compute the 8 tac planes natively. Train a small from-scratch conv (E2 harness lineage, `scripts/dvderisk_e2_featablation.py` / `closeout_probe.py`): **4-plane CONTROL vs 8-plane TREATMENT**, light top-block + value head, on a TURN-PHASE-MATCHED set (planes 2,3 equalized — strip the AUC-0.807 shortcut; verify within-matched turn-phase AUC ∈ [0.45, 0.55] = control valid). Split verified `shared_games == 0`; effective-n via **distinct-game bootstrap** (§D-ARGMAX, NOT raw count — ~13 held-out pairs from fewer distinct games, so report the bootstrap CI honestly).

**THE single decision metric (the one the register never ran):**
```
PAIRWISE-RERANK = P( value(saving-sibling) > value(blunder) )   on the GAME-DISJOINT held-out pairs
   pre-FT control ≈ 0.50 by construction (the net is value-blind)
```

**PASS (all three):**
1. `PAIRWISE-RERANK ≥ 0.70` (8-plane treatment), distinct-game bootstrap CI_lo > 0.55
2. turn-phase-matched held-out `KILL-C ≥ 0.85` (win-preservation; the D-FULLSPEC bar, obeyed literally)
3. `|mean_v drift| ≤ 0.10` on 300 unrelated positions (anti-correlation canary)

**KILL (any one):**
- `PAIRWISE-RERANK < 0.60` OR `KILL-C < 0.75` OR `mean_v < −0.15`

**What each outcome IMPLIES:**
- **PASS** → the PAIRED/co-evolved signal makes the LOCAL cut the class-bag could not → the representational gamble is retired CHEAPLY → fund L2 (Stage-1B + GPU-week).
- **KILL** → L2 (value-restart) is dead. **Fall back to L1-only** (decode/policy lever, 16–42% held-out internalized) + the validated deploy-backup oracle visit-floor (+0.165/+0.195 in-window, 59–69% oracle reach) carrying the deep tail. Total L2 spend capped at ~1 GPU-day. This is the gate WORKING.
- **INDETERMINATE** (CI straddles, likely given ~13 pairs / ~5 short-catchable positions) → do NOT fund the GPU-week; either expand the corpus via offline solver replay on more buckets first, or proceed L1-only. Honest: the judges flagged thin power as the #1 reason this could return INDETERMINATE rather than PASS — budget for a corpus expansion pass.

**Second cheap gate (prereq for the loop, graft B), `SOLVER-RECALL`, ~hours:** run the BUILT native perf body offline on the same 38-core at a self-play-affordable budget (target ≤ ~1s/pos amortized at root with net ordering OFF). Require **prove-rate ≥ 90% (34/38)** with **0 soundness violations** (brute cross-check, 0 false-LOSS). RATIONALE: if the quiet-move body still ceilings near 8%, the training loop STARVES regardless of separability — no GPU-week helps. KILL → ship the remaining perf body (aged-TT/PVS/LMR) before re-running; do NOT start L2 self-play.

---

## 5. Staged plan + pre-registered kill criteria

**Stage-1A — L1 DECODE/POLICY (cheap, ships first, no headline gate needed):** wire kcluster `v6_live2_ls` no-drop decode into self-play + deploy (already validated: full-game off-window forced 0.0), warm-start `checkpoint_00200000` clean (no input-plane change), solver visit-injection into the policy target. **Explicit L1 hypothesis under test:** multi-window no-drop decode recovers the 20/32 single-window-stranded priors → lifts **game-disjoint held-out** in-window trap-flip *above* the P1b single-window held-out ceiling (16%), toward the 42% clean-transfer rate. **Measurement is held-out (NOT in-sample — in-sample 31% is the memorization upper bound and proves nothing).** The smoke routes through the fitted-NET deploy path (run_gumbel_on_board has no external prior hook — root prior comes from the net only). Smoke (5–10k steps, ~1 GPU-day) **PASS:** held-out trap-flip **≥ 25%** (a real margin above control 12% / single-window held-out 16%, NOT the unreachable 30%) AND deploy off-window forced rate HOLDS 0.0 AND C1–C3 PASS. **KILL** if held-out trap-flip ≤ ~16% (no lift over the single-window ceiling → the decode-recovers-stranded-priors wager is FALSE) OR off-window forced rate rises OR C1–C3 regress → L1-as-internalization invalid; the **deploy-backup oracle visit-floor (already built, +0.165) still ships** and carries the in-window saves at deploy time. **Visit-floor decision (B2):** ship the deploy-time solver visit-FLOOR (= deploy-backup) REGARDLESS of L1's trained yield — L1 only reduces deploy-time solver reliance, it does not replace the floor.

**Stage-0 — headline cheap gate (§4):** ≤ 1 GPU-day. PASS → fund L2. KILL → L1-only + deploy-backup, STOP L2.

**Stage-1B — L2 build (only if Stage-0 PASSES):** tac-plane precision audit on 300-won bank (FP ≤ 5%, judge-demand a) → `v6_live2_ls_tac` encoding + audit + bench-gate → finalize_game solver hook + R3 guard + REALIZE/STAMP + corrected-z math + warm-restart conv1 4→8.

**Stage-2a — L2 smoke (5–10k steps, ~1 GPU-day):** corrected-z incidence ≥ 0.5% of value rows (target ≥ 2%; < 0.5% = labels too sparse, HALT) AND realized-outcome sign-AUC ≥ 0.99 vs proof AND held-out trap-flip ≥ 30% AND live-position KILL-C ≥ 0.85 (deploy WR vs SealBot-d5 ≥ 0.49, no regression) AND `mean_v` on 300 neutral within ±0.10 of bootstrap (ENTANGLED_LT canary) AND C1–C3 PASS. **KILL any** → revert to L1-only + deploy-backup.

**Stage-2b — L2 full (~1 GPU-week, only if 2a clears):** decision ckpt ~150–200k. **SUCCESS metric = STANDALONE net strength with deploy-backup OFF** (Z2/Z3 discipline — NEVER deploy-backup-vs-SealBot, that re-measures the crutch): `deploy_strength_eval.py` g=0 multi-window legal-set vs fixed-depth-5 SealBot, n ≥ 200 DISTINCT games, distinct-game bootstrap BT-Elo, `CI_lo(WR@restart-best) > CI_hi(WR@bootstrap-baseline)`; held-out in-window trap-flip ≥ 50%; D-LOCALIZE value-blind rate ≤ 30% (vs 92%) on a held-out trap corpus. **Co-gate (false-clear rule):** off-window forced rate held 0.0 on a spread-uncapped/adversarial probe using ModelPlayer (immune to the exploit_probe arm-aliasing bug) — a fixed-bot WR can false-clear an off-window defect. **Live KILLs every 5k:** `mean_v` drift > 0.2 below bootstrap (anti-correlation); argmax self-ladder WR regressed > 5%; C1–C3 regress; off-window forced rate rises. **Fail-fast at 100k:** no trap-flip movement above control AND mean_v depressing → KILL before the second GPU-week.

**Pre-registered final prediction (D-PERCEPT-consistent):** the value head absorbs only the short/mid catchable ~15–33%; the deep tail STAYS a deploy-time solver-backup residual, NOT learned → WR lift is BOUNDED and the deploy-backup is a PERMANENT co-component (Z2 "DOESN'T-TEACH" branch), not a crutch to be removed.

---

## 6. Cost + reuse

**Reused (no rebuild):**
- **P2 perf body** (scored α-β + 729-eval) — LANDED. Remaining: aged-TT/PVS/LMR/net-ordering (multi-day, gates the SOLVER-RECALL gate + L2 loop).
- **Clean corpus** `reports/d_tactical_2026-06-26/corpus.jsonl` (38 proven-core, game-disjoint, net-parity 61/61) — drives both cheap gates.
- **Deploy-backup** (D-SOLVER A1, `solver_probe` DI hook, +0.165/+0.195 in-window) — ships now as the short-band bonus AND the permanent deep-tail carrier.
- **kcluster `v6_live2_ls`** multi-window no-drop decode (full-game forced 0.0, shipped Rust `legal_set_scatter_max` head, DeployHeadBot `legal_set=True`).
- Anchors: `checkpoint_00200000` (L1 clean warm-start), `bootstrap_model_v6_live2.pt` (L2 restart).

**Cost ladder (downside-capped):**
| Item | Cost | Where |
|---|---|---|
| SOLVER-RECALL gate | ~hours, eval-only | laptop / 1×5080 |
| **Stage-0 headline gate** | **≤ 1 GPU-day** | laptop -j4 feasible (corpus exists) |
| Stage-1A L1 smoke | ~1 GPU-day | vast 5080 |
| Perf-body remainder (if recall gate fails) | ~3–7 eng-days | vast release-LTO (thermal rule: NOT laptop LTO) |
| `v6_live2_ls_tac` encoding + audit + bench | ~1 eng-day | laptop |
| finalize_game hook + corrected-z + R3 | ~3–5 eng-days | laptop -j4 |
| Stage-2a L2 smoke | ~1 GPU-day | vast |
| Stage-2b L2 full week | ~1.5–2.5 GPU-weeks (solver CPU + ~3× multi-window forward) | vast |

**Downside if both cheap gates KILL:** total ~2 days, program lands at **L1 decode/policy + deploy-backup-only** — the deploy-backup oracle visit-floor (59–69% in-window at deploy) banked, plus whatever L1 internalized (16–42% held-out), the deep tail carried as computation, ZERO value-restart GPU-week spent. The gate working = the win.

---

## 7. Honest most-likely outcome + the single biggest risk

**Most-likely outcome:** L1 (decode/policy) ships and banks most of the in-window WR lift cheaply. The **Stage-0 headline gate returns INDETERMINATE or marginal** on first run — ~13 held-out pairs / ~5 short-catchable positions is thin power (every judge flagged this); pairwise-rerank likely lands in [0.55, 0.70] with a CI that straddles. Expect a corpus-expansion pass (offline solver replay over more s150k/175k/200k buckets) before Stage-0 is decisive. If it then PASSES, L2 absorbs only the ~15% short/mid catchable, leaving the deploy-backup permanent — a bounded, real WR gain, not a blind-spot cure.

**The single biggest risk:** **co-evolution still does not SEPARATE win/loss in the value-blind neighborhood, and pairwise-rerank passes Stage-0 on the 38-core but does NOT generalize** — the same in-sample-vs-holdout divergence that was D-FULLSPEC's smoking gun (in-sample 46% vs holdout crater), now hiding in a thin pairwise CI. If that happens, Stage-2a's mean_v canary fires mid-week (the anti-correlation re-emerging as global depression), the GPU-week is KILLED at ≤100k by fail-fast, and the verdict converges on D-FULLSPEC's "DEEP/HORIZON architectural limit": the value-restart is NOT the lever, only L1 + permanent deploy-backup are. Mitigation is structural, not certain — the two cheap gates + the live mean_v KILL cap the loss at one GPU-day of L2 commitment plus the fail-fast week-half, never the full bundled restart the judges warned against.

---

## 8. Review-mandated corrections (pre-exec gate, 2026-06-30)

The pre-exec review (`wmlgvwwyp`) returned: **P2 = GO, Stage-0 = GO-after-B1, L1 = BLOCK-until-reframed**. Resolutions folded in above; carried items:

**B1 — z-label unit [RESOLVED, §1.1 + §3]:** gate = terminal mate `|score|≥99_999_000` (CF-1 sign); `d` in `z_loss` = corpus `mate_distance` (TURNS, 2–9); `proven_depth` {6:34,7:3,8:1} is the search-depth band, NOT a filter (all 38 proven-core qualify); "depth≥7" = SEARCH budget (plies≈2×turns) for minting NEW labels only.

**B2/B3 — L1 reframe [RESOLVED, §0 + §1.4 + Stage-1A]:** the 59–69% is the deploy-backup oracle visit-floor, not trained yield; L1 realizes 16–42% held-out (value-capped); Stage-1A measures GAME-DISJOINT held-out vs control 12%/16% (PASS ≥25%, KILL ≤16%), routes through the fitted-net path (no external prior hook in `run_gumbel_on_board`), and the deploy-backup visit-floor ships REGARDLESS of L1's trained yield.

**Carried non-blocking concerns (execution must respect):**
- **Stage-0 power is likely permanently thin in the PASS direction:** offline SealBot re-mine via `sourcing.py` only RE-MINES the same 68-decisive-game / 38-core pool. *Genuinely new* proven pairs need either deeper mining of the other ~212 of the 280 self-play games (non-decisive proven losses / other plies — assess yield) or NEW self-play (GPU). Treat a non-PASS as **underpowered**, not a clear KILL; report the bootstrap CI honestly (distinct-game, §D-ARGMAX — NOT raw count).
- **Stage-0 decides on the 4-plane CONTROL vs 8-plane TREATMENT *delta*, NOT the absolute 0.70/0.85 thresholds** (those may not transfer from a small from-scratch conv to L2's warm-restarted 12-block trunk). E2's 1-ply planes ceilinged 0.646; the 8 planes are genuinely 2-ply/quiet-forcing but still an input-plane readout on a small conv → low prior of cleanly clearing 0.85; lean on the delta.
- **Baseline hygiene:** use the corrected **54%** value-blind baseline (D-PERCEPT honest core), NOT the 92% over-count, for the Stage-2b "vs" comparison.
- **Stage-2b distinct-game generation:** g=0 deterministic deploy collapses to ~2 distinct games/pair (D-ARGMAX) → the "n≥200 distinct games + distinct-game bootstrap" requires explicit **injected opening/opponent diversity** to GENERATE the distinct games (currently unspecified — mandate it, else the CI is √(copies) over-confident). Off-window co-gate uses **ModelPlayer** (immune to the exploit_probe arm-aliasing bug); opponent = fixed-depth-5 SealBot.
- **Honest false-PASS cost:** ~eng-days + ~0.75 GPU-week (Stage-1B 3–5 eng-days + Stage-2a 1 GPU-day + up to half of Stage-2b before the mean_v fail-fast@100k), NOT the "≤1 GPU-day" the cost table implies for L2.
- **Cheap code asserts (Stage-0/L1 harness):** `head_logp(cache_trunk_out)==forward()[0]` parity at fit start (silently wrong on `gpool_bias_active=True`/PMA checkpoints); game-disjoint LOO `shared_games==0` assert in `load_traps`; read `kept_plane_indices` from the encoding registry, not hardcoded `[0,8,16,17]`.

---

## 9. GATE RESULTS (run 2026-06-30 — laptop, eval-only, zero GPU)

**L1 multi-window-fit probe (`scripts/dpfit_l1_mwfit_probe.py`) — PARTIAL: architecture SOUND.**
- The head CAN reach the multi-window root prior: a multi-window-aware fit drives the decoded prior 0.091→0.980 on 20/20 stranded traps; in-sample deploy flip = the oracle 59%. P1b's stranding was a single-window TARGETING artifact, NOT a head-capacity limit. **L1's architecture premise is confirmed.**
- **Free, training-free lift:** switching deploy to the correct multi-window decode (`legal_set=True`; P1b used `legal_set=False`) lifts raw-net flip 16%→22% (recoverable-stranded 21%→36%), zero training. **SHIP `legal_set=True` regardless.**
- Held-out static LOO fit ≈16% (no generalization) BUT that's the wrong instrument (tiny-data memorization regime; self-play's on-policy regen trains on the held-out distribution, a static fit cannot). Valid upper bound = in-sample 57–59% (value-capped); the 12–15% is a memorization floor, not the smoke ceiling. **L1 GPU smoke JUSTIFIED but risky on generalization → use SOFT visit-injection on proven traps, NOT one-hot distill (probe shows one-hot is collaterally destructive).**

**Stage-0 PAIRWISE-RERANK gate (`scripts/dpfit_stage0_rerank.py`) — KILL on the CHEAP PROXY (does NOT cleanly kill L2's real mechanism).**
- 4-vs-8-plane held-out rerank DELTA = **+0.000** (distinct-game bootstrap CI [−0.19,+0.19]); both at 0.583 (straddle chance). KILL-C craters (treat 0.36 / ctrl 0.22, bar 0.85). **mean_v canary FIRES: treatment −0.281** (the open-4 planes fire in win AND loss neighborhoods → "threat present" misread as loss → anti-correlation WORSE).
- **Load-bearing caveat:** the cheap Python-prototyped planes are a documented WEAK proxy — literal mate-in-1 planes are 100% DEAD at these mate-in-3..9-TURN traps; only the confounding open-4 `threat_moves` planes fire. The REAL solver-derived `forced_loss/win_within_2` maps (precise, fire only on proven-loss structure) are what L2 uses and are NOT built here. **So the cheap gate CANNOT pre-clear L2 cheaply — a valid test requires the real solver planes (Stage-1B, the expensive step the gate was meant to de-risk). Chicken-and-egg.**
- The mean_v warning REINFORCES that the §1.1 precision audit (FP≤5% on a won-bank) is mandatory and load-bearing: imprecise "threat-present" planes provably worsen the anti-correlation.
- **Corpus expansion:** ~140 genuinely-new distinct-game pairs minable offline from 85 untapped model-lost games (`--expand-scan`, no GPU) — 4× the corpus; fixes power-in-n but cannot flip a structurally-zero delta on dead proxy planes.

**Consolidated decision:** L1 (decode/policy) + deploy-backup oracle floor are GO — ship the free `legal_set=True` lift + the validated +0.165 backup, L1 GPU smoke justified (soft visit-injection). **L2 (value-restart) is HELD**: no cheap positive signal, mean_v warning fired, and the cheap gate proved L2 cannot be cheaply de-risked — testing its real mechanism needs the expensive solver-derived-plane build (Stage-1B) + precision audit + expanded corpus, with a still-live D-FULLSPEC-wall risk. Do NOT fund the L2 GPU-week now.

**USER-CONFIRMED 2026-06-30: Hold L2, ship L1 + backup.** Execution plan: `docs/handoffs/l1_backup_shipping_runbook.md`. The native perf body (increments 1-5, 45 tactics tests + randomized verdict-invariance fuzz green, proof-vs-ordering separation confirmed) is COMPLETE in worktree `wf_02f2bc67-5fa-1` (UNMERGED) — it underpins the deploy-backup Z1 native hook (next: merge → vast exhaustive sweeps → Z1 hook + mandatory `make bench` on vast).