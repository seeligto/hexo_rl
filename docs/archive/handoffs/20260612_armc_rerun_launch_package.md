# Arm-C Fixed-Loop Re-run — Launch Package (DESIGN-ONLY, operator-gated GPU)

Assembled by **D-RERUNPREP** (Phases 0–2 complete + Phase-3 package ready). The 50k
re-run does NOT launch until Phase 3 (GPU smoke) PASSES and the operator gates it.

Companion artifacts:
- Watch sheet: `docs/archive/handoffs/20260612_armc_rerun_watchsheet.md`
- Runbook (unchanged, still canonical): `docs/archive/handoffs/20260612_loopfix_armc_rerun_runbook.md`
- Phase-3 smoke: `configs/variants/v6_live2_ls_ab_smoke.yaml` + `scripts/run_phase3_armc_smoke.sh`
- Phase-1/2 raw verdicts: `reports/audits/d_rerunprep_phase12.workflow.js` (workflow), run `wf_6d6a620e-0d7`

---

## 1 · GO / NO-GO sheet

| Gate | Verdict | Evidence |
|---|---|---|
| **Phase 0 — post-merge hygiene** | ✅ **P0-PASS** | FF linear merge; all 7 loopfix commits (`a4d43fe`→`ab8d71d`) reachable from HEAD `1845d47`; `make test` **1994 passed / 0 failed**; `engine/src` diff EMPTY → bench-SKIP justified (Python-only; hammerhead is a submodule pointer); laptop+vast trees clean (only untracked scratch); gate fix `b340e99` in master both hosts; `master==origin/master`. |
| **Phase 1/2 — static sweep** | ✅ **0 BLOCKERS** | 6 buckets B1–B6 + aggregate/review/red-team. See §2. |
| **Phase 3 — GPU smoke (run 1)** | ❌ FAIL → diagnosed | Caught **W2-VACUOUS** (launch fresh-init skipped the pin) + **host bootstrap DRIFT** (vast `4198d5cb` ≠ laptop-derived pin `aba28e10`) + F2/F3. (Initial "fp16" mechanism was a misdiagnosis, corrected by a GPU discriminator.) Record: `phase3_smoke_results.md`. |
| **Phase 3 — GPU re-smoke (run 2)** | ✅ **GREEN** | On the fixed code + reconciled pin `4198d5cb`. Stage A (closeout integration) PASS; criteria 1–7,9 PASS (crit4/crit5 confirmed via raw logs after a self-check grep fix), crit8 SKIP (no promotion; W3 unit-proven). **crit7 (pin verified) + stage-D resume PASSING the pin** (hard-failed in run 1) are the on-GPU proof the W2-VACUOUS fix + reconciliation work. terminal eval `completed=true, wr_best=0.565`. |
| **Phase 4 — launch package** | ✅ this document | Design-only; no launch. |

**Bottom line:** the Phase-3 GPU smoke FAILED on run 1 — catching a real code hole (**W2-VACUOUS**:
the launch never verified the incumbent pin), a **host bootstrap drift**, and an initial misdiagnosis
(corrected by a cheap GPU discriminator). After the fixes + pin reconciliation to the de-facto
`4198d5cb`, the **Phase-3 re-smoke is GREEN** (pin now verified; the resume that hard-failed now
passes). **The 50k is CLEARED for operator launch** — pending authorization to push `master` to origin
(currently blocked) so vast can pull the fixes, OR an operator-run launch on the rsync'd vast code.

---

## 2 · Bucket verdicts (per-acceptance, with evidence)

| Bucket | Verdict | Evidence / note |
|---|---|---|
| **B1 — terminal eval is TERMINAL** | **PASS** (was GAP→overturned) | Forward isolation: terminal result emits distinct `terminal_eval_complete` (never `eval_complete` → never enters steering), runs outside `step()` (`step_coordinator.py:1465-1577`). Resume isolation: `resolve_anchor` loads only the `best_model.pt` resilient chain (a promoted *model*, never the terminal *record*) (`anchor.py:310-315`). **Patch DROPPED as redundant** — `test_terminal_eval_runs_full_battery_ignoring_stride` + `test_terminal_eval_skipped_on_sigint` already cover it (verified by reading the tests). |
| **B2 — planted incumbent hard-fails** | **PASS** | `resolve_anchor` raises `RuntimeError` (non-zero exit) when loaded sha ≠ config pin, **pre-training** (`loop.py:145-160` before `pool.start()`), message names both shas (`anchor.py:346-361`). `test_planted_wrong_incumbent_fails_loudly` PASSES. sha canonicalises compile prefixes (`anchor.py:120-128`). |
| **B3 — runbook↔config parity** | **PASS** | sha `aba28e10…` verified vs `bootstrap_model_v6_live2.pt`; bar **220/400=0.550** (`evaluate_gate`); bootstrap_anchor stride 1 / sims 128; nnue+offwindow stride 2; **corpus `bootstrap_corpus_v6_live2.npz` present on laptop+vast** (closes B3's own caveat); config loads no missing-key; `bootstrap_model.pt` (unversioned random footgun) **unreachable** from `resolve_anchor`. Objective-A in-run reads: nnue+offwindow at **rounds 2,4** (2 each) + terminal — W1's zero-reads sin closed. |
| **B4 — opening-jitter pre-flight** | **PASS (moot)** | Jitter is **UNDEPLOYED** in the re-run (`random_opening_plies=0`), so the off-window-injection hazard cannot fire. Mechanism validated if ever used: 100% in-window on bootstrap, 55% distinct @3-ply, dedupe + effective-n guard work. **Caveat:** an off-window-specialised future checkpoint could drift jitter off-window — future-run validation, named in SYNTHESIS. |
| **B5 — GREENLIGHT wiring** | **PASS** | (1) `exploit_probe.py` runnable, in-run offwindow stride-2 schedule; (2) `KClusterMCTSBot` 17 unit tests pass + instantiation smoke; (3) Arm-B **0.03** overlay source in design doc, explicit "beat Arm-B overlay" in runbook L64-66 (prose, not a separate template — operationally sufficient); (4) golong@50k `checkpoint_00050000_PEAK_sb0.38.pt` loadable (7 keys, 48.7 MB). |
| **B6 — watch sheet** | **DELIVERED** | `docs/archive/handoffs/20260612_armc_rerun_watchsheet.md` (authored; stride→round firing **corrected** — B6's draft mis-stated rounds 1,2,3, true mapping is rounds 2,4). |

**Adversarial passes:** review = CONCERNS (correctly flagged the redundant B1 patch); red-team =
AGGREGATE-WEAK with `git_diff_clean=true` (no tracked source mutated during Phase 1 — Explore
agents were read-only as designed; **I re-verified `git status` myself: clean**).

**Open (non-blocking) items:**
- **TICKET (doc-nit):** mirror the runbook's W1/W2/W3/POWER confound disclosure into the
  `v6_live2_ls_ab.yaml` preamble, so the "5 non-encoding changes" are acknowledged at the config
  (they are the corrected instrument, not a second experimental variable — already disclosed in
  the runbook). Optional.
- **ACTION before quoting cost:** the ~$67 / 4.25-day estimate rests on an **unverified** 490
  steps/hr. Read the golong 50k start→step-50000 wall-clock from its logs and re-derive.

---

## 3 · What Phases 1–3 changed vs the runbook

The runbook (`loopfix_armc_rerun_runbook.md`) and config (`v6_live2_ls_ab.yaml`) are **unchanged
and remain canonical**. D-RERUNPREP added, did not alter:
- **Corrected** the Objective-A firing cadence in the watch sheet: stride-2 → **rounds 2,4** (not 1,2,3).
- **Verified** the corpus file exists on both hosts (was an unverified B3 caveat).
- **Added** the Phase-3 smoke variant + self-checking run script (the deferred "GPU-bound" acceptance).
- **Dropped** B1's proposed test as redundant (existing closeout tests cover it).

---

## 4 · Confound disclosure (VERBATIM — state before any greenlight read)

> **PRIMARY — absolute off-window robustness (NO matched arm needed).** `exploit_probe.py
> --arms exploit,control` ≤ **0.06** AND the causal-uncapping counterfactual via
> `KClusterMCTSBot` (NOT single-window `ModelPlayer`, which false-clears). Absolute
> thresholds. Arm C must also beat the Arm-B free `KClusterMCTSBot` overlay (**0.03**).
>
> **SECONDARY — strength non-inferiority vs golong@50k, WITH THE DISCLOSED CONFOUND.**
> The fixed loop biases Arm C **UP** (pinned bootstrap incumbent, fixed cadence, n=400 power,
> unloaded terminal eval — the original Arm C had none). The guard can **DETECT HARM** (Arm C
> materially weaker than golong50k → multi-cluster costs strength → do NOT scale, route strength
> to Gumbel) but **CANNOT ATTRIBUTE GAINS** to the encoding. Any Arm-C ≥ golong50k is confounded
> by the loop fixes and must NOT be read as "the encoding made it stronger."
>
> Effective-n caveat (§D-ARGMAX): dedupe byte-identical games + bootstrap CI over **distinct**
> games before trusting any strength gap.

**One variable between arms = ENCODING.** The W1/W2/W3/POWER changes are the corrected
instrument applied to make the re-run valid at all, not a second experimental variable.

---

## 5 · Abort gates (for the live 50k)

- **draw_rate** hard-abort: ≥0.55 for **3 consecutive** evals (min_step 0) → run ABORTED, close-out does NOT run. *Actively-breaking* signal.
- **grad_norm** hard-abort: 10.0.
- **Divergence signatures to watch (not auto-abort):** components_count collapse <15 (colony), off_window_rate rising >55% with low components (encoding not helping), 0 promotions with best_model.pt stuck at bootstrap (encoding weaker than bootstrap — the SECONDARY guard's harm signal).

---

## 6 · Next-theme teaser (routes the post-run session instantly)

Each is a **separate variable** — do NOT fold any into this re-run.

1. **Temp-schedule smoke (draw-gated).** The draw-rate gate (L9) is the standing risk on this
   line. A temp-schedule variation is a distinct knob; smoke it draw-gated AFTER the encoding
   variable is isolated. Falsified-register guard: **no across-training cosine temp**.
2. **Gumbel #1 / #4.** If the SECONDARY strength guard DETECTS HARM (Arm C < golong50k), strength
   routes to Gumbel root-action search — the runbook's pre-named fallback. #1/#4 are the
   candidate configs to smoke next.
3. **Value un-HELD on fixed-loop data.** §D-VALCEIL left value HELD (CEIL-HEADROOM, Idea-3
   reopened). With the loop now fixed, re-opening the value head on fixed-loop data is the next
   value-axis probe — gated on this re-run completing so the data regime is stable.
