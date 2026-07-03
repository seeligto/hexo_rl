# D-RESUME-FORENSIC — Stage 0 report (read-only, no restart)

**Date:** 2026-06-24. **Scope:** interrogate the ~93k resume boundary of the live d1m run
(`longrun_v6_live2_ls_gumbel_m16`, PID 1512427 on vast) before any relaunch lever is committed.
Eval-only / log-forensic. **Live run NEVER touched** — all analysis on rsync'd-to-laptop assets.

## Headline verdict

**F1 CLEAN · F2 NON-EMPTY (eval-pipeline ONLY) · F3 STEP-at-resume (benign) · F4 NONE.**
The resume is a **red herring for the 150k stall** — the resume-artifact hypothesis is **FALSIFIED**
on the stall. Anchor wiring survived; **selfplay/training/search/LR config did NOT drift** and LR did
not re-peak. Two real discontinuities at 94k, both off the stall's critical path:
1. a **depth step** (3.05→2.55) — engine-state, not config; decoupled from the stall; benign.
2. a **silent EVAL-regime down-powering** — `best_checkpoint.n_games 400→200`, eval `model_sims
   128→64`, `bootstrap_anchor` monitor enabled, `expected_anchor_sha256` pin removed. This does NOT
   touch the learning loop, but it means the post-93k **"parity stall" was measured under a
   cheaper, noisier promotion eval** than run-1's.

**→ The learning-loop relaunch baseline is clean; the depth step does not block. BUT the live
"parity stall" reads (150/180/210k at 0.47/0.53/0.495, n=200/64-sims) are DOWN-POWERED and must be
discounted — which directly strengthens the case for Stage-1's high-power BT-MLE Elo ladder to
resolve measurement-ceiling-vs-true-stall. Proceed to Stage 1; restore eval power (n_games→400,
model_sims→128) and/or rely on the BT ladder.**

No kill-now trigger. Nothing changed on the host.

> **Correction note (post-audit):** an earlier draft of this report and the workflow's IMPL agent both
> reported F2=EMPTY. That was a profiler bug — the active log is run-1++run-2 **concatenated**, so the
> "first startup config" read run-1's twice. The fresh REVIEW + RED-TEAM agents independently caught
> it; a corrected three-way diff confirms NON-EMPTY (eval-pipeline only). Selfplay/training/search/LR
> are byte-identical across the boundary — the depth cliff is still NOT config-explained.

---

## Inputs pulled (rsync OFF vast → laptop, read-only)
- `logs/d1m/d1m_gumbel_m16_n150.jsonl` (349 MB) — **the active log is run-1 ++ run-2 CONCATENATED**
  (operator backed up run-1 to `*.run1_pre_restart.jsonl` but run-2 kept appending to the same file).
- `logs/d1m/d1m_gumbel_m16_n150.run1_pre_restart.jsonl` (147 MB) — run-1-only backup (0→94k).
- `events_3b853…jsonl` (run-2) + `events_7a155…jsonl` (run-1) per-game streams.
- `checkpoints/best_model.pt.provenance.json`.
- Read-only metadata ssh: vast HEAD = `e132e67` (Jun 22 21:50); live cmdline; checkpoint listing.

## Instrument-match (load-bearing)
- Live cmdline: `train.py --variant longrun_v6_live2_ls_gumbel_m16 --checkpoint
  checkpoints/checkpoint_00094000.pt --iterations 1000000 --run-name d1m_gumbel_m16_n150`.
- Run-2 started **2026-06-22 19:54** — BEFORE `e132e67` was committed (21:50). So the live process
  loaded its python from **~`fa48910`/`872787f`** (Jun 21 14:41).
- `resume_anchor_step_mismatch` is a **Python** `log.warning` (`anchor.py:481`), added in
  `8c0bec1`/`da7a51e` — both **ancestors of `fa48910`** ⇒ **WIRED in the live run (NOT BLIND).**
- The scheduler reads `T_max` from **config `total_steps`** (`trainer.py:432`), not `--iterations`.

---

## F1 — Resume boundary + anchor-desync ruling → **CLEAN**

Process-start timeline (from the active log):
| start | step | meaning |
|-------|------|---------|
| 2026-06-21 12:57 | 0 | launch attempt 1 (from 8300 bootstrap) — aborted ~4 min |
| 2026-06-21 13:01 | 0 | run-1 proper (from 8300 bootstrap), trains 0→94k |
| 2026-06-22 19:54 | **94000** | **run-2 = the single mid-training resume** (from `checkpoint_00094000`) |

- **Single mid-training resume at 94k.** The two Jun-21 launches are fresh-from-bootstrap at step 0
  (trainer.step=0=anchor 0 → no mismatch possible). `resume_anchor_step_mismatch` count: run-1=0,
  run-2=**1**.
- The mismatch fired **once**, at the resume: `{"trainer_step": 94000, "best_model_step": 60000}`.
  This is the **benign, legitimate "training continued past last promotion"** case the M2 docstring
  describes — at 94k the last *completed* promotion was 60k (the 90k eval was mid-flight,
  `best_arena` game 202/400, partial WR 0.58, when the restart killed it → never completed). Anchor =
  60k is **correct**, not stale/rolled-back. (60k promotion is real: run-1 `evaluation_round_complete`
  promoted=True, wr_best=0.585.)
- **120k promotion is correctly anchored:** the first run-2 eval (120k) scored the 120k candidate
  against the 60k anchor → wr_best=0.62 → promoted → `best_model.pt.provenance.json` = `{step:120000,
  run_id:3b853 (run-2), promoted:true}`. Only **two promotions ever**: 60k (run-1), 120k (run-2).
- **Post-150k evals correctly anchored to 120k:** 150k/180k/210k all promoted=False on the point
  estimate (0.47/0.53/0.495 < 0.55) vs the 120k best_model — a genuine parity self-compare, not a
  mis-anchor.

**⇒ Anchor-desync-as-stall-cause FALSIFIED.** The mismatch warning is informational; gate wiring
survived the resume. The 0.47/0.53/0.495 are a real self-compare → Stage 1.

---

## F2 — Effective-config diff across the resume boundary → **NON-EMPTY (eval-pipeline ONLY)**

Diffed the logged **startup `config` dict** (run-1 @12:57:10 vs run-2 @19:54:08, taking the *second*
config event from the concatenated active log). **173 → 179 keys; 10 differing keys — ALL under
`eval_pipeline.opponents`. Confirmed three independent ways** (REVIEW agent, RED-TEAM agent, corrected
local profiler) and corroborated by the live `evaluation_games_start` events.

| key | run-1 | run-2 | meaning |
|-----|-------|-------|---------|
| `opponents.best_checkpoint.n_games` | 400 | **200** | promotion eval games **halved** |
| `opponents.best_checkpoint.model_sims` | (default 128) | **64** | eval search sims **halved** |
| `opponents.best_checkpoint.opponent_sims` | — | 64 | (symmetric, no directional bias) |
| `opponents.bootstrap_anchor.enabled` | False | **True** | vs_boot monitor ON (n=50, stride4, 8300) |
| `opponents.bootstrap_anchor.{path,n_games,stride,model_sims,opponent_sims}` | — | added | (the monitor's params) |
| `gating.expected_anchor_sha256` | `ebf2ed39…` | **removed** | anchor-sha pin disabled (expected for resume) |

Live corroboration (`evaluation_games_start`, best_arena): run-1 `n_games=400, model_sims=128` →
run-2 `model_sims=64` (and the 120k promotion's `ci_best=[0.551,0.684]` is a wide n=200-class CI).

**Zero selfplay/training/search/LR keys differ** — `n_sims_full=150, n_sims_quick=150,
full_search_prob=1.0, fast_prob=0.0, total_steps=1000000, random_opening_plies=0, temp_min=0.5,
gumbel_m=16, gumbel_explore_moves=10, dirichlet_enabled=false, recency_weight=0.75, draw_value=-0.5,
ply_cap_value=0.0, buffer_schedule` all byte-identical. `--iterations 1000000` is a loop-count, not a
config key (total_steps=1M both sides).

**Interpretation per the dispatcher F2 pre-reg:** the changed keys are **NOT depth/game-length/
selfplay-affecting** ⇒ the depth cliff (F3) is **still not config-explained** (it is engine-state).
BUT the **eval/promotion MEASUREMENT regime drifted** at the resume — a "cheaper promotion eval"
(later committed as `872787f`) + a vs-bootstrap monitor (`e132e67`) were applied to the vast working
tree at relaunch and logged in run-2's effective config. **Consequence: the post-93k "parity stall"
(150/180/210k below the 0.55 gate) was scored at HALF the games and HALF the eval sims of run-1's
regime → wider CIs, weaker eval play → the stall is partly a measurement-power effect, not provably a
true strength wall.** This **re-weights Stage 1's read** and is a fresh, independent reason to run the
high-power BT-MLE Elo ladder rather than trust the live promotion-gate point estimates. The learning
loop itself is uncontaminated (selfplay/training clean).

---

## F3 — Depth-cliff timing → **STEP at resume (benign, engine-state — not config/LR/weights)**

Clean within-active-log 1k-bin `mcts_mean_depth`:
```
80–94k : 3.056 → 3.032   (flat; slight model-sharpening creep over 90–94k)
94–95k : 2.720           (straddle bin — the discontinuity)
95k →  : 2.544 → 2.552   (flat for the rest of the run, to 220k)
```
A **~0.49 STEP DOWN exactly at the 94k resume**, inside one continuous file (the pre-94k region of
the active log = run-1 @3.05; post-94k = run-2 @2.55). Not a smooth decline → **STEP_AT_RESUME.**

**Defusing facts:**
- **LR did NOT re-peak.** Empirical `lr` from `train_step`: 1.987e-3@60k → 1.967e-3@94k →
  1.966e-3@96k → 1.947e-3@120k → 1.857e-3@200k. Smooth cosine straight through the resume.
  `--iterations 1000000` did NOT reset the schedule (T_max=config 1M; trainer.step restored to 94k).
  The scariest F2/F3 hypothesis is **empirically dead.**
- **Depth is DECOUPLED from the stall:** depth is constant (~2.55) across BOTH the still-promoting
  region (95–120k, incl. the 120k promotion) AND the stalled region (150k+). The stall onset (150k)
  has no depth change. So the resume depth-step does not drive the stall.
- **Weights restored** (resume from 94k checkpoint), **main buffer restored full** (`buffer_restored
  positions=250000`).

**Named engine-state resets at the resume** (the candidate cause, none in config):
- `recent_buffer_init capacity=125000, recency_weight=0.75` → the **recency buffer reset to EMPTY**
  (recency-weighted sampling, weight 0.75, reshuffled at resume).
- `worker_pool_started n_workers=32` → 32 self-play workers restarted fresh; game counter → 0;
  self-play RNG reseeded.

**⇒ Unexplained-in-config but ENGINE-STATE-explained resume discontinuity on depth; benign given A1
FLAT + the downstream 120k promotion + depth-stall decoupling. Flag for the relaunch monitor (watch
depth re-step on the fresh-from-120k restart); do NOT block.**

---

## F4 — Second-boundary check at ~150k (stall onset) → **NONE**

- **Zero** startup/resumed/process-start markers after 2026-06-22 20:00 → **no second restart**
  between 95k and 220k.
- No eval-harness/eval-config change; buffer-capacity step is scheduled at 300k (none earlier — the
  schedule's only pre-300k entry is step 0); LR is a smooth single cosine (no crossing).
- **However — a strong Stage-1 LEAD at the stall onset:** `value_spread_alert` / repeated
  **`SOFT-ABORT "dual-bank colony capture — discriminator degrading"`** fire densely from ~140.5k
  (t3_spread → −0.47, −0.53). This is a **value-head discriminator-degradation signature right at the
  stall onset** — a training DYNAMIC, not a resume artifact (consistent with the runbook's
  colony-attractor kill-gate concern, §157). It does NOT re-scope this forensic but is a high-value
  pointer for Stage 1's parity-stall investigation.

**⇒ The 150k stall is genuine training/eval dynamics → Stage 1 owns it. Resume forensic does not
explain the stall (hypothesis cleanly falsified on the stall).**

---

## Decision impact

| F1 | F2 | F3 / F4 | Outcome |
|----|----|---------|---------|
| CLEAN | NON-EMPTY (eval-only, **not** a depth key) | STEP at resume (benign), F4 NONE | **Learning-loop relaunch baseline is clean** (selfplay/training/LR/anchor all survived the resume). The depth step is a benign engine-state resume blip (recency-buffer/worker reset), decoupled from the stall. The eval-pipeline drift does NOT explain the cliff but **down-powers the live stall measurement** → re-weight Stage 1. **Stall is Stage 1.** |

- **No fix-and-resume needed** (F1 not MISMATCH-bad → anchor wiring intact, stall metric not corrupted
  by desync).
- **No selfplay/depth key to restore** (F2 changes are eval-only); the depth cliff is engine-state, not
  a dropped key.
- **DO restore eval power on the relaunch** — `best_checkpoint.n_games 400` and `model_sims 128` (or
  rely on the Stage-1 BT ladder), because the live 150/180/210k "parity stall" reads are n=200/64-sims
  and therefore noisier than the runbook's framing assumed.
- **Proceed to Stage 1** — the self-anchored BT-MLE Elo ladder on the banked checkpoints, exactly as
  the relaunch runbook planned. The down-powered live eval (F2) is a fresh, independent reason the BT
  ladder is the right GO gauge. Add the depth-re-step watch + the value_spread/colony-capture watch to
  the relaunch monitor.

## REVIEW + RED-TEAM (fresh independent agents) — audit outcome

Workflow `wf_2e2ecfa4-8c2`, 3 agents (Sonnet 4.6), independent re-derivation on the same local logs.

- **F1 CLEAN — CONFIRMED ×3.** RED-TEAM additionally pinned the `anchor_identity` event at 19:54:50:
  `step=60000, sha256=bb7e986f` (≠ preD1M bootstrap sha `ebf2ed39`) → best_model.pt was the **60k
  checkpoint, not rolled back** to the bootstrap. 90k eval produced no `evaluation_round_complete` (run-1
  killed mid-eval) → no hidden 90k result lost. 120k `ci_best=[0.551,0.684]` lower bound 0.551 > 0.55 →
  promotion valid even at the reduced n=200.
- **F2 EMPTY → OVERTURNED to NON-EMPTY (eval-only).** Both audit agents independently found the 10
  eval-pipeline diffs; my corrected profiler agrees. This is the audit's load-bearing catch.
- **F3 STEP_AT_RESUME — CONFIRMED ×3, not a binning artifact.** Timestamp-separated: last run-1
  `train_step_summary` step=94590 @19:52:13 depth=**3.025**; first run-2 step=94010 @19:55:37 depth=
  **2.528** (after `buffer_warmup_ended` @19:55:22). Δ=−0.497, immediate. `mcts_mean_depth` emitter
  confirmed at `fa48910:events.py L258+L308`, sourced from the **selfplay** pool rstats (not eval) — so
  the cliff is a self-play search change. LR strictly monotone (no re-peak), T_max not reset by
  `--iterations`.
- **F4 NONE — CONFIRMED ×3.** RED-TEAM enumerated the full restart-marker set the live rev emits
  (`resumed`, `buffer_restored`, `buffer_warmup_ended`, `worker_pool_started`, `selfplay_pool_started`,
  `seeded`, `cuda_warmup_start/done`, `inference_trace_compiled`) — each appears **exactly 3×** (pids
  1232448 + 1232916 + 1512430 = the three process starts), no 4th near 150k. `viewer_engine_reloaded`
  at 14:20/16:01 are promotion-triggered, not restarts. (IMPL's marker list was incomplete but its
  NONE conclusion held.)

**Net:** F1/F3/F4 verdicts survive adversarial attack; F2 corrected EMPTY→NON-EMPTY(eval-only). The
headline (resume not the stall cause; depth cliff benign; proceed to Stage 1) stands, now with the
added, decision-relevant caveat that the live promotion eval was down-powered at the resume.

### Notes for the relaunch (carry forward)
1. The active log is run-1++run-2 concatenated — any future log parser must not double-count the
   pre-94k region (the run1 backup duplicates it).
2. The restart interrupted a likely 90k promotion (partial WR 0.58 at game 202/400) — a one-time cost
   of the kill, not a defect.
3. Stage-1 lead: value-discriminator degradation / dual-bank colony-capture SOFT-ABORTs cluster at
   140–150k, coincident with the stall onset — the most concrete mechanistic pointer the log offers.
4. **Eval regime drifted at 94k (F2):** restore `best_checkpoint.n_games 400` + eval `model_sims 128`
   on the relaunch (or rely on the Stage-1 BT ladder), and treat the live 150/180/210k point estimates
   as down-powered (n=200/64-sims). The selfplay/training loop is uncontaminated.
