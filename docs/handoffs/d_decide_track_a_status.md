# D-DECIDE — Track A status (paused 2026-06-24)

Paused mid-execution at operator request. Track A (decision inputs) done except A2 strength
curve (running in background). Track B (code bundle) NOT launched. Resume notes below.

## Live run — UNTOUCHED
- PID 1512427 on vast, `d1m_gumbel_m16_n150` (`longrun_v6_live2_ls_gumbel_m16`), started 94k → step
  ~200898 at pause. GPU 86%, healthy. All reads were single read-only rsyncs OFF vast. No commits.
  No compute on vast.

## A0 — DISCOVERY → **DATA-PARTIAL**
- Self-play recording ON. Current window = 417 decisive games (`games_2026-06-24.jsonl`) + 499
  per-game records. d1m-lineage earlier windows ROTATED — only today survives. Baselines available
  are CROSS-LINEAGE (pre-d1m Jun5-6, alt234k Jun21).
- **BUG FOUND:** `checkpoint_step` frozen at 120000 + `model_version` frozen at 1 for every current
  record → recorder step/version threading dead. No within-run step axis. (Feeds B3.)
- A1 instrument already existed: `scripts/d1m_replay_analyzer.py` (engine-pinned). A1 = validate+run.

## Data staged (laptop, gitignored): `reports/d_decide_2026-06-24/`
- `replays/games_2026-06-24.jsonl` (current, 419 recs) · `replays_vast/` (Jun5-6 pre-d1m) ·
  `games_current/` (499) · `games_alt234k/` (50) · `checkpoints/` (95k/120k/150k/175k/200k, 49MB ea) ·
  `inference_only.pt` · `a2_eval/strength_curve.log` (running).

## A1 — COHERENCE → **FLAT (coherence NOT broken)**  [Workflow wf_dfdc3fbd-7d6]
- trust_gate PASS (A1-validate 20/20 math primitives vs engine source; analyzer engine-faithful).
- confound_gate FAIL (cross-window delta invalidated; only baseline is cross-lineage; frozen step).
- 0/3 trip-gate legs fire. Current d1m (n=419): longest_line 6.01, longest_line_frac 0.263,
  n_components 1.03, colony_ext 0.002, forced_win_conversion 0.538.
  - n_components ~1.0 across ALL windows (no fragmentation; negatively corr w/ stone-count = not a
    length artifact, also no real signal).
  - longest_line saturated at 6 (decisive winners reach the 6-run by construction); `_fraction` is
    pure inverse stone-count artifact (Spearman −0.9997 vs stones).
  - colony co-signal INACTIVE everywhere (0.002 / 0.000 / 0.002).
  - Absolutes sit ENTIRELY BELOW golong terminal-break signature (n_comp 26→42, ll 9.3→8.4,
    conv 0.89→0.66).
- **Verdict = kill-gate CLEARANCE on degraded evidence**, NOT a confirmed positive trend. Selects the
  "coherence-not-broken" aggregation row → final cell gated on A2 + vs_boot@240k.
- Effective-n caveat for A2: `hash_moves` is byte-dedup only (over-counts vs symmetry-canonical);
  canonicalize over 6 hex reflections/rotations + color swap before trusting any "CI-resolved" gap.

## A2 — HONEST STRENGTH
- Harness BUILT + validated: `scripts/eval/gumbel_greedy_bot.py` (UNTRACKED; new file, nothing shared
  edited). Smoke 4/4 + 2-game SealBot smoke PASS on CUDA. Uses SAME engine planner (SH winner =
  `masked_argmax(gumbel+logits+completedQ)`, no temp, no PUCT-visit leak). Planner-rank divergence
  50% on openings (the PUCT+temp proxy materially mis-picks vs deploy head).
- **ready=FALSE — OPEN off-window fidelity gate:** Rust MCTSTree carries only a flat 362-logit vector
  → drops off-window candidates; a SealBot WR would *false-clear* an off-window defect by
  construction. A1 shows off-window play ~0%, so the strength *curve* (UP/FLAT/DOWN, in-window) is
  valid — label results in-window-only; do NOT use to clear an off-window defect.
- **RUNNING (background):** `a2_eval/strength_curve.log` — 5 checkpoints × 120 games vs SealBot,
  opening-plies 4, distinct seed/step. ETA ~2h from ~08:5x. On resume: parse per-step WR; bootstrap
  CI over DISTINCT games (cluster_bootstrap_ci), NOT raw count. UP/FLAT/DOWN read.
  - Background bash IDs at pause: A2 eval = `b9d70us6k`.

## Aggregation matrix (pending A2 + vs_boot@240k)
- A1 FLAT → coherence cleared as a kill cause. If A2 strength UP → CONTINUE/golong. If A2 FLAT/DOWN →
  plateau/regression on the STRENGTH axis (relaunch + Dirichlet-off, the one untested handicap).
  vs_boot@240k (scheduled, free) lands ~13h out; A2 dominates it on signal quality.

## TRACK B — NOT LAUNCHED (staged as Workflow 2)
Decoupled / restart-only; B3 gated on A1 defs (now validated). Buckets: B1 Dirichlet-off (gumbel
branch, byte-pure PUCT path), B2 CI half-draw, B3 structural engine-add (longest_line_fraction +
n_components emit + value-calibration + distinct-hash + FIX the frozen step/version axis; bench ≥73k),
B4 S6 single monitor, B5 stat deletes (incl. `strength_aggregate` dead code) + gumbel-regime guards +
sims_per_sec ~21× fix.

**EXECUTION-MODEL DECISION for resume (important):**
- Engine is `maturin`-installed into `.venv/` (main tree). A `git worktree` would NOT have `.venv`
  → worktree agents can't `import engine` without a full `maturin develop` (LTO) rebuild = thermal
  cutoff risk (per memory `dev-laptop-build-thermal-cutoff`) + slow.
- → Run Track B **serially in the MAIN tree** (engine pre-built, incremental rebuilds only): each
  bucket change → test → `git add -A && git diff HEAD > reports/d_decide_2026-06-24/patches/Bx.patch`
  → revert clean → next. Python buckets (B2/B4) run targeted `pytest` (NOT full `make test`, avoids
  Rust rebuild). Only B3 needs `make bench` (release LTO) — run it alone, GPU idle.
- Commit only on operator ask → produce patches + evidence for review, do not commit/merge.

## Hard invariants honored at pause
No vast contact for analysis · PID 1512427 untouched · no git commits (HEAD 11bd734) · no shared
source edited (only untracked `scripts/eval/gumbel_greedy_bot.py` added) · all analysis laptop-only.

## B3a ITEM-4 — frozen step/version axis DIAGNOSIS (diagnose-only, NOT blind-fixed)
The "frozen@120000 step + model_version@1" the structural emit would inherit is NOT dead wiring —
`set_step` IS live (pool.py:607-609 ← step_coordinator.py:417-418 startup-seed + eval_drain.py:113-114
promotion-refresh) and `model_version` bumps **on promotion only**. Frozen@120000 + mv@1 means ZERO
promotions have occurred since the startup seed: the frozen axis is a **promotion-cadence** artifact,
not broken threading. A per-train-step bump was rejected — it would over-tag records past the actual
inference weights (step_coordinator.py:858-862 already warns about this), so threading is left
UNCHANGED. Consequence for B3a: the new PER-PLAYER (winner) `longest_line_fraction` / `n_components`
emit has **no reliable step axis until promotions resume** — trend-over-training reads on these
metrics are unusable while the run sits at mv@1. Bucket the structural emit by wall-clock / game-count
(or by the `game_id_byte_hash` distinct-game effective-n) rather than checkpoint_step until promotions
re-fire; revisit once mv advances.

---

# RESUMED 2026-06-24 — A2 RESULT + DECISION + TRACK B COMPLETE

## A2 — HONEST STRENGTH → **FLAT (plateau, real not proxy-artifact)**
Deploy-matched gumbel-greedy vs SealBot, 100 DISTINCT games/checkpoint (copy_mult 1.00 — no
pseudo-replication; §D-ARGMAX trap avoided):

| step | 95k | 120k | 150k | 175k | 200k |
|---|---|---|---|---|---|
| WR | 45% | 45% | 51% | 39% | 48% |
| Wilson95 | 36–55 | 36–55 | 41–61 | 30–49 | 38–58 |

Linear slope ≈ 0.001 pp/1k (r=0.01); two-proportion z(95k vs 200k)=0.43 → NO significant change. All
CIs overlap. **FLAT plateau confirmed.** Caveat: SealBot = in-window instrument (off-window
false-clear by construction) — but A1 showed off-window play ~0%, so the in-window plateau is the
honest read. Model sits ~even with SealBot (45–51%), not improving across 105k steps.

## D-DECIDE AGGREGATION → **PLATEAU → relaunch same lineage + Dirichlet-off**
- A1 coherence = **FLAT** (kill-gate cleared; not broken).
- A2 honest strength = **FLAT** (real plateau, deploy-matched head, honest distinct-n CI).
- vs_boot@240k = scheduled/free, ~13h out — not required; A2 dominates it on signal quality and
  already resolves the strength axis.

Matrix cell (A1 FLAT, A2 FLAT) = Plateau at ~120k-anchor parity. **Coherence is NOT the kill cause;
strength has plateaued.** ⚠️ **CORRECTION (2026-06-24, post-investigation):** the dispatcher's
"relaunch + Dirichlet-off" lever is **FALSIFIED** — Dirichlet was OFF for the entire d1m run
(`dirichlet_enabled:false` since config creation ca621f2, 2026-06-21). B1 is a NO-OP for this variant
(hygiene for others). ALL dispatcher lever-premises were falsified (Dirichlet, 94k-anchor, §104 bake,
zero-promotions/staleness, LR-rise, n_sims, B2). Real picture: 2 promotions fired (60k wr=0.585, 120k
wr=0.62), then 150k/180k/210k failed the 0.55 gate on the **point estimate** (0.47/0.53/0.495) → genuine
parity stall vs the model's own 120k anchor. **The actual plan is in
`docs/handoffs/d_decide_relaunch_runbook.md`**: PRIMARY = a cheap eval-only self-anchored BT-MLE Elo
discriminator on the 5 banked checkpoints (decides measurement-ceiling vs true-stall; gates all restart
levers) BEFORE any restart. (No kill-now trigger; live run is a disposable comparator until the
relaunch is staged.)

## TRACK B — code bundle (patches-only, NO commits; HEAD still 11bd734)
Combined Python diff: `reports/d_decide_2026-06-24/patches/track_b_python_bundle.patch` (19 files,
+2452/−790). All review-verified (71 targeted tests green, no rebuild, zero hardcoded literals).
- **B2** CI half-draw → `p_hat=(W+0.5D)/n` Wilson (no double-trials, no knob). 18 tests. (NOTE: the
  illustrative tally W104/L80/D16 does NOT flip — too marginal at n=200; real flip case W112/L82/D6
  pinned instead. Root cause + fix correct.)
- **B4** single read-only `run_feed_reader.py` + thin `d1m_monitor.py` (golden-parity verified). 19 tests.
- **B3a** per-player `longest_line_fraction` + `n_components` emit + byte-hash `game_id`; threshold →
  `configs/monitoring.yaml` (hard-error). 16 tests. (B3b value-calibration + symmetry-canonical hash
  = DEFERRED, needs Rust + bench.)
- **B5** deleted 4 dead stats (`value_loss_main`, `model_version_range_size`, `aux_loss_rows`,
  `frac_fullsearch_in_batch`); `root_concentration`+cluster-trio gumbel-regime-guarded
  (`mcts_mean_depth` KEPT); `sims_per_sec` per-move fix (~21×). 18 tests. **`strength_aggregate`
  EXCLUDED from deletes — phantom producer but LIVE consumers → fix-in-place (B2 domain), NOT a delete.**
- **B1** Dirichlet-off on the Gumbel branch (Rust) — **DONE**. Deleted the gumbel-arm root Dirichlet
  (`if gumbel_mcts` block); PUCT `else` arm UNTOUCHED. Added S4_PUCT_DIRICHLET byte-purity golden
  (seed `0x5_4D_1_5EED`): `test_golden_s4_puct_dirichlet_unchanged` **PASSES** — PUCT root-noise
  f32::to_bits-IDENTICAL to HEAD (proves the deletion can't perturb the PUCT path). Full
  `cargo test -p engine` = **326 passed / 0 failed / 2 ignored** (dirichlet_parity + inv25 green).
  Bench SKIPPED (removes hot-path work — refactor-template skip). Entropy-drop behavior check DEFERRED
  (needs a `maturin develop` rebuild → thermal; checkpoint + gumbel_sims harness staged for a cool-host
  run). Patch → `patches/B1_dirichlet_off.patch` (3 engine files). Restart-only; live run already has
  `dirichlet_enabled:false` so unaffected.

## BUNDLE FINAL STATE (all of Track B complete)
Working tree: 19 tracked files modified (+705/−802) + 4 new untracked (`run_feed_reader.py`,
`test_run_feed_reader.py`, 2 fixtures). Patches: `patches/track_b_python_bundle.patch` (170KB, incl.
new files) + `patches/B1_dirichlet_off.patch` (9KB, engine). HEAD `11bd734`, nothing staged, NO commits.
Deferred to a later restart-bundle pass: B3b (Rust value-calibration + symmetry-canonical hash + bench),
B1 entropy-drop confirmation.

## On operator ask → commit
Bundle is staged in the working tree + as patches. On "commit it": branch off master, one commit per
bucket (conventional prefix, NO Co-Authored-By), `make test` green + (for B1) the cargo byte-purity
golden before landing.
