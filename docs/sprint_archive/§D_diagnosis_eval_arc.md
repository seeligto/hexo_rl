# §D_diagnosis_eval_arc

_Relocated from `docs/07_PHASE4_SPRINT_LOG.md` (D-DOCS-DEBLOAT split, 2026-06-23). Scope: §P6/§O1/§PRELONG + §D-WALLCAUSATION…§D-STRENGTHAXIS diagnosis & eval-foundation arc. Verbatim; falsified-register rows also consolidated into the sprint-log index register section._

## §P6 — v6_live2 flatness re-measure + Hammerhead NNUE 2nd opponent — 2026-06-01

Dispatcher for the post-fix flatness verdict. The 3 correctness fixes (DRAW-MASK
value-target mask on ply-capped rows, CF-5 `aux_opp_reply_weight: 0.0`, CF-6
no-bug) are already landed (`3291ebe`); this is the wire→run→eval→verdict layer.
Report + pre-registered verdict: `reports/investigations/v6_live2_flatness_30k_20260601.md`.

**STEP 0 (recorded, NOT changed):** `v6_live2` **k_max=1 (single-cluster)**. The
§169 K-cluster min/max 12pp-argmax lead lives in the separate v6w25/v7mw k_max=8
line, never ported. This is **line-divergence / parsimony, NOT a silent
regression** — the v6→v6tp→v6_live2 production lineage was always k_max=1; the
H-PLANE fix touched only the plane set. The prior 40%-greedy baseline shares
k_max=1, so the flatness delta is uncontaminated. K-cluster restore = a separate
single-variable arm w/ its own matched baseline, flagged for the rethink session.

**STEP 1 DONE — Hammerhead NNUE wired as eval-only 2nd opponent.** Vendored
`vendor/bots/hammerhead` (submodule of github.com/seeligto/hammerhead) + root
`Cargo.toml` `workspace.exclude`; built into `.venv` (maturin release). The
engine `Bot` is stateful/incremental-from-origin (hard `MustStartAtOrigin`, no
set-position API) and hexo_rl keeps no ordered history → `NnueBot(BotProtocol)`
**diff-syncs** the board each move + applies one **translation** to the origin.
Correctness rests on translation-invariance (Hammerhead's static search depends
only on occupied-set + side-to-move, not replay order) so the exact opening is
irrelevant; ranges compatible (hh `max_piece_distance=8` ≥ r5; suggests within
r2 ≤ r5). Reused the SealBot path (no bespoke): `Evaluator.evaluate_vs_nnue`,
`opponent_runners._run_nnue` (appended LAST → byte-for-byte insert order),
`EvalRoundResult.wr_nnue`, `eval_pipeline` cfg/pid, `configs/eval.yaml`
`opponents.nnue` (**default OFF** — keeps the in-run 30k single-variable),
`run_sealbot_eval.py --opponent {sealbot,nnue}`. **Zero hot-path touch** (no
bench), pinned by `test_nnue_eval_path_only.py`. TDD `test_nnue_bot.py` 11/11
green (incremental + cold-start sync == board, non-origin opening, full game,
auto-reset; caught a real premature-origin-lock bug). End-to-end smoke on the
real baseline 30k checkpoint PASS. **Second-opponent interpretation:** general
strength → HeXO_vs_NNUE ≫ HeXO_vs_SealBot; SealBot-overfit → ≈/< despite NNUE
being weaker.

**STEP 2 launch-ready (operator/vast ~13hr):** `train.py --checkpoint
checkpoints/bootstrap_model_v6_live2.pt --variant v6_live2_smoke --iterations
30000`. Single variable vs the prior 30k = the 3 fixes only (verified config
delta; cosine OFF L9; stride5_p90 + grad-norm + SealBot-WR hard-aborts live).

**STEP 3/4 pending the run:** eval the post-fix 30k dual-opponent × dual-temp
(SealBot/NNUE, T=0/T=0.5, n=100, mcts-128) → pre-registered flatness bins
(BUGS-WERE-IT / TUNING-NEEDED / REGRESSION) vs baseline greedy 40% / sampled
~0.20 / ~2× gap. Canary = spam (stride5); V_spread DEMOTED. Not committed
(operator-gated).

**Local NNUE baseline matches (2026-06-01, operator request, mcts-128 T=0 n=20,
NNUE 500ms):** bootstrap v6_live2 **70%** [48,86] (14/6); pre-fix 30k **80%**
[58,92] (16/4) — both ~2× their SealBot WR (34% / 40%) ⇒ **general-strength
direction** (NNUE a genuine lower rung); wiring validated end-to-end.

**Post-fix 30k smoke RUNNING on vast (RTX 5080, 2026-06-01), NNUE ON in-run
(n=50).** Launched once on the laptop by mistake (operator: smoke belongs on
vast) → killed. vast was on pre-Phase-6 `a604804` (no DRAW-MASK engine, no NNUE)
→ rsync'd source, `make build` rebuilt engine WITH DRAW-MASK (cargo-PATH gotcha
[[feedback_vast_bench_scripting]] hit + fixed), built hammerhead for in-run NNUE,
verified engine+nnue import. §149 buffer trap pre-cleaned (archived stale
buffer+ckpts). Confirmed clean: encoding v6_live2, buffer_size_before_corpus_load
=0, step 20 healthy (loss 3.07, grad_norm 1.26), no early_game_probe_failed,
no abort. tmux `v6l2`, log `/tmp/v6l2_smoke.log`. ~13hr est; 1hr health-watch
running.

**Probe fix — early_game_probe for the 4-plane model (resolver, no hardcode).**
The in-run probe failed `expected 4 channels, got 8`: a stale version-matched
8-plane `early_game_probe_v6_live2_v1.npz` loaded blindly (the BUILDER was
encoding-aware, the LOADER had no channel check). Fixed TDD: `load_fixture`
validates the loaded plane count against `resolve_arch(encoding_name).in_channels`
and regenerates a stale fixture; `compute()` gained a model-channel guard.
On-disk fixture auto-healed 8→4. Probe-only (never affected training/verdict);
`tests/test_early_game_probe_encoding.py` +2.

**VERDICT (2026-06-02) — run completed 30k on vast (exit 134 = benign teardown
abort post-save; model safe). Bin = TUNING-NEEDED + GENERAL strength.**
- T=0 greedy n=100 standalone: **SealBot 0.32** [0.24,0.42] vs baseline greedy
  **0.40** (within noise, NOT improved); **NNUE 0.77** [0.68,0.84]. In-run sampled
  (5–25k): SealBot 0.10/0.25/0.18/0.15/0.18, NNUE 0.30/0.56/0.54/0.64/0.50.
- **Flatness PERSISTS** (greedy/sampled ≈ 1.8×, the 2× gap survives) → NOT
  BUGS-WERE-IT. Threat/value-head probe healthy + climbing across ckpts
  (C1 3.81→4.88→5.35→5.47, all C1–C3 PASS) ⇒ DRAW-MASK/CF-5 fixed the VALUE side;
  residual is **policy-side flatness, genuine**. Next lever per memo = **O1
  forced-win one-hot POLICY target** (rethink session). Not a regression (within
  baseline noise, spam clean, no promotion any round).
- **Second-opponent RESOLVED = GENERAL strength** (not SealBot-specific overfit):
  HeXO vs NNUE 0.77 ≫ vs SealBot 0.32, and NNUE is the weaker bot ⇒ the SealBot
  ceiling is a real general strength gap. KrakenBot blocker closed via NNUE.
- **NNUE wrapper cold-start bug** (the `(2,-10)` OutOfRange that crashed the first
  NNUE T0 eval) fixed: within-range replay filter + full-reset retry + legal
  fallback; resilience test added; re-run clean (4/5000 fallback moves).

**Parallelization (operator asked; DECIDED — not implemented):** in-run eval is
already async daemon-thread + skip-if-busy, so slow eval never blocks training;
**leave it sequential** — parallelizing it contends for the GPU (eval MCTS shares
the GPU-forward-bound selfplay path) = breaks training. Standalone
`run_sealbot_eval.py` is safely parallelizable (`--jobs N` subprocess-sharding,
each shard the proven sequential path, deterministic merge) — biggest payoff is
STEP-3 4×n=100 on vast; deferred (GPU busy with the 30k smoke, can't verify
locally). Future item; design captured in the report.

## §O1 — flatness diagnostics (D1-D4) + forced-win one-hot POLICY target — 2026-06-02

Report: `reports/investigations/diagnostics_o1_20260602.md`. Resolved the open
§P6 flatness diagnostics on the post-fix v6_live2 30k (vast run `f8aaf414`,
artifacts pulled to `/tmp/hexo_postfix/`), then implemented O1 gated on D1.

**Diagnostics (CPU, read-only on the existing run):**
- **D1 = HEALTHY** (gate for O1). Value loss = binary CE on scalar win-prob
  (`losses.py:75-102`, floor log(2)=0.693). Stratified by plies-to-terminal on
  371 real-terminal games: 0-4 bucket **0.271** ≪ floor (mean logit +1.0),
  5-12 0.519, 13-30 0.690, 31+ 0.762; overall 0.684 dragged to floor by genuinely
  uncertain deep-early positions. Value head reads near-won positions → bottleneck
  is policy. Independent re-bucket reviewer AGREED (exact match).
- **D2 = NATURAL-LENGTH dominant (70.3%).** 38/128 caps weak-conversion (depth-1
  missed win / open-5), 90/128 natural-length — but 67/90 carry unconverted open-4
  (2-move-win) pressure (only 23 truly balanced). O1 depth-2 reaches the open-4
  subset; residual length = cap-height (arch session). Spot-trace: model walked
  past a completed-6 win ~40 plies.
- **D3 = ALREADY-RECENTERED → SKIP.** Window = bbox centroid recomputed every
  `apply_move` (`core.rs:345-351,480-494`); first-move-(0,0) pin is a no-op; NOT a
  flatness lever.
- **D4 = soft-but-HEALTHY.** disagreement flat ~0.68 (lagging, not collapsed),
  early-game entropy ~3.15 stable, losses falling (value best 0.42), aux null
  (CF-5 confirmed), grad ~1.08. One self-corrected V_spread transient (s15-22k
  negative → +0.349, §S181 signature, didn't propagate).

Working theory CONFIRMED (soft policy → indecisive finishing → value parks at
log(2) as genuine uncertainty, one mechanism) → lever = policy sharpening (O1),
NOT value-head reroute.

**O1 LANDED (uncommitted working tree, NOT pushed):** `Board::forced_win_move`
(depth-1 + depth-2, turn-phase via `moves_remaining`, NOT ply parity) +
`first_winning_move` (`moves.rs`); `apply_forced_win_one_hot` (`records.rs`)
overrides the training policy target to a (near-)one-hot at the proven winning
move; the hardened row is forced full-search so PCR's `full_search_mask` can't drop
it. Config `forced_win_policy_{enabled,depth,weight}` as `#[pyo3(get,set)]`
(default OFF/2/1.0 — INV19/25/26 byte-equivalence untouched) → `pool.py` ←
`configs/selfplay.yaml`. INV pin `inv_o1_forced_win_one_hot_wiring.rs`
(source-presence) + behavioral unit tests + Python prune-survival test. **305 Rust
+ 4 Python tests green**; soundness reviewed **5/5, no silent drop** (one
latent-unreachable fast-game branch documented). Downstream survival proven at
every stage (aggregate/rotate/augment/prune/loss). Default-OFF = byte-identical.

**Stage C:** `configs/eval.yaml` `eval_profiles` (cheap n=50/10k SealBot-only +
milestone n=200/20k dual-opponent dual-temp) — config-only, declarative, base
defaults unchanged, wiring flagged for arch session. k_max NOT changed.

**Bench gate (vast 5080, n=5, make bench):** baseline MCTS 112,135 sim/s (≫73k) /
worker 96,043 pos/hr / all PASS; post-O1 (default OFF) — see report (no hot-path
touch: O1 fires once per move at target extraction, not per-sim).

**NOT done this batch:** O1 validation smoke (next GPU run, pre-registered: O1 ON
must narrow greedy/sampled gap 1.8×→~1.3× on a matched 30k). Handoff →
claude.ai arch-finalization session (O1 ready; cap-height; K-cluster restore;
eval_profiles wiring).

## §PRELONG — pre-long-run triage (T1-T3 → window-WIDEN arm) — 2026-06-04

Report: `reports/investigations/prelong_triage_2026-06-04.md`. Separated the causes
the "flatness" frame conflated after the O1 smoke dissolved the 1.8× premise.
Local artifacts (vast 30k gone); the perception geometry is encoding-invariant so the
local **pre-fix** v6_live2 checkpoints answer the routing question. Scripts in
`scripts/structural_diagnosis/prelong_triage_*.py`.

- **Mechanism (proven):** v6_live2 = `k_max=1` single 19×19 window. A win cell at
  chebyshev > 9 from the bbox-centroid centre → `to_flat=usize::MAX` → **no policy
  logit** (362=19×19+pass, prior 0) AND truncated out of the MCTS child array on
  spread boards (`backup.rs` MAX_CHILDREN 192) → **genuinely unreachable**.
- **T1 = DECISIVE GAME** (qualitative). SealBot self-play (global eval) misses **0/69**
  forced wins, caps low at the 150 cap where HeXO caps ~12% (noise-free) / 25.7% (run).
  *Red-team caught* the SealBot leg was deterministic/seed-blind (n=120→~9 lines, gap
  not significant); re-run with random openings (27 shapes, cap ~5%) restores it.
  Human corpus is six-in-a-row-filtered ⇒ length only (median 49), not draw rate.
- **T2 (keystone) = OFF-WINDOW.** Trained 30k (157 misses): **94.9% off-window**
  (independent re-derive 94.6%), lift **2.02×**, 97% on ≥2-cluster boards, far-cheb
  median 11; model takes **every in-window immediate win** (0/18 in-window depth-1);
  off-window fraction **training-invariant** (bootstrap 97%→30k 95%). All depth-2 =
  within-turn open/closed-4→6 (operator note); detector verified genuine (turn-phase
  guard, real check_win, dedup).
- **T3 = FLAT/structural** (cap 14.7/13.3/12.2% n=90, statistically flat); off-window
  fraction flat-high → the residual is the wall, not training horizon. Off-window is
  the dominant **miss** mode but a **minority cap** cause (24% of caps had no
  forced-miss; 64% of forced-miss games won).
- **ROUTING (post-red-team):** the binding lever is window **RADIUS, not count** —
  every off-window cell is cheb ≤ 11, so a **23×23 single-window WIDEN covers 100%**
  (width oracle: 19×19=5%, 23×23=100%). Stand up a **window-widen arm (primary) vs
  K-cluster (comparator)** BEFORE the 300k — NOT K-cluster-first (it failed 3× per
  §174 + doesn't fix single-cluster off-window misses + v8-25×25 argmax-dilution).
  Do NOT go-long-blind on `k_max=1`. O1 stays banked (P4; ~5% in-window depth-2 only).
- **Adversarial review:** 10-agent workflow (4 lenses CONFIRM mechanism, 3 red-team
  attacks LANDED → routing corrected not buried, 2 stat-gap agents → ranked
  next-diagnostics incl. multi-window value oracle, off-window-DEFENSE on losses,
  value-calibration-at-cap). Verbatim: reports/investigations/prelong_triage_data/verify_results.json.
- **Process fixes (working tree, NOT committed):** P1 eval-profile opening-plies pin
  (kills the greedy@4-vs-sampled@0 artifact class) + P2 cadence (`configs/eval.yaml`);
  P3 V_spread canary CORE met (T3 bank fires on v6_live2, `t3_spread=0.528`; alt-bank
  4-plane rebuild deferred); P4 O1 shelved+armed comment (`configs/selfplay.yaml`);
  P5 `--no-web-dashboard` flag (`train.py`/`lifecycle.py`, kills exit-134 teardown).
  Owed (operator): commit the hammerhead/NNUE stack; CONFIG-3.

## §PRELONG-CLOSE — v6_live2 + cleanup milestone committed, FF-merged, tagged — 2026-06-04

Closed the v6_live2-adopt + cleanup milestone: the §PRELONG process fixes + the
Hammerhead/NNUE eval stack are committed on `phase4.5/v6_live2`, FF-merged to
`master`, and tagged. This is the clean base for the perception arm — which is
**NOT** started here (no 23×23 / window-widen / K-cluster code lands in this
milestone; routing call deferred to the arch-finalization session per §PRELONG).

**Owed-work commit wave (5 commits, each its own concern):**

| commit | what |
|---|---|
| `65d6b30` feat(eval) | Hammerhead minimax+NNUE 2nd ladder opponent — submodule `vendor/bots/hammerhead`, Cargo `workspace.exclude`, `NnueBot` + eval wiring (`evaluator`/`opponent_runners`/`eval_pipeline`/`result_types`), `run_sealbot_eval.py --opponent {sealbot,nnue}`, `eval.yaml`/`v6_live2_smoke` cfg. **Eval-path only** (grep-pinned `test_nnue_eval_path_only.py`); zero hot-path touch → no bench gate. |
| `fa4850c` feat(eval) | P1/P2 canonical prolonged-run `eval_profiles` (cheap/milestone) — opening-plies pinned IDENTICALLY across temps (kills the greedy@4-vs-sampled@0 O1-smoke artifact class) + 12.5k cadence. Config-only (SoT); selection hook deferred. |
| `0c7474a` docs(selfplay) | P4 O1 forced-win policy = SHELVED + ARMED comment (value unchanged, stays `false`); reactive lever IFF the long-run V_spread excursion fails to self-correct at 200-300k. |
| `10922b6` feat(train) | P5 `--no-web-dashboard` — suppress only the Flask-SocketIO dashboard (kills the benign exit-134 SIGABRT teardown), keep the terminal dashboard. + gate test. |
| `c7ca6ef` fix(monitoring) | early_game_probe 8→4-plane auto-heal (resolver-derived `in_channels`, regenerate stale fixture) + compute() channel-mismatch skip + canonical v6_live2/v7full fixtures. The v6_live2-adoption monitoring fix. |

**P3 V_spread canary = NO-OP here.** The core T3-bank fix already landed on
`master` (`321b136`, unmodified in the working tree). Alt-bank 4-plane rebuild
stays DEFERRED (operator follow-up; the T3 bank alone arms the long-run canary).

**Verification (pre-merge gate):**
- `make test` GREEN — **1768 passed, 54 skipped, 1 xpassed, 0 failed** (Rust +
  Python). The v6w25-roundtrip PYTHONHASHSEED flake did not fire (clean run).
- `make bench` n=5 laptop (`n_workers=14`) — **all 10 targets PASS**. MCTS (CPU,
  no NN) median **89,891 sim/s** (range 86.9k-90.5k, IQR ±1,109) ≫ ≥73,000 floor;
  flat vs the §S183 baseline 88,006 → O1 (default-OFF) + B1 cold-path = no
  regression, as expected (neither touches the hot path).
- Fresh-context review: O1 `forced_win_policy_enabled: false` in the merged
  config; hammerhead referenced only on the eval path; no perception-arm/23×23
  leak in the owed commits.

**Merge + tag:** FF-merge `phase4.5/v6_live2` → `master`; tag
**`v6_live2-adopt-close`** at the last CODE commit `c7ca6ef` (not this
docs/sprint-log commit, per archive-tag convention). Pushed `master` + tag.

**Left uncommitted (out of scope — investigation/perception-arm tooling,
NOT this milestone):** `scripts/structural_diagnosis/*` (§PRELONG triage
probes), `investigation/`, `docs/compression/`, `scripts/{export_selfplay_games,
generate_demo_replays,transfer_v6_to_v6w25,update_manifest}.py`,
`scripts/s174_bootstrap_fix_run.sh`. The §PRELONG triage report itself is
gitignored under `reports/`.

**Next:** the perception arm (23×23 single-window WIDEN primary vs K-cluster
comparator) on this clean `master`, per the §PRELONG routing — separate chapter.

## §D-WALLCAUSATION — does the off-window wall CAUSE colony? + recorder/tripwire fixes — 2026-06-05

Branch `phase4.5/wallcausation_fixes` (Phase B, not committed pending operator).
Report `reports/investigations/wallcausation_2026-06-04.md`; go-long validation
`reports/investigations/golong_validation_v2.md`.

**Phase A — causation: INCONCLUSIVE, leaning AGAINST.** Regenerated single-window
`ModelPlayer` self-play from archived colony checkpoints (s180b 10k–53.5k, s179
10k–60k, + healthy v6live2) → forced-win detector both sides → correlated off-window
incidence vs recorded colony signal (`metadata.json eval_trajectory`). s180b corr
0.96 but off-window is **coincident-not-leading** (flat 0–2% through 10k–40k while
colony_anchor climbs 36→43 / elo 422→330; spikes only at the 50k hard-fail, CI[13,32]
disjoint from earlier [0,7.7]); healthy v6live2 carries 11% off-window > colony
checkpoints; s179-60k (colony 77%) below its own 50k peak. **Decisive caveat
(structural):** `ModelPlayer` drops off-window at selection (`evaluator.py:113`) →
`max_spread≤18` (window diameter) → it CANNOT reach the training-self-play spread-306
regime where the wall fires (§OFFWINDOW 25.6%). Instrument is asymmetrically biased
against detecting causation → can't claim clean FALSIFIED; lag+non-monotonicity kill
FIRMED. → **wall→colony NOT firmed → D-SCATTER NOT triggered.** Decisive follow-up =
Rust worker_loop self-play regen (off-window-searchable), operator-gated (§A.5).

**Phase B — recorder + tripwire fixes (3, tested, suite green; commit pending).**
(1) Live forced-win tripwire was INERT: `_emit_forced_win_trend` passed
`mover_side=0`, never matches engine {1,−1} → n=0. Fixed: fold both engine sides via
`forced_win_detector.engine_player_sides(enc)` (zero-literal, derived from a fresh
board); `update_trend_from_file_incremental` generalized to `int|Sequence[int]`.
(2) `checkpoint_step` stuck at 0 (every historical record): `pool.update_checkpoint_step`
had zero callers. Fixed: seed at StepCoordinator init + refresh at the promotion sync
(`eval_drain.py`, the ONLY `sync_inference_weights` site) — NOT per train step
(red-team caught that over-attribution: self-play runs the inference model, swapped only
on promotion). (3) Moveset: VERIFIED already-correct on master (`move_history.push`
unconditional, inner.rs:968) — empty historical records are legacy. Bench-exempt
(Python-only, off hot path). +6 tests in test_forced_win_detector/test_step_coordinator,
new test_eval_drain.py.

**Phase C — mechanism + precedent (confirmed on master).** Off-window drop is 3-layer:
target drop `records.rs:62` (via `usize::MAX` from `core.rs:409`) + uniform 1/n_ch prior
`backup.rs:112` + 192-child cap `backup.rs:105` (binding limiter). records is the SOLE
TRAINING-TARGET drop site (eval/ModelPlayer drop at inference only). **§S181 did NOT pin
the single-window action space** — it pinned value-head discrimination collapse (H6) +
permissive value-head arch (H7); off-window is a DISTINCT (policy/action-target)
mechanism → the "no-reopen colony" fence basis is WEAKER than assumed; the open question
(unsettled) is whether off-window is the upstream cause of H6.

**Phase D — go-long validation: GO-WITH-NOTE.** Single-window go-long READY iff the
fixed tripwire is a HARD gate (n=game-sides semantics). All 8 standing checks PASS; #9
wall-honesty / #11 tripwire-live (n>0) / #12 causation-not-firmed satisfied; no blockers.

**Lessons.** L: `ModelPlayer` (deploy/eval path) is spread-capped at the action window
(`evaluator.py:113`) and reproduces neither the spread-306 wall regime nor the
colony/draw regime — both are training-self-play (Rust worker) phenomena; use the Rust
path, not ModelPlayer, to study training-time pathologies. L: the recorder's
`checkpoint_step` must be tagged at the inference weight-sync (promotion), not per train
step — self-play uses the promoted model, not the live trainer model.

## §D-MULTICLUSTER S-PRE — §174 kill-gate precheck (CONDITIONALLY CLEAN) — 2026-06-06

**Goal.** Predict the §174 argmax-degeneracy for the legal-set ACTION space CHEAPLY,
BEFORE paying S0 (the Rust refactor). The handoff brief specced a static argmax-distribution
probe; CORRECTED — §174 is a bootstrap→selfplay HANDOFF collapse (policy entropy ~2.4 nats,
explicitly "not the lever"), so a static probe would false-CLEAR. Faithful instrument =
MCTS viability, Python-runnable via `KClusterMCTSBot` (no Rust; §173 multi-window machinery).

**Method.** A/B isolating ONLY the off-window drop (`records.rs:62`): CONTROL = single-window
362 (`to_flat ≥ n_actions-1` dropped) vs TREATMENT = legal-set no-drop, same v6_live2 head,
same Python K-cluster MCTS. The self-play A/B was UNDER-POWERED — sims-64 self-play stays
K=1 (0% K>1 over 33,715 expansions, all models) → pivoted to MOVE-AGREEMENT on REAL spread
positions (K≥2 AND off-window present), the regime where the drop can bite.

**Result.** Move-agreement 0.85 (30k) / 0.79 (54.5k); TREATMENT picks an off-window move
the single-window CONTROL cannot reach ~9% (checkpoint-independent). No collapse / no
degeneracy → §174's "single-window-tuned policy breaks under multi-cluster scatter-max" is
REFUTED for the 362-multiwindow legal-set. Real self-play geometry IS multi-window (K up to
7; 79% of positions have ≥1 off-window legal move). Code-faithfulness PASS (5/5); red-team +
completeness reviewers applied (3 agents).

**Conditions (all must hold for GO):** (1) **S1 MANDATORY** — eval-only clears the *argmax*
mechanism, NOT the training-loop handoff (value-head over-fit, §174 e50 mode); §174 failed
that 3× → S1 still >50%-likely to fail. (2) **S0 MUST lock 19×19 / 362-head + multi-window**
— NOT a single larger head (that reintroduces the §174-626 risk; registry has no multi-window
362 encoding yet). (3) efficacy + off-window-pick QUALITY untested → S3 adversarial gate
(`exploit_probe.py` ≤ 0.06), NEVER vs-bot WR.

**Verdict: CONDITIONALLY CLEAN** — §174 *argmax* kill-gate CLEARED → the multi-cluster line
is NOT killed; GO/no-go on S0 is now an EFFICACY + COST + S1-risk call, not a §174-argmax call.

**Lessons.** L: a cheap pre-S0 §174 prediction IS possible via eval-path multi-window MCTS,
but it clears only the argmax mechanism — the bootstrap→selfplay→training handoff is
irreducibly an S1 (post-S0) gate (consistent with the existing "ModelPlayer eval-path ≠
training self-play" lesson; the off-window drop is 3-layer in training — `records.rs:62` +
`backup.rs:112` uniform prior + `backup.rs:105` 192-child cap — the eval probe models only
the first). L: sims-64 Python self-play never spreads (K=1) regardless of model; use
move-agreement on REAL replay positions, not self-play, to exercise the multi-window regime
cheaply.

Full: `reports/investigations/multicluster_s174_precheck_2026-06-06.md`.
Probe: `scripts/multicluster_s174_precheck.py` + `_moveagreement.py` + `_measure_k.py`.

## §D-FRAGILITY — why do the long runs break + spread? (A/B/C diagnosis) — 2026-06-07

Verdict: **A (LR/training-stability transient); B (off-window value-corruption) DECISIVELY
FALSIFIED; MODERATE confidence (~0.72–0.75).** Eval-only on the LIVE v6_live2 §D-GOLONG run
(30k→90k arc pulled from vast); diagnosis only, no re-pretrain/Rust/engine change.

PHASE 0 — the run did NOT "break twice." ONE causal chain: a FALSE single-point SealBot-WR
gate abort at 87.5k (`v6l2golong.log:146636`, wr_history `[[62500,0.29],[75000,0.05]]` — only
2 points; the 87.5k drain re-eval had already RECOVERED to ~0.19) + a BENIGN exit-134 SIGABRT
(`terminate called without an active exception`, `:147047`) during graceful teardown AFTER
checkpoint+buffers saved. Gate fix committed `b340e99` (B/C triggers now require
`wr_collapse_consecutive_evals=3`; the 2-point history cannot fire it). The "second abort" is
the benign teardown the mandate anticipated, not an independent divergence.

PHASE 1 (decisive) — value-head won/lost discrimination (AUC) across 10 checkpoints (40k→90k)
on a matched 8000-position arc pool (`scripts/structural_diagnosis/fragility_value_discrim.py`,
reusing `forced_win_detector` + `golong_game_analysis` + `load_inference_model`):
  • B(iii) FALSE (lead refutation, confound-free): AUC_spread PEAKS at 75k under ALL spread
    metrics while `value_fc2` weight-norm COLLAPSES (0.224→0.143) + g4 band fails → the 75k
    event is a weight/CALIBRATION wobble, NOT a ranking-discrimination collapse (AUC is
    scale-invariant). fc2 self-corrected non-monotonically to ~0.20 by 85–90k.
  • B(i) FALSE: AUC_compact never reaches 0.70 (max 0.688); spread-vs-compact direction FLIPS
    by metric (bbox/ncomp spread>compact; density spread<compact) = a ply confound (spread ply
    72.9 vs compact 21.6), NOT a clean spread deficit.
  • B(ii) = off-window artifact: clear-won OFF-window-only spread reads ~0/neg (CORRECT —
    unconvertible, no logit); clear-won IN-WINDOW (convertible) spread reads positive
    (+0.13..+0.47, peaks 75k).
  • B(iv) precondition IS MET (independently replicated): per-game off-window forced-win rate
    RISES with spread (Spearman(bbox, off-window rate) +0.356, p≈9e-11). → **B is dead on its
    CONSEQUENCE (i+iii), not its precondition** (the stronger refutation): the throttle is
    real, but the value head handles it correctly and spread-discrimination does not collapse.

PHASE 2 — 75k event at FLAT lr ~0.00197 (53k re-warm 2e-3 plateau; did NOT decay through
recovery → A's literal lr-decay mechanism doesn't hold, but sustained-high-lr wobble does);
grad_norm stable 0.9–1.4; t3_spread dipped 0.30@77.5-83k → recovered 0.43.

PHASE 3 — SealBot-WR softening (b340e99) is necessary+sufficient; stride5 (passive p90=4) +
grad_norm stay FAST (never false-fired, don't self-correct); g4 band is already warn-only
(fired a TRUE warning at 75k without aborting).

Verification: 7-agent fresh-review + red-team workflow (`wf_72614f66-0b1`) — 6/7 upheld A
(methodology SOUND); 1 red-team (recovery-illusory, conf 0.25) capped confidence, did not
overturn. Caught + corrected a reviewer error (3 agents cited the in-run EMA to wrongly claim
B(iv) fails).

Decision: fix LR/stability; bank `checkpoint_00087500.pt` as deployable single-window
baseline; **multi-cluster PARKED on the fragility basis** (B falsified → not a fragility fix;
off-window remains a real ACTION blind spot for the SEPARATE adversarial-human-exploit question
only). **REVISIT if the 100k full SealBot eval ≤ ~0.15** or loss/draw/grad destabilize. No
evidence prior long runs all broke at 75–90k (first run to traverse it; recovered) → fragility
NOT systematic.

Full: `reports/investigations/fragility_diagnosis_2026-06-07.md`.
Instruments (local, uncommitted): `scripts/structural_diagnosis/fragility_value_discrim.py`,
`investigation/fragility_2026-06-07/`.

## §D-COHERENCE — in-window vs off-window forced-win conversion decomposition — 2026-06-08

Verdict: **V-INWINDOW** (significant sub-material) — the golong `forced_win_conversion`
decline is driven by IN-WINDOW finishing, NOT the off-window structural defect. Eval-only,
read-only on banked golong self-play records + checkpoints; ZERO engine/config/Rust/pretrain
change (`git diff --stat` empty). Source data = vast
`checkpoints/_archive_golong_kill_20260608T065342Z/` (the killed run, pulled). Usable arc
buckets (promoted-inference checkpoint_step tags on Rust GameRecorder self-play records):
30k (n=256 gs) / 53k (377) / 87.5k (293); 75k=1, step-0 legacy unusable.

**CORRECTION (2026-06-08 — RETRACTED by §D-GLOBALCONC Phase 2b; see that entry below).** The
V-INWINDOW verdict and the "**NOT multi-cluster (off-window not the driver, 19 %)**" routing in this
entry are **RETRACTED**. They rested on the legacy depth-1/ply flatten win-cell unit, which mislabeled
depth-2 wins that COMPLETE off-window but place their FIRST stone in-window as in-window. Under the
turn-correct completing-cell unit (`forced_win_detector.winning_turn_cells`, the cell that LANDS the
win — re-run `coherence_decomposition.py --unit turn`): the in-window decline LOSES CI significance
(per-game-side d−0.036 CI[−0.098,+0.028]; turn-level d−0.116 CI[−0.258,+0.034] — both straddle 0),
the off-window share RISES significantly (+0.074 CI[+0.009,+0.141]), and the decline shift-share flips
**81/19 → 46/54** → off-window is the larger, significant, rising leg. The GLOBAL `forced_win_conversion`
decline (−0.075) and the over-spread WHAT (own ncomp 14→22) are **unit-INVARIANT and STAND**; only the
in/off-window SPLIT and the multi-cluster dismissal change. This RE-OPENS the off-window / multi-cluster
line (→ §D-GLOBALCONC Phase-3 Branch C). History preserved below as originally written.

**Phase 1 (decomposition).** Off-window share is FLAT (turn-level 0.516→0.480→0.554, Δ+0.038;
per-game-side Δ−0.006). Shift-share of the global turn-level decline (off-window converts ≈0
⇒ global = conv_in·(1−share)): **in-window-drop = 81 %, off-window-rise = 19 %** → the
red-team "decline over-determined by off-window rise" premise is REFUTED. In-window
conversion drops: recurrence-robust per-game-side Δ−0.089 (CI[−0.156,−0.016], seed-stable,
**just SUB-material** vs the 0.10 bar); turn-level Δ−0.151 is ~40 % recurrence-inflated (don't
trust as "material"); non-converting game-side fraction rose 17.0→25.7 %. Survivorship
inverted: in-window forced-win COUNT per game-side ROSE 1.49→1.90.

**Phase 2 (mechanism, self-contained — re-derived, NOT leaning on §D-FRAGILITY).** On a fixed
common pool of in-window forced-win positions, NEITHER NN head degrades: POLICY finishing-mass
flat (p_win 0.237→0.246, top1 ~0.30, entropy flat), VALUE healthy (AUC won>not-won flat
0.79→0.81; 75k global-sharpness peak independently reproduced). The drop is DISTRIBUTIONAL:
under the same models, p_win on 87.5k-sourced positions (0.187) ≪ 30k-sourced (0.256), Δ−0.069
(~27 %, ≈ the conversion drop), uniform at every checkpoint, `#win-cells`-invariant.

**Phase 2b (WHY — operator-driven).** Operator read the games: "defending + scattered attacks,
way too spread out." Confirmed: the mover's OWN force OVER-FRAGMENTS along the arc — own
components 14.0→16.6→**22.1** (+58 %) while stones rise only +9 % (components/stone 0.34→0.49,
elevated within matched ply- AND stone-bands → training-checkpoint property, not game-length);
largest-blob fraction 0.35→0.24; local support around the win 1.59→1.45 falls; opponent
interference 0.90→1.12 rises. So forced wins are thin, unsupported, single-threat opportunities
in scattered structure. This is the **opposite pole** from the §175/§S181 colony attractor
(over-homogenization) — spread ran too far into force-fragmentation.

**Lever (to strategy layer).** = reward/target shaping for in-window line-COHERENCE, sharpened
to **force-CONCENTRATION of the attacking mass**; NOT O1 (policy-target — Phase 2 refutes the
policy-head mechanism, O1 stays falsified), NOT multi-cluster (off-window not the driver, 19 %).
Guard rail: a concentration lever must land BETWEEN over-spread and colony-homogenization — gate
on conversion/fragmentation here AND colony_fraction/stride5 monitors AND SealBot-WR (the spread
also drove the 24→38 % WR gains; don't trade strength for finishing efficiency blindly).

Verification: independent REVIEW (UPHELD — all numbers reproduced, leak-check clean) +
RED-TEAM (6/7 pillars clean; over-spread reinforced by matched-band controls; one owed
qualification = sub-material magnitude, folded in). NOTE: the off-window target drop is the
`window_flat_idx_at_geom` reprojection in `engine/src/game_runner/records.rs`; the widely-quoted
"`records.rs:62`" is the pass-slot skip, not the off-window drop (mechanism right, line wrong;
the `forced_win_detector` docstring perpetuates it — left untouched per read-only mandate).

Full: `reports/investigations/coherence_decomposition_2026-06-08.md`. Instruments (local):
`scripts/structural_diagnosis/coherence_decomposition.py` (Phase 1),
`coherence_inwindow_policy.py` (Phase 2), `coherence_overspread.py` (Phase 2b);
`investigation/coherence_2026-06-08/` (replays + checkpoints + JSON + REVIEW.md/REDTEAM.md).

## §D-OVERSPREAD — WHY does the model over-spread? (5-driver discriminator) — 2026-06-08

Verdict: **NO clean driver; the hypothesized D1→D3→D5 value-first stack is FALSIFIED.** Eval-only,
read-only on banked golong replays + the 11-rung checkpoint ladder; ZERO engine/config/Rust/pretrain
change (`git diff --stat` = only this sprint-log). 12-agent parallel workflow (5 driver probes →
per-driver red-team + independent review + ordering attack). Pre-registration (lighting + thresholds
locked before any probe): `investigation/overspread_2026-06-08/PREREGISTRATION.md`.

**Drivers.** D1 value-discrimination ceiling **OUT** — value RANKS concentration; the clean
(turn-fork, stone-matched) strand AUC 0.69–0.79 RISING (the mover_ncomp/largest_frac headline
strands are stone-confounded — red-team correction; OUT rests on the fork-redundancy strand only).
D2 off-window-structure-biases-play **OUT** — abandonment flat/down, model commits to boundary lines
MORE (P(pick interior) 0.64→0.56); the boundary-share rise is a SYMPTOM of over-spread, not a driver.
D3 target-doesn't-credit-forks **OUT** — fork-affinity HIGH (110–180× the no-credit null after
removing the 52% finishing-move confound) and does NOT fall (75k sharpness peak). D4 exploration
**OUT by constancy** — 33 exploration knobs byte-identical across 3 relaunches; cosine on LR not temp.
D5 self-play-co-adaptation **INCONCLUSIVE** — Part-2 (losses are spread-force) instrument-blocked (no
eval move-sequences banked + window-masked ModelPlayer); Part-1 decline-leg "spearman ±1.0" is a
training-step monotonicity artifact (87.5k WR injected; colony_wins co-falls → partly the opposite
pole). All §2 re-validation guards PASS (each OUT ruled out by in-context evidence, not a borrowed
prior). Review UPHELD (git-clean, numbers re-derived, no leaks).

**Ordering red-team — DECISIVE co-movement.** value_mean, policy p_win, policy-fork-mass AND
MCTS-fork-mass ALL peak at 75k (the §D-FRAGILITY/§D-COHERENCE sharpness transient) while over-spread
(ncomp 14→22) rises MONOTONICALLY → a monotone phenomenon cannot be caused by a non-monotone one ⇒
value+search are a COUPLED WAVE riding ON TOP of the spread substrate, not its generator. Value-first
falsified in the wrong direction; value/search are clean signals to lean ON, not holes to patch.

**D5 follow-on (2 purpose instruments; reframe: over-spread = own-force FRAGMENTATION, a single-window
weights property, NOT off-window reach — the block was un-RECORDED moves, not the window mask).** Leg A
INTERNAL (banked self-play, n≈210–320/bucket): the more-fragmented side is NEVER the loser
(P(more-frag lost)=0.40–0.50 ≤0.5 everywhere; winner has MORE components; mildly favored early, neutral
by the finish) → **over-spread NOT punished internally** → the co-adaptation PRECONDITION is CONFIRMED.
Leg B EXTERNAL (generated SealBot games, KClusterMCTSBot @ temp 1.0, REGIME=REPRO verified): the naive
cut-frac Δ(loss−win) (+0.16/+0.04/+0.21, growing) is a **GAME-LENGTH confound** (losses longer: 30k
69v46, 87.5k 63v47 plies); STONE-MATCHED (fixed-ply) Δ mostly NULL/reversed (30k ply40 −0.13, 50k
−0.08/−0.01 straddle; 87.5k ply60 +0.16 CI>0 but n_win=2, uninterpretable) — win class underpowered (WR
0.09–0.14). → **external loss-spread NOT established** (length-mediated, not clean spread-force-loses).
**D5 = precondition CONFIRMED, external clause INCONCLUSIVE** — leading framing, not a clean LIT driver.
Branch 2 (compact-reference regularizer) rests on the confirmed internal neutrality. Instruments:
`overspread_d5_internal_punish.py` + `overspread_d5_sealbot_lossspread.py`.

**TURN-vs-PLY standing hole (operator insight, folded in).** A turn = 2 stones; `count_winning_moves`
/ quiescence are depth-1 (single-stone) — wrong unit. New turn-correct primitive
`scripts/structural_diagnosis/turn_wins.py::count_winning_turns` = `|depth1 ∪ {depth2 second-stones}|`.
Empirics: depth-1 undercounts the turn win-set in **86.5%** of threat snapshots; at in-window
forced-win turn-starts the engine quiescence `credit_gap=1.000` / `ply_blind≈0.95` (NEVER fires +1).
Strength order: this-turn depth-2 completion > next-turn depth-1 ≥3 fork > single ply threat — the
engine credits only the weaker, blind to the stronger. But the gap is FLAT across the arc → a STANDING
structural hole that *permits* uncorrected spread, NOT the trend driver (a constant cannot drive +58%
— same logic that ruled out D4). Eval-MEASURED, engine NOT changed (Phase-B; depth-2 is O(threats²)/
leaf, deliberately omitted §28/§30). Recommend promoting `count_winning_turns` into
`forced_win_detector.py` (unify the f-vs-s inconsistency); audit `probe_threat_logits.py` for the
same depth-1 blind spot.

**Routed fix (NOT value-first; operator-gated; 30k SIGNATURE smoke before any sustained run).**
(1) Close the credit-gap hole — turn-correct HARD concentration/fork credit (promote
`count_winning_turns` into the value/target/quiescence path; aux value target predicting turn-fork
redundancy) so a thin win ≠ a concentrated turn-fork (heads aren't broken — *strengthen the soft
signal into a hard one*). GUARDS: not A2 avg-pool, not a config knob; quiescence variant bench-gated.
(2) D5 compact-reference self-play regularizer — the co-adaptation backstop (GUARD: not bot-mix
anti-colony, not a PFSP league). **Decisive missing instrument FIRST:** D1/D3 OUT ⇒ the soft signal
already failed to prevent spread, so the self-play DYNAMIC (D5) may be load-bearing — but D5 is
instrument-blocked. Before paying for a fix, build a **spread-uncapped, move-recording** SealBot-eval
(lift the n_actions window mask) and test directly whether losses are spread-force. Gate any lever on
the fragmentation/conversion metrics AND colony_fraction/stride5 (opposite pole) AND SealBot-WR.

**Lessons.** L: when every menu-driver is OUT/INCONCLUSIVE, the over-spread trend is generated by the
self-play DYNAMIC upstream of value/search — diagnosed via CO-MOVEMENT (a monotone effect can't come
from a peaked cause), not by a single lit driver. L: a STATIC structural hole (depth-1 credit gap,
constant) is a standing CONDITION, not a trend DRIVER — apply the D4 constancy logic to any constant.
L: `count_winning_moves`/quiescence are depth-1; the turn-correct unit (`count_winning_turns`)
matters wherever forks/winning-counts are credited (value override, target shaping, threat probe).
L (citation): §D-WALLCAUSATION's `evaluator.py:113`/`max_spread≤18` is imprecise — the spread bound
is the n_actions window mask at `hexo_rl/eval/evaluator.py:108-118`, no `max_spread` variable
(mechanism holds). Falsified-register candidate: **§D-OVERSPREAD D1 (value-discrimination ceiling
drives over-spread) FALSIFIED** — value ranks turn-fork concentration 0.69–0.79 RISING.

Full: `reports/investigations/overspread_driver_2026-06-08.md`. Instruments (local, uncommitted):
`scripts/structural_diagnosis/turn_wins.py` + `overspread_forkredundancy.py` + `overspread_d{1..5}_*.py`
+ red-team scripts; `investigation/overspread_2026-06-08/` (PREREGISTRATION + notes + JSON + workflow).

## §D-GLOBALCONC — mid-game GLOBAL-concentration discriminator + turn-unit fix — 2026-06-08

Verdict: **GLOBAL-SIGNAL-ABSENT — neither head carries a clean concentration signal at the build-up
scale.** Eval-only GATE for the §D-OVERSPREAD follow-on (strategy-layer red-team: every §D-OVERSPREAD
concentration signal was measured at LOCAL TACTICAL positions; the fragmentation it explains is a
GLOBAL build-up property — different scales). Read-only on banked golong replays + the 11-rung
ladder; tracked-source change = the sanctioned Phase-2 detector edit + tests (+ turn_wins shim +
coherence_decomposition `--unit`). Pre-reg locked before any probe:
`investigation/globalconc_2026-06-08/PREREGISTRATION.md`. Verified by a 6-agent fresh-context
REVIEW + 4-lens RED-TEAM (UPHELD_WITH_CAVEATS; it refuted the initial policy claim + caught a
determinism bug — folded in below).

**Phase 1 (the gate).** Pool = BUILD-UP turn-starts (ply-band swept, immediate-forced-win turns
EXCLUDED — NOT the §D-OVERSPREAD tactical regime), n=9000, 30k/53k/87.5k buckets, corr(ncomp,stones)
=0.604. (a) **VALUE** `AUC_globalconc` = P(value ranks CONCENTRATED build-up > SCATTERED) at matched
stones AND matched eventual outcome (stratified Mann-Whitney): **0.40–0.42 mean every band, max
anywhere 0.579, never reaches 0.60**; stone-confound REFUTED as the cause (AUC invariant across
stone-band widths 1–12 incl. EXACT width-1 match; value is stone-agnostic); CI 2–12 SD below 0.50;
won-only ~0.50 (NEUTRAL) / lost-only ~0.33 (INVERTED). → value does NOT favor global concentration.
(b) **POLICY PRIOR** main-vs-PERIPHERAL `AUC` (adjacency controlled) = **0.565 mean, 0.547–0.582,
upper CI never reaches 0.60, flat-declining** — the prior carries only generic ADJACENCY
(`AUC_adj`≈0.78), NOT main-mass concentration. → **GLOBAL-SIGNAL-ABSENT on BOTH heads** (NOT the
clean MIXED first reported; the red-team showed the 0.78 was the trivial adjacency floor). Consistent
with §D-OVERSPREAD "no clean driver." Re-scopes **§D-OVERSPREAD D1**: value ranks LOCAL turn-fork
concentration (0.69–0.79) but is GLOBALLY absent (0.41) — D1 was mis-scoped onto local tactics
(frame meta-lesson #1, empirically confirmed). Value-sign audited (won +0.017 ≫ lost −0.166).

**LIVE COMPETING HYPOTHESIS (un-refuted; the fix must discriminate).** value AUC<0.50 = value rates
more-fragmented build-up HIGHER. H1: value is a HOLE (over-credits threat-spread w/o convertibility).
H2: value is CORRECT — §D-OVERSPREAD D5 internal-neutrality (more-fragmented side NEVER the self-play
loser) means a value head that doesn't penalise build-up fragmentation is a FAITHFUL estimator of the
spread-tolerant equilibrium, and the liability is at the FINISH not build-up. H1→Branch A; H2→Branch C
+ Branch A pushes away from self-play-optimal (risks the 24→38% WR the spread bought).

**Phase 2a (turn-unit fix; one clean commit; full `make test` green: 1831 pass; 31 detector tests).**
Promoted `winning_turn_cells`/`count_winning_turns`/`is_fork_turn`/`FORK_THRESHOLD` from turn_wins.py
into `forced_win_detector.py` (turn_wins → re-export shim). Unified the f-vs-s inconsistency onto the
COMPLETING cell `pair[1]`. PROVEN bounded: `forced`/`converted` invariant to the f-vs-s choice
(non-empty iff any win) → `forced_win_conversion` unchanged; only off-window classification of depth-2
wins shifts. **DETERMINISM FIX (red-team-caught live-metric regression):** `get_threats()` order is
unstable → `winning_turn_cells`/off-window binding were non-deterministic (65/4068 mismatches); fixed
by sorting `depth2_wins` candidates (0/3069 after; regression test). `probe_threat_logits.py` audited
— NOT depth-1-count based (threat HEAD + level≥3); unaffected.

**Phase 2b (re-validation — FLAG HARD; RETRACTS the 19% routing).** Deterministic (two turn runs
bit-identical). GLOBAL conversion IDENTICAL across units (0.750→0.676 — invariance proven). The
§D-COHERENCE in-window ATTRIBUTION does NOT survive: in-window decline loses CI significance at BOTH
levels (per-game-side −0.036 CI[−0.098,+0.028]; turn-level −0.116 CI[−0.258,+0.034] — both straddle
0), off-window share RISES significantly (+0.074 CI[+0.009,+0.141]), shift-share flips **81/19 →
46/54**, V-INWINDOW → AMBIGUOUS. Cause: depth-2 wins completing off-window but first-stone in-window
were mislabeled in-window by the legacy flatten unit. → **RETRACTS §D-COHERENCE's "NOT multi-cluster
(19%)"** — off-window is the larger, significant, rising leg. Does NOT touch the Phase-1 finding nor
the over-spread WHAT (ncomp 14→22).

**Routed fix DESIGN (no impl, no engine change, no run; operator-gated).** Lead with the H1/H2
DISCRIMINATOR (fine-tune value/aux head on a banked ckpt → re-run globalconc_probe [value AUC>0.50?]
AND overspread_d5_internal_punish [does internal-neutrality reverse?]). Branch A: value-side (+ opt.
policy-side, since policy is ALSO absent ~0.565) GLOBAL-concentration AUXILIARY PREDICTION target
(largest-region-share / support-weighted attacking-mass concentration). GUARDS: not A2 avg-pool, not
a config knob, not LOCAL turn-fork credit (wrong scale); reward supported attacking mass not raw
clustering, gate between over-spread and the §175/§S181 colony pole; apply the WR guard-rail to
itself (H2 risk); won't fix the turn-pair scale. Branch C (RE-OPENED, possibly SAFER primary under
H2): off-window/multi-cluster for the CONVERSION/finishing leg (where §D-OVERSPREAD Leg B showed the
loss actually happens — length/finishing). Smoke reads TRAJECTORY (components/stone reverses, value
AUC>0.50, D5 internal-neutrality reverses, colony_fraction/stride5 clean, SealBot-WR holds).

**FOURTH SCALE still unprobed (red-team-named): the TURN-PAIR / second stone.** Both arms are
per-single-stone; neither measures the JOINT concentration of the two stones a turn places. Branch A
(per-position) wouldn't fix turn-pair sequencing. Next probe if the build-up fix doesn't move
components/stone.

**Lessons.** L: a concentration signal can exist at the LOCAL tactical scale (D1 0.69–0.79) yet be
ABSENT at the GLOBAL strategic scale where the decision is made — match the instrument's regime to the
decision's (frame meta-lesson #1, empirically confirmed). L: a policy "concentration" AUC that lumps
isolated moves into "spread" collapses to the trivial adjacency floor — control for adjacency
(main-vs-peripheral) or you measure "plays near its stones," not concentration (red-team catch). L:
value-absent at a scale is direction-ambiguous (hole vs faithful-estimator) — discriminate with a
self-play-internal-punishment counterfactual before forcing the feature. L: the depth-1→turn-correct
unit change is `forced`/`converted`-invariant but moves the in/off-window SPLIT — re-validating
§D-COHERENCE RETRACTED its 81/19 → 46/54 (off-window re-opened). L: `get_threats()` order is unstable
→ any primitive selecting a single threat cell (`winning_turn_cells` `pair[1]`) must sort or it makes
a live metric non-deterministic. Falsified-register: **§D-COHERENCE "in-window finishing, NOT
multi-cluster (off-window 19%)" RETRACTED** under the turn-correct unit (off-window ≈54%, the
significant rising leg). Initial §D-GLOBALCONC "MIXED / policy-sees-concentration" SELF-CORRECTED to
GLOBAL-SIGNAL-ABSENT by the red-team (0.78 = adjacency floor).

Full: `reports/investigations/globalconc_probe_2026-06-08.md`. Instruments (local):
`scripts/structural_diagnosis/globalconc_probe.py`; `coherence_decomposition.py --unit {ply,turn}`;
`investigation/globalconc_2026-06-08/` (PREREGISTRATION + JSON + run{,2}.log + verify_workflow.js).

## §D-RECONVERGE — off-window PLACEMENT conversion-lift discriminator (THE GATE) + corrections — 2026-06-08

Converges the §D-OVERSPREAD/§D-GLOBALCONC arc back to the ORIGINAL §PRELONG off-window frontier. A
unit bug in the founding §D-COHERENCE conversion metric (depth-1/ply flatten) mis-routed the
investigation away from off-window ("NOT multi-cluster, 19%"); §D-GLOBALCONC Phase-2b corrected the
unit (completing cell) and re-opened off-window as the larger, rising leg (46/54). The retraction was
BORDERLINE (off-window share +0.074 CI[+0.009,+0.141], point-pinned by the determinism fix) — enough
to KILL the 19% dismissal, NOT enough to commit multi-cluster Rust-weeks. This is the cheap
discriminator the borderline demanded.

**Phase 0 (COMMITTED + PUSHED `origin/phase4.5/overspread_driver`).** `7e786b9` (detector turn-unit
promotion + f-vs-s unification onto the completing cell + `get_threats()` determinism SORT, bundled so
no intermediate commit carries the non-determinism + 7 tests + `turn_wins` shim) + `1361ca0`
(`coherence_decomposition --unit` + `globalconc_probe.py` + sprint-log §D-GLOBALCONC + §D-COHERENCE
CORRECTION note). Commit-safety PROVEN: GATED `forced_win_conversion` invariant (forced iff win-set
non-empty; converted reads outcome — both f-vs-s-independent), only the WATCH `off_window_forced_win_rate`
shifts toward correctness+determinism. `make test` green (1829 py + rust). **Flake (`test_shape_fallback`)
verified PRE-EXISTING** (registry `(8,626)→v6w25` shape-probe fragility): parent suite 1822 + change
1829 BOTH green; the change adds ZERO registry registration → cannot trigger it (STOP condition
impossible).

**Phase 1a — THE GATE — VERDICT: LIFT (off-window PLACEMENT is the binding conversion constraint).**
EVAL-ONLY, read-only, NO Rust/training. Reuse `_ControlDropMCTSBot` (single-window, off-window priors
dropped — production path) vs `KClusterMCTSBot` (multi-window legal-set), SAME model, ONE switch. Pool
= golong self-play off-window forced-win turn-starts (corrected `winning_turn_cells`+`is_off_window`);
160 OFF-leg (**all-off 151 = 94%**, control conversion ≈0), 160 IN-leg reference. `recovery =
MULTI_off / R_in` (R_in = in-window finishing skill). Pre-registered LIFT (recovery≥0.50 + every-ckpt
lift-CI>0 + placement≥0.5) — **all met**:

| ckpt | R_in | OFF control | OFF multi | lift CI | recovery | off-placement-frac | all-off multi |
|---|---|---|---|---|---|---|---|
| 30k | 0.581 | 0.025 | 0.412 | [+0.310,+0.464] | 0.71 | 0.73 | 0.424 |
| 50k-PEAK | 0.631 | 0.056 | 0.512 | [+0.376,+0.544] | 0.81 | 0.65 | 0.523 |
| 87.5k | 0.519 | 0.050 | 0.312 | [+0.193,+0.340] | 0.60 | 0.66 | 0.311 |
| **pooled** | — | ~0.04 | — | **+0.369** | **0.71** | **0.68** | **0.419** |

Multi recovers 60–81% of in-window finishing on off-window forced wins, MOSTLY by placing the
off-window cell (placement 0.65–0.73); single-window production is structurally walled (~0.02–0.06).
Ablation-clean (drop fired 28–30k expansions, K>1 ~96%, max_k 7–8, `dropped_all_turns=0` — NOT
vacuous). The all-off subleg (control ≈0 → multi ~0.42; ANY conversion there REQUIRES off-window
placement) is the clincher. → the §D-COHERENCE +54% off-window leg is a REAL finishing liability the
action space addresses → **Branch C (off-window/multi-cluster) is the validated lever.** Phase 2
(H1/H2) is CONDITIONAL on NO-LIFT → NOT run.

**Phase 1b — determinism CLEAN.** Repeat-call sweep (951 games × 8): **0 mismatches** across the live
chain — `analyze_recorded_game` 0/13314, `winning_turn_cells`+binding 0/6657 over **3069** turn-starts
(the exact 65/4068 class, now 0), depth1/2 SETs 0/6657, `coherence` both units 0/6657. The
`depth2_wins` sort closed the only live-chain leak. **Two LATENT risks flagged OUTSIDE the chain** (NOT
fixed — design-only): `offwindow_adversary_bot.py:261` `blocks[0]` from unsorted `get_threats()`
(affects the ARMED exploitability monitor's reproducibility — one-line `sorted()` fix, operator-gated);
`generate_threat_probe_fixtures.py:129` sorts by level only → ties retain unstable order (fixture
REGENERATION non-deterministic; live gate reads the baked fixture → unaffected at runtime).

**Phase 3 — routing DESIGN-only (operator-gated).** Branch C re-validated on the new conversion basis
(NOT the retracted 19%); also covers the unprobed TURN-PAIR fourth scale (legal-set is over both
stones). §D-MULTICLUSTER gates re-evaluated: **S0** (Rust 362-multiwindow head) still needed — the
inference lift raises the EFFICACY prior, does not pay for it; **S1** is the DOMINANT residual risk and
UNCHANGED — the probe models only the FIRST of the THREE training-layer off-window drops
(`records.rs:62`); the bootstrap→selfplay→training-loop handoff (`backup.rs:112`/`:105` + value
over-fit, §174 e50) is irreducibly post-S0 and **§174 failed S1 3× (>50%-likely to fail)**; **S3**
(`exploit_probe`≤0.06, NEVER vs-bot WR) unchanged. **GATE EXT-LINK (OPEN, BLOCKING S0):** the LIFT is
SELF-PLAY conversion; the self-play→external link is **SIGN-AMBIGUOUS** — §PRELONG-BRIDGE's 0.0pp
(n=400, Wilson95 upper 1.52pp) + D5 Leg B null/underpowered say a vs-SealBot-WR A/B reads ≈0 BY
CONSTRUCTION (false-clears), while §D-EXPLOIT's 18%-vs-6% adversary (p=0.00017) says off-window matters
MORE vs adversarial/human play. → the open gate must be the ADVERSARIAL/spread-uncapped instrument
(`exploit_probe`≤0.06 OR a spread-uncapped move-recording eval, off-window-targeting WR Δ CI-lower>0),
NOT a SealBot-WR tourney. Discharge BEFORE any Rust-weeks; do not assert the link a 4th time.

**Verification — 4-agent fresh-context REVIEW + 3-lens RED-TEAM (NOT the implementer): UPHELD WITH
CAVEATS.** REVIEW re-derived 30k BIT-IDENTICALLY + confirmed corrected unit / one-switch ablation /
non-vacuous drop / clean Phase-0 tree. RED-TEAM: lift GENUINE off-window placement (all-off alone
clears all gates; binding-only DILUTES) — correction: control ≈0 not ==0 (residual nets out,
conservative); retraction is a CORRECT monotone reclassification (387/3069 flip IN→OFF-only, 0 reverse;
`pair[1]`=landing stone 12/12; forced/converted invariant 0/37279); frame-gate HIGH → the EXT-LINK gate
sharpened to the adversarial instrument (above). Zero tracked-source contamination.

**Lessons.** **L (CLAUDE.md candidate, promoted): a unit error in a founding measurement mis-routes
every downstream investigation — verify the measurement UNIT before building a frame on it.** The
§D-COHERENCE depth-1/ply flatten counted a depth-2 win's in-window FIRST stone as in-window
convertibility, hiding that the win LANDS on the off-window completing stone; that one-cell mislabel
sent a multi-week detour ("NOT multi-cluster, 19%") that §D-GLOBALCONC Phase-2b + this gate reversed.
L: a BORDERLINE retraction earns a CHEAP eval-only discriminator before an expensive lever — the
LIFT (recovery 0.71) cost one GPU run and converted "re-opened, magnitude-borderline" into "validated
binding constraint." L: an inference conversion-lift is a NECESSARY-condition / capability probe — name
the self-play→external KILL link as an explicit OPEN gate, and pick the RIGHT external instrument
(adversarial, not a fixed-bot WR that false-clears). Falsified-register: no new falsification (confirms
§D-GLOBALCONC's off-window re-opening + Branch C).

Full: `reports/investigations/offwindow_reconverge_2026-06-08.md`. Instruments (local):
`scripts/structural_diagnosis/offwindow_placement_lift.py` + `determinism_audit.py`;
`investigation/reconverge_2026-06-08/` (PREREGISTRATION + JSON + review_scratch). Phase-0 commits
`7e786b9`,`1361ca0` pushed.

## §D-EXTLINK — discharge the off-window external gate before any S0 Rust-weeks — 2026-06-08

Verdict: **EXT-LINK-REAL.** The off-window blind spot is a real EXTERNAL / adversarial defect, not just
the §D-RECONVERGE self-play conversion constraint → Branch C justified for **least-exploitability
(Objective A)**; S0 spec MAY proceed (still S1-dominant-risk gated). Discharged on the ADVERSARIAL /
spread-uncapped instrument, NEVER SealBot-WR (false-clears by construction). Eval-only, frozen
checkpoints; pre-registration LOCKED before the run (`investigation/extlink_2026-06-08/PREREGISTRATION.md`).

**Phase 0 (COMMITTED `a7ba110`, sole sanctioned commit).** Fixed the gate instrument's determinism:
`offwindow_adversary_bot.py` blocked level-5 threats via `blocks[0]` from UNSORTED `get_threats()`.
Characterized (pure-engine, 24612 replay positions): `get_threats()` order unstable **74.6%** (intra-
process, every run); OLD raw `blocks[0]` varied **165/166** block-relevant (≥2 level-5) positions → the
§D-EXPLOIT numbers were computed on a non-deterministic instrument (the §114 lesson). One-line
`sorted(...)` (block SET unchanged; representative pinned). Post-fix `get_move` **0 mismatches / 24612 ×
8 repeats**. `make test` green (1829 py + rust).

**Phase 1 — THE GATE (deterministic instrument, {30k / 50k-PEAK / 87.5k}, n=200/arm, sims=128, random×6).**
- **1a (adversarial forced-win) reproduces 18v6 STRONGER.** All 3 ckpts FORCEABLE: exploit
  0.255 / 0.235 / 0.215, control 0.075 / 0.06 / 0.05; pooled margin **+0.173 [+0.134, +0.213]**, every
  per-ckpt margin CI-lower > 0. Checkpoint-INDEPENDENT (drifts slightly DOWN with training). CLEAN
  one-switch ablation: `any_offwindow_forcing_position_rate` equal across arms (0.325/0.325, 0.29/0.30,
  0.295/0.30) — builder skill constant; arms diverge ONLY at conversion. The point estimate SHIFTED UP
  vs the pre-fix §D-EXPLOIT 0.18 — exactly the consequence of the block non-determinism (Phase 0 real).
- **1b (off-window-targeting WR Δ at power).** WR Δ +0.162 [+0.122, +0.202]; off-window win class
  **n=178** (~89× the §D-OVERSPREAD Leg-B n_win=2 trap). NOTE 1a ≈ 1b (exploit wins ~100% off-window →
  not independent corroboration); per-ckpt FORCEABLE margins carry the verdict, pooled z=8.45 decorative
  (same run).
- **1b-causal (the genuine independent leg; uncapped KClusterMCTSBot vs capped ModelPlayer defender,
  one switch = the off-window cap, 50k-PEAK n=100).** Uncapping CLOSES the margin **+0.16 → +0.03
  (−81%, drop z=2.56)**: the uncapped defender faces the forcing setup MORE (0.38 vs 0.27) yet loses 7×
  LESS (0.03 vs 0.22) — it BLOCKS, not avoids. The off-window advantage is CAUSALLY the action cap.
  Residual 0.03 ≤ the S3 gate (0.06) = a fix-DIRECTION efficacy prior — NOT an S3/S1 clearance (an
  inference-time multi-window overlay on single-window-TRAINED weights, not a trained multi-cluster model).

**Phase 2 (routing DESIGN-only, operator-gated).** **Objective A** (least-exploitability — gated REAL,
the deployment-vs-humans goal §D-FRAGILITY kept off-window alive for) vs **Objective B** (recover
self-play strength — NOT gated). The golong collapse (−0.32 external, peak→trough 0.38@50k→0.05@75k,
recovered ~0.19@87.5k) is dominated by the over-spread fragmentation self-play dynamic (§D-OVERSPREAD
no-clean-driver, value-first falsified); the off-window conversion leg ≈ **−0.040** (54% of the −0.075
GLOBAL conversion decline — NOT −0.075; that is the total) = ~13% of the −0.32 → **Branch C alone will
NOT recover Objective B** (bounded-small). Branch C's effect on over-spread is TWO-SIDED (entrench vs
channel), resolvable ONLY by the S1 TRAJECTORY smoke. §D-MULTICLUSTER gates unchanged: S0 (Rust 362 +
multi-window, NOT a single larger head) expensive, not paid by the lift; **S1 the dominant residual
(>50% fail, §174 ×3) UNCHANGED**; S3 post-fix off-window-pick quality ≤0.06. Optional cheap Python
multi-window S1 pre-check is KILL-ONLY (can falsify, cannot clear). Latent determinism carry-forwards:
`generate_threat_probe_fixtures.py:129` (fixture-regen, baked gate unaffected) + legacy §PRELONG copies.

**Verification — 4-agent fresh-context REVIEW + 3-lens RED-TEAM (`wf_8e547df9`, NOT the implementer):
UPHELD, no REFUTE.** REVIEW re-derived 30k 51/200=0.255 BIT-EXACTLY + confirmed pre-reg locked before
runs (mtimes) + Phase-0 sole-commit + read-only. RED-TEAM: instrument (low — regime genuinely off-window
cheb 10-15, win class powered, 1b-causal sound) / frame (low) / **magnitude (medium — caught the
off-window-leg mislabel: −0.075 is the GLOBAL total, the leg is its 54% share ≈ −0.040; corrected
throughout, conclusion STRENGTHENED)**. Zero tracked-source contamination.

**Lessons.** L: a gate instrument must be DETERMINISTIC before a load-bearing decision rides on it
(§114) — `get_threats()` order was unstable 74.6% intra-process; the §D-EXPLOIT 18v6 was computed on a
non-deterministic adversary and the deterministic rate is HIGHER (0.215-0.255). L: discharge the
off-window external gate on the ADVERSARIAL / spread-uncapped instrument, NEVER vs-bot WR (false-clears
— off-window wins need a dominant-but-exploitable state a fixed bot never creates). L: a one-switch
causal defender swap (capped→uncapped) is the cleanest external-defect proof AND a fix-direction prior;
it is NOT an S1/S3 clearance. L (CLAUDE.md re-validate-unit, again): the off-window LEG (−0.040) ≠ the
GLOBAL conversion decline (−0.075) — the dispatcher's "−0.075 off-window conversion" mislabeled the
total; the red-team caught it. EXT-LINK gates Objective A only; the −0.32 driver is elsewhere
(over-spread). No new falsification (confirms §D-EXPLOIT + §D-RECONVERGE Branch C on the external axis).

Full: `reports/investigations/extlink_gate_2026-06-08.md`. Instruments (local):
`scripts/exploit_probe.py` (deterministic), `investigation/extlink_2026-06-08/` (PREREGISTRATION +
`determinism_verify.py` + `analyze_p1.py` + `uncapped_defender_causal.py` + JSON + `review_workflow.js`).
Phase-0 commit `a7ba110`.

## §D-FOUNDING — re-establish the golong failure signal on the right instrument (checkpoint-relative round-robin) — 2026-06-08

Tested the unexamined premise every prior §D assumed: did the golong suffer a self-play STRENGTH
regression? Six investigations chased the CAUSE of the "−0.32 collapse" — a vs-SealBot WR (the
project's own flagged-WRONG strength instrument), never re-established on the instrument-matched
(checkpoint-relative MCTS-vs-MCTS) measure. EVAL-ONLY, git-diff clean (all code untracked under
`investigation/founding_2026-06-08/`).

**DATA RECOVERY (the enabler).** The post-peak ladder (75k…112.5k) believed lost was still on the live
vast box in `checkpoints/_archive_golong_kill_20260608T065342Z/` — full ladder 5k→112.5k + BANK (50k
PEAK / 85k PRE / 90k POST) + `best_model_75k_deceptive.pt` + 80M final log + AS-RUN yaml + replays.
Pulled + `torch.load`-verified (v6_live2 4-plane auto-detect). First §D able to measure the post-peak
segment checkpoint-relatively.

**VERDICT — TWO-FACED, and that is the result.**
- **On-distribution (standard openings): FLAT.** 12-rung all-pairs round-robin, MCTS-vs-MCTS, 64 sims
  (compute-bound at 128; relative Elo robust to sims), temp 0.5, color-balanced, n=40/pair (2640
  games), Bradley-Terry Elo. Slope **+0.13/1k, bootstrap CI [−0.25,+0.55], P(>0)=0.73, r²=0.014**; late
  rungs 90–112.5k statistically = 50k; heavily non-transitive (25/66 pairs invert; s45k beats s112.5k
  15–4). **No CI-resolved self-play strength regression → Objective B (recover self-play strength) is
  ILL-POSED.**
- **Off-distribution (6 random opening plies): RESOLVED late FELL.** Instrument 2×2 (temp × opening,
  rungs 50/75/90/100/112.5k, 400g/cell): holding temp=0.5, open0→open6 drops 90–112.5k from ≈50k
  (n.s.) to −55/−83/−88 Elo (CIs exclude 0); argmax (temp0) deepens only modestly. **Opening
  randomization, NOT temperature, is the lever.** Slope −1.4 to −1.7/1k, P(neg) 0.95–0.98, growing
  post-75k. **→ Objective A (off-distribution exploitability / off-window brittleness) is REAL**,
  triangulating SealBot-WR collapse + §D-EXTLINK off-window. Mechanism: scatter enlarges live-stone
  bbox → more completing cells off-window → spread-specialized late model (more off-window-dependent)
  punished hardest = the single-window × over-spread interaction, surfaced by scatter not a bot.

**Over-spread = style/symptom correlate, NOT the strength driver** (powered, n=759 self-play): the
naive "loser more-spread" (colony z=+11.5) is a SHORT-GAME blowout artifact — length-controlled it is
neutral/inverted (glen≥95 colony z=−1.7; winner more-spread by pw-dist z=−5.9); the loser−winner gap
is stable/shrinking with step; monotone-rising spread co-moves with FLAT on-distribution strength.
Its causal role is on the EXPLOITABILITY axis (the off-window mechanism), not Objective B.

**Premise corrections** (CLAUDE.md re-validate; stated plainly): **C1** no "canonical 200k floor"
exists anywhere in this log — unsourced/unverified. **C2** the run was NOT auto-killed by the SealBot
gate — the Wave3-B WR gate fired a `level:"warning"` at 87.5k (5.0% < 14.5%), the operator ran 25k
MORE steps to 112722, then an EXTERNAL process kill mid-eval; failure mode = instrument MISDIAGNOSIS
(matchup-WR read as a strength meter) misdirecting six investigations, not a premature auto-kill.
**C3** STRENGTH-ROSE/FELL/FLAT + §D-FOUNDING were dispatcher-defined, not pre-existing.

**INTELLECTUAL-HONESTY / falsified-register:** the first-pass reported **STRENGTH-ROSE +1.46/1k
"CI-cleared"** — a BUG (inverse-CI-variance WLS gave the zero-width BT anchor ~10¹⁵× weight, pinning
the fit through (35k,0)). The fresh-context red-team caught it; corrected → FLAT. L: never weight a LS
fit by a CI that can be exactly zero (the BT anchor); use a game-level bootstrap. L (re-validate the
unit, again): the SealBot −0.32 "collapse" was REAL but it measures OFF-DISTRIBUTION EXPLOITABILITY
(Objective A), not self-play STRENGTH (Objective B) — the founding measurement's MEANING was never
validated against the decision it gated (matchup-WR ≠ strength), the exact failure CLAUDE.md's
"verify the measurement unit" rule warns of. L: a borderline retraction earns a CHEAP eval-only
discriminator before any lever — the temp×opening 2×2 (minutes on the 5080) flipped "FLAT" into the
correct two-faced read; the red-team did not just verify, it changed the conclusion.

**ROUTING (design only, operator-gated):** Objective B ill-posed → do NOT open a 7th self-play-
strength cause-hunt. Pour effort into Objective A — the off-distribution/off-window exploitability the
spread specialization buys: (1) the single-window→multi-cluster/K-window ENCODING decision (§D-GOLONG
4d) now has measured evidence (the opening-scatter FELL is a measured off-window blind-spot instance);
Branch C (compact-reference regularizer) addresses the spread STYLE, the encoding swap the MECHANISM;
gate on an adversarial/spread-uncapped eval, never SealBot-WR. (2) Any fresh canonical run: steer+abort
on a checkpoint-relative mini-round-robin (Objective-B floor) PLUS an off-distribution/adversarial gate
(Objective-A), SealBot-WR demoted to logged style-diagnostic; run length governed by these, not a
guessed floor (C1).

Full: `reports/investigations/founding_signal_2026-06-08.md`. Banked data:
`reports/eval/golong_vast_pull_20260608/` (arena DB + ratings curve), pulled ladder in
`checkpoints/_archive_golong_kill_20260608T065342Z/`. Instruments (local, untracked):
`investigation/founding_2026-06-08/` — `rr_driver.py` (round-robin + BT + bootstrap slope),
`argmax_discriminator.py` (temp×opening 2×2), `overspread_causal.py`, `spread_trajectory.py`, +
`rr_agg`/`argmax_agg`/`ctrl_agg`/`rr_5rung_agg` outputs. Housekeeping: `a7ba110` (off-window
determinism fix) still UNPUSHED on `phase4.5/overspread_driver` (2 ahead) — operator-gated.

**§D-LAND — branch consolidated + FF-merged to master (2026-06-08).** Hygiene close of the §D arc
(no investigation, no instrument, no run). 4 hygiene commits landed on `phase4.5/overspread_driver`:
`936c7ad` (this §D-FOUNDING entry), `a07d405` (gitignore `investigation/` — local-evidence dir, mirrors
the `reports/**` do-not-travel convention; no tracked files lived under it), `c221398` (force-add the 11
untracked `scripts/structural_diagnosis/` evidence scripts, preserved as-is — not pytest-collected, zero
test impact), `1d40d6f` (deterministic `(q,r)` tiebreak in `generate_threat_probe_fixtures.py` regen —
the latent risk flagged in `aa2833a`/§D-RECONVERGE; regen-time only, the baked fixture NPZ is unchanged
at runtime). Pre-merge gate PASS: `make test` 1829 passed / 0 failed (the `test_shape_fallback`
full-suite flake did not appear); clean bisect (11 commits, all conventional, no WIP/fixup); FF-able.
**FF-merged master `7e35290 → 1d40d6f`** (no merge commit). NO tag — investigation arc, not a refactor
cycle; `checkpoint-archive-policy` defines no git-tag convention here. Supersedes the stale Housekeeping
line above: all §D commits are now pushed and on master. NOTE: two earlier pushed commits (`9085df9`,
`c339688`) carry the repo-standard `Co-Authored-By` trailer (present in 277/500 recent master commits) —
left as-is; the 4 new hygiene commits omit it.

**NEXT FEATURE (own branch, NOT built in this merge):** promote the two reusable instruments into clean
tracked primitives — (a) the checkpoint-relative round-robin Elo (`rr_driver`, currently an untracked
workaround around the hardcoded-v6 4-plane loader) as the Objective-B steer/abort floor; (b) the
adversarial/off-distribution gate (`exploit_probe` / off-window probe / opening-scatter Elo) as the
Objective-A gate — the steer/abort instruments the eval-swap lesson needs. Deferred fold-in for that
branch (no tracked target here, `rr_driver` lives under the now-gitignored
`investigation/founding_2026-06-08/`): log the play command (sims/temp) alongside round-robin results
(closes the docstring-says-128 / run-used-64 reproducibility gap).

## §D-EVALFOUND — build the eval/run foundation (right steer signal + eval throughput) — 2026-06-08

The next major phase after the §D diagnostic arc. The arc's central lesson: a run was steered/judged
on vs-SealBot WR — the flagged-wrong instrument for self-play strength — misdirecting six
investigations. Build the two foundations every downstream fork (Objective-A off-window / Phase-4.5
features / any canonical run) depends on: (1) steer/abort on the RIGHT signal; (2) eval throughput
(serial-eval halves training throughput). DESIGN → REVIEW → IMPL → REVIEW+RED-TEAM, pre-registered.
Eval/infra only — NO training run, NO encoding change, NO multi-cluster Rust, NO Phase-4.5 features.
Commits HELD (operator-gated).

**DESIGN (Phase 1).** `docs/designs/D_EVALFOUND_design.md` (v2). Pre-registered: a TWO-TIER instrument
(offline all-pairs round-robin = Phase-3 tool; live current-vs-FIXED-reference-set = cycle-robust
steer, dodging the running-best anchor-reset confound that inflated §D-FOUNDING's "100k lost 0.33");
cycle-aware abort (a high directed-3-cycle density = non-transitive equilibrium, NOT a regression →
suppress the STRENGTH abort, NEVER the robustness one); SealBot-WR → logged diagnostic; serial-eval
cross-game batching. **Fresh-context 4-lens REVIEW (`wf_8eb7e9d8`) PASS_WITH_CHANGES, 11 MAJORs — all
dispositioned in v2.** Two were resolved by MEASUREMENT (eval-only, local 4060): **M-VAR** —
batch-size FP variance persists under float32 (logit |Δ| 3.7e-3, not zero) but **0/32 argmax flips**,
so literal "bit-identical" is unachievable even on float32 (the dispatcher's premise corrected) →
replaced by an explicit **G1–G5 gate hierarchy** whose primary scatter proof is a deterministic
inference stub (FP-independent); **M-TP** — serial eval 9,252 games/hr @ GPU 53.3% (the bug confirmed
exactly); **M-CYC** — banked-ladder directed-3-cycle density **0.073** (38% pairwise inversions but few
full RPS triples) → `cycle_density_max` data-grounded.

**VERDICT — foundation BUILT + verified (PASS_WITH_CHANGES, all follow-ups resolved).** 5 features,
TDD, 0 commits (held), `make test` 1872 py PASS, `make bench` 10/10 PASS (MCTS floor **88,989 ≥ 73k**
— the pure-Python eval changes are off the benched Rust path, as predicted):
- **C1 round-robin primitive** (`hexo_rl/eval/round_robin.py` + `scripts/eval_round_robin.py`):
  registry-by-name 4-plane loader fix (`validate_arch_against_spec` guards in_channels==n_planes AND
  policy==policy_logit_count; `spec.name` stamped, not literal "v6"/"v8"); records full move lists +
  checkpoint steps + play command (the §D-FOUNDING per_game.jsonl lacked all three); emits win-matrix
  + BT-Elo + Copeland + median-rank + inversion-fraction + 3-cycle-density + Kendall-τ. **Reproduces
  §D-FOUNDING EXACTLY** (rr_pull: 2640 games, 25/66 inversions, s50k=87.5, s75k=100.0, s85k=101.4).
- **C2 robustness gate** (`hexo_rl/eval/robustness_gate.py`): config-keyed wrapper around
  `offwindow_probe.run_adversary_games` (single source, deterministic post-`a7ba110`); threshold 0.06
  fix-acceptance; hard-error on encoding mismatch; the ONLY instrument that sees the off-window defect
  (vs-SealBot false-clears).
- **C3 cross-game batched evaluator** (`hexo_rl/eval/eval_batcher.py`): generator/scheduler interleaves
  N games (own MCTSTree + per-game RNG each), one combined `infer_batch`, scatter by game-index. **G1
  deterministic-stub scatter proof byte-identical batched==serial** (parametrized N∈{8,16,32}); G3
  repeat-deterministic; **measured 2.24× games/hr (8,730→19,545), GPU 47→63.5%, |ΔWR|=0.000.** G5
  reseed-equivalence DEFERRED to Phase-3 banked data.
- **C4 steer/abort rewire** (`alert_rules.py` + `config.py` + `step_coordinator.py` + `eval_pipeline.py`
  + `gate_logic.py`): SealBot-WR DEMOTED to logged diagnostic (never feeds shutdown; honesty knob
  `sealbot_wr_revert_to_abort` default False); cycle-aware `check_strength_regression_abort` +
  `check_strength_warn` + robustness WARN/abort (never cycle-suppressed) + `check_objective_a_coverage`
  pre-flight (loud WARN if NO Objective-A signal active); promotion conjunction `decide_promotion`
  (PROMOTE iff strength_ok AND robustness_ok; fixed-ref aggregate REPLACES wr_best when present, else
  falls back; missing robustness = pass). Back-compat: no new signals → identical to legacy.
- **C5 §2.5 calibration** (`hexo_rl/eval/strength_calibration.py`): pre-registered separation method.
  `cycle_density_max`=**0.15** LOCKED. `strength_abort_floor` = **UNCALIBRATABLE on the §D-FOUNDING
  ladder** — healthy/post-peak per-rung Copeland OVERLAP because on-distribution strength is FLAT +
  non-transitive (the founding verdict, Objective B ill-posed). The method correctly REFUSED to guess
  → strength abort stays DISABLED, robustness gate carries Objective-A. This CONFIRMS the SealBot
  demotion: there is no on-distribution strength regression for any abort to catch.

**REVIEW+RED-TEAM (post-IMPL, `wf_f42ce70d`, fresh 5-lens, PASS_WITH_CHANGES).** Verified every claim
by RUNNING (re-derived §D-FOUNDING Elo from raw; confirmed loader stamps spec.name; G1 anti-trivial;
54+ tests). Two lens BLOCKs (strength path inactive) ADJUDICATED DOWN — honestly-disclosed
operator-gated follow-up, not a hidden defect (`decide_promotion` correctly falls back; result fields
guarded `is not None`). Genuine issues = hygiene/enforcement, FIXED: lost-signal pre-flight WARN +
single-eval strength WARN + revert-honesty knob + G1 N-sweep + G5 defer-note. Residual BLOCKER (the
pre-registered spec is untracked) = the operator's COMMIT decision (held per dispatcher).

**HONESTLY NOT DONE (operator-gated / coupled to at-power data):** the per-round Tier-B ref-set
strength PRODUCER (decision-4's data source — without it, strength_aggregate is never produced →
promotion uses wr_best fallback + robustness); the at-power strength floor (uncalibratable until a run
yields a separable collapse); **Phase 3 validation** (on-distribution plateau at power + off-distribution
FELL mechanism trace — the round-robin now records the moves it needs; heavy compute → operator-gated).
The Objective-A guard is CONFIG-GATED OFF by default — the operator MUST enable
`opponents.offwindow_adversary` before a live run (design §7; pre-flight WARNs if absent).

**FORKED roadmap (Phase 3 tees up; operator-owned, NOT actioned):** plateau-real → strength/
value-ceiling investigation; off-window-mediated FELL → multi-cluster encoding (S0/S1/S3);
opening-overfit FELL → opening-diversity. Report the fork; do not pick it.

**Lessons.** L (CLAUDE.md "verify the measurement unit/premise"): the dispatcher's literal
"bit-identical to serial" was UNACHIEVABLE (M-VAR: GPU batch-size FP variance survives float32) AND
conflicts with fixing the global-RNG concurrency hazard — corrected to a CPU-byte-identical +
GPU-determinism + statistical-equivalence hierarchy with a deterministic-stub scatter proof. L: a
calibration that REFUSES to fabricate a floor on inseparable data is the method working, and the
refusal is itself the finding (flat ladder ⇒ no strength-abort target ⇒ the SealBot abort was firing
on a non-strength signal). L: demoting a wrong abort without re-arming the right one is a lost-signal
trap — the robustness gate is config-gated OFF by default, so a pre-flight coverage WARN is mandatory.

Full: `docs/designs/D_EVALFOUND_design.md` (DESIGN v2 + both review dispositions). Code: tracked edits
to `hexo_rl/eval/{bradley_terry,checkpoint_loader,eval_pipeline,evaluator,gate_logic,result_types}.py`,
`hexo_rl/monitoring/{alert_rules,config}.py`, `hexo_rl/training/step_coordinator.py`; new
`hexo_rl/eval/{round_robin,robustness_gate,eval_batcher,strength_calibration}.py`,
`scripts/eval_round_robin.py`, 7 new test files. Measurements (local, untracked):
`investigation/evalfound_2026-06-08/{batch_variance_probe,batched_eval_measure}.py`. ALL COMMITS HELD
(operator-gated) — nothing committed; `git diff` = the 9 sanctioned tracked files + untracked new
modules/tests/design.

### §D-EVALFOUND Phase 3 — VALIDATE at power (eval-only, banked golong ladder) — 2026-06-09

> **⚠ PARTIALLY RETRACTED by §D-ARGMAX (2026-06-09, below).** The on-distribution **ARGMAX −109 Elo
> "CI-resolved late FALL"** in the table below is a **measurement artifact**: the t0_o0 cell has only
> ~2 effective (distinct) games/pair (deterministic argmax + no opening diversity → 40 byte-identical
> copies), so the raw n=800 BT-CI is over-confident by √40 = 6.32×; the honest deduped CI [−427,+208]
> **straddles 0**. **Objective B (on-distribution self-play strength) is FLAT / ill-posed — §D-FOUNDING
> stands.** The off-distribution **t05_o6 FELL (−40..−54, Objective A) is UNAFFECTED** (powered, 150
> distinct/pair) and remains the one real CI-resolved deficit. The "(2) on-distribution argmax fall →
> strength/value-ceiling investigation" fork arm is CLOSED. See §D-ARGMAX for the full reversal.

Ran the new round-robin primitive (+`opening_plies`) at power to resolve the two §D-FOUNDING open
questions. Compute: **vast 5080, 3-way pair-sharded** (the serial RR underutilizes the GPU; sharding →
~82%); contested rungs {50,75,90,100,112.5k}, sims=64; temp×opening **2×2** (temp{0,0.5}×open{0,6}),
n=1500 (temp0.5) / 800 (temp0). Host hygiene: cleared a stale `/tmp/tmux-*/default` socket (the
"wedged tmux"); synced code by **push+pull** (vast on `phase4.5/evalfound`), not scp. SealBot xval ran
on the local 4060. Aggregated with `aggregate_games` + the bootstrap slope; mechanism via
`analyze_recorded_game` (turn-correct completing-cell `pair[1]` unit). Data:
`reports/eval/phase3_20260609/` + `reports/eval/phase3_sealbot_xval_20260609/`; analyzer
`investigation/evalfound_2026-06-08/phase3_analyze.py`.

**VERDICT — two CI-resolved late regressions; §D-FOUNDING's "on-distribution FLAT / Objective B
ill-posed" is CORRECTED (it was a temp-0.5 sampling artifact).** The temp×opening 2×2 (late-rung Elo,
anchor s50k=0):

| cell | s75k | s90k | s100k | s112.5k | 3-cycle | verdict |
|---|---|---|---|---|---|---|
| on-dist T0.5 (t05_o0, n1500) | +34 | −3.7 | −26.6 | +14.9 | **0.30** | FLAT, non-transitive |
| on-dist **ARGMAX** (t0_o0, n800) | +75.6 | −36.3 | **−109.5** [−160,−59] | **−109.5** [−160,−59] | 0.00 | **CI-RESOLVED FALL** |
| off-dist T0.5 (t05_o6, n1500) | −19.6 | −37.2 [−73,−2] | −40.0 [−75,−5] | **−54.5** [−90,−19] | 0.00 | **CI-RESOLVED FELL** |
| off-dist ARGMAX (t0_o6, n800) | +1.7 | −8.7 | −37.5 | −29.6 | — | FELL (compressed by scatter) |

- **On-distribution: temperature-dependent.** temp0.5 → FLAT + **non-transitive** (3-cycle 0.30, slope
  −0.69/1k CI [−1.64,+0.14]). temp0 (ARGMAX, best-play) → a **CI-resolved ~109-Elo late FALL**
  (100k/112.5k vs 50k, transitive 3-cycle 0.00). **Temp-0.5 sampling masks a real argmax regression
  AND injects the non-transitivity.** → §D-FOUNDING "Objective B ill-posed" CORRECTED: at best-play
  there IS a resolved late strength regression. (CLAUDE.md re-validate-unit: temperature is a
  load-bearing measurement choice, not secondary — the founding's temp-0.5-only on-dist read was
  incomplete; it lacked the temp0×open0 cell, added here.)
- **Off-distribution: CI-resolved FELL** (open6, temp0.5): 90/100/112.5k = −37/−40/−54 Elo, CIs exclude
  0, transitive (3-cycle 0.00). Opening randomization is the lever (confirms §D-FOUNDING 1b at power).
- **MECHANISM (the novel deliverable): off-distribution losses are ~96% OFF-WINDOW-MEDIATED.**
  Loser-side off-window-forced-turn rate (completing cell `pair[1]`, turn-correct) = **0.96 pooled**,
  **uniform** across checkpoint (late 0.96 / early 0.96, both t05_o6 + t0_o6) and opening-distance bin.
  → **Branch A (off-window → multi-cluster) confirmed; Branch B (opening-diversity, criterion <0.15)
  REFUTED.** The off-window single-window action blind spot is the STRUCTURAL off-distribution failure
  mechanism (uniform, not late-specific) — matching the SealBot xval below.
- **SealBot cross-validation (§1c, demotion safety):** `exploit_probe` off-window forced-win on banked
  {50,75,90,100,112.5k}: all **FORCEABLE 0.20–0.275** (margins +0.15..+0.24), **flat across the
  wr_sealbot 0.38→0.05 crash** (reproduces EXT-LINK 50k 0.225≈0.235). → off-window defect is
  structural/persistent + checkpoint-independent; **demoting SealBot-WR loses NO off-window signal**
  (exploit_probe is its deterministic superset), and the wr_sealbot collapse is **NOT off-window-
  mediated** (exploit_probe flat across it) — reinforcing that wr_sealbot was a confounded matchup
  signal, not a clean Objective-A meter.

**FORK (resolved data; operator-owned, NOT actioned) — BOTH arms now live:**
1. **Off-distribution off-window FELL → multi-cluster / K-window ENCODING (S0/S1/S3).** Strongest
   evidence yet: 96% of off-distribution losses are off-window-mediated; the single-window action
   blind spot IS the mechanism. (S1 remains the dominant residual risk, §174 ×3.)
2. **On-distribution ARGMAX fall (−109, in-window — NOT off-window) → strength/value-ceiling
   investigation.** Objective B is **NOT** ill-posed: the late checkpoints genuinely regressed at
   best-play (sampling masked it). A distinct mechanism from the off-window FELL.

**Lessons.** L (CLAUDE.md re-validate-unit, again): the temperature of an Elo measurement is
load-bearing — temp-0.5 sampling compressed a CI-resolved −109 argmax regression to "flat" AND
manufactured a non-transitive cloud (3-cycle 0.30→0.00 at argmax). The founding's "FLAT, Objective B
ill-posed" rested on the temp-0.5-only on-distribution read; the missing temp0×open0 cell inverts it.
L: the mechanism trace's turn-correct completing-cell unit (`analyze_recorded_game`, depth-1 ∪
depth-2-`pair[1]`) was essential — a depth-1-only detector would have undercounted the off-window
forced-win set ~86.5% and could not have established the 96% off-window attribution. Compute: vast
ssh background/tmux were wedged (stale socket + missing `ssh -n`/`ControlPath=none`); foreground ssh +
tmux (post-socket-fix) is the working pattern; push+pull beats scp for code sync.

## §D-ARGMAX — diagnose the in-window ARGMAX −109 regression → it is a measurement artifact — 2026-06-09

Dispatched to diagnose the §D-EVALFOUND Phase-3 "CI-resolved ~109-Elo in-window ARGMAX strength
regression" as the DOMINANT deficit and resolve the unification fork (does B unify with the
single-window encoding → multi-cluster fixes both, or is B distinct). **VERDICT — the −109 is a
LOW-EFFECTIVE-POWER / over-confident-CI ARTIFACT; Objective B (on-distribution self-play strength
regression) is NOT real; the fork DISSOLVES — one real deficit remains (A, off-window).** EVAL-ONLY,
read-only (`git diff` empty); all artifacts under gitignored `investigation/argmax_2026-06-09/`.
Verified by a fresh-context red-team (NOT the implementer): **SOUND**.

**PHASE 1 — CONFIRM: the −109 does NOT survive at power.** The §D-EVALFOUND Phase-3 read reproduces
exactly (t0_o0 argmax: s100k=s112.5k=−109.5 [−160,−59]\*, transitive). But the power is fake:
**t0_o0 (argmax, opening_plies=0) has exactly 1 distinct game-sequence per directed pair — 40
byte-identical copies; true effective n = 20 games total, ~2/pair.** temp=0 + no opening diversity ⇒
deterministic self-play ⇒ games repeat (the 60-20 splits = {one color→draw, other→win}, ×40).
Dedupe to distinct sequences → **s100k=−109.5 [−427,+208], STRADDLES 0**; the raw n=800 BT-CI is
inflated **exactly 6.32× = √40** (textbook pseudo-replication; L23/§176 generalised). The −109 is NOT
CI-resolved at its real sample size. Corroborating: **sims=128 INVERTS** the ranking (s100k strongest
+148, also 2 games/pair); a deterministic t0_o0 game **flips winner across GPUs** (4060 vs banked 5080,
CUDA-float) after the shared opening — not even cross-hardware reproducible. The properly-powered cells
are FLAT: t05_o0 (sampled, 150 distinct/pair) FLAT; **t0_o6 (argmax + 6 opening plies = 80 distinct/pair,
real power) FLAT** — every rung straddles 0 (s100k −37.5 [−85.8,+10.9]). The ONLY CI-resolved,
properly-powered late regression is OFF-distribution: **t05_o6 (sampled, opening-6) s90k/100k/112.5k
= −37\*/−40\*/−54\*** = the off-window FELL (Objective A, §D-EXTLINK). → §D-FOUNDING's original
"Objective B ill-posed / FLAT" was CORRECT; §D-EVALFOUND Phase-3's reversal was itself the artifact.

**COMPONENT LOCALIZATION (valid as STYLE, not a strength driver).** Fixed-pool head scoring over the
full ladder (in-window forced-win turn-starts, n=7498): no head/search collapse — POLICY top1
−0.02..−0.04 (entropy de-sharpens@100k, RECOVERS@112.5k); VALUE won-vs-lost AUC healthy 0.84–0.88;
SEARCH MCTS-lift +0.05..+0.10 STABLE. Largest fixed-pool effect is DISTRIBUTIONAL — late-checkpoint-
sourced positions ~0.10–0.14 harder to finish in-window, STONE-MATCHED (not depth); mechanism: late
in-window wins are THIN/UNDER-SUPPORTED (local own-stone support 2.04→1.35), genuine finishing-failure
~10%→34% (NOT a benign multi-win artifact). REAL position-structure facts, but they co-move with FLAT
powered strength (§D-FOUNDING) = a style/finishing-efficiency correlate, NOT a net strength regression.

**PHASE 2 — MECHANISM (all 4 arms OUT, coherent with "no real B").** A1 encoding-unification **OUT,
fork_collapses=FALSE** (centering churn loser_late−early +0.004 CI[−0.007,+0.015] FLAT; the game-level
churn-LIT was a SHORT-GAME blowout confound — paired loss-gap −0.039 at argmax, boundary-pressure
INVERTED). A2 value-ceiling **OUT as driver** (conc-AUC ~0.40 <0.70 ✓ but FLAT, 75k→112.5k +0.009
CI[−0.012,+0.031]; a constant PERMITS, cannot drive — D4-constancy; confirms §D-GLOBALCONC absence at
full ladder incl. late). A3 de-sharpening **OUT** (argmax −0.034\* + sampled −0.026\* fall TOGETHER, gap
narrows, entropy recovers; L9 cosine-temp = draw-collapse context, does NOT transfer). A4 over-spread
**OUT as global driver** (matched-buildup WINNER more-spread; game-end loser-spread = short-game
blowout; global frag doesn't predict finishing — replicates §D-FOUNDING length-confound at argmax).

**UNIFICATION ANSWER — the fork DISSOLVES.** A1 OUT (no centering link) AND the −109 is not a real
deficit → there is no "dominant B" to route around. ONE real, CI-resolved, properly-powered deficit:
**A — off-distribution / off-window FELL (Objective A, −40..−54 Elo, t05_o6)**. The sustained late
SealBot decline is NEITHER on-distribution strength (B, artifact) NOR off-window-mediated (exploit_probe
flat 0.20–0.275, §D-EVALFOUND xval) → confounded matchup-WR (§D-FOUNDING), not a strength canary.

**ROUTING — DESIGN only, operator-gated.** (1) **B: nothing to fix** (artifact); do NOT open a
strength/value-ceiling lever; the thin-wins style finding does not justify one (co-moves with flat
strength). (2) **EVAL-METHOD fix (the real Phase-3 deliverable, no training/Rust):** argmax
(deployment-regime) strength CANNOT be measured by a deterministic round-robin from a fixed opening
(~2 effective games/pair, over-confident BT-CI) — the steer instrument
(`hexo_rl/eval/round_robin.py`) must inject independent variation (on-distribution opening jitter OR a
diverse opponent panel) AND bootstrap the CI over DISTINCT games (dedupe copies first); add a
distinct-sequence-count / effective-n guard + warning. (3) **A: off-window / multi-cluster lever**
(Objective A) on the properly-powered off-distribution + §D-EXTLINK adversarial basis (exploit_probe
≤0.06, NEVER SealBot-WR), NOT the −109 — clean Rust path (registry.toml source-of-truth,
registry-by-name, zero literals, `make bench` ≥73k sim/s); **S1 still the dominant residual (>50% fail,
§174×3)**; the 30k SIGNATURE smoke must use the powered/dedupe-bootstrap instrument or it repeats the
artifact.

**LESSONS.** **L (promote, CLAUDE.md candidate): a self-play strength CI's effective sample size is the
number of DISTINCT games, not the game count** — deterministic regimes (argmax/temp-0 + fixed opening)
collapse to ~2 games/pair, and a BT/Wilson CI over the raw count is over-confident by √(copy-multiplier)
(here exactly √40 = 6.32×; L23/§176 generalised). Verify the distinct-game count before trusting any
strength CI; a "CI-resolved" Elo gap on a deterministic round-robin is the over-confidence trap, not a
finding. Corollary to "verify the measurement unit/premise": temperature is load-bearing AND so is
opening/opponent DIVERSITY — the argmax deployment regime needs injected variation to be measurable at
all. **L:** a deterministic argmax game can flip winner across GPUs (CUDA-float) — single-config argmax
outcomes are not strength statements. **Falsified-register: §D-EVALFOUND Phase-3 "on-dist ARGMAX
CI-RESOLVED FALL −109.5, transitive" RETRACTED** (counted 40 copies as independent; honest CI [−427,+208]
straddles 0). §D-FOUNDING "Objective B ill-posed/FLAT" RE-AFFIRMED. The "−109 → strength/value-ceiling"
fork arm is CLOSED; off-window/multi-cluster (Objective A) is the only surviving live arm.

Full: `investigation/argmax_2026-06-09/SYNTHESIS.md` + `REDTEAM_reversal.md`. Instruments (local,
gitignored): `component_localize.py` (+component_all/_inwindow.json), `distributional_confound.py`,
`distributional_mechanism.py` (+json), `search_arm.py` (+json), `a1_centering*.py` (+json),
`a2_value_conc.py` (+json), `a3_desharpen.py` (+json), `a4_overspread_argmax.py` (+json),
`sims128_redteam.py` (+sims128_check/). ALL COMMITS HELD (operator-gated).

## §D-STRENGTHAXIS — fix the strength instrument, then answer the hyperparameter question — 2026-06-09

The strength axis flip-flopped THREE times (FOUNDING flat → EVALFOUND −109 → ARGMAX flat) for one root
cause: the eval instrument could not measure argmax strength (deterministic argmax from a fixed opening
pseudo-replicates to ~2 effective games/pair → over-confident BT-CI either way). Fix the instrument,
then test whether there is even a hyperparameter-shaped problem before any hyperparameter hunt.
EVAL/CONFIG-only (Phase 1 eval-path Python off the benched Rust hot path; Phases 2–3 read-only; Phase 4
design-only). Full writeup `investigation/strengthaxis_2026-06-09/SYNTHESIS.md`. COMMITS HELD.

**PHASE 1 — the fixed steer instrument (`hexo_rl/eval/round_robin.py`, TDD, +12 tests, `make test.py`
1904 passed / 0 failed; no Rust → no bench).** (1) dedupe byte-identical `(p1,p2,moves)` sequences
(`distinct_game_key`/`distinct_games`; no-move records cannot be claimed as copies — legacy-safe); (2)
**game-level (cluster) bootstrap CI over DISTINCT games** (`bootstrap_ratings_ci`, wired into
`aggregate_games(n_boot=…)` per-rung `ci_*_boot`; `aggregate_to_dir` defaults 1000) — copies cannot
narrow it; (3) **effective-n / distinct-per-pair guard + WARN** (`effective_n_guard`: `copy_multiplier`,
`distinct_per_pair_min`, `low_power_warning`, gated on move-data presence; always emitted by
`aggregate_games`); (4) **on-distribution opening jitter** (`opening_jitter_plies`/`_temp`: the player's
OWN model plays the opening at a sampling temp then argmax — breaks argmax determinism WITHOUT
scattering the bbox off-window; mechanically distinct from the uniform `opening_plies`, which the
measured evidence shows is the OFF-distribution / Objective-A scatter — but jitter is
CHECKPOINT-CONDITIONAL: in-window only insofar as the model's own opening policy is, so a
spread/off-window-specialized checkpoint could drift jitter off-window; (4) is unit-tested for ROUTING
only, NOT run-validated — verify the jitter-region bbox span in the future Objective-A smoke).
**VALIDATE on banked Phase-3
cells:** reproduces the raw Hessian read EXACTLY; the guard fires only on t0_o0 (copy_mult 40, WARN);
the honest bootstrap CI **straddles 0 for on-dist argmax** (the −109 dissolves), is FLAT for the
powered cells (t05_o0, t0_o6), and resolves only the off-distribution **t05_o6 FELL (−40..−54 =
Objective A)**. → §D-ARGMAX / §D-FOUNDING "on-distribution FLAT at power" CONFIRMED on the productionised
instrument; every future strength/abort call now dedupes copies + bootstraps over distinct games.

**PHASE 2 — plateau vs equilibrium (pre-registered, fixed instrument, §D-FOUNDING full 12-rung ladder
35k→112.5k temp0.5 2640g).** EARLY (≤75k) slope **+2.80 [+1.6,+4.1]** EXCL-0 (real climb to a
peak ~75–85k); LATE (≥75k) slope **−1.37 [−2.5,−0.2]** EXCL-0 (**CI-resolved DECLINE from the peak**);
FULL +0.53 [+0.17,+0.89]; 3-cycle density **0.073** (< 0.15, NOT an RPS cycle); inversion 0.379.
**Pre-registered binary = AMBIGUOUS** (E1 STALL fails on inversions; E2 EQUILIBRIUM fails on the low
3-cycle + non-straddling full slope). Per the probe-smoke-verdict discipline thresholds are NOT moved —
the ambiguity is the finding: a **RISE-to-peak-then-LATE-DECLINE**, neither flatten-and-hold nor a
non-transitive cloud. Relative to s50k the END ≈ the early-trained level (§D-ARGMAX powered cells
straddle 0); the decline is the drop FROM the peak (≈40–60 Elo, CI-resolved at temp0.5; an unresolved
~40-Elo point drop at the powered argmax cell t0_o6). It is NOT a plateau to push past, and it occurred
at HEALTHY LR — a constant/healthy hyperparameter cannot manufacture a time-localized inflection.

**PHASE 3 — Dirichlet/temp/LR provenance + sanity (parallel audit + adversarial verify; `wf_129aa8df`).**
All SANE, all NOT_IMPLICATED. **Dirichlet** 0.05/0.10 (config_key, HTTT-tuned §115/§116/§143 per the
config comment; alpha landed §118): 10/N against
the ACTUAL windowed root n_children (~25 opening, 216–347 mid/late = 0.029–0.046) puts 0.05 at/above the
mid-game band; §156 R11 (eps→0 NULL) + §S181-T3 (alpha×4 + no_noise < 3pp) falsify a Dirichlet stall.
**Temperature** fixed τ=0.5, cosine OFF (config_key, §156 R12 / L9): the large branching is the REASON
the cosine floor is dangerous (R12 cosine-on → 91% draws); constant, applied to MCTS-improved visits.
**LR** 2e-3/eta_min 5e-4 cosine (mixed; KataGo never-below-0.5× plasticity floor, raised 2e-4→5e-4
§S181 PR-B): the §D-STRENGTHAXIS "cosine collapses LR by 50–75k → plateau" hypothesis is FALSIFIED **by
direct live-log measurement** — the documented segment-1 footgun (launch omitted
`--override-scheduler-horizon` → T_max baked at 30k → LR toward eta_min by ~50–53k) was fixed at the
**53k restart re-warm to 2e-3 BEFORE the peak/decline region**; segment-2 LR stayed high
(1.971e-3@75k, 1.919e-3@90k); the §D-FRAGILITY 75k event was at FLAT ~1.97e-3 LR, not eta_min.

**PHASE 4 — ROUTING (DESIGN only, operator-gated).** GATE (lever IFF Phase2=STALL AND Phase3 implicated)
NOT satisfied: Phase 2 is not a clean stall and Phase 3 implicates nothing → **no hyperparameter/LR
lever for on-distribution strength.** The **LR-for-long-runs schedule is HYGIENE not a strength lever**
— prevent the segment-1 footgun on any fresh run (a clean single cosine over the true 0→N horizon
avoids the rewarm-anchor hack; plasticity floor already in place) — but it was fixed before the region
and the decline occurred at healthy LR, so it is not the cause. **The only live lever is Objective A —
off-window / multi-cluster** (S0 clean-Rust 362-multiwindow registry-by-name zero-literals
`make bench`≥73k; S1 the dominant residual >50% fail §174×3; S3 exploit_probe ≤0.06), justified on the
powered off-distribution FELL + the §D-EXTLINK adversarial basis, NEVER the −109 or SealBot-WR. Any
Objective-A 30k smoke MUST use the Phase-1 fixed instrument (dedupe-bootstrap + effective-n guard) or
it repeats the pseudo-replication artifact — and if it uses the on-distribution opening-jitter control,
it MUST first verify the jitter-region bbox span stays in-window for that checkpoint (the
checkpoint-conditional caveat above).

**Lessons.** L: the three-reversal recursion is closed by fixing the INSTRUMENT, not by a fourth
cause-hunt — a steer signal that cannot measure the deployment (argmax) regime at power will whipsaw
every call that depends on it. L: a pre-registered binary that returns AMBIGUOUS is data, not failure —
the rise-peak-decline is a real third pattern; do not move thresholds to force a verdict. L: "the LR was
misconfigured" (segment-1 footgun, TRUE) and "the LR caused the dynamics" (FALSE — fixed before the
region; decline at healthy LR) are different claims; a live-log LR trajectory separated them. Full:
`investigation/strengthaxis_2026-06-09/{SYNTHESIS.md,validate_instrument.py,phase2_plateau_vs_equilibrium.py}`;
Phase-3 audit `wf_129aa8df`. ALL COMMITS HELD (operator-gated).

**§D-LAND-EVALFOUND — LANDED to master 2026-06-09.** Branch `phase4.5/evalfound` FF-merged to master
(no merge commit); master HEAD **`ac628e7`** (was `e7bbee7`). Landed = the 9 already-pushed D-EVALFOUND
commits (`bbfc493..920b387`: RR primitive + robustness gate + cross-game batched evaluator = the
serial-eval GPU-~50% fix + steer/abort rewire + calibrate + opening_plies) **plus** the 3
§D-STRENGTHAXIS commits previously held: **`af22d09`** feat(eval) fixed steer instrument
(dedupe-bootstrap CI + effective-n guard + opening-jitter, 12 tests), **`e286ec8`** docs(sprint-log)
(§D-ARGMAX artifact verdict + Phase-3 RETRACTED banner + this §D-STRENGTHAXIS entry), **`ac628e7`**
docs(claude-md) effective-n lesson. Gate: `make test` green (1904 py passed / 0 failed + Rust, no
`test_shape_fallback` flake); clean 3-commit bisect, conventional, no Co-Authored-By; FF (master linear
ancestor). Adversarial pre-merge verify (`wf_b49587ae`, 4 read-only auditors) caught a message off-by-one
("13 new tests" → corrected to 12; the §D-ARGMAX irony noted) — reworded before merge. EVAL-path Python
only, zero Rust/hot-path → no `make bench` needed.

**RUN-READINESS FLAGS (pre-RUN items, NOT merge-blockers — for the run-design phase, NOT changed here):**
(1) **Robustness gate is config-OFF by default** — `opponents.offwindow_adversary` MUST be enabled
before any live run (the pre-flight WARN exists); default intentionally NOT flipped in this merge.
(2) **Opening-jitter is routing-tested but run-UNVALIDATED** (jitter_count=0); on-distribution-ness is
checkpoint-conditional — before any use, add/validate the bbox-span pre-flight (jitter span < the
checkpoint's uniform-scatter span). Do not rely on it yet.
(3) **Tier-B ref-set strength producer still deferred** — promotion uses the `wr_best` fallback +
robustness gate until it lands.

