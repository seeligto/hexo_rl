# D-RUN2 — first multi-window production run, FRESH FROM BOOTSTRAP: DESIGN

Date: 2026-07-03. Status: **DESIGN ONLY — not launched.** Launch is a separate operator call, gated
on the D-WS3V3 v3 smoke qualifying the seeding/injection machinery (`docs/handoffs/d_ws3v3_smoke_runbook.md`).
Branch: `phase4.5/d-solver`. This doc is written to the working tree, uncommitted (operator commits on ask).

Companion: `configs/variants/run2_mw_fresh.yaml` (the run's variant config; §4 below maps every
non-base key to the reasoning that put it there).

---

## 0. Status / premise

`§D-FORENSIC F1` (`memory/d-forensic-f1-lineage-single-window-cascade.md`) found the entire d1m
lineage (the run this design supersedes as "the baseline") self-played **single-window** its whole
life — a string-form `encoding:` bug plus checkpoint-metadata self-perpetuation silently downgraded
every declared `v6_live2_ls` (multi-window legal-set) launch to plain `v6_live2` (single-window,
off-window-drop) self-play. `§D-LADDER`'s TRUE-STALL verdict (120k→226k flat, deploy-matched
Gumbel@150 self-ladder) is **re-scoped to characterize the single-window regime only** — it is
context, not a ceiling, for a genuinely multi-window run.

**Consequence for this design:** there is no clean control. A warm-start from any existing
checkpoint inherits up to 272k steps of crippled-action-space training (single-window self-play,
off-window wins dropped at 5 layers — `docs/designs/dmulticluster_362_legalset_design.md` §2.1).
A fresh bootstrap run is therefore the *less* confounded experiment, not a decision to throw away
sunk cost. This is the new baseline; comparisons against d1m are **context, never evidence**
(§8 restates this as a hard rule).

**Deliberate bundling.** Per dispatcher instruction, this run bundles several validated-but-never-
jointly-run components (§1), makes seven explicit config-level decisions with pre-registered checks
(§2), and lands three restart-owed engine/eval additions while the run is fresh anyway (§3). Bundling
sacrifices clean single-variable attribution. §8 pre-commits the post-run ablation order so that cost
is paid consciously, not silently.

**Load-bearing correction found while researching this design (§1.6):** the `legal_move_radius_jitter`
mechanism, which CLAUDE.md's component inventory treats as a validated, LOCKED anti-colony pairing
partner for the (already-OFF) cosine-temperature schedule, is **dead code** for every registry-spec
encoding — which today means **every** encoding, including plain `v6` — since the §172/§173 registry
migration made `encoding_name` non-optional end to end. This is argued from source + a named
regression test in §1.6; it changes the radius-diversity story for this run (§2 Decision 2) and the
LOCKED-list wording (§1).

---

## 1. LOCKED components (validated; not up for re-litigation this run)

### 1.1 Multi-window legal-set self-play (first ever at production scale)

`v6_live2_ls` (`engine/src/encoding/registry.toml:325-348`) is the encoding: 4-plane wire-identical
to `v6_live2` ([0,8,16,17], 362-logit head, `v6_live2` weights load with **no reshape**), but
`is_multi_window=true`, `cluster_threshold=5`, `k_max=8`, `policy_pool=legal_set_scatter_max`. The
ragged legal-set fix (`docs/designs/dmulticluster_362_legalset_design.md` §9, "Design-B,
AUTHORITATIVE") is a Rust-internal global intermediate (`LegalSetPolicy{dense, overflow}`) — it never
crosses PyO3, the buffer stays dense-362-per-cluster-row, symmetry/persist/trainer are **unchanged**.
Landed end-to-end: commit `5c90988` "feat(mcts,game_runner): wire legal-set ragged path (v6_live2_ls)
end-to-end". This is the FIRST production run to self-play multi-window from step 0 — d1m never did
(§0), so there is no prior sustained-scale evidence for this mechanism's own training dynamics; that
absence is itself part of the premise, not a defect of this design.

### 1.2 Encoding gates + `--expect-encoding` everywhere + stamped anchors

`D-EVALGATE` (commit `9729e5b`) drew the two-semantic line: `--encoding` = decode-time override
(loud, never raises), `--expect-encoding`/`declared_encoding` = assertion (raises on stamp
disagreement). `da29fa1` wired `--expect-encoding` into the L1 trap-flip evaluator; `4d573c0` made
`probe_threat_logits.py` stamp-aware and refuse silent cross-fixture fallback; `a33f7df` (D-WS3V3 A1)
re-baselined and closed the loop with the HOST-MATCH rule (§1.5). `402bbed` declared
`bootstrap_anchor.encoding` in every variant that repoints the anchor path. This run's yaml declares
`encoding: v6_live2_ls` at the top level and pins `bootstrap_anchor.encoding: v6_live2` (the anchor's
own stamp) per the D-EVALGATE deep-merge warning in `configs/eval.yaml`.

### 1.3 R3 LOSS guard

Native `engine::tactics` LOSS-side proof soundness (emergent-sound, prune symmetry) confirmed active
at all tiers: `engine/src/tactics/search.rs:253-270` + recall-verify (`§D-FORENSIC F2` closeout,
2026-07-03, sprint log line 2785). This run's solver-in-loop lever (if activated, §2 Decision 3) uses
**WIN-only** injection (no LOSS-derived z; the R3 guard's relevance is to a future Track-3
quiet-move-widening body that is explicitly NOT in this run — `memory/d-reconfirm-r3-loss-soundness.md`).

### 1.4 Mixing schedule canonical

`decay_steps: 200_000`, `initial_pretrained_weight: 0.8`, `min_pretrained_weight: 0.1` — the
`w_pre = max(min_pretrained_weight, initial_pretrained_weight · e^{-step/decay_steps})` formula
(`configs/training.yaml:169-188`). This is the value every `v6_live2_ls` production variant to date
has used (`v6_live2_golong.yaml`, the D-1M `longrun_v6_live2_ls_gumbel_m16.yaml`, all `ws3v3_arm_*`).
Kept as-is; not re-opened.

### 1.5 HOST-MATCH baseline protocol

`docs/handoffs/d_ws3v3_smoke_runbook.md` §0.6: the baseline any arm/checkpoint is compared against
must come from the **same host** as the read (a laptop-vs-vast discriminator run found ~1-2 flip
scatter attributable to floating-point host drift, not signal). This run's §5/§6 gates that compare
against a fixed reference (threat-probe anchor, trap-flip baseline) must be re-minted on whichever
host actually runs the gate before trusting the comparison.

### 1.6 LEGAL_MOVE_RADIUS jitter — CORRECTED, not simply "locked"

**Original claim (dispatcher inventory):** "LEGAL_MOVE_RADIUS jitter (mandatory pairing, L9 register:
NO cosine temp)" is validated and LOCKED. **Finding from this session's source read:** the jitter
mechanism is dead code for this run's encoding, and in fact for every encoding in the current
codebase. Evidence, file:line:

- `engine/src/game_runner/worker_loop/inner.rs:630`:
  `if init_ctx.legal_move_radius_jitter && worker_registry_spec.is_none() && ro < 0 { ... JITTER_RADII = [4,5,6] ... }`
  — jitter fires only when `worker_registry_spec.is_none()`.
- `engine/src/game_runner/mod.rs:334-344`: `spec_static` (which becomes `worker_registry_spec`) is
  `Some(spec)` whenever `encoding_name` resolves in the registry — and `hexo_rl/selfplay/pool.py:134,
  201` show `encoding_name` is now a **non-optional** `str` always set to `spec.name`, for every
  encoding including plain `"v6"` (which is itself a registry entry, `registry.toml:58-82`, since the
  §172/§173 registry migration). There is no special case anywhere in the call chain
  (`mod.rs:465` → `worker_loop/mod.rs:262` → `inner.rs:344,377`) that forces `worker_registry_spec`
  back to `None` for the "default" v6 shape.
- `engine/src/game_runner/mod.rs:1036-1039` (comment on a retired test):
  *"Jitter behaviour is now guarded by `worker_loop.rs:321` `registry_spec.is_none()` branch ...
  registry_spec is the surviving sentinel for 'non-default perception bound'."* The named test,
  `test_worker_loop_jitter_does_not_override_v6w25_radius`, was **removed outright** in the §P3.2
  test retirement — verified at review: neither replacement test
  (`engine/tests/test_worker_loop_v6w25_smoke.rs`, `engine/tests/inv17_pyregistryspec_supersedes_pyencodingspec.rs`)
  contains any jitter reference. The retirement comment documents the guard's intent, but the
  jitter-inertness behavior is currently UNTESTED (coverage gap — Open Questions).

**Read:** this guard's *original intent* (per `engine/src/board/state/core.rs:299-307`,
`§171 P3 e6682f6` precedent) was "don't let the legacy v6 jitter mechanism silently stomp a
registry-bound encoding's own canonical radius (v6w25, v8, ...)." Its *actual effect*, after the
Wave-8 migration made `encoding_name` unconditional, is that **jitter never fires for any run
launched today**, including the plain-v6 runs the mechanism was originally written for. `vast.yaml`
and `v6_sustained.yaml` both set `legal_move_radius_jitter: true` and have run this way for months —
their per-game radius has, in all likelihood, been static at whatever `legal_move_radius_schedule`
(if any) or the registry's static field dictates, not jittered r∈{4,5,6} per game as their own
comments claim.

**What still works:** `override_legal_move_radius` (`core.rs:313-322`, "Works WITH encoding" —
written explicitly for training-time curriculum scheduling) is **not** guarded by
`registry_spec.is_none()` (`inner.rs:620-623`) and does fire under `v6_live2_ls`. The
`legal_move_radius_schedule` config key (read by `hexo_rl/training/step_coordinator.py:502-514`,
`_resolve_radius`) is therefore the **only live per-run radius-diversity mechanism** available on
this encoding today.

**Action taken in this design:** downgrade the LOCKED-list wording from "jitter, mandatory pairing"
to "a live radius-diversity mechanism, mandatory pairing with cosine-temp-off" and route it through
`legal_move_radius_schedule` (§2 Decision 2) instead of `legal_move_radius_jitter`, which this run's
yaml does NOT rely on. **Open question, not resolved here** (§ open questions at the end of this doc):
is this a genuine multi-month latent regression on `vast.yaml`/`v6_sustained.yaml` worth its own
bug-fix ticket, or was there a compensating mechanism this 40-minute source read missed? A 20-game
smoke (log `board.legal_move_radius` per game with `legal_move_radius_jitter: true`,
`legal_move_radius_schedule` **absent**, on any registry-spec encoding) would settle it in minutes
and should be run once, cheaply, before or alongside this launch — it does not block this design.

### 1.7 Opening-diversity injection for distinct-game eval

`§D-ARGMAX` corollary (CLAUDE.md "Verify the measurement unit..." section): a deterministic deploy
regime (argmax/temp-0 Gumbel-SH) collapses to ~2 distinct games per opponent pair without injected
opening variation; a raw-count Bradley-Terry/Wilson CI is then over-confident by `sqrt(copy_multiplier)`.
`hexo_rl/eval/deploy_strength_eval.py` implements this correctly (RNG-seeded opening plies,
color-balanced, byte-identical-game dedup before bootstrap CI, `effective_n_guard` /
`min_distinct_per_pair` floor). This run's `eval_pipeline.opponents.deploy_strength` block (§2
Decision 4, §4) is pinned on from step 0 specifically so the 100k+ deploy-matched slope read (§5) has
this protection built in rather than retrofitted after a stall is already suspected.

---

## 2. DECIDE — seven decisions, each with a pre-registered check

### Decision 1 — Dirichlet: OFF (banked, already engine-level, not a config knob this run turns)

**Re-validation (CLAUDE.md protocol).** Cite: `§D-DECIDE` Track-B B1, "Dirichlet-off on the Gumbel
branch" — flagged in the dispatcher's own status doc as "**DONE**... Restart-only; live run already
has `dirichlet_enabled:false` so unaffected" (`docs/handoffs/d_decide_track_a_status.md` line 145-153).
**Context it was validated in:** removing a *redundant* Dirichlet call stacked on top of Gumbel-Top-k
root exploration (double root-noise, ~2× completed-Q policy-target-floor inflation), landed as an
**engine-level code deletion**, not a config flag — commit `f4413d6` "refactor(mcts): drop redundant
Dirichlet on the Gumbel root". Verified still present on this branch
(`engine/src/game_runner/worker_loop/inner.rs:889-892`: *"NO Dirichlet root noise under Gumbel...
Dirichlet lives in the PUCT `else` arm only"* — unconditional, no flag check on the Gumbel branch at
all). **Transfer test:** this run uses `gumbel_mcts: true` throughout (§4), the exact regime the fix
targets; `mcts.dirichlet_enabled: false` is set in the yaml purely for hygiene/documentation (the
Gumbel branch does not read it), matching every other `v6_live2_ls` variant's convention.
**Decision: OFF, structurally guaranteed by the engine, not a run-time bet.** No pre-registered check
needed beyond confirming `cargo test -p engine` (326/0/2 per the B1 commit) is still green at launch
time — a one-line precondition in §10, not a monitoring gate.

There is a second, narrower Dirichlet-adjacent question the dispatcher's inventory conflates with
this: **entropy floor / policy-target-floor inflation from the OLD stacked Dirichlet**, which this
run never sees (it launches fresh on the already-fixed engine). The early-run entropy/diversity gate
the dispatcher asks for (per its own item-1 framing) is retained anyway as a general health check, not
an abort-to-ON switch (there is nothing to turn back on — the code path is gone):

- **Pre-registered check:** entropy `H(π)` at steps 1k/5k/25k. Read cadence: every
  `training_step` log line (`log_interval: 10`), watched at the 5k/25k gate reads (§5).
  **Band (corrected at review):** the historically-quoted "~3-6 nats" (sprint log §1, the
  entropy-regularisation note near line 77 — NOT `configs/training.yaml`, which carries no band)
  contradicts MEASURED healthy-run history: the 148-stat audit banked `policy_entropy` at
  **min 2.141 / max 2.889 / mean 2.455 (n=76)** on a healthy Gumbel run
  (`docs/audits/stat_audit_2026-06-23/bucket_B1.md` § policy_entropy), and `entropy_reg_weight`
  has been halved 0.01 → 0.005 (§S181 PR-B) since the 3-6 band was minted — the band was never
  re-confirmed post-halving. Operative thresholds are the live monitoring ones
  (`hexo_rl/monitoring/config.py:35-36`): **warn < 2.0 nats, collapse alert < 1.0 nats**.
  Healthy expectation for this run: **~2.1-2.9 nats**; KILL floor stays **< 1.0** (§6).
  A stable band systematically below 2.1 is drift-to-investigate (plausibly the weight halving),
  not an automatic KILL.
  **Action on breach:** if entropy floors below 1.0 nat and stays there past 3 consecutive eval
  points, this is a genuine new collapse signature (not explained by the removed-Dirichlet fix, since
  that fix *reduces* an inflation, it does not introduce a new collapse mode) — treat as a KILL
  condition per §6, not a re-enable-Dirichlet reflex (there is no config knob to flip; investigate
  `entropy_reg_weight` / `policy_prune_frac` instead).

### Decision 2 — Radius curriculum: IN, staged, via `legal_move_radius_schedule` (NOT jitter)

**Re-validation.** Cite 1: `§174` "LEGAL_MOVE_RADIUS 8→5 at bootstrap fixes v6w25 selfplay collapse"
— FALSIFIED (median game length identical across R∈{1..8} at bootstrap time). **Context:** static
bootstrap-time radius vs a collapse-quality outcome. **Transfer test:** this decision is a
*training-time schedule* aimed at *wall-clock economics + early coherence*, not a bootstrap-time fix
for a collapse — different variable, different objective. Per `docs/handoffs/d_strix_s2_portability.md`
§"Re-validation against the prior falsified rows": **prior does not block this driver**, but sets
correct expectations (no quality miracle, only economics + dense z-signal, itself unmeasured).
Cite 2: `L9`/§156-157 — cosine-temperature is the load-bearing colony-attractor knob; mandatory
pairing with radius diversity when cosine-temp is active. **Context:** this run has cosine-temp OFF
(`temperature_threshold_compound_moves: 0`, fixed τ=0.5, the canonical default) — the hazard is
INACTIVE, not triggered by adding a schedule. Stated as a hard DO-NOT: never re-enable cosine-temp on
this run without independently re-confirming a working radius-diversity source (see §1.6 — jitter is
NOT that source here; the schedule only provides step-keyed diversity, not per-game diversity, so this
run has *strictly less* radius diversity than the historical v6/v7full jitter regime believed to
exist). Cite 3 (transfer-test correction owed to this session's own §1.6 finding):
`v6_live2_golong.yaml`'s precedent explicitly declined `vast.yaml`'s 5→8 schedule because "vast's
5→8 is a 25×25 (v6w25) curriculum; v6_live2 is 19×19" — the mandatory pre-flight is
`radius ≤ floor(window_size/2)` for the target encoding, verified via
`engine/src/encoding/registry.toml` (`hexo_rl.encoding.lookup(name)`), never copied from another
variant. For `v6_live2_ls`: `cluster_window_size=19` → half=9; the official rule ceiling is 8
(`LEGAL_MOVE_RADIUS`, sprint log line 109) — 8 ≤ 9, fits.

**A tension this design does not paper over:** `docs/handoffs/d_strix_s2_portability.md` explicitly
recommends the radius schedule be introduced "as its own, later, single-variable arm — never bundled
with the solver/seeding variable in the same run." This run's §0 premise already accepts bundling
(deliberately, with attribution limits documented in §8); adding the radius schedule on top of a
possible solver/seeding activation (Decision 3) is a **further** deviation from that recommendation,
not a free one. Recorded here rather than silently applied: if Decision 3 activates seeding, this run
has THREE co-active novel mechanisms (multi-window self-play, radius schedule, solver+seeding) with no
single-variable disentanglement possible — §8's ablation order is the honest response, not a claim
that the bundle is clean.

**Staging (adapted from `docs/handoffs/d_strix_s2_portability.md` "Curriculum variant SPEC", re-derived
for `v6_live2_ls`'s 19×19 window; band-jitter is not achievable per §1.6, so each stage is a fixed
point, not a jittered set):**

| Stage | step trigger | radius | rationale |
|---|---|---|---|
| S1 | 0 | 4 | cheapest/shortest games, coherence-first |
| S2 | 200,000 | 5 | current registry static default — known-safe baseline |
| S3 | 400,000 | 6 | widen toward the rule bound |
| S4 | 600,000 | 8 | hard rule ceiling (`LEGAL_MOVE_RADIUS`); no further widening possible |

**Signature escape hatch (OQ8 amendment, 2026-07-04, D-PRELAUNCH) — widen EARLY on
`forced_win_conversion` saturation.** The step table above becomes a hard **ceiling**, not the only
trigger: each stage advances at **min(step threshold, signature fire)**. The signature: the net has
*mastered conversion at the current radius*, so cheap early games have stopped teaching.
**"Saturates" is defined concretely:** EMA `forced_win_conversion` (the `ForcedWinTrend` snapshot
field, `hexo_rl/diagnostics/forced_win_detector.py:392`) **≥ 0.85 with no upward slope (inter-read
delta ≤ +0.02), sustained across ≥ 2 consecutive signature-cadence reads (§5, 25k cadence), each
read informed by n ≥ 30 forced-win games** (`ForcedWinTrend` skips games with no forced win; its own
docstring flags near-zero-n reads as advisory-only — an under-n read never fires the gate). **Band
derivation (cite):** the only measured healthy sustained-run conversion in this lineage is golong's
pre-break **0.89** (`docs/handoffs/d_decide_track_a_status.md` line 34 — the same anchor §3.1/§6
already use for the terminal-break signature `0.89→0.66`); d1m's mid-run, clearly-non-saturated read
is **0.538** (same doc, line 28). 0.85 sits just under the healthy anchor and far above the
non-saturated read — labeled per this section's own convention: derived-from-anchor-points, **not** a
falsified-tested threshold. **RED-TEAM constraints (mandatory):** (i) **WIDEN-ONLY** — the gate can
never shrink radius nor delay the step schedule (the table stays the latest-start ceiling); (ii)
**hysteresis** — the ≥2-consecutive-reads + n-floor requirement above, so a single noisy EMA read
cannot fire it; (iii) **IRREVERSIBLE** once fired — no rollback if conversion dips after the widen
(an untested transition shape, same rationale as this section's no-mid-schedule-rollback rule). The
draw/colony **breach action below (revert to flat radius 5) is a separate pre-registered safety and
takes precedence** — a breach revert is not a violation of (i), it is a different gate.
**Machinery status (checked 2026-07-04):** `_resolve_radius`
(`hexo_rl/training/step_coordinator.py:502-514`, re-evaluated at log cadence, line 1257) is a pure
step-keyed table lookup — **no conditional/metric-triggered advancement machinery exists**, and none
is built for this gate. It lands as an **OPERATOR-PROCEDURAL rule**: on fire, the operator edits
`legal_move_radius_schedule` in the yaml in place (drop the next stage's `step` to the current step,
later stages untouched) and resumes per §10 step 8's resume rules; the yaml carries a matching
comment block next to the schedule.

**Radius ↔ cluster-threshold interaction (surfaced at review, own it rather than bury it):** with
`cluster_threshold=5` (registry), disjoint stone clusters cannot form while `radius ≤ 5` — every
legal move lands within hex-distance 5 of an existing stone, so the BFS-union at `≤ 5`
(`engine/src/board/moves.rs`, `get_clusters`) keeps all stones in one component. Under the S1/S2
stages (steps 0-400k, radius 4-5), K>1 cluster views can therefore arise only via the
massive-cluster multi-anchor mechanism (a single component spanning wider than the 19×19 crop —
`dmulticluster_362_legalset_design.md` §9.2a mentions "massive-cluster anchor dedup"; D-WS3V3 v2 DID
observe max_k=3 at radius 5, so this path is real, but its rate at fresh-bootstrap play strength is
unknown). **Consequence:** the run's headline novel mechanism (K>1 multi-window self-play with
disjoint clusters) phases in substantially only at S3 (radius 6, step 400k). This is a genuine
tension between the curriculum's economics rationale and the §0 premise ("first multi-window
self-play from step 0") — recorded as an operator alternative, not silently resolved: if K>1
exposure from step 0 is prioritized over cheap early games, start the schedule at radius 6 flat and
drop S1/S2. This design keeps the 4→5→6→8 staging (economics-first) and instruments the K
distribution instead (§5 M6 liveness counter — informational during S1/S2, decisive from S3).

**Pre-registered check (draw-collapse canary):** `monitors.hard_abort_draw_rate` /
`hard_abort_draw_rate_consec` (already wired, `configs/monitors.yaml:15-23`) watched for **3
consecutive evals** immediately after each stage transition (200k/400k/600k), not just steady-state.
**Band (citation corrected at review):** draw rate must not exceed 0.20 sustained across those 3
evals. The intrinsic-draw-rate reference is `configs/monitors.yaml:20-21`'s own §D-GOLONG note —
"a decisive game has an intrinsic draw rate ~0-5% (v6_live2 30k smoke: 0.05)" — which IS a
v6_live2-family source. (The previous draft cited sprint log line 1192, which is §157's 5k
**v7full** cosine-temp validation smoke — a different encoding family; citation withdrawn.)
The 0.20 watch band itself is **house convention with no falsification precedent** — no run has
ever been killed or saved by it. Contrary precedent stated honestly: §157 follow-up #5 (sprint log
line 1194 + lessons table line 1634, `feedback_draw_rate_not_abort_signal.md`) is an explicit user
verdict that "draw_rate is NOT abort signal (draws are model missing open-4s, not pathology)" —
hence this canary is a WATCH that escalates to operator review, never a standalone abort; only the
0.55 sustained hard-abort (§6, the later §D-GOLONG draw-LOCK signature, a different mechanism than
ordinary draws) stops the run on this axis. **Colony canary (same cadence):**
`colony_extension_fraction` per-game telemetry (already emitted, `hexo_rl/selfplay/instrumentation.py`)
watched alongside SealBot-WR trajectory at each transition. **Band (added at review — no
judgment-call reads):** rolling mean `colony_extension_fraction` **> 0.10 sustained across 2
consecutive signature-cadence reads (§5) = alarm → operator review**. Derivation: benign precedent
max 0.086 (§157 5k validation smoke, sprint log line 1192 — usable here because
colony_extension is encoding-family-agnostic telemetry, unlike the draw-rate band above), d1m
healthy read 0.002 (`d_decide_track_a_status.md` A1); captured-attractor reads sit far above
(§176 Gate 2: 100% col>0.3 rate on winners). 0.10 sits above all benign precedent with margin.
Labeled: derived-from-two-anchor-points convention, not a falsified-tested threshold.
**Action on breach:** revert `legal_move_radius_schedule` to a flat `[{step:0, radius:5}]` (the S2
baseline) for the remainder of the run; do not attempt a mid-schedule radius rollback to an
intermediate value (untested transition shape).

### Decision 3 — Seeding + z-labels: CONDITIONAL on the v3 smoke's GENERALIZES verdict

This is the load-bearing gate for this entire run's most expensive lever. **IN only if** the D-WS3V3
v3 smoke (`docs/handoffs/d_ws3v3_smoke_runbook.md`) returns **GENERALIZES**; **OFF** (solver +
seeding both stay at their config defaults, `solver_enabled: false`, `seed_fraction: 0.0`) on
**MEMORIZES**, **KILL**, **THIN-STILL** (pending one escalation, not an automatic IN), or
**INDETERMINATE**. §7 below names the exact smoke counters and bands that constitute this handshake —
this section states the decision rule, §7 states the mechanics.

**Seed corpus caveat (stated plainly, not hidden in a footnote):** the 125 mined traps
(`reports/d_tactical_2026-06-26/heldout_traps_all.jsonl`) were mined from an **old net's** blunders.
The *class* of tactic (short-horizon forced-win-after-blunder) transfers to any net; the *specific
instances* may not — a fresh-bootstrap net trained under multi-window self-play may blunder into
entirely different positions. **Mitigation, not a full fix:** keep the trap-corpus miner in-loop,
re-mined from the run's OWN checkpoint's blunders on a cadence (every ~100k steps, piggy-backing on the
existing eval-checkpoint cadence — no new infrastructure) rather than treating the initial 125-trap
corpus as fixed for the whole 1M-step run. This re-mining is a config/ops action (re-run
`scripts/build_ws3v3_seed_corpus.py`-equivalent against the latest checkpoint), not a code change; it
is listed here as an operator action for §10, not committed to the yaml (which cannot express "re-run
a script periodically").

**Value if activated (per `docs/handoffs/d_ws3v3_smoke_runbook.md` §0.5, "DUAL mechanism"):** (i)
trap-decision z-labels densify ALWAYS on seeded games — no solver proof required; (ii) solver
POLICY-injection densifies only on natively-provable conversions, honest ceiling
**≤8.8% (Wilson 95% upper bound, measured 0/40 sample, `reports/d_ws3v3/native_provable_fraction_sample40.json`)**
— `§D-FORENSIC F2` (2026-07-03) *independently* reconfirmed this at 20k→3M nodes / 271× SealBot's own
median node cost (0/40 proven at every tier) and **localized the drop to candidate-generation
truncation on deep (mate≥3) LOSS-side proofs**, not budget. **Re-validation:** this does not change
Decision 3's SEED-STARVED default expectation (already the pre-registered default per the runbook's
§3.5); it *strengthens* it — F2 shows the injection half is not merely under-measured but
structurally capped at the current search depth/candidate-gen, so do not expect seeding density
escalations (visit_weight 0.5→1.0, §"Verdict table" THIN-STILL row) to move the injection-fire-rate
ceiling — only the z-label half (mechanism (i)) is a genuinely free lever.

**Solver-tuning values, per the coordinator's supplemental (measured, not the stale `_full.yaml`
snapshot):** `n_workers: 24` (not 32 — 32 oversubscribes the 5080/9900X host, load ~29, under the
per-move solver CPU cost), `solver_node_budget: 20000` (not 50000 — MEASURED 2026-07-01: no-win moves
exhaust the budget at BOTH depth 10 and 16, so budget not depth is the throughput lever; ≤8-ply corpus
wins prove well under 20k, recall-safe, ~2.5× faster). `solver_visit_weight` is **explicitly NOT
hardcoded here** — the 0.3→0.15 softening rationale in `z2_solver_in_loop.yaml` is FALSIFIED (a
solver-OFF control under the same confounded regime reproduced the identical KILL signature; the
weight itself was never shown guilty, `docs/handoffs/d_ws3v3_smoke_runbook.md` §0). This run's yaml
carries `solver_visit_weight: 0.5` as a **placeholder equal to whatever ARM-INJECT/ARM-SEEDED validate
under the fixed v3 regime** — if the smoke's GENERALIZES verdict lands at a different weight
(including the THIN-STILL escalation to 1.0), **overwrite this value at launch time**, do not launch
with 0.5 by default inheritance.

**Pre-registered check (the smoke-qualification handshake):** see §7 (named counters + bands, as the
dispatcher's structure requests it as its own numbered section).

**Cost tradeoff, surfaced not resolved:** activating solver+seeding is measured at ~9.9× slower per
v2 (276 steps/hr at budget=50000) and ESTIMATED ~4× slower at this run's `solver_node_budget: 20000`
(~690 steps/hr vs `~2721 steps/hr` control-arm measured, `docs/handoffs/d_ws3v3_smoke_runbook.md` §2).
Applied to the FULL 1M-step run this is the difference between ~15.3 days and ~60.4 days wall-clock
(§9 works this out in full). **This design does not pick a window extent** (whole-run vs a bounded
window, e.g. steps 0-100k only) — that is a GPU-budget-appetite call belonging to the operator at
smoke-verdict time, not something inferable from docs. Flagged again in Open Questions.

### Decision 4 — Net size: keep 4.25M, bank the width probe

**Re-validation.** Cite: `§D-STRIX` S2 "Tiny net" verdict — TESTABLE-CHEAP, not a portable win;
production net measures 4,254,283 params (not the "2.9M" a prior brief mis-cited — that number matches
an older res10/f112 trunk, `docs/handoffs/d_strix_dispatcher_report.md` line 53-54). **Context:** the
2×2 width×aux-heads economics probe (`docs/handoffs/d_strix_s2_portability.md` §"Verdict 3") is a
*separate, cheap, standalone* ablation with an F1-anchor-cleanliness precondition — it does not gate
or block a production run at the current size. **Transfer test:** nothing about this run's premise
(fresh bootstrap, multi-window action space) changes the width-probe's cost/benefit case. **Decision:
keep 4.25M net, do not block this run on the probe.** Bank the probe as a candidate NEXT-run-only
ablation (its own bootstrap, since width changes are restart-class). No config action.

**Related, also inherited pinned:** `deploy_strength_eval.py`'s architecture-shape assertions
(`policy_fc.out_features`, `in_channels`) are unaffected by this decision since the net size is
unchanged from every prior `v6_live2`/`v6_live2_ls` bootstrap.

### Decision 5 — Sims/m: keep m16/n150, deploy eval = same regime

**Re-validation.** Cite: `§D-GUMBELSIMS` closed NULL / affordability-PARITY
(`docs/handoffs/d_strix_dispatcher_report.md` "Aggregation table": *"Train==deploy search consistency
... already live on D-WS3V3 arms"*). **Context:** matched-total-sim (m, n) sweep found no
strength-vs-cost win from moving off m=16/n_sims_full=150 at the current net size/encoding family.
**Transfer test:** this run is the SAME net size, same self-play search algorithm family
(Gumbel-Top-k Sequential Halving), just a new encoding (`v6_live2_ls` vs `v6_live2`) and fresh
bootstrap — none of which the D-GUMBELSIMS result was conditioned on being held fixed. **Decision:
`gumbel_m: 16`, `n_sims_full: 150`, `full_search_prob: 1.0`** (every position is a valid Gumbel policy
target — the 0.5 `full_search_prob` used pre-D-1M-audit was silently discarding half; the D-1M audit's
GB-2 fix is inherited here, not re-litigated). Deploy-matched eval (§1.7, §3) reads through the
**identical** `(gumbel_m, n_sims_full, c_visit, c_scale)` — `hexo_rl/eval/deploy_strength_eval.py`
hard-errors on a missing knob specifically to prevent a silent proxy-regime substitution
(`§D-LOCALIZE` "triple miss" precedent: PUCT-visit-policy + `eval_temperature=0.5` + `model_sims=64`
measured a head the model never deploys).

### Decision 6 — Value target: raw z stays; no value levers this run

**Re-validation.** Cite: `§D-INJECT` VERDICT=NO-GO (constant-−1 value-distill KILL-A⊥KILL-C
anti-correlated across all weights — can't fix losses without flipping ~25% wins,
`memory/dvderisk-ds1-labels-and-distill-overgeneralization.md`); `§D-FULLSPEC` = ENTANGLED (full-
spectrum class-balanced distill on a frozen trunk still can't separate win/loss,
`memory/d-fullspec-entangled-feature-problem.md`). **Context:** both results are on a **frozen-trunk,
distillation-style** value-target intervention, tested against the single-window `v6_live2` action
space. **Transfer test:** this run's action space changes (multi-window legal-set), but neither
falsified result's mechanism (distillation collapsing win/loss separability) is specific to the
single-window action space — the entanglement was representational/feature-level, not action-space-
level (`D-FULLSPEC`'s own closing note: "representational fix needed... light-trunk unfreeze then
E2 threat planes" — a different, larger intervention class, not something this run is doing).
**Decision: keep raw outcome z as the value target, no distillation, no value-head architecture
change.** The ~31% value-bound tail (`§D-LOCALIZE`/`§D-PERCEPT` deep value-blind cells, honest-PROVEN
core 33/61=54% after the over-count correction) is **accepted as deploy-backup territory** for this
run — i.e., the native-solver deploy-time backup (already shipped, `docs/handoffs/d_ws3v3_smoke_runbook.md`
predecessor work, D-SOLVER A1) remains the answer for that tail, not a training-time value fix. No
config action; this is a decision to NOT touch `uncertainty_weight`, `ownership_weight`,
`threat_weight`, or the value loss function from their current base defaults.

### Decision 7 — Mixing corpus: reuse the v6_live2 NPZ vs regenerate an _ls corpus (OPERATOR-OWNED)

> **RESOLVED — REGENERATED (D-PRELAUNCH, 2026-07-04).** The RECOMMENDED path below was taken:
> `--encoding v6_live2_ls` wired into `export_corpus_npz.py` (per-cluster-row scatter,
> `dmulticluster_362_legalset_design.md` §4 Tier-2 variant-(b); emission in
> `hexo_rl/bootstrap/dataset.py::replay_game_to_triples_ls`), corpus rebuilt from the same
> static human-game scrape tree (`data/corpus/raw_human` — the May-29 NPZ's true source per its
> `.metadata.json`; `data/hexo_human_corpus.jsonl` never existed locally). Output
> `data/bootstrap_corpus_v6_live2_ls.npz`, sha256 `3813edc2…345c97`, 610,954 rows
> (472,537 plies + 138,417 scatter rows = 22.7% previously-zeroed containing-window mass;
> 0.53% outside-all-windows plies dropped, unrepresentable in dense-362, same as the v6 path).
> Red-team spot-check: 8/8 off-window-adjacent positions show secondary-window one-hot rows the
> v6_live2 corpus zeroed. Snapshot caveat, recorded: the source tree is a newer snapshot than
> the May-29 build (8,300 games vs 7,199 — 1,101 scraped since). The yaml's
> `mixing.pretrained_buffer_path` now points at the _ls NPZ; the fallback text below is
> retained for the record only.
>
> **Ablation flag (2026-07-05, D-WS3V3 r1-vs-r2, note-don't-transfer):** in the v3 smoke's
> WARM-START regime (200k net, w_pre ≈ 0.283), swapping this _ls corpus in for v6_live2
> measured combined-125 trap-conversion saves 23 → 17 (a ~1.5σ regression, n=125;
> `reports/d_ws3v3/control_r2_*`). Different regime from this run (fresh bootstrap,
> w_pre 0.8 from step 0, no prior single-window lineage) — the regeneration rationale
> above stands on its own; but if run2's 5k/25k conversion gates read LOW, this
> datapoint names the mixing corpus as an early suspect to ablate.

Two facts, stated together so neither hides the other:

- **(a) Provenance is CLEAN — F1-cascade-exempt by construction.** `data/bootstrap_corpus_v6_live2.npz`
  is built by `scripts/export_corpus_npz.py` from the **static human-game corpus JSONL**
  (`scripts/export_corpus_jsonl.py` output — the script's own header: "any registered --encoding can
  be rebuilt from the same move lists"), NOT from self-play. The F1 single-window cascade poisoned
  self-play lineages; it cannot have touched this corpus. The scarier worry — "the pretrain corpus is
  itself F1-contaminated" — is dead on arrival; say so explicitly.
- **(b) Encoding semantics are STALE for this run's purpose.** The NPZ's policy targets were encoded
  under **off-window-DROP** semantics: zero probability mass on exactly the class of moves this run
  exists to stop dropping. At `w_pre` 0.8 → 0.1 over 200k decay steps (§1.4), that is single-window-
  shaped policy supervision riding inside the multi-window run for its whole first phase — a real,
  directional confound on the run's own thesis, not noise.

The registry already anticipates this: `hexo_rl/encoding/resolvers.py` `_CORPUS_PATHS` declares the
canonical `v6_live2_ls` corpus as `data/bootstrap_corpus_v6_live2_ls.npz` — **which does not exist**
— with the inline note "regenerate-vs-reuse is a run-design call (AB_RUNBOOK:
export_corpus_npz.py --encoding v6_live2_ls)". **Verified at review: that command does NOT currently
run** — `export_corpus_npz.py`'s `--encoding` choices are `("v6", "v6tp", "v6_live2", "v6w25",
"v8")` (`export_corpus_npz.py:274`); `v6_live2_ls` is not among them, and a faithful _ls corpus
additionally needs the K-cluster-row / off-window-scatter generation the resolvers note alludes to
("multi-window corpus generation differs (K cluster rows/ply)") — the shape sketched by
`dmulticluster_362_legalset_design.md` §4 Tier-2 variant-(b) (scatter the played move across ALL
containing windows), which was specced but never built. Regeneration = a small, bounded script
change, not a one-flag rerun.

**RECOMMENDATION: regenerate pre-launch.** Wire `--encoding v6_live2_ls` into `export_corpus_npz.py`
(choices + per-cluster-row emission), rebuild from the same JSONL (provenance stays clean by
construction), point `mixing.pretrained_buffer_path` at the new NPZ. This removes a directional
confound from the run's founding 200k steps for roughly a day of script work + one corpus-build.
**FALLBACK if regeneration is deferred (the current yaml state):** reuse
`data/bootstrap_corpus_v6_live2.npz` (wire-compatible, loads no-reshape) AND (pre-registered check,
cheap, run once before launch) **measure the fraction of corpus positions that have ≥1
off-window-eligible legal move** (a one-pass script over the NPZ/JSONL: for each position, does any
legal move fall outside the board-centered 19×19 window?). If that fraction is small (human games
are compact; plausibly ≪5%), the stale-semantics confound is quantified-small and reuse is
documented as a conscious confound in §1.1 + §8; if it is large, regeneration stops being optional.
Either way the measured number goes into the run log before step 0. The yaml ships the fallback
(reuse, loudly commented) because the regeneration work is not this DESIGN dispatch's to build;
the operator picks at launch time.

### Omitted lever noted for the record — bot-mix

`mixing.bot_corpus_path`/`bot_batch_share` stay at base defaults (null/0.0). §S181 Track 4A found
bot-mix mildly protective in its own lineage (removing it ACCELERATED decline), but that result's
anchor lineage (`bootstrap_model_v6`, v6 corpus/encoding family) is a different corpus and encoding
family — transfer to a fresh `v6_live2_ls` bootstrap is unestablished, so the lever is omitted
deliberately rather than half-inherited; the colony canary (§2 Decision 2 band) covers the watch
that motivated it.

---

## 3. BUNDLE — restart-owed adds, verified landed or verified missing

### 3.1 Structural-metrics instrumentation — LANDED, verify-in-doc

`longest_line_fraction` + `n_components` (PER-PLAYER/winner) are emitted on every `game_complete`
event when `monitoring.log_investigation_metrics: true` (base default, `configs/monitoring.yaml:26`).
Commit `5686b5c` "feat(selfplay): per-game structural metrics..." — confirmed an ancestor of this
branch's HEAD (`git merge-base --is-ancestor 5686b5c HEAD` → yes). Implementation:
`hexo_rl/selfplay/instrumentation.py:213-350` (`_compute_longest_line`, `_compute_n_components`,
pinned to the engine's own `_CLUSTER_THRESHOLD=5` / `_WIN_LENGTH=6` constants so the Python emit stays
numerically comparable to a future Rust-side emit). This makes the "golong kill-gate" *runnable* in
the sense that the reference signature (`n_components 26→42`, `longest_line 9.3→8.4`,
`forced_win_conversion 0.89→0.66` at terminal-break, per `docs/handoffs/d_decide_track_a_status.md`
line 34) is now comparable against live telemetry without a separate offline replay-analysis pass
(`scripts/d1m_replay_analyzer.py` remains available as an offline cross-check but is no longer the
only source). **Caveat carried from `d_decide_track_a_status.md` "B3a ITEM-4":** the per-game emit has
no reliable step axis until at least one promotion fires (checkpoint_step / model_version freeze at
their startup-seed value between promotions) — bucket by wall-clock or game-count for early-run reads,
not by `checkpoint_step`, until the first promotion.

### 3.2 CI half-draw fix — LANDED

Commit `ab92a29` "fix(eval): draw-aware promotion-gate CI corrects half-draw false-negative" —
confirmed an ancestor of HEAD. `_draw_aware_ci` centers the Wilson interval on the same
`p_hat=(W+0.5D)/n` the point estimate already used (`hexo_rl/eval/gate_logic.py:103-156`), closing a
conservative (false-negative-only) promotion-gate bug. Direction-safe: can only un-block a deserving
checkpoint, never falsely promote one. No config action needed; this run inherits the fix
automatically via `hexo_rl/eval/eval_pipeline.py` / `opponent_runners.py`.

### 3.3 Deploy-matched eval pinned from step 0 — NEW for this run (not previously pinned this early)

`eval_pipeline.opponents.deploy_strength` (`configs/eval.yaml`, D-LOCALIZE P4 Track B,
`hexo_rl/eval/deploy_strength_eval.py`) exists and is **default OFF** in every shipped variant to
date, including the D-1M `longrun_v6_live2_ls_gumbel_m16.yaml` (which relied on the legacy
`best_checkpoint`/`sealbot` opponents and only got a deploy-matched read via the separate, ad-hoc
`gumbel_greedy_bot.py` harness built mid-run during `§D-DECIDE`, per
`docs/handoffs/d_decide_track_a_status.md` "A2 — HONEST STRENGTH"). This run's yaml sets
`eval_pipeline.opponents.deploy_strength.enabled: true` **from launch**, paired with
`best_checkpoint.enabled: true` (its opponent) per the module's own docstring requirement. This
closes the "instrument built mid-investigation, after the plateau was already suspected" pattern —
the deploy-matched, distinct-game-bootstrapped strength read (§1.7) is live from step 0, so the 100k+
slope gate (§5) has real trend data rather than 5 retroactively-computed points. Adaptive
screen→confirm (`screen_n:80`/`confirm_n:200`/`screen_confirm_hi:1.0`) keeps the added eval cost
bounded — base defaults are used unmodified (§4 lists only the `enabled: true` flip).

### 3.4 Summary-JSON loader-semantics recording — PARTIALLY landed, gap flagged

`scripts/eval/run_l1_trapflip_smoke.py:276-278` writes `encoding_decode` / `expect_encoding` / `loader`
fields into its `l1_trapflip_summary.json` — confirmed landed and load-bearing (this is the mechanism
that made the D-WS3V3 A1 re-baseline provenance record possible, `memory/d-ws3v3-a1-rebaseline-match-hostmatch.md`).
`scripts/exploit_probe.py`'s `*.summary.json` records `checkpoint`/`encoding`/`defender`/`sims` but
**not** an explicit `loader`/decode-path field (`exploit_probe.py:219-224`) — a strictly weaker record
than the trapflip evaluator's. `scripts/probe_threat_logits.py` resolves and *prints* the stamp source
to stderr (`probe_threat_logits.py:664-677`, "encoding from checkpoint stamp: ... " /
"auto-detected encoding...") and hard-refuses a silent cross-fixture fallback
(`probe_threat_logits.py:693-715`), but its persisted baseline JSON schema
(`save_baseline_json`, lines 121-141) does **not** carry a `loader`/`encoding_decode` field either —
the provenance lives only in the console log, not the artifact. **Verification result: landed on the
evaluator that most needed it (trap-flip), NOT yet on every eval path.** This is flagged, not fixed,
in this design (a small addition to `exploit_probe.py`'s and `probe_threat_logits.py`'s summary
schemas — mirroring the trapflip evaluator's 3 fields — is a cheap follow-up, listed in Open
Questions, not committed here since it is out of this DESIGN dispatch's scope).

---

## 4. Config — `configs/variants/run2_mw_fresh.yaml`

Base configs consulted: `configs/training.yaml`, `configs/selfplay.yaml`, `configs/eval.yaml`,
`configs/model.yaml`, `configs/monitors.yaml`, `configs/monitoring.yaml`. Every key below either
matches an established `v6_live2_ls` variant's precedent (cited) or is a new key introduced by one of
the numbered Decisions above (cited by decision number). No key duplicates a base-default value
(hygiene per `v6_sustained.yaml`'s own "redundant-to-base scalars dropped" precedent) except where a
loud comment explains why the explicit restatement matters (e.g. `mcts.dirichlet_enabled: false`,
which is a documentation no-op on the Gumbel branch per Decision 1, kept for grep-ability).

| Key | Value | Why |
|---|---|---|
| `encoding` | `v6_live2_ls` | §1.1 |
| `in_channels` | `4` | matches `v6_live2_ls` wire shape (registry `n_planes=4`) |
| `lr` / `eta_min` | `2e-3` / `5e-4` | matches every `v6_live2_ls` production precedent (`v6_live2_golong.yaml`, D-1M longrun) |
| `total_steps` | `1_000_000` | matches the D-1M production-run precedent for this encoding (`longrun_v6_live2_ls_gumbel_m16.yaml`); sets the CosineAnnealingLR `T_max` — MUST be launched with `--iterations 1000000` (D-1M audit C2 lesson, an open-ended launch silently inherits the base 200k denominator and mis-anneals) |
| `min_buffer_size` | `25000` | POSITIONS — real self-play prefill gate before the first training step (`step_coordinator.py:736`). Base default 256 is not a real gate; inheriting it reproduces the v2 cold-buffer confound (training began on ~0 real self-play positions, `d_ws3v3_smoke_runbook.md` §0 V0b). §5's 5k/25k buffer-health gate reads this value |
| `aux_opp_reply_weight` | `0.0` | CF-5: the opp-reply aux head trains on the current-player MCTS visit target — redundant with the policy head, a flat-policy-reinforcing gradient spending 15% of loss weight (`configs/variants/v6_live2_golong.yaml:70-76`, citing `reports/investigations/phase6_scoping_memo_20260531.md` §3); every `v6_live2`-family variant since carries it |
| `training_steps_per_game` | `1.0` | matches `v6_live2_ls` precedent (throughput-tuned for the 5080/9900X host) |
| `selfplay.completed_q_values` | `true` | matches precedent; Gumbel completed-Q targets |
| `selfplay.n_workers` | `24` | operator-measured correction (coordinator supplemental): 32 oversubscribes the 5080/9900X host (load ~29) under any per-move solver CPU cost; `24 == nproc`. Kept at 24 even when solver is OFF for byte-identical worker-pool sizing across the solver-off→solver-on transition if Decision 3 fires mid-run |
| `selfplay.inference_batch_size` / `inference_max_wait_ms` / `leaf_batch_size` | `64` / `2.0` / `8` | matches `v6_live2_ls` precedent |
| `selfplay.random_opening_plies` | `0` | matches precedent; deterministic from move 1 |
| `selfplay.gumbel_mcts` / `gumbel_m` / `gumbel_explore_moves` / `c_visit` / `c_scale` | `true` / `16` / `10` / `50.0` / `1.0` | Decision 5 |
| `selfplay.playout_cap.full_search_prob` / `n_sims_full` / `n_sims_quick` | `1.0` / `150` / `150` | Decision 5, inherits D-1M audit GB-2 fix |
| `selfplay.legal_move_radius_schedule` | 4-stage table, §2 Decision 2 | Decision 2; NOTE `legal_move_radius_jitter` is NOT set here — it is dead code for this encoding (§1.6) and the schedule is the only live diversity source |
| `selfplay.solver_enabled` | `false` (launch default) | Decision 3 — flips to `true` ONLY on smoke GENERALIZES, per §7 |
| `selfplay.solver_depth` | `16` | matches v3 smoke precedent (plies, ~2× mate-distance-in-turns) |
| `selfplay.solver_node_budget` | `20000` | operator-measured (coordinator supplemental): NOT the stale `_full.yaml` 50000 — no-win moves exhaust the budget at both depth 10 and 16, so budget not depth is the cost lever; corpus wins (≤8-ply) prove well under 20k |
| `selfplay.solver_neighbor_dist` | `2` | matches v3 smoke precedent (quiet-move widening) |
| `selfplay.solver_visit_weight` | `0.5` (PLACEHOLDER) | Decision 3 — explicitly a placeholder; overwrite with whatever the smoke's GENERALIZES arm validates before launch, do not inherit silently |
| `selfplay.seed_fraction` / `seed_corpus_path` | `0.0` / `null` (launch default) | Decision 3 — flips only on GENERALIZES |
| `selfplay.forced_win_policy_enabled` | `false` | matches precedent; solver supersedes O1 when active, single-variable discipline |
| `mcts.dirichlet_enabled` | `false` | Decision 1 — documentation no-op on the Gumbel branch, kept for grep-ability / hygiene parity with every other `v6_live2_ls` variant |
| `mixing.pretrained_buffer_path` | `data/bootstrap_corpus_v6_live2.npz` | §2 Decision 7 FALLBACK state (deliberate reuse, loudly commented in the yaml): provenance CLEAN (human-game JSONL, F1-exempt), semantics STALE (off-window-drop policy targets); recommended pre-launch action is regeneration to `bootstrap_corpus_v6_live2_ls.npz`. NOTE `"<auto>"` would resolve to the nonexistent _ls path (resolvers `_CORPUS_PATHS`) and fail — the explicit path makes the reuse decision visible |
| `mixing.initial_pretrained_weight` / `min_pretrained_weight` / `decay_steps` | `0.8` / `0.1` / `200000` | §1.4 — RESTATED-FOR-VISIBILITY ONLY, byte-identical to `configs/training.yaml` base defaults (loud comment in the yaml per this section's own hygiene rule); matches the D-1M variant's same convention |
| `eval_interval` (root) | `25000` | this is the LIVE trigger read by `loop.py:192` (D-1M audit F1 finding — `eval.yaml`'s `eval_interval` is dead), aligned to the §5 "milestone" cadence |
| `eval_pipeline.opponents.best_checkpoint` | `{enabled:true, stride:1, n_games:200, model_sims:64, opponent_sims:64}` | matches D-1M audit's cost-reduced precedent (n=400→200, sims 128→64, ~4× cheaper, still turns the loop) |
| `eval_pipeline.opponents.sealbot` | `{enabled:true, stride:1, n_games:100}` | matches precedent; Wilson95 ±10pp |
| `eval_pipeline.opponents.random` | `{enabled:true, stride:1, n_games:20}` | matches precedent |
| `eval_pipeline.opponents.bootstrap_anchor` | `{enabled:true, stride:4, n_games:50, model_sims:64, opponent_sims:64, path: checkpoints/bootstrap_model_v6_live2_8300.pt, encoding: v6_live2}` | matches D-1M precedent; `encoding:` declared per the D-EVALGATE deep-merge warning (§1.2) — the anchor's OWN stamp, never the candidate's |
| `eval_pipeline.opponents.deploy_strength.enabled` | `true` | §3.3 — the one opponent-block flip that is NEW vs every prior `v6_live2_ls` variant; all other `deploy_strength.*` keys left at `configs/eval.yaml` base defaults (`screen_n:80`, `confirm_n:200`, `sealbot_max_depth:5`, `n_boot:1000`, `min_distinct_per_pair:10`) |
| `eval_pipeline.opponents.offwindow_adversary` | `{enabled:true, stride:1, n_games:100, arm:"exploit", model_sims:128}` | OQ10 RESOLVED (2026-07-04, D-PRELAUNCH): arms the D-EVALFOUND §7 IN-loop Objective-A guard — `offwindow_forced_win_rate` feeds `decide_promotion` (BLOCKS at code-default `gating.robustness_threshold=0.06`, `eval_pipeline.py:443`) + the step-coordinator WARN; closes the `objective_a_coverage_gap` warning. Shape = `v6_live2_golong.yaml` precedent EXCEPT `stride:1` (a promotion gate reads missing = pass; a skipped round is an unguarded promotion). TRUST SCOPE: absolute exploit-arm rate only — `memory/exploit-probe-arm-aliasing-bug.md` (see §5) |
| `eval_pipeline.gating.expected_anchor_sha256` | `ebf2ed39fd64db525864aeade75972372da9418cc8179347afc8cecb60ba6db8` | fresh-launch incumbent pin (`§D-LOOPFIX` W2) — asserts `best_model.pt` resolves to the intended bootstrap anchor at launch; **must be cleared/repointed on any resume after the first promotion** (same caveat the D-1M variant documents) |
| `eval_pipeline.gating.promotion_winrate` / `require_ci_above_half` | `0.55` / `true` | matches precedent |
| `eval_pipeline.gating.bootstrap_floor.enabled` | `false` | matches D-1M precedent (diagnostic-only `wr_bootstrap_anchor`, does not gate — audit PGL-2) |
| `telemetry.emit_outcome_distribution` / `emit_value_pred_at_ply_cap` | `true` / `true` | matches precedent |
| `monitors.hard_abort_grad_norm` | `10.0` | matches precedent |
| `monitors.hard_abort_draw_rate` / `_consec` / `_min_step` | `0.55` / `3` / `0` | matches precedent; also the §6 KILL condition and the Decision 2 stage-transition canary's hard backstop |

Every key not listed above is left at its `configs/{training,selfplay,eval,model,monitors,monitoring}.yaml`
base default. No `board_size`/`in_channels`/`n_planes` scalar is declared beyond `in_channels: 4`
(required so the model constructor matches the checkpoint before the encoding resolver runs) —
everything else about the encoding's geometry is read from the registry by name, per the
`_check_scattered_keys` discipline in `CLAUDE.md`.

---

## 5. Monitoring gates (pre-registered)

| Cadence | What | Metric / band | Action |
|---|---|---|---|
| 5k, 25k | Machinery gates | fire counters (`solver_eligible_per_step`/`solver_injected_per_step`/`solver_fire_rate`, only if Decision 3 fired), seeding fraction realized vs `seed_fraction` config, encoding stamp on the saved checkpoint (`metadata['encoding_name'] == 'v6_live2_ls'`), buffer health (`min_buffer_size: 25000` reached, no drop-queue overflow — `results_queue_cap: 10000`, `positions_dropped` getter), **multi-window liveness check (F1-recurrence guard — see below this table)** | Abort-fix-relaunch. Pre-committed as CHEAP at this stage — do not tolerate a broken machinery gate past 25k hoping it self-corrects. |
| 25k, 50k, 100k | Signature reads, NOT strength bars | entropy in the ~2.1-2.9 nat healthy band, warn <2.0, KILL <1.0 (Decision 1, corrected band); draw rate (intrinsic ~0-5% per `configs/monitors.yaml:20-21`, watch above 0.20 sustained — escalate-only, never standalone abort, Decision 2); colony_extension rolling mean vs the 0.10 alarm band (Decision 2); game length distribution; structural metrics trend (`longest_line`, `n_components` — §3.1, bucket by game-count until first promotion); threat-probe C1-C3 **against the RIGHT fixture** (`--positions tests/fixtures/threat_probe_positions_v6_live2.npz` on this `_ls`-stamped run — the cross-fixture fallback trap, `memory/d-ws3v3-a1-rebaseline-match-hostmatch.md`) — **early-run caveat: the C1-C3 baseline (`threat_probe_baseline_anchor200k.json`) is anchored to a 200k-step warm-start; a 25k fresh-bootstrap net has NO expected band against it — early reads are purely directional trend (is C1 rising?), never PASS/FAIL, so a low C1 at 25k is NOT a regression signal**; trap-flip on held-out 31/125 (`scripts/eval/run_l1_trapflip_smoke.py`) vs a HOST-MATCH-minted baseline (§1.5) — only meaningful once Decision 3 has fired, otherwise this is a no-op (no solver/z-label lever to move it); `forced_win_conversion` EMA vs the §2 Decision 2 **saturation band** (≥0.85, slope ≤ +0.02, ≥2 consecutive reads, n≥30 — the OQ8 widen-early signature gate); `offwindow_forced_win_rate` in-loop trend (stride 1, every eval round — OQ10, absolute-rate read only, see the trust-scope block below this table) | Canonical runs for this encoding go 200k-1M (D-1M precedent); these are trend checks, not pass/fail bars. Escalate to operator review on any 2-of-5 metric simultaneous drift. EXCEPTIONS with their own actions: `forced_win_conversion` saturation fire → operator widens the radius schedule EARLY per §2 Decision 2 (widen-only, irreversible, yaml edit + resume); `offwindow_forced_win_rate` > 0.06 → promotion BLOCKED automatically in-loop (`decide_promotion` robustness gate — not an operator read). |
| 100k+ | Deploy-matched SealBot-d5 slope; self-ladder BT | `deploy_strength` WR trajectory read in this doc's own distinct-game bootstrap-CI terms (never an absolute threshold — `§D-LADDER`'s own lesson: a wall-clock time-limited bar vs a fixed-depth bar flipped a verdict): **UP** = the inter-milestone WR delta's 95% distinct-game bootstrap CI excludes zero from above; **FLAT** = the delta CI straddles zero across **2 consecutive milestone reads**; **DOWN** = excludes zero from below. Self-play BT-Elo same discipline: distinct-game bootstrap (dedupe byte-identical sequences before the CI — CLAUDE.md effective-n corollary, `§D-ARGMAX`), decision on `ci_lo_boot`/`ci_hi_boot`, never the raw-count CI | UP → continue; FLAT/DOWN per the CI definition above → escalate to a `§D-DECIDE`-style plateau review (the exact aggregation-matrix pattern that closed the d1m lineage) before spending further GPU-weeks. |

Read cadence for the 25k/50k/100k row rides the yaml's `eval_interval: 25000`; the 5k/25k machinery
row is read via the `checkpoint_interval: 500` / `anchor_every_steps: 5000` guarantees (base config,
unmodified) — a checkpoint always exists at every 5k boundary regardless of the eval cadence, so the
machinery gate does not need its own eval-pipeline round.

**In-loop Objective-A robustness guard (OQ10 RESOLVED 2026-07-04, D-PRELAUNCH — supersedes the
"unguarded in-loop promotion path" caveat this doc previously carried).** The
`offwindow_adversary` eval opponent is now **ENABLED** in the yaml (`{enabled:true, stride:1,
n_games:100, arm:"exploit", model_sims:128}` — `v6_live2_golong.yaml` precedent shape, stride
tightened to 1 because the promotion gate treats a missing `robustness_rate` as PASS, so any skipped
round would be an unguarded promotion). `offwindow_forced_win_rate` now flows to BOTH the promotion
gate (`decide_promotion` BLOCKS at the code-default `gating.robustness_threshold=0.06`,
`hexo_rl/eval/eval_pipeline.py:443` — no yaml key needed) and the step-coordinator WARN, per
`D_EVALFOUND_design.md` §7; the `objective_a_coverage_gap` startup warning no longer fires. The §5
M6 paired-decode `exploit_probe` reads and the §6 seeding co-gate remain the decisive OUT-of-loop
instruments; this monitor is the always-on in-loop backstop. **TRUST SCOPE (arm-aliasing bug,
`memory/exploit-probe-arm-aliasing-bug.md`):** the shared adversary code path
(`hexo_rl/eval/offwindow_probe.py`, mirrored by `scripts/exploit_probe.py`) has a known
arm-aliasing bug — the adversary's `_is_off` classifies against the CURRENT board, not
`model_last_snapshot`, so exploit-vs-CONTROL **contrast** verdicts are contaminated
(exploit==control byte-identical under centroid-shifting defenders). The in-loop monitor reads the
**single exploit arm's ABSOLUTE forced rate only**, which that memo (and `offwindow_probe.py`'s own
2026-06-28 `adv_reference="current"` empirical resolution) holds VALID — the gate as configured does
not inherit the contamination. Do **NOT** add a control-arm contrast read to the in-loop gate, and
do not cite in-loop exploit-vs-control deltas anywhere, until that bug is fixed. Secondary value: on
a live multi-window decode the absolute rate should sit ~0.0 (D-DECODE legal-set precedent,
`memory/d-decode-offwindow-decoding-reframe.md`) — a sustained high read is an additional
F1-recurrence signal alongside the M6 checks below.

**Multi-window liveness check (F1-recurrence guard — red-team addition).** The machinery gate's
`metadata['encoding_name'] == 'v6_live2_ls'` stamp check trusts exactly the signal class that failed
silently in F1 (checkpoint metadata self-perpetuation). Two checks independent of the checkpoint
dict, run at the 5k gate and re-run at 25k:

1. **Paired-decode probe (PRIMARY, decisive).** Run `scripts/exploit_probe.py` TWICE on the SAME 5k
   checkpoint: once `--expect-encoding v6_live2_ls --defender deploy --legal-set`, once identical but
   WITHOUT `--legal-set` (single-window decode of the same weights). Named metric:
   `off_window_forced_win_rate` (exploit arm; the ABSOLUTE rate — valid per the arm-aliasing memo,
   `memory/exploit-probe-arm-aliasing-bug.md`; the exploit-vs-control contrast is NOT used).
   **Band:** the two runs must NOT be byte-identical, AND
   `rate(legal_set) ≤ rate(single_window)` with a gap consistent with the D-SOLVER/D-DECODE
   precedent (single-window deploy 0.335 forced / multi-window no-drop 0.0 on the SAME weights —
   `memory/d-solver-offwindow-deploy-head-hole.md`). The paired-on-same-weights form is deliberately
   weak-net-robust: a 5k net is weak in BOTH runs, so a decode-driven difference (not a strength
   difference) is what the pair isolates. Byte-identical output across the pair = the legal-set path
   is not live in eval decode; `rate(legal_set) ≥ 0.15` with no gap vs single-window = the
   single-window-regime signature → **abort-fix-relaunch** (this is exactly the F1 failure re-caught
   at 5k instead of at a 272k forensic).
2. **Self-play-side counter (SECONDARY, informational during S1/S2).** Fraction of recorded self-play
   positions emitting K≥2 cluster rows (readable from replay records / buffer row-count-per-position;
   D-WS3V3 v2 forensics measured max_k=3 this way). Named counter: `k_ge2_row_fraction` (derived
   per-checkpoint from game records, not a live emit). **No pre-derivable band during S1/S2** — per
   §2 Decision 2's radius↔cluster-threshold interaction, disjoint clusters cannot form at radius ≤5
   and K>1 arises only via massive-cluster multi-anchoring, whose rate at fresh-bootstrap strength is
   unknown; a K=1-dominant read at 5k/25k is NOT evidence of the F1 failure. From S3 (radius 6,
   step 400k) onward this counter becomes decisive: `k_ge2_row_fraction == 0` across a full 25k
   window at radius ≥6 is not plausible under a live multi-window path → treat as check-1-failed.

---

## 6. KILL conditions (pre-set, hard abort or operator-mandatory stop)

- **Draw-collapse signature (L9 class):** `monitors.hard_abort_draw_rate: 0.55` for
  `hard_abort_draw_rate_consec: 3` consecutive eval points. The mechanism is wired in base, but the
  base ships it **DEFAULT-OFF** (`configs/monitors.yaml:25`, `hard_abort_draw_rate: 0.0` — its own
  comment says the operator arms it per run variant); this run's yaml's 0.55/3 is a **deliberate
  per-variant arm** (matching every armed `v6_live2_ls` production variant), not a base default
  carried through (§4).
- **Stride-5 spam (lattice-spam draw-collapse variant):** `monitors.hard_abort_stride5_p90` — base
  default (30) inherited; §CANARY-VAL validated (radius-5 benign P90 ≤4, cosine-temp collapse P90
  86-133, `configs/monitors.yaml:8-14`).
- **Fragmentation co-occurrence:** `n_components` UP **and** `longest_line` DOWN **and**
  `forced_win_conversion` DOWN, simultaneously, sustained across 2+ signature-cadence reads (§5) — the
  exact golong terminal-break shape (`n_comp 26→42, ll 9.3→8.4, conv 0.89→0.66`,
  `docs/handoffs/d_decide_track_a_status.md` line 34). No single-metric threshold; co-occurrence is
  the signal (a single metric moving alone has repeatedly been a length/stone-count artifact, per the
  same doc's A1 finding that `longest_line_fraction` alone is "pure inverse stone-count artifact,
  Spearman −0.9997 vs stones").
- **Entropy floor breach:** Decision 1's check — `H(π) < 1.0` nat sustained past 3 consecutive eval
  points.
- **Seeding KILL (re-derived for fresh-run scale, per the dispatcher's ">16% normal-position
  corruption analog" framing):** the D-WS3V3 v3 runbook's own KILL condition is `deploy-disagree > 0.16`
  on an ON-arm with a clean control (§6 verdict table in that runbook). This run does not re-derive a
  NEW threshold — it inherits the smoke's own KILL bar verbatim, since the mechanism (solver-in-loop
  soft visit-injection corrupting normal play) and its measurement instrument
  (`scripts/eval/run_l1_trapflip_smoke.py` deploy-disagree co-gate) are identical whether run at
  smoke scale (8k steps) or production scale (this run). If Decision 3 fires, this run must
  periodically re-run the deploy-disagree co-gate (piggy-backing the 25k/50k/100k signature cadence,
  §5) — a sustained breach past 0.16 is a hard stop on the solver lever specifically (revert to
  `solver_enabled: false`, keep the rest of the run going on its multi-window-only merits), not a
  whole-run abort.
- **Not a KILL — armed promotion-blocker (OQ10 RESOLVED, for completeness of this table):** the
  in-loop `offwindow_adversary` robustness gate (§5) BLOCKS individual promotions at
  `offwindow_forced_win_rate > 0.06`; it never stops the run. The previously-documented gap ("the
  IN-loop promotion path has no Objective-A guard as configured") is closed — a checkpoint can no
  longer promote off-window-exploitable unguarded. A *persistently* blocked promotion stream
  (3+ consecutive eval rounds robustness-BLOCKED) is an operator-review escalation on the
  multi-window machinery (cross-check the §5 M6 paired-decode probe), not an automatic abort.

---

## 7. Smoke-qualification handshake (D-WS3V3 v3 → this run's Decision 3)

Named counters + bands, read directly from `docs/handoffs/d_ws3v3_smoke_runbook.md` §3/§5/§6 (verbatim,
not re-derived):

**Control validity gate (§3 of the runbook) must PASS first — if this fails, Decision 3 stays OFF
regardless of what the ON-arms show, full stop:**
- `deploy-disagree ≤ 0.16` on ARM-CONTROL.
- registered-31 trap-flip count within the HOST-MATCH-recentered band (laptop: [1,5]; vast: re-mint
  the strip self-baseline once, band re-centers on that read, §1.5).
- combined-125 trap-flip count within the HOST-MATCH-recentered band (laptop: [17,27]; vast:
  re-derive per §1.5).
- threat-probe C1 ≥ 5.848, C2 ≥ 25%, C3 ≥ 40% vs `threat_probe_baseline_anchor200k.json`.

**Headline gate per ON-arm (§4):** held-out flip on registered-31 AND combined-125
(`scripts/eval/run_l1_trapflip_smoke.py`, `--expect-encoding v6_live2_ls --legal-set`, baseline =
`ws3v3_warmstart_200k.pt`).

**Co-gates, all must clear (§5):** off-window forced rate holds 0.0 (`exploit_probe.py --defender
deploy --legal-set`); ModelPlayer false-clear cross-check also 0.0; threat-probe C1-C3 vs the anchor
baseline hold.

**SEED-STARVED pre-check for ARM-SEEDED specifically (checked BEFORE any verdict-table row for that
arm, §6 of the runbook):** `solver_fire_rate_seeded < 11.2%` (2× the organic v2-measured 5.6% rate) →
do NOT read ARM-SEEDED's flip standalone as MEMORIZES; the z-label mechanism (§0.5 (i)) may still be
live even when the solver-injection mechanism (§0.5 (ii)) is starved. `§D-FORENSIC F2` (§2 Decision 3
above) makes SEED-STARVED the *expected*, not edge-case, outcome for the injection half — plan Decision
3's activation around the z-label mechanism primarily.

**GREENLIGHT (this run's Decision 3 flips to IN) iff:** verdict table (`docs/handoffs/d_ws3v3_smoke_runbook.md`
§6) resolves **GENERALIZES** for ARM-INJECT or ARM-SEEDED — held-out flip ≥25% AND control holds flat
(§3 gate passed) AND KILL ≤16% (deploy-disagree co-gate) AND all §5 co-gates clear. **On THIN-STILL:**
do not activate Decision 3 for this run yet — the runbook's own next-step is one escalation
(`solver_visit_weight: 1.0`) before declaring; wait for that escalation's verdict, do not treat
THIN-STILL as a green light. **On MEMORIZES, KILL, or INDETERMINATE:** Decision 3 stays OFF for the
duration of this run; the multi-window self-play + radius curriculum (Decisions 2, unaffected) still
proceed as the run's own worthwhile experiment (§0 premise — this run is valuable even without the
solver lever, since multi-window self-play from step 0 has never been measured at all).

---

## 8. Attribution limits + pre-committed ablation order

**No in-run component attribution is possible for this bundle.** If the run succeeds (sustained
strength gain past the d1m single-window TRUE-STALL context, §0), the following components are
CO-ACTIVE and cannot be individually credited from this run's data alone: multi-window legal-set
self-play (§1.1), the radius curriculum (§2 Decision 2, if it stays IN), and — conditionally — the
solver+seeding lever (§2 Decision 3, if GENERALIZES). **Pre-committed post-run ablation order, IF
SUCCESS:**

1. **Curriculum-off first.** Re-run (or a short warm-start continuation) with
   `legal_move_radius_schedule` removed (flat radius=5 throughout) — cheapest ablation (no engine
   change, one config flip), and Decision 2's own economics claim (cheap early games) is the
   least-validated of the bundle's mechanisms (§2 Decision 2's own text: "payoff is economics + dense
   z-signal, and that payoff itself is UNMEASURED").
2. **Seeding-off second** (only if Decision 3 fired) — re-run with `solver_enabled: false`,
   `seed_fraction: 0.0`, everything else held. More expensive (loses the z-label densification the
   smoke validated), so ordered second, not first.

**Pre-committed debug path, IF A KILL FIRES (§6):** the specific KILL condition that fired determines
the branch — draw-collapse/stride-5/fragmentation → suspect the radius curriculum interacting with the
(already-inert, §1.6) jitter gap first (this run has LESS per-game radius diversity than the historical
v6/v7full regime believed to have); entropy floor → suspect `entropy_reg_weight`/`policy_prune_frac`,
NOT a Dirichlet re-enable (there is no Dirichlet knob left on the Gumbel branch, §2 Decision 1);
seeding KILL (deploy-disagree > 0.16) → revert Decision 3 to OFF, keep the rest of the run.

**d1m comparisons are CONTEXT ONLY, never evidence.** Any read of this run's trajectory against d1m's
120k→226k TRUE-STALL numbers must be captioned as "different action space, not a controlled
comparison" — per §0's premise, d1m characterizes the single-window regime, this run is the first
measurement of the multi-window regime, and the two are not exchangeable data points for a single
strength curve.

---

## 9. Budget honesty

**Solver+seeding overhead — MEASURED-FROM-SMOKE placeholder, named counters that will fill it:**
`training_step.solver_fire_rate`, `solver_eligible_per_step`, `solver_injected_per_step`,
`solver_fire_rate_seeded` (per-arm, from the D-WS3V3 v3 smoke's in-run JSONL logs, once it runs). The
number that matters for THIS run's wall-clock projection is the **steps/hour delta** between
solver-OFF and solver-ON at `solver_node_budget: 20000`, which the smoke will measure directly
(ARM-CONTROL vs ARM-INJECT/ARM-SEEDED `train_step` timestamp deltas in
`logs/ws3v3_{control,inject,seeded}.jsonl`). Until that lands, this section uses the runbook's own
pre-registered ESTIMATE (§2 of that runbook, "Per-arm wall-clock estimate"), explicitly marked
ESTIMATED there, not yet MEASURED for v3's exact config.

**Bench-gate evidence available now (does NOT measure solver overhead — see caveat):** four `make
bench` snapshots pulled from vast 2026-07-02 (`reports/vast_dsolver_bench/bench_{pre,pre2,post,post2}.log`,
gitignored, local-only), bracketing the D-WS3V3 engine bench-gate PASS (`commit a481ea3`, "bench gate
PASS on vast (10/10 x4 snapshots)"). These are the STANDARD `scripts/benchmark.py --mcts-sims 50000
--pool-workers 22` harness — it does not enable the solver (no solver flag exists in that script) —
so they confirm the bundled engine changes (Dirichlet-drop `f4413d6`, legal-set wiring `5c90988`,
seeding/counter instrumentation `87b9c96`) cost **no throughput regression**: MCTS sim/s 109.8k-111.2k
(floor ≥73,000, comfortably clear), worker throughput 78.2k-87.8k pos/hr, `Worker pool throughput
games/hr` medians 689.9-869.8 across the 4 snapshots (generic-workload noise band, not a solver
measurement — flagged explicitly here so this number is not mistaken for solver-specific overhead).
**This is confirmation the bundle is bench-clean, not a solver cost measurement.**

**Real per-arm numbers that DO exist (v2 measured, v3 config, `docs/handoffs/d_ws3v3_smoke_runbook.md`
§2):**

| Arm | steps/hr | positions/hr | Basis |
|---|---|---|---|
| Control (solver OFF) | 2721 | ~129,000 | MEASURED, v2's control arm actual `train_step`/`game_complete` timestamps, same 5080 host, `n_workers:24` |
| Inject/Seeded (solver ON, budget 20000) | ~690 (ESTIMATE) | ~40,000 (ESTIMATE) | v2's candidate arm measured 276 steps/hr at budget=50000; scaled ×2.5 for the budget reduction (itself a measured ratio from the trap-population timing study, not a guess) — **re-measure against the v3 smoke's own first-hour log before trusting for scheduling** |

**Wall-clock projection to this run's `total_steps: 1_000_000` (vast 5080 + Ryzen 9 9900X):**

| Scenario | Rate | Wall-clock for 1,000,000 steps |
|---|---|---|
| Decision 3 stays OFF (multi-window + radius curriculum only) | 2721 steps/hr | **~367.5 hours ≈ 15.3 days** |
| Decision 3 ON for the ENTIRE run | ~690 steps/hr (ESTIMATE) | **~1449 hours ≈ 60.4 days** |
| Decision 3 ON for a bounded window only (e.g. first 100k steps, then OFF) | mixed | 100,000/690 + 900,000/2721 ≈ 145 + 331 = **~476 hours ≈ 19.8 days** |

**This table is presented, not resolved into a single decision** — §2 Decision 3 explicitly declines
to pick the window extent, since a 4×+ wall-clock multiplier is a GPU-budget-appetite call, not a
docs-inferable one. The bounded-window row is included specifically to show the operator that a
partial-run activation is far cheaper than whole-run and may be the practical middle path if
GENERALIZES lands.

**Vast $ estimate:** using the reference-hardware convention (`docs/rules/perf-targets.md`,
5080+9900X is the "vast" mirror) and the `v6_sustained.yaml` header's own cost convention
(~$0.207/hr cited there for an older/smaller instance — **this run's actual current vast 5080 rate
should be re-confirmed at launch time**, rental pricing drifts): at a placeholder **~$0.35-0.50/hr**
for a 5080+9900X-class instance (typical vast.ai spot range for this GPU tier as of this design's
writing — NOT independently verified against a live vast.ai listing this session, flagged as an
assumption): Decision-3-OFF run ≈ 15.3 days × 24h × $0.40/hr ≈ **$147**; Decision-3-ON-whole-run ≈
60.4 days × 24h × $0.40/hr ≈ **$580**; bounded-window ≈ 19.8 days × 24h × $0.40/hr ≈ **$190**.
**Re-verify the hourly rate against the actual vast.ai listing before committing to a multi-week
run** — this is a placeholder assumption, explicitly marked as such per the dispatcher's "state vast $
estimate with assumptions" instruction.

---

## 10. Launch checklist (operator steps)

1. **Vast host state (per coordinator supplemental, 2026-07-03):** canonical checkout
   `/workspace/hexo_rl`, branch `phase4.5/d-solver`, origin now synced to `a33f7df` — `git fetch` +
   verify before launch (per `memory/fetch-before-merging-to-master.md` discipline, even though this
   is not a master merge — stale-branch risk is the same class). GPU idle, 59G free disk (re-check
   `disk_guard` thresholds, `configs/training.yaml:210-214`, against this free-space figure before a
   multi-week run). **Pre-launch cleanup on vast:** discard the dirty `configs/variants/z2_solver_in_loop.yaml`
   working-tree diff (the operator's tuning patch is already archived locally at
   `reports/vast_dsolver_bench/z2_solver_in_loop_operator_tuning.patch` — safe to `git checkout --`
   the file on vast) and remove the 4 bench logs sitting in the vast repo root (not `reports/`, which
   is gitignored — these are presumably loose files from an ad-hoc redirect, confirm before deleting).
2. **HOST-MATCH baseline first (§1.5):** before reading ANY gate that compares against a fixed
   reference (threat-probe anchor, trap-flip baseline), re-mint the self-baseline ONCE on whichever
   host runs this launch (vast) — do not reuse the laptop-side A1 numbers cross-host.
3. **Anchor + corpus preflight** (mirrors the D-1M launch's own checklist,
   `configs/variants/longrun_v6_live2_ls_gumbel_m16.yaml` header):
   - **CORRECTIVE STEP (AMENDED at D-PRELAUNCH dry-run, 2026-07-04 — the D-1M raw-bootstrap cp
     is INSUFFICIENT for this variant):** copying the RAW `bootstrap_model_v6_live2_8300.pt` into
     `best_model.pt` passes the sha pin but hard-fails `resolve_anchor` on encoding
     (`anchor.py:249` — raw stamp resolves `v6_live2` vs the variant's declared `v6_live2_ls`;
     `anchor_encoding_mismatch`, "refusing to fall through" — observed live in the acceptance
     dry-run). The anchor incumbent must be the step-3a MINTED `_ls`-stamped artifact instead:
     `cp checkpoints/run2_bootstrap_v6_live2_ls.pt checkpoints/best_model.pt && rm -f
     checkpoints/best_model.pt.bak`, then RE-VERIFY:
     `scripts/anchor_sha256.py checkpoints/best_model.pt` → `ebf2ed39…ba6db8` — the pin is
     UNCHANGED because `checkpoint_state_sha256` hashes MODEL WEIGHTS ONLY (sorted keys + raw
     tensor bytes; its own docstring) and the mint carries byte-identical weights. Ordering
     consequence: step 3a (mint) must run BEFORE this anchor copy.
   - `rm -f checkpoints/replay_buffer.bin checkpoints/replay_buffer.bin.recent.npz` (no stale buffer
     from a prior run bleeding into this fresh bootstrap).
   - `sha256sum data/bootstrap_corpus_v6_live2.npz` matches the expected corpus — OR, if Decision 7's
     recommended regeneration happened, the new `bootstrap_corpus_v6_live2_ls.npz` sha is recorded in
     the run log and the yaml's `mixing.pretrained_buffer_path` is repointed at it.
   - SealBot native import preflight (`python -c "from hexo_rl.bots.sealbot_bot import SealBotBot;
     SealBotBot(time_limit=0.1)"`).

3a. **Mint the re-stamped weights-only launch artifact (BLOCKER — the raw bootstrap cannot be fed to
   `--checkpoint` directly).** `bootstrap_model_v6_live2_8300.pt` is an unstamped bare state_dict
   (`configs/eval.yaml`'s own note: "currently-shipped bootstrap_model_*.pt files are unstamped bare
   state_dicts"); the loader's filename disambiguation resolves it `v6_live2` (the filename lacks
   `_ls`), and the explicit-encoding reconciliation then raises
   (`hexo_rl/training/trainer_ckpt_load.py:148-156` — "Encoding version disagrees ... Refusing to
   silently override either direction") against the variant's `encoding: v6_live2_ls`. **No CLI
   override exists.** Produce `checkpoints/run2_bootstrap_v6_live2_ls.pt` mirroring
   `scripts/make_ws3v3_warmstart.py` semantics: payload EXACTLY `{model_state, metadata, step}`
   (weights-only — the variant's fresh `lr`/`lr_schedule`/`total_steps` win, which is exactly what a
   fresh launch wants), `metadata['encoding_name'] = 'v6_live2_ls'` after asserting the wire
   signature `(4, 19, 362, True, "size_19")` is shared (it is — the weights load no-reshape), and
   **`step = 0`** (the run's absolute-step schedules — radius curriculum at 200k/400k/600k, mixing
   decay, cosine T_max — all key off the trainer's absolute step, which starts at the artifact's
   `step` field). NOTE: the existing script cannot be reused as-is — its `restamp_encoding` hard-
   requires an existing `metadata['encoding_name']` on the input (`make_ws3v3_warmstart.py:109-113`),
   which a bare state_dict lacks; either extend it with a `--synthesize-metadata` path (bare input →
   fresh `{encoding_name}` metadata + `step=0`) or mint via an equivalent 15-line one-off, pasting
   the printed sha + stamp line into the run log either way.
   **Intent owned explicitly:** this run WARM-STARTS its NN weights from the v6_live2-pretrained
   8300 bootstrap — "fresh from bootstrap" means fresh *training lineage* (step 0, fresh LR schedule,
   fresh buffer, no self-play inheritance), NOT random-init weights. This differs from the D-1M
   precedent, whose header `cp`'d the bootstrap only into `best_model.pt` (the eval incumbent);
   feeding it to `--checkpoint` as the weight init is this design's deliberate choice, consistent
   with the resolvers' own `_ANCHOR_PATHS` note ("TREATMENT launches from the CONTROL's bootstrap —
   the 4-plane/362 v6_live2 weights load into v6_live2_ls with NO reshape"). The pretrain weights
   were produced from the human-game corpus (F1-exempt provenance, Decision 7(a)), not from any
   single-window self-play lineage.
4. **Engine test green:** `cargo test -p engine` (326+/0/2, per Decision 1's precondition) and
   `make bench` ≥73,000 sim/s floor (bench-gate skill; the 4 pulled logs in §9 already confirm this,
   re-run once more immediately pre-launch if any engine file changed since 2026-07-02).
5. **Registry audit:** `python -m hexo_rl.encoding audit` — confirms `v6_live2_ls`'s
   `PolicyPool::LegalSetScatterMax` parses, the `is_multi_window ⇒ value_pool≠None` invariant holds,
   and the Python `_REGISTERED_NAMES`/resolver paths agree with the Rust registry.
6. **Launch command** (real entrypoint — NOT `-m hexo_rl.training.run`, per the D-WS3V3 runbook's own
   corrected-mistake note; `--checkpoint` = the step-3a minted artifact, NEVER the raw bootstrap):
   ```bash
   .venv/bin/python scripts/train.py \
       --checkpoint checkpoints/run2_bootstrap_v6_live2_ls.pt \
       --variant run2_mw_fresh \
       --iterations 1000000 \
       2>&1 | tee logs/run2_mw_fresh.log
   ```
   `--iterations 1000000` is MANDATORY alongside `total_steps: 1000000` in the yaml (§4) — a mismatch
   silently mis-anneals the cosine schedule (D-1M audit C2). Verify the startup log's
   `checkpoint_encoding_resolved` / `train_encoding_resolved` both read `v6_live2_ls`
   (`is_multi_window: true`) and no `checkpoint_encoding_overrides_variant` warning fires — the same
   in-run confirmation the ws3v3 FIX1 validation used.
7. **5k gate (first mandatory stop):** run `make probe.latest` per `docs/rules/workflow.md`'s
   session-start-independent rule ("run at training step 5000 as the kill criterion for every new
   sustained run"), plus the §5 machinery-gate checklist. Do not proceed past this point unattended.
8. **Decision 3 activation, if/when the v3 smoke lands GENERALIZES (§7):** edit
   `configs/variants/run2_mw_fresh.yaml` in place — flip `solver_enabled: true`, set
   `seed_fraction`/`seed_corpus_path` per the smoke's winning arm, overwrite the `solver_visit_weight`
   placeholder with the validated value (§2 Decision 3) — and resume training with
   `--checkpoint <latest run2 checkpoint>` (weights-only or full, per whichever the operator wants the
   LR schedule to inherit; note the RESUME_CHECKPOINT_OWNED_KEYS precedence rules,
   `memory/resume-variant-override-bug-fixed.md`, before choosing).
9. **Re-mine the seed corpus periodically** if Decision 3 is active (§2 Decision 3's "in-loop miner"
   mitigation) — an operator/ops action, not expressible in the yaml.
10. **Session hygiene:** kill any lingering training/benchmark processes before this launch
    (`docs/rules/workflow.md` "Kill running processes before starting new ones"); this is a
    multi-week run — no other training process may share the GPU concurrently.

---

## Open questions (could not resolve from docs — do not guess silently)

1. **§1.6 `legal_move_radius_jitter` dead-code finding:** is this a genuine multi-month latent
   regression on `vast.yaml`/`v6_sustained.yaml` (both declare `legal_move_radius_jitter: true` and
   have run this way in production), or is there a compensating mechanism this session's ~40-minute
   source read missed? A 20-game smoke would settle it cheaply (§1.6) but was not run this session
   (out of scope for a DESIGN dispatch with no engine/test-execution mandate). Recommend running it
   before or alongside this launch, and separately flagging the historical `vast.yaml`/`v6_sustained.yaml`
   runs' documentation as potentially overclaiming jitter's role in their own anti-colony story.
2. **Decision 3's window extent** (whole-run vs bounded, §2 Decision 3 / §9): a GPU-budget-appetite
   call for the operator at smoke-verdict time, not resolved here by design.
3. **RESOLVED at review (was: unexplained `aux_opp_reply_weight: 0.0` convention).** The rationale
   exists: CF-5, `configs/variants/v6_live2_golong.yaml:70-76` — the opp-reply aux head is trained on
   the current-player MCTS visit target, redundant with the policy head, a flat-policy-reinforcing
   gradient spending 15% of loss weight; source analysis
   `reports/investigations/phase6_scoping_memo_20260531.md` §3. §4's config table now cites it.
   The rationale is head-wiring-level, not encoding-level — it transfers to this run unchanged.
4. **Vast.ai hourly rate for a 5080+9900X instance** (§9): used a placeholder ~$0.35-0.50/hr range,
   not verified against a live listing this session. Re-confirm before committing budget.
5. **`exploit_probe.py` / `probe_threat_logits.py` summary-JSON schema gap** (§3.4): a cheap follow-up
   (mirror the trap-flip evaluator's `encoding_decode`/`expect_encoding`/`loader` fields) is flagged
   but not built — out of this DESIGN dispatch's scope; worth a small dedicated commit before this
   run's data is used for a load-bearing verdict at 100k+.
6. **`MAX_CHILDREN_PER_NODE=192`** (`engine/src/mcts/mod.rs:46`) remains a hardcoded Rust const, not
   the `config_key` the `dmulticluster_362_legalset_design.md` §3.2/§9.6 flagged as a TODO. Per §9's
   ragged fix, off-window-covered cells now carry real priors (truncation is by true prior, not
   auto-last), so this is a **residual, not a known-active** risk for this run — but it was not
   independently re-verified this session that no spread position in early multi-window self-play
   exceeds `n_legal=192` in a way that still matters post-fix. Worth a cheap telemetry check (log
   `n_legal` distribution per game) at the 25k gate if spread/fragmentation canaries (§6) look
   borderline.
7. **Jitter-inertness coverage gap (from the M4 review correction to §1.6):**
   `test_worker_loop_jitter_does_not_override_v6w25_radius` was removed outright in the §P3.2 test
   retirement and its two named replacement tests contain zero jitter references — the
   registry-spec-suppresses-jitter behavior §1.6's finding rests on is currently UNTESTED at the
   engine-test level (the source-level argument stands; the test that would pin it does not exist).
   The §1.6 20-game smoke doubles as the cheapest re-cover; a proper regression test is a small
   follow-up commit.
8. **PARTIALLY RESOLVED 2026-07-04 (D-PRELAUNCH) — K>1 exposure timing under the radius curriculum
   (from the Decision 2 review addition):** at radius ≤ 5 (< `cluster_threshold=5` adjacency bound),
   disjoint clusters cannot form, so the S1/S2 stages (steps 0-400k) exercise multi-window K>1
   dynamics only via massive-cluster multi-anchoring at an unknown rate.
   **Resolution taken: the 4→5→6→8 economics-first staging is KEPT, with a signature escape hatch
   added** — the §2 Decision 2 widen-early gate (operator-procedural, no trainer machinery) advances
   a stage before its step threshold when `forced_win_conversion` saturates at the current radius
   (EMA ≥ 0.85, slope ≤ +0.02, ≥2 consecutive §5 reads, n ≥ 30; widen-only, hysteresis, irreversible
   — full definition and band derivation in §2 Decision 2; matching comment block in the yaml). This
   bounds the worst case of the economics-first choice: a net that masters the cheap-radius regime
   early no longer waits idle for the step threshold to buy K>1 exposure. **Still OPEN:** the flat-
   radius-6-from-step-0 alternative remains an operator call at launch, and the massive-cluster
   multi-anchor mechanism's K>1 rate at fresh-bootstrap play strength has never been measured (v2's
   max_k=3 read was on a 200k-trained net); the §5 `k_ge2_row_fraction` counter measures actual
   exposure either way.
9. **RESOLVED 2026-07-04 (D-PRELAUNCH) — Decision 7 regeneration wiring BUILT and corpus
   REGENERATED.** (Was: `export_corpus_npz.py --encoding v6_live2_ls` does not run today.) The
   per-cluster-row emission was built (`hexo_rl/bootstrap/dataset.py::replay_game_to_triples_ls`),
   the choice wired in, and `data/bootstrap_corpus_v6_live2_ls.npz` regenerated — see the
   resolution note at the top of §2 Decision 7 for shas, row counts, and the snapshot caveat.
10. **RESOLVED 2026-07-04 (D-PRELAUNCH) — decision: ENABLED.** (Was: D-EVALFOUND
   `objective_a_coverage_gap` warning fires on this config — with `offwindow_adversary.enabled:
   false` the pipeline warned "no Objective-A signal active ... a run can promote an
   off-window-exploitable checkpoint unguarded"; the IN-loop promotion path had no Objective-A
   guard as configured.) The yaml now sets `eval_pipeline.opponents.offwindow_adversary:
   {enabled:true, stride:1, n_games:100, arm:"exploit", model_sims:128}` per the
   `D_EVALFOUND_design.md` §7 recipe (config shape = `v6_live2_golong.yaml` precedent; stride
   tightened to 1 because the promotion gate treats missing `robustness_rate` as pass). The rate
   feeds `decide_promotion` (blocks at code-default `gating.robustness_threshold=0.06`) + the
   step-coordinator WARN — see the §5 "In-loop Objective-A robustness guard" block and the §4
   config-table row. **Trust-scope caveat carried with the resolution:** the adversary shares the
   `exploit_probe` code path's known arm-aliasing bug (`memory/exploit-probe-arm-aliasing-bug.md`)
   — only the single-arm ABSOLUTE forced rate (what the in-loop monitor reads) is trustworthy;
   exploit-vs-control CONTRAST verdicts stay contaminated until that bug is fixed and must not be
   added to (or cited from) the in-loop gate.
