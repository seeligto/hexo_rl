# D-STRIX S2 — mechanism attribution + portability judgments

**Role:** S2 of the D-STRIX dispatcher. Consumes S1's economics diff
(`docs/handoffs/d_strix_s1_economics_diff.md`, DONE). This doc judges each
candidate driver PORTABLE / NOTE-ONLY / REJECT and specs the two driver
build-outs (radius curriculum, tiny-net probe) plus one banked card
(axis-graph). No hexo-strix code vendored. No HeXO source changed — this is
the only file this session writes.

## Governing caveat (inherited from S1, restated because every verdict below depends on it)

hexo-strix's repo is a **code-only export** — `docs/`, `runs/`, `data/`,
`*.pt`, `trajectories.json` are all gitignored; zero committed run logs,
zero SealBot results, zero self-play throughput number. Every "their X
works" claim below is about a **configured-recipe mechanism** (what the
TOML says it intends to do), not a verified realized outcome. Where a
HeXO-side prior exists (falsified register, banked memory), that prior is
weighted as REAL evidence; where the only evidence is "strix's config says
so," it is weighted as a **hypothesis worth testing cheaply**, not a proven
result to import wholesale. Days-to-parity is not comparable (S1 §
"SealBot-anchor comparability verdict").

---

## Verdict 1 — Board-size curriculum: **PORTABLE (radius axis only)**

### Mechanism

Small legal-move space early → short decisive games → dense z-signal →
coherence learned before the search space explodes → widen on a schedule.
No encoding change, no GNN, no restart required for the radius axis.

### What strix actually has (read from `configs/axis/curriculum.toml` +
`configs/curriculum.toml` + `hexo_a0/curriculum.py`)

Two curricula exist (axis-graph 5-stage, hex-native 6-stage), both varying
**two** axes per stage: `win_length` (4→5→6) **and** `placement_radius`
(2→4/6→6→8). Stage advance uses `convergence_mode` (checkpoint-plateau:
win-rate vs a lagged checkpoint drops below `improve_threshold` for
`patience` evals; or SPRT: log-likelihood-ratio promotion test with a
pooled-Wilson plateau detector and a promotion-velocity drought detector).
Confirmed via `hexo_rs.GameConfig(win_length=..., placement_radius=...,
max_moves=...)` in `curriculum.py:697` — both are genuine **game-rule**
parameters, not just NN-input-window parameters, in their engine.

**Stage semantics DO transfer to HTTT.** Checked their turn structure
directly (`hexo-rs/tests/integration.rs`, `hexo-mcts/src/axis_graph.rs:551`,
`README.md:3`): "P1 opens with 1 move, then P2/P1 alternate 2 moves per
turn" — **byte-identical compound-turn rule to HeXO's**, not a plain
one-stone-per-turn Hex clone. `win_length` = length of the winning line
(same concept as our `WIN_LENGTH`); `placement_radius` = same mechanism
class as our `LEGAL_MOVE_RADIUS` (a global cap on how far a new stone may
land from any existing stone). No turn-vs-ply unit mismatch risk here
(§D-COHERENCE re-validation check, satisfied) — both projects count wins in
stones-in-a-row and turns in 2-move compound units identically.

### The axis split: radius is ALREADY BUILT; win_length is NOT

This is the load-bearing finding of this driver. HeXO's radius axis has a
**live, wired, production-used analog already**:

- `engine/src/board/moves.rs` — `Board.legal_move_radius` is a per-game
  runtime field (not a compile-time const), with `set_legal_move_radius`
  PyO3 binding, `legal_move_radius_jitter` (samples r ∈ {4,5,6} per game),
  and a full `legal_move_radius_schedule` mechanism already threaded
  through `hexo_rl/selfplay/pool.py` → `engine/src/game_runner/{config,mod,
  worker_loop}.rs`.
- **Already live in the canonical `configs/variants/vast.yaml`:**
  `legal_move_radius_schedule: [{step:0,radius:5},{step:50000,radius:6},
  {step:100000,radius:7},{step:150000,radius:8}]`, paired with
  `legal_move_radius_jitter: true` — this is literally a step-keyed radius
  curriculum, running today, calibrated for the **v6w25 (25×25 window)**
  encoding.
- A 2-stage smoke variant (`configs/variants/smoke_radius_curriculum.yaml`,
  R=5→6 at step 5000) exists but **has no committed eval result** in the
  sprint log — the schedule mechanism is wired but its own isolated effect
  (schedule vs flat-radius control) has never been measured.

By contrast, `win_length` is a **hardcoded Rust `const WIN_LENGTH: usize =
6`** (`engine/src/board/moves.rs:44`, mirrored in `threats.rs:20`), baked
into `count_in_line`, `has_player_long_run`, threat-window sliding-window
scans, and MCTS backup comments ("`WIN_LENGTH - 1`" appears in
`mcts/backup.rs`). Making it a runtime, schedulable field is a genuine
engine refactor across board move-generation and threat-detection **hot
paths** — squarely inside `bench-gate` skill scope (any edit to
`engine/src/board/**` touching MCTS/threat hot loops requires a `make
bench` baseline/diff before landing), not a config-only change.

**Scope decision for this spec: ship radius-only.** Treat `win_length`
curriculum as a separate, larger, bench-gated future item — NOT part of
this spec's deliverable. HeXO stays fixed at 6-in-a-row (current phase
status) for the curriculum's lifetime; only `placement_radius` schedules.

### Re-validation against the prior falsified rows (CLAUDE.md protocol)

- **§174 "LEGAL_MOVE_RADIUS 8→5 at bootstrap fixes v6w25 selfplay
  collapse" — FALSIFIED** by the §174 R∈{1..8} smoke: median game length
  identical across all radii: **radius does not move bootstrap quality**.
  Context: this tested a **static, bootstrap-time** radius value against a
  **quality/collapse** outcome. It does **not** transfer to reject a
  **schedule-during-training** aimed at **wall-clock economics** (cheap
  early games, more positions/hr) — a different variable (schedule vs
  static value) against a different objective (economics vs
  collapse-quality). The sprint log itself draws this line explicitly:
  "`legal_move_radius_schedule` retained as downstream-training lever, not
  bootstrap-time hyperparameter" (line 1385). Verdict: **prior does not
  block this driver**, but it does set correct expectations — don't expect
  the schedule alone to fix quality; its payoff is economics + dense
  z-signal, and that payoff itself is UNMEASURED (no eval exists yet).
- **L9 / §156-§157 — cosine-temperature is the load-bearing knob in
  draw-collapse; mandatory pairing with `legal_move_radius_jitter` when
  cosine-temp is active.** Context: R12 found cosine-temp schedule alone
  drives ~5%→91% draws, and re-enabling it reopens the §147/§154
  colony-attractor (rate 67%, mitigated to trace levels only by jitter).
  Current canonical default has cosine-temp **OFF**
  (`temperature_threshold_compound_moves: 0`, fixed τ=0.5) and jitter
  **already default-on** — so this hazard is currently INACTIVE, not
  triggered by adding a radius schedule. It transfers as a **hard
  guard**: this curriculum spec must never combine a radius schedule with
  a re-enabled cosine-temp schedule unless jitter is confirmed on. State
  as an explicit DO-NOT in the spec below, not a live risk today.
- **§S178-§S181 colony-attractor arc (10+ config-level anti-colony levers,
  ALL exhausted with zero escape at sustained scale)** — this is the
  single biggest banked-result caution against adding "one more training
  schedule knob" as a cure-all. Context: bot-mix share, `ply_cap_value`
  split, `completed_q_values` flip, `game_length_weights`, cosine-temp —
  every config-level lever failed to prevent the colony attractor past
  ~step 40-50k; the arc's own closing verdict routes to **code-level**
  levers (PSW, corpus refresh). Does this transfer to reject the radius
  curriculum? **Partially, as a scope caveat, not a kill:** the radius
  curriculum's claimed mechanism is economics (cheaper early wall-clock)
  + early coherence, not "escape the colony attractor" — it was never
  pre-registered as an anti-colony lever, so the exhausted-arc rows don't
  directly falsify it. But it IS a NEW schedule knob riding on the same
  config surface that arc exhausted, so this spec's canary list (below)
  explicitly includes a colony-fraction / SealBot-WR trajectory watch, not
  just a draw-rate watch — cheap insurance given that history.
- **v6_live2_golong.yaml's own precedent — do not reuse `vast.yaml`'s
  5→8 schedule verbatim.** That variant explicitly declined `vast.yaml`'s
  `legal_move_radius_schedule` because "vast's 5→8 is a **25×25** (v6w25)
  curriculum; v6_live2 is **19×19**" — keeping only the base jitter set
  {4,5,6}. The NEXT full run (per the D-WS3V3 track: `ws3v3_arm_control/
  inject/seeded.yaml`, all on `encoding: v6_live2_ls`, a 19×19-window
  encoding) must NOT blindly inherit vast.yaml's radius endpoints; any
  radius schedule's max must be checked against that encoding's window
  half-size via the registry before use. **This is the one concrete
  conflict this audit found between the curriculum idea and a banked
  result** — flagging it prevents silently re-running an
  already-rejected-for-window-mismatch schedule.

### Curriculum variant SPEC (not created — spec only)

**Stages** (radius-only; win_length fixed at 6 throughout):

| Stage | `placement_radius` (legal_move_radius_schedule step trigger) | rationale |
|---|---|---|
| S1 | jitter{3,4,5}, schedule floor = current DEFAULT_LEGAL_MOVE_RADIUS-1 | cheapest/shortest games, coherence-first |
| S2 | jitter{4,5,6} | current production default jitter set — known-safe baseline |
| S3 | jitter{5,6,7} | widen toward true rule bound |
| S4 (final) | jitter{6,7,8} — **8 is the hard rule ceiling** (`LEGAL_MOVE_RADIUS` official rule, `docs/07_PHASE4_SPRINT_LOG.md:109`) | matches deployed game rule; no further widening possible |

Trigger keying: **step-count** for the MVP (zero new code — reuses the
existing `legal_move_radius_schedule: [{step, radius}, ...]` list
mechanism verbatim, same as `vast.yaml`/`smoke_radius_curriculum.yaml`
already do). A convergence-gated version (advance on SealBot-WR or
`wr_best` plateau, à la strix's `checkpoint`/`sprt` modes) is a **follow-on
only** — it requires new code (a stage-advance decision function reading
the existing `eval_pipeline` trajectory) and is not needed to validate the
core economics hypothesis. Ship step-keyed first.

**Encoding/window gate (mandatory pre-flight, per the v6_live2_golong
conflict above):** before setting any stage's radius, confirm
`radius ≤ floor(window_size / 2)` for whichever encoding the target run
declares (check `engine/src/encoding/registry.toml` via
`hexo_rl.encoding.lookup(name)`), not by copying another variant's numbers.

**Draw-collapse canary:** reuse the existing live monitor
(`monitors.hard_abort_draw_rate` + `hard_abort_draw_rate_consec`, already
wired — see `configs/variants/ws3v3_arm_control.yaml:122-123`) at each
stage boundary; a stage transition is exactly the kind of distribution
shock L9's cosine-temp lesson warns about, so watch draw-rate for **3
consecutive evals post-transition**, not just steady-state.

**Jitter pairing (mandatory, per L9):** `legal_move_radius_jitter: true`
at every stage — non-negotiable, already the safe default, do not disable
to "clean up" the schedule.

**Colony canary (per the exhausted §S178-181 arc caution above):** watch
`colony_extension_fraction` / `colony@sealbot` alongside SealBot-WR
trajectory at each stage boundary — not because the curriculum is expected
to cause colony capture, but because it's a new schedule knob on the same
surface that arc exhausted, and the canary is nearly free (already-emitted
telemetry).

**Composition with v3 seeding + solver-z machinery (D-WS3V3 track):** the
radius schedule is orthogonal to `solver_enabled`/`seed_fraction` (both
gate the self-play move-injection path; radius gates the legal-move
generator upstream of that). No interaction expected, but the NEXT full
run should hold radius-schedule OFF (flat, current default) in the
ARM-CONTROL validity-gate arm (per `docs/handoffs/d_ws3v3_smoke_runbook.md`
§3 — that arm's whole point is a single-variable diff vs the warm-start
anchor) and introduce the radius schedule only as its own, later,
single-variable arm — never bundled with the solver/seeding variable in
the same run.

---

## Verdict 2 — Axis-graph line-topology prior: **NOTE-ONLY (banked card)**

**Banked representation card**

- **Mechanism:** bake win-line/axis topology directly into the input
  representation (graph edges along the 3 hex axes, or an equivalent
  CNN-side structural prior) so the net doesn't have to learn "a line is 6
  colinear stones" from raw stone-presence planes.
- **Why not now:** this is a genuine **input-representation change** =
  encoding-class change = restart-class cost (new bootstrap, new
  compatibility surface), and the one HeXO experiment closest to this
  mechanism — §D-FULLSPEC's threat-plane full-spectrum distill — found the
  representation change **still couldn't separate win/loss** on a
  **frozen v6_live2 trunk** (matched-KILL-C craters 0.32-0.46; in-sample
  46% rules out under-power). That result is scoped to a **frozen-trunk,
  input-plane-only** addition, not a from-scratch retrain with the
  topology baked into the trunk architecture itself — so D-FULLSPEC is
  NOTE, not KILL, for this exact mechanism (context does not fully
  transfer: frozen-trunk feature-injection ≠ from-scratch structural
  prior). Still too expensive to justify absent a restart already being
  evidence-justified for other reasons.
- **When to unlock:** only after a restart is independently justified
  (e.g. a genuinely new encoding migration already on the roadmap for
  other reasons) — bundle this mechanism into that restart's design
  rather than paying restart cost solely to test it.
- **Evidence that would justify unlocking early:** a **from-scratch**
  (not frozen-trunk) ablation showing line-topology input features improve
  early-training sample efficiency at bootstrap-cheap scale (e.g. inside
  an existing curriculum S1-scale run, near-zero marginal restart cost) —
  that would be the cheap discriminator to run *before* committing to a
  full production restart.

---

## Verdict 3 — Tiny net: **TESTABLE-CHEAP**

### Mechanism hypothesis

19× fewer params (222K vs 4.25M, both MEASURED per S1) → cheaper
forward+backward → more games/hr per fixed compute → faster wall-clock
learning curve, independent of any final-strength-ceiling claim (a
smaller net may plateau lower; this probe does not test that).

### Confound to isolate first: aux-head count is a SEPARATE axis from width

HeXO carries 6 active-weighted auxiliary heads (`aux_opp_reply_weight:
0.15`, `ownership_weight: 0.2`, `threat_weight: 0.2`, `aux_chain_weight:
1.0`, plus value-uncertainty 0.1; `ply_index_weight: 0.0` disabled) vs
strix's bare 2-head net (policy + value only) — confirmed via
`configs/training.yaml` weight block vs `hexo-a0/src/hexo_a0/model.py`
(S1's diff table). Width and aux-head-density are two independent axes
that both currently distinguish HeXO from strix. **The probe must not
collapse them into one "strix-like" arm** — that would leave the result
uninterpretable (can't tell if a throughput gain came from fewer
parameters or fewer loss heads).

### Probe spec (economics only — no strength claim; spec only, GPU decision is operator's)

**Design: 2×2 grid, short run, throughput + loss-curve slope only.**

| Arm | Width (`filters`/`res_blocks`) | Aux heads |
|---|---|---|
| A (baseline) | 128 filters / 12 res_blocks (current) | all 6 active |
| B (half-width) | 64 filters / 12 res_blocks | all 6 active |
| C (quarter-width) | 32 filters / 12 res_blocks | all 6 active |
| D (baseline, aux-off) | 128 filters / 12 res_blocks | policy+value only (all aux weights → 0) |

(Res-block depth held constant across A-C; only channel width varies —
narrower-but-same-depth is the cleanest single-variable width probe.
Depth is a separate, larger architecture question, out of scope here.)

**What to measure (each arm, same short run length — e.g. 2-5K steps,
whatever the existing smoke cadence budget allows):**
- games/hr and pos_gen/hr (direct throughput multiplier vs baseline A;
  this is the entire economics claim).
- wall-clock-to-fixed-loss-milestone (e.g. steps × step-time to reach a
  fixed policy-loss value) — the "faster learning per wall-clock"
  hypothesis needs a **wall-clock** x-axis, not a step-count x-axis (the
  same distinction §D-GUMBELSIMS had to correct for after its Phase-3 A/B:
  step-matched comparisons hide a throughput multiplier that only shows up
  wall-clock-matched).
- threat-probe C1-C3 (existing gate, cheap, already required per every
  5k-step checkpoint) — sanity-only, not a strength verdict.

**What this probe does NOT tell us (state explicitly, don't let it be
over-read later):** final strength ceiling at full training budget, value-
head quality at deep/hard positions (D-LOCALIZE's value-blind-cell class),
or whether a narrower net can still carry 6 aux heads' gradient signal
without degrading the main policy/value heads (arm B/C vs D structure
partially probes this, but only at short-run scale — a real answer needs
the full sustained-run budget, out of scope for a cheap probe by
definition).

**Decision criterion:** if B or C's games/hr multiplier × wall-clock-to-
milestone improvement is large (e.g. ≥2× either axis) with threat-probe
C1-C3 still PASS, that's sufficient justification to escalate to a real
GPU-week sustained comparison (operator decision, not this probe's call).
If the multiplier is small (<1.3×) or threat-probe regresses, the tiny-net
mechanism is not worth the restart-adjacent risk (new bootstrap needed for
any width change — width is a checkpoint-shape-breaking change, same
restart class as the axis-graph driver, just far cheaper to iterate on
since no encoding change is required).

---

## Verdict 4 — No-window simplicity: **already actioned (confirmation only)**

hexo-strix's "no window, no cross-cluster K-aggregation, single flat
graph" simplicity is exactly the class of defect the single-window cascade
was (D-FORENSIC F1: whole d1m lineage self-played single-window from a
string-form `encoding:` bug; D-DECODE: off-window defense needed
multi-window no-drop decoding, not a single-window action space). HeXO's
fix direction is the OPPOSITE of strix's simplicity (multi-window,
no-drop, kcluster) and that fix is validated (D-DECODE kcluster defends at
0.0 forced-loss vs g=0 Gumbel's 0.335/0.165). No new work from this
driver — strix's simpler regime is a different point on the tradeoff
curve (never needed windowing because the axis-graph fully covers the
board), not evidence HeXO's window-fix direction was wrong.

---

## Verdict 5 — Train==deploy search consistency: **NOTE-ONLY (open, not a blanket recommendation)**

### What S1 surfaced

strix uses Gumbel-root+PUCT-interior for **both** self-play data-gen and
deploy/eval — one algorithm, no divergence. HeXO's canonical production
default (`configs/selfplay.yaml`: `gumbel_mcts: false`) self-plays with
PUCT+Dirichlet-root; deploy/eval uses Gumbel g=0 SH m=16/n=150 — a real
root-exploration-mechanism mismatch (Dirichlet-noise root vs
Gumbel-Top-k root). Not in the original dispatcher list; judged fresh here
per instruction.

### Re-validation: this is NOT untested territory — cite precisely

- **`gumbel-interior-puct-canonical` memory (2026-06-23):** the **interior**
  algorithm (PUCT at non-root nodes) is shared and paper-canonical in
  BOTH regimes already — Gumbel-root uses PUCT-interior whenever Gumbel
  runs at all (production default self-play doesn't run Gumbel at all,
  but when it does — e.g. the WS3V3 arms below — the interior mechanism
  is byte-consistent with deploy's). So the real mismatch is narrower
  than "different algorithms end to end": it's specifically **root
  exploration mechanism** (Dirichlet vs Gumbel-Top-k), not tree search.
  That memory also flags a **redundant-Dirichlet-on-Gumbel-root** leak
  (paper: Gumbel MuZero doesn't use Dirichlet at all) as the actual
  non-canonical bug when Gumbel self-play IS used.
- **§D-GUMBELSIMS (2026-06-18, `reports/gumbelsims/SESSION_VERDICT.md`)
  ALREADY TESTED the adjacent question** — "does switching self-play from
  PUCT to Gumbel change final trained strength" — via a direct one-variable
  A/B (`p3_armc_{gumbel,puct}.yaml`, matched encoding/anchor/corpus,
  15k steps each). Result: **PUCT self-play produced a tactically
  sharper/stronger net at matched step** (paired-bootstrap sign robust,
  Gumbel−PUCT Elo [−143,−38.5] at 99.98%; +89 Elo H2H, but **n=1 training
  seed** and explicitly flagged as an **algorithm+sim-budget BUNDLE**
  (Gumbel-100 vs PUCT-600, not matched total sims).
  **[CORRECTED per REVIEW]** The 15k read did NOT stand as the closing
  result: the same investigation ran a cost-matched 50k follow-up
  (Gumbel-opt-50k ties PUCT-15k at ~72% cost; the 15k negative was
  diagnosed as "dominated by undertraining") and closed with verdict
  **GUMBEL-SIMS-NULL / affordability-PARITY** (`FINAL_VERDICT.md`,
  sprint log §D-GUMBELSIMS 50k entry). Only the clean matched-sim
  (m,n) sweep remains OPEN; the matched-wall-clock comparison WAS run.
  Context match: this is the SAME question (self-play algorithm choice
  affecting trained strength) at the SAME encoding family, so it
  transfers directly and must be weighed — it is a real, if underpowered
  and confounded, prior **against** naively switching production self-play
  to Gumbel purely for train==deploy consistency.
- **Independent signal the operator has already partially converged
  here without a formal decision:** the current D-WS3V3 research track
  (`configs/variants/ws3v3_arm_{control,inject,seeded}.yaml`) already runs
  self-play with `gumbel_mcts: true` AND has already applied the
  redundant-Dirichlet fix (`mcts.dirichlet_enabled: false` with the
  comment "Gumbel-Top-k IS root exploration; no double root-noise"). This
  is a narrower research arm (D-SOLVER track), not the canonical
  production default, but it shows the "match deploy's Gumbel root" idea
  is already live in practice on at least one active branch, with the
  known leak already closed there.

### Verdict reasoning

**[CORRECTED per REVIEW]** Not REJECT — the interior-algorithm mismatch
is already resolved (shared, sound), and one active branch has already
adopted root-consistency correctly. Not PORTABLE-as-a-blanket-
recommendation either — but the standing prior is NOT a clean negative:
§D-GUMBELSIMS's cost-matched 50k follow-up closed at
**GUMBEL-SIMS-NULL / affordability-PARITY** (Gumbel-opt ties PUCT at
~72% wall-clock cost; the 15k "PUCT wins" read was diagnosed as
undertraining). **NOTE-ONLY**: train==deploy root consistency is
strength-neutral and mildly cheaper per the one direct test; the still-
open lever is the clean matched-total-sim (m,n) sweep. Already
informally adopted on the D-WS3V3 arms (with the redundant-Dirichlet
leak closed); not a change to force on the canonical default without
the m-sweep.

---

## Summary table

| # | Driver | Verdict |
|---|---|---|
| 1 | Board-size curriculum (radius axis) | PORTABLE — spec above; win_length axis split out as a separate bench-gated engine item, NOT in scope |
| 2 | Axis-graph line-topology prior | NOTE-ONLY — banked representation card, unlock only alongside an independently-justified restart |
| 3 | Tiny net | TESTABLE-CHEAP — 2×2 width×aux-heads probe spec above, economics-only, no strength-ceiling claim |
| 4 | No-window simplicity | Already actioned (D-FORENSIC F1 / D-DECODE) — confirmation only, no new work |
| 5 | Train==deploy search consistency | NOTE-ONLY [CORRECTED per REVIEW] — interior already consistent; §D-GUMBELSIMS closed at GUMBEL-SIMS-NULL / affordability-PARITY (cost-matched 50k tie at ~72% cost, NOT a standing negative); open item is the matched-total-sim (m,n) sweep |

## REVIEW

**Verdict: PASS-WITH-CORRECTIONS** (Verdict 1 and its cites hold up well;
Verdict 5's central citation is stale/incomplete and needs rework; one
mandated re-validation check — D-ARGMAX/opening-diversity — was skipped
entirely).

**Verified correct (no correction needed):**

- `configs/axis/curriculum.toml` (5 stages) / `configs/curriculum.toml`
  (6 stages) both confirmed: `win_length` 4→5→6, `placement_radius`
  2→3/4→6→6→8 across stages — matches the doc's "4→5→6 / 2→4/6→6→8"
  characterization.
- `hexo_a0/curriculum.py:697` confirmed verbatim: `hexo_rs.GameConfig(
  win_length=stage_cfg.game.win_length, placement_radius=..., max_moves=...)`.
- strix's compound-turn rule independently confirmed from their own code
  (`README.md:3`: "two players alternate placing stones (two per turn)";
  `axis_graph.rs` test `mid_game()`: "P1 starts at (0,0). P2 gets 2 moves,
  then alternating 2 each.") — genuinely the same rule class as HeXO's,
  not a plain one-stone Hex clone. No turn-vs-ply unit mismatch.
- `L9` cite verified verbatim against `docs/07_PHASE4_SPRINT_LOG.md:805`
  ("Cosine temperature schedule is the load-bearing knob in draw-collapse.
  Pair with LEGAL_MOVE_RADIUS jitter when active.") — context transfers
  correctly (cosine-temp is OFF in canonical default, jitter is ON; the
  doc correctly frames this as a DO-NOT guard, not a live risk).
- `§S178-§S181` colony-attractor arc: counted ~12 distinct falsified rows
  across the sprint log (§S179, §S180a, §S180b, §S181-T1, §S181-T3, plus
  the static-corpus/alt-V_spread/refresh-hook/plateau/promotion-gate/
  bot-mix/multi-aux-density bullets) — "10+" is accurate, not inflated.
- `v6_live2_golong.yaml` conflict is real: lines 20-21 and 82-83 of that
  file confirm the explicit decline of `vast.yaml`'s 5→8 schedule with the
  exact stated reason (v6w25=25×25 vs v6_live2=19×19). Registry confirms
  `v6_live2_ls` window=19, `v6w25` window=25 (`engine/src/encoding/
  registry.toml`), so the "check radius ≤ floor(window/2) via the registry"
  guidance is sound and the cited conflict is genuine, not fabricated.
- `vast.yaml`'s `legal_move_radius_schedule` "already wired and live" claim
  verified end-to-end, not just config-present: `configs/variants/
  vast.yaml:44-48` → `hexo_rl/training/step_coordinator.py:496-514`
  (`_resolve_radius` reads `selfplay.legal_move_radius_schedule`, calls
  `pool.set_radius_override`) → `hexo_rl/selfplay/pool.py:1015-1022`
  (`set_radius_override` → `self._runner.set_radius_override`). The
  mechanism does what the doc says, not just a same-named key sitting
  unused in two files.

**Correction — Verdict 5's central citation is stale, and the doc's own
re-validation-discipline check (which it explicitly invokes) was not
actually applied to it:**

The doc cites `reports/gumbelsims/SESSION_VERDICT.md` for "PUCT self-play
produced a tactically sharper/stronger net at matched step" (15k-step
result, Elo gap ≈ +89, sign-robust at 99.98%) and states "the matched-
wall-clock (not matched-step) comparison ... left explicitly OPEN by that
investigation's own closing text." **This is incomplete and the "left
open" claim is false.** The same directory contains
`reports/gumbelsims/FINAL_VERDICT.md` (explicitly titled as the closing
adjudication, "every session addition ... the 50k affordability run ...
folded in"), and — more importantly — `SESSION_VERDICT.md` *itself*
already contains the 50k follow-up (its own "FINAL:" paragraph, lines
~49-63) that the doc's citation stops short of. The sprint log's own
sequential, dated record confirms the same thing at
`docs/07_PHASE4_SPRINT_LOG.md:2174-2200` ("§D-GUMBELSIMS 50k affordability
test (2026-06-18) — undertraining dominated the 15k read").

What that follow-up actually found, none of which appears in S2:
- Gumbel-opt-50k **ties** PUCT-15k in absolute strength (Elo 136 vs 135,
  H2H 51-49 dead even) at **~72% the GPU-hour cost** (6.9h vs 9.6h) — this
  IS a cost/wall-clock-matched comparison, and it did not confirm PUCT's
  superiority; it found approximate parity.
- The matched-step gap itself **narrows** with more training (60% H2H at
  15k → 54% at 50k, Elo +89 → +72) and the sprint log states plainly: "the
  15k Phase-3 verdict ... was dominated by undertraining artifacts."
- The off-window "Gumbel defends / PUCT forceable" coherence read (which
  elsewhere in this codebase's memory feeds the D-DECODE/off-window
  narrative) was found to be a **seed-parity exposure confound** on
  red-team (forced test: Gumbel loses 17/17 when actually forced) — by
  50k both regimes defend. Not degraded, not a real Gumbel advantage.
- The actual FINAL_VERDICT.md verdict label is **"GUMBEL-SIMS-NULL /
  affordability-PARITY,"** not the flat "PUCT wins, Gumbel switch
  discouraged" reading the doc presents. The one item genuinely left open
  per that document is an **m-sweep training A/B** (m≈8 vs 16 vs 32 at
  n=100) — not a "matched-wall-clock" sweep, which was in fact run (the
  50k affordability arm) and returned closer to parity.

Net effect: Verdict 5's conclusion ("NOTE-ONLY... a real, context-matched
negative prior... blocks a confident switch-to-Gumbel recommendation")
overstates the strength of the evidence against train==deploy consistency.
The re-validation discipline this doc explicitly invokes elsewhere ("cite
the prior → state the exact context it was falsified in → test whether
that context transfers") was not applied to its own headline citation here
— it cited the early section of a document whose own later section (and a
sibling FINAL_VERDICT.md in the same folder) already revises the finding.
Recommend Verdict 5 be rewritten around "affordability-parity, m-lever
still open" rather than "PUCT wins at matched step."

**Gap — one mandated re-validation check was skipped:**

The dispatcher's review mandate explicitly asked whether small-radius-early
conflicts with "opening diversity (D-ARGMAX effective-n lesson)" alongside
L9 and the colony-attractor arc. **D-ARGMAX / opening-diversity does not
appear anywhere in this document** (`grep -c ARGMAX` → 0 hits) — only L9
and the colony arc were actually checked, despite the doc's framing
implying a complete re-validation sweep. This is a live, non-trivial
question the doc should have addressed: D-ARGMAX's lesson is that
argmax/temp-0 deployment collapses to near-duplicate games without
injected opening variation — and a small-`placement_radius` early
curriculum stage mechanically *shrinks* the legal-move space, which could
plausibly *reduce* opening diversity at exactly the training stage where
it matters most for effective-n. This should be checked (or explicitly
scoped out with a reason) before the curriculum spec ships, not silently
dropped.

## RED-TEAM

### Target: conclusion B — "radius curriculum = the big portable win"

**(i) Does HeXO's OWN history show a benefit from the schedule it already
runs?** Checked directly (`grep -n legal_move_radius_schedule` across
`configs/`, `hexo_rl/`, `engine/`, sprint log): the schedule appears in
exactly two places — `configs/variants/vast.yaml` (live, 5→6→7→8 at
steps 0/50k/100k/150k) and `configs/variants/smoke_radius_curriculum.yaml`
(2-stage smoke, R=5→6 at step 5000). **Zero sprint-log hits for either
config name in the context of a schedule-vs-flat-radius comparison, and
zero hits for "radius" co-occurring with "density," "z-signal,"
"decisive," "dense," "early-signal," or "coherence"** in the whole
sprint log. §175 (the vast.yaml lineage the CLAUDE.md phase header calls
"§175 v6 sustained... closed by interrupt at step 70176") ran *past* the
schedule's first transition (step 50000, radius 5→6) — meaning the
schedule fired for real, in production, for tens of thousands of steps —
and **no diagnostic anywhere ever isolated what that transition did**, not
even a before/after game-length or draw-rate glance. The doc's own S2
text already half-admits this ("its own isolated effect... has never been
measured") but still labels the driver "the big portable win" rather than
"an untested mechanism riding on an unexamined multi-week production run."
**Confirmed: the "big win" framing rests on zero internal evidence, for or
against.** This doesn't kill the idea (nothing falsifies it either) but
the doc's summary-table language ("PORTABLE") should be downgraded to
match Verdict 3's own honesty standard ("TESTABLE-CHEAP") — it's the same
epistemic state (plausible, cheap to spec, zero measured effect) as the
tiny-net driver, not a stronger one.

- **Adjacent tension found, not cited by the doc:** the one committed HeXO
  number that *does* vary radius (`docs/07_PHASE4_SPRINT_LOG.md:1250`, the
  v8/v7full argmax cross-encoding eval) shows SealBot WR **rising** with
  radius (v7full: r=5 6.5% → r=8 12.5% → r=10 15%; B1 arm r=8/10/12: flat
  0%/0%/0%, different regime). Context doesn't transfer cleanly (that's a
  static-radius argmax-eval comparison at fixed training, not a
  small-then-widening training curriculum), so this is NOT a kill of the
  curriculum hypothesis — but it is the one existing HeXO data point on
  "does smaller radius correlate with better measured outcome," and it
  points the opposite direction of the curriculum's implicit intuition
  ("small is good, early"). Should be named as an open tension, not
  omitted.

**(ii) Radius-shrink vs D-ARGMAX opening-diversity — REVIEW already flagged
this gap; re-tested here, not just re-asserted as missing.** A
small-radius early stage caps the legal-move set near existing stones,
which mechanically shrinks the *combinatorial* opening space available to
Dirichlet/jitter-driven exploration at exactly the stage with the fewest
stones on the board (i.e., the stage where the absolute legal-move count
is smallest to begin with). D-ARGMAX's lesson is about deployment-time
argmax collapse, not training-time self-play (which already carries
Dirichlet noise + temperature + `legal_move_radius_jitter` — mechanisms
D-ARGMAX's own fix target, deployment argmax, lacks). So the tension is
**real in mechanism-class (both are "shrunk move-space → shrunk
diversity") but weaker in context-transfer than the review implied**,
because self-play (unlike deploy) already has three independent
diversity injectors active. Net: worth a cheap canary (distinct-opening
count per S1-stage window, already loggable), not a blocking objection —
but should be named in the spec's canary list alongside draw-rate/colony,
which it currently is not.

**(iii) Does "short decisive games → dense z" transfer given known
draw-collapse and colony attractors?** This is the sharpest attack. The
curriculum's entire mechanism claim is "small radius → short decisive
games → dense z-signal." But L9 (cited by the doc itself) found that a
**different** early-stage-narrowing lever (cosine-temp, which also
functions by narrowing effective move diversity early) drove draws
5%→91% and reopened the §147/§154 colony attractor — mitigated only by
`legal_move_radius_jitter`. A small-radius stage is a **second,
independent narrowing of the same effective search/move space** (radius
narrows the *legal* set; cosine-temp narrows the *sampled* set) stacked at
the exact same lifecycle point (early training). The doc treats jitter as
sufficient insurance because it was sufficient against cosine-temp — but
that's re-using a fix validated against a *different* narrowing mechanism
without checking whether a radius-driven narrowing produces the same
failure mode through a different path (fewer OPTIONS available at all,
vs fewer options SAMPLED). This is exactly the kind of "prior doesn't
automatically transfer" case CLAUDE.md's re-validation protocol exists
for, and the doc's own L9 section doesn't run this specific check — it
only checks "is cosine-temp re-enabled" (no) rather than "does
radius-narrowing independently reproduce cosine-temp's failure mode."
**Verdict: WEAKENED.** Not falsified (radius ≠ cosine-temp mechanistically
— one caps the *set*, one caps the *sample* — and the S178-181 arc note
already flags "new schedule knob, same surface" as a canary trigger) but
the "short decisive games → dense z" transfer claim is asserted, not
demonstrated, and the one lever in this codebase's history that most
resembles it (early-stage move-space narrowing) has a **directly
opposite** track record (91% draws, colony reopened) before jitter fixed
it. The existing canary list (draw-rate + colony-fraction, already in the
spec) is the right insurance — but the doc should say so explicitly
instead of treating the mechanism claim as established.

**B overall: SURVIVES as "worth cheaply testing," WEAKENED as "the big
win."** No internal evidence for the payoff (i), a real but non-fatal
tension with opening-diversity (ii), and a directly-analogous prior
failure mode from a different narrowing lever that the doc doesn't
explicitly clear (iii). Downgrade PORTABLE → TESTABLE-CHEAP in the
summary table, matching Verdict 3's own honesty bar.

### Target: conclusion C — "tiny-net probe worth running"

Checked whether the ACTUAL production net is smaller than the 4.25M this
doc (and S1) uses for the 19.1× ratio: `configs/model.yaml` declares
`res_blocks: 12, filters: 128, in_channels: 8, board_size: 19` (the exact
dims S1 instantiated). Grepped every variant that could plausibly override
architecture — `configs/variants/vast.yaml` (canonical production
variant) and `configs/variants/ws3v3_arm_control.yaml` /
`v6_live2_golong.yaml` (current active research lineage) — **zero
`res_blocks`/`filters`/`model:` overrides in any of them.** The currently
deployed/production net genuinely is 4.25M, not 2.9M; the review's
"likely a smaller historical trunk config, not this doc's number" finding
is correct and **does not change this probe's premise** — 19.1× is the
right ratio for the net that's actually running today. **C's premise
question: SURVIVES unchanged** (no hidden production net rescues or
undercuts the ratio).

Second half of C — is the 4-arm probe actually cheap given F1 (single-
window cascade) confounds all historical iteration-speed reads? This is a
real, under-examined risk the doc doesn't address: F1
(`d-forensic-f1-lineage-single-window-cascade`, memory) found the *whole*
d1m lineage self-played single-window from a metadata bug that
self-perpetuates across resumes — i.e., HeXO's own historical throughput/
learning-curve numbers carry a live contamination risk from checkpoint
metadata inheritance, not just from the width/aux-head axes this probe
controls for. A 2-5K-step, 4-arm short probe launched from whatever
warm-start/bootstrap anchor is convenient risks *inheriting* that same
metadata-cascade class of defect (wrong encoding/window silently carried
forward) if it isn't launched from a freshly-verified single-window-clean
anchor — which would make arms A-D's throughput comparison internally
consistent (same contamination across all 4) but still not informative
about production reality. **This is a real, cheap-to-close precondition
the probe spec should state explicitly** ("verify anchor checkpoint
encoding via `hexo_rl.encoding.lookup` + F1's audit method before
launching any arm") and currently doesn't. Doesn't kill the probe — it's
still cheap relative to a GPU-week sustained run — but "actually cheap"
should be conditioned on that one-line precondition, not assumed.

**C overall: SURVIVES**, with one added precondition (anchor
single-window-clean verification before launch) the doc should state.

### Target: conclusion D — "kernel REJECT" (S3)

No dedicated S3 doc exists in this repo (checked: no
`docs/handoffs/d_strix_s3*`, no "kernel" hits in any of the three D-STRIX
docs, no S3 file in `docs/`) — the REJECT reasoning under attack here is
the dispatcher-brief's own summary ("their kernel solves GNN variable-size
batching; our dense CNN has cuDNN/flash paths"), not a written HeXO
artifact. Sanity-checked the one hole named in the task: does K-cluster
multi-window batching create variable-shape batches on OUR side, the way
ragged edge lists do on theirs?

Read `hexo_rl/selfplay/inference.py` (`infer_batch`,
`infer_batch_per_cluster`): each board's K-cluster windowing produces a
*list* of fixed-shape `(C, board_size, board_size)` tensors — one per
cluster — and `infer_batch` does `torch.cat(all_tensors, dim=0)` across
**all clusters from all boards** before a single forward pass. K (cluster
count per board) varies board-to-board, so the **total batch dimension**
varies run-to-run — but every individual tensor in that batch is the
*same shape* (fixed window geometry from the encoding registry). This is
architecturally just a variable **batch size** (N varies), which is the
single most standard case cuDNN/flash paths are built for (any
convolution kernel handles arbitrary N along the batch dim with zero
padding/scatter machinery) — categorically different from a GNN's ragged
per-graph node/edge count, which varies the **tensor shape within a single
graph's forward pass** and genuinely needs `edge_budget`-style dynamic
batching or padding+masking to batch at all. **No hole found: the K-cluster
axis that superficially resembles "variable-size batching" does not
actually stress the same mechanism hexo-strix's kernel exists for.**

**D: SURVIVES.** The REJECT reasoning holds under a direct code check of
the one plausible counter-example (K-cluster multi-window batching);
variable cluster-*count* is not variable tensor-*shape*, and dense
CNN/cuDNN handles the former natively with no analog to GNN edge-budget
batching needed.
