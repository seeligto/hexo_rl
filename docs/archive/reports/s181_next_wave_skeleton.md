# §S181 Next-Wave Operations Skeleton

**Source:** §S181 structural-diagnosis aggregation
(`audit/structural/00_aggregation.md`). **Date:** 2026-05-22.

Each operation below = a single Claude Code prompt to draft later. Ranked
by leverage. Dependencies noted. All op prompts MUST open with the
standard pre-flight: *Read CLAUDE.md + `docs/07_PHASE4_SPRINT_LOG.md`
§S181 + `audit/structural/00_aggregation.md` first.*

---

## Leverage ranking

```
OP-1  value-spread ladder probe       leverage HIGHEST   cost ~30–60min   dep: none
OP-3  probe + dashboard wave (PR-A)    leverage HIGH      cost ~2 hr       dep: none (parallel)
OP-2  value-head re-arch A/B (A2+A3)   leverage HIGH      cost ~3–5 days   dep: OP-1
OP-4  PSW / refresh-hook               leverage MED       cost ~12–17 hr   dep: OP-2 verdict
```

OP-1 and OP-3 are independent — launch both first, in parallel. OP-2 is
gated on OP-1 (the ladder probe pins the canary target). OP-4 is the
fallback, gated on OP-2's verdict — deploy ONLY if the architecture A/B
shows the loop installs the bias regardless of architecture.

---

## OP-1 — Value-spread canary + §S180b checkpoint-ladder probe

**Leverage:** HIGHEST. Pins WHEN the value head flattens — the single
fact that tells every downstream op where to act. Cheap, zero-training.

**Scope.**
- Extend `scripts/structural_diagnosis/mcts_colony_probe.py` (value-spread
  mode) or `probe_value_bias.py` with a ladder loop.
- Probe `archive/s180b_3knob_fail/ckpts/ckpt_step{10,20,30,40,50}k.pt`
  for `value_spread = mean V(colony bank) − mean V(extension bank)` over
  the 40-position canonical bank (20 colony + 20 extension, already in
  the T3 probe).
- Plot spread vs training step. Cross-plot against the §S180b eval
  trajectory (wr_sealbot 11→7→12→19→0, colony_a 36→35→40→43→59).
- Anchor reference: spread +0.617 (healthy). §S180b step-50k: −0.016.

**Deliverable.** Report pinning the flatten step + a plot. Verdict:
monotonic degradation from step 10K vs late cliff.

**Cost.** Wall-clock ~30–60 min. Dev-hours ~2. Compute ~0 (CPU fine).

**Dependency.** None. Launch first.

**Decision output.** Monotonic → OP-2 canary must hold from step 0,
OP-4 (if used) must act before step 20K. Late cliff → target the cliff
step.

---

## OP-2 — Value-head re-architecture A/B (T2 A2 + A3)

**Leverage:** HIGH. Directly attacks H6/H7 — removes the coverage-blind
`v_max` route and adds a direct anti-colony value gradient.

**Scope.**
- **A2** — replace the `v_max` global-max-pool half with multi-scale
  `v_avg` (global mean + 2×2-block mean pooled). `network.py` +
  `network_min_max_head.py`, ~40 LOC. Changes `value_fc1` input dim
  (`2C → 3C`) → state-dict shape break → **fresh bootstrap re-pretrain
  mandatory**.
- **A3** — add a colony-penalty value-head auxiliary loss: penalize
  value > threshold on positions flagged colony by the existing colony
  detector (Q11 RESOLVED on detection). `losses.py` + trainer wiring,
  ~60 LOC. No shape break.
- Wire the OP-1 `value_spread` metric as a first-class dashboard canary
  + hard-abort gate (abort if spread < +0.20).
- A/B: A2+A3 arm vs a stock-architecture control re-run from the same
  anchor pretrain recipe.
- `make bench` gate before any commit (A2 touches `network.py` forward).

**Cost.** Wall-clock ~3–5 days (re-pretrain ~0.5 day + sustained run
~1 day/arm + bench). Dev-hours ~12–16. Compute ~3–4 GPU-days vast 5080.

**Dependency.** OP-1 (canary target). Do NOT launch before OP-1.

**Decision output.** Spread holds > +0.20 AND wr_sealbot does not
collapse → architecture was the load-bearing permissive element; adopt
A2+A3, close the structural line. Spread still collapses → escalate to
OP-4.

**Constraints.** Do NOT re-propose PMA/gpool-bias (§170 FALSIFIED), no
cosine (L9), no config-knob anti-colony lever (L38). A2 is the "mean"
arm of Q2 applied to the within-window spatial reduction.

---

## OP-3 — Probe + dashboard implementation wave (T4)

**Leverage:** HIGH. Closes the L2/L42 blindness. PR-A alone would have
fired 20–40K steps before the §S180b crash. Instrumentation — not a
fix, but unblocks every future diagnosis.

**Scope (land order, lowest-LOC highest-leverage first).**
- **PR-A** — `colony_a` / `colony_a_frac` first-class eval metric +
  dashboard panel + `check_colony_a` alert + 2 `MonitoringConfig`
  fields. ~40 LOC, ~2 hr. Data already in the
  `evaluation_round_complete` payload (`colony_wins_bootstrap_anchor`).
  **Land first, alone, no dependencies.**
- **PR-B** — L34 anchor↑/sealbot↓ divergence metric + `check_l34_divergence`
  alert (3-eval rolling window). ~50 LOC. Add ALERT-6 `check_wr_best_regression`
  alongside (~20 LOC — catches the §S180a not-learning signature).
- **PR-D** — training-side value-head colony-bias probe (clone
  `ValueProbe` with a colony-vs-extension fixture) + `check_value_colony_bias`
  alert. ~90 LOC + fixture. Earliest leading indicator — runs every K
  training steps, no eval round needed. Shares the OP-1 / OP-2 canary.
- **PR-C** — per-opponent colony-fraction matrix + game-length split.
  ~120 LOC, pure surfacing.
- **PR-E** — MCTS-in-loop probes P1–P4 + `probe_complete` event +
  ALERT-4/5. ~400 LOC + 4 fixtures. Land P4 (anti-colony) first.
  First validate thresholds via `scripts/structural_diagnosis/new_probes.py
  --probe p4` against the 5 §S180b archived checkpoints — confirm
  `colony_pull` crosses 0.20 at step 10K.

**Cost.** PR-A ~2 hr. Full wave ~32 LOC-hours scoped in T4 §7
(~5 dev-days total). Compute ~0 for PR-A/B/C; PR-D/E need short
fixture-eval GPU passes.

**Dependency.** None for PR-A. Launch in parallel with OP-1. PR-D
shares the value-spread fixture with OP-1/OP-2 — sequence after OP-1
defines the fixture.

**Constraint.** Do NOT remove C1–C4 — keep as decode/sharpness sanity
checks; stop treating them as a sufficient pre-promotion gate.

---

## OP-4 — PSW / bot-corpus refresh-hook (FALLBACK)

**Leverage:** MED. Demoted by §S181 — both levers reshape the *buffer*;
they only help if the reshaped buffer retrains value-head discrimination.
Deploy ONLY if OP-2 shows the architecture A/B does not hold the
value-spread canary (i.e. the loop installs the bias regardless of
architecture).

**Scope.**
- **PSW** — Policy Surprise Weighting per
  `docs/designs/s179b_policy_surprise_weighting.md`. Floor(0.5+KL)
  upsample on policy-shift buffer rows. ~395 LOC, ~12.5 dev-hr.
- **Refresh hook** — activate the inert bot-corpus refresh hook at
  `step_coordinator.py:651-666` per `docs/designs/s179c_bot_refresh_hook.md`.
  Regenerate the bot corpus against the PROMOTED model. ~100–200 LOC,
  ~4–6 dev-hr.

**Cost.** Wall-clock per lever ~1–2 days impl + ~1 day sustained run.
Compute ~1–2 GPU-days/arm.

**Dependency.** OP-2 verdict. Do NOT launch before OP-2 returns
"architecture insufficient".

**Success criterion.** The value-spread canary (L42), NOT loss /
value-acc — loss and value-acc improved through every §S178-line crash
(Goodhart). A PSW/refresh arm PASSES only if value-spread holds
> +0.20 AND wr_sealbot does not collapse.

---

## Cross-op constraints

- Anchor = `bootstrap_model_v6.pt` (SHA `7ab77d2c…`). Never
  `bootstrap_model.pt` (quarantined).
- §S178 line encoding = `v6` (k_max=1). K-cluster min-pool does NOT
  fire — do not analyze it as the §S178 capture channel.
- Cosine annealing permanently OFF (L9). No config-knob anti-colony
  lever (L38). No c_puct/Dirichlet retune (L41).
- Any op touching `engine/src/mcts/**`, `network.py` forward, or a
  training hot path → `make bench` gate before commit.
- §175 / §S179 / §S180a / §S180b all FAILED — do not cite as a working
  baseline.
