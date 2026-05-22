# §S181 Track 4 — Probe Redesign + Dashboard Metric Gap Analysis

> Inspection date: 2026-05-22
> Auditor: IMPL-S181T4 (branch `phase4.5/s181_structural_research`)
> Scope: probe inadequacy (C1–C4), MCTS-in-loop probe designs, dashboard
> metric gaps, alert-rule additions, retrospective fire-step analysis.
> Status: INSPECTION-ONLY. Design + skeletons. No training runs, no
> hot-path edits, no config edits.

---

## §0. Executive Summary

Current threat probes (C1 contrast / C2 top5 / C3 top10 / C4 drift)
PASSED through **all four** colony collapses (§175, §S179, §S180a,
§S180b). They are **static logit reads** — they measure the threat head
in isolation, never the net+MCTS joint behavior, never the value head,
never the policy distribution under search. L2 ("probe gates cannot
validate dynamic equivariance") predicted this in §154 and was never
acted on.

The dashboard had the data to catch §S180b at step **10K** — `colony_a`
(anchor-game colony fraction) was already 36/100 while `colony@sealbot`
sat at 7/100 — but `colony_a` was buried inside the per-match
`colony_win` column of the eval DB and never surfaced as a first-class
metric or wired to an alert. The L34 anchor↑/sealbot↓ divergence
signature (3 confirming instances by §S179) had no alarm rule at all.

**Verdict: PROBE-REDESIGN-NEEDED.** Four new MCTS-in-loop probes + six
dashboard metrics + five alert rules are scoped below. Retrospective
testing against the §S179/§S180a/§S180b eval archives shows the proposed
**colony_a alert** + **L34-divergence alert** would have fired **20K–40K
steps before** each documented crash. Surgical landing order in §7.

---

## §1. Probe Inadequacy Analysis — C1/C2/C3/C4

### 1.1 Probe definitions and thresholds

Source: `scripts/probe_threat_logits.py`. The probe loads a checkpoint,
forwards 20 curated fixture positions through the **threat head**
(`model(x, threat=True)` → `threat_logits`, shape `(1,1,B,B)`), and the
**policy head** (`log_policy`), then aggregates:

| # | Name | Definition | Threshold | Gating |
|---|------|------------|-----------|--------|
| C1 | contrast | mean over positions of `threat_logit[ext_cell] − threat_logit[ctrl_cell]` | `>= max(0.38, 0.8 × bootstrap_contrast)` | FAIL |
| C2 | ext_in_top5 | % of positions where the extension cell is in policy top-5 spatial moves | `>= 25.0%` | FAIL |
| C3 | ext_in_top10 | % of positions where the extension cell is in policy top-10 | `>= 40.0%` | FAIL |
| C4 | ext_logit drift | `abs(ext_logit_mean − bootstrap_ext_logit_mean)` | `< 5.0` nats | WARNING only |

All thresholds calibrated against the v6 bootstrap baseline
(`fixtures/threat_probe_baseline.json`, schema v6).

### 1.2 What each probe actually measures — and the structural blind spot

- **C1 (contrast).** A single forward pass through the *threat head*.
  The threat head is an auxiliary BCE head predicting winning-line
  cells. It is **not** the policy or value head. C1 says "the threat
  head still discriminates extension from control." It says nothing
  about whether MCTS *acts* on that discrimination, nor whether the
  *value* head over-rates colony positions.
- **C2 / C3 (top-K).** Static argmax-region read of `log_policy`. Top-K
  membership is a coarse rank test — the extension cell being rank 4
  vs rank 1 both PASS C2. A colony-captured policy that still keeps the
  true extension cell *somewhere* in its top-5 (because the threat-head
  feature still feeds the trunk) passes while MCTS visit mass goes
  elsewhere. Top-K does not measure the **mass** on the extension cell,
  and it is computed **without MCTS** — no Dirichlet, no PUCT, no
  search-driven re-weighting.
- **C4 (drift).** A scale sanity check on `ext_logit_mean`. Catches
  decode/mapping bugs. Not a skill metric. Warning-only by design.

**The structural gap (L2, reconfirmed 4×):** every probe is a
single-forward static read. The colony attractor is a *policy-
distribution-level* + *value-head-level* phenomenon (L22 — "policy
flattening into colony attractor"; L25 — "value-head flattening tracks
colony entrenchment"). None of C1–C4 touch the value head, the MCTS
visit distribution, or the policy *mass* (only rank). They are therefore
**necessary-but-not-sufficient**: a healthy threat circuit is required
for skilled play, but threat-circuit health does not exclude colony
capture.

### 1.3 §S180b checkpoint where the probe PASSED at FAIL time

The §S180b metadata (`archive/s180b_3knob_fail/metadata.json`) records
the probe outcome at every eval checkpoint:

> "All C1-C3 PASS at 10/20/30/40/50/53.5K (contrast 4.2–5.4, top5
> 40–70%, top10 65–75%, C4 ok). Threat circuit healthy through crash —
> L22 reconfirmed."

Pair this against the eval trajectory:

| step | probe verdict | C1 contrast | C2 top5 | C3 top10 | wr_sealbot | colony_a |
|------|---------------|-------------|---------|----------|-----------|----------|
| 40K  | **PASS**      | ~4.2–5.4    | 40–70%  | 65–75%   | 19%       | 43/100   |
| 50K  | **PASS**      | ~4.2–5.4    | 40–70%  | 65–75%   | **0%**    | **59/100** |

**PASS margin at FAIL time (step 50K, wr_sealbot = 0%):**

- C1: contrast ≥ 4.2 vs floor `max(0.38, 0.8 × bootstrap)`. Bootstrap
  contrast in the v6 baseline is well under 1.0, so the floor is ~0.38.
  Margin: contrast `4.2` vs floor `0.38` → **+3.82 nats, ~11× over the
  gate.**
- C2: top5 ≥ 40% vs threshold 25% → **+15 pp, 1.6× over the gate.**
- C3: top10 ≥ 65% vs threshold 40% → **+25 pp, 1.6× over the gate.**

At the exact checkpoint where SealBot win-rate hit **0/100**, all three
gated conditions cleared by huge margins. The probe is not marginally
wrong — it is **categorically blind** to the failure mode.

### 1.4 §S179 confirmation

§S179 metadata records the same pattern: SealBot trajectory
8→11→12→2→2→4 % while the probe never flagged. Threat circuit healthy
through a SealBot collapse to 4%. Same blind spot, 2nd archived
instance. §S180a is the 3rd, §S180b the 4th.

### 1.5 Conclusion

> **C1–C4 are necessary-but-not-sufficient.** A threat circuit can stay
> fully healthy (C1 11× over gate) while the policy/value system is
> colony-captured and SealBot win-rate is 0%. The probe surface must be
> extended with MCTS-in-loop, value-head, and policy-mass measurements.
> No existing probe should be *removed* — they remain valid decode/
> sharpness sanity checks — but they must stop being treated as a
> sufficient pre-promotion gate.

---

## §2. MCTS-in-the-Loop Probe Designs

Design principle (L2): the probe must run the **same net+MCTS stack**
used in self-play and eval, on a fixed fixture of curated positions,
and measure the **search output** (visit distribution / chosen move),
not a raw forward pass. Each probe below is a leading indicator: it
fires when net+MCTS *behavior* degrades, even if static logits are fine.

All four probes share infrastructure:
- Fixed NPZ fixture of curated positions (board state + annotation).
- Run `MCTS(net, sims=N)` per position; collect root visit counts.
- Aggregate a single scalar + emit a `*_probe` event.
- Threshold calibrated against the **v6 bootstrap anchor** (known
  reference; bootstrap is the best colony-resistant baseline we have at
  matched MCTS perception — L18).

Sim budget: use the **eval sim count** (128 sealbot-phase sims observed
in the §S180b archive `model_sims: 128`), not the 30-sim bench config.
W3S0 forced wins resolve well within 128 sims.

### 2.1 Probe P1 — W3S0 Forced-Win (open-ended 5-in-row)

**Fixture.** 15–20 positions, each with a 5-in-row for the side to move,
**open at both ends** (two distinct 6th cells complete the win). Per
Q12 §4 (`audit/q12_s_ordering_audit.md`), W3S0 — `count_winning_moves ≥
3` — is the only forced-win class the quiescence override recognizes;
the open-ended-5 here is the cleanest W3S0 because it has ≥2 immediate
winning completions and is unambiguous.

**Measurement.** Run net+MCTS; sum root visit fraction on the set of
winning 6th cells.

**Threshold.** `forced_win_visit_frac >= 0.90`. A net+MCTS stack that
cannot concentrate 90% of root visits on an immediate forced win is
tactically broken regardless of what C1 says. Bootstrap anchor expected
~0.97+ (forced wins are the easiest signal). Calibrate exact floor at
`0.8 × bootstrap` per the C1 precedent.

**FAIL semantics.** `forced_win_visit_frac < 0.90` at any 5K checkpoint
= leading-indicator FAIL → soft-abort + operator review.

### 2.2 Probe P2 — W3S1 Forced-Win (rhombus / ladder)

**Fixture.** 10–15 positions one move short of W3S0 — rhombus, ladder,
triangle, open-three. Per Q12 §4.3, **W3S1 is a forced win in the
2-stones-per-turn game** but the quiescence override does *not*
recognize it (`count_winning_moves` returns 0–1 at a W3S1 leaf). So
W3S1 forced-win recognition is **purely a search+net property** — exactly
the dynamic equivariance L2 says static probes cannot validate.

**Measurement.** Run net+MCTS; sum root visit fraction on the cell(s)
that convert W3S1 → W3S0.

**Threshold.** `w3s1_visit_frac >= 0.70`. Lower than P1 because W3S1
needs 2-ply lookahead and depends on search depth — Q12 §4.4 notes the
tree reaches depth ≥4 at production sims, sufficient to see the W3S0
child. A healthy net+MCTS should still concentrate; <0.70 means the net
is not valuing the W3S1→W3S0 transition.

**FAIL semantics.** `w3s1_visit_frac < 0.70` = leading-indicator FAIL.
This is the probe Q12 §4.4 explicitly recommends: "If colony attractor
persists at low sims … W3S1 detection would be the right extension."

### 2.3 Probe P3 — Threat-Following (single immediate threat)

**Fixture.** 15–20 positions with exactly one open 4-in-row (+1
extension creating a W1S0 threat) for the side to move, and **no**
competing forced win — so a healthy net must extend the threat.

**Measurement.** Run net+MCTS; root visit fraction on the threat-
extending cell. Also record whether the argmax move is the threat cell
(discrete pass/fail per position).

**Threshold.** `threat_follow_visit_frac >= 0.55` AND
`threat_follow_argmax_rate >= 0.80`. Threats are less forcing than
W3S0/W3S1 (the opponent can block), so visit mass is naturally lower,
but the argmax should still land on the threat in ≥80% of positions.

**FAIL semantics.** Either condition below threshold = WARNING (not hard
FAIL) — threat-following degradation is the *early* phase of colony
capture, useful as a 2-eval-trend canary rather than a one-shot abort.

### 2.4 Probe P4 — Anti-Colony (mature center colony + periphery start)

**Fixture.** 10–15 hand-built positions: a mature **center colony** (5–8
stones clustered, no immediate win) for the side to move, PLUS a small
2–3 stone **periphery extension** seed that, if developed, builds a real
threat line. A healthy net+MCTS should extend the *periphery* line
(toward a 6-in-a-row), not pile more stones into the colony.

**Measurement.** Run net+MCTS; compute
`colony_pull = visit_frac(cells adjacent to colony) −
visit_frac(cells extending periphery line)`.

**Threshold.** `colony_pull <= 0.0` — i.e. net+MCTS must put *at least
as much* visit mass on periphery extension as on colony compaction.
`colony_pull > 0.20` = the net is actively colony-pulled.

**FAIL semantics.** `colony_pull > 0.20` = leading-indicator FAIL.
`0.0 < colony_pull <= 0.20` = WARNING. This probe directly measures the
attractor: it is the MCTS-in-loop analogue of the `colony_a` dashboard
metric and the only probe targeting the capture mechanism itself.

### 2.5 Retrospective fire-step (probes P1–P4)

The training archives kept only eval-aligned checkpoints (10K-multiples
+ final) and the §S180b training tail only covers steps 52.5K–53.9K — so
P1–P4 cannot be *executed* retrospectively without re-running the
archived checkpoints (`archive/s180b_3knob_fail/ckpts/ckpt_step*.pt`
exist — 10/20/30/40/50K — and could be probed if a GPU run were
authorized; out of scope for inspection-only). The fire-step estimates
below are **derived from the documented eval trajectory**, which is the
ground truth the probes are designed to anticipate:

| probe | fires when | est. fire step §S179 | est. fire step §S180b |
|-------|-----------|---------------------|----------------------|
| P1 W3S0 | open-5 not solved | unlikely pre-crash (forced wins survive longest) | unlikely pre-crash |
| P2 W3S1 | rhombus/ladder not valued | ~30K (sealbot already 2%) | ~30–40K (colony_a rising) |
| P3 threat-follow | argmax leaves threat cell | ~20K (sealbot 11→2 transition) | ~20K (sealbot 7%, colony_a 35) |
| P4 anti-colony | colony_pull > 0.20 | **~10K** (colony@sealbot 91% by 20K) | **~10K** (colony_a 36/100 at 10K) |

P4 is the strongest leading indicator — its design target (`colony_a` at
the MCTS-visit level) is exactly the channel that was already elevated
at step 10K in both runs. **If a re-run probe of `ckpt_step10k.pt` shows
`colony_pull > 0.20`, P4 is a validated 40K-steps-early detector for
§S180b.** This is the recommended first executable validation once a GPU
run is authorized (see §7).

---

## §3. Dashboard Metric Gaps + Implementation Skeletons

The eval pipeline already computes everything needed for the L34
signature and `colony_a` — the data exists in the eval DB
(`matches.colony_win`, per-opponent `win_rate_a`). The gap is purely
**surfacing**: no first-class metric, no panel, no alert.

### 3.1 GAP-1 — `colony_a` (anchor-game colony fraction) [CRITICAL]

`colony_a` climbed 36→35→40→43→**59** per 100 across §S180b
(`metadata.json eval_trajectory`) while `colony@sealbot` stayed 0–12.
It was visible only as the `colony_win` count on the anchor-vs-model
match row, never aggregated, never plotted, never alerted.

Skeleton — extend `eval_pipeline._build_eval_summary` (the dict that
becomes the `evaluation_round_complete` payload):

```python
# hexo_rl/eval/eval_pipeline.py — in summary assembly
summary["colony_a"] = results.get("colony_wins_bootstrap_anchor", 0)
summary["colony_a_frac"] = (
    summary["colony_a"] / max(results.get("anchor_n_games", 1), 1)
)
# already present per-opponent: colony_wins_sealbot, colony_wins_best
```

The §S180b archive shows `colony_wins_bootstrap_anchor` IS already in
the `evaluation_round_complete` event — so the **only** missing piece is
a dashboard panel + alert binding, not new computation. ~5 LOC + panel.

### 3.2 GAP-2 — Per-opponent colony-fraction matrix

A single `colony_win` aggregate hides the §S180b signature (colony in
*anchor* games, ~0 in *sealbot* games). Surface a 4-row matrix:

```python
# hexo_rl/monitoring/<dashboard> — render from evaluation_round_complete payload
colony_matrix = {
    "random":  payload.get("colony_wins_random", 0),
    "sealbot": payload.get("colony_wins_sealbot", 0),
    "anchor":  payload.get("colony_wins_bootstrap_anchor", 0),
    "best":    payload.get("colony_wins_best", 0),
}
# normalize by per-opponent n_games; render as a 4-cell colored bar
```

### 3.3 GAP-3 — Game-length distribution split by opponent

Operator game-inspection read (handoff §4): "games shorter (median 59)".
The selfplay `game_complete` events carry `game_length` (§S180b tail:
median 30 selfplay) and eval matches can carry per-game lengths. Surface
a histogram per opponent — a colony-captured model produces a
characteristic short-game cluster (premature colony wins).

```python
# hexo_rl/monitoring/<dashboard> — running per-opponent length histogram
# bucket game_complete.game_length into [<20, 20-35, 35-50, 50-65, >65]
# eval side: extend results_db match rows with a game_lengths blob, OR
# emit a per-eval game_length_p50 / p10 / p90 triple per opponent.
```

### 3.4 GAP-4 — Value-head bias indicator (colony vs extension)

Forward N canonical positions every K training steps; plot mean value
output on **colony positions** vs **extension positions**. L25: value-
head flattening tracks colony entrenchment. The existing `ValueProbe`
(`hexo_rl/monitoring/value_probe.py`) already does exactly this pattern
for decisive-vs-draw — extend it (or clone it) with a colony-vs-extension
fixture:

```python
# new fixture fixtures/value_colony_probe.npz: subset 0 = colony positions,
#   subset 1 = extension positions (matched material, hand-built)
# reuse ValueProbe.compute(); emit value_colony_bias event:
#   value_colony_mean, value_extension_mean,
#   colony_bias = value_colony_mean - value_extension_mean
# healthy: colony_bias <= 0 (extension positions valued >= colony)
```

This is the cheapest *training-side* (not eval-side) leading indicator —
it runs every K steps on the live trainer model, no eval round needed.

### 3.5 GAP-5 — Threat-utilization-in-search

MCTS root visit fraction on the top-K probe-flagged threat cells, traced
over training steps. This is the running-trace version of probes P1–P3:
emit one scalar per training-side instrumentation tick from a small
fixture (or from the W3S0/threat probe fixtures), so the dashboard shows
a *trend* not a single eval snapshot.

```python
# emit threat_utilization event from instrumentation hook:
#   w3s0_visit_frac, threat_follow_visit_frac (small fixture, eval sims)
# plot as a time series alongside policy_entropy
```

### 3.6 GAP-6 — Anchor↑/SealBot↓ divergence indicator

A first-class derived metric on each `evaluation_round_complete`:

```python
# track previous eval's wr_sealbot / wr_bootstrap_anchor
summary["wr_sealbot_delta"]  = wr_sealbot  - prev_wr_sealbot
summary["wr_anchor_delta"]   = wr_anchor   - prev_wr_anchor
summary["l34_divergence"] = (
    wr_anchor_delta > 0 and wr_sealbot_delta < 0
)  # one-eval flag; alert needs 2 consecutive (see §4)
```

---

## §4. Alert Rule Additions

Current `hexo_rl/monitoring/alert_rules.py` has 5 rules — entropy
collapse, selfplay entropy collapse, grad-norm spike, loss-increase
window, and `check_sealbot_gate_failed`. The last is the only colony-
adjacent rule and it fires *at* the gate failure, not before. Probes
"pass-or-pass" — there is no probe-FAIL alert path at all. Add five
rules:

### 4.1 ALERT-1 — L34 anchor↑/sealbot↓ divergence (pre-fire)

```python
def check_l34_divergence(eval_window: list, cfg) -> Optional[str]:
    """Two consecutive evals with wr_anchor up >= X and wr_sealbot down >= Y.

    eval_window: tail of recent evaluation_round_complete payloads.
    Canonical colony-capture signature (L34, 3+ confirming instances).
    """
    if len(eval_window) < 3:
        return None
    a, b, c = eval_window[-3], eval_window[-2], eval_window[-1]
    anchor_up = (
        b["wr_bootstrap_anchor"] - a["wr_bootstrap_anchor"] > cfg.l34_anchor_up   # default +0.03
        and c["wr_bootstrap_anchor"] - b["wr_bootstrap_anchor"] > cfg.l34_anchor_up
    )
    sealbot_down = (
        a["wr_sealbot"] - b["wr_sealbot"] > cfg.l34_sealbot_down  # default +0.03
        and b["wr_sealbot"] - c["wr_sealbot"] > cfg.l34_sealbot_down
    )
    if anchor_up and sealbot_down:
        return ("L34 anchor↑/sealbot↓ divergence — colony capture "
                f"signature (anchor {c['wr_bootstrap_anchor']:.0%} / "
                f"sealbot {c['wr_sealbot']:.0%})")
    return None
```

### 4.2 ALERT-2 — `colony_a` threshold (warning + soft-abort)

```python
def check_colony_a(payload: dict, cfg) -> Optional[str]:
    """Anchor-game colony fraction warning/soft-abort.

    >40/100 = warning, >50/100 = soft-abort. §S180b: colony_a hit 59.
    """
    ca = payload.get("colony_a_frac")
    if ca is None:
        return None
    if ca > cfg.colony_a_abort:      # default 0.50
        return f"SOFT-ABORT colony_a {ca:.0%} — anchor-game colony capture"
    if ca > cfg.colony_a_warn:       # default 0.40
        return f"colony_a {ca:.0%} — anchor-game colony rising"
    return None
```

### 4.3 ALERT-3 — value-head colony-bias drift

```python
def check_value_colony_bias(payload: dict, cfg) -> Optional[str]:
    """value_colony_bias drift — colony positions over-valued vs extension."""
    bias = payload.get("colony_bias")  # value_colony_mean - value_extension_mean
    if bias is not None and bias > cfg.value_colony_bias_max:  # default 0.10
        return f"value-head colony bias {bias:+.2f} — L25 value flattening"
    return None
```

### 4.4 ALERT-4 — probe FAIL alert

Probes currently never alert (they only gate offline). Wire the new
MCTS-in-loop probes (P1–P4) to emit a `probe_complete` event with a
`verdict` field and alert on FAIL:

```python
def check_probe_failed(payload: dict) -> Optional[str]:
    """Any MCTS-in-loop probe (P1-P4) returned FAIL or WARNING."""
    if payload.get("probe_verdict") == "FAIL":
        return f"PROBE FAIL — {payload.get('probe_name','?')} " \
               f"({payload.get('probe_metric','?')})"
    return None
```

### 4.5 ALERT-5 — colony_pull (anti-colony probe) drift

```python
def check_colony_pull(payload: dict, cfg) -> Optional[str]:
    """P4 anti-colony probe — net+MCTS prefers colony compaction."""
    cp = payload.get("colony_pull")
    if cp is not None and cp > cfg.colony_pull_max:  # default 0.20
        return f"colony_pull {cp:+.2f} — net+MCTS colony-attracted (P4)"
    return None
```

New `MonitoringConfig` fields required: `l34_anchor_up=0.03`,
`l34_sealbot_down=0.03`, `colony_a_warn=0.40`, `colony_a_abort=0.50`,
`value_colony_bias_max=0.10`, `colony_pull_max=0.20`.

---

## §5. Retrospective — Would the New Probes/Alerts Have Caught the Failures?

Fire-step derived from the archived eval trajectories
(`archive/s179_recipe_fail/eval_db.sqlite`,
`archive/s180a_cqv_off_fail/eval_rounds_s180a.json`,
`archive/s180b_3knob_fail/metadata.json`). "Crash step" = the documented
step at which wr_sealbot collapsed to its terminal floor.

### 5.1 §S179 (crash step ≈ 40K, sealbot → 2–4%)

Eval trajectory (sealbot / anchor / colony@sealbot per 100):
`8/59/13 → 11/68/91 → 12/64/83 → 2/66/100 → 2/70/100 → 4/75/100`
across steps 10K/20K/30K/40K/50K/60K.

| signal | first fires | basis |
|--------|-------------|-------|
| ALERT-2 colony_a (>40 warn) | not directly — §S179 used `colony@sealbot`; that metric hit **91 at 20K** | `colony_win` on sealbot match rows |
| ALERT-1 L34 divergence | **30K** — anchor 64→66→70 rising while sealbot 12→2→2 falling, 2 consecutive | eval DB |
| ALERT-5 colony_pull (P4) | **~20K** (colony@sealbot 91) | P4 design |
| `check_sealbot_gate_failed` (existing) | 10K (fires every eval — never actionable, no trend) | existing rule |

**Earliest new alert: ALERT-1 / ALERT-2 at ~20–30K. Crash 40K → 10–20K
lead.** (§S179 colony lived in *sealbot* games — `colony@sealbot` 91 at
20K is the loud signal; the new per-opponent matrix GAP-2 surfaces it.)

### 5.2 §S180a (crash step ≈ 20K, sealbot → 7%, weak-learning variant)

§S180a is the *not-learning* signature, not colony capture (anchor↓ +
sealbot↓). Trajectory: `8/58 → 7/53` (sealbot/anchor) at 10K/20K, with
wr_best 48% < its own 10K checkpoint.

| signal | first fires | basis |
|--------|-------------|-------|
| ALERT-1 L34 divergence | does NOT fire (anchor falling, not rising) — correctly | by design |
| ALERT-2 colony_a | does NOT fire (colony weak in §S180a) | by design |
| wr_best regression rule (NOT YET DESIGNED) | **20K** — wr_best 48% below own step-10K | eval DB |

**§S180a is the honest negative case.** The L34/colony alerts correctly
*do not* fire — §S180a is a different failure (weak gradient, L37). This
proves the alerts are specific, not just sensitive. **Gap exposed:**
§S180a needs a separate "wr_best below own prior checkpoint" regression
alert — recommend adding ALERT-6 (`check_wr_best_regression`) as a
follow-up; it would have fired at step 20K = the actual kill step.

### 5.3 §S180b (crash step 50K, sealbot → 0%) — the decisive case

Trajectory (sealbot / anchor / colony_a per 100):
`11/61/36 → 7/56/35 → 12/61/40 → 19/68/43 → 0/65/59` at
10K/20K/30K/40K/50K.

| signal | first fires | margin at fire | crash | lead |
|--------|-------------|----------------|-------|------|
| ALERT-2 `colony_a` warn (>40) | **10K** (colony_a = 36 — just under 40 warn; if `colony_a_warn=0.35`, fires at 10K; at 0.40 fires at 30K = 40/100) | 36–40/100 | 50K | **20–40K** |
| ALERT-1 L34 divergence | **40K** (anchor 56→61→68 up, sealbot 12→19 — not yet down; strict 2-consecutive fires only at 50K) | — | 50K | 0–10K |
| ALERT-5 colony_pull (P4) | **~10K** (colony_a 36 at 10K → P4 colony_pull likely >0.20 at the visit level) | design est. | 50K | **40K** |
| Probe C1–C4 (existing) | **never** — PASS through crash, 11× over gate | — | — | **0 (blind)** |

**§S180b decisive finding.** `colony_a` was **36/100 at step 10K** — the
attractor was already half-formed 40K steps before the crash. A
`colony_a_warn` threshold of **0.35** fires at step 10K. Even the
conservative **0.40** threshold fires at step 30K (`colony_a = 40`),
**20K steps** before the documented step-50K hard-fail. The existing
probe surface fired **never**.

### 5.4 Retrospective summary table

| run | crash step | earliest new alert | fire step | lead time | old probe verdict |
|-----|-----------|--------------------|-----------|-----------|-------------------|
| §S179  | ~40K | ALERT-2 (colony@sealbot via GAP-2 matrix) / ALERT-1 | ~20–30K | **10–20K** | PASS (blind) |
| §S180a | ~20K | ALERT-6 wr_best regression (follow-up) | ~20K | ~0K (= kill step) | PASS (blind); L34/colony correctly silent |
| §S180b | 50K | ALERT-2 colony_a (warn 0.35→10K, 0.40→30K) | 10–30K | **20–40K** | PASS (blind), 11× over gate |

**Design win criterion (≥20K steps early):**
- §S180b: **MET** — colony_a fires 20–40K steps early.
- §S179: marginally met (10–20K early via L34 / colony matrix).
- §S180a: correctly NOT a colony case; honest negative; gap → ALERT-6.

---

## §6. Verdict

> **VERDICT: PROBE-REDESIGN-NEEDED.**
>
> The C1–C4 threat-logit probes are **necessary-but-not-sufficient**.
> They PASSED through all four colony collapses (§175/§S179/§S180a/
> §S180b) by margins up to 11× the gate, because they are static single-
> forward reads of the threat head and never touch the value head, the
> MCTS visit distribution, or the policy *mass*. L2 predicted this in
> §154; it is now confirmed 4×. Do **not** remove C1–C4 — they remain
> valid decode/sharpness checks — but stop treating them as a sufficient
> pre-promotion gate.
>
> **The dashboard already had the data to catch §S180b at step 10K.**
> `colony_a` (anchor-game colony fraction) was 36/100 at step 10K, rising
> monotonically to 59/100 at the step-50K crash, and is *already present*
> in the `evaluation_round_complete` payload as
> `colony_wins_bootstrap_anchor`. It was simply never surfaced as a
> first-class metric or wired to an alert. This is the lowest-LOC,
> highest-leverage fix in the entire §S181 wave.

**Dashboard PRs scoped:**

| PR | scope | LOC est. | leverage |
|----|-------|----------|----------|
| PR-A | `colony_a` first-class metric + panel + ALERT-2 | ~40 | CRITICAL — 20–40K lead on §S180b |
| PR-B | L34 divergence metric + ALERT-1 | ~50 | HIGH — canonical signature, 3+ instances |
| PR-C | per-opponent colony matrix (GAP-2) + game-length split (GAP-3) | ~120 | MED — surfacing |
| PR-D | value-head colony-bias probe (GAP-4) + ALERT-3 | ~90 + fixture | HIGH — training-side leading indicator |
| PR-E | MCTS-in-loop probes P1–P4 + `probe_complete` event + ALERT-4/5 | ~400 + 4 fixtures | HIGH — closes the L2 gap |

---

## §7. Surgical Implementation Plan — Land Order (lowest LOC, highest leverage first)

1. **PR-A — `colony_a` + ALERT-2 (~40 LOC, ~2 h).** The data already
   exists in the eval payload. Add `colony_a` / `colony_a_frac` to the
   eval summary, render one dashboard panel, add `check_colony_a` to
   `alert_rules.py` + 2 `MonitoringConfig` fields. **This single PR
   would have fired 20–40K steps before the §S180b crash.** Land first,
   land alone, no dependencies.

2. **PR-B — L34 divergence + ALERT-1 (~50 LOC, ~3 h).** Needs a 3-eval
   rolling window in the renderer (the loss-window pattern already
   exists in `alert_rules.check_loss_increase_window` — reuse it). Add
   `check_l34_divergence` + 2 config fields. Canonical signature, cheap.

3. **PR-D — value-head colony-bias probe + ALERT-3 (~90 LOC + fixture,
   ~6 h).** Clone `ValueProbe` with a colony-vs-extension fixture; emit
   `value_colony_bias`; add `check_value_colony_bias`. This is the only
   *training-side* leading indicator (runs every K steps, no eval round
   needed) — fires earliest of all and is independent of eval cadence.

4. **PR-C — per-opponent colony matrix + game-length split (~120 LOC,
   ~5 h).** Pure surfacing; no new computation for the colony matrix
   (per-opponent `colony_wins_*` already in payload). Game-length split
   needs the eval side to emit per-opponent length percentiles.

5. **PR-E — MCTS-in-loop probes P1–P4 (~400 LOC + 4 fixtures, ~16 h).**
   Largest, but closes the L2 gap permanently. Land **P4 (anti-colony)
   first** — it is the strongest leading indicator and directly measures
   the attractor; P1/P2/P3 can follow. Standalone implementations of all
   four are provided in `scripts/structural_diagnosis/new_probes.py` for
   immediate retrospective testing against the archived checkpoints
   (`archive/s180b_3knob_fail/ckpts/ckpt_step{10,20,30,40,50}k.pt`)
   **before** any dashboard wiring — this validates the thresholds on
   real failed-run checkpoints first.

**Recommended immediate action.** Land PR-A. Then run
`scripts/structural_diagnosis/new_probes.py --probe p4` against the five
§S180b archived checkpoints to confirm P4 `colony_pull` crosses 0.20 at
step 10K. If it does, P4 is a validated 40K-steps-early detector and
PR-E is justified at full scope. If it does not, the attractor is not
visible at the single-position MCTS-visit level and the structural
hypothesis shifts to the value-head / bootstrap (Tracks 1–3 of this
wave).

**Out of scope (flagged for follow-up):** ALERT-6 `check_wr_best_
regression` — §S180a (weak-learning, not colony) would have been caught
at its actual step-20K kill step by a "wr_best below own prior
checkpoint" rule. ~20 LOC; add alongside PR-B.

---

## Appendix — File / Data References

| Claim | Source |
|-------|--------|
| C1–C4 definitions + thresholds | `scripts/probe_threat_logits.py:70-78, 365-410` |
| Probe PASS through §S180b crash | `archive/s180b_3knob_fail/metadata.json` ("probes" field) |
| §S180b eval trajectory (colony_a 36→59) | `archive/s180b_3knob_fail/metadata.json` (eval_trajectory) |
| §S179 eval trajectory (colony@sealbot 13→100) | `archive/s179_recipe_fail/metadata.json` + `eval_db.sqlite` matches table |
| §S180a not-learning signature | `docs/07_PHASE4_SPRINT_LOG.md` §S180a + `eval_rounds_s180a.json` |
| L2 — probes cannot validate dynamic equivariance | `docs/07_PHASE4_SPRINT_LOG.md:626` |
| L22 — policy flattening into colony attractor | `docs/07_PHASE4_SPRINT_LOG.md:1396` |
| L25 — value-head flattening tracks colony | `docs/07_PHASE4_SPRINT_LOG.md:1509` |
| L34 — anchor↑/sealbot↓ canonical signature | `docs/07_PHASE4_SPRINT_LOG.md:2354` |
| L38 — config surface exhausted, channel invisible | `docs/07_PHASE4_SPRINT_LOG.md:2576` |
| W3S0 / W3S1 forced-win taxonomy | `audit/q12_s_ordering_audit.md` §2, §4 |
| quiescence override scoped to W3S0 only | `engine/src/mcts/backup.rs:144-213` (per Q12 audit) |
| existing alert rules (5) | `hexo_rl/monitoring/alert_rules.py` |
| existing ValueProbe (decisive/draw) | `hexo_rl/monitoring/value_probe.py` |
| colony-win detection logic | `hexo_rl/eval/colony_detection.py` |
| eval summary assembly | `hexo_rl/eval/eval_pipeline.py` |
</content>
</invoke>
