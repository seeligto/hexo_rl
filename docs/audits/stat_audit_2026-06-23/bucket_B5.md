# Bucket B5 — monitor stats audit (D-STATAUDIT Phase 1)

Source pin `52067631` (worktree). Empirical claims from §4 banked logs ONLY.
Key banked fact (train_cdf… step 500, the ONLY value_spread event in the sample):
`t3_spread=-0.5249 (mean_colony=-0.081 < mean_ext=+0.4439, n_col=n_ext=20)`,
`alt_spread=NaN` with `alt_components.n=0`, plus a sibling log line
`value_spread_alt_skipped_plane_mismatch {alt_planes:8, model_in_channels:4}`.
Entropy in sample: combined `policy_entropy` 2.14–2.89 nats; selfplay 2.09–3.59 nats.
grad_norm max 2.88; colony_extension_fraction 0.0–0.026.

---

## t3_spread  (value_spread_canary.py:469 emit / :324 compute / :291 dataclass)

- **A formula**: `spread = mean V(colony) − mean V(extension)` (l.193-202). Math sound; round to 4dp; n=40 fixed bank. OK.
- **B eff-n**: 40 fixed positions, deterministic forward, no aggregation over games. Not a comparison stat needing CI. OK.
- **C planner-semantics**: value-head probe, NOT a PUCT/visit quantity — Gumbel-agnostic. OK under Gumbel-SH.
- **D band calibration**: anchor +0.617 measured on *synthetic T3 builder* positions for an 8-plane v6 model; gates WARN 0.30 / SOFT-ABORT 0.20 are anchor-derived. But banked run has `model_in_channels=4` (l. plane_mismatch), a DIFFERENT encoding regime than the 8-plane anchor — and T3=-0.5249 is INVERTED (colony > ext expected; here colony < ext). Band borrowed from a regime that does not match the audited run. BIASED.
- **E construct validity**: the code's OWN docstring (alert_rules.py:107-108, config.py:42-45 / L50) records that this canary FAILED to track eval: alt held +0.18-0.30 across 46k steps while wr_sealbot collapsed 33%→5%; canary demoted to INFORMATIONAL. Documented Goodhart failure — the named "colony-capture" signal does not predict play-quality regression. BIASED (signal real but demoted/unreliable).
- **F redundancy**: `spread` (l.305 back-compat) is the SAME number re-emitted under a second key in the same payload. t3_spread is canonical; `spread` is the duplicate (see `spread` row).
- **Verdict: BIASED** (D borrowed/mismatched band + E demoted construct). **Action: fix** — re-anchor gates to the audited encoding regime (4-plane) and label INFORMATIONAL in the surface, OR drop gates and keep as logged diagnostic only. Keep the number, fix the band.

## alt_spread  (value_spread_canary.py:469 / :385,393 — SEED 7)

- **A formula**: when applicable, same `mean(col)−mean(ext)` (l.247-279). BUT in the banked run it is the **skip sentinel** `float("nan")` (l.385) emitted when alt plane count (8) ≠ model in_channels (4) and `encoding_spec is None` (re-encode branch l.362 requires a non-None spec, which `fire_canary` only supplies when `encoding` is passed; banked run passed None → re-encode never attempted → fell to skip). `alt_components.n=0` proves no value was computed. NaN here means "NOT MEASURED", conflated on the wire with a finite degraded measurement. Axis-A fail: a sentinel masquerading as a metric value with no companion "measured?" flag in the payload. WRONG.
- **B eff-n**: n=0 in the banked sample; never aggregates a real distribution this run.
- **C**: value-head probe, Gumbel-agnostic. N/A.
- **D band**: ALT_WARN 0.10 / ALT_SOFT_ABORT 0.07 gates are DEAD when alt is skipped — the alt branch of `check_value_spread_canary` (alert_rules.py:352,357) never fires on NaN (`_bad` guards it). Band uncalibrated-for-this-run because never evaluated.
- **E**: same L50 demotion as t3_spread.
- **F**: redundant-by-design with t3 when present (PR-C bills it as an independent 3×-amplified bank), but the dominating failure is A.
- **Verdict: WRONG** (Axis A — NaN skip-sentinel emitted as a metric value; plane-mismatch path is the live behavior, not the documented "dual-bank" path). **Action: fix** — emit an explicit `alt_applicable: bool` / `alt_status` field and a numeric NaN-guard so dashboards/gates distinguish "skipped" from "degraded"; re-encode path should be the default (wire `encoding` through `fire_canary` on the real run). Stat is salvageable, not Gumbel-meaningless → fix, not drop.
- **SEED 7 RESULT: CONFIRMED-but-INVERTED-cause.** alt_spread DOES emit NaN this run (handoff claim "does NOT NaN via re-encode path" is FALSE for the banked run). Cause is NOT numerical instability and NOT the re-encode path succeeding — it is the plane-mismatch SKIP (in_channels=4 vs 8-plane fixture, encoding=None so re-encode skipped). Marking INVERTED: documented mechanism (re-encode rescues it) did not occur.

## both_pass  (value_spread_canary.py:469 / :352,376,390)

- **A formula**: `t3 ≥ 0.20 AND alt ≥ 0.07` (l.352-353); when alt skipped, degrades to `t3 ≥ 0.20` only (l.390). Logic correct for inputs, but inherits alt's skip → "both" is a misnomer when only one bank ran. In banked run `both_pass=false` driven by t3=-0.52, alt absent. Minor A concern (label vs behavior).
- **B**: boolean over the 40-pos banks, no game aggregation. OK.
- **C**: Gumbel-agnostic. OK.
- **D**: no own band (it IS the gate). Inherits t3/alt band issues.
- **E**: same L50 demotion — the PASS verdict failed to track eval regression. BIASED (construct).
- **F**: derivable from t3_spread + alt_spread + the four thresholds → zero marginal signal beyond the two spreads. Redundant, but E dominates.
- **Verdict: BIASED** (E — demoted gate that mislabels "both" when alt is skipped). **Action: fix** — rename/flag to reflect single-bank degradation and mark INFORMATIONAL per L50.

## spread  (value_spread_canary.py:307 — back-compat alias)

- **A**: identical value to t3_spread (to_payload l.305 sets `spread=self.t3_spread`). Correct math, but pure duplicate emit.
- **F redundancy**: exact byte-duplicate of t3_spread under a second key, same payload. Canonical = **t3_spread**.
- **Verdict: REDUNDANT.** canonical=`t3_spread`. **Action: drop** the alias once pre-PR-C single-bank renderers/tests migrate (it exists only for back-compat).

## mean_colony  (value_spread_canary.py:307)

- **A**: `mean V(colony bank)` (l.191), component of spread. Correct. Banked: -0.081.
- **B/C**: fixed 20-pos mean, Gumbel-agnostic. OK.
- **D**: no own band (`has_band=false` in inventory). OK.
- **E**: diagnostic component; only meaningful as part of spread (same L50 caveat applies to the parent). Standalone it is an honest mean.
- **F**: not redundant — spread does not expose the colony level alone; needed to read inversion (here colony<ext). Keep.
- **Verdict: CORRECT** (no band, honest component mean). **Action: keep.**

## mean_ext  (value_spread_canary.py:307)

- Same analysis as mean_colony; banked +0.4439. Honest component mean, no band, not recoverable from spread alone.
- **Verdict: CORRECT.** **Action: keep.**

## wr_sealbot_sliding_window_alert  (alert_rules.py:101 `check_sealbot_wr_hard_abort`)

- **A formula**: three triggers (rolling-mean, collapse-from-peak, early-death), all over caller-owned `wr_history` ring. `peak_wr = max(...)`, consecutive-N guards. Math sound (l.135-196).
- **B eff-n** (PREREG seed5 lens): wr values are win-rates over eval games. This rule consumes pre-aggregated wr scalars with NO CI and NO distinct-game (n_eff) accounting. §D-ARGMAX warns argmax/temp-0 deterministic eval collapses to ~2 distinct games/pair → a wr point is far noisier than its game count implies. The abort fires on raw wr thresholds with no CI band → a √(copies) over-confidence risk on the trigger. BIASED (B) — but partly mitigated by the consecutive-N requirement (3 evals) which is a crude noise filter.
- **C**: WR vs a fixed bot — opponent-side, not a PUCT/visit quantity. Gumbel-agnostic. OK.
- **D band calibration** (PREREG seed4 lens): thresholds are LOW floors used as ABORT triggers — rolling 0.10, early-death 0.05, collapse peak×0.5. This is the OPPOSITE error from seed4's "green-at-0.55": these are abort-on-collapse floors, not a strength bar, and 5-10% floors are defensible against SealBot reality (~18% at fair temp-0.5). NOT the seed4 failure. Bands OK directionally.
- **E**: this IS the L50 replacement that the V_spread canary deferred to — chosen BECAUSE V_spread failed construct validity. But config.py:64-93 / §D-FOUNDING flags SealBot-WR itself as the project's "flagged-wrong instrument" (misread Objective-A off-distribution as Objective-B regression, misdirected 6 investigations); it is supposed to be DEMOTED in favor of the strength/robustness aborts, yet this sliding-window abort keeps it as `wr_hard_abort_enabled: bool = True` (PRIMARY). Construct-validity tension: re-arming the demoted instrument as primary. BIASED (E).
- **F**: overlaps `strength_regression_abort` (the intended replacement) in role, but measures a different opponent set; not a duplicate emit.
- **Verdict: BIASED** (B no-CI/no-n_eff on a wr-trigger + E re-arms the flagged-wrong instrument as primary). **Action: fix** — attach a deduped-bootstrap CI / n_eff to each wr point before it can trip an abort, and reconcile the True default against the §D-FOUNDING demotion (strength/robustness aborts are the spec primary).

## strength_regression_abort_alert  (alert_rules.py:199 `check_strength_regression_abort`)

- **A**: mean of last-N reference-set aggregates < floor, consecutive-N, with cycle-density suppression (l.221-238). Math sound.
- **B**: aggregate is over a FIXED frozen reference set (cycle-robust by design); consumes a pre-aggregated scalar, no CI exposed. Same no-CI concern as wr, but lower-stakes (`strength_abort_enabled=False` default until §2.5 calibration locks the floor). BIASED-lite.
- **C**: checkpoint-relative strength vs frozen set; opponent-side, Gumbel-agnostic. OK.
- **D band**: `strength_abort_floor=0.0` is a PLACEHOLDER (config.py:81 "TBD — locked at §2.5 calibration"); abort disabled until set. `strength_cycle_density_max=0.15` vs observed 0.073 — calibrated from this regime. Floor uncalibrated but explicitly gated OFF, so not live-misleading. Borderline.
- **E**: this is the §D-FOUNDING-correct instrument (replaces SealBot-WR). Cycle-suppression encodes the right asymmetry (cheaper error = missed abort). Construct-valid by design. OK.
- **F**: distinct from wr abort. OK.
- **Verdict: BIASED** (B — no CI on the aggregate trigger; D floor is an un-calibrated placeholder). **Action: fix** — lock `strength_abort_floor` via §2.5 calibration and attach a CI/n_eff to the aggregate before enabling. (Default-off keeps it from being WRONG today.)

## robustness_abort_alert  (alert_rules.py:295 `check_robustness_abort`)

- **A**: mean off-window forced-win rate > warn_threshold for consecutive-N past min_step (l.307-320). Math sound; no cycle suppression (correctly — robustness rejection unambiguous).
- **B**: rate over off-window probe games; no CI exposed; default `robustness_abort_enabled=False` (PROMOTE+WARN). BIASED-lite (no CI), mitigated by consecutive-N + default-off.
- **C**: off-window exploitability — the RIGHT adversarial instrument per CLAUDE.md "self-play→external kill" guidance; Gumbel-agnostic. OK.
- **D**: `robustness_warn_threshold=0.06` is the EXT-LINK fix-acceptance bar (config.py:85) — derived, not invented; min_step 30k. Calibrated. OK.
- **E**: construct-valid — only instrument that sees the off-window defect a fixed-bot WR false-clears. OK.
- **F**: distinct. OK.
- **Verdict: BIASED** (B — no CI/n_eff on the exploit-rate trigger). **Action: fix** — attach CI to exploit_rate. Closest-to-CORRECT of the abort family; band + construct are sound.

## entropy_collapse_alert  (alert_rules.py:26 `check_entropy_collapse`)

- **A**: `policy_entropy < alert_entropy_min` (l.31). Correct comparison.
- **B**: per-step scalar, no game aggregation. OK.
- **C**: policy entropy under Gumbel root — Gumbel adds gumbel noise at the ROOT for sampling but the network `policy_entropy` is the raw NN head distribution, independent of the planner. Gumbel-agnostic. OK.
- **D band** (PREREG seed3 lens): `alert_entropy_min=1.0` nats. Banked combined entropy 2.14-2.89 nats; max=ln(362)=5.89. Alert fires when entropy DROPS below 1.0 → flags COLLAPSE (low entropy). Direction CORRECT (low=collapse=bad). This is NOT the seed3 "healthy ≤2.6 backwards" band — that backwards band lives at a different emit site (terminal_dashboard/early_game_probe normalized-entropy), NOT here. This raw-nats floor at 1.0 sits safely below the observed healthy 2.1-2.9 band. OK.
- **E**: mode-collapse is a real failure mode; entropy is its direct measure. OK.
- **F**: pairs with selfplay variant but different stream. OK.
- **Verdict: CORRECT.** **Action: keep.**

## selfplay_entropy_collapse_alert  (alert_rules.py:36 `check_selfplay_entropy_collapse`)

- **A**: prefers `selfplay_model_entropy_batch`, falls back to `policy_entropy_selfplay`; NaN/finite-guarded (l.47-52). Correct.
- **B**: per-step scalar. OK.
- **C**: selfplay NN-head entropy, planner-independent. OK.
- **D**: `collapse_threshold_nats=1.5` nats. Banked selfplay entropy 2.09-3.59 → floor at 1.5 below healthy band, fires only on true collapse. Direction correct, calibrated below observed regime. OK. (Again NOT the seed3 ≤2.6 backwards band.)
- **E**: direct measure of selfplay mode collapse. OK.
- **F**: distinct stream from combined entropy. OK.
- **Verdict: CORRECT.** **Action: keep.**

## grad_norm_spike_alert  (alert_rules.py:57 `check_grad_norm_spike`)

- **A**: `gn == gn` NaN-filter + `gn > alert_grad_norm_max` (l.63). Correct; NaN handled.
- **B**: per-step, no aggregation (`aggregates=false` in inventory). OK.
- **C**: optimizer-side, planner-irrelevant. OK.
- **D**: `alert_grad_norm_max=10.0`. Banked grad_norm max 2.88 → 10.0 is a generous spike ceiling above the observed healthy band. Reasonable. OK.
- **E**: gradient instability is a real training-health failure; grad norm is its direct measure. OK.
- **F**: unique. OK.
- **Verdict: CORRECT.** **Action: keep.**

## loss_increase_window_alert  (alert_rules.py:68 `check_loss_increase_window`)

- **A**: fires when last `n+1` losses strictly increasing, `n=alert_loss_increase_window=3` (l.78-84). Window-tail logic correct; off-by-one guarded (`len(window) <= n` early return, tail `[-n-1:]`). OK.
- **B**: consumes caller-owned deque tail, not game aggregation. OK.
- **C**: loss is optimizer-side, planner-irrelevant. OK.
- **D**: `n=3` consecutive increases — a slope/run band, not a magnitude threshold; cannot be "backwards". Reasonable trip count. OK.
- **E**: a 3-step monotone loss rise is a weak but honest divergence canary; it is named for exactly what it measures (loss increasing). OK.
- **F**: distinct from grad-norm and entropy. OK.
- **Verdict: CORRECT.** **Action: keep.**

---

## Seeds re-derived

### Seed 7 — `alt_spread` "goes NaN" (Axis A) — **INVERTED**
Re-derived from value_spread_canary.py:382-390 + banked train_cdf… step-500 event.
Banked event: `alt_spread=NaN`, `alt_components.n=0`, sibling line
`value_spread_alt_skipped_plane_mismatch {alt_planes:8, model_in_channels:4}`.
alt_spread DOES emit NaN this run → handoff claim "does NOT NaN (re-encode path)" is FALSE.
Cause = plane-mismatch SKIP (model in_channels=4 ≠ 8-plane fixture) with `encoding=None`, so the
re-encode branch (l.362, gated on `encoding_spec is not None`) was never entered and code fell to the
skip sentinel (l.385). Not numerical NaN, not the re-encode path working. INVERTED: documented
rescue mechanism did not run. Drives the alt_spread WRONG verdict (sentinel-as-value, Axis A).

### Seed 3 — entropy band "≤2.6 backwards" (Axis D) — **STALE (not at B5 alert emit sites)**
Re-derived against my entropy alert stats (alert_rules.py:31 floor 1.0; :52 floor 1.5).
Both are raw-nats COLLAPSE floors that fire on LOW entropy (correct direction); banked healthy
entropy 2.1-2.9 (combined) / 2.09-3.59 (selfplay) sits safely above the floors; max ln(362)=5.89.
The "healthy ≤2.6 backwards" band is NOT present at these emit sites — it lives at a different
surface (terminal_dashboard / early_game_probe normalized-entropy, outside this bucket's emit sites).
For B5 alert_rules entropy: band CORRECT. Flagging the seed as not reproducible at my assigned sites.

### Seed 2 — invented depth ~3.0/3.4 literal (Axis A) — **NOT PRESENT (coverage gap)**
No `depth` / `root_concentration` stat at any B5 emit site (value_spread_canary.py, alert_rules.py),
and absent from §4 banked logs. Per PREREG §4, flag absence-of-emit as a coverage finding rather than
inventing a distribution. The depth-literal seed belongs to B3 (search) emit sites, not B5.

### Seed 6 — colony/bce-gap/conversion bands mislabeled as gates (Axis D) — **PARTIAL (colony at B5)**
`colony_extension_fraction` appears in banked logs (0.0-0.026) but the colony 0.15/0.25 bands are not
in alert_rules.py; the value-spread canary uses colony as a value-head bank, not a fraction-gate. The
only colony-derived gate in my files is the V_spread SOFT-ABORT 0.20 / WARN 0.30 (T3) — and L50 itself
(alert_rules.py:334) already DEMOTES that gate to INFORMATIONAL, conceding it is a heuristic not a hard
gate. So for B5 the seed is self-consistent with the code's own demotion comment; bce-gap/conversion
bands are not at B5 emit sites. Confirmed in the limited sense that the colony V_spread gate is labeled
INFORMATIONAL (heuristic) by the code, matching seed6's "heuristic not gate" claim.
