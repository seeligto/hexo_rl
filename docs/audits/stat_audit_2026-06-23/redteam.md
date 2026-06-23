# Phase 3 — RED-TEAM (integrity stress + strongest-claim refutation)

Pin: worktree HEAD `5da9b24…` (audit-commit tip); SOURCE pin `52067631…` per PREREG §0.
Lens: adversarial — distinct from Phase-2 review. Tried to REFUTE the audit's load-bearing
claims from source + §4 banked logs only. Read-only throughout.

---

## A. Dispatcher self-test (PREREG §7)

| # | Requirement | Result | Evidence |
|---|---|---|---|
| (a) | Inventory from EMIT SITES, not monitor surface | PASS | `inventory.md` header: method = grep over `log.info`/`log.warning`/`emit_event`/`"event":` keys + PyO3 getters + `WorkerStats` struct; "Per-stat metadata derived from source only — NOT from banked logs". 113 distinct emit-crossing stats enumerated (vs 97 prior). Spot-checked: `root_concentration` traced to `mcts/mod.rs:199-212` getter → `events.py:260`, not lifted off the dashboard. |
| (b) | Seeds RE-DERIVED, not inherited | PASS | All 8 seeds carry source line-cites re-run this phase (see §B). 3 INVERTED + 4 STALE/FALSIFIED outcomes — impossible from inheritance; inheritance would rubber-stamp CONFIRM. seed5 INVERTED rederived independently below. |
| (c) | Every CORRECT re-audited + instrument-distribution lens | PASS (asserted on review) | `review.md` covers the PREREG §5 subset = 30% deterministic (i%10∈{0,3,7}) + ALL CORRECT + ALL WRONG. Red-team re-ran the instrument-distribution lens on the highest-traffic CORRECT stats (`mcts_mean_depth`, `policy_entropy`, `value_accuracy_*`) — no hidden band; depth `:.1f` display, entropy floors raw-nats. |
| (d) | git diff clean + PREREG unmoved | PASS | `git status --porcelain \| grep -v '_audit/'` → empty (exit 1, no match). `git log -p _audit/PREREG.md` → single commit `5da9b24` "prereg: freeze … BEFORE Phase 1", no later edits. PREREG frozen. |
| (e) | Master table = 100% of inventory rows (assert; synth runs after) | ASSERTED + bucket-coverage confirmed | Buckets B1–B5 + inventory.md + coverage_gaps.md all present under `_audit/`. Buckets partition the inventory by subsystem (training/self-play/search/eval/monitor). Synthesis (post-Phase-3) MUST emit one master-table row per inventory stat with coverage gaps separated; the per-bucket files supply that 100% input. No orphan subsystem. |

Self-test verdict: all five PASS. No contamination, PREREG genuinely pre-committed.

---

## B. Seed re-derivation cross-check (adversarial)

- **seed1 (root_concentration C)** UPHELD. `selection.rs:124-126` root descent (`cur==0`) uses
  `forced_root_child`, bypassing PUCT; `inner.rs:702-769` Gumbel SH allocates root sims by halving
  schedule. Root visit distribution is SH-pinned → `root_concentration` (`mcts/mod.rs:199-212`,
  max-child/total) is mechanically determined, "collapse=search-drop" is Gumbel-meaningless.
  Refinement holds: `terminal_dashboard.py:474` displays it `:.2f` with NO threshold → construct-validity
  defect, not a firing alarm. depth half correct (interior PUCT descent valid). WRONG/DROP stands.
- **seed2 (invented depth ~3.0/3.4 literal A)** FALSIFIED — UPHELD. grep for `3.0`/`3.4`/depth-band over
  all monitoring: only `mcts_mean_depth` displayed `:.1f`, NO literal band constant. The invented-literal
  doc claim does not reproduce → stale heuristic. depth = CORRECT.
- **seed3 (entropy ≤2.6 backwards D)** FALSIFIED at B1 / STALE at B5 — UPHELD. `config.py:35-36`
  `alert_entropy_min=1.0`, `alert_entropy_warn=2.0`; `alert_rules.py:31` fires on LOW entropy (correct
  collapse direction). Banked `policy_entropy` 2.141–2.889 (n=93) sits above floors. The "≤2.6 backwards"
  band is NOT at any B1 formula or B5 alert emit site — lives only in a normalized-entropy dashboard
  surface (out of audited emit scope). Correctly relocated.
- **seed4 (wr_sealbot green-at-0.55 + absolute-not-slope D)** CONFIRMED — UPHELD, see §C.1.
- **seed5 (§D-ARGMAX n_eff→2 B)** INVERTED — UPHELD with MODIFICATION, see §C.2.
- **seed6 (colony bands as gates D)** FALSIFIED for B2 emit / CONFIRMED at B5 script layer — UPHELD.
  `colony_extension_fraction` has NO band (inventory has_band=false). The 0.15/0.25 appear only in
  `check_phase_c.py:187` (bar-display) + `d1m_monitor.py:779` (comment). The one colony-derived gate
  in alerts (`alert_rules.py:334` V_spread SOFT 0.20/WARN 0.30) is self-DEMOTED to INFORMATIONAL
  (§S181-AUDIT Wave 3 L50) → matches seed6 "heuristic not hard gate".
- **seed7 (alt_spread "does NOT NaN" claim)** INVERTED — UPHELD, see §C.3.
- **seed8 (forced_win_conversion 0.686→1.0 recovery E)** STALE — UPHELD. `forced_win_detector.py:392`
  is a replay-EMA diagnostic, not a tracked B4 `EvalRoundResult` stat; B4 robustness uses
  `offwindow_forced_win_rate`. No `forced_win_trend` events in §4 banked logs → recovery not re-derivable.

---

## C. Refutation of the 7 most consequential findings

### C.1 — `sealbot_gate_passed` BIASED (Axis D) + feeds PRIMARY auto-abort
TARGET: highest-consequence — drives the live HARD-ABORT trigger.
CLAIM: `opponent_runners.py:155` sets `sealbot_gate_passed = win_rate >= 0.5` — a self-play-style PUCT
bar borrowed onto an external bot whose FAIR-regime WR is ~18% (temp-0.5 `defaults.py:32`, 128 sims
`defaults.py:30`); absolute not slope; effectively unreachable.
REFUTE ATTEMPT: is the 0.5 actually load-bearing, or cosmetic? Traced consumers:
`alert_rules.py:88-89 check_*` fires WARNING when `sealbot_gate_passed is False`. Source verified.
Banked §4 has ZERO eval events → no live WR distribution (verified emit site + threshold from source).
**OUTCOME: UPHELD.** 0.5 vs ~18% fair WR ⇒ flag ~always-False ⇒ alert spam. Genuine Axis-D
miscalibration. NUANCE the audit got RIGHT: the PRIMARY auto-abort is NOT this flag but
`check_sealbot_wr_hard_abort` (`alert_rules.py:96`), which uses RELATIVE triggers (rolling-mean,
peak×ratio collapse, early-death) — the correctly-designed slope instrument seed4 said the absolute
flag should have been. The two SealBot stats are correctly differentiated by the audit:
`sealbot_gate_passed` BIASED-D (absolute) vs `wr_sealbot_sliding_window_alert` BIASED-B/E (proxy
construct-validity, not band-direction). Internally consistent.

### C.2 — `strength_aggregate` WRONG (Axis A,E) — REPLACES the H2H promotion gate
TARGET: feeds a promotion gate directly.
CLAIM: WRONG on aggregation/eff-n + construct validity.
REFUTE ATTEMPT: confirm it actually gates. `gate_logic.py:57-58`: when present,
`strength_ok = strength_aggregate >= strength_floor` REPLACES `wr_best`. `eval_pipeline.py:390-395`
populates from `results.get("strength_aggregate")` and feeds `decide_promotion`. So a flawed aggregate
SILENTLY supplants the CI-guarded wr_best gate.
**OUTCOME: UPHELD, MODIFIED severity note.** The replace-semantics is real and high-consequence (Axis E:
an aggregate that doesn't track play-quality would gate promotion with NO Wilson-CI guard — the wr_best
path's `_binomial_ci` protection is bypassed). Axis-A concern (eff-n over raw count) is plausible but
NOT re-derivable from §4 — no eval events banked, and the rr_5rung path uses Bradley-Terry
(`round_robin.py:28`) whose distinctness depends on `distinct_game_key`. Recommend the master table
carry strength_aggregate as WRONG-with-promotion-gate-flag (S7/gate-feeding) — its kill is well-founded
on the replace-without-CI structural fact even if the magnitude is unmeasured here.

### C.3 — `alt_spread` WRONG (Axis A,E) — NaN sentinel; seed7 INVERTED the handoff
TARGET: feeds `both_pass` / V_spread canary.
CLAIM: handoff "does NOT NaN via re-encode" is FALSE.
REFUTE ATTEMPT: maybe re-encode (l.362) DID run and a numerical NaN crept in elsewhere?
Source `value_spread_canary.py:362`: re-encode branch guarded by `encoding_spec is not None and
_in_ch is not None`. Skip branch l.383-386 sets `alt_spread = float("nan")` + components NaN, n=0.
Banked `train_cdf…`: `value_spread_alt_skipped_plane_mismatch`, `alt_planes:8`, `model_in_channels:4`,
`"alt_spread": NaN` — the SKIP fired, not the re-encode. So `encoding_spec` was None (or re-encode threw,
but no `value_spread_alt_reencode_failed` event present → it was None / branch not entered).
**OUTCOME: UPHELD.** Documented rescue (re-encode) did NOT run; NaN is a plane-mismatch SKIP sentinel,
not numerical, not the re-encode-succeeds path. Handoff INVERTED. Construct impact: `both_pass` degrades
to `t3_spread >= SOFT_ABORT` only (l.390) — alt half silently dropped. Both Axis-A (sentinel masquerading)
+ Axis-E (canary measures half what it claims). Note: `both_pass` BIASED-E and `alt_spread` WRONG verdicts
are mutually consistent with the demotion-to-INFORMATIONAL (l.334) — the dead alt channel is no longer a
hard gate, which CAPS the blast radius. Severity WRONG stands (the stat itself is broken) but the
operational damage is bounded by the L50 demotion — worth a master-table note.

### C.4 — `mcts_mean_root_concentration` WRONG (Axis C) — feeds S7 drop
TARGET: a kill that feeds S7 deletion.
CLAIM: Gumbel-meaningless (SH-pinned), drop.
REFUTE ATTEMPT: is it EVER PUCT-meaningful — e.g. on a non-Gumbel fallback path? `inner.rs:735`
branches `if gumbel_mcts`; the non-Gumbel branch would give PUCT root visits where concentration IS
meaningful. So under a PUCT config the stat is valid.
**OUTCOME: MODIFIED.** The DROP is correct for the AUDITED REGIME (live 1M Gumbel run, `gumbel_mcts=true`
— the whole point of the sandbox pin). Under that regime root visits are SH-pinned (banked flat
0.545–0.576) → WRONG/Axis-C holds. But the kill is REGIME-SCOPED, not absolute: the stat is sound under
PUCT self-play. Per CLAUDE.md re-validation discipline, S7 deletion must scope the drop to the Gumbel
regime, NOT delete the emit unconditionally (a PUCT ablation would lose a valid diagnostic). Recommend
S7 carry "DROP-under-Gumbel / KEEP-emit-guarded-by-planner" rather than hard delete. The Axis-C verdict
is UPHELD; the deletion action needs the regime caveat.

### C.5 — seed5 INVERTED: n_eff does NOT collapse to ~2 (Axis B)
TARGET: an INVERTED prior — highest risk of an audit error (inversions are bold).
CLAIM: rr_5rung shows 26-37 distinct game-signatures/pair, NOT 2; live eval injects opening_plies=4 +
color-alt + temp-0.5 breaking all three §D-ARGMAX collapse preconditions.
REFUTE ATTEMPT (hardest): I re-derived distinctness from §4 `rr_5rung.jsonl` INDEPENDENTLY.
Finding-1: records have NO `moves` field (keys: pair, game_idx, p1, p2, winner, plies, n_stones_*,
colony_fraction_winner, …). So the SHIPPED `distinct_game_key` (`round_robin.py:179-183`) falls to the
`__nomoves__`+idx fallback → treats ALL 40 as distinct by idx. The audit's "26-37 distinct signatures"
must use TERMINAL-STAT signatures, NOT the shipped key.
Finding-2: I computed terminal-stat distinct counts per pair: **30–40 distinct/40 raw** (e.g.
(s90k,s100k) 37, (s50k,s100k) 40, (s90k,s112.5k) 30). Confirms wide diversity.
Finding-3: source grounding solid — `defaults.py:32-33` temp=0.5, opening_plies=4; `evaluator.py:212`
applies opening plies; `evaluator.py:259` color i%2 alternation.
**OUTCOME: UPHELD, MODIFIED.** INVERTED conclusion (no collapse on as-shipped defaults) is CORRECT and
double-confirmed. MODIFICATION: the "26-37 distinct signatures" support evidence used a terminal-stat key,
NOT the code's `distinct_game_key` — which on this moveless log cannot prove distinctness at all (it
fallback-distincts everything by idx). Both routes agree "no collapse," but the audit should state the
distinct-count came from terminal-stat signatures, not the shipped dedup utility. The residual hazard the
audit correctly flagged on `ci_argmax_n` (`_binomial_ci` keys raw n, pathology returns if a variant zeros
opening_plies/temp) STANDS.

### C.6 — `strength_cycle_density` WRONG (Axis A,D)
TARGET: WRONG verdict, paired with strength_aggregate.
CLAIM: WRONG.
REFUTE ATTEMPT: `round_robin.py:80 directed_three_cycle_density` + alert `alert_rules.py:223`
`cycle_density >= cfg.strength_cycle_density_max` → non-transitivity abort. Cycle-density over a tiny
5-rung ladder with eff-n already tight: a 3-cycle metric on ~5 players is high-variance, and the abort
band (`strength_cycle_density_max`) is calibrated by `strength_calibration.py:35 calibrate_cycle_density_max`
— need to confirm it's regime-derived not invented. Banked §4 has no eval events → live density absent.
**OUTCOME: UPHELD (Axis A high-variance small-ladder) but Axis-D PARTIALLY MITIGATED.** Unlike the
hand-set 0.5 sealbot bar, cycle_density_max is CALIBRATED (`calibrate_cycle_density_max` exists), so the
band is not "invented" — it's derived. The Axis-A small-n instability of a 3-cycle stat over a 5-node
ladder is the real defect and stands. Recommend down-weighting the Axis-D component of this verdict:
the threshold has a calibration path; the dominant failure is eff-n/variance (Axis A/B), not an
invented band.

### C.7 — `axis_q` / `axis_r` BIASED (Axis D) — warn fires on HEALTHY corpus
TARGET: a BIASED band claim, checkable from source + a baseline value.
CLAIM: `axis_warn=0.45` (`events.py:90`) but corpus baseline `axis_r≈0.4526` > 0.45 → warn fires on
healthy corpus-matching self-play.
REFUTE ATTEMPT: is 0.4526 from the §4 sample? NO — it's in `smoke_baseline.jsonl`
(`axis_distribution_baseline_loaded`, axis_r=0.4525828…), outside §4. So the empirical value is
out-of-sample. But the THRESHOLD (0.45) and the baseline-LOAD emit site are both in source, and §4
banked live `axis_r` runs 0.50–0.54 (`events_fede…`, `events_13046…`) — i.e. live axis_r is ALSO
> 0.45, so the warn band fires on the actually-observed healthy regime regardless of the smoke baseline.
**OUTCOME: UPHELD.** The Axis-D miscalibration is double-supported: (i) loaded corpus baseline 0.4526 >
0.45 from source, AND (ii) §4 banked live axis_r 0.50–0.54 > 0.45. Warn fires perpetually on healthy
data. BIASED-D stands on §4-internal evidence (the live banked values), so the verdict does NOT depend
on the out-of-§4 smoke baseline. Robust.

---

## D. Integrity findings beyond the verdict set

1. **No eval events in §4 banked logs.** Every B4 stat (wr_*, ci_*, strength_aggregate,
   cycle_density, sealbot_gate, elo) and the eval-gate chain is verdicted from EMIT SITE + threshold
   only, never a live distribution. This is correctly disclosed per PREREG §4 (absence-of-emit =
   coverage finding), but it means all B4 BIASED/WRONG verdicts are STRUCTURAL (source-derived), not
   magnitude-measured. The master table must NOT imply B4 magnitudes were observed. coverage_gaps.md
   should enumerate the missing B4 live distributions explicitly.

2. **Regime-scoping of two WRONG/DROP kills.** Both `mcts_mean_root_concentration` (C.4) and the
   depth-family are sound under PUCT; the Gumbel kill is regime-specific. S7 deletion must inherit the
   CLAUDE.md re-validation discipline — scope drops to the Gumbel regime, guard the emit rather than
   delete, so a PUCT ablation keeps the diagnostic.

3. **strength_aggregate replace-without-CI is the sharpest live risk.** Of all gate-feeding stats it
   is the one that silently REPLACES a CI-guarded gate (`gate_logic.py:57`) with an unguarded
   threshold compare. Highest promotion-integrity exposure; warrants the strongest master-table flag.

4. **Verdict internal consistency confirmed.** The two SealBot stats (absolute flag BIASED-D vs
   relative sliding-window BIASED-B/E) and the alt_spread/both_pass pair (WRONG stat + BIASED-E
   consumer, both capped by the L50 INFORMATIONAL demotion) are mutually consistent — no contradictory
   verdicts found across buckets.

---

## E. Overall integrity verdict

**PASS.** The audit is sound and adversarially robust.
- Self-test: all 5 PASS (inventory from emit sites; seeds genuinely re-derived — 3 INVERTED + 4
  STALE/FALSIFIED prove non-inheritance; diff clean; PREREG frozen pre-Phase-1; buckets cover all
  subsystems for the master table).
- Of the 7 stress-tested consequential findings: 4 UPHELD outright (sealbot_gate, alt_spread,
  strength_aggregate replace-semantics, axis_q/r band), 3 MODIFIED (root_concentration kill →
  regime-scope; strength_cycle_density → demote the Axis-D component, band is calibrated; seed5 →
  inversion correct but the distinct-count evidence used terminal-stat key not the shipped dedup key).
- ZERO findings OVERTURNED. No verdict was refuted from source.
- Required follow-through for synthesis/S7: (a) scope the two Gumbel kills to the planner regime
  (guard, don't hard-delete); (b) flag strength_aggregate as the top promotion-integrity risk
  (replace-without-CI); (c) state B4 verdicts are structural (no live eval distribution in §4);
  (d) correct seed5's distinct-count attribution (terminal-stat signatures, not `distinct_game_key`).
