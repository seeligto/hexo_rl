# §1 GATE-TRACE — which gate fired the live run's promotions

**Resolves the one UNRESOLVED item in `STAT_AUDIT_STRATEGY.md` §1.** Source pin `52067631` (PREREG §0).
Read-only trace; evidence cited to file:line.

## Question (from addendum §1)

The addendum listed two load-bearing promotion gates and could not adjudicate which fires:
- `strength_aggregate` (ref-set Bradley-Terry) — docstring says it "**REPLACES wr_best in decide_promotion**"
  (`gate_logic.py:57-59`)
- `wr_best`/`ci_best` — the CI gate (`gate_logic.py:120-122`, `evaluate_gate`)

Addendum **read (a guess):** "strength_aggregate likely gated the live run's 2 promotions; wr_best/ci_best
is the no-moves fallback." It flagged: "**ACTION: trace which path the 2 live promotions took.**"

## Finding — INVERTED

**`wr_best`/`ci_best` is the SOLE live promotion gate. The `strength_aggregate` path is dead code in this
source pin.** The addendum's read is backwards.

### Evidence chain

1. **`decide_promotion` is the fork** (`gate_logic.py:57-62`):
   ```
   if strength_aggregate is not None:          # strength path
       strength_ok = strength_aggregate >= strength_floor
   else:                                        # fallback
       strength_ok = wr_best_promoted
   ```
   Which branch runs is decided entirely by whether `strength_aggregate` is `None`.

2. **The value comes only from a `.get`** (`eval_pipeline.py:390`):
   `strength_aggregate = results.get("strength_aggregate")` → `None` whenever the key is absent.

3. **No producer writes the key — tree-wide.** Every `results[...] =` site in `run_evaluation` is
   enumerated (`eval_pipeline.py:293,294,398,434,440`): `value_fc2_weight_abs_max`,
   `g4_value_head_band_pass`, `promoted`, `ratings`, `elo_estimate`. **`strength_aggregate` is never
   among them.** No `results.update(...)`, no `asdict(...)`, no `**splat` that could inject it. The
   results dict is initialized minimal at `eval_pipeline.py:288`
   (`{"step":…, "promoted":False, "eval_games":0}`).
   - Rust engine: `grep strength_aggregate|strength_cycle|ref_set engine/` → **0 hits**.
   - Configs: `grep promotion_strength_floor|ref_set|strength_aggregate configs/` → **0 hits**.

4. **The TypedDict says so explicitly** (`result_types.py:97-99`):
   > "Written by the per-round ref-set producer **(operator-gated follow-up)**; consumed by the
   > strength-regression abort + decide_promotion."

   The field is declared for a *follow-up that is operator-gated and unimplemented in this pin*. Declared,
   never produced.

**∴** `results.get("strength_aggregate")` is always `None` → `decide_promotion` always takes the `else`
branch → `strength_ok = wr_best_promoted`. The live gate is `wr_best`/`ci_best` (`evaluate_gate`,
`gate_logic.py:100-129`), AND-combined with the §155-T2 bootstrap floor when `floor_enabled`
(`eval_pipeline.py:372-378`) and the robustness off-window gate when
`offwindow_forced_win_rate` is present (`eval_pipeline.py:391`, `decide_promotion` `robustness_ok`).

## Consequences — revised severity ordering (supersedes addendum §1)

1. **The half-draw CI bug IS the live promotion gate — not a "fallback / next-run" issue.** The addendum
   §2 routed it `BANK → next run` on the rationale "bug is fallback." That rationale is wrong; the gate is
   live. **The routing verdict (BANK, don't hot-patch a running loop) still stands** — but on the
   *no-restart* rule, not on "it's only the fallback."

2. **The 2 live promotions remain TRUSTWORTHY.** The CI bug direction is **conservative** (bare `wins`
   numerator in `_binomial_ci` while `winrate_ok` uses half-draw `wr_best` → CI strictly tighter →
   false-NEGATIVES). Promotions that fired cleared a *stricter-than-intended* bar. What the bug can cause
   is **missed** promotions (loop slower), never false ones. Live-trust conclusion HOLDS.

3. **`strength_aggregate`'s "uncalibrated 0.55 floor" is NOT a live exposure in this pin.** The floor can
   never be reached because the producer never runs. It is a pure *when-implemented / next-run* concern.
   Likewise the strength-regression **abort is inert** for the same root cause (producer never runs) —
   not merely because `floor=0.0 + enabled=False` (`config.py:81`, `alert_rules.py:229,249`). Both
   `step_coordinator.py:1087` and `alert_rules.py:247` guard on `is not None`, so both are dead paths.

4. **The real live promotion-gate surface = the `wr_best`/`ci_best` arithmetic + the §155-T2 bootstrap
   floor AND-combine + the robustness off-window gate.** That is where promotion-correctness scrutiny
   belongs for THIS run.

## Caveat (coverage gap — honest scope)

Live host is REMOTE + unreachable (PREREG §0); its exact source commit is not verifiable. This trace is
sound **for the audited pin `52067631`** (the closest reachable proxy). If the live remote ran a *newer*
source that implemented the ref-set producer, the strength path could be live there — unverifiable from
this sandbox. Per PREREG philosophy, **absence-of-producer is itself an Axis-A/coverage finding**, logged
here. The B4 (eval) bucket of the audit re-run should independently re-derive this as a cross-check.

## Empirical note

The frozen banked-log sample (PREREG §4) contains **zero** `checkpoint_promoted` events from the live
run. The only local promotion event is an April-8 record (`wr_best=1.0`, no `strength_aggregate`/`reason`
fields → predates this gate code, legacy `wr_best` path). The live run's 2 promotions are not in any local
log. So §1 is resolved **structurally** (code reachability), not from live promotion records.
