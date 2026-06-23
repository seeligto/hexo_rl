# STAT_AUDIT — strategy addendum (consumption guardrails + live-run routing)

Companion to `STAT_AUDIT.md` (D-STATAUDIT, 97 stats). Strategy-layer synthesis on top of the
audit's own verdicts. **Persist this NEXT TO the audit** — it carries carve-outs not in the table.

> **PROVENANCE NOTE (2026-06-23):** The companion `STAT_AUDIT.md` (41KB, 97-stat audit) plus the
> `bucket_B1–B5.md`, `redteam.md`, `review.md`, `inventory.md`, `coverage_gaps.md` working files were
> **LOST** — they lived untracked in a `/tmp` scratchpad worktree (session `f9112b6e`) that was cleaned
> up before being committed. Only `PREREG.md` (committed `5da9b24`) and this addendum survived. The
> audit must be **re-run** to restore the table this addendum annotates. Re-run pins: source
> `52067631`, PREREG frozen at `5da9b24`.

Audit verdict: **passes red-team.** Integrity self-evidencing (PREREG single-commit `5da9b24`,
diff-clean outside `_audit/`, source pinned `52067631`). Seed re-derivation honest — falsified 4 of
its own priors. The items below are strategy reads on TOP of a sound audit, not corrections to it.

---

## 1. UNRESOLVED IN THE AUDIT — resolve before acting

**Which path gates live promotion?** The audit lists TWO load-bearing promotion gates and does not
adjudicate which fires:
- `wr_best`/`ci_best` — "Load-bearing (promotion gate)" — `gate_logic.py:120-122`
- `strength_aggregate` — "**Replaces wr_best in decide_promotion**" — `gate_logic.py:57-62`

Read of the line ranges + `n_distinct_games` ("Live RR writes moves so prod OK"): **strength_aggregate
(ref-set BT) likely gated the live run's 2 promotions; wr_best/ci_best is the no-moves fallback.**

Consequence if true: the headline **half-draw CI bug is a fallback / next-run issue, NOT the live
promotion gate.** The live exposure is instead `strength_aggregate`'s **uncalibrated 0.55 floor**
(`strength_abort_floor=0.0` "TBD") + an **inert regression-abort** (floor 0.0 + `enabled=False` → can
never fire even if turned on). → **ACTION: trace which path the 2 live promotions took.** That decides
the severity ordering below.

> **RESOLVED 2026-06-23 — the read above is INVERTED. See `_audit/gate_trace.md` for the full chain.**
> No code writes `strength_aggregate` into `results` anywhere (Python eval, Rust engine, configs, dynamic
> injection — all checked). `result_types.py:97-99` confirms the ref-set producer is an "operator-gated
> follow-up" — declared, unimplemented in pin `52067631`. So `results.get("strength_aggregate")` is always
> `None` → `decide_promotion` always takes the `else` branch → **`wr_best`/`ci_best` is the SOLE live
> promotion gate; the strength path is dead code, not a fallback.**
> Revised severity: (1) the **half-draw CI bug IS the live gate**, not a next-run issue — but its direction
> is **conservative** (false-negatives), so the 2 promotions remain trustworthy and the only harm is a
> possibly-slower loop; the §2 `BANK` routing still holds, now on the no-restart rule not "it's a fallback."
> (2) The `strength_aggregate` 0.55 floor + regression-abort are **not live exposures in this pin** (producer
> never runs) — pure when-implemented/next-run concerns. (3) Promotion-correctness scrutiny for THIS run
> belongs on `wr_best`/`ci_best` + §155-T2 bootstrap floor + robustness off-window gate.
> Caveat: live host is remote/unverifiable; sound for the audited pin. Re-validation discipline observed —
> prior read cited, context tested, INVERTED on fresh trace.

---

## 2. LIVE-RUN ACTION ROUTING

Standing rule: no-restart, eval/monitor fixes offline; never mutate the running loop's hot path.

| Finding | Touches | Route | Why |
|---|---|---|---|
| half-draw CI/WR cluster (`ci_*`,`wr_*`) | promotion gate (or fallback — see §1) | **BANK → next run** | bug direction is CONSERVATIVE (bare-wins CI stricter → false-negatives, not false-promotes). Live promotions are real; loop only possibly slower. Hot-patching a running promotion gate = live-loop mutation, forbidden. |
| `mcts_mean_root_concentration` D5 co-read | monitor display only | **offline now** | relabel "SH budget-share NOT conviction" + delete `d1m:312-315` co-read; fix getter f64. Zero live risk. |
| `mcts_quiescence_fires` / `_per_step` triangular over-count | monitor diagnostic | **offline now** | uninterpretable as-is; delta-the-tree-counter fix is monitor-side. Zero live risk. |
| `sims_per_sec` 21× undercount | dashboard number | **VERIFY then offline** | confirm e614327 read pos/hr (clean) not sims_per_sec before trusting/distrusting the tuning. Error is undercount = SAFE direction for affordability. |
| `mcts_mean_depth` ~3.0/3.4 literal + <2.5 band | monitor display | **offline now** | delete invented literal `d1m:308` + band `d1m:335`; keep run-relative `depth_health`. |
| `colony_extension_fraction` docstring | diagnosis routing | **offline now** | relabel "disjoint-cluster spam"; route colony diagnosis to `terminal_reason`. No code-gate exists, so no live risk. |
| `strength_aggregate` uncalibrated floor + inert regression-abort | **promotion safety net** | **CONFIRM before next run** | promote-side 0.55 needs a calibrated floor from THIS regime; abort-side is a dead safety net. Not urgent for a HEALTHY run, but it's the real live gate per §1. |
| `early_game` probe `probe_failed` (encoding mismatch) | live coverage hole | **offline now** | the opening-diversity instrument is DEAD → §D-ARGMAX can't be settled empirically until fixed. |

**Net live read:** the run stays healthy and its promotions are trustworthy. Nothing here forces a
restart. Everything is offline-monitor or bank-for-next-run.

---

## 3. S7 GUARDRAIL — DELETE ONLY THESE FOUR

The master table stamps all 16 WRONG as `S7-delete=Y` by the mechanical rule. **DO NOT bulk-delete
S7=Y rows** — that nukes the promotion instrument. Real deletes:

1. `selfplay_model_entropy_batch` — exact alias of `policy_entropy_selfplay`
2. `value_loss_main` — exact alias of `value_loss`
3. `value_spread_both_pass` — re-derivable from two spreads + thresholds, no consumer
4. `mcts_mean_root_concentration` — **as a health signal** (Gumbel-unfixable). Keep the raw emit if a
   PUCT-deployment A/B is ever on the table (deployment regime is undecided — see handoff open
   problem); just strip it from the Gumbel monitor surface.

**Everything else WRONG = FIX-IN-PLACE** (CI numerators, sims_per_sec accumulator, quiescence delta,
wr_argmax_n label). The underlying quantity is recoverable.

---

## 4. S6 GUARDRAIL — residual the audit could not adjudicate

The audit fixed the wr_* **arithmetic** (half-draw/CI) but did not resolve their **planner-mismatch
construct validity** — the handoff's #1 open problem: wr_* are computed under **PUCT+temp-0.5 on a
Gumbel-trained net** → biased-low, noisier PROXY for deployed strength. That's not auditable against an
**undecided deployment regime**, so it was correctly out of scope. **S6 must surface every `wr_sealbot`/
`wr_best` WITH the "PUCT-proxy of a Gumbel net" caveat** until the deployment-regime decision lands.
Surfacing the number clean without that caveat re-imports the exact proxy confusion the handoff warned
about.

---

## 5. HANDOFF CORRECTIONS (stale claims the audit falsified)

Fix in `CONTEXT_HANDOFF` so the next session doesn't re-import them:
- Entropy band "≤2.6 backwards" — **NOT real**; source direction is correct (`early_game` fires HIGH).
- §D-ARGMAX "n_eff→~2 under argmax" — **does NOT reproduce** (eval temp-0.5 samples → distinct games
  real). The n≈400 bar still stands on **raw-power** grounds; the mechanism story is dead.
- `alt_spread` "does NOT NaN this run" — **CONTRADICTED** (NaN at step500; alt bank inert; dual-bank
  advertising is false, value-spread canary runs T3-only).
- depth + root_concentration "absent from banked logs" — **present** (depth med 3.73/3.79, conc flat
  0.545-0.576).

Meta-read: 4 stale claims in one handoff = re-verify handoff stat-claims against source before any
rest on them. Standing discipline, re-confirmed.
