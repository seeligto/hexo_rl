# D-C VALPROBE — Session Handoff

**Branch:** `phase4.5/valprobe_dc` (off confres `2d3335d`). **Date:** 2026-07-10.
**Persistent state memory:** `~/.claude/.../memory/d-c-valprobe-state.md` (loaded into every session).

---

## TL;DR — the one line

**V-CONFIRM.** Run2's value head has a confirmed **value-representation weakness**: on a robust **33–54% of loss games** it stays optimistic (v ≥ −0.5) **≥2 compound turns after the position is irreversibly lost** (point-of-no-return), with only **1.75% false-pessimism** on won games. The alternative — the head *sees* the loss but policy/search walks in (V-KILL) — is **rejected** (EARLY 35%, far below the 60% bar). → **Run3 card #1 (distributional value head) premise CONFIRMED.** The fix belongs in value **representation/targets**, not policy/search.

LATE effect-size range (all clear the 30% threshold): maximal-adversarial **33.3%** / robust-core **38.6%** / as-frozen **45.6%** / censoring-corrected **54.4%** / guard-off **68.4%**. Report as a RANGE — 7 "never-crossed→LATE" games rest on terminal-only provability, so the core-CI just straddles 0.30 (FRAGILE by the design's own convention).

---

## Findings (every important one)

1. **WP1 recognition-lag = V-CONFIRM** (248k primary; 175k secondary = MIXED, different book r4). Validated by an independent REVIEW (5 games hand-derived, all match) + a 6-attack RED-TEAM (survives all; verdict *under*-states the effect).
2. **WP4 value-health corroborates:** value head poorly calibrated across the run2 ladder — ECE 0.30–0.60, decided-sign accuracy near/below chance, MAE never <0.85, a calibration cliff at the **200k r4→r5** transition (recovers by 210k, the best ckpt), and the **248k ckpt regressed** to 90k-era calibration.
3. **Three independent lines converge** on "value head is the problem": WP1 (this) + WP4 + prior D-LOCALIZE (value blind to SealBot-reachable losses).
4. **Native TacticalSolver CEILING-CONFIRMED** — it can only prove *terminal-adjacent* losses (0/6 quiet positions at 1–2M nodes, always budget-exhausted, >1500× node gap). **Perf was never the barrier** (the whole D-PFIT-P2 perf body is already landed+wired; a threat_moves fix gave 11.4× but recall unchanged). It's architectural (full-legal branching at quiet nodes explodes). **Do NOT re-attempt the native solver for early T_provable.** Extends D-FORENSIC-F2 / D-SOLVER-A2 post-widening.
5. **`threat_moves` 11.4× speedup banked** (sound, output-equivalent, 45/45 tactics tests) — useful for the deploy-backup / Z2 solver hooks even though it didn't unblock WP1.
6. **Methodological arc (why it took a detour):** native dead → **SealBot as T_provable prover** (the lineage's real definition of "provable"). Key correction (operator's catch): provable-loss is **non-monotone** in the actual game (opponent blunders → position oscillates lost↔won), so T_provable = **point of no return** (backward from terminal, earliest turn of the final unbroken lost streak), NOT "earliest-ever" — which would have manufactured a spurious inflated V-CONFIRM. **26/57 loss games actually oscillate**, so this correction was load-bearing.
7. **Verdict is host-independent:** the red-team recomputed v_t on the laptop (game-gen host) for all 1333 positions → max|Δv| 9.7e-3, **zero class changes**. The 91.5% replay-match (< frozen 95%) is inherent GPU-Gumbel non-determinism on flat-prior openings; it touches only descriptive q_t, never the verdict-bearing v_t.
8. **WP3-C1** (event-contract test) + **WP3-C2** (canary revive): the `value_spread` canary's run2 empty-chart root-caused (live 4-plane v6_live2_ls model vs an old 8-plane v6 alt bank → NaN-skip 227/227); bank regenerated 4-plane, alt arm now fires (+0.292). Thresholds held conservative (no run2 series to percentile-calibrate).

## Run3 direction (value-head enhancement)

The scalar `tanh` value head is a point estimate `E[outcome]` that **averages the developing losing tail away**. **Card #1 = distributional value head** (65-bin categorical, C51-style): represents the full outcome distribution so the losing tail is explicit; cross-entropy over bins gives a richer gradient. Likely paired with (a) **outcome/search-distilled value targets** (not TD-bootstrap that propagates the net's own optimism — but the distributional head is what lets loss-mass in without flipping win predictions, avoiding the D-INJECT anti-correlation), and (b) possibly a **representational/trunk fix** (D-FULLSPEC: frozen v6 features can't even linearly separate the blind losses). **The WP2 probe set is the run3 validation harness** — does the new head assign real losing-tail mass at these positions the scalar missed?

---

## Marked files (read in this order)

| File | What it is | Priority |
|---|---|---|
| `reports/valprobe/AGGREGATE.md` (`81b96b8`) | THE verdict write-up (one line first) | **READ FIRST** |
| `reports/valprobe/recognition_lag.md` | Frozen WP1 design + §1 verdicts + §4.3 point-of-no-return revision + §7 filled | High |
| `reports/valprobe/248k/summary.json` + `games.jsonl` | Primary measurement (V-CONFIRM, class counts, sweep, cross-tab, oscillation) | High |
| `reports/valprobe/probe_set_v1.jsonl` (**234 positions**) | **Run3 card #1 eval/validation set** (supersedes the 41-position card1_probe_set.jsonl) | High |
| `reports/valprobe/value_health_series.jsonl` + `value_health.md` | WP4 value-health trend (calibration, radius cliff, 248k regression) | High |
| `reports/valprobe/175k/summary.json` | Secondary arm (MIXED, r4) | Med |
| `reports/valprobe/canary_revive.md` | WP3-C1 + C2 (event contract + canary root-cause/fix) | Med |
| `reports/valprobe/solver_abort_evidence.json` + `.../tactics_ceiling_probe.py` | Native-solver ceiling evidence (why native is dead for T_provable) | Med |
| `scripts/valprobe/measure_recognition_lag.py` + `run_valprobe_sealbot.py` | The WP1 pipeline (SealBot point-of-no-return + v_t/q_t) | Ref |
| `scripts/valprobe/sealbot_instrument_check.py` | SealBot feasibility harness | Ref |
| `scripts/valprobe/value_health.py` | WP4 pipeline | Ref |
| `tests/test_event_contract.py` | WP3-C1 CI test | Ref |
| `~/.claude/.../memory/d-c-valprobe-state.md` | Persistent state / resume point | Ref |

## Commit chain (`phase4.5/valprobe_dc`)

`a4e7a13` WP3-C1 · `555ed43` WP1 design freeze · `c23fe23` WP1 native abort · `d12e3a1` install.sh fix · `4692350` threat_moves 11.4× · `c8635d7` WP4 · `6c11db3` SealBot feasibility gate · `ada51c2` WP1 re-run (V-CONFIRM) · `81b96b8` AGGREGATE · `7702b13` WP3-C2 · `9d4c94d` handoff · `cfa42b0` WP2 fix (sealbot_depth relative-path) · `f812dd3` WP2 (153) · `d381672` WP2 top-up (**234 distinct**).

## Open / next — NONE (investigation complete)

- **WP2 DONE: 234 distinct card1 positions** (`probe_set_v1.jsonl`; 41 WP1 + 193 over 5 batches, seeds 20260711/+1000/+2000/20263711/20264711). Exceeded the 200 target (5.7× the original 41). Integrity verified (234/234 distinct, 0 criterion violations). This is run3's card #1 validation set.
- **Run3 next:** build the distributional value head, validate against these 234 positions.
- **Run3:** build card #1 distributional value head; validate against the probe set; consider search-distilled targets + trunk/representation fix.
- **Infra note:** vast secondary box `02e023b4079b` (RTX 5080, ssh -p 13279, key ~/.ssh/vast_hexo) prepped + native-tuned but **ephemeral** — sync anything off it. install.sh hardened for private submodules.

## Reusable lessons

- Native `engine::tactics` solver is **recall-bound on quiet developmental losses** — architectural, not a perf/budget issue. Use SealBot (window-guarded) for that class.
- "Provably lost" is **non-monotone** in a real (suboptimal) game line — use **point-of-no-return** (backward from terminal), never earliest-ever, or you inflate value-blindness metrics.
- Cross-host v_t is bit-stable for classification; GPU-Gumbel replay is *not* 100% same-host either (don't trust a 95% same-host determinism assumption).
