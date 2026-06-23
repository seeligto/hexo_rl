# PREREG — D-STATAUDIT (tracked-stat correctness + bias audit)

**Frozen BEFORE Phase 1. Do not move. Red-team verifies via `git log -p _audit/PREREG.md`.**

## 0. Sandbox pin

- **Baseline SHA (worktree HEAD):** `52067631f07b2ac2a83d25840d1a0b35cc29c90b`
- **Composition:** branch `phase4.5/gumbelprep` tip `e132e67` + tracked WIP (`value_spread_canary.py`,
  `gumbel_search_py.py`, `export_corpus_jsonl.py`, sprint log) + untracked text (configs/variants,
  docs/handoffs, scripts). Captures the **local working state** = closest reachable proxy to the live
  1M Gumbel run's source. Live host is REMOTE + must not be touched; its exact commit is unreachable
  from this sandbox, so per dispatcher §1 the fallback is local — pinned to local working state rather
  than a stale clean commit so the audited monitor/search WIP matches what the run actually uses.
- **Audit is read-only.** Auditors read source; they never edit it. All audit artifacts live under
  `_audit/` ONLY. Any working-tree change OUTSIDE `_audit/` = contamination → that phase VOID, re-run
  from clean.
- **Contamination check (every phase boundary):**
  `git -C <WT> status --porcelain | grep -v '_audit/'` → MUST be empty.

## 1. The 6-axis rubric (frozen — every stat runs all six)

| Axis | Question | Fail signature |
|---|---|---|
| **A — Formula** | Implemented math == intended definition? | wrong math; invented literal/host-cruft constant (e.g. "~3.0/3.4" depth); off-by-one window; unnormalised denom |
| **B — Eff-n honesty** | If it aggregates over games/sims, is n the DISTINCT count? argmax/no-plies collapse n_eff (§D-ARGMAX→~2)? CI width reported and < effect? | counts non-distinct games; √n over-confidence; no CI on a comparison stat; promotion gate < n≈400 |
| **C — Planner-semantics** | Is it a PUCT heuristic applied to a Gumbel-SH regime? Does the quantity mean anything under SH-pinned root visits? | D5 root_concentration/depth "collapse=search-drop" under Gumbel; temperature framing on temp-less Gumbel; visit-distribution health where SH mechanically pins it |
| **D — Band calibration** | Healthy/warn/alert threshold derived from THIS regime, or borrowed/inverted/invented? | entropy "healthy ≤2.6" (backwards; healthy ~3–6 nats, max ln(362)=5.89); wr_sealbot green-at-0.55 (borrowed from wr_best PUCT bar; SealBot reality ~18% at fair temp-0.5); invented depth band |
| **E — Goodhart / construct validity** | Does it measure the thing it's named for? If a lever optimises it, does play-quality follow? | internal metric (pe_self, probe pass, loss) moves while H2H-vs-anchor regresses; "spread = strength" (falsified 3×); colony label measuring a symptom not the disease |
| **F — Redundancy** | Does another tracked stat carry the same info at zero marginal signal? | duplicate emit under two names; derived stat trivially recoverable from a surfaced one |

## 2. Verdict bins + precedence (frozen; most-severe wins)

- **WRONG** — Axis A fail (math/constant) OR Axis C fail (meaningless/misleading under Gumbel).
  → fix-or-drop; Gumbel-meaningless + unfixable → **drop**.
- **BIASED** — Axis B or D fail, OR Axis E fail where signal is real but skewed in magnitude/direction.
  → **fix** (specify it).
- **REDUNDANT** — Axis F fail, no other failure. → **drop**, point to canonical stat.
- **CORRECT** — passes all six (or has no band to mis-calibrate and aggregates nothing). → **keep**.
- Multiple failures → **most severe**: WRONG > BIASED > REDUNDANT > CORRECT.

## 3. Seeds — RE-DERIVE from source, do NOT inherit (a seed confirms only when re-derived)

If you cannot reproduce the documented problem from the code, **that is a finding** (stale doc/heuristic — flag it).

1. `root_concentration`/depth D5 "collapse=search-drop" reading — Axis C.
2. invented depth ~3.0/3.4 literal — Axis A.
3. entropy band ≤2.6 backwards — Axis D.
4. `wr_sealbot` green-at-0.55 + absolute-not-slope — Axis D.
5. §D-ARGMAX `n_eff`→~2 under argmax+no-plies — Axis B.
6. `colony` 0.15/0.25, `bce-gap` 0.1/0.05, `conversion` 20% — bands mislabeled as gates not heuristics — Axis D.
7. `alt_spread` "goes NaN" claim — handoff says it does NOT NaN this run (re-encode path) → verify which is true — Axis A.
8. `forced_win_conversion` 0.686→1.0 "recovery" read on partial evidence — Axis E + coverage gap.

## 4. Frozen banked-log sample (auditors may read ONLY these for empirical claims)

Absolute paths in the MAIN repo (read-only; outside the worktree — reading them cannot contaminate):

- `/home/timmy/Work/Hexo/hexo_rl/logs/train_cdf24392b8414486a28424673f221575.jsonl` (3054 lines — has `alt_spread`, `colony_extension_fraction`, all `policy_entropy*`)
- `/home/timmy/Work/Hexo/hexo_rl/logs/events_cdf24392b8414486a28424673f221575.jsonl` (2335 lines)
- `/home/timmy/Work/Hexo/hexo_rl/logs/train_76cf0f3925ab45e889c7ff12a5b6b2ea.jsonl` (801 lines — newest; `colony_extension_*`, `policy_entropy*`)
- `/home/timmy/Work/Hexo/hexo_rl/logs/events_76cf0f3925ab45e889c7ff12a5b6b2ea.jsonl` (623 lines)
- `/home/timmy/Work/Hexo/hexo_rl/investigation/founding_2026-06-08/rr_5rung.jsonl` (round-robin eval — wr_*/eff_n/CI empirical)

Empirical claim NOT covered by the banked sample (`depth`, `root_concentration` absent from these logs) →
auditors verify the EMIT SITE exists in source and flag "not present in banked sample" rather than
inventing a distribution. Absence-of-emit is itself an Axis-A / coverage finding.

## 5. Review subset (frozen — no cherry-pick)

Phase-2 REVIEW re-audits, on the inventory sorted ascending by canonical `Stat` name (0-based index `i`):
- **30% deterministic sample:** every stat with `i % 10 ∈ {0, 3, 7}` (pre-registered pattern, no RNG).
- **PLUS every `CORRECT`** verdict from Phase 1 (rubber-stamp guard).
- **PLUS every `WRONG`** verdict from Phase 1 (confirm the kill before it feeds S7 deletion).

## 6. Phase→model→bucket assignment (frozen)

- Phase 0 enumeration: 1 agent, sonnet.
- Phase 1: B1 training (sonnet), B2 self-play (sonnet), B3 search (opus/high), B4 eval (opus/high), B5 monitor+summary (opus/high) — parallel.
- Phase 2 review: 1 agent opus/high, MUST NOT be a Phase-1 implementer (fresh agent — guaranteed).
- Phase 3 red-team: 1 agent opus/high, distinct lens from review.
- (Dispatcher names "Opus 4.6 / Sonnet 4.6"; this sandbox maps to available `opus`/`sonnet` tiers.)

## 7. Dispatcher self-test (filled in deliverable)

Inventory from emit sites (not monitor surface); seeds re-derived not inherited; every CORRECT
re-audited + red-teamed on instrument-distribution lens; git diff clean each boundary + PREREG unmoved;
master table = 100% of inventory rows, coverage gaps separated.
