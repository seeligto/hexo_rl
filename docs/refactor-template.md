# Refactor Session Template

Use for any refactor candidate from the `docs/FUTURE_REFACTORS.md` deferred section or archived audits under `docs/archive/audit/` (or future audits). One file per session.

## Preamble (every session, every subagent)
Read in order:
1. `CLAUDE.md` + `docs/rules/*.md`
2. `docs/07_PHASE4_SPRINT_LOG.md` — most recent § entries (single source of truth, never re-litigate)
3. `docs/FUTURE_REFACTORS.md` deferred section + any relevant archived audit report under `docs/archive/audit/`
4. The TARGET file + every importer (`rg 'from <module>' --type py` or `rg 'use <module>' --type rust`)

## TARGET (paste audit row)
<file_path>: <LOC> LOC
Responsibilities (current, mixed): <list>
Pain: <why split helps>
Proposed split: <new_file>: <responsibility> per line
Effort: <S/M/L>
Bench impact: <neutral / regression risk / improvement>
Test impact: <test files needing update>

## Goal
Zero behavior change. Decompose per audit's proposed split. All callers updated. All tests pass. Perf-sensitive paths require zero regression vs `make bench` baseline.

## Gate 1 — Pre-flight + diagnosis [READ-ONLY, ~30min]
1. `git status` clean, on master, pulled
2. Verify TARGET LOC matches audit (`wc -l`). If >10% drift → STOP, audit stale
3. Map every external caller. Count by file
4. Map every test touching TARGET (direct + indirect)
5. Identify perf-sensitive boundary: MCTS / NN forward / Buffer push or sample / SelfPlayRunner / hot-path serialization → bench gate REQUIRED. Else `make test.py` sufficient

Output to chat: caller count, test count, perf gate yes/no + reason, confirmed split vs audit (any deviations).

## Gate 2 — Bench baseline [iff perf-sensitive, ~10min]
1. `make bench` — capture all 10 metrics per `docs/rules/perf-targets.md`
2. Save to `/tmp/bench_baseline_pre_refactor.json`
3. If any metric out of band → STOP, refactor on broken baseline = noise

Skip if not perf-sensitive.

## Gate 3 — Surgical extract [one responsibility per commit]
For each new file, leaf-most first:
1. Create new file with extracted responsibility
2. Update TARGET to delegate
3. Update every caller per Gate 1 inventory
4. `make test.py` — verify pass
5. If perf-sensitive: smoke-bench (subset, ~2min) — verify no obvious regression
6. Commit: `refactor(<scope>): extract <responsibility> to <new_file>`

Hard rules:
- Each commit independently compilable + testable
- Pure move, NOT behavior change. Zero logic edits during move. Logic changes = own commit AFTER move lands
- Resist fix-while-moving. Note in `/tmp/refactor_followups.md` for separate commit
- TARGET public API stays identical until last commit (delegation, not removal). Last commit removes shim.

Stop signals:
- Test failure → revert that commit, debug, retry
- Caller count grows → audit's split crosses wrong boundary, STOP
- Hidden coupling (private fns reaching across proposed boundary) → STOP, surface

## Gate 4 — Verification
1. Final `make test.py` — full pass
2. Final `make bench` if perf-sensitive — diff vs Gate 2:
   - Each metric within ±5% of baseline AND in target band
   - Regression >5% or out-of-band → revert to last passing commit, surface
3. `git diff master --stat` sanity check
4. Diff summary line for sprint log: "<original LOC> → <sum of new LOCs>; <N> callers updated; <bench delta or N/A>"

## Gate 5 — Sprint log + handoff
Draft `/tmp/sprint_log_<§>_refactor_<scope>_draft.md`:
- TARGET file + responsibilities split
- New files + boundaries
- Caller update count
- Bench delta (if perf-sensitive)
- Test pass count
- Follow-ups in `/tmp/refactor_followups.md`
- Reference to audit candidate

Do NOT commit sprint log yet — surface to user, await go.
Final commit: `docs(sprint): §<N> refactor <scope> landed`

## Hard constraints (DO NOT)
- Behavior change during move commits
- Public API change before final commit
- Bench regression >5% or out-of-band — revert
- Touch any file outside TARGET + callers + new split files
- Mix this refactor with any other audit candidate (one file per session)
- Skip bench gate on perf-sensitive paths
- Modify config defaults
- Change test assertions to make tests pass

## Soft constraints
- One responsibility per commit, conventional prefix `refactor(<scope>):`
- `make test.py` after every commit
- Branch: `refactor/<file_short_name>` off master
- Subagent prompts always preamble: "Read CLAUDE.md and docs/07_PHASE4_SPRINT_LOG.md first"

## Decision boundaries — surface to user
- Audit description vs current state mismatch (Gate 1)
- Hidden coupling discovered (Gate 3)
- Bench regression >5% (Gate 4)
- Follow-ups discovered during extract

## Done-when
- TARGET LOC reduced per proposed split
- All public API consumers compile + test pass
- `make test.py` clean
- `make bench` within ±5% if perf-sensitive
- Sprint log draft exists, surfaced
- Refactor branch ready to merge

## Bench gate matrix
**Mandatory:** mcts/mod.rs, board/state.rs, selfplay/pool.py, worker_loop.rs, training/trainer.py
**Optional:** training/loop.py, eval_pipeline.py, corpus_analysis.py, terminal_dashboard.py
**Verify-first:** board/moves.rs (if move-gen in MCTS hot-path → mandatory)
