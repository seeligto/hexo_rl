---
name: wave-audit
description: >
  Run when auditing a cross-subsystem change, a batch of perf fixes, or accumulated
  drift across the codebase. Trigger phrases include "master review", "audit pass",
  "cross-subsystem findings", "parallel subagent", "fix wave", "supply wave",
  "W1", "W2", "W3", "W4", "W5", "batch review before commit", "pre-gate sweep",
  "pre-Phase gate". Use whenever multiple independent findings need to be bucketed
  and dispatched to parallel fix agents with clear surgical scope, or when a
  refactor touches three or more subsystems and needs a single coordinated review
  before merging.
---

# Wave audit

Two-phase discipline: a read-only audit that produces a prioritised fix ledger,
followed by a surgical fix wave where each bucket lands in one commit. Prevents
the "one giant PR that touches everything" failure mode and keeps the bench
baseline interpretable commit-by-commit.

## When to use

- Post-incident audit (e.g. after a correctness bug surfaces, sweep related code).
- Pre-Phase-gate sweep (before declaring Phase 4.5 entry, before promoting a
  checkpoint).
- Perf-wave planning across MCTS / replay buffer / game runner / NN boundaries.
- Dependency refactor or API change touching three or more subsystems.
- Drift review when memory or sprint log flags multiple loosely-related findings.

## Phase 1 — read-only audit

1. **Bucket findings by subsystem.** Group by owner (engine/src/mcts/,
   engine/src/replay_buffer/, engine/src/game_runner/, hexo_rl/selfplay/,
   hexo_rl/training/, configs/, etc.). One subsystem per bucket.

2. **Dispatch parallel Explore subagents per bucket.** Send one Explore agent
   per bucket with a narrow brief: "audit <subsystem> for <finding-type>; list
   lines, severity, and proposed fix". Parallel dispatch — single message,
   multiple tool calls.

3. **Aggregate into a fix ledger.** Sort findings P0/P1/P2:
   - P0 = correctness, data corruption, probe canary fail, active-run blocker.
   - P1 = perf regression > 5%, test flake root cause, stale config key.
   - P2 = nitpicks, style, doc drift.

   Write the ledger to `reports/audits/<topic>_<date>.md`.

## Phase 2 — surgical fix wave

4. **One commit per bucket.** Each bucket lands as a single conventional commit
   with a descriptive subject (`fix(mcts): ...`, `perf(replay-buffer): ...`).
   No batched "fix wave" commits — each fix is independently revertible.

5. **Bench gate between waves.** For perf-touching buckets, capture `make bench`
   before the wave commit and again after. See `.claude/skills/bench-gate/SKILL.md`.

6. **Revert policy.** Any wave commit that causes >5% regression OR a target
   FAIL gets reverted unless the commit body explicitly justifies the tradeoff.

## Anti-patterns

- **Lumping unrelated findings into one commit.** Makes bisect useless and
  blocks partial reverts. The whole point of the wave is surgical scope.
- **Skipping bench baseline between waves.** If W3 regresses and you didn't
  re-baseline after W2, you can't tell whether W3 or W2 caused it.
- **Un-dispatching subagents to chase a hunch mid-audit.** Finish Phase 1 as
  planned; start a new audit cycle if a new hunch emerges.
- **Auditing and fixing in the same agent.** Read-only audit and surgical fix
  are different mindsets — separate them.
- **Re-scoping P0/P1/P2 after seeing the fix cost.** Severity is about impact,
  not convenience.

## Reference waves

- **W1–W5 (Q27 fix wave)**: five surgical buckets after the perspective-flip
  investigation — each landed as a single commit with bench baseline between.
- **Master review 2026-04-18**: cross-subsystem audit before the 2026-04-18
  bench rebaseline; produced the P0/P1/P2 ledger that drove the next two weeks.
- **Supply-wave §113 (2026-04-22)**: 5-commit perf wave with one reverted item
  (#3) and pos/hr +8.5%, bench +12.6%. Left Q35 follow-up for ReplayBuffer
  GIL-release refactor — discipline preserved even when one bucket failed the
  gate.
