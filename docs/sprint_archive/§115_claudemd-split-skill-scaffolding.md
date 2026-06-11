<!-- Forensic archive extracted from docs/07_PHASE4_SPRINT_LOG.md during compression sprint 2026-05-13. Compressed counterpart in docs/07_PHASE4_SPRINT_LOG.md. -->

## §115 — CLAUDE.md split + skill scaffolding — 2026-04-22

### Motivation

CLAUDE.md had drifted to 734 lines, well over the z.ai instruction-memory
target (<200 lines). It mixed two categories: instruction memory (rules that
apply every session) and learning memory (dated benchmark history, §114
bootstrap-v4 narrative, §102 variance anecdotes). Applying z.ai principles
— scoped rule loading, instruction/learning separation, concrete verifiable
rules — the file was split into seven topic-scoped rule files under
`docs/rules/`, and three workflow skills were scaffolded under
`.claude/skills/` so OpenCode, Claude Code, and Codex all discover them
without duplication.

### Commit sequence (13 atomic commits)

1. `chore(docs): scaffold docs/rules/ directory with topic stubs`
2. `docs(rules): move board-representation content from CLAUDE.md`
3. `docs(rules): move workflow content from CLAUDE.md`
4. `docs(rules): move build-commands content from CLAUDE.md`
5. `docs(rules): move phase-4-architecture content from CLAUDE.md`
6. `docs(rules): move perf-targets content from CLAUDE.md`
7. `docs(rules): move bot-integration + background-tasks content from CLAUDE.md`
8. `docs(claude): shrink CLAUDE.md to index + prime directive + MCP tools`
9. `feat(skills): draft investigation-probe-smoke-verdict skill`
10. `feat(skills): draft wave-audit skill`
11. `feat(skills): draft bench-gate skill`
12. `docs(claude): add .claude/skills/ reference to CLAUDE.md root`
13. `docs(sprint): §115 CLAUDE.md split + skill scaffolding` (this entry)

### Layout delta

| Scope | Before | After |
|---|---|---|
| CLAUDE.md | 734 lines | 87 lines |
| docs/rules/ | — | 7 files (board-representation, workflow, build-commands, phase-4-architecture, perf-targets, bot-integration, background-tasks) |
| .claude/skills/ | — | 3 skills (investigation-probe-smoke-verdict, wave-audit, bench-gate) |

### Learning-memory preservation

The §114 bootstrap-v4 corpus-filter narrative remains at sprint-log line 3764;
CLAUDE.md / workflow.md now carry only the distilled rule ("Corpus + probe
discipline"). The 2026-04-06, 2026-04-09, 2026-04-16, and 2026-04-17 dated
bench variance notes were dropped from the perf-targets rule file and
preserved via pointer to §98 / §102 — no history is lost; authoritative
history lives in this log.

### Zero code or config touched

This refactor is doc-only. No file under `configs/`, `engine/`, `hexo_rl/`,
`tests/`, or `scripts/` was modified. Sustained RL runs on both hosts
continued unaffected.

