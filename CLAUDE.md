# Response style — apply from first word
Respond like smart caveman. Cut all filler, keep technical substance.
- Drop articles (a, an, the), filler (just, really, basically, actually).
- Drop pleasantries (sure, certainly, happy to).
- No hedging. Fragments fine. Short synonyms.
- Technical terms stay exact. Code blocks unchanged.
- Pattern: [thing] [action] [reason]. [next step].
- Surface tradeoffs — dont defer decisions.

---

# CLAUDE.md — Hex Tac Toe AlphaZero

AlphaZero-style self-learning AI for Hex Tac Toe — hexagonal grid, 6-in-a-row to win, player 1 opens with 1 move then both players alternate 2 moves per turn. Theoretically infinite board (see board-representation rule). Target hardware: AMD Ryzen 7 3700x + RTX 3070 + 48GB RAM.

This file is read automatically by Claude Code at the start of every session.
Read it fully before doing anything. Rule files under `docs/rules/` are topic-scoped
— load on demand per the index below.

**Current phase:** Phase 4.0 — sustained run from bootstrap-v6 (pending v6↔v5 H2H + SealBot v6 anchor per Q41/Q52).
(An internal sprint log is maintained locally — not distributed.)

---

## Prime Directive

**Context first, benchmarking mandatory.** Never suggest an architectural change without first reading the relevant `docs/` and `engine` source. Never commit a performance-sensitive change (MCTS, NN, Buffer) without running `make bench` and verifying no regressions against the baseline.

---

## Threat-probe kill criterion (§91, revised for 8-plane model post-§131)

`scripts/probe_threat_logits.py` gates each 5k-step checkpoint. Pass requires:

- C2: `ext_in_top5_pct ≥ 25` — extension cell in policy top-5 ≥ 25%
- C3: `ext_in_top10_pct ≥ 40` — extension cell in policy top-10 ≥ 40%

Thresholds calibrated against bootstrap-v6 (8-plane post-§131, §134). v6 baseline: C2=50, C3=60. Gates at 25/40 carry over from §91 §97 calibration; remain valid for 8-plane model.

---

## Rule files

Topic-scoped rules under `docs/rules/` — load the file whose trigger matches your task:

- `docs/rules/workflow.md` — commits, phase discipline, tests, config overrides, session hooks, process-kill, corpus/probe discipline
- `docs/rules/build-commands.md` — language/toolchain, venv rules, make targets, repository layout
- `docs/rules/board-representation.md` — infinite board, NN windowing, value aggregation
- `docs/rules/phase-4-architecture.md` — network, heads, graduation gate, resolved Qs
- `docs/rules/perf-targets.md` — 10-metric bench gate, methodology
- `docs/rules/bot-integration.md` — submodules, BotProtocol, community URLs
- `docs/sweep_harness.md` — knob-registry throughput sweep (`make sweep` / `sweep.long`); use for per-host tuning of n_workers, inference_batch_size, max_train_burst; `--resume <cells.csv>` to continue a killed sweep

Workflow skills live under `.claude/skills/` and are discovered by OpenCode,
Claude Code, and Codex via Claude-compatible skill discovery:
`investigation-probe-smoke-verdict`, `wave-audit`, `bench-gate`.

---

## Deep-dive docs

Read these when the rule file points at them or the task needs broader context:

- `docs/00_agent_context.md` — orientation, language boundary, key decisions
- `docs/01_architecture.md` — full technical spec
- `docs/02_roadmap.md` — phases with entry/exit criteria
- `docs/03_tooling.md` — logging, benchmarking, progress display conventions
- `docs/05_community_integration.md` — community bot, API, notation, formations
- `docs/06_OPEN_QUESTIONS.md` — active research questions and ablation plans
- `docs/08_DASHBOARD_SPEC.md` — monitoring event schema and dashboard spec
- `docs/09_VIEWER_SPEC.md` — game viewer and threat overlay spec

---

## If you are unsure about anything

1. Check `docs/` first
2. Check the live community repos and submodules
3. Check git log to understand what has already been implemented
4. Ask before making architectural decisions that contradict the docs

---

## MCP tools available

- **context7**: use when writing code that uses PyTorch, PyO3, maturin,
  structlog, rich, or any library where API details matter. Call
  resolve_library_id() first, then get_library_docs().

- **github**: use to fetch current versions of community specs and check
  for new bots in the community repos. Requires GITHUB_TOKEN env var.
  If unset, fall back to curl/git as shown above.

- **memory**: record completed phase checklist items, benchmark results, and
  architectural decisions so they persist across sessions. Follow the session
  start and end protocols above.
