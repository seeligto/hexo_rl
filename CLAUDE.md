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

AlphaZero-style self-learning AI for Hex Tac Toe — hexagonal grid, 6-in-a-row to win, player 1 opens with 1 move then both players alternate 2 moves per turn. Theoretically infinite board (see board-representation rule).

This file is read automatically by Claude Code at the start of every session.
Read it fully before doing anything. Rule files under `docs/rules/` are topic-scoped
— load on demand per the index below.

**Current phase:** §175 v6 sustained closed-by-interrupt at step 70176 (2026-05-14T20:56Z); §176 Phase A merged 2026-05-15 with step-20K promoted as canonical anchor (`bootstrap_model_v6_step20k.pt`, SHA `297e0ce0…2bce6a`, 18.0% n=100 vs SealBot [11.7, 26.7]) via Gate 3. Phase B pending launch on fresh branch. §173 α multi-window K-cluster selfplay infrastructure operational (bench gate held). v7full (§150 anchor, 17.4% n=500 vs SealBot), v7e30, v7, v6, v6w25, v6_step20k retained as versioned baselines.
**Reference hardware:** vast 5080 + Ryzen 9 9900X is primary; laptop Ryzen 7 8845HS + RTX 4060 Max-Q is dev; desktop Ryzen 7 3700x + RTX 3070 is legacy reference. See `docs/rules/perf-targets.md`.
(An internal sprint log is maintained locally — not distributed.)

---

## Prime Directive

**Context first, benchmarking mandatory.** Never suggest an architectural change without first reading the relevant `docs/` and `engine` source. Never commit a performance-sensitive change (MCTS, NN, Buffer) without running `make bench` and verifying no regressions against the baseline.

---

## Threat-probe kill criterion

`scripts/probe_threat_logits.py` gates each 5k-step checkpoint and any pre-promotion checkpoint. Full C1–C4 criterion + baseline values + run cadence live in `docs/rules/workflow.md` § "Threat-logit probe". C1–C3 must all PASS; C4 is a warning-only BCE-drift canary.

---

## Encoding registry

`engine/src/encoding/registry.toml` is canonical single source of truth for every encoding (v6, v6w25, v7full, v8, v8_canvas_realness, …). Encoding-aware code: check registry first. Lookup via `hexo_rl.encoding.lookup(name)` (Python) or `engine::encoding::lookup_or_panic(name)` (Rust) — don't reinvent fields, don't hardcode plane counts.

New encoding: add 1 TOML entry, run `python -m hexo_rl.encoding audit` to verify Rust/Python parity + invariants. No factory functions, no scattered constants — both parsers read the same TOML at load. Variant configs use `encoding: <name>`; redundant scalar overrides (e.g. `board_size:`, `n_planes:`) are tolerated by `_check_scattered_keys` when consistent with the named encoding, and rejected when they disagree. Forward pointers: `docs/designs/encoding_registry_design.md` (registry spec, field semantics, migration policy), `docs/designs/encoding_alpha_multiwindow_selfplay_design.md` (α multi-window K-cluster selfplay, §173+).

---

## Rule files

Topic-scoped rules under `docs/rules/` — load the file whose trigger matches your task:

- `docs/rules/workflow.md` — commits, phase discipline, tests, config overrides, session hooks, process-kill, corpus/probe discipline
- `docs/rules/build-commands.md` — language/toolchain, venv rules, make targets, repository layout
- `docs/rules/board-representation.md` — infinite board, NN windowing, value aggregation
- `docs/rules/phase-4-architecture.md` — network, heads, graduation gate, resolved Qs
- `docs/rules/perf-targets.md` — 10-metric bench gate, methodology
- `docs/rules/bot-integration.md` — submodules, BotProtocol, community URLs
- `docs/rules/background-tasks.md` — daily scrape, manifest commit policy
- `docs/rules/checkpoint-archive-policy.md` — eval checkpoint retention, archive cadence
- `docs/designs/encoding_registry_design.md` — encoding registry spec: canonical TOML at `engine/src/encoding/registry.toml`, new encodings go in TOML (not factory functions)
- `docs/designs/encoding_alpha_multiwindow_selfplay_design.md` — α multi-window K-cluster selfplay (§173+)
- `docs/refactor-template.md` — refactor discipline, pure-move rule, bench-gate skip conditions
- `docs/sweep_harness.md` — knob-registry throughput sweep (`make sweep` / `sweep.long`); per-host tuning, `--resume <cells.csv>` continues killed sweep

Workflow skills live under `.claude/skills/` and are discovered by OpenCode,
Claude Code, and Codex via Claude-compatible skill discovery:
`investigation-probe-smoke-verdict`, `wave-audit`, `bench-gate`, `rsync-vast`, `hf-upload`.

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
  start/end protocols in `docs/rules/workflow.md`.
