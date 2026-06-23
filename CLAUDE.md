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

**Current phase:** §S178 v6 bot-mix recipe launch pending vast operator-run — pre-launch baseline committed 2026-05-18 (9 commits `b26999b..22597fc` on `phase4.5/s178_botmix`: design + T1 bot corpus generator + T2 Rust ply_cap_value split + T3-T7 training-path + T6 yaml). Two mechanism levers vs §175/§177 colony-attractor: SealBot-vs-anchor bot-corpus slot at `bot_batch_share=0.15` + `ply_cap_value` split from `draw_reward` (operator override: -0.5 vs design -0.8). Anchor `bootstrap_model_v6.pt` (SHA `7ab77d2c…372103`); INV26 + INV19 extension pin the outcome split. Bench gate DEFERRED to vast (operator-run post-impl). Hygiene H1-H5 (per `docs/designs/S178_design.md` §10) DEFERRED. §176 Phase B launch prompt at `docs/handoffs/s176_phase_b_prompt.md` retained for follow-up. §175 v6 sustained closed-by-interrupt at step 70176; §177 closed by interrupt at vast (recipe-attractor reproduction, L24 candidate). §173 α multi-window K-cluster selfplay infrastructure operational. v7full (§150 anchor 17.4% n=500), v7e30, v7, v6, v6w25, v6_step20k retained as versioned baselines.
**Reference hardware:** vast 5080 + Ryzen 9 9900X is primary; laptop Ryzen 7 8845HS + RTX 4060 Max-Q is dev; desktop Ryzen 7 3700x + RTX 3070 is legacy reference. See `docs/rules/perf-targets.md`.
(An internal sprint log is maintained locally — not distributed.)

---

## Prime Directive

**Context first, benchmarking mandatory.** Never suggest an architectural change without first reading the relevant `docs/` and `engine` source. Never commit a performance-sensitive change (MCTS, NN, Buffer) without running `make bench` and verifying no regressions against the baseline.

---

## Re-validation discipline

Falsifications here are **objective- and regime-specific** (anti-colony ≠ over-spread-correction ≠ conversion-decline ≠ draw-collapse). Do **NOT** drop a candidate driver or fix by citing a prior falsified-register row, banked result, or prior § without re-validating that the prior finding's **context transfers**. Protocol: cite the prior → state the exact context it was falsified in → test whether that context transfers to the current objective → only then keep/drop. A drop resting on an un-re-validated prior = reject. (§D-OVERSPREAD 2026-06-08: every driver had a tempting prior to drop it on; tested fresh, D1 INVERTED its prior.)

The falsified register itself (the consolidated table — central rows + the additions relocated from split-out phase bodies) lives in `docs/07_PHASE4_SPRINT_LOG.md` § "Falsified Hypotheses Register". This file holds only the protocol above, not the table.

---

## Verify the measurement unit before building a frame on it

A unit error in a **founding measurement** mis-routes every downstream investigation. Before a metric becomes a frame's load-bearing number, confirm its UNIT matches the decision it gates — especially turn-vs-ply (a HTTT turn places TWO stones; the depth-1 single-stone unit undercounts turn wins) and which cell of a multi-stone win is the reachability-relevant one (the COMPLETING cell `pair[1]` that LANDS the win, not the first stone). (§D-COHERENCE 2026-06-08 counted a depth-2 win's in-window FIRST stone as in-window convertibility; the win actually lands on the off-window completing stone — that one-cell mislabel sent a multi-week "NOT multi-cluster, 19%" detour that §D-GLOBALCONC + §D-RECONVERGE reversed, off-window 19%→54%.) Corollary: a BORDERLINE retraction earns a CHEAP eval-only discriminator before any expensive lever; and an inference/self-play lift is a necessary-condition probe — name the self-play→external **kill** link as an explicit OPEN gate and pick the RIGHT external instrument (adversarial / spread-uncapped — a fixed-bot WR can false-clear an off-window defect by construction). Corollary (effective-n, §D-ARGMAX 2026-06-09): a self-play strength CI's effective sample size is the number of **DISTINCT games, not the game count** — a deterministic regime (argmax / temp-0 from a fixed opening) collapses to ~2 games/pair, and a BT/Wilson CI over the raw count is over-confident by √(copies) (a measured √40 = 6.32× narrowing manufactured a spurious "CI-resolved −109 Elo" that an honest deduped-bootstrap CI straddles). Dedupe byte-identical sequences and bootstrap the CI over distinct games before trusting any "CI-resolved" strength gap; opening/opponent **DIVERSITY** is as load-bearing as temperature — the argmax deployment regime needs injected variation to be measurable at all.

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
`investigation-probe-smoke-verdict`, `wave-audit`, `bench-gate`, `hf-upload`.

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
