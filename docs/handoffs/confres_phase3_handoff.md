# CONFRES Phase-3 handoff — resume from here (fresh session)

**Branch:** `phase4.5/confres` (branched from master `9aab184`). NOT merged to master.
**Spec:** `docs/designs/confres_design.md` (v4 FINAL) — read it fully first. Audit:
`docs/audits/confres_knob_inventory.md`. This file = state + remaining work + the gating discipline.

Operator (Tom) chose to bank the CONFRES core and resume the highest-blast-radius quarter
(6c/6d/7 + Phase 4/5) in a fresh session with integration gating. Operator directive for impl:
**delegate impl to subagents (sonnet/opus) + a review subagent per batch; commit each clean batch
(one feature = one commit, conventional prefix, no Co-Authored-By, no push).**

---

## DONE — 11 commits (all TDD'd, subagent-reviewed, unit suite green ~2340, launch integration-verified)

| Commit | Batch | What |
|---|---|---|
| 016c61f | — | docs(audit): knob inventory |
| ea27f39 | — | docs(design): ResolvedRunConfig design v4 |
| 20fa291 | 1 P3 | `resolve/bootstrap.py` — validate checkpoint at launch (not late torch.load) |
| 30c4d2e | 2 P5 | `resolve/nsims.py` — collapse eval model_sims 96/128-vs-100/200 to one authority |
| 2fe2f3a | 3 P4 | `resolve/temperature.py` — self-play schedule (moved from pool, L9-ban intact) + eval constant |
| 0253e4e | 4 S2 | `events.py regime_gated_cluster_stats` — iteration_complete schema regime-STABLE (null under Gumbel, not dropped) |
| a19bdab | 5 S1 | `resolve/lr.py` — WARN when declared lr ≠ ckpt-baked on full resume (declared-vs-baked, NOT annealed-effective) |
| a39679b | 6a-i | `resolve/run_config.py` — pure ResolvedRunConfig builder + variant_layers (kind-tagged) + provenance + two-phase |
| 9dfd81b | 6a-ii | wire builder into launch (`orchestrator._build_load_paths`/`build_and_emit_resolved_config`, `scripts/train.py`) + emit `resolved_config` event |
| 4d5dd9c | 6b | F1 resume-precedence FIX (base-inherited defers to ckpt-baked) + B3 null provenance + lr→dict render |
| 6d4bf71 | fix | assert F2 invariant at LOAD time only (not against post-load-mutated config) — fixed a launch-abort |

**Live now:** every launch emits a `resolved_config` event (per-knob value/source/precedence_family/
inputs_seen). F1 behavior change is IN: on a full-checkpoint resume, a variant-wins knob only
inherited from a base config now PRESERVES the checkpoint-baked value (was: base default clobbered
it), with a loud `resume_base_default_deferred_to_baked` WARN. Declared (variant/--config/cli) knobs
still win (E0 preserved); owned keys still checkpoint-owned.

---

## CRITICAL patterns already established (reuse; don't reinvent)

- **`hexo_rl/config/resolve/`** is the resolution authority. `resolve/__init__.py` is intentionally
  EMPTY — no eager imports (the §8 N3 import-cycle guard: importing an eval-touching resolve module
  would cycle via `hexo_rl/eval/__init__.py`). Import specific submodules; keep them cycle-free
  (stdlib + resolve.* + encoding + utils.config only, NO eval/selfplay/training imports in the
  builder).
- **Declaration vs inherited** = `_is_declaration_layer` (kind ∈ {config,variant}), driven by the
  `kind` tag `capture_config_layers` sets (base|config|variant). Do NOT use filename heuristics.
  `compute_declared_keys(layers)` is the single "what did the operator declare" rule.
- **F2 invariant** (`merge(variant_layers) == combined_config`) is asserted at LOAD time in
  `load_train_config` against the PRISTINE config. NEVER re-assert it against a config that
  downstream legitimately mutates (mixing/buffer `<auto>`-path expansion, resume encoding back-prop)
  — that was the 6a-ii launch-abort.
- **Two-phase build (I7/F3):** `resolve_preload_config` (seed/device/tf32, no checkpoint inputs) is
  Phase-A; `resolve_run_config` is Phase-B (post-`init_trainer`). Emission fires post-Phase-B.
- **The 6a-ii except is `except ConfigConflictError`** (NOT ValueError) — a variant-vs-stamp encoding
  conflict is 6b-deferred and must not abort the launch NOR mask a builder bug. NOTE: 6c/7 may make
  the encoding conflict a real hard-raise on the live path — but the existing WARN-log-ckpt-wins
  (D-WS3V3 FIX1c, §171 P3) is intentional for bootstrap forks; treat that as a SEPARATE contentious
  decision, do not bundle it.

---

## GATING DISCIPLINE (the closeout bug taught this — DO NOT skip)

The per-batch gate `pytest -m "not slow and not integration"` MISSES launch/eval-path aborts (it
skipped a 6a-ii launch-abort that only the 23-min integration test caught). For ANY batch touching
the launch path (orchestrator/train.py) or eval construction (6c, 6d, 7): **run the integration
gate before committing** — at minimum:
```
.venv/bin/python -m pytest -q -k closeout_lifecycle tests/     # ~23 min, real training subprocess
```
Also `tests/test_train_lifecycle.py` (`make test.slow`). Budget for these; they are the only thing
that catches launch-abort / eval-dispatch regressions.

---

## REMAINING WORK

### 6c — encoding/radius adoption (HIGH blast radius; byte-pure on no-conflict)
Migrate encoding CONSUMERS onto the resolved spec (design §4, §8, B5b):
- ReplayBuffer built from RAW `config.get("encoding")` (`orchestrator.py:481`) — `normalize(None)→v6`
  on a no-declaration resume. Migrate to the resolved spec.
- recent-buffer sized from the PRE-checkpoint `_registry_spec` (`scripts/train.py:196-198,233-235`)
  — stale after a metadata-wins resume. Migrate. (This is a latent-bug FIX, byte-pure on no-conflict.)
- `allocate_batch_buffers_for_config` v6-shaped literal fallback (`orchestrator.py:670-673`, warn-only)
  → hard-error under CONFRES.
- Extract encoding/radius resolution rules into `resolve/encoding.py` / `resolve/radius.py` (the
  existing raise/normalize logic + `eval_board.resolve_eval_radius`) so both the launch builder AND
  the per-artifact loaders (`checkpoint_loader.load_model_with_encoding`) DELEGATE to the same rule.
- grep-gate: after migration, zero raw reads/writes of the migrated keys outside `resolve/`
  (incl `scripts/` — the 8th-seam class).

### 6d — offline radius hard-error (B6; smaller)
`scripts/evalfair/core.py radius_from_checkpoint` returns None → silent registry default when a ckpt
has no baked schedule (the sanctioned weights-only-strip produces exactly this). HARD-ERROR when
radius is unresolvable + no explicit `--radius-stage`; the strip must preserve the stage. Also name
`run_sealbot_eval.py:439` hardcoded radius defaults + `round_robin.py:485` bare `Board()` for the
grep-gate.

### Batch 7 — full `build_player` factory (BIGGEST)
Single `build_player(resolved_cfg, context, subcontext) -> Player` all ~8 play seams route through
(design §4, §6, B1). Enumerated seams: `evaluator.py:93,115` ModelPlayer +
`defender_dispatch.build_model_bot`→`KClusterMCTSBot` (in-loop eval), `eval_batcher.py` (batched
carve-out: HARD-REFUSE `batched:true` on legal-set OR a KCluster-batched branch — do not run the
off-window-dropping ModelPlayer physics), `deploy_strength_eval.py` (deploy + legal_set),
`evalfair/core.py:217 make_head_bot` + `scripts/eval/gumbel_greedy_bot.py` (offline, dup
`_REQUIRED_KNOBS`), `round_robin.py:440`, `run_sealbot_eval.py:252`, `inference_methods.py:112,129`.
`PlannerSpec` fields: root, interior, bot_impl, n_sims, c_puct, fpu, virtual_loss, legal_set,
temperature_ref, gumbel_scale. UNIFY the construction ENTRY, NOT the physics (ModelPlayer /
KClusterMCTSBot / Gumbel-g0 stay distinct). Arch from the resolved authority, NOT shape-inference
(inference stays a loud `require_encoding_source` fallback). N5: factory passes the encoding label
explicitly; the bot must STOP reading `model.encoding` (make the ctor param required). Off-loop
tools (`our_model_bot`, `viewer/model_loader`, `pretrain_cli.py:157`) are Surface-B delegated
(§11) — route through `resolve.encoding` with `require_encoding_source`, do not restructure physics.
Pin test = the resolved DISPATCH RESULT (which bot impl each (context,subcontext) runs), so a
`policy_pool`/`mode=` swap FIRES the test.

### Phase 4 — frozen mutation tests M1–M7
Already covered by sub-batch tests: M3/M6/M7 + M2-partial + absent-encoding (in
`test_confres_run_config.py`). STILL TO ADD as committed tests: M1 (eval radius S2→6 / S3 via the
SAME call path eval uses — needs 6c radius adoption), M5 (registry5==curriculum5, mutate curriculum→6
→ eval follows), M4 (decode_override → resolve+loud; unset+mismatch→raise). Verdict gate: M1–M7 green
+ golden-run (THREE regimes: PUCT + Gumbel + resume-from-full-ckpt) + full `make test` green.

### Phase 5 — fable5 red-team on the IMPL
3 novel config/checkpoint attacks NOT in M1–M7 (resume-with-CLI-override precedence; anchor-opponent
encoding keys; a SINGLE-RESOLVER row the refactor didn't touch — verify it was single). Plus:
fresh-clone `make test` green; build-from-master clean for the next run's rsync+bundle.

### Final gate + housekeeping
- Full integration gate (closeout + test_train_lifecycle).
- `make bench` ONLY if any Rust file was touched (none so far; 6c/7 are Python).
- **Prune stale agent worktrees before merging to master** — one is on `phase4.5/confres` at an old
  commit with the pre-tuple `load_train_config` caller (locked, harmless now, breaks if reused).
- Aggregation report (`reports/confres/AGGREGATE.md`) per the dispatcher: knob-table summary, M1–M7
  + red-team results, golden-run diff, commit-list proposal for the operator.

---

## Also note
- `bot_batch_share` is RETIRED (base 0; bot-mix proved ~useless) — see memory `bot-mix-retired-s178-useless`.
  Under F1(A), resuming a §178/§180 ckpt that baked 0.15/0.30 WITHOUT an explicit override PRESERVES
  bot-mix ON; declare `bot_batch_share: 0` to force off (the WARN surfaces it).
- Pre-existing flaky/unrelated test failures seen this session: `test_hexb_v7_v6w25_roundtrip`
  (random-sample flake, passes in isolation), `test_closeout_lifecycle` (was the 6a-ii abort — now
  FIXED). Not CONFRES regressions.
