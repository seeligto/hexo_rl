# CONFRES Knob Inventory (D-B Phase 1 audit)

**Purpose:** complete regime-knob inventory for the CONFRES refactor (goal: exactly ONE
resolver per regime knob, a `ResolvedRunConfig` built once at launch; eval reads the SAME
seam self-play does — `eval_board.py` / dffd5aa is the proven pattern to generalize).

**This table IS the Phase-2 refactor scope.** Phase 2 touches MULTI-RESOLVER rows +
SILENT-OVERRIDE rows only. SINGLE-RESOLVER rows are NOT beautified. DEAD rows are cleanup-only.

**Method:** 4 read-only sonnet5 agents, one per bucket, traced the ACTUAL code on `master`
(HEAD 9aab184), git-diff/git-log to verify "already fixed" claims. Read-only; no edits.
**Status: DRAFT — awaiting operator verdict gate before Phase 2 begins.**

---

## 1. Verdict summary

| Verdict | Count | Knobs |
|---|---|---|
| **SINGLE-RESOLVER** (clean) | 15 | encoding name · window/cluster_window_size · declared_encoding/decode_override · opponent+anchor encoding keys · radius (self-play / eval / deploy / opening ×4) · n_sims self-play · n_sims deploy · Dirichlet · LR schedule · augment · mixing/decay_steps · resume EXCLUDE-set |
| **MULTI-RESOLVER** (bug-class) | 5 | **n_sims eval (model_sims)** · **planner regime** · **temperature** · **seeds** · **bootstrap** |
| **DEAD** (set but never read) | 1 (+3 sub) | `legal_move_radius_jitter`; sub-flags: `temperature_threshold_ply` (training), `dirichlet_enabled` (under Gumbel), `KEPT_PLANE_INDICES` (module hoist) |

Plus **2 SILENT-OVERRIDE flags** (structurally single-resolver, but violate the CONFRES
design law "checkpoint may INFORM, never SILENTLY override a declared value"):
LR-on-full-resume, and the `events.py:259` planner-gated event schema.

---

## 2. Phase-2 scope (MULTI-RESOLVER + SILENT-OVERRIDE), prioritized

| # | Knob | Why it's in scope | Severity | CONFRES action |
|---|---|---|---|---|
| P1 | **planner regime** (Gumbel/PUCT + interior_selector) | 3 independent construction sites — self-play (Gumbel-root+PUCT-interior, `pool.py:434,470`), live eval (bare PUCT+temp0.5+128sim, `evaluator.py:93,115`), deploy (Gumbel g=0, `deploy_strength_eval.py:76,155`). No shared resolver. Confirmed eval⊥deploy⊥selfplay planner divergence. | **HIGH** | The planners SHOULD differ (train vs measure), so do NOT unify. Resolve each from ONE `ResolvedRunConfig` accessor; emit the resolved planner-per-context in the `resolved_config` event; guard so a regression can't silently swap which planner eval uses. |
| P2 | **seeds** | Python/NumPy/Torch seeded once from YAML (`orchestrator.py:115`); Rust self-play RNG (`inner.rs:373`) + ReplayBuffer RNG (`replay_buffer/mod.rs:200`) are per-thread OS entropy — UNREACHABLE from config `seed:`. Self-play trajectories + augment symmetry non-reproducible. | **MED** | Decide scope: plumb `seed` into Rust runner + buffer for reproducibility, OR document Rust-RNG as explicitly out-of-scope and emit that fact. Either way: one resolver, one emitted value. |
| P3 | **bootstrap** | `Makefile:11` `BOOTSTRAP ?= checkpoints/bootstrap_model_v6.pt` — file does NOT exist on disk. Two resolvers (Makefile var + `--checkpoint`). No early-existence guard → `make train` fails late at `torch.load` with an uninformative error. | **MED (cheap)** | Fix the stale default (point at the live bootstrap) + add an existence guard at launch that names the resolved path. Low-effort, high-value. |
| P4 | **temperature** | Two structurally-separate resolvers: self-play quarter-cosine via `resolve_playout_cap_temperature` (`pool.py:399`, Rust `inner.rs:1038`) vs live-eval constant via `eval_pipeline.py:303`→`evaluator.py:163`. Numerically AGREE at current defaults (both τ=0.5) but by different mechanisms. | **MED** | Single resolver + typed accessor per context; emit. L9 cosine-ban surface is SOUND (both resolvers pin threshold→0) — preserve it. |
| P5 | **n_sims eval (model_sims)** | Two seams: pipeline-injected defaults (96/128, `eval_pipeline.py:297`) vs direct-`Evaluator` fallback (100/200, `evaluator.py:157`/`defaults.py:40`). Production always injects → deterministic; diverges only in direct-construction (tests/ad-hoc scripts). | **LOW** | Collapse to one default source so direct construction and pipeline can't disagree; pin the composed value with a test. |
| S1 | **LR schedule** (SILENT-OVERRIDE) | Single resolver, precedence CORRECT (checkpoint-owned on full resume, by design), but a declared variant `lr:` SILENTLY loses to the checkpoint optimizer-state blob with no warning. Weights-only strip remains the required manual procedure. | **MED** | Do NOT change precedence. Make it LOUD: emit declared-vs-effective LR in `resolved_config`; log when they differ. This is the exact CONFRES design-law case. |
| S2 | **event schema under Gumbel** (SILENT-OVERRIDE) | `events.py:259` (`_puct_regime = not gumbel_mcts`) gates `mcts_root_concentration` + cluster stats OUT of `iteration_complete` under Gumbel. Regime switch silently changes the emitted schema; a consumer wired to those fields flatlines with no error. | **LOW-MED** | Emit the fields as explicit `null` under Gumbel (schema-stable) OR document the regime-gated schema in the dashboard spec + guard the consumer. |

---

## 3. Per-bucket detail

### B1 — Encoding & window (all SINGLE-RESOLVER)

| Knob | Resolver(s) | Precedence / silent? | Verdict |
|---|---|---|---|
| encoding name | `resolvers.py:344` (fresh/selfplay), `trainer_ckpt_load.py:78,365` (resume), `checkpoint_loader.py:390` (eval/anchor) — all funnel through registry TOML as canonical | Resume: metadata stamp → `config.encoding` → shape-infer; declared config must AGREE or RAISES (`trainer_ckpt_load.py:399`). Eval: all stamp sources must agree or RAISES (`checkpoint_loader.py:196`). No silent override. | SINGLE |
| window / cluster_window_size | `pool.py:140` (Python spec), Rust runner re-derives from registry by name | Registry TOML → name string → Rust lookup. `config["cluster_window_size"]` is a DERIVED provenance scalar (`trainer_ckpt_load.py:209`), NOT read at runtime. | SINGLE |
| declared_encoding / decode_override | `checkpoint_loader.py:362-428` (single path) | `declared_encoding` = assertion (raises on mismatch); `decode_override` = always wins, logs loudly, never silent; mutually exclusive. | SINGLE |
| opponent + anchor encoding keys | `opponent_runners.py:270-298` (single call site) | Base `eval.yaml` deliberately has NO `encoding:` in `bootstrap_anchor` (deep-merge trap avoided); each variant declares per-anchor; missing → `require_encoding_source=True` → loud fail. | SINGLE |

### B2 — Radius & n_sims

| Knob | Resolver | Precedence / silent? | Verdict |
|---|---|---|---|
| radius — self-play | `step_coordinator.py:503` `_resolve_radius(step)` | schedule ordered; `None`→registry default; variant wins on resume (not in EXCLUDE-set) | SINGLE |
| radius — eval | `step_coordinator.py:517` `_resolve_eval_radius` → `eval_board.py:41` — reads the SAME `self._current_radius` self-play set | override key wins if set, else tracks self-play; zero honoured (not falsy) | SINGLE |
| radius — deploy | `opponent_runners.py:436` → `deploy_strength_eval.py:412` (same `_resolve_eval_radius` value) | single source | SINGLE |
| radius — opening (evalfair) | `evalfair/core.py:49` `radius_from_checkpoint` — calls the SAME `_resolve_radius` unbound | book must match ckpt radius or hard-abort (`core.py:395`) | SINGLE |
| `legal_move_radius_jitter` | `pool.py:453`→Rust `inner.rs:630` guard `... && worker_registry_spec.is_none()` — always `Some` post-§172/§173 | body NEVER executes for registry-spec encodings; `selfplay.yaml:107 true` is a dead write | **DEAD** |
| n_sims — self-play | `pool.py:290-372` | variant wins on resume; hard-raise on misconfig | SINGLE |
| n_sims — eval (model_sims) | `eval_pipeline.py:297` inject (96/128) **vs** `evaluator.py:157`/`defaults.py:40` fallback (100/200) | production always injects → deterministic; direct-`Evaluator` paths see 100/200 | **MULTI** |
| n_sims — deploy | `deploy_strength_eval.py:81` `extract_deploy_knobs` (hard-error on gap); evalfair reads same key from ckpt config | single key path `selfplay.playout_cap.n_sims_full` | SINGLE |

### B3 — Planner / temperature / Dirichlet

| Knob | Resolver(s) | Precedence / silent? | Verdict |
|---|---|---|---|
| planner regime | `pool.py:434,470` (self-play) · `evaluator.py:93,115` (live eval) · `deploy_strength_eval.py:100-165` (deploy) — 3 sites, no shared config | structural seam gap; no checkpoint override (all from YAML/hardcode) | **MULTI** |
| temperature | `pool.py:399` (self-play cosine) · `eval_pipeline.py:303`→`evaluator.py:163` (eval constant) · legacy `utils.py:37` (Python worker only) | numerically agree at defaults, separate mechanisms; L9 cosine ban SOUND (threshold pinned 0 both sides); `temperature_threshold_ply` DEAD for training | **MULTI** |
| Dirichlet | `pool.py:437-439` (self-play only); f4413d6 split confirmed on master | Gumbel arm: NO Dirichlet (removed); PUCT arm: applied when `dirichlet_enabled && !intermediate`; eval/deploy: none. `dirichlet_enabled` flag LIVE for PUCT, structurally DEAD under Gumbel (intentional) | SINGLE |

### B4 — Training regime

| Knob | Resolver | Precedence / silent? | Verdict |
|---|---|---|---|
| LR schedule | `orchestrator.py:260` `build_resume_config_overrides` (LR keys excluded) + `trainer_ckpt_load.py:521-558` (state restore) | checkpoint WINS on full resume (resume-owned); declared variant `lr:` SILENTLY ignored, no warning; weights-only strip = required manual procedure | SINGLE + **SILENT-OVERRIDE (S1)** |
| augment | `loop.py:201` reads merged `train_cfg` | variant wins on resume (not in EXCLUDE-set); reads merged config not `trainer.config` | SINGLE |
| mixing / decay_steps | `orchestrator.py:684` `read_mixing_params` + `train.py:214` | variant wins on resume; schedule evaluates at actual `trainer.step` (correct) | SINGLE |
| seeds | Python: `orchestrator.py:115` `seed_everything` (pre-checkpoint) · Rust self-play `inner.rs:373` + buffer `replay_buffer/mod.rs:200` = OS entropy | Python single-source; Rust RNG UNREACHABLE from config `seed:` — non-reproducible self-play | **MULTI** |
| bootstrap | `Makefile:11` `BOOTSTRAP` var (stale, file missing) + `--checkpoint` | two resolvers; no existence guard; late `FileNotFoundError` | **MULTI** |
| resume EXCLUDE-set | `orchestrator.py:238-257` `RESUME_CHECKPOINT_OWNED_KEYS` (18 keys) | explicit frozenset; variant wins for all non-excluded keys; weights-only bootstrap forks use `fallback_config` entirely | SINGLE |

---

## 4. Historical-strike cross-check (7 strikes + 2 instrument deaths)

| Strike | Lives in knob | Current master permits it? |
|---|---|---|
| **F1 lineage window** (string-form `encoding:` isinstance-dict bug → whole d1m lineage single-window from step 0) | encoding name | **NO.** `normalize_encoding_name` (`resolvers.py:33`) runs before every isinstance dispatch; `explicit_encoding` triggers on both string + dict form (`trainer_ckpt_load.py:129`). Dead by construction. |
| **v2 encoding** (ckpt metadata overriding declared variant encoding) | encoding name | **NO.** `trainer_ckpt_load.py:399` RAISES on stamp≠declared. Weights-only re-stamp is the sanctioned path. |
| **eval radius** (eval pinned to registry default, invisible until 400k) | radius — eval | **NO.** `_resolve_eval_radius` reads self-play's `_current_radius`; guarded by `test_400k_boundary_advances_eval_off_the_registry_default`. Residual: test only covers `run2_mw_fresh`, not all variants. |
| **E0 resume whitelist** (variant overrides dropped on resume) | resume EXCLUDE-set | **NO.** 18-key `RESUME_CHECKPOINT_OWNED_KEYS`; variant wins otherwise; explicit + tested (`test_resume_variant_overrides.py`). |
| **v2 LR** (declared LR silently lost on resume) | LR schedule | **YES (by design, silent).** LR keys resume-owned; declared `lr:` silently ignored on full resume; weights-only strip still required. → **S1**, make loud. |
| **canary plane-mismatch** (probe reading wrong plane count) | encoding / window | **NO.** `probe_threat_logits.py:209` + `validate_arch_against_spec` (`checkpoint_loader.py:244`) raise on plane/channel mismatch. Residual footgun: `LocalInferenceEngine` default `encoding_spec=lookup("v6")`. |
| **monitor never-emitted `value_spread`** | (telemetry, B3) | **STALE — not a strike on master.** Emitter `value_spread_canary.py:419` (called `trainer.py:1332`), consumer `terminal_dashboard.py:309` both present. Memory `t3-vspread-instrument-void` reflected an older state. Live analog is the `events.py:259` regime-gated schema (**S2**). |
| **+2 unnamed** (dispatcher WHY) | — | Not individually named in the WHY; consolidated register lives in `docs/07_PHASE4_SPRINT_LOG.md § Falsified Hypotheses Register`. No new un-covered strike surfaced by the 4 buckets. |

---

## 5. DEAD rows (cleanup only, not Phase-2 refactor)

- `legal_move_radius_jitter` — Rust guard `inner.rs:630` permanently blocks it for registry-spec encodings; `selfplay.yaml:107 true` is a dead write. **No test pins the inertness** (single enforcement point).
- `temperature_threshold_ply` (`selfplay.yaml:30`) — dead for the training path; only the legacy Python worker reads it, and returns 0.0 in eval mode regardless.
- `dirichlet_enabled` under Gumbel — structurally unreachable (Gumbel arm never enters the Dirichlet block). Intentional, not a bug.
- `KEPT_PLANE_INDICES` (`inference.py:26`) — module-level hoist of `lookup("v6").kept_plane_indices`, not read at runtime (infer uses `self.encoding_spec`). Dead code.

---

## 6. Cautions / latent footguns (single-resolver but worth a guard)

1. **`LocalInferenceEngine` default `encoding_spec=lookup("v6")`** (`inference.py:51`) — a caller omitting the kwarg with a non-v6 model silently gets v6 plane geometry. Not triggered in production (all call sites pass the resolved spec); no test pins it. Flag any NEW consumer.
2. **`resolve_from_checkpoint` bypasses `require_encoding_source`** — the gate lives only in `load_model_with_encoding`. A script calling `resolve_from_checkpoint` on an unstamped `v6_live2_ls` checkpoint with a non-descriptive filename silently resolves to `v6_live2` (`resolvers.py:489`).
3. **n_sims eval two-seam** — no end-to-end test pins the composed 96/128; only isolated constants.
4. **evalfair opening-radius silent `None`** — `radius_from_checkpoint` on a checkpoint without a baked schedule returns `None`→registry default with no loud warning; book/ckpt guard only fires when `book_stage is not None`. Check `evalfair_r4_v2.json` has `radius_stage` set.
5. **`trainer.config` vs `combined_config` split ownership** (B4) — mixing/augment/seed read from the merged YAML config; loss weights + train hyperparams read from `trainer.config`. Latent multi-seam not fully covered by tests. `ResolvedRunConfig` should collapse this split.

---

## 7. Corrections to prior beliefs (memory updates pending operator OK)

- `t3-vspread-instrument-void` — **superseded.** `value_spread` IS emitted + consumed on master. Update memory to point at the `events.py:259` regime-gated schema as the live analog.
- `legal-move-radius-jitter-dead-code` — confirmed still DEAD; add that **no test enforces the inertness** (Rust guard is sole enforcement).

---

## 8. Design-law read for Phase 2

Two clean patterns to generalize into `ResolvedRunConfig`:
- **radius** (all 4 contexts share `_current_radius`) and **encoding** (registry TOML canonical + raise-on-conflict) are the model: one resolver, typed accessors, raise (not silently override) on conflict.
- The 5 MULTI + 2 SILENT rows are the delta. Note P1 (planner) and P2 (seeds) are cases where the correct fix is **explicit + emitted divergence**, NOT unification — self-play and eval SHOULD run different planners; the bug is that the divergence is implicit and unguarded, not that it exists.
