# CONFRES Design — resolution authority per regime knob (D-B Phase 2, v2)

**Status:** v4 FINAL — folds challenge rounds 1–3. Round-3 verified B1 + B3-round-2 CLOSED.
F1 RESOLVED by operator (2026-07-10): **(A) preserve ckpt-baked**. B1 scope: **full `build_player`
factory**. Operator GREENLIT Phase-3 (round-4 re-challenge WAIVED — residuals are textual, caught
by the per-batch sonnet5 review + Phase-4 mutation tests). Proceeding to impl per the §8 order.

**Scope (operator-confirmed at the Phase-1 gate):** the 5 MULTI-RESOLVER + 2 SILENT-OVERRIDE
rows: P1 planner · P2 seeds · P3 bootstrap · P4 temperature · P5 n_sims-eval · S1 LR-loud ·
S2 event-schema. Challenge-v1 established that doing P1 and the radius/eval knobs *correctly*
pulls a wider CONSUMER surface than the audit listed (the `KClusterMCTSBot` eval-dispatch and
the offline eval fleet) — same knobs, more seams. That expanded consumer surface is called out
for the operator at the Phase-2→3 boundary (§14). SINGLE-RESOLVER rows are behaviour-unchanged
but are ADOPTED as the shared rule + surfaced in emission (§2, clarifies B5b).

**Changelog v1→v2:** §13 maps every challenge-v1 finding → resolution → section. Core reframe:
"one *resolution authority* per knob" (a shared rule module invoked at launch AND per-artifact),
replacing the v1 "one builder absorbs everything, built once" overreach (B4). Read §13 first if
re-reviewing.

---

## 1. Goal + design law

**One resolution authority per regime knob.** Each knob has exactly ONE rule function in a
shared module. That function is the only path that decides the knob's value — invoked once at
launch to build the RUN config, and invoked per-artifact by eval/offline loaders. Because both
invocation surfaces import the same rule, they cannot diverge.

Design law (7 historical strikes):
1. **One authority per knob.** Consumers read a resolved value; no consumer re-derives.
2. **Checkpoint may INFORM, never SILENTLY override a declared value.** Declared-vs-stamp
   conflict on an *authority* knob (encoding/arch) = hard `ConfigConflictError` naming both
   sources + values. Where the checkpoint legitimately OWNS the value (optimizer/scheduler
   state), it wins but the divergence is EMITTED + WARN-logged. Never silent. (S1, B2, B3.)
3. **Normalize before dispatch; absent ≠ default on resume.** Polymorphic encoding inputs
   (string/dict/spec) normalize to a canonical type before any `isinstance` branch (F1 dead).
   An ABSENT declaration is a distinct `UNSPECIFIED` sentinel — NOT coerced to the "v6" default
   on resume (B5a): absent → the checkpoint stamp is authoritative (metadata-wins compat), a
   PRESENT declaration that disagrees → raise.
4. **Eval reads the same authority self-play does.** `eval_board.resolve_eval_radius` (dffd5aa)
   is the template; generalise it, and extend it to the OFFLINE fleet (B6).
5. **Emit provenance.** One `resolved_config` event at launch carries the resolved table +
   per-knob `inputs_seen` (every source that OFFERED a value, including the checkpoint-baked
   config for variant-wins knobs — B3). This is the F1-forensic artifact.

---

## 2. Architecture — one rule module, two invocation surfaces (fixes B4, B5b)

```
hexo_rl/config/resolve/                 # the resolution AUTHORITY (new)
  __init__.py     resolve_run_config(...) -> ResolvedRunConfig   # launch/attach surface
  encoding.py     resolve_encoding(variant, stamp, is_resume) -> ResolvedValue   # ADOPTS existing
  radius.py       resolve_radius(schedule, step, override)      # wraps eval_board (existing)
  planner.py      resolve_planner(context, subcontext, cfg, registry) -> PlannerSpec  # P1
  temperature.py  resolve_temperature(context, cfg)             # P4
  nsims.py        resolve_nsims(context, opponent, cfg)         # P5
  lr.py           resolve_lr(variant, ckpt_state)               # S1/B2 — reads STATE blob
  bootstrap.py    resolve_bootstrap(cli, makefile_env, anchor_paths)  # P3
  provenance.py   ResolvedValue, ConfigConflictError, emit_resolved_config
```

**Surface A — launch builder.** `resolve_run_config(registry, variant_layers, combined_config,
checkpoint_stamps, checkpoint_state, cli)` builds a frozen `ResolvedRunConfig` ONCE for THE
RUN's knobs. `variant_layers` is the RAW pre-merge yaml layer chain in LOAD ORDER (base → --config → variant;
later-wins per `_deep_merge`, so the variant is HIGHEST — matching `load_train_config`
orchestrator.py:63-75), carried so the builder has per-key PROVENANCE — which layer PRESENTED a
key, needed to distinguish an operator declaration (incl explicit `null`) from an inherited/absent
key (B3). Build-time INVARIANT (F2): `merge(variant_layers) == combined_config` (asserted), so the
resolver's precedence can only ADD provenance + the ckpt-baked position, never diverge from the
merged value the rest of the code reads. `--config` is a first-class chain position in BOTH the
fresh and resume chains (F2 — the round-3 miss was omitting it).
`checkpoint_state` is the loaded optimizer/scheduler STATE blob (not just metadata/config), so
effective LR is visible (B2). Consumed by the self-play loop, trainer, `StepCoordinator`,
monitor, and the in-loop `Evaluator` (passed the object, not raw config).

**Surface B — per-artifact resolvers.** `load_model_with_encoding`
(`checkpoint_loader.py`), `radius_from_checkpoint` (`evalfair/core.py`), and each offline
instrument's config builder call the SAME rule functions per checkpoint/artifact. They are
DELEGATED-TO, not absorbed. The mid-run anchor/promotion encoding gate (`checkpoint_loader.py:196`)
STAYS — it now imports `resolve.encoding` so its rules are provably identical to the launch
resolution. "Built once" applies to the RUN object; per-artifact loads legitimately re-invoke
the rule.

**Invariant that ties them:** there is no encoding/radius/planner value produced by any path
that does not come from a `resolve.*` function. Grep-gate (§8) enforces: no raw read OR write of
a migrated knob outside `hexo_rl/config/resolve/`.

**`ResolvedRunConfig` object:**
```python
@dataclass(frozen=True)
class ResolvedValue:
    value: Any
    source: str            # registry|variant|checkpoint|checkpoint_state|cli|derived|default|external
    precedence_family: str # raise-on-conflict|checkpoint-wins-loud|variant-wins|derived|documented-external
    inputs_seen: dict      # {source: raw_value} for EVERY source that offered — incl ckpt-baked (B3)

@dataclass(frozen=True)
class ResolvedRunConfig:
    _values: Mapping[str, ResolvedValue]
    def encoding_name(self) -> str: ...
    def window_set(self) -> RegistrySpec: ...                      # ← encoding (registry)
    def eval_radius(self, step: int) -> int | None: ...           # ← curriculum (template)
    def planner(self, context: str, subcontext: str = "default") -> PlannerSpec: ...  # B1
    def temperature(self, context: str, subcontext: str = "default") -> TemperatureSpec: ...
    def n_sims(self, context: str, opponent: str = "default") -> int: ...
    def loop_horizon(self) -> int: ...                            # total_steps split (MEDIUM)
    def anneal_horizon(self) -> int: ...                          # scheduler T_max split
    def bootstrap_path(self) -> str: ...
    def stats_emitted_under(self, regime: str) -> frozenset[str]: ...
```

**Builder invariants:**
- **I1 normalize-first + absent-sentinel (presence BEFORE normalize).** Encoding inputs run
  through `normalize_encoding_name` (`resolvers.py:33`) ONLY after a key-PRESENCE test — because
  `normalize_encoding_name(None)` itself coerces to `"v6"` (`resolvers.py:49`), normalizing an
  absent key first would make absent look like a PRESENT "v6" and I2-raise against any non-v6
  stamp, breaking all 12 no-`encoding:` variants on resume. So: absent key → `UNSPECIFIED`
  sentinel (resume → stamp authoritative; fresh → the "v6" compat default), distinct from a
  PRESENT "v6"; a PRESENT declaration that disagrees with the stamp → raise (B5a). NB: base
  `model.yaml:23` declares `encoding: v6` — under variant-layer presence this base value is
  INHERITED, not a variant declaration, so a no-`encoding:` variant + non-v6 stamp now resolves to
  the STAMP with a WARN instead of today's hard raise (`trainer_ckpt_load.py:396`). Stated, emitted
  delta — not silent; add a mutation case (no-enc variant + non-v6 stamp → stamp wins, WARN).
- **I2 conflict = raise.** raise-on-conflict knobs: two PRESENT differing normalized values →
  `ConfigConflictError(knob, {sourceA: valA, sourceB: valB})`. Reuses `trainer_ckpt_load.py:399`
  / `checkpoint_loader.py:196` logic, moved into `resolve.encoding` and called from both surfaces.
- **I3 checkpoint-wins-loud reads STATE (B2).** LR/scheduler effective values come from
  `checkpoint_state` (optimizer_state `param_groups[i].lr`, scheduler_state `T_max`/`last_epoch`),
  NOT metadata/config. When declared ≠ state-effective → WARN + flag. §5 example is producible.
  Emission keys off the SAME `is_full_ckpt` predicate the restore uses
  (`trainer_ckpt_load.py:332`), NOT mere `optimizer_state` presence — a partial ckpt (e.g.
  `scaler_state` missing) that never restores must fall to the variant LR, not emit a phantom
  state-LR (B2 caveat). The post-emission T_max mutate under `--override-scheduler-horizon`
  (`:551`) MUST gain a log (currently silent).
- **I4 variant-wins: per-layer provenance + ckpt-baked source (B3).** The builder reads
  `variant_layers` (raw pre-merge chain), so it can tell a key PRESENT in the operator's
  `--variant` layer (a declaration, incl explicit `null`) from a key merely inherited/absent.
  Resume precedence for variant-wins knobs is `cli → variant-layer → ckpt-baked config → base
  default` (§3). A variant-layer-present value (incl null) WINS + is emitted. Absent from the
  variant layer but present in the ckpt-baked config → the ckpt-baked value is EFFECTIVE
  (matching today's `config = dict(ckpt["config"])`, `trainer_ckpt_load.py:338`), recorded as
  `source: checkpoint`, WARNed if it differs from the base default. Base-config nulls
  (`training.yaml:182 bot_corpus_path`, `selfplay.yaml:97 seed_corpus_path`) are DEFAULTS, not
  declarations — null-skip applies to them so a resumed seeded/bot-mix run keeps its baked path.
  Closes B3: emission can no longer lie — every variant-wins knob's effective source is recorded.
- **I5 required-key.** Missing required knob → `ValueError` at build (M7). Bootstrap existence
  check (P3) is CONDITIONAL: only when a checkpoint is actually required (resume/bootstrap
  present) so fresh/test runs without `checkpoints/` never raise (build-hygiene, challenge-v1).
- **I6 frozen** (M6).
- **I7 two-phase for pre-load-consumed knobs (F3).** `seed`, `device`, `tf32` are consumed at
  `scripts/train.py:193` BEFORE the checkpoint loads (`:202`), so the full builder (which needs
  `checkpoint_state`) cannot resolve them. Phase-A resolves these launch-only knobs from
  registry + `variant_layers` + cli with NO checkpoint ingestion (their emitted value IS what the
  run used); Phase-B (post-load) resolves everything else. Pre-load knobs are a distinct
  precedence family (launch-only, NO ckpt-baked source — I4 does not apply to them) and are
  flagged `consumed_pre_resolution: true` in the event.

---

## 3. Precedence model (explicit, per-knob)

| Knob | Family | Sources (highest→lowest) | Conflict policy |
|---|---|---|---|
| encoding | raise-on-conflict | stamp ⇔ variant(if PRESENT) → registry; absent=UNSPECIFIED→stamp | RAISE both (I2); absent→stamp, recorded |
| arch | raise-on-conflict | ckpt-inferred ⇔ variant → registry | RAISE (`_resolve_model_hparams`) |
| radius (selfplay/eval/deploy/opening) | derived | ← schedule@step (one source); override pins | curriculum; offline HARD-ERROR on unresolvable (B6) |
| n_sims selfplay | variant-wins | cli→variant→default | variant-wins |
| n_sims eval | variant-wins | cli→variant→ONE default source | variant-wins (P5) |
| n_sims deploy | variant-wins | variant `n_sims_full` (hard-error gap) | variant-wins |
| planner[context,subctx] | variant-wins per-(ctx,subctx) | variant/registry `policy_pool`→dispatch | variant-wins; dispatch-pinned + emitted (B1) |
| temperature[context,subctx] | variant-wins | variant per ctx → default | L9 ban preserved (P4) |
| Dirichlet | variant-wins | variant→default | already single |
| interior_selector | variant-wins (HARD-READ) | variant | KeyError if absent — preserved, no silent default (MEDIUM) |
| lr/optimizer/scheduler-state | checkpoint-wins-loud | ckpt STATE (resume)→variant (fresh) | I3 loud (S1/B2) |
| loop stop (`--iterations`) | cli-only | cli `--iterations` (RELATIVE to resume step); absent → run-forever | NO variant/`total_steps` auto-stop (N6 — operators rely on run-forever); emitted for provenance only |
| anneal_horizon (was scheduler_t_max) | checkpoint-wins-loud | ckpt scheduler_state → variant under `--override-scheduler-horizon` | unchanged semantics, emitted; `total_steps` feeds THIS, never the loop stop |
| augment/mixing/seeds(py) | variant-wins | variant→default; I4 ingests ckpt-baked | variant-wins loud |
| seed (rust selfplay/buffer) | documented-external | OS entropy | emitted external, not config-seeded (P2, §7) |
| eval_seed_base | variant-wins | eval cfg | emitted as its own knob (challenge note) |
| bootstrap path | variant-wins + validate | cli `--checkpoint`→Makefile `BOOTSTRAP`→`_ANCHOR_PATHS`/`<auto>` | existence-validated conditionally (P3, 3rd seam folded) |

`RESUME_CHECKPOINT_OWNED_KEYS` (`orchestrator.py:238`) becomes the membership table the builder
consults for raise-on-conflict vs checkpoint-wins-loud on the resume path; the per-call-site
copies collapse into the builder + the delegated per-artifact resolvers.

For **variant-wins** knobs the RESUME source chain is `cli → variant-layer → ckpt-baked → base
default` (I4, B3) — distinct from the FRESH chain `cli → variant → default` shown in the rows.
The ckpt-baked position is what makes a variant-only key (absent on resume-without-variant,
present in the ckpt config) resolve truthfully instead of the resolver emitting a phantom default.

**RESOLVED (F1 — operator, 2026-07-10): (A) PRESERVE ckpt-baked.** A variant-wins knob with a
CONCRETE base default now RESOLVES to the checkpoint-baked value on resume when no cli/variant/
`--config` layer declares it — a deliberate resume-precedence FIX (the E0-strike direction: a
resume continues the run's baked runtime knobs instead of silently reverting to base defaults).
This RETRACTS §11's blanket "changing resume precedence out of scope" for this case. Emission:
`source: checkpoint`, WARN-loud when ckpt-baked ≠ base default; golden regime (c)
(resume-without-variant of a variant-mismatched ckpt) pins the delta byte-visibly. Example that
MATTERS: a §S178 ckpt baked `completed_q_values` / `ply_cap_value` at a tuned value, resumed for a
follow-up without re-passing the variant — (A) preserves the tuned value instead of the old silent
E0-class revert to base.

**`bot_batch_share` carve-out (operator).** bot-mix (§S178 SealBot-vs-anchor corpus slot) is
RETIRED — proved ~useless — so `bot_batch_share` base stays 0 (`training.yaml:183`, already 0) and
0 is DESIRED regardless of F1; it is NOT a live (A) case. NB under (A), resuming a §178/§180 ckpt
that baked 0.15/0.30 WITHOUT an explicit override would PRESERVE the baked share (bot-mix ON) —
those checkpoints are the retired branch; if ever resumed, declare `bot_batch_share: 0` explicitly
to force it off. The WARN surfaces exactly this case.

---

## 4. Derived + planner knobs — one authority each

- **window_set ← encoding.** `registry.lookup(name)`. The stale `_registry_spec`-sized
  recent-buffer + ReplayBuffer (`train.py:196-198`, `orchestrator.py:481`) migrate to the
  RESOLVED spec — a latent bug fix (byte-pure on no-conflict configs; correct on metadata-wins
  resume where today they size from the pre-checkpoint spec). (B5b, caution #5.)
- **eval_radius ← curriculum.** REUSE `eval_board.resolve_eval_radius(_resolve_radius(step),
  override)`. Extended to offline: `radius_from_checkpoint` calls the same rule and HARD-ERRORS
  when the ckpt carries no baked schedule AND no explicit `--radius-stage` (B6). The sanctioned
  weights-only-strip path must PRESERVE the radius/curriculum stage in metadata (or the offline
  tool requires `--radius-stage`) — no silent registry default.
- **planner[context, subcontext] ← dispatch-aware resolver (B1).** `PlannerSpec` fields:
  `root` (gumbel|puct), `interior` (puct|…), `bot_impl` (dispatch target: `ModelPlayer` |
  `KClusterMCTSBot` | `argmax` | Rust-selfplay), `n_sims`, `c_puct`, `fpu`, `virtual_loss`,
  `legal_set` (decode flag), `temperature_ref`, `gumbel_scale`. (`c_puct` added — N2: a
  search-physics knob read raw at `evaluator.py:93,201`, `eval_batcher.py:205`, deploy
  `_REQUIRED_KNOBS mcts.c_puct`, `build_model_bot` default 1.5, `inference_methods.py:70`,
  `pool.self.c_puct`; without a field the grep-gate + pin test cannot cover it.)
  Resolved per (context, subcontext):
  - selfplay: Rust worker (Gumbel-root+PUCT-interior, fpu_reduction=0.25, virtual_loss on).
  - eval subcontexts: `best`/`bootstrap_anchor`/`sealbot` → **dispatch via `policy_pool`**:
    legal-set → `KClusterMCTSBot` (fpu_q=0, no virtual loss, K-batched); flat → `ModelPlayer`
    (bare PUCT). `random` (96). `argmax_n` (sims=1). `offwindow` adversary (τ=0.0 hardcoded).
  - deploy: Gumbel g=0 + `legal_set` decode flag (`deploy_strength_eval.py:426`).
  The resolver resolves the **dispatch** (`policy_pool → bot_impl`) explicitly, not just a spec
  dict. The per-context pin test (Phase 4) asserts the RESOLVED DISPATCH RESULT — which bot impl
  + physics each (context, subcontext) actually runs — so a `registry.policy_pool` edit or a
  `defender_dispatch.py:65` `mode=` force FIRES the test. Enumerated construction seams the
  resolver must cover: `pool.py` (selfplay), `evaluator.py:93,115` ModelPlayer +
  `defender_dispatch.build_model_bot`→`KClusterMCTSBot` (in-loop eval),
  `eval_batcher.py:35-51,137,205` (in-loop BATCHED eval — a 7th seam that re-implements physics
  inline; batched carve-out below), `deploy_strength_eval.py:100-165,426` (deploy),
  `evalfair/core.py:217-237 make_head_bot` + `scripts/eval/gumbel_greedy_bot.py:167,336` (offline
  deploy-head, with a DUPLICATE `_REQUIRED_KNOBS:71` to collapse into the shared resolver),
  `round_robin.py:440,485`, `run_sealbot_eval.py:252,439`, `inference_methods.py:112,129`
  (offline). This is the B1 fix (round-2 residual folded).

  **Player factory — single construction authority (operator ask, folded).** All player
  construction routes through ONE factory `build_player(resolved_cfg, context, subcontext) ->
  Player` (`hexo_rl/eval/player_factory.py`, new). It reads ARCHITECTURE from the resolved
  encoding/arch authority (registry TOML + stamp — never a fresh raw-config read, never shape-
  inference as the happy path) and SEARCH PHYSICS from the resolved `PlannerSpec.bot_impl`,
  returning the correct player object. This upgrades B1 from "emit + pin the dispatch" to a
  single construction authority: the six enumerated seams above all become thin callers of
  `build_player`, so a mis-dispatch (eval builds deploy's player, or a `policy_pool` edit swaps
  the eval bot) is impossible by construction, not merely detectable.
  Two hard constraints: (i) the factory UNIFIES the construction ENTRY, NOT the physics —
  ModelPlayer (bare PUCT) / KClusterMCTSBot (K-batched, fpu_q=0) / Gumbel-g0 stay distinct
  algorithms the factory SELECTS among from the resolved spec (flat vs legal-set vs deploy
  legitimately differ; forced merger is out of scope, §11). (ii) architecture comes from the
  resolved authority; `detect_encoding_from_state_dict` shape-inference stays a LOUD last-resort
  gated by `require_encoding_source` (warns), never the factory's default path — inference-as-
  happy-path is the F1/canary silent-mislabel class (audit cautions #1,#2).

  **Batched carve-out (B1a).** `eval_batcher.batched_evaluate` disassembles physics into a
  coroutine + shared `infer_fn` and cannot receive a `Player` object. It MUST consult the resolved
  dispatch and either (a) take a KCluster-batched branch for legal-set encodings, or (b)
  HARD-REFUSE `batched: true` on a legal-set encoding (fail loud) — never silently run the
  single-window ModelPlayer physics that drops off-window legal moves (`eval_batcher.py:42`). The
  full batched-KCluster path is a follow-up (§11); the CONFRES fix is the guard. **Label channel
  (N5):** the factory passes the encoding label EXPLICITLY to `KClusterMCTSBot`, never via the
  `model.encoding` attribute mutation (`defender_dispatch.py:100`) — a model shared across two
  arms with different labels is last-writer-wins. **Off-loop tools (B1c):** `our_model_bot.py`,
  `viewer/model_loader.py`, `pretrain_cli.py:157` are Surface-B delegated (scoped in §11).
- **temperature[context,subctx] ← per-(ctx,subctx).** self-play cosine params vs eval-subcontext
  constants (sealbot τ0.5, offwindow τ0.0). L9 cosine-ban preserved (resolver pins threshold=0
  unless explicit opt-in). (P4.)
- **stats_emitted_under(regime) ← planner (S2).** Regime-gated stats emit explicit `null` under
  Gumbel instead of dropping keys (`events.py:259`) — schema-stable.

---

## 5. Emission — the `resolved_config` event

One event at launch, via `emit_event`:
```json
{"event": "resolved_config", "knobs": {
  "encoding": {"value":"v6_live2_ls","source":"checkpoint","precedence_family":"raise-on-conflict",
               "inputs_seen":{"variant":"UNSPECIFIED","checkpoint":"v6_live2_ls"}},
  "lr": {"value":0.0018569,"source":"checkpoint_state","precedence_family":"checkpoint-wins-loud",
         "inputs_seen":{"variant":0.002,"checkpoint_state.param_groups":0.0018569}},   // S1/B2 differ → WARN
  "planner.eval.best": {"value":{"root":"puct","bot_impl":"KClusterMCTSBot","fpu":0.0,
                        "virtual_loss":false,"n_sims":64,"legal_set":true},"source":"registry.policy_pool"}, // B1
  "planner.eval.sealbot": {"value":{"bot_impl":"KClusterMCTSBot","n_sims":128,"temperature":0.5}},
  "planner.deploy": {"value":{"root":"gumbel","gumbel_scale":0.0,"legal_set":true}},
  "bot_batch_share": {"value":0.15,"source":"variant","precedence_family":"variant-wins",
                      "inputs_seen":{"variant":0.15,"checkpoint":0.0}},   // B3: ckpt-baked recorded; differ → WARN
  "seed.rust_selfplay": {"value":"os_entropy","source":"external","precedence_family":"documented-external"},
  "loop_horizon":{"value":400000,"source":"variant"}, "anneal_horizon":{"value":300000,"source":"checkpoint_state"}
}}
```
`decode_override` stays a SEPARATE loud event (unchanged). Post-emission mutations that change
effective LR (`optimizer_state_skipped` fallback `trainer_ckpt_load.py:526`,
`--override-scheduler-horizon` T_max mutate `:551`) each log separately (challenge-v1 note).

**Emission point (F3).** `resolved_config` fires immediately after the Phase-B build, before the
training loop's first consumer read; the Phase-A pre-load knobs are already applied and appear
flagged `consumed_pre_resolution: true`. This is distinct from — and supersedes as the truth
source — the existing pre-checkpoint `train_encoding_resolved` event (`orchestrator.py:179`),
which is explicitly "declared intent, not truth"; `resolved_config` is the post-resolution truth.

---

## 6. Per-target design (current → resolved)

- **P1 planner** — dispatch-aware resolver + single `build_player` factory (§4); covers all
  enumerated seams incl KClusterMCTSBot (B1). No behaviour change (byte-pure): same bots + same
  physics, one construction authority, emitted + dispatch-pinned.
- **P3 bootstrap** — `resolve_bootstrap` (cli→Makefile→`_ANCHOR_PATHS`/`<auto>`, 3rd seam folded)
  + conditional existence-validate (I5). Fix the branch-correct Makefile default. Ships first.
- **P4 temperature** — one authority, per-(ctx,subctx) accessors; byte-pure at defaults; L9 ban.
- **P5 n_sims eval** — collapse the two default seams (`eval_pipeline.py:297` 96/128 vs
  `evaluator.py:157`/`defaults.py:40` 100/200) to ONE default; pin composed value with a test.
- **S1 LR-loud** — I3 reads `checkpoint_state`; emits declared-vs-state-effective + WARN (B2).
  No precedence change → training hot path byte-pure.
- **S2 event schema** — emit `null` for regime-gated stats (schema-stable).
- **P2 seeds** — document-external + emit (§7); `eval_seed_base` emitted as its own knob.

---

## 7. Seeds (P2) — document-external + emit (SURVIVED challenge-v1)

Recommendation stands: do NOT plumb Rust seeding in CONFRES core. OS-entropy Rust RNG
(`inner.rs:373`, `replay_buffer/mod.rs:200`) affects trajectory sampling, reaches no resolved
CONFIG knob — challenge-v1 found no config-knob divergence path. Close the HIDDEN-seam bug by
EMITTING `seed.rust_selfplay: os_entropy`. Plumbing is a bench-gated Rust hot-path change for a
reproducibility property this stochastic codebase does not require; if later needed it is a
scoped follow-up with its own bench gate. Also emit `eval_seed_base` and note the per-game
global np/random reseed (`evaluator.py:215`) as a documented second python-seed seam (config-inert).

---

## 8. Migration order (lowest-risk first) + grep-gate covers WRITERS

1. **P3 bootstrap** — isolated. 2. **P5 n_sims eval** — eval-only. 3. **P4 temperature** —
no-op at defaults. 4. **S2 event schema** — dashboard-only. 5. **S1 LR-loud** — emission only.
6. **encoding/radius adoption** — extract existing logic into `resolve.*`, migrate ReplayBuffer
+ recent-buffer + offline `radius_from_checkpoint` (B6) — byte-pure on no-conflict. 7. **P1
planner** — highest blast radius (self-play/eval/deploy + KClusterMCTSBot dispatch); last.

After each batch: **grep-gate covers reads AND writes** — zero raw reads AND zero `setdefault`/
mutation of a migrated key outside `resolve/` (closes the `eval_pipeline.py:296` setdefault-write
channel, MEDIUM). Paired with the Phase-4 mutation test for that knob. **Gate scope = the whole
repo including `scripts/`** — offline instruments (`scripts/eval/run_a1_solver_backup.py`,
`gumbel_ladder.py`, `run_z2_standalone_ladder.py`, `scripts/exploit_probe.py`, `scripts/d_decode/*`)
construct players directly (the 8th-seam class); each must sit on an explicit EXEMPT-or-MIGRATE
list (§11), not silently uncovered (round-3 F-caveat). The two `_REQUIRED_KNOBS` dicts differ
(gumbel_greedy 6 keys incl `dirichlet_enabled` vs deploy_strength_eval 5) — the collapse must
preserve both shapes.

**Build hygiene (N3 import cycle).** `resolve.radius` "wraps eval_board", and `hexo_rl/eval/__init__.py`
eagerly imports `eval_pipeline`→`evaluator`→(batch 7)`player_factory`→`config.resolve` — a cycle.
Break it with LAZY imports inside `resolve/` (import `eval_board` at call time), or invert so
`eval_board` delegates to `resolve.radius`. No name shadow (`hexo_rl/config` is new; no collision
with `hexo_rl.utils.config`/`hexo_rl.monitoring.config`). Fresh-clone `make test` stays green
(cargo+pytest, no checkpoint dependency; I5's existence check is conditional).

---

## 9. Golden-run byte-pure — THREE regimes

Run THREE reference launches `--iterations 3 --min-buffer-size 64`: (a) base-default (PUCT arm);
(b) a Gumbel variant — exercises the S2 absent≡null exception (the default reference is PUCT;
live runs are Gumbel); (c) **resume-from-full-checkpoint** — B3(b) and N6 bite ONLY on resume
(the launch-only arms never exercise the ckpt-baked source chain or the loop-stop semantics).
Pre-existing behaviour events must be byte-identical pre/post; the new `resolved_config` event is
additive/excluded. The ONLY intended schema delta: S2 fields absent (pre) ≡ null (post),
key-scoped to the enumerated S2 fields.

---

## 10. Mutation-test satisfaction map (Phase 4, frozen M1–M7)

| Test | Satisfied by |
|---|---|
| M1 eval radius S2→6/S3 same call path | `eval_radius(step)` wraps `resolve_eval_radius(_resolve_radius(step))` (§4) |
| M2 ckpt v6_live2 + variant v6_live2_ls → raise both | I2 (§2) |
| M3 string ≡ dict encoding | I1 normalize-first (§2) |
| M4 decode_override → resolve+loud; unset+mismatch→raise | separate loud event (§5)+I2 |
| M5 registry5==curriculum5, curriculum→6 → eval follows | curriculum-derived, all variants (§4) |
| M6 raw key edited post-resolution → consumer unchanged | I6 frozen (§2) |
| M7 missing required key → ValueError at launch | I5 (§2) |
| (added) M2b ckpt-baked variant-wins key differs from resolved → WARN emitted | I4 (B3) |
| (added) M1b offline radius unresolvable → HARD-ERROR not registry default | §4 (B6) |
| (added) M-plan eval dispatch = KClusterMCTSBot for legal-set → pinned | §4 dispatch pin (B1) |
| (added) M2c absent-encoding variant + stamped ckpt → resolves to stamp, NO raise | I1 presence-before-normalize (B5a) |
| (added) M-batch `batched:true` on legal-set → HARD-REFUSE (no silent ModelPlayer physics) | §4 batched carve-out (B1a) |
| (added) M-stop resume with variant `total_steps` set → loop does NOT auto-stop (cli-relative only) | §3 loop-stop (N6) |

---

## 11. Out of scope (explicit)

- **Unifying the planners** — self-play Gumbel vs eval PUCT/KClusterMCTSBot is correct AlphaZero;
  CONFRES makes the divergence explicit + emitted, not removed.
- **Changing resume precedence** — owned keys stay owned; only emission (I3/I4) added. EXCEPTION
  (F1, operator-approved 2026-07-10): variant-wins knobs with a concrete base default now PRESERVE
  the ckpt-baked value on resume (was: base-default silently reverted) — a deliberate fix, WARN-loud.
- **Rust RNG behaviour** (P2 → document+emit).
- **DEAD-code removal** (`legal_move_radius_jitter`, `temperature_threshold_ply`,
  `dirichlet_enabled`-under-Gumbel, `KEPT_PLANE_INDICES`) — separate hygiene pass.
- **Off-loop play tools** (`our_model_bot.py`, `viewer/model_loader.py`, `pretrain_cli.py:157`
  `resolve_from_checkpoint`) — Surface-B DELEGATED (§2): they MUST call `resolve.encoding` with
  `require_encoding_source` (no silent shape-inference; a test asserts the inference path warns),
  but they do NOT build the RUN config object and their game-play physics is not restructured.
- **Full batched-KCluster physics** — the batched carve-out (§4) provides only the guard
  (hard-refuse / correct-branch); implementing a K-cluster batched-eval path is a follow-up, not
  a CONFRES batch.
- **Beautifying SINGLE-RESOLVER rows** (but encoding/radius ARE adopted as the shared rule +
  emitted — behaviour unchanged).

---

## 12. Fable5 re-challenge surface

Re-attack the 6 folded BLOCKERs specifically: (B1) can any eval/deploy path construct a bot the
dispatch resolver doesn't cover, or a dispatch swap the pin test won't catch? (B2) can effective
LR differ from the emitted `checkpoint_state` value with no WARN? (B3) can a checkpoint override
a declared or explicit-null variant key without a recorded `inputs_seen` + WARN? (B4) is any
per-artifact encoding gate deleted rather than delegated? (B5) is any absent-declaration variant
mis-resolved, or encoding-consumer left on a raw dict? (B6) can any offline instrument read a raw
ckpt knob or silently default radius? Plus novel constructions + build hygiene (fresh-clone
`make test`, two-regime golden).

---

## 13. Challenge-v1 finding → resolution map (verify these in re-challenge)

| # | Finding | Resolution | §|
|---|---|---|---|
| B1 | planner misses KClusterMCTSBot live eval seam | dispatch-aware `PlannerSpec` (bot_impl/fpu/legal_set) + single `build_player` factory (all play paths route through it; arch from resolved authority not inference); dispatch-pinned test | §4,§6 |
| B2 | S1 can't see state-blob-effective LR | I3 reads `checkpoint_state` (param_groups/scheduler_state) | §2,§5,§6 |
| B3 | null-skip / variant-wins silent ckpt override | I4 ingests ckpt-baked config into inputs_seen + WARN; explicit null = declaration | §2,§3,§5 |
| B4 | "built once" deletes per-artifact gate | reframe: one rule module, two surfaces; per-artifact loaders delegate | §2 |
| B5a | absent encoding ≡ "v6" breaks no-decl variants | UNSPECIFIED sentinel; absent→stamp on resume, present-mismatch→raise | §2 I1 |
| B5b | scope incoherent + 3-dict encoding split | encoding/radius ADOPTED (behaviour-unchanged) + surfaced; ReplayBuffer/recent-buffer→resolved spec | §2,§4,§8 |
| B6 | offline fleet never resolves; evalfair silent radius | offline builders call `resolve.*`; radius unresolvable→HARD-ERROR; strip preserves stage | §4,§8 |
| M total_steps: NO variant auto-stop | loop-stop stays cli `--iterations` (N6); `total_steps`→anneal_horizon only | §3 |
| M intra-eval subcontexts + fpu/legal_set/c_puct | (context, subcontext) resolver + spec fields | §4 |
| M grep-gate writers | gate covers setdefault/writes too | §8 |
| M interior_selector hard-read | preserved (KeyError, no silent default) | §3 |
| M resolve_from_checkpoint bypass (pretrain_cli.py:157) | Surface-B delegated: require_encoding_source + test | §11 |
| M golden regimes | PUCT + Gumbel + resume-from-full-ckpt reference launches | §9 |
| M conditional P3 existence | validate only when checkpoint required | §2 I5 |
| **— round-2 residuals folded (v3) —** | | |
| B1-r2 | eval_batcher 7th in-loop seam (drops off-window) + offline make_head_bot/gumbel_greedy | batched carve-out guard + seams enumerated; off-loop scoped Surface-B | §4,§11 |
| B3-r2 | explicit-null uncomputable from merged config + ckpt-baked chain missing | `variant_layers` raw provenance + cli→variant→ckpt-baked→default | §2,§3 |
| B5a-r2 | normalize coerces None→v6 before sentinel | UNSPECIFIED = key-PRESENCE test BEFORE normalization; +M2c | §2 I1,§10 |
| N2 c_puct | not in PlannerSpec | added field | §4 |
| N3 import cycle (resolve↔eval) | lazy imports in resolve/, or eval_board delegates | §8 |
| N5 model.encoding mutation | factory passes label explicitly + bot stops READING the attribute (required ctor param) | §4 |
| N6 loop stop | keep cli-relative, no variant-sourced stop | §3 |
| **— round-3 findings folded (v4) —** | | |
| F1 | ckpt-baked vs base-default fork (resume-without-variant) | **RESOLVED (A) preserve-baked** (operator 2026-07-10); WARN-loud; `bot_batch_share` carve-out (bot-mix retired, base 0) | §3,§11 |
| F2 | variant_layers order wrong + --config layer unmapped | order = base→--config→variant; `merge(variant_layers)==combined_config` invariant; --config first-class both chains | §2,§3 |
| F3 | seed consumed pre-resolution + emission point unpinned | I7 two-phase (pre-load knobs launch-only, no ckpt ingestion); emission fires post-Phase-B pre-first-read | §2,§5 |
| F-caveat | grep-gate scope for scripts/*; 8th-seam offline instruments (run_a1_solver_backup, gumbel_ladder, exploit_probe) | state gate scope + exempt-or-migrate list for scripts/eval/*, scripts/exploit_probe.py | §8,§11 |
| B5a-caveat | base model.yaml declares encoding:v6 → today-firing raise on no-enc+non-v6-stamp gets demoted | state the delta + mutation case (no-enc variant + non-v6 stamp → stamp wins, WARN) | §2 I1 |

---

## 14. Note for operator (Phase-2→3 boundary)

Challenge-v1 expanded the CONSUMER surface (not the knob set): P1 must cover the
`KClusterMCTSBot` eval dispatch (the live run2 eval planner) and B6 pulls the offline eval
fleet (evalfair/exploit_probe/round_robin/run_sealbot_eval) under the resolver. Same 7 targets,
more call sites → larger Phase-3 blast radius, especially batch 6 (encoding/radius adoption) and
batch 7 (planner). Flagged for the impl go-ahead.

**Operator-requested addition (folded into B1):** a single `build_player` factory as the ONE
play-construction entry (§4). This is the correct way to make B1 unbreakable, but it turns
batch 7 from "resolver + emission" into "resolver + emission + a player-factory refactor across
~6 construction sites" — the largest single batch. It stays inside P1 scope (same knob, stronger
impl) and forces NO physics merger (ModelPlayer/KClusterMCTSBot/Gumbel stay distinct). CONFIRMED
by operator (2026-07-10): **full `build_player` factory** in batch 7 (not the lighter
emit-and-pin-only variant).
