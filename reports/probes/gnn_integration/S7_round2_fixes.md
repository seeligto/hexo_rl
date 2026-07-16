# S7 round-2 — closing the "Re-run after blocker fixes" findings (F5a/F5b/F6/F7/F8) — 2026-07-15

**Input:** `reports/probes/gnn_integration/S7_smoke_gate.md` § "Re-run after blocker fixes".
**Worktree HEAD at start:** `80f7c54`. Machine: laptop RTX 4060 Max-Q (8 GiB), idle before
each verification run (GPU 2 MiB used, no orphan python/torch procs).

**STATUS: DONE.** All five findings closed. No Rust changes (`engine/src/` untouched) — no
`cargo`/bench-gate run needed. Not committed, not `git add`ed per instructions.

---

## Per-fix summary

**F5a — `run4_gnn.yaml` missing `eval_pipeline.gating.best_model_path` namespace.**
1-line fix (`configs/variants/run4_gnn.yaml`): added
`eval_pipeline.gating.best_model_path: checkpoints/best_model_run4_gnn.pt`, exact template of
run3_dist65.yaml:172 (§RUN3-STEP0 law). Launch-path test extended to assert BOTH
`buffer_persist_path` and `best_model_path` are namespaced (never the bare shared defaults)
for `run4_gnn` AND `run4_gnn_smoke`, plus a dedicated test that the two variants' namespaced
paths never collide with each other.

**F5b/F7/F8 — ONE bug class: dense-only `.in_channels` reads on a live model instance.**
Fixed via one shared primitive, `hexo_rl.model.build_net.model_representation(model) ->
"grid"|"graph"` (isinstance-based, unwraps `torch.compile`'s `_orig_mod`), reused at all three
sites:

- **F5b** (`hexo_rl/training/anchor.py::resolve_anchor`): arch-sync is now representation-aware.
  Cross-representation anchor (namespace collision, §RUN3-STEP0-class trap) raises
  `RepresentationMismatch` LOUD — never `AttributeError`. Within one representation: grid path
  is byte-identical to before (same `in_channels`/`value_head_type` compare, pinned by the
  pre-existing `tests/training/test_anchor_branches.py` suite, untouched); graph path compares
  its own arch fields (`representation.output_dim`, `value_head.fc2_bins.out_features` — the
  graph-analogue "does this state_dict even fit" descriptors) and skip-logs (no raise) on a
  genuine net-scale mismatch (WP-C Cost 1, out of scope for run4).

- **F7** (`hexo_rl/eval/eval_pipeline.py::run_evaluation`): root cause traced past the report's
  own hypothesis — the per-opponent crash was NOT `k_cluster_mcts_bot.py` (that class is
  guarded upstream by `defender_dispatch.needs_no_drop_bot`, which never routes a GnnNet there
  for `gnn_axis_v1`'s `policy_pool="none"`); it was the SAME `inference.py:82` site as F8,
  reached via `ModelPlayer.__init__` → `LocalInferenceEngine.infer_batch`. Checked
  `gnn_integration_scope.md` §C5 first, as instructed: the in-loop promotion-gate arena for a
  graph candidate is explicitly out of scope (needs C3, "mixed-representation in-loop anchor",
  OQ-8 open) — so the fix is a round-level skip, not per-opponent repair. Added a
  representation guard before the opponent-dispatch loop: a graph candidate emits ONE
  structured `eval_round_skipped_graph_representation` warning (step, representation,
  `opponents_skipped=len(OPPONENTS)`, reason) and sets `results["eval_opponents_skipped"]`
  (new `EvalRoundResult` field) instead of silently 0-gaming through N per-opponent crash logs.
  Dense (grid) candidates take the unchanged path — byte-identical (verified: existing
  `tests/test_eval_pipeline.py` suite passes untouched).

- **F8** (`hexo_rl/selfplay/inference.py::LocalInferenceEngine`): `infer_batch` now branches on
  `model_representation` at `__init__` (hoisted, dense path byte-identical). The graph leg
  (`_infer_batch_graph`) reuses the WP-3 production graph inference seam —
  `InferenceBatcher.submit_graphs_and_wait` (native `build_axis_graph`, the SAME WP-1 seam
  guards) → a background `InferenceServer` graph loop (`collate_graph_batch` →
  `GnnNet.forward_batch` → segment-softmax → `assemble_ls_from_gnn_probs`) → the assembled
  `LegalSetPolicy.dense[362]` — never a hand-rolled graph encode. Off-window `overflow` mass is
  dropped, matching the EXISTING dense single-window `infer_batch`'s own off-window-drop
  contract (not a new approximation; the GNN is whole-board per OQ-6 so overflow is empty in
  practice anyway). A single `LocalInferenceEngine` instance owns one background
  `InferenceServer` thread for its lifetime; added `close()`/`__del__` for cleanup (mirrors
  `test_gnn_seam_smoke.py`'s try/finally). Sibling fix: `infer_batch_per_cluster` (the no-drop
  legal-set decode, which has no graph analogue per OQ-6) now raises a named `NotImplementedError`
  for a graph model instead of the same bare `AttributeError` two lines down.

  **Real circular-import bug found and fixed while verifying F8:** a top-level
  `from hexo_rl.model.build_net import model_representation` in `inference.py` closed a real
  cycle (`gnn_net.py` → `hexo_rl.bots.strix_v1_net` → `hexo_rl.bots.__init__` →
  `our_model_bot.py` → `selfplay/worker.py` → `selfplay/inference.py` → back to `build_net.py`
  → `gnn_net.py`, partially initialized). Fixed by deferring the import inside `__init__`
  (this codebase's established convention for exactly this class of cycle, matching
  `opponent_runners.py`'s own "lazy import → no import cycle" comment). Caught by actually
  running the code, not by the test suite (pytest's collection order happened not to trigger
  it) — confirms the value of the live proof-run beyond unit coverage.

**F6 — `configs/variants/run4_gnn_smoke.yaml` (pinned ruling, implemented).** Full duplicate of
`run4_gnn.yaml` (no variant-of-variant inheritance mechanism exists in this repo — confirmed by
reading `orchestrator.py`) plus labeled smoke overrides: `batch_size: 64` (production stays
implicit-256, header comment states the OQ-2 5080 rider is still open), `min_buffer_size: 64` +
`buffer_capacity: 4096` (minimal pool sizing for a 3-step gate), and — not explicitly asked but
required by §RUN3-STEP0 discipline once F5a exists — DISTINCT namespaced
`buffer_persist_path`/`best_model_path` from run4_gnn.yaml's own (a smoke run sharing either
file with a real production run4_gnn launch is exactly the collision class F5a exists to kill,
just at variant-vs-variant instead of run-vs-run granularity). Header comment matches the
requested wording. Launch-path test covers resolution-clean, forbidden-weights-zero,
corpus-mix-off, and both namespacing properties.

---

## `.in_channels` sweep census

`grep -rn "\.in_channels" hexo_rl/ scripts/ --include="*.py"` — **24 files found, 0 untriaged.**

| Classification | Count | Notes |
|---|---|---|
| **fixed** (this wave) | 2 | `hexo_rl/training/anchor.py` (F5b), `hexo_rl/selfplay/inference.py` (F7-root/F8) |
| **already-safe** | 4 | `orchestrator.py` (explicit graph short-circuit BEFORE the read, pre-existing WP-4 fix), `lifecycle.py` + `loop.py` (both read `InfModelArch.in_channels`, a dataclass field always declared for either representation, never a live model attribute), `early_game_probe.py` (reads `resolve_arch(...).ArchSpec.in_channels`, same class — confirmed live: the S7 rerun's own artifact list shows `early_game_probe_gnn_axis_v1_v1.npz` was minted successfully with zero crash) |
| **already-safe (upstream-guarded)** | 1 | `k_cluster_mcts_bot.py` — only ever constructed via `defender_dispatch.build_model_bot`, which dispatches on `policy_pool` (`gnn_axis_v1` is `"none"`, not `"legal_set_scatter_max"`) — a GnnNet never reaches this class in production today |
| **dense-only-scope** (out of scope, not reachable from run4 launch/eval path) | 15 | `v6_argmax_bot.py`, `analyze_api.py` (standalone dashboard dev-tool), `bench_v6w25_nn.py`, `global_token.py` (internal CNN-only submodule), `probe_threat_logits.py` (CNN plane-specific, no graph analogue — flagged as an open gap for a future graph launch, not fixed here), and 10 historical CNN-only investigation/probe scripts (`dvderisk_*`, `dpfit_*`, `diagnosis/overspread_d3_*`, `e1/validate_ckpt.py`, `headswap/model_heads.py`) |
| **definition / docstring-only** | 2 | `hexo_rl/model/network.py` (the attribute's own declaration site), `hexo_rl/model/build_net.py` + `hexo_rl/eval/eval_pipeline.py` (prose in this fix wave's own comments, not code) |

Full per-file classification + rationale lives in `tests/test_s7_in_channels_sweep_completeness.py`'s
`_CENSUS` dict — that test re-runs the grep and fails loud if a new, untriaged `.in_channels`
consumer site appears anywhere in the tree.

**One residual gap found live, NOT in the `.in_channels` sweep, NOT fixed (different bug class,
out of task scope):** `hexo_rl/monitoring/value_spread_canary.py` calls the model with a bare
dense tensor (same class as F2's `cuda_warmup`, i.e. "no representation guard before a bare
`forward()` call" — NOT the `.in_channels` class this task targets) and fails with
`Module [GnnNet] is missing the required "forward" function` on every graph train step. It is a
caught, warning-only canary (`value_spread_canary_failed`, non-fatal) — observed live while
minting the F8 proof-run checkpoint. Flagging for future triage, not fixing here (F2 is closed;
this is a second, independent site of F2's bug class, not F5b/F7/F8's).

---

## Tests added

| File | New tests | Purpose |
|---|---|---|
| `tests/training/test_anchor_graph_representation.py` | 4 | F5b: graph-vs-graph sync works + mutates weights; cross-representation raises `RepresentationMismatch` loud; graph net-scale mismatch skip-logs (no raise, no mutation); graph fresh-init (no `.in_channels` dependency) |
| `tests/test_eval_pipeline_graph_representation.py` | 3 | F7: graph candidate skips whole round + counter, no opponent calls; exactly ONE structured skip log per round; dense candidate byte-identical (existing opponents still run) |
| `tests/selfplay/test_gnn_local_inference_engine.py` | 7 | F8: no `AttributeError`, correct policy shape/finiteness; argmax is a legal move; empty-boards no-op; **`run_gumbel_on_board` end-to-end plays a legal searched move** (the live-proof criterion, mirrored as a fast CPU regression); `infer_batch_per_cluster` raises named `NotImplementedError` not bare `AttributeError`; graph server thread starts/stops/idempotent-close; dense engine unaffected (no graph server spun up) |
| `tests/model/test_build_net.py` (+3) | 3 | `model_representation` unit coverage: graph, grid, `torch.compile` `_orig_mod` unwrap |
| `tests/test_run4_gnn_launch_path.py` (+8) | 8 | F5a/F6: both variants declare namespaced `best_model_path`/`buffer_persist_path` (raw yaml + resolved-config level); the two variants' paths never collide; `run4_gnn_smoke.yaml` resolves clean with the labeled overrides (batch_size 64, min_buffer_size 64) while production `run4_gnn.yaml` stays at the base 256 |
| `tests/test_s7_in_channels_sweep_completeness.py` | 2 | Census pin: no untriaged `.in_channels` site; both `fixed`-class files still guarded |
| **Total new** | **27** | |

## Verification run

- `tests/training tests/selfplay tests/model tests/eval -q -m "not slow and not integration"`:
  **365 passed, 2 skipped** (pre-existing skips, unrelated).
- `tests/ -q -m "not slow and not integration" --ignore={training,selfplay,model,eval}` (every
  other top-level suite, incl. the launch-path + F3 + F7 + sweep-completeness tests above):
  **2296 passed, 134 skipped, 11 deselected, 1 xpassed** — 0 failures.
- `tests/selfplay/test_gnn_record_dispatch.py -m integration` (the e2e dispatch test named in
  the task): **2 passed**.
- `scripts/evalfair/tests/ -m "not slow and not integration"`: **60 passed, 3 deselected**
  (the 3 deselected are pre-existing `@pytest.mark.slow`/`@pytest.mark.integration` real-game
  tests in `test_worker_invariance.py`/`test_anchor.py`, unrelated to this task's change surface
  — zero files under `scripts/evalfair/` were touched. `test_anchor.py` alone runs >2 min;
  not exhaustively re-run given zero code-path overlap with this fix wave. The graph-specific
  coverage that DOES exercise `scripts/evalfair/core.py` — `tests/test_s7_f3_graph_eval_book_radius.py`
  — already passed in the top-level batch above, and the F8 live proof-run below exercised the
  real `scripts/evalfair/core.py::run_arm` end-to-end successfully).
- No `engine/src/` changes — `cargo test`/bench-gate not applicable.

## Live proof (F8)

Minted a diagnostic `gnn_axis_v1` checkpoint via the exact S7-agent recipe: fresh `GnnNet` →
BC-prefit warm-start (`checkpoints/probes/gnn_bc/gnn_bc_040000.pt`, 46/46 tensors verified) → 3
`Trainer.train_step`s on the WP-A frozen `wpa_positions.json` set → `save_checkpoint`. Ran the
REAL production entrypoint end-to-end:

```python
from scripts.eval.mantis_pull_eval import stage2_d5_eval
stage2_d5_eval(
    ckpt_path=<minted ckpt>, book_r4=".../evalfair_r4_v2.json", book_r5=".../evalfair_r5_v2.json",
    out_dir=<scratch>, workers=1, n_boot=20, expect_encoding="gnn_axis_v1", n_pairs=1,
)
```

Result: `graph_ckpt_evalfair_book_r5 radius=5` (F3 still correctly clear) → `run_arm` → deploy
head → `gumbel_search_py` → `LocalInferenceEngine.infer_batch` → **got PAST the F8 crash site,
played 2 games** (`{'wr': 0.0, 'n': 2, 'eff_n': 2, 'radius': 5, 'deploy_matched': True,
'bad_pairs': 0, 'censored_games': 0, ...}` — `wr=0.0` is expected/uninteresting for a 3-step
fresh-init-adjacent net against SealBot, not a defect). Confirms F8 end-to-end through the exact
path `mantis_pull_eval` stage-2 drives, not just the isolated unit tests.

## Artifacts (scratch, not committed)

`/tmp/claude-1000/.../scratchpad/`: `s7f8_mint_ckpt.py` (mint script) + `s7f8_ckpt/` (minted
checkpoint), `s7f8_stage2_proof.py` (live-proof driver) + `s7f8_stage2_out/result.json`,
`s7_f8_scratch.py` + `s7_f8_gumbel.py` (initial API-verification scratch scripts).

## Micro-fix (S-1/S-2/S-4)

Closes the three non-blocking findings from `S7_round2_review.md` (independent-reviewer pass,
HEAD `80f7c54` + round-2 uncommitted). All three are additive (new tests) or self-contained
try/finally wraps — no behavior change on the dense path, no new call-site plumbing required.

**S-1 [MEDIUM] smoke-yaml drift trap — closed.** Added
`test_run4_gnn_smoke_resolved_config_parity_with_production` to
`tests/test_run4_gnn_launch_path.py` (beside the rest of the launch-path suite). Resolves BOTH
`run4_gnn.yaml` and `run4_gnn_smoke.yaml` through the real `orchestrator.load_train_config` +
`flatten_config_and_resolve_encoding` entrypoint, dotted-flattens both `combined_config` dicts,
and asserts every key matches except an explicit `_SMOKE_ALLOWED_DIVERGENT_KEYS` allowlist
(`batch_size`, `min_buffer_size`, `buffer_capacity` — the three S7 F6 labeled overrides — plus
the two §RUN3-STEP0 namespaced paths). Ran it against the real on-disk files first without the
allowlist to confirm the diff set is EXACTLY those 5 keys (no silent extra divergence today);
then pinned the allowlist. A future prod edit to `draw_value`/`ply_cap_value`/`recency_weight`
or any `GRAPH_FORBIDDEN_NONZERO_WEIGHTS` key that isn't mirrored into the smoke file now fails
this test loudly instead of silently validating different training semantics.

**S-2 [LOW] graph InferenceServer thread cleanup relying on GC — closed.**
- `scripts/evalfair/core.py::_play_pair` (S7 F8's `eng = _build_engine_for_model(...)` call,
  ~line 415): wrapped the pair's play loop in `try/...: finally: eng.close()`. Matches the
  finding's explicit recommendation ("`try/finally: eng.close()` in `run_arm`" — the actual
  per-pair engine construction/use site is `_play_pair`, `run_arm`'s worker function; `run_arm`
  itself never builds an engine).
- `hexo_rl/eval/deploy_strength_eval.py::DeployStrengthEvaluator` (the `self._engine =
  _build_engine_for_model(...)` in `__init__` at ~429, and `_best_bot`'s own engine at ~455):
  added a `close()` method (`self._engine.close()`, idempotent, mirrors
  `LocalInferenceEngine.close`'s own docstring which already named this class as an intended
  caller). `run()` — the one place both engines (`self._engine` via `self._cand`, and
  `best_bot`'s via `_best_bot()`) are actually used, across every one of its exit paths (early
  `best_model is None` return, screen-reject early return, and the full confirm path) — now
  closes both in a `finally` (plus the dedicated call in the `best_model is None` branch, which
  returns before the `try` is entered). Single caller (`opponent_runners.py::_run_deploy_strength`
  constructs, calls `.run()` once, discards) already matches this lifetime; no caller-side change
  needed. Dense engines: `close()` is a no-op by construction (no graph server started), so
  `tests/test_deploy_strength_eval.py`'s dense-only fixtures are unaffected.

**S-4 [LOW] anchor cross-representation test coverage gap — closed.** Added
`test_resolve_anchor_reverse_cross_representation_raises_loud_not_attributeerror` to
`tests/training/test_anchor_graph_representation.py`, mirroring the existing
`test_resolve_anchor_cross_representation_raises_loud_not_attributeerror` with the roles
swapped: `anchor_model = GnnNet()` (graph anchor on disk) + `inf_model = HexTacToeNet(...)`
(grid candidate) — asserts `resolve_anchor` still raises `RepresentationMismatch`, never
`AttributeError`. The reviewer verified this direction live by hand; this closes the gap so a
future regression narrowing the guard to one direction (e.g. an `isinstance(inf_model,
GnnNet)`-only special case) fails the suite instead of passing clean.

**Verification (laptop, this worktree, HEAD `80f7c54` + round-2 + this micro-fix,
uncommitted):**

- `tests/test_run4_gnn_launch_path.py tests/training/test_anchor_graph_representation.py`
  (`-m "not slow and not integration"`): **18 passed** (16 pre-existing + 2 new: S-1's parity
  test, S-4's reverse-direction test).
- `tests/test_deploy_strength_eval.py` (`-m "not slow and not integration"`): **13 passed**
  (S-2 touch site's own suite, unaffected).
- `scripts/evalfair/tests` (`-m "not slow and not integration"`): **60 passed, 3 deselected**
  (matches round-2's own baseline — S-2's `core.py` touch site's suite, unaffected).
- `tests/{training,selfplay,model,eval}` (`-m "not slow and not integration"`): **366 passed, 2
  skipped, 18 deselected** (round-2 baseline 365 + 1 = S-4's new test; S-1's test lives directly
  under `tests/`, outside this glob).
- `tests/eval/test_turn_veto.py tests/selfplay/test_gnn_local_inference_engine.py
  tests/test_confres_player_factory.py tests/test_eval_opponent_runners.py
  tests/test_offwindow_eval_opponent.py` (every other `DeployStrengthEvaluator`/
  `deploy_strength` referencing suite, `-m "not slow and not integration"`): **42 passed, 1
  deselected**.
- `tests/selfplay/test_gnn_record_dispatch.py` (`-m integration`, e2e graph dispatch): **2
  passed**.
- Full `--collect-only -m "not slow and not integration"`: **2799/2828 collected** (round-2
  baseline 2797/2826 + 2 = the two new tests), 0 errors.
- Fresh `import hexo_rl.selfplay.inference`: clean (S-2 touches neither this module nor its
  circular-import fix).

Tree otherwise unchanged from round-2 HEAD; not `git add`ed per instructions.
