# WP-4 — registry-driven model construction + loader family (C4 subset)

**Status: DONE.** Kills the three construction-family SILENT-CORRUPT sites (contract audit
rows 11a-c) via one `build_net(spec, config, **kwargs)` authority, and converts the two
loader-family rows (11d-e) from a confusing deep-strict-load crash / a hard raise into a
correct RAGGED-OK graph path. Full suite green (details below). Worktree
`worktree-gnn-integration`; no `engine/**`, `game_runner/**`, `inference_bridge.rs`, or
`inference_server.py` touched (respected the concurrent-agent file lock).

**Path correction (verify-against-source caveat in the dispatch prompt fired):** the dispatch
named the shared detector `hexo_rl/training/resolvers.py` — that file does not exist. The
actual single-sourced `detect_encoding_from_state_dict` (imported identically by both
`checkpoint_loader.py` and `trainer_ckpt_load.py`) lives at `hexo_rl/encoding/resolvers.py`.
Edited there instead — it is the literal function the C4/C7 single-source finding names, just
under its real path. This is the one file I touched outside the constraint's literal
`hexo_rl/{model,training,eval}/` list; flagging it explicitly since the general constraint and
the task's own explicit item 2 disagreed on the path, and the task's explicit function name won.

---

## 1. Construction-family sites killed (contract rows 11a-c)

New authority: `hexo_rl/model/build_net.py` — `build_net(spec, config, **kwargs)` dispatches on
`spec.representation`. `"grid"` forwards `**kwargs` byte-identically to
`HexTacToeNet(**kwargs)`; `"graph"` builds `GnnNet` from `spec.node_feat_dim`/`edge_feat_dim`
+ `config`'s `gnn_hidden`/`gnn_num_layers`/`gnn_policy_hidden`/`gnn_value_hidden` (default to
the probe-284k class that carries the +414 evidence) + `n_value_bins`; validates
`value_head_type` is `"dist65"` or absent (`GnnNet` ships only `GnnDist65ValueHead`). Any other
representation, or a graph spec missing its schema-v4 geometry, raises `RepresentationMismatch`
(`ValueError` subclass, message prefixed `"RepresentationMismatch: "` — mirrors the Rust seam's
`inference_bridge.rs:492-497` convention).

| # | Site (before) | file:line (verified against current source) | After |
|---|---|---|---|
| 11a | `orchestrator.py` fresh-run `HexTacToeNet(...)` | was `:677`, now the `build_net` call sits at `:695` (mission's `~677` drifted ~18 lines from intervening commits) | `build_net(_fresh_spec, combined_config, **same kwargs)`; `_fresh_spec = _registry_resolve(combined_config)` |
| 11b | `lifecycle.py` inf_model + eval_model | was `:66`/`:172`, now `:87` (`build_inference_model`) / `:202` (`build_eval_model`) | Both dispatch through `build_net`. `InfModelArch` gained `spec: Any = None` + `gnn_hparams: dict` fields so `build_eval_model` (which only sees `arch`, not the full trainer config) reconstructs the IDENTICAL architecture `build_inference_model` built |
| 11c | `anchor.py` in-loop anchor/eval `best_model` | was `:569`, now `:577` (fresh-init fallback branch) | `_anchor_spec = resolve_from_config(config)`; `build_net(_anchor_spec, config, **same kwargs)` |

**Blast-radius fix required beyond the 3 named sites:** `orchestrator.py::_resolve_fresh_in_channels`
(a helper `init_trainer`'s fresh branch calls **before** reaching the `build_net` call) crashed
on a graph encoding — `resolve_arch()` unconditionally indexes `kept_plane_indices` (empty `[]`
for `gnn_axis_v1`) to find `cur_stone_slot`/`opp_stone_slot`, raising `ValueError: x not in list`
before `build_net` ever ran. Fixed with a representation check that short-circuits to an inert
`(0, None)` placeholder for `representation="graph"` (in_channels/input_channels are grid-only
concepts `build_net`'s graph branch never reads). Found via the new
`tests/test_orchestrator_gnn_build.py` fresh-run test — first run failed here, confirming this
was a real gap the audit's "3 named sites" framing didn't anticipate, not a hypothetical.

**Grid regression proof:** `tests/model/test_build_net.py::test_grid_dispatch_matches_direct_construction_{v6,dist65}`
pin `build_net(spec, {}, **kwargs)` and `HexTacToeNet(**kwargs)` to identical `state_dict()`
keys+shapes. `test_none_spec_defaults_to_grid` pins that every pre-WP-4 caller/test that never
threads a `spec` (`spec=None`) still gets a `HexTacToeNet` — this is load-bearing for
`InfModelArch(spec=None)`'s default and every existing test that constructs it directly.

---

## 2. Loader-family branches landed (contract rows 11d-e)

### 11d — `hexo_rl/encoding/resolvers.py::detect_encoding_from_state_dict`

Added a graph-detect branch **before** the grid `trunk.input_conv(.conv)?.weight` probe: a
`GnnNet` state dict has `representation.input_proj.weight` (no grid marker at all) → returns
`lookup("gnn_axis_v1")` unconditionally (only one graph encoding registered today, so the match
is unambiguous — noted as a future disambiguator gap if a second graph encoding ever lands).
Must run first in **both** `strict` modes: the grid branches would otherwise raise
("cannot detect encoding", strict=True) or silently return `None` (strict=False) — neither
correct for a graph state dict.

Single-sourced consumers, both fixed by this one change:
- `hexo_rl/eval/checkpoint_loader.py::detect_encoding_label` (strict=True call site, C4).
- `hexo_rl/training/trainer_ckpt_load.py::_detect_encoding_from_state_dict` (strict=False call
  site, C7) → `_resolve_checkpoint_encoding` now correctly resolves a bare weights-only GNN
  checkpoint to `gnn_axis_v1` instead of silently falling through to grid hparam inference.

### 11e — `hexo_rl/eval/checkpoint_loader.py`

- `load_model_with_encoding`: new `label == "gnn_axis_v1"` branch → `spec = _registry_lookup("gnn_axis_v1")`.
- `_build_model_from_spec`: now checks `spec.representation == "graph"` **before** the
  `has_pass_slot` branch — `gnn_axis_v1` has `has_pass_slot=True` too (registry: unchanged action
  space), so a naive has_pass_slot-only dispatch would have mis-routed a GNN state dict into
  `_build_min_max_model`, which reads `trunk.input_conv.weight` (absent → `KeyError`).
- New `_build_gnn_model(state, spec)`: ground-truths `hidden`/`num_layers`/`policy_hidden`/
  `value_hidden` from the checkpoint's own tensor shapes (`infer_gnn_hparams_from_state_dict`,
  new in `hexo_rl/training/checkpoints.py`, single-sourced with the C7 resume branch — not
  duplicated), cross-checks `node_feat_dim`/`edge_feat_dim` against the registry spec
  (`validate_arch_against_spec`-style guard), `strict=True` loads, then the **landed-verify**
  (C7 red-team demand): `torch.allclose` over every `representation.*`/`policy_head.*`/
  `value_head.*` tensor — representation+policy coverage, not value-only (mirrors
  `_build_min_max_model`'s existing E1 dist65-only allclose guard, extended to all three
  submodules since `GnnNet` has no legacy `strict=False` tolerance to hide a partial load behind).

### 11d (C7 resume path) — `hexo_rl/training/trainer_ckpt_load.py::load_checkpoint`

Restructured the single `HexTacToeNet(...)` construction into an `if representation == "graph"`
branch (new) / `else` (existing grid body, pure-moved one indent level, byte-identical). Graph
branch: `infer_gnn_hparams_from_state_dict(model_state)` ground-truths hparams from the
checkpoint, persists the `gnn_*`-prefixed keys into `config` (→ `trainer.config`) so a later
`lifecycle.build_inference_model`/`build_eval_model`/`anchor.resolve_anchor` rebuild reconstructs
the **identical** architecture (not `build_net`'s probe-284k defaults, which could silently drift
from what was actually trained), then `build_net(resolved_spec, config, value_head_type=...,
n_value_bins=...)`. `resolved_spec is None` (non-canonical test-fixture shape) still falls
through to the pre-WP-4 grid `_resolve_model_hparams` path byte-for-byte.

### New shared helper: `assert_full_gnn_checkpoint_or_raise` (`checkpoints.py`)

Single-sourced named-raise for the one case both loaders can't support: a **BC-prefit-only**
state dict (`representation.*`/`policy_head.*` present, no `value_head.fc2_bins.weight` —
e.g. `gnn_bc_040000.pt`). The BC warm-start *transfer* (representation+policy onto a fresh value
head) is a separate, not-yet-built loader (`gnn_integration_scope.md` §C7 lead-time table: "GNN
warm-start loader (new)") — explicitly out of WP-4 scope. Without this check a bare
`strict=True` load would eventually raise on the missing keys anyway; this gives the identical
failure a clear, named diagnosis instead of a generic missing-key dump. Called at the top of
both `_build_gnn_model` (C4) and the trainer resume graph branch (C7) — same message either side.

---

## 3. Loader branch behavior table

| Input state dict | `detect_encoding_from_state_dict` | `checkpoint_loader.load_model_with_encoding` | `trainer_ckpt_load.load_checkpoint` (resume) |
|---|---|---|---|
| Grid (v6/v6w25/v6tp/v6_live2/v8/…) | unchanged (grid branches, byte-identical) | unchanged — `_build_min_max_model`/`_build_kata_model` | unchanged — `_resolve_model_hparams` + `HexTacToeNet` |
| Full production `GnnNet` (representation+policy_head+`GnnDist65ValueHead`) | → `gnn_axis_v1` (both strict modes) | → `GnnNet`, hparams ground-truthed from tensor shapes, landed-verified | → `GnnNet`, hparams ground-truthed + persisted into `trainer.config` |
| BC-prefit-only (`representation.*`/`policy_head.*`, no `value_head.fc2_bins.*`) | → `gnn_axis_v1` (detection still succeeds — shape marker present) | **named raise**: `assert_full_gnn_checkpoint_or_raise` — "looks like a BC-prefit-only state dict … not a full production GnnNet checkpoint" | same named raise (single-sourced) |
| Garbage / truncated graph sd (has `representation.input_proj.weight` but corrupt/mismatched shapes elsewhere) | → `gnn_axis_v1` (marker-key-only detection) | `strict=True` load raises `RuntimeError` (missing/unexpected/shape-mismatched keys) — loud, not silent | same |
| Non-canonical shape (neither grid nor graph marker, e.g. tiny test fixture) | `None` (lenient) / raise (strict) — unchanged | unchanged fallback chain | unchanged fallback to `_resolve_model_hparams` |

---

## 4. Tests

New/extended test files, all under the top-level `tests/` collection root (`pytest.ini`
`testpaths = tests`, `Makefile:test.py` — confirmed `hexo_rl/*/tests/*` files, e.g. the existing
`hexo_rl/training/tests/test_lifecycle_dist_head.py`, are **not** part of the standard gate's
collection; every new test lands under `tests/` instead so it counts):

- `tests/model/test_build_net.py` (9 tests) — grid byte-identical dispatch (2), `spec=None`
  defaults to grid (1), graph dispatch + config-hparam overrides (2), omitted `value_head_type`
  defaults cleanly (1), `RepresentationMismatch` on scalar-vs-graph / missing geometry / unknown
  representation (3).
- `tests/encoding/test_detect_encoding_from_state_dict.py` (+3 tests) — graph-detect by marker
  key, strict-mode-doesn't-hit-grid-raise, sanity that the fixture lacks the grid marker.
- `tests/test_checkpoint_loader_gnn.py` (5 tests) — weights-only round-trip via shape detection,
  full-ckpt-with-metadata round-trip, byte-exact landed-verify over every tensor, BC-shaped named
  raise, grid v6 round-trip unaffected.
- `tests/training/test_trainer_ckpt_load_gnn_resume.py` (4 tests) — weights-only resume builds
  `GnnNet` with ground-truthed dims + persists `gnn_*` into `trainer.config`, full-checkpoint
  round-trip (save→resume, byte-exact weights, step preserved), BC-shaped named raise via the
  live `Trainer.load_checkpoint` entrypoint, grid resume unaffected.
- `tests/training/test_lifecycle_gnn_build.py` (3 tests) — `build_inference_model` builds
  `GnnNet` + clean strict load, `build_eval_model` reconstructs an identical-shape architecture
  from `arch` alone, grid trainer's `arch.gnn_hparams == {}` (pure no-op for non-graph runs).
- `tests/training/test_anchor_branches.py` (+2 tests) — graph fresh-init builds `GnnNet` with the
  config's `gnn_*` hparams, `config={}` (every pre-WP-4 caller) still resolves to grid — no
  regression from threading `config` into `build_net`.
- `tests/test_orchestrator_gnn_build.py` (2 tests) — fresh-run graph build (this is the test that
  caught the `_resolve_fresh_in_channels` gap above), fresh-run grid unaffected.

**Total: 28 new tests across 7 files** (5 new files, 2 extended-in-place), all green. No
production run4 variant yaml created (launch scope, per instructions) — tests build inline
minimal configs.

---

## 5. Suite result

```
.venv/bin/python -m pytest -q -m "not slow and not integration"
2533 passed, 134 skipped, 14 deselected, 1 xpassed, 69 warnings in 323.75s (0:05:23)
```

Zero collection errors, zero failures, exit code 0. The `1 xpassed` and skip/deselect counts are
pre-existing (unrelated to this change — spot-checked, no WP-4 test is skipped/xfailed).

---

## 6. Deferred / explicitly out of scope

- **BC-prefit warm-start transfer loader.** Loading `gnn_bc_040000.pt`-shaped checkpoints
  (representation+policy_head onto a *fresh* value head) is a separate C7 task per the scope
  doc's lead-time table ("GNN warm-start loader (new)", 3-6 pd). WP-4's loaders correctly refuse
  it with a named, diagnostic raise (`assert_full_gnn_checkpoint_or_raise`) rather than silently
  mishandling it — this is the "named raise where the path can't support graph yet" case the
  mission's task 3 anticipated.
- **`resolvers.py` graph-detect is single-encoding.** Only `gnn_axis_v1` is registered; the
  branch returns it unconditionally on the marker key. A second graph encoding would need a real
  disambiguator (label hint or explicit declaration), noted inline in the code comment — mirrors
  the existing v6_live2-vs-v6_live2_ls precedent in the same function.
- **No `engine/**`, `inference_bridge.rs`, `inference_server.py`, `graph_collate.py` touched** —
  confirmed via `git status` throughout; those are the concurrent agent's files (WP-3 seam work,
  already landed dormant at `871ea41`/`fc9d3bb`). This report's scope is Python-side
  construction/loading only, as directed.
