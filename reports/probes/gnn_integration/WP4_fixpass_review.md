# WP-4 review fix pass — fresh-eyes verification

**Reviewer:** independent fresh-eyes fix-pass reviewer (did not write the fix). Read-only on
source; live reproductions run from the worktree venv (`.venv/bin/python`, symlinked to master
checkout venv per worktree-setup convention). Inputs: `WP4_review.md` (findings, @ f30cc43),
`WP4_construction.md` §7 (implementer claims), uncommitted fix-pass diff (`wp4fix_diff.patch`,
17 files + report). Date: 2026-07-15.

## Verdict

**CLOSED.** All four fixes verified live (not just by reading the diff): the F1 landmine
scenario reproduced end-to-end through all three fresh-construction sites and builds `GnnNet`
dist65 with no raise; grid byte-identical passthrough holds (no `value_head_type` key injected);
the real banked BC artifact now reaches the actionable BC-prefit diagnostic; the compat.py
filename-beats-shape exploit is closed at both the direct and the `resolve_from_checkpoint`
level; reverse-F1 raises the named `RepresentationMismatch` in both grid builders while
malformed-grid keeps its original `KeyError`. Suite green on rerun (first run had one
statistical flake in a file untouched by the fix pass — adjudicated below, non-blocking).

Two cosmetic notes (N1 count-tally off-by-one in §7, N2 commingled concurrent-WP changes in the
worktree — a commit-hygiene note for the controller, not a defect in the fix pass). No REOPEN
findings.

---

## Fix 1 (F1 MUST-FIX) — representation-aware `value_head_type` default: VERIFIED

**Helper exists and is representation-aware.** `hexo_rl/model/build_net.py:55`
`resolve_value_head_type(spec, config)`: declared config value wins (`str()`-wrapped);
omitted/explicit-null → `"dist65"` when `getattr(spec, "representation", "grid") == "graph"`,
else `str(MODEL_HPARAM_DEFAULTS["value_head_type"])` (= `"scalar"`) — byte-identical to the
pre-WP-4 grid default. `spec=None` → grid → scalar (pinned by
`test_resolve_vht_none_spec_defaults_grid_scalar`).

**All FOUR merge points wired** (grep + read live source):

| Site | Wiring | Verified |
|---|---|---|
| `orchestrator.py:697-716` fresh branch | `_fresh_vht = resolve_value_head_type(_fresh_spec, combined_config)`; graph-only persist `combined_config["value_head_type"] = _fresh_vht` (line 707, guarded by `representation == "graph"`); grid config contents untouched | live repro below |
| `lifecycle.py:75` `build_inference_model` | `value_head_type = resolve_value_head_type(_spec, trainer.config)` — the old `str(trainer.config.get(..., MODEL_HPARAM_DEFAULTS[...]))` (incl. the `str(None)`→`"None"` trap) is GONE | live repro below |
| `anchor.py:408` `resolve_anchor` | signature default `"scalar"` → `None`; fresh-init branch (line 589-593) resolves via helper only when caller omitted; explicit caller value (loop.py:159 passes `_arch.value_head_type`) used as-is | live repro below |
| `trainer_ckpt_load.py:632` graph resume | inline `config.get("value_head_type", "dist65")` replaced by `resolve_value_head_type(resolved_spec, config)` — fourth copy killed | code read + 4 resume tests green |

**Live landmine reproduction** (the exact scenario the review reproduced pre-fix — graph config
that OMITS `value_head_type`, driven through all three fresh sites in one process):

```
init_trainer OK: GnnNet True
trainer.config value_head_type: dist65
build_inference_model OK: GnnNet arch.value_head_type= dist65
resolve_anchor OK: GnnNet
ALL THREE MERGE POINTS PASS — no RepresentationMismatch
```

(`init_trainer` log line: `new_run ... value_head_type=dist65`; `resolve_anchor` called with
`value_head_type` param omitted entirely → fresh-init built `GnnNet`.)

**Grid byte-identical claim verified live** — grid config omitting `value_head_type` through
`init_trainer`:

```
grid model: HexTacToeNet value_head_type= scalar
value_head_type in grid_config after init_trainer: False
value_head_type in trainer.config: False
```

No key injected into a grid `combined_config` — the graph-only-persist guard holds.

---

## Fix 2 (F2 HIGH) — unknown STAMP falls through to strict detection: VERIFIED

`checkpoint_loader.py:472-499`: on `EncodingRegistryError` from `_registry_lookup(label)`, the
fall-through fires **only** when `declared_name is None and override_name is None and
label == ckpt_stamp_name` — i.e. exactly when the unknown label came from the checkpoint's own
stamp. (Chain check: `stamped_or_declared = override_name or ckpt_stamp_name or declared_name`,
and a no-source case already went through `detect_encoding_label`, which only returns registered
names — so the condition is airtight.) Declared/override unknown names keep the loud generic
raise.

**Real-artifact repro** (`checkpoints/probes/gnn_bc/gnn_bc_040000.pt` exists, top keys
`['arm','lr','steps','n_params','model_state_dict','encoding']`, stamp `strix_axis_graph`):

```
[warning] unknown_ckpt_encoding_stamp_falling_through_to_detection
          checkpoint=checkpoints/probes/gnn_bc/gnn_bc_040000.pt stamp=strix_axis_graph
          stamp_source="checkpoint ... raw['encoding']"
RAISED ValueError: gnn checkpoint: graph state dict has representation.*/policy_head.* keys but
NO value_head.fc2_bins.weight — ... BC-prefit-only ... The BC-prefit warm-start path is
`hexo_rl.model.gnn_net.load_representation_policy_from_bc` ...
```

Log event present, reaches `assert_full_gnn_checkpoint_or_raise`'s actionable diagnostic,
message enriched to name the warm-start entry point. **Declared unknown name stays loud**:

```
declared_encoding="strix_axis_graph" -> RAISED ValueError: checkpoint ...: unknown encoding
label 'strix_axis_graph'
```

Tests: `test_unknown_stamp_label_falls_through_to_detection_full_gnn` (synthetic full-GnnNet
round-trip, asserts label==`gnn_axis_v1` + isinstance GnnNet),
`test_real_banked_bc_prefit_reaches_named_diagnostic` (skipif-guarded on the real artifact,
asserts `load_representation_policy_from_bc` in the message — RAN, not skipped, artifact
present), `test_unknown_declared_encoding_still_raises`. All green.

---

## Fix 3 (F3 HIGH) — compat.py graph marker beats filename; marker single-sourced: VERIFIED

`compat.py:93`: `if _GNN_GRAPH_MARKER_KEY in state_dict: return "gnn_axis_v1"` sits BEFORE
`_filename_match` — state-dict shape is ground truth, filename is a hint.

**Single-sourcing check** (`grep -rn "representation.input_proj.weight\|GNN_GRAPH_MARKER_KEY"
hexo_rl/`): canonical definition only in `hexo_rl/encoding/_probes.py:32`; consumers import it —
`compat.py:27/93`, `resolvers.py:19/469` (the WP-4 detect branch now uses the import),
`training/checkpoints.py:89/123/175` (`infer_gnn_hparams_from_state_dict` +
`assert_full_gnn_checkpoint_or_raise`), `eval/checkpoint_loader.py:656/659/663` (the new guard).
Remaining literal occurrences are comments/docstrings only. No inlined copies.

**Review exploit reproduced live** (bare `GnnNet` state_dict, no metadata):

```
compat.infer_encoding_from_state_dict(sd, "checkpoints/gnn_v6_checkpoint_bare.pt") -> gnn_axis_v1
resolve_from_checkpoint(<bare weights saved as gnn_v6_checkpoint_bare.pt>) -> gnn_axis_v1
```

Pre-fix this silently resolved `v6`. Also verified through the live higher-level
`resolve_from_checkpoint` path the review named (DeprecationWarning steer still fires,
resolution correct). Inverse direction pinned by
`test_compat_grid_sd_with_gnn_in_filename_still_resolves_grid` (grid sd + 'gnn' in path →
v6w25 by shape) — no regression.

---

## Fix 4 (F5 LOW + F4 doc) — reverse-F1 named raise + landed-verify scope lines: VERIFIED

`checkpoint_loader.py:646` `_reject_graph_shaped_state_for_grid_spec(state, spec, missing_key)`
— raises `RepresentationMismatch` only when `missing_key not in state AND
GNN_GRAPH_MARKER_KEY in state`. Called at the top of BOTH `_build_min_max_model` (line 671,
key `trunk.input_conv.weight`) and `_build_kata_model` (line 763, canvas-aware `inp_w_key`).

Live:

```
declared v6 on bare GNN sd -> RepresentationMismatch: encoding 'v6' (representation='grid')
  declared for a GRAPH-shaped state dict — no 'trunk.input_conv.weight', but the GNN marker ...
declared v8 on bare GNN sd -> RepresentationMismatch: encoding 'v8' ... GRAPH-shaped ...
malformed grid sd (no marker) declared v6 -> KeyError: 'trunk.input_conv.weight'  (original
  behavior preserved)
```

F4 doc lines present: `gnn_net.py:272` ("Scope (by design): the landed-verify proves the LOAD
landed what the file said — it does NOT ... validate that the file's own contents are sane")
and `checkpoint_loader.py:629` (same scope note at `_build_gnn_model`'s landed-verify loop).
Docs-only as claimed.

---

## Regression-test quality

18 new test functions (see N1 — §7 claims 19; actual `def test_` count in the diff is 18):
6 helper-unit (`tests/model/test_build_net.py`, incl. explicit-null→dist65, declared-wins both
directions, `None`-spec→scalar, and the full launch-config-omits-key build), 4 site-level F1
(orchestrator graph-omits + grid-omits-stays-scalar-and-no-key-injected, anchor
param-AND-config-omitted, lifecycle omits→arch carries dist65→build_eval_model rides it),
3 loader F2, 3 compat F3, 2 named-raise F5. Two existing tests extended with assertions
(`test_resolve_anchor_fresh_init_empty_config_still_grid` pins scalar;
`_make_gnn_trainer` parameterized).

**Would the demanded test have caught the original landmine?** Yes — spot-checked by reading
the fixtures against the pre-fix code (not by reverting):
`test_init_trainer_fresh_run_graph_without_value_head_type` deletes the key from
`_graph_config()` and drives real `init_trainer`; pre-fix that path computed
`combined_config.get("value_head_type", _MHPD["value_head_type"])` = `"scalar"` →
`_build_gnn_net` raises `RepresentationMismatch` → test errors. Same logic holds for the
lifecycle test (pre-fix `str(config.get(...))` = `"scalar"`) and the anchor test (pre-fix
signature default `"scalar"`). All three assert `isinstance(..., GnnNet)` + dist65 bins, and
the orchestrator one additionally pins `trainer.config["value_head_type"] == "dist65"`
(persist) — behavior asserts, not implementation mirrors. The grid-side tests pin the
no-key-injected byte-identical claim (`assert "value_head_type" not in trainer.config`).

---

## Suite results

**Targeted (9 WP-4 files):**

```
tests/model/test_build_net.py tests/encoding/test_detect_encoding_from_state_dict.py
tests/test_checkpoint_loader_gnn.py tests/training/test_trainer_ckpt_load_gnn_resume.py
tests/training/test_lifecycle_gnn_build.py tests/training/test_anchor_branches.py
tests/test_orchestrator_gnn_build.py tests/test_encoding_registry.py
tests/scripts/test_t10_manifest.py
-> 92 passed, 5 warnings in 4.03s
```

(t10_manifest included per §7's claim that the graph-marker fix kills that flake class at the
root for graph sds — green.)

**Full suite** (`pytest -q -m "not slow and not integration"`), run 1:

```
1 failed, 2556 passed, 134 skipped, 15 deselected, 1 xpassed in 251.27s
FAILED tests/test_game_length_weight_schedule.py::test_sampling_distribution_matches_weight_schedule
```

Count reconciliation vs the review baseline (2533 passed): +18 fix-pass tests +6 tests from the
untracked concurrent-WP file `tests/training/test_gnn_hexg_buffer.py` (collected by testpaths)
= 2557 ran = 2556 + 1. Skip count unchanged at 134 (the banked-BC skipif test RAN — artifact
present).

**Flake adjudication:** the one failure is a χ² goodness-of-fit test over an UNSEEDED Rust RNG
(`sample_batch`); the test's own comment documents the spurious-failure mode and commit
`7f16cb6` already relaxed α 0.01→0.001 for exactly this. File untouched by the fix pass (not in
the diff); test passed 5/5 on immediate re-runs; full-suite rerun (run 2) green:

```
2557 passed, 134 skipped, 15 deselected, 1 xpassed, 69 warnings in 235.13s (0:03:55)   (0 failed)
```

Not one of the two pre-adjudicated flakes (kraken determinism, validate_ckpt float-tol) — noted
per contract, classified statistical-flake, non-blocking on the rerun evidence. Suggest the
dispatcher add it to the adjudicated-flake list (third occurrence class; α relaxation history
shows it has fired spuriously before).

---

## Scope check

Fix-pass diff (`wp4fix_diff.patch`) touches exactly 17 files = the §7 tally: 11 source
(`build_net.py`, `gnn_net.py`, `encoding/{_probes,compat,resolvers}.py`,
`training/{orchestrator,lifecycle,anchor,trainer_ckpt_load,checkpoints}.py`,
`eval/checkpoint_loader.py`) + 6 test files + `WP4_construction.md`. Nothing extra in the
fix-pass diff itself.

**N2 (commit-hygiene note for the controller, not a fix-pass defect):** the WORKTREE carries
additional uncommitted changes outside the fix pass — `hexo_rl/training/losses.py`
(`ragged_policy_ce`, docstring says WP-5 §6.1), `engine/src/**` (5 files) + untracked
`engine/src/replay_buffer/hexg/` + `tests/training/test_gnn_hexg_buffer.py` (the concurrent
replay-buffer WP, explicitly out of this review's scope, not inspected). The WP-4 fix commit
must be staged file-selectively; a blanket `git add -A`/`commit -a` would sweep concurrent-WP
work into it.

## Findings

| # | Severity | Finding |
|---|---|---|
| N1 | COSMETIC | §7 tally arithmetic claims 19 new tests; actual new `def test_` count in the diff is 18 (their bucket math 6+5+3+3+2 double-counts one; buckets themselves are all present). No coverage gap — the review-demanded cases all exist. |
| N2 | NOTE | Worktree commingles concurrent-WP uncommitted work (`losses.py` WP-5, `engine/**` + hexg tests) with the fix pass — controller must stage the 17 fix-pass files + this report's sibling selectively. |
| F1..F5 | — | All review findings verified FIXED (see per-fix sections). No new defects found. |

## One-line summary

**CLOSED** — all four fixes verified by live reproduction of each original exploit/landmine;
targeted 92/92 green; full suite green on rerun (2557 passed, 0 failed; run-1 single failure
adjudicated as a documented pre-existing statistical flake in a non-WP-4 file, 5/5 green on
re-run).
