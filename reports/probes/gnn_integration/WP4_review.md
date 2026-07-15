# WP-4 review + red-team — build_net authority + loader graph branches

**Reviewer:** fresh-eyes WP-4 REVIEW+RED-TEAM agent, one combined pass. Read-only on source;
scratch under `/home/timmy/.claude/jobs/7d6e8877/tmp/`. Target: commit `895d786`. Cross-referenced
`docs/designs/gnn_ragged_contract_v1.md` rows 11a-e, `docs/designs/gnn_integration_scope.md`
§C4/§C7, `reports/probes/gnn_integration/WP4_construction.md` (author claims), and re-ran the
full suite + 48 targeted tests + a battery of manual adversarial probes not in the test suite.

## Verdict

**REVIEW: FIXES-REQUIRED.** **Red-team: GAPS-FOUND** (3 concrete gaps beyond
`WP4_construction.md`'s claims — one MUST-FIX launch blocker, two real-but-adjacent silent/loud
diagnosability holes). Grid byte-identical passthrough (mission priority 1's core question) is
CLEAN. `detect_encoding_from_state_dict`'s own filename-vs-shape ordering (priority 5) is CLEAN.
Suite health confirmed: full re-run `2533 passed, 134 skipped, 15 deselected, 1 xpassed, 0
failed`, exit 0 (matches the construction report's claim); the 48 WP-4-specific tests all pass in
isolation too.

---

## Finding 1 — MUST-FIX: `value_head_type` default landmine blocks every graph launch that
doesn't explicitly declare `value_head_type: dist65`

**Severity: MUST-FIX (launch-blocking for run4-GNN).** Loud, not silent — but undocumented and
self-contradicting.

All three "killed" construction sites resolve `value_head_type` via
`config.get("value_head_type", MODEL_HPARAM_DEFAULTS["value_head_type"])`, and
`MODEL_HPARAM_DEFAULTS["value_head_type"] == "scalar"` (`hexo_rl/training/model_defaults.py:35`).
That means `build_net`'s graph branch NEVER actually sees `None` ("omitted") from any production
caller — it always receives a concrete `"scalar"` unless the launch config explicitly sets
`value_head_type: dist65`. `build_net._build_gnn_net` (`hexo_rl/model/build_net.py:118-124`) then
raises `RepresentationMismatch` with the advice `"Declare value_head_type: dist65 (or omit it)"` —
but "omit it" is unreachable from any of the three real call sites:

- `hexo_rl/training/orchestrator.py:703` (fresh-run) — `combined_config.get("value_head_type", _MHPD["value_head_type"])`.
- `hexo_rl/training/lifecycle.py:68` (`build_inference_model`) — `str(trainer.config.get("value_head_type", MODEL_HPARAM_DEFAULTS["value_head_type"]))` (the `str()` wrap even breaks the `value_head_type: null` workaround, turning `None` into the literal string `"None"`).
- `hexo_rl/training/anchor.py:408`'s `resolve_anchor(value_head_type: str = "scalar", ...)` default, fed transitively from `_arch.value_head_type` at the one real caller, `hexo_rl/training/loop.py:159`.

Reproduced live end-to-end (`init_trainer` with a graph config that omits `value_head_type`):

```
RepresentationMismatch: build_net: representation='graph' (encoding 'gnn_axis_v1') only ships
GnnDist65ValueHead; got value_head_type='scalar'. Declare value_head_type: dist65 (or omit it)
for a graph encoding.
```

**The fix pattern already exists in this same commit** — `trainer_ckpt_load.py`'s C7 resume
branch (`hexo_rl/training/trainer_ckpt_load.py:629`) correctly computes
`config["value_head_type"] = config.get("value_head_type", "dist65")` (representation-aware
default), so a *resumed* graph run never hits this. Only the three *fresh*-construction sites
lack the same treatment. **Fix:** in `orchestrator.py`'s fresh branch and
`lifecycle.py::build_inference_model`, resolve the default from the spec's representation before
the `.get()` call (e.g. `_default_vht = "dist65" if _fresh_spec.representation == "graph" else
_MHPD["value_head_type"]`) instead of unconditionally defaulting to `"scalar"`; thread the same
fix into `anchor.py`'s `resolve_anchor` default or (simpler) rely on `_arch.value_head_type`
already being correct once `lifecycle.py` is fixed.

**Why the 28 new tests never caught this:** every graph config fixture across all 5 new test
files (`test_orchestrator_gnn_build.py::_graph_config`, `test_lifecycle_gnn_build.py::_make_gnn_trainer`,
`test_anchor_branches.py::test_resolve_anchor_fresh_init_graph_builds_gnn_net`,
`test_trainer_ckpt_load_gnn_resume.py::_base_gnn_cfg`) explicitly sets
`"value_head_type": "dist65"` — the author clearly knew the net needs it, but never wrote a test
for the launch config that *doesn't* declare it, which is the realistic operator mistake this
landmine punishes. Recommend adding exactly that regression test once fixed.

---

## Finding 2 — HIGH: BC-prefit named raise (mission priority 4) is NOT reliably reached via the
eval loader on the real banked artifact

**Severity: HIGH** — directly falsifies a specific `WP4_construction.md` claim on the exact file
the mission named. Not silent, but not the intended diagnostic either.

Ran `load_model_with_encoding` against the real `checkpoints/probes/gnn_bc/gnn_bc_040000.pt`
(not a synthetic fixture):

```
RAISED: ValueError
checkpoint checkpoints/probes/gnn_bc/gnn_bc_040000.pt: unknown encoding label 'strix_axis_graph'
```

Root cause: the checkpoint carries a top-level `encoding: "strix_axis_graph"` stamp, written by
`hexo_rl/probes/gnn_bc/train_bc.py:409` (`"encoding": "v6_live2_ls" if arm == "cnn" else
"strix_axis_graph"`) — a label that was **never registered** anywhere in
`engine/src/encoding/registry.toml` (the registry name is `gnn_axis_v1`). The eval-side stamp
gate (`_resolve_ckpt_stamped_encoding`, pre-existing D-EVALGATE machinery, unmodified by WP-4)
treats a top-level `raw['encoding']` stamp as authoritative and NEVER overridden by shape
inference — so `_registry_lookup("strix_axis_graph")` fails before `_build_gnn_model` /
`assert_full_gnn_checkpoint_or_raise` is ever reached. Tried all three natural access patterns:

| Call | Result |
|---|---|
| `load_model_with_encoding(p, device)` (plain) | `ValueError: unknown encoding label 'strix_axis_graph'` — generic, no mention of BC-prefit/warm-start |
| `declared_encoding="gnn_axis_v1"` | `DeclaredEncodingMismatchError` (declared vs. stamp disagree) — also not the BC-prefit message |
| `decode_override="gnn_axis_v1"` | reaches the correct, actionable `assert_full_gnn_checkpoint_or_raise` message |
| `Trainer.load_checkpoint(p, fallback_config=...)` (C7 resume) | reaches the correct message **directly, no override needed** |

The C7 resume path only reads `ckpt['config']['encoding']` (absent on this artifact — its top
keys are `['arm','lr','steps','n_params','model_state_dict','encoding']`, no `'config'` key), so
it falls through to shape detection → WP-4's graph branch → correct spec → the intended raise
fires cleanly. The C4 eval loader's stamp-priority gate short-circuits before that ever happens.
`WP4_construction.md`'s claim ("Called at the top of both `_build_gnn_model` (C4) and the trainer
resume graph branch (C7) — same message either side") is **true only when there is no
unregistered top-level stamp** — it does not hold for the one artifact the mission explicitly
named, and none of the 28 tests exercise the real file (synthetic fixtures use either no stamp or
a matching `metadata['encoding_name']='gnn_axis_v1'` — never the probe pipeline's actual
`"strix_axis_graph"` label).

**Root cause is outside WP-4's file scope** (`train_bc.py` predates and is untouched by this
diff). Recommended fix (follow-up, not WP-4 itself): re-stamp `train_bc.py:409` to emit the
registry name `"gnn_axis_v1"` going forward, and re-stamp/migrate already-produced probe
checkpoints (including the banked `gnn_bc_040000.pt`) the same way the A5 migration script
handles legacy grid checkpoints.

---

## Finding 3 — HIGH (pre-existing, still open, same failure class the whole contract targets):
`resolve_from_checkpoint` / `resolve_encoding_for_eval` silently mis-detects a GNN checkpoint as
grid via filename substring

**Severity: HIGH.** Not a WP-4 regression (file untouched by this diff), but a live, exploitable,
still-open SILENT-CORRUPT hole in the loader family that the contract's audit table (rows 11a-e,
`gnn_ragged_contract_v1.md`) never named, and that directly contradicts the "zero SILENT-CORRUPT"
design target end-to-end.

`hexo_rl/encoding/resolvers.py::resolve_from_checkpoint` (used by
`resolve_encoding_for_eval`, `pretrain_cli.py`, `scripts/smoke_selfplay_bootstrap.py`,
`scripts/diagnosis/selfplay_fixture_gen.py`, `scripts/diagnosis/wallcausation_rust_regen.py`)
falls back to `hexo_rl.encoding.compat.infer_encoding_from_state_dict` whenever a checkpoint
lacks `metadata['encoding_name']`. That function (untouched by WP-4, **no graph-detect branch at
all**) tries `_filename_match(path_hint)` — longest registered-name substring in the path — BEFORE
shape inference. This is the exact same failure class WP-3 flagged as a "pre-existing flaky trap"
(`test_t10_manifest.py::test_shape_fallback`), just reachable through a live GNN-relevant path
that WP-4 never touched. Reproduced:

```python
net = GnnNet(...)  # bare weights-only, no metadata
torch.save(net.state_dict(), "/tmp/xxx/gnn_bare_weights.pt")
resolve_from_checkpoint(p)   # -> EncodingRegistryError (no filename match, no shape match) -- LOUD, fine

torch.save(net.state_dict(), "/tmp/xxx/gnn_v6_checkpoint_bare.pt")   # plausible real filename
resolve_from_checkpoint(p)   # -> spec.name == 'v6'   *** SILENT MISDETECT, grid spec for a GNN net ***
```

A "safe" filename raises loud (no probe key matches, no filename match) — correct. But a
perfectly plausible real filename containing a registered grid substring (`v6`, `v7`, `v8`, or
any `v6*` family name) makes `resolve_from_checkpoint` silently return the WRONG (grid) spec for
a genuine GNN checkpoint. This is exactly the F1-class hole (row 11a-c) restated for the
checkpoint-encoding-resolution surface instead of the model-construction surface — and it
survives WP-4 intact because the audit table never named `resolve_from_checkpoint`/`compat.py` as
a node to fix.

**Recommendation:** add this path to the ragged-contract audit table as a follow-up node, and
either (a) add the same `"representation.input_proj.weight" in state_dict` graph-marker probe to
`compat._shape_match`/`_filename_match` (require the marker to win over any accidental grid
filename substring), or (b) mandate `metadata['encoding_name']` stamps on all GNN artifacts before
this path is reachable (closes the hole by construction, matches the resolver's own
`DeprecationWarning` steer), or (c) at minimum, make `_filename_match` refuse to win when the
state dict shows ANY graph marker key, regardless of filename.

---

## Finding 4 — MEDIUM: "landed-verify" does not catch shape-preserving corruption

**Severity: MEDIUM.** Matches the code's own honest docstring hedge ("belt-and-suspenders...not
the only defense") but over-promises relative to the mission's literal ask and the construction
report's framing ("torch.allclose over every representation/policy/dist65 tensor").

Per mission priority 3's explicit test ("transpose one tensor in a saved ckpt and confirm the
`torch.allclose` verify FAILS loud"): transposing a **non-square** tensor (e.g.
`representation.input_proj.weight`, shape `[16,11]`) IS caught — shape mismatch surfaces earlier,
via the `node_feat_dim`/`edge_feat_dim` cross-check in `_build_gnn_model`
(`hexo_rl/eval/checkpoint_loader.py:565-580`). But transposing a **square** tensor (several exist
in `GnnNet`'s GINE conv layers, e.g. `representation.convs.0.lin.weight`, shape `[16,16]`) round-
trips with **NO raise**:

```
orig shape torch.Size([16, 16])
NO RAISE -- model loaded, landed-verify PASSED
matches transposed src: True
matches ORIGINAL (untransposed) net: False
```

The `torch.allclose` loop (`checkpoint_loader.py` ~587-605) only verifies "what got loaded into
the model equals what was in the saved state dict" — it cannot detect that the state dict's own
values are semantically wrong when the shape happens to be preserved. This is a narrow,
adversarial-only gap (requires a corrupted checkpoint with shape-compatible corruption, not a
realistic accidental failure mode) — low urgency, but worth either scoping down the "landed-
verify" claim in docs/comments, or (if real protection against this class is wanted) adding a
coarse content checksum alongside the weights.

---

## Finding 5 — LOW: reverse-F1 (grid-declared + GNN state dict) raises a raw, non-diagnostic `KeyError`

**Severity: LOW.** Loud, not silent (satisfies the contract's core requirement) — just
inconsistent polish vs. the nice named raises WP-4 added elsewhere.

`load_model_with_encoding(gnn_bare_weights.pt, device, declared_encoding="v6")` (bare GNN
weights, no metadata to contradict the declaration) →

```
RAISED: KeyError 'trunk.input_conv.weight'
```

from `_build_min_max_model`. This is the exact behavior the audit table already predicted for row
11e ("both read `trunk.input_conv.weight` → `None`/`KeyError` on a GNN sd") — pre-existing, not a
WP-4 regression, and not in WP-4's stated scope to improve. Noting only because mission priority 2
explicitly asked to verify this direction: it IS a named raise in the sense of "not silent", but
the message gives zero hint that a graph checkpoint was mis-declared as grid. Optional follow-up:
`_build_min_max_model`/`_build_kata_model` could catch the `KeyError` and re-probe with
`detect_encoding_from_state_dict(state, ..., strict=False)` to enrich the message when the shapes
actually look like a graph net.

---

## Finding 6 — INFO, no action needed: hybrid state dict resolves deterministically to graph

Constructed a state dict containing BOTH `representation.input_proj.weight` AND
`trunk.input_conv.weight` + `policy_fc.weight`. `detect_encoding_from_state_dict` returns
`gnn_axis_v1` in both `strict=True` and `strict=False` modes, deterministically (the graph check
is an unconditional `if` at the top of the function, always short-circuiting before the grid
probe). This is an inevitable, harmless consequence of the branch ordering — no real checkpoint
would ever carry both marker sets (they'd have to come from manually concatenating two different
nets' state dicts). Not tested in the 28 new tests; worth a one-line comment/test for
completeness but not a bug.

---

## Finding 7 — CLEAN: `detect_encoding_from_state_dict`'s own filename-vs-shape ordering (mission
priority 5) is NOT exploitable

The graph-detect branch WP-4 added (`hexo_rl/encoding/resolvers.py:468-469`) is purely
shape-marker-gated — `if "representation.input_proj.weight" in state: return
lookup("gnn_axis_v1")` — and never consults `ckpt_label`/filename at all. It runs unconditionally
FIRST, before the grid `inp_w` probe and before any of the grid-family filename disambiguators
(`v6w25`, `v6tp`, `v6_live2[_ls]` substring checks, which only ever run on grid-shaped state
dicts that already passed the `inp_w` probe). Concretely: a CNN state dict saved as
`gnn_v6_foo.pt` CANNOT get graph-detected (no marker key present → falls straight to the grid
path; `"gnn_v6_foo.pt"` matches none of the grid-family substring checks either, so it resolves
purely by shape as intended). The WP-3-flagged "filename beats shape" flaky trap lives entirely in
the **separate, untouched** `hexo_rl.encoding.compat.infer_encoding_from_state_dict` — see
Finding 3, where that exact bug resurfaces on a live (if secondary) GNN-relevant code path WP-4
did not close.

---

## Priorities 1/2 (grid byte-identical + F1-closed) — confirmed CLEAN

- **Grid passthrough**: `build_net`'s grid branch is `return HexTacToeNet(**kwargs)` — zero added
  defaults, zero dropped kwargs, zero device/dtype handling changes. `test_grid_dispatch_matches_direct_construction_{v6,dist65}`
  pins state_dict key+shape equality; read line-by-line against pre-diff call sites
  (`orchestrator.py`, `lifecycle.py`, `anchor.py`) — every kwarg forwarded byte-identically, just
  routed through one extra function call. `_resolve_fresh_in_channels`'s new graph short-circuit
  (`orchestrator.py:256-266`) returns an inert `(0, None)` that `build_net`'s graph branch never
  reads — verified it doesn't leak into grid behavior (grid path untouched, `_MHPD` defaults
  unaffected).
- **F1 scenario** (graph config through all 3 killed sites → `GnnNet` or named raise, never a
  CNN): confirmed for all three sites once `value_head_type: dist65` is explicit in config —
  built real `GnnNet` instances with matching shapes across `init_trainer` → `build_inference_model`
  → `build_eval_model`. **Never silently builds a CNN** — the constraint the whole contract exists
  to enforce holds, modulo the launch-blocking Finding 1 gate sitting in front of the happy path.
- **Reverse F1** (grid declared + GNN sd, and graph declared + CNN sd): both directions raise loud
  (`KeyError`/`RuntimeError`), never a silent partial load — confirmed via direct reproduction
  against `load_model_with_encoding`.
- **Doctored hparams**: an internally-inconsistent state dict (claimed `hidden=32` on
  `input_proj` while conv layers stay `hidden=16`) raises a clear `RuntimeError: size mismatch`
  from `load_state_dict(strict=True)` — loud, correct.

## Test quality (priority 6)

The 28 new tests assert **behavior**, not implementation details — state_dict key/shape equality,
output feature dims, exception types + message substrings. No grid-coverage regression: every
pre-existing grid dispatch path is preserved byte-identically; the new branches are pure-added
`if representation == "graph"` guards ahead of existing logic (verified via the diff, not just the
construction report's word). Real gap: as documented in Finding 1, every graph-config test
fixture across all 5 new test files explicitly sets `value_head_type: "dist65"`, which
systematically avoided exercising the exact landmine a real launch config is likely to hit.
Recommend, once Finding 1 is fixed: (a) a test that constructs a graph launch config WITHOUT
`value_head_type` and asserts a clean build; (b) a `checkpoint_loader.py` test against the real
`checkpoints/probes/gnn_bc/gnn_bc_040000.pt` artifact (Finding 2) instead of only synthetic
BC-shaped fixtures; (c) at least one coverage note pointing at `resolve_from_checkpoint`'s open
hole (Finding 3), even if fixing it is out of WP-4's scope.

---

## Suite verification

- Targeted 48 tests (7 WP-4 files): `48 passed` in isolation, `3.96s`.
- Full suite re-run (`pytest -q -m "not slow and not integration"`): `2533 passed, 134 skipped,
  15 deselected, 1 xpassed, 69 warnings in 239.17s`, exit `0`. Zero failures, zero collection
  errors — matches `WP4_construction.md`'s claimed baseline (minor `15` vs `14` deselected count
  is environment/ordering noise, not a regression).

## One-line summary for the dispatcher

**REVIEW-FIXES-REQUIRED / GAPS-FOUND.** Grid path and the graph-detect branch itself are clean;
the graph *construction* path has a launch-blocking `value_head_type` default landmine (Finding
1, fix pattern already exists in the C7 resume branch — port it to orchestrator/lifecycle/anchor);
the BC-prefit diagnostic doesn't reliably fire on the real banked artifact via the eval loader due
to an unregistered `train_bc.py` stamp (Finding 2); and a sibling, WP-4-untouched resolver
(`resolve_from_checkpoint`/`compat.py`) still silently mis-detects a GNN checkpoint as grid via
filename substring — the exact SILENT-CORRUPT class the whole program exists to kill, just not on
WP-4's named node list (Finding 3). Hybrid-sd: deterministic graph-wins, untested but harmless.
Filename-ordering exploit (priority 5) inside `detect_encoding_from_state_dict` itself: CLEAN — no
finding.
