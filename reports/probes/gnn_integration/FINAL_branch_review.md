# FINAL whole-branch review — GNN-integration → master

**Scope:** 9 commits `f4fc523..9ba7464` (HEAD `9ba7464`, worktree clean). Cross-commit
seam review only — each commit already passed its own IMPL→REVIEW→RED-TEAM→bench chain.
Read-only; `.venv/bin/python`, no maturin, no git mutation.

## VERDICT: READY-TO-MERGE

Zero cross-commit blockers. Dense (grid/CNN) path proven behavior-identical across the sum
of 9 commits; every single-source primitive stayed single-source; the one shared native
builder carries the WP-1 fix in the built `.so`; config/launch artifacts and the design-doc
pin table cohere; full non-slow/non-integration sweep green.

---

## Cross-commit findings (severity)

All CLEAR — no seam composes wrongly. Enumerated for the record:

1. **amp_dtype seam (commit-A × commit-B × S7 bf16) — CLEAR.** `amp_dtype_for` is the ONE
   resolver; trainer `__init__` (`amp_dtype_for(model_representation(self.model), config)`),
   `InferenceServer.__init__` (`amp_dtype_for("graph" if self._is_graph else "grid", config)`),
   and `lifecycle.cuda_warmup` (warmup autocast dtype) all consult it. Grid = config knob
   (fp16 default), byte-identical to the removed inline `_resolve_amp_dtype`; graph = bf16
   unconditional. `scaler_enabled = fp16 and amp_dtype==float16` → auto-False on bf16. No
   composition defect.

2. **Dispatch seam (commit-A recorder × commit-B trainer × sample builder) — CLEAR.** All
   three graph producers call the SINGLE shared Rust `hexo_graph::build_axis_graph`:
   `inference_bridge.rs` (self-play inference), `game_runner/records.rs` (commit-A worker-loop
   recorder), `replay_buffer/hexg/sample.rs` (commit-B trainer sample-time rebuild). The WP-1
   empty-board 5×5 fallback therefore lands identically on all three + the export producer
   (which rebuilds through the same path). Empirically confirmed the built `.so` carries WP-1:
   empty-board push → `sample_graph_batch` → `node_offsets [0, 26]` = 25 legal + 1 dummy. The
   sweep thus ran against current native code, not a stale `.so`.

3. **Dense-path integrity across all 9 commits — CLEAR.** The only dense-path edit in
   `train_step` is one added `if isinstance(buffer, HexgBuffer): return self._train_on_graph_batch(...)`
   early-return; a dense `ReplayBuffer` never matches → dense body untouched. `__init__`
   amp resolver, `inference_server` resolver, and `cuda_warmup` grid branch are all
   byte-identical for grid (`model_representation(HexTacToeNet)=="grid"`). `from engine import
   HexgBuffer, ReplayBuffer` present in the built `.so`. 2675 dense+graph tests pass.

4. **Single-source discipline — CLEAR (no drift).** One definition each, all consumers import
   it: `GRAPH_FORBIDDEN_NONZERO_WEIGHTS` (trainer.py; launch-path test imports the same
   constant), `amp_dtype_for` / `model_representation` / `resolve_value_head_type` (build_net.py),
   `GNN_GRAPH_MARKER_KEY` (encoding/_probes.py → compat, resolvers, checkpoint_loader,
   checkpoints), mass-drop tolerance (`REL_TOL=1e-4` in sample.rs `mass_drop_check`, sole def).
   No duplicated copies anywhere.

5. **recency × bf16 seam — CLEAR.** Orthogonal axes. `_train_on_graph_batch` passes
   `recent_frac=recency_weight` (0.75) to `sample_graph_batch`; `recent_frac=0.0` is
   byte-identical to pre-commit-B. Graph runs set `recent_buffer=None` (commit-A P3) so
   recency is applied exactly once (no dense RecentBuffer double-application). bf16 is an
   autocast-dtype concern, disjoint from index selection.

6. **warm-start × amp_dtype — CLEAR.** `maybe_warmstart_gnn_from_bc` loads fp32 weights on
   CPU before `Trainer` wraps the net; autocast dtype is applied at runtime — dtype-agnostic,
   no seam.

7. **cuda_warmup spec threading — CLEAR.** Sole caller `loop.py:129` passes `spec=_arch.spec`,
   so a graph run warms the real graph seam (`forward_batch`); `spec=None` default keeps the
   grid branch byte-identical for every pre-S7 caller/test.

---

## Minor-findings register — rulings

| # | finding | ruling | basis |
|---|---------|--------|-------|
| A-F1 | `_in_ch` rehoist nit | ACCEPT-WITH-NOTE | dense-path recording var; graph push never touches it (records.rs:208); dense byte-identity proven + 2675 pass. Cosmetic. |
| A-F2 | unlabeled graph drops in `positions_dropped` | ACCEPT-WITH-NOTE | graph-only run → no dense drops to conflate; each dropped `GraphRecord` == 1 position so name is accurate; documented intentional hygiene reuse (inner.rs:1763). |
| A-F5 | `getattr` default | ACCEPT-WITH-NOTE | the `getattr(spec,"representation","grid")` / `getattr(model,"_orig_mod",model)` defaults are the byte-identity-preserving fall-throughs for non-spec callers — consistent with build_net's own default contract. |
| B-F1 | export mtime-cutoff fragility | ACCEPT-WITH-NOTE | offline one-shot operator corpus mint; NOT on launch/self-play path; run4 ships corpus-mix OFF (`pretrained_buffer_path: null`) so unused at launch. |
| B-F3 | collate `trunk_size=19` default | ACCEPT-WITH-NOTE | 19 & `win_length=6` == gnn_axis_v1 registry values (verified); param feeds ONLY the semantic VALIDATION cross-check (`_canonical_slot_vec`), which raises LOUD on mismatch — never silently corrupts tensors. A 2nd graph encoding fails safe. Recommend threading `spec.trunk_size`/`spec.win_length` when a 2nd graph encoding is registered. |
| B-F4 | recency dedup full-ring dilution | ACCEPT-WITH-NOTE | game_id dedup replaces recent-window collisions with full-ring draws → mild statistical dilution of effective `recent_frac`; not a correctness bug; matches dense RecentBuffer §8.5 approximation. |
| B-F5 | event field names | ACCEPT-WITH-NOTE | `fp16_scale=self.scaler.get_scale()` returns 1.0 on the bf16 (disabled-scaler) branch — slight misnomer, harmless; the hard-required `loss/policy_loss/value_loss` keys are present (delta §10 consumers satisfied). |
| R2-S3 | `model_representation` vs future wrapped models | ACCEPT-WITH-NOTE | unwraps only `_orig_mod` (torch.compile). No DDP/other wrapper exists in-tree today. A future wrapper would misclassify graph→grid (wrong amp + wrong dispatch) — add its unwrap when introduced. Latent, not live. |
| R2-S5 | stale precedent phrasing | DROP | doc phrasing only, no code effect. |
| — | `value_spread_canary` bare-`forward()` | ACCEPT-WITH-NOTE (do-not-fix-here) | fires per graph checkpoint save (not gated), hits `GnnNet` `NotImplementedError` (no bare `forward()`), swallowed by the trainer's outer `try/except` → `value_spread_canary_dispatch_failed` log line, save proceeds. NON-FATAL. `value_spread_canary.py` is a run3-shared DENSE file — do not edit here. Cheap future fix = representation-gate the `fire_canary` call at trainer.py:1533 (graph-only skip, zero dense-path risk). Not a blocker. |
| — | `test_gnn_seam_smoke.py` stale docstring fp16 | ACCEPT-WITH-NOTE | docstring-only; code path is bf16. Optional one-line docstring fix, no behavior impact. |

**Net: 0 fix-before-merge, 10 accept-with-note, 1 drop.** The two worth a post-merge
follow-up ticket (cheap, graph-only, zero dense risk): representation-gate `fire_canary`
(kills per-checkpoint log noise on graph runs); thread `spec.trunk_size`/`win_length` into
the three `collate_graph_batch` call sites before a 2nd graph encoding lands.

---

## Config / launch coherence — CLEAR

`run4_gnn.yaml` ↔ `run4_gnn_smoke.yaml` ↔ `run4_gnn_design.md §9.3` pin table ↔
`test_run4_gnn_launch_path.py` all agree:
- shared keys identical across both yamls: `encoding: gnn_axis_v1`, `value_head_type: dist65`,
  `in_channels: 0`, `draw_value: -0.5`, `ply_cap_value: 0.0`, `bot_batch_share: 0`, all 7
  `GRAPH_FORBIDDEN_NONZERO_WEIGHTS` = 0, `recency_weight: 0.75`, `pretrained_buffer_path: null`,
  `gnn_warm_start.enabled: true` + `gnn_bc_040000.pt`, `promotion_gate_subprocess_isolation: true`.
- namespaced paths correctly DISTINCT (prod `replay_buffer_run4_gnn.hexg` / `best_model_run4_gnn.pt`
  vs smoke `..._run4_gnn_smoke.*`) per §RUN3-STEP0 law — no variant-vs-variant buffer/anchor collision.
- smoke overrides labeled (bs=16, min_buffer_size=64, buffer_capacity=4096, inference_batch_size=16,
  stall 5400s); prod stays silent on bs (inherits 256, OQ-2 5080 rider OPEN — matches doc).
- design §9.3 as-built net = 286,082 params; §9.2 bf16 LAW matches `amp_dtype_for` code exactly.

---

## Full sweep

`.venv/bin/python -m pytest -q -m "not slow and not integration"` — HEAD `9ba7464`:

```
2675 passed, 136 skipped, 29 deselected, 1 xpassed, 69 warnings in 247.86s
```

- exit 0. **No failures, no errors, no collection errors** (fresh-clone worktree correctly set up).
- collected 2841 (selected 2812 = 2675 passed + 136 skipped + 1 xpassed). Matches the
  ~2811/2840 expectation (+1; branch added tests). First recorded single all-green post-`9ba7464`.
- `1 xpassed` = an xfail-marked test that passed; non-blocking (suite not `xfail_strict`, exit 0).
- warnings: 40 DeprecationWarning (checkpoint `encoding_name` metadata migration — known/expected),
  4 UserWarning, 1 TracerWarning (torch.jit trace_inference test) — all benign.
- parity test `test_hexo_graph_parity.py` passes standalone (87.6s), corroborating `.so` currency.
