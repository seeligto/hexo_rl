# S7 round-2 fixes — combined REVIEW + RED-TEAM (fresh, one pass) — 2026-07-15

**Reviewer:** independent (did not author the fixes). **Worktree HEAD:** `80f7c54`, round-2
fixes UNCOMMITTED. **As-built doc:** `reports/probes/gnn_integration/S7_round2_fixes.md`.
**Findings context:** `S7_smoke_gate.md` § "Re-run after blocker fixes" (F5a/F5b/F6/F7/F8).

## VERDICT: **PASS** (non-blocking findings only)

F5a/F5b/F6/F7/F8 all genuinely closed. F8 graph branch is a **true single-source reuse** of the
WP-3 production seam, not a reimplementation. Dense arm byte-identical. Every red-team attack landed
on an existing guard. Findings below are hygiene / test-coverage; one MEDIUM is a gate-integrity
recommendation, none block launch.

---

## Per-priority status

| # | Priority | Status |
|---|---|---|
| 1 | F8 infer_batch graph branch | **PASS** — single-source WP-3 seam; dense byte-identical; deploy-matched Gumbel preserved; mixed pool safe |
| 2 | anchor.py arch-sync | **PASS** — cross-rep raises `RepresentationMismatch` BOTH directions (verified live); graph fields correct; dense byte-identical |
| 3 | eval_pipeline in-loop skip | **PASS** — LOUD (log + counter in result_types + surfaced in round summary); graceful; dense byte-identical |
| 4 | `model_representation` primitive | **PASS** for all reachable sites; census + 3 spot-audits hold |
| 5 | Circular-import (deferred import) | **PASS** — real cycle, correct fix, order-independent |
| 6 | Smoke yaml | **FIXES-RECOMMENDED** — drift trap (no parity test); MEDIUM |
| 7 | Test suites | **PASS** — all green (counts below) |

---

## Priority detail

**P1 — F8 (`selfplay/inference.py`).** `_infer_batch_graph` calls
`self._graph_batcher.submit_graphs_and_wait(positions)` where `positions =
[(list(board.get_stones()), current_player, moves_remaining), …]`. The Rust entry
(`inference_bridge.rs:1105`) builds each graph via **native `build_graph_from_request` (runs the
WP-1 seam guards)** → `submit_batch_and_wait_graph_rust` → the background `InferenceServer` graph
loop (`_run_graph_loop`: `collate_graph_batch` → `GnnNet.forward_batch` → `segment_softmax` →
`assemble_ls_from_gnn_probs`). No hand-rolled encode/marshal. The engine's server is constructed
with `batcher=self._graph_batcher` — the server thread and the submitter **share one batcher**, so
it is the same seam self-play's `infer_and_expand_graph` rides at the server-loop level.
- *Entry-point nuance (correct, not a defect):* `submit_graphs_and_wait` is documented as the
  eval/offline round-trip driver, NOT the per-leaf self-play hot path (workers never cross PyO3 per
  leaf). That is the right entry for a synchronous eval engine; the shared machinery below it is the
  reused part.
- *Dense byte-identical:* `__init__` hoists `model_representation(model)` into `_is_graph`; dense →
  `_graph_batcher/_graph_server = None`, no server started, `infer_batch` skips the branch. One
  isinstance at construct-time only, not per-inference — no hot-path cost (not a bench-gated surface;
  self-play uses the Rust loop, not this class per leaf).
- *Deploy-matched Gumbel (D-LADDER law):* `gumbel_search_py._infer_and_expand` runs the full Gumbel
  search in Python on `tree`; only leaf eval is routed to `engine.infer_batch`. The search regime is
  untouched → deploy-matched preserved (proof-run reported `deploy_matched: True`).
- *Mixed pool:* verified live — a dense `LocalInferenceEngine` and a graph one in the SAME process
  hold **distinct batchers**, and the dense engine spins **no server**. No shared queue, no routing
  hazard.
- *Off-window drop:* graph leg drops `overflow`, same contract as the dense single-window branch;
  for whole-board `gnn_axis_v1` (OQ-6) overflow is empty in practice (near-full board test: dense
  sums to 1.0000). Acceptable and documented.

**P2 — F5b (`training/anchor.py`).** `_inf_repr != _anc_repr` → `RepresentationMismatch` is
**symmetric** — I drove the reverse direction (grid inf_model + graph anchor) live: raises correctly
(the test file only covers graph-inf/grid-anchor). Graph-vs-graph compares
`representation.output_dim` + `value_head.fc2_bins.out_features` and skip-logs on net-scale
mismatch (no raise, no mutation — verified). Grid path is the byte-identical `in_channels`/`vht`
compare (pinned by untouched `test_anchor_branches.py`).

**P3 — F7 (`eval/eval_pipeline.py`).** Skip is LOUD and honest:
- structured `log.warning("eval_round_skipped_graph_representation", …)` (exactly once/round);
- `results["eval_opponents_skipped"] = len(OPPONENTS)` — new `EvalRoundResult` field, and
  `evaluation_round_complete` spreads `results` (L542) so the counter **is** in the round summary;
- graceful: `eval_games` initialized to 0 at L319 (no KeyError), `wr_best` absent → `wr_best_promoted
  = False` → no promotion, no divide-by-zero. Dense path unchanged (test asserts byte-identical).
- *Operational note (by design, OQ-8):* graph runs now **never promote in-loop** — best_model
  advances only via offline EVALFAIR. Correct honest behavior; ensure the run4 operator plan uses the
  offline promotion path.

**P4 — `model_representation` (`model/build_net.py`).** `isinstance(base, GnnNet) → "graph" else
"grid"`, unwraps `_orig_mod`. Correct for every reachable site (anchor built-nets, LocalInference
torch models, eval dispatch). Census (24 files) spot-audited:
- `orchestrator.py` — explicit `representation=="graph"` short-circuit at L265 **before** the read ✓
- `probe_threat_logits.py`, `analyze_api.py` — not imported by any training/eval/selfplay path
  (manual/standalone) ✓
- `v6_argmax_bot.py` — v6-dense by construction ✓

**P5 — circular-import fix.** Real cycle: `build_net → gnn_net.py:54 (from hexo_rl.bots.strix_v1_net
…) → bots.__init__ → our_model_bot → selfplay.worker → selfplay.inference`. A top-level build_net
import in inference.py would close it; fix defers into `__init__` (AST check: 0 top-level build_net
imports). Collection-order-independent (fresh import + full `--collect-only` both clean). anchor.py's
top-level build_net import is safe (training package not in the bots cycle — verified via passing
anchor tests + live import).

**P6 — smoke yaml.** See Finding S-2 (MEDIUM drift trap).

---

## Red-team results (all attacks defended)

| Attack | Result |
|---|---|
| (a) Poisoned/oversized positions → `submit_graphs_and_wait` | WP-1 guards fire at THIS entry: duplicate-stone / player-99 / current_player-77 → `ValueError` (player-range); moves_remaining=-5 → `ValueError` (WP1 Attack-4 narrowing-cast). huge coords (1e5,-1e5) → legal infinite-board position, finite `dense[362]`, no crash (correct). Near-full board → policy sums 1.0000, finite. |
| (b) Mixed-representation eval process | Graph engine + dense engine, one process: distinct batchers, dense spins no server. No cross-talk. |
| (c) Skip-counter honesty | Forced graph skip: exactly 1 structured warning, `eval_opponents_skipped=5`, `eval_games=0`, `promoted=False`, 0 db rows, no `eval_opponent_failed` noise. |
| (d) Anchor cross-rep, both directions | graph-inf/grid-anchor AND grid-inf/graph-anchor both raise `RepresentationMismatch` (reverse direction verified live, not just the one direction the test covers). |
| (e) Smoke §RUN3-STEP0 collision | `buffer_persist_path` + `best_model_path` both DISTINCT from run4_gnn's (`…_run4_gnn_smoke.*` vs `…_run4_gnn.*`); pinned by `test_run4_gnn_and_smoke_variant_paths_never_collide`. |

---

## Findings

- **S-1 [MEDIUM] Smoke-yaml drift trap (Priority-6 flag).** No test pins `run4_gnn_smoke.yaml ≡
  run4_gnn.yaml` except the labeled SMOKE OVERRIDE keys. The launch-path test checks
  forbidden-weight-zeros / corpus-mix-off / namespacing / batch_size resolve clean through the smoke
  file, but **outcome-lever parity is unguarded**: `draw_value`, `ply_cap_value`, `recency_weight`
  (the very §178 levers run4 is ABOUT) could diverge on a future prod edit and the smoke gate would
  silently validate DIFFERENT training semantics. They match today (-0.5 / 0.0 / 0.75) but only by
  hand. Recommend a parity test asserting all keys equal except the labeled override set (the yaml
  header already declares the contract — "must be mirrored here by hand" — it just isn't enforced).

- **S-2 [LOW] Graph InferenceServer thread cleanup relies on `__del__`/GC.** The report names
  `close()` as the intended teardown, but NO offline-eval caller calls it — `evalfair/core.py:415`
  (`run_arm`), `deploy_strength_eval.py:429/455`. Works today: `eng` (and the deploy bot holding it)
  drop to refcount 0 at `run_arm` return with no engine↔bot cycle, so CPython promptly runs `__del__`
  → `close()` → `join(5s)`; daemon thread dies at process exit regardless; the proof-run finished with
  zero orphans. Fragile under a future refactor that holds engines in a list or introduces a cycle.
  Recommend `try/finally: eng.close()` in `run_arm`. Not blocking.

- **S-3 [LOW] `model_representation` isinstance-else-grid.** Any non-`GnnNet` → "grid". Correct for
  all current call sites; a hypothetical future ONNX-exported / TorchScripted `GnnNet` reaching
  anchor-sync or LocalInferenceEngine would misclassify as grid. Out of scope now; note for a future
  graph-deploy-export path.

- **S-4 [LOW] Test-coverage gap (not a code defect).** `test_anchor_graph_representation.py` covers
  cross-rep in one direction only (graph-inf/grid-anchor). Code is symmetric and I verified the
  reverse live, but a regression narrowing the guard to one direction would pass the suite. Add the
  reverse-direction case.

- **S-5 [INFO] Doc citation nuance.** The as-built doc cites `opponent_runners.py`'s "lazy import →
  no import cycle" comment as the deferred-import precedent; that exact phrasing isn't in that file.
  The deferred-import fix itself is correct and convention-consistent — cosmetic only.

---

## Verification runs (this review, laptop, HEAD 80f7c54 + uncommitted)

- 6 new/modified test files (`-m "not slow and not integration"`): **46 passed**.
- `tests/{training,selfplay,model,eval} -m "not slow and not integration"`: **365 passed, 2 skipped,
  18 deselected** (matches as-built).
- `scripts/evalfair/tests -m "not slow and not integration"`: **60 passed, 3 deselected**.
- `tests/selfplay/test_gnn_record_dispatch.py -m integration` (e2e graph dispatch): **2 passed**.
- Full `--collect-only -m "not slow and not integration"`: **2797/2826 collected, 0 errors**.
- Fresh `import hexo_rl.selfplay.inference`: clean (circular-import proof).
- Red-team script (attacks a–e): all defended (table above).

Tree left byte-identical apart from this report.
