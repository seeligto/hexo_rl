# WP-2 REVIEW — production GnnNet (C2), commit `0a06273`

Fresh-eyes review of `hexo_rl/model/gnn_net.py` + `tests/model/test_gnn_net.py` +
`reports/probes/gnn_integration/WP2_net.md` against `docs/designs/gnn_ragged_contract_v1.md`
(amended), `docs/designs/run4_gnn_design.md` net-scale ruling, and
`docs/designs/gnn_integration_scope.md` §C2/§C7. Read-only review — ran the shipped test suite
(`.venv/bin/python -m pytest tests/model/test_gnn_net.py -v`, 10/10 PASS, 1.08s) and wrote a
standalone counter-case probe (`/tmp/wp2_review_probe.py`, not committed) exercising cross-graph
leakage, off-by-one `node_offsets`, the zero-stone fallback branch, and dtype robustness.

**Verdict: REVIEW-PASS** (with 5 SHOULD-FIX items — none block correctness of what's shipped; all
are test-coverage / defense-in-depth / documentation-precision gaps worth closing before WP-3
wires a real caller against this module).

---

## 1. Block-diagonal forward correctness — NO LEAK FOUND (verified, not just read)

Constructed a counter-case myself (2 graphs, 3+4 nodes and 2+5 nodes, random features/edges,
each graph's own internal ring topology only) and ran it against `forward_batch`:
- Perturbing graph A's node features by +1000 leaves graph B's policy logits and pooled value
  **byte-unchanged** (`torch.allclose` exact), while graph A's own outputs visibly move. No leak
  via `legal_mask` boolean-indexing (`gnn_net.py:172` `legal_emb = emb[legal_mask]`) or via
  `segment_mean_with_fallback`'s `index_add_` routing (`gnn_net.py:102-134`).
- This is independently corroborated by the shipped
  `test_block_diagonal_batch_matches_per_graph_forward` (`test_gnn_net.py`): it compares
  `forward_batch`'s per-graph slice against a **fully independent** `forward_single` call on that
  graph alone (no other graph present in memory at all) — a real leak would show up as a
  numerical mismatch there, and it doesn't (atol/rtol 1e-5, 3 real WP-A positions).
- No `BatchNorm` anywhere in the trunk (`RepresentationNetwork` uses `LayerNorm`, which
  normalizes per-node over the feature dim, no cross-node statistics) and `PolicyHead.mlp` is a
  plain per-row `nn.Sequential` (weights shared, but no cross-row coupling) — so criterion 4's
  named confound classes (global pooling in the policy head, shared-bias leakage) don't exist in
  this trunk. Verified by reading `strix_v1_net.py:88-116` (`RepresentationNetwork.forward`) and
  `:119-126` (`PolicyHead`).

**Off-by-one counter-case (real finding, see 4 below):** shifting the interior `node_offsets`
boundary by +1 does **not** raise — it silently produces a *different* (wrong) pooled value with
no error. This is exactly the class `docs/designs/gnn_ragged_contract_v1.md` assigns to the WP-B
resolver (`NodeCountChecksum` / `EdgeCrossesGraphBoundary`), which doesn't exist yet — see
finding 4.

## 2. State-dict transfer honesty — mostly real, one redundant layer

- `load_representation_policy_from_bc` (`gnn_net.py:236-291`) pre-checks
  `own_keys_for_prefixes - src.keys()` / `src.keys() - own_keys_for_prefixes` **exhaustively**
  (all 46 keys, not sampled) and raises `RuntimeError` **before** calling `load_state_dict` on any
  mismatch. Verified this actually raises, not just the happy path:
  `test_state_dict_mismatch_raises` drops one `representation.*` key and the loader raises — ran
  it, passes.
- The downstream `torch.allclose` "landed-verify" (`gnn_net.py:264-284`, sampled `verify_n=3` in
  production, `verify_n=5` in the test) is **functionally redundant** given the above pre-check,
  for a reason worth naming precisely: PyTorch's `load_state_dict(..., strict=False)` only
  suppresses the *missing/unexpected-key* error — a shape-mismatched-but-present key **always**
  raises regardless of `strict` (verified against PyTorch's documented behavior), and a
  present+shape-matching tensor is **always** copied via `copy_()`. Since the exhaustive
  missing/unexpected check already ran first, there is no remaining code path through which
  `load_state_dict` would silently no-op a tensor here. The `torch.allclose` step also compares
  the reloaded tensor only against **the same `src` dict it was loaded from** — it cannot catch a
  "transposed/wrong-value-but-right-shape" bug in the *source* checkpoint (there's no independent
  oracle to compare against), so it does not answer "would it catch a transposed tensor" in the
  sense of a corrupted source. It genuinely only guards against a bug in this function's own
  plumbing (e.g. a future refactor that re-adds a remapping shim between the pre-check and the
  load call).
  This differs from the E1 precedent it's modeled on (`checkpoint_loader.py:590-603`): E1's
  `_build_min_max` calls `model.load_state_dict(state, strict=False)` with **no** prior
  missing/unexpected pre-check, so E1's `torch.allclose` on `value_fc2_bins.weight` is the *only*
  thing standing between a mis-detected architecture and a silently-random value head — it is
  load-bearing there. Here the exhaustive pre-check already closes that hole, making the sampled
  allclose insurance against a narrower, already-mostly-closed risk.
  **SHOULD-FIX 1** (`gnn_net.py:241`, `verify_n: int = 3` default): given only ~46 tensors total
  and the check costs microseconds, default to verifying **all** of them (or at least
  `min(verify_n, len(keys))` should not undersell "landed-verify" — WP2_net.md is honest that the
  *test* strengthens to exhaustive `torch.equal`, but the **production** function callers (WP-3/
  WP-5, not built yet) get only the 3-sample default). Fix: change default to `verify_n=None` →
  verify all, or bump default to a value ≥ max tensors-per-prefix.

## 3. dist65 primitive reuse — CONFIRMED, not re-implemented

- `gnn_net.py:26` imports `N_VALUE_BINS, decode_binned_value` directly from
  `hexo_rl.training.binned_value` — same module the CNN head uses.
  `network_min_max_head.py:103-105` imports and calls the identical
  `decode_binned_value(bin_logits)` for the CNN's dist65 branch — grepped and read both call
  sites; they're the same function object, not parallel reimplementations.
- `GnnDist65ValueHead.forward` (`gnn_net.py:75-80`) does exactly what
  `gnn_integration_scope.md` §C2 prescribes: keep the `value_hidden`-width hidden layer, replace
  the final `Linear(hidden,1)+Tanh` tail with `Linear(hidden,65)` and route through
  `decode_binned_value` — matches the charter's own "concretely" worked example almost verbatim.
- No bin edges, two-hot logic, or loss math appears in `gnn_net.py` at all (`scalar_to_two_hot` /
  `binned_value_loss` aren't even imported here — correct, they belong to the trainer, out of
  this module's scope). **No drift risk found on this axis.**

## 4. Reachability test — honestly stated, structurally sound, but leaves the actually-interesting case untested

- Verified the WP2_net.md claim against source: `strix_v1_graph.py:220-262`'s axis-window walk
  does emit a **direct** edge from node `i` to every same-axis node within `window = win_length-1
  = 5` steps (the loop appends an edge at **every** `d` in `range(1, window+1)`, not just the
  final `d`), so for the test's clean 5-stone line + empty completing cell, `(0,0)` and `(5,0)`
  really do get a direct 1-hop edge (confirmed by hand-tracing the loop: same-owner stones never
  trigger `should_stop`, so the walk never breaks before `d=5`). The "1-hop by construction, not 4
  hops load-bearing" finding in WP2_net.md is **accurate**, not overclaimed.
- The gradient path itself is unconfounded (see finding 1's LayerNorm/no-pooling-in-policy-head
  argument) — `target_logit.backward()` reaching nonzero `x.grad[far_end_idx]` really does prove
  the direct edge carries gradient through `input_proj → GINE conv → policy MLP`, not some
  incidental bias/normalization shortcut.
- **SHOULD-FIX 2**: WP2_net.md itself names the gap — a line broken by an opponent stone (which
  triggers `should_stop` and truncates the direct-edge walk, forcing genuine multi-hop routing) is
  the scenario where "does information reach across N layers" would actually be tested, and it's
  explicitly **not built**. Given this is presented as "the correctness risk named in the WP-2
  charter," and the shipped test can only ever pass trivially for any win-line shape (window=5
  means every winning line's endpoints are *always* ≤1 hop apart by construction — there is no
  win-line geometry for which this builder needs multi-hop reachability), the interesting
  未-tested case is specifically **blocked/discontinuous threat patterns**, not longer win lines.
  Recommend either building that second case (5-10 lines: same stones but insert one opponent
  stone at, say, `(2,0)` so the direct walk from `(0,0)` breaks, and check the completing cell
  still gets nonzero gradient through a 2-hop path) or explicitly deferring it to WP-3/WP-5 in the
  "explicitly out of scope" list (currently absent — WP2_net.md discloses the gap in prose but
  doesn't move it to the scope table).

## 5. Deploy-path parity — duplicated pooling logic, undocumented rationale

- `forward_batch` (`gnn_net.py:160-202`) already handles `node_offsets=None` as an implicit `B=1`
  single-graph case (`gnn_net.py:185-187`). `forward_single` (`gnn_net.py:205-234`) does **not**
  call `forward_batch(..., node_offsets=None)` or `segment_mean_with_fallback`; it re-implements
  the stone-mask-mean-with-all-nodes-fallback branch inline
  (`gnn_net.py:222-225`: `if stone_mask.any(): pooled = emb[stone_mask].mean(dim=0) else: pooled =
  emb.mean(dim=0)`). This is a **second, independent implementation of the same fallback rule**,
  and nothing pins them together except the general-purpose real-position parity test (which, per
  finding 6, never exercises the fallback branch itself).
- I confirmed by direct construction that a synthetic zero-stone graph produces identical results
  through both paths today (`forward_single` vs `forward_batch(node_offsets=None)`, atol 1e-5) —
  so there's **no live bug**, but the two implementations can silently diverge on a future edit to
  either one (e.g. someone changes `segment_mean_with_fallback`'s fallback condition from
  "masked_counts==0" to something else and forgets `forward_single`).
- No comment anywhere states *why* this duplication exists (e.g. a perf argument — avoiding
  `_node_offsets_to_batch_vec` + 4 `index_add_` calls' allocation overhead for the single-graph
  deploy hot path would be a legitimate reason, since `forward_single` is presumably the
  self-play MCTS leaf-eval path WP-A benchmarked). Given the ambiguity, this is a live tradeoff to
  name, not silently accept:
  **SHOULD-FIX 3**: either (a) delegate `forward_single` to
  `forward_batch(x, edge_index, edge_attr, legal_mask, stone_mask, node_offsets=None)` +
  `.squeeze(0)` on `value`/`bin_logits` — guarantees byte-identical semantics by construction,
  costs one extra `repeat_interleave` + a few `index_add_` calls on a single-graph-sized tensor
  (cheap unless benched otherwise), or (b) keep the duplicated fast path but add a one-line
  comment stating the perf rationale, and add the test named in finding 6 so the two paths' *fallback
  branch* — not just the common-case branch — is pinned by an explicit test, not just accidentally
  covered by real positions that happen to always have ≥1 stone.

## 6. Test-quality gaps

- **SHOULD-FIX 4 (empty-stone-mask fallback, real production scenario, currently untested
  end-to-end):** confirmed via `reports/probes/gnn_integration/wpa_positions.json` that the
  min `n_stones` across all 320 frozen WP-A positions is **3** (`ply_range_pool: [2, 149]`) — the
  shipped real-position tests (`test_forward_single_...`, `test_forward_batch_...`,
  `test_block_diagonal_batch_matches_per_graph_forward`) therefore **never** exercise the
  zero-stone fallback branch in `forward_single` or `forward_batch` (only
  `test_segment_mean_with_fallback_matches_manual_per_graph_mean` hits it, and only against the
  raw pooling function with synthetic embeddings — not through either forward path). The empty
  board (ply 0, before P1's opening move) is a real MCTS root position a deployed net will face.
  Recommend adding a synthetic 0-stone graph test through both `forward_single` and
  `forward_batch`, asserting finite output and (per finding 5) that the two paths agree.
- **SHOULD-FIX 5 (input dtype/shape trust boundary undocumented):** confirmed empirically that
  `forward_single`/`forward_batch` accept an `int32` `edge_index` silently (no error; the WP-B
  contract requires `i64` and calls a wrong dtype `DtypeMismatch`, a named LOUD-FAIL owned by the
  not-yet-built `graph_collate.collate_graph_batch` resolver). This is a reasonable scope boundary
  (the module's docstring says it "consumes already-collated torch tensors" and the
  producer/resolver are WP-1/WP-3's job) but the **"explicitly out of scope" list in WP2_net.md
  doesn't say this explicitly** — it lists the Rust producer and the resolver as out of scope, but
  doesn't say "this module performs zero dtype/shape/monotonicity validation of its own inputs and
  is unsafe to call directly outside the resolver-guarded path." One sentence would close the gap
  between what a future reader assumes ("net.py forwards → must be safe") and what's actually true
  ("net.py trusts its caller completely"). Same applies to the `node_offsets` off-by-one silent
  divergence from finding 1 — worth a one-line note, not a code change (WP-B already owns the
  fix).
- **NIT:** no test for a minimum/degenerate graph (e.g. a single legal move, no stones at all
  except via the fallback case already flagged) or for a fully-legal-no-stones opening-move graph
  specifically at `moves_remaining=1` (asymmetric turn structure per HTTT's "P1 opens with 1 move"
  rule) — lower priority than 4/5 above since it's a variant of the same gap.

## 7. Repo idiom / doc-comment honesty / dead code

- Style matches neighboring `hexo_rl/model/*.py` (e.g. `network_min_max_head.py`): module
  docstring with attribution + design notes, `from __future__ import annotations`, fully typed
  signatures. No idiom violations found.
- `forward_single` uses `@torch.no_grad()` where the reference `HeXONet.forward`
  (`strix_v1_net.py:162`) uses the stricter `@torch.inference_mode()`. Not a bug (no_grad is a
  superset-permissive choice), but worth a NIT: if there's no reason to allow a `.requires_grad_()`
  re-entry after a `forward_single` call, matching the reference's `inference_mode` would be
  marginally faster and closer to the ported precedent this module explicitly claims to mirror.
- No dead code found — all imports (`random`, `Optional`, `Sequence`, `Tuple`) are used;
  `BC_TRANSFER_PREFIXES` is consumed by both the module and the test.
- WP2_net.md's claims were cross-checked line-by-line against the diff and the source it cites
  (`strix_v1_graph.py:220-262`, `network_min_max_head.py:98-105`) — all accurate, including the
  self-critical disclosures (finding 4's reachability caveat, finding 2's "strengthened beyond
  sampling" note). No overclaiming found in the doc itself; the gaps above are omissions, not
  misstatements.

---

## Summary table

| # | Severity | Location | Finding |
|---|---|---|---|
| 1 | (informational) | `gnn_net.py:172`, `:102-134` | Block-diagonal forward has NO cross-graph leak — verified by construction + existing test. No leak found. |
| 2 | SHOULD-FIX | `gnn_net.py:241` (`verify_n=3` default) | Landed-verify sampling is redundant given the exhaustive pre-check + PyTorch's own shape-raise; default to exhaustive (cheap) rather than sampled. |
| 3 | (informational) | `gnn_net.py:26`, `network_min_max_head.py:103-105` | dist65 primitive reuse CONFIRMED — same imported function, not re-implemented. |
| 4 | SHOULD-FIX | `WP2_net.md` reachability section; `strix_v1_graph.py:220-262` | Reachability test is honest and structurally sound but can only ever pass trivially (window=5 makes every win-line 1-hop); the actually multi-hop-requiring case (opponent-blocked line) is undisclosed-as-scope, not built. |
| 5 | SHOULD-FIX | `gnn_net.py:205-234` vs `:160-202` | `forward_single` duplicates `segment_mean_with_fallback`'s fallback logic instead of delegating to `forward_batch(node_offsets=None)`; no documented rationale (perf or otherwise) for the duplication. |
| 6 | SHOULD-FIX | `tests/model/test_gnn_net.py` | Zero-stone-mask fallback branch (real MCTS-root scenario) is untested end-to-end for both forward paths — WPA positions never go below 3 stones. |
| 7 | SHOULD-FIX | `hexo_rl/model/gnn_net.py` (module docstring / "out of scope" list) | Module performs zero dtype/monotonicity/boundary validation of its inputs (confirmed: int32 edge_index and an off-by-one interior `node_offsets` are both silently accepted); state this trust boundary explicitly rather than leaving it implicit. |
| 8 | NIT | `gnn_net.py:204` vs `strix_v1_net.py:162` | `forward_single` uses `@torch.no_grad()` vs reference's `@torch.inference_mode()` — likely fine, flagged for awareness. |

**None of the above block landing** — the core C2 deliverable (block-diagonal forward, BC
state-dict transfer, dist65 primitive reuse, reachability) is correct and honestly documented.
The SHOULD-FIX items are cheap (mostly test additions + a default-value change + doc sentences)
and worth closing before WP-3 builds a real caller on top of this module's currently-unvalidated
trust boundary.

**Verdict: REVIEW-PASS**
