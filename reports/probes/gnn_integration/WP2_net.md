# WP-2 (C2) — production GNN net + dist65 pooled value head

**Status: BUILT + TESTED, all green.** GNN-integration program (R4 ratified b+),
build wave WP-1 (Rust builder) ∥ WP-2 (this doc) — file+review disjoint.

**File:** `hexo_rl/model/gnn_net.py` (new). **Tests:** `tests/model/test_gnn_net.py` (new, 10 tests, CPU-only, 0.92s).

## Architecture

`GnnNet` — probe-284k class per `run4_gnn_design.md` §0/§4.2 net-scale ruling (the exact
architecture that measured +414 [+320,+560] BT-Elo, D-M R-LADDER R1/R2): GINE
`RepresentationNetwork` (hidden=128, num_layers=4, JK-cat, EDGE_DIM=5, in_dim=11) +
`PolicyHead`, both imported unmodified from `hexo_rl.bots.strix_v1_net` (same modules the
probe's `GnnBcNet` wraps) — byte-identical construction, so `representation.*` /
`policy_head.*` state-dict keys match the BC-prefit checkpoint by construction, not by a
remapping shim.

**dist65 pooled value head (NEW):** `GnnDist65ValueHead` — stone-masked mean pooling
(`emb[stone_mask].mean(dim=0)`, all-nodes fallback when a graph has zero stone nodes;
batched generalization is `segment_mean_with_fallback`, a `index_add_`-based vectorized
per-graph mean+fallback) over the JK-cat 512-wide (`num_layers*hidden`) embeddings, then
`Linear(512→32)→ReLU→Linear(32→65)`, decoded via
`hexo_rl.training.binned_value.decode_binned_value` — the SAME primitive
`network_min_max_head.py:98-105` uses for the CNN dist65 head. `scalar_to_two_hot` /
`decode_binned_value` / `binned_value_loss` are imported, not re-implemented (verified
pool-agnostic per `gnn_integration_scope.md` §C2). Fresh-init, as expected (E1 REVIVE:
dist65 warm-starts fine from an absent value head).

## Param count

| Component | Params |
|---|---|
| `representation.*` | 201,728 |
| `policy_head.*` | 65,793 |
| `representation` + `policy_head` (BC-transferable) | **267,521** |
| `value_head.*` (NEW dist65, 512→32→65) | 18,561 |
| **`GnnNet` total** | **286,082** |

Cross-check: probe `GnnBcNet.num_params() == 283,970` (frozen, `test_gnn_bc_probe.py`) =
267,521 (representation+policy) + 16,449 (probe's own unsupervised scalar value head,
512→32→1 + tanh). `267,521 == 283,970 − 16,449` confirmed
(`test_param_count_probe_284k_class`). `GnnNet` = 267,521 + 18,561 (new dist65 head) =
**286,082**, i.e. "~284k + new value head" as specified — the trunk+policy are exactly the
measured 284k class; only the value-head tail differs in size/architecture from the probe's.

## State-dict compatibility (mission item 1)

`load_representation_policy_from_bc(net, bc_state_dict)` loads ONLY `representation.*` /
`policy_head.*` from `checkpoints/probes/gnn_bc/gnn_bc_040000.pt` (`model_state_dict` key,
46 tensors across the two prefixes — incl. the `eps` buffers). Strict on those two prefixes
(missing/unexpected keys raise `RuntimeError`); `value_head.*` untouched (fresh-init).

**Result: loaded 46/46 tensors, landed-verify PASSED.** Test
`test_state_dict_loads_and_landed_verifies_from_bc_prefit`:
- `torch.allclose` landed-verify sampled 5 tensors per prefix (10 sampled, 9 actually
  verified — `representation.` has ≥5 keys, `policy_head.` has only 4, so `min(5,4)=4`
  sampled there + 5 from `representation.` = 9) — **covers BOTH representation AND policy,
  not value-only**, per the C7 red-team demand.
- Strengthened beyond sampling: the test additionally asserts `torch.equal` (byte-exact) on
  **every one of the 46 loaded tensors**, not just the sampled subset.
- `test_state_dict_mismatch_raises`: dropping one `representation.*` key from the source
  dict raises `RuntimeError` (no silent partial-load — the F1 guard).

## Reachability verdict: **PASS at hop-count 1** (honest finding, not the naive "4 layers barely make it")

Test: `test_reachability_completing_cell_sees_far_end_stone_gradient`. Position = 5 own
stones `(0,0)..(4,0)` (a 6-in-a-row minus the last cell) + empty completing cell `(5,0)`,
`win_length=6, radius=6`. Assert nonzero gradient of the completing cell's policy logit
w.r.t. the far-end stone `(0,0)`'s input node features, through the FULL 4-layer GINE trunk
+ policy MLP.

**Result: PASS, gradient nonzero.** But the honest hop-count finding (read
`strix_v1_graph.py:220-262`, not assumed): the axis-window walk connects node `i` to
**every** same-axis node within `window = win_length − 1 = 5` steps via a **single direct
edge per distance** (`for d in range(1, window+1): ... edge_src.append(i); edge_dst.append(j)`)
— it is NOT a chain of adjacent-cell edges. A 5-stone line spans exactly `window` cells, so
the builder emits a **direct 1-hop edge** from `(0,0)` to `(5,0)`. This test therefore proves
the direct edge's information survives `input_proj → GINE conv → policy MLP` with nonzero
gradient (a real risk on its own — an `index_select`/`index_add_` off-by-one, a dead ReLU,
or an accidental `.detach()` could all silently zero it) — it does **not** stress-test
whether 4 message-passing hops are load-bearing for THIS exact span, because 1 already
suffices structurally. For a line that also involves a blocking opponent stone between the
ends (which truncates the direct-edge walk via `should_stop`), multi-hop routing through
other same-axis or dummy-node paths would matter more, but the mission scope named exactly
the clean 5-line + empty-cell case, which is what was built and tested. No edge type is
missing; no schema change made.

## Forward signature (the contract)

- `forward_batch(x, edge_index, edge_attr, legal_mask, stone_mask, node_offsets=None) →
  (policy_logits (Lg,), value (B,1), bin_logits (B, 65))` — pure torch, block-diagonal
  disjoint union, `node_offsets` is the WP-B `[B+1]` ptr array (`None` = single graph).
  No PyG dependency (matches the probe / strix's plain-torch `_GINEConv`).
- `forward_single(x, edge_index, edge_attr, legal_mask, stone_mask) → (policy_logits (n_legal,),
  value scalar, bin_logits (65,))` — deploy path, `@torch.no_grad()`.

## Tests (10, all pass, CPU, 0.92s)

1. `test_param_count_probe_284k_class` — rep+policy == probe minus its scalar value head; total accounting.
2. `test_state_dict_loads_and_landed_verifies_from_bc_prefit` — 46/46 tensors load byte-exact + sampled `torch.allclose` landed-verify.
3. `test_state_dict_mismatch_raises` — dropped key raises (F1 guard).
4. `test_forward_single_shape_and_finiteness_on_real_positions` — 8 real WP-A self-play positions, shapes + finiteness + value bounds.
5. `test_forward_batch_shape_and_finiteness_on_real_positions` — 6-graph batch, same checks.
6. `test_block_diagonal_batch_matches_per_graph_forward` — 3-graph batch vs 3 single forwards, atol/rtol 1e-5, policy + value + bin_logits all match.
7. `test_segment_mean_with_fallback_matches_manual_per_graph_mean` — synthetic 2-graph case incl. the zero-stone fallback branch.
8. `test_dist65_head_decode_matches_shared_primitive` — head's internal decode == `decode_binned_value` directly.
9. `test_dist65_decode_extreme_bins_hit_support_endpoints` — one-hot bin 0 → −1, bin 64 → +1.
10. `test_reachability_completing_cell_sees_far_end_stone_gradient` — see verdict above.

Full-suite collection check: `tests/ --co -q` → **2623 tests collected, no errors** (no
collection breakage introduced).

## Explicitly out of scope (per WP-2 charter, other workpackages)

- Rust ragged producer / `graph_collate.collate_graph_batch` resolver (WP-1/WP-3): this
  module consumes already-collated torch tensors; it does not read the WP-B wire arrays.
  `forward_batch`'s `node_offsets` param is shaped to match WP-B's `node_offsets` field
  directly so the resolver can hand tensors straight through.
- Trainer forward-input branch (dense vs graph), `build_net(spec, state)` dispatch
  authority, `resolvers.py` graph-detect branch (WP-5/C4/C7).
- Registry TOML `representation` discriminant (WP-4).
- `engine/**` (WP-1).
