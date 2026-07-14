# WP-2 RED-TEAM — adversarial audit of production `GnnNet` (commit 0a06273)

**Overall verdict: RED-TEAM-CLEAN.** No net bug, no crash, no silent numerical corruption.
Two LOW-severity latent dtype-defense gaps (net delegates dtype enforcement entirely to the
WP-3 resolver — not defense-in-depth) + one load-bearing INFORMATIONAL finding (measured D6
equivariance gap, feeds DS-3 axis-bias canary calibration).

Target: `hexo_rl/model/gnn_net.py` (`GnnNet`: GINE trunk + dist65 pooled head). Attacks
EXECUTED (not reasoned) via `.venv/bin/python`; harness archived at
`/home/timmy/.claude/jobs/7d6e8877/tmp/redteam_attacks.py`. Banked weights
`checkpoints/probes/gnn_bc/gnn_bc_040000.pt`; 20 real positions
`reports/probes/gnn_integration/wpa_positions.json`; build args match the BC training regime
(`prune_empty_edges=True, threat_features=True, relative_stones=True`, verified against
`train_bc.py:103-106`).

---

## Attack 2 (HEADLINE) — functional equivalence to the frozen BC probe

**Setup.** Load the SAME 40k `model_state_dict` into `GnnBcNet` (probe, `strict=True`, full sd
incl. its scalar value head) and into `GnnNet` (production, via
`load_representation_policy_from_bc`, representation+policy only). Forward 20 real WP-A graphs
through each; compare per-legal-node policy logits (`GnnBcNet.policy_logits_for_graph` vs
`GnnNet.forward_single`).

**Expected.** Bit-identical policy logits — same modules (`RepresentationNetwork`/`PolicyHead`
from `strix_v1_net`), same op order, same weights. If they diverge, the +414 [+320,+560] BT-Elo
evidence carried by the probe does NOT transfer to the production module.

**OBSERVED.**
- Loader: 46/46 tensors, 9 landed-verified.
- Single-graph path: **20/20 positions EXACT-EQUAL, max |Δ policy_logit| = 0.000e+00.**
- Batch path (`forward_batch`, block-diagonal 20-graph union) vs probe single: max |Δ| =
  **4.77e-7**, i.e. pure float32 accumulation order from block-diagonal batching, **within the
  WP-B prod parity gate (max|Δ| < 6.6e-7).**

**Verdict: PASS — evidence transfers.** The production trunk+policy is numerically bit-identical
to the +414-carrying probe on the deploy path. The batch path is parity-clean to prod tolerance.
No remapping-shim drift; the "byte-identical construction" claim in `WP2_net.md` holds under
execution.

---

## Attack 3 — D6 equivariance gap (unmeasured until now; feeds DS-3)

**Setup.** For each of 11 non-identity D6 elements (6 rotations × reflection), rotate a position's
stone coords by the hex-axial rotation `(q,r)→(−r,q+r)` (+ reflection `(q,r)→(q,−q−r)`), rebuild
the graph via the Python builder, forward. Compare original legal-node logit at coord `c` vs
rotated-graph logit at coord `D6(c)`. 8 positions × 11 elements.

**Coverage check (soundness gate).** 23859/23859 = **100.0%** of original legal coords land on the
rotated graph's legal set → the rotation is a genuine node relabeling (radius/hex_distance are
D6-invariant), so the gap is real net non-equivariance, NOT a coord-mismatch artifact. Confirmed
by the tiny VALUE gap below (a wrong rotation would move the board and blow up value).

**OBSERVED.**

| init | max \|Δ policy-logit\| | mean | median | p90 | max value(scalar) gap |
|---|---|---|---|---|---|
| **40k-trained (prefit init)** | **2.196** | 0.221 | 0.157 | 0.485 | 0.0048 (mean 0.0014) |
| random-init | 0.423 | 0.122 | 0.107 | 0.250 | 0.0033 (mean 0.0010) |

Per-element trained max gap is uniform across rotations+reflections (1.7–2.2), no privileged axis.

**Verdict: NOT invariant — as the contract predicts (Part 3: FEASIBLE-ON-LEGACY-V1, "do not drop
aug"). Two load-bearing findings for DS-3 calibration:**
1. **The TRAINED net is MORE non-equivariant than random** (mean 0.22 vs 0.12, max 2.2 vs 0.42 —
   ~1.8×/5×). BC on the un-augmented corpus (no aug in `bc_data.py`/`train_bc.py`, grep-clean)
   BAKED IN orientation-specific policy preferences. This is direct empirical support for the
   contract's "~12× fewer effective samples, do not drop aug" ruling.
2. **Value is ~equivariant (gap ~0.005); policy is NOT (gap ~2.2).** Mechanism: value pools stone
   embeddings symmetrically and the dominant threat features are axis-permutation-invariant
   (`node_threat_features` = max/count over the 3 axes); policy is per-node and reads
   `norm_q/norm_r` + axis-specific edge structure. **DS-3 axis-bias canary MUST read POLICY
   logits — a value-side axis canary is insensitive (0.005 floor) by construction.** Calibration
   number for the prefit-40k init: expect max policy-logit D6 gap ≈ 2.2, mean ≈ 0.22.

---

## Attack 1 — cross-graph contamination + degenerate graphs

**Setup.** (1a) Block-diagonal batch `[A, B]` where A is pathological (8 same-player stones, node
features ×1e6) and B is a tiny 2-stone graph; compare B's outputs in-batch vs B-alone. (1b) empty
board (0 stones). (1c) zero legal nodes (mask all-False). (1d) single dummy-node graph. (1e) batch
containing a 0-stone graph (segment_mean fallback path).

**Expected.** Block-diagonal isolation ⇒ B unaffected by pathological A; degenerate graphs handle
without crash or div-by-zero.

**OBSERVED.**
- Contamination: B policy max|Δ| = **1.49e-8**, value |Δ| = 1.12e-8, bin_logits 1.19e-7 (float
  noise). A's own outputs stay finite despite ×1e6 features. **No leak** — LayerNorm is per-node,
  message-passing is edge-scoped, pooling is `batch_vec`-segmented; no batch-global op.
- Empty board → 1 dummy node, 0 legal: policy shape `(0,)`, value = −0.011 via all-node fallback,
  bins finite. **No crash.**
- Zero-legal (mask all-False): policy `(0,)`, value computes. **No div0.**
- Single dummy node: policy `(0,)`, value finite.
- Batch with 0-stone graph: `segment_mean_with_fallback` fallback fires, value shape `(2,1)`,
  finite. `denom.clamp(min=1.0)` guarantees no div0 even at zero masked+total count.

**Verdict: PASS.** Block-diagonal isolation is exact under adversarial magnitude; every degenerate
graph is crash-free. NOTE (not a net gap): the net returns an empty policy row for a zero-legal
graph rather than rejecting it — the contract's `EmptyLegalSet` assertion is the WP-3 resolver's
job (correctly out of this module's scope per `WP2_net.md`), but the boundary is worth naming: the
net alone would emit a `(0,)` policy into the dense scatter silently.

---

## Attack 4 — dist65 head under adversarial pooling

**Setup.** Same graph, value via stone-mask mean vs forced all-node (fallback) mean; plus extreme
pooled inputs (×1e3, ×1e6, ×−1e6).

**OBSERVED.**
- stone-pool value = −0.0049 (argmax bin 35); all-node value = −0.0075 (argmax bin 45).
  **Fallback discontinuity in value = 0.0026** for this position.
- Value stays in support [−1,1] in every case. pooled×1e6 → 0.094; pooled×−1e6 → 0.156 — bounded,
  because `decode_binned_value` is softmax·support then `clamp(-1,1)`; the fixed support endpoints
  ±1 cannot be exceeded.

**Verdict: PASS.** Support and scale are structurally fixed to [−1,1]; no discontinuous
support/scale change. The stone→fallback transition introduces a bounded value jump (0.0026 here,
fresh-init head) but never leaves valid support. The discontinuity magnitude scales with how far
stone-mean and all-node-mean embeddings diverge — for a trained value head it could be larger, but
still hard-clamped to [−1,1]. (Head is always fresh-init per E1 REVIVE, so this fresh-head measure
is representative.)

---

## Attack 5 — dtype / contract hostility

**Setup.** Feed off-contract dtypes and observe die-loud vs silent-cast.

**OBSERVED.**

| Input | Result | Note |
|---|---|---|
| `edge_index` int8 / int16 | **DIED LOUD** — `index_select(): Expected dtype int32 or int64` | torch rejects sub-i32 index |
| `edge_index` int32 | **SILENTLY ACCEPTED**, output == i64 result (no drift) | contract declares i64; net does not enforce |
| `node_feat` float64 | **DIED LOUD** — `mat1 and mat2 must have the same dtype (Double vs Float)` | good |
| `edge_attr` float64 | **DIED LOUD** — same | good |
| `legal_mask` uint8 | **works TODAY** (correct rows) but emits `UserWarning: indexing with dtype torch.uint8 is now deprecated` | latent risk (below) |
| `stone_mask` uint8 | works TODAY (pooling correct), same deprecation warning | latent risk |
| `node_offsets` int32 (batch path) | accepted; `batch_vec` still int64 | no corruption |

**Verdict: PASS-with-2-LOW-severity-gaps.** Float dtype mismatches die loud (good). But:

- **GAP-1 (LOW): uint8 mask is a deprecated silent-corruption timebomb.** `emb[legal_mask]` /
  `emb[stone_mask]` (gnn_net.py:220,223, forward_single; same in forward_batch:196) accept a uint8
  mask TODAY only via torch's DEPRECATED uint8-as-boolean-mask path (warning fires). When torch
  removes that path, a uint8 tensor becomes an INTEGER index → `emb[u8]` silently gathers rows
  0/1 instead of masking → wrong policy/pooling with NO error. The contract's `DtypeMismatch`
  (legal_mask/stone_mask must be bool) is the designed defense and lives in the WP-3 resolver
  (not yet built); the net itself has no dtype guard. A one-line `assert mask.dtype == torch.bool`
  in the two forwards would be cheap defense-in-depth against the exact F1 silent-corruption class
  this program exists to kill.
- **GAP-2 (LOW): int32 `edge_index` silently accepted.** Numerically safe (i32 covers any batch
  range) and the ADV-4 u16-wrap trap can't reach the net (it wraps in the wire arithmetic before
  becoming a valid i64), so enforcement correctly belongs to the resolver's `DtypeMismatch`. Flag
  only as "net is not defense-in-depth on edge_index dtype" — it trusts upstream to hand it i64.

Neither gap is a current corruption; both are the net delegating dtype enforcement wholly to the
(unbuilt) resolver. Not a blocker for WP-2; a candidate hardening line.

---

## Summary

| Attack | Verdict | Headline number |
|---|---|---|
| 2 — probe equivalence (HEADLINE) | PASS | single-path 20/20 exact (max|Δ|=0.0); batch 4.77e-7 < 6.6e-7 gate |
| 3 — D6 equivariance | NOT-INVARIANT (expected) | trained max 2.196 / mean 0.221 policy; value 0.005; coverage 100% |
| 1 — contamination + degenerate | PASS | leak 1.5e-8; no crash on empty/terminal/single-node |
| 4 — dist65 pooling | PASS | value clamped [−1,1]; fallback jump 0.0026 |
| 5 — dtype hostility | PASS + 2 LOW gaps | float→loud; uint8-mask deprecated timebomb; i32 edge_index silent |

**Overall: RED-TEAM-CLEAN.** The production `GnnNet` is a faithful, bit-exact carrier of the
+414 BC-prefit evidence (Attack 2), isolates batched graphs under adversarial magnitude (Attack 1),
and keeps the value head in valid support (Attack 4). The two dtype gaps (Attack 5) are LOW-severity
defense-in-depth misses delegated to the WP-3 resolver, not net bugs. The measured D6 gap (Attack 3)
is the deliverable for DS-3: **canary on POLICY logits, expect max ≈ 2.2 / mean ≈ 0.22 on prefit-40k;
value is D6-blind (~0.005) and useless as an axis canary.**
