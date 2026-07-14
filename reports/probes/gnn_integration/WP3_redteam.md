# WP-3 RED-TEAM — LIVE graph inference seam (`871ea41` + `5479199`)

**Agent:** WP-3 red-team. Attacks EXECUTED against the production path (runner-built
graph batcher + `InferenceServer._run_graph_loop`), driven exactly as
`tests/selfplay/test_gnn_seam_smoke.py` does. Read-only on source; `.venv`
engine `.so` used as-built (no rebuild). Scratch under
`/home/timmy/.claude/jobs/7d6e8877/tmp/`.

## OVERALL VERDICT: **GAPS-FOUND** (all latent / by-design — none a WP-3 grid regression)

The **structural** contract holds: shape / segment-length / offset / legal-node
geometry violations all **die loud**, S2 window-center threading is **correct**,
no **waiter leak** on server death, the **grid path is byte-clean**, and the seam
is **deterministic**. The gaps are all in **model-OUTPUT value validation** and
the **release-inertness of the numeric `debug_assert`s**:

| # | Finding | Severity | Regression? |
|---|---------|----------|-------------|
| G1 | No finiteness check on model policy/value output — **NaN/Inf silently poisons MCTS** | MEDIUM | No (parity with CNN path) |
| G2 | The one guard that would catch NaN priors (`assemble` sum≈1) is a `debug_assert` — **compiled OUT of the release build** | MEDIUM | No |
| G3 | Positional `id↔segment` binding: equal-legal-count mis-ordered submit **silently swaps priors** | LOW (latent) | No |
| G4 | S2 / trunk / sum invariants are all `debug_assert`s — inert in release (correct-by-construction today) | LOW | No |

**Build note (enabler for G1/G2/G4):** the installed `.so`
(`.venv/.../engine.cpython-314-...so`) is **byte-identical to `target/release`**
(md5 `dcd82e3e…`). Every numeric `debug_assert` string
(`"…probs do not sum to 1…"`, `"…builder window_center != Board…"`,
`"graph trunk mismatch"`) is **absent from the binary** (`strings | grep -c` = 0)
— confirming `debug_assertions=off`. This is the production-representative build,
so any correctness that rests on a `debug_assert` is **not enforced in self-play**.

---

## Attack 1 — contract-violation injection at the LIVE seam

Monkeypatched `GnnNet.forward_batch` on the running server; drove the 26-position
smoke fixture through `submit_graphs_and_wait` (same
`next_graph_batch → collate → forward_batch → segment_softmax → submit_graph_inference_results
→ assemble_ls_from_gnn_probs` path the worker's `infer_and_expand_graph` rides).

| Variant | Expected | OBSERVED | Verdict |
|---|---|---|---|
| **1a** logits permuted within each graph | (undetectable) | NO-ERROR; valid distribution (sum=1, finite) but priors **scrambled** — parity vs oracle **0.031** (clean = 1.3e-8) | **SILENT — by design** |
| **1b** logits for N−1 legal nodes | die loud | `segment_softmax` `scatter_reduce_` size mismatch → `RuntimeError` → `submit_graph_inference_failure` → caller `ValueError`. Named error logged (`graph_inference_forward_failed`). Worker would skip batch. | **DIE-LOUD ✔** |
| **1c-value** NaN value | die loud (per claim) | NO-ERROR; **all 26** positions returned a **NaN value**. No finiteness check anywhere. | **SILENT ✖ (G1)** |
| **1c-policy** NaN policy logits | die loud (per claim) | NO-ERROR; **all 26** dense priors NaN, sum≠1. `assemble` sum≈1 guard is `debug_assert` → inert in release. | **SILENT ✖ (G1/G2)** |
| **1c-inf** +Inf policy logit | die loud (per claim) | NO-ERROR; `inf−inf` in stable-softmax → NaN → 4 fused batches' lead node poisoned. | **SILENT ✖ (G1)** |
| **1d** results for wrong request-id order | die loud | equal legal-count pair → **SILENT SWAP** (A's leaf got B's priors, `a_vs_B`=3.7e-9, no error); unequal count → die loud via `meta.policy_dst_slot.len() != leaf_probs.len()` | **SILENT (latent) (G3)** |

### Headline SILENT finding (G1): model-output finiteness is NOT validated
`grep -niE 'isfinite|isnan|isinf'` over `inference_server.py`, `graph_collate.py`,
`records.rs`, `inference_bridge.rs` = **zero hits**. A NaN/Inf from the net —
policy OR value — flows straight through `submit_graph_inference_results` (which
validates only lengths/offsets) and `assemble_ls_from_gnn_probs` (which validates
only slot ranges) into `expand_and_backup_ls_at`. A NaN value hits
`backup(): node.w_value += value` and **poisons `w_value` up to the root** (NaN +
x = NaN → undefined PUCT argmax for the rest of the game). This is **reachable in
production**: the graph loop runs **fp16 autocast on CUDA** (the exact reason the
smoke pins CPU fp32), where policy/value overflow to Inf/NaN is a real event.

- **Not a graph-seam regression** — the CNN dense path (`run()`:
  `probs = log_policy.float().exp(); probs/=sum`; `value` submitted raw) shares the
  identical gap. But the seam design's **"die-loud everywhere, no silent
  fallback"** claim is **overstated**: it is true for *structure*, false for
  *value finiteness*, and the graph path's **only** would-be numeric catch
  (`assemble` `debug_assert` sum≈1) is **compiled out** (G2).
- **Cheap fix (closes the design's own claim):** one always-on check —
  `torch.isfinite(policy_logits).all() & torch.isfinite(value).all()` before submit
  (raise → existing `submit_graph_inference_failure` die-loud), or an always-on
  Rust finiteness scan in `submit_graph_inference_results`. Converts silent
  NaN-poison → die-loud + batch-skip.

### 1a / 1d are by-construction, not new holes
- **1a** (within-graph permutation) has **no redundancy to detect** — the net's
  per-legal-node output order *is* the contract; a scrambled CNN policy vector is
  equally undetectable. Flag only: the ADV-set validates the *wire* (input), never
  the *net output ordering*.
- **1d** binding is **positional** (`id=request_ids[i]`, `leaf_probs=probs[lo[i]..lo[i+1]]`,
  then look up `id`'s own `meta`). The length check catches every mis-order where
  the swapped graphs differ in legal count; an **equal-count** swap is invisible.
  **No production path reorders** (`request_ids` + `wire` come from one
  `next_graph_batch`; probs/offsets built from the same `batch` in lockstep) →
  latent robustness gap, not a live defect. (First test attempt was degenerate:
  two translation-equivalent boards are policy-isomorphic under the relative-coord
  GNN, so the swap was invisible for the wrong reason; re-ran with same-coords /
  different-colour boards → clean SILENT-SWAP confirmation, policy divergence
  0.021.)

---

## Attack 2 — S2 falsification (window-center drift / frame correctness)

**Result: S2-SOUND — no F1 mismatch achievable.** 12 deep / negative /
asymmetric-bbox positions (window "slid" far from origin; truncate-toward-zero
`(min+max)/2` sign-sensitive):

- in-window parity at the **builder** center: max **1.1e-8** (correct);
- in-window parity read at a deliberately **wrong** center (+1): min **5.1e-3**,
  **0** positions where the wrong center still matched → the frame is **load-bearing**
  and the seam threads the **right** one.

**Source proof it cannot drift:** `hexo_graph::window_center` (lib.rs:239) and
`Board::window_center` (board/state/core.rs:345) are the **identical**
truncate-toward-zero `((min_q+max_q)/2, (min_r+max_r)/2)` formula. `Board`'s
`min_q/max_q/min_r/max_r` are **pure running stone extents** (updated only in
`place_stone`, restored on undo — core.rs:483-493), and the worker feeds the
builder the **same stone set** (`leaf.cells_iter()`, inner.rs:887). Same formula ×
same stones ⇒ same center. The fresh-expand path (`expand_and_backup_ls_at`) reads
at `centers[i]` = the builder center **directly**; the TT-hit re-read path
(`expand_and_backup_single_ls`, board frame) reads at `board.window_center()` =
the same value. **Residual (G4):** the coincidence guard `debug_assert_eq!` is
**release-inert** — correct today, but a future refactor that diverged the two
formulas (or an infinite-canvas encoding whose `Board` extents include a
legal-radius margin) would misread the **TT-hit** path silently in production.

---

## Attack 3 — mid-episode server death / waiter leak (N5)

**Result: N5 VERIFIED — no waiter leak.** All three death modes released every
blocked waiter with **no hang** (20 s watchdog):

| Mode | Injection | Caller | Hung? |
|---|---|---|---|
| (a) exception in **collate** | `collate_graph_batch` raises | `ValueError` (die-loud) | **No** |
| (b) exception in **forward** | `forward_batch` raises | `ValueError` (die-loud) | **No** |
| (c) **server thread exit** | `forward_batch` raises a `BaseException` (escapes BOTH `except Exception` handlers → run() unwinds) | `ValueError` | **No** |

(a)/(b) route through the inner `except Exception` →
`submit_graph_inference_failure(request_ids)` → each waiter `Err` + notify. (c)
proves the **`finally: self._batcher.close()`** path: `close_rust`
(inference_bridge.rs:394) sets `closed=true` and notifies **every**
`graph_waiter.cv` (line 402-404); the blocked `submit_batch_and_wait_graph_rust`
loop wakes, sees `closed`, returns `Err` (line 456). The caller was released by
the dying server's `finally` **before** any external `stop()`.

**Residual (LOW, theoretical):** `submit_batch_and_wait_graph_rust` has a
lost-wakeup TOCTOU — it checks `closed` (line 456) *then* `cv.wait` (line 459)
while holding only the `waiter.result` mutex, but `close_rust` sets `closed`
(atomic) and `notify_all` **without** taking that mutex. A `close` landing between
the check and the park would be lost → that one waiter hangs. Extremely tight
window; all empirical runs released cleanly. The result-delivery path
(`submit_graph_inference_results`) is correct (sets result *under* the waiter
mutex before notify).

---

## Attack 4 — off-window integrity under load (56 positions)

**Result: CLEAN.**

- 56 positions, **44** with off-window overflow, **11 582** overflow nodes total;
- **0** illegal overflow keys (every key ⊆ the position's legal set),
  **0** stone-keyed overflow, **0** duplicate overflow keys;
- distribution mass `sum(dense)+sum(overflow)` ≈ 1 on **all** 56 (0 bad).

`off_fraction = 0.785` here is an **artifact of adversarial max-spread synthetic
boards** (clusters 22-45 apart, far beyond real 19×19 bounds) — **not** a
contradiction of the §1.4 **20% real-chosen-move** prior (a different
distribution; on an in-bounds real board the trunk-19 window covers everything →
~0% off-window). The security-relevant invariant — option (b) overflow is always
a **legal, empty, unique** cell and mass is conserved — **holds under load**.

---

## Attack 5 — grid-path regression canary

**Result: CLEAN.** `test_v6w25_microsmoke` (full v6w25 pool self-play: Python pool
+ Rust `worker_loop` + `ReplayBuffer`, NaN-free shape-checked rows) +
`test_inference_server` + `test_selfplay_encoding_aware` → **23 passed** under
`-W error::UserWarning`. The 8 warnings are **pre-existing** CNN TorchScript
`TracerWarning`s from `torch.jit.trace` — the trace path WP-3 wraps **verbatim** in
the non-graph `else` arm (the graph path drops trace entirely). No new warnings;
grid behavior signature unchanged.

---

## Attack 6 — determinism

**Result: IDENTICAL.** Same 26-position batch twice through the live seam (CPU,
eval, fixed seed): **max|Δ| = 0.0** across dense + value, **0** overflow-key
mismatches.

---

## Answers to the four required returns

- **Overall verdict:** **GAPS-FOUND** — all latent / by-design; **no WP-3 grid
  regression**. Structural contract dies loud, S2 correct, no waiter leak, grid
  byte-clean, deterministic.
- **SILENT finding (headline):** **G1 — model-output finiteness is unvalidated on
  BOTH paths.** NaN/Inf policy *or* value from `forward_batch` flows silently into
  MCTS (`w_value` NaN-poisons the tree); reachable under the graph loop's fp16
  CUDA autocast. The graph path's only would-be catch (`assemble` sum≈1) is a
  `debug_assert`, **compiled out of the release build** (G2). Contradicts the
  seam's literal "die-loud everywhere" claim; cheap always-on `isfinite` check
  closes it. Secondary silent: **G3** equal-legal-count mis-ordered submit swaps
  priors (latent — no production reorder path).
- **S2 result:** **SOUND.** Builder/`Board` `window_center` are the identical
  formula over the identical stone set (cannot drift through legal play); the
  correct frame is threaded and is load-bearing (wrong-center misreads by 5e-3,
  right-center 1e-8). Only residual: the coincidence `debug_assert_eq!` is
  release-inert (G4).
- **Waiter-leak result:** **NO LEAK.** collate-raise / forward-raise /
  thread-killing `BaseException` all release every waiter with no hang (the
  `finally: close()` → `close_rust` notifies each `graph_waiter.cv`). One
  theoretical lost-wakeup TOCTOU in the `closed`-check-then-`wait` sequence; not
  hit empirically.

## Reproduce
`PYTHONPATH=<worktree> .venv/bin/python /home/timmy/.claude/jobs/7d6e8877/tmp/attack_seam.py`
(A1/A4/A6), `.../attack_seam2.py` (A2/A3/A1d), `.../attack_1d_redo.py` (A1d
non-isomorphic). Grid canary:
`pytest tests/selfplay/test_v6w25_microsmoke.py tests/test_inference_server.py tests/test_selfplay_encoding_aware.py -W error::UserWarning`.
