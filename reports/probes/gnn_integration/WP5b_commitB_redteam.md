# WP-5b COMMIT B — RED-TEAM (distinct from reviewer; assume-guilty)

**Verdict: GAPS-FOUND.** One **CRITICAL launch-blocker** (BREAK-1: run4's fp16 regime
hard-crashes the graph train-step on step 0), one confirmed minor break (BREAK-2:
reviewer-F2 export truncation blind spot, reproduced — skip-count integrity violated),
one INFO (warm-start↔resume silent-skip). Everything else HELD under weaponized probes.

Red-team worktree HEAD `a1549f0`, commit-B uncommitted. Source byte-identical at end
(`git status` == entry state; HEAD unchanged; probes in scratch `wp5bB_redteam/`, no temp
Rust file, no source edited, report NOT git-added). Extension freshly built (Jul-15 19:51,
`sample_graph_batch(recent_frac=…)` present). CUDA available — fp16 path tested on real GPU.

Reviewer (`WP5b_commitB_review.md`) verdict was PASS w/ F1–F5 LOW/INFO. **The reviewer's own
32-green ran the graph branch ONLY under `fp16: False`** — the exact hole BREAK-1 lives in.

---

## BREAK-1 (CRITICAL — LAUNCH-BLOCKING) — graph train-step crashes under fp16 autocast

**The production run4 regime does not train a single step.**

### Repro (end-to-end, real Trainer, real GPU, exact run4 config)
1. run4_gnn.yaml merged through the REAL chain (`load_train_config(variant="run4_gnn")`,
   validator passes) → **`fp16=True`, `amp_dtype=fp16`** (base `configs/training.yaml:4,15`;
   run4_gnn.yaml does NOT override — confirmed via merge, `probe_yaml_merge.py`).
2. `_train_on_graph_batch` wraps forward **and both losses** in
   `with autocast(device_type=…, dtype=self.amp_dtype, enabled=self.fp16)` (`trainer.py:681`).
   `GnnNet.forward_batch` does NOT disable autocast internally → `policy_logits` come out **fp16**.
3. `ragged_policy_ce` → `segment_softmax` (`graph_collate.py:588`):
   `ex = torch.exp(logits - seg_max[seg])`. Under autocast `torch.exp` is on the fp32 cast-list
   → **`ex` is fp32**, but `denom = torch.zeros(b, dtype=logits.dtype)` is **fp16** →
   `denom.scatter_add_(0, seg, ex)` raises:

   `RuntimeError: scatter(): Expected self.dtype to be equal to src.dtype`

   Reproduced through `Trainer.train_step(HexgBuffer, …)` with `fp16=True` on CUDA — **crashes on
   step 0, deterministically** (dtype mismatch is value-independent). `probe_trainer_fp16.py`.
4. Second, independent fp16 failure on the same path (surfaces if the dtype mismatch were fixed
   naïvely without full fp32): with fp16 `policy_logits`, `ragged_policy_ce`'s
   `torch.log(probs.clamp(min=1e-12))` → **`1e-12` underflows to 0 in fp16**, so the clamp is a
   no-op, `log(0) = -inf`, and `target·(-inf)` with a 0-target node = **NaN loss**
   (`probe_fp16_ragged.py`: fp16 tensors → `loss: nan`).

### Root cause
`ragged_policy_ce`/`segment_softmax` are the **only** loss path with **no `.float()` cast**.
Contrast: `binned_value_loss` casts `bin_logits.to(torch.float32)` (`binned_value.py:49`); the
dense CNN losses `.float()` their inputs (`losses.py:211,218,223,259`); `F.log_softmax` (dense
policy) is autocast-safe by construction. The GNN policy loss inherited none of these guards.

### Why every test missed it
`tests/training/test_gnn_train_step.py:68` and the graph test in
`tests/test_training_loop_event_schema.py:186` both hardcode **`"fp16": False`**. There is NO
graph train-step test under fp16. The reviewer re-ran the same fp16=False suites (32 green) — the
production regime is entirely untested.

### Severity
CRITICAL. run4 launches with `fp16=True`; the first `_train_on_graph_batch` raises. This is the
one commit-B leg that "trains end-to-end" — it does not, under the shipped config.

### Fix direction (surface tradeoff, do not defer)
- **Correct fix:** cast to fp32 inside the loss — `policy_logits = policy_logits.to(torch.float32)`
  at the top of `ragged_policy_ce` (mirror `binned_value_loss`), or wrap the segment-softmax+log in
  `with torch.autocast(enabled=False):`. Fixes BOTH the scatter crash AND the log-underflow NaN, and
  keeps fp16 throughput on the matmuls (forward stays fp16; only the small reduction is fp32).
- **Do NOT** just set `fp16: false` in run4_gnn.yaml — it hides the latent NaN, leaves a fp16-graph
  landmine for any future run, and forfeits the OQ-2 5080 throughput the whole GNN case rides on.
- **Test debt:** add a `fp16=True` graph train-step test (CUDA-gated) — the missing regime is the
  entire reason this shipped green.

---

## BREAK-2 (LOW–MEDIUM — confirmed) — export truncation blind spot + skip-count lies (reviewer F2, weaponized)

### Repro (`probe_export_f2.py`)
Synthetic 20-move record, plies 0–18 legal, ply-19 = duplicate of an occupied cell (illegal),
`expected_last_ply = min(20,150)-1 = 19`:
- `replay_positions` yields the position for ply 19 **before** `apply_move` raises (yield-then-apply),
  so the yielded `move` IS the illegal cell.
- `replay_game`'s truncation proxy `positions[-1].ply < expected_last_ply` is `19 < 19` = **False**
  → **NOT detected** → returns 18 rows including a one-hot policy target on the **occupied (illegal)
  cell**.
- **No skip counted** (`stats.n_games_skipped` stays 0 for this game — the export summary LIES).
- `push_graph_position` **accepts** it (push validates finite outcome / finite≥0 prob / ±1 player
  only — NOT visit-cell legality).
- Crash is **DEFERRED to training-sample time**: sampling that slot trips the mass-drop guard LOUD
  (`HEXG sample: visit mass dropped … illegal/off-window visit coord`).

### Severity
LOW–MED. Fail-loud, but the failure moves from export (cheap, one-shot) to a **non-deterministic
mid-run training abort** (fires whenever the poisoned slot is first sampled — step 5 or step 50k),
and a poisoned row reaches the persisted `.hexg`. Narrow trigger: an illegal move at exactly the
window-boundary ply; run4's curated six-in-a-row human corpus makes it rare (last move is the legal
winning move). But the export is a general tool and its skip-count is an integrity number that
under-reports here.

### Fix direction
Gate the built row on visit-cell legality (cell ∉ stones) before push, OR make the truncation proxy
also assert the last position's move is a legal cell. Cheap (~4 lines), converts a deferred
training-crash into an export-time LOUD-skip-with-count (the stated §5.4 contract).

---

## INFO-1 — warm-start ↔ `--checkpoint` collision is a silent skip

`maybe_warmstart_gnn_from_bc` fires ONLY in `init_trainer`'s fresh (`else`) branch (orchestrator
diff). Launch with `--checkpoint <ckpt>` **and** `gnn_warm_start.enabled: true` (e.g. resuming run4
from a mid-run ckpt with the SAME run4_gnn.yaml) → the resume branch runs, warm-start is **silently
skipped**, no warning that a declared config knob was ignored. Behavior is CORRECT (resume weights
must win), but silent — and run2/run3 history shows resumes-with-same-yaml are routine. One-line
warn ("gnn_warm_start.enabled ignored on a --checkpoint resume") closes it.

---

## ADV-GRADUATION LIST (attacks that HELD)

**Recency × wrap × resize (surface 2)**
- `recent_frac=0.0` byte-identical to pre-commit-B (Rust `…zero_frac_is_byte_identical…`) — HELD.
- `recent_frac=1.0` draws newest slots; 700→600 wrap; clamp-before-fill (`size<window`) — HELD
  (3 Rust tests pass here; `probe_recency_persist.py` case A → 1.0).
- persist→save→load→`recent_frac=1.0` draws newest (load sets `head=size%cap`, records linearized
  oldest→slot0, newest survives ordering) — HELD (case C → 1.0).
- resize-UP then recency: newest survives, but `recent_window=min(size, new_cap/2)` means the whole
  pre-grow set counts as "recent" until refill → recency temporarily dilutes to ~0.5 (case B → 0.55).
  Accepted approximation (standing §8.5), NOT wrong — **noted** as a transient post-grow recency dip.
- resize-DOWN: API rejects (`new_capacity ≤ capacity` → PyValueError) — N/A by construction.
- retry-exhaustion under-fill: `sample_indices` always returns exactly `batch_size` (in-place
  replacement, never removal); untagged `-1` game_ids skip the dedup entirely — batch never
  under-fills — HELD.
- recency threads end-to-end (NOT a dead knob): config `recency_weight=0.75` → trainer forwards
  `recent_frac=0.75` to `sample_graph_batch` (spy-verified, `probe_yaml3.py`) — HELD.

**Train-step ragged payloads (surface 1)**
- No second entry point: `sample_graph_batch`/`_train_on_graph_batch` have a single caller
  (`train_step` `isinstance(HexgBuffer)` dispatch); collate runs `semantic="full"` (18-assertion
  set) every batch — no wire consumed before/without the resolver — HELD.
- NaN/Inf past guards: push validates finite outcome + finite≥0 visit prob + ±1 stone player before
  any mutation (`push.rs`); the sole buffer entry is `push_graph_position` (worker loop + export both
  use it) — NaN cannot reach the loss via data. Finite-extreme outcome clamped by
  `scalar_to_two_hot`; finite-extreme visit prob stays finite through CE — HELD.
- all `value_valid=0` batch → `binned_value_loss` returns `zeros(())` (no div-by-zero), value head
  gets no grad, policy stays connected, loss finite — HELD.
- all quick-search batch → `ragged_policy_ce` returns 0 (connected via ×0, keeps grad_fn), finite — HELD.
- both-degenerate batch (all value-invalid AND all quick-search) → loss=0, grad_norm=0, finite; NO
  "does not require grad" backward crash (the ×0 policy path preserves grad_fn) — HELD
  (`probe_degenerate.py`).

**Warm-start (surface 4)** — all raise LOUD (`probe_warmstart.py`)
- missing policy_head keys → RuntimeError (key mismatch); trunk shape mismatch → RuntimeError (size
  mismatch, even under strict=False); grid ckpt on graph run → RuntimeError (key mismatch); graph
  guard (warm-start on grid spec) → ValueError; checkpoint unset → ValueError; default-OFF → False
  no-op. Real `gnn_bc_040000.pt` happy path transfers 46/46 tensors, landed-verify fires. — HELD.

**YAML / merge (surface 5)** — `probe_yaml_merge.py`, `probe_yaml3.py`
- run4 merges clean through the real base+variant chain + variant-validator; `fp16=True`(!),
  `recency_weight=0.75`, `value_head_type=dist65`, `in_channels=0`, levers as declared.
- `value_head_type: scalar` user/CLI override → **RepresentationMismatch (LOUD)** at build_net
  (declared-wins semantics correctly fail-loud, not silent scalar head) — HELD.
- `in_channels=0` vs base `8`: scattered-key gate satisfied by the explicit override; inert — HELD.

**Monitoring (surface 6)** — `probe_yaml3.py`
- Negative direction: `emit_training_events` **direct-indexes** `loss`/`policy_loss`/`value_loss` →
  **KeyError (loud)** on each missing key. The contract test drives the REAL `train_step` → REAL
  `emit_training_events`, so a trainer regression that drops a key is caught at runtime — HELD.

**Export sha/held-out gates (surface 3)** — 8 export tests green in this worktree (incl.
manifest-mismatch hard-fail, held-out-input hard-fail, output-sha collision, malformed LOUD-skip);
the manifest/held-out gates fire. Only the F2 boundary case (BREAK-2) escapes.

---

## Test evidence (re-run by red-team)
- `pytest test_gnn_train_step + test_gnn_bc_warmstart + test_gnn_hexg_corpus_export +
  test_training_loop_event_schema -m "not slow"` → **28 passed** (all fp16=False).
- `cargo test -p engine --lib recency` → **3 passed** (byte-identical, newest-slot, clamp).
- End-to-end fp16 crash reproduced on CUDA through the real `Trainer`.

## One-line summary for the caller
Ship-blocked: BREAK-1 (fp16 graph train-step crash on step 0 — the run4 config's own regime, untested
because every graph test forces fp16=False) must be fixed before launch; fix by casting the graph
policy loss to fp32. BREAK-2 (export F2) and INFO-1 (warm-start resume silent-skip) are cheap
follow-ons. All other weaponized attacks HELD.
