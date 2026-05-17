# Wave 10 Batch B — IMPL Recon (P69 inline-test scaffold)

**Cycle:** 3
**Wave:** 10
**Batch:** B — P69 inline-test scaffold for `InferenceBatcher` early-return paths
**Entry HEAD:** `f53975e` (Wave 10 Batch A close — worker_loop split into 7 sibling modules)
**Subagent:** IMPL Batch B
**PERF_SENSITIVE:** NO (additive `#[cfg(test)]` — compiled out of release builds; production surface untouched)
**Operator-resolved decisions:** U9 = Option B.2.c (minimum-viable scaffold; `inference_bridge.rs` only); U8 = DEFER worktree-parallel pilot (sequential after Batch A)

---

## §0 — Recon evidence

### 0.1 — PREP §B.1 verification re-confirmed at HEAD `f53975e`

`grep -n '#\[cfg(test)\]\|#\[test\]\|mod tests' engine/src/inference_bridge.rs` → **EMPTY.**

P69 inline-test coverage gap is at-zero. The Batch A split did NOT touch `inference_bridge.rs` (verified via `git show --stat f53975e | grep inference_bridge` → no hits).

### 0.2 — Line-number verification (PREP §B.2 + §C.1 anchors)

| PREP cite | Anchor | HEAD `f53975e` actual | Delta |
|---|---|---|---|
| `:181-184` | `submit_batch_and_wait_rust` closed-channel guard | L182-184 (`if self.inner.closed.load(...) { return Err(()); }`) | 0 (PREP cited 181-184 inclusive of the fn-signature line above) |
| `:194-196` | length-mismatch guard inside per-features loop | L194-196 (`if features.len() != self.feature_len { return Err(()); }`) | 0 |
| `:178-180` | `submit_batch_and_wait_rust` signature | L178-181 (4-line wrap) | trivial |

Both early-return paths exist at the cited locations; substring `return Err(())` confirmed via direct file read.

### 0.3 — `InferenceBatcher` ctor surface for tests (PREP §B.3 / Option B.2.c)

- `pub fn new(encoding_spec: Option<PyRegistrySpec>, feature_len: Option<usize>, policy_len: Option<usize>, pool_size: Option<usize>) -> PyResult<Self>` is the only ctor (`#[new]` PyO3 + plain `pub fn`).
- **PyO3-free ctor path:** `InferenceBatcher::new(None, Some(feature_len), Some(policy_len), None)` — explicit-kwarg arm at `inference_bridge.rs:297` (`(Some(f), Some(p), _) => (f, p)`) skips the `encoding_spec` dereference entirely. This is the exact pattern `engine/tests/batcher_default.rs:12` uses for a pure-Rust integration test. Pure `#[cfg(test)] mod tests` inside `inference_bridge.rs` will work end-to-end without any Python runtime.
- `close_rust(&self)` is `pub(crate)` — accessible from the same crate's `#[cfg(test)] mod tests` via `super::*` (since the test module is INSIDE `inference_bridge.rs`, `super::*` brings `InferenceBatcher` + `pub(crate)` methods into scope).
- `submit_batch_and_wait_rust(&self, batch_features: Vec<Vec<f32>>) -> Result<Vec<(Vec<f32>, f32)>, ()>` is `pub(crate)` — same accessibility from inline tests.

**Verdict:** inline `#[cfg(test)] mod tests` at end of `inference_bridge.rs` works without any visibility changes. No `pub(crate)` upgrades, no fallback to integration-test pattern.

### 0.4 — Baseline test count at HEAD `f53975e`

`cargo test --package engine --release` → **274 passed** (verified). Matches Batch A close baseline.

---

## §1 — SD4 corrections

| # | PREP claim | Reality at HEAD `f53975e` | Action |
|---|---|---|---|
| 1.1 | PREP §B.2 cites `:181-184` for closed-channel guard | Closed-channel guard sits at L182-184; PREP cite is fn-start-inclusive (181 = fn body open brace). Cosmetic 1-line drift. | No-op. |
| 1.2 | PREP §B.2 sketch shows test stubs but does not verify the PyO3-free ctor path works | Verified: `InferenceBatcher::new(None, Some(f), Some(p), None)` is PyO3-free per ctor `(Some(f), Some(p), _) => (f, p)` arm — `engine/tests/batcher_default.rs:12` precedent. | Use the same kwarg pattern. |
| 1.3 | PREP §B.3 acknowledges the closure-vs-fn problem for `infer_and_expand` and defaults to Option B.2.c (test only `inference_bridge.rs`) | Confirmed; operator binding decision U9 = B.2.c. | Land 2 tests in `inference_bridge.rs` only. Defer `infer_and_expand` test target to a future opportunistic wave; disclose in commit body. |

**No fragile-claim escalations. All PREP claims either confirmed or trivially-corrected (1-line drift / verification follow-on). Scope unchanged.**

---

## §2 — Final test design

### 2.1 — Test fixture

Pure-Rust `#[cfg(test)] mod tests` block appended at end of `engine/src/inference_bridge.rs`. Tests use `super::*` to bring `InferenceBatcher` into scope; ctor pattern matches `engine/tests/batcher_default.rs` (explicit `feature_len` + `policy_len` kwargs → PyO3-free arm at L297).

Geometry: `feature_len = 8 * 19 * 19 = 2888`, `policy_len = 19 * 19 + 1 = 362`. Matches v6 default + existing test precedent; arbitrary values would work too (the tests exercise control-flow, not geometry).

### 2.2 — Cell 1: `submit_batch_and_wait_rust_returns_err_when_closed`

**Path under test:** `inference_bridge.rs:182-184` — `if self.inner.closed.load(Ordering::SeqCst) { return Err(()); }`.

**Mechanism:**
1. Construct batcher via PyO3-free kwargs.
2. Call `batcher.close_rust()` to set `inner.closed = true` (legitimate public-crate API — no test-only field-poke required).
3. Call `submit_batch_and_wait_rust(vec![vec![0.0; 2888]])`.
4. Assert returned `Result` is `Err(())`.

**LOC budget:** ~10 lines including assertion.

### 2.3 — Cell 2: `submit_batch_and_wait_rust_returns_err_on_length_mismatch`

**Path under test:** `inference_bridge.rs:194-196` — `if features.len() != self.feature_len { return Err(()); }` inside the per-features loop.

**Mechanism:**
1. Construct batcher via PyO3-free kwargs with `feature_len = 2888`.
2. Submit `vec![vec![0.0; 2889]]` (one extra element → length mismatch).
3. Assert returned `Result` is `Err(())`.

**Note:** the length-mismatch guard sits INSIDE a `self.inner.queue.lock()` scope, so the test exercises that the early-return drops the lock guard correctly via RAII. No deadlock observable from outside.

**LOC budget:** ~8 lines including assertion.

---

## §3 — File-touch manifest

| File | Operation | LOC added (est.) |
|---|---|---:|
| `engine/src/inference_bridge.rs` | MODIFY — append `#[cfg(test)] mod tests { ... }` block at end | ~25 (2 tests × ~10 LOC + module-block boilerplate) |
| `audit/rust-engine/cycle_3/wave_10/B_recon.md` | NEW (this file) | n/a (audit doc) |

**Per U9/Option B.2.c: `engine/src/game_runner/worker_loop/inner.rs` is NOT touched.** Deferred `infer_and_expand` test target disclosed in commit body per master prompt.

**No production logic touched. No fn signatures changed. No `pub`/`pub(crate)` visibility upgrades. No `#[inline]` added.**

---

## §4 — Commit-body preview

```
refactor(engine): add P69 inline test scaffold for InferenceBatcher early-return paths (cycle 3 Wave 10 Batch B)

PURPOSE
- minimum-viable P69 inline-test scaffold per operator U9 = Option B.2.c.
- 2 #[cfg(test)] tests in engine/src/inference_bridge.rs covering
  submit_batch_and_wait_rust early-return paths:
    1. returns Err when batcher is closed (close_rust path; L182-184)
    2. returns Err on feature length mismatch (per-features loop; L194-196)
- establishes inline-test pattern for inference_bridge.rs; full P69
  coverage (incl. infer_and_expand partial-batch handler in
  worker_loop/inner.rs) deferred to a future opportunistic wave —
  requires closure-vs-fn extraction decision; not in scaffold scope.

OPERATOR-RESOLVED DECISIONS
- U9: Option B.2.c (minimum-viable scaffold; inference_bridge.rs only).
- U8: DEFER worktree-parallel pilot; Batch B is sequential after Batch A.

NO `!`-MARKER
- additive test-only code (#[cfg(test)] excluded from release builds).
- production surface untouched.
- no PyO3, no Python, no behavior change.

SD4 CORRECTIONS
- PREP §B.2 cited :181-184 for closed-channel guard; HEAD actual is
  L182-184 (fn signature occupies L178-181). Cosmetic 1-line shift; no
  scope impact.
- PREP §B.2 sketch did not verify the PyO3-free ctor path; verified
  here: InferenceBatcher::new(None, Some(f), Some(p), None) takes the
  explicit-kwarg arm at inference_bridge.rs:297 and skips the
  encoding_spec dereference entirely. Pattern mirrors
  engine/tests/batcher_default.rs:12.

VERIFICATION
- cargo build --release: clean
- cargo test --package engine: 276 passed (was 274; +2 P69 tests)
- pytest tests/: unchanged from Batch A close
- cargo clippy --release: no new warnings (floor 42 unchanged)

Files: 1 changed (+~25 / -0)
```

---

*End of B_recon.md.*
