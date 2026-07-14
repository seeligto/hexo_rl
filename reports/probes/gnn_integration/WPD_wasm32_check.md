# WP-D: wasm32-unknown-unknown compilability check — `hexo-graph` crate skeleton

**Program:** GNN-integration, R4 ratified (b+). **Work package:** WP-D (COND-4).
**Date:** 2026-07-14. **Verdict: WASM32-COMPILES.**

## Mission

Standing ruling: ONE Rust graph-builder crate compiled to BOTH native and
wasm32-unknown-unknown. WP-D's job was to prove wasm32 compilability of the
crate *skeleton* now — before the axis-graph builder (C1,
`docs/designs/gnn_integration_scope.md`) has real logic — so a dependency
poison (std-only dep, rayon pulled into the wasm feature set, a threading
assumption) surfaces as a one-line revert, not a redesign after the fact.

## What was built

- New crate `engine/hexo-graph/` (`engine/hexo-graph/Cargo.toml`,
  `engine/hexo-graph/src/lib.rs`). Added to the root workspace
  `Cargo.toml` `members` list. The `engine` crate's own `Cargo.toml` and
  build path are **untouched** — `engine` does not depend on `hexo-graph`,
  and `make build` invokes maturin directly on `engine/Cargo.toml`, so the
  new member adds zero build-risk to `make build` / `make bench`.
- Payload types only (structs, `pub const`s, one `todo!()` function
  signature) — no real builder logic. Types track
  `docs/designs/gnn_integration_scope.md` C1/C3/C4 field names
  (`node_feat`/`x (N,11)`, `edge_index (2,E)`, `edge_attr (E,5)`,
  `legal_mask`, `stone_mask`) plus the two names the mission brief called
  out that the scope doc doesn't yet name explicitly: `graph_offsets`
  (batched disjoint-union node-range boundaries, the Rust-side
  `_collate_gnn` equivalent) and `policy_scatter_index` (per-legal-node →
  dense-362 action-space re-projection, the Rust-side home for what
  `strix_v1_bot.py::get_move` does today in Python).
- Makefile target `check.wasm` (installs the wasm32 target idempotently,
  then runs the gated check) — this repo has no Rust CI workflow
  (`.github/workflows` is Claude-workflows-only), so **the gate is
  Makefile-based**, same as every other Rust check in this repo
  (`test.rust`, `bench`, etc.). `make help` lists it.

## Feature matrix

| Feature | Default? | Purpose | Deps pulled in |
|---|---|---|---|
| `native` | **yes** (in `default`) | self-play worker / offline corpus builder; future home for rayon/std::thread per-leaf parallel graph construction | none yet (skeleton) |
| `wasm` | no | browser inference companion to onnxruntime-web | none |
| `python` | no | PyO3 glue for the Python builder parity harness (`hexo_rl/probes/gnn_bc/*`, `scripts/research/gnn_infer_bench.py`) | `pyo3` (optional, version-pinned to match `engine/Cargo.toml`'s `0.28`) |

`wasm` and `python` are mutually exclusive by construction (pyo3 links
libpython and does not target wasm32-unknown-unknown) — enforced by neither
being in `default` and by `python`'s only dependency (`pyo3`) being
`optional = true`, gated `dep:pyo3`.

`std::thread` usage (currently just a placeholder `parallelism_hint()`, not
called by anything) is double cfg-gated:
`#[cfg(all(feature = "native", not(target_arch = "wasm32")))]` — belt and
suspenders so even a misconfigured build (`--features native --target
wasm32-unknown-unknown`) can't compile it in. This is the pattern the real
C1 builder must follow once it adds actual threading.

## Dependency list

Core path (native default build, `wasm` build): **empty**. The only
dependency in the manifest is `pyo3`, `optional = true`, gated behind the
non-default `python` feature — never pulled into a `native` or `wasm`
build. Confirmed by the check outputs below: the wasm32 check has nothing
to fetch or compile beyond the crate itself.

## Verification

All four checks run at `-j4` (thermal rule). Ran from repo root
`/home/timmy/Work/Hexo/hexo_rl/.claude/worktrees/gnn-integration`.

### 1. Native check (default features = `native`)

```
$ cargo check -p hexo-graph -j4
    Checking hexo-graph v0.1.0 (…/engine/hexo-graph)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 2.88s
```

### 2. wasm32-unknown-unknown check (the gate)

```
$ rustup target add wasm32-unknown-unknown   # was not installed; added
$ cargo check -p hexo-graph --no-default-features --features wasm \
    --target wasm32-unknown-unknown -j4
    Checking hexo-graph v0.1.0 (…/engine/hexo-graph)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.10s
```

Also verified via `make check.wasm` (idempotent `rustup target add` +
the same `cargo check` invocation) — same clean result.

### 3. `engine` crate untouched

```
$ cargo check -p engine -j4
    Checking engine v0.1.0 (…/engine)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 7.07s
```
No warnings, no errors, no changes to `engine/Cargo.toml` or any file under
`engine/src/`. `engine` does not reference `hexo-graph` anywhere.

### 4. Bonus: `python` feature check (not required by the mission, run for
extra confidence since the feature exists in the manifest)

```
$ cargo check -p hexo-graph --no-default-features --features python -j4
    Checking pyo3 v0.28.3
    Checking hexo-graph v0.1.0 (…/engine/hexo-graph)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 4.60s
```

### 5. Bonus: `cargo test -p hexo-graph` (native, default features)

```
running 2 tests
test tests::payload_types_default_construct ... ok
test tests::build_axis_graph_is_unimplemented_by_design - should panic ... ok
test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```
(`make test.rust` runs plain `cargo test` over the whole workspace, so
these two cheap tests now ride along with the existing `engine` test suite
automatically — no separate invocation needed going forward.)

## Verdict detail

**WASM32-COMPILES.** No blocked dep, no redesign needed. The skeleton's
type boundary (flat `Vec<f32>`/`Vec<u32>`/`Vec<bool>` structs, no
`std::thread`/rayon/pyo3 in the shared core path) is wasm32-clean by
construction — this was a low-risk skeleton (no external deps to poison
it), so the main value of this check is establishing the feature-gate
*pattern* and the two-target verification *habit* before C1 adds real
logic (candidate poison sources for that future PR: rayon for parallel
per-leaf graph construction, `dashmap`/`fxhash` if the builder reuses
`engine`'s hashing helpers instead of reimplementing them, any use of
`std::time::Instant` — wall-clock timing is unsupported the same way on
plain wasm32-unknown-unknown without a JS shim).

## Files written / modified

- **New:** `engine/hexo-graph/Cargo.toml`
- **New:** `engine/hexo-graph/src/lib.rs`
- **New:** `reports/probes/gnn_integration/WPD_wasm32_check.md` (this file)
- **Modified:** `/Cargo.toml` (root workspace `members` + explanatory
  comment; `[profile.release]` and `exclude` untouched)
- **Modified:** `Makefile` (added `check.wasm` target under the existing
  `# ── Tests ──` section, between `test.slow` and the `# ── Benchmarks ──`
  section; listed by `make help` via the standard `##` comment convention)
- **Untouched:** `engine/Cargo.toml`, everything under `engine/src/`,
  everything under `docs/designs/` (per instruction — other agents own
  those concurrently), `.github/workflows/` (confirmed no Rust CI exists
  there; none added — Makefile is the gate).

## Carried design notes for the future browser path (verbatim per dispatch)

- ONNX opset ≥18 (export clean after data-dependent-guard strip, D-N
  verified; WP-A found ORT-CUDA Expand E×H memory blow-up — keep op-set
  minimal and avoid Expand-materializing patterns).
- onnxruntime-web WASM SIMD+threads in a web worker.
- COOP/COEP required for prod repo.
- AMD note: ORT ROCm EP removed ≥1.23 — MIGraphX or WGPU-Vulkan for AMD
  inference.

These are browser/onnxruntime-web-path notes, not `hexo-graph`-crate
findings — carried here because WP-D is the wasm-path work package and this
is the natural place for the next agent to find them alongside the
compilability verdict. WP-A's full finding (self-play inference rides
torch-CUDA; ORT/wasm is browser-only) is in
`reports/probes/gnn_integration/WPA_cuda_bench.md`.
