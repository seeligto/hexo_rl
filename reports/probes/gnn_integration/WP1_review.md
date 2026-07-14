# WP-1 REVIEW — fresh-eyes pass on commit `bfca786`

**Reviewer role:** WP-1 REVIEW (independent, read-only on source). **Scope:**
`engine/hexo-graph/src/lib.rs`, `src/bin/harness.rs`, `benches/build_bench.rs`,
`tests/test_hexo_graph_parity.py`, vs `hexo_rl/bots/strix_v1_graph.py` (oracle) and
`docs/designs/gnn_ragged_contract_v1.md` (amended contract).

**Verdict: REVIEW-FIXES-REQUIRED.** Core builder logic, wasm/clippy discipline, and bench
methodology are sound and I independently re-verified the headline parity claim (see below). One
MUST-FIX (a real, frequently-hit board state — game start — is never checked against the oracle)
and several SHOULD-FIX items (the producer-side contract mirror is dormant in the exact build
profile production runs in; one wire type silently drifted from the ratified contract) keep this
short of a clean pass.

---

## Independent verification performed (not just re-reading claims)

- `cargo test -p hexo-graph` — 5/5 pass.
- `cargo clippy -p hexo-graph --all-targets --features harness` (forced rebuild, not cached) —
  **0 warnings**, confirms the commit-message claim.
- `cargo build --release -j4 -p hexo-graph --features harness` then
  `cargo check -p hexo-graph --no-default-features --features wasm --target wasm32-unknown-unknown`
  — both clean, confirms wasm-GREEN claim.
- `RUSTFLAGS="-C panic=unwind" cargo bench -j4 -p hexo-graph --features harness` — reproduced
  ~926–945 µs/pos (report claims ~929 µs/pos) — **within budget, claim confirmed**.
- `.venv/bin/python -m pytest tests/test_hexo_graph_parity.py -v -s` — **reproduced**:
  `n=1588 positions, 17126772 edges, max_int_diff=0, max_float_diff=0.00e+00`. The byte-exact
  parity claim is real, not just asserted in the report.
- Empirically confirmed (via `rustc -v` invocation dump + a standalone `cfg!(debug_assertions)`
  probe under matching profile flags) that `-C debug-assertions` is **never explicitly set** and
  the workspace `[profile.release]` (`opt-level=3`, no override) means `cfg!(debug_assertions) ==
  false` in the exact release build the harness/parity run and any production self-play run use.
  This underpins finding #1.

---

## Parity-test coverage enumeration (priority 1)

**COMPARED, byte-exact or ≤1e-6, against the real oracle or a shared/tested Python geometry fn:**
`num_nodes`, `n_nodes_checksum` (vs recomputed `num_nodes`), `n_stones`, `edge_src`/`edge_dst`
(exact order, not set-equality), `legal_mask`, `stone_mask`, `node_coords`, `current_player`,
`node_feat` (float ≤1e-6, oracle f64→f32 cast to match the deploy path), `edge_attr` (same),
`window_center` (vs `hexo_rl.diagnostics.forced_win_detector.window_center` — a real, shared,
tested production geometry function, not a test-local reimplementation).

**COMPARED but NOT against an independent oracle value (weaker):**
- `legal_node_gather` — checked via the formula `[n_stones+j for j in range(len(g["legal_coords"]))]`,
  i.e. it verifies the contiguous `[stones|legal|dummy]` layout invariant, not a field the oracle
  itself emits.
- `policy_dst_slot` — checked against `_canonical_slot()`, a **third, test-local** reimplementation
  of `core.rs::window_flat_idx_at_geom`'s formula. `build_axis_graph_raw` has no concept of a dense
  action slot at all (WP1_builder.md §4.1 is honest about this — "no oracle in
  `build_axis_graph_raw`"), so this is parity between two independently-written from-spec
  implementations (Rust builder vs. test helper), not oracle-verification. A bug shared by both
  from misreading the spec would not be caught. (`is_off_window()` in `forced_win_detector.py`
  calls the *actual* live `board.to_flat(...)` PyO3 binding for the same purpose elsewhere in the
  codebase — that would have been a stronger, already-existing oracle to route through instead of
  re-deriving the formula inline in the test.)
- `builder_impl` — only checked `== 1` (tautological; the oracle has no such field).

**NOT COMPARED / correctly out of scope for a single-leaf (non-batched) builder:** `node_offsets`,
`edge_offsets`, `legal_offsets`, `contract_version`, cross-graph boundary checks, dtype-on-wire
checks, `AugRoundTripMismatch` (no augmentation exists in this commit — that's WP-3/HEXG). `n_edges`
is emitted by the harness but never asserted (redundant with the `edge_src`/`edge_dst` equality
already checked).

**Degenerate-case coverage, in the 1588-position set:**
| case | covered? |
|---|---|
| empty board (0 stones) | **NO** — see finding #2 |
| 1 stone | yes (3 instances via prefix truncation) |
| terminal (`current_player=None` in the oracle) | not separately exercised, but the oracle's `None` branch is code-identical to its `-1` branch (`x==1 else -1`) which IS exercised 800× — no incremental oracle behavior exists to miss; accepted per WP1_builder.md §4.3 |
| duplicate-coord input | **NO** in the cross-language set — oracle's dict interface structurally can't represent duplicates, so a true Rust-vs-Python check is impossible by construction; validated only same-side (Rust vs Rust) in `duplicate_coord_dedups_last_wins`. Documented, accepted (§4.4). |

**What the augmentation scheme does NOT cover:** every position is built at exactly ONE
geometry parameterization (`win_length=6, radius=6, trunk_size=19` — module-level constants in
`test_hexo_graph_parity.py`, never varied by `_build_inputs()`) even though `BuildParams`/
`build_axis_graph` and the harness JSON schema are fully parameterized per-position. Prefix
truncations are never combined with the player/moves swap (partial boards are only tested with the
original side to move). See findings #12/#13.

---

## Findings

### 1. [SHOULD-FIX] Producer-side contract mirror is compiled out in the exact profile production runs use
`engine/hexo-graph/src/lib.rs:672` — `debug_assert_contract` is gated `if cfg!(debug_assertions)`.
The workspace `[profile.release]` (root `Cargo.toml:21-28`) sets no `debug-assertions` override, so
`cfg!(debug_assertions) == false` there (verified empirically above). Consequence: the release
harness binary used to *produce* the "1588 positions, byte-exact" headline number never ran this
check — the byte-exactness rests entirely on the external Python-side asserts, which is fine, but
the "producer-side contract invariants... die LOUD" defense this whole WP exists to add (mirroring
`gnn_ragged_contract_v1.md` §2.5) will also be silently absent from every real self-play/training
run, which always builds `--release` (perf discipline, `docs/rules/perf-targets.md`). The in-code
doc comment (lib.rs:667-669, "dies LOUD in debug + test builds") is honestly scoped, and this
matches an established repo-wide `debug_assert!` convention (38 uses in `engine/src`) — so this is
not a novel mistake. But the contract doc's own framing (§2.5: "a mirror in the Rust producer's
debug-assert") implies defense-in-depth in production, which this doesn't deliver. **Fix:** either
(a) make the check unconditional — cost is trivial (~2932 edges/graph vs. 930 µs/pos build time,
sub-percent overhead), or (b) explicitly state in WP1_builder.md that the Rust-side mirror is
debug/test-only and production semantic-layer coverage is 100% the Python resolver's job (no
Rust-side defense in depth) — today the report's wording ("fires in debug/test builds... Oracle
divergence dies loud") reads as reassuring without spelling out the release-mode gap.

### 2. [MUST-FIX] Empty board (game start, ply 0) never checked against the oracle
`tests/test_hexo_graph_parity.py::_build_inputs()` (lines 65-86) draws only from
`wpa_positions.json`, whose minimum stone count is 2 (verified: `min(len(p["stones"]) for p in
positions) == 2`), and its own augmentation (player/moves sweep, prefix truncation with `k>=1`)
never produces a 0-stone input. The empty board is the actual pre-move-1 state of every single game
— not a hypothetical edge case. It IS covered by a same-side Rust-only unit test
(`empty_board_builds_dummy_only`, lib.rs:737-745), but that only proves the Rust builder is
internally consistent with itself, not that it matches `build_axis_graph_raw({}, ...)`. **Fix:**
add `{"stones": [], "current_player": ±1, "moves_remaining": 1|2}` to `_build_inputs()` (or a
standalone parametrized case) and assert against the real oracle call.

### 3. [SHOULD-FIX] `policy_dst_slot` wire width silently drifted from the ratified contract
`gnn_ragged_contract_v1.md` §2.1/§2.2 rules `policy_dst_slot` as `u16` ("u16 is correct and halves
its H2D" — an explicit, reasoned decision). `engine/hexo-graph/src/lib.rs:135`
(`PolicyScatterIndex(pub Vec<u32>)`) and `lib.rs:61` (`OFF_WINDOW_SLOT: u32 = u32::MAX`) use `u32`.
Valid slots are `0..361` and the off-window sentinel would work identically as `u16::MAX` — nothing
about the domain requires the wider type, and no comment explains the widening. This is exactly the
class of silent contract drift `gnn_ragged_contract_v1.md` was written to prevent (it is not a
runtime bug — it's a local, per-leaf representation choice WP-3 will need to remember to narrow —
but leaving it as-shipped means the contract's own documented rationale for `u16` is now
contradicted by the reference implementation). **Fix:** `Vec<u16>` / `OFF_WINDOW_SLOT: u16 =
u16::MAX`.

### 4. [SHOULD-FIX] `debug_assert_contract` mirrors only a subset of the leaf-checkable §2.5 assertions
Of the contract's 18 named checks (13 structural + 4 semantic + 1 handshake), roughly 9 are
meaningful at single-leaf (non-batched) scope — the rest (`BatchCountMismatch`,
`OffsetsNonMonotonic`, `EdgeCrossesGraphBoundary`, `ScatterGatherCrossesGraph`,
`GraphContractVersionMismatch`, `DtypeMismatch`, `AugRoundTripMismatch`) require batch/wire context
that doesn't exist until WP-3. `debug_assert_contract` (lib.rs:671-703) mirrors: `NodeCountChecksum`,
the edge_attr/node_coords length checks, `EdgeIndexOutOfBounds`, `GatherNotLegalNode`, and
`ScatterSlotCanonicalMismatch`+`ScatterSlotOutOfBounds` (combined into one block). **Missing, though
fully leaf-computable and cheap (~2932 edges):**
- `ScatterSlotAliasing` — two legal nodes in the same graph mapping to the same `policy_dst_slot`
  (contract's ADV-2b). One pass over `policy_scatter_index` (excluding the sentinel) with a seen-set.
- `EdgeAttrGeometryMismatch` — the contract's own headline semantic finding (ADV-8): recompute each
  edge's expected `axis_onehot`/`signed_dist`/`src_player` from `node_coords` + `current_player` and
  compare. This is the single check the red-team called out as defeating a purely structural
  validator, and it's absent from the producer mirror.
- `EmptyLegalSet` — a graph with zero legal moves (unlikely on real boards but checkable and named).

### 5. [NIT] `legal` Vec not capacity-reserved, contradicting the report's own claim
`engine/hexo-graph/src/lib.rs:368` (`legal_moves_from_stones`) — `let mut legal: Vec<(i32,
i32)> = Vec::new();`, grown only by repeated `.push()`. The sibling `seen` set two lines above IS
reserved (`stones.len() * offsets.len()`). WP1_builder.md §2 states "every `Vec` pre-sized from
stone-count bounds" — this one isn't. **Fix:** `Vec::with_capacity(stones.len() * offsets.len())`
(same upper bound as `seen`) or `Vec::with_capacity(seen.capacity())`.

### 6. [NIT] Dedup packed key gives `src` only 30 of its 32 bits
`engine/hexo-graph/src/lib.rs:652` — `(u64::from(src[rd]) << 34) | (u64::from(dst[rd]) << 2) |
u64::from(axis)`. `dst` gets the full 32-bit range (`<<2` leaves bits 2-33 free); `src` only gets
bits 34-63 (30 bits) — the top 2 bits of a `src >= 2^30` would silently fall off the top of the u64
and could alias another key. No real exposure today (self-play graphs cap at ~900 nodes,
WP-A-measured), and it's implicitly documented ("src/dst < ~1000 nodes") — but the asymmetry between
src's 30-bit budget and dst's 32-bit budget isn't called out specifically. A
`debug_assert!(src[rd] < (1 << 30) && dst[rd] < (1 << 30))` would make the real limit explicit
instead of incidental.

### 7. [NIT] Threat-feature stack array size relies on an incidental panic, not an explicit assert
`engine/hexo-graph/src/lib.rs:302` — `let mut cells = [0i8; 64]; // 2*wl-1 <= 63 for wl<=32`. For
`win_length > 32` the subsequent `cells[slot] = ...` (line 309) would panic on out-of-bounds
indexing — a loud failure, but via an incidental array-bounds panic rather than a clear upfront
`debug_assert!(win_length <= 32, "…")` documenting the real constraint at the point it's assumed.

### 8. [NIT] No compile-time guard against `wasm` + `python` feature combination
`engine/hexo-graph/Cargo.toml:36-38` documents "NEVER combine with `wasm`" as a comment only.
Nothing in `lib.rs` enforces it (e.g. `#[cfg(all(feature = "wasm", feature = "python"))]
compile_error!(...)`). Today it would fail anyway (pyo3 doesn't target wasm32-unknown-unknown), but
with a confusing dependency-resolution error rather than the crate's own clear message — worth
adding given how deliberately this crate documents its feature boundaries elsewhere.

### 9. [NIT] `harness.rs::get_i64` silently defaults on wrong-typed JSON, not just missing keys
`engine/hexo-graph/src/bin/harness.rs:27-29` — `v.get(k).and_then(Value::as_i64).unwrap_or(dflt)`.
A present-but-wrong-typed field (e.g. `"win_length": "6"`, a string) silently falls through to the
default rather than erroring, unlike every other field in this file which uses `.expect(...)` and
dies loud. Dev/test-tool only, low real exposure, but inconsistent with the file's own style.

### 10. [NIT] Harness JSON payload carries no schema/version tag
`engine/hexo-graph/src/bin/harness.rs:78-95` — the emitted JSON object has no `contract_version` or
equivalent. Explicitly out of scope per the module doc (WP-3 owns the real wire format), so not a
bug — flagging only so WP-3 doesn't vendor this test-scaffolding schema forward as-is without adding
the version field `gnn_ragged_contract_v1.md` §2.1/§2.5 requires as "First field checked."

### 11. [NIT] `test_hexo_graph_parity.py`'s `max_int_diff` is dead/always-zero
`tests/test_hexo_graph_parity.py:117` declares `max_int_diff = 0`, printed at line 166-167, but
every integer field is checked via hard `assert ==` (not a diff computation) — a real mismatch would
raise and abort the test before the print is ever reached. The printed "max_int_diff=0" is therefore
tautological, not a measured quantity (unlike the genuinely-computed `max_float_diff` on the same
line). Harmless but slightly misleading framing; either compute a real diff (e.g. count of would-be
mismatches under a soft-check mode) or drop the variable.

### 12. [SHOULD-FIX] Parity suite exercises exactly one geometry parameterization
`win_length=6, radius=6, trunk_size=19` are fixed module constants in
`tests/test_hexo_graph_parity.py` (never varied by `_build_inputs()`/`_run_harness()`), even though
`BuildParams`/`build_axis_graph` are fully parameterized and the harness JSON schema supports
per-position overrides (`harness.rs:61-66`). Other registry encodings use different trunk sizes
(e.g. v6w25 at 25, per `docs/rules workflow` / registry). If WP-1's scope is genuinely "only the
19/6/6 legacy schema, forever" that should be stated as a hard constraint (the code doesn't enforce
it); otherwise the non-default parameter paths through `window_flat_idx`/`window_center`/threat
window sizing are shipped with zero oracle verification.

### 13. [NIT] Prefix truncations never combine with the player/moves swap
`tests/test_hexo_graph_parity.py::_build_inputs()` — truncated (partial-board) positions are always
built at the ORIGINAL `(current_player, moves_remaining)`, never the swapped pair, so no partial
board is tested with the opponent to move. Minor combinatorial gap in the augmentation scheme.

---

## Summary

| # | Severity | One-liner |
|---|---|---|
| 1 | SHOULD-FIX | `debug_assert_contract` compiled out under `cfg!(debug_assertions)` in the exact release profile production/harness runs use — no Rust-side defense-in-depth in production |
| 2 | MUST-FIX | Empty board (game start) never checked against the Python oracle — only a same-side Rust unit test |
| 3 | SHOULD-FIX | `policy_dst_slot`/`OFF_WINDOW_SLOT` shipped as `u32`, contract ratifies `u16` |
| 4 | SHOULD-FIX | Producer debug-assert mirror missing 3 leaf-computable §2.5 checks: `ScatterSlotAliasing`, `EdgeAttrGeometryMismatch`, `EmptyLegalSet` |
| 5 | NIT | `legal` Vec not capacity-reserved (contradicts report's "every Vec pre-sized" claim) |
| 6 | NIT | Dedup packed key gives `src` only 30 of 32 bits, unguarded |
| 7 | NIT | Threat-feature `[i8;64]` win_length≤32 bound relies on incidental panic, not an assert |
| 8 | NIT | No `compile_error!` guard for `wasm`+`python` feature combo |
| 9 | NIT | `harness.rs::get_i64` silently defaults on wrong-typed (not just missing) JSON fields |
| 10 | NIT | Harness JSON has no version tag (explicitly out of scope, flag only for WP-3 handoff) |
| 11 | NIT | `max_int_diff` in parity test is dead/always-zero, misleadingly printed alongside the real float diff |
| 12 | SHOULD-FIX | Parity suite only tests one geometry (19/6/6) despite fully parameterized code |
| 13 | NIT | Truncated positions never combined with the player/moves swap |

**Verdict: REVIEW-FIXES-REQUIRED** — driven by #2 (MUST-FIX) plus the cluster of contract-fidelity
SHOULD-FIX items (#1, #3, #4, #12). The core port itself is faithful (independently reproduced
byte-exact parity, clean clippy/wasm, sound bench methodology, no unnecessary clones, correct
last-wins/keep-first dedup semantics matching the oracle) — none of these findings indicate the
builder is wrong on the domain it was tested against, only that the domain and the production-mode
safety net are narrower than the commit message's claims suggest.
