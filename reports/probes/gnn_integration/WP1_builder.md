# WP-1 — C1 native axis-graph builder (`engine/hexo-graph`) + BUILD-HOT package

**Date:** 2026-07-14 · **Program:** GNN-integration (R4 ratified b+) · **Role:** WP-1 IMPL
**Scope:** `engine/hexo-graph/**` + one Python parity test (`tests/test_hexo_graph_parity.py`) + this report.
**Verdict: PARITY-EXACT + WASM-GREEN + BUILD-WITHIN-BUDGET.**

Faithful Rust port of `hexo_rl/bots/strix_v1_graph.py::build_axis_graph_raw` (the LEGACY
relative+threat schema the amended ragged-contract fixes as `node_feat_dim=11`). The Python
builder is now the TEST ORACLE only; the Rust builder is the production path. The builder emits
the full amended-contract v1 single-graph wire slice: `node_feat` (N×11), `edge_index` (local u32,
block-diagonal-ready), `edge_attr` (E×5), `legal_mask`/`stone_mask`, `node_coords` (2N),
`policy_scatter_index` (dense slot per legal node), `legal_node_gather`, `n_stones`,
`n_nodes_checksum`, `window_center`, `current_player`, `builder_impl=1`.

---

## 1. Parity result — BYTE-EXACT

`tests/test_hexo_graph_parity.py` drives the Rust builder (via a native JSON harness, no PyO3, no
`engine`-crate touch) over **1588 positions** and compares against `build_axis_graph_raw` +
oracle-geometry references.

| metric | value |
|---|---|
| positions compared | **1588** |
| total edges compared | **17,126,772** |
| max integer diff (node order, edge_index, masks, node_coords, n_stones, gather, slots, window_center, checksums) | **0** |
| max float diff (node_feat, edge_attr; both cast float32) | **0.00e+00** |

Integer outputs are byte-identical; float features are BIT-identical at f32 (features accumulate in
f64 then cast to f32, mirroring the oracle's Python-float → `torch.float32` deploy path — the same
path strix's `get_move` takes at `strix_v1_bot.py:193`). This exceeds the ≤1e-6 float tolerance and
the WP-A prod parity gate (`max|Δ|=6.6e-7`).

**Position coverage.** The frozen WP-A self-play set (320 real positions,
`wpa_positions.json`, mean 490 nodes/2932 edges) augmented to ≥1000 by (a) a
current-player/moves-remaining sweep — exercises `player_feat`/`own_is_p1`/`moves_feat` — and (b)
prefix-truncations of the real stone lists (distinct real-board geometries). **NOTE on the replay
augmentation the mission suggested:** `logs/replays/*.jsonl` holds only 33 EMPTY draw games (0 moves
each), so per-ply sweeping them yields nothing — every parity geometry is a real self-play board or
a real-board prefix instead.

Rust unit tests (`cargo test -p hexo-graph`, 5 pass) additionally lock: empty-board (dummy only),
duplicate-coord dedup (last-player-wins, see §4.4), gather-in-legal-subrange + canonical-slot
invariants, `builder_impl`/checksum.

---

## 2. Bench (BUILD-HOT sub-package) — WITHIN BUDGET

Capacity reserves first (§S182 lesson): every `Vec` pre-sized from stone-count bounds
(`node_feat = N·11`, edges = `n_real·3·2·window·2 + 2·n_real`, coords/masks = N); in-place edge
dedup (single pass, packed-u64 key, no reallocation); dep-free FNV-1a coordinate hashing (the
~30k point-lookups per position). NO parallelism inside the build (once-per-leaf; the caller
parallelizes over leaves; rayon stays out of the core path).

Criterion bench (`benches/build_bench.rs`) over the REAL WP-A self-play set (deserialize once, time
build only). Instrument: laptop Ryzen 7 8845HS, native release, `-j4` (the WP-A 0.539 ms proxy was
measured on the same laptop class).

| bench | median | spread | ns/pos |
|---|---|---|---|
| `full_set/all_positions` (320 pos, one loop) | 297.2 ms | MAD 3.85 ms; CI [296.3, 297.8] | **929 µs/pos** (MAD ~12 µs) |
| `per_position` (iter_batched, cycled) | 925.5 µs | MAD 47.3 µs; CI [915, 936] | **926 µs/pos** |

**Headline: ~0.93 ms/pos** (warm steady-state; cold-start first-run reads ~0.84 ms/pos —
laptop throttles under sustained load).

- vs the **strix proxy 0.539 ms/pos**: **1.72×** — does NOT beat it.
- vs the **contract ≤1.5 ms/pos budget**: **WITHIN** (0.93 < 1.5) → the mission's ">1.5ms ⇒ profile"
  obligation is NOT triggered.
- vs the throughput analysis: option-(c) rebuild-at-sample = 0.93 ms × batch 256 ≈ **0.24 s/step**
  (contract's ~0.14 s/step used the biased 0.539 proxy; on the honest full distribution it is 0.24
  s/step — still bounded, ~6% of a 4 s/step budget, vs the Python builder's 26× trap at 3.58 s/step).

**Why it doesn't beat 0.539 (stated, per mission):**
1. The 0.539 proxy is strix's OWN builder on a **biased 55-pos origin-compat subsample** (WP-A:
   "only positions with a stone at (0,0) reconstruct through their public API") — skewed and lighter.
   This bench is the **full 320-pos self-play distribution** (mean 490 nodes/2932 edges), unbiased.
2. This builder additionally computes the contract semantic-layer wire (`window_center`,
   `policy_scatter_index` via `window_flat_idx` per legal node, `node_coords`) the strix proxy does
   not emit.
3. Laptop thermal throttling: warm ~0.93 ms vs cold ~0.84 ms; cross-run absolute ms drift ±16%
   under sustained load (within-run CIs are <0.5%). Production self-play runs warm.
4. **5080 rider (WP-A) applies:** absolute ms do not transfer across GPUs/CPUs; re-bench on the 5080
   box when it frees (post-run3). The GNN-vs-CNN ratio is the transferable quantity.

**Bench run command** (the `panic=unwind` override is REQUIRED — workspace `profile.release` sets
`panic="abort"` for the MCTS hot path, which criterion cannot link, and this crate's `cdylib`
artifact makes the mismatch a hard error otherwise):

```
RUSTFLAGS="-C panic=unwind" cargo bench -j4 -p hexo-graph --features harness
```

---

## 3. wasm discipline — GREEN

```
cargo check -p hexo-graph --no-default-features --features wasm --target wasm32-unknown-unknown   # make check.wasm
```
Finishes clean. The core builder is dep-free (FNV hasher inlined, no rayon/std::thread/PyO3); the
JSON harness (`serde_json`) and criterion bench are behind the `harness`/dev-dep surface and
feature-gated OUT of the wasm build (the harness bin's `required-features` is unmet under
`--features wasm`, so `cargo check` never builds it; benches aren't built by `cargo check`).
Clippy (pedantic) is **0 warnings** across lib + bin + bench.

**Contract assertions producer-side.** `verify_contract` (ALWAYS-ON, every profile — promoted
from debug-assert per review #1, see §6) fires on the §2.5 invariants: `n_nodes_checksum == N`,
`N == stones+legal+dummy`, `node_coords` len `== 2N`, `edge_attr` len `== 5E`, all edge ids
in-bounds, every `legal_node_gather` row in the legal subrange `[n_stones, n_stones+n_legal)`,
every `policy_dst_slot` either the honest off-window sentinel or the canonical `window_flat_idx`
of its gathered coord, no slot aliasing, per-edge attr↔geometry consistency
(`EdgeAttrGeometryMismatch`, ADV-8), and `EmptyLegalSet` (empty-board escape hatch only). A
failing payload panics with the NAMED contract error and is never emitted.

---

## 4. Oracle AMBIGUITIES found (surfaced, not silently picked)

### 4.1 `policy_dst_slot` off-window legal cells — the load-bearing one
`build_axis_graph_raw` NEVER computes a dense action slot: the strix bot re-projects by COORDINATE
match (`strix_v1_bot.py:224-229`, `strix_legal[idx]`), not through the 362-slot dense space. The
contract's `policy_dst_slot` (`window_flat_idx` at the bbox `window_center`, `gnn_ragged_contract_v1.md`
§2.4) is a WP-B addition with **no oracle in `build_axis_graph_raw`**. Measured: **43.55% of legal
cells (57567/132174) across the 320-set are OFF-WINDOW** — outside the trunk-19 window, where
`core.rs::window_flat_idx_at_geom` returns `usize::MAX`. The oracle is silent on their slot.
**Resolved (not silently):** the builder emits `OFF_WINDOW_SLOT` = `-1` (i32 since the §6 fix
pass; originally u32::MAX with a harness `-1` remap) and
defers off-window handling to the Python resolver — consistent with engine `records.rs:62` which
SKIPS off-window legal cells at record time. **Decision WP-3/WP-B must ratify:** whether an
off-window legal node is dropped, masked, or gets a synthetic slot in the dense `[B,362]` scatter.
The parity test verifies the slot IS the canonical `window_flat_idx` for in-window cells (byte-exact)
and IS the sentinel for off-window — but the *downstream* meaning of the sentinel is unspecified by
the oracle.

### 4.2 Two distinct "centres" in one builder
`norm_q`/`norm_r` use the **float CENTROID** (mean of stone coords, `strix_v1_graph.py:148-156`);
`policy_dst_slot`'s window uses the **bbox `window_center`** (`(min+max)/2` truncate-toward-zero,
`core.rs::window_center`). Different origins, easy to conflate. Pinned `window_center` to the engine
`core.rs` truncate-toward-zero semantics (byte-parity with `window_flat_idx_at_geom`; `midpoint()`
rounds differently and would break negative-coord boards — allowed the `manual_midpoint` clippy lint
with that note).

### 4.3 `current_player == None` (terminal)
Oracle: `player_feat = 1.0 if current_player == 1 else -1.0`; `to_move = 1 if current_player == 1
else -1` → a terminal `None` is treated as P2 (−1). `BuildParams.current_player` is `i8` with no
`None`; any value ≠ 1 maps to −1. Faithful for `current_player ∈ {1, −1}` (all real positions); the
`None`→−1 terminal branch is unreachable through a real board. Documented, not exercised.

### 4.4 Duplicate stone coordinates
The oracle's input is a **dict**, so a repeated coord silently collapses (last-player-wins). A raw
`Vec` input could carry duplicates. **Resolved:** stone nodes are derived from the deduped
`stone_map` (last-wins), not the raw Vec — byte-matching the dict on the whole input domain. Real
boards never duplicate a cell; matching the dict is what makes the port faithful rather than merely
correct-on-clean-input. Unit-tested (`duplicate_coord_dedups_last_wins`).

### 4.5 dedup keep-FIRST is emission-order-dependent (fragility, not divergence)
The axis-edge dedup keys `(src, dst, axis_idx)` and keeps the FIRST occurrence's `edge_attr`
(`strix_v1_graph.py:264-282`). A reverse+forward duplicate of the same `(j,i,axis)` survives with
whichever attr was emitted first in the i-loop order. Behavior is deterministic, but any reordering
of the walk (outer node → axis → sign∈(1,−1) → distance) silently changes which attr survives. The
Rust port replicates the exact emission order + keep-first; the 17.1M-edge byte-parity confirms it.
Flagged so a future "optimize the walk order" change knows it is parity-load-bearing.

### 4.6 Only the default schema is ported (scoping, not ambiguity)
The oracle supports flag combinations the Rust builder does NOT: absolute base
(`relative_stones=False`, base_dim 8 with a `to_move` column), `prune_empty_edges=True`,
`threat_features=False`. The contract fixes `node_feat_dim=11` = relative-7 + threat-4 (§2.1), so
only that default schema is implemented. Named here so WP-2/WP-3 know the Rust builder is
single-schema by design.

---

## 5. Files written

| file | change |
|---|---|
| `engine/hexo-graph/src/lib.rs` | skeleton → real builder (`build_axis_graph`), FNV hasher, threat/legal/window ports, in-place dedup, `debug_assert_contract`, 5 unit tests |
| `engine/hexo-graph/src/bin/harness.rs` | NEW — native JSON parity harness (`hexo_graph_harness`), feature `harness`, no PyO3 |
| `engine/hexo-graph/benches/build_bench.rs` | NEW — criterion BUILD-HOT bench over the WP-A set |
| `engine/hexo-graph/Cargo.toml` | `harness` feature (`serde_json`), `[[bin]]`+`[[bench]]` (required-features `harness`), criterion dev-dep |
| `tests/test_hexo_graph_parity.py` | NEW — 1588-position byte-parity vs `build_axis_graph_raw` |
| `reports/probes/gnn_integration/WP1_builder.md` | this report |

**Out of scope, untouched:** `engine/src/**`, `hexo_rl/**`, `Makefile`, workspace `Cargo.toml`. PyO3
wiring into the engine seam + the dense `[B,362]` re-projection are WP-3; the block-diagonal collate
resolver + off-window ruling (§4.1) are WP-B/WP-3.

---

## 6. Fix-pass addendum (2026-07-14, post WP1_review.md REVIEW-FIXES-REQUIRED)

Applied per dispatcher ruling; red-team (`WP1_redteam.md`) came back CLEAN over the reachable
domain, so no red-team-driven code change beyond the harness cast validation it flagged.

| review item | action |
|---|---|
| #2 MUST-FIX | Empty board (4 variants) + 1-stone boards (4 variants) now IN the oracle-compared parity set (`_build_inputs()` section (a)) — not just Rust-side unit tests |
| #1 (ruling: promote) | `debug_assert_contract` → **`verify_contract`, ALWAYS-ON in every profile** (release included). A failing payload PANICS with the NAMED contract error and is never emitted. Cost measured below |
| #3 (ruling: i32) | `policy_dst_slot`/`OFF_WINDOW_SLOT` = **`i32` / −1** (was u32/MAX; contract said u16). Contract §2.1/§2.2/§2.5 AMENDED citing the WP-3 option-(b) ruling (`gnn_inference_seam_design.md` §1: field demoted to training/probe metadata; u16 can't carry the sentinel). Harness serializes the i32 natively (no −1 remap) |
| #4 | `verify_contract` extended with the leaf-checkable missing checks: **`ScatterSlotAliasing`** (bitset over trunk²), **`EdgeAttrGeometryMismatch`** (per-edge recompute from WIRE arrays only — node_coords endpoints, axis one-hot, integral signed_dist ≤ window, src_player from stone/own columns × current_player; dummy edges all-zero), **`EmptyLegalSet`** (legal ≥ 1; EXPLICIT terminal escape hatch: the empty board — n_stones == 0 — is the ONLY input with a legitimately empty legal set, matching the strix bot's pre-net short-circuit). Two new should-panic unit tests prove ADV-2b and ADV-8 die with the named error |
| #12 | Second geometry in the parity suite: 100 base positions at **win_length=5 / radius=4** (threat-window sizing, walk depth, legal ring) — oracle-compared, byte-exact |
| #5 | `legal` Vec capacity-reserved (same bound as its sibling `seen` set) |
| #7 | `win_length ∈ 1..=32` asserted at builder entry (the `[i8;64]` threat-buffer bound), `trunk_size ≥ 1` |
| #8 | `compile_error!` guard for `wasm`+`python` (verified: fires with the crate's own message) |
| #9 + red-team Attack-4 | Harness parsing STRICT: present-but-wrong-typed field dies loud (defaults only on ABSENT); every narrowing cast range-validated first (`moves_remaining` must fit u8, `current_player` i8 — a −1 can no longer wrap to 255) |
| #10 | Harness output now `{"harness_schema_version": 1, "graphs": [...]}` — version-tagged test scaffolding |
| #11 | Dead always-zero `max_int_diff` deleted; parity print now says "ints byte-asserted" (hard equality, not a computed diff) |
| #6 | 30-bit src dedup-key ceiling made EXPLICIT: `assert!(n < (1 << 30))` once per build + comment (red-team proved ≥1e9 nodes unreachable — no further change) |
| #13 | SKIPPED (not in the dispatcher fix list; marginal combinatorial coverage) |
| #1-alt (b) | Moot — option (a) taken (always-on), so the "document release-mode gap" alternative doesn't apply |

**Parity rerun (all fixes in):** n=**1696** positions (8 degenerate + 1588 default-geometry +
100 at wl=5/r=4), 17,742,046 edges, ints byte-asserted (hard equality), max_float_diff = 0.00e+00.

**Always-on verify_contract cost:** see the bench table below — measured on the same frozen
position set, same instrument, single post-fix run vs the §2 pre-fix baseline.

| bench | pre-fix (debug-assert, compiled out) | post-fix (always-on) | delta (criterion vs-prev) |
|---|---|---|---|
| `full_set/all_positions` median | 297.2 ms (929 µs/pos) | 306.0 ms (**956 µs/pos**), MAD ~15 µs/pos | **+2.14%** CI [+1.74, +2.54], p<0.05 |
| `per_position` median | 925.5 µs | 955.1 µs, CI [942, 968] | +1.56% CI [−1.5, +4.6], p=0.31 (n.s.) |

Both runs cooled (same protocol as §2). Verdict vs the ~3% always-on budget: **+2.1%
(full-set, the tighter instrument) < 3% → ALWAYS-ON STANDS** per the dispatcher ruling; the
cost is the O(E) edge-geometry recompute + the O(n_legal) slot bitset, ~20 µs/pos. Still
comfortably within the ≤1.5 ms contract budget (0.956 < 1.5). wasm check GREEN, clippy 0
warnings (the `float_cmp` allow on `verify_contract` is intentional — the compared floats are
exact constants the builder itself wrote; approximate comparison would weaken the check).
