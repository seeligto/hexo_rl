# WP-1 RED-TEAM — native axis-graph builder (`engine/hexo-graph/src/lib.rs`)

**Target:** commit `bfca786` — `build_axis_graph` vs Python oracle
`hexo_rl/bots/strix_v1_graph.py::build_axis_graph_raw`.
**Method:** real side-by-side execution. Release harness
(`cargo build --release -j4 -p hexo-graph --features harness`) vs the oracle,
compared field-by-field (integers byte-exact, floats ≤1e-6) in the exact wire
format of `tests/test_hexo_graph_parity.py`. Drivers:
`/home/timmy/.claude/jobs/7d6e8877/tmp/redteam_driver.py`,
`redteam_det.py`.

## OVERALL VERDICT: RED-TEAM-CLEAN over the reachable domain — 1 latent GAP (LOW-reachability, SILENT in release)

The builder is a faithful port. Every attack landed inside the reachable
self-play domain (|q|,|r| bounded by game length, ~hundreds) MATCHED the oracle
byte-for-byte. The single class of divergence found requires coordinates within
`radius` of `i32::MAX`/`i32::MIN` (~2.147e9) — **unreachable in real self-play**
but **unguarded and silent in the release/production build** (overflow wraps, no
panic, no assertion). Reported as **GAP-LATENT (severity LOW-reachability /
MEDIUM-latency)**, not a parity break of the tested contract.

---

## Attack 1 — Dedup collision (packed-key / FNV) — CLEAN

**Construction.** `pack(q,r) = (i64::from(q) << 32) | i64::from(r as u32)`
(lib.rs:104). High 32 bits carry `q` (sign-extended then shifted — low 32 bits
zeroed by `<<32`); low 32 bits carry `r`'s raw two's-complement (`r as u32`,
zero-extended). The two halves never overlap → `pack` is a **bijection over
i32×i32**. FNV (`FnvHasher`) is only the bucket hash; `HashMap`/`HashSet` still
compare the full i64 key with `Eq`, so a hash collision re-buckets but **cannot
merge two distinct coords**. Fired adversarial "collision-looking" pairs anyway:
sign-vs-low-16 aliases `(-1,0)`/`(65535,0)`, `(0,-1)`/`(0,65535)`; q/r swaps
`(1,2)`/`(2,1)`, `(-2,3)`/`(3,-2)`; byte-shift aliases `(256,0)`/`(1,0)`,
`(0,256)`/`(0,1)`; low-bit adjacency `(0,0)`/`(0,1)` — placed as opposite-player
stones so any silent merge would collapse `n_stones` 2→1.

**Expected.** Distinct nodes, `n_stones=2`, exact oracle match.
**Observed.** All 7 in-range pairs `match` (byte-exact). The only two A1
divergences (`A1[3]`,`A1[6]`) contain an `i32::MAX` coordinate → that is the
Attack-2 overflow, NOT a pack collision.

**Verdict: CLEAN.** No dedup collision exists in the reachable domain. The
edge-dedup key `(src<<34)|(dst<<2)|axis` (lib.rs:652) is injective while
`src < 2^30` and `dst < 2^32`; node count here peaked at 3127 (≪ 2^30), so proven
non-colliding by construction — a collision needs >1e9 nodes (OOM long before).

## Attack 2 — Coordinate extremes / i32 overflow — GAP-LATENT (headline finding)

**Construction.** Stones at `i32::MAX`, `i32::MIN`, `(MAX,MAX)`, `(MIN,MIN)`,
`MAX-3`, and mid-range `1e6`/`1e9`. Same integer inputs to both sides (Python
uses bigint → the divergence is pure Rust i32 wrap).

**Observed (release harness ran to completion, rc=0, NO panic in every case):**

| input | rust | oracle | divergence |
|---|---|---|---|
| `[(1e6,1e6),(1e6+3,1e6)]` | — | — | **MATCH** |
| `[(1e9,0),(1e9+3,0)]` | — | — | **MATCH** |
| `[(i32::MAX,0,1)]` | `window_center=(-1,0)`, coords wrap to `MIN..` | `wc=(2147483647,0)` | node_coords, features, all slots wrong |
| `[(i32::MAX,0),(i32::MIN,0)]` | `num_nodes=141` (one merged cluster) | `num_nodes=255` (two clusters) | **topology differs** — MAX's `+1` legal cell wraps to MIN's neighborhood, so far-apart stones become "adjacent" |
| `[(i32::MAX,i32::MAX,1)]` | `wc=(-1,-1)` | `wc=(2147483647,2147483647)` | node_feat maxdiff `4.295e9` (=2^32) |
| `[(0,i32::MAX),(0,i32::MIN)]` | `num_nodes=141` | `num_nodes=255` | topology differs |

**Root cause.** Silent i32 wrap at multiple sites (release `overflow-checks=off`):
`legal_moves_from_stones` `sq+dq` (lib.rs:371 — first site to trip), `window_center`
`(min_q+max_q)/2` (lib.rs:237), `node_threat_features` `coord.0+k*dq` (lib.rs:308),
`hex_distance` subtraction + `i32::MIN.abs()` (lib.rs:258). Python's bigint never
wraps → total structural divergence (node count, coords, edges, features).

**Reachability.** In self-play, stones are placed within `radius` of existing
stones from an origin; coords are bounded by ply count (~hundreds), ≪ 2^31. The
crossover is empirical: `1e9` MATCHES, boundary values diverge. So NOT reachable
under normal self-play. **But there is NO clamp/assert on the coordinate range**,
and the CLAUDE.md board-representation rule literally calls the board infinite —
a corrupt corpus row, an out-of-range coord from a future PyO3 caller (WP-3
seam), or an off-board wander would produce a **silently wrong graph, not a
crash**. Recommend the WP-3 resolver assert `|q|,|r| < i32::MAX - radius` (or the
builder itself), since the sole existing guard is debug-only (see Attack 6).

**Verdict: GAP-LATENT.** Parity of the *tested* domain intact; unbounded-coord
contract undefined and unguarded in release.

## Attack 3 — Oracle-fragility: duplicate coord, conflicting players — CLEAN

**Construction.** Same coord, opposing players, multiple emission orders:
`[(0,0,1),(2,0,-1),(0,0,-1)]` vs reorderings; 4× alternating dupes; `[(1,1,-1),(1,1,1)]`.
**Expected.** Rust `HashMap::insert` = **last-player-wins by input order**;
Python test caller `{(q,r):pl for ...}` dict-comp = **last-wins** too. Must agree
per ordering.
**Observed.** All 5 `match`. The `stones.sort_unstable_by_key` on both sides
canonicalizes post-dedup order, and there are no duplicate keys left to expose
stable-vs-unstable sort differences.
**Verdict: CLEAN.** Rust is last-wins (not keep-first); matches the oracle's dict
last-wins under every ordering tested. (Note the lib.rs:409 comment says
"last-player-wins" — accurate; the `duplicate_coord_dedups_last_wins` unit test
name agrees.)

## Attack 4 — Degenerate boards & param edges — CLEAN

**Construction / Observed (all `match`):** empty board (dummy-only, `wc=(0,0)`);
single stone; 6-in-a-row saturation line (`cp=±1`); `moves_remaining ∈ {0,1,3}`;
`current_player ∈ {0,2}` (oracle rule `!=1 → -1`, Rust `cur==1` else `-1` — agree);
stones on the 19-window boundary `(0,0)+(18,0)`, `(0,0)+(17,1)`; a **mixed
on/off-window board** `(0,0)+(24,0)` → 68 on-window slots + 184 `-1` off-window
sentinels, **slot-for-slot exact**. `window_center=(12,0)` both sides.
**Verdict: CLEAN.** `OFF_WINDOW_SLOT`→`-1` sentinel consistent per-cell between
`policy_dst_slot` and the gather rows; degenerate/param edges all faithful.

**Boundary note (wire, not builder):** harness casts `moves_remaining as u8` and
`current_player as i8` (bin/harness.rs:65). A negative `moves_remaining=-1`
wraps to 255 → `moves_feat=127.5` vs oracle `-0.5` (confirmed). This is a
harness JSON→u8 cast, and `BuildParams.moves_remaining: u8` structurally forbids
negatives, so it is not a builder-vs-oracle break — but the WP-3 seam must
range-validate `moves_remaining`/`current_player` before the cast.

## Attack 5 — Float / accumulation determinism — CLEAN

**Construction.** 149-stone real self-play board (densest in `wpa_positions.json`,
768 nodes / 17422 edges); 13 random shuffles of the stone list through the
release harness. Plus a centroid-precision stress: 200 stones near `1e9`
(3127 nodes) where f64 sums are large but < 2^53.
**Observed.**
- 13 shuffles → **ALL BYTE-IDENTICAL** across every field (node_feat, edge_attr,
  edge_index, coords, slots, …). Real board also matches the oracle exactly
  (`node_feat_maxdiff = 0.00e+00`).
- 200-stone @1e9: `node_feat_maxdiff = 0.0`, `edge_attr_maxdiff = 0.0`,
  coords exact.
**Why.** `stones.sort_unstable_by_key((q,r))` (lib.rs:415) runs BEFORE centroid
and every float accumulation, so input order is canonicalized; f64 addition of
integer coords is exact while partial sums stay < 2^53 (≈9e15 — needs >4e6
stones at 2e9 to break). Order-dependence would only appear via duplicate-coord
player resolution (Attack 3), which is itself deterministic per input.
**Verdict: CLEAN.** The D6-rebuild determinism claim holds.

## Attack 6 — Release-mode safety (guard audit) — CONFIRMED debug-only guard

**Construction.** Same overflow input `[(i32::MAX,0,1)]` through the **release**
harness and a **debug** harness (`cargo build -p hexo-graph --features harness`).
**Observed.**
- Release: `rc=0`, runs to completion, emits a silently-wrong graph.
- Debug: **PANIC** rc=101 — `lib.rs:371:22: attempt to add with overflow`
  (`legal_moves_from_stones`, `sq + dq`).
**Analysis.** `debug_assert_contract` (lib.rs:671) is gated on
`cfg!(debug_assertions)` and vanishes in release; overflow-checks also vanish. So
the ONLY thing catching a malformed/extreme coordinate is the debug overflow
check, absent from the production build. No `unsafe`/`get_unchecked` in the crate
— all `Vec` indexing stays bounds-checked, so the worst case is a **panic (debug)
or silently-wrong output (release), never UB**. Confirms the Attack-2
recommendation: the resolver/seam must own the range guard, because the builder's
is compiled out where it ships.

---

## Headlines (for the caller)
- **Overall:** RED-TEAM-CLEAN over reachable domain; 1 latent gap.
- **Collision attack:** CLEAN — `pack` is a bijection over i32×i32; FNV can't
  merge nodes (HashMap `Eq` on i64 keys); every adversarial alias pair produced
  distinct nodes matching the oracle.
- **Coordinate bound:** builder is exact for |q|,|r| up to ~`i32::MAX-radius`
  (1e9 MATCHES); beyond that, silent i32 wrap → wrong topology/features, no
  panic in release, no guard. Unreachable in self-play; recommend a resolver/seam
  range assert (only guard today is debug-only, proven by the debug-build panic
  at lib.rs:371).
- **Determinism:** CLEAN — 13 shuffles of a 149-stone real board → byte-identical;
  the pre-accumulation coord sort canonicalizes input order; f64 sums exact.
