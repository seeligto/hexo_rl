# v8 encoding contract — Phase A integration spec

**Status:** Locked at branch creation. Source of truth for Phase A code.
**Branch:** `encoding/phase_a_pipeline`
**Date:** 2026-05-07
**Master HEAD at branch:** `5c82cd5`
**Strategy:** Gated coexistence. v6 path remains canonical default; v8 path
added behind config knob `encoding.version == "v8"`. v6 byte-exact
(no observable behavior change).

This contract supersedes `docs/designs/encoding_migration_v8.md` Phase A
on **scope** only: that doc described a hard cutover at Phase A; the
operator chose gated coexistence instead. All other v8 design choices
(Path β 25×25, 11 planes, R=8, K removed, 96-channel trunk + GPool
in Phase B, etc.) carry over unchanged.

---

## 1. Constants

### 1.1 v6 (canonical default — UNCHANGED)

| Symbol | Rust | Python | Value |
|---|---|---|---|
| `BOARD_SIZE` | `BOARD_H = BOARD_W` | `BOARD_SIZE` | 19 |
| `HALF` | `HALF` | `HALF` (in `coordinates.py`) | 9 |
| `N_CELLS` | `N_CELLS` | `NUM_CELLS` | 361 |
| `N_ACTIONS` | `N_ACTIONS` | (n/a, used inline) | 362 (361 + pass) |
| `N_PLANES` | `N_PLANES` | `BUFFER_CHANNELS` | 8 |
| `KEPT_PLANE_INDICES` | `KEPT_PLANE_INDICES` | `KEPT_PLANE_INDICES` | [0,1,2,3,8,9,10,11] |
| `DEFAULT_LEGAL_MOVE_RADIUS` | (Rust only) | (n/a) | 5 |
| `CLUSTER_THRESHOLD` | (Rust only, private) | (n/a) | 5 |
| `STATE_STRIDE` | `N_PLANES * N_CELLS` | (derived) | 2888 |
| `CHAIN_STRIDE` | `N_CHAIN_PLANES * N_CELLS` | (derived) | 2166 |
| `POLICY_STRIDE` | `N_ACTIONS` | (derived) | 362 |
| `AUX_STRIDE` | `N_CELLS` | (derived) | 361 |

### 1.2 v8 (gated, Path β)

| Symbol | Value | Notes |
|---|---|---|
| `BOARD_SIZE_V8` | 25 | bbox-of-all-stones fixed-max |
| `HALF_V8` | 12 | (BOARD_SIZE_V8 - 1) / 2 |
| `N_CELLS_V8` | 625 | 25 × 25 |
| `N_ACTIONS_V8` | 625 | spatial-only; no pass slot (P1 close-out: pass dead in HTTT) |
| `N_PLANES_V8` | 11 | 8 KEPT + 1 off_window + 2 scalar broadcasts |
| `LEGAL_MOVE_RADIUS_V8` | 8 | HTTT rule baseline; replaces v6's r=5 cap |
| `CLUSTER_THRESHOLD_V8` | N/A | K-aggregation removed in v8 |
| `MARGIN_M_V8` | 8 | bbox dilation margin = LEGAL_MOVE_RADIUS_V8 |
| `STATE_STRIDE_V8` | 6875 | 11 × 625 |
| `CHAIN_STRIDE_V8` | 3750 | 6 × 625 (Q13 chain planes ride v8 spatial; output-head only, untouched in Phase A) |
| `POLICY_STRIDE_V8` | 625 | spatial-only |
| `AUX_STRIDE_V8` | 625 | single-plane spatial |

---

## 2. Plane schema — v8 (11 planes)

Source-of-truth: plan §GATE 2 + `audit/encoding_spikes/SPIKE_SUMMARY.md` §2.2.

| Idx | Name | Source | Semantics |
|---:|---|---|---|
| 0 | cur ply-0 | KEPT_PLANE_INDICES[0] (= src 0 of 18-plane game state) | current player's stones (this ply) |
| 1 | cur ply-1 | KEPT[1] (src 1) | current player's history (1 ply back) |
| 2 | cur ply-2 | KEPT[2] (src 2) | current player's history (2 plies back) |
| 3 | cur ply-3 | KEPT[3] (src 3) | current player's history (3 plies back) |
| 4 | opp ply-0 | KEPT[4] (src 8) | opponent's stones (this ply) |
| 5 | opp ply-1 | KEPT[5] (src 9) | opponent history (1 ply back) |
| 6 | opp ply-2 | KEPT[6] (src 10) | opponent history (2 plies back) |
| 7 | opp ply-3 | KEPT[7] (src 11) | opponent history (3 plies back) |
| 8 | off_window | NEW | 1.0 outside dilated hex (padding cell), 0.0 inside (valid cell). KataGo `mask = 1.0 − off_window` computed once at trunk forward entry. |
| 9 | moves_remaining_bcast | NEW | `(MAX_MOVES − moves_played) / MAX_MOVES`, broadcast across spatial. |
| 10 | ply_parity_bcast | NEW | `ply % 2`, broadcast across spatial. |

**NOTE:** `docs/designs/encoding_migration_v8.md` §2.3 plane labels for indices
4-7 reference threat/chain planes; that table is inconsistent with v6
KEPT_PLANE_INDICES which are stone-history-only. **This contract locks the
plan + SPIKE_SUMMARY interpretation: planes 0-7 are stone history (matching
v6 KEPT semantics), 8-10 are new.** Design doc §2.3 plane label rows 4-7
should be revised on Phase E doc rewrite.

---

## 3. NPZ schema v8

| Field | v6 | v8 |
|---|---|---|
| `states` shape | (N, 8, 19, 19) fp16 | (N, 11, 25, 25) fp16 |
| `policy_targets` shape | (N, 362) fp16 | (N, 625) fp16 |
| `aux_targets` shape (ownership / winning_line) | (N, 19, 19) u8 | (N, 25, 25) u8 |
| `chain_planes` shape (Q13, output-only) | (N, 6, 19, 19) fp16 | (N, 6, 25, 25) fp16 |
| Compression | optional via flag | unchanged |
| mmap-compatible | yes (preserves `pretrain.py:669` path) | yes |

**Path:** v6 → `data/bootstrap_corpus.npz` (canonical, unchanged).
v8 → `data/bootstrap_corpus_v8.npz` (separate file; no overwrite).

NPZ schema_version key bumped 6 → 7 only when v8 file is written; v6 file
remains schema_version=6.

---

## 4. Config knob shape

### 4.1 `configs/model.yaml`

Add a new section. Existing `board_size: 19`, `in_channels: 8` remain
unchanged when `encoding.version == "v6"`. Under v8, model code must
treat `board_size: 25` and `in_channels: 11` as derived from
`encoding.version`, NOT from these legacy keys (which stay at v6 values
for the canonical default path).

```yaml
# Existing keys (v6 defaults; do not change)
board_size: 19
in_channels: 8
res_blocks: 12
filters: 128
se_reduction_ratio: 4

# New v8 gate
encoding:
  version: "v6"   # default; "v8" routes to Path β (25×25 + 11 planes + R=8)
```

### 4.2 EncodingSpec resolver

New module `hexo_rl/utils/encoding.py` exposes:

```python
class EncodingSpec(NamedTuple):
    version: Literal["v6", "v8"]
    board_size: int
    half: int
    n_cells: int
    n_actions: int
    n_planes: int
    legal_move_radius: int
    cluster_threshold: int | None  # None for v8
    state_stride: int
    chain_stride: int
    policy_stride: int
    aux_stride: int

def resolve_encoding(config: Mapping[str, Any]) -> EncodingSpec: ...
```

Behavior:
- `config.get("encoding", {}).get("version", "v6") == "v6"` → v6 spec (defaults).
- `... == "v8"` → v8 Path β spec.
- Any other version → `ValueError`.

All v8-aware Python code threads an `EncodingSpec` rather than reading
loose constants. v6-only code paths can keep importing
`hexo_rl.utils.constants.*` directly (zero-touch).

### 4.3 Rust resolution

Rust constants resolved at config-load time, NOT compile time. Two paths:

* **v6 (default):** existing const symbols stay (`N_PLANES = 8`, `BOARD_H = 19`,
  …). v6 code path unchanged.
* **v8 (gated):** new const symbols (`N_PLANES_V8 = 11`, `BOARD_H_V8 = 25`, …)
  exposed alongside. Functions that need to operate on either path take an
  `EncodingVersion` enum or shape parameters; v6-only callers keep their
  fixed constants.

PyO3 `feature_len` / `policy_len` defaults stay at v6 values
(`8 * 19 * 19 = 2888`, `19 * 19 + 1 = 362`). v8 callers pass explicit values
sourced from EncodingSpec.

---

## 5. PyO3 boundary

### 5.1 `inference_bridge.rs:295` — `InferenceBatcher::new`

Current:
```rust
#[pyo3(signature = (feature_len = 8 * 19 * 19, policy_len = 19 * 19 + 1))]
pub fn new(feature_len: usize, policy_len: usize) -> Self { ... }
```

After Phase A: signature unchanged. v6 callers omit args (defaults preserved).
v8 callers pass explicit `feature_len = 11 * 25 * 25 = 6875`,
`policy_len = 25 * 25 = 625`. PyO3 default behavior: byte-exact v6.

### 5.2 `game_runner/mod.rs` PyO3 surface

`feature_len` / `n_actions` / shape signatures parameterized similarly.
v6 defaults preserved at PyO3 boundary; v8 callers pass explicit values.

### 5.3 sym_tables LUT regeneration

`engine/src/replay_buffer/sym_tables.rs` LUTs are static at v6 dimensions.
v8 path needs:
- New static LUTs at `N_CELLS_V8 = 625`, `N_PLANES_V8 = 11` for the 12-fold
  hex symmetry scatter — distinct symbols, NOT a replacement of v6 LUTs.
- Identity check: sym=0 is identity scatter; rotations compose.
- Validated by new test `engine/tests/test_sym_tables_v8.rs`.

### 5.4 Replay buffer wire format

`replay_buffer/{push,sample,persist,mod}.rs` parameterize on encoding spec.
v6 wire format unchanged at runtime. v8 wire format is a new format with
its own slot stride; persistence `version` byte distinguishes.

**This contract does NOT require runtime support for mixing v6 and v8 buffers
in a single run.** A run is one or the other; encoding is a startup-pinned
config knob. Cross-version buffer load is a Phase E cutover concern.

---

## 6. Hotfix-(c) bundling

P2 catastrophic verdict closure (probe SUMMARY §31): v7full loses 22% to
brain-dead far-line script because LEGAL_MOVE_RADIUS=5 perception cap is
narrower than HTTT rule's r=8.

### 6.1 Under v8 path

- `LEGAL_MOVE_RADIUS_V8 = 8` (HTTT baseline)
- `CLUSTER_THRESHOLD_V8 = None` (K-aggregation retired; single bbox replaces)
- `tests/probes/p2_far_placement_opponent.py` under v8 must show
  `far_line` opp_winrate < 5% (probe SUMMARY exit gate).

### 6.2 Under v6 path

- `DEFAULT_LEGAL_MOVE_RADIUS = 5` (unchanged)
- `CLUSTER_THRESHOLD = 5` (unchanged)
- Probe under v6 still shows ~22% (unchanged; this is the catastrophic
  baseline, NOT a regression — confirms Phase A doesn't accidentally fix
  v6 path).

### 6.3 Gate

`engine/src/board/moves.rs:20,32` constants stay at v6 values as the
**Board default**. v8-gated code paths pass explicit radius/threshold to
Board init or use a v8-construction helper. The two paths share the
moves.rs implementation; only the constants used at Board init differ.

---

## 7. What this contract does NOT cover

Out of Phase A scope (Phase B+):
- Model architecture changes (96-channel trunk, KataConvAndGPool, KataGo
  policy head replacement) — Phase B.
- Corpus regeneration on real data — Phase C.
- Bootstrap retrain (`bootstrap_model_v8full.pt`) — Phase C.
- Self-play encoding-awareness end-to-end — Phase D (partial in Bucket D
  here for hotfix-(c) probe).
- Sustained smoke runs / SealBot evals — Phase D.
- Cutover (canonical pointer flip, `bootstrap_model.pt` overwrite) — Phase E.
- Documentation rewrites of `docs/rules/board-representation.md` and
  `docs/rules/perf-targets.md` — Phase E.
- `audit/probes/SUMMARY.md` close-out note — Phase E.

Hardcoded constants cleanup (898 hits per `audit/encoding_spikes/hardcoded_constants.txt`)
beyond the Phase A buckets is OUT OF SCOPE — separate cleanup sprint after
Phase A validates v8 path works end-to-end on test fixtures.

---

## 8. Contract update protocol

If implementation reveals a contract issue (a constraint that prevents
clean gated coexistence, or a numerical mismatch at the v6/v8 boundary),
update this doc FIRST, then change code. Bucket commits should reference
the contract section being implemented.
