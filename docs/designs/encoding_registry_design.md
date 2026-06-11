# §172 Phase A2 — Encoding Registry Design

*Architectural sprint: §172 · Phase A2 (design) · Branch: `phase4/encoding_registry`*

Date: 2026-05-09

---

## 0. Purpose & status

This doc is the contract A3 implements against and A9 reviews against. It locks
the registry surface (file, schema, Rust API, Python API), the metadata schemas
for checkpoints + corpora, the audit utility, the round-trip test fixture, and
the A3-A9 sequencing.

**Inputs:**

- `docs/designs/encoding_registry_analysis.md` (A1 — 184 surfaces inventoried,
  23 load-bearing, 5 open questions resolved 2026-05-09).
- `docs/designs/encoding_alpha_multiwindow_selfplay.md` (α scope memo —
  forward-pointer for A7).
- §171 P3 blocker entry in `docs/07_PHASE4_SPRINT_LOG.md` (failure mode that
  motivated this sprint).
- Operator decisions recorded in A1 §6.6 (`EncodingMeta` struct), A1 §6.11
  (Option 3 — thread `EncodingSpec`, retire `config["board_size"]`).

**Status:** locked at A2 commit; A3 implements without re-litigating. Any
design change A3 needs returns to A2.

---

## 1. Problem statement

Encoding-derived values are scattered across **184 surfaces** in 4 layers
(Rust engine, Python runtime, Python offline, artifacts). The §171 P3
pre-flight blocker chain demonstrated the failure mode end-to-end:

- A2.1 (trainer) wrote `config["board_size"] = 25` (trunk semantics).
- A2.2 (pool guard) compared against canvas=19 — same key, different
  semantics.
- `Board::to_planes()` (engine/src/board/state.rs:641) ignored
  `cluster_window_size`, emitting 18×19×19 regardless.
- v6w25 model expected (B, 8, 25, 25) input. Selfplay delivered (B, 8, 19,
  19). Either crash or silent shape corruption.
- 25 checkpoints + 8 corpora carry **no** explicit encoding metadata; one
  artifact (`bootstrap_model_v8full_warm.pt`) is unresolvable without it.

Each surface has its own copy of "the rules". Adding an encoding means
editing N places. Mis-syncing one of them is the §171 P3 failure.

**Fix:** single source of truth. One file. Every consumer reads from it.

---

## 2. Design principle

`engine/src/encoding/registry.toml` is the canonical declaration of every
encoding the codebase supports. Both Rust and Python parse the same file and
build immutable `EncodingSpec` records keyed by name. Every encoding-aware
surface — Rust or Python, runtime or offline, training or eval — reads from
this registry. No surface carries its own copy of `board_size`,
`cluster_window_size`, `n_planes`, or any other encoding-derived constant.

Constraints enforced by this principle:

- Adding an encoding = one new TOML entry + (if multi-window) implementing
  the dispatch path.
- Renaming a field = grep one file (the schema validator), then mechanical
  fix-up of consumers.
- `EncodingSpec` instances are `frozen` / `Clone + 'static` — never
  mutated after registry init.
- Variant configs override **only** `encoding: <name>` — the resolver
  looks up everything else.

This supersedes A1 §6.6's "named const presets" pattern: lookup-returned
`&'static EncodingSpec` replaces `V6_META, V6W25_META, V8_META` const
presets. Stable address from `once_cell::sync::Lazy` preserves the
cache-line invariant for hot-path consumers (ReplayBuffer push/sample).

---

## 3. TOML schema

`engine/src/encoding/registry.toml` is the authoring file. One table per
encoding under `[encodings.<name>]`. Required keys are validated at parse
time; missing or wrong-typed keys fail loud.

### 3.1 Per-encoding schema

```toml
[encodings.<name>]
# Geometry
board_size              = <int>          # canvas side; action space = board_size² + (1 if has_pass_slot else 0)
trunk_size              = <int>          # NN trunk input side (= board_size for single-window;
                                         #   = cluster_window_size for K-cluster)
cluster_window_size     = <int> | "none" # K-cluster window side, or "none" if single-window
cluster_threshold       = <int> | "none" # min stones to trigger clustering, or "none"
legal_move_radius       = <int>          # legal-move radius around stone bbox

# Tensor shape
n_planes                = <int>          # input channels
plane_layout            = [<str>, ...]   # semantic plane names; len = n_planes
policy_logit_count      = <int>          # = board_size² + (1 if has_pass_slot else 0)
has_pass_slot           = <bool>         # A3 amendment 2026-05-09 — v8/v8_canvas_realness=false (existing
                                         #   ckpts have policy_fc.out_features=625, no pass); v6/v6w25/v7full=true

# Dispatch
is_multi_window         = <bool>         # true → workers emit K cluster views per position
value_pool              = <str>          # "min" | "max" | "mean" | "none"
                                         #   (K-aggregation at root; ignored if !is_multi_window)
policy_pool             = <str>          # "scatter_max" | "scatter_mean" | "none"
                                         #   (per-move cluster dispatch; ignored if !is_multi_window)

# Schema-level
sym_table_id            = <str>          # ID into sym_tables.rs registry
schema_version          = <int>          # bump when a field is added/removed/renamed
notes                   = <str>          # free-form provenance (which sprint anchored, what for)
```

### 3.2 Worked examples

```toml
# v6 — canonical 19×19 single-window, 8 planes (history-only)
[encodings.v6]
board_size              = 19
trunk_size              = 19
cluster_window_size     = "none"
cluster_threshold       = "none"
legal_move_radius       = 5
n_planes                = 8
plane_layout            = [
  "current_player_t0", "current_player_t-1", "current_player_t-2", "current_player_t-3",
  "opponent_t0",       "opponent_t-1",       "opponent_t-2",       "opponent_t-3",
]
policy_logit_count      = 362            # 19*19 + 1
has_pass_slot           = true
is_multi_window         = false
value_pool              = "none"
policy_pool             = "none"
sym_table_id            = "size_19"
schema_version          = 1
notes                   = "v6 canonical encoding (Phase 1–3 anchor)."

# v7full — same wire format as v6, distinct variant tag for §150+ anchor
[encodings.v7full]
board_size              = 19
trunk_size              = 19
cluster_window_size     = "none"
cluster_threshold       = "none"
legal_move_radius       = 5
n_planes                = 8
plane_layout            = [
  "current_player_t0", "current_player_t-1", "current_player_t-2", "current_player_t-3",
  "opponent_t0",       "opponent_t-1",       "opponent_t-2",       "opponent_t-3",
]
policy_logit_count      = 362
has_pass_slot           = true
is_multi_window         = false
value_pool              = "none"
policy_pool             = "none"
sym_table_id            = "size_19"
schema_version          = 1
notes                   = "§150 anchor (17.4% n=500 vs SealBot); selfplay-canonical pending α."

# v6w25 — K-cluster 25×25 windows, matched-perception variant
[encodings.v6w25]
board_size              = 25
trunk_size              = 25
cluster_window_size     = 25
cluster_threshold       = 8
legal_move_radius       = 8
n_planes                = 8
plane_layout            = [
  "current_player_t0", "current_player_t-1", "current_player_t-2", "current_player_t-3",
  "opponent_t0",       "opponent_t-1",       "opponent_t-2",       "opponent_t-3",
]
policy_logit_count      = 626            # 25*25 + 1
has_pass_slot           = true
is_multi_window         = true
value_pool              = "min"          # per §170 P4 P1 NULL — K-invariant value head
policy_pool             = "scatter_max"  # per-move cluster dispatch (α design Option iii)
sym_table_id            = "size_25"
schema_version          = 1
notes                   = "§170 P4 P1 canonical for pretrain+eval+matched-MCTS; selfplay blocked on α."

# v8 — bbox-of-stones 25×25, 11 planes (history + 3 aux)
[encodings.v8]
board_size              = 25
trunk_size              = 25
cluster_window_size     = "none"
cluster_threshold       = "none"
legal_move_radius       = 8
n_planes                = 11
plane_layout            = [
  "current_player_t0", "current_player_t-1", "current_player_t-2", "current_player_t-3",
  "opponent_t0",       "opponent_t-1",       "opponent_t-2",       "opponent_t-3",
  "off_window_mask", "moves_remaining_bcast", "to_play_bcast",
]
policy_logit_count      = 625            # 25*25 (no pass slot — A3 amendment)
has_pass_slot           = false
is_multi_window         = false
value_pool              = "none"
policy_pool             = "none"
sym_table_id            = "size_25"
schema_version          = 1
notes                   = "v8 contract — single-bbox 25×25; plane-8 polarity OFF→OUTSIDE (vanilla); no pass slot (existing v8 ckpts have policy_fc.out_features=625)."

# v8_canvas_realness — v8 with plane-8 polarity flipped (inside→canvas)
[encodings.v8_canvas_realness]
board_size              = 25
trunk_size              = 25
cluster_window_size     = "none"
cluster_threshold       = "none"
legal_move_radius       = 8
n_planes                = 11
plane_layout            = [
  "current_player_t0", "current_player_t-1", "current_player_t-2", "current_player_t-3",
  "opponent_t0",       "opponent_t-1",       "opponent_t-2",       "opponent_t-3",
  "canvas_realness", "moves_remaining_bcast", "to_play_bcast",
]
policy_logit_count      = 625            # 25*25 (no pass slot — A3 amendment)
has_pass_slot           = false
is_multi_window         = false
value_pool              = "none"
policy_pool             = "none"
sym_table_id            = "size_25"
schema_version          = 1
notes                   = "§169 A4 / §170 P4 P2 variant; plane-8 polarity INSIDE→CANVAS for PartialConv2d trunk-entry. Pair with bootstrap_corpus_v8_canvas_realness.npz (sha 110ea6b2…)."
```

### 3.3 Schema validation

Parse-time validator (run by both Rust and Python at registry init):

- All required keys present; types match.
- `len(plane_layout) == n_planes`.
- `policy_logit_count == board_size * board_size + (1 if has_pass_slot else 0)`
  (A3 amendment 2026-05-09: v8 / v8_canvas_realness have no pass slot;
  per-encoding `has_pass_slot` bit added to schema).
- `cluster_window_size == "none"` ⇔ `cluster_threshold == "none"` ⇔
  `is_multi_window == false`.
- `is_multi_window == true` ⇒ `value_pool ∈ {"min","max","mean"}` and
  `policy_pool ∈ {"scatter_max","scatter_mean"}`.
- `is_multi_window == false` ⇒ `value_pool == "none"` and
  `policy_pool == "none"`.
- `trunk_size == cluster_window_size if is_multi_window else board_size`.
- `sym_table_id` resolves to a known sym table.

Parse failure raises a structured error (Rust: `EncodingRegistryError::Schema`;
Python: `EncodingRegistryError`) listing all offending keys, not just the
first.

---

## 4. Rust side — `engine/src/encoding/`

### 4.1 Layout

```
engine/src/encoding/
├── registry.toml         # canonical authoring file (§3)
├── mod.rs                # public API + lookup
├── spec/mod.rs           # EncodingSpec struct + Schema validator
└── (meta.rs — never created; merged into registry internals per §9 amendment)
```

### 4.2 EncodingSpec

```rust
pub struct EncodingSpec {
    pub name: &'static str,          // "v6" | "v6w25" | "v8" | ...
    pub board_size: usize,
    pub trunk_size: usize,
    pub cluster_window_size: Option<usize>,
    pub cluster_threshold: Option<usize>,
    pub legal_move_radius: usize,
    pub n_planes: usize,
    pub plane_layout: &'static [&'static str],
    pub policy_logit_count: usize,
    pub has_pass_slot: bool,
    pub is_multi_window: bool,
    pub value_pool: ValuePool,
    pub policy_pool: PolicyPool,
    pub sym_table_id: &'static str,
    pub schema_version: u32,
    pub notes: &'static str,
}

pub enum ValuePool  { None, Min, Max, Mean }
pub enum PolicyPool { None, ScatterMax, ScatterMean }
```

`EncodingSpec` is `Clone + Send + Sync + 'static` (string fields are
`&'static str` — they live in the registry's `Lazy` allocation, leaked at
init time so addresses are stable for the lifetime of the process).

### 4.3 Lazy parse + lookup

**Operator decision (A1 §6.6 + this doc §2): runtime parse via
`once_cell::sync::Lazy`.** No `build.rs`. `~1ms` parse cost, amortized to
zero across process lifetime. Simpler build, no codegen step, no
synchronization between TOML edits and a generated Rust file.

```rust
static REGISTRY: Lazy<HashMap<&'static str, &'static EncodingSpec>> =
    Lazy::new(|| { /* parse registry.toml, validate, leak strings, return map */ });

pub fn lookup(name: &str) -> Option<&'static EncodingSpec> { REGISTRY.get(name).copied() }
pub fn lookup_or_panic(name: &str) -> &'static EncodingSpec { /* clearer error */ }
pub fn all() -> impl Iterator<Item = &'static EncodingSpec> { REGISTRY.values().copied() }
```

`registry.toml` is embedded via `include_str!` at compile time (file
location relative to `engine/src/encoding/mod.rs`). This keeps the binary
self-contained — runtime doesn't need the file on disk at all. Python
side reads the same file from the repo tree (§5.3).

### 4.4 EncodingMeta hot-path companion

`engine/src/encoding/meta.rs` houses the `EncodingMeta` struct for
ReplayBuffer/sym_tables (A1 §6.6). Populated lazily at the same time as
`EncodingSpec`, one `EncodingMeta` per registered encoding, stored in a
parallel `Lazy<HashMap<&'static str, &'static EncodingMeta>>`.

`EncodingMeta` is the strided-storage view (n_cells, state_stride,
chain_stride, policy_stride, aux_stride, sym group). Derived from
`EncodingSpec` at init; never authored independently.

**Why this beats const presets:** single source of truth (TOML), still
gets `&'static` stable addresses for buffer push/sample. Bench-gate per
the `bench-gate` skill before A4 commit lands; expectation is no
measurable regression vs current `pub const V6_META` (struct-fields-on-
cache-line invariant preserved by `Lazy`).

> **Implementation note (post-§172):** The implementation collapsed
> `EncodingMeta` and `EncodingSpec` into a single `RegistrySpec` struct
> with `#[inline]` accessors. Authored fields and derived strided-storage
> fields share one source of truth; the separate
> `engine/src/encoding/meta.rs` module was not created. The rationale
> documented in §4.4 still applies — derived fields are computed once
> during registry load via `Box::leak` and accessed via `&'static`
> reference at hot-path call sites.

### 4.5 `Board::with_encoding(spec)`

Already exists from §171 A1 (`engine/src/board/state.rs:219`). A4 adapts
the signature to take `&'static EncodingSpec` from the registry instead
of a `PyEncodingSpec` round-tripped through Python. Validator unchanged.

### 4.6 Multi-window dispatch guard

`Board::to_planes_windowed(idx)` is an A7 (α) deliverable, not §172. For
this sprint, every multi-window dispatch site reads `spec.is_multi_window`
and, on `true`, surfaces:

```rust
if spec.is_multi_window {
    unimplemented!(
        "multi-window selfplay deferred to α; see \
         docs/designs/encoding_alpha_multiwindow_selfplay_design.md \
         (§172 Phase A7)"
    );
}
```

The error message is identical at every dispatch site. Loud, never silent.

---

## 5. Python side — `hexo_rl/encoding/`

### 5.1 New module layout

```
hexo_rl/encoding/
├── __init__.py           # re-exports lookup, EncodingSpec, resolve_*
├── registry.py           # parse + cache + lookup
├── spec.py               # EncodingSpec dataclass + validator
├── resolvers.py          # resolve_from_config, resolve_from_checkpoint
├── audit.py              # `python -m hexo_rl.encoding audit` entry point
└── compat.py             # backward-compat shape inference (with DeprecationWarning)
```

### 5.2 EncodingSpec dataclass

```python
@dataclass(frozen=True, slots=True)
class EncodingSpec:
    name: str
    board_size: int
    trunk_size: int
    cluster_window_size: int | None
    cluster_threshold: int | None
    legal_move_radius: int
    n_planes: int
    plane_layout: tuple[str, ...]
    policy_logit_count: int
    has_pass_slot: bool
    is_multi_window: bool
    value_pool: Literal["none", "min", "max", "mean"]
    policy_pool: Literal["none", "scatter_max", "scatter_mean"]
    sym_table_id: str
    schema_version: int
    notes: str

    @property
    def n_actions(self) -> int: return self.policy_logit_count

    def to_pyo3(self) -> "engine.PyEncodingSpec":
        # already exists post-§171 A2; A4 keeps signature, sources from registry
        ...
```

### 5.3 Registry parsing

```python
# registry.py
import tomllib
from importlib.resources import files

@functools.cache
def _load_registry() -> dict[str, EncodingSpec]:
    # Path: <repo_root>/engine/src/encoding/registry.toml
    # Discovery: walk up from hexo_rl/encoding/ until we find engine/src/encoding/registry.toml.
    # Sanity-check that the engine/ tree is co-located (in-repo dev install only;
    # wheel/sdist installs ship the file under hexo_rl/encoding/_registry.toml — see §5.4).
    raw = tomllib.loads(_read_registry_toml())
    return {name: _validate_and_build(name, body) for name, body in raw["encodings"].items()}

def lookup(name: str) -> EncodingSpec:
    spec = _load_registry().get(name)
    if spec is None:
        raise EncodingRegistryError(f"unknown encoding {name!r}; "
                                    f"registered: {sorted(_load_registry())}")
    return spec

def all_specs() -> Iterable[EncodingSpec]: return _load_registry().values()
```

### 5.4 Single-canonical-file decision

**Operator decision: single canonical file at
`engine/src/encoding/registry.toml`, parsed by both Rust and Python.**

- Rust embeds via `include_str!` at compile time (binary is
  self-contained).
- Python reads via path traversal from the package root, with a sanity
  check that `engine/src/encoding/registry.toml` is co-located in the repo.
  Dev install (the only supported install today) always satisfies this.
- For wheel/sdist parity (future), `setup.py` copies `registry.toml` into
  `hexo_rl/encoding/_registry.toml` at build time; Python falls back to
  the package-local copy if the engine/ tree isn't present.
- Symlink alternative rejected (Windows portability, pre-commit hook
  divergence).

### 5.5 Resolvers

```python
def resolve_from_config(cfg: Mapping[str, Any]) -> EncodingSpec:
    """
    Variant configs override only `encoding: <name>`. Resolver looks up.
    Backward-compat: if `encoding` is absent but legacy keys are present
    (e.g., `board_size: 19`), infer v6 (or v8 by board_size=25 + n_planes=11)
    with a DeprecationWarning. Reject scattered overrides (§10).
    """

def resolve_from_checkpoint(path: str | Path) -> EncodingSpec:
    """
    Read ckpt['metadata']['encoding_name'] if present. Else fall back to
    shape inference via compat.py with DeprecationWarning. Save path
    (in trainer) always writes new metadata.
    """

def validate_against_state_dict(spec: EncodingSpec, state_dict: Mapping[str, Tensor]) -> None:
    """
    Cross-check spec.policy_logit_count, spec.n_planes, spec.trunk_size
    against state_dict shapes (policy_fc out_features, first conv in_channels,
    feature-map side after trunk). Raise ShapeMismatch on disagreement.
    """
```

### 5.6 Compat shim

`hexo_rl/utils/encoding.py` was planned as a thin shim (as-built, the compat shim is `hexo_rl/encoding/compat.py`; neither `hexo_rl/utils/encoding.py` nor `hexo_rl/encoding/spec.py` was created standalone):

```python
# as-built: hexo_rl/encoding/compat.py
import warnings
from hexo_rl.encoding import lookup, resolve_from_config

warnings.warn(
    "hexo_rl.utils.encoding is deprecated; import from hexo_rl.encoding instead",
    DeprecationWarning, stacklevel=2,
)

# Re-export the names the codebase currently imports:
def v6_spec(): return lookup("v6")
def v6w25_spec(): return lookup("v6w25")
def v8_spec(): return lookup("v8")
def resolve_encoding(cfg): return resolve_from_config(cfg)

EncodingSpec = ...  # re-export
```

A4 lands the new module + shim in one commit; the shim survives until
every import site is migrated (target: same A4 chain). Removal lands
behind a separate commit when grep is clean.

---

## 6. TOML location — RESOLVED

See §5.4. Single canonical file at `engine/src/encoding/registry.toml`.
Embedded in Rust via `include_str!`; Python reads via path traversal with
package-local fallback for wheel/sdist installs.

---

## 7. Surface plumbing pass (A4 spec)

For each of the 23 load-bearing surfaces in A1 §5.1, the change pattern
is uniform:

**Rust (5 surfaces — A1 §5.1):**

```rust
// before
let board = Board::new();              // implicit v6 defaults

// after
let spec = encoding::lookup_or_panic(&runner.encoding_name);
let board = Board::with_encoding(spec);
```

**Python (10 runtime + 3 offline):**

```python
# before
N_ACTIONS = BOARD_SIZE * BOARD_SIZE + 1   # selfplay/utils.py:9
# after
from hexo_rl.encoding import lookup
def n_actions(spec): return spec.policy_logit_count
```

```python
# before
chain_planes = np.zeros((B, 6, 19, 19), dtype=np.float16)
# after
chain_planes = np.zeros((B, 6, spec.trunk_size, spec.trunk_size), dtype=np.float16)
```

**Boards carry encoding by reference, not by copy.** Each `Board` holds
`encoding: &'static EncodingSpec` (Rust) / `EncodingSpec` (Python). All
encoding-derived values read from `self.encoding.<field>`, never from a
module-level constant.

**Configs carry encoding by name, not by scalar.** `encoding: <name>` is
the only encoding-related key a variant config may set. Resolver looks
up the spec at config-load time. Validator rejects scattered overrides
(§10).

**Multi-window dispatch:** every site that reads `spec.is_multi_window`
must implement the `false` branch (single-window) and guard the `true`
branch with the `unimplemented!()` from §4.6. A7 fills in the `true`
branch.

**Per-game mutator audit:** per `feedback_encoding_post_mutators_audit.md`,
every mutator that runs after `Board::with_encoding(spec)` (legal_move
jitter, rotation, anything else) must guard with
`encoding.is_none()` unless its composition with the encoding is
explicitly designed. A4 review checklist includes this audit pass.

A4 is a sequence of small commits, one per subsystem (trainer, selfplay,
eval, model, bootstrap). Each commit has green tests for its subsystem.
13-task checklist (per A1 close-out §7) carries the per-subsystem
sequencing.

---

## 8. Checkpoint metadata

New mandatory `metadata` field at every checkpoint save site:

```python
{
    "step": int,
    "model_state": dict,
    "optimizer_state": dict,
    "scaler_state": dict,
    "scheduler_state": dict,
    "metadata": {
        "encoding_name": str,            # MANDATORY — resolves via registry
        "commit_sha": str,               # `git rev-parse HEAD` at save
        "training_date": str,            # ISO 8601 UTC
        "train_config_path": str,
        "corpus_sha256": str | None,
        "model_architecture": str,       # e.g. "HexTacToeNet_v8"
        "model_variant": str | None,     # sub-arch tag — e.g. "B1_128x12_GPool6_10",
                                         # "PMA_global", "min_max"; None when no
                                         # ablation arm. Not load-bearing for shape
                                         # inference; differentiator for audit + ablation
                                         # bookkeeping. Stamp `None` if no variant.
        "schema_version": int,           # = 1 for this design
    },
}
```

**Save path:** `Trainer.save_checkpoint` writes the metadata dict
unconditionally. No code path may save without it.

**Load path:** `Trainer.load_checkpoint` and `eval/checkpoint_loader.
detect_encoding_label` read `metadata["encoding_name"]` if present, fall
back to `compat.infer_encoding_from_state_dict(state_dict)` with a
DeprecationWarning otherwise. Inferred encoding is treated as a
suggestion — operator can override via CLI flag.

**Backward-compat (one-time):** `scripts/migrations/2026_05_09_stamp_artifact_metadata.py`
(date-prefix naming, preserved post-run) stamps the 25 existing
checkpoints with metadata driven by an opt-in operator-authored manifest
YAML, not best-effort guess. `bootstrap_model_v8full_warm.pt` is
archived in this pass (operator confirmed: interim v6 ckpt mislabeled).

---

## 9. Corpus metadata

New `<corpus_name>.metadata.json` next to each `.npz` (sidecar, not
embedded — keeps `.npz` archive bytes immutable for sha-stability):

```json
{
    "encoding_name": "v6w25",
    "sha256": "<self-hash of .npz>",
    "n_positions": 353091,
    "source_manifest": "<path or url>",
    "created_at": "2026-05-09T14:30:00Z",
    "created_by_commit": "<git rev-parse HEAD>",
    "schema_version": 1,
    "extra": {
        "cluster_window_size": 25,
        "cluster_threshold": 8,
        "canvas_realness": false
    }
}
```

`pretrain.load_corpus`, `make_augmented_collate`, and every other corpus
consumer call `validate_corpus_metadata(npz_path)` at load. Mismatch
between sidecar `encoding_name` and resolved-config `encoding_name`
raises `EncodingMismatchError`. Closes §171 P3 deltas item 2.

Backward-compat: `validate_corpus_metadata` accepts a missing sidecar
with DeprecationWarning + treats the corpus as `encoding_name=<inferred
from filename>`. The migration script (§8) generates sidecars for the 8
existing corpora from operator manifest.

---

## 10. Variant config schema

**Allowed encoding-related keys in variant configs:**

```yaml
encoding: v6w25                # MANDATORY if differing from base default
```

**Scattered-key semantics (A8 amend, 2026-05-10) — consistency-not-equality:**

Scattered encoding-derived keys (`board_size`, `cluster_window_size`,
`cluster_threshold`, `n_planes`, `in_channels`) are accepted **iff they
agree with the registry value implied by `encoding:`**. Disagreement
raises `EncodingRegistryError` at load.

This is a deliberate relaxation of the original "always reject" rule.
Rationale:
- Variant configs inherit from `configs/model.yaml:5` which carries
  `board_size: 19` for v6 base defaults. Forcing literal-rejection would
  require either rewriting every base config or deleting `board_size`
  from `model.yaml` entirely (breaks any non-registry consumer).
- Consistency-checked acceptance preserves the inheritance flow while
  catching real mismatches (e.g. `encoding: v6w25` + `board_size: 19`
  fails loud).

Validator message on disagreement:

```
EncodingRegistryError: variant 'sprint_171_p3_5080' declares
'encoding: v6w25' but also sets 'board_size: 19'. Registry says v6w25
has board_size=25. Either remove the scattered key or correct it to 25.
Registered encodings: v6, v6w25, v7full, v8, v8_canvas_realness.
```

A4 retires this relaxation transitively via Option 3 (full
`config["board_size"]` retirement — see §14): once no consumer reads
the legacy scalar, the scattered keys can be stripped from configs.
Until then, consistency-not-equality is the contract.

`resolve_corpus_path(spec)` and `resolve_anchor_path(spec)` helpers
remove the need for variants to spell out corpus + bootstrap_anchor
paths separately (the sprint_171_p3_5080.yaml example variant was never committed to master; A4 cleans up
once helpers land).

Backward-compat: legacy configs with `board_size` and no `encoding` key
are accepted with DeprecationWarning + inferred encoding (v6 by default;
v8 if `n_planes == 11`). DeprecationWarning carries the correct
replacement spelling for copy-paste.

---

## 11. Audit CLI utility

```
$ python -m hexo_rl.encoding audit [--strict] [--checkpoints-dir DIR]
                                   [--corpora-dir DIR] [--variants-dir DIR]
```

**Output sections:**

1. **Registered encodings:** name, board_size, n_planes, n_actions,
   is_multi_window, schema_version.
2. **Checkpoints:** for every `.pt` under `checkpoints/`, declared
   `metadata["encoding_name"]` vs inferred (filename + state-dict
   shape). Flag mismatches (esp. `v8full_warm.pt` ambiguity).
3. **Corpora:** for every `.npz` under `data/`, sidecar
   `encoding_name` vs inferred. Flag mismatches; flag missing sidecars.
4. **Variants:** for every `configs/variants/*.yaml`, resolved encoding
   under loadout (from variant explicit, from base default, from CLI).
   Flag scattered-override rejections.
5. **Hardcoded-literal grep:** scan encoding-aware surfaces (A1 §3.1
   load-bearing list + A1 §5.1 +A1 §5.2) for `19`, `25`, `361`, `5`,
   `8` literals not paired with a `spec.<field>` reference. Flag any
   not justified by a comment pointing at the registry.
6. **Cross-table consistency** (A4 amend, 2026-05-10): joins §2
   checkpoints to §3 corpora via `metadata.corpus_sha256` ↔ corpus
   sidecar `sha256`. Per-checkpoint verdict in `{OK, ENC-MISMATCH,
   ORPHAN-SHA, NO-CORPUS-SHA, NO-META}`. Plus orphan-corpus pass: any
   corpus sha not referenced by any stamped ckpt is reported (info by
   default; warn under `--strict`). Severity matrix:
   - `error` — ckpt + corpus both stamped, encodings disagree (INV-1).
   - `error` — ckpt has `corpus_sha256` but no corpus matches (INV-2).
   - `warn` — ckpt stamped but `corpus_sha256 is None` (INV-3).
   - `warn` — ckpt has no metadata at all (INV-4).
   - `info` — clean match (INV-5) or orphan corpus (INV-6).

   Skip rule: when `corpora_dir` is empty AND any ckpt carries
   `corpus_sha256`, emit one section-level warn ("§6 skipped — no
   corpora to join against") and produce no per-row findings.

**Exit codes:**

- `0` — all sections clean.
- `1` — at least one warning (missing sidecar, deprecated config,
  hardcoded literal not annotated).
- `2` — at least one error (mismatch, schema violation, unknown
  encoding referenced).

**Use cases:**

- Pre-flight gate before any sustained run (operator runs before
  launching tmux session).
- CI gate — A8 wires it into `make test` or `make lint`.
- Onboarding — single command surfaces the encoding posture of every
  artifact.

---

## 12. Cross-encoding round-trip test

`tests/test_encoding_round_trip.py`, parameterized over every encoding
in the registry:

```python
@pytest.mark.parametrize("name", [s.name for s in all_specs()])
def test_round_trip(name):
    spec = lookup(name)

    # 1. Spec ↔ TOML round-trip.
    assert lookup(spec.name) is spec  # registry returns stable instance.

    # 2. Board construction.
    board = Board.with_encoding(spec)  # PyO3
    assert board.board_size == spec.board_size

    # 3. Tensor projection.
    if spec.is_multi_window:
        with pytest.raises(NotImplementedError, match="α"):
            board.to_tensor()        # single-window path; multi-window deferred to α
        views = board.get_cluster_views()
        # K cluster views, each shape (n_planes, cluster_window_size, cluster_window_size)
        assert all(v.shape == (spec.n_planes, spec.cluster_window_size, spec.cluster_window_size)
                   for v in views)
    else:
        tensor = board.to_tensor()
        assert tensor.shape == (spec.n_planes, spec.trunk_size, spec.trunk_size)

    # 4. Network ctor.
    net = HexTacToeNet(encoding=spec)
    out = net(torch.zeros(1, spec.n_planes, spec.trunk_size, spec.trunk_size))
    value, policy, aux = out
    assert policy.shape == (1, spec.policy_logit_count)

    # 5. Checkpoint compat (if a fixture ckpt exists for this encoding).
    fixture = _find_fixture_ckpt(spec.name)
    if fixture:
        state = torch.load(fixture, map_location="cpu")
        validate_against_state_dict(spec, state["model_state"])

    # 6. MCTS one-sim smoke.
    if spec.is_multi_window:
        with pytest.raises(NotImplementedError, match="α"):
            run_one_mcts_sim(board, net)
    else:
        run_one_mcts_sim(board, net)   # green
```

**Fixture checkpoints** — to keep the test fast and reproducible without
shipping multi-MB `.pt` files in-tree:

- Synthetic state-dicts generated at module-scope (random init, correct
  shapes), one per encoding.
- Real fixture checkpoints (small, e.g., 5k-step) committed under
  `tests/fixtures/encoding/<name>.pt` — only if A6 finds the synthetic
  path doesn't catch enough regressions.

A6 deliverable; integrates into `make test` and CI.

---

## 13. α design doc — forward pointer

A7 produces `docs/designs/encoding_alpha_multiwindow_selfplay_design.md`,
expanding the scope memo (`encoding_alpha_multiwindow_selfplay.md`) into
a full design:

- Option iii (single tree, per-move cluster dispatch) chosen — see scope
  memo §2.
- New Rust API: `Board::to_planes_windowed(window_idx, cluster_window_size)`.
- Inference batcher K-per-position fan-out.
- MCTS expansion-time per-move cluster dispatch.
- Replay buffer schema: store board state, recompute K windows on read.
- Value head: K-invariant min-pool at root (per §170 P4 P1 NULL verdict).
- Test plan: K-window round-trip, cluster dispatch correctness, value-head
  K-invariance regression.
- Estimated effort: 1–2 weeks on laptop.

This doc is the contract for §173+ implementation. A7 produces it; A8
adds the cross-references in `01_architecture.md` and `02_roadmap.md`.

---

## 14. Sequencing — A3 through A9

Each phase is a separate commit (or commit chain) on
`phase4/encoding_registry`. Each ends with green tests and an updated
sprint log entry.

| Phase | Scope | Done-when |
|---|---|---|
| **A3** | Registry implementation. New `engine/src/encoding/{registry.toml, spec.rs, meta.rs}` + `hexo_rl/encoding/{registry.py, spec.py, resolvers.py, audit.py, compat.py}`. Schema validator + parse-time errors. Unit tests for registry parse + lookup + schema rejection. No consumer rewrites yet. | All 5 worked TOML examples (§3.2) registered. `pytest tests/test_encoding_registry.py` green. `cargo test --test encoding_registry` green. Compat shim (`hexo_rl/utils/encoding.py`) re-exports correctly. |
| **A4** | Plumbing pass — 23 load-bearing surfaces (A1 §5.1) + retire `config["board_size"]` (A1 §6.11 Option 3). One commit per subsystem (trainer, selfplay, eval, model, bootstrap). Per-game mutator audit per `feedback_encoding_post_mutators_audit.md`. **Bench-gate** the EncodingMeta migration (§4.4) per the `bench-gate` skill. | All consumers read `spec.<field>` not module constants. `make test` green. `cargo test` green. Bench positions/hr within ±2% of pre-A4 baseline. §171 P3 v7full smoke pre-flight passes (cold smoke ≥1k positions, no shape errors). |
| **A5** | Audit CLI (§11) + checkpoint metadata schema (§8) + corpus metadata schema (§9). Save path writes new metadata; load path reads with backward-compat fallback. One-time migration script `scripts/migrations/2026_05_09_stamp_artifact_metadata.py` + operator manifest. Archives `bootstrap_model_v8full_warm.pt`. | `python -m hexo_rl.encoding audit` runs clean (exit 0) on stamped artifacts. New ckpts/corpora always carry metadata. Migration script idempotent + dry-runnable. |
| **A6** | Cross-encoding round-trip test (§12). Parameterized over registry; covers Board, Tensor, Network, ckpt-compat, 1-sim MCTS smoke. | `pytest tests/test_encoding_round_trip.py -v` shows 5+ green test nodes (one per registered encoding); v6w25 NotImplementedError-asserts on the multi-window path with the α deferral message. CI green. |
| **A7** | α design doc (§13). Full spec for §173+ implementation. Scope memo elevated. | `docs/designs/encoding_alpha_multiwindow_selfplay_design.md` committed. Cross-references added in scope memo + sprint log + this doc. |
| **A8** | Doc cleanup. `01_architecture.md` v8 section. `04_bootstrap_strategy.md` v8 corpus. `board-representation.md` post-§172 invariants. `perf-targets.md` v8 baseline. `encoding_migration_v8.md` §2.3 plane-label fix per v8 contract. `00_agent_context.md` registry pointer. | `make doctest` green. Manual review of changed docs. |
| **A9** | Review pass. Spawn `wave-audit` against the full A3-A8 changeset. Bench gate sign-off. §171 P3 cold smoke re-run on `phase4/encoding_registry` HEAD. | `wave-audit` returns no critical findings. Cold smoke ≥1k positions clean. Sprint log entry §172 close-out + memory updates. Branch ready to merge to master (or to feed §171 P3 sustained smoke direct, operator's call). |

**Gating:**

- A3 must land before A4 (consumer rewrites need the registry).
- A4 must land before A5 (metadata schema changes touch the same save/load
  paths the plumbing rewrites).
- A6 can run in parallel with A5 (no shared files; `tests/test_encoding_round_trip.py`
  stands alone).
- A7 can run in parallel with A4-A6 (pure docs, no code dependency).
- A8 follows A7 (doc cross-references depend on A7's filename).
- A9 is the final gate.

**Out of scope for §172:** α implementation (§173+), sym-table
parameterization for v8 if it requires more than `EncodingMeta` fields
already covers (defer to §173+ per A1 §6.6.1), Rust `MCTSTree`
parameterization (defer to §173+).

---

## 15. Done-when (this doc)

- All 14 sections populated. ✓
- TOML schema worked example for v6, v6w25, v7full, v8,
  v8_canvas_realness. ✓
- Rust + Python module API surface specified. ✓
- A3-A9 sequencing locked. ✓
- 1 commit on `phase4/encoding_registry`:
  `docs(172): A2 design — encoding registry single-source-of-truth`.
- Surface to operator: design ready for A3 implementation; A3 implements
  without re-litigating; any design change A3 needs returns to A2.

---

**End of A2 design doc.** Proceed to A3 dispatch.
