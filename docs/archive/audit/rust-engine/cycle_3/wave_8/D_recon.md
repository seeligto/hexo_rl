# Wave 8 Batch D — IMPL recon

**Branch:** `refactor/rust-engine-cycle-3`
**Entry HEAD:** `9f0f2dc` (Batch C close — FF.10 SelfPlayRunner encoding_name + WireFormatSpec retire + legacy-v6-fallback collapse)
**Subagent:** Batch D IMPL — naming-fold sweep (model layer + Batch A deferred fold-credits)
**Date:** 2026-05-17

---

## §1 — PREP claim vs HEAD verification (SD4)

Each rename target verified by `rg` at HEAD `9f0f2dc`.

### 1.1 — `min_max_v6_head` / `network_v6_head` (PREP §F item 1, CONSOLIDATE #5 + GENERICISE #5)

| Site | PREP value | HEAD actual | Status |
|---|---|---|---|
| `hexo_rl/model/network_v6_head.py` | file exists | **exists** (88 LOC; function def at L37 + docstring mentions at L17) | match |
| Import in `hexo_rl/model/network.py:796` | 1 import | **L45 import + L781 docstring + L796 call + L929 docstring + L932 call** (5 references total) | PREP undercount: actual = 5 references |
| External callers | "1 file + 1 import site" | **0 external imports** (only the model module imports it) | match |
| Tests | not specified | **0 test imports** | clean |

Net touch: `network_v6_head.py` (1 file rename + 1 fn rename + 1 docstring mention), `network.py` (1 import line + 4 internal references — 1 import + 2 docstring + 2 call sites).

### 1.2 — Replay aliases (PREP §B.1 GENERICISE #1-#3, deferred from Batch A)

| Site | PREP value | HEAD actual | Status |
|---|---|---|---|
| `hexo_rl/bootstrap/replay_triples.py:22-24` | 3 aliases | **3 aliases at L22/L23/L24** + 3 call-sites at L82/L85/L95 | match |
| External callers | not specified | **0 external imports** (private-prefix module-locals); 1 hit at `tests/test_dataset_v8.py:398` is a function name `test_v8_replay_board_uses_R_8_perception` (test pin name, NOT an import of `_v8_replay`) | clean |

Net touch: `hexo_rl/bootstrap/replay_triples.py` only.

### 1.3 — `_build_v6_model` / `_build_v8_model` (PREP §B.1 GENERICISE #4, deferred from Batch A)

| Site | PREP value | HEAD actual | Status |
|---|---|---|---|
| `hexo_rl/eval/checkpoint_loader.py:149,197` | 2 defs | **L149 def `_build_v6_model`** + **L197 def `_build_v8_model`** + dispatch at L134/L139/L142 | match |
| External callers | "param type EncodingSpec migrates; functions take RegistrySpec" | **0 external imports**; comment-only mentions at `tests/test_gpool_bias_eval_plumbing.py:35` + `tests/test_k_cluster_mcts_bot.py:356, 383` | comments need update |

Net touch: `hexo_rl/eval/checkpoint_loader.py` (dispatcher + 2 fn renames) + 3 comment references in 2 test files (cosmetic doc sync).

### 1.4 — `_v6_net` / `_v8_net` test helpers (PREP §B.2 CONSOLIDATE #4, deferred from Batch A)

| Site | PREP value | HEAD actual | Status |
|---|---|---|---|
| `hexo_rl/training/tests/test_checkpoint_metadata.py:47,52` | 2 helpers | **L47 def `_v6_net`** + **L52 def `_v8_net`** + 5 call sites at L96/L122/L138/L164/L180 | match |
| External callers | not specified | **0 imports**; `tests/test_encoding_auto_detect.py:43,48` has identical-shape `_v6_net`/`_v8_net` test helpers but is OUT OF PREP §F SCOPE | out-of-scope per PREP |

Net touch: `hexo_rl/training/tests/test_checkpoint_metadata.py` only. Sibling `tests/test_encoding_auto_detect.py` deferred (not in PREP scope; opportunistic-cleanup-suppression per SD3).

### 1.5 — `compute_v8_mask` (PREP §F item 3 + §B.1 GENERICISE #6; operator-locked DEFER)

| Site | PREP value | HEAD actual | Status |
|---|---|---|---|
| Operator pre-decision | **DEFER to cycle 4+** | **N/A** | untouched |
| Verification | `rg` confirms no Batch D touch | **6 hits across 3 files** (`tests/model/test_gpool.py:8, 22, 122, 131`, `hexo_rl/model/network.py:14, 41, 717`, `hexo_rl/model/gpool.py:243`) — all preserved | confirmed untouched in Batch D scope |

Action: **NONE** — operator-locked DEFER.

---

## §2 — `_build_v6_model` vs `_build_v8_model` body analysis (collapse decision)

Read `hexo_rl/eval/checkpoint_loader.py:149-235` in full.

### 2.1 — Body shape

`_build_v6_model` (L149-194):
- Detects `pool_type` ∈ {`pma_global`, `pma`, `min_max`} via `global_encoder.*` / `cluster_pool.*` key presence.
- Detects `gpool_bias_active` via `gpool_bias_branch.*` key presence + cross-check vs `pool_type='min_max'`.
- Constructs `HexTacToeNet(encoding="v6", pool_type=pool_type, gpool_bias_active=gpool_bias_active, ...)`.
- `strict=False` load (allows legacy `tower.*` ↔ `trunk.tower.*` aliasing).

`_build_v8_model` (L197-235):
- Detects `canvas_realness` via `trunk.input_conv.conv.weight` key presence (PartialConv2d wrap).
- Detects `gpool_indices` per-block via `trunk.tower.{i}.conv1.conv1g.weight`.
- Detects `head_use_gpool` via `policy_head.conv1g.weight` key.
- Constructs `HexTacToeNet(encoding="v8", gpool_indices=..., head_use_gpool=..., canvas_realness=..., ...)`.
- `strict=True` load.
- Asserts `in_channels == 11`.

**Verdict:** the two bodies are architecturally distinct (different feature detection logic, different constructor kwargs, different strict-load policy). A naive "single body with internal branching" inlining would produce a ~80 LOC method with two unrelated code paths gated on `spec.has_pass_slot`.

### 2.2 — Collapse target

Per inventory recommendation "fold into a single `_build_model_from_spec(spec)`": the cleanest fold preserves the architectural split internally but exposes a single unified entry point + renames the internal helpers to architectural names (not registry-version names).

**Decision:**
1. Add `_build_model_from_spec(state, spec)` entry-point dispatcher that branches on `spec.has_pass_slot` (the registry-fielded discriminator: True for v6/v6w25/v7full/v7/v7e30/v7mw; False for v8/v8_canvas_realness).
2. Rename `_build_v6_model` → `_build_min_max_model` (descriptive: builds the min_max-family network used for v6 / v6w25 / v7full).
3. Rename `_build_v8_model` → `_build_kata_model` (descriptive: builds the KataGo-family network used for v8 / v8_canvas_realness; reflects `KataGoPolicyHead` usage).
4. Replace dispatch site at L134-142 with a single call to `_build_model_from_spec(state, spec)`.

This satisfies inventory recommendation (single registry-spec-keyed entry point) without destabilising the two architecturally-distinct construction paths. SD4-permitted scope tightening preserves separate-body architecture.

### 2.3 — Comment-sync at external sites

Three comment references in tests reference the old function names:
- `tests/test_gpool_bias_eval_plumbing.py:35` — "Mirrors ``checkpoint_loader._build_v6_model`` which constructs..."
- `tests/test_k_cluster_mcts_bot.py:356` — "_build_v6_model rewrites the encoding to 'v6'..."
- `tests/test_k_cluster_mcts_bot.py:383` — "checkpoint_loader._build_v6_model must read 'pma_global'..."

Update comments to cite the new dispatcher (`_build_model_from_spec` / `_build_min_max_model`). Cosmetic doc sync; not behavioural.

---

## §3 — SD3 forced expansion

Per Batch D scope (cosmetic renames only):

| Site | In PREP §F? | Reason for forced expansion |
|---|---|---|
| `tests/test_gpool_bias_eval_plumbing.py:35` | NO | Comment references retired name `_build_v6_model`; doc sync required for accuracy. Minimal edit (1 string update). |
| `tests/test_k_cluster_mcts_bot.py:356, 383` | NO | Same — 2 comment references; minimal edit (2 string updates). |

Total forced expansion: 2 test files (3 comment-only string updates). No behavioural changes; no opportunistic cleanup. Per SD3 rule "minimal edit; no opportunistic cleanup".

---

## §4 — `!`-marker decision

All renames target **private-prefix** Python identifiers:
- `min_max_v6_head` → module-private function (used only inside model layer).
- `network_v6_head.py` → module-private file (no top-level `__all__` export; only one import path `hexo_rl.model.network`).
- `_v6_replay` / `_v6w25_replay` / `_v8_replay` → underscore-prefix module-local aliases.
- `_build_v6_model` / `_build_v8_model` → underscore-prefix module-private functions.
- `_v6_net` / `_v8_net` → underscore-prefix test-local helpers.

`rg` sweep across `hexo_rl/`, `tests/`, `scripts/` confirms **zero `import` / `from ... import` statements** reach any of these names from outside their defining modules. The 3 comment references in `tests/test_gpool_bias_eval_plumbing.py` + `tests/test_k_cluster_mcts_bot.py` are documentation strings, not import bindings.

**Decision: NO `!`-marker.** Pure-Python-internal rename; zero externally-observable contract change.

---

## §5 — Commit-body disclosure draft

### Scope
- Original PREP §F renames: file rename `network_v6_head.py` → `network_min_max_head.py` (CONSOLIDATE #5); function rename `min_max_v6_head` → `min_max_window_head` (GENERICISE #5).
- Batch A's deferred fold-credits: replay aliases collapse to `_REPLAYERS` dispatcher dict (GENERICISE #1-#3); `_build_v6_model`/`_build_v8_model` → `_build_min_max_model`/`_build_kata_model` + unified `_build_model_from_spec` dispatcher (GENERICISE #4); `_v6_net`/`_v8_net` → `_net_from_spec(spec)` (CONSOLIDATE #4).

### Pre-decision citation
- U3 `compute_v8_mask` polarity-flip rename: **DEFERRED** to cycle 4+ encoding-schema wave per operator pre-decision (PREP §M item 1). A `mask_polarity` TOML field would force registry schema bump (8 entries to re-validate) — too much scope for naming-fold batch. `spec.notes` is a stale-comment vector. `compute_v8_mask` left untouched at HEAD `9f0f2dc`.

### Batch A scope-deferral resolution
Batch A (`a6ca01b`) deferred PREP §B.5 GENERICISE/CONSOLIDATE fold-credits (#1-#4 + CONSOLIDATE #4) to keep the FF.2 EncodingSpec retirement anchor diff minimal (A_recon §3 verdict). Batch D picks these up alongside the original PREP §F naming-fold scope.

### `!`-marker decision
**NO `!`-marker.** All renames target private-prefix Python identifiers (module-private file `network_v6_head.py`, private functions `min_max_v6_head`, `_v6_replay`, `_build_v6_model`, `_v6_net`, etc.). SD4 `rg` sweep confirms zero `import` statements outside defining modules.

### SD3 forced expansion items
- `tests/test_gpool_bias_eval_plumbing.py:35` — 1 comment-string update (rename mention).
- `tests/test_k_cluster_mcts_bot.py:356, 383` — 2 comment-string updates (rename mentions).

### SD4 PREP corrections
- PREP §F item 1: "1 import site" at `network.py:796` → actual = **5 references** (L45 import + L781 docstring + L796 call + L929 docstring + L932 call). All updated in this commit.
- PREP §B.1 GENERICISE #4: `_build_v6_model`/`_build_v8_model` collapse target = single `_build_model(spec)` body. SD4 analysis (D_recon §2.1) found the two bodies are architecturally distinct (different feature detection, different `strict` load policy, different constructor kwarg set). Adopted SD4-permitted scope tightening: introduce `_build_model_from_spec` entry-point dispatcher + rename internal helpers `_build_min_max_model` / `_build_kata_model` (architectural names) — preserves separate bodies while satisfying inventory's "unified entry point keyed on spec" recommendation.
- PREP §F item 2 implicit assumption that comment-only test references can be left stale: SD4 verdict = update for documentation accuracy (3 comment updates total).

### Per-rename mapping table

| # | Old | New | File | Caller count |
|---|---|---|---|---|
| 1 | `network_v6_head.py` (file) | `network_min_max_head.py` | `hexo_rl/model/` | 1 import (network.py:45) |
| 2 | `min_max_v6_head` (fn) | `min_max_window_head` | `hexo_rl/model/network_min_max_head.py` | 2 calls in network.py (L796 + L932) |
| 3 | `_v6_replay` / `_v6w25_replay` / `_v8_replay` (3 aliases) | `_REPLAYERS` dict | `hexo_rl/bootstrap/replay_triples.py` | 3 internal call sites collapsed to dispatcher lookup |
| 4 | `_build_v6_model` (fn) | `_build_min_max_model` | `hexo_rl/eval/checkpoint_loader.py` | 0 external; called via new `_build_model_from_spec` dispatcher |
| 5 | `_build_v8_model` (fn) | `_build_kata_model` | `hexo_rl/eval/checkpoint_loader.py` | 0 external; called via new dispatcher |
| 6 | dispatch at L134-142 | `_build_model_from_spec(state, spec)` | `hexo_rl/eval/checkpoint_loader.py` | 1 internal call site |
| 7 | `_v6_net` / `_v8_net` (fns) | `_net_from_spec(spec)` | `hexo_rl/training/tests/test_checkpoint_metadata.py` | 5 internal call sites collapsed |

### Gate results (Phase 3)
- pytest: 1585 → 1585 ± Δ (Δ predicted ≤2; helper collapse may reduce by 0–2 test count via fixture consolidation; SD4 confirms at Phase 3 run-time).
- cargo test --package engine: **268** (unchanged — zero Rust touch).
- cargo clippy --package engine --release: **42 warnings floor** (unchanged — zero Rust touch).
- Module load smoke: `from hexo_rl.model.network_min_max_head import min_max_window_head` → ok.
- `git status` shows R (renamed) for `network_v6_head.py` → `network_min_max_head.py`.

---

## §6 — Operator-escalation items

**None.** Operator pre-decisions cover:
- U3 `compute_v8_mask` DEFER (operator-locked).
- SD3 expansion scope-by-deletion permitted (used for 3 comment-only updates).
- SD4 collapse-target tightening for `_build_v6_model`/`_build_v8_model` (architecturally-distinct bodies preserved as separate internal helpers).

Sibling test file `tests/test_encoding_auto_detect.py` has identical-shape `_v6_net`/`_v8_net` helpers but is OUT OF PREP §F SCOPE (only `hexo_rl/training/tests/test_checkpoint_metadata.py:47,52` was flagged). Per SD3 no-opportunistic-cleanup rule: not touched. Inventory may flag this in Phase 5 sweep.
