# S5 — fresh held-out game set mint (post-R2c BURN)

Status: **DONE**. Untracked report (per instruction — do NOT git-add).

## Context

`docs/registers/run3_corpus_manifest.md` §2 ruled to KEEP the 1796-game
laptop-only delta INSIDE the frozen run3 training corpus
(`data/bootstrap_corpus_v6_live2_ls.npz`, sha `3813edc2…`), which BURNED it
as a held-out set for future BC/architecture reads (it was the D-M R2
BC-scaling held-out set, R2's read is done, and run3 doesn't need held-out
purity). §2 named the fix explicitly: mint a fresh held-out from "the 29
build-excluded games + all subsequent daily-scrape tail". This is that mint.

## Source store + count scanned

- Source: `/home/timmy/Work/Hexo/hexo_rl/data/corpus/raw_human` (main
  checkout, **READ-ONLY** — never written to). 8,698 `.json` game files as
  of scan time (2026-07-15), matching `data/corpus/manifest.json`
  (`last_updated: 2026-07-07T18:52:35Z`, `human_games: 8698` — unchanged
  since, i.e. the daily scrape has not produced new games since 2026-07-07).
- All 8,698 files scanned (full `raw_dir.glob("*.json")`).

## Exclusion logic vs the canonical NPZ (how membership was computed)

The canonical NPZ (`data/bootstrap_corpus_v6_live2_ls.npz`, sha `3813edc2…`)
was built 2026-07-04 by globbing `raw_human` at build time; `run3_corpus_
manifest.md` §2 documents 8669-of-8698 went in, 29 did not (build cutoff
09:44 local). Since `raw_human` is append-only (verified: box-6902 ⊆
laptop-8698 byte-identical), file **mtime** is a sound membership proxy —
independently re-derived on the current 8,698-file tree:

| cutoff (local) | mtime <= cutoff | mtime > cutoff |
|---|---:|---:|
| 2026-07-04 09:44:00 | 8,669 | 29 |
| 2026-07-04 09:56:30 (NPZ completion mtime) | 8,669 | 29 |

Zero files land in the `(09:44, 09:56]` gap, so both candidate cutoffs agree
exactly and reproduce the manifest doc's documented 8669/29 split — this is
the correctness check for the method. Held-out candidates = every file with
mtime **strictly after `2026-07-04T09:44:00` local**. Today that is exactly
the same 29 "build-excluded" games (no newer daily-scrape tail exists yet).

## v7-filter provenance

No literal string "v7-filter" exists anywhere in the codebase (searched
`hexo_rl/`, `scripts/`, `docs/`, `engine/`) — read this as the corpus
pipeline's own **ingestion filter**, applied identically to the canonical
corpus and re-validated (not just trusted) here:

1. `HumanGameSource._passes_filter` (`hexo_rl/corpus/sources/human_game_source.py:97`):
   `rated == true`, `moveCount >= 20`, `gameResult.reason ==
   "six-in-a-row"`. Matches `data/corpus/manifest.json`'s `"filter"` field
   verbatim.
2. Corpus-pipeline discipline (`hexo_rl/probes/gnn_bc/bc_data.py`, mirrored
   by `scripts/export_corpus_npz.py`): `MIN_GAME_LENGTH=15`, position window
   `[2, 150)`.

Cited precedent for "same ingestion filter" framing:
`reports/probes/gnn_bc/r2/R2C_REDTEAM.md` §1 (R2c's own held-out caveat:
"same source population... same ingestion filter: rated, >=20 moves,
six-in-a-row").

All 29 candidates passed both filters — **zero drops**
(`n_ingestion_filter_dropped: 0`, `n_min_game_length_dropped: 0`).

## Final held-out set

- **Count: 29 games** (2,963 `v6_live2_ls` rows — 2,090 plies emitted, 10
  plies dropped outside all cluster windows).
- **Date range** (`startedAt_day`, UTC day-granularity): **2026-07-04 →
  2026-07-07**. Scrape (mtime) range: 2026-07-04T17:34:16 →
  2026-07-07T20:35:46 local.
- Elo range 877–1616 (n=58 rated-player values); band counts sub_1000: 6,
  1000_1200: 14, 1200_1400: 7, 1400_plus: 2.
- **NPZ sha256: `88f99c2b5fea7495484e4e9cc1af831d1e053221dc7e0f9c8f5d3ab6f27aa69e`**
  (independently verified via `sha256sum`, matches `compute_npz_sha256`).
  Size 12,872,280 bytes.
- Games-manifest sha256 (sorted per-game hash list, mirrors
  `corpus_check.py`'s `corpus_sha256`):
  `61f4f227e1d79af55f9435bd5b82cc044d71c9c4a04d1766a7bae4b6d80858e2`.
- Caveat (small n — mirrors R2c's own): this is genuinely everything scraped
  since the canonical corpus's build cutoff; the daily scrape has been idle
  since 2026-07-07. Not a different population, mild distribution shift
  (later calendar window) only.

## Loader-assertion design

Extends the WP0.4 single-resolver (`docs/registers/run3_corpus_manifest.md`
§3) with a second, symmetric sha registry:

- `_HELDOUT_CORPUS_SHAS: dict[str, tuple[str, int]]` in
  `hexo_rl/encoding/resolvers.py` — `label -> (sha256, size_bytes)`.
- **Resolver-level static invariant** `_assert_no_registry_overlap()`,
  called at import time: `_CORPUS_SHA_PINS` ∩ `_HELDOUT_CORPUS_SHAS` (by
  sha) must be empty. Fails at the FIRST import of the resolvers module on
  a bad entry, not at some later launch.
- **Load-level hard gate** in
  `hexo_rl.training.batch_assembly.load_pretrained_buffer`: before the
  existing WP0.4 pin check, a cheap `os.path.getsize()` (metadata-only, no
  stream) checks whether the target file's size matches ANY registered
  held-out artifact's size. Only on a size match does it stream a sha256
  and call `assert_not_heldout_sha`, which raises `ValueError` naming the
  held-out label + path + the BURN-ruling requirement. Fires
  **unconditionally** — NOT gated on the target encoding having a
  `_CORPUS_SHA_PINS` entry — because held-out artifacts are inherently far
  smaller than any real training corpus, so the size guard makes this
  effectively free on every launch it doesn't apply to. The pin-check path
  reuses the held-out gate's already-streamed sha (never hashes twice —
  same FIX-1 discipline as the WP0.4 pin gate).
- Both "directions" collapse to one runtime mechanism by construction: a
  held-out sha can never simultaneously be a launch pin (forbidden by the
  static invariant), so "held-out loaded where training corpus expected"
  and "training corpus turns out to be the held-out set" are the same
  check.
- `hexo_rl/encoding/__init__.py` exports `assert_not_heldout_sha`,
  `held_out_shas`, `heldout_size_bytes` alongside the existing
  `resolve_corpus_sha_pin`.

## Test evidence

New file `tests/test_s5_heldout_corpus_gate.py` (7 tests, synthetic tiny
corpora + monkeypatched registries — no dependency on the real gitignored
held-out NPZ):

- `test_heldout_sha_on_unpinned_encoding_raises` — direction 1 (held-out sha
  on an UNPINNED encoding still raises — proves unconditional coverage).
- `test_heldout_sha_on_pinned_encoding_raises_heldout_error_not_pin_mismatch`
  — direction 2 (held-out sha where a pinned corpus is expected raises the
  held-out-specific message, firing BEFORE the generic pin-mismatch check).
- `test_pinned_corpus_loads_normally_with_unrelated_heldout_entry_registered`
  — non-overlap happy path (legitimate pinned load unaffected by an
  unrelated held-out registration).
- `test_unpinned_unrelated_corpus_unaffected_by_heldout_registry` — proves
  the size-guard short-circuits BEFORE any sha256 stream for a non-matching
  file (0 `compute_npz_sha256` calls).
- `test_registry_overlap_raises` / `test_registry_no_overlap_happy_path` —
  the resolver-level static invariant, both directions.
- `test_real_registries_have_no_overlap` — regression guard over the REAL
  production registries (not monkeypatched).

Run:

```
.venv/bin/python -m pytest tests/test_s5_heldout_corpus_gate.py -q
# 7 passed
```

Full regression sweep of adjacent corpus/batch-assembly/resolver test files
(all pre-existing, none touched except `batch_assembly.py` /
`resolvers.py` / `encoding/__init__.py`):

```
.venv/bin/python -m pytest \
  tests/test_corpus_sha_pin_gate.py tests/test_s5_heldout_corpus_gate.py \
  tests/test_encoding_resolver_paths.py tests/test_run3_corpus_launch_path.py \
  tests/test_augment_plumbing.py tests/test_b3_augment_opp_slot_required.py \
  tests/test_batch_assembly_bot_slot.py tests/test_batch_aug_uniform.py \
  tests/test_corpus_chain_target.py tests/test_generate_bot_corpus_encoding.py \
  tests/test_inv_refresh_hook.py tests/test_no_stale_plane_refs.py \
  tests/test_pipe1_corpus_position_index.py tests/test_recent_buffer_augment.py \
  tests/test_training_registry_plumbing.py -q
# 97 passed, 1 skipped (test_run3_corpus_launch_path.py — real 2.65GB
# corpus not present on this worktree host; pre-existing skip, unrelated
# to this change)
```

## File paths (exact)

Code (touched — all within scope; did NOT touch
`hexo_rl/selfplay/pool.py`, `engine/src/game_runner/**`, or orchestrator
worker/restore code, which a concurrent implementer owns and had already
modified in this worktree before S5 started):

- `/home/timmy/Work/Hexo/hexo_rl/.claude/worktrees/gnn-integration/hexo_rl/encoding/resolvers.py`
  (new: `_HELDOUT_CORPUS_SHAS`, `held_out_shas`, `assert_not_heldout_sha`,
  `heldout_size_bytes`, `_assert_no_registry_overlap` + import-time call)
- `/home/timmy/Work/Hexo/hexo_rl/.claude/worktrees/gnn-integration/hexo_rl/encoding/__init__.py`
  (exports)
- `/home/timmy/Work/Hexo/hexo_rl/.claude/worktrees/gnn-integration/hexo_rl/training/batch_assembly.py`
  (`load_pretrained_buffer` — held-out gate call site)
- `/home/timmy/Work/Hexo/hexo_rl/.claude/worktrees/gnn-integration/tests/test_s5_heldout_corpus_gate.py`
  (new, 7 tests)
- `/home/timmy/Work/Hexo/hexo_rl/.claude/worktrees/gnn-integration/scripts/mint_s5_heldout_corpus.py`
  (new mint script)

Docs (new, committable per dispatcher instruction):

- `/home/timmy/Work/Hexo/hexo_rl/.claude/worktrees/gnn-integration/docs/registers/s5_heldout_manifest.md`

Data artifacts (new, gitignored — worktree-local only, per §6 of the
manifest doc; NOT present on the main checkout or any other host):

- `/home/timmy/Work/Hexo/hexo_rl/.claude/worktrees/gnn-integration/data/corpus/heldout_s5/` (29 raw JSON game files)
- `/home/timmy/Work/Hexo/hexo_rl/.claude/worktrees/gnn-integration/data/bootstrap_corpus_v6_live2_ls_heldout_s5.npz`
- `/home/timmy/Work/Hexo/hexo_rl/.claude/worktrees/gnn-integration/data/bootstrap_corpus_v6_live2_ls_heldout_s5.npz.metadata.json`

This report (untracked, per instruction):

- `/home/timmy/Work/Hexo/hexo_rl/.claude/worktrees/gnn-integration/reports/probes/gnn_integration/S5_heldout_mint.md`
- Companion machine-readable summary (untracked):
  `/home/timmy/Work/Hexo/hexo_rl/.claude/worktrees/gnn-integration/reports/probes/gnn_integration/s5_heldout_summary.json`
