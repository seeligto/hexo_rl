# S5 held-out corpus manifest — fresh held-out mint (post-R2c BURN)

Durable pin doc for the GNN-integration program's S5 work package: mint a
fresh held-out game set after `docs/registers/run3_corpus_manifest.md` §2
BURNED the prior held-out set (the 1796-game "R2c" delta) for run3-lineage
architecture reads.

**PURPOSE: future BC / value / architecture reads ONLY. This set must NEVER
enter any training corpus.** Any future entry that wants to move a held-out
game into a training corpus requires an operator-ratified BURN ruling,
exactly like the R2c precedent (`docs/registers/run3_corpus_manifest.md` §2).

---

## 1. Why a fresh mint (context)

`docs/registers/run3_corpus_manifest.md` §2 ruled to KEEP the 1796
laptop-only games (mtimes 2026-05-24 → 2026-07-02+) inside the frozen
`data/bootstrap_corpus_v6_live2_ls.npz` (sha `3813edc2…`) — they were the
D-M R2 BC-scaling held-out set, that read is done, and run3 is production RL
where held-out purity doesn't matter. Consequence (§2, verbatim): "the
1796-game R2c held-out set is BURNED for run3-lineage architecture reads.
Any future BC/value architecture adjudication needing a clean held-out MUST
mint a fresh held-out from the post-2026-07-04 scrape (the 29
build-excluded games + all subsequent daily-scrape tail), NOT reuse the
1796." This doc is that mint.

---

## 2. Membership — how "not in the canonical NPZ" was computed

The canonical training corpus is `data/bootstrap_corpus_v6_live2_ls.npz`
(sha `3813edc2fb10a7c5ab976a0293e38cbba0fd6b84e5295630f339ca421b345c97`),
built 2026-07-04 by globbing `data/corpus/raw_human` at build time (8669 of
the then-8698 games went in; 29 did not — `run3_corpus_manifest.md` §2).

`raw_human` is **append-only** (the daily scrape only adds new UUID-named
`.json` files; existing files are never edited — verified: box 6902 ⊆
laptop 8698 byte-identical, `run3_corpus_manifest.md` §1). File mtime is
therefore a sound, monotonic proxy for "existed at build time."

Independent re-derivation on the laptop `data/corpus/raw_human` (8698 files,
2026-07-15):

| cutoff (local) | files with mtime <= cutoff | files with mtime > cutoff |
|---|---:|---:|
| 2026-07-04 09:44:00 | 8669 | 29 |
| 2026-07-04 09:56:30 (NPZ completion mtime) | 8669 | 29 |

**Zero files fall in the `(09:44, 09:56]` gap** — the two candidate cutoffs
agree exactly, so membership is unambiguous today: every `raw_human` file
with mtime **strictly after 2026-07-04T09:44:00 local** is a held-out
candidate (the canonical NPZ build could not possibly have read it). This
exactly reproduces the manifest doc's documented split (8669 in / 29 out),
which is the correctness check for the method.

No newer scrape has landed since (`data/corpus/manifest.json` last_updated
`2026-07-07T18:52:35Z`, `human_games: 8698` — unchanged through 2026-07-15),
so the held-out candidate set today is exactly those same **29 games** (the
build-excluded stragglers; "all subsequent daily-scrape tail" is currently
empty).

---

## 3. Filter provenance (the ingestion / corpus-pipeline filter)

Two filters apply, matching the canonical corpus's own conventions exactly
(`run3_corpus_manifest.md` §1 "Filter parity"):

1. **Ingestion filter** (`HumanGameSource._passes_filter`,
   `hexo_rl/corpus/sources/human_game_source.py:97`) — re-validated
   independently here, not just trusted from scrape time:
   `rated == true`, `moveCount >= 20`, `gameResult.reason ==
   "six-in-a-row"`. Matches `data/corpus/manifest.json`'s `"filter"` field
   verbatim (`{"rated": true, "min_moves": 20, "reason": "six-in-a-row"}`).
2. **Corpus-pipeline discipline** (`hexo_rl/probes/gnn_bc/bc_data.py`,
   mirrored by `scripts/export_corpus_npz.py`): `MIN_GAME_LENGTH=15`,
   position window `[2, 150)` (skips the forced P1 opener, caps at the
   P95.5 long-game tail).

All 29 candidates passed BOTH filters with zero drops (`n_ingestion_filter_dropped: 0`,
`n_min_game_length_dropped: 0` — `moveCount` range 20–605, well above both floors).

---

## 4. Final held-out set

| field | value |
|---|---|
| **Game count** | **29** |
| Date range (`startedAt_day`, UTC, day-granularity — scrubbed field) | **2026-07-04 → 2026-07-07** |
| Scrape (mtime) range, local | 2026-07-04T17:34:16 → 2026-07-07T20:35:46 |
| Elo range (rated players present) | 877 – 1616 (n=58 player-Elo values) |
| Elo band game counts | sub_1000: 6, 1000_1200: 14, 1200_1400: 7, 1400_plus: 2 |
| Cutoff used | `2026-07-04T09:44:00` local (matches `run3_corpus_manifest.md` §2 exactly) |

**Caveat (mirrors the R2c precedent's own caveat,
`reports/probes/gnn_bc/r2/R2C_REDTEAM.md` §1):** small n (29 games / 2,963
`v6_live2_ls` rows) — this is genuinely all that has been scraped since the
canonical corpus's build cutoff; the daily scrape has not produced new games
since 2026-07-07. Same source population as training (same site, same
ingestion filter) — mild distribution shift only (later calendar window),
not a different population.

### Artifacts (BOTH minted — flexible for either a CNN-arm or GNN-arm read,
mirrors how R2c fed the SAME raw JSON games to both arms):

1. **Raw JSON directory** — `data/corpus/heldout_s5/` (29 files, byte-copies
   of the source `raw_human/*.json`, gitignored — same convention as every
   other corpus data artifact in this repo). Point `HumanGameSource` or
   `hexo_rl.probes.gnn_bc.bc_data.iter_corpus_positions` at this directory
   for a future BC/architecture read of EITHER arm.
2. **`v6_live2_ls`-schema NPZ** — `data/bootstrap_corpus_v6_live2_ls_heldout_s5.npz`
   (gitignored data artifact; sidecar `data/bootstrap_corpus_v6_live2_ls_heldout_s5.npz.metadata.json`
   committed-convention but also not tracked, same as every other corpus
   NPZ). Built via the SAME `replay_game_to_triples_ls` builder the
   canonical training corpus used — schema-identical (`states` / `policies`
   / `outcomes` / `weights`), so it drop-in-works with any tool that reads
   the canonical NPZ's schema, EXCEPT it must never be pointed at as a
   training corpus (see §5).

| artifact | sha256 | size | rows/games |
|---|---|---:|---:|
| `data/bootstrap_corpus_v6_live2_ls_heldout_s5.npz` | `88f99c2b5fea7495484e4e9cc1af831d1e053221dc7e0f9c8f5d3ab6f27aa69e` | 12,872,280 bytes | 2,963 rows (2,090 plies emitted, 10 plies dropped outside all cluster windows) |
| games-manifest (sorted per-game hash list, mirrors `corpus_check.py`'s `corpus_sha256`) | `61f4f227e1d79af55f9435bd5b82cc044d71c9c4a04d1766a7bae4b6d80858e2` | — | 29 games |

Mint date: **2026-07-15** (`created_at` in the NPZ sidecar:
`2026-07-15T13:51:53Z`; git commit at mint time `f4fc523ee0aed4db6dc0b13aa4bf688699ce1138`).

Mint tool: `scripts/mint_s5_heldout_corpus.py` — re-run with:

```
.venv/bin/python scripts/mint_s5_heldout_corpus.py \
    --raw-dir <path-to-raw_human> \
    --out-dir data/corpus/heldout_s5 \
    --out-npz data/bootstrap_corpus_v6_live2_ls_heldout_s5.npz
```

The script re-validates the ingestion filter per file (does not just trust
scrape-time filtering), copies the raw JSON files, and prints the full
summary (also dumped to
`reports/probes/gnn_integration/s5_heldout_summary.json`, untracked).

**Regeneration note:** if the daily scrape resumes and new games land after
2026-07-07, re-running the mint script with the SAME `--cutoff` will pick up
a LARGER held-out set (the append-only tail grows) and mint a NEW sha — that
is expected and fine (this is not a launch-critical byte-identical-across-
hosts pin like the training corpus; it only needs internal provenance). If
that happens, update the registered sha in
`hexo_rl/encoding/resolvers.py:_HELDOUT_CORPUS_SHAS` (see §5) and this doc's
table.

---

## 5. Loader assertion (enforcement)

Extends the WP0.4 single-resolver (`docs/registers/run3_corpus_manifest.md`
§3) with a second, symmetric registry:

- **Held-out registry:** `_HELDOUT_CORPUS_SHAS` in
  `hexo_rl/encoding/resolvers.py` — `label -> (sha256, size_bytes)`. Today:
  `"s5_post20260704" -> (88f99c2b…, 12872280)`.
- **Resolver-level static invariant** (`_assert_no_registry_overlap`,
  called at import time): `_CORPUS_SHA_PINS` and `_HELDOUT_CORPUS_SHAS` must
  never share a sha256. A bad registry entry fails loudly at the FIRST
  import of `hexo_rl.encoding.resolvers`, not at some later training
  launch.
- **Load-level hard gate** (`hexo_rl.training.batch_assembly.load_pretrained_buffer`):
  before the existing WP0.4 pin check, the loader checks whether the target
  file's on-disk **size** matches any registered held-out artifact's size
  (a cheap `stat()`, no stream). Only on a size match does it stream a
  sha256 and call `assert_not_heldout_sha` — which raises `ValueError`
  naming the held-out label, the path, and the R2c-style BURN-ruling
  requirement. This fires **unconditionally** (not gated on the target
  encoding having a `_CORPUS_SHA_PINS` entry) — held-out corpora are always
  much smaller than any real training corpus (MB vs. hundreds-of-MB/GB), so
  the size pre-check makes this effectively free on every launch it doesn't
  apply to, while still catching a held-out artifact pointed at ANY
  encoding's `mixing.pretrained_buffer_path`.
- Both directions collapse to the same mechanism by construction: a
  held-out sha can never simultaneously be a registered launch pin (the
  static invariant forbids it), so "held-out loaded where a training corpus
  is expected" and "a would-be training corpus turns out to be the held-out
  set" are the same runtime check.

Tests: `tests/test_s5_heldout_corpus_gate.py` — held-out sha on an unpinned
encoding raises; held-out sha on a pinned encoding raises the held-out-
specific message (not a generic pin-mismatch); a legitimate pinned load is
unaffected by an unrelated held-out entry; an unrelated corpus's load never
even streams a sha for the held-out check (size-guard proof); the resolver
overlap invariant raises on a contrived collision and passes on the real,
non-overlapping production registries.

---

## 6. Adjacent — NOT this doc's scope

- The raw JSON directory (`data/corpus/heldout_s5/`) and the NPZ
  (`data/bootstrap_corpus_v6_live2_ls_heldout_s5.npz`) currently exist ONLY
  on this worktree checkout (`.claude/worktrees/gnn-integration`) — `data/`
  is gitignored repo-wide and each checkout/host holds its own copy by
  convention (mirrors how the canonical training corpus itself isn't
  tracked in git). A future BC/architecture read running on a different
  host (main checkout, vast box) needs its own copy — re-run the mint
  script there, or copy the artifacts across; either way, verify the sha256
  matches this doc's table before trusting the copy.
- This doc does not itself run any BC/architecture probe against the
  held-out set — it only mints and gates the artifact. A future probe
  consuming it should cite this doc + verify the sha before use.
