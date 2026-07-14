# run3 corpus manifest — canonical ruling (WP0.4)

Durable pin doc for the run3-CNN pre-launch corpus-manifest unification.
Source design: WP0.4 dispatcher design (2026-07-14, D-L run3 convene chain).
Verified independently on laptop (`/home/timmy/Work/Hexo/hexo_rl`) at
implementation time; box shas below are as reported by the design's
cross-host read (`ssh vast`, `/workspace/hexo_rl`) and are NOT re-verified
by this doc's author — re-run the one-liner in §4 on the box before launch
if box drift is suspected.

Two distinct artifacts — do not conflate them:

- **Human-corpus MANIFEST** — the raw scraped game set + its
  `hexo_rl.probes.gnn_bc.corpus_check` reproducibility sha. Bookkeeping only.
  run3 does **NOT** read this at train time.
- **run3 TRAINING corpus** — the frozen `_ls` NPZ. This is the load-bearing
  artifact `load_pretrained_buffer` actually reads. **This is the pin that
  matters for launch.**

---

## 1. Canonical shas (ratified)

| Artifact | Scope | Count | sha256 | Source |
|---|---|---|---|---|
| Laptop human-corpus manifest | `data/corpus/raw_human/` (laptop) | 8698 games | `a4d27e3f7c33b44c766fb137eb917beed7e81dfbd1bd00bf54c2eee0e3942017` | `reports/probes/gnn_bc/corpus_step0.json` |
| Box human-corpus manifest | `/workspace/hexo_rl/data/corpus/raw_human/` (box) | 6902 games | `2916cca59898e125b75f9b5e437c02d523beb7bf336e7f43afd84582ef3b9f0c` | `reports/probes/gnn_bc/corpus_step0_box_r2.json` |
| **run3 TRAINING corpus (load-bearing)** | `data/bootstrap_corpus_v6_live2_ls.npz` | 610954 positions (8669 games, Jul-4 build) | **`3813edc2fb10a7c5ab976a0293e38cbba0fd6b84e5295630f339ca421b345c97`** | verified via `sha256sum` on laptop 2026-07-14, matches sidecar `sha256` field + design's cross-host read |

Sidecar (`data/bootstrap_corpus_v6_live2_ls.npz.metadata.json`, byte-identical
both hosts): `encoding_name: v6_live2_ls`, `n_positions: 610954`,
`ls_plies_emitted: 472537`, `created_at: 2026-07-04T07:56:30Z`,
`created_by_commit: f409c9e908ff8d62a941dc32689f5a6a2a13f051`,
`source_manifest: "scripts/export_corpus_npz.py --human-only"`.

**RULING: canonical human-corpus manifest = laptop 8698, sha `a4d27e3f…`.**
Box 6902 (`2916cca…`) is a strict subset (laptop-only delta = 1796 games,
box-only = 0; box is a May-23 scrape snapshot, mostly frozen since), retained
ONLY as the frozen R2 BC-scaling TRAIN snapshot — not a rival manifest.

**Canonical run3 TRAINING corpus = `_ls` NPZ sha `3813edc2…`.** This is the
launch-pinned value enforced in code (§3 below).

Filter parity (both hosts, identical): `min_game_length: 15`,
`position_window: [2, 150]`, 4/4 Elo bands non-empty, `elo_weighting_active:
true`, `winner_replay_disagree: 0/500`, `step0_pass: true`. Minor tooling
ambiguity (not a host divergence — same on both hosts): `min_game_length`
reads `15` in `bc_data.py` / `corpus_check`, `22` in `configs/corpus.yaml:5`
(a *different* consumer — the pretrain-mode source-weighting knob, not the
run3 training path), "≥20" in R2c prose. Pin the exact filter here:
**`min_game_length=15`, `position_window=[2,150]`, rated, six-in-a-row.**

---

## 2. Ruling on the 1796 held-out games: **ACCEPT (they enter run3 training)**

The 1796 laptop-only games (mtimes 2026-05-24 → 2026-07-02+, the post-box-
snapshot append-only daily-scrape tail) are already baked into the frozen
`_ls` NPZ pinned above — the NPZ was built 2026-07-04 09:56 by globbing
`data/corpus/raw_human` at build time (8669 of the then-8698 games; 29 were
scraped after the 09:44 build cutoff). Position-count triangulation
(`ls_plies_emitted=472537`, far above the box-6902-derived 393,957, near the
laptop-8698-derived 500,494) confirms the NPZ was built from the laptop
superset, not the box snapshot.

**Ruling: KEEP the frozen NPZ `3813edc2…` as-is — the 1796 enter run3
training.** Rationale:

1. **Held-out purpose is SPENT.** The 1796 were the D-M R2 BC-scaling
   held-out set; R2 reads are DONE (2026-07-14, verdict banked: BC-saturated,
   held-out top1 FLAT past 40k, banked-40k best-generalizing —
   `reports/probes/gnn_bc/R2_scaling.md` §2/§3, `r2/R2C_REDTEAM.md`).
2. **run3 is production RL, not a held-out-gated architecture probe.** Its
   objective is strength; held-out purity is irrelevant to run3's own read.
   The 1796 extra human games are mildly beneficial training data.
3. **Rebuild-to-exclude is disruptive + breaks comparability.** Excluding
   them mints a new NPZ sha, requires a 2.6 GB laptop→box rsync, re-mints the
   weights-only anchor lineage, and re-runs `corpus_check` + bench-gate on
   both hosts. **run2 already trained on `3813edc2…`** — a different run3
   corpus would break run3-vs-run2 corpus comparability for negligible gain.
4. **Known, accepted caveat at run2 pre-launch** already
   (`docs/designs/d_run2_multiwindow_run_spec.md` §2 Decision 7 snapshot
   caveat); run2 launched 2026-07-05 on this NPZ anyway.

**Consequence: the 1796-game R2c held-out set is BURNED for run3-lineage
architecture reads.** Any future BC/value architecture adjudication needing a
clean held-out MUST mint a **fresh held-out from the post-2026-07-04 scrape**
(the 29 build-excluded games + all subsequent daily-scrape tail), NOT reuse
the 1796.

### 2.1 Flip condition (operator override)

The only scenario warranting KEEP-OUT + rebuild is if the operator wants
run3's lineage to preserve the 1796 as a pristine held-out for a *future*
run3-net architecture comparison. Given R2 is done and run3 is production RL,
this is not indicated by default — but it is available as an explicit
operator override.

**Cost of the override** (if flipped): rebuild `_ls` from the box-6902
snapshot (or laptop-8698 minus the 1796) via
`export_corpus_npz.py --encoding v6_live2_ls` against a pinned raw_human
subset (needs a subset-glob flag — does not exist today, must be added); new
NPZ sha; new sidecar; rsync 2.6 GB laptop→box; re-run `corpus_check` on both
hosts; re-mint/re-pin the anchor lineage; re-run the bench gate. Breaks
run3-vs-run2 corpus comparability. Multi-step; not recommended absent an
explicit operator ask.

---

## 3. Enforcement (code)

- **Single resolver:** `configs/variants/run3_dist65.yaml` `mixing.pretrained_buffer_path`
  is `"<auto>"`, routed through `resolve_corpus_path` (registry-keyed,
  `hexo_rl/encoding/resolvers.py:265`, `_CORPUS_PATHS` line ~129) via
  `expand_auto_paths` (called from `hexo_rl/training/orchestrator.py`, section
  6). For `v6_live2_ls` this resolves to `data/bootstrap_corpus_v6_live2_ls.npz`
  — the exact path this doc pins.
- **Sha-pin registry:** `_CORPUS_SHA_PINS` in `hexo_rl/encoding/resolvers.py`,
  keyed by encoding name, `resolve_corpus_sha_pin(spec)` accessor.
  `v6_live2_ls → 3813edc2fb10a7c5ab976a0293e38cbba0fd6b84e5295630f339ca421b345c97`.
- **Hard gate:** `load_pretrained_buffer`
  (`hexo_rl/training/batch_assembly.py`) recomputes the corpus file's sha256
  (`hexo_rl.bootstrap.corpus_io.compute_npz_sha256`, streaming, not trusting
  the sidecar) and raises `ValueError` on mismatch against the registered
  pin, naming expected vs actual sha + the file path. Also calls
  `hexo_rl.bootstrap.corpus_io.load_corpus(path, expected_encoding=...)` as a
  belt-and-braces check — this catches a stale/desynced sidecar (sidecar
  declares a sha that no longer matches the on-disk file) even when the
  on-disk file happens to still match the launch pin. The pin is what proves
  "this is the RIGHT corpus"; the sidecar only proves internal (npz↔sidecar)
  consistency. This fires on EITHER host, at step 0, for ANY path the config
  resolves to reading — a hardcoded-path config that names the wrong-but-
  same-shaped NPZ is caught by the sha check even though it bypasses the
  `<auto>` resolver route.
- Only `run3_dist65.yaml`'s launch path was collapsed onto `<auto>` in this
  pass. The ~30 other hardcoded-path variants (`e1_scalar.yaml`,
  `run2_mw_fresh.yaml`, etc.) are unchanged — out of scope, no launch-safety
  need (not active launch configs).
- No crosstalk risk from `configs/corpus.yaml:corpus_npz_path` (a different
  v7full NPZ, read only by the standalone pretrain CLI's `--corpus-npz`
  default, `hexo_rl/bootstrap/pretrain_cli.py`) — the run3 training loop
  reads `mixing.pretrained_buffer_path` exclusively, a different key with no
  fallback into `corpus.yaml`.

---

## 4. Cross-host parity — verification one-liner

Run on **both** hosts before launch:

```
sha256sum data/bootstrap_corpus_v6_live2_ls.npz
# must print 3813edc2fb10a7c5ab976a0293e38cbba0fd6b84e5295630f339ca421b345c97
```

Also confirm the sidecar's `sha256` field matches (`jq .sha256
data/bootstrap_corpus_v6_live2_ls.npz.metadata.json`). Both were verified
identical on laptop + box at design time (2026-07-14); the code-level gate in
§3 makes this a hard launch-time assertion, not just a pre-launch manual
check.

**Do not re-export** `data/bootstrap_corpus_v6_live2_ls.npz` — `raw_human`
grows daily (background-scrape); re-running `export_corpus_npz.py --encoding
v6_live2_ls` on either host mints a NEW, non-reproducible NPZ that no longer
matches this pin and will hard-fail the launch gate (by design).

---

## 5. Adjacent — NOT this doc's scope

Anchor checkpoint parity (`checkpoints/run2_bootstrap_v6_live2_ls.pt`,
`checkpoints/bootstrap_model_v6_live2_8300.pt`) must independently be
sha-identical both hosts before launch — carried to the launch dispatcher /
operator checklist, not enforced by this WP (memory:
`vast-stale-checkpoint-name-collision`, `d1m-resume-eval-downpowered`).
