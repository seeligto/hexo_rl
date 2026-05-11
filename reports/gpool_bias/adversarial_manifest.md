# §170 P4 P2 — adversarial corpus manifest

**Date:** 2026-05-09
**Sprint:** §170 P4 P2 — corpus prep for §171 A4 distribution-shift
fine-tune.
**Branch:** `encoding/gpool_bias_a1`.
**Producer:** `scripts/generate_adversarial_corpus.py` (committed in
this sprint).
**Predecessor:** §170 P0 verdict SPATIAL_RICH
(`reports/investigations/a4_scalar_ablation_20260508/VERDICT.md`).

---

## TL;DR

`data/adversarial_corpus_v8.npz` exists on **5080 vast.ai
(`/workspace/hexo_rl/data/`)** and **laptop
(`/home/timmy/Work/hexo_rl/data/`)** at sha256
`e6c1b9b921492d9b23f825cce26e99b818285743fffef8aec3ae47532ef84c2c`.
12,781 positions across 5 sources, 198 MB uncompressed NPZ, schema
byte-compatible with `data/bootstrap_corpus_v8_canvas_realness.npz`
(canvas_realness=True variant). NO RETRAIN performed in this sprint —
corpus prep only.

§171 A4 fine-tune entry-point:
`data/bootstrap_corpus_v8_canvas_realness.npz` (base v8 canvas_realness
corpus, sha256 `110ea6b2…`) + this adversarial corpus → A4 fine-tune
resume from `checkpoints/ablation_169/A4_canvas_realness.pt`.

> **Correction (§171 P0.1, 2026-05-09):** original draft of this manifest
> cited `data/bootstrap_corpus_v8.npz` with sha `110ea6b2…` as the §171
> A4 entry-point. That was a path/sha transposition: the file at
> `data/bootstrap_corpus_v8.npz` is the **vanilla v8 baseline corpus**
> (sha `adb88412…`, plane-8 polarity off→outside, used by §167 B1 / §169
> A2 / A3 retrain). The §171 A4 fine-tune trains from
> `A4_canvas_realness.pt`, which was pretrained on the canvas_realness
> variant `data/bootstrap_corpus_v8_canvas_realness.npz` (sha
> `110ea6b2…`, plane-8 polarity inside→canvas, per
> `reports/ablation_169/A4_corpus_export.log`). The corrected entry-
> point uses the canvas_realness file to preserve plane-8 polarity match
> with the resume checkpoint.

---

## Output

| field | value |
|---|---|
| path | `data/adversarial_corpus_v8.npz` |
| sha256 | `e6c1b9b921492d9b23f825cce26e99b818285743fffef8aec3ae47532ef84c2c` |
| size | 198.2 MB (uncompressed; mmap-ready) |
| total positions | **12,781** |
| target positions | 15,000 (12,781 / 15,000 = 85% of target — KrakenBot pair-bug + scripted-game brevity left ~2.2 k short; in 10–20 k operator band) |
| seed (master) | 20260509 |
| host | 5080 vast.ai (`ssh6.vast.ai:13053`, `/workspace/hexo_rl`) |
| wall time | ~14 min (459 s sealbot_vs_a1 + 39 s far_line + 25 s far_placement + 214 s krakenbot + 147 s sealbot self-play) |

### NPZ schema (column-by-column parity with bootstrap_corpus_v8_canvas_realness.npz)

| key | shape | dtype | notes |
|---|---|---|---|
| `states` | (12781, 11, 25, 25) | float16 | v8 + canvas_realness=True (plane 8 inside-mean = 217.0 = R=8 hex cell count, identical to base corpus) |
| `policies` | (12781, 625) | float32 | one-hot on played move (sum-per-row = 1.0) |
| `outcomes` | (12781,) | float32 | ±1 from current player's POV |
| `weights` | (12781,) | float32 | uniform 1.0 (per-position fine-tune weights) |
| `source_labels` | (12781,) | object (will be `S40` on future runs) | per-position source tag (diagnostic; pretrain loader ignores) |

The `bbox_clip_fired` telemetry total was 26,674 — matches the same
~2-stones-clipped-per-encode rate as B1 / A4 corpus regen, expected
under R=8 stones spilling outside the 25×25 envelope on long games.

---

## Per-source breakdown

| source | weight | games attempted | games kept | win-P1 / win-P-1 / dropped | mean ply (kept) | median ply | positions kept | opponent strength band |
|---|---:|---:|---:|---|---:|---:|---:|---|
| **sealbot_vs_a1** | 0.45 | 351 | 347 | 177 / 170 / 4 | 59.4 | 37 | **6,750** | SealBot minimax (t=0.1 s) vs A1 v6w25 argmax |
| **scripted_far_line** | 0.13 | 102 | 90 | 90 / 0 / 12 | 27.4 | 29 | **1,860** | FarLineOpponent (§164 P2 catastrophic) vs SealBot (t=0.1 s) |
| **scripted_far_placement** | 0.12 | 94 | 62 | 62 / 0 / 32 | 17.6 | 17 | **958** | FarPlacementOpponent (§164 P2) vs SealBot (t=0.1 s) |
| **krakenbot_vs_sealbot** | 0.15 | 117 | 39 | 11 / 28 / 0 | 32.0 | 31 | **963** | KrakenBot (t=0.1 s) vs SealBot (t=0.1 s) |
| **sealbot_vs_sealbot** | 0.15 | 117 | 117 | 55 / 62 / 0 | 39.0 | 31 | **2,250** | SealBot self-play (t=0.1 s) |

**Total: 781 games attempted, 655 usable, 12,781 positions kept.**

Notes per source:

- **sealbot_vs_a1** is the *primary* signal per §170 P4 P2 prompt — A1 vs SealBot positions are exactly what A4 needs to learn under the SPATIAL_RICH (§170 P0) corpus-overfitted-spatial hypothesis. Side-alternation (seed parity) gives A1 both colors. SealBot at t=0.1 is at the same strength used for §170 eval (1×0.5 s budget, but per-move t=0.1 here for bulk generation; same pattern as §168 SealBot eval).
- **scripted_far_line** and **scripted_far_placement** are §164 P2's catastrophic asymmetric-perception adversaries. They place stones at hex_dist ∈ [6, 8] from any bot stone — exactly the OOD distribution A4 collapses on. Note these are run against **SealBot** here (not A4 / not the v6w25 anchor), to harvest SealBot's response as ground-truth policy.
- **krakenbot_vs_sealbot** introduces a peer-project minimax style different from SealBot's 140×140 flat board. KrakenBot has a **known pair-bug** under tight time budgets where MinimaxBot returns the same cell twice as a 2-move pair; the wrapper's pre-cache validity check (`rust_board.get(q2, r2) == 0` _before_ move 1 plays) doesn't catch the move1==move2 case. The generator catches the resulting `apply_move` exception and skips the game. Effective game-yield: 39/117 ≈ 33 %. Position target shortfall (963 vs 2,250) was absorbed by other sources without exceeding the 20 k operator cap.
- **sealbot_vs_sealbot** is the lowest-weight source per §170 P4 P2 prompt ("less informative since SealBot is the eval anchor — generating SealBot vs SealBot positions overfits to its style"). Retained at 0.15 weight for distribution coverage of typical SealBot opening / middlegame patterns; not the primary signal.

---

## Position filter criteria

Identical to `bootstrap_corpus_v8_canvas_realness.npz`
(`scripts/export_corpus_npz.py --canvas-realness`) for training-loop
schema compatibility:

- Drop draws / no-winner games (`board.winner() is None`).
- Drop games shorter than 15 plies (`min_game_plies=15`).
- Sample plies in range `[2, 150)` per game (`position_start=2, position_end=150` —
  skip P1 forced opener; cap at P95.5 of human-game length).
- Cap per-game position count at 25 (`max_positions_per_game=25`) — uniform
  random subsample within ply range, deterministic per (master_seed XOR
  game_seed).
- Plane-8 polarity is `canvas_realness=True` (1 inside, 0 outside) — required
  by `A4_canvas_realness.pt` PartialConv2d trunk entry; matches base
  `bootstrap_corpus_v8_canvas_realness.npz` (canvas_realness variant, sha
  `110ea6b2…`).
- `legal_move_radius=8` on every Board (matches v6w25 / v8 perception);
  `cluster_threshold=8` and `cluster_window_size=25` set on the Board only
  for the sealbot_vs_a1 source (A1's V6ArgmaxBot reads the v6w25 cluster
  view from `state.to_tensor()`).
- Random-opening plies = 4 (matches §170 / §168 eval convention; injects
  diversity per seed).

---

## §171 entry-point

```bash
# §171 A4 fine-tune corpus inputs (base + adversarial):
data/bootstrap_corpus_v8_canvas_realness.npz    sha256 110ea6b20ad3140d2791a1ca72c5c36076a75913e9fe5f9574fa3a1d45dc8cb3  (347,142 pos)
data/adversarial_corpus_v8.npz                  sha256 e6c1b9b921492d9b23f825cce26e99b818285743fffef8aec3ae47532ef84c2c  ( 12,781 pos)

# §171 A4 fine-tune resume base (frozen architecture from §169 P4):
checkpoints/ablation_169/A4_canvas_realness.pt  21.9 MB, encoding=v8, canvas_realness=True, PartialConv2d trunk entry, no gpool, 3.85 M params

# Mixing ratio (operator-tunable for §171; not committed here):
#   Bootstrap-only batches : adversarial-only batches ≈ 0.97 : 0.03
#   (adversarial is ~3.7 % of base corpus by position count — match the
#    natural ratio so fine-tune converges without destabilising the
#    base distribution).
```

**Recommendation for §171 scoping (deferred to next sprint):** use a
small fine-tune budget (~2 – 5 k steps, peak LR ≤ 5 e-5, frozen
trunk-entry PartialConv2d, unfreeze res_blocks 8–11 + heads only) per
the §170 P0 SPATIAL_RICH verdict. The hypothesis is that A4's spatial
features generalise from bootstrap_corpus to adversarial under modest
distribution shift — full retrain is not warranted. Fine-tune from the
A4 checkpoint, not from scratch.

---

## Reproducibility

```bash
# On vast 5080 (~14 min wall):
.venv/bin/python -u scripts/generate_adversarial_corpus.py \
    --target-positions 15000 \
    --max-positions-per-game 25 \
    --sealbot-time-limit 0.1 \
    --krakenbot-time-limit 0.1 \
    --out data/adversarial_corpus_v8.npz \
    --stats-out reports/gpool_bias/adversarial_stats.json \
    --seed 20260509
```

Default source weights:
- `--weight-sealbot-vs-a1 0.45`
- `--weight-scripted-far-line 0.13`
- `--weight-scripted-far-placement 0.12`
- `--weight-krakenbot-vs-sealbot 0.15`
- `--weight-sealbot-vs-sealbot 0.15`

Pass `--no-a1` to skip the SealBot vs A1 source (smoke / no-GPU runs).
The script auto-detects A1 encoding (v6 / v6w25) from the checkpoint.

**Artefacts (gitignored — not committed; reproducible from the script):**

- `data/adversarial_corpus_v8.npz` — the corpus.
- `reports/gpool_bias/adversarial_stats.json` — JSON sidecar with full
  per-source counts, schema, config, sha256.
- `reports/gpool_bias/adversarial_gen.log` — stdout/stderr from the run
  (35 KB, mostly the SealBot 5+ cluster colony-bug-risk warnings).

---

## Done-when (§170 P4 P2 close-out)

- [x] `data/adversarial_corpus_v8.npz` on 5080 + laptop (rsync down,
      `/workspace/hexo_rl/data/` → `/home/timmy/Work/hexo_rl/data/`).
- [x] sha256 captured (`e6c1b9b9…`); also stamped into the JSON sidecar.
- [x] Per-source breakdown captured here (counts, win splits, mean ply,
      strength band).
- [x] §171 entry-point documented (this manifest §"§171 entry-point").
- [x] 1 commit on `encoding/gpool_bias_a1` (this section + script +
      sprint-log P4 P2 entry).
- [x] NO RETRAIN performed.
