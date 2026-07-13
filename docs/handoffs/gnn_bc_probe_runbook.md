# GNN-BC Probe — operator runbook (D-L WP3)

Runs the axis-graph BC-prefit discriminator: train THEIR representation on OUR data,
against a matched CNN control, then a raw-policy round-robin to read the isolating
comparison. Design + frozen verdicts: `docs/designs/gnn_bc_probe_design.md`.

**Venue:** a 5080 box (training is the multi-hour part). CPU verification (STEP-0,
cross-check, param/forward smoke) is already done + committed — see §0.
**Everything here is operator-run; nothing below has been executed.**

---

## 0. Environment + what's already verified (no action)

- **venv:** the main `.venv` (torch + engine). **`torch_geometric` is NOT required** — the
  ported GINE net is plain-torch (F2 in `reports/tourney/adapters_strix.md`). No PyG install.
- Already RUN + committed (see `reports/probes/gnn_bc/`):
  - **STEP-0 corpus check:** PASS (8698 games, 500,494 positions, 4 Elo bands populated,
    0 hash collisions, 500/500 winner agreement) — `corpus_step0.json`.
  - **Graph-builder cross-check:** EXACT MATCH 10/10 (edge set + 5-dim edge attrs + dummy) —
    `cross_check_edges.md`.
  - **Param + forward smoke:** GNN 283,970 params, CNN 571,501 params, both finite forwards,
    GNN grad flows through the trained path. Covered by `tests/test_gnn_bc_probe.py` (6 pass).
- **Re-run the CPU verifications** (optional sanity, ~2 min):
  ```bash
  .venv/bin/python -m hexo_rl.probes.gnn_bc.corpus_check          # STEP-0 (gate)
  .venv/bin/python -m hexo_rl.probes.gnn_bc.cross_check           # builder cross-check
  .venv/bin/python -m pytest tests/test_gnn_bc_probe.py -q        # param/forward/grad/bot
  ```
  STEP-0 must exit 0 before any training.

---

## 1. Corpus + assets

- Human corpus: `data/corpus/raw_human/` (already present, 8698 games).
- External anchors for the eval field:
  - `strix_checkpoint_00237000.pt` (repo root) — strix-raw.
  - `checkpoints/run2_final/checkpoint_00261500.pt` — mantis-261k-raw.
  - SealBot (vendored) — sealbot-d5 (via `scripts/arena/bots/sealbot_child.py`).
- Book: `scripts/arena/book_r5_5ply_32.json` (`book_r5_5ply_32_seed42`), already committed.

---

## 2. Train both arms, two LRs each (the four BC runs)

Each arm at LR-lo=1e-3 and LR-hi=3e-3 (§3.3 fairness sweep). 40,000 steps, batch 256.
On a 5080 use `--device cuda`. These are the multi-hour runs; run them however you prefer
(tmux, backgrounded — see the vast process-hygiene memory before killing anything).

```bash
cd /workspace/hexo_rl   # or the box's checkout

# GNN-BC arm
.venv/bin/python -m hexo_rl.probes.gnn_bc.train_bc --arm gnn --lr 1e-3 \
    --steps 40000 --batch-size 256 --device cuda \
    --out checkpoints/probes/gnn_bc/gnn_lr1e-3
.venv/bin/python -m hexo_rl.probes.gnn_bc.train_bc --arm gnn --lr 3e-3 \
    --steps 40000 --batch-size 256 --device cuda \
    --out checkpoints/probes/gnn_bc/gnn_lr3e-3

# CNN-BC control arm (SAME steps/batch/corpus/protocol)
.venv/bin/python -m hexo_rl.probes.gnn_bc.train_bc --arm cnn --lr 1e-3 \
    --steps 40000 --batch-size 256 --device cuda \
    --out checkpoints/probes/gnn_bc/cnn_lr1e-3
.venv/bin/python -m hexo_rl.probes.gnn_bc.train_bc --arm cnn --lr 3e-3 \
    --steps 40000 --batch-size 256 --device cuda \
    --out checkpoints/probes/gnn_bc/cnn_lr3e-3
```

Each run writes `<out>/<arm>_bc_040000.pt` + `<out>/train_log.jsonl` (step / loss / top1 / lr).

**Notes / tradeoffs**
- The GNN arm's graph-building + segmented per-graph CE is Python-side per position; it is
  slower per step than the CNN's dense batch. If throughput binds, raise `--batch-size` (the
  disjoint-union batch is memory-light) before touching steps — keep steps EQUAL across arms
  (the matched-protocol constraint).
- FP16 AMP is on for CUDA in both arms. The value head is present but never supervised.
- **Do NOT change steps between arms.** The isolating comparison requires matched steps.

### 2b. Pick each arm's best LR

Best LR = higher held-out corpus policy top-1 (the tail of each `train_log.jsonl`), confirmed
by the eval (§4). Record which LR won for each arm in the verdict report. Symlink/copy the two
winners to stable paths:

```bash
# example — replace with whichever LR won per arm
cp checkpoints/probes/gnn_bc/gnn_lr1e-3/gnn_bc_040000.pt checkpoints/probes/gnn_bc/gnn_best.pt
cp checkpoints/probes/gnn_bc/cnn_lr3e-3/cnn_bc_040000.pt checkpoints/probes/gnn_bc/cnn_best.pt
```

---

## 3. Raw-policy round-robin (640 games)

Field: `{gnn-bc, cnn-bc, mantis-261k-raw, strix-raw, sealbot-d5}`. All net bots raw-policy
(0 sims, temp 0). Book = the fair origin-start 5-ply. Headless stdio children (no
ref-server/bridge).

```bash
.venv/bin/python scripts/arena/run_gnn_bc_tourney.py \
    --gnn-ckpt checkpoints/probes/gnn_bc/gnn_best.pt \
    --cnn-ckpt checkpoints/probes/gnn_bc/cnn_best.pt \
    [--resume]
```

Writes `reports/probes/gnn_bc/games_raw_gnn_bc.json` (640 games, checkpointed every 20,
`--resume` re-entrant). On the 5080, net bots can be flipped to `--device cuda` inside
`scripts/arena/run_gnn_bc_tourney.py::build_field` if CPU is the bottleneck (mantis/strix/
sealbot children already read the box device).

---

## 4. Bradley-Terry analysis (isolating comparison)

Reuse the D-K argmax BT + pair-cell bootstrap (the same instrument that produced
`reports/tourney/argmax/ARGMAX_FINAL.md`):

```bash
# adapt build_analysis_argmax_final.py to read games_raw_gnn_bc.json (5-bot field);
# it already does BT MLE (win=1, draw=0.5) + bootstrap resampled at (pairing, opening_idx)
# cell level, N=1000, and emits the ratings table + pairwise matrix + distinct-N per pairing.
.venv/bin/python reports/tourney/argmax/build_analysis_argmax_final.py \
    --games reports/probes/gnn_bc/games_raw_gnn_bc.json \
    --out   reports/probes/gnn_bc/GNN_BC_VERDICT.md
```
(If that script's `--games/--out` flags differ, copy it to
`reports/probes/gnn_bc/build_analysis.py` and point `MONGO_EXPORT`/`OUT` at the WP3 paths —
the BT + bootstrap functions transfer unchanged.)

Read out **Δ = BT(gnn-bc) − BT(cnn-bc)** (Elo scale) + its pair-cell bootstrap 95% CI, plus
`gnn-bc` vs `mantis-261k-raw` and both vs `strix-raw`.

---

## 5. Verdict (frozen — do not re-interpret)

Apply verbatim from `docs/designs/gnn_bc_probe_design.md §6`:

- **ARCH-DOMINANT:** Δ ≥ +100 Elo (CI excludes 0) AND gnn-bc ≥ mantis-261k-raw
  → representation effect real → run3 primary variable RE-OPENS.
- **ARCH-NULL:** Δ ≈ 0 (CI spans 0) → recipe/data/scale, not architecture
  → run3 launches as specced (CNN + dist head); GNN stays re-opened-unproven.
- **MIXED:** Δ > cnn-bc but < +100, OR gnn-bc beats cnn-bc while both ≪ strix-raw
  → report gap; run3 as specced; GNN = run4 pre-registered card #1.

Threshold rationale (+100 < half the observed ~229 D-K raw gap) is in the design doc §6.
Write the LR-per-arm, Δ + CI, the field ratings, and the verdict cell into
`reports/probes/gnn_bc/GNN_BC_VERDICT.md`.

---

## 6. Gotchas

- **STEP-0 is a gate.** If it fails (positions=0 / <2 Elo bands / winner disagreement), STOP —
  the corpus is wrong, not the probe.
- **Matched steps across arms is load-bearing.** Never let the CNN train fewer/more steps than
  the GNN "to save time" — that silently confounds the very variable the probe isolates.
- **Two LRs per arm is the red-team guard** against "you under-tuned the CNN." Do not drop it to
  one LR.
- **Param asymmetry is documented, not a bug** (GNN 284k vs CNN 571k, the CNN floor is its
  fixed 261k policy FC — design doc §5). The CNN has MORE capacity, so a GNN win despite that
  strengthens ARCH-DOMINANT. Optional tighter parity: `--hidden 160` on the GNN net (edit
  `gnn_bc_net.py` default or add a flag) → ~430k.
- **vast process hygiene:** `tmux kill-session` orphans the python (SIGHUP won't reach it);
  kill only comm=python by PID; verify 0 procs before relaunch (see the memory note).
