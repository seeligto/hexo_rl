# GNN-BC Probe — axis-graph BC-prefit discriminator (D-L WP3)

Status: **DESIGN + BUILD, pre-training.** Frozen verdicts below are written BEFORE any
training run and may not be edited after the first BC checkpoint is produced without a
dated amendment note (§9). Training is 5080/operator-gated — this doc + the shipped code
are build-and-CPU-verify only.

Port attribution: axis-graph builder + GINE net ported from **SootyOwl/hexo-strix**
(`hexo-rs` + `hexo-a0`) @ SHA `c381ffbeb248313a1ec177eb650d9c3c2380caa8` (MIT). The port
already lives, fidelity-gated, in `hexo_rl/bots/strix_v1_graph.py` +
`hexo_rl/bots/strix_v1_net.py` (Δ=0 vs strix's own PyG `HeXONet` on 20 fixtures — see
`reports/tourney/adapters_strix.md`). This probe REUSES that builder unchanged and reuses
the net's `RepresentationNetwork`/`_GINEConv`/head modules (see §3.1).

---

## 1. Question

The D-K argmax tournament (`reports/tourney/argmax/ARGMAX_FINAL.md`, the strix-FIXED panel)
ranks **strix-raw #1 at +121 Elo (69.1%)** and **mantis-261k-raw last-but-one at −108 Elo
(32.8%)** — a **~229 BT-Elo raw-policy gap**. strix and mantis differ on BOTH axes at once:

- **Architecture:** strix = axis-graph GINE GNN (stones-as-nodes, 3 win-axis edge families,
  5-dim edge features, ~0.28M params). mantis = our CNN (K-cluster windowed ResNet, ~4.25M).
- **Recipe / data / scale:** strix = train==deploy Gumbel-128 self-play, 5-stage
  win-length+radius curriculum. mantis = PUCT-train/Gumbel-deploy self-play, no curriculum,
  248k→261k steps, off a human-corpus BC bootstrap.

The raw gap could be EITHER. §D-STRIX (sprint log ~2840) banked the **axis-graph
representation as NOTE-ONLY / restart-gated, NEVER falsified** — only the strix CUDA KERNEL
was rejected (ragged-batching perf we don't need, falsified register row §D-STRIX S3). WP3 is
the discriminator that decides whether to reopen the representation as a run3/run4 variable.

**The isolating move: train THEIR representation on OUR data, same protocol as a CNN control.**
Behavior-clone both a GNN (axis-graph) and a CNN (v6_live2_ls) on the identical human corpus,
identical BC protocol, identical steps, then play both raw-policy in the same field. The
comparison that isolates architecture from recipe is **gnn-bc vs cnn-bc, matched everything.**

### Why the control arm is LOAD-BEARING

Without cnn-bc, "gnn-bc beats mantis-261k-raw" is uninterpretable: mantis-raw is a
SELF-PLAY net, gnn-bc is a BC net. BC-vs-selfplay is a training-regime confound that swamps
architecture. cnn-bc removes it: gnn-bc and cnn-bc share corpus, protocol, steps, BC-vs-search
regime, and the only free variable is the representation (axis-graph GNN vs windowed CNN).
mantis-261k-raw and strix-raw stay in the field only as **external anchors** — to place both
BC nets on the D-K argmax Elo scale — not as the isolating comparison.

---

## 2. Both arms — architecture

Both arms train a POLICY on the human corpus. Value head is OPTIONAL and NOT trained/scored —
this is a **policy probe** (matching the raw-policy argmax deploy regime that produced the
~229 gap). Both arms replay the SAME games through `hexo_rl.env.GameState` so the per-position
target set is byte-identical; only the per-position representation differs.

### Shared per-position target (from `GameState` replay)

For each game `(moves, winner)` from `HumanGameSource`, replay ply-by-ply via
`GameState.from_board(Board()).apply_move(...)`. At each ply within the §114 window
(`POSITION_START=2 .. POSITION_END=150`, `MIN_GAME_LENGTH=15`):

- **policy target** = the move actually played `(q,r)` (one-hot / cross-entropy target).
- **value target** (unused by the probe) = `+1 if current_player == winner else −1`.
- **Elo-band sampling weight** = §114 band weights (`sub_1000=0.5 / 1000_1200=1.0 /
  1200_1400=1.5 / 1400_plus=2.0`) — applied identically to both arms.

This is the exact ground truth `hexo_rl/bootstrap/dataset.py::replay_game_to_triples` uses for
the CNN corpus; the probe reuses the `GameState`/`Board` replay so no winner-mapping or
position-window drift can enter.

### Arm A — GNN-BC (axis-graph GINE)

- **Representation:** strix's axis-graph, built by
  `hexo_rl/bots/strix_v1_graph.py::build_axis_graph_raw` (REUSED unchanged). Per position:
  stones→nodes, 3 WIN_AXES edge families with a both-direction walk + stop-at-opponent,
  5-dim edge features `[axis one-hot ×3, signed dist, src-player]`, a global dummy node, and
  11-dim node features (relative base 7 + 4 threat dims). `win_length=6, radius=6,
  prune_empty_edges=True, threat_features=True, relative_stones=True` (the strix ckpt config).
- **Net:** `hexo_rl/probes/gnn_bc/gnn_bc_net.py::GnnBcNet` — GINE representation + per-NODE
  policy head over legal-move nodes. REUSES strix's `RepresentationNetwork` / `_GINEConv` /
  `PolicyHead` / `ValueHead` modules from `strix_v1_net.py` (those classes are grad-capable;
  only `HeXONet.forward` carries `@torch.inference_mode()`, which blocks BC — so the probe net
  has its own grad-capable forward, NOT `HeXONet`).
- **Policy target mapping:** the played `(q,r)` is a graph node (a legal-move / empty node).
  Cross-entropy is over the legal-move node logits; target = index of the played move's node
  in `g["legal_coords"]` order. Skip a ply if the played move is not a representable node
  (out of `radius=6` of every stone — extremely rare on real games; counted + logged).
- **Params (VERIFIED):** hidden=128, num_layers=4 → **283,970**. (Config-parametric; the
  operator may set `--hidden 160` for tighter param parity — see §5 param note.)

### Arm B — CNN-BC (v6_live2_ls, the CONTROL)

- **Representation:** OUR v6_live2_ls encoding. Per position, `GameState.to_tensor()` yields
  K cluster-window views; the probe emits the SAME per-cluster-row scatter the corpus uses
  (`dataset.py::replay_game_to_triples_ls`), kept planes `[0,8,16,17]`, dense 362 policy over
  the window-local cell, `k_max=8`.
- **Net:** `hexo_rl/model/network.py::HexTacToeNet(encoding="v6_live2_ls")` — OUR ResNet trunk
  + policy head, **fresh-init** (no warm-start — the probe measures the representation's BC
  ceiling from scratch, matching gnn-bc's from-scratch fit). Policy-only loss; value head
  present but not supervised/scored.
- **Params (VERIFIED):** at `filters=24, res_blocks=3` → **571,501** (the small config used
  for the probe — see §5). Full-size (128/12) 4.25M is NOT used: a size-matched trunk keeps
  the comparison about representation class, not capacity.

### Both arms — BC head geometry

The GNN scores a variable legal-node set (per-node MLP → 1 logit each); the CNN scores a fixed
362-way window-local softmax. Both are trained by cross-entropy on the played move. Argmax
deploy: GNN picks `legal_coords[argmax(node_logits)]` filtered through the board's legal set
(exactly the shipped `strix_v1_bot` path); CNN picks the window-local argmax decoded to a board
cell (exactly the shipped `our_model_bot` / K-cluster decode path).

---

## 3. BC protocol (both arms, matched)

### 3.1 Reuse map (what is ported vs new)

| Component | Source | Status |
|---|---|---|
| axis-graph builder | `hexo_rl/bots/strix_v1_graph.py` | REUSED unchanged (fidelity-gated) |
| GINE conv / representation / heads | `hexo_rl/bots/strix_v1_net.py` modules | REUSED (grad-capable; new forward) |
| CNN trunk + policy head | `hexo_rl/model/network.py::HexTacToeNet` | REUSED unchanged |
| per-position target replay | `GameState`/`Board`, mirrors `dataset.py` | REUSED via `hexo_rl.env` |
| corpus source | `HumanGameSource` → `GameRecord` | REUSED unchanged |
| policy CE / value BCE | `hexo_rl/training/losses.py` | REUSED (CNN); CE reimpl for GNN's ragged legal set |
| eval harness | `scripts/arena/run_argmax_tourney.py` pattern | REUSED (adds 2 stdio children) |
| BT + bootstrap | `reports/tourney/argmax/build_analysis_argmax_final.py` | REUSED |

### 3.2 Optimizer / schedule (matched to the corpus pretrain path)

Mirrors `hexo_rl/bootstrap/pretrain_trainer.py`:

- Optimizer **AdamW**, `weight_decay=1e-4`.
- Scheduler **CosineAnnealingLR**, `T_max = total_steps`, `eta_min=1e-5`.
- Loss = **policy cross-entropy on the played move only** (value head not supervised).
  Label smoothing `ε=0.05` (corpus default), applied identically to both arms.
- Batch size **256** (graphs are variable-N → the GNN arm batches by disjoint-union /
  block-diagonal concatenation of graphs; the CNN arm batches dense tensors normally).
- **Steps: 40,000** per arm (≈ the pretrain `T_max` order; enough to converge a sub-1M net on
  ~470k weighted positions). Both arms run the SAME step count.
- FP16 AMP on CUDA (both arms).

### 3.3 Two LRs per arm — RED-TEAM fairness sweep

A single LR could handicap one architecture (GNNs and CNNs have different loss-landscape
curvature). To keep the verdict from resting on an LR that happens to favor one arm, **each arm
trains at TWO learning rates** and the arm's probe-strength = the BEST of its two LRs (by
held-out corpus policy top-1 accuracy, then confirmed by the eval). LRs:

- **LR-lo = 1e-3**, **LR-hi = 3e-3** (bracketing the pretrain default 2e-3).

Four BC runs total: `gnn-bc@1e-3, gnn-bc@3e-3, cnn-bc@1e-3, cnn-bc@3e-3`. The verdict compares
`best(gnn-bc)` vs `best(cnn-bc)`. This closes the "you just under-tuned the CNN" red-team.

### 3.4 Corpus + STEP-0 verification (§114)

Corpus = `data/corpus/raw_human/` (human games), parsed by `HumanGameSource`. **STEP-0 runs
BEFORE any training** (`hexo_rl/probes/gnn_bc/corpus_check.py`) and must pass, documenting:

1. game count parsed + games surviving the §114 filter (`MIN_GAME_LENGTH=15`, decisive only);
2. total positions in the `[POSITION_START=2, POSITION_END=150]` window;
3. Elo-band histogram (raw + weighted) — confirms band weights are non-degenerate;
4. winner-derivation agreement: `HumanGameSource.winner` vs a fresh
   `GameState`-replay-to-terminal winner on a sample (catches winner-mapping drift, the §114
   bug class);
5. a sha256 over the sorted game-hash list (dedup / reproducibility key), written to a manifest
   sidecar so the two arms provably train on the same games.

Both arms consume the SAME STEP-0 manifest. A STEP-0 failure blocks the training run.

---

## 4. Eval design

- **Regime:** raw-policy **argmax, NO search** (temp 0) — the exact regime that produced the
  ~229 D-K gap. Search would confound representation with search strength.
- **Field (5 bots):** `{gnn-bc, cnn-bc, mantis-261k-raw, strix-raw, sealbot-d5}`.
  - `gnn-bc`, `cnn-bc` = the two BC probe nets (best-LR checkpoint each).
  - `mantis-261k-raw` = our self-play net raw-policy (the D-K −108 anchor).
  - `strix-raw` = the shipped `strix_v1_bot` raw-policy (the D-K +121 anchor).
  - `sealbot-d5` = minimax depth-5 fixed baseline (the neutral cross-family anchor both D-K
    argmax panels use).
- **Book:** the fair origin-start 5-ply book `scripts/arena/book_r5_5ply_32.json`
  (`book_r5_5ply_32_seed42`, 32 openings, radius 5, origin-start) — the same book the D-K
  tournaments froze, so gnn-bc/cnn-bc land on a comparable Elo scale.
- **Games:** C(5,2)=10 pairings × 32 openings × 2 colors = **640 games** (matches the D-K
  argmax panel size).
- **Harness:** `scripts/arena/run_gnn_bc_tourney.py` — a thin copy of
  `run_argmax_tourney.py` with `gnn_bc` + `cnn_bc` stdio children added and the field swapped.
  Headless (no ref-server / bridge); referee = the same in-file legality + 6-in-a-row check.
- **Ranking:** **Bradley-Terry MLE** (win=1, draw=0.5), CI **bootstrap resampled at the
  (pairing, opening_idx) cell level** (per §D-ARGMAX effective-n — game-level bootstrap
  over-narrows deterministic replays), N=1000. Reuses
  `reports/tourney/argmax/build_analysis_argmax_final.py::bt_mle` + the pair-cell bootstrap.
- **Effective-n:** report distinct-trajectory count per pairing; the 5-ply origin book gives
  32 distinct openings × both colors, well above the §D-ARGMAX floor.

---

## 5. Param counts + the honest asymmetry

| Net | Config | Params (VERIFIED) |
|---|---|---|
| gnn-bc (GINE) | hidden=128, layers=4, policy_hidden=128 | **283,970** |
| cnn-bc (v6_live2_ls) | filters=24, res_blocks=3 | **571,501** |
| cnn-bc floor | filters=16, res_blocks=3 | 548,599 |
| (mantis full ref) | filters=128, res_blocks=12 | 4,249,675 |

**Asymmetry, documented not hidden:** the CNN cannot go below ~548k because its policy head
`Linear(2·361, 362)` alone is ~261k FIXED by the v6 362-action space — a per-node GNN head has
no such fixed cost. So gnn-bc (284k) vs cnn-bc (571k) is a ~2× param gap, entirely in the
head/trunk floor, NOT a tuned advantage. This CUTS AGAINST a spurious ARCH-DOMINANT: the CNN
has MORE capacity, so if the GNN still wins by ≥100 Elo the representation effect is real
despite the capacity handicap. Operator may set `gnn-bc --hidden 160` (~430k) to tighten
parity; the verdict thresholds are chosen to be robust to a 2× param difference (a
representation-class question, not an exact-param question). Both nets are sub-1M — neither is
"the big net."

---

## 6. Frozen verdicts (VERBATIM — written before training)

Let `Δ = BT(gnn-bc) − BT(cnn-bc)` (Elo scale), both at their best-LR checkpoint, from the 640-game
eval; `CI(Δ)` = the pair-cell bootstrap 95% CI on that difference.

- **ARCH-DOMINANT:** gnn-bc beats cnn-bc by **≥ +100 BT Elo (CI excludes 0)** AND
  **gnn-bc ≥ mantis-261k-raw** → the axis-graph representation effect is REAL on our data →
  the run3 primary variable RE-OPENS (axis-graph representation becomes a launch variable).

- **ARCH-NULL:** gnn-bc **≈ cnn-bc (CI spans 0)** → the strix edge is recipe / data / scale,
  NOT architecture → run3 launches AS SPECCED (CNN + distributional value head) and GNN stays
  re-opened-unproven (banked, not falsified — it just isn't the lever).

- **MIXED:** gnn-bc **> cnn-bc but < +100** Elo, OR gnn-bc beats cnn-bc while **both ≪
  strix-raw** → report the gap; run3 launches AS SPECCED; GNN becomes the **run4
  pre-registered card #1** (a proven-partial lever queued behind the specced run3).

### Threshold rationale (documented)

**+100 Elo is < half the observed ~229 raw gap** (strix-raw +121 vs mantis-261k-raw −108 in
the D-K argmax FINAL panel). It is deliberately a **conservative architecture-effect floor**:
a full architecture explanation of the raw gap would predict Δ near the whole ~229; requiring
only ≥100 lets ARCH-DOMINANT fire even if architecture is only ~half the story, while the
CI-excludes-0 clause guards against calling noise a win. Symmetrically, ARCH-NULL requires the
CI to actually span 0 (not merely a point estimate < 100) so a real-but-small effect routes to
MIXED, not to a false NULL. The `gnn-bc ≥ mantis-261k-raw` co-clause in ARCH-DOMINANT prevents
declaring an architecture win when BOTH BC nets are simply weak (a BC-vs-selfplay ceiling could
put both below the self-play anchor — that is a recipe story, MIXED, not an arch win).

---

## 7. What this probe does NOT resolve (scope honesty)

- **BC ceiling ≠ self-play ceiling.** Both BC nets may sit below strix-raw (which is a
  self-play net, not BC). A gnn-bc that beats cnn-bc but trails strix-raw still ISOLATES the
  representation (that is the MIXED cell) — it does not claim the axis-graph reaches strix's
  full deployed strength, which would need porting strix's self-play recipe too.
- **Search is out of scope.** Raw-policy only. Whether the axis-graph representation helps
  UNDER search (Gumbel/PUCT) is a separate, later question.
- **Curriculum / scale confound is REMOVED by construction** (both BC nets, same corpus, same
  steps, sub-1M) — that is the whole point; it is not a residual.

---

## 8. Deliverables (this WP)

- This design doc.
- Port code under `hexo_rl/probes/gnn_bc/`: `graph_check.py` (independent builder for the
  cross-check), `gnn_bc_net.py` (grad-capable GINE net), `cnn_bc_net.py` (CNN control factory),
  `corpus_check.py` (STEP-0), `bc_data.py` (shared replay → per-arm examples),
  `train_bc.py` (BC trainer, both arms, 2 LRs), `gnn_bc_bot.py` / `cnn_bc_bot.py` (raw-policy
  BotProtocol wrappers for eval).
- Eval glue: `scripts/arena/bots/gnn_bc_child.py`, `scripts/arena/bots/cnn_bc_child.py`,
  `scripts/arena/run_gnn_bc_tourney.py`.
- Operator runbook: `docs/handoffs/gnn_bc_probe_runbook.md`.
- Cross-check report: `reports/probes/gnn_bc/cross_check.md` (edge-set match on 10 positions,
  STEP-0 output, param counts, forward-pass smoke).

## 9. Amendment log

- (none — pre-training freeze.)
</content>
</invoke>
