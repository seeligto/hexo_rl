# §S181 IMPL-S181T1 — Bootstrap + corpus colony-bias audit

**Wave:** §S181 structural-diagnosis research. **Mode:** INSPECTION-ONLY.
**Branch:** `phase4.5/s181_structural_research`. **Date:** 2026-05-22.

**Goal.** Quantify whether `bootstrap_model_v6.pt` and/or the v6 human
corpus encode colony bias in the value head or policy head *before* any
self-play touches them.

**Verdict (jump ahead):** **BIAS-RULED-OUT** for the value head;
**BIAS-RULED-OUT** for the policy head on the discriminating tests.
The pre-self-play bootstrap is not the colony source. See §6.

---

## 0. Artifact-identity corrections (read first)

The task brief names three files that do **not** exist as named on disk.
The substitutions below are forced and are load-bearing for every number
in this report.

| Brief name | On-disk reality | Action taken |
|---|---|---|
| `data/bootstrap_corpus_v6.npz` | **absent**. The v6-encoding corpus is `data/bootstrap_corpus.npz` — `states (353091, 8, 19, 19) f16`, `policies (…, 362)`, matches the v6 anchor arch (8-plane, 362 actions). | Used `bootstrap_corpus.npz` as the v6 position corpus + `data/corpus/raw_human/*.json` (6857 game files) + `data/corpus/quality_scores.json` for per-game Elo. |
| `archive_quarantine/bootstrap_model_random_init_v6w25.pt` | exists at `checkpoints/archive_quarantine/bootstrap_model_random_init_v6w25.pt` but is **v6w25 architecture** (626 actions, 25×25, `value_pool=min`). Not arch-compatible with the v6 anchor. | Cannot be the bias floor. Used a fresh `HexTacToeNet(encoding="v6")` (seed 20260522) as the same-architecture floor instead. |
| `checkpoints/pretrain_*.pt` epoch checkpoints | **absent**. No pretrain epoch checkpoints retained anywhere under `checkpoints/` or `reports/sprint_archive/`. | Pretrain trajectory (§4) is **NOT reconstructable**. |

**Anchor confirmed.** `checkpoints/bootstrap_model_v6.pt` — wrapped
training checkpoint, `model_state` key, `metadata.encoding_name = "v6"`,
SHA prefix `7ab77d2c`, verify sidecar verdict `TRAINED`. v6 encoding =
8-plane history, 19×19, 362-action policy (`361 cells + 1 pass`).
`metadata.corpus_sha256 = None` — corpus provenance is **not pinned** in
the anchor (minor hygiene gap, not a bias finding).

**Architecture note (kills structural hypothesis #2 up front).** The §S181
handoff hypothesis "value head min-pool asymmetric ⇒ architectural colony
bias" does **not apply to the v6 anchor**. v6 registry spec has
`value_pool = "none"`; v6 routes through `min_max_window_head`
(`hexo_rl/model/network_min_max_head.py`) whose value branch is a
**symmetric global avg+max pool** → `value_fc1` → ReLU → `value_fc2` →
tanh. `MinMaxPool` (the min/max K-aggregation) is only reached on
multi-window encodings (`v6w25`, `v7mw`), which the v6 anchor is not. No
min-pool runs in the v6 value path. Architectural-asymmetry hypothesis is
**N/A for this anchor** — investigate it only if a v6w25 anchor is revived.

---

## 1. Corpus colony/extension statistics per Elo band

Source: `data/corpus/raw_human/*.json` (6857 games, 0 skipped), Elo =
mean of the two players' `players[].elo`. Banding matches
`data/corpus/manifest.json`. "Winning line" = the full stone set of the
winning player at game end (the line that produced the 6-in-row).

Operational definitions (chosen + justified):
- **extension** (winning line) := max straight co-linear chain ≥ 5, no
  gaps. Justified: every human win ends on a 6-in-row, so a winning line
  *must* contain a long run; the ≥5 cut captures the near-complete
  winning extension and is the natural "is this an extension shape" test.
- **compact / colony-like** (winning line) := mean pairwise hex distance
  ≤ 2.6 over the winning stone set. Justified: a true colony keeps all
  stones mutually close; 2.6 ≈ the mean pairwise distance of a tight
  ~7-stone hex blob.
- Position-level (§1b): **extension** := any player's stone set has a
  ≥4-chain; **colony** := mean pairwise dist ≤ 2.2 AND max chain ≤ 3.

| Elo band | n games | median glen | mean max-chain (win line) | mean pairwise dist | mean win-stones | P1 win rate (CI95) | frac win-line **extension** (≥5) | frac win-line **compact** (≤2.6) |
|---|---|---|---|---|---|---|---|---|
| sub_1000   | 1829 | 41 | 5.42 | 4.29 | 24.2 | 0.550 [0.527, 0.573] | **0.900** | 0.159 |
| 1000_1200  | 3872 | 49 | 5.49 | 5.20 | 28.5 | 0.549 [0.533, 0.564] | **0.912** | 0.150 |
| 1200_1400  | 1125 | 59 | 5.62 | 6.86 | 38.1 | 0.517 [0.488, 0.546] | **0.934** | 0.109 |
| 1400_plus  |   31 | 89 | 6.06 | 8.20 | 53.9 | 0.516 [0.348, 0.680] | **1.000** | 0.032 |
| **all**    | 6857 | 49 | 5.50 | 5.25 | 29.1 | 0.544 [0.532, 0.555] | **0.913** | 0.145 |

**Findings.**
1. **The corpus is overwhelmingly extension-shaped, monotonically with
   Elo.** Winning lines are ≥5-chain extensions in 90.0% of sub-1000
   games rising to 100% of 1400+ games. Only 14.5% of winning lines are
   compact (≤2.6 mean pairwise dist), *falling* with Elo (15.9% → 3.2%).
   Human players at every Elo band win by extending, not by colonizing.
2. **Higher Elo = longer, more spread-out games.** Median game length
   41 → 89, mean win-stones 24 → 54, mean pairwise dist 4.3 → 8.2 as Elo
   climbs. No band plays colony-compact winning lines.
3. **Elo bands contributing most signal under current weighting.** The
   corpus weight is the quality score `0.4·elo + 0.3·len + 0.3·entropy`
   (`hexo_rl/bootstrap/corpus_metrics.py:502`). Position weights in
   `bootstrap_corpus.npz`: min 0.00158, max 0.0714, mean 0.0158. Band
   `1000_1200` dominates raw count (3872/6857 = 56%) and, because Elo and
   length both feed the score and that band has the most games, supplies
   the bulk of the weighted signal. The 1200_1400 band is up-weighted
   per-game (higher Elo, longer games → higher score) but is only 16% of
   games. Net: **no band feeds colony-style data** — every band's winning
   lines are extension-dominated, so weighting cannot manufacture a
   colony bias the data does not contain.
4. P1 advantage present (54.4% overall), consistent with Q8 (57.1%
   first-player-advantage open question) — not a colony finding.

### 1b. Position-level colony/extension fraction (`bootstrap_corpus.npz`)

40,000 positions sampled (seed 20260522), 71,703 player-stone-sets with
≥4 stones classified:

| metric | value |
|---|---|
| frac **extension** (≥4-chain) | **0.500** |
| frac **colony** (compact, ≤2.2 dist & ≤3-chain) | **0.046** |
| frac neither | 0.454 |
| mean position weight — extension | 0.01418 |
| mean position weight — colony | 0.01919 |
| max-chain histogram | {1: 1975, 2: 12088, 3: 21806, 4: 31920, 5: 3914} |

Mid-game positions split ~50% extension-bearing / ~5% colony-compact /
~45% transitional. The colony positions carry a *slightly higher* mean
weight (0.0192 vs 0.0142) — but they are only 4.6% of stone-sets, so the
weighted contribution of colony-shaped positions is tiny
(0.046 · 0.0192 / (0.046·0.0192 + 0.50·0.0142) ≈ **11%** of the
extension-vs-colony weighted mass). The corpus does not over-represent
colony shapes.

---

## 2. Bootstrap value-head bias probe + random baseline

**Position bank.** 50 colony + 50 extension positions, v6 8-plane
tensors, built by `scripts/structural_diagnosis/probe_value_bias.py`:
- **colony**: BFS-grown compact hex blob of S stones (S ∈ {4..9},
  weighted mid), opponent given a fixed 3-stone scattered filler.
- **extension**: straight co-linear chain of S stones (same S range,
  same opponent filler).
- Matched on stone count S; both colors (current / opponent) covered;
  small positional jitter. History planes replicate the t0 occupancy
  across all 4 history slots (quiescent-position model).

Value-head output (tanh, current-player perspective), N=50 each:

| model | colony value (CI95) | extension value (CI95) | Δ colony−ext | Welch p |
|---|---|---|---|---|
| **anchor** v6 | −0.0879 [−0.288, +0.112] | +0.0623 [−0.185, +0.310] | **−0.150** | **0.355** |
| fresh-init v6 floor | −0.826 [−0.856, −0.796] | −0.873 [−0.883, −0.863] | +0.047 | 0.003 |

**Findings.**
1. **No colony value bias.** The anchor rates colony positions if
   anything *lower* than extension positions (Δ = −0.150), and the
   difference is **not statistically significant** (Welch p = 0.355,
   CIs overlap by a wide margin). The colony-bias hypothesis predicts the
   opposite sign with significance — it is not observed.
2. **Fresh-init floor.** The random net rates everything near −0.85
   (tanh of small-logit noise) with a tiny +0.047 colony−ext gap that is
   "significant" only because the random net's outputs have near-zero
   variance (sd 0.04–0.11). This is a degenerate floor — it shows the
   bias *measurement* has resolution ~0.05, and the anchor's −0.150 is
   inside measurement noise of zero, not a real effect.
3. **Discriminating near-win sub-probe.** A sharper hand-built test
   (4 positions, current-player), reproduced by
   `build_subprobe_positions()` in `probe_value_bias.py` and recorded in
   the JSON sidecar under `subprobe_near_win.anchor`. Construction: the
   open-N positions are straight co-linear chains built by
   `build_extension_position(N, color="cur", jitter=0)` — the open-5 is
   the line (0,0)..(4,0) with both axis endpoints (−1,0)/(5,0) empty, so
   either completes a 6-in-row; the compact-N positions are BFS-grown hex
   blobs from `build_colony_position(N, color="cur", jitter=0)`. The
   open-5 argmax cells uniquely fix this geometry.

   | position | anchor value | anchor argmax cell (top-1) |
   |---|---|---|
   | open 5-in-row (one move from win) | **+0.978** | (−1,0) — co-linear win completion; top-2 = (−1,0),(5,0), the two cells that complete the 6-run |
   | open 4-in-row | +0.704 | (0,1) — blob-adjacent off-axis |
   | compact 5-blob | +0.583 | (−2,0) — blob-edge |
   | compact 4-blob | +0.378 | (−2,0) — blob-edge |

   The value head rates the **near-win extension highest of all**
   (+0.978) — strictly above the equivalent-size compact blob (+0.583)
   and above the open-4 (+0.704). Full ordering open-5 > open-4 >
   5-blob > 4-blob: at matched stone count the open extension outscores
   the compact blob in both pairs (5: +0.978 vs +0.583; 4: +0.704 vs
   +0.378). A colony-biased value head would invert this. It does not.

---

## 3. Bootstrap policy-head bias probe results

Same 50+50 bank, anchor argmax + top-5 recorded.

| metric | anchor | fresh-init floor |
|---|---|---|
| argmax = pass slot | 0/50 colony, 0/50 ext | 0/50, 0/50 |
| argmax adjacent to own stones — colony | 50/50 [0.929, 1.0] | 5/50 [0.043, 0.214] |
| argmax adjacent to own stones — extension | 50/50 [0.929, 1.0] | 7/50 [0.070, 0.262] |
| **extension endpoint frac** (argmax = co-linear endpoint extension) | **0.900 [0.786, 0.957]** — 45 endpoint / 5 off-axis / 0 other | n/a (random) |
| **colony breakout frac** (argmax ≥2 hex from blob) | **0.000 [0.0, 0.071]** — 0 breakout / 50 stay-compact | n/a (random) |

**Findings.**
1. **Policy plays the threat on extension lines.** On extension
   positions the anchor argmax is the **co-linear endpoint extension**
   90% of the time (45/50) — i.e. it grows the run toward the 6-in-row.
   The near-win sub-probe confirms: given an open 5-in-row the top-2
   argmax cells are exactly the two cells that complete the win. This is
   correct threat-seeking play, not colony bias.
2. **Policy compacts on colony positions — but this is not
   discriminating.** On a compact blob the anchor argmax is blob-adjacent
   100% of the time and never breaks out (0/50). However, a colony with
   no embedded near-win has **no "correct extension" move** — local
   thickening is a defensible move there too. The probe cannot separate
   "colony-biased" from "correct local play" on these synthetic colony
   positions, because the synthetic colonies contain no threat to
   compare against. What the probe *can* say: the anchor never
   *spontaneously* abandons a cluster to start a far-away line. That is a
   compaction *preference*, but the §1 corpus shows human winning play is
   itself "extend the cluster you have" — the preference matches the
   training distribution and is not pathological pre-self-play.
3. The fresh-init floor argmax is ~10% adjacent (chance for a 19×19
   board with ~6 stones), confirming the anchor's 100%-adjacency is
   learned structure, not an artifact.

---

## 4. Pretrain learning trajectory

**NOT RECONSTRUCTABLE.** No pretrain epoch checkpoints
(`checkpoints/pretrain_*.pt` or under `reports/sprint_archive/`) were
retained. `hexo_rl/bootstrap/pretrain_trainer.py:292` writes only a
single final weights-only file; no per-epoch snapshots exist. The
question "when did colony bias appear during pretrain" cannot be
answered from local artifacts.

Mitigating fact: §2/§3 show there is **no colony bias in the final
bootstrap to trace the onset of** — so the missing trajectory does not
leave an open colony question. If a future wave wants the trajectory,
pretrain must be re-run with `--save-every-epoch` (does not currently
exist as a flag; ~10 LOC to add to `pretrain_cli.py`).

---

## 5. Encoding channel asymmetry (v6)

v6 registry spec (`engine/src/encoding/registry.toml`): 8 planes =
`[cur_t0, cur_t-1, cur_t-2, cur_t-3, opp_t0, opp_t-1, opp_t-2, opp_t-3]`.
Every plane is a **binary stone-occupancy map** — one bit per cell per
player per history step. There is **no density channel, no cluster-size
channel, no chain-length channel** in the v6 input.

**Consequence — symmetry, not asymmetry.** Colony and extension shapes
are represented by *identical machinery*: both are just sets of 1-bits in
the occupancy planes. The encoding does not privilege compact clusters
over thin lines, nor vice versa — a 6-stone blob and a 6-stone line are
the same number of set bits in the same planes. Any colony/extension
preference must be **learned by the conv trunk**, not handed to it by the
encoding. There is no channel asymmetry to quantify because there are no
shape-specialized channels at all.

(Contrast: the retired v6w25 path has `value_pool=min` /
`policy_pool=scatter_max` K-aggregation, which *could* introduce a
shape-dependent asymmetry across cluster windows — but v6w25 is not the
anchor and was retired §174. Not in scope for the v6 anchor.)

The aux chain-length head exists (`_compute_chain_planes`, 6 planes) but
is a *training target*, not an input channel, and is symmetric over
chain length by construction.

---

## 6. VERDICT

### Value head: **BIAS-RULED-OUT**
The v6 anchor does not rate colony positions above extension positions.
Δ(colony−extension) = −0.150 (wrong sign for the hypothesis), Welch
p = 0.355 (not significant). The near-win sub-probe shows the value head
rates an open 5-in-row extension (+0.978) strictly **above** an
equivalent-size compact 5-blob (+0.583), and the open-4 extension
(+0.704) above the 4-blob (+0.378) — the open extension outscores the
compact blob at both matched stone counts. The value head is
extension-favouring, consistent with a corpus whose winning lines are
91% extensions.

### Policy head: **BIAS-RULED-OUT** (on the discriminating test)
On extension positions the anchor argmax is the co-linear win-completing
endpoint 90% of the time; given an explicit open 5-in-row it plays the
exact 6th-move win. The colony-compaction behaviour on threat-free
synthetic blobs is non-discriminating (no correct alternative exists in
those positions) and matches the human corpus's own "extend your
cluster" winning style.

### Corpus: **BIAS-RULED-OUT**
Human winning lines are ≥5-chain extensions in 91.3% of games, rising
monotonically with Elo to 100% at 1400+. Compact winning lines are 14.5%
and *falling* with Elo. Position-level: only 4.6% of stone-sets are
colony-compact; their slight weight up-lift nets to ~11% of the
extension-vs-colony weighted mass. No Elo band feeds colony-style data.

### Architecture: hypothesis **N/A for this anchor**
v6 `value_pool="none"` — symmetric avg+max pool, no min-pool in the v6
value path. The "min-pool asymmetric ⇒ architectural colony bias"
hypothesis applies only to multi-window v6w25/v7mw, which the active
anchor is not.

### Overall: **BIAS-RULED-OUT — the pre-self-play bootstrap is not the
colony source.**

This **falsifies §S181 structural hypothesis #1** ("bootstrap + corpus
jointly encode colony bias in the value head"). The colony attractor
that captured §175 / §S179 / §S180a / §S180b is **not present at
self-play step 0**. It is generated *by the self-play training loop
itself* — consistent with L38 (config-invisible capture channel) and the
operator's read that the issue is structural-dynamic, not static.

**Falsified Hypotheses Register candidate row:**

| § | Hypothesis | Falsified by | Mechanism |
|---|---|---|---|
| §S181-T1 | `bootstrap_model_v6.pt` + v6 human corpus jointly encode a colony bias in the value head (and/or policy head) before self-play | §S181-T1 bootstrap+corpus bias audit (`audit/structural/01_bootstrap_corpus_bias.md`) | Value-head Δ(colony−ext)=−0.150, Welch p=0.355 (wrong sign, n.s.); near-win sub-probe rates open-5 extension +0.978 > 5-blob +0.583 (and open-4 +0.704 > 4-blob +0.378); policy plays the 6th-move win 90%; corpus winning lines 91.3% extension rising to 100% at 1400+ Elo. No colony bias exists pre-self-play. Attractor is generated by the training loop. |

---

## 7. Surgical follow-up experiments

§6 verdict is BIAS-RULED-OUT, so no causality-confirmation experiments
for *bootstrap* bias are needed. The productive follow-ups **redirect the
search to the training loop**, where the attractor must originate:

1. **Checkpoint-trajectory bias probe (highest value, cheap).** Re-run
   this exact probe (`probe_value_bias.py`, point `load_anchor` at a
   checkpoint path) on the §S180b archive ladder
   (`archive/s180b_3knob_fail/ckpt_step{10,20,30,40,50}k.pt`). If
   Δ(colony−ext) flips from −0.15 (step 0) to a large positive value by
   step 40–50K, that pins the **step at which the value head acquires
   the colony bias** and proves the loop — not the bootstrap — installs
   it. ~30 min, zero training. **Do this first.**

2. **Self-play target audit.** The colony must enter via the training
   targets the loop writes. Audit the CQV / visit-count policy targets
   and the bootstrapped value targets on colony-vs-extension positions
   sampled from a captured run's replay buffer — does the *target*
   (not the net output) over-value colony? This localizes the channel
   to either MCTS search dynamics (target inflation) or value
   bootstrapping (TD-style self-reinforcement).

3. **MCTS-in-loop probe (handoff hypothesis #3 / #5).** This audit's
   probes are static logit reads — same class as the threat probes that
   PASS through every collapse (L22, 4×). The colony-breakout result
   (0/50) is suggestive but non-discriminating statically. An
   MCTS-matched probe — give the net+search a colony position with an
   embedded reachable extension, measure visit fraction on the
   breakout move vs the compaction move — would discriminate. This is
   the natural T-track successor and aligns with the §7 dashboard /
   probe-redesign gaps in the handoff.

4. (Lower priority) Add `--save-every-epoch` to `pretrain_cli.py` and
   re-run pretrain to fill the §4 trajectory gap — only worthwhile if
   #1–#3 somehow reopen a bootstrap-side question.

---

## Files produced by this track

- `audit/structural/01_bootstrap_corpus_bias.md` (this file)
- `scripts/structural_diagnosis/probe_value_bias.py` (standalone probe;
  imports only `HexTacToeNet` — rationale in its docstring; all probe
  logic is self-contained; no selfplay/training/MCTS code touched)
- `scripts/structural_diagnosis/probe_value_bias_results.json` (JSON
  sidecar with full per-position results, regenerated by the script)
