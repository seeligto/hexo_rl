# Q27 — ZOI reachability audit of threat-probe fixture positions

> **SUPERSEDED 2026-04-19** by
> `reports/q27_zoi_reachability_realpositions_2026-04-19.md` (Probe
> 1b). The "0/20 outside ZOI on synthetic" result below was correct
> for the synthetic fixture; its generality caveat (§3 "Caveat on
> generality") is the real finding — the fixture could not exercise
> §77's ply > `zoi_lookback` or disjoint-cluster truncation modes. On
> the real mid/late Probe 1b fixture the count is 1/20 outside ZOI,
> and bootstrap C2/C3 shifts from 20%/20% to 60%/65%. See sprint log
> §106. Body below retained as the record of what Probe 1 (synthetic)
> said at the time.

**Date:** 2026-04-19
**Fixture:** `fixtures/threat_probe_positions.npz` (N=20, all synthetic, phase=early, side=P1)
**Scope:** §77 follow-up item #3 — do probe-position extension cells fall within
the live ZOI window, such that the post-search ZOI mask could cap policy top-K?

---

## 1. Where in the probe is top-K computed. Is ZOI applied?

**Answer: No. ZOI is NOT applied anywhere in the probe.**

Top-K is computed in `scripts/probe_threat_logits.py::_probe_one`, line 219–221:

```python
policy_spatial = log_policy[0, :BOARD_SIZE * BOARD_SIZE].float().cpu()   # (361,)
top10_indices  = policy_spatial.topk(min(10, len(policy_spatial))).indices.tolist()
top5_indices   = top10_indices[:5]
```

The full 19×19 = 361-cell log-policy is used verbatim — no ZOI mask, no legal-move
mask, no occupied-cell mask. `grep -i 'zoi\|mask\|legal\|distance'
scripts/probe_threat_logits.py` returns zero hits.

Contrast with the live self-play path (`engine/src/game_runner/worker_loop.rs:410-427`),
which filters the legal-move set to cells within hex-distance ≤ `zoi_margin` (= 5)
of the last `zoi_lookback` (= 16) moves, when `zoi_enabled` is true and
`move_history.len() >= 3`. This filter is **post-search, move-selection only** —
it does not reduce MCTS tree branching (§77).

**Implication:** the probe's C2/C3 metrics are computed against an unmasked policy
distribution. The ZOI mask, if it had been applied, would only remove candidate
cells from top-K; it cannot push a non-ZOI cell INTO top-K. So ZOI is not
asymmetrically biasing the probe against the extension cell. However, if many
probe extensions are geometrically near the cluster core (within ZOI), and many
distractor cells in the current top-K are ALSO within ZOI, the ZOI mask would
not change top-K ordering — it can only eliminate far-away distractors.

The more interesting question is the converse: **are there probe positions where
the extension cell is OUTSIDE ZOI?** If so, the trained policy can never select
them during self-play (so the training signal never pushes them up in policy
rank), yet the probe still demands that they rise into top-K. That would be a
structural C2/C3 failure mode not fixable by training.

---

## 2. Per-position ZOI reachability

Config (`configs/selfplay.yaml:64-66`): `zoi_enabled: true, zoi_lookback: 16,
zoi_margin: 5`.

All 20 fixture positions are **synthetic, early-phase** (ply = 7: one P1 opener
+ two P2 + two P1 + two P2 = 7 stones placed). Since ply = 7 ≤
`zoi_lookback` = 16, the full board stone set is exactly the ZOI anchor set —
no stones fall outside the 16-move lookback window. Therefore `min_distance =
min over all stones on board`, computed in window-local axial coordinates
(translation-invariant).

Stone counts per position: 4–5 stones visible inside the 19×19 K=0 cluster
window (3 P1 stones forming the 3-in-a-row + 1–2 P2 stones near the window
edge; the remaining 2–3 P2 "far" stones were placed outside the cluster window
by construction and are not encoded in the NPZ, but on the global board they
remain ZOI anchors — not needed for this analysis).

| #  | phase | side | n_stones_in_window | ext (wq,wr) | min d (all current stones) | in ZOI (d ≤ 5)? |
|---:|------:|-----:|-------------------:|------------:|---------------------------:|:---------------:|
|  0 | early |  P1  | 5 | (9, 13)  | 1 | **yes** |
|  1 | early |  P1  | 5 | (6, 9)   | 2 | **yes** |
|  2 | early |  P1  | 4 | (12, 9)  | 2 | **yes** |
|  3 | early |  P1  | 5 | (10, 13) | 2 | **yes** |
|  4 | early |  P1  | 5 | (2, 13)  | 3 | **yes** |
|  5 | early |  P1  | 5 | (8, 13)  | 2 | **yes** |
|  6 | early |  P1  | 4 | (5, 9)   | 3 | **yes** |
|  7 | early |  P1  | 4 | (5, 9)   | 3 | **yes** |
|  8 | early |  P1  | 5 | (11, 6)  | 1 | **yes** |
|  9 | early |  P1  | 5 | (10, 6)  | 2 | **yes** |
| 10 | early |  P1  | 5 | (9, 5)   | 3 | **yes** |
| 11 | early |  P1  | 5 | (11, 7)  | 1 | **yes** |
| 12 | early |  P1  | 5 | (1, 13)  | 3 | **yes** |
| 13 | early |  P1  | 5 | (8, 11)  | 2 | **yes** |
| 14 | early |  P1  | 5 | (13, 9)  | 1 | **yes** |
| 15 | early |  P1  | 5 | (9, 5)   | 3 | **yes** |
| 16 | early |  P1  | 4 | (9, 6)   | 2 | **yes** |
| 17 | early |  P1  | 5 | (13, 11) | 3 | **yes** |
| 18 | early |  P1  | 5 | (13, 10) | 3 | **yes** |
| 19 | early |  P1  | 5 | (13, 10) | 3 | **yes** |

### Distance distribution — extension cell vs nearest stone

| min distance | count |
|---:|---:|
| 1  | 4  |
| 2  | 7  |
| 3  | 9  |

Max observed `min_distance` = 3. Radius-5 floor gives 2 cells of head-room on
every position. Median extension-to-nearest-stone distance = 2.

### Control cells

All 20 control cells sit at window corner flat index 0 = (wq, wr) = (0, 0),
with distances to nearest stone in `[14, 21]`. All control cells are FAR outside
ZOI by construction (`find_control_cell` demands min-dist ≥ 4 from all stones
and walks the window in row-major order, so (0, 0) is picked whenever empty).

---

## 3. Headline result

**Count of fixture positions where the extension cell is outside ZOI: 0 / 20.**

Every probe extension is at hex-distance 1–3 from the nearest placed stone,
well within the ZOI radius of 5. The ZOI mask cannot explain C2/C3 failures
on this fixture — the extension cells are always reachable by the live policy.

**Caveat on generality:** this fixture is 100% synthetic early-phase positions
with ≤ 5 stones in-window. It does NOT sample mid-game (ply 15–49) or late-game
(ply ≥ 50) positions where ZOI reachability could fail in two ways:

1. **Truncation past the lookback window.** Once ply > `zoi_lookback` = 16,
   older stones stop being ZOI anchors. An extension cell that extends a
   3-in-a-row laid down before the lookback window, with no recent moves in its
   neighborhood, would be outside ZOI.
2. **Disjoint-cluster threats.** A threat in a cluster the opponent has
   ignored for ≥ 16 moves would have no recent ZOI anchor nearby.

If/when the fixture is regenerated from real game records (the
`--run-dir` path in `scripts/generate_threat_probe_fixtures.py`, currently
unexercised — see `phase_buckets: {early: [], mid: [], late: []}` in
`_sample_from_games`), this audit must be re-run and the count may differ.

---

## 4. Conclusion for Q27 (§77 item #3)

ZOI post-search masking is **not** the cap on probe top-K for the current
fixture. The §77 hypothesis — that extension cells sit outside ZOI "by
construction" — is **falsified for this fixture**. C2/C3 failures on
synthetic early-phase positions must be explained by:

- policy trunk not routing attention to the threat-scalar signal (Q27 main),
- value-aggregation hijacking (Open Question 2: min vs mean), or
- `aux_threat_weight` under-tuning.

Before escalating to any of those, regenerate the fixture with mid/late-game
real positions and re-run this audit to confirm the same conclusion holds
for longer plies.

---

*Generator:* ad-hoc analysis; no code changed. NPZ read via
`.venv/bin/python` direct NumPy inspection. Window-local axial distances via
`max(|dq|, |dr|, |dq + dr|)` (matches `hexo_rl/utils/coordinates.py:axial_distance`).
