# Open Questions — HeXO Phase 4+

## Resolved

| # | Question | Resolution | Commit |
|---|---|---|---|
| Q5 | Supervised→self-play transition schedule | Exponential decay 0.8→0.1 over 1M steps; growing buffer + mixed data streams | `a6e5a79` |
| Q6 | Sequential vs compound action space | Sequential confirmed — 2 MCTS plies per turn, Q-flip at turn boundaries, Dirichlet skipped at intermediate plies | `5be7df7`, `9b899e9` |

## Active (Phase 4.0)

| # | Question | Experiment Design | Estimated Cost | Priority |
|---|---|---|---|---|
| Q2 | Value aggregation: min vs mean vs attention | Train 4 variants, compare value MSE + win rate | ~4 GPU-days | HIGH |
| Q3 | Optimal K (number of cluster windows) | Ablation K=2,3,4,6 | ~6 GPU-days | MEDIUM |
| Q8 | First-player advantage in value training | Measure P1 win rate by Elo band; adjust value targets if >60% | ~2 GPU-days | MEDIUM |

## Community Watch (pending external validation)

### Q9 — KL-Divergence Weighted Buffer Writes

**Source:** Kubuxu (bot dev Discord, 2026-04-01). Confirmed used in KataGo codebase
(not the paper). Kubuxu's 1.1M param transformer beats his heuristic bot 310–10
on 8 playouts vs 5,000 heuristic playouts.

**Technique:** Weight each sample written to the replay buffer by:

```
weight = 0.5 + KL(π_prior ∥ π_target)
```

where `π_prior` is the NN policy before MCTS and `π_target` is the MCTS visit
distribution. Write the sample `floor(weight)` times, plus probabilistically
`frac(weight)`.

**Expected benefit:** Positions where the network most disagrees with MCTS search
get written more frequently, concentrating learning signal on the highest-uncertainty
positions. KataGo reports this as a major training efficiency improvement.

**Prerequisite:** Requires a stable self-play baseline checkpoint (Phase 4.5 gate).
Cannot ablate without a reference point. Implementation touches `ReplayBuffer::push`
(Rust) and the self-play sample writer.

**Estimated cost:** ~2 GPU-days (A/B comparison against uniform-weight baseline).

---

### Q10 — Torus Board Encoding (Architectural Alternative)

**Source:** imaseal (bot dev Discord, 2026-03-31). Phoenix expressed interest.

**Technique:** Encode board state on a torus (wrap-around / circular padding in CNN).
Enables full rotational symmetry without re-encoding, reduces edge artifacts,
compatible with standard PyTorch `circular` padding modes.

**Tradeoff vs current approach:** Our attention-anchored windowing handles the
infinite board via K cluster snapshots. Torus encoding is a fundamentally different
architectural bet — it assumes the game is "local enough" that wrap-around doesn't
create false connectivity across the board's virtual edges. imaseal has **not yet
confirmed** whether wrap-around causes false-line artifacts (phantom 6-in-a-row
across the torus seam). Phoenix noted this risk explicitly.

**Prerequisite:** imaseal's results on wrap-around artifact frequency. **Do not
implement until community data is available.** Watch item only.

**Estimated cost:** Unknown until artifact risk is quantified. If clean, ~4 GPU-days
for a full architectural comparison against windowed baseline.

---

## Deferred (Phase 5+)

| # | Question | Reason deferred |
|---|---|---|
| Q1 | 2-moves-per-turn MCTS convergence rate | Requires fixed-board ablation harness not yet built |
| Q4 | 12-fold augmentation equivariance on infinite boards | Equivariance test not yet implemented |
| Q7 | Transformer stone-token encoder | CNN baseline must be established first |
