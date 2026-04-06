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

**Q2 interaction note (2026-04-04):** Ownership and threat auxiliary heads added in §37
interact with value aggregation strategy — both heads provide spatial value grounding that
may shift the relative advantage of min vs mean aggregation. Run Q2 ablation before and
after head stabilisation (~10k RL steps) to avoid confounding.

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

### Q11 — Colony win detection over-inclusion

Colony detection (hexo_rl/eval/colony_detection.py) examines ALL winner's
stones at game end, not just the winning 6-in-a-row line. A player who
places scattered exploratory stones early but wins with a connected group
is incorrectly flagged as a colony win if any orphaned stones exist with
centroid distance ≥ 6.0 from the winning group.

Fix: BFS from the 6 stones comprising the winning line only, not from the
full winner stone set. Requires win-line coordinates to be passed into
colony_detection.py from the game result.

Priority: LOW. Does not affect training correctness. Affects interpretability
of eval metrics only. Defer until after Phase 4.0 exit criteria are met.

---

### Q12 — Shaped Reward S-Ordering Correctness

**Priority:** MEDIUM
**Source:** Threat Theory document cross-reference, 2026-04-06

**Question:** Does the current shaped reward table respect W/S/C Strength ordering?
Specifically, is "Double threat created" (+0.08) correctly categorised — is it detecting
W2×(S0) (blockable) or something higher? Is the reward magnitude ordering consistent
with the principle that lower-S formations should always reward more than higher-S
formations of equal Weight?

**Action:** Inspect `FormationDetector` in `engine/` and map each detected formation to
its W/S/C value using the table in `docs/10_COMMUNITY_BOT_ANALYSIS.md §7.1`. Revise
reward table if S-ordering is violated.

**Cost:** ~2 hours inspection, no GPU cost. Fix before Phase 4.5 training run.

---

### Q13 — Chain Length Planes as Input Tensor Augmentation

**Priority:** MEDIUM (Phase 4.5 architectural decision)
**Source:** KrakenBot chain head analysis, threat theory framework, 2026-04-06

**Question:** Should we add 6 chain-length planes (per-cell, per-direction unblocked
run length, 3 hex axes × 2 players) to the input tensor, changing from 18 to 24
planes? These planes are the spatial substrate of W-values across the board and give
the network geometric awareness as an inductive bias rather than learning it from
scratch.

**Implementation path:** Compute in Python `GameState.to_tensor()` — no Rust changes
required. Pure board dict scan. Fast enough for the hot path.

**Constraints:** Breaking change — requires retrain from scratch, new replay buffer
layout, new augmentation scatter tables. Must be decided before the next sustained
training run.

**Experiment design:** Train 18-plane baseline vs 24-plane variant for 500K steps;
compare tactical accuracy (S0 block rate) and early Elo trajectory.

**Cost:** ~4 GPU-days.

---

### Q14 — KrakenBot MinimaxBot as Eval Ladder Opponent

**Priority:** LOW (Phase 4.5 target, blocked on submodule add)

**Question:** Add KrakenBot `MinimaxBot` as a third eval ladder opponent (alongside
SealBot and RandomBot). Provides tactical diversity — pattern-based evaluation vs
SealBot's tree search. Pure Python import, no build step.

**Prerequisites:** `git submodule add vendor/bots/krakenbot`, write `BotProtocol`
wrapper (~30 lines). See `docs/10_COMMUNITY_BOT_ANALYSIS.md §1.9`.

**Note:** Do NOT use KrakenBot `MCTSBot` as a Bradley-Terry anchor — it is actively
training, making it a moving target. SealBot stays as the primary gate.

---

### Q15 — Corpus Tactical Quality Filtering

**Priority:** LOW (Phase 4.5 target)

**Question:** Should corpus game sampling be weighted by peak tactical complexity
(maximum threat strength reached during the game)? Positionally quiet games with no
S1+ structures developed by either side provide weaker training signal for tactical
pattern recognition.

**Implementation:** During manifest analysis pass, compute `max_threat_strength` per
game using `board/threats.rs`. Add field to manifest. Soft-weight buffer sampling
toward tactically richer games.

**Cost:** Low — manifest field only, no training changes.

---

## Deferred (Phase 5+)

| # | Question | Reason deferred |
|---|---|---|
| Q1 | 2-moves-per-turn MCTS convergence rate | Requires fixed-board ablation harness not yet built |
| Q4 | 12-fold augmentation equivariance on infinite boards | Equivariance test not yet implemented |
| Q7 | Transformer stone-token encoder | CNN baseline must be established first |
