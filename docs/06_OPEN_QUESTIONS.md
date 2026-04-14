# Open Questions — HeXO Phase 4+

## Resolved

| # | Question | Resolution | Commit |
|---|---|---|---|
| Q5 | Supervised→self-play transition schedule | Exponential decay 0.8→0.1 over 1M steps; growing buffer + mixed data streams | `a6e5a79` |
| Q6 | Sequential vs compound action space | Sequential confirmed — 2 MCTS plies per turn, Q-flip at turn boundaries, Dirichlet skipped at intermediate plies | `5be7df7`, `9b899e9` |
| Q12 | Shaped reward S-ordering correctness | Won't implement shaped rewards — formation taxonomy bias outweighs sample efficiency benefit at current compute scale; quiescence override covers forcing without encoding human formations; revisit at Phase 5 if training stagnates tactically | — |
| Q13 | Chain-length planes as input tensor augmentation | 6 chain-length planes added at indices 18..23 (3 hex axes × 2 players, /6-normalised). Includes chain_head auxiliary regression loss (smooth-L1 on `input[:, 18:24]` target). Atomic 18→24 plane break with fresh pretrain v3. See sprint log §92. | `feat/q13-chain-planes` branch |
| Q19 | Threat-head BCE class imbalance | `pos_weight = 59.0` (theoretical `(1−p)/p` at ~1.6% positive fraction) added to threat-head `BCEWithLogitsLoss`. Landed atomically with Q13 as a fresh bootstrap. `scripts/compute_threat_pos_weight.py` recomputes empirically from a replay buffer when available. §91 C4 monitoring hook stays in place. | `feat/q13-chain-planes` branch §92 |

## Active (Phase 4.0)

| # | Question | Experiment Design | Estimated Cost | Priority |
|---|---|---|---|---|
| ~~Q17~~ | ~~Phase 4.0 self-play mode collapse — root cause and remediation~~ | **RESOLVED 2026-04-10** — Dirichlet ported to `engine/src/game_runner.rs` (commit `71d7e6e`). Verified via `debug_prior_trace` (commit `4a3149e`). Awaiting §71 checklist walk before sustained run. See sprint log §73. | done | resolved |
| Q2 | Value aggregation: min vs mean vs attention | Train 4 variants, compare value MSE + win rate | ~4 GPU-days | HIGH — unblocked now Q17 resolved |
| Q3 | Optimal K (number of cluster windows) | Ablation K=2,3,4,6 | ~6 GPU-days | MEDIUM |
| Q8 | First-player advantage in value training | Measure P1 win rate by Elo band; adjust value targets if >60% | ~2 GPU-days | MEDIUM |

**Q17 (2026-04-09, RESOLVED 2026-04-10):** The P3 overnight run
collapsed to deterministic carbon-copy self-play games between
ckpt_13000 / 14000 / 15000 despite healthy dashboard metrics.

**Confirmed findings (diagnostics A/B/C + follow-up 2026-04-10):**

- **Root cause:** `engine/src/game_runner.rs` (live Rust training path)
  has zero calls to `apply_dirichlet_to_root` — unported feature since
  2026-03-30 Phase 3.5 migration.
- **Failure mode: stuck fixed point, not progressive collapse.** Entropy
  oscillates in a ~1.49–1.70 nat band across ckpt_13000–17428 with no
  downward trend. The system locked in early and maintained the fixed
  point; subsequent training did not deepen the collapse.
- **Temperature sampling is working.** A separate check (2026-04-10,
  ckpt_15000 vs itself, 20 games, τ=1.0) produced 13 distinct game
  lengths — sampling diversifies play when temperature > 0. The
  collapse is purely the missing Dirichlet path, not a second bug.
- **Eval identical games = argmax by design.** 100% identical round-robin
  games are expected: `ModelPlayer` uses `temperature=0.0` (argmax).
  Not a seeding or sampling bug.
- **`best_model.pt` is NOT an independent reference.** Weight fingerprint
  `ed07ecbe6a73` matches `bootstrap_model.pt` exactly — was initialised
  from bootstrap weights at training start and never promoted. There is
  no pre-collapse independent checkpoint in the P3 dataset.
- **Restart point:** do not use entropy rank to choose. No checkpoint in
  13k–17k is less collapsed than any other. Restart from
  `bootstrap_model.pt` once the Dirichlet port is complete, or from the
  earliest checkpoint before self-play dominated the buffer (~step 10k).

Full details in sprint log §70 and `archive/diagnosis_2026-04-10/`. No
fixes proposed in this pass — findings only.

**2026-04-10 update (§71):** Gumbel fallback verified — static audit and
runtime trace confirm `gumbel_mcts: true` provides functionally active root
noise on the training path (visit concentration 0.24 vs 0.65 for PUCT;
workers diverge in candidate selection per §71 verdict). Policy-entropy split
monitoring landed: `policy_entropy_pretrain` and `policy_entropy_selfplay`
now emitted separately on every `train_step` event; selfplay collapse
threshold at 1.5 nats visible in both dashboards. Pre-run checklist
documented in §71. **Dirichlet port is the only remaining blocker before
restart.** A new sustained run should start from `bootstrap_model.pt` after
the port is unit-tested and the §71 pre-run checklist is walked.

**2026-04-10 RESOLUTION:** Dirichlet root noise ported to `engine/src/game_runner.rs` (commit `71d7e6e`). Runtime-verified via `debug_prior_trace` smoke from `ckpt_15000`: `apply_dirichlet_to_root` records now appear (10/10 with unique noise), top-1 visit fraction drops 0.65 → 0.47. See sprint log §73 and `archive/dirichlet_port_2026-04-10/verdict.md`. Remaining blocker: §71 pre-run checklist walk (archive buffer, move collapsed ckpts, run 2hr smoke from `bootstrap_model.pt`).

**Q2 blocking on Q17:** Q2 requires a stable baseline to ablate value
aggregation strategies against, but every post-bootstrap checkpoint has
the same collapse signature. Q17 is now resolved — Q2 unblocked once
the first sustained run from `bootstrap_model.pt` confirms stable entropy.

**Q2 interaction note (2026-04-04):** Ownership and threat auxiliary heads added in §37
interact with value aggregation strategy — both heads provide spatial value grounding that
may shift the relative advantage of min vs mean aggregation. Run Q2 ablation before and
after head stabilisation (~10k RL steps) to avoid confounding.

## Community Watch (pending external validation)

### Q9 — KL-Divergence Weighted Buffer Writes

**Source:** Kubuxu (bot dev Discord, 2026-04-01). Confirmed used in KataGo codebase
(not the paper). Kubuxu's 1.1M param transformer beats his heuristic bot 310-10
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


### Q13 — Chain Length Planes as Input Tensor Augmentation

**RESOLVED 2026-04-14** — sprint log §92. See the "Resolved" table at the
top of this file. Kept inline here for historical context on the pre-
landing design discussion.

**Priority (historical):** MEDIUM (Phase 4.5 architectural decision)
**Source:** KrakenBot chain head analysis, threat theory framework, 2026-04-06,
plus `reports/literature_review_26_04_24/review.md`.

**Question:** Should we add 6 chain-length planes (per-cell, per-direction unblocked
run length, 3 hex axes × 2 players) to the input tensor, changing from 18 to 24
planes? These planes are the spatial substrate of W-values across the board and give
the network geometric awareness as an inductive bias rather than learning it from
scratch.

**Resolution:** yes. Implemented as a fresh-start bundle with Q19 and a
new auxiliary `chain_head` regression loss (smooth-L1 on a slice-from-
input target). The literature review's 1.65× KataGo expectation was
explicitly downgraded — our aux target is an input slice, not forward
information — so realistic uplift is 1.1–1.3× on tactical probe
convergence, not on raw loss magnitude. The wider-window aux variant
that matches KataGo's structure is parked as Q21.

**Implementation actual:** Python `GameState.to_tensor()` calls a
module-private numpy-vectorised helper (slicing + zero-pad shifts, NOT
`np.roll`). Rust self-play path has a parity helper in
`engine/src/board/state.rs` (`encode_chain_planes`) so the feature
tensors from both paths are byte-exact. Replay buffer scatter kernel
remaps plane indices 18..23 through a per-symmetry axis permutation
table; tested byte-exact against fresh ground-truth recomputation.

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

### Q16 — leaf_batch_size round-trip hypothesis [WATCH]

**Priority:** MEDIUM (Phase 4.5 target, blocked on baseline)

**Question:** Does a `game_runner.rs` refactor to coalesce Phase-i candidate
inference across a single batch recover the theoretical `leaf_batch_size` gain?
The §69 sweep showed `leaf_bs=16` consistently hurts throughput (−19–30% games/hr)
and inflates draw rate (+25pp) — the opposite of the theory that larger leaf batches
reduce inference round-trips. The suspected mechanism: current game_runner submits
leaves per-worker, so `leaf_bs=16` just delays submission without reducing total calls
(calls/move actually *increases*). A coalesced batch across workers might change this.

**Prereq:** Phase 4.5 baseline established. Do not attempt without a reference point.

**Negative result reference:** Sprint log §69.

**Estimated cost:** ~2 GPU-days (implementation + A/B comparison).

---

### Q18 — NN forward latency ceiling [WATCH, Phase 4.5]

**Priority:** WATCH (do not touch during Phase 4.0 sustained runs)
**Source:** Sprint log §90 / `/tmp/gpu_util_phase1.md`, 2026-04-13

Live steady-state NN forward latency is 12.5 ms vs 1.6 ms in isolated bench
(7.8×). GPU util is 84% — the GPU is busy but producing less work per unit
time than it does in isolation. The §90 config sweep **falsified the
`inference_batch_size` / `inference_max_wait_ms` lever**: raising batch cap
64 → 128 grew mean batch 60 → 85 (+42%) but crashed `nn_forwards/sec`
88 → 53 (−39%), for a net −14% `nn_pos/sec`. Headline `pos/hr` read flat
only because game length doubled, hiding a **−37% `steps_in_window`**
regression. The live batcher is starved, not the GPU; the remaining
inefficiency is architectural.

**Remaining levers (all architectural):**

- **CUDA stream separation** — training gradient kernels and inference forward
  kernels share the default stream; training-step kernels pollute the
  inference kernel/autocast cache. A dedicated inference stream would let the
  inference server run without cross-contamination.
- **Process split** — run the Python training loop in a second process,
  leaving the inference server + worker pool in the primary. Trades IPC and
  duplicate weight hosting for zero kernel-cache interference.
- **`torch.compile` re-enable** — currently disabled per §25/§30/§32 for
  Python 3.14 CUDA graph TLS conflict. Revisit when PyTorch + Python 3.14
  CUDA graph support stabilizes. Expected to cut per-forward Python dispatch
  overhead substantially.
- **Mixed-precision tuning** — BF16 vs FP16 on Ada Lovelace; FP8 speculation.
- **py-spy flame graph on live training** — blocked on `py-spy` Python 3.14
  support (0.4.1 fails with "Failed to find python version from target
  process"). Re-attempt when upstream lands. Expected to confirm NN forward
  dominates wall-time; if otherwise, reopen the worker-parallelism hypothesis.

**Priority rationale:** WATCH. Don't touch during Phase 4.0 sustained runs.
Revisit only if (a) Q2 (value aggregation) lands and throughput becomes the
gating factor for sustained training length, or (b) py-spy with Python 3.14
support reveals a bottleneck the §90 diagnosis missed.

**Prerequisite:** Stable Phase 4.0 baseline checkpoint (shared with Q2).

**Reference:** Sprint log §90, Phase 1 diagnosis in `/tmp/gpu_util_phase1.md`.

---

### Q19 — Threat-head BCE class imbalance

**RESOLVED 2026-04-14** — sprint log §92. `pos_weight = 59.0` landed
atomically with Q13 on the fresh pretrain-v3 bootstrap. See the
"Resolved" table at the top of this file. Historical ticket preserved
below.

**Priority (historical):** WATCH (Phase 4.0+)
**Source:** Sprint log §85, §91; ckpt_00014344 probe 2026-04-14

Probe at step 14344 (§91) shows threat head logits drifted −5.6 nats from
bootstrap baseline (ext −0.60 → −6.21) while contrast grew 8× (+0.50 → +3.94).
Dashboard aux loss trends upward across the run. Mechanism: `winning_line`
labels are ~1.6% positive (6/361 cells per terminal position, 0 for draws).
`BCEWithLogitsLoss` without `pos_weight` drives all logits strongly negative;
positive-class loss climbs while negative-class loss drops.

**Effect on training (pre-fix):** not directly hurting. Aux weight 0.1 × 2 heads
means trunk gradients are dominated by policy + value. Policy head top-10 IS
improving (65% → 70% vs bootstrap), so trunk is reconciling the signals.

**Fix as landed:** `pos_weight ≈ 59` (theoretical `(1−p)/p`) added to
threat-head BCE. Configured via `configs/training.yaml:threat_pos_weight`
(default 59.0) and cached once per Trainer instance in
`self._threat_pos_weight`. `scripts/compute_threat_pos_weight.py`
recomputes empirically from a replay buffer when one exists. Ownership
head deliberately does NOT receive a pos_weight — stone density is
20–40%, already balanced.

**Prereq for landing (satisfied):** fresh training restart. Bundled with
Q13 in the fresh bootstrap v3 cycle.

**Escalation (post-landing):** §91 C4 `abs(Δ ext_logit_mean) < 5.0` warning
stays active as a drift canary. If it trips on post-fix checkpoints,
re-open and investigate.

**Reference:** Sprint log §85 (aux target alignment), §91 (probe revision +
C4 warning hook), §92 (Q13 + Q13-aux + Q19 atomic landing).

---

### Q21 — Wider-window chain-aux target for forward information injection [PARKED]

**Priority:** LOW (post-baseline research question)
**Source:** Sprint log §92; literature review §"Recommended encoding specification"
**Status:** parked after Q13 landing, to be revisited once the 24-plane
baseline is established.

The current Q13-aux target (`chain_head` smooth-L1 loss) is a slice of
the INPUT tensor — `states[:, 18:24]`. This gives the network
regularization and intermediate supervision on a feature we know matters,
but NOT forward information. The trunk can already see the chain values
directly in its input, so the head's job is near-identity and the initial
pretrain-v3 chain_loss drops to ~0.01 within the first epoch (basically
just a conv-through-residual preservation task).

KataGo's auxiliary targets are FUTURE information (game-end ownership,
score, etc.) that the network cannot trivially reproduce from the current
board state. This is where the 1.65× speedup in Wu 2019 Table 2 comes
from — the auxiliary loss teaches the trunk to build prediction circuits
for counterfactual information.

**Proposed experiment (Q21).** Compute chain targets on a WIDER window
than the NN input window. Concrete example: NN sees a 19×19 cluster window;
chain target is computed on the 25×25 region centred on the same point,
clipped back to 19×19 at the head output. Now the target values near the
edges reflect stones that the network CANNOT see — it has to learn to
extrapolate chain structure from partial information. This matches
KataGo's structure and would hopefully deliver the "genuine" speedup
the Q13-aux slice-from-input variant does not.

**Complications:** the wider chain target is no longer derivable from
the input at training time. Two options:
- **(a) Store in replay buffer** as a separate spatial target (6 × 361 u8
  per row, adds ~22% to state size). Requires HEXB v4 with an additional
  column, migration path, and aux reprojection in self-play game-end.
- **(b) Compute on-the-fly at push time** by passing the wider chain
  values from the self-play worker (Python has access to the full board;
  Rust has the 2-plane cluster view only). Requires a new Rust path that
  takes a wider window for the aux target while still feeding the NN a
  19×19 view.

Option (a) is cleaner and matches the existing ownership/winning_line
pattern. Option (b) avoids buffer-size growth but adds a Rust-side
geometric computation.

**Prereq:** Q13-aux baseline established (one sustained self-play run
post-Q13). Measure realistic Q13 uplift before trying the harder variant.

**Cost:** ~3–4 GPU-days (implementation + A/B comparison).

---

## Deferred (Phase 5+)

| # | Question | Reason deferred |
|---|---|---|
| Q1 | 2-moves-per-turn MCTS convergence rate | Requires fixed-board ablation harness not yet built |
| Q4 | 12-fold augmentation equivariance on infinite boards | Equivariance test not yet implemented |
| Q7 | Transformer stone-token encoder | CNN baseline must be established first |
