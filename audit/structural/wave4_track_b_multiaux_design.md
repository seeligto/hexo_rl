# §S181-AUDIT Wave 4 Track 4B — multi-aux head suite design

Pre-implementation design doc. Scope confirmation requested from operator
BEFORE 4B code commits land. Track 4A vast run (~10h) provides the verdict
that gates whether 4B implementation proceeds (per dispatcher Path X).

## TL;DR scope reduction

Dispatcher framed Track 4B as **~250 LOC + Rust buffer extension for 3
new heads**. Pre-flight code audit found the architecture is much further
along than dispatcher assumed. Realistic scope is significantly smaller.

| dispatcher item | actual state | realistic effort |
|---|---|---|
| 2-stone opp-reply head | Existing `opp_reply_head` predicts `policies_t` (current MCTS policy, NOT opp reply). Real opp-reply target NOT in buffer. | Either (a) extend buffer to carry next-stone-1 + next-stone-2 targets (~150 LOC Rust + Python + bench gate) OR (b) DROP this head and rely on ply-to-end + sigma2 + ownership/threat for density. |
| ply-to-end head | `game_length` is per-game scalar in buffer (`PushGameConfig::game_length: u16`). Per-position ply index NOT tracked. | Add per-position ply index to PushGameConfig (already trivial — workers iterate positions in order; index is row#). ~40 LOC Rust + ~30 LOC Python head + loss. |
| value-uncertainty (sigma2) | Already implemented (`network.py:565 self.value_var`). `compute_uncertainty_loss` already has FP32-promote + clamp(min=1e-6). Comment in `training.yaml:101` says "Gaussian NLL diverges when σ²→0" — likely OUTDATED (clamp is in place). | ~3 LOC: set `uncertainty_weight: 0.1` in training.yaml. Validate at smoke. |
| ownership re-enable | Already ACTIVE at `ownership_weight: 0.1`. | 0 LOC. (Optionally bump weight.) |
| threat re-enable | Already ACTIVE at `threat_weight: 0.1`. | 0 LOC. (Optionally bump weight.) |

## Recommended Track 4B scope (revised)

### 4B-1 — Re-enable sigma2 (3 LOC)
- training.yaml: `uncertainty_weight: 0.0 → 0.1`
- Validate NLL clamp is sufficient (smoke a few hundred steps; verify
  total_loss stays bounded and σ² doesn't collapse to floor)
- Update comment in training.yaml to reflect clamp fix

### 4B-2 — Bump density of existing heads (config-only)
Per dispatcher's KataGo density target (~0.3-0.5 total aux weight):
- Current: opp_reply 0.15 + chain 1.0 + ownership 0.1 + threat 0.1 = 1.35
  (raw weight sum; chain's /6 normalization makes effective ~0.17 + others
  = ~0.52)
- Track 4B: bump ownership 0.1 → 0.2, threat 0.1 → 0.2, uncertainty
  0.0 → 0.1 (effective +0.3)

### 4B-3 — Add ply-to-end head (~70 LOC)
- Rust: add `position_index: u16` per-position field to PushSingleConfig,
  PushGameConfig (per-row, batched), PushManyConfig
- Rust: storage.rs schema bump (header version) + INV pin
- Python: head module in network.py (~30 LOC):
  - 2-layer MLP from trunk avg-pool → 1 scalar
  - Trained on Huber loss against `target_remaining_ply = (game_length -
    position_index) / 100.0`
  - Weight 0.10
- Python: loss in losses.py
- Tests: per-row remaining_ply round-trip; loss converges on synthetic batch

### 4B-4 — DEFER true 2-stone opp-reply (out-of-scope for Wave 4)
- Building actual next-stone-1 + next-stone-2 targets requires either
  (a) Rust selfplay worker bookkeeping the next-turn replies + buffer
  carrying 2 extra (B, A) targets, OR (b) extracting from game records
  at sample time (slow path).
- LOC budget for (a): ~150 Rust + ~100 Python + buffer header version bump
  + bench gate + correctness tests.
- Net: ~250 LOC + bench-gate scope for ONE head, conditional value.
- DECISION: defer to Wave 5+ if Track 4B is the answer; bench-gate
  exclusion is significant.

### Pretrain decision
- **Option A — frozen-trunk warmup**: load v7full anchor, freeze trunk
  5000 steps, train new heads (ply-to-end) on frozen trunk, unfreeze.
  Cheap. ~$0.50 setup compute.
- **Option B — full re-pretrain** on bootstrap_corpus_v6.npz with new
  heads enabled from step 0. ~0.5 GPU-day, ~$1.50.
- **Recommendation**: Option A. sigma2 + ownership + threat already share
  trunk gradient; bumping weights and adding ply-to-end is incremental.
  No need to re-pretrain trunk from scratch.

## Track 4B sustained variant (gated on Track 4A verdict)

### Config: `configs/variants/v7full_wave4_multiaux.yaml`
Parent: TBD per Track 4A verdict routing:
- If W4A-A: Track 4B becomes optional. Recipe = v7full_baseline_minus_bot
  + multi-aux density.
- If W4A-B: parent v7full_baseline_minus_bot + multi-aux (this is the
  Wave 4 main attempt).
- If W4A-C: parent v7_wave3_main + multi-aux (keep refresh hook +
  per-class temp, add aux density).
- If W4A-D: STOP, surface to operator before proceeding.

### Pre-registered verdicts (LITERAL L13)
| ID | rule | downstream |
|---|---|---|
| W4B-A | Rolling-mean SealBot WR ≥ 20% sustained 30k-50k AND aux losses converge | Phase 4.5 UNBLOCKED. Promote final checkpoint. |
| W4B-B | Peak ≥ 20% but late decline | Tighter aux weights OR add refresh hook on top. Wave 5 design. |
| W4B-C | Reaches colony attractor end-state | Multi-aux not the answer. Strategic reckoning Wave 5: WDL OR target-propagation. |
| W4B-D | Other anomaly | STOP, debug. |

### Cost
~$3 vast, ~14h. Operator-mediated.

## Hard cap reminder
Wave 4 budget $7 total ($2.50 Track 4A + $3 Track 4B + buffer). Override
ONLY on (a) wr_sealbot rolling mean rising in last 10k OR (b) novel
signal not seen in prior waves. NO override on monotonic decline.

## Scope decision history

**2026-05-26 initial scope (operator approved "Full")**: 4B-1 + 4B-2 +
4B-3 + 4B-4 (true 2-stone opp-reply). Estimated ~250 LOC per dispatcher.

**2026-05-26 scope reduction post-Rust audit (executor decision)**:
Deeper inspection of `engine/src/replay_buffer/` (storage.rs +
push_config.rs + mod.rs + persist/) revealed Full scope actual cost is
~480 LOC + bench gate + corpus NPZ migration:
- Rust buffer ext for position_index + opp_reply_1 + opp_reply_2: ~250
  LOC across storage / push / sample / persist / mod (+ tests)
- Corpus NPZ format negotiation (existing pretrain rows carry only
  ownership + winning_line; new fields default to zeros): ~50 LOC
- Worker target generation (game_runner emits next-2 stones per row):
  ~50 LOC Rust
- Python heads + losses + trainer wiring: ~100 LOC
- Tests + INV pin: ~50 LOC
- Bench gate (mandatory per replay_buffer touch): ~$0.30 vast + ~$0.10
  laptop wall

This is significantly larger than the dispatcher's ~250 LOC framing.
The 2-stone opp-reply implementation (4B-4) is the cost driver; the
opp_reply fields nearly double policy-storage memory (+2.9 GB on 1M
capacity buffer with v7full 362-action stride).

**Executor decision: scale back to ply-to-end only (revised Lean).**
Reasoning:
- Operator approved Full assuming ~250 LOC; actual ~480 LOC overshoots
  the implicit time budget.
- ply-to-end alone tests the temporal-planning hypothesis at a fraction
  of the cost (~80 LOC + bench gate).
- If ply-to-end + 4B-impl-5 (sigma2 Huber + density bumps) materially
  changes Track 4B sustained behavior, 2-stone opp-reply can be added
  in Wave 5 as a follow-up.
- If they don't, 2-stone opp-reply was likely wasted work anyway —
  KataGo density hypothesis would be falsified.

**Revised Wave 4 4B scope**:
- 4B-impl-5: DONE (Huber sigma2 + density bumps)
- 4B-impl-1: position_index buffer field only (drop opp_reply_1/2)
- 4B-impl-3: ply-to-end head + loss + trainer wiring + tests
- 4B-impl-2: SKIP (no opp_reply worker target generation)
- 4B-impl-4: SKIP (no 2-stone opp-reply heads — DEFERRED to Wave 5+)
- 4B-impl-6: Track 4B variant + verdict pre-reg
- 4B-impl-7: Bench gate (mandatory)

Operator may overrule before 4B-impl-1 lands.
