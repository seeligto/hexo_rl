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

## Open question for operator

Confirm revised scope before implementation:
- (recommended) **Lean**: 4B-1 + 4B-2 + 4B-3 + DEFER 4B-4. ~70 LOC + tests
  + INV pin. ~3h dev.
- (full per dispatcher) **Full**: 4B-1 + 4B-2 + 4B-3 + 4B-4 build out
  true 2-stone opp-reply. ~250 LOC + Rust buffer ext + bench gate. ~8h
  dev + ~$0.30 bench compute.

Recommendation: **Lean** because the 4B sustained run cost ($3) is the
limit, not LOC. We get most of the aux density benefit from 4B-1/2/3
alone, and 4B-4 buffer extension is a long-term feature worth doing on
its own time (not a Wave 4 rush). Operator may overrule.
