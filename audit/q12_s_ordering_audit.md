# Q12 — Shaped Reward S-Ordering Correctness Audit

> Inspection date: 2026-05-20
> Auditor: Claude (master branch, §S180b launch day)
> Scope: Shaped-reward table, W/S/C S-ordering correctness, quiescence override coverage
> Status: INSPECTION ONLY — no source edits, no config changes

---

## §1. Shaped-Reward Table: Location and Full Mapping

### 1.1 Design-spec table (never implemented)

The shaped-reward table lives in `docs/01_architecture.md` lines 404–416:

| Formation event | Shaped reward |
|---|---|
| Singlet (pre-emp) created | +0.01 |
| Singlet blocked | +0.005 |
| Threat created | +0.03 |
| Threat blocked | +0.02 |
| Double threat created | +0.08 |
| Triangle created (`x A0 A1`) | +0.20 |
| Open Three created (`x A0 A3`) | +0.15 |
| Rhombus / Arch created | +0.18 |
| Ladder created | +0.25 |
| Bone created | +0.15 |
| Opponent reached unstoppable formation | −0.30 |

### 1.2 Implementation status: NOT IMPLEMENTED

`engine/src/formations/mod.rs` was deleted at commit `de41875` (2026-04-18):

> chore(engine): remove dead formations module (superseded by quiescence override §28)
> FormationDetector::has_forced_win had no callers across the workspace.

No formation detection code exists in the current codebase. Verified with:

```
rg 'formation|Formation|shaped_reward' engine/src/ hexo_rl/   # zero non-test hits
```

The config reward surface contains only:
- `draw_reward: 0.0` (`configs/training.yaml:draw_reward`)
- `ply_cap_value: 0.0` (`engine/src/game_runner/config.rs:55`)
- `quiescence_blend_2: 0.3` (`engine/src/mcts/mod.rs:122`)

No shaped-reward keys exist in any YAML config or Python training code.

**The S-ordering table is a dead design spec. It has no runtime effect.**

---

## §2. W3 Family Analysis — W3S0 / W3S1 / W3S2 Weights

### 2.1 W/S/C taxonomy applied to the design-spec table

Using the community Threat Theory framework (W = moves opponent needs to block;
S = stages from forced win; C = moves to advance):

| Design-spec entry | Shaped reward | W/S/C class | Reasoning |
|---|---|---|---|
| Opponent reached unstoppable | −0.30 | W3S0 (opponent) | ≥3 unblockable completions exist NOW |
| Triangle created | +0.20 | **W3S1** | One move away from W3S0 (rhombus/ladder class) |
| Rhombus / Arch created | +0.18 | **W3S1** | Same class; community-classified forced win in 2 steps |
| Ladder created | +0.25 | **W3S1** (or W3S2 if long) | Forced win class; exact S depends on ladder length |
| Open Three created | +0.15 | **W3S1** (pre-W3S0) | Creates W3S0 in exactly 1 move |
| Bone created | +0.15 | Unknown | No community W/S/C assigned; see §2.2 |
| Double threat created | +0.08 | W2S0 | Two immediate threats; blockable (opponent uses 2 stones) |
| Threat created | +0.03 | W1S0 (ambiguous) | Single immediate threat; scope unclear — see §3.2 |
| Threat blocked | +0.02 | N/A (defensive) | |
| Singlet created | +0.01 | W1S3+ | Pre-emptive; furthest from forcing |
| Singlet blocked | +0.005 | N/A (defensive) | |

### 2.2 W3 family reward comparison

| Class | Entries | Reward range |
|---|---|---|
| W3S0 (opponent's forced win) | "Opponent unstoppable" | −0.30 |
| W3S0 (own forced win) | Not explicitly listed | NOT PRESENT in table |
| W3S1 | Triangle, Rhombus/Arch, Ladder, Open Three | +0.15 to +0.25 |
| W3S2+ | Not defined separately | — |

**Critical gap:** There is no explicit reward entry for W3S0 — reaching a position
where the current player has ≥3 unblockable winning moves. The "Threat created"
entry at +0.03 is the nearest candidate but it is ambiguous (see §3.2).

If "Threat created" fires on W3S0, the reward (+0.03) is far lower than W3S1
formations (+0.15 to +0.25), which is a S-priority inversion. S0 is more urgent
than S1; it cannot have lower reward within the same W class.

---

## §3. S-Priority Violations

### 3.1 S-priority rule

Within the same W class, lower S (more forcing) must receive equal or higher
reward. Across W classes, higher W (fewer blocks needed) dominates.

### 3.2 Violation V1 — W3S0 absent; "Threat created" scope undefined (CRITICAL if W-scope = W3)

"Threat created" at +0.03 is ambiguous:

- **Option A:** fires on W1S0 only (single-axis immediate threat). No violation.
- **Option B:** fires on any S0 regardless of W, including W3S0 (≥3 immediate
  winning moves). Then W3S0 gets +0.03 while W3S1 gets +0.15 to +0.25.
  **That is an S-priority inversion: S0 rewarded 5-8× less than S1.**

Since the formation module was deleted before Option A or B was codified, there
is no implementation to audit. But the design spec leaves this undefined.

**Status:** LATENT violation in the design spec. No runtime impact because
shaped rewards are not implemented.

### 3.3 Violation V2 — S1 rewards span +0.15 to +0.25 (no S-ordering violation, but magnitude spread is unexplained)

All four W3S1 entries (Triangle, Rhombus/Arch, Open Three, Ladder) receive
different rewards (+0.15 to +0.25). Community framework assigns identical
W3S1 class to all. The spec creates a preference ordering within S1 that is
not grounded in threat theory.

**Status:** Not a strict S-ordering violation (all are S1; rule requires equal-or-higher,
not equal). But the 67% spread (Ladder +0.25 vs Open Three +0.15) has no
stated justification in the spec or docs.

### 3.4 Violation V3 — Cross-S inversion: S1 > S0 by weight

W3S1 rewards (+0.15 to +0.25) exceed all S0 rewards (+0.03 to +0.08 for W1/W2).
Under the S-first priority rule, S0 (immediate forcing) should dominate S1.
The spec inverts this for the W3 family: the network would be incentivized to
build toward W3S1 formations rather than exploit existing W3S0 positions.

**Status:** Design-level tension documented in the prior audit (`docs/q12_s_ordering_audit.md`,
2026-04-06). Not a runtime bug (no implementation). If shaped rewards were
ever implemented, this would bias the network toward farming S1 formations —
the exact "reward farming" hazard the architecture doc warns about at line 422.

### 3.5 S-ordering summary table

| W family | S0 reward | S1 reward | Priority-correct? |
|---|---|---|---|
| W3 (own) | ABSENT (or +0.03 if "Threat" fires on W3S0) | +0.15 to +0.25 | **NO** — S0 absent or underweighted vs S1 |
| W3 (opponent) | −0.30 | — | N/A — terminal penalty |
| W2 | +0.08 (Double threat) | — | OK (no W2S1 defined) |
| W1 | +0.03 (Threat) | — | OK (no W1S1 defined) |

---

## §4. Quiescence Override Scope Analysis

### 4.1 Implementation

`engine/src/mcts/backup.rs:144–213` (`apply_quiescence`):

```rust
let current_wins = board.count_winning_moves(current_player);
if current_wins >= 3 { return 1.0; }   // W3S0: forced win override
if opponent_wins >= 3 { return -1.0; } // opponent W3S0: forced loss override
if current_wins == 2 { return (value + blend_2).min(1.0); } // W2S0: blend
if opponent_wins == 2 { return (value - blend_2).max(-1.0); } // opp W2S0: blend
```

`board.count_winning_moves(player)` (`engine/src/board/moves.rs:261–283`) counts
empty cells that, if occupied by `player`, complete a 6-in-a-row along any of
the three hex axes. A count ≥ 3 means the opponent cannot block all threats
in their 2-stone response.

### 4.2 What the quiescence override covers

| Condition | Board state | Quiescence result |
|---|---|---|
| current_wins ≥ 3 | **W3S0** — ≥3 immediate winning cells exist NOW | → +1.0 (override) |
| opponent_wins ≥ 3 | Opponent W3S0 | → −1.0 (override) |
| current_wins == 2 | **W2S0** — 2 immediate wins; blockable but strong | → NN value + 0.3 blended |
| opponent_wins == 2 | Opponent W2S0 | → NN value − 0.3 blended |

### 4.3 What the quiescence override does NOT cover: W3S1

**W3S1** (rhombus, ladder, triangle, open three) is the class where the current
player has a forced win in 2 steps: one move to create W3S0, then the resulting
W3S0 is unblockable. At a leaf node evaluated mid-game:

- `count_winning_moves` on a W3S1 position returns 0 or 1 (no immediate 6-in-a-row
  is completable yet — the threat requires one more stone to be placed first).
- The quiescence check therefore sees `current_wins < 3` and passes the NN value
  through unchanged.

Example: rhombus with 4 stones on two parallel lines, one move from creating 3
simultaneous winning completions. At this leaf: `count_winning_moves = 0`. The
quiescence check fires on neither branch. The NN value is returned as-is.

**W3S1 is theoretically a forced win but the quiescence override does not
recognise it.** The NN must learn the W3S1 → W3S0 → terminal chain entirely
from MCTS tree search (i.e., by searching deep enough to hit the W3S0 branch
in a child node). The override fires only when the W3S0 position is the leaf
being evaluated.

### 4.4 Is W3S1 quiescence coverage feasible / necessary?

Detecting W3S1 at a leaf requires looking one ply ahead: for each legal move,
call `count_winning_moves` on the resulting board, check if any move yields
count ≥ 3. Cost: O(legal_moves × count_winning_moves) per leaf — significantly
more expensive than the current O(1 count_winning_moves) check.

With 600 simulations per root the tree routinely reaches depth sufficient to
observe W3S0 nodes as children of W3S1 leaves. The practical forcing effect is
already captured via search. Quiescence override extension to W3S1 would reduce
dependency on search depth but at meaningfully higher per-leaf cost.

**Conclusion:** W3S1 gap is a design choice, not a correctness bug. The override
is correctly scoped to W3S0 (immediate forcing). W3S1 is handled implicitly via
tree expansion. If colony attractor persists at low sims (fast-path configs
where depth is insufficient to see W3S0 children), W3S1 detection would be the
right extension.

---

## §5. Verdict

**SHAPED REWARDS: NOT IMPLEMENTED — no S-ordering table is active in training.**

The formations module was removed at commit `de41875` (2026-04-18). The shaped-reward
table in `docs/01_architecture.md:404–416` is a dead design spec with no runtime
path. S-ordering violations in the spec are irrelevant to any current or in-flight
training run (§S180b included).

**QUIESCENCE OVERRIDE: CORRECT for its stated scope (W3S0).**

The override at `engine/src/mcts/backup.rs:144–213` correctly identifies the
W3S0 forced-win condition and overrides the NN value to ±1.0. It does not extend
to W3S1 (rhombus/ladder class) by design. This is a scope choice, not a bug.

**VERDICT: CORRECT** — no S-ordering table exists to be wrong; quiescence scope
is as designed.

---

## §6. No Source Fix Required

No correctness bug exists in active code. Shaped rewards are not implemented;
no diff is proposed.

If shaped rewards are ever revisited (Phase 5+), the design-spec must be revised
before implementation to address:

1. Define "Threat created" scope: W1S0 only vs any S0. If any S0, add explicit
   W3S0 entry at reward ≥ −(opponent unstoppable) = +0.30.
2. S-first ordering: S0 rewards must exceed S1 rewards within each W class.
   Current spec inverts this (+0.03 S0 < +0.15 S1 for W3 family).
3. Bone/Arch/Trapezoid: no community W/S/C classification; assign before coding.

Proposed corrected table (S-first, not a diff — for spec revision only):

| Formation event | Reward | W/S/C |
|---|---|---|
| Opponent reached W3S0 | −0.30 | W3S0 opp |
| Own W3S0 (triple threat) | +0.30 | W3S0 |
| Double threat (W2S0) | +0.10 | W2S0 |
| Threat (W1S0 only) | +0.05 | W1S0 |
| Ladder | +0.18 | W3S1+ |
| Triangle | +0.15 | W3S1 |
| Rhombus | +0.15 | W3S1 |
| Open Three | +0.15 | W3S1 |
| Bone (if W3S1) | +0.15 | W3S1 TBD |
| Singlet | +0.01 | W1S3+ |

---

## §7. Colony Attractor Interaction

### 7.1 Does the S-ordering gap explain colony bias?

**No direct mechanism.** Shaped rewards are not implemented, so the S-ordering
gap in the design spec cannot contribute to training dynamics. The colony
attractor (§175, §S179, §S180a reproductions) is a training-time phenomenon;
with zero shaped-reward signal in the selfplay reward path, the formations table
has no causal lever.

### 7.2 What the quiescence gap could contribute

The W3S1 quiescence gap is the only live mechanism related to threat ordering.
If MCTS search at production sim counts (600 full / 100 fast) is insufficient
depth to observe W3S0 positions as children of W3S1 leaves, the colony attractor
could partially arise from the NN failing to correctly value rhombus/ladder
positions. However:

- At 600 sims the tree routinely reaches depth ≥ 4 (two full turns), which is
  sufficient to observe the W3S0 branch from a W3S1 root.
- The colony attractor manifests as systematic preference for a fixed spatial
  cluster of moves, not as short-horizon tactical failure — the signature of a
  distribution-shift / value-head collapse, not a quiescence scope miss.
- §S180a (CQV-flip) being NULL/FAIL rules out completed-Q-value amplification
  as the mechanism; the attractor survives with sharp targets disabled.

**Verdict:** S-ordering gap is INDEPENDENT of the colony attractor. The attractor
is not caused or amplified by the quiescence W3S1 gap. The S-ordering audit
trigger condition (§S178 + §S179 + §S180a all fail) has been met, but the audit
finds no active shaped-reward mechanism to fix. Investigation should focus on
value-head target construction and training distribution, not reward shaping.

---

## Appendix — File/Line Reference

| Claim | File | Line(s) |
|---|---|---|
| Shaped-reward design table | `docs/01_architecture.md` | 404–416 |
| Formation module deleted | commit `de41875` | — |
| Quiescence `apply_quiescence` | `engine/src/mcts/backup.rs` | 144–213 |
| `count_winning_moves` | `engine/src/board/moves.rs` | 261–283 |
| `has_player_long_run` | `engine/src/board/moves.rs` | 230–248 |
| W3S0 override (≥3 wins) | `engine/src/mcts/backup.rs` | 188 |
| W2S0 blend | `engine/src/mcts/backup.rs` | 200–205 |
| `draw_reward` key | `configs/training.yaml` | — |
| `ply_cap_value` field | `engine/src/game_runner/config.rs` | 55 |
| Q12 entry (trigger condition) | `docs/06_OPEN_QUESTIONS.md` | 10 |
| Prior Q12 audit (2026-04-06) | `docs/q12_s_ordering_audit.md` | — |
