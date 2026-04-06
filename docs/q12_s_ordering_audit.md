# Q12: Shaped Reward S-Ordering Audit

> Inspection date: 2026-04-06
> Auditor: Claude (laptop branch, Phase 4.5 prep)
> Scope: Formation detection code, shaped reward table, W/S/C classification
> Status: **INSPECTION ONLY** — no code or config changes made

---

## 1. Formation Detector State

### 1.1 Rust Enum (`engine/src/formations/mod.rs`)

The `Formation` enum defines 7 variants:

| Variant | Implemented? | Detection logic |
|---|---|---|
| `Triangle` | **No** — enum only | No pattern matching code |
| `OpenThree` | **No** — enum only | No pattern matching code |
| `Rhombus` | **No** — enum only | No pattern matching code |
| `Arch` | **No** — enum only | No pattern matching code |
| `Trapezoid` | **No** — enum only | No pattern matching code |
| `Ladder` | **No** — enum only | No pattern matching code |
| `Bone` | **No** — enum only | No pattern matching code |

### 1.2 Only Implemented Method: `has_forced_win()`

`FormationDetector::has_forced_win()` (line 21-53) detects **open 4-in-a-row** —
4+ consecutive same-colour stones along any hex axis with both endpoints empty.
This is a single boolean check used for analysis/logging. It does NOT:

- Classify which formation type was detected
- Return the formation variant
- Emit any event
- Connect to any reward pathway

### 1.3 Threat Detection (`engine/src/board/threats.rs`)

Viewer-only sliding window scanner. Detects N-of-6-in-a-window (N >= 3) along
all 3 hex axes. Returns empty cells within threatening windows, tagged with
severity levels:

| Level | Meaning | N stones in window |
|---|---|---|
| 3 | WARNING | 3 stones, 3 empty |
| 4 | FORCED | 4 stones, 2 empty |
| 5 | CRITICAL | 5 stones, 1 empty |

**Explicitly marked `NEVER called from MCTS or training. Viewer only.`**

This module detects linear threats only — no geometric formations (Triangle,
Rhombus, etc.). It does not distinguish open vs closed ends.

### 1.4 Reward Pathway: **Does Not Exist**

Searched all configs and Python/Rust source:

- `configs/training.yaml` — no shaped reward keys. Only `draw_reward: -0.5`,
  `threat_weight: 0.1` (auxiliary head loss weight, NOT shaped reward),
  `ownership_weight: 0.1`.
- `configs/selfplay.yaml` — no shaped reward keys.
- `engine/src/game_runner.rs` — outcome is purely terminal: +1.0 (win),
  -1.0 (loss), or `draw_reward` (configurable, default -0.5). No shaped
  reward signals are added.
- No Python code in `hexo_rl/selfplay/` references formations or shaped rewards.

**The shaped reward table in `docs/01_architecture.md` (lines 283-295) is a
design spec that has never been implemented.** No formation events are emitted,
no shaped rewards are computed, no decay schedule exists in config.

---

## 2. W/S/C Classification of Design-Spec Formations

Even though nothing is implemented, the architecture doc specifies rewards for
11 formation events. Here is the W/S/C classification using the community's
Threat Theory framework from `docs/10_COMMUNITY_BOT_ANALYSIS.md` §7.1:

| Formation Event (from arch doc) | Spec'd Reward | W/S/C | Classification Notes |
|---|---|---|---|
| Singlet (pre-emp) created | +0.01 | **W1S3+** (estimated) | Pre-emptive — furthest from forcing. No community W/S/C defined for singlets. Estimated S3+ based on distance from any forcing line. |
| Singlet blocked | +0.005 | N/A (defensive) | Defensive action, not a formation. W/S/C doesn't apply to blocks. |
| Threat created | +0.03 | **W1S0** (if single axis) | Ambiguous — "threat" could mean any S0 structure. If it fires on W3S0 (triple threat / forced win), +0.03 is far too low. See Open Question #1 below. |
| Threat blocked | +0.02 | N/A (defensive) | Defensive action. |
| Double threat created | +0.08 | **W2S0** | Two simultaneous W1S0 threats. Blockable (opponent uses both stones). NOT a forced win per community framework. |
| Triangle created | +0.20 | **W3S1** | Forced win class — one step from unblockable. |
| Open Three created | +0.15 | **Pre-W3S0, so S1** | Creates triple threat in 1 move. The formation itself is S1 (one tempo from forcing). |
| Rhombus / Arch created | +0.18 | **W3S1** (Rhombus) / **Unknown** (Arch) | Rhombus is W3S1 per community table. Arch has no community classification — see Open Question #2. |
| Ladder created | +0.25 | **W3S1+** | Forced win class, exact S depends on length. |
| Bone created | +0.15 | **Unknown** | Not in community table. See Open Question #3. |
| Opponent unstoppable | -0.30 | **W3S0** (opponent) | Opponent reached forced win. Terminal-adjacent penalty. |

---

## 3. S-Ordering Analysis

### 3.1 The S-Ordering Rule

Within the same Weight class, lower S (more urgent) formations MUST have
equal or higher shaped reward than higher S formations.

### 3.2 Violations Found

**Violation 1: "Threat created" (+0.03) vs "Double threat created" (+0.08)**

- "Threat created" is classified as W1S0 (single immediate threat).
- "Double threat created" is W2S0 (two simultaneous threats, blockable).
- Within S0, W2 > W1 in urgency, and the reward correctly reflects this
  (+0.08 > +0.03). **No violation here if "threat" means W1S0 only.**
- **BUT:** If "threat created" also fires on W3S0 (triple threat = forced win),
  then a forced win (+0.03) would be rewarded LESS than a blockable double
  threat (+0.08). This would be a **critical S-ordering violation**.

**Violation 2: Open Three (+0.15) < Rhombus/Arch (+0.18) < Triangle (+0.20)**

- All three are W3S1 formations (forced win class, one step from forcing).
- Community framework assigns identical W3S1 to Triangle, Rhombus, and
  Open Three. They are threat-theoretically equivalent.
- Current ordering: Ladder (+0.25) > Triangle (+0.20) > Rhombus/Arch (+0.18)
  > Open Three (+0.15).
- **This is NOT an S-ordering violation** (all are S1, so the rule only
  requires equal-or-higher within S). The differences create a preference
  ordering among S1 formations that the community framework doesn't specify.
- **However:** The magnitude spread (+0.15 to +0.25) implies the network
  should prefer Ladders over Open Threes by 67%. Whether this matches
  tactical reality is an open question — a Ladder is not inherently 67%
  more valuable than an Open Three.

**Violation 3: "Threat created" (+0.03) < any S1 formation (+0.15 to +0.25)**

- A W1S0 threat (immediately forcing) is rewarded LESS than W3S1 formations
  (one step from forcing).
- **This IS an S-ordering violation across weight classes.** An S0 formation
  of any weight is more urgent than an S1 formation. The opponent must respond
  to S0 NOW; they can delay responding to S1.
- Counter-argument: W3S1 formations are strategically more valuable (they lead
  to forced wins), even though S0 threats are more urgent. This is the
  Weight-vs-Strength tension the community framework acknowledges.
- **Verdict:** This is a design choice, not a bug. But it should be documented
  as intentional. The risk is that the network learns to build S1 formations
  instead of creating immediate threats, which could lead to the "farming"
  behavior the architecture doc warns about.

**Violation 4: "Singlet" (+0.01) reward magnitude**

- A singlet is a pre-emptive move (S3+). It correctly has the lowest positive
  reward. **No violation.**

### 3.3 Cross-Weight Ordering Summary

| S Level | Formations | Reward Range | Ordering |
|---|---|---|---|
| S0 | Threat (+0.03), Double threat (+0.08) | 0.03 — 0.08 | W-ordered within S0: OK |
| S1 | Open Three (+0.15), Bone (+0.15), Rhombus/Arch (+0.18), Triangle (+0.20), Ladder (+0.25) | 0.15 — 0.25 | All equal S, spread by unspecified preference: ACCEPTABLE |
| S3+ | Singlet (+0.01) | 0.01 | Lowest: OK |
| Defensive | Singlet blocked (+0.005), Threat blocked (+0.02) | 0.005 — 0.02 | Defensive, not comparable: OK |

**Cross-S comparison:** S1 rewards (0.15-0.25) > S0 rewards (0.03-0.08).
This means the network will prefer building strategic W3S1 formations over
creating immediate W1S0 threats. Whether this is correct depends on the
training phase — early training may benefit from tactical S0 rewards being
higher, while late training benefits from strategic S1 rewards dominating.

---

## 4. Recommended Reward Adjustments (Config Only)

These are recommendations for when shaped rewards are eventually implemented.
No changes should be made now since no implementation exists.

### 4.1 Critical: Define "Threat Created" Scope

Before assigning any values, define precisely what "threat created" detects:

- **Option A:** W1S0 only (single-axis immediate threat). Keep +0.03.
- **Option B:** Any S0 threat (including W2S0, W3S0). Must have W-scaled
  reward: W1S0=+0.03, W2S0=+0.08, W3S0=+0.30 (matching "opponent unstoppable").
- **Recommendation:** Option A. Have separate events for each W level at S0.

### 4.2 Suggested Revised Table (Respecting S-Ordering)

Two viable orderings depending on philosophy:

**Option 1: S-first ordering (urgency dominates)**

| Formation Event | Reward | W/S/C | Rationale |
|---|---|---|---|
| Opponent unstoppable (W3S0) | -0.30 | W3S0 opp | Terminal-adjacent, keep |
| Triple threat created (W3S0) | +0.30 | W3S0 | Forced win = near-terminal |
| Double threat created (W2S0) | +0.08 | W2S0 | Keep (blockable but urgent) |
| Threat created (W1S0) | +0.05 | W1S0 | Slight bump from +0.03 |
| Ladder created | +0.18 | W3S1+ | S1 < S0 strictly |
| Triangle created | +0.15 | W3S1 | |
| Rhombus created | +0.15 | W3S1 | Same as Triangle |
| Open Three created | +0.15 | W3S1 | Same as Triangle |
| Arch created | +0.15 | W3S1 (TBD) | Pending community classification |
| Bone created | +0.12 | TBD | Pending community classification |
| Threat blocked | +0.02 | Defensive | Keep |
| Singlet created | +0.01 | W1S3+ | Keep |
| Singlet blocked | +0.005 | Defensive | Keep |

**Option 2: W-first ordering (strategic value dominates) — current approach**

Keep current values but add explicit W3S0 event at +0.30. Accept that S1 > S0
for lower-W formations as a deliberate "build toward forced wins" bias.

### 4.3 Recommendation

**Option 1 (S-first)** is safer for early training where the network needs to
learn tactical responses before strategic planning. The decay schedule will
phase out shaped rewards before strategic formation-building becomes important.

---

## 5. Open Questions Requiring Tom's Judgment

### Open Question 1: "Threat created" scope

Does "threat created" fire on:

- (a) Any S0 threat regardless of W? If so, a W3S0 (forced win) gets only +0.03.
- (b) W1S0 only? Then we need a separate "triple threat created" event at much
  higher reward.

**Impact:** If (a), this is a critical S-ordering violation. If (b), no violation.

Since there's no implementation, this is a design question to resolve before
coding the FormationDetector's incremental detection.

### Open Question 2: Arch classification

"Arch" is listed in the Formation enum and the reward table but has no
community W/S/C classification. What geometric pattern is an Arch?

- If it's a variant of Rhombus (two parallel lines with a bridge): W3S1.
- If it's an open-ended structure: could be S2 or higher.
- **Need:** Geometric definition from community knowledge base or Tom's
  specification.

### Open Question 3: Bone classification

"Bone" appears in both the Formation enum and reward table (+0.15, same as
Open Three) but has no community W/S/C classification.

Possible interpretations:

- **Dog-bone shape:** Two parallel short lines connected at center (like H or
  dumbbell). If it creates threats on two axes: could be W2S1 or W3S1.
- **Literal bone (straight with knobs):** A line with perpendicular extensions
  at endpoints. Fork potential depends on geometry.

**Need:** Geometric definition. If Bone is W2S1 (not forced win class), its
reward should be lower than W3S1 formations (currently equal at +0.15).

### Open Question 4: Trapezoid

`Trapezoid` appears in the Formation enum but NOT in the reward table. Either:

- It was added after the reward table was written (needs a reward assigned), or
- It's a sub-component of another formation (shouldn't be rewarded separately).

**Need:** Clarification on whether Trapezoid should have its own shaped reward.

### Open Question 5: Implementation priority

The entire shaped reward system is unimplemented. The formation enum is a stub.
`has_forced_win()` is the only detection logic. Before implementing:

1. Is shaped reward still planned for Phase 4.5, or has the quiescence override
   (already implemented) reduced the need?
2. The architecture doc warns about reward farming (confirmed community incident).
   Should we start with only S0 events (Threat/Double/Triple) and add S1
   formations later after confirming no farming behavior?

---

## 6. Bone: Community Consensus Needed?

**Yes.** Bone has no W/S/C classification in the community Threat Theory
framework. Before assigning a shaped reward:

1. Define the exact geometric pattern (which cells, which axes).
2. Determine the threat-theoretic properties (how many moves to block, how
   many steps from forcing).
3. Ideally, validate against SealBot/KrakenBot evaluations — do positions
   with "Bone" formations have higher win rates?

If Bone turns out to be W2S1 rather than W3S1, its current reward (+0.15,
tied with W3S1 Open Three) is too high. If it's W3S1, it's correctly placed.

The same applies to Arch and Trapezoid — all three lack community
classification and should be flagged for review.

---

## 7. Summary

| Finding | Severity | Action |
|---|---|---|
| Shaped rewards are entirely unimplemented | INFO | No current training impact |
| Formation enum is a stub (7 variants, 0 detectors) | INFO | Full implementation needed before Phase 4.5 |
| `has_forced_win()` detects open-4 only, no classification | INFO | Covers W2S0 case only |
| "Threat created" scope undefined — may create W3S0 violation | **HIGH** | Resolve before implementing (Open Question 1) |
| S0 rewards < S1 rewards in current spec | **MEDIUM** | Intentional? Document or fix (Section 4) |
| Bone/Arch/Trapezoid lack community W/S/C | **MEDIUM** | Get definitions before assigning rewards (Section 6) |
| No shaped reward keys in any config file | INFO | Must be added to `configs/training.yaml` when implemented |
| threats.rs is viewer-only, not connected to rewards | INFO | Correct separation of concerns |
