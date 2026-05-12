# §174 Game Quality Investigation

**Date:** 2026-05-12
**Scope:** Investigate short games, win-rate imbalance, and LEGAL_MOVE_RADIUS status
for the v6w25 α sustained run (§173 → §174).

---

## Dashboard observations (from operator brief)

| Metric | Value |
|---|---|
| P0 wins | 34.8% |
| P1 wins | 65.2% |
| Draw rate | 0.0% |
| Median game length | ~13 compound moves |
| Game pattern | Isolated cluster building, no defensive play |
| Axis bias | axis_r max frac 0.9555 (strong axis bias) |

---

## LEGAL_MOVE_RADIUS status

### Current value for v6w25

**`legal_move_radius = 8`** — confirmed by three independent sources:

1. **`engine/src/encoding/registry.toml`** (canonical source of truth):
   ```toml
   [encodings.v6w25]
   legal_move_radius = 8
   ```

2. **`engine/src/board/state.rs:260`** — `Board::with_registry_spec()` binds the
   radius from the spec at construction time:
   ```rust
   b.legal_move_radius = spec.legal_move_radius as i32;
   ```

3. **`engine/src/game_runner/worker_loop.rs:301`** — jitter is **guarded** by
   `encoding.is_none()`, so v6w25 runners (which always pass `encoding_spec`)
   are **not jittered**:
   ```rust
   if legal_move_radius_jitter && encoding.is_none() {
       board.set_legal_move_radius(r);  // skipped for v6w25
   }
   ```

Test `legal_move_radius_jitter_skipped_when_encoding_present`
(`engine/src/game_runner/mod.rs:862-896`) asserts this explicitly:
```rust
assert_eq!(board.legal_move_radius(), 8);
assert!(!([4, 5, 6].contains(&board.legal_move_radius())));
```

### §146 rationale still applies?

**Partially — the cap was for v6, not v6w25.**

- §146 capped `DEFAULT_LEGAL_MOVE_RADIUS` from 8→5 to prevent **fragmentation**
  past the 19×19 single-window encoding (§142).
- v6w25 uses **25×25 cluster windows** (`cluster_window_size = 25`), so the
  fragmentation pivot that hit v6 at ply ~31 is pushed much further out.
- v6w25 already uses radius **8** (from the registry) and the jitter guard
  keeps it at 8. The §146 constant (`DEFAULT_LEGAL_MOVE_RADIUS = 5`) only
  affects bare-defaults v6 paths.

### Recommendation

**Keep radius 8 for v6w25.** Raising it further (e.g., back to the official
rule's unbounded behaviour) would re-introduce the fragmentation risk even
inside a 25×25 window once games reach ply 60+. The current value is the
registry's intentional setting and is verified by test.

If game length remains critically short (sub-20 plies) after the model gains
basic threat awareness, then radius could be revisited — but that is a
separate tuning decision, not an immediate fix.

---

## `legal_move_radius_jitter` effect

**Effectively DISABLED for v6w25.**

- `vast.yaml` has `legal_move_radius_jitter: true`.
- Because v6w25 passes `encoding_spec` to `SelfPlayRunner`, the jitter branch
  in `worker_loop.rs:301` is never taken.
- Jitter only affects **legacy v6 paths** (no encoding_spec, bare `Board::new()`).
- For v6w25 sustained runs, this knob is a no-op.

**Implication:** the §157 Q2 jitter benefit (mitigating stride-5 fixed point,
−9% colony fraction) does NOT apply to v6w25. If the v6w25 run later exhibits
a colony-attractor signature, jitter would need to be extended to encoding-aware
paths — but that is an engine change, not a config change.

---

## Pretrain quality comparison

| Bootstrap | WR vs SealBot | Corpus | Encoding |
|---|---|---|---|
| v6 | 15% (3/20) | 353K human positions, Elo-weighted | 19×19 single-window |
| v6w25 | 0% (0/100) | Same corpus, re-encoded to 25×25 | 25×25 K-cluster |

### Possible causes for the 0% gap

1. **Re-encoding signal loss** — Positions that were tight and tactical in
   19×19 become sparse in 25×25. Local structure (threats, blocks) that the
   network learned to recognise at v6 density is now diluted across a larger
   canvas. The pretrain objective (policy cross-entropy) does not penalise
   spatial dilution directly.

2. **Policy head capacity** — v6w25 policy logits = 626 (25×25 + pass) vs
   v6 = 362 (19×19 + pass). The same 128-channel trunk now distributes
   representational capacity over ~73% more output units. A head trained on
   19×19 density may generalise poorly to the expanded action space.

3. **Bootstrap eval regime** — v6w25 bootstrap eval was n=100 vs SealBot
   (more rigorous than v6's n=20). A weak but non-zero bootstrap might have
   scored 5–10% at n=20 and 0% at n=100 by sampling variance.

4. **No v6w25-specific pretrain** — The corpus was re-encoded from v6; there
   is no v6w25-native human-game corpus. The model has never seen a
   25×25-geometry position that was produced by strong play at that scale.

---

## Root cause analysis: short games + no defence

### Expected from weak pretrain

The observations are **consistent with a weak bootstrap** and do NOT indicate
a config or architecture bug:

- **0% vs SealBot** means the bootstrap has essentially no strategic
  understanding beyond "place stones legally."
- **13 compound-move median** means games resolve quickly — the first player
  to complete a line wins because the opponent does not block.
- **No defensive play** means the value head has not yet learned that
  `opponent_has_5_in_a_row → loss`. This is a training-signal issue, not a
  structural one: the value head learns from game outcomes, and with
  `draw_rate = 0`, every game provides a crisp ±1 outcome signal.
- **P0/P1 imbalance (34.8/65.2)** is typical of random/weak play where the
  second player has a structural advantage in Hex Tac Toe (first player opens
  with 1 stone, both players then place 2 per turn). The first move advantage
  is small; the second player's double-move tempo dominates at low skill.

### Training-signal trajectory

Threat awareness should emerge from selfplay value-head learning:

- **Step 0–10K:** Model plays like bootstrap (~0% vs SealBot). Games short,
  no defence. Value head learns `terminal_win = +1`, `terminal_loss = -1`.
- **Step 10K–30K:** Value head begins to recognise intermediate threats.
  Game length may INCREASE as players start blocking each other.
  Draw rate may rise from 0% toward 10–20%.
- **Step 30K–60K:** Policy head catches up to value head. Defensive moves
  enter the policy. Win rate vs SealBot should begin rising from 0%.
- **Step 60K+:** If trajectory is healthy, WR vs SealBot should cross 10%,
  then 20%, approaching the v6 baseline of 15% and beyond.

**No step count guarantee** — this depends on the policy/value head
synchronization, the buffer mixing ratio (pretrain vs selfplay), and the
MCTS sim count (400 sims × 128 batch = modest exploration).

### Is the current game length / quality blocking learning?

**No — short games are actually high-signal.**

- Every game produces a terminal outcome (no draws = 100% outcome rate).
- The value head gets maximum gradient per game.
- The policy head learns from the actual winning sequence.
- Longer games (with blocking) would emerge naturally as the model improves;
  forcing longer games artificially (e.g., by lowering `legal_move_radius`)
  would not accelerate learning and might reduce signal density.

The risk is NOT game length; it is **stagnation** (WR vs SealBot stays at 0%
forever). The operator should monitor:
1. WR vs SealBot at 10K intervals.
2. Draw rate trend (should rise from 0%).
3. Mean game length trend (should rise as defence emerges).

---

## Recommendations (OPERATOR DECIDES — do not implement)

1. **Keep current config and let training run longer.**
   The v6w25 bootstrap is weak but selfplay is the correct learning signal.
   Monitor WR vs SealBot, draw rate, and mean game length. No config or
   architecture changes needed at this stage.

2. **LEGAL_MOVE_RADIUS for v6w25 — keep at 8 (registry default).**
   Raising it further risks fragmentation in 25×25 windows at late plies.
   Lowering it to 5 would shorten games further and is counterproductive.

3. **Consider a stronger pretrain for v6w25 only if WR stays at 0% past 40K steps.**
   Options: (a) generate a v6w25-native corpus from SealBot games,
   (b) retrain the pretrain with data augmentation at 25×25 scale,
   (c) bootstrap from a v6 checkpoint that already has 15% WR.

4. **Extend `legal_move_radius_jitter` to encoding-aware paths if colony
   attractor emerges.** Currently jitter is a no-op for v6w25. If dashboard
   shows colony fraction rising above trace levels, extend the jitter guard
   in `worker_loop.rs:301` to sample from a v6w25-appropriate range
   (e.g., `{6, 7, 8}` instead of `{4, 5, 6}`).

---

*Investigation completed 2026-05-12. No code changes recommended.*
