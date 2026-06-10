# §178 design — bot-corpus mixing + ply-cap-value split

**Date:** 2026-05-18. **Anchor:** `checkpoints/bootstrap_model_v6.pt` (canonical v6 base bootstrap, §175 anchor).
**Predecessor closed:** §175 v6 sustained from same v6.pt anchor — recipe alone cannot escape attractor (SealBot 18→4% across 20K→70K).
**§177** (different anchor: step-20K weights, same recipe): reproduced same colony trajectory (2→0% across 10K→40K). Confirms recipe-dependent, anchor-independent.
**Investigation:** `reports/s178_pre_design_investigation.md` (5-way subagent dispatch, 2026-05-18).

---

## 0. Executive summary

§178 introduces SealBot-vs-anchor pre-generated games as a separate top-level batch slot
(NOT inside selfplay slice, NOT decaying with pretrain_weight) at `bot_batch_share=0.15`.
Pairs with `ply_cap_value` split from `draw_value` to penalise truncation while leaving
organic-draw treatment as operator pre-commits (`-0.5` → `-0.1`).

Two new mechanism levers:
1. **Bot-corpus slot** — fixed 15% of batch, sourced from SealBot vs v6.pt NPZ.
   Doesn't compete with pretrain_weight schedule; doesn't get LRU-evicted with selfplay churn.
2. **Ply-cap-value split** — separates "ran out of moves" from "neutral organic draw".
   Restores resolution pressure on long colony-prone games without affecting genuine draws.

**Anchor switch rationale (operator pick 2026-05-18):** v6.pt instead of v6_step20k.pt. Cleaner
experimental control — §175 (same v6.pt anchor, no bot) vs §178 (same v6.pt, +bot+ply_cap) is a
direct A/B testing the mechanism intervention. v6_step20k.pt has 20K steps of selfplay state
leakage; v6.pt is fresh post-pretrain.

KrakenBot DROPPED for §178 (operator call, supported by Wave C BT −494 to −3072 Elo).
Bot vs bot games SKIPPED (no anchor-mistake signal). Bootstrap corpus UNCHANGED (human-only per §148).

---

## 1. Direct answers to operator's 6 questions

| Q | Answer | Citation |
|---|---|---|
| Drop KrakenBot, SealBot only? | **YES.** Kraken −494 to −3072 Elo, sentinel-fallback 0.42/game distorts strength, MCTSBot weights blocked. | Wave C `ratings.csv`, §176 risk #1+#2, A1 §b |
| Bot vs bot games weird/skip? | **YES skip.** Bot-vs-bot = strong-vs-strong, no anchor mistakes to punish. Anchor-side-only retains the colony→loss signal. | §176 V6 PASS (colony opponent-coupled), L22 |
| Bot games in bootstrap corpus? | **NO.** v7 corpus human-only since §148; pre-v7 41% bot retired (L1/L15). Bot games go in BUFFER as separate slot. | §148, L15 |
| 10-15% bot in buffer? | **15%.** As separate top-level batch slot. Naive 10-15% inside selfplay is HB-2-CONDITIONAL undersized; top-level slot bypasses the 23-34% selfplay constraint. | SA-B HB-2, batch math §3 |
| Pre-gen + regen on anchor promotion? | **YES** for §178 static. Refresh hook wired with `enabled: false`; flip on for §179 if §178 outcome shows aging. | L8 head-drift analogue |
| Buffer turnover ~hours — retain bot games? | **Separate NPZ corpus path** (SA-D 1-WEEK surgery), reuse `pretrained_buffer` infra. No LRU eviction, stays constant 200K steps. | SA-D VD-1, `batch_assembly.py:122-238,318-455` |

---

## 2. Three findings beyond original prompt scope

**F1 — `bootstrap_model.pt` UNKNOWN-provenance v6w25.** Makefile default `BOOTSTRAP ?= checkpoints/bootstrap_model.pt`
silently loads a 21M v6w25-architecture checkpoint with 0/143 tensors matching `bootstrap_model_v6w25.pt`.
Likely §174 transfer/FT intermediate (per `transfer_v6_to_v6w25.py`, `s174_bootstrap_fix_run.sh:CKPT_FT`).
**§178 mitigation:** explicit `--checkpoint=checkpoints/bootstrap_model_v6_step20k.pt` in launch command.
**Separate hygiene task:** identify or rename `bootstrap_model.pt`. Not §178-blocking.

**F2 — `decay_steps=200_000` is canonical.** Three-way drift (live yaml 200K, memory 20K, sprint §40b 300K) —
yaml authoritative. At step 40K: corpus 65.6% of batch. Pretrain corpus DOMINATES the entire 200K run
(76% at 10K → 29% at 200K). This is the structural reason selfplay slice stays small while still
being colony-saturated. **§178 does NOT change `decay_steps`** — anti-colony human signal is the
existing protection; bot-corpus is the new lever.

**F3 — Ply-cap truncation = organic draw in outcome column.** SA-C VC-2 confirms ply-cap-hit games
write `outcome = draw_reward = -0.5`, identical to genuine draws. Only `terminal_reason` metadata
distinguishes; outcome col does not. Operator pre-commits `draw_value: -0.5 → -0.1` (BCE target
0.25 → 0.45). On its own, this REMOVES the "finish faster" pressure — counter-colony intent breaks.
**Fix: split `ply_cap_value` from `draw_value`.** Penalise truncation harder (`-0.8`) while
softening organic draws (`-0.1`). Restores resolution pressure on long colony games.

---

## 3. Batch composition math

Architecture: bot-corpus is a TOP-LEVEL slot, parallel to human-corpus + selfplay.

```
batch[256] = n_pre + n_bot + n_recent + n_uniform
```

- `n_bot = round(bot_batch_share × batch_size) = 0.15 × 256 = 38`
- `n_pre = round(w_pre(step) × (batch_size − n_bot))`
- `n_self = batch_size − n_pre − n_bot`
- `n_recent = round(n_self × recency_weight)` | `n_uniform = n_self − n_recent`

`w_pre(step) = max(0.1, 0.8 × exp(-step / decay_steps))` with decay_steps=200000.

| Step | w_pre | n_pre | n_bot | n_self | n_recent | n_uniform |
|---:|---:|---:|---:|---:|---:|---:|
| 10K | 0.761 | 166 | 38 | 52 | 39 | 13 |
| 20K | 0.724 | 158 | 38 | 60 | 45 | 15 |
| 40K | 0.655 | 143 | 38 | 75 | 56 | 19 |
| 100K | 0.485 | 106 | 38 | 112 | 84 | 28 |
| 200K | 0.100 (floor) | 30 | 38 | 188 | 141 | 47 |

### 3.1 Anti-colony force decomposition at step 40K

| Force | Batch share | Quality |
|---|---:|---|
| Human corpus (anti-colony, diverse) | 56% | INDIRECT — different distribution, doesn't directly correct colony |
| Bot corpus (direct anti-colony, anchor-loses-to-SealBot) | 15% | DIRECT policy + value correction (anchor positions → outcome −1) |
| ply_cap_value on ~10% of selfplay rows | ~3% (10% × 29%) | DIRECT value correction on long games |
| Selfplay uniform | 7% | NEUTRAL (mix colony + tactical) |
| **Total non-colony pressure** | **~81%** | mixed quality |

Colony force: recent-selfplay slice **22%**, colony-shifted by step 5K per HB-1.

**Aggregate anti-colony : colony = 81 : 22 ≈ 3.7 : 1**. Looks safe on indirect-inclusive math.

**DIRECT anti-colony only (bot + ply_cap): 18% vs colony 22% — ratio 0.82:1.** Colony slightly wins on direct gradient.

### 3.2 Honest verdict

Math is **borderline-PASS** on direct corrective signal. Differentiators vs §175 (which had only
human corpus 76% vs colony 24% = 3.2:1 aggregate but ZERO direct corrective):
- §178 has 18% DIRECT corrective force (NEW)
- ply_cap_value gives value head GROUND TRUTH that long colony ≠ win
- Bot games give policy head SHARP one-hot targets on positions where anchor's prior is wrong
- Pretrain corpus floor at 56% at step 40K means colony selfplay never gets to dominate buffer

**Why not 20-25% bot share for §178:** at 25% bot, n_pre drops to 126 at step 40K, reducing the
human-corpus anti-colony signal that's currently the bulk of total anti-colony pressure (HB-1
implication). 15% balances both levers without over-rotating either. **Reserve 20-25% for §179
if §178 colony_frac trajectory shows bot signal underdamping.**

### 3.3 §179 escalation triggers

If §178 V178-3 (colony_frac ≤50% at step 30K) FAILS, escalation order:
1. Bump `bot_batch_share` to 0.20 → direct bot:colony 0.20:0.22 = 0.91:1 (parity)
2. Neutralize game_length_weights to uniform 1.0 → colony per-row gradient drops ~2× → bot:colony ≈ 1.8:1
3. Add length-discounted value target γ=0.99 → structural fix for VA-2

Each lever is a 1-line yaml change (1) + (2) or ~15 LOC Rust (3). Pre-registered in §178 sprint log.

---

## 4. Bot pool design

### 4.1 Source A static NPZ

- **Path:** `data/bot_corpus_s178_sealbot_vs_v6.npz`
- **Schema:** identical to existing corpus NPZ (states fp16, policies fp32, outcomes fp32, weights fp32 — `dataset.py:37,61` convention)
- **Policy target:** one-hot SealBot move (`policies[t, target_idx] = 1.0`). Anchor moves also one-hot from anchor's argmax post-MCTS (records the MOVE not the visit distribution; consistent with existing bot-game replay).
- **Outcome:** game-end z ∈ {−1, +1} from cur-player POV per row.
- **Aux:** ownership=neutral (1), winning_line=zeros, chain recomputed at load time (`batch_assembly.py:184-190` existing pattern).
- **Encoding:** v6 single-window 19×19 (anchor is v6 architecture).

### 4.2 Game-generation knobs

- **Strong side:** SealBot @ 0.5s/move (matches Wave C baseline, BT-anchor).
- **Weak side:** anchor `bootstrap_model_v6.pt` with MCTS @ n_sims=200, T=0.5, dirichlet=true.
  Mirrors §175 sampled-eval setup at run start. Anchor is fresh post-pretrain — Wave C BT delta
  −62 vs SealBot at MCTS-128 → at sims=200 anchor loses ~75-85% (extrapolation; verify in smoke).
  High loss rate produces high colony-mistake fraction in anchor's plies.
- **Game count:** 700 games target (~ 60-90 plies × 700 = 42K-63K positions; matches pool size).
- **Random openings:** 4 random opening plies per game (off-canonical diversity, matches eval.yaml convention).
- **Generation cost:** ~50 sec/game × 700 = ~10 hours single-thread, ~2.5 hours 4-parallel on vast.
- **Output:** sidecar `<npz>.metadata.json` with sha256+encoding_name+n_positions+anchor_sha+sealbot_version+generation_date+anchor_path.

### 4.3 Pool sizing rationale

50K positions = ~700 games. SealBot has ~80-150 distinct opening patterns vs given anchor;
700 games × random_opening_plies covers the opening tree adequately. Larger pool (100K) marginal
benefit since SealBot vs frozen anchor has bounded position diversity. Smaller pool (25K) risks
behavioural cloning overfit on small set.

### 4.4 Refresh-on-anchor-change hook (§179 wire)

```yaml
bot_corpus:
  refresh:
    enabled: false  # §178 static
    trigger: best_model_promotion
    cooldown_steps: 25_000
    min_new_games: 200
```

`step_coordinator.py` near eval-result-handler emits regen request when `best_model_winrate ≥ 55%`
vs current bot-corpus anchor. Async regen via `tournament_validate.py` subprocess pattern.
For §178: hook present, disabled — first iteration is static.

**Aging risk note (v6.pt anchor):** since v6.pt is FRESH (zero selfplay steps), the model
will diverge from v6.pt rapidly during §178's first 5-10K steps. Bot games baked against v6.pt
remain valuable longer than bot games against v6_step20k (which is closer to where the model
will be at step 20K). v6.pt-baked bot games teach "what SealBot punishes about FRESH v6 weights"
— a stable target since v6.pt never moves. Refresh becomes relevant only once model surpasses
v6.pt SealBot WR significantly (i.e. the model can win games v6.pt can't).

---

## 5. Pair-A — `ply_cap_value` separate from `draw_value`

### 5.1 Surgery scope

- **Rust:** add `ply_cap_value: f32` to `SelfPlayRunnerConfig` (`engine/src/game_runner/config.rs:103`-area).
  Default = `draw_reward` for backward compat (when not set explicitly).
- **Branch:** `engine/src/game_runner/worker_loop/inner.rs:1069-1074` `finalize_game`:
  ```rust
  let outcome = match winner {
      Some(p) => if p as i8 == player as i8 { 1.0 } else { -1.0 },
      None => {
          if terminal_reason == TerminalReason::PlyCap {
              ply_cap_value
          } else {
              draw_reward
          }
      }
  };
  ```
- **Python wire:** `hexo_rl/selfplay/pool.py:299`-area pass `ply_cap_value=training_cfg.get("ply_cap_value", training_cfg.get("draw_value", -0.5))`.
- **Yaml:** `configs/training.yaml` add `ply_cap_value: -0.8`.

### 5.2 LOC + bench

~30 LOC Rust + 5 LOC Python. `finalize_game` runs once per game (~770 games/hr), not per MCTS sim
(~150K/s). Bench-neutral expected. **Bench gate required** to confirm.

### 5.3 Tradeoff surfaced

- `draw_value: -0.1` softens organic draws (operator pre-commit; rationale: infinite-board draws are
  legitimate "no chain emerged" outcomes, not pathology).
- `ply_cap_value: -0.8` penalises truncation hard (close to loss). Restores finish-pressure that
  `draw_value: -0.5` was implicitly providing for both classes.
- Net effect on value head: 10% organic-draw rows → BCE target 0.45 (gentler); 10% ply-cap rows →
  BCE target 0.10 (harsher). 80% win/loss rows unchanged.

---

## 6. Locked decisions

### Changing in §178
- Anchor: `bootstrap_model_v6.pt` (operator pick 2026-05-18; cleaner control vs §175)
- §177 buffer: discarded; `mixing.buffer_persist_path` re-pointed or deleted before launch
- `bot_corpus_path`: new key, points to pre-generated NPZ baked vs v6.pt
- `bot_batch_share`: 0.15
- `draw_value`: −0.5 → **−0.1** (operator pre-commit)
- `ply_cap_value`: NEW key, **−0.8** (Pair-A)
- Variant: `configs/variants/v6_botmix_s178.yaml` (NEW)

### Unchanged in §178 (locked)
- Network architecture (12 res × 128ch, GroupNorm(8), SE, 8-plane, dual-pool value head)
- Encoding (v6, registry.toml)
- MCTS (PUCT, n_sims=400, top-K=192, quiescence)
- `completed_q_values: true` (inherited from §177; SA-A verdicts hold under both paths)
- `full_search_prob: 0.5`
- Buffer augmentation (12-fold lazy)
- Eval opponents (SealBot anchor + bootstrap_anchor=v6.pt + best_checkpoint + random)
- `decay_steps: 200_000` (F2 rationale — corpus dominance IS the existing anti-colony lever)
- `recency_weight: 0.75`
- `cosine_temp + jitter` mandatory pairings (L9)
- `game_length_weights` (flagged as colony-biasing; §179 candidate if §178 modest)
- Loss function weights (all aux unchanged)
- Ply cap: 150 plies
- Bootstrap corpus: human-only v7 (UNCHANGED — bot games go in separate pool)
- Pretrained_buffer_path: `data/bootstrap_corpus_v6.npz` (same as §177)

### NOT in §178 scope
- Source B live cross-bot games (subprocess infra absent per SA-D VD-2; §179+ design)
- KrakenBot reintroduction (`enabled: false` in §178 variant)
- Bot pool refresh on promotion (hook wired, `enabled: false`)
- `bootstrap_model.pt` provenance investigation (separate hygiene prompt — CONFIRMED fresh-init v6w25 random; cleanup pending)
- Chain length planes (Q12 deferred per CLAUDE.md current phase)
- Gumbel MCTS full implementation (Phase 4.5 queue item #5)
- Length-discounted value target (§179 candidate if bot+ply_cap insufficient)
- game_length_weights neutralization (§179 candidate)

---

## 6.5 Loss function structure — what §178 changes and what it doesn't

Investigation SA-A flagged VA-1 FAIL (policy KL colony-diffuse), VA-2 FAIL (value BCE binary),
VA-3 HIGH (opp_reply colony-reinforce). These are loss-DESIGN failures, not loss-IMPLEMENTATION
bugs. §178 does NOT change the loss function structure. Rationale: every structural fix
considered requires either Rust path changes more invasive than Pair-A or buffer schema changes
that block atomic landing with bot-corpus surgery.

**Considered and rejected for §178:**

| Option | Mechanism | Why rejected |
|---|---|---|
| Length-discounted value target `z' = γ^game_length × z` | Penalises long colony wins on value head | More invasive than Pair-A; per-game length stamping needed in buffer; reverting needs careful test. Pair-A `ply_cap_value` captures most of the benefit by separating truncations. **§179 candidate.** |
| Tactical-quality auxiliary head (new) | Predicts S0/W3 forced-win/lose flags | New buffer column + retrain from scratch. Existing threat head partially covers it; better fix is feeding threat-head TARGET on colony positions (currently empty when terminal_reason=colony) — still a buffer schema change. |
| `completed_q_values: false` switch | Move from Gumbel-CQ KL to visit-count CE | Under colony, both targets ≈ prior. Empirical Δ likely 5-10%; not the lever §178 needs. |
| Per-position policy weight by entropy | Up-weights hard positions | Complicates batch assembly; conflicts with `is_full_search` masking. Bot games already provide this — sharp targets on positions where anchor's prior is weak. |
| Detach value gradient from improved-policy target | Breaks value→policy feedback loop | In Rust there's no autograd; the "detach" semantically collapses to switching to visit-count CE (option above). Skip. |

**§179 first lever (if §178 colony reduction modest):** length-discounted value target with γ=0.99.
Surgical Rust change in `inner.rs` `finalize_game` (compute z' from terminal_ply at game-end). ~15 LOC.
Tests the structural fix for VA-2 directly.

---

## 7. Pre-registered §178 verdicts

| ID | Hypothesis | PASS | FAIL | NULL |
|---|---|---|---|---|
| V178-1 | §178 SealBot WR at step 10K > §177 step-10K WR (was 2%) | ≥5% | ≤2% | 2-5% |
| V178-2 | §178 SealBot WR at step 30K > 0% (was 0/100) | wins ≥1, n=100 | 0/100 | — |
| V178-3 | §178 colony_frac in bootstrap_anchor wins at step 30K < §177 step-30K (~64%) | ≤50% | ≥70% | 50-70% |
| V178-4 | §178 non-monotone-declining SealBot WR (at least one round WR ≥ prev) | observed | strict decline all 4 rounds | — |
| V178-5 | G4 value_fc2_weight_abs_max enters band [0.154, 0.462] by step 20K | in-band | below-band all rounds | — |
| V178-6 | Bot-pool gradient non-negligible: avg policy-loss-per-batch on `is_bot` rows > 0.5 nat through step 30K (training jsonl probe) | observed | bot-rows loss ≈ corpus rows | — |
| V178-7 | Draw rate stable/down vs §177 step-30K (~10%); ply-cap-truncation rate down (10.5% → ?) | both stable/down | either rises ≥5pp | — |
| V178-8 | Value head ply-cap-position prediction tightens around new `ply_cap_value=-0.8` target by step 15K | mean v_pred(ply≥140) ∈ [-0.9, -0.7] | drifts toward 0 | — |

Freeze at variant-yaml commit SHA per L13 + A4 do-not #9.

---

## 8. Risk register

| # | Risk | Likelihood | Mitigation | Source |
|---|---|---|---|---|
| 1 | `draw_value=-0.1` removes ply-cap finish-pressure if Pair-A not landed atomically | HIGH if Pair-A separated | Land Pair-A in same launch commit as variant yaml; reject launch if `ply_cap_value` not in config | F3, SA-C VC-2 |
| 2 | Bot pool 50K = ~700 games → behavioural cloning overfit on small set | MEDIUM | Augmentation 12-fold (already on for pretrained_buffer); restart bot pool gen with larger n_games if overfit signature in policy entropy probes | SA-D, dataset.py:61 |
| 3 | SealBot one-hot argmax target flattens model entropy too much | MEDIUM | `entropy_reg_weight=0.01` retained; monitor policy entropy on bot-row vs corpus-row vs selfplay-row | SA-A entropy section |
| 4 | Anchor `bootstrap_model_v6` frozen → bot signal weakens as model surpasses v6.pt strength | MEDIUM | Refresh hook wired; flip `enabled: true` in §179; for §178 accept and watch colony_frac trajectory. v6.pt fresh-anchor signal stays valid LONGER than v6_step20k would (frozen target, no selfplay drift in anchor itself) | L8 head-drift |
| 5 | Bot batch_share=15% undersized vs colony 22% recent-selfplay (HB-2 CONDITIONAL) | MEDIUM | Pair-A `ply_cap_value=-0.8` adds value-head pressure that bot share alone cannot deliver | SA-B HB-2 |
| 6 | §175 buffer-persist file at `checkpoints/replay_buffer.bin` carries §175 colony positions into §178 | HIGH if not cleared | Pre-launch: delete or rename `checkpoints/replay_buffer.bin`; verify clean start in trainer ckpt-load log | mixing.buffer_persist=true, training.yaml:140 |
| 7 | `bootstrap_model.pt` (no suffix) is FRESH-INIT v6w25 random per provenance investigation; Makefile default loads it if launch command sloppy | LOW (if explicit `--checkpoint`) | §178 launch script asserts `--checkpoint=...v6.pt`; abort if any other path. Separate hygiene commit pending: repoint Makefile/anchor.py/opponent_runners | F1, SA-E, reports/bootstrap_model_pt_provenance.md |
| 8 | `data/bootstrap_corpus_v6.npz` vast-only; §178 launch fails if not present | HIGH on vast | Pre-flight: scp from current vast §177 dir or regen with `scripts/generate_v6_corpus.py` on vast | SA-E vast-only finding |
| 9 | Game length weights `[0.15, 0.50, 1.0]` bias gradient toward long colony games | OBSERVED | Documented; defer to §179 if bot+ply_cap_value insufficient | training.yaml:120-124 |
| 10 | `opp_reply` head trains on one-hot SealBot move (target = SealBot's next move) on bot rows; semantically muddled vs MCTS-distribution targets on selfplay rows | LOW | weight=0.15 bounds impact; monitor opp_reply loss on bot vs selfplay rows | losses.py:86-107 |
| 11 | Ply-cap rate increases above 10.5% baseline due to changed value-head landscape | MEDIUM | V178-7 verdict catches this; soft-abort if `pct_at_cap` rises ≥5pp without WR recovery | §177 trajectory |
| 12 | KrakenBot omission removes opponent diversity per §176 V6 (colony opponent-coupled) | LOW for §178 (single-anchor static pool already opponent-coupled) | Reintroduce in §179 Source B if §178 colony reduction modest | §176 V6 PASS |

---

## 9. Implementation task breakdown

| # | Task | Files | LOC est | Bench | Commit |
|---|---|---|---|---|---|
| T1 | Bot game NPZ generator script | `scripts/generate_bot_corpus.py` (NEW), `Makefile` (add `corpus.bot`) | ~250 | N | feat(bootstrap): bot-game NPZ generator (SealBot vs anchor) §178 |
| T2 | Rust `ply_cap_value` knob + branch | `engine/src/game_runner/{config.rs, worker_loop/{params,inner}.rs}` | ~30 | **Y** | feat(engine): split ply_cap_value from draw_reward §178 |
| T3 | Python wire ply_cap_value | `hexo_rl/selfplay/pool.py`, `configs/training.yaml` | ~10 | N | feat(training): wire ply_cap_value config knob §178 |
| T4 | batch_assembly + load_pretrained_buffer extend for bot slot | `hexo_rl/training/batch_assembly.py` | ~80 | **Y** | feat(training): bot-corpus second-buffer slot §178 |
| T5 | step_coordinator n_pre/n_bot/n_self split | `hexo_rl/training/step_coordinator.py` | ~30 | N | feat(training): n_pre/n_bot/n_self split §178 |
| T6 | Variant yaml | `configs/variants/v6_botmix_s178.yaml` (NEW) | ~100 | N | feat(config): v6_botmix_s178 variant §178 |
| T7 | Refresh hook (disabled) | `hexo_rl/training/step_coordinator.py` (continuation of T5) | ~40 | N | feat(training): bot-corpus refresh hook (disabled) §178 |
| T8 | §177 close-out forensics + sprint log entry | `docs/07_PHASE4_SPRINT_LOG.md`, archive paths | ~80 | N | docs(sprint): §177 close-out + verdicts §178 |
| T9 | §178 launch entry + L24+ candidates | `docs/07_PHASE4_SPRINT_LOG.md` | ~50 | N | docs(sprint): §178 launch + L24-L26 §178 |
| T10 | INV pin for `ply_cap_value` distinct outcome path | `engine/tests/inv26_ply_cap_value.rs` (NEW), `tests/test_inv26_*.py` | ~60 | N | test(inv): INV26 ply_cap_value outcome path §178 |

Total ~730 LOC across ~10 commits. Two bench gates (T2 Rust hot-path adj, T4 batch assembly).

Dispatch pattern: PREP → IMPL Batch A (T2+T3+T6+T10 Rust+Python+yaml+INV; sequential same files) +
IMPL Batch B (T1 isolated script) + IMPL Batch C (T4+T5+T7 training path; sequential same module
chain) → BENCH-GATE (T2+T4 changed perf-sensitive paths) → REVIEW N parallel → AGGREGATION.
Worktree-parallel viable: Batch B is fully disjoint from A and C.

---

## 10. Open hygiene items (NOT §178-blocking)

| # | Item | Path | Owner |
|---|---|---|---|
| H1 | `bootstrap_model.pt` UNKNOWN-provenance v6w25 — identify or rename | `checkpoints/bootstrap_model.pt` | Separate Claude Code investigation prompt |
| H2 | `data/bootstrap_corpus_v6.npz` vast-only — scp to local or regenerate | `data/` | Operator pre-flight |
| H3 | §177 vast tmux liveness — SSH check before §178 prep | vast REMOTE_HOST:REMOTE_PORT | Operator pre-flight |
| H4 | `game_length_weights` colony-bias — flag for §179 review | `configs/training.yaml:118-124` | §179 design |
| H5 | `bootstrap_model.pt` Makefile default — replace with explicit error | `Makefile:1` | §178 close-out commit |

---

## 11. §178 launch command (pre-flight contract)

```bash
# vast pre-flight checks (operator-mediated):
# 1. tmux kill-session -t s177 if still alive
# 2. scp/regen data/bootstrap_corpus_v6.npz to ${WORKDIR}/data/
# 3. rm checkpoints/replay_buffer.bin OR redirect mixing.buffer_persist_path
# 4. generate bot corpus on vast (anchor = v6.pt, NOT v6_step20k.pt):
#    make corpus.bot ANCHOR=checkpoints/bootstrap_model_v6.pt \
#        N_GAMES=700 OUT=data/bot_corpus_s178_sealbot_vs_v6.npz
# 5. launch:
python scripts/train.py \
  --checkpoint checkpoints/bootstrap_model_v6.pt \
  --variant v6_botmix_s178 \
  --iterations 100000

# launch script must verify:
# - --checkpoint == bootstrap_model_v6.pt (NOT bootstrap_model.pt = random init Makefile default)
# - configs/variants/v6_botmix_s178.yaml.ply_cap_value is set
# - data/bot_corpus_s178_sealbot_vs_v6.npz exists with valid sidecar (anchor_sha matches v6.pt SHA)
# - checkpoints/replay_buffer.bin does not exist (clean start)
```

---

## 12. Anchor switch implications (v6.pt vs v6_step20k.pt)

§178 uses v6.pt per operator 2026-05-18. Key implications:

**Pro:**
- Direct A/B vs §175 (same anchor, different recipe → mechanism delta isolated)
- No selfplay-state leakage from 20K steps of §175 self-training
- Bot games baked vs a FROZEN target — v6.pt never moves, signal stays valid longer
- Cleaner mechanism-validation experiment

**Con:**
- Initial WR slightly lower than v6_step20k peak (extrapolated ~18% vs measured 18%)
- Less data on this exact anchor + recipe combination (v6_step20k had §177 trajectory)
- Model diverges rapidly from v6.pt in first 5-10K steps; bot game positions may not match model's eventual playstyle as closely as bot vs v6_step20k would

**Net:** experimental cleanliness wins. The diverge-from-anchor concern is exactly what the bot pool refresh hook is designed for (§179 trigger).

---

## 13. Forward pointer

§178 closes when:
- 100K training steps completed OR soft-abort fires (per V178 verdicts)
- All V178-1..8 verdicts assigned
- Sprint log §178 entry written with mechanism lessons L24+
- §179 design queued: refresh hook activation, Source B subprocess infra design,
  game_length_weights rebalance + length-discounted value target if §178 outcome shows
  insufficient colony reduction
