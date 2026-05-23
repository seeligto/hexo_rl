# §S181-AUDIT Track D — data-pipeline regression audit

**Wave:** §S181-AUDIT Track D. **Date:** 2026-05-23. **Mode:**
document-only; no code/config/training touched. **Branch context:** master
`b1d98f0` post §S181-AUDIT close.

**Purpose.** Walk codebase / config / data transitions from §150 (v7full
anchor, working) through §S180b (4th colony reproduction) to §S181 close.
Identify smoking-gun candidate changes that plausibly carry the
training-loop value-head discrimination collapse (L42). Rank by
joint (likelihood × impact). Cite SHA or §-number on every claim. No
proposals that re-litigate the Falsified-Hypotheses Register.

**Companion artifacts.** Track A `audit/structural/track_a/A6_aggregation.md`
(no single corpus source dominant, L49). FU-1
`audit/structural/05_fu1_value_spread_ladder.md`. FU-1.5
`audit/structural/06_fu1_5_finer_ladder.md`. A3 bank confound
`audit/structural/track_a/A3_h_bank_confound.json`.

---

## 1. Transition table

Rows are training campaigns; columns are pipeline knobs. Cell cites the
SHA / § that introduced the value. Blank cell = unchanged from prior column.
"—" = field N/A.

### 1.1 Anchor + encoding + corpus

| field | §150 v7full (anchor only) | §172 v7full sustained | §175 v6 sustained | §S178 (= §S179 launch) | §S180b 3-knob | §S181 close |
|---|---|---|---|---|---|---|
| anchor file | `bootstrap_model_v7full.pt` SHA `29306533…` (§150 `1f822ae`) | same | `bootstrap_model_v7full.pt` (per `v6_sustained.yaml`:13 §175 `6e13768`) → run actually used `bootstrap_model.pt` v6 default per Track A finding | `bootstrap_model_v6.pt` SHA `7ab77d2c…` (§S178 5dbdf88 + `c1173a8` H1 quarantine) | same | same |
| anchor encoding | v7full (8-plane, 19×19, K=1) §150 | v7full | v6 (`v6_sustained.yaml` per §175 `6e13768`) | v6 | v6 | v6 |
| bootstrap-corpus path | `data/bootstrap_corpus.npz` (v7full Elo-weighted §148 `4cc8791`) | same | `data/bootstrap_corpus_v6.npz` (explicit; `ef268d2` §175) | same `bootstrap_corpus_v6.npz` (T1 corpus 100% human) | same | same |
| corpus regeneration | v7 human-only Elo-weighted (§148) | same | v6-encoding regen `ee7f88c` §174 (383k positions human-only) | same | same | same |
| encoding registry | n/a (pre-§172) | landed §172 `138768a..8ba6060` | same | same | same | same |
| bot-corpus | none | none | none | `data/bot_corpus_s178_sealbot_vs_v6.npz` 700g static `5b9ad09` §S178 | same | same |
| bot_batch_share | 0 | 0 | 0 | 0.15 (§S178 `22597fc`) | **0.30** (§S180b `3146144`) | — |

### 1.2 Reward shaping + value targets

| field | §150 v7full | §172 v7full sustained | §175 v6 sustained | §S178 | §S180b | §S181 close |
|---|---|---|---|---|---|---|
| draw_reward / draw_value | -0.5 (§40, pre-§148) | -0.5 | -0.5 | **-0.1** (§S178 `5dbdf88`) | -0.1 | -0.1 (training.yaml current `03425de`) |
| ply_cap_value | == draw_value (no split) | == | == | split + revised **0.0** (§S178 T12 `1b47eb1` + `e3ede86`; engine split `b84006d`) | 0.0 | 0.0 default training.yaml |
| game_length_weights | `[0.15, 0.50]` @ `[10,25]` (training.yaml legacy) | same | same | same `[0.15, 0.50]` | **uniform [1.0, 1.0]** (§S180b `3146144` K2) | uniform |
| entropy_reg_weight | 0.01 (legacy) | 0.01 | 0.01 | 0.01 | 0.01 | **0.005** (`03425de` §S181 PR-B) |
| weight_decay | 0.0001 (training.yaml) | 0.0001 | 0.0001 | 0.0001 | 0.0001 | 0.0001 + param-group split (`c2a0f31` §S181 PR-B) |
| eta_min | 5e-5 anchor pretrain (§150 `1f822ae`) | 2e-4 (v7mw/vast.yaml era) | 2e-4 (`v6_sustained.yaml` cosine over 100K) | 2e-4 | 2e-4 (variant overrides) | **5e-4** default `a43d5eb` §S181 PR-B (variant overrides still 2e-4) |

### 1.3 MCTS / selfplay

| field | §150 | §172 v7full sustained | §175 | §S178 | §S180b | §S181 close |
|---|---|---|---|---|---|---|
| n_simulations | 800 (selfplay legacy) | 400 | 400 (per `v6_sustained.yaml`) | 400 | 400 | 400 |
| c_visit / c_scale | 50.0 / 1.0 default | same | 50.0 / 1.0 | 50.0 / 1.0 | 50.0 / 1.0 | same |
| dirichlet | active root-only since §73 `71d7e6e` | same | same | same | same | same |
| completed_q_values | true | true | true | true (`v6_sustained.yaml` §175) | true (§S180a flipped false → FAILED; §S180b restored) | true |
| temperature schedule | cosine on @ §156 default | cosine on | cosine OFF in `v6_sustained.yaml` (§156 R12 / L9; folded to base @ `96337c4`) | cosine OFF + jitter ON (`96337c4`) | same | same |
| legal_move_radius_jitter | (n/a pre-§156) | on (§157) | on | on | on | on |
| playout_cap | n_sims_quick=100, n_sims_full=600, full_search_prob=0.25 (§100) | 0.25 | **0.5** (`v6_sustained.yaml` §175 inheritance) | 0.5 | 0.5 | 0.5 |
| max_game_moves | 100 (§144 `e4c8b29`) → 150 (§144 calibrated 150) | 150 | 150 (`v6_sustained.yaml`) | 150 | 150 | 150 |
| random_opening_plies | 1 default | 0 (vast.yaml) | 0 (`v6_sustained.yaml`) | 1 (variant override) | 1 | 1 |
| rotation_enabled | true (§121 e37e977 per-game rotation) | true | true | true | true | true |

### 1.4 Augmentation / buffer / arch

| field | §150 | §172 v7full sustained | §175 | §S178 | §S180b | §S181 close |
|---|---|---|---|---|---|---|
| augment (12-fold) | true (training.yaml) | true | true | true | true | true |
| buffer_schedule | `[250k → 500k @ 300k → 1M @ 1M]` (training.yaml legacy) | same | same `v6_sustained.yaml` | same | same | same |
| recency_weight | 0.75 (training.yaml) | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 |
| pretrain_mix decay_steps | 200k | 200k | 200k | 200k | 200k | 200k |
| min/initial pretrained_weight | 0.1 / 0.8 | same | same | same | same | same |
| aux_chain_weight | 1.0 (§95) | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| aux_opp_reply_weight | 0.15 (§85) | 0.15 | 0.15 | 0.15 | 0.15 | 0.15 |
| threat_weight | 0.1 (§91 / §S178 F-fix-1 `1aa0c8f`) | 0.1 | 0.1 | 0.1 + colony-fix target | 0.1 | 0.1 |
| ownership_weight | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 |
| input planes | 8 (§131 `1bf20b5`) | 8 | 8 | 8 | 8 | 8 |
| trunk arch | 128×12 + SE blocks + GN(8) (§99) | same | same | same | same | same |
| value head | dual-pool v_max + v_avg (per §S181-T2 `42a47ed`) | same | same | same | same | same |
| FxHashSet child order | pre-§S182 raw insert order | same | same | same | same | canonical `(prior desc, flat asc)` `de149e6` §S182 follow-on (behaviourally inert per 5 tests) |

### 1.5 Eval cadence + opponents

| field | §150 | §172 v7full sustained | §175 | §S178 | §S180b | §S181 |
|---|---|---|---|---|---|---|
| eval_interval | 5000 (training.yaml base) | 5000 → mid-run 5000→**10000** (§172 B2 `e90e49d`) | 10000 (`v6_sustained.yaml` §175 `6e13768`) | 5000 launch → **10000** §S179 relaunch `98296f9` | 10000 | 10000 |
| eval_temperature | 0.5 (eval.yaml default) | 0.5 | 0.5 (§175 continuity L21) | 0.5 | 0.5 | 0.5 |
| eval_random_opening_plies | 4 → **0** (`cda9933` §174) | 0 | 0 | 0 | 0 | 0 |
| SealBot stride | 4 | 4 | 4 | 4 → **1** (`3b708c3` + `d9b3a1d` per §S179 relaunch) | 1 | 1 |
| n_games SealBot | 100 (eval.yaml) | 20 (§172 B2) | 100 (`v6_sustained.yaml`) | 100 | 100 | 100 |
| bootstrap_floor.enabled | (off) | off | **on** min_winrate 0.45 (§157 / `v6_sustained.yaml`) | on | on | on |

### 1.6 Outcomes

| field | §150 | §172 v7full sustained | §175 | §S178 / §S179 | §S180a | §S180b | §S181 close |
|---|---|---|---|---|---|---|---|
| SealBot WR (sampled, peak) | 17.4 % anchor n=500 (`§150`) | 0.05 stalled (`§172 B2`) | 18 % @ 20K → 4 % @ 70K | 12 % @ 30K → 2-4 % @ 40-60K (`§S179`) | 8 % @ 10K → 7 % @ 20K (`§S180a`) | 19 % @ 40K → **0 %** @ 50K (`§S180b`) | n/a (no run) |
| anchor↑/sealbot↓ divergence | — | — | yes | yes (L34 confirm 2) | not capture, just weak | yes @ 50K | — |
| colony@sealbot | — | — | — | 91% @ 20K, 100% @ ≥40K | — | 0-12 % throughout (visible-metric crushed) | — |
| V_spread @ run end | +0.617 anchor | n/a | n/a | n/a | n/a | **−0.016** @ 50K (T3 bank); FU-1.5 alt-bank +0.093 @ 20k | — |

---

## 2. Per-transition diff narrative

### 2.1 §150 → §172 v7full sustained

**What changed.** §150 was a one-shot pretrain promotion (no sustained
selfplay). §172 ran the first 30K v7full **sustained** on the encoding
registry branch (`§172 B2`, mid-run `e90e49d`). MCTS, augment, draw_reward,
ply_cap (no split), arch all identical. The new surface is selfplay
itself + bootstrap_floor gating + the encoding-registry plumbing landed
across `138768a..8ba6060` (§172 A3-A10).

**Why.** Pre-launch design `docs/designs/encoding_registry_design.md`
made registry single-source-of-truth to unblock §173 α multi-window.

**Mechanism for plausible attractor contribution.** Low — pure plumbing.
v7full sustained's failure was characterized as "encoder-specific gap, not
value-drift pathology" (§172 B verdict, line 1167) with argmax_n DRIFT
detector 0/20 — at this point the colony attractor was not yet visible
*at sampled-eval* because n=20 was too noisy. **§172 is the
predecessor pattern not the cause.**

**Cite.** Sprint log §172 lines 1151-1171; commit `e90e49d`.

### 2.2 §172 → §175 v6 sustained (FIRST sampled-eval crash)

**What changed.**
1. Anchor encoding: v7full → v6 sustained (selfplay path `v6` 19×19 K=1).
2. SealBot eval n=20 → n=100 (`v6_sustained.yaml` `6e13768`).
3. eval_interval 5000 → 10000 (matched cosine denominator math).
4. Bootstrap_floor wired ON @ min_winrate 0.45 (§157 propagated to base
   via `96337c4`).
5. playout_cap full_search_prob 0.25 → 0.5 (`v6_sustained.yaml`).
6. Cosine-temperature disabled (`v6_sustained.yaml`; matched §156 R12 /
   L9). Later folded to base default `96337c4` 2026-05-18.
7. eval_random_opening_plies 4 → 0 (`cda9933` §174 — Track 2 audit).

**Why.** §174 closed v6w25 sustained as DEAD bootstrap (3 recipes fail
viability); §175 escalated to v6 sustained from `bootstrap_model_v7full.pt`
per recipe design. Encoding-aware checkpoint loader fix `d763fa8` was
landed mid-§175 (`normalize_encoding_name` + Trainer mutation).

**Mechanism for plausible attractor contribution.** This is the
transition where the colony attractor became visible. Candidates folded
in the wash:
- v7full anchor → v6 selfplay encoding (model trained on v7full weights but
  selfplay K=1 perception is identical wire format) — likely NOT the lever
  (T2 confirms architecture permits but does not compel).
- playout_cap full_search_prob 0.25→0.5 doubles fraction of full-search
  positions in the buffer. Possible policy-target sharpness amplifier.
- eval_random_opening_plies 4→0 removes free positional diversity that
  masked weaknesses in §168 (per `1220` sprint-log note).
- bootstrap_floor on prevents step-0 to step-50K from looking
  catastrophic at eval gate, *delaying detection* (not causing the
  attractor) — but it does not create it.

L42 names this transition the first one where the value-head
discrimination collapse was observable.

**Cite.** §175 entries; `6e13768`, `ef268d2`, `cda9933`, `d763fa8`,
`96337c4`.

### 2.3 §175 → §S178 (bot-mix + ply_cap split + draw_value softening)

**What changed.**
1. Anchor: `bootstrap_model_v7full.pt` → **`bootstrap_model_v6.pt`** (`5dbdf88`).
2. Bot-corpus slot at `bot_batch_share=0.15` introduced (`5b9ad09` +
   `9dbfb69`).
3. ply_cap_value split from draw_reward as new outcome arm (Rust
   `b84006d`; Python `062dc62`; INV26 `bbacbae`).
4. ply_cap_value revised −0.5 → **0.0** (T12 `1b47eb1` + `e3ede86`).
5. draw_value −0.5 → **−0.1** (operator pre-commit on `5dbdf88` design).
6. F-fix-1 threat-target colony fix (`1aa0c8f` T11) — `find_winning_line`
   now scans all stones in `player_wins` fallback so threat-head target
   non-empty under HTT 2-moves-per-turn off-line second-move.
7. SealBot stride 4 → 1 (`3b708c3` + `d9b3a1d`) + eval_interval 5000
   → 10000 in §S179 relaunch (`98296f9`).

**Why.** §175 + §177 jointly falsified "recipe alone" hypothesis on two
different anchors (L24). §S178 design `docs/designs/S178_design.md` §3.1
predicted DIRECT-corrective : recent-selfplay-colony force ratio of
**0.82:1 (borderline)** at `bot_batch_share=0.15`.

**Mechanism for plausible attractor contribution.**
- **draw_value −0.5 → −0.1.** Halves the magnitude of every draw target.
  In a selfplay regime with mounting truncation rate, this halves the
  push-back signal on draw-shaped colony stalemates. PLAUSIBLE
  contribution — INCONCLUSIVE.
- **ply_cap_value 0.0.** "Literature-canonical" but neutralizes the
  outcome target on the truncation arm. Any colony game that reaches
  ply 150 now contributes a 0.5 BCE target (neutral) instead of -0.25
  (penalty). PLAUSIBLE — INCONCLUSIVE.
- **Bot corpus + ply_cap split together = ≈0.82:1 borderline force
  ratio (S178_design §3.1).** §S179 close confirmed: the mechanism bought
  ~one extra promotion vs §175 trajectory and did not escape (L35).
- **F-fix-1 (`1aa0c8f`).** Plausibly neutral or anti-colony; expands
  threat-target coverage. Unlikely to cause attractor.
- **Anchor swap v7full → v6.pt.** v6.pt itself was characterized as the
  §175 anchor (per `1236`); switching from a v7full-recipe-trained
  anchor to a v6-recipe-trained anchor changes the starting V_spread
  T3-bank value (per FU-1: +0.617 at v6.pt → expectation that v7full's
  separation is even higher pre-collapse) — but L34 third confirming
  instance proved colony attractor reproduces under either anchor.

**Cite.** §S178 sprint-log lines 2124-2196; commits cited inline.

### 2.4 §S178 / §S179 → §S180a → §S180b

**§S179 → §S180a.** Single-knob: `completed_q_values: true → false`
(`6f08042`). FAILED: not colony capture, just weaker learning signal
(`9df6e1c`). CQV ruled out as colony lever. L37 captured the empirical
finding; L43 later inverted its proposed mechanism.

**§S180a → §S180b.** Three-knob escalation (`3146144`):
1. CQV restored true (KNOB 3 = no-op vs §S179).
2. **bot_batch_share 0.15 → 0.30** (KNOB 1) — force ratio 0.82:1 →
   1.64:1 per S178_design §3.1.
3. **game_length_weights neutralized** `[0.15, 0.50] → [1.0, 1.0]` (KNOB 2).

**Why.** §S179 + §S180a established 2 baseline arms with 4 unranked
suspects; combine 3 cheapest unfired levers (L36 per `46fa489`).

**Mechanism for plausible attractor contribution.** §S180b crashed at
50K with wr_sealbot 0 % but **every visible colony metric crushed**:
self-play colony 0.04 %, colony@sealbot 0-12 %. L38: "config-level
anti-colony surface area exhausted". L42 measured the actual mechanism:
**value-spread +0.617 (anchor) → −0.016 (§S180b step-50k)** = value-head
discrimination collapse.

The 3-knob delta from §S179 did *not* cause the attractor — every arm
back to §175 reproduces it. It exposed that the attractor lives in a
channel none of the YAML knobs touch (L38).

**Cite.** §S180a `9df6e1c`; §S180b `46fa489`; variant `3146144`.

### 2.5 §S180b → §S181 close (research wave, no training)

**What changed.** No training run. Inspection-only 4-track structural
diagnosis on `phase4.5/s181_structural_research`. Subsequent commits
landed:
- value_spread canary `879bcc8` (PR-A).
- FU-1 + FU-1.5 ladder probes (`c41d76c`, `9cfadd7`).
- §S181-AUDIT Track A (5 source-decomposition subtasks) +
  PR-B (3 hygiene changes).

**Why.** §S181 verdict (`f51dea8` audit aggregation): training-loop
value-head discrimination collapse. T1 falsified bootstrap+corpus
bias (H1); T3 falsified MCTS dynamics (H5); T2 confirmed architecture
permits but does not compel (H7); T4 confirmed metrics blind (H3/H4).

**PR-B hygiene changes (`c2a0f31`, `a43d5eb`, `03425de`).** Mechanism-
aligned with L47 but explicitly NOT a colony-attractor fix:
- Param-group split for AdamW: standard nanoGPT/KataGo pattern;
  orthogonal.
- `eta_min` 2e-4 → 5e-4: prevent loss-of-plasticity at late-run when
  attractor manifests (KataGo precedent: never < 0.5× peak).
- `entropy_reg_weight` 0.01 → 0.005: counter-pressure without
  over-penalising confident tactical moves.

**Mechanism for plausible attractor contribution.** PR-B is the only
mechanism-aligned change so far on master; it is too small to be
load-bearing alone (per L47 the loop overrides architecture by
≥0.5 V_spread per 1000 steps). PR-B is hygiene to clean the baseline
before the loop-side intervention.

**Cite.** §S181 sprint-log lines 2595-3064; commits cited inline.

---

## 3. Cross-reference with V_spread + SealBot trajectory

V_spread on the §S180b ladder (FU-1 + FU-1.5 + A3 alt bank). SealBot WR
sampled at T=0.5 n=100 (§175 continuity per L21). Anchor V_spread =
v6.pt forward on bank.

### 3.1 Ladder

| anchor | step | run | V_spread (T3 bank) | V_spread (alt bank) | SealBot WR (sampled T=0.5) | source |
|---|---:|---|---:|---:|---:|---|
| `bootstrap_model_v6.pt` | 0 | (probe) | **+0.6173** | +0.2119 | 18 % @ §175 step-20K (`297e0ce0`) | FU-1 `05_*.md`; A3 ladder `A3_h_bank_confound.json`; §176 Gate 3 |
| §S180b ladder | 10 000 | `fd9ea56e` | +0.2601 | n/a | 11 % | FU-1 §2; §S180b sprint-log L2531 |
| §S180b ladder | 20 000 | `fd9ea56e` | −0.1103 | n/a | 7 % | FU-1 §2 |
| §S180b ladder | 30 000 | `fd9ea56e` | +0.1395 | n/a | 12 % | FU-1 §2 |
| §S180b ladder | 40 000 | `fd9ea56e` | −0.0506 | n/a | 19 % | FU-1 §2 |
| §S180b ladder | 50 000 | `fd9ea56e` | **−0.0159** | n/a | **0 %** | FU-1 §2; §S180b 50K L2535 |
| §S180b ladder | 53 500 | `fd9ea56e` | +0.0990 | n/a | n/a | FU-1 §2 |
| FU-1.5 re-run | 2 000 | (FU-1.5) | +0.1752 | +0.1998 | n/a | FU-1.5 §3 + A3 ladder |
| FU-1.5 re-run | 4 000 | (FU-1.5) | −0.1179 | +0.2581 | n/a | FU-1.5; A3 |
| FU-1.5 re-run | 10 000 | (FU-1.5) | +0.3901 | +0.1703 | 9 % (§176-cross-check ±2pp of §S180b) | FU-1.5 §"§S180b cross-check" |
| FU-1.5 re-run | 14 000 | (FU-1.5) | **+0.5226** | +0.2462 | n/a | FU-1.5 — peak post-onset oscillation |
| FU-1.5 re-run | 20 000 | (FU-1.5) | +0.1075 | +0.0930 | n/a | FU-1.5; A3 |
| §150 anchor (reference) | 0 | v7full | (not measured on T3 bank with this probe) | n/a | 17.4 % n=500 | §150 |
| §175 step-20K (promoted) | 20 000 | `c7e74d…` | (not measured) | n/a | 18.0 % [11.7, 26.7] | §176 Gate 3 |
| §175 step-70K | 70 000 | `c7e74d…` | (not measured) | n/a | 4.0 % | §175 close; L22 |

### 3.2 Observations

- **Anchor V_spread on T3 bank = +0.6173 (FU-1).** Alt-bank corpus-
  derived = +0.2119, ~34% of T3 (`A3_h_bank_confound.json`). L48
  revises L47's ≥1.0/1000-step claim to ~+0.33/1000 steps (alt-bank).
- **Front-loaded collapse.** FU-1.5 single-interval 0→2k loses
  **−0.4421 = 86.7 %** of the trajectory total. Cliff is *between* step
  0 and step 2k; step 4k already negative. L44.
- **Post-onset behaviour is reproducible-near-zero but path-divergent.**
  §S180b reached permanent flat-dead band post-20K; FU-1.5 oscillates
  with peak +0.523 @ 14k. Both 20K endpoints near zero. L46.
- **SealBot WR + V_spread.** SealBot WR is decoupled from V_spread at
  10K-40K and only crashes after value-head discrimination is sustained
  near-zero for ~30k steps (§S180b 50K crash). V_spread leads SealBot
  WR by ~30K steps.

---

## 4. Smoking-gun candidates (rank-ordered)

Joint (likelihood × impact). Each candidate is a CHANGE that survived
all prior falsification (L18 / L47 + L49 + §S181-T2 / T3 confine the
remaining surface to training-loop / value-target / buffer-side levers).

### Candidate 1 — Bot-corpus value-target imprint at small share

**Mechanism.** `bot_batch_share=0.15` on a static SealBot-vs-anchor
corpus is enough to imprint a colony-favouring direction on the
**value-target distribution** the trainer sees. Track A A1 measured
bot-corpus position asymmetry +0.078 vs anchor +0.617 (8× smaller on
T3-bank scale, ~+0.014/step upper-bound contribution to V_spread).
Coupled to the L37/L43 finding (colony positions yield
LOWER-entropy MCTS targets — sharp-and-wrong CE gradient), the bot
corpus's modest position-level pull is amplified at gradient time. The
static corpus also goes stale relative to the model (no refresh hook
fires) — exactly the L34 anchor↑/sealbot↓ divergence channel.

**Falsifiability.** Track B (planned) — short instrumented run logging
per-sample gradient magnitude bucketed by colony / extension / neither
on the live buffer. If bot rows dominate the colony gradient at
disproportional weight vs their 15% mass, candidate confirmed.
Alternatively: bot-corpus refresh hook A/B at fixed share (already
infrastructure-landed; see `27849e8` + design `98a84a8`).

**Cite.** §S178 `5b9ad09`; Track A `A1_h_bot_corpus_position_bias.md`;
L34, L37, L43.

**Cost.** Track B short instrumented run = ~5k steps on vast 5080
(~1 hour wall). PRIMARY recommendation per A6 routing.

---

### Candidate 2 — Pretrain-corpus colony fraction × `recency_weight=0.75` cross

**Mechanism.** A5 measured `bootstrap_corpus_v6.npz` colony fraction
**31.2 %**, asymmetry **+0.157** — ~25% of the anchor V_spread direction
in T3-bank units. Pretrain mass enters the batch via the static
`pretrain_weight` decay from 0.8 → 0.1 over decay_steps=200K. **Crucially**,
`recency_weight: 0.75` sources 75 % of every batch from the recent-position
ring buffer (training.yaml:100); the recent ring fills with selfplay
positions whose value targets are increasingly colony-biased once the
attractor activates. The static pretrain corpus then has a *modest*
counter-pressure but its colony-positive direction (per A5) means the
~25% of the batch sourced from pretrain *also* pulls in the colony
direction during the front-loaded window (0-2k). The combined
effect creates the V_spread cliff FU-1.5 measured.

**Falsifiability.** Per-class buffer sampling weight (PSW) experiment
on `recency_weight × pretrain mix`. Combine with Track B gradient
instrumentation — measure pretrain-batch contribution to per-step
gradient bucketed by class. PSW is the A6 §3 lever; targets the
H-PRETRAIN largest single contributor without re-pretrain.

**Cite.** Track A `A5_h_pretrain_position_z.md`; training.yaml:100
recency_weight; A6 §1 ranked contribution.

**Cost.** Short smoke (~5k steps) with PSW knob + value_spread canary
gate (the canary fires at step ≤ 2k per FU-1.5; rapid kill criterion
exists). ~2 hours wall on vast 5080.

---

### Candidate 3 — ply_cap_value=0.0 × playout_cap full_search_prob=0.5 cross

**Mechanism.** Two changes interact non-additively:
1. **ply_cap_value=0.0** (§S178 T12 `1b47eb1`): truncation positions
   (ply≥150) now carry value target 0 instead of `draw_value=-0.5`.
   Honest-neutral on truncation, but every position in the ply-150
   game inherits target 0 via mc_outcome.
2. **playout_cap full_search_prob=0.5** (`v6_sustained.yaml`): half of
   moves run full 600-sim search vs 100-sim fast. Full-search games
   are *longer* on average (better defense) → more ply-cap truncations.

Together: more truncated games → more value-target=0 examples → value
head learns flat targets faster → V_spread collapse to ≈0. This matches
the **FU-1.5 collapse shape exactly**: ≈0 floor with oscillation, not a
negative spike. Both changes were introduced when §175 baseline rolled
forward to §S178 — no run since has tested either independently.

**Falsifiability.** Single-knob A/B: revert ply_cap_value 0.0 → -0.25
(midpoint between L2-canonical -0.5 and current 0.0) holding all else
constant. If V_spread canary holds > +0.20 through step 5k, candidate
confirmed. L36 retired pure single-knob A/B discipline ("4 unranked
suspects = 120h") but value_spread canary at 2k gives a 2-hour kill
criterion — cheap.

Alternative: A/B on full_search_prob 0.5 → 0.25 alone; same canary gate.

**Cite.** §S178 T12 `1b47eb1` + `e3ede86`; `v6_sustained.yaml`:138; FU-1.5
front-loaded shape; L44.

**Cost.** Short smoke (~2k steps with value_spread canary) per arm.
~30 min wall on vast 5080. Two arms = 1 hour total.

---

### Candidate 4 — Static `bot_corpus` staleness × outcome-channel feedback

**Mechanism.** The SealBot-vs-anchor bot corpus (`5b9ad09`) is **static** —
baked once vs `bootstrap_model_v6.pt` and never regenerated. As selfplay
training drifts the model state into the colony attractor, the bot
corpus's value/policy targets become increasingly *off-distribution* —
they teach what an early-bootstrap-anchor would do, not what the model
*should* do against the live selfplay distribution. The refresh hook
exists but is disabled (`27849e8` `bot_corpus_refresh.enabled: false`).
Static off-distribution gradient may pull *with* the colony direction
(L34 mechanism: anchor↑/sealbot↓ divergence is the canonical signature),
not against it.

**Falsifiability.** Bot-corpus refresh hook on @ `min_new_games=200`
+ `cooldown_steps=25000` (existing design `98a84a8`). A/B vs §S180b
recipe. Pair with value_spread canary as success metric, NOT WR (per
A6 §4 "DO NOT re-propose A1's H-BOT lever in isolation").

**Cite.** Refresh-hook infra `27849e8`; `98a84a8` design; L34; A6 §3
combined-lever guidance.

**Cost.** Full sustained 30K run (refresh hook needs anchor-promotion
trigger cadence to fire ≥2-3 times). ~6-8 h wall on vast 5080. More
expensive than C1-C3. Pair with C1 in single instrumented run.

---

### Candidate 5 — `recency_weight=0.75` × selfplay-buffer compounding (INCONCLUSIVE)

**Mechanism.** `recency_weight: 0.75` (training.yaml:100) sources 75% of
each batch from the recent-position ring buffer (newest ~50% of buffer).
Once the attractor activates and recent selfplay games shift colony,
the ring is dominated by colony positions inside ~few k steps. The
trainer then samples colony-biased value targets 3× more often than the
fixed-corpus offset would suggest. This is a **positive feedback loop**
the §S181 H6 narrative names. Compounds with Candidate 1 (bot rows
~static) and Candidate 2 (pretrain pull) but is itself unmeasured.

**Falsifiability.** A/B with `recency_weight` 0.75 → 0.50 → 0.25.
Pair with value_spread canary @ step 2k gate; short smoke per arm.
**Caveat:** flat recency weight (0.0) was the pre-§33 default and changed
recency was a deliberate trainer tuning (§33). This proposal does not
contradict §33 — it tests whether the cross with the colony attractor
mechanism is load-bearing under §S180b recipe specifically. INCONCLUSIVE
on dominance vs C1-C4 without Track B.

**Cite.** training.yaml:100; §33 (recency-weighted replay origin); L42.

**Cost.** Short smoke per arm (~2k steps + canary). 3 arms = 1.5 h
total on vast 5080.

---

### Ranking summary

| rank | candidate | likelihood | impact | falsifiability cost |
|---:|---|:--:|:--:|:--:|
| 1 | Bot-corpus value-target imprint (+ static staleness) | HIGH (Track A A1 measured; L34 confirmed 3×) | HIGH | cheap-probe (Track B short smoke ~1h) |
| 2 | Pretrain corpus colony pull × recency_weight cross | HIGH (A5 measured +0.157 asym; A6 ranks #1 contributor) | HIGH | short smoke ~2h (PSW + canary) |
| 3 | ply_cap_value=0.0 × playout_cap=0.5 cross | MED (mechanism plausible; FU-1.5 shape matches; not measured) | HIGH (front-loaded collapse explained) | short smoke ~1h (single-knob A/B + canary) |
| 4 | Bot-corpus staleness × outcome feedback | MED (mechanism credible; L34 channel; refresh-hook untried) | MED | full sustained ~6-8h |
| 5 | recency_weight cross with selfplay buffer | INCONCLUSIVE (unmeasured surface) | MED | short smoke ~1.5h |

**Explicit non-proposals per Falsified-Hypotheses Register / L47 guard.**
NOT proposed: cosine-temp toggle (§156 L9 closed); beta2=0.95 (PR-B
L47-counter-indicated); A2-class arch-only fix (§S181 FU-2 V-FU2-C
falsified); WDL migration (same cost as A2 with no L47 evidence); c_puct
/ Dirichlet retune (§S181-T3 falsified, L41); EMA (changes selfplay
inference model; A6 deferred).

---

## 5. Open questions surfaced

Items Track D could not pin down from sprint-log + configs + commits.

### 5.1 §175 anchor file — config vs run

`configs/variants/v6_sustained.yaml`:13 declares
`bootstrap_model_v7full.pt`. §175 sprint-log lines 1232-1242 list
`bootstrap_model.pt` (v6) as "§175 anchor". The §S178 design (`5dbdf88`)
explicitly switches to `bootstrap_model_v6.pt`. **Question.** Did §175
actually launch with v7full anchor or v6 default? Sprint-log §S178a
A item (`c1173a8`) quarantined `bootstrap_model.pt` as random-init
v6w25 architecture — but §175 launched *before* that hygiene wave.
Run-id `c7e74d28…` artifact metadata would settle this. **Impact.** If
§175 used v6 anchor (not v7full), the §175 → §S178 anchor "swap" is no
swap and Candidate 4 staleness story strengthens (same anchor lineage
since §175); if v7full, the anchor swap is real and may interact with
the colony attractor onset.

### 5.2 §S180b launch-time `draw_value`

§S178 variant + base `training.yaml`:92 both retain `draw_value=-0.5`
on master at commit `5dbdf88` (§S178 launch), but the §S178 variant
explicitly overrides to **-0.1** at variant file commit. The §S180b
variant (`3146144`) inherits this `-0.1`. **Question.** Was the
draw_value -0.1 propagation explicit in §S180a / §S180b launch logs?
The current training.yaml:92 still reads "-0.5 (reverted from -0.75 for
experiment A; isolate chain aux kill)" — comment is stale.
**Impact.** If draw_value=−0.1 only fired in §S178/§S179/§S180a/b
variants, Candidate 3 (ply_cap × draw_value softening) needs the
draw_value half re-cited as variant-side; on master default the
-0.5 number still applies for non-variant runs. Hygiene-deferred.

### 5.3 FU-1.5 +0.523 step-14k oscillation

L46 names the oscillation as "chaotic post-onset divergence in the
colony-attractor neighbourhood". The peak +0.523 at step 14k is
≈85% of the anchor +0.617 — *brief value-head recovery to anchor
levels* mid-run. **Question.** Is this transient recovery
correlated with a checkpoint-save event, a buffer-flush, or a specific
arena-promotion? FU-1.5 doc does not cross-reference an
`evaluation_round_complete` step-14k payload. **Impact.** If a discrete
training event (e.g. arena promotion + best_model sync to inference
server) maps to the recovery spike, that event is a candidate
mechanism-disruptor — could be exploited as an explicit refresh
mechanic. INCONCLUSIVE pending FU-1.5 re-cross-check against
`eval_db.sqlite` of the FU-1.5 run.

### 5.4 H1 quarantine retroactive impact

§S178a A (`c1173a8`) repointed Makefile/`anchor.py`/`opponent_runners.py`
canonical defaults from `bootstrap_model.pt` (random-init v6w25) to
`bootstrap_model_v6.pt`. **Question.** Did any pre-§S178a run silently
use the random-init `bootstrap_model.pt` as `bootstrap_anchor` in
eval_pipeline.opponents? The fix landed mid-§S178 sustained run on
vast (`sS178` tmux LIVE per `2203-2207`). **Impact.** If §175 or
§S178 eval_pipeline's `bootstrap_anchor` opponent was silently a
random-init model, the L34 anchor↑/sealbot↓ "canonical colony-capture
signature" reads partially against a random opponent, weakening L34's
3× confirming-instance count. Would need to grep §175 eval-DB rows
for `bootstrap_anchor` checksums to confirm.

### 5.5 §S180b post-50k recovery hint

§S180b @ step 53.5k V_spread = +0.0990 (FU-1 §2), rebounding from
−0.016 @ 50k. **Question.** Is this within-noise (per FU-1 §4
classifier note: 1.3× SE) or a genuine *post*-crash recovery sign?
SealBot WR data stops at 50K (SIGINT). **Impact.** If genuine, the
attractor is metastable and a small perturbation could escape — argues
for a perturbation-style lever (e.g. value-target Bayesian noise or
periodic warm restart). If noise, no implication. Probably noise per
FU-1's SE computation but not pinned.

### 5.6 Track A H-CE-STRENGTH grad-L2-ratio 1.21× — colony vs extension

A4 measured gradient L2 ratio 1.21×; A6 calls this "below 1.5 threshold"
hence INCONCLUSIVE. **Question.** Is 1.21× the average across the
trajectory or the post-collapse value? FU-1.5 endpoints differ from
§S180b paths (L46) — pre-onset gradient asymmetry might differ
significantly. **Impact.** Per-class target temperature on colony
positions (T_colony > 1.0) is A6's recommended cheapest lever — its
relevance depends on whether the 1.21× holds *during* the 0→2k cliff
window, where it would compound H-PRETRAIN's +0.157 asymmetry.
Operator-decides whether to instrument inside Track B.

---

## 6. Summary for parent task

Single highest-leverage recommendation per Track D: **launch Track B
short instrumented run (~1 h vast 5080) measuring per-sample value-target
direction bucketed by class (colony / extension / neither) and by source
(pretrain / bot_corpus / selfplay)**, gated by the value_spread canary
(`879bcc8`) firing at step 2k. Track B output discriminates among
Candidates 1, 2, 5 and supplies the missing per-step pull accounting that
A6 ranks as the largest unmeasured surface (+0.18/step gap between
source upper-bound and observed-after-damping rate).

If Track B confirms loop-side imbalance: combined-lever next wave is
**PSW + bot-corpus refresh hook + per-class target temperature on
colony**, value_spread canary as success metric (NOT loss / value-acc
— Goodhart per A6 §3 / L42).

Candidates 3 (ply_cap × playout_cap cross) is a CHEAP discriminator
(~1 h) and could run in parallel with Track B prep on a separate vast
instance or laptop smoke if available.

NO architecture-only fixes. NO MCTS-knob retunes. NO repropose of
falsified items. L47 stands.
