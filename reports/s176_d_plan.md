# §176 Phase A — Plan synthesis (Wave D)

**Branch:** `phase4.5/s176_phase_a_validation` | **Date:** 2026-05-14 | **Author:** Wave D synthesizer.

This plan synthesizes Waves A1-A4, B, and C into the §176 Phase B implementation
roadmap. Every claim cites a Wave A/B/C source file + section/line. No new
hypotheses. No design proposals beyond empirical data.

---

## Section 7 — Opening

### Goals

Land KrakenBot eval-ladder integration + a colony POC metric + design docs for
Source A (static bot-game corpus mix) and Source B (live cross-bot games) — all
within the empirical envelope set by §175 forensics + Wave C tourney + A4
falsified register. Close Q14 (`docs/06_OPEN_QUESTIONS.md:444-456`). Plan
sequences six subtasks S1-S6 across ≤10 commits, no bench gate (cold paths
only), zero edits to MCTS / replay buffer / inference bridge.

### References

Wave A1: `reports/s176_a1_kraken_smoke.md` — KrakenBot submodule + build outcome,
INTEGRABLE_NOW verdict + MCTSBot NEEDS_WEIGHTS_DOWNLOAD.
Wave A2: `reports/s176_a2_eval_arch.md` — eval dispatch trace + CACHING_CLEAN
verdict + minimal-diff plan (~150-180 LOC).
Wave A3: `reports/s176_a3_selfplay_forensics.md` — §175 colony attractor
REFUTED single-cluster; POC = `n_components` (Cohen's d −0.822).
Wave A4: `reports/s176_a4_falsified_scan.md` — 15-item do-not list + L1-L17
mechanism lessons.
Wave B: `reports/s176_b_smoke.md` — 3 wrappers + harness + 15-game smoke PASS.
Wave C: `reports/s176_c_tourney/{summary.md,verdicts.txt,ratings.csv,h2h_matrix.csv,colony_table.csv,per_game.jsonl}`
— 1050 games / 7 bots / BT ladder + V1-V6 + D1-D5 verdicts.

### 3 most important findings Wave D readers must internalize

1. **our_v6 bootstrap @ MCTS-128 ≈ SealBot @ 0.5s.** Wave C BT delta −62 Elo,
   25/50 H2H (`reports/s176_c_tourney/summary.md:9-19`, finding #1). The
   §175 degradation (18% → 4% SealBot WR by step 50K, A3 §b) is NOT a
   weak-bootstrap problem — it is internal regression of the self-play loop
   AWAY from a SealBot-comparable start point. This recalibrates the §176
   framing: Source A/B work must not start from "v6 is weak vs strong bots".

2. **§175 colony attractor is multi-island fragmentation, NOT one large
   diffuse cluster.** Operator's qualitative claim REFUTED by A3 (§d):
   single-cluster fraction monotone-down 18.1% → 6.3% strict (4483 → 7267
   games / cohort). `n_components` grows 9.29 → 13.63 across the 40K
   step-change (A3 §c). Cohen's d −0.822 — largest of 8 candidate metrics
   (A3 §e). POC metric = raw `n_components`, not single-cluster fraction.

3. **15-item empirical do-not list dominates the constraint surface** (A4 §e).
   Top three: (#1) NO in-process Python daemon for opponent mixing — GIL
   3.3× regression killed §17 (sprint log line 597); (#3) training loss is
   NOT a proxy for SealBot WR (L4 / A4 §a); (#5) e30 pretrain epoch ceiling
   (L8). Source A/B design docs are CONSTRAINED, not free design space.

---

## Section 1 — D1 to D5 verdicts

Reproduced + per-row justification from `reports/s176_c_tourney/summary.md:100-108`.

| ID | Decision | Verdict | Cited from | What would close NEEDS_REVISION / PARTIAL |
|---|---|---|---|---|
| D1 | Extend BotProtocol for compound bots via caching (not `get_turn`) | **BACKED** | A2 §i CACHING_CLEAN; Wave B 15-game + Wave C 1050-game stability (`reports/s176_b_smoke.md:97`, `summary.md:6`) | n/a |
| D2 | Tourney includes all KrakenBot variants + checkpoints | **PARTIAL** — MCTSBot blocked on weights | A1 §b: `training/{mcts,resnet}_results/best.pt` missing, `.gitignore:8`, no HF/S3 mirror; Wave C ran 7/9 bots | Operator supplies the two `best.pt` files OR contacts WolverinDEV upstream (A1 §g). Cython build + `vendor/bots/krakenbot/setup_puct.py build_ext --inplace` (A1 §f). |
| D3 | MinimaxBot colony-rate < MCTSBot colony-rate (αβ punishes spread) | **PARTIAL** — modified-D3 (vs SealBot) BACKED at 27.9pp gap; MCTSBot comparison BLOCKED | V3 PASS in `summary.md:76-83`: sealbot 35.0% [29.6%, 40.9%] vs kraken_minimax_strong 7.1% [4.2%, 11.8%], CIs non-overlapping by 18pp | Needs MCTSBot weights to test original D3 hypothesis (kraken-MCTS vs kraken-minimax). Same blocker as D2. |
| D4 | Mix ratio target ~75/15/10 (selfplay/bot/corpus); per-source weights configurable | **NEEDS_REVISION** — Kraken games ~500 Elo weaker than selfplay | `ratings.csv`: kraken_minimax_strong −494, kraken_minimax_fast −499 vs sealbot 0, our_v6_mcts128 −62. Naive 15% bot pool with uniform per-source weight would dilute. | Per-source Elo-derived weighting (S4 design doc): sealbot 50% / our_v6 30% / kraken_strong 15% / kraken_random 5% of bot-game pool. See Section 3. |
| D5 | Source A static corpus first, Source B (live + cross-bot G4/G5) target | **BACKED** | V6 PASS in `summary.md:90-96`: colony rate opponent-coupled, 41pp spread across opponents for sealbot alone. A4 §e do-not #1: subprocess isolation MANDATORY for Source B. | n/a |

**Justification paragraphs.**

D1 stays BACKED because the caching pattern (`hexo_rl/bots/sealbot_bot.py:39-106`,
A2 §c) handled all 1050 Wave C games with zero compound-turn bugs
(`summary.md:6`). No evaluator-side refactor needed (A2 §h Approach 2: 150-180
LOC for new bots vs +40 LOC `get_turn` opt-in).

D2 is PARTIAL only because MCTSBot weights are gitignored upstream
(`vendor/bots/krakenbot/.gitignore:8` per A1 §b) and have no committed mirror.
A1 §g recommends escalation to operator: "do you have a copy of the trained
KrakenBot resnet weights, or should we contact the WolverinDEV/krakenbot
maintainer?" Wave D defers MCTSBot integration to §177+ pending operator
action.

D3 modified-PASS is supported by V3 (`summary.md:76-83`) but the original
hypothesis (kraken-MCTSBot colony < kraken-MinimaxBot colony, "αβ punishes
spread") cannot be tested until MCTSBot weights arrive. The modified-D3
(MinimaxBot colony < SealBot colony, 27.9pp gap, CIs non-overlapping by 18pp)
is the empirically defensible substitute.

D4 needs revision because the BT ladder
(`reports/s176_c_tourney/ratings.csv`) shows ~500 Elo separation between
sealbot/our_v6 and the kraken-minimax cluster. Uniform per-bot weighting in
the 15% bot-game pool would feed the trainer noise rather than signal. L1
(corpus quality = model quality floor, A4 §b) + L15 (v6 corpus retired for
exactly this contamination class) cement the Elo-weighting requirement.
Section 3 below has the adjusted defaults.

D5 BACKED because V6 PASS (opponent-coupled colony emergence,
`summary.md:90-96`) empirically demonstrates that cross-bot games carry
signal selfplay alone can't. A4 §e do-not #1 (§17 3.3× GIL regression,
sprint log line 597) blocks the live implementation until subprocess
architecture is in place — Source A first because the static corpus path is
unblocked.

---

## Section 2 — Sub-task sequence S1-S6

### S1 — KrakenBot wrappers + production integration

**Current state.** Three wrapper files exist in working tree
(`hexo_rl/bots/krakenbot_bot.py` MODIFIED + `krakenbot_random.py`,
`krakenbot_mcts.py` UNTRACKED per `git status`) plus
`tests/test_krakenbot_wrappers.py` (3 tests PASS, `s176_b_smoke.md:18-25`).
`_smart_legal_fallback` already lives in `krakenbot_bot.py:37-69` with
neighbour-2 candidate pool from `_D2_OFFSETS` (vendor `bot.py:42`). Smart
fallback fired 438/1050 games = 0.42/game over Wave C with zero
uniform-random degradations (`summary.md:118`, finding #3).

**Remaining file work.**
- Review quality of three wrappers (DONE per A1 §g + B §a — INTEGRABLE).
- `krakenbot_mcts.py` (152 LOC, B §a): keep as skeleton; raises
  `FileNotFoundError` at construction until weights arrive. Decision: KEEP
  skeleton — A1 §d already showed `mcts_bot.py:46` `torch.load` failure path;
  preserving the wrapper means S2's eval ladder can probe `kraken_mcts_*`
  names without resurrection later. Cost is 152 LOC dead code, value is
  zero-effort enablement when weights arrive (S5/§177+ scope).

**Commit boundary 1.** `feat(bots): KrakenBot wrappers + smart neighbour-2 fallback (§176 S1)` —
single commit covering all three wrappers (`krakenbot_bot.py` MOD,
`krakenbot_random.py` NEW, `krakenbot_mcts.py` NEW) + smart-fallback fix +
`tests/test_krakenbot_wrappers.py` (NEW, 64 LOC, 3 tests).

**Test count.** 3 tests in `tests/test_krakenbot_wrappers.py` (B §a, PASS
1.90s). No new tests required.

**LOC delta.** ~300 LOC new code (B §a: 148 + 152 wrappers; the
`_smart_legal_fallback` patch in `krakenbot_bot.py` adds ~30 LOC over the
A1-state).

**INV risk.** 0. A2 §f: no INV pin fires on new BotProtocol implementation
or new opponent name.

**Bench gate.** NO. Cold path (bot wrappers, not MCTS / replay / inference).

---

### S2 — Eval pipeline integration (configs/eval.yaml + opponent_runners.py + Q14 close)

**Plan basis.** A2 §h Approach 2 (caching, lowest friction): +80-100 LOC per
new bot wrapper (DONE in S1) + ~10 LOC per opponent in `configs/eval.yaml`
+ ~20 LOC per opponent in `hexo_rl/eval/opponent_runners.py` + ~5 LOC
`eval_pipeline.py` wiring.

**Q14 close.** `docs/06_OPEN_QUESTIONS.md:444-456` (verified read 444-456):
priority LOW, blocked on submodule add. Wave A1 §a verifies submodule
pinned at `d9c5bfb` ("better eval"). Wave C BT delta places KrakenBot
MinimaxBot 494 Elo BELOW SealBot. Q14 close text: Kraken-Minimax is NOT a
Bradley-Terry anchor candidate (already noted in `docs/06_OPEN_QUESTIONS.md:455`
"Do NOT use KrakenBot MCTSBot as a Bradley-Terry anchor"); SealBot remains
primary gate (`reports/s176_a4_falsified_scan.md` Q14 row, §d).

**Specific opponents to wire.**
- `kraken_minimax_strong` (think_time=1.0s). Latency: per Wave C
  `summary.md:50`, total_wall_s mean = 12.087, max_move_s = 2.049.
- `kraken_random` (no think time). Latency negligible per `summary.md:51-52`
  (0.005s total).

**OMITTED.** `kraken_minimax_fast` (0.1s). V2 finding (`summary.md:73-75`):
0.1s vs 1.0s BT delta only −5 Elo, within CI. 10× budget does NOT translate
to strength — iterative deepening saturates at depth 4 on dense mid-board
positions (A1 §e: `last_depth` plateaus). Keeping only the strong dial
prevents wasted eval budget. (NEW FALSIFIED row, see S6.)

**OMITTED.** `kraken_mcts_resnet` / `kraken_mcts_selfplay`. NEEDS_WEIGHTS
(A1 §g). Deferred until S5 / §177+.

**Eval-ladder justification.** Even though kraken_minimax_strong loses to
SealBot (V1 modified-FAIL, `summary.md:69-71`), V3 PASS
(`summary.md:76-83`) shows Kraken colony rate (7.1%) differs from SealBot
(35.0%) by 27.9pp with non-overlapping CIs. This is a **distribution-diversity
signal** worth tracking on the eval ladder: Kraken probes a different
failure mode (tight αβ-punished play) than SealBot (tree-search-with-extensions).

**Test update.** `tests/test_eval_opponent_runners.py::test_opponents_canonical_order`
(referenced in A2 §f INV table, "0 pins fire on new opponent" — but the
canonical-order list itself is an explicit string assertion that must be
updated for new bot names). Cost: ~2 LOC in the canonical list + 0 LOC
elsewhere.

**Commit boundary 2.** `feat(eval): kraken_minimax_strong + kraken_random eval ladder + Q14 close (§176 S2)` —
covers `configs/eval.yaml` additions, `opponent_runners.py` two new runner
closures, `eval_pipeline.py` wiring loop update, `tests/test_eval_opponent_runners.py`
canonical-order update, `docs/06_OPEN_QUESTIONS.md:444-456` Q14 RESOLVED.

**LOC delta.** ~50 LOC code (20 yaml + 40 runners + 5 wiring) + ~5 LOC test
+ ~10 LOC Q14 close text = ~65 LOC.

**INV risk.** 1 (canonical-order test).

**Bench gate.** NO. Eval runs per-checkpoint, cold path.

---

### S3 — Colony POC metric (Python-side first, Rust deferred)

**A3 verdict.** POC metric = `n_components` (raw BFS, all components incl.
orphans). Cohen's d = −0.822 between 20K and 50K cohorts
(`reports/s176_a3_selfplay_forensics.md` §e, table line 184). Largest |d|
of 8 candidates. Strict_n_components==1 fraction 18.1% → 0.2% across the
§175 cohort window (A3 §d).

**Implementation site.** `hexo_rl/selfplay/pool.py` `game_complete` event,
the existing colony emission locus (sprint log line 1431 "per-game
`game_length` in structlog `game_complete` (pool.py:593)"). The Wave C
harness already proves the shape works — `per_game.jsonl` (~1050 lines) has
`n_components_winner` (B §b schema: pair, game_idx, p1, p2, winner,
winner_bot, plies, n_stones_p1/p2, colony_fraction_winner,
**n_components_winner**, mean_pairwise_hex_dist_winner, max_pairwise_hex_dist_winner, ...).

**Algorithm.** Hex-adjacency BFS on winner's stones, threshold connectivity
distance = 1. Reused from `hexo_rl/eval/colony_detection.py:31-52`
`_connected_components` (cited A3 file:line list).

**LOC.** ~25 LOC in `pool.py` (BFS helper inline + emit field). The Wave B
harness implementation in `scripts/tournament_validate.py` is the reference
pattern.

**Constraint.** Python-only this sprint per master prompt constraint #7.
Rust port deferred to a future micro-refactor.

**Threshold for §176 sustained.** Warning-only signal: `n_components ≥ 15`
matches §175 50K cohort mean 13.63 (A3 §b cohort table). Do NOT make it a
hard abort gate. A4 §e do-not #9 ("no moving goalposts") + L12 (sprint log
line 636) require pre-registration; A4 §e #15 requires fresh-context review.
Warning-only is the conservative call.

**Commit boundary 3.** `feat(selfplay): emit n_components colony POC metric in game_complete (§176 S3)` —
single commit, `pool.py` + emit-schema test.

**INV risk.** The schema test in `tests/refactor_invariants/` may snapshot
`game_complete` shape. Wave D plan author has not confirmed any active INV
on the `game_complete` schema (A2 §f lists 6 INVs, none on selfplay event
schema). Likely 0, possibly 1.

**Bench gate.** NO. Cold path, runs per-game at terminal state.

---

### S4 — Source A static corpus mixing design doc

**File.** `docs/designs/bot_game_mixing_design.md` (NEW). Follow format of
`docs/designs/encoding_alpha_multiwindow_selfplay_design.md` (Status,
Problem statement, References, Numbered sections).

**Scope.** Spec what + why. No implementation. The contract precedes the
code per A4 §b L13 ("pre-registered pass criteria; implicit done-when
causes scope creep") + L17 ("§122 rotation 'one-liner' was a ~50-80 line
port").

**Sources.** Sealbot + our_v6_mcts128 + kraken_minimax_strong + kraken_random.

**Per-source weighting (Elo-derived, per D4 NEEDS_REVISION).** Adjusted
defaults in Section 3 below. Mechanism justification: A4 §b L1 (corpus
quality = model quality floor) + L15 (v6 corpus retired wholesale due to
~41% bot games at uniform source_weight=1.0).

**Interleave strategy.** Sample-uniform from union with per-source weights,
NOT chronological mix. Per A4 §b L1: silent filter bugs (rng.choice on
uniform weights → degeneracy at §147) do NOT show in loss curves. Explicit
per-source weight enforcement at sample time is the only safe design.

**Live opponent constraint.** A4 §e do-not #1 (§17 sprint log line 597): no
in-process daemon for Source B. Source A is corpus-only — no live opponent,
no GIL issue. Subprocess architecture for Source B is S5 scope, not S4.

**Commit boundary 4.** `design(corpus): Source A static bot-game mixing design doc (§176 S4)` —
single commit, ~200 LOC markdown.

**INV risk.** 0 (docs-only).

**Bench gate.** NO.

---

### S5 — Source B live bot games + cross-bot G4/G5 design doc

**File.** `docs/designs/bot_live_selfplay_design.md` (NEW). End-state spec,
not §176 implementation.

**Constraint.** A4 §e do-not #1 (sprint log line 597): subprocess isolation
MANDATORY. No in-process Python daemon. 3.3× GIL regression
(1.52M → 464K pos/hr) is the §17 verdict, citation is in
`docs/07_PHASE4_SPRINT_LOG.md:597`.

**Architecture.** Cross-bot G4/G5 protocol: kraken-vs-sealbot games,
sealbot-vs-our_v6 games, etc. Reuse `scripts/tournament_validate.py`
(Wave B §b) as the launcher — single-game-at-a-time, fresh bot per game.
Already validated end-to-end on 1050 games (`summary.md:6`).

**Why design-only now.** §177+ implementation, not §176. Wave C 1050 games
took 5111.2s wall on laptop (`summary.md:6`); scaling to a sustained
selfplay loop needs subprocess pool sizing + corpus-ingest schema +
inference-server routing. None of that is empirically pinned by Wave A/B/C.

**Signal justification.** V6 PASS (`summary.md:90-96`): colony rate is
opponent-coupled (sealbot colony vs kraken_minimax_strong = 0.412 vs
sealbot vs our_v6_mcts128 = 0.154 vs sealbot vs randombot = 0.024 — 41pp
spread). Cross-bot games give a signal selfplay alone can't.

**Commit boundary 5.** `design(corpus): Source B live bot games + cross-bot eval design doc (§176 S5)` —
single commit, ~250 LOC markdown.

**INV risk.** 0 (docs-only).

**Bench gate.** NO.

---

### S6 — Sprint log §176-Phase-A entry + Q14 close + L18+ lessons

**Files touched.**
- `docs/07_PHASE4_SPRINT_LOG.md` — append §176-Phase-A entry; add new
  Falsified Register row + L18+ Mechanism Lessons.
- `docs/06_OPEN_QUESTIONS.md:444-456` — Q14 RESOLVED 2026-05-14
  (already-touched in S2, but the wording lives here).

**Sprint log §176-Phase-A entry contents.**
- TL;DR: V1-V6 results + D1-D5 verdicts (from `reports/s176_c_tourney/summary.md`).
- Report pointers: A1, A2, A3, A4, B, C summary + Wave D plan.
- New Falsified Register row: "Kraken MinimaxBot @ 1.0s > @ 0.1s by ≥ 30 Elo
  — FALSIFIED per Wave C V2 side-finding (`summary.md:73-75`), BT delta −5
  Elo, within CI."
- Forward pointer to §176 Phase B (S1-S5 implementation work).

**L18+ candidates (from `summary.md:113-122` "Surprising findings").**

L18: our_v6 bootstrap @ MCTS-128 ≈ SealBot @ 0.5s. §175 degradation locus
is internal (policy/value head drift under self-play), not "bootstrap too
weak vs opponent" (`summary.md:114`).

L19: KrakenBot MinimaxBot time_limit is inert — iterative deepening
saturates at depth 4 on dense mid-board positions (A1 §e + V2 side-finding).
10× budget is not 10× strength.

L20: Smart neighbour-2 fallback prevents fb_rand degradation —
`fb_n2=438/1050 fb_rand=0` over Wave C 1050 games confirms the wrapper-side
fix sustained over a long run (`summary.md:118`).

L21: argmax-only baseline (n_sims=1 / MCTS-1) is below RandomBot —
our_v6_argmax 0 wins / 300 games < randombot 0 wins / 300 + 99 draws
(`verdicts.txt:10-11`). The "argmax baseline" in §175 reports is
structurally below random (`summary.md:120-121`); confirms
`feedback_v6_v8_same_training_data.md` argmax-handicap memory.

L22: Colony rate is opponent-coupled. 41pp spread for sealbot across
opponents (V6 PASS). Cross-bot games carry signal selfplay alone can't —
this is the empirical basis for Source B design (S5).

**Commit boundary 6.** `docs(sprint): §176-Phase-A close-out + Q14 RESOLVED + L18-L22 lessons (§176 S6)` —
single commit, ~150 LOC markdown.

**INV risk.** 0 (docs-only).

**Bench gate.** NO.

---

## Section 3 — Mix-ratio defaults adjusted per C findings

D4 NEEDS_REVISION basis: Wave C BT ladder (`reports/s176_c_tourney/ratings.csv`):

| Bot | Elo | Justification |
|---|---:|---|
| sealbot | 0.0 | Anchor |
| our_v6_mcts128 | −62.1 | Tied with sealbot at MCTS-128 (`summary.md:14`, 25/50 H2H) |
| kraken_minimax_strong | −493.8 | ~500 Elo below sealbot |
| kraken_minimax_fast | −499.4 | Inert vs strong (V2 side-finding) — OMITTED from mix |
| kraken_random | −3072.2 | Saturated L2-reg corner; near random — diversity-only |
| randombot | −3090.8 | Floor |
| our_v6_argmax | −3102.4 | Structurally below random (L21 candidate) — OMITTED |

### Original master-prompt 75/15/10 default

```
75% selfplay
15% bot-game pool (uniform per-source)
10% human corpus
```

### Adjusted per-source weighting (Source A design doc target)

```
75% selfplay                           [unchanged]
15% bot-game pool, Elo-derived split:
    sealbot              50% pool → ~7.50% corpus
    our_v6_mcts128       30% pool → ~4.50% corpus
    kraken_minimax_strong 15% pool → ~2.25% corpus
    kraken_random         5% pool → ~0.75% corpus
10% human corpus (v7+ Elo-weighted per L15)
```

### Per-source one-sentence justifications

- **sealbot 50%** — BT-anchor at 0 Elo (`ratings.csv`), 91.3% win rate
  including draws (`verdicts.txt:5`); highest-quality bot signal in the mix.
- **our_v6_mcts128 30%** — BT −62 Elo, within CI of sealbot (`ratings.csv`);
  87.7% win rate including draws (`verdicts.txt:9`); avoids over-fitting
  exclusively to sealbot's idiosyncrasies.
- **kraken_minimax_strong 15%** — BT −494 Elo (`ratings.csv`); colony rate
  7.1% vs sealbot 35.0% (`verdicts.txt:38-44`, V3 PASS in `summary.md:76-83`);
  injects αβ-punished low-spread distribution.
- **kraken_random 5%** — BT −3072 Elo (`ratings.csv`); only 7 wins / 300
  games (`verdicts.txt:8`); diversity-injection per V6 PASS (opponent-coupled
  colony, `summary.md:90-96`); low weight prevents corpus-quality
  degradation per L1.
- **kraken_minimax_fast 0%** — OMITTED. V2 side-finding (`summary.md:73-75`):
  inert vs strong dial (−5 Elo, within CI).
- **our_v6_argmax 0%** — OMITTED. 0 wins / 300 games (`verdicts.txt:10`);
  L21 candidate "structurally below random".

---

## Section 4 — Risk register

| # | Risk | Likelihood | Mitigation | Source |
|---|---|---|---|---|
| 1 | KrakenBot MinimaxBot sentinel-fallback fires 0.42/game (438/1050 over Wave C) — smart fallback prevents crash but distorts strength | OBSERVED | Smart-neighbour-2 fallback in `hexo_rl/bots/krakenbot_bot.py:37-69`; document caveat in S2 eval-ladder report | A1 §e (sentinel `[(0,0)]` returned), Wave C `summary.md:118` |
| 2 | Source A bot-game mixing degrades corpus quality if Kraken games (weak) weighted naively (D4 NEEDS_REVISION) | HIGH if uniform | Elo-derived per-source weights per S4 design doc + Section 3 table | A4 §b L1 (corpus quality = model floor); A4 §b L15 (v6 corpus retired for this exact failure mode); Wave C `ratings.csv` (~500 Elo gap) |
| 3 | KrakenBot MCTSBot weights never arrive — D2 stays PARTIAL indefinitely | MEDIUM | Skeleton wrapper retained (S1); operator escalation per A1 §g; deferred to §177+ | A1 §b (`.gitignore:8`, no HF/S3 mirror); A1 §g |
| 4 | Colony POC threshold `n_components ≥ 15` becomes a moving goalpost if §177+ runs underperform | HIGH if not pre-registered | Warning-only signal per S3 plan; do NOT promote to hard abort gate; require fresh-context review per L13 | A4 §e do-not #9 (no moving goalposts); L12 (sprint log line 636); A3 §c (mean reached 13.63 at 50K) |
| 5 | Source B live bot-mixing re-introduces §17 GIL regression (3.3× pos/hr drop) | CRITICAL if attempted in-process | S5 design doc mandates subprocess isolation; do NOT in-process daemon; `scripts/tournament_validate.py` (Wave B) is the launcher template | A4 §e do-not #1; sprint log line 597; §17 c9f39de revert |
| 6 | Operator's qualitative "one large diffuse cluster" intuition still drives §177+ work despite A3 REFUTED verdict | MEDIUM | A3 forensics REFUTED single-cluster (6.3% non-orphan at 50K); §176 Phase B work must cite §175 multi-island fragmentation as the attractor, not unified colony | A3 §d, §verdict |
| 7 | Wave D plan author mis-allocates §176 Phase B bot-integration LOC budget as "one-liner" | HIGH per L17 | S2 budget explicit: ~65 LOC opponent integration + 1 INV test update; L17 origin §122 rotation was ~50-80 LOC port | A4 §b L17; sprint log line 641 |

---

## Section 5 — Open questions deferred to §177+

- **Source B implementation.** S5 is design only. Subprocess pool sizing,
  corpus-ingest schema, inference-server routing all unpinned by Wave A/B/C.
- **MCTSBot weights acquisition.** A1 §g escalation pending. Without
  `vendor/bots/krakenbot/training/{resnet,mcts}_results/best.pt`, D2 + D3
  original-hypothesis stay BLOCKED.
- **Rust port of `n_components` POC.** S3 is Python-only per master prompt
  constraint #7. Rust port would land alongside any other engine-side colony
  detector consolidation (current locus
  `engine/src/game_runner/worker_loop.rs:806-808` per A3 file:line list).
- **Q15 integration with Source A.** Corpus tactical quality filtering
  (`docs/06_OPEN_QUESTIONS.md:460-475`) overlaps with Source A weighting.
  If max_threat_strength manifest field lands, Source A weights become
  Elo × tactical-quality joint. Out of §176 scope per A4 §d (Q15 watch).
- **Q-§176-residual P24b/c.** `docs/06_OPEN_QUESTIONS.md:752-770` —
  HexTacToeNet decomposition + seed_everything shim. Both cold-path; hot-path
  edits in `forward()` would require bench gate. Deferred per A4 §d.
- **§175 internal regression locus.** L18 candidate flags policy/value head
  drift under self-play as the §175 degradation mechanism (`summary.md:114`).
  Mechanism not pinned by Wave A/B/C — explicit OPEN for §177+ trainer
  diagnostics.
- **Strict argmax bot path.** B §e item #5: `our_v6_argmax` is currently
  `OurModelBot(n_sims=1)` MCTS-1 proxy. True policy-head argmax (no MCTS)
  would require an `argmax=True` kwarg on `OurModelBot`. Out of §176 scope.

---

## Section 6 — Commit boundary table

| # | S# | Commit message | LOC delta | Bench-gate | INV touched |
|---|---|---|---|---|---|
| 1 | S1 | `feat(bots): KrakenBot wrappers + smart neighbour-2 fallback (§176 S1)` | +300 (148+152 wrappers, +30 fallback, +64 tests + minus existing diff) | N | 0 |
| 2 | S2 | `feat(eval): kraken_minimax_strong + kraken_random eval ladder + Q14 close (§176 S2)` | +65 (20 yaml + 40 runners + 5 wiring + 2 test update + 10 Q14 close text) | N | 1 (test_opponents_canonical_order) |
| 3 | S3 | `feat(selfplay): emit n_components colony POC metric in game_complete (§176 S3)` | +25 (BFS + emit) | N | 0-1 (possible game_complete schema snapshot) |
| 4 | S4 | `design(corpus): Source A static bot-game mixing design doc (§176 S4)` | +200 (markdown) | N | 0 |
| 5 | S5 | `design(corpus): Source B live bot games + cross-bot eval design doc (§176 S5)` | +250 (markdown) | N | 0 |
| 6 | S6 | `docs(sprint): §176-Phase-A close-out + Q14 RESOLVED + L18-L22 lessons (§176 S6)` | +150 (sprint log entry + L18-L22 lessons + Falsified row) | N | 0 |

**Total: 6 commits, ~990 LOC delta, 0 bench gates, 1-2 INV touched.**

Master prompt cap: ≤10 commits. Headroom of 4 commits available for split
options:
- S2 can be split into commit-2a (`feat(eval): wire kraken bots`) + commit-2b
  (`docs(open-questions): Q14 RESOLVED`).
- S6 can be split into commit-6a (`docs(sprint): §176-Phase-A close-out`) +
  commit-6b (`docs(open-questions): Q14 RESOLVED`) if S2 doesn't carry Q14.
- S1 could be split if the wrappers introduce more friction than projected
  during execution.

Recommended path: SHIP AS 6 commits. Splits add diff-noise without value
when the LOC delta per S# stays under ~300.

---

## Compliance check

### A4 15-item do-not list

- #1 (no in-process daemon): S5 design doc mandates subprocess isolation. PASS.
- #2 (no stacked knob changes): Plan touches no cosine-temp / Dirichlet /
  opening_plies / playout_cap. PASS.
- #3 (training loss ≠ SealBot WR proxy): S3 metric is `n_components`, not loss
  variant. S2 eval-ladder relies on H2H + Elo, not loss. PASS.
- #4 (no PMA / learned aggregation replacement): No NN architecture changes.
  PASS.
- #5 (e30 epoch ceiling): No bootstrap retraining proposed. PASS.
- #6 (no LEGAL_MOVE_RADIUS alone tuning): No radius changes. PASS.
- #7 (no cosine-temp without radius jitter): No cosine-temp changes. PASS.
- #8 (no v6 corpus baseline): S4 Source A uses sealbot + our_v6 + kraken +
  v7+ human, NOT v6 corpus contamination. PASS.
- #9 (no SealBot WR gate recalibration): S3 POC threshold is warning-only,
  not WR gate. PASS.
- #10 (no smoke step-budget extension): No smoke runs in §176 Phase B plan
  (only eval ladder + corpus design). PASS.
- #11 (no frozen-spine fine-tune): No fine-tuning proposed. PASS.
- #12 (no probe-only eval gates): S3 POC is warning-only; S2 retains
  MCTS-matched SealBot as primary gate per Q14 close. PASS.
- #13 (no dev-default cold smoke): No smoke runs. PASS.
- #14 (no "one-line config change" for BotProtocol integration): S1 budget
  is +300 LOC (matches L17 magnitude). PASS.
- #15 (no implicit done-when): Wave D plan IS the pre-registered done-when.
  Wave E fresh-context review locks it. PASS.

### Master-prompt 10-item do-not list

- (#1) In-process GIL daemon: S5 subprocess isolation. PASS.
- (#2) Stacked knob changes: none. PASS.
- (#7) Python-only POC: S3 explicit. PASS.
- (#12) MCTS-matched eval: SealBot primary gate retained. PASS.
- All others: not invoked by S1-S6 scope. PASS.

No violations to flag.

---

## Done-when checklist

- [x] Section 7 (opening): goals + references + 3 most important findings
- [x] Section 1: D1-D5 verdict table + per-row justification
- [x] Section 2: S1-S6 subtask sequence with file/LOC/INV/commit-boundary
- [x] Section 3: mix-ratio defaults adjusted, per-source justified
- [x] Section 4: risk register, 7 rows, empirically grounded
- [x] Section 5: open questions deferred to §177+
- [x] Section 6: commit boundary table, 6 commits (≤10 cap)
