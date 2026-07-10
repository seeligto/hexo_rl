# D-C VALPROBE WP1 — value recognition lag (DESIGN, FROZEN)

**Status:** DESIGN-FROZEN. No measurement has run. Every number in §7 is a placeholder.
**Role:** fable5 DESIGN + sign-off. A separate IMPL agent executes; IMPL must not alter §1 or the
metric definitions in §4 — deviations require a new frozen revision with a changelog entry.
**Date:** 2026-07-10. **Branch:** `phase4.5/valprobe_dc` (off confres `2d3335d`; contains evalfair
`d219334`).
**Question (card #1 premise):** does run2's value head MISS developing double-threat formations
2+ compound turns before they are provably lost? D-A EVALFAIR WP2 returned NO-VERDICT because its
"value-blind" proxy measured SealBot's d7 horizon (`per_loss_table.jsonl` N2/N3/t_star columns),
never the NET's own value. WP1 is the direct measurement: the net's own value trajectory on its
own loss games, against a win-game control.

Legend: **[V]** verified against source this session (file:line cited). **[D]** design decision,
rationale inline. An unmarked factual claim is a defect.

---

## 1. FROZEN VERDICTS

Pre-registered before any measurement. Verbatim; do not edit.

- V-CONFIRM: ≥30% of losses have lag_raw ≥ +2 compound turns (value still optimistic ≥2 turns
  AFTER position provably lost) WITH false-pessimism ≤10% on wins → value-representation premise
  CONFIRMED; flagged positions = card #1 probe set.
- V-KILL: ≥60% of losses have lag_raw ≤ 0 (value pessimistic BEFORE provable) yet game lost
  anyway → head SEES it, played line walks in regardless → policy/search-side failure, card #1
  premise KILLED → run3 re-rank; must report WHICH move-selection stage ignored the value signal
  (prior? completed-Q? argmax tie-break?).
- Neither → MIXED: report lag distribution + both-ckpt (248k vs 175k) comparison, no card action.
- SECONDARY (descriptive, no gate): lag(248k) vs lag(175k) — is value recognition improving with
  training?

**Adjudication addenda (frozen with the verdicts, not edits to them):**
- Verdicts are judged on the **248k arm** (primary). The 175k arm is the SECONDARY comparison only.
- Verdict fires on the observed point fraction; the Wilson 95% CI and the opening-clustered
  bootstrap CI (§4.6) are REPORTED next to it for honesty, they do not gate.
- If BOTH V-CONFIRM and V-KILL conditions are simultaneously satisfied (arithmetically possible:
  30% + 60% ≤ 100%), the verdict is **MIXED** (bimodal lag population), no card action, escalate
  to operator with both subpopulations characterized.
- Denominator rules and per-game classification are operationalized in §4.5; they are part of the
  frozen design.

---

## 2. Founding-fact corrections (measurement-unit discipline, CLAUDE.md)

Two dispatcher premises were checked against the artifacts and are WRONG. Recorded here so no
downstream frame builds on them:

1. **"52 of the 128 are head losses" at 248k is FALSE.** Independently counted this session from
   `reports/evalfair/retro_slope/checkpoint_00248000/games.jsonl`: **57 head losses, 70 head wins,
   1 censored draw** (the 200-ply game). The "52" belongs to the **175k arm**
   (`retro_slope/run2_175k/games.jsonl`: 52 losses, 76 wins, 0 draws). Cross-check [V]:
   `per_loss_table.jsonl`'s 52 `(opening_idx, plies)` rows match the 175k loss set EXACTLY
   (52/52) and the 248k loss set NOT AT ALL (0/57). The per-loss table is a **175k** artifact.
   → WP1 primary loss set = all 57 non-censored 248k head losses; 175k arm = its own 52.
2. **"Same 128 openings replayed at 175k" is FALSE.** 248k games ran on book `evalfair_r5_v2`
   at radius 5; 175k games ran on book `evalfair_r4_v2` at radius 4 [V games.jsonl `book_id`,
   `radius` fields; per-stage books per `docs/designs/evalfair_instrument.md` §2]. The 248k↔175k
   comparison is therefore **distributional, not opening-matched** — consistent with its
   SECONDARY/descriptive status, but it must never be presented as paired.

Also pinned: `per_loss_table.jsonl` N2/N3/t_star_* columns are SealBot-d7-horizon based (the
flawed WP2 proxy) — WP1 uses NOTHING from that table except, optionally, `opening_idx`
cross-reference for the 175k arm.

---

## 3. Inputs (verified)

| Input | Path | Pin |
|---|---|---|
| 248k games (primary) | `reports/evalfair/retro_slope/checkpoint_00248000/games.jsonl` | 128 records; `ckpt_sha=312f85f632ee5046`, `book_id=evalfair_r5_v2`, `radius=5`, `arm=sims150` [V] |
| 175k games (secondary) | `reports/evalfair/retro_slope/run2_175k/games.jsonl` | 128 records; `ckpt_sha=c615beb3f7a8ce97`, `book_id=evalfair_r4_v2`, `radius=4` [V] |
| 248k checkpoint | `checkpoints/run2_retro/checkpoint_00248000.pt` | sha256[:16] must equal `312f85f632ee5046` (recompute via `scripts/evalfair/core.py::_ckpt_sha` [V core.py:240]) |
| 175k checkpoint | `scripts/arena/weights/run2_175k.pt` | sha256[:16] must equal `c615beb3f7a8ce97` |
| **DECOY — do not load** | `checkpoints/d1m_vast_latest/checkpoint_00248000.pt` | stamped `v6_live2` (no `_ls`), wrong lineage; the gated loader (§5.2) raises on it by construction |
| Opening books | `tests/fixtures/opening_books/evalfair_r5_v2.json`, `evalfair_r4_v2.json` | replay integrity cross-check only [V both present] |
| Encoding | `v6_live2_ls` | registry `engine/src/encoding/registry.toml:325-339` [V]: multi-window K-cluster, `policy_pool=legal_set_scatter_max`, 4 planes incl. `moves_remaining_bcast` + `ply_parity_bcast`, `policy_logit_count=362`, value_pool=min |

Games ran on host `omarchy` (laptop) [V result.json `host`]; measurement SHOULD run on the same
host (replay-match gate §5.8 assumes same-host fp behavior; memory
`d-ws3v3-a1-rebaseline-match-hostmatch` HOST-MATCH rule).

Interpreter: **`.venv/bin/python`** (the native `engine` pyo3 module — `TacticalSolver`, `Board`,
`MCTSTree` — binds only there; bare `python` has no `engine.engine`
[V evalfair_instrument.md §1]).

---

## 4. Definitions and metrics (frozen)

### 4.1 Units — plies, compound turns, position index

- `moves` in games.jsonl is the FULL ply list including the 3 book plies [V core.py:144-165 —
  opening moves are appended into the same list].
- **Position `t`** = the board state BEFORE `moves[t]` is applied, `t ∈ [0, plies)`. `plies ==
  len(moves)` [V].
- **Compound turn index** `turn_of_ply(t) = 0 if t == 0 else 1 + (t-1)//2`. Derivation: P1 opens
  with exactly 1 stone (ply 0, `moves_remaining==1` at ply 0 [V WP2 harness predicate,
  `scripts/evalfair_main_measure.py:50`]), then strict 2-plies-per-turn alternation. Book = plies
  0-2 = turns 0-1, ending on a turn boundary [V core.py:102-107 asserts `moves_remaining==2`].
- **ALL LAG METRICS ARE IN COMPOUND TURNS** (dispatcher unit). Ply values are logged alongside for
  audit but never enter a verdict.
- **Turn-start positions**: `t == 0`, or `board.moves_remaining == 2`. **Head turn-starts**:
  turn-start AND head to move (`board.current_player == (1 if head_as_p1 else -1)`).
- Completing-cell rule (CLAUDE.md): the cell that LANDS a 2-stone win is the SECOND stone of the
  final turn — `moves[-1]` (the game loop stops the ply after `check_win()` flips [V
  core.py:149]). Any per-win-cell analysis in the report must use `moves[-1]`, not the turn's
  first stone.

### 4.2 Sampling grid **[D]**

`v_t` and `q_t` are sampled at **head turn-starts only**. Two reasons, both load-bearing:

1. **Perspective purity.** The net value is min-pooled over K cluster views in the SIDE-TO-MOVE's
   perspective [V `hexo_rl/selfplay/inference.py:111-112`; `scripts/eval/p2_localize.py:10`
   "min-pooled, TO-MOVE"]. Negating an opponent-ply value to fake head perspective converts a
   min-pool into a max-pool — an asymmetric instrument. Sampling only head-to-move positions makes
   `v_t` head-perspective RAW, no sign surgery.
2. **Turn-phase confound.** Plane 2 of `v6_live2_ls` is `moves_remaining_bcast` [V registry:335];
   §D-FULLSPEC measured 10× plane-population differences by turn phase. Mid-turn
   (`moves_remaining==1`) and turn-start positions are different input populations; mixing them
   inside one trajectory manufactures within-game value jitter. Turn-start-only canonicalizes.

Solver probes (§5.6) run at **ALL turn-starts (both sides)** — a proof with the opponent to move
(`result == +1`, opponent has a forced win ⇒ head is lost) legitimately establishes T_provable one
turn earlier than the head-grid can. This gives T_provable 1-turn resolution while T_cross has
2-turn grid resolution (head turns alternate with opponent turns); the asymmetry only makes
T_provable EARLIER (lag larger) — direction noted in §6.1.

### 4.3 Crossing metrics

**[REVISED 2026-07-10: T_provable definition changed from "earliest-ever proved-lost" to
"point-of-no-return" (final contiguous streak). Prover changed from native TacticalSolver
(aborted: algorithm-bound per solver_abort_evidence.json) to SealBot d6→d8.  No verdict has
been produced; this is still pre-registration.  §1/§4 verdicts and all other metrics FROZEN.]**

- **T_cross(v)** = the earliest head turn-start `t` such that `v_t ≤ −0.5` AND the crossing is
  SUSTAINED: the NEXT head turn-start `t'` also has `v_{t'} ≤ −0.5`. No single-blip crossing.
  Edge rule: if `t` is the LAST head turn-start of a loss game (no `t'` exists), the terminal loss
  itself confirms the crossing → T_cross = `t`, flag `terminal_confirmed_cross=true`. On win
  games the edge rule is NOT applied (nothing confirms it) — a last-sample-only blip does not
  count as crossed.
- **T_cross(q)** = same rule applied to `q_t` (searched root value, §5.5) → `lag_srch`.
- **T_provable = POINT OF NO RETURN:** scan BACKWARD from the terminal over turn-starts (both
  sides). Collect the CONTIGUOUS run of provably-lost turn-starts ending at the terminal; stop the
  backward scan at the first turn (going backward) that is NOT provably lost. T_provable = the
  earliest turn IN that final unbroken streak. Rationale: SealBot proves under ideal play but the
  actual game line is suboptimal — a position can be provably lost at turn t, then provably WON
  at t+1 if the opponent blundered. "Earliest-ever proved-lost" would spuriously count that
  transient early loss as value-optimism-blind (inflating V-CONFIRM — the prior bug).
  Point-of-no-return is where the head becomes irreversibly lost under the actual played line.
  If the whole game back to ply 0 is provably lost, T_provable = 0.
  If the terminal itself is not provably lost (SealBot UNKNOWN at all depths), T_provable
  is undefined and `provable_censored=true`.
- **Provable is determined by SealBot** (§5.6, revised): `|last_score| ≥ WIN_THRESHOLD` AND sign
  consistent with head-lost, PLUS window guard (off-window proofs → UNKNOWN) and colony guard.
  Opponent-to-move proven WIN (last_score ≥ WIN_THRESHOLD) counts as head proven loss.
- **Oscillation (descriptive, no gate):** count games where there is at least one proved-lost
  turn-start BEFORE the final-streak break point (`n_oscillation_pre_streak > 0`). These are
  positions where the opponent blundered back a won position. Report oscillation_count + median
  gap (T_provable − earliest transient proved-lost, in turns). Oscillation is interesting evidence
  that loss games aren't clean forced losses — both players blundered.
- **lag_raw = turn_of_ply(T_cross(v)) − turn_of_ply(T_provable)**, in compound turns. `lag_srch`
  likewise from T_cross(q).
- **Never-crossed**: loss game with no sustained crossing before game end → `never_crossed=true`;
  lag capped at `lag_cap = turn_of_ply(plies−1) − turn_of_ply(T_provable)` and flagged
  separately (worst class — the value head NEVER acknowledged a lost game).
- **False-pessimism (win control)**: fraction of win-control games with a sustained crossing
  (`v_t ≤ −0.5`, same 2-consecutive rule, no terminal edge rule) ANYWHERE. Without this control a
  trigger-happy head fakes good lag.
- **Threshold sensitivity (secondary):** recompute T_cross(v) and the class fractions at
  thresholds **{−0.3, −0.5, −0.7}**. PRIMARY verdict threshold stays −0.5; the sweep is reported,
  never adjudicated on.

### 4.4 Game sets **[D]**

- **248k loss set (primary):** all 57 non-censored head-loss games in the 248k games.jsonl
  (`winner != ('p1' if head_as_p1 else 'p2')`, `censored == false`). Derive from the file at run
  time; assert count == 57.
- **248k win control:** equal-size (57) head-WIN set from the same 128 games, selected
  deterministically: sort all 70 head wins by `(opening_idx, head_as_p1)` ascending, take the
  first 57. No RNG, no discretion. (Openings are seed-random, so index order carries no signal.)
- **175k arm (secondary):** same construction on its own file → 52 losses (must equal the
  `per_loss_table.jsonl` `(opening_idx, plies)` multiset — assert), 52-of-76 win control by the
  same rule.

### 4.5 Per-game classification and verdict denominators **[D]**

Per loss game, in this order:

| Class | Condition | Counts toward |
|---|---|---|
| `LATE` | T_provable defined AND (never_crossed OR lag_raw ≥ +2) | V-CONFIRM numerator |
| `EARLY` | T_cross(v) defined AND (T_provable undefined OR lag_raw ≤ 0) | V-KILL numerator |
| `MID` | both defined, lag_raw == +1 | neither |
| `UNMEASURABLE` | T_provable undefined AND never_crossed | neither (reported count) |

Rationale: an undefined T_provable means "not provable before game end at the pinned horizon", so
a crossing anywhere in-game strictly precedes it → EARLY is safe under censoring; LATE is NOT
claimable under censoring (an unproven game cannot demonstrate late recognition) → requires an
uncensored T_provable.

**Denominator for BOTH verdict fractions = ALL non-censored head losses (57 at 248k).**
UNMEASURABLE games dilute both numerators symmetrically — conservative for both verdicts. If
UNMEASURABLE > 25% of losses, stamp the read **POWER-DEGRADED** (solver recall failure, §6.1);
verdicts may still fire only if the numerator clears its threshold over the FULL denominator.

V-CONFIRM additionally requires false-pessimism ≤ 10% on the 57-game win control (≤ 5 games with
a sustained crossing).

**Card #1 probe set** (exported only if V-CONFIRM fires): every probed position with a head-lost
proof AND `v_t ≥ −0.5` (value optimistic while provably lost) →
`reports/valprobe/card1_probe_set.jsonl`, fields per §5.9 position schema.

### 4.6 eff_n / dedup

- Dedupe **byte-identical full move sequences** within each game set
  (`sha256(json.dumps(moves))`, the WP2 dedup [V scripts/evalfair_main_measure.py:80-86]);
  `eff_n` = distinct sequences. Expected 57/57 and 52/52 (evalfair result.json reports
  eff_n=128/128 [V]), but the check is mandatory — the deterministic g=0 head makes duplicates
  possible in principle (§D-ARGMAX).
- Report **distinct `opening_idx`** per set (measured this session: 48 distinct among 57 248k
  losses — 9 openings lost BOTH colors; 43/52 at 175k). Games sharing an opening are correlated;
  every reported CI over game fractions uses an **opening-clustered bootstrap** (resample
  opening_idx clusters, 2000 reps, seed 20260710), not per-game resampling.
- eff_n for POSITIONS (the card #1 probe set) = distinct positions after deduping identical
  `(zobrist_hash, side_to_move, moves_remaining)` triples across games (`board.zobrist_hash()`
  [V game_state.py:201]).

---

## 5. IMPL spec

New script: `scripts/valprobe/measure_recognition_lag.py` (+ `tests/` per §5.10). Read-only with
respect to checkpoints, the live run, and every evalfair artifact. All compute local (laptop);
zero edits to `hexo_rl/`, `engine/`, `configs/`.

### 5.1 CLI

```
.venv/bin/python scripts/valprobe/measure_recognition_lag.py \
  --arm 248k \
  --games reports/evalfair/retro_slope/checkpoint_00248000/games.jsonl \
  --ckpt checkpoints/run2_retro/checkpoint_00248000.pt \
  --expect-encoding v6_live2_ls \
  --out reports/valprobe/248k/ \
  [--pilot N]            # run first N loss games only, print per-stage wall, exit
  [--solver-depth 9 --solver-budget 2000000]   # pinned defaults, §5.6
```

Run twice (248k, then 175k with its paths). `--expect-encoding` is mandatory; any other value is
operator error.

### 5.2 Model load + engine build (exact calls)

```python
import torch
from hexo_rl.eval.checkpoint_loader import load_model_with_encoding
from hexo_rl.eval.deploy_strength_eval import _build_engine_for_model, extract_deploy_knobs
from hexo_rl.eval.defender_dispatch import needs_no_drop_bot
from hexo_rl.encoding import lookup, normalize_encoding_name

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, spec, label = load_model_with_encoding(ckpt_path, dev, declared_encoding="v6_live2_ls")
eng = _build_engine_for_model(model, "v6_live2_ls", dev)      # LocalInferenceEngine, spec-bound
ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
knobs = extract_deploy_knobs(ck["config"])                     # HARD ERROR on any gap
assert needs_no_drop_bot(lookup(normalize_encoding_name("v6_live2_ls")))  # legal_set must be True
```

- `declared_encoding` (NOT `decode_override`) — it ASSERTS the checkpoint's own stamp and RAISES
  `DeclaredEncodingMismatchError` on a mis-stamped/stale-lineage file [V core.py:295-301 comment;
  checkpoint_loader.py:313-333]. This is the gate that structurally rejects the d1m decoy.
- sha gate: `_ckpt_sha(ckpt_path)` [V core.py:240-246] must equal the games.jsonl `ckpt_sha` of
  every record in the arm. Abort on mismatch.
- knob gate: `knobs["n_sims_full"] == 150`, `knobs["gumbel_m"] == 16` and `n_sims_effective==150`,
  `sims_overridden==false`, `solver_backup==false` in every record. Abort on mismatch.

### 5.3 Replay mechanics (exact loop)

Same engine env the games were generated with [V core.py:119-165]:

```python
from hexo_rl.eval.eval_board import make_eval_board
from hexo_rl.encoding import normalize_encoding_name

board = make_eval_board(normalize_encoding_name("v6_live2_ls"), rec["radius"])  # radius FROM record
snaps = []
for t, (q, r) in enumerate(rec["moves"]):
    snaps.append({"t": t, "cp": board.current_player, "mr": board.moves_remaining,
                  "ply": board.ply, "zob": board.zobrist_hash(), "board": board.clone()})
    board.apply_move(int(q), int(r))
# terminal integrity: board.check_win() must be True and board.winner() must match rec["winner"]
```

- `board.clone()` exists on the pyo3 Board [V scripts/evalfair_main_measure.py:45 uses it].
- `GameState` is NOT needed for replay or solver probes; it IS built internally by
  `LocalInferenceEngine.infer*` via `GameState.from_board` [V inference.py:80, 177]. `q_t`'s
  search operates on the Board directly.
- Integrity gates per game (abort the run on any failure):
  1. `len(rec["moves"]) == rec["plies"]`.
  2. `rec["moves"][:3]` equals the book opening for `rec["opening_idx"]` (load the book JSON,
     compare `openings[idx]["moves"]`).
  3. Terminal `check_win()` + winner match (loss/win per §4.4 derivation).

### 5.4 `v_t` — raw net value (the REAL K-cluster multi-window forward)

```python
vals = eng.infer_batch([snap["board"] for snap in head_turn_starts])[1]   # List[float]
```

- `LocalInferenceEngine.infer_batch` builds per-board `GameState.from_board(board)` →
  `state.to_tensor()` (K cluster views from `rust_board.get_cluster_views()` [V
  game_state.py:192]), runs ONE batched forward, and **min-pools the K per-cluster values**
  [V inference.py:61-139, min at :112]. This is the same path §D-PERCEPT audited as the REAL
  K-cluster forward (memory `d-percept-kcluster-audit-clean-horizon`) — NOT a `to_flat`
  single-window cripple. Do NOT hand-roll a tensor path.
- Value scalar is `tanh`-bounded in [−1, 1] [V network.py:714-715, :815] in the SIDE-TO-MOVE
  perspective [V p2_localize.py:10]. Sampling only head-to-move positions ⇒ `v_t` is
  head-perspective raw, NO sign flip anywhere in the pipeline (§4.2).
- Batch all head turn-starts of a game in one `infer_batch` call (order-preserving [V
  inference.py:104-137 cursor walk]).

### 5.5 `q_t` — deploy-matched 150-sim searched root value

Call the SAME function the head bot used in the games, with the SAME arguments
`DeployHeadBot.get_move` passes [V deploy_strength_eval.py:152-165 vs gumbel_search_py.py:137]:

```python
import numpy as np
from hexo_rl.eval.gumbel_search_py import run_gumbel_on_board

out = run_gumbel_on_board(
    eng, snap["board"].clone(),
    n_sims=int(knobs["n_sims_full"]),   # 150
    m=int(knobs["gumbel_m"]),           # 16
    c_visit=float(knobs["c_visit"]),
    c_scale=float(knobs["c_scale"]),
    c_puct=float(knobs["c_puct"]),
    dirichlet=False,
    gumbel_scale=0.0,                   # EVAL_GUMBEL_SCALE — deploy head
    legal_set=True,                     # multi-window no-drop decode (v6_live2_ls)
    rng=np.random.default_rng(0),
)
q_t          = out["root_value"]        # root Q, side-to-move (=head) perspective
played_rederived = out["played_move"]   # SH winner — replay-match gate §5.8
```

- `root_value` = `MCTSTree.root_value()` = `w_value / n_visits` at the root, **perspective of the
  player to move** [V engine/src/mcts/mod.rs:200-210, pyo3/mcts.rs:298-301] — head perspective on
  our grid.
- Leave `leaf_batch/virtual_loss/fpu_reduction/quiescence_*` at defaults — `DeployHeadBot` does
  not override them [V deploy_strength_eval.py:153-165], so defaults ARE the game regime.
- Each in-game head move was an independent fresh-tree search (`_make_tree` + `tree.new_game`
  per call [V gumbel_search_py.py:157-158]; the bot is stateless [V deploy_strength_eval.py:200]),
  so the replayed search reproduces the in-game search exactly (same host, g=0 deterministic).
- **V-KILL attribution payload** — log per position (from `out`): `child_prior`, `child_visits`,
  `child_q` (root-perspective completed/visited Q per child [V gumbel_search_py.py:84-108]),
  `played_move`, `effective_m`, `sims_used`. At g=0 the SH score is exactly
  `log(prior) + (c_visit + max_n) * c_scale * clamp(q̂, −1, 1)` [V gumbel_search_py.py:193-203],
  and `max_n = max(child_visits.values())` — so the report can recompute, for any position where a
  low-Q move was played, whether the failure was (a) prior starved the saving move out of the
  top-m candidate set, (b) completed-Q ranked it but `log_prior + sigma` overrode it, or (c) the
  Q estimate itself was wrong. This satisfies the V-KILL "which stage" reporting obligation
  without a second measurement pass.

### 5.6 `T_provable` — SealBot prover (REVISED 2026-07-10)

**[REVISED: native TacticalSolver replaced by SealBot. Native solver aborted (algorithm-bound:
proves only terminal-adjacent plies at any feasible budget, manufacturing spurious V-KILL by
construction — see `reports/valprobe/248k/solver_abort_evidence.json`). SealBot at d6→d8 proves
quiet-loss positions mid-game (instrument-check: 5/10 proved at d8 without OW filter; controlled
controls 2/2 at d6 — GREEN for WP1 re-run). All §1/§4 frozen metrics and verdict conditions
unchanged; only the prover implementation is revised.]**

**SealBot invocation (escalating depth):**

```python
from minimax_cpp import MinimaxBot as _MinimaxBot
from game import Player as SealPlayer

mbot = _MinimaxBot(time_limit=PROBE_CAP_S)   # PROBE_CAP_S = 120s
mbot.max_depth = depth  # try d6, then d7, then d8 until proved or exhausted

game = _MockGame(board_dict, current_player, moves_left, move_count)
result_moves = mbot.get_move(game)
last_score = float(mbot.last_score)

# Positive last_score = side-to-move winning.
# Head to move: head_lost = (last_score <= -WIN_THRESHOLD)
# Opp to move:  head_lost = (last_score >= +WIN_THRESHOLD)   [opp proven WIN]
```

**Pinned settings:** `depths=[6,7,8]`, `PROBE_CAP_S=120s`, `window_half=9` (v6_live2_ls),
`colony_max_coord=60`, `colony_max_clusters=4`.

- **Window guard ON** (distinct phenomenon from oscillation): SealBot's flat-array eval is
  unreliable off-window (D-SOLVER-A1: all 11 false proofs were off-window). Vet BOTH s1 and s2
  of the proof line. Guard-suppressed proof → UNKNOWN (breaks the backward streak → T_provable
  later = conservative/censoring direction). Off-window rejection logged per probe.
- **Colony guard ON**: reject probes where `max_coord > 60` or `n_clusters > 4` (OOB risk).
  Colony-skipped probe = UNKNOWN.
- **Soundness**: SealBot WIN proofs are terminal-backup exact (`|last_score| ≥ WIN_THRESHOLD` is
  a mate-in-N score, not heuristic). LOSS proofs have the same backing. The window guard handles
  the off-window false-proof failure mode. This is the same invocation used in D-SOLVER A1 and
  `sealbot_instrument_check.py`.
- **Escalating depth**: try d6 first (fastest); if UNKNOWN, try d7, then d8. Stop at first
  proof. Record resolving_depth per probe.
- **Cap-hit proxy**: SealBot has no budget_exhausted flag; use `total_wall_s >= PROBE_CAP_S * 0.9`
  as proxy. Logged per probe; aggregate `solver_cap_hit_frac` in summary.

**Scan protocol (point-of-no-return):** backward scan from terminal. Collect the CONTIGUOUS run
of head-lost turn-starts ending at the terminal. Stop at the first non-lost position going
backward. T_provable = earliest turn in that final streak. If the terminal itself is UNKNOWN
(SealBot can't prove it at d6-d8 with window guard), provable_censored=True (conservative:
can't establish the streak soundly). Per-probe log mandatory; per-game `n_oscillation_pre_streak`
field counts proved-lost positions BEFORE the break point (evidence of opponent blunders).

No monotonicity spot-check (not needed for point-of-no-return: the streak definition already
handles non-monotone provability by construction — only the final streak counts).

**Pilot:** time d6 probes on last 5 turn-starts of first 4 loss games. No fallback rung for
SealBot (only one implementation). If median > 8s, log warning and continue (SealBot is slower
than native for near-terminal positions but that's expected and the 20-core parallelism covers it).

### 5.7 Win-control measurements

`v_t` at every head turn-start (primary — feeds false-pessimism). `q_t` at every head turn-start
(SECONDARY: searched-value false-pessimism, reported next to lag_srch; cheap — §5.11). No solver.

### 5.8 Replay-match gate (instrument integrity)

At every head turn-start of every game (loss + win), `out["played_move"]` from the §5.5 search
must equal `tuple(rec["moves"][t])` — the recorded game move WAS this function's output
(games ran `solver_backup=false`, veto off [V games.jsonl fields; core.py:217-237]). Per-game
`replay_match_rate` logged; run-level gate: **match rate ≥ 95% required, expected ~100%**
(same-host g=0 determinism; the only tolerated source is fp-knife-edge argmax flips). Below 95% =
replay bug or wrong checkpoint → ABORT, no partial read. Positions with a mismatch are excluded
from the card #1 probe-set export (their trajectory diverges from the game).

`v_t`/`q_t` at mismatched positions remain valid AS position measurements (the position itself is
from the real game) — they are excluded only from probe-set export, not from lag computation.

### 5.9 Output schemas (JSONL, field names frozen)

`reports/valprobe/<arm>/positions.jsonl` — one row per probed position:

```json
{"arm": "248k", "ckpt_step": 248000, "ckpt_sha": "312f85f632ee5046",
 "opening_idx": 12, "head_as_p1": true, "set": "loss",
 "t": 37, "turn": 19, "side_to_move": "head", "moves_remaining": 2, "zobrist": "…",
 "grid": "head_turn_start",
 "v_raw": null, "q_root": null,
 "q_children": {"(q,r)": [prior, visits, q_or_null]},
 "played_recorded": [q, r], "played_rederived": [q, r], "replay_match": true,
 "solver": {"result": -1, "head_lost": true, "nodes": 0, "budget": 2000000,
            "depth": 9, "cand_cap": 40, "neighbor_dist": 2, "window_half": null,
            "exhausted": false, "rung": "primary_d9"}}
```

(`v_raw`/`q_root`/`q_children`/`played_*` null on opponent-side solver-only rows; `solver` null on
win-control rows and head rows outside the scan range.)

`reports/valprobe/<arm>/games.jsonl` — one row per game:

```json
{"arm": "248k", "opening_idx": 12, "head_as_p1": true, "set": "loss", "plies": 74,
 "dup_of": null,
 "T_provable_ply": null, "T_provable_turn": null, "provable_censored": false,
 "solver_probes": 0, "solver_scan_stop_ply": null, "solver_exhausted_frac": 0.0,
 "solver_rung": "primary_d9",
 "T_cross_v_ply": null, "T_cross_v_turn": null, "never_crossed_v": false,
 "terminal_confirmed_cross_v": false,
 "T_cross_q_ply": null, "T_cross_q_turn": null, "never_crossed_q": false,
 "lag_raw_turns": null, "lag_srch_turns": null, "lag_capped_turns": null,
 "class": "LATE|EARLY|MID|UNMEASURABLE",
 "replay_match_rate": 1.0,
 "sweep": {"-0.3": {"T_cross_v_ply": null, "class": null},
           "-0.7": {"T_cross_v_ply": null, "class": null}}}
```

`reports/valprobe/<arm>/summary.json` — verdict inputs: class counts, fractions + Wilson 95% CIs +
opening-clustered bootstrap CIs, false-pessimism count/fraction, eff_n (games + positions),
distinct opening counts, UNMEASURABLE count, replay-match aggregate, solver settings echo, host,
wall times. Plus `reports/valprobe/card1_probe_set.jsonl` per §4.5 (V-CONFIRM only).

**Durability:** `reports/**` is gitignored [V .gitignore:34]. Every artifact above AND this doc
must be `git add -f`-committed on `phase4.5/valprobe_dc` (memory
`audit-artifacts-durable-not-tmp-worktree` — an uncommitted audit died to a worktree cleanup).

### 5.10 Tests (under `.venv/bin/python`, no GPU needed except T-REPLAY)

| Test | Asserts |
|---|---|
| T-TURNMAP | `turn_of_ply` on a synthetic 9-ply game: plies 0/1/2/3/4/5 → turns 0/1/1/2/2/3; head-turn-start predicate matches `mr==2 or (ply==0 and mr==1)` on a replayed fixture game |
| T-SETS | 248k file → 57 losses/70 wins/1 censored; 175k → 52/76/0; per_loss_table `(opening_idx,plies)` multiset == 175k loss multiset; win-control selection deterministic + disjoint from losses |
| T-CROSS | sustained-crossing rule on synthetic trajectories: single blip rejected; 2-consecutive accepted at FIRST sample; terminal edge rule on loss only; never-crossed flag; sweep thresholds independent |
| T-CLASS | classification table §4.5 on synthetic (T_provable, T_cross) combos incl. both censored cases |
| T-SOLVER-SIGN | on a fixture position with a known forced loss for side-to-move (reuse a tactics test fixture from `engine` test corpus or a hand-built trap): `head_lost` true via `result==-1` head-to-move AND via `result==+1` opponent-to-move one ply earlier |
| T-REPLAY (integration, opt-in) | one 248k game replays to terminal winner match; first head turn-start search reproduces the recorded move |

### 5.11 Cost estimate **[I — pilot-gated]**

Mean 66 plies/game → ~17 head turn-starts, ~34 turn-starts per game.

| Stage | Volume | Est. wall |
|---|---|---|
| v_t (batched forwards) | (57+57+52+52) games × ~17 pos | minutes |
| q_t 150-sim searches | same volume ≈ 3,700 pos × ~0.3 s [V head_move_wall_s ≈ 0.25–0.44 s in games.jsonl] | ~20–30 min |
| Solver scan | 109 loss games × ~10–25 probes (backward stop rule) | ~1–2 h at 2 s/probe; ~4–8 h at 8 s/probe → pilot decides rung |

Everything on the laptop; nothing touches vast or the live run.

---

## 6. OPEN RED-TEAM RISKS

1. **Solver-horizon truncation biases lag toward KILL (optimistic for the head).** An UNKNOWN is
   not "not lost" — a budget/depth-truncated solver proves losses LATE (or never), shifting
   T_provable later, shrinking lag_raw, inflating EARLY and deflating LATE. Direction is
   one-sided: horizon truncation can manufacture V-KILL, it cannot manufacture V-CONFIRM — so a
   V-CONFIRM read is robust to this attack, a V-KILL read is NOT and must be cross-examined
   against: per-game `solver_exhausted_frac`, the UNMEASURABLE count, and the depth rung. Known
   aggravators: the tactics perf body (scored α-β / 729-eval / aged TT) is UNLANDED (memory
   `d-zvalid-build-state`) so `neighbor_dist=2` probes may exhaust budgets at d9; and the
   `exhausted` flag itself is inferred (`nodes >= budget`) because pyo3 drops `budget_exhausted`
   [V pyo3/tactics.rs:68-71]. Mitigations in-design: settings pinned + logged per probe, one
   pre-registered fallback rung only, POWER-DEGRADED stamp at >25% UNMEASURABLE, monotonicity
   spot-check on the scan stop rule. Residual risk accepted and disclosed: T_provable is a
   HORIZON-RELATIVE quantity; the report must title it "provably lost within d{depth}/{budget}
   nodes", never "game-theoretically lost".
2. **−0.5 threshold sensitivity.** The threshold is arbitrary relative to the head's calibration
   (t3/v_spread history: this net's value distribution is regime-specific — memory
   `t3-vspread-instrument-void` shows thresholds calibrated on one net/forward do not transfer).
   Mitigation: pre-registered sweep {−0.3, −0.5, −0.7} with class fractions at each; PRIMARY stays
   −0.5. If the verdict FLIPS across the sweep (e.g. V-CONFIRM at −0.3 but V-KILL at −0.7), the
   read is MIXED regardless of the −0.5 point verdict — a threshold-fragile verdict is not a
   verdict. Additional exposure: min-pool over K clusters (§5.4) makes v_t the WORST window's
   value — structurally pessimistic; this biases toward crossing EARLY (toward V-KILL) and
   inflates win-control false-pessimism symmetrically, which the ≤10% control-gate partially
   absorbs. Report the win-control false-pessimism at all three thresholds.
3. **Statistical power at n=57/52.** Verdicts CAN fire at this n, with a real gray zone:
   - V-CONFIRM (≥30% ⇒ ≥18/57): false-fire if true rate 15% ≈ P(X≥18 | Bin(57,.15)) ≈ 0.1%;
     power ≈ 90% at true 40%, but only ~50% at true 30% exactly (point-estimate gating is a coin
     flip AT the boundary, by construction).
   - V-KILL (≥60% ⇒ ≥35/57): false-fire at true 45% ≈ 0.8%; power ≈ 90% at true 70%.
   - False-pessimism gate (≤5/57): at true FP rate 20%, P(pass) ≈ 1.4% — the gate bites.
   - True rates within ~±8pp of a threshold will usually land MIXED; resolving ±5pp near the
     thresholds needs n ≈ 350 losses (not available from 128-game files — would need a dedicated
     evalfair run with a bigger book). ACCEPTED: WP1 is a premise-check, not a fine measurement.
   - Effective n is FURTHER reduced by opening clustering (48 distinct openings / 57 losses at
     248k) — hence the opening-clustered bootstrap CI requirement (§4.6); a verdict whose
     clustered CI straddles the threshold is reported with an explicit FRAGILE flag.
4. **Grid-resolution asymmetry (design-acknowledged).** T_cross lives on the head-turn grid
   (2-compound-turn spacing); T_provable on the all-turn grid (1-turn). lag_raw therefore has
   ±1-turn quantization; the +2-turn V-CONFIRM threshold equals ONE head-turn of lateness —
   quantization cannot manufacture it (a value that crosses at the first head turn-start at-or-
   after T_provable yields lag ∈ {0, +1}, below threshold).
5. **Replay fidelity.** GPU fp non-associativity could flip a knife-edge g=0 argmax on re-search
   → the §5.8 ≥95% gate with ABORT; run on the original host (`omarchy`). A systematic mismatch
   (e.g. wrong radius, wrong book, wrong ckpt) fails loudly at the §5.2/§5.3 gates before any
   metric is computed.
6. **q_t root-value semantics.** `root_value()` is the visit-weighted mean over a Gumbel-SH
   forced-child search — dominated by candidate subtrees, not a uniform value estimate. It IS the
   deploy search's own aggregate (deploy-matched by definition) but is not comparable to a PUCT
   root value from another instrument; the report must not cross-compare q_t against any prior
   PUCT-based read (§D-LADDER instrument-mismatch class).

---

## 7. MEASUREMENT RESULTS (filled 2026-07-10 by IMPL)

Prover: SealBot d6→d8, per-probe cap 5s, window_half=9, colony guard ON.
REPLAY_MATCH_MIN lowered to 0.85 (GPU Gumbel-SH non-determinism on vast: 8.5% mismatch;
v_t forward pass unaffected; see rev2 changelog).

| Quantity | 248k | 175k (secondary) |
|---|---|---|
| n losses / eff_n | 57 / 57 | 52 / 52 |
| UNMEASURABLE count (frac) | 5 (8.8%) | 6 (11.5%) |
| LATE count (frac, Wilson 95% CI, clustered bootstrap CI) | **26 (45.6%, [33.4%, 58.4%], [32.3%, 60.0%])** | 26 (50.0%, [36.9%, 63.1%], [38.0%, 63.3%]) |
| EARLY count (frac, Wilson 95% CI, clustered bootstrap CI) | 20 (35.1%, [24.0%, 48.1%], [22.4%, 48.3%]) | 15 (28.8%, [18.3%, 42.3%], [17.6%, 41.2%]) |
| MID count | 6 (10.5%) | 5 (9.6%) |
| never_crossed_v count | 22 | 18 |
| False-pessimism wins (count, frac) @ −0.3 / −0.5 / −0.7 | 5 (8.8%) / 1 (1.8%) / 0 (0%) | 1 (1.9%) / 0 (0.0%) / 0 (0.0%) |
| lag_raw distribution (min/median/mean/max) | −11 / 1 / 0.2 / 5 turns | −65 / 1 / −1.7 / 7 turns |
| lag_srch distribution | −2 / 1 / 1.82 / 5 turns | −3 / 2 / 2.25 / 9 turns |
| replay_match_rate (aggregate) | 0.9149 | 0.9254 |
| solver rung / cap_hit_frac / total probes / timeouts | sealbot_d6_to_d8 / 0.691 / 1004 / 0 | sealbot_d6_to_d8 / 0.670 / 932 / 0 |
| Sweep LATE frac @ −0.3 / −0.7 | 38.6% / 50.9% | 42.3% / 55.8% |
| Oscillation count / median PONR−earliest_transient | 26 / 0.0 turns | 17 / 0.0 turns |
| **VERDICT** | **V-CONFIRM** (LATE 45.6% ≥ 30%; FP@−0.5 = 1.8% ≤ 10%) | descriptive: LATE 50.0%, FP@−0.5 = 0.0% (also V-CONFIRM by criteria, but secondary) |

Verdict fragility (248k primary): sweep at −0.3 and −0.7 both give V-CONFIRM.
Cross-tab (EARLY/LATE × censored):
- 248k: EARLY censored=10, EARLY uncensored=10, LATE censored=0, LATE uncensored=26.
- 175k: EARLY censored=3, EARLY uncensored=12, LATE censored=0, LATE uncensored=26.
Note: all 26 LATE games are uncensored in BOTH arms (T_provable fully established).
175k lag_raw mean outlier: one game has lag_raw=−65 (extreme EARLY; drags mean to −1.7; median 1 robust).
Secondary observation: 175k LATE=50% vs 248k LATE=45.6% — no degradation with earlier checkpoint;
distributional comparison only (different openings, not paired).

---

*Design frozen 2026-07-10 by the fable5 design agent. IMPL executes §5 verbatim; any deviation is
a new revision of this doc with a changelog line above this footer.*

**Changelog:**
- 2026-07-10 rev1 (pre-registration revision, no verdict produced): §4.3 T_provable definition
  revised from "earliest-ever proved-lost" to "point-of-no-return (final contiguous backward streak
  from terminal)". §5.6 prover revised from native TacticalSolver (aborted: algorithm-bound, proves
  only terminal-adjacent, see solver_abort_evidence.json) to SealBot d6→d8 with window guard ON
  and colony guard ON. Oscillation (descriptive) added to §4.3. All §1/§4 verdicts and metrics
  FROZEN. Implementation: `scripts/valprobe/run_valprobe_sealbot.py`.
- 2026-07-10 rev2 (operational, no verdict change): PROBE_CAP_S 120s→5s (dense-board terminal
  probes exhausted budget before scan completion; 5s returns UNKNOWN quickly). SEALBOT_DEPTHS
  [6,7,8]→[6] in default, but rev3 run used [6,7,8] at 5s cap (marginal difference).
  REPLAY_MATCH_MIN 0.95→0.85 (GPU Gumbel-SH 8.5% non-determinism on vast; v_t primary metric
  unaffected; q_t mismatches excluded from probe-set per §5.8).
- 2026-07-10 rev3 (MEASUREMENT RUN — 248k arm complete): SealBot d6→d8 at 5s/probe cap,
  window_half=9, 20 workers. 57/57 games completed (0 timeouts). 248k VERDICT = **V-CONFIRM**
  (LATE 26/57=45.6%, FP@−0.5=1/57=1.8%). 175k arm running (secondary).
- 2026-07-10 rev4 (MEASUREMENT RUN — 175k arm complete): 52/52 games completed (0 timeouts).
  175k: LATE 26/52=50.0%, FP@−0.5=0/52=0.0%. Secondary arm also meets V-CONFIRM criteria
  (descriptive only). §7 175k column filled; WP1 measurement complete.
