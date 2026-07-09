# EVALFAIR instrument — design (D-A / WP1)

**Status:** DESIGN. Authored in the main loop (the fable5 design agent died on an API limit;
substrate was already fully in-context). IMPL is a follow-on sonnet5 dispatch gated on this doc.
**Date:** 2026-07-09. **Run under read:** run2_mw_fresh LIVE on vast — this instrument touches
NONE of it; all compute on the laptop (8845HS + RTX 4060 Max-Q) against banked checkpoints.

Legend: **[V]** verified against source this session (file:line cited). **[I]** inferred/estimated.
An unmarked factual claim is a defect.

The purpose is ONE reusable offline strength instrument that replaces the temp-0.5 PUCT proxy
(`ModelPlayer`, `eval_temperature=0.5`, 128 sims — the §D-LADDER "triple miss") for every offline
read: WP2 loss-harvest, WP3 sims ladder + solver arm, WP4 retro slope, WP5 first-turn read. It is
the productionized generalization of the VERIFIED verdict-2 probe
(`scripts/watchguard/verdict2_opening_line_probe.py`, committed `9aab184`), which already ran 130
games in 1979 s and produced WR 0.594, CI [0.508, 0.672] over 64 pairs.

---

## 0. The one measured fact that recosts the WP

verdict-2 ran **130 games in 1979 s = 15.2 s/game** [V `reports/watchguard/verdict2/result.json`
`wall_sec=1979.05`, `n=130`] at SealBot **d5** + Gumbel-150, r4, on this laptop. The
`retro_slope_DESIGN.md` §7 estimate of ~90 s/game is **wrong by 6×** (it guessed SealBot d5 at
50–105 s/game; the measured cost is 15.2 s/game *total*). Every wall estimate below is rebuilt from
15.2 s/game. This alone flips the WP4 feasibility verdict from "2 overnight shards" to "one ~8 h
serial run, ~2 h at 4 workers."

---

## 1. Module layout

New package `scripts/evalfair/` (thin CLI drivers) over an importable core. Zero edits to
`hexo_rl/`, `engine/`, `configs/`, the live run, or the in-loop eval — the whole instrument reuses
existing machinery (§5).

```
scripts/evalfair/
  __init__.py
  core.py         # play/pair/score — the verdict-2 loop, lifted verbatim (§4)
  book.py         # book_v2 generation + load/verify (§3)
  run_eval.py     # single ckpt, single arm  -> games.jsonl + result.json  (WP3 unit, WP5)
  retro_slope.py  # WP4 driver: N ckpts x book -> per-ckpt JSON + slope     (refuses --override-n-sims)
  sims_ladder.py  # WP3 driver: sims arms + solver arm on one ckpt
  tests/
    test_book_determinism.py
    test_radius_fold.py
    test_pairing_from_records.py
    test_sims_override_refusal.py
    test_mismatch_stamp_raises.py
    test_worker_invariance.py
```

**Interpreter [V]:** the native `TacticalSolver` (WP3 solver arm) and the engine live in
`.venv/lib/python3.14/site-packages/engine/engine.cpython-314-*.so`; the bare mise `python` (3.14)
has NO `engine.engine` and `from engine import TacticalSolver` raises `ModuleNotFoundError`. Every
driver + test runs under `.venv/bin/python` (Makefile `PY=.venv/bin/python:4-5` [V]). Confirmed
this session: `.venv/bin/python -c "from engine import TacticalSolver"` binds; bare `python` fails.

**Relationship to the verdict-2 file.** The dispatcher says "extend it, do NOT rewrite; it is the
verified instrument." It is also committed and its result is a FROZEN anchor. Resolution: the
core functions (`build_book`, `play_from_opening`, `suffix_key`, `bootstrap_mean`) are lifted
**byte-for-byte** into `scripts/evalfair/core.py` and `book.py`; the acceptance test proves the
lift is faithful by reproducing `book_v1` (sha256[:16] `a1763be0c32ab4c1` over the sorted dump [V
`meta.json`]) AND the 175k WR 0.594 (§10 test T-ANCHOR). `verdict2_opening_line_probe.py` stays on
disk unchanged as the frozen reference; it is NOT re-pointed at the new module (re-editing a
committed anchor to import a fresh module is a needless risk — the determinism test is the
guarantee that they compute the same thing).

### Core signatures (`core.py`)

```python
HEAD = "head"; OPP = "sealbot"; MAX_PLIES = 200

def radius_from_checkpoint(ck: dict) -> Optional[int]:
    # drive the REAL StepCoordinator._resolve_radius over ck["config"], ck["step"]

def sealbot_depth_from_config(path="configs/eval.yaml") -> int:  # -> 5

def play_from_opening(p1_bot, p2_bot, p1_label, p2_label, encoding, radius,
                      opening: Sequence[Sequence[int]]) -> dict:
    # replay opening stones verbatim, then bots alternate; record full move list,
    # per-move wall times (head vs sealbot), sealbot search times, head_fired, censored

def suffix_key(game: dict, n_open: int) -> tuple  # post-opening move suffix (eff_n)

def bootstrap_mean(vals, n_boot, seed) -> tuple[float, float, float]  # pair-level CI

@dataclass(frozen=True)
class ArmSpec:
    label: str
    n_sims_override: Optional[int] = None   # WP3 only; None = deploy-matched
    solver_backup: bool = False             # WP3.3
    window_half: Optional[int] = 9          # v6_live2_ls in-window band
    first_turn_book: Optional[list] = None  # WP5 arm (a)

def make_head_bot(eng, knobs, arm: ArmSpec, legal_set: bool) -> BotProtocol:
    kb = dict(knobs)
    if arm.n_sims_override is not None:
        kb["n_sims_full"] = int(arm.n_sims_override)
    head = DeployHeadBot(eng, kb, label=HEAD, seed=0, legal_set=legal_set)
    if not arm.solver_backup:
        return head
    return SolverBackupBot(head, probe_engine="native", window_half=arm.window_half,
                           depth=6, cand_cap=40, node_budget=200_000)

def run_arm(ckpt, arm: ArmSpec, book, *, out_dir, workers=1, n_boot=2000, book_seed) -> dict:
    # loads model via gated loader, resolves radius+knobs from ckpt, F6 knob-drift abort,
    # plays 64 pairs (colors swapped), streams games.jsonl, returns result dict
```

`run_gumbel_on_board` accepts `n_sims` directly [V `deploy_strength_eval.py:194`
`n_sims=self._n_sims`], and `DeployHeadBot.__init__` reads `self._n_sims = int(knobs["n_sims_full"])`
[V `:150`], so the sims override is a single dict overwrite (§5) — no engine change.

---

## 2. The book contradiction — ADJUDICATED

**The conflict.** The dispatcher orders one book **per radius stage** ("r4, r5, r6 stages exist
across run2 ckpts … one book per stage … openings sampled INSIDE that stage's radius"). The prior
`retro_slope_DESIGN.md` §2.3–2.4 / R4b explicitly **REJECTS** per-stage books in favour of a SINGLE
r4-sampled book used byte-identically across all checkpoints (monotone-permissivity: r4-ball ⊂
r5-ball, so an r4 book is legal + in-distribution for every checkpoint).

**Adjudication — ADOPT per-stage books (the dispatcher), and here is why the prior rejection does
not bind.** The retro design's rejection rests on its R2 disqualification: sampling *one* book at
r5 and running r4-*trained* checkpoints on it manufactures a slope (early ckpts weak on shell cells
they never trained on). But per-stage-book-**matched-to-ckpt** is a different construction: every
checkpoint reads a book sampled at **its own training radius** — r4 ckpts on the r4 book, r5 ckpts
on the r5 book. **No checkpoint is ever read off its own training distribution.** R2's manufactured
slope requires a cross-stage book mismatch; matched per-stage books have none.

The retro design's R4a (one r4 book everywhere) was solving *cross-stage comparability* — but the
FROZEN verdict 3 forbids any cross-stage slope comparison anyway ("no cross-instrument slope
comparisons, ever" [V `retro_slope_DESIGN.md` §10]). Cross-stage book identity therefore buys
**nothing** for the slope, and it carries a real cost: under R4a the r5-stage checkpoints read on
an r4-narrower opening book — off-training-regime for Series B, which the retro design itself flags
(R1) as UNDERSTATING later checkpoints → biasing Series B toward plateau *at exactly the live
frontier where run2 now sits*. Per-stage books eliminate that bias. **Per-stage is strictly better
for Series-B validity; the loser (R4a single r4 book) biases Series B toward false-plateau.**

Slope is read PER-STAGE regardless (Series A = r4 <200k, Series B = r5 ≥200k, never spliced across
the 200k boundary). Within a stage the book is constant → any book choice is a level offset, never a
within-series slope bias. So the adjudication changes only Series-B *typicality*, not Series-A (which
is byte-identical either way).

### The 175k anchor (WP4-mandated)

175k is step 175 000 < 200 000 → radius 4 [V schedule `run2_mw_fresh.yaml:94-98` `{0:4, 200000:5,
400000:6, 600000:8}`]. So 175k reads the **r4-stage book**. **Design guarantee: the r4-stage
book's `openings[i].moves` are byte-identical to `book_v1`** — same sampler (`core.build_book`
lifted verbatim), same seed `20260709`, same 64 openings, same 3-ply turn-clean structure. The
book_v2 wrapper adds `book_id`/`radius_stage`/per-opening `id` fields but the `moves` arrays are
untouched; a sha256 over just the concatenated `moves` reproduces `a1763be0c32ab4c1`. Therefore the
WP4 175k point byte-reproduces verdict-2's 0.594 and NO both-books run is needed. Test T-ANCHOR
(§10) enforces this; if a future sampler change breaks it, the fallback is the dispatcher-authorised
both-books 175k run, reported side by side.

### book_v2 format (tracked fixtures — `reports/**` is gitignored [V `.gitignore:34`])

```json
{
  "book_id": "evalfair_r4_v2",
  "seed": 20260709,
  "radius_stage": 4,
  "sampler_commit": "<git sha of core.build_book at gen time>",
  "openings": [
    {"id": 0, "moves": [[q,r],[q,r],[q,r]], "rng_seed": null},
    ...
  ]
}
```

- **One book per stage:** `evalfair_r4_v2` (seed 20260709, radius_stage 4, 64 openings — moves ≡
  book_v1), `evalfair_r5_v2` (seed **20260710**, radius_stage 5, 64 openings sampled on an r5
  board). r6/r8 books are generated on demand when run2 crosses 400k/600k (not needed now — latest
  banked is 248k).
- Committed to `tests/fixtures/opening_books/` (same tracked dir as `book_v1.json`, committed in
  WP0).
- **`rng_seed` per opening: DROPPED (set `null`).** Both bots are deterministic — DeployHeadBot is
  g=0 (`EVAL_GUMBEL_SCALE=0.0`, RNG × 0 at the root [V `deploy_strength_eval.py:53`, `:200-205`
  comment "MULTIPLIED BY ZERO"]) and SealBot is fixed-depth with a cold TT per game. A game is a
  pure function of (ckpt, opening stones, color); a per-opening play-time seed has **no consumer**.
  The field is retained as an explicit `null` for format-forward-compat only and MUST NOT be read
  by the player (documenting the §9-F1 `hash()`-seed trap: no seed is ever derived at play time).
- **Turn-clean [V]:** book = 2 turns = 3 plies (P1 opener 1 ply, P2 compound turn 2 plies), ending
  on a turn boundary asserted `moves_remaining == 2` [V `verdict2_opening_line_probe.py:112`]. A
  1-turn book has ≤25 distinct openings (fixed 5×5 empty-board legal set,
  `engine/src/board/moves.rs:95-101` [I — per retro_slope_DESIGN §9-F2]) < 64. Red-team F2 honoured.

---

## 3. Play + scoring

- **Head side:** `DeployHeadBot` — Gumbel-SH winner, `gumbel_scale=0.0`, Dirichlet OFF, no
  temperature, multi-window legal-set decode `legal_set=needs_no_drop_bot(spec)` (True for
  v6_live2_ls) [V `deploy_strength_eval.py:109-207`, `:426`]. Veto OFF (D-VETO fire≈0; keep the
  instrument minimal — memory `d-veto-fire-rate-zero`).
- **Knobs FROM the checkpoint** via `extract_deploy_knobs(ck["config"])` [V `:104`, `_REQUIRED_KNOBS
  :81-87` hard-errors on any gap]: `gumbel_m`, `c_visit`, `c_scale`, `n_sims_full`, `c_puct`.
  Expected `gumbel_m=16`, `n_sims_full=150` [V `run2_mw_fresh.yaml:69,75`]. **F6 abort:** across a
  multi-ckpt series, assert every resolved knob is identical to the first ckpt's; abort on drift
  (a mid-series knob change splices instruments). Never a literal — the only sanctioned literal is
  WP3's explicit `--override-n-sims` (§4).
- **Opponent:** `SealBotBot(time_limit=600.0, max_depth=5)` [V `sealbot_bot.py:30`]. d5 is the
  canonical external bar [V `eval.yaml … deploy_strength.sealbot_max_depth=5`, read not hardcoded via
  `sealbot_depth_from_config`]. The 600 s ceiling ≫ the in-loop 60 s so DEPTH binds and the bar is
  machine-independent (F7: a thermally-throttled 8845HS core extends wall time, never truncates
  depth; assert per-search max < 60 s, memory `dev-laptop-build-thermal-cutoff`).
- **Model load:** `load_model_with_encoding(ckpt, dev, decode_override="v6_live2_ls")` — the
  gated loader that RAISES on a mis-stamped file [V `checkpoint_loader.py`; verdict-2 uses exactly
  this at `:203`]. Driver flag `--expect-encoding v6_live2_ls` (§8). Explicit `_ls` fixture wherever
  a probe is involved (the C1–C3 fixture-fallback trap — memory `d-ws3v3-a1-rebaseline-match-hostmatch`).
- **Radius:** `make_eval_board("v6_live2_ls", R_ckpt)` where `R_ckpt = radius_from_checkpoint(ck)`
  via the real `StepCoordinator._resolve_radius` [V `verdict2…:64-73`]. Series assert: Series-A
  ckpts must resolve 4, Series-B 5; abort on mismatch.
- **Pairing:** each opening played twice, colors swapped, SAME stones, SAME per-side sim budget.
  Pair score `S = 0.5*(cand_outcome(g_a, HEAD) + cand_outcome(g_b, HEAD))` [V `a1_stats.cand_outcome:16`
  = 1.0 win / 0.5 draw / 0.0 loss]. Checkpoint WR = mean(S). CI = pair-level bootstrap over 64 pairs
  (`bootstrap_mean`, 2000 reps, percentile) [V `verdict2…:169-176`].
- **Draw handling — symmetric half-point, documented.** A draw scores 0.5 for the head in BOTH
  colors of the pair via `cand_outcome`; there is no asymmetric "best gets the half-point" path
  (the `wr_best` half-draw CI bug — memory `deploy-strength-inloop-cost` lineage — is avoided by
  construction: both arms of every pair receive identical draw treatment).
- **Dedup / eff_n:** `eff_n = distinct post-opening suffixes` reported next to nominal n ALWAYS
  [V verdict-2 reported 128/128]. Use `round_robin.distinct_games` [V `round_robin.py:186`] over the
  full game list for the byte-identical-trajectory count; additionally report `suffix_key` distinct
  count (the honest one — catches SealBot transposing into shared development lines). If distinct <
  nominal, the result JSON lists WHICH `opening_idx` collided into a shared suffix.
- **Integrity gates** (per verdict-2 [V `:253-258`]): `bad_pairs` = pairs where the move records do
  NOT show both arms sharing the opening prefix or the head did not fire in both; `censored_games`
  = games hitting `max_plies=200`. Both must be 0 for a clean read; >0 flags the operator.

### Move-record schema (`games.jsonl`, write-and-flush per game)

```json
{"ckpt_step": 175000, "ckpt_sha": "<sha256[:16] of the .pt>", "radius": 4,
 "book_id": "evalfair_r4_v2", "arm": "sims150", "opening_idx": 12, "head_as_p1": true,
 "p1": "head", "p2": "sealbot", "winner": "p1", "plies": 54,
 "moves": [[q,r], ...],
 "n_sims_effective": 150, "n_sims_from_ckpt": 150, "sims_overridden": false,
 "solver_backup": false, "solver_fired_win": 0, "solver_fired_loss": 1,
 "head_move_wall_s": [0.11, ...], "sealbot_search_wall_s": [3.9, ...],
 "sealbot_search_max_s": 5.1, "head_fired": true, "censored": false}
```

`n_sims_effective`/`sims_overridden` per record satisfy the WP3 REVIEW gate ("verify per-arm sim
budget from move-record metadata, not config echo"). `head_move_wall_s`/`sealbot_search_wall_s`
feed WP3's wall-time table (s/move split head-side vs SealBot-side).

### result.json schema

```json
{"wr": 0.594, "pair_ci": [0.508, 0.672], "n": 128, "eff_n": 128,
 "n_pairs": 64, "per_pair_scores": [...64...], "wall_per_move_head_s": 0.14,
 "wall_per_move_sealbot_s": 3.9, "wall_sec": 1979.0, "host": "laptop-8845hs",
 "ckpt_sha": "...", "ckpt_step": 175000, "book_id": "evalfair_r4_v2",
 "radius": 4, "knobs": {...}, "arm": "sims150",
 "n_sims_effective": 150, "sims_overridden": false, "solver_backup": false,
 "solver_counters": {"fired_win": 0, "fired_loss": 3, "skipped_offwindow": 1, "probes": 832},
 "deploy_matched": true,
 "bad_pairs": 0, "censored_games": 0, "suffix_collisions": []}
```

`deploy_matched = (not sims_overridden) and (not solver_backup)` — makes the JSON self-identifying
as a deploy-matched read or not.

---

## 4. Sims override (WP3) — the sanctioned escape hatch

The "knobs from the checkpoint, never a literal" rule and WP3's `n_sims ∈ {75,150,300,600}` arms
directly conflict. Resolution:

- `run_eval.py --override-n-sims N` mutates ONLY `knobs["n_sims_full"]` in a copy (`make_head_bot`,
  §1) — every other knob still comes from the checkpoint.
- Every move record carries `n_sims_effective`, `n_sims_from_ckpt`, `sims_overridden`; result JSON
  carries `deploy_matched=false` whenever overridden. The read is self-identifying as NON-deploy.
- `retro_slope.py` and the WP5 driver **hard-refuse** any `--override-n-sims` (assert the flag is
  None at entry; raise otherwise) — they are deploy-matched-only reads. Test T-REFUSE (§10).
- Solver-backup ON likewise sets `deploy_matched=false` (it changes the played move) and is refused
  by `retro_slope.py`/WP5.

---

## 5. Solver-backup arm (WP3.3) — one variable

ON = wrap the head in `SolverBackupBot`; OFF = bare head. Single toggle (`ArmSpec.solver_backup`).

- Entry point [V `hexo_rl/eval/solver_backup_bot.py:69`]:
  `SolverBackupBot(inner=DeployHeadBot(...), probe_engine="native", window_half=9, depth=6,
  cand_cap=40, node_budget=200_000)`.
- `probe_engine="native"` drives `engine.TacticalSolver` [V `:171-178`] — engine-native, no SealBot
  flat-array OOB, multi-window-correct. **WIN-only override**, terminal-mate-only (`|score| ≥
  WIN_THRESHOLD=99_999_000`) [V `:49, :211`]; proven LOSS is flagged not overridden [V `:242-244`].
- `window_half=9` — the v6_live2_ls in-window band [V `:92` comment]. The bare-default native probe
  with `window_half=None` WARNS that off-window WINs fire unguarded [V `:120-129`]; pass 9 to
  restrict to the sound in-window offense band. NB this measures the **in-window** tactical lift;
  the D-SOLVER off-window deploy-head hole (memory `d-solver-offwindow-deploy-head-hole`) is NOT
  what this arm reads.
- Counters `fired_win`/`fired_loss`/`skipped_offwindow`/`probes` [V `:140-144`] surfaced in the
  result JSON.
- **First read on a multi-window net.** The prior +0.195 lift (memory
  `d-solver-a1-instrument-paired-gate`) was validated on a d1m-era **single-window** net; run2 is
  multi-window. A null here is NEWS, not noise — report the paired-CI either way (WP3.3 pre-reg).
- Requires `.venv/bin/python` (engine `.so`, §1).

---

## 6. Parallelism + determinism

**Determinism claim.** A game is a pure function of `(ckpt, opening stones, color)`: DeployHeadBot
is g=0 (RNG × 0 at the root [V `:53,:200-205`]) → deterministic given the position; SealBot is
fixed-depth with a cold TT rebuilt per game in `reset()` [V `sealbot_bot.py` reset pattern; verdict-2
builds a fresh `SealBotBot` per game `:209-211`]. There is **no cross-game mutable state** — each
game builds a fresh board and fresh bots. The only residual nondeterminism source is GPU fp
non-associativity in the net forward, which a g=0 argmax could in principle flip on a knife-edge.

**Worker design.** `--workers N` partitions the 64 openings across N OS processes; each process
owns its own CUDA context + model + SealBot; results merged by `opening_idx`. `games.jsonl` is
written per-shard then concatenated and **sorted by `(arm, opening_idx, head_as_p1)`** before any
artifact hash, so write-interleaving cannot perturb the artifact. SealBot is the bottleneck
(single-threaded CPU, ~3.9 s/search; the GPU is mostly idle), so N processes ≈ N× on the SealBot
term with negligible GPU contention (model ≈ 49 MB, 4060 Max-Q 8 GB fits ≥4 contexts). 8845HS = 8
cores → default `--workers 4`.

**The invariance gate (red-team (c) + F10, this is the acceptance test, not a hope).** Run a 4-pair
smoke (i) twice at `--workers 1`, and (ii) once at `--workers 1` vs once at `--workers 4`; all three
result JSONs + sorted games.jsonl must be **byte-identical**. If they are, worker count is proven
irrelevant (games are pure functions of their inputs) and workers are free for WP3/WP4. If GPU
nondeterminism breaks it, the fallback ladder is: (1) `torch.use_deterministic_algorithms(True)` +
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, re-test; (2) if still nondeterministic, **anchor/verdict reads
(WP2 175k, WP5, the WP4 175k anchor point) run `--workers 1`**; WP4 non-anchor ckpts may use workers
only if the per-ckpt WR jitter is demonstrably below the MDE (0.13, retro_slope §8) — but the honest
default is serial for anything a frozen verdict reads. Given verdict-2 already ran deterministically
single-process, I expect the gate PASSES; the test decides, not this doc.

---

## 7. Recosted wall table (15.2 s/game; SealBot term sims-INVARIANT, head term ~linear in sims)

SealBot d5 dominates (~11–13 s/game, sims-independent — SealBot does not scale with net sims);
the head at 150 sims is ~2–4 s/game and scales ~linearly with `n_sims`. Per-arm game counts:
64 pairs × 2 colors = **128 games/arm/ckpt**.

| WP | Work | Games | Wall serial | Wall @4 workers |
|---|---|---|---|---|
| WP3 sims ladder | 4 arms (75/150/300/600) × 128 | 512 | ~2.4 h [I] | ~0.7 h |
| WP3 solver arm | 1 arm × 128 (native probe +~50% game) | 128 | ~0.8 h [I] | ~0.25 h |
| WP4 retro slope | 14 ckpts × 128 + 28 canonical | 1820 | **~7.7 h** [I] | **~2 h** |
| WP5 quick-win | 32 pairs×2 + 1 canonical pair | ~66 | ~0.3 h | ~0.1 h |

WP4 is the long pole — start it first of the parallel group (dispatcher §SEQUENCING). All [I] rows
carry the mandatory 4-game timing probe as step 0 of impl; if measured >2× the 15.2 s/game central,
re-cost before committing to the full run (but do NOT drop below deploy-match — fewer sims reads a
different head, and shorter `max_plies` censors long games into draws).

---

## 8. Gated loading + mis-stamp raise

- `--expect-encoding v6_live2_ls`; the driver passes `decode_override`/`declared_encoding` to
  `load_model_with_encoding`, which raises on a mis-stamped file [V verdict-2 `:203`].
- Test T-MISSTAMP (§10): construct a mis-stamped ckpt by loading a real run2 ckpt dict, overwriting
  `ck["config"]["...encoding..."]`/the stored encoding-name stamp to a wrong value (e.g. `v6w25`),
  saving to tmp, and asserting `load_model_with_encoding(..., decode_override="v6_live2_ls")` RAISES.
  (Exact stamp key: `_resolve_ckpt_stamped_encoding` at `checkpoint_loader.py:137` [V] — the impl
  agent reads that resolver to pick the key it must corrupt.)

---

## 9. Red-team checks (the three FROZEN acceptance gates)

The dispatcher's WP1 red-team is three checks; ALL must PASS or the instrument is not accepted:

- **(a) mis-stamped ckpt → RAISE** via the gated loader. → Test T-MISSTAMP (§8, §10).
- **(b) regenerate the book from the same seed → byte-identical JSON.** → Test T-BOOK: `book.py`
  generating `evalfair_r4_v2` from seed 20260709 twice yields byte-identical `moves`, and those
  moves reproduce sha `a1763be0c32ab4c1`.
- **(c) same 4-pair smoke twice → byte-identical results** (instrument self-determinism). → the §6
  invariance gate (also covers workers).

Plus the REVIEW (fresh context): re-derive pairing correctness from the MOVE RECORDS of a 4-pair
smoke (not the harness log) — Test T-PAIRING asserts, from `games.jsonl` alone, that each pair's two
games share the opening prefix and swap colors, and that the head moved in both. And verify
radius/knob resolution reads the checkpoint (T-RADIUS: fold vs the run2 schedule table at 50k→4,
199999→4, 200000→5, 400000→6).

---

## 10. Test plan (unit tests the impl agent MUST write, all under `.venv/bin/python`)

| Test | Asserts |
|---|---|
| T-BOOK | book gen from seed 20260709 is byte-identical across two runs; r4 `moves` ≡ book_v1, sha `a1763be0c32ab4c1`; every opening ends `moves_remaining==2`; 64 distinct |
| T-RADIUS | `radius_from_checkpoint` fold matches the schedule at steps 50000→4, 199999→4, 200000→5, 400000→6 |
| T-PAIRING | from a 4-pair `games.jsonl` alone: both games/pair share the 3-ply opening, colors swap, head_fired both |
| T-REFUSE | `retro_slope.py` and WP5 driver raise on `--override-n-sims`; `run_eval.py` accepts it and stamps `deploy_matched=false` + `n_sims_effective` in every record |
| T-MISSTAMP | mis-stamped ckpt → `load_model_with_encoding(..., decode_override="v6_live2_ls")` raises |
| T-WORKERS | 4-pair result byte-identical at `--workers 1` (twice) and `--workers 1` vs `--workers 4` |
| T-ANCHOR (integration, `@pytest.mark.integration`, opt-in — needs GPU + 175k ckpt) | 175k on evalfair_r4_v2 reproduces WR 0.594 within CI; skipped in `make test` |

`make test` excludes the integration marker [V Makefile] — T-ANCHOR is operator-run on the laptop,
the rest run in CI.

---

## 11. Conflicts with the prior design, recorded

1. **Per-stage books vs single r4 book (§2).** Dispatcher wins over `retro_slope_DESIGN.md`
   R4a/R4b. Loser's bias: single r4 book biases **Series B toward false-plateau** (r5 ckpts read
   off-regime). Adopted scheme has no within-series bias identified.
2. **book_v1 32 openings vs 64 (retro_slope §3 says "Size: 32").** The VERIFIED verdict-2 run used
   **64** [V `result.json n_openings=64, n_pairs=64`]. 64 is adopted (more power; retro §8's 32-pair
   MDE of 0.13 tightens to ~0.09 at 64). No conflict with any frozen verdict.
3. **`rng_seed` per opening (dispatcher's book_v2 spec) has no consumer (§2).** Set `null`, retained
   for format compat, never read at play time. Recorded rather than silently omitted.

---

## 12. Open questions for the operator

1. **Worker default.** §6 recommends `--workers 4` gated on T-WORKERS passing. If T-WORKERS FAILS
   (GPU nondeterminism), the anchor reads go serial and WP4 stretches to ~7.7 h. Accept serial, or
   invest in `use_deterministic_algorithms` first? (Impl runs T-WORKERS in step 0; the answer is
   data, not a guess — flagged only so the operator isn't surprised by a serial WP4.)
2. **"latest banked" for WP3 (dispatcher WP3.1 says 175k or latest).** Latest local is **248k**
   (Series B, r5). Recommend running WP3 on **175k** (comparable to verdict-2, r4) as the primary
   and adding 248k only if wall permits — 248k needs the r5 book, which is fine but not
   verdict-2-comparable. Confirm the primary ckpt.
3. **r5 book seed.** Proposed `20260710` (retro_slope §3 escalation seed). Any objection before it
   is committed as a fixture?
