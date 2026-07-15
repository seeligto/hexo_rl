# Run3-CNN Pre-Registration — stop rule + verdict defs + baseline pins

**Status:** PRE-REGISTERED (WP0.6, 2026-07-15). Committed, binding record of the run3-CNN
comparison protocol — the final Phase 0 gate. Not yet launched.
**Frozen-after-launch:** every number, estimator, artifact SHA, and verdict boundary in
§0–§7 below is FROZEN the instant run3 launches. Launch commit SHA: `<LAUNCH_COMMIT_SHA>`
(placeholder; the Phase 1 launch step stamps the actual launch commit here and never edits
thereafter). Launch config SHA: `<LAUNCH_CONFIG_SHA>` (placeholder; sha256 of the resolved
launch YAML — `configs/variants/run3_dist65.yaml` post-merge — stamped at the same time).
Pre-launch, this doc's §0–§3 hard pins / stop rule / verdict definitions are FROZEN AS OF
THIS COMMIT (WP0.6 mandate: "stop-rule numbers and verdict definitions are FROZEN as
drafted incl. its DESIGN ADJUSTMENTS A1–A3"); any change requires a dated addendum, not a
silent edit. §4/§5 pin tables and gap-analysis MAY still receive dated addenda pre-launch as
remaining backfill SHOULD-items land (tracked in §5, §8 risk 6). Post-launch, every section
closes to append-only per §9.
Design baseline commit at drafting: `c2e2c3d` (master). WP0.6 finalization baseline:
`f614430` (WP0.5 fresh-books commit) + the fold-ins pinned in §4b/§4d/§4e/§4f below,
committed by WP0.6 (see this file's own commit in `git log` for the exact finalization SHA).

Run3 = fresh-init CNN + dist65 value head, bounded ≤300k steps. Purpose: E1-at-scale
one-variable read — dist65 head-shape vs run2's banked scalar baseline on the same fair
instrument (fair-strength slope + value-health trajectory). NOT an architecture run.

---

## 0. Hard pins (void the whole comparison if violated)

- **Encoding = `v6_live2_ls`.** Books, 234-probe, and 651-control reconstruct under this
  encoding via the gated loader; the E1 dist65 head was frozen under it. If run3 trains
  a different encoding, EVERY comparator below is void. Assert at launch
  (`--expect-encoding v6_live2_ls`, D-EVALGATE two-semantic gate).
- **Head = dist65** (65-bin distributional value head, E1-graduated, `value_head_type`
  carries the E1 REVIVE card). Scalar comparator = run2 lineage.
- **Deploy/eval regime matched:** wr_d5 read is deploy-matched — DeployHeadBot Gumbel
  g=0 argmax, m=16, 150 sims, no solver, no sims-override. Never the training-loop
  PUCT+temp eval (D-LADDER: live eval ≠ deploy regime).

---

## 1. Instruments — VERIFIED EXIST + reach the regime

| Instrument | Producer (path) | Output | Verified |
|---|---|---|---|
| **wr_d5 EVALFAIR** (fair strength) | `scripts/eval/mantis_pull_eval.py` Stage 2 → `scripts/evalfair/run_retro_ckpt.py` / `core.py` / `book.py` | WR vs SealBot-d5, DeployHead Gumbel-150 g=0, 64 pairs=128 games, pair-bootstrap CI, eff_n=distinct games | ✅ ran over run2 (`reports/evalfair/retro_slope.md`, 15 ckpts, eff_n=128/128 clean) |
| **fair-slope** (Theil-Sen) | `scripts/evalfair/retro_slope.py` / `compute_slope_report.py` | Theil-Sen slope of wr_d5 vs step, pair-bootstrap CI, ≥7-pt power floor, per-stage (no r4/r5 splice) | ✅ `retro_slope.md` Series A/B verdicts produced |
| **M1/M2/M4 value-health** (234-probe) | `scripts/eval/mantis_pull_eval.py` Stage 1 → `scripts/e1/validate_ckpt.py` (+`scripts/valprobe/value_health.py compute_ece`) | M1 mean-v on 234 losses; M2 ECE on 234∪651; M4 false-pessimism on 651; pos-level bootstrap 10k | ✅ E1 series produced (`reports/e1/t7_series.jsonl`) |
| **verdict compute** | `scripts/e1/compute_verdict.py` | frozen REVIVE/CONFIRM-DEMOTE per `e1_metric_freeze.md` §5 | ✅ produced `E1_VERDICT.md` |
| **strix-g128 bar** (D-K ceiling, +313 Elo) | `mantis_pull_eval.py` Stage 3b → `scripts/evalfair/head_vs_strix.py` | WR vs strix-g128, 32 pairs, own venv | ✅ present (`--with-strix`, run3 bar from day 1) |

**Probe-set version match (red-team standing order):** the 234-probe used by run3's
value-health = the SAME frozen set E1 gated on. `reports/valprobe/probe_set_v1.jsonl`
sha256 `7899fa136ac083f0a428f5f6fa4c89918f1ba82c85618e8c7369a19506a9adb6` (234 rows, all
`set:loss`) — MATCHES the brief's `7899fa13…` and the hard SHA-guard hardcoded in
`validate_ckpt.py:106` (`_PROBE_SHA256`, HALTS on mismatch). Controls =
`negatives_v1.jsonl` (651 rows, sha256
`8faa6af74a7640f869cc3b1c4cb058b62660a052c5381e8ab7ad740a38cafef3`). Metric freeze:
`docs/designs/e1_metric_freeze.md` (§4 M1/M2/M4 defs, §5 verdicts) @ `685c617`, sha256
`f658a86f3d56a6d687d89651fa3c4eb35f719f181d88059963bb5d983d0d496d`.

**mantis_pull_eval = the unified backfill+forward tool.** One script emits Stage-1
value-health series + Stage-2 wr_d5 retro + Stage-3b strix bar; `--book-r4/--book-r5`
flags accept the WP0.5 fresh books; done-marker idempotent. Point it at
`checkpoints/run2_retro` (backfill) and `checkpoints/run3` (forward) with the SAME books.

---

## 1a. Instrument caveats (WP0.6 fold-in, 2026-07-15)

(i) **kraken/strix hardcode the r5 book at every step, regardless of checkpoint radius.**
Pre-existing `mantis_pull_eval.py` design (Stage 3 `_stage3_kraken` / Stage 3b
`_stage3b_strix`): both opponents take only `--book-r5` (no `--book-r4` param exists) and
load it unconditionally. At 50k/100k/150k (run3's native curriculum radius = r4) the
kraken and strix-g128 bars still play on an r5-sampled book. This is
instrument-internally consistent ACROSS every run3 checkpoint (same book, same sampler, at
every read step) — the kraken/strix SLOPE reads across steps remain valid; **wr_d5
(radius-matched via `resolve_book_for_radius`) is the primary strength read** (§3);
kraken/strix-g128 are secondary/ceiling bars only, not gating on the stop rule. (Source:
WP0.5 report §"Eval-wiring verification".)

(ii) **Two known pre-existing flaky tests, adjudicated non-blocking:**
- `tests/test_krakenbot_v1_mcts_bot.py::test_krakenbot_v1_mcts_determinism_temp0` —
  order-dependent (fails in a full-suite run, passes in isolation, reproduced isolated-pass
  twice).
- `scripts/e1/tests/test_validate_ckpt.py::test_scalar_row_schema_and_metrics` —
  float-tolerance assert, observed value `1.7039926935857874e-05` against a `< 1e-6`
  bound; reproduces byte-identically on clean master (confirmed via `git stash -u`
  isolation at WP0.5, commit `f614430`'s parent `42e4e90`).

Neither touches the eval/evalfair/value-health code paths this preregistration's
instruments depend on. Both instruments were independently verified sound via exact
reproduction instead: WP0.6 backfill-A reproduced the 50k value-health row byte-exact
against `t7_value_health_series.jsonl` (§4d); backfill-B reproduced ckpt SHAs and clean
book selection (§4e). A test-suite run matching EXACTLY these two failure signatures is
non-blocking for a preregistration/docs-only commit gate.

---

## 2. Stop rule (FROZEN at launch)

- **HARD STOP at 300k steps.**
- **EARLY STOP at 150k iff BOTH flat:**
  - **(a) fair-strength slope** — Theil-Sen on run3 wr_d5 (64-pair EVALFAIR reads, n≥5
    rounds between 50k and 150k, ≥7-pt power floor honored per WP1 frozen verdict-1); flat
    := 95% pair-bootstrap CI of slope includes 0.
  - **(b) value-health** — M1 level-gap AND M2 ECE-gap (234-probe) at 150k show no
    improvement vs run2-scalar at matched step (150k); flat := paired pos-level bootstrap
    CI of Δ (10k resamples) includes 0 on BOTH.
  - Both (a) AND (b) flat ⇒ early stop. Either trending ⇒ continue to 300k.
- **draw_rate hard-abort:** draw_rate ≥ 0.55 on 3 consecutive evals → abort; watchdog
  armed. Entropy warn < 1.5 nats, collapse-abort < 1.0 (healthy band 2.1–2.9, D-RUN2).

---

## 3. Primary read — verdict at stop (whichever rule fires)

`gap = run2scalar_M − run3dist_M` sign convention per `e1_metric_freeze.md` (positive =
run3/dist better). "matched step" = **min(run3_stop_step, 261500)** — run2 lineage's
banked terminal is 261500; run3 is ALSO read at that step so no side gets a free-steps
handicap (see §6 adjustment A1).

- **IMPROVED** iff (M1 OR M2 beats run2-scalar-at-matched-step, paired CI excluding 0)
  AND (fair strength: run3 wr_d5 Theil-Sen LEVEL at stop ≥ run2 matched-step, with pair-CI
  not excluding run3-worse-by > 0.05 WR).
- **FLAT** iff both value-health Δ CIs include 0.
- **REGRESSED** iff strength CI excludes 0 negative OR either value-health Δ worse with CI
  excluding 0.
- **Any fourth bucket → escalate to operator, NO folding.** (R2 precedent: D-M R2's
  4th-bucket regressed-vs-40k was NOT folded into a clean verdict; MEMORY d-l-strixprobe.)

---

## 4. Baseline pin table (all local; laptop `omarchy` authoritative)

Host note: run2 lineage checkpoints + all E1/EVALFAIR artifacts are LOCAL and complete.
Vast box (35883053) is NOT required and NOT a pin source — its corpus is a stale May-23
6902-game snapshot and its `checkpoints/` has step-name lineage collisions (MEMORY
`vast-stale-checkpoint-name-collision`); do not pin from it. Read-only, box untouched.

### 4a. run2 lineage checkpoints (comparator nets)

| step | path | sha256 | host |
|---|---|---|---|
| 50k | `checkpoints/run2_retro/checkpoint_00050000.pt` | `a6b9d861…c53f447` | omarchy |
| 150k | `checkpoints/run2_retro/checkpoint_00150000.pt` | `57cf9d4d…ba1ecf9` | omarchy |
| 200k | `checkpoints/run2_retro/checkpoint_00200000.pt` | `d4d994b1…20170c` | omarchy |
| 250k | `checkpoints/run2_retro/checkpoint_00250000.pt` | `768cc85c…d1f2a11` | omarchy |
| 248k (E1 warm-start base) | `checkpoints/run2_retro/checkpoint_00248000.pt` | `312f85f6…a1a6e50` | omarchy |
| 261500 (banked terminal) | `checkpoints/run2_final/checkpoint_00261500.pt` | `b340a8ed…44c9f69` | omarchy |

Full run2_retro ladder present: 50/70/90/110/130/150/170/195/200/210/220/230/240/248/250/255k
(no exact 100k ckpt — 90k+110k bracket it). run2_final: 260500/261000/261500.
Cross-check: the E1 series' short `ckpt_sha` = sha256[:16] of the .pt (verified: 50k
`a6b9d861…` = t7_value_health 50k row; 150k `57cf9d4d…` = retro 150k row). **Re-confirmed
at WP0.6:** both backfill-A and backfill-B independently re-verified the 50k/150k/200k/
250k/261500 SHA256s against this table BEFORE use — all MATCH (§4d, §4e).

### 4b. Instrument artifacts (frozen sets + metric procedure)

| artifact | path | sha256 | rows |
|---|---|---|---|
| 234-probe (loss set) | `reports/valprobe/probe_set_v1.jsonl` | `7899fa13…adb6` | 234 |
| 651-controls (safe set) | `reports/valprobe/negatives_v1.jsonl` | `8faa6af7…afef3` | 651 |
| metric freeze | `docs/designs/e1_metric_freeze.md` | `f658a86f…0d496d` | — |
| **books r4 (CURRENT, WP0.5 fresh — fold-in #1)** | `tests/fixtures/opening_books/evalfair_r4_v3.json` | `50c3fcc77609f140b56a82371f5384b14149b4effc42107130cb2118a5b7621f` | 64 openings |
| **books r5 (CURRENT, WP0.5 fresh — fold-in #1)** | `tests/fixtures/opening_books/evalfair_r5_v3.json` | `906451e8e25253daa1d969739b9a57bcffb0d54dbec8e529f9789a56ae1ecd08` | 64 openings |

Seeds: r4_v3 = `20260714`, r5_v3 = `20260717` (r5's first two candidate seeds — `20260715`,
`20260716` — were rejected on a real D6-symmetry-fold training-overlap collision, see
below). **0-overlap: 0/64 exact AND 0/64 symmetry-folded**, both books, vs all 8669
training-corpus games (the exact game set that fed the pinned run3 NPZ, §4f). Symmetry-fold
is LOAD-BEARING here (not a cosmetic upper bound): `configs/training.yaml: augment: true`
trains on all 12 hex-symmetric images of every corpus position, so a symmetry-collision
opening genuinely was seen (in some frame) at train time. Verified via
`scripts/evalfair/book_overlap.py` (committed WP0.5):
```
[book_overlap] raw_human training(<=cutoff)=8669 excluded(>cutoff)=29
[book_overlap] evalfair_r4_v3: EXACT 0/64 (games=0)  SYMFOLD 0/64 (games=0)  -> CLEAN
[book_overlap] evalfair_r5_v3: EXACT 0/64 (games=0)  SYMFOLD 0/64 (games=0)  -> CLEAN
```
r6/r8 books deliberately NOT generated — provably unreachable at the read grid:
`resolve_radius_from_schedule` over `{50k,100k,150k,200k,250k,300k}` resolves r4 (≤150k) /
r5 (200k–300k) only; r6 needs step≥400000, r8 needs step≥600000 — both past run3's 300k
HARD STOP (`legal_move_radius_schedule` = `{0:r4, 200000:r5, 400000:r6, 600000:r8}` per
`configs/variants/run3_dist65.yaml`).

Superseded (kept for provenance, NOT a valid comparator under this preregistration):
`evalfair_r4_v2.json` (sha `9cf1f839…6963db`) / `evalfair_r5_v2.json` (sha
`075a950a…f21410`) — different seeds/openings (`20260709`/`20260710`); the §4c "wr_d5
retro (OLD books)" row below used these and is superseded by §4e.

### 4c. run2 matched-step EVAL RECORDS (what exists vs backfill)

| record | artifact | sha256 | run2 coverage on 234-probe / wr_d5 |
|---|---|---|---|
| value-health 50k (234-probe, true run2 lineage) | `reports/e1/t7_value_health_series.jsonl` | `ee0b1b05…06e21e` | M1/M2/M4 @ **50k only** (see §4d for the full matched-step ladder) |
| value-health 250k–298k (E1 arms) | `reports/e1/t7_series.jsonl` | `cc7d688d…5c8dfe` | scalar+dist @ 250/255/260/265/270/275/280/285/290/295/298k — **E1 paired warm-start regime, divergent from vanilla run2 (see §5-B2, quantified at §4d)** |
| **wr_d5 retro (OLD v2 books — SUPERSEDED by §4e, kept for provenance only)** | `reports/evalfair/retro_slope.md` + `reports/evalfair/retro_slope/*/result.json` | `c4409d6b…1502ca` (md) | wr_d5 @ 50–248k, but books = r4_v2/r5_v2 (**not run3's fresh books — do not use as a level comparator**) |
| book-position value-health (NOT M1/M2) | `reports/valprobe/value_health_series.jsonl` | `f7c044eb…d2d5fbc` | evalfair_r4_v2 game-position ECE/decided-acc — DIFFERENT instrument, NOT the 234-probe M1/M2; do NOT use as the value-health comparator |
| E1 verdict | `reports/e1/E1_VERDICT.md` | `b58adfc7…4d98c7` | dist65 REVIVE (head-shape evidence, not a run3 comparator) |

### 4d. BACKFILL-A — vanilla run2 value-health (234-probe M1/M2/M4), matched-step, true retro lineage (WP0.6 fold-in #3, COMPLETE)

Source: `reports/eval/run3_cnn/run2_valuehealth_backfill.jsonl` (sha256
`e1a7eaab0f1709cb6dd708ca6237fa7ce5f34c47a738cdd65c03db9787307ee2`), produced by
`scripts/e1/validate_ckpt.py --arm scalar` against the frozen 234-probe (§4b), laptop
4060, zero skips, encoding `v6_live2_ls` asserted.

| step   | M1 (mean_v_on_losses) | M2 (ECE) | M4 (false-pess) | decoded AUC |
|--------|----------------------|----------|-----------------|-------------|
| 50000  | +0.3183 | 0.1232 | 0.0046 | 0.7450 |
| 150000 | -0.0228 | 0.1086 | 0.0108 | 0.8212 |
| 200000 | +0.2701 | 0.1373 | 0.0092 | 0.7517 |
| 250000 | +0.3311 | 0.1520 | 0.0046 | 0.7406 |
| 261500 | +0.0967 | 0.1034 | 0.0046 | 0.8075 |

Verification: 50k reproduces byte-identical to `t7_value_health_series.jsonl` (M1
0.31834655, ECE 0.12318188, AUC 0.74498799, M4 0.00460829); all 5 ckpt SHA256s match §4a
exactly.

**Interpretation note (per WP0.6 mandate — does NOT change the §2 stop-rule text above):**
value health is NON-MONOTONE across run2's lineage. **150k is run2's STRONGEST
value-health point** (M1 near 0, AUC 0.82); 200k/250k REGRESS below even the 50k level on
M1/ECE/AUC (coincides with the r4→r5 radius boundary, consistent with D-C VALPROBE "200k
radius cliff" + 248k regression); 261500 partially recovers. Consequence for §2 early-stop
rule (b): the 150k comparator is run2's BEST value-health reading, not a representative
mid-run point — it is a **conservative (strict) bar**. Early stop at 150k requires run3 to
be flat vs THIS strict bar; a run3 that merely matches run2's weaker 250k/261500 level
would NOT satisfy rule (b) at the 150k gate. Both conditions — (a) strength-slope flat AND
(b) value-health flat vs this strongest-point bar — must hold for early stop; a run3 that
only matches run2's terminal-average level is not stopped early at 150k under this rule.

Also flagged: the existing `t7_series.jsonl` 250k–298k reads are a DIFFERENT (E1
warm-started, paired) lineage, materially divergent from vanilla run2 at the same nominal
step (250k: M1 +0.403/ECE 0.161/AUC 0.782 there vs +0.331/0.152/0.741 here) — confirms
§5-B2's caveat was load-bearing; do not substitute `t7_series.jsonl` for this table.

### 4e. BACKFILL-B — run2 wr_d5, fresh v3 books, matched-step (WP0.6 fold-in #4, PARTIAL — HARD requirement complete)

Source: `reports/eval/run3_cnn/run2_wrd5_v3books_backfill.jsonl` (sha256
`8dc0526830785eaabcffd1939faac0d62289ceda0af10d2f6c6dfe513a4bfa8b`), produced by
`scripts/evalfair/run_retro_ckpt.py` (frozen EVALFAIR instrument), DeployHead Gumbel g=0
argmax, 150 sims deploy-matched, 64 pairs = 128 games, pair-bootstrap CI,
`--expect-encoding v6_live2_ls`, laptop, GPU uncontended.

| step | wr_d5 | pair-CI | raw_n | eff_n | book used |
|---|---|---|---|---|---|
| 50k  | 0.547 | [0.461, 0.633] | 128 | 128 | `evalfair_r4_v3` (sha `50c3fcc7…5b7621f`) |
| 150k | 0.586 | [0.492, 0.672] | 128 | 128 | `evalfair_r4_v3` (sha `50c3fcc7…5b7621f`) |

Ckpt SHAs verified against §4a BEFORE use (50k `a6b9d861…c53f447`, 150k `57cf9d4d…ba1ecf9`
— MATCH). Book auto-resolved per checkpoint radius (both native r4, run2 schedule
r4≤195k); `result.json` `book_id` confirms v3 (not v2) was actually used. **No eff_n
collapse:** 128/128 distinct post-opening-suffix AND 128/128 distinct full-trajectory
hashes at both steps, zero duplicate games/suffix collisions — the 64-opening book
supplies sufficient diversity per §7. Clean integrity: `bad_pairs=0`, `censored_games=0`
both steps.

**Read caveat (mandatory per fold-in #4):** these are the §2/§3-required PRE-150k HARD
reads only. **Both CIs straddle 0.5** (50k and 150k are each individually
indistinguishable from a coin-flip vs SealBot-d5 at n=128) and **the two CIs overlap
heavily** (Δwr=+0.039) — this pair of points is a LEVEL bank, not yet a slope: the §2(a)
Theil-Sen slope read needs "n≥5 rounds between 50k and 150k" per the frozen stop rule and
is UNDEFINED from 2 points alone. **Status: PRE-150k HARD requirement (§5-B) is MET** (50k
+ 150k on fresh books, both banked, both clean, no anomalies). **PRE-LAUNCH SHOULD
requirement (200k/250k/261500 on fresh books) is STILL OPEN** — not done in this WP0.6
pass; carried as a residual (§5, §8 open risk 6). Comparison against the OLD-book (v2)
retro reads (§4c) is explicitly NOT made — different openings, not a valid level
comparison per §5-B.

### 4f. run3 TRAINING CORPUS pin (WP0.6 fold-in #2)

Canonical pin doc: `docs/registers/run3_corpus_manifest.md` (WP0.4, commits `1d4a206` +
`42e4e90`).

| artifact | path | sha256 | scope |
|---|---|---|---|
| run3 training corpus (load-bearing) | `data/bootstrap_corpus_v6_live2_ls.npz` | `3813edc2fb10a7c5ab976a0293e38cbba0fd6b84e5295630f339ca421b345c97` | 610954 positions, 8669 games, built 2026-07-04 |

Enforcement (code, live): `configs/variants/run3_dist65.yaml`
`mixing.pretrained_buffer_path: "<auto>"` routes through `resolve_corpus_path` →
`_CORPUS_SHA_PINS` (keyed by encoding, `hexo_rl/encoding/resolvers.py`);
`load_pretrained_buffer` recomputes the on-disk sha256 and hard-fails (`ValueError`,
expected-vs-actual named) on any mismatch, on EITHER host, at step 0 — the
**`<auto>`-requires-pin law**. Ruling on the 1796 laptop-only held-out games: ACCEPT —
already baked into the frozen NPZ, enters run3 training (manifest §2; the 1796 are BURNED
as a future clean held-out set as a consequence — do not reuse for a future architecture
read). Cross-host parity (`sha256sum` on the vast box) must be independently confirmed
before launch per manifest §4 — not re-verified by this preregistration (vast box is
read-only, not a pin source, per §4 host note above).

---

## 5. GAP ANALYSIS + backfill requirements

*(WP0.6 status, 2026-07-15: both backfills below are RESOLVED for their HARD requirement;
the drafted rationale is kept in full as the record of what was required and why. See §4d/
§4e for the completed pin tables and §8 risk 6 for the one remaining SHOULD-item.)*

Two comparators the stop rule needs are UNDEFINED or STALE. Both backfill via
`mantis_pull_eval.py` over `checkpoints/run2_retro` (+`run2_final`) with WP0.5's fresh
books. Both are CHEAP relative to a training run.

### A — value-health (234-probe) matched-step reads for run2 lineage → MOSTLY MISSING
- **Have:** 50k (`t7_value_health_series.jsonl`, true run2 retro lineage).
- **Missing (true run2 lineage):** 150k, 200k, 250k, 261500. (100k has no ckpt; use
  90k+110k or drop.)
- **Caveat B2 — the existing 250k–298k 234-probe reads (`t7_series.jsonl`) are NOT the
  vanilla run2 lineage.** E1 warm-started BOTH arms from run2-248k + headswapped heads,
  then trained 50k more under the E1 experimental regime (eval OFF, paired). The scalar
  arm there diverges from what run2 would have done past 248k. For a clean matched-step
  value-health comparator, re-score the run2 RETRO ckpts (true lineage).
- **Backfill:** `validate_ckpt.py --arm scalar` on run2_retro/{150k,200k,250k} +
  run2_final/261500. NO search — fixed-position forward, deterministic given (ckpt,probe).
  Est. **~1–3 min/ckpt on laptop 4060; ≤15 min total** for the 4 missing steps.
- **REQUIREMENT:** the **150k** read is a **PRE-150k HARD REQUIREMENT** (early-stop rule
  (b) is undefined without it). The 200k/250k/261500 reads are a **PRE-LAUNCH SHOULD**
  (needed only if run3 runs past early-stop; cheap enough to do all up front — recommended).
- **STATUS (WP0.6, 2026-07-15): COMPLETE.** All 5 matched steps (50k/150k/200k/250k/
  261500) banked, byte-verified at 50k, ckpt SHAs match §4a. See §4d.

### B — wr_d5 matched-step reads use STALE books → RE-READ REQUIRED
- **Have:** run2 wr_d5 @ 50–248k, but on books `evalfair_r4_v2`/`r5_v2`. WP0.5 generates
  FRESH books (new openings). Different openings → WR not directly comparable at the
  pre-registered LEVEL bar. (Same book recipe/sampler, new seed → distributionally similar
  but not identical; a frozen LEVEL comparison demands identical books.)
- **Backfill:** re-run `mantis_pull_eval.py` Stage 2 over run2_retro with
  `--book-r4/--book-r5 <WP0.5 fresh books>` at the matched steps the primary/early-stop
  read needs: 50k, 150k (early-stop), 200k, 250k, 261500 (terminal). ~640–810 s/ckpt (128
  games) → **~1 h for 5 ckpts** on laptop workers=4.
- **REQUIREMENT:** 50k+150k reads on fresh books = **PRE-150k HARD** (early-stop (a)+(b)
  reference); 200k/250k/261500 = **PRE-LAUNCH SHOULD**. Fresh-book generation (WP0.5) is a
  BLOCKING upstream dep — books must exist and be SHA-pinned into this doc BEFORE the run2
  re-read, else the retro reads inherit the same stale-book defect.
- **Radius match:** run3 curriculum radius in the 50k–150k window MUST match the book
  radius used (run2 was r4 through 195k, r5 from 200k; r4→r5 boundary is a structural
  discontinuity — never splice slopes across it, per frozen verdict-3). Pick the book
  whose radius == run3's live curriculum radius at each read step.
- **STATUS (WP0.6, 2026-07-15): PARTIAL.** PRE-150k HARD requirement (50k+150k on fresh
  books) COMPLETE and clean — no anomalies (§4e). PRE-LAUNCH SHOULD (200k/250k/261500 on
  fresh books) NOT done this pass — open residual, §8 risk 6.

### C — hard-stop-300k comparator: run2 never reached 300k (see §6-A1)

---

## 6. DESIGN ADJUSTMENTS (with rationale)

**A1 — hard-stop comparator: run2 lineage max = 261500, not 300k.** run2 banked-terminal
is 261500; a "run2-scalar at 300k" does not exist. Proposal number said "run3 at stop vs
run2 matched-step" — undefined at a 300k stop. **ADJUSTMENT:** matched step :=
`min(run3_stop_step, 261500)`; read run3 ALSO at 261500 for the frozen LEVEL bar so run3
gets no free 38.5k-step advantage. Additionally report run3@300k-terminal vs
run2@261500-terminal as a SECONDARY (terminal-vs-terminal) descriptive line. Rationale:
step-matched removes the training-budget confound from the load-bearing IMPROVED bar;
terminal-vs-terminal is the honest "where each run ended" descriptor.

**A2 — 100k matched step has no run2 ckpt.** run2_retro has 90k+110k, not 100k. The stop
rule never reads 100k (early-stop is 150k), so this is cosmetic — but the brief's pin
grid listed 50/100/150/200/250. **ADJUSTMENT:** drop 100k from the pin grid; matched
comparator steps = {50k, 150k, 200k, 250k, 261500}. If a 100k point is ever wanted,
bracket with 90k+110k (both present).

**A3 — book radius selection is per-step, per WP0.5 fresh books.** The retro instrument
already auto-selects the book by `radius_from_checkpoint`; freeze that behavior for the
backfill and require run3's read-step radius to equal the book radius (no cross-radius
comparison). No number change; makes the WP1 fix load-bearing here.

No other proposal numbers changed. Stop-rule thresholds (150k early, 300k hard, 0.55×3
draw abort, entropy 1.5/1.0, slope-CI-includes-0, Δ-CI-includes-0) all retained.

---

## 7. eff_n / dedupe discipline (Re-validation + D-ARGMAX, MANDATORY every round)

- **eff_n = number of DISTINCT games, not the game count.** argmax/g=0 deploy from a fixed
  opening is deterministic → byte-identical sequences collapse to ~1–2 games/pair. A
  BT/Wilson CI over the raw count is over-confident by √(copies) (D-ARGMAX: a √40=6.32×
  spurious narrowing manufactured a phantom "−109 Elo").
- **Every eval round:** hash each game's move trajectory; report distinct/nominal and
  eff_n; the CI is the PAIR-bootstrap over DISTINCT games (retro instrument already does
  this — `result.json` carries `eff_n`, `per_pair_scores`; keep it). Reject any wr_d5
  round with distinct/nominal < 1.0 as underpowered unless opening/opponent DIVERSITY is
  injected (the 64-opening book IS the injected diversity — do not shrink it).
- **Value-health bootstrap is position-level, 10k resamples, paired** (resample once per
  replicate, recompute both arms, take the Δ). M1 resamples 234; M2 resamples 234∪651.
  Frozen caveat: position-level bootstrap ignores source-game clustering (anti-conservative
  if losses cluster by game) — accepted per E1 dispatcher, NOT re-opened.
- **Re-validation discipline:** any drop of a run3 driver by citing a run2/E1 prior must
  re-validate that the prior's context (init regime, encoding, radius, deploy regime)
  transfers to run3 fresh-init before keeping the drop. A drop on an un-re-validated prior
  = reject.

---

## 8. Open risks

1. **Init-regime confound (load-bearing).** run2 lineage warm-started from the
   `bootstrap_v6_live2_ls` mint (a pre-trained net); run3 = fresh RANDOM init. At 50k–150k
   the absolute M1/M2 and wr_d5 LEVELS conflate architecture with init-regime. The
   TRAJECTORY/SLOPE reads (stop rule (a), and the value-health slope) are far more robust
   to this than the LEVEL bar in §3. Recommend the operator weight slope over level when
   they diverge; record if IMPROVED rests solely on a level bar.
2. **E1 head-strength caveat.** E1 REVIVE = dist65 better at KNOWING it is losing (M1/M2),
   but net-vs-net argmax showed WR 0.45 [0.30,0.60] — a TIE, no play-strength edge at 40
   games. Run3's fair-strength wr_d5 is the necessary external kill link; do not read a
   value-health IMPROVED as a strength claim without the wr_d5 LEVEL clearing §3.
3. **WP0.5 fresh-book dependency is BLOCKING.** Both backfills (A,B) and every wr_d5 read
   depend on the fresh books existing + SHA-pinned into this doc pre-launch. **CLOSED at
   WP0.6:** books exist, are SHA-pinned (§4b), 0-overlap confirmed (both exact and
   symmetry-folded), and the 50k+150k run2 re-reads on fresh books are banked (§4e).
4. **Radius-curriculum divergence.** If run3's fresh-init radius schedule differs from
   run2's (r4→195k, r5→200k), the per-step book selection must still match run3's live
   radius; a mismatch silently reads the wrong book (D-EVALFAIR radius-guard should trip,
   but confirm run3's schedule is declared and books cover its radii).
5. **300k may be short for a fresh CNN.** run2 needed ~150k+ to reach its wr_d5 peak from a
   warm start; a fresh net may still be climbing at 300k → a FLAT verdict could be
   "not-yet-converged" not "head-shape null". The bounded ≤300k is a cost cap, not a
   convergence claim; record FLAT-at-cap distinctly from FLAT-converged.
6. **Backfill-B SHOULD-item still open (WP0.6 residual).** run2 wr_d5 on fresh v3 books is
   banked at 50k+150k only (§4e); 200k/250k/261500 on v3 books were NOT re-read in this
   pass. Does not block early-stop (rule (a)/(b) only need 50k–150k), but the
   terminal-vs-terminal descriptive line (§6-A1) and any 200k+/250k+ run3 comparison
   against run2 currently has NO fresh-book comparator until this residual is closed.
   Cheap (~1 h/ckpt per WP0.5 estimate, ~40 min total for 3 ckpts) — recommend closing
   before run3 crosses 150k in wall-clock, not strictly before launch.
7. **150k early-stop value-health bar is run2's STRONGEST point, not typical (§4d).** A
   run3 that is FLAT vs 150k under rule (b) is flat vs a conservative bar — good news if
   it clears; a run3 that FAILS rule (b) at 150k should not be read as "run3 is bad at
   value health generally" without also checking the weaker 250k/261500 run2 points for
   context.

---

## 9. Frozen-after-launch clause

**WP0.6 status (this commit, 2026-07-15):** the book/corpus SHA-pinning originally
planned as a launch-time step is DONE NOW (§4b/§4f) — the fresh books and corpus pin are
folded into this preregistration at WP0.6 commit time, not deferred to launch. What
remains for the actual Phase 1 launch step:

1. Stamp `<LAUNCH_COMMIT_SHA>` above with the launch commit.
2. Stamp `<LAUNCH_CONFIG_SHA>` above with sha256 of the resolved launch YAML.
3. Confirm run3's declared `legal_move_radius_schedule` matches the schedule assumed by
   §4b's r6/r8-unreachable proof (`{0:r4, 200000:r5, 400000:r6, 600000:r8}`) — if the
   launch config changes this schedule, the r6/r8-unreachable claim and the
   book-radius-match (§6-A3) must be re-verified before launch, not assumed.
4. Close open risk 6 (§8) if wall-clock allows, or explicitly accept it open.
5. Confirm cross-host corpus parity (manifest §4 one-liner) on the vast box if a box run
   is planned.

Thereafter every §0–§8 number, estimator, and artifact SHA is FROZEN; changes are dated
APPEND-only addenda. No silent edits — this is the pre-registration of record.
