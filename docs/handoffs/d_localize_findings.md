# D-LOCALIZE — FINDINGS (gap-localization + eval-recipe + search-scaling)

Follow-on to D-LADDER (TRUE-STALL + robust model⊥minimax intransitivity, P=0.004).
D-LADDER asked WHETHER the model relates non-scalarly to SealBot; D-LOCALIZE asks
**WHERE the gap lives** (LINES / VALUE / TACTICS) → the Stage-2 lever class.
All offline / read-only re-eval of banked games + one in-loop code deliverable. No
GPU-week, no lever-training launched.

## VERDICT — VALUE-TARGET (value head blind to SealBot-reachable losses)

Re-evaluated all **68 lost mid-cluster games** (model net-vs-SealBot@depth-5, model
LOST; s150k=18, s175k=26, s200k=24 — the intransitive triangle, 8/9 D-LADDER cycles
route through SealBot). For each game: replayed the banked move-seq, took the net's
value+policy at every model decision ply, found the **decisive blunder** via a FULL
per-ply SealBot-d6 `last_score` scan + a WIN→LOSS-persists-to-terminal filter, and
classified by the pre-registered gates.

### Raw gate counts (68 games)
| ckpt | n | VALUE | LINES | TACTICS | VALUE+LINES | ALREADY-LOST |
|---|---|---|---|---|---|---|
| s150k | 18 | 17 | 3 | 2 | 3 | 1 |
| s175k | 26 | 22 | 5 | 1 | 5 | 4 |
| s200k | 24 | 22 | 5 | 1 | 5 | 2 |
| **TOTAL** | 68 | 61 | 13 | 4 | 13 | 7 |

Plurality = **VALUE**. **LINES never fires standalone** (all 13 are VALUE co-fires);
TACTICS never standalone. UNCLASSIFIED = 0.

### Corrected VALUE-vs-SEARCH discriminator (the load-bearing result)
The raw VALUE gate fires at the *decisive* ply, which is by construction d6-WIN-side,
so `net_value≥0` there only confirms the position was winning — it does **not**
diagnose a value error (RED-TEAM concern). Corrected test re-uses the per-ply
net_value already logged, evaluated at the FIRST **post-blunder** model ply (where d6
flips to a terminal-persistent LOSS):

- **VALUE-BLIND** (net_value(post-blunder) ≥ −0.05 — value head still thinks it's OK at
  a spot deep search calls lost): **56/61 = 92%**
- SAW-LOSS / TERMINAL (net sees the loss; deploy search/policy walked in or forced): 5/61 = 8%
- LINES-genuine co-signal (real missed forced line, ref_mass<0.05, pv≥3): 13/61

The value head reports **net_value 0.6–1.0 (winning)** at positions d6 calls **forced
mate against** (`d6_post = −99999997`). E.g. s150k idx150: net_value(post)=**0.99** at a
forced loss; idx108: **0.85**; idx38: **0.91**. This is a genuine, severe value-head
blindness, **not** an instrument artifact.

⇒ Stage-2 lever = **value-target / horizon-aware value calibration** (the value head
systematically over-values positions SealBot will convert several turns deep).
Predicts **P3 = PLATEAU**: more search budget searches deeper but the value head still
mis-evaluates the leaves — search can't fix a mis-calibrated value.

## Adversarial validation
- **REVIEW** (held-out s175k, independent re-derivation): **CONFIRMS**, 11/11 per-game
  agreement on decisive-ply + class set, 0 discrepancies; d6 determinism proven
  (0 mismatches across two processes); persistence filter rejects transient flips.
- **RED-TEAM**: the 13 LINES are **real** — d7 re-eval, **5/5 sampled LINES survive**
  (the MODEL missed the line — ref_mass 0.0000–0.0102, rank 16–216, gumbel never
  played ref-best — the *reference* did not miss it). p2_verdict_survives = true.
- **Persistence filter**: changed the decisive ply on **0/68** games (d6 oscillation
  real but did not bite the WIN→LOSS-persistent crossing).

## Off-window completing-cell sub-check (descriptive, NO lever)
- terminal-rate 12/68 = 18% (window re-centred at terminal — biased in-window);
  **decision-time-rate 26/61 = 43%** (correct test; `to_flat` single-window perception).
- So the scout's "0/18 off-window" null was a terminal-recentring artifact — at the
  model's decision ply, **~43% of SealBot completing cells fall outside the single
  window the net can see.** BUT D-LADDER already FALSIFIED off-window as a *deployable*
  defect (fixed-depth SealBot inversion s150k 0.35→0.55), so this is descriptive of the
  SealBot matchup, not a deployable defect. No lever built on it — consistent with the
  value-blindness story (SealBot wins via lines the net's window+horizon can't see).

## Stage-2 routing (final)
- **#1 lever = VALUE-TARGET** (value-head / horizon calibration on the deploy-matched
  Gumbel@150 head). The model+d6 both call the decisive position winning, yet it's lost
  → value/horizon misassessment, not missed-line policy gap, not search budget.
- **bot-mix / SealBot-vs-anchor imitation corpus = DROPPED** (the LINES+off-window
  branch did not fire; LINES never plurality). §S178 bot-mix recipe stays UNLAUNCHED.
- **Corpus regen = OFF / spec-only** (gated on P2=LINES, which did not fire). No
  GPU-week launched.
- **n stays 150.** P3 search-scaling DEFERRED + running on vast (see below); the
  value-blindness verdict predicts PLATEAU.

## Deliverables
- **P4 (in-loop deploy-matched strength eval) — SHIPPED + committed.**
  `hexo_rl/eval/deploy_strength_eval.py`: Gumbel SH greedy, root noise **g=0**
  (`gumbel_scale=0.0`), no temperature, deploy sims; fixed-depth-5 SealBot; adaptive
  screen(80)→confirm(200); distinct-game bootstrap BT-Elo gate; fail-safe (no
  PUCT/temp/64-sim fallback). Default-OFF opponent so existing runs are bitwise
  unchanged. RED-TEAM: g0 verified, gate traced end-to-end, false-negative bounded
  (~2.4% only at the exact 0.55 bar; stride re-screens), **14 tests pass**. Residual:
  screen-band lo/hi are config-overridable — pin defaults before deploy.
- **P0:** depth-5 = reproducible SealBot bar; **unit = HTTT turns** (depth-5 ≈ 10 stones;
  5 = median+1). **P1:** banked jsonl PARTIAL (moves, no value) → re-eval, no fresh games.
- **P3 (search-scaling 150-vs-256): COMPLETE → PLATEAU-by-150.** All 1200 games on vast
  (~7h, co-tenant). **No checkpoint has CI_lo(WR@256) > CI_hi(WR@150)** — 150→256 drops or
  flatlines everywhere (s120k .525→.350, s150k .575→.400, s175k .350→.250; only s200k
  .450→.525, within noise); n512 no climb; depth-4 red-team matches depth-5 (bar-independent).
  ⇒ keep n=150; 256 not worth the 1.7×. CONFIRMS the value-blindness is NOT a search-budget
  artifact — more sims search deeper to the same mis-calibrated leaf values. Verdict
  `reports/d_localize_p3/P3_VERDICT.md` (gitignored).

## Caveats
1. `SealBot.extract_pv()` walks only the forced-mate PV (|score|≥~1e8); heuristic
   positions return empty PV → LINES/TACTICS gates structurally under-fire on non-mate
   decisive plies. The **corrected discriminator is independent of this** (uses net_value,
   not PV), and its 92% VALUE-BLIND stands regardless.
2. The raw `classify()` VALUE gate has a polarity quirk (fires at the d6-WIN decisive ply);
   the corrected post-blunder discriminator is the trustworthy VALUE measure. Even
   discounting the raw VALUE inflation, LINES/TACTICS never reach standalone plurality.
3. distinct-n = game count per bucket (copy_mult=1.0); read-only re-eval of banked
   distinct games, no pseudo-replication.
4. Live d1m run was **NOT stopped** (healthy + actively training at ~249k; see
   `d_localize_run_status.md`). To free the full 5080 for P3: `ssh vast 'kill -INT 1512430'`.

## Artifacts (durable, in-repo)
`scripts/eval/p2_localize.py` (classifier), `p2_localize_review.py`, `p2_spotcheck.py`,
`p2_redteam_d7.py`, `p2_value_discriminator.py` (corrected test),
`scripts/eval/gumbel_ladder.py` (+`--n-sims-override`), `scripts/run_d_localize_p3_vast.sh`.
Data under `reports/d_localize_2026-06-25/` (gitignored): `p2_decisions.jsonl`,
`p2_summary.md`, `p2_corrected_summary.md`, `p2_redteam_*`.
