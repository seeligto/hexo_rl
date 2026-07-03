# D-LADDER → Stage-2 relaunch runbook

**Date:** 2026-06-24. **Predecessor:** §D-LADDER (sprint log) + `reports/d_ladder_2026-06-24/{PREREG,FINDINGS}.md`.
**Verdict driving this:** d1m is **TRUE-STALL** (deploy-matched Gumbel@150 self-ladder 120k→226k flat;
brief s200k peak not held). Bot corpus was SILENT (n_bot=0). The colony / off-window mechanism is
FALSIFIED (bar-dependent SealBot artifact). One robust extra: the model relates non-scalar-ly to a
minimax opponent (self-play Elo is an incomplete order).

## HARD INVARIANTS
- The live d1m run (PID 1512430, `/workspace/hexo_rl` on vast) is disposable but **NEVER touched** for
  this work. No `git pull` on the live host; no compute on the live training host. A relaunch is a
  SEPARATE process (new box, or after the live PID is retired) — operator-gated.
- Single-variable forks only. No lever committed on a story; every lever = banked checkpoint + one
  variable + honest deploy-matched eval delta.

## Stage-2 levers (ranked)
**#1 — pretrain-floor 0.10 → lower (training-fork A/B).** The only always-on external-data injection
this run had was the pretrained-buffer mix decaying 0.8→floor **0.10**. With bot SILENT, the floor is
the remaining "external anchor in the loop" suspect for the stall. Fork from a banked checkpoint
(e.g. 120k or 150k), single variable `mixing.pretrained_buffer_floor` 0.10 → {0.05, 0.0}, hold all
else; measure with the deploy-matched eval below. OFF (0.0) beats DECAY if the floor is not useful;
a lower floor (0.05) is the gentler arm.

**#2 — opening-diversity (selfplay).** Deferred behind #1. Inject opening variation in self-play data
generation (random opening plies / opening-book sampling) to widen the self-play distribution the
argmax regime collapses. Diagnostic: top1-mass / early-game entropy (§D-MONITORFIX Tier-1).

**#3 — bot-mix (SealBot-vs-anchor corpus, §S178) 0.0-vs-0.15 strength A/B.** UNTESTED as a strength
lever; ranks below #1/#2 for the stall. Its anti-colony rationale is FALSIFIED (A1 colony FLAT +
§D-LADDER colony/off-window mechanism falsified), but §D-LADDER's robust **model⊥minimax intransitivity**
gives a fresh, specific rationale: the corpus IS SealBot games, and self-play leaves the model
non-transitive vs the minimax style. So test bot-mix as an **EXTERNAL-axis lever, not a stall fix**.
REFRAME the hypothesis (anti-colony → close the external/minimax gap) and RE-INSTRUMENT the eval
(deploy-matched Gumbel@150 + fixed-depth SealBot external bar, NOT the old PUCT+temp+time-limited stack
that false-nulled it). Single variable: `bot_batch_share` 0.0 vs 0.15. This also re-points the **pending
§178 v6 bot-mix launch** — worth running, but on the new eval + the reframed success criterion (external
strength), or it false-clears/false-nulls on the instrument problems §D-LADDER exposed.

**Do NOT** (per §D-LADDER): chase the colony-attractor / off-window mechanism (falsified — bar
artifact); treat bot-mix as a #1 stall fix (the stall is self-play strength; bot-mix targets the
external axis — #3); use a wall-clock time-limited SealBot as the bar.

## Deploy-matched eval recipe (fold in REGARDLESS of lever)
Replace the live in-loop eval (PUCT visit-policy + `eval_temperature=0.5` + 64 sims — a head the model
never deploys) with:
- **Head:** Gumbel SH greedy, `n_sims_full=150` (deploy-matched), **no temperature**.
  Instrument: `scripts/eval/gumbel_ladder.py` (`play` → `aggregate`; supports `--sealbot-max-depth`,
  `--sealbot-only`). The in-loop gate should likewise drop temperature + restore sims to deploy.
- **Diversity:** opening **book** (balanced, color-reversed) preferred over random opening plies
  (AZ: random plies "resulted in more losses"); random-4-plies is the validated fallback (copy_mult 1.0).
- **External bar:** **fixed `max_depth` SealBot** (`SealBotBot(max_depth=D)`, deterministic /
  machine-independent), NOT `time_limit` (SealBot@0.5s reached median depth 4 here → hardware-dependent).
  D≈5 ≈ the 0.5s-equivalent. Add a **2nd** fixed-compute external bar (a single bot can false-clear an
  off-window defect by construction). Keep an external bar as a NON-REDUNDANT axis — the model⊥minimax
  intransitivity means self-play Elo alone misses the external ranking.
- **Stats:** distinct-game dedup + **bootstrap CI over distinct games** (§D-ARGMAX; the existing
  `round_robin` stack). Color-balanced, paired.
- **Power:** restore to ≥400-game-equivalent pooled (the live n=200/64-sim was down-powered). Adaptive:
  cheap ~80-game screen at deploy sims every cadence → full confirm only near the bar; trim descriptive
  opponents to lower cadence.

## Optional / lower-priority
- **Gumbel SH budget sweep:** 150 is non-canonical for m=16 (phase profile 2/4/10/23 vs 128's clean
  2/4/8/16). Sweep {128, 150, 256} — but a budget change must move training+deploy+eval TOGETHER (the
  net is calibrated to the 150-sim search). Second-order; won't fix a training stall.
- distinct-corpus-game-rate log (byte-hash game_id) so "is it repeating" is MEASURED next run.
- promotion-health monitor (model_version/promoted) before any relaunch.

## Most-likely outcome (pre-registered)
Given TRUE-STALL + bot SILENT + the clean learning loop (forensic), expect the pretrain-floor fork to
be the highest-information single-variable test. If floor→0.0 does NOT lift deploy-matched strength
over the 120k–200k band, the stall is intrinsic to the v6_live2_ls self-play regime at this capacity,
and the next lever is architectural/data (not a knob). The model⊥minimax intransitivity is the
standing open question — worth a dedicated external-opponent strength axis, not a one-off.
