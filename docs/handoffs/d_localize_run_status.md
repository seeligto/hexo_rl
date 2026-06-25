# D-LOCALIZE — live run status (2026-06-24/25, autonomous session)

TL;DR — two long jobs running in parallel; the live d1m run was **NOT** stopped.

## Decision log (autonomous, operator away)
- **Live d1m run LEFT RUNNING.** Probed vast: PID 1512430 is **healthy + actively
  training** (step 248k→249k+ in real time, GPU ~82%, checkpoints every 500 steps —
  the "24h-stale log" was a vast-clock-skew illusion, not a hang). The dispatcher
  PRECONDITION "stop live PID" was labeled operator-manual and never run; killing a
  healthy 2-day run unattended was not explicitly authorized, so I did **not**.
  - D-LADDER's TRUE-STALL verdict still holds strength-wise (248k still in the flat
    regime), so continuing it costs compute but isn't producing new strength.
  - **To free the full 5080 for P3:** `ssh vast 'kill -INT 1512430'` (graceful;
    it checkpoints on SIGINT). Then P3 can be relaunched on the whole box.
- **P3 → vast co-tenant (no kill).** P3 is SealBot-CPU-bound + GPU-light; the live
  run is GPU-bound → complementary. Running `nice -19`. Measured impact: live run
  kept training (step advanced), GPU mem +~0.5 GB. Strictly more conservative than
  the dispatcher's "stop it".
- **P2 → laptop 4060** (no GPU contention between boxes).
- **Loss insurance:** pulled latest d1m checkpoints (246.5k–248k) to
  `checkpoints/d1m_vast_latest/` (you had nothing past 226.5k locally).

## WF1 (done) — foundation
- **P0:** depth-5 = the reproducible SealBot bar; **unit = HTTT turns** (depth-5 ≈
  10 stones; median 4 + chosen 5 same unit, 5 = median+1). Settles the unit rule.
- **P1:** banked jsonl = PARTIAL (moves, no per-move value) → P2 re-evals, no fresh
  games.
- **P4 (SHIPPED, working tree, uncommitted):** deploy-matched in-loop strength eval
  — `hexo_rl/eval/deploy_strength_eval.py` (Gumbel SH greedy, **g=0** via
  `gumbel_scale=0.0`, no temp, deploy sims; fixed-depth-5 SealBot; adaptive
  screen(80)→confirm(200); distinct-game bootstrap BT-Elo gate; fail-safe, no
  PUCT/temp/64 fallback). Default-OFF opponent `deploy_strength` so existing runs
  are bitwise-unchanged. Tests added. Pending RED-TEAM validation (WF2) before commit.
- **Scout:** P2 mechanism verified live; gates pre-registered (see below).

## WF2 (RUNNING, laptop) — task ww1c1rpi9
P2 gap-localization on 68 lost mid-cluster games (s150k=18, s175k=26, s200k=24) via
full per-ply SealBot-d6 `last_score` scan + persistence-to-terminal filter →
classify each decisive blunder LINES / VALUE / TACTICS (pre-registered numeric
gates). Then held-out REVIEW (s175k re-derive) + RED-TEAM (d7 reference-artifact
check, decision-time off-window, P4 g0/gate/false-negative + run P4 tests) →
synthesize the Stage-2 lever. Outputs: `reports/d_localize_2026-06-25/`
(`p2_decisions.jsonl`, `p2_summary.md`) + `scripts/eval/p2_localize.py`.

## P3 (RUNNING, vast tmux `p3sweep`) — search-scaling sweep
Driver: `scripts/run_d_localize_p3_vast.sh`. Order (decision-relevant first):
n150 → n256 → n128 → n512 → depth-4 red-team@n256. SealBot-only, depth-5,
40 games/pair, 6 ckpts. Outputs stream to `reports/d_localize_p3/<n>/sealbot_games.jsonl`.
Pre-registered gate: PLATEAU-by-150 (keep n=150) vs CLIMBS-past-256 (adopt 256);
decision-relevant comparison is 150→256. Expected PLATEAU. ~11h.
- **Check:** `ssh vast 'tmux capture-pane -pt p3sweep | tail'` or
  `tail reports/d_localize_p3/run.log`.
- **Aggregate + pull back** when done: see
  `docs/handoffs/d_localize_p3_search_scaling_vast_runbook.md` §Step 6 / §OUTPUT.
  (Aggregate needs the banked net-vs-net rows — run on laptop after rsync if simpler.)

## DONE (post-WF2)
- **P2 verdict = VALUE-TARGET** (value head blind to SealBot-reachable losses;
  corrected discriminator VALUE-BLIND 56/61 = 92%). bot-mix DROPPED, corpus regen
  OFF/spec-only. Full record: `docs/handoffs/d_localize_findings.md`.
- **Committed** on `phase4.5/d-decide-track-b` (NOT master, not pushed):
  `4449a15` P4 deploy-matched in-loop gate · `cf89801` localization + ladder n-override
  + findings + sprint-log §D-LOCALIZE · (+ p3_verdict aggregator). 14 P4 tests pass.
- Corpus regen SPEC-ONLY (gated on P2=LINES, did not fire). No GPU-week launched.

## P3 (DONE) — PLATEAU-by-150
- All 1200 games complete on vast (~7h, co-tenant); results pulled to
  `reports/d_localize_p3/` (gitignored) incl. `P3_VERDICT.md`. tmux sessions cleaned up.
- **Verdict: PLATEAU-by-150** — no checkpoint has CI_lo(WR@256) > CI_hi(WR@150); 150→256
  drops/flatlines (s150k .575→.400, s175k .350→.250), n512 no climb, depth-4 ≈ depth-5.
  ⇒ keep n=150, 256 not worth 1.7×. Confirms value-blindness ≠ search-budget artifact.
- Live d1m run unaffected: advanced 249k→269k during the sweep, still healthy.
