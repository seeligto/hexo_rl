# D-1M Plateau Relaunch Runbook (READ-ONLY synthesis, 2026-06-24)

Run: `longrun_v6_live2_ls_gumbel_m16` (d1m), live PID 1512427 on vast, ~step 220k. All findings from LOCAL pulled assets + committed code + git history; one prior light read-only ssh grep of the live log. Live PID untouched. NO edits, NO builds performed.

---

## 1. CORRECTION HEADER — dispatcher lever FALSIFIED

**"Relaunch + Dirichlet-off (untested handicap)" is NOT a new lever. Dirichlet was OFF for the ENTIRE d1m run.**

- Variant created at commit `ca621f2` (2026-06-21) with `mcts.dirichlet_enabled: false` ALREADY at config line 43 (operator GO, §D-1M PUCT-1). `git log --all` on the file = 3 commits, none ever flipped Dirichlet ON. `git blame` line 43 = ca621f2 only.
- A2 honest strength log header corroborates independently: `[knobs] ... dirichlet_enabled:false, n_sims_full:150` (reports/d_decide_2026-06-24/a2_eval/strength_curve.log).
- Therefore "Dirichlet-off" is the LIVE STATUS QUO, not a plateau-breaker. The just-committed **B1** (drop redundant Gumbel-root Dirichlet) is a **NO-OP for this variant** (already off) — hygiene for OTHER configs only.

**Five further dispatcher/session premises FALSIFIED (do not carry forward):**
1. "Started from a 94k Dirichlet-ON prior-lineage checkpoint" — FALSE. Fresh from the **8300 SUPERVISED bootstrap** (`bootstrap_model_v6_live2_8300.pt`, sha ebf2ed39); 94k is an intermediate d1m step.
2. "§104 ~2x policy-target floor baked into ≤94k weights" — FALSE. §104 is a self-play completed-Q artifact; the 8300 anchor is a supervised corpus pretrain (no self-play/Dirichlet path → nothing to bake). Live Gumbel-target entropy 0.208 = SHARP, opposite of an inflated floor.
3. "model_version frozen@1 ⇒ zero promotions ⇒ self-play staleness" — **FALSE at the root.** Live log shows **2 real promotions** (60k wr_best=0.585, 120k wr_best=0.62). frozen@1 is a process-RESTART artifact: `InferenceBatcher.model_version` is an in-memory `AtomicU64` (engine/src/inference_bridge.rs:58, init 0) that resets to 0 on each batcher reconstruction; the single startup best_model sync bumps it to 1 — so every restart segment shows frozen@1 regardless of promotions (two distinct run_ids in the log = ≥1 restart). Local `games_current` snapshot (model_version@1, checkpoint_step@120000) is a STALE early-window pull, not a wiring bug.
4. "LR 'floors at 200k then RISES' pathology in effect" — FALSE. That is the COUNTERFACTUAL the C2 fix (`total_steps:1000000`, T_max=1M) prevents. Verified at trainer.py:446 `CosineAnnealingLR(T_max=total_steps, eta_min=5e-4)`. Actual LR is flat-near-peak, NOT rising.
5. "n_sims_full=150 a handicap" — FALSE. 150 > validated-100 = larger Sequential-Halving budget = BETTER targets. Least-likely lever.

Net: of the four R1 levers, three are non-causal status-quo; the fourth (LR) is real but secondary. **B2** (draw-aware promotion-gate CI) is near-inert here (draw_rate≈0.001) and could not have caused the post-120k stall (those evals failed on the wr_best POINT estimate <0.55, not on a CI false-negative). B2 ships as hygiene/insurance.

---

## 2. ROOT-CAUSE READ

**Verdict: genuine FLAT strength trajectory / parity fixed-point — NOT a wiring/Dirichlet/LR/n_sims defect. Confidence HIGH that it is flat vs current instruments; MEDIUM on whether it is a TRUE strength stall vs a MEASUREMENT ceiling (rank-1 discriminator resolves this).**

Live promotion + eval signature (read-only ssh grep, 6-row table):

| eval step | wr_best | promoted | wr_sealbot |
|-----------|---------|----------|------------|
| 30k  | 0.51  | no  | 0.23 |
| 60k  | 0.585 | YES | 0.24 |
| 120k | 0.62  | YES | 0.27 |
| 150k | 0.47  | no  | 0.28 |
| 180k | 0.53  | no  | 0.24 |
| 210k | 0.495 | no  | 0.18 |

Loop turned twice (60k, 120k) then STALLED: self-play stuck on the 120k best_model anchor since ~120k; post-120k candidates sit at parity (~0.50) with their own anchor and never clear 0.55.

Corroborating evidence (verified locally):
- **A2 honest gumbel-greedy curve** (strength_curve.log, recomputed): A_wins **45 / 45 / 51 / 39 / 48** over 95k/120k/150k/175k/200k, **all distinct=100/100** (copy_mult 1.00). Slope ~0. This is the deploy-matched head (no temperature, no PUCT-visit-argmax) — the honest instrument, NOT the biased ModelPlayer proxy.
- **Live wr_sealbot** flat/declining 0.23→0.18.
- §D-LONGRUN-C independently documents the alternating-high/low WR pattern as **healthy loop-turning** (best_model hard-reset on promotion → eval target gets harder → WR drops → climbs back). So the post-promotion dip is expected; the PROBLEM is the climb-back never re-crosses 0.55 after 120k.

Self-play opponent provenance (verified, anchor.py:451-466): self-play consumes the PROMOTED `best_model.pt` anchor (graduation-gate design), re-synced only on promotion via `promote_anchor → pool.sync_inference_weights`. It advanced exactly twice — by design, not a bug. Promotions correctly STOPPED because the candidate stopped beating its 120k anchor at the 0.55 bar.

**Strongest confound on the "true stall" label (why rank-1 runs first):** SealBot is a COMPACT in-window bot. Per gumbel_greedy_bot.py header fidelity gate + CLAUDE.md fixed-bot warning, a fixed in-window bot (a) ceilings-out once the model reaches parity and (b) FALSE-CLEARS off-window defects by construction (the flat PyO3 policy_logit_count vector cannot score off-GLOBAL-window overflow candidates). The A1 "off-window ~0%, colony~0" read was measured vs this same compact instrument on self-play — must be re-validated with an adversarial/spread-uncapped probe before concluding multi-cluster is irrelevant. So a chunk of the "plateau" may be measurement saturation, not a real strength wall.

**Secondary co-driver (medium confidence, mechanism verified):** flat-high LR. `CosineAnnealingLR(T_max=1M)` holds LR near peak across the whole window: ~1.97e-3 @94k → ~1.86e-3 @200k (93-98% of 2e-3 peak). Zero annealing-driven sharpening through the plateau. Compounds a parity fixed-point (no late sharpening on top of stale targets) but does not by itself create one. This is the C2-fix tradeoff: 1M horizon trades the rise-pathology for near-constant high LR.

---

## 3. RECOMMENDED RELAUNCH (single-variable discipline)

**Two-stage. Stage 1 is the load-bearing decision and costs ZERO against the live run.**

### Stage 1 — RANK-1 DISCRIMINATOR (eval-measurable, no restart, no contention) — DO THIS FIRST

Decide measurement-ceiling vs true-stall on the 5 banked local checkpoints. Per CLAUDE.md "a BORDERLINE retraction earns a CHEAP eval-only discriminator before any expensive lever."

Build a **self-anchored BT-MLE Elo ladder** with the deploy-matched gumbel-greedy head:
- Player: `scripts/eval/gumbel_greedy_bot.py` (SH-winner head; NO temperature, NO PUCT-visit-argmax — exactly the live deploy action). **Do NOT use `scripts/generate_trajectory_games.py` / ModelPlayer** (biased PUCT+temp-0 proxy the A2 harness explicitly replaced).
- Ladder set: frozen {8300_bootstrap, 60k(promo1), 120k(promo2), 95k, 150k, 175k, 200k}. Round-robin.
- Elo: `scripts/tournament_validate.py` emits BT-MLE `ratings.csv` anchored at first bot. **GAP: tournament_validate has NO gumbel head wired** — its player args are generic. Either wire `GumbelGreedyBot` as a tournament player, or fit BT-MLE offline over the gumbel_greedy pairwise results. This is the one piece of glue to confirm before launch (see §6).
- Honesty knobs (§D-ARGMAX): opening_plies ≥4 RNG-seeded per game; n_games ≥100/pair; DEDUP byte-identical sequences (distinct_game_count); BOOTSTRAP the Elo CI over DISTINCT games, not raw count.
- Cost: offline, off the training host, ~5 checkpoints. Run on laptop/desktop GPU. ZERO contention with PID 1512427.

**Decision:**
- Rising self-anchored Elo line (Theil-Sen slope >0, bootstrap-CI-lower >0) ⇒ model IS improving, SealBot/A2 was a MEASUREMENT ceiling ⇒ swap the eval instrument for the relaunch (§4), do NOT spend a config restart on the learning loop.
- Flat self-anchored Elo (slope CI straddles 0) ⇒ TRUE stall ⇒ proceed to Stage 2 + open the rank-2 diversity / rank-5 capacity backlog.

Also add the **off-window kill probe (C)** here: enable a spread-uncapped / adversarial opponent (`offwindow_adversary` slot exists, `enabled:false`) on the banked checkpoints to re-validate A1 "off-window ~0%". This is an OPEN gate, not a clear (self-play→external kill link, per CLAUDE.md).

### Stage 2 — RESTART (restart-only; ONLY if Stage 1 shows a true stall, or to reproduce the stall under merged instrumentation)

**Anchor: restart from the 120k promoted best_model. Do NOT continue 200k as-is** (continuing just re-runs the stalled regime). 120k is the last GOOD promoted anchor.

**Config: UNCHANGED** for the first relaunch (single-variable discipline). The variant already has both §D-LONGRUN-C bugfixes baked (`full_search_prob:1.0` line 37, `bootstrap_floor.enabled:false` line 76) + `dirichlet_enabled:false` line 43. The clean first move is: reproduce-or-break the stall under the merged-master instrumentation (B3a structural emit + promotion-health watch), THEN change exactly one knob if it reproduces.

**LR-anneal pin (load-bearing):** restarting mid-cosine, `--iterations` MUST be the REMAINING horizon to 1M (≈ 880000 from the 120k anchor), NOT `1000000` fresh — else CosineAnnealingLR re-peaks LR to 2e-3 (trainer.py:446, T_max derives from total_steps/iterations). Confirm total_steps/scheduler_t_max resolves to the FULL 1M target on resume (trainer_ckpt_load restores scheduler state + repoints T_max), so the LR continues its curve rather than restarting it.

**Pre-launch host steps (config header lines 4-9, operator-run on vast, NEVER under PID 1512427):**
1. `cp` the 120k promoted best_model → `checkpoints/best_model.pt`; re-point `expected_anchor_sha256` (line 72, currently disabled-for-resume) to the **120k** sha and re-enable it as the fresh-launch incumbent assertion.
2. `rm` stale `checkpoints/replay_buffer.bin` + `.recent.npz`.
3. corpus sha-check `data/bootstrap_corpus_v6_live2.npz` → 8f7115ab.
4. SealBot native import preflight.
5. Rebuild the vast `.so` from MERGED master (rsync + git bundle, never `git pull`, never under the live PID).

**Exact command (operator-run on vast only):**
```
.venv/bin/python scripts/train.py \
  --variant longrun_v6_live2_ls_gumbel_m16 \
  --checkpoint checkpoints/checkpoint_00120000.pt \
  --iterations <REMAINING_to_1M, ~880000> \
  --run-name d1m_gumbel_m16_n150_r2 \
  --log-dir logs/d1m_r2
```
Change NO other knob on this relaunch.

---

## 4. EVAL + MONITORING

**Replace the point-in-time SealBot GO-bar with THREE deploy-matched, §D-ARGMAX-honest instruments. SealBot stays DESCRIPTIVE only (at n=100 its Wilson CI is ±10pp — cannot resolve "improving" alone; CLAUDE.md effective-n).**

- **(A) Self-anchored BT-MLE Elo ladder** (PRIMARY GO gauge): gumbel-greedy head round-robin vs frozen past checkpoints {8300, 60k, 120k, then every banked 50k}. Rising Elo = the load-bearing "improving" signal, immune to a fixed-bot ceiling. Cadence A every 50k banked ckpt, offline off the training host.
- **(B) vs_boot absolute reference**: keep the frozen 8300 bootstrap_anchor opponent (config stride4/n50) but score it with the gumbel-greedy head + opening plies + dedup; bump n 50→100. Non-rotating absolute climb signal.
- **(C) Off-window adversarial kill probe** (OPEN gate, every 100k): enable `offwindow_adversary` / a spread-uncapped external bot so off-window play (~0% now) is actually probed. Name the self-play→external kill link as an explicit OPEN gate.
- All three: read DISTINCT-game counts, bootstrap CIs over distinct games.

**Promotion-health watch (must-add before relaunch — TOP priority):** `hexo_rl/monitoring/run_feed_reader.py` (single read-only consumer) currently has NO model_version/promotion watch. Add: tail `evaluation_round_complete` for (step, wr_best, promoted, wr_sealbot) — the 6-row table above — and watch `model_version_max` in `game_complete`. Caveat: model_version frozen@1 is a benign restart artifact (AtomicU64 resets); a TRUE frozen model_version (workers not syncing promoted weights to inf_model) IS the classic stall — watch it live, do not assume from the local snapshot.

**Structural watch (committed 5686b5c, native game_complete emit):** `longest_line_fraction`, `n_components` (per-player, _CLUSTER_THRESHOLD=5 pinned), `colony_extension_fraction`, byte-hash `game_id` (resolves the I7a distinct-game blind spot). A1 baseline: n_components~1, longest_line~6, colony~0, off-window~0%.

**Committed D-MONITORFIX gauges:** F2 robust-slope wr_sealbot (Theil-Sen + measurement-error CI-lower>0, NOT normal-z — old gauge false-greened plateaus at n=5-8); F3 low-floor entropy (warn <1.5 / collapse <1.0; live healthy 2.668); F5 self-baseline depth; F6 Tier-1 (Gumbel-target entropy live 0.208, opening diversity, fp16 AMP-scale canary). HOLD-PARTIAL: F2-C1 cannot green a live SealBot climb until n≥9 — the self-anchored Elo ladder (A) is the real GO gauge.

### Pre-registered SUCCESS criteria (single run, no peeking-to-stop)
IMPROVING (continue) — ALL of:
1. **Promotion health**: ≥1 promotion beyond the restart anchor by +50k AND ≥1 per ~80k thereafter; the 3-consecutive-failed-eval stall must NOT recur for 3+ rounds.
2. **Self-anchored Elo (A)**: Theil-Sen slope >0, bootstrap-over-DISTINCT-games CI-lower >0 (NOT normal-z); effect size (slope×span) ≥ a pre-set Elo floor.
3. **vs_boot (B)**: wr vs frozen 8300 rises, robust-slope CI-lower >0.

### Pre-registered KILL gates
HARD-ABORT monitors already in config (lines 82-87, KEEP): `hard_abort_grad_norm 10.0`; `hard_abort_draw_rate 0.55 ×3 consec`. Add:
1. **PROMOTION-DEAD** (plateau signature): NO promotion within **150k** steps of the restart anchor AND self-anchored Elo slope CI straddles 0 over ≥6 points ⇒ ABORT (same regime as observed 120k→210k+ dead zone).
2. **LOOP-TURNING gate** (Phase-C pre-reg, relaxed for warm restart): first promotion MUST fire by +50k past anchor or FLAG for review.
3. **COLONY/STRUCTURAL**: kill if `colony_extension_fraction` last-50 P90 climbs sustained >0.05 (§157 watch — colony attractor re-engaging).
4. **STRIDE5 SPAM canary**: hard-abort on stride5_run P90 spike (§152 dominant failure mode).
5. **ENTROPY collapse floor**: policy_entropy_selfplay <1.5 warn / <1.0 collapse. NO upper bar.
6. **VALUE-SPREAD**: keep value_spread_alert/canary as canary (already firing 428/14 in live log), not abort.

---

## 5. SECONDARY LEVERS (ranked backlog — only if rank-1 confirms a TRUE stall)

| rank | lever | restart/eval | one-line rationale |
|------|-------|--------------|--------------------|
| **2** | **Training-time opening diversity**: `random_opening_plies` 0 → ≥4 | restart-only | §D-ARGMAX applies to TRAINING: random_opening_plies:0 (line 30) + low opening coverage may collapse self-play to a narrow trajectory distribution → buffer under-covers state space → self-consistent parity fixed point. NEW context (buffer diversity), not the eval-side falsification — test fresh. Cheap pre-check: measure self-play distinct-game/opening-entropy on the live replay stream first (eval-measurable canary). |
| **3** | **Promotion-gate threshold/cadence**: 0.55 bar vs a parity-self model | eval-measurable (analysis) / restart (live change) | After 2 promotions the model sits ~0.50 vs its newest anchor; a real +2-3pp edge is sub-noise at n=320 (Wilson ±5pp). Re-score banked checkpoints offline at varied n/threshold first. TRADEOFF: lowering the bar risks promoting noise (§155 bootstrap-floor / §101 CI guard exist to prevent that) — not a free win. |
| **4** | **n_sims revert 150 → 100** (the ONE deliberate delta vs validated-200k, §D-1M GB-2) | restart-only | Low prior — Gumbel targets valid at all sim counts, and 150>100=larger budget. But it is the named single-variable, so it belongs on the ablation list as a clean A/B (throughput-positive). |
| **5** | **Architecture/anchor ceiling** — larger net OR re-bootstrap from stronger corpus/anchor | restart-only, EXPENSIVE | ONLY if rank-1 confirms a true stall vs a HARDER adversarial instrument AND ranks 2-4 are cleared. LR still ~1.86e-3 (not an under-LR stall) ⇒ if it is real, the limit is capacity/regime. GATE this behind the rank-1 eval discriminator per CLAUDE.md cheap-discriminator-before-expensive-lever. |

---

## 6. OPEN QUESTIONS / confirm before launch

1. **tournament_validate gumbel-head wiring (BLOCKER for rank-1 instrument A):** `scripts/tournament_validate.py` emits BT-MLE `ratings.csv` but has NO gumbel head — only generic player args. Wire `GumbelGreedyBot` as a tournament player OR fit BT-MLE offline over gumbel_greedy_bot.py pairwise results. Confirm before running the ladder.
2. **run_feed_reader promotion/model_version watch (must-add):** not present yet — the most load-bearing live signal (does model_version advance past the restart seed) is currently unmonitored. Add before relaunch.
3. **Remaining-iterations arithmetic:** confirm `--iterations` = exact remaining-to-1M from the 120k anchor (~880000) and that resume re-points scheduler T_max to the full 1M so LR continues (does NOT re-peak). Verify against trainer_ckpt_load scheduler-restore path before launch.
4. **Re-validate A1 off-window ~0% with an adversarial probe** before concluding multi-cluster is irrelevant — it was measured vs the compact instrument that false-clears off-window by construction (gumbel_greedy_bot.py fidelity gate).
5. **Number reconciliation (already understood, restate for the operator):** live PRODUCTION sealbot reads (30k=0.23/60k=0.24/120k=0.27) are §D-ARGMAX-collapsed (argmax + no opening plies → n_eff~2); the HONEST A2 reads (45-51%) are load-bearing. Do not use production-sealbot point reads as a GO bar.
6. **best_model sha for the 120k anchor:** capture/verify it on the host before re-pointing `expected_anchor_sha256` (line 72) for the fresh relaunch.

---

### Restart-only vs eval-measurable summary
- **eval-measurable (no restart, no contention):** rank-1 self-anchored Elo discriminator; off-window adversarial probe; promotion-gate re-scoring (rank-3 analysis); self-play distinct-game/opening-entropy canary (rank-2 pre-check). DO THESE FIRST.
- **restart-only:** the 120k relaunch itself; rank-2 opening-diversity; rank-4 n_sims revert; rank-5 capacity/anchor. GATE all behind the rank-1 result.

Key files (absolute):
- /home/timmy/Work/Hexo/hexo_rl/configs/variants/longrun_v6_live2_ls_gumbel_m16.yaml (line 43 dirichlet:false, 37 full_search_prob:1.0, 20 total_steps:1M, 30 random_opening_plies:0, 73 promotion_winrate:0.55)
- /home/timmy/Work/Hexo/hexo_rl/scripts/eval/gumbel_greedy_bot.py (deploy-matched SH-winner head; off-window fidelity gate in header)
- /home/timmy/Work/Hexo/hexo_rl/scripts/tournament_validate.py (BT-MLE ratings.csv; gumbel head NOT wired — glue needed)
- /home/timmy/Work/Hexo/hexo_rl/scripts/generate_trajectory_games.py (BIASED ModelPlayer proxy — do NOT use for strength)
- /home/timmy/Work/Hexo/hexo_rl/hexo_rl/training/trainer.py:426-446 (CosineAnnealingLR T_max=total_steps)
- /home/timmy/Work/Hexo/hexo_rl/hexo_rl/training/anchor.py:451-466 (self-play consumes best_model anchor)
- /home/timmy/Work/Hexo/hexo_rl/hexo_rl/monitoring/run_feed_reader.py (single read-only consumer; promotion/model_version watch MISSING)
- /home/timmy/Work/Hexo/hexo_rl/reports/d_decide_2026-06-24/a2_eval/strength_curve.log (A2 flat: 45/45/51/39/48, all 100/100 distinct)
- /home/timmy/Work/Hexo/hexo_rl/reports/d_decide_2026-06-24/checkpoints/ (95k/120k/150k/175k/200k .pt — rank-1 inputs)
- /home/timmy/Work/Hexo/hexo_rl/docs/07_PHASE4_SPRINT_LOG.md (§D-LONGRUN-C 2254-2326 healthy-loop-turning confirmation; §D-MONITORFIX 2330+)