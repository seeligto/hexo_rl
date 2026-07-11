# D-F HEADSWAP — build spec of record

Frozen-trunk value-head-swap discriminator for run3 card #1 (distributional
65-bin value head). This file is the load-bearing spec: harness build, negatives
build, scoring, and the final report all bind to it. Verdicts are pre-registered
(§Verdict) BEFORE any arm trains.

## REGISTER GUARD (INV-D1 — non-negotiable)
Every value target = game OUTCOME z only (two-hot over 65 bins, `soft_z_lambda=0`).
NO search-distilled targets, NO TD bootstrap, NO teacher/distillation loss, NO
SealBot/solver/other-net value in ANY gradient path. SealBot appears ONLY as a
probe-set LABEL (evaluation instrument). "Richer search target" is matched-KILL-C
on the falsified register and is BLOCKED. If any step proposes adding it: refuse,
log, continue.

## Question
Does a distributional 65-bin value head, trained on the SAME outcome targets,
recover the losing-tail signal the scalar head misses — and if not, is the failure
in the head or in the frozen trunk features (D-FULLSPEC)?

## Arms (4-arm factorial: head-shape × unfreeze-depth)
- A: scalar head, fresh-init, TRUNK FROZEN         (control: "any retraining helps")
- B: 65-bin head, fresh-init, TRUNK FROZEN         (head-shape effect, frozen)
- C: 65-bin head, fresh-init, LAST BLOCK UNFROZEN  (head-shape + local features)
- D: scalar head, fresh-init, LAST BLOCK UNFROZEN  (control for C)

Trunk init (SHA-verified, run2 lineage — NOT contaminated d1m):
- primary:     checkpoints/run2_retro/checkpoint_00248000.pt (sha 312f85f632ee5046)
- sensitivity: checkpoints/run2_retro/checkpoint_00210000.pt (best-calibrated, WP4)
On box: /workspace/headswap_data/checkpoint_00248000.pt (+210000, +replay_buffer.bin).

All arms: SAME trunk instance, SAME replay sample stream (identical draw seeds),
SAME step count, SAME LR schedule, 3 seeds each (report per-seed + pooled).
Re-init BOTH value layers (fc1+fc2). Unfreeze arms (C/D): unfreeze ONLY
`model.trunk.tower[11]`, LR = 0.1× head LR (two optimizer param groups), else
byte-identical to A/B.

## KEY TRAINING FACT (corrects design doc D3.1)
The replay buffer stores ONE window per row (K=1). The K-cluster `value_pool=min`
argmin does NOT happen at training — it is inference-only. So the 65-bin arm at
training = PLAIN per-row CE against two-hot(z). The argmin-cluster logic lives ONLY
in the probe-SCORING path (§Scoring).

## Model surface (from code scope)
- Encoding v6_live2_ls, in_channels=4, filters=128, res_blocks=12, pool_type=min_max.
- Construct: `HexTacToeNet(board_size=19, in_channels=4, filters=128, res_blocks=12,
  encoding="v6_live2_ls", pool_type="min_max")`.
- Load trunk: `torch.load(ckpt, map_location=..., weights_only=False)["model_state"]`,
  `load_state_dict(state, strict=False)`.
- Value branch (replicate; do NOT rely on min_max_window_head at train time):
  `out = model.trunk(states)  # (B,128,19,19)`
  `v = cat([out.mean((2,3)), out.amax((2,3))], 1)  # (B,256)`
  `v = relu(model.value_fc1(v))  # (B,256)`
  scalar arm:  `v_logit = value_fc2(v)  # (B,1)`  -> compute_value_loss (BCE vs (z+1)/2, mask)
  65-bin arm:  `logits65 = value_fc2_bins(v)  # (B,65)` -> two_hot_ce_loss (scripts/headswap/targets.py)
- Freeze: `requires_grad=False` on all params; then `True` on value_fc1 + value_fc2(_bins);
  unfreeze arms also `True` on `model.trunk.tower[11]`. Optimizer sees only requires_grad params.
- Batch draw: `ReplayBuffer.load_from_path(...)`, `sample_batch_with_pos(n, augment=True)` ->
  states f16 (n,4,19,19), ..., outcomes f32 (n,), ..., value_target_valid u8 (n,). Need
  states, outcomes, value_target_valid. Same draw seed across arms for the shared stream.
- Loss mask = value_target_valid (buffer is 100% valid, but keep the mask for correctness).
- Production knobs: batch 256, LR/optimizer per production defaults; ~20–30k steps/arm.
  A/B are cheap (frozen) — run FIRST for an early read. LR mini-grid {1×, 0.3×} applied
  symmetrically to both arms, best-of by holdout metric (ii) chosen BEFORE any probe scoring.

## Probe set (single-phase!)
All 234 positives are `(side_to_move=head, moves_remaining=2)` — ONE turn phase. This
auto-controls the D-FULLSPEC turn-phase confound. Positive class = loss positions where
the ORIGINAL scalar head stayed optimistic (v_raw ≥ -0.5); selection bias favors a PASS,
so ALSO score the ORIGINAL un-retrained 248k scalar head as a reference anchor.
- 41 WP1 positives: boards reconstruct locally from retro_slope/checkpoint_00248000/games.jsonl
  (VERIFIED 41/41 exact zobrist). Match probe row -> game by (opening_idx, head_as_p1), replay
  to ply `t`, board is the position BEFORE move `t`.
- 193 WP2 positives: source games cleaned from both boxes; boards recovered by regeneration
  (§WP2). Frozen labels/solver proofs retained; only boards regenerated.
Negatives ("safe") = head-turn-start (head, mr=2) positions from run2@248k WON games
(retro_slope 248k, 57 head-won games), SealBot-verified d7 score ≥ 0 AND no mate, matched
1:1 to positives on ply_band = t//10 (single phase, so ply_band only). Phase+ply canary
must be ≤ 0.55 or NO-TEST.

## Scoring (multi-window inference path)
For each probe position (pos + neg), reconstruct the board, run the K>1 multi-window
inference (real legal-set windows). For each arm's trained head:
- scalar arm: per-cluster tanh value -> argmin cluster -> that cluster's scalar v.
- 65-bin arm: per-cluster logits65 -> decode each to scalar (E[softmax·support]) ->
  argmin cluster -> report THAT cluster's decoded scalar v AND its tail-mass P(v≤-0.5).
Argmin in DECODED-SCALAR space (matches production min-on-tanh; logit-space pooling barred).

## Metrics (all pre-registered; reports/headswap/)
PRIMARY gate = paired ΔAUC (arm-D-shape − arm-S-shape) of lost-vs-safe on the matched set.
Report AUC under BOTH score functions for 65-bin arms:
  (a) decoded scalar v  — the FAIR comparate to the scalar head (design-doc primary; the gate).
  (b) loss-tail mass P(v≤-0.5) — the mechanism readout the dispatcher headlines (descriptive).
Gate on (a); headline (b). If (b) passes but (a) does not: NULL-with-signal — the 4-arm
structure (C vs D) then adjudicates the feature-adaptation route. Pre-commit tail threshold =
bins 0..16 (support[16] = -0.5 exact). Red-team: monotone-transform invariance + threshold sweep.
Secondary (sanity, quote all): (ii) decided-row sign accuracy on holdout (row-random split —
buffer game_id all -1, game-disjoint impossible; flagged); (iii) 10-bin value ECE; false-pessimism
on the matched WON-control set (a doom-happy head must not score free points). CI = cluster
bootstrap by SOURCE GAME (10k resamples) over paired per-position ΔAUC. Report nominal n,
distinct positions, distinct source games, achieved SE, empirical between-arm score r.

## PRE-REGISTERED VERDICTS (write into report BEFORE training)
- PASS:       B > A by AUC ≥ +0.05 (bootstrap CI excl 0) AND false-pessimism ≤ A + 5pp
              → card #1 CONFIRMED for run3; frozen-trunk sufficiency noted.
- PASS-JOINT: B ≈ A but C > D by ≥ +0.05 (CI excl 0) → head-shape needs feature adaptation
              → card #1 proceeds, run3 trains head jointly from start; D-FULLSPEC partly corroborated.
- TRUNK-FORK: B ≈ A AND C ≈ D → head shape buys nothing even with local features → card #1
              DEMOTED below card #2; D-FULLSPEC representation hypothesis PROMOTED; ESCALATE to
              operator before RUN3SPEC freezes. Do not soften this in the report.
- Any arm diverges/NaNs → fix or report, never swap thresholds.
Pre-registered expectation (honesty): D-FULLSPEC raises P(B≈A); the informative arm is likely C-vs-D.

## WP2 board recovery (regeneration fallback)
Pull impossible (games cleaned). Regenerate the 5 WP2 batches on the box from LOCAL books
(reports/valprobe/wp2/evalfair_r5_wp2_b*.json) at ckpt 248k, deploy head 150 sims m=16 vs
SealBot-d5, KEEPING games.jsonl. Smoke-check 1 batch: how many frozen WP2 positives' zobrists
reproduce (cross-GPU Gumbel non-determinism expected). If reproduction × 193 + 41 ≥ 200 →
recover-exact (B1). Else fresh-extract equivalent trap-formation positives (same criterion) to
top up to ≥200, flagged as fresh-equivalent. Report reproduction rate + composition either way.

## Deliverable
reports/headswap/VERDICT.md (one line first: PASS / PASS-JOINT / TRUNK-FORK) + per-arm JSONs +
trained heads. rsync OFF the ephemeral box after EVERY completed stage (box vanishes; laptop is
the record). Existence-checked.
