# Escalation E1 — Paired short-self-play trajectory: does a distributional value head help WHEN the trunk co-adapts?
# (Follow-up to D-F HEADSWAP. Dispatch-ready spec for the overarching agent.)

## 0. Why this exists (context — do not re-derive)
D-F HEADSWAP (frozen-trunk 4-arm discriminator: scalar/65-bin × frozen/last-block-unfrozen) returned
**NULL-with-signal**, NOT a clean TRUNK-FORK:
- On the pre-registered decoded-**MEAN** gate, the 65-bin distributional value head buys nothing over
  scalar on the 234-position value-tail probe (ΔAUC B−A = −0.017; C−D = −0.001), frozen or last-block.
- BUT the distributional **TAIL** readout (P(v≤−0.5)) PASSES on the matched gate set:
  **C.tail − D.v = +0.052, CI[+0.018,+0.087]**, threshold-robust (flat 0.927–0.931 across bins 8–24).
  → card #1 = **demoted-not-DEAD**.
- Red-team correction (load-bearing): the frozen trunk is **NOT rank-blind** — a fresh scalar head hits
  AUC 0.885 and recognizes off-probe kraken-MCTS losses 20–40 plies earlier than the production head.
  The production defect is **LEVEL / CALIBRATION** (original head reads mean v = +0.73 on lost positions
  while ranking them fine, anchor AUC 0.824). The experiment gated on **RANK (AUC)**, which is partly
  blind to a LEVEL defect. → **D-FULLSPEC (representation-blind) is UNSUPPORTED**; the lever is
  head-training/calibration + the distributional tail, NOT an expensive trunk/representation restart.

The frozen discriminator is structurally blind to the distributional head's actual mechanism:
**CE-over-a-support reshaping the TRUNK during JOINT training.** A frozen null can only DEMOTE, never
KILL. E1 resolves that confound cheaply.

## 1. Question (frozen)
When the trunk is allowed to CO-ADAPT to a 65-bin distributional value head during real self-play
(trunk trainable, not frozen), does the head's **LEVEL/calibration** on the value-tail probe improve
over a matched scalar head — and does that improvement **GROW with training steps** (the
representation-shaping signature the frozen test cannot see)?

## 2. Design — paired short-self-play trajectory
- TWO runs, **identical config/seed/data, ONLY the value head differs**:
  - **65-BIN arm**: distributional value head; two-hot outcome-z target (λ=0); MCTS/deploy consume
    decode = E[softmax·linspace(−1,1,65)].
  - **SCALAR arm**: production scalar value head (control).
- **WARM-START from the converged HEADSWAP heads** on the run2 248k trunk (65-bin ← arm-B head,
  scalar ← arm-A head). Skips the fresh-head garbage transient (heads already trained) and continues
  the HEADSWAP ladder seamlessly: frozen (0 adapt) → last-block (minimal) → **full-trunk self-play (E1)**.
- **UNFREEZE the FULL trunk** — the whole point; let it co-adapt. Else = production self-play config.
- **RESUMABLE TRAJECTORY**: configure a full ("1M-step") run but CHECKPOINT + EVALUATE at
  **5k / 10k / 20k / 50k**. **First read at 5k** — cheapest, and warm-start makes 5k a meaningful
  trunk-FINE-TUNE nudge (not from-zero); below ~2k is transient noise. Extend ONLY if the gap opens —
  do NOT commit the GPU-week until the trajectory earns it. (5k aligns with run2's existing bank grid.)

## 3. Metrics (per checkpoint, per arm) — the LEVEL correction is load-bearing
PRIMARY = **LEVEL / CALIBRATION** on the 234-probe (the ACTUAL defect; AUC misses it):
  - recognition-lag = mean v on matched lost positions (D-C WP1) — drops below the optimism threshold sooner?
  - value ECE (D-C WP4) + mean-v-on-losses.
SECONDARY (rank, continuity with HEADSWAP): decoded-v AUC + tail-mass AUC on the 234 probe, paired 65-bin vs scalar.
TERTIARY (strength sanity, weak early): ~50-game paired H2H (65-bin-ckpt vs scalar-ckpt), DIVERSE openings —
  directional only; the value head barely drives move choice at 5–10k.
**KEY READOUT = the TRAJECTORY**: does the 65-bin arm's calibration/tail advantage GROW 5k→50k
(representation-shaping compounding) or stay flat (confirm demote)?

## 4. Pre-registered verdict (see R2 AMENDMENTS below — A1 softens REVIVE, A5 fixes the map)
- **REVIVE**: 65-bin's calibration/tail advantage over scalar (A1) has a POSITIVE SLOPE across 5k→50k
  AND a final-point gap CI (bootstrap over the 234 probe positions) EXCLUDING 0. (Strict monotonicity on
  4 noisy points is DROPPED.) → representation-shaping is real → card #1 to full run3 primary.
- **CONFIRM-DEMOTE**: gap flat/absent through 50k → joint training doesn't rescue the distributional head
  → run3 primary = scalar head + retrain/recalibration + tail-style monitoring; distributional head PARKED,
  register-noted as calibration-equivalent.
- Asymmetry: a POSITIVE at 5k is strong+cheap; a NULL at 5k/10k is weak (compounds later) → read to 50k
  before CONFIRM-DEMOTE.

## 5. Register guard (INV-D1, binds every stage)
Value target = game OUTCOME z ONLY (two-hot, soft_z_lambda=0). No teacher / TD-bootstrap / distill /
solver value in ANY gradient. SealBot = probe LABEL only. λ=0 pinned (root_value blend is a separate card).

## 6. Cost / integration
- CHEAP: 5k–50k paired self-play ≈ hours-to-a-day (vs GPU-weeks for a full run); fits a 1–2 day box.
- INTEGRATION (bigger than HEADSWAP, which bypassed the trainer): wire the 65-bin value head + two-hot CE
  loss into the PRODUCTION self-play trainer (scripts/train.py + net value head + MCTS value decode).
  Reuse: two-hot primitive scripts/headswap/targets.py; converged heads; 234 probe; D-C recognition-lag/ECE.
- OPTIONAL ~2h derisk: a bootstrap-corpus paired probe (65-bin vs scalar pretrain) gives a directional read
  + derisks the integration — but it is OOD (bootstrap corpus ≠ probe's run2 self-play distribution), so read
  only the RELATIVE sign, not absolutes. Lower priority than E1 itself.

## 7. Assets (all exist)
- Converged HEADSWAP heads: /home/timmy/headswap_safe/box_results/headswap/{ab,cd}/arm_*/head_*.pt
- run2 248k trunk (sha 312f85f632ee5046): checkpoints/run2_retro/checkpoint_00248000.pt
- 234 probe + 4204 negatives: reports/valprobe/{probe_set_v1,negatives_v1,negatives_v2_wp2}.jsonl
- two-hot loss: scripts/headswap/targets.py · recognition-lag/ECE: scripts/valprobe/{measure_recognition_lag,value_health}.py

## 8. R2 AMENDMENTS (BINDING — operator ruling 2026-07-11)
E1 = run3 card #1 trainer integration + validation + full-path run3 SMOKE, in one.
- **A1 (REVIVE softened):** positive SLOPE of the between-arm calibration gap 5k→50k AND final-point gap
  CI (bootstrap over the 234 probe positions) excluding 0. Strict monotonicity on 4 noisy points dropped. (folded into §4)
- **A2 (freeze the metric):** freeze the numeric gap definition — metric = recognition-lag mean-v-on-losses
  delta + ECE delta; the bootstrap scheme; the thresholds — in the OFFICIAL spec BEFORE the 5k read. No post-hoc.
- **A3 (run3-lineage code):** both arms run on run3-lineage code — CONFRES-resolved config + watchdog ARMED
  (`selfplay_stall_timeout_sec` explicit) + the promotion-gate isolation fix if landed. E1 IS the run3
  launch-path rehearsal → treat ANY E1 infra failure as a RUN3-BLOCKING bug, not an E1 nuisance.
- **A4 (venue):** second box if it survives the integration lead-time; else ORIGINAL box immediately
  post-run2-stop (R3). Do NOT rush integration to beat the box clock.
- **A5 (verdict map):** REVIVE → card #1 to full run3 primary. CONFIRM-DEMOTE (flat through 50k) → run3
  primary = scalar head + retrain/recalibration + tail-style monitoring; distributional head parked,
  register-noted calibration-equivalent. (folded into §4)

## 9. STANDING GUARDS (R5, restated)
No search-distilled value targets / TD-bootstrap / teacher-in-loss in run3 v1 (blocked 3x). ONE variable =
the value head. Training sims = 150. Every strength claim carries: protocol + n + eff_n + per-side compute.
