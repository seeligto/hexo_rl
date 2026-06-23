# D-TEMPDECAY Phase-2 — draw-safety probe runbook (operator-run on vast)

Static-weights generation probe. Discriminates the **L9-transfer = UNKNOWN** gate
(does within-game cosine → draw-collapse transfer to a floor of 0.20–0.45, vs the
§143 toxic 0.05?) **before** any training spend. Pre-registration + baselines + the
verbatim caveats live in `reports/investigations/tempdecay_phase0_2026-06-12.md`.

- **Host:** new vast `ssh -i ~/.ssh/vast_hexo -p 31743 -t root@ssh9.vast.ai`.
  Do NOT touch the Arm-C re-run host. Cost ≈ $1–2.
- **Model (frozen):** `checkpoints/golong_bank/checkpoint_00050000_PEAK_sb0.38.pt`
  (golong@50k PEAK, v6_live2). No training — generation only.
- **Arms (4):** `tempdecay_probe_{control,a45,a30,a20}` — control τ=0.5 (schedule OFF)
  + quarter-cosine threshold=12 turns with floor {0.45, 0.30, 0.20}. They share
  golong's generation regime (cap+CQV, jitter, n_sims) — the **only** variable is
  temperature. Configs validated to load + resolve (`make`-free `load_config` check).

## Run (per arm)

Pre-flight — 2-game smoke per arm FIRST (confirms the config loads, the model
loads, and the schedule fires before the full spend):

```bash
CKPT=checkpoints/golong_bank/checkpoint_00050000_PEAK_sb0.38.pt
for arm in control a45 a30 a20; do
  python scripts/smoke_selfplay_bootstrap.py --checkpoint "$CKPT" \
    --variant "tempdecay_probe_${arm}" --n-games 2 --mode mcts
done
```

Full run — N=2000 distinct games per arm (`--mode mcts`; the harness prints
`draw_rate`, game-length percentiles, x/o/draw counts):

```bash
for arm in control a45 a30 a20; do
  python scripts/smoke_selfplay_bootstrap.py --checkpoint "$CKPT" \
    --variant "tempdecay_probe_${arm}" --n-games 2000 --mode mcts \
    | tee reports/tempdecay_probe_${arm}.json
done
```

Each arm records ALL games to `logs/replays_tempdecay_<arm>/*.jsonl`
(`game_replay.sample_rate: 1`) for the secondary analysis.

## Measure

| metric | source |
|---|---|
| **draw rate** (PRIMARY gate) | harness `draw_rate` (= `pool.draws/games`) |
| game length dist | harness `{median,mean,std,min,max}_game_length_plies` |
| opening diversity (distinct first-2-turn sets) | `logs/replays_tempdecay_<arm>/*.jsonl` |
| forced_win_conversion, off-window | `python scripts/golong_game_analysis.py --encoding v6_live2 logs/replays_tempdecay_<arm>/*.jsonl` |
| **τ-fired verification** | opening diversity of schedule-ON arms (early τ=1.0) MUST exceed control (τ=0.5); reference curve = `hexo_rl.selfplay.utils.quarter_cosine_temperature` |

**Dedupe byte-identical move sequences before any draw-rate CI** (effective-n =
distinct games; §D-ARGMAX). N=2000 gives draw-rate SE ≈ 0.007 at p≈0.09 → the 0.10
abort margin resolves at >10σ.

## Pre-registered verdicts (FIXED — no post-hoc moves)

- **PROBE-ABORT (per arm):** draw rate > **0.20** absolute **OR** > (control + **0.10**).
  → arm DEAD, do not train it.
- **PROBE-PASS (per arm):** draw rate ≤ (control + 0.10) AND ≤ 0.20 AND the τ-fired
  check confirms the schedule ran.
- If A45 **and** A30 **and** A20 all abort → schedule family questioned; STOP, report,
  no training smoke.
- **Static-regime caveat (restate verbatim in the report):** this reads the STATIC
  regime only. L9's collapse was a TRAINING dynamic (feedback through generations).
  Probe-PASS ≠ collapse-safe; it de-risks only the immediate generation-side effect.
  A "~0" here from the wrong regime is NOT evidence of absence. The Phase-3 training
  smoke carries its own in-run draw gate (recent draw ≥ 0.30 / 3-consec).

## Decide → Phase 3

Smoke (Phase 3) runs control vs the **lowest draw-safe survivor** (tie-break: best
probe finishing/diversity). golong@50k baselines the smoke must move:
draw 0.093, value_loss 0.544, forced_win_conversion 0.887 (in-window 0.85),
off_window 0.219. `value_accuracy_masked` is NOT logged anywhere — Phase-3 must add
per-source logging before the smoke (Phase-1b item, not yet built).
