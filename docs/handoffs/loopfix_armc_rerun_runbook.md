# §D-LOOPFIX — Arm-C re-run runbook (DESIGN-ONLY, operator-gated GPU)

Status: **DESIGN-ONLY. DO NOT LAUNCH.** Gated on the proven fixed loop (this pass).
Supersedes the as-run Arm C, which §D-PROMOGATE ruled **A/B-INVALID-AS-RUN** (stale
golong@50k incumbent + frozen self-play generator, n=50 underpowered gate, zero
in-run Objective-A reads, terminal eval killed by the 900 s drain).

## What the fixed loop changed (why a re-run is now meaningful)

| defect (as-run) | fix landed (§D-LOOPFIX) |
|---|---|
| terminal promotion round killed at 900 s drain; stride-4 nnue/offwindow got 0 reads | W1 close-out: budgeted drain + TERMINAL full-battery eval (stride ignored) on the final checkpoint, run UNLOADED (pool stopped first) |
| best_model.pt silently = golong@50k via .bak restore (wrong incumbent + generator) | W2 pin: `expected_anchor_sha256` HARD-FAILS launch on the wrong incumbent; `anchor_identity` logged |
| promoted anchors logged step=0 (indistinguishable from bootstrap) | W3 stamp: step + run_id + encoding + `.provenance.json` sidecar |
| n=50 + CI guard → bar 0.64, P(promote\|0.55)=0.127, dead band 0.55-0.62 | POWER: n=400 → bar 0.55, P(promote\|0.55)=0.521, P(false\|0.50)=0.0255, dead band gone |
| eval WR load-dependent (0.36 live / 0.52 idle) | co-tenancy: terminal eval runs UNLOADED; n=400 makes in-run small-n non-decisive |

## Re-run config (variant `configs/variants/v6_live2_ls_ab.yaml`)

Treatment = encoding `v6_live2_ls` (multi-window legal-set; flips is_multi_window +
k_max 1→8 + legal-set action policy) vs the same `bootstrap_model_v6_live2` bootstrap
as Arm A. The variant is made re-run-ready:

- **Incumbent PINNED:** `eval_pipeline.gating.expected_anchor_sha256:
  "aba28e10bd80b2bac65e9b33e109cb9dc36a3a83871bf3a3fff0ca0f96d27165"` — the
  state-dict sha256 of `bootstrap_model_v6_live2.pt` (reproduce:
  `python scripts/anchor_sha256.py checkpoints/bootstrap_model_v6_live2.pt`). Launch
  HARD-FAILS if best_model.pt is anything else (closes W2). **Preflight: `rm -f
  checkpoints/best_model.pt checkpoints/best_model.pt.bak` so resolve_anchor
  fresh-inits the pinned bootstrap — do NOT inherit a stale anchor.**
- **Cadence (W1):** `best_checkpoint.stride: 1` + `bootstrap_anchor.stride: 1` so
  EVERY eval interval (12500) is promotion-capable AND has the bootstrap-floor
  measurement (else the floor's missing-measurement default blocks odd rounds).
  `best_checkpoint.n_games: 400` (POWER). The `bootstrap_anchor` floor opponent is
  raised off `model_sims/opponent_sims: 1` (the §D-PROMOGATE 11,032 s sims=1 colony
  cell — 551 s/game) to a decisive sim count so it does not dominate every round +
  the terminal battery.
- **Objective-A in-run (the GREENLIGHT axis):** `nnue.stride: 2` and
  `offwindow_adversary.stride: 2` so the strength-ladder + robustness reads land at
  rounds 2 and 4 IN-RUN (not just the terminal battery). `offwindow_adversary.enabled:
  true` is the standing run-readiness flag.
- **Close-out:** `terminal_eval_enabled: true` (default) → the final checkpoint gets a
  full-battery, stride-ignored, promotion-capable decision.

## Launch (operator-run on vast; DO NOT run from here)

```
rm -f checkpoints/best_model.pt checkpoints/best_model.pt.bak checkpoints/replay_buffer.bin
python scripts/train.py \
  --checkpoint checkpoints/bootstrap_model_v6_live2.pt \
  --variant v6_live2_ls_ab \
  --iterations 50000
```
Verify at startup: `anchor_identity sha256=aba28e10… pinned=aba28e10…` (no mismatch
RuntimeError); `eval_schedule_capability promotion_capable_in_run_rounds=4`.

## GREENLIGHT routing (axes restated)

- **PRIMARY — absolute off-window robustness (NO matched arm needed).** The greenlight
  rests on Objective-A robustness, per §D-EXTLINK / the multicluster design §2:
  `scripts/exploit_probe.py --checkpoint <Arm-C-final> --arms exploit,control` ≤ **0.06**
  AND the causal-uncapping counterfactual read, routed through `KClusterMCTSBot`
  (NOT the single-window `ModelPlayer`, which drops off-window moves and FALSE-CLEARS).
  These are ABSOLUTE thresholds — they do not need Arm A/B for comparison. Arm C must
  also beat the Arm-B free `KClusterMCTSBot` overlay on robustness (the training-half's
  justification).
- **SECONDARY — strength non-inferiority guard vs golong@50k, WITH THE DISCLOSED
  CONFOUND.** The strength read is a SAFETY guard (does multi-cluster COST strength?),
  not a credit claim. **DISCLOSED CONFOUND: the fixed loop biases Arm C UP** — it now
  has a pinned bootstrap incumbent (not golong), a fixed promotion cadence, n=400
  power, and an unloaded terminal eval; the original Arm C had none of these. So the
  guard can **DETECT HARM** (Arm C materially weaker than golong50k → multi-cluster
  costs strength → do NOT scale, route strength to Gumbel search) but it **CANNOT
  ATTRIBUTE GAINS** to the encoding — any Arm-C ≥ golong50k result is confounded by the
  loop fixes and must NOT be read as "the encoding made it stronger." State this in any
  greenlight write-up so nobody later claims encoding credit for loop gains.

## Confounds that remain (state before any greenlight read)

- Fixed-loop bias on the strength axis (above) — the strength guard is harm-detection
  only.
- Arm C vs Arm A still differ beyond the encoding on launch warmth (golong warm from
  the 30k baseline vs Arm C fresh bootstrap) — but the encoding robustness axis is
  absolute (exploit_probe), so this does not gate the PRIMARY read.
- The strength instrument's effective-n caveat (§D-ARGMAX): if comparing via argmax/
  temp-0 from a fixed opening, dedupe byte-identical games + bootstrap the CI over
  DISTINCT games before trusting any strength gap.

## Do NOT

- Do NOT launch (operator-gated GPU).
- Do NOT read a greenlight off strength alone, or off a SealBot-WR (the project's
  flagged-wrong strength instrument, §D-FOUNDING).
- Do NOT route the robustness read through the single-window `ModelPlayer` (false-clears).
