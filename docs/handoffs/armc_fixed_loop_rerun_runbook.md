# Arm-C fixed-loop RE-RUN runbook — the FALLBACK encoding-strength read

Status: **DESIGN-ONLY. DO NOT LAUNCH from here (operator-gated GPU).** ~$22 / ~1.5 days on
vast (5080). **This is the dispatcher's FALLBACK, not the primary call** — current lean is
**route strength to Gumbel**; launch this re-run only if Gumbel stalls (or the operator wants
the encoding-ceiling read directly). Supersedes the as-run `armc50k_20260614T154502Z` for the
STRENGTH axis only — its PRIMARY off-window GREENLIGHT is banked and clean
(`armc50k_kcluster.summary.json` = DEFENDED 0.0) and is NOT re-opened.

## Why a re-run is a legitimate fallback (what the crashed loop left unknown)

The as-run 50k read "flat ≈ bootstrap → route to Gumbel." The headline is RIGHT but for a
reason the run could not settle (see `reports/investigations/armc50k_strength_reconcile_20260616.md`):

1. **LOOP-CRIPPLED.** nnue/hammerhead `ModuleNotFoundError` crashed the eval at rounds 2, 4,
   and terminal → every earned promotion discarded (25k=0.668, 50k=0.668, terminal=0.685, all
   > the 0.55 bar). The self-play generator weights swap **only on promotion**
   (`eval_drain.py:111`); **0 promotions → generator frozen at bootstrap all 50k**, with a
   material selfplay share (0.2→0.375). The loop never turned — so the encoding's strength
   ceiling *under a working loop* is genuinely unknown. That gap is what this re-run closes.
2. **The flat SealBot read is VALID (not an instrument artifact).** A first clean-ladder pass
   APPEARED to lift 50k to 35% and invert the lean — but that was a methodology error:
   `--random-opening-plies 4` (§174 inflates WR) + a temp change. The matched read (temp 0.5,
   0 opening plies = exact in-run conditions) = **18.3%**, reproducing the in-run 14–20% band; a
   drop-vs-no-drop control showed the off-window drop is **immaterial** for on-distribution SealBot
   strength. So the model IS flat-to-weak vs SealBot (~0.15–0.20), with a FROZEN generator. Weak
   evidence a working-loop re-run cracks SealBot → hence Gumbel leads and this re-run is the
   fallback.

## The ONE thing this re-run fixes vs the crashed run

| as-run defect | fix (must be live on the host before launch) |
|---|---|
| nnue opponent `ModuleNotFoundError: hammerhead` → eval crash → promotions eaten → generator frozen | hammerhead built into the run venv — `scripts/install.sh` step 8 (master `3e878fb`): `maturin develop --release` on hammerhead-engine + `pip install -e hammerhead`. **Verify `from hammerhead import Bot` resolves on the host.** |
| in-run SealBot / off-window reads via single-window `ModelPlayer` (drops off-window legal moves → understates strength, false-clears robustness) | `hexo_rl/eval/defender_dispatch.py` + the routed `Evaluator` (this session, uncommitted): legal-set model-under-test → no-drop `KClusterMCTSBot`. Sync `defender_dispatch.py`, `evaluator.py`, `inference_methods.py` to the host so in-run reads are the correct instrument. |

Everything else is IDENTICAL to the as-run (the W1–W3 loopfix machinery — pinned incumbent,
n=400 power, stride-1 cadence, terminal close-out — was already in the run and worked; only
the eval-opponent build was missing).

## Launch identity (unchanged from as-run)

- Variant `configs/variants/v6_live2_ls_ab.yaml`; anchor `bootstrap_model_v6_live2.pt`
  (state-dict sha256 `4198d5cb…b0a186`, pinned via `expected_anchor_sha256`).
- 50k iterations from-bootstrap; n_workers 18 (host-fit) or the committed 16.

```
# PRE-FLIGHT (the as-run's lesson — ABORT if any fails):
.venv/bin/python -c "from hammerhead import Bot; print('hammerhead OK')"
.venv/bin/python -c "from hexo_rl.bots.nnue_bot import NnueBot; print('nnue OK')"
.venv/bin/python -c "from hexo_rl.eval.defender_dispatch import build_model_bot; print('no-drop dispatch OK')"
python scripts/anchor_sha256.py checkpoints/bootstrap_model_v6_live2.pt   # == 4198d5cb…
rm -f checkpoints/best_model.pt checkpoints/best_model.pt.bak checkpoints/replay_buffer.bin

# LAUNCH:
python scripts/train.py \
  --checkpoint checkpoints/bootstrap_model_v6_live2.pt \
  --variant v6_live2_ls_ab \
  --iterations 50000
```
Startup must show: `anchor_identity sha256=4198d5cb… pinned=4198d5cb…` (no mismatch) and
`eval_schedule_capability promotion_capable_in_run_rounds=4`.

## Pre-registered reads (fixed BEFORE the numbers)

**Loop-turned check (the necessary condition this re-run exists to satisfy):** ≥1 promotion
must LAND (the as-run cleared the bar 3× and promoted 0×). Expect promotions at the rounds
that beat 0.55 vs the *advancing* best; `best_model.pt` sha must change from `4198d5cb…`.
If 0 promotions again → eval still broken, the read is null again, do NOT conclude strength.

**STRENGTH verdict — MATCHED CONDITIONS ARE MANDATORY.** Measure SealBot at the *exact* in-run
eval settings — `eval_temperature 0.5`, `eval_random_opening_plies 0` (NOT 4 — §174 inflates),
mcts-128, SealBot t=0.5 — and compare to the as-run **frozen-loop baseline ≈ 0.18 (50k, matched,
`armc_50k_matched_inrun.json`; in-run band 0.14–0.20)** and to golong measured **at the same
matched settings** (re-measure golong; do NOT use the bare 0.38 PEAK label — provenance/settings
unknown). The drop instrument is immaterial here, so single-window ModelPlayer or KClusterMCTSBot
both work for the strength number; use KClusterMCTSBot only because it is the architecturally
faithful decode. n≥100, distinct-games CI.
- **CRACKS** — re-run final SealBot **CI-above golong@matched AND ≥ ~0.30** → the working loop
  turned the encoding into a real lead over its frozen-loop ~0.18 → the encoding is the strength
  path; scale it, Gumbel becomes additive.
- **PARITY/FLAT** — final SealBot ≈ the frozen-loop ~0.18–0.20 (loop added little) → the encoding
  plateaus → route remaining strength upside to **Gumbel**; bank the encoding as robustness only.
- **REGRESS** — final SealBot **< 0.14** (below the frozen-loop baseline) → the working loop HURT
  (colony-attractor / over-spread on the legal-set policy) → stop scaling; diagnose loop dynamics.

**Robustness:** re-confirm off-window via standalone `exploit_probe.py … --arms exploit,control
--encoding v6_live2_ls` (KClusterMCTSBot dispatch) ≤ 0.06 on the re-run final — the PRIMARY
greenlight must survive the working loop (a stronger generator must not re-open the blind spot).

## What to watch live
- **best_model trajectory** advancing (the as-run's frozen-bootstrap is the failure signature).
- **clean** SealBot rising past 0.35 across rounds (the in-run eval now no-drop-routed).
- Colony / components / draw-rate abort gates (unchanged from as-run watch sheet).

## Do NOT
- Do NOT read strength off the single-window `ModelPlayer` decode or a raw in-run SealBot-WR.
- Do NOT claim encoding credit if strength rises without confirming ≥1 promotion landed (else
  it is loop-fix bias, not the encoding) — though here the loop fix IS the point (the as-run
  proved the encoding reaches parity with NO working loop).
- Do NOT re-open the banked PRIMARY off-window greenlight; this re-run is the STRENGTH read.
