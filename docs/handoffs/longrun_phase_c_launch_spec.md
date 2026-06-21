# §D-LONGRUN-READY Phase C — Long-Run Launch Spec

**Status: DESIGN-ONLY.** Do NOT launch until:
1. Phase B (m-gate) completes and verdict is in.
2. Operator gives GO.

**Resolved by Phase B:** which config runs (Gumbel@best-m OR PUCT-600).
**One variable in C:** the resolved search config. Everything else canonical.

---

## Pre-registered routing from Phase B

| Phase B verdict | Long-run config |
|---|---|
| **GUMBEL-EARNS-IT** — best m-arm per-step non-inferior or close to PUCT-600, AND cheaper-per-strength | Gumbel at best m (expected m=8) + n=100, optimised throughput |
| **GUMBEL-PARITY-ONLY** — best m-arm still clearly weaker per-step, edge ≈ 28% cost margin only | **PUCT-600** (stronger per-step, simpler, no host-tuning dependency) |
| **GUMBEL-WORSE** — m doesn't recover the gap at all | **PUCT-600** |

**PUCT-600 long run = safe default.** The 4.5× throughput multiplier is real but
mostly consumed offsetting Gumbel's shallower (depth 2.83 vs 3.52) search. If the
m-lever doesn't flip parity, PUCT-600 is the stronger-per-step, host-tuning-free choice
for a GPU-weeks run.

---

## Invariants (apply regardless of which search config wins)

- **Encoding:** `v6_live2_ls` (GREENLIT — 0/200 off-window, causal counterfactual confirmed).
- **Bootstrap:** new 8300-game pretrain from Phase A2 (NOT the old `bootstrap_model_v6_live2.pt`).
  Comparability break with prior runs accepted (fresh endeavor on new corpus).
- **Corpus schedule: PROVEN canonical (0.8 → 0.1, floor 0.1).** Do NOT change.
  `decay_steps` scaled to run length: for 200k canonical, the floor is reached at
  ~200k. For longer runs, scale proportionally (floor 0.1 ALWAYS retained — F2 coherence
  anchor). **Corpus-to-zero is NOT in this run** (separate coherence-gated ablation, later).
- **ONE variable:** the resolved search config (m + search type). No corpus co-tuning.

---

## Config template (fill after Phase B)

### If GUMBEL-EARNS-IT (example: m=8 wins)

```yaml
# configs/variants/longrun_v6_live2_ls_gumbel_m8.yaml
encoding: v6_live2_ls
in_channels: 4
lr: 2e-3
eta_min: 5e-4
aux_opp_reply_weight: 0.0
training_steps_per_game: 1.0

selfplay:
  completed_q_values: true
  n_workers: 32              # vast Gumbel optimum
  inference_batch_size: 64
  inference_max_wait_ms: 2.0
  leaf_batch_size: 8
  random_opening_plies: 0
  gumbel_mcts: true
  gumbel_m: 8                # Phase B winner — fill actual value
  gumbel_explore_moves: 10
  c_visit: 50.0
  c_scale: 1.0
  playout_cap:
    full_search_prob: 0.5
    n_sims_full: 100
    n_sims_quick: 100

mixing:
  pretrained_buffer_path: "data/bootstrap_corpus_v6_live2.npz"
  pretrain_weight: 0.8           # canonical start
  pretrain_decay_rate: 0.1       # → floor 0.1 over decay_steps
  decay_steps: 200000            # scale to run length (see §§ below)

eval_interval: 5000              # longer cadence for a sustained run
eval_pipeline:
  opponents:
    best_checkpoint: { enabled: true,  stride: 1, n_games: 400 }
    sealbot:         { enabled: true,  stride: 4, n_games: 50  }
    random:          { enabled: true,  stride: 1, n_games: 20  }
    bootstrap_anchor: { enabled: true, stride: 2, n_games: 100 }
    nnue:            { enabled: true,  stride: 2, n_games: 50  }
    offwindow_adversary: { enabled: false }
  gating:
    expected_anchor_sha256: "FILL_FROM_A2"    # sha256 of bootstrap_model_v6_live2_8300.pt
    promotion_winrate: 0.55
    require_ci_above_half: true

telemetry:
  emit_outcome_distribution: true
  emit_value_pred_at_ply_cap: true

monitors:
  hard_abort_grad_norm: 10.0
  hard_abort_draw_rate: 0.55
  hard_abort_draw_rate_consec: 3
  hard_abort_draw_rate_min_step: 0
```

### If PUCT-600 wins (or GUMBEL-PARITY/WORSE)

Same as above except:
```yaml
selfplay:
  ...
  n_workers: 18               # PUCT optimal (18 for 24-thread host; re-tune if host changes)
  inference_batch_size: 128
  inference_max_wait_ms: 16.0
  gumbel_mcts: false          # standard PUCT
  playout_cap:
    full_search_prob: 0.5
    n_sims_full: 600          # PUCT-600
    n_sims_quick: 100
```
No `leaf_batch_size` needed (PUCT default).

---

## Corpus schedule — canonical pins

```
pretrain_weight start : 0.8   (heavy corpus anchor at step 0)
decay_rate            : 0.1   (floor, not zero)
decay_steps           : scale to run length
```

**decay_steps scaling rule:** the floor (0.1) should be reached at ~70% of the
target run length, to give the model time to learn from pure self-play in the
final third. For target 200k: `decay_steps=200000`. For target 500k: `decay_steps=500000`.

**Do NOT set decay_steps low to "fix" colony** — that's the F2 coherence
trap (§D-GOLONG: lowering the floor collapsed coherence). The floor 0.1 is
load-bearing.

---

## Loop-turning gate (mandatory pre-launch check)

The training loop has NEVER turned end-to-end for v6_live2_ls. Catch loop failures
early, not at 200k steps.

**Gate:** within the first 15k steps (3-4 eval rounds), a promotion MUST fire AND
the self-play generator MUST sync to the promoted checkpoint. Verify:
1. A checkpoint hits `wr_best ≥ 0.55` (or whatever promotion bar is live).
2. The log shows `promoted: true` AND `generator synced`.
3. The self-play replays from after that point show the NEW policy (not the bootstrap).

**If the loop does NOT turn by step 15k → ABORT and diagnose before continuing.**
Known failure modes from prior runs: eval opponent import fail (hammerhead), anchor
SHA mismatch (wrong checkpoint as pinned anchor), promotion disabled by `require_ci_above_half`
at very low n (raise n_games temporarily to verify, then restore).

**Pre-flight (before launch):**
```bash
# Verify eval opponents import on the host
python -c "from hammerhead import Bot; print('OK')"
python -c "from hexo_rl.eval.k_cluster_mcts_bot import KClusterMCTSBot; print('OK')"
# Also: 2-game smoke with each eval opponent at eval_interval=100 (then restore)
```

---

## Abort / monitor gates

| Signal | Gate | Action |
|---|---|---|
| `draw_rate` | ≥ 0.55 for 3 consecutive evals | Hard abort (already in monitors) |
| `grad_norm` | ≥ 10.0 | Hard abort (already in monitors) |
| `forced_win_conversion` | DECLINING trend over 3 evals | Flag; if reaches <0.5 investigate coherence |
| Colony fraction | >15% of board area | Investigate (not abort by itself) |
| Fragmentation | components >20 (and rising) | Investigate (golong failure signature) |
| Longest line stall | longest_line_fraction flat at <0.4 by step 50k | Investigate |
| Loop NOT turning | No promotion by step 15k | **ABORT, diagnose** |
| Promotion cadence | <1 promotion per 50k after step 50k | Investigate stall |
| value_accuracy_masked | Decline >10pp over 10k steps | Flag |

**SealBot read:** every 50k steps from step 50k onward. Judge as trajectory/signature,
not a point-in-time bar (the 0.38 ceiling at golong step 50k is context-dependent;
SealBot-beating lives in the 200k-1M regime). Do NOT abort on a single bad SealBot step.

**Coherence monitor — ACTIVE not passive:** the golong failure (components 26→42,
conversion 0.89→0.66) progressed over ~40k steps. If the fragmentation signature
appears, it's already in the collapse arc — act early (bank, diagnose, consider abort)
rather than waiting to confirm.

---

## Banking protocol

**Bank EVERY promotion checkpoint.** Also bank on schedule:
- Step 25k, 50k, 75k, 100k, 150k, 200k (and beyond at 50k cadence).
- Bank the "final" checkpoint regardless of step count (before kill).

**Transport:** rsync from vast to laptop (vast fetch dead; push works). Example:
```bash
# From laptop — pull latest promotion checkpoint:
rsync -avz -e "ssh -p 13053 -i ~/.ssh/vast_hexo" \
  "root@ssh6.vast.ai:/workspace/hexo_rl/checkpoints/longrun/" \
  "~/Work/Hexo/hexo_rl/checkpoints/longrun_bank/"
```

**Safety:** bank from step 1. Arm-C had no banking for the first 15k steps; if the
vast instance had died, the run would have been lost entirely.

---

## Target steps and cost estimate

| Target | Steps | Est. cost (Gumbel w32/n100) | Est. cost (PUCT-600 w18) |
|---|---|---|---|
| Loop-turn gate | 15k | ~2h | ~12h |
| SealBot-first-read | 50k | ~7h | ~40h |
| First milestone | 100k | ~14h | ~80h |
| Full sustained | 200k | ~28h | ~163h |
| Long run | 500k | ~69h | ~400h |

(Gumbel ~7.2k steps/hr at w32/n100. PUCT ~1.23k steps/hr at w18/n600.)

---

## What comes BEFORE this launches

1. Phase A1: vast consolidation → /workspace clean ✓
2. Phase A2: bootstrap pretrain → `bootstrap_model_v6_live2_8300.pt` + SHA ✓
3. Phase B: m-gate → verdict → routing decision ✓
4. Operator GO

**Then and only then:** create the actual config (fill from template above), launch.

---

## Separate, NOT in this run: corpus-to-zero ablation

If the long run shows value divergence (selfplay BCE floors while corpus BCE continues
dropping) OR model plateaus at 0.1 corpus floor, a late-phase 0.1→0 decay ablation
may be warranted. Pre-register the gate criteria before running it; never co-tune
corpus schedule with the search config; never drop the anchor while coherence is unproven.
