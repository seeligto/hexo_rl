# Q27 — Perspective-flip fix smoke test

**Date.** 2026-04-18 → 2026-04-19 (crossed midnight on desktop arm).

**Question.** Does the W1 perspective-flip fix (commit `e9ebbb9` —
Q-negation at intermediate ply in `get_improved_policy`, Gumbel score,
`get_top_visits`) measurably affect self-play entropy and/or the
threat-probe C1–C3 criteria at 5K steps from `bootstrap_model.pt`?

**Short answer.** **No — not at 5K.** Both arms fail the threat probe
on C2/C3 with identical 20%/20% scores. The fix is still correct
(training targets were inverted at ~50% of move steps pre-fix; that is
a separate correctness argument and not what this smoke tested), but
it does **not** independently close the attention-hijacking symptom
within this budget.

---

## Setup

| Arm | Commit | Variant | Machine | Workers |
|---|---|---|---|---|
| pre_fix | `723615e` (parent of `e9ebbb9`) | `gumbel_targets` | laptop (8845HS / RTX 4060) | 14 |
| post_fix | `a7efa78` (HEAD, fix in tree) | `gumbel_targets_desktop` | desktop (3700x / RTX 3070) | 10 |

Both arms:

- Started from `checkpoints/bootstrap_model.pt` (18-plane, post-§99 GroupNorm).
- Ran 5000 training steps via `scripts/train.py --iterations 5000
  --no-dashboard`.
- Ran `make probe.latest` against `checkpoints/checkpoint_00005000.pt`
  immediately after training stopped.
- Logged `policy_entropy_selfplay` per step via structlog JSON.

The desktop `gumbel_targets_desktop` variant derives from
`gumbel_targets` (same selfplay semantics; `n_workers 14 → 10` and
`inference_max_wait_ms 4 → 5` per §79 D3 sweep). Per the variant
header, it "shares selfplay semantics" — targets, search, playout cap,
Gumbel disable, completed-Q are identical. See Caveats for the
implications of the split.

Desktop run required one restart (run1 partial → step 450 after
accidental window close; run2 resumed from `bootstrap_model.pt`, not
from the partial checkpoint — no contamination).

---

## Results

### Threat probe at step 5000

| Criterion | Target | pre_fix (723615e) | post_fix (HEAD) | Δ |
|---|---|---|---|---|
| C1 contrast_mean | ≥ +0.380 | **+3.665** PASS | **+3.891** PASS | +0.226 |
| C2 ext_in_top5_pct | ≥ 25% | **20%** FAIL | **20%** FAIL | 0 |
| C3 ext_in_top10_pct | ≥ 40% | **20%** FAIL | **20%** FAIL | 0 |
| C4 \|Δ ext_logit_mean\| | < 5.0 (warn) | 0.078 ok | 0.505 ok | +0.427 |
| **Exit code** | 0 = PASS | **1 (FAIL)** | **1 (FAIL)** | — |

Both arms: threat head learned the extension-vs-control contrast (C1
easily, ~10× threshold in both cases), but the **policy head does not
route top-K to the extension cell** — C2 and C3 are pinned at 20% in
both arms, consistent with "attention hijacking" (threat knowledge
present in the scalar head, policy ignoring it).

### Entropy at step 5000

| Arm | H(policy) @ 5K | Settled band (500 → 5000) |
|---|---|---|
| pre_fix | 5.3733 | 5.17 – 5.57 |
| post_fix | 5.6284 | 5.51 – 5.72 |
| **Δ** | **+0.2551 (post higher)** | post band sits ~0.2 higher |

Uniform entropy over the ~361-cell action space is ≈ 5.88. Post-fix
policy sits closer to uniform than pre-fix by ~0.25 nats of H, a small
but consistent offset for the length of the trajectory (see snapshots
below).

### Entropy trajectory snapshots

| Step | pre_fix | post_fix |
|---|---|---|
| 10 | 1.90 | 3.46 |
| 200 | 5.52 | — |
| 500 | 5.57 | 5.51 |
| 1000 | 5.42 | 5.72 |
| 1500 | 5.46 | 5.65 |
| 2000 | 5.17 | 5.70 |
| 2500 | 5.39 | 5.60 |
| 3000 | 5.32 | 5.59 |
| 3500 | 5.27 | 5.59 |
| 4000 | 5.39 | 5.61 |
| 4500 | 5.38 | 5.62 |
| 5000 | 5.37 | 5.63 |

### Other post-fix metrics (desktop only, pre-fix not captured)

- Games completed: 1253.
- Replay buffer: 250000 / 250000 (saturated).
- Policy loss: 1.9619 → 1.6544 (Δ −0.3075).
- Run wall-clock (run2): ~5h (20:08 → 01:18 local).

---

## Interpretation

**W1 is necessary, not sufficient (at 5K).** Pre-fix, the intermediate
ply Q-sign inversion corrupted ~50% of training targets at three sites
(`get_improved_policy`, Gumbel score, `get_top_visits`). The fix is
required for correctness — that argument stands on the code change
itself. What this smoke tested is whether the fix closes the
threat-probe failure as a side-effect at 5K. It does not: C2/C3 are
identically FAIL at 20% in both arms.

**Entropy delta is in the wrong direction for a "fix helps" story.**
If the fix had let the policy start committing to extension cells, we
would expect post-fix entropy to be **lower** than pre-fix. We observe
the opposite (+0.25 nats, closer to uniform). Two plausible readings:

1. Post-fix policy is healthier in that it is still exploring; pre-fix
   policy had committed earlier but to wrong targets. Consistent with
   the hypothesis that W1 unblocks *further* learning beyond 5K.
2. Post-fix arm is simply a noisier, slower trajectory because of the
   variant/machine split, and the +0.25 is not signal.

Cannot discriminate with n=1 per arm. Both readings agree that the
**attention-hijacking symptom persists** at 5K regardless.

**C1 in both arms is ~10× threshold** — the threat head carries
extension information just fine; the failure is downstream in how the
policy trunk uses (or fails to use) that signal.

---

## Caveats

Large enough to call out individually:

1. **Two machines, two variants.** The split was deliberate (saves
   wall-clock), but it means pre- vs post-fix differ in `n_workers`
   (14 vs 10) and `inference_max_wait_ms` (4 vs 5). Selfplay target
   construction is identical per the variant header, but game
   generation rate and GIL-stall pattern are not. The +0.25-nat
   entropy gap could partly be worker-count noise.
2. **n = 1 per arm.** No confidence intervals. A fair A/B repeat on
   the same machine with the same variant would cost ~10h wall-clock;
   the question "does W1 close the symptom at 5K" is answered
   decisively enough by the identical 20%/20% C2/C3 that the n=1
   limitation does not flip the conclusion, but the entropy delta is
   within plausible run-to-run noise.
3. **Desktop arm had a restart.** Run1 reached step 450 before the
   window closed; run2 started from `bootstrap_model.pt` again (not
   from a mid-run checkpoint). No state contamination, but the
   reported trajectory is run2 only.
4. **Cross-machine CPU / GPU / RNG all differ.** Seed is the same
   (config-driven) but RNG-user ordering differs with worker count.
5. **Pre-fix arm policy_loss trajectory not captured.** Only post-fix
   summary cites it; the pre-fix summary was constructed after the
   log rotation archived older step-level entries.

None of these flip the core finding (identical C2/C3). They do mean
the entropy finding is directional, not quantitative.

---

## Verdict

**W1 (perspective fix) does not close the attention-hijacking symptom
at 5K.** Both arms fail the threat probe on C2/C3 with identical
20%/20% scores. The fix remains necessary on correctness grounds
independent of this result.

**Q27 (attention hijacking) remains open.** The next probes should
look past the search-side perspective fix:

- **Value aggregation (Open Question 2).** If min-pool over cluster
  windows is silently discarding the extension-cell evidence, the
  threat scalar would still learn contrast (seen in C1) while the
  policy gradient never routes attention to it. Worth an ablation
  with mean-pool and a logged comparison at 5K.
- **Threat head → policy gradient coupling.** The threat head
  (BCE, weight 0.1) and policy head share a trunk. Check whether the
  trunk gradient contribution from the threat BCE is being drowned
  out by the policy CE loss. An `aux_threat_weight` sweep 0.1 → 0.5
  at fixed 5K budget would test it.
- **ZOI post-search filter (§77).** ZOI restricts top-K to hex
  distance ≤ 5 of last 16 moves. If extension cells live outside that
  window in the threat-probe positions, C2/C3 can cap at the non-ZOI
  baseline by construction — worth logging whether the probe
  positions' extension cells are reachable under the active ZOI mask.

A second smoke with n=3 per arm on a single machine would tighten the
entropy reading — defer until one of the three probes above has a
concrete hypothesis attached.

---

## Files

- `pre_fix/summary.txt` · `pre_fix/probe_output.txt` ·
  `pre_fix/entropy_trajectory.tsv` · `pre_fix/train_stdout.txt`
- `post_fix/summary.txt` (only summary preserved post-restart; full
  logs rotated)
- `reports/probes/latest_20260418_223903.md` (pre_fix probe detail)
- `reports/probes/latest_20260419_011839.md` (post_fix probe detail)
