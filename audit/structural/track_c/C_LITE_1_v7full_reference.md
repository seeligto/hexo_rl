# C-LITE-1 — v7full anchor V_spread (dual bank)

§S181-AUDIT Wave 1 / Track C-LITE-1. Pre-§175 era reference point.

## Hypothesis (pre-registered)

If `bootstrap_model_v7full.pt`'s V_spread (alt bank) ≥ +0.15 AND v7full's
documented 17.4% SealBot WR (§150, n=500) proves it was sustainable,
then **encoding regression v7full → v6w25 → v6 is the load-bearing
change** in the §150 → §175 → §S178 trajectory.

## Setup

| field | value |
|---|---|
| anchor | `checkpoints/bootstrap_model_v7full.pt` |
| anchor SHA-256 | `568d8a33…d61e8e98` |
| anchor encoding (metadata) | `v7full` (from ckpt `metadata.encoding_name`; viewer loader logged `?` — minor display gap, registry comparison authoritative) |
| anchor commit | `b947659d1b71ee209d66c16b4841aa567774deb5` (2026-05-11 training_date) |
| T3 bank fixture | `tests/fixtures/value_spread_bank.json` SHA `934204…3991` |
| alt bank fixture | `tests/fixtures/value_spread_bank_alt.json` SHA `a68b810f…20a20ff` |
| device | CPU |
| forward path | dual-bank canary (PR-C `compute_value_spread_dual`) |

**Encoding compatibility (verified).** `engine/src/encoding/registry.toml`
shows v6 and v7full are identical at the plane / spatial level:
`board_size=19`, `n_planes=8`, identical `plane_layout` (X_t0..t-3,
O_t0..t-3), identical `kept_plane_indices=[0,1,2,3,8,9,10,11]`,
`has_pass_slot=true`, `is_multi_window=false`. The only registry-level
difference is `notes` (selfplay-canonical vs Phase 1-3 anchor). Both
banks' (8, 19, 19) state arrays and T3 Board sequences feed v7full
cleanly.

## Result

| bank | mean(colony) | mean(extension) | **V_spread** | gate |
|---|---|---|---|---|
| T3  | `+0.0169` | `-0.2002` | **`+0.2171`** | PASS (≥ +0.20, borderline) |
| alt | `+0.0837` | `-0.3241` | **`+0.4078`** | PASS (≫ +0.07) |

`both_pass = True`. Wall: 1.1 s on CPU.

### v6 baseline (for direct comparison)

| bank | V_spread (v6 anchor) | V_spread (v7full) | Δ |
|---|---|---|---|
| T3  | `+0.6173` | `+0.2171` | **−0.40** (v7full WORSE on T3) |
| alt | `+0.2119` | `+0.4078` | **+0.20** (v7full BETTER on alt, nearly 2×) |

**The two banks invert.** T3 was specifically synthesized to highlight
colony-vs-extension contrast under the v6 anchor's value head. v7full —
trained on a different selfplay corpus — discriminates the synthetic T3
constructions less sharply but discriminates the real corpus-drawn alt
positions MORE sharply (nearly 2× v6).

This is **independent corroboration of L48**: T3 is partly bank-specific.
The alt bank — drawn from real `bot_corpus_s178_sealbot_vs_v6` mid-game
positions — is the corpus-grounded reference, and on it v7full
materially outperforms v6.

## Verdict (LITERAL, L13 guard)

**C-LITE-1-A.** v7full alt-bank V_spread = `+0.4078` ≥ +0.15.

**Encoding regression candidate CONFIRMED.** v7full has a stronger
value-head colony/extension discriminator (on the corpus-grounded alt
bank) than v6 by ~2×. Combined with §150's documented 17.4% SealBot WR
(n=500), the v7full anchor is the strongest pre-collapse reference
point we have.

**Recommended downstream (Task 5 / REAL_RUN_RECIPE).** Consider
`bootstrap_model_v7full.pt` as the real-run anchor candidate, gated on
C-LITE-2's verdict on whether the K-cluster-with-min-pool encoding
class (v6w25) protects in training or whether the encoding question is
v6 vs v7full at the single-window level.

## Caveats

1. **T3 borderline pass on v7full.** `+0.2171` is `+0.017` above the
   T3 SOFT-ABORT gate. Per L48 this is exactly the predicted
   confound: T3's +0.617 calibration was specific to the v6 anchor's
   value head; cross-anchor reads should be measured primarily on
   alt.
2. **No retraining was performed.** This is a single inspection-only
   forward (~1 sec CPU). The verdict is "v7full's value head, AS IT
   SITS, discriminates colony/extension better than v6 on real
   positions." It does NOT directly test whether v7full's training
   trajectory under the §S180b recipe would resist colony capture —
   that is C-LITE-2's province (and ultimately the real run's).
3. **Encoding metadata propagation gap.** The viewer's `load_model`
   returned `encoding_name='?'` while the checkpoint's
   `metadata.encoding_name` is `v7full`. Confirmed by direct
   `torch.load` inspection. Cosmetic only — does not affect the
   forward result.
4. **bootstrap_model_v6.pt baseline values are from prior audits.**
   T3 `+0.6173` from `audit/structural/05_fu1_value_spread_ladder.md`;
   alt `+0.2119` from `audit/structural/track_a/A3_h_bank_confound.json`.
   Both reproduced under PR-C unit tests within ±0.005 tolerance.

## Sidecar

`audit/structural/track_c/C_LITE_1_v7full_reference.json` — full
component breakdown, SHAs, thresholds, verdict object.

## Cross-references

- `audit/structural/05_fu1_value_spread_ladder.md` (FU-1 anchor +0.617)
- `audit/structural/track_a/A3_h_bank_confound.json` (alt anchor +0.212, L48 source)
- `docs/07_PHASE4_SPRINT_LOG.md` §S181-AUDIT — L48 / L49
- `engine/src/encoding/registry.toml` — v6 / v7full registry entries
