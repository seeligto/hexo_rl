# §S181 Track A — A3 H-BANK: alt bank confound cross-validation

**Wave:** §S181-AUDIT Track A. **Subtask:** A3. **Branch:**
`phase4.5/s181_audit_track_a`. **Date:** 2026-05-23.

> **VERDICT: CONFOUND — literal — T3 bank is partially confounded
> w.r.t. real corpus positions.** Pearson correlation between V_spread
> on T3 bank and V_spread on a real-corpus-derived alt bank is
> **r = 0.27** across the FU-1.5 ladder (11 paired checkpoint
> measurements, step 0 + 2k-20k). Below the 0.7 CONFOUND threshold.
> The V_spread metric — as currently scoped on T3 bank — is partly a
> T3-bank-specific artifact, not a clean diagnostic for value-head
> colony-extension discrimination on game-play positions.

---

## 0. Inputs

| | |
|---|---|
| Alt bank source corpus | `data/bot_corpus_s178_sealbot_vs_v6.npz` |
| Corpus SHA-256 | `2209b76fd82e943d…f885b8691900bd4ff` |
| Alt bank fixture | `tests/fixtures/value_spread_bank_alt.json` |
| Alt bank SHA-256 | written into fixture `meta.sha256` (reproducibility pin) |
| Anchor model | `checkpoints/bootstrap_model_v6.pt` |
| Ladder | `archive/s181_fu1_5/ckpts/checkpoint_{00002000..00020000}.pt` (10 ckpts, 2k cadence) |
| T3 V_spread reference | `audit/structural/06_fu1_5_finer_ladder.md` §3 (hard-coded into A3 script for reproducibility) |
| Sampler seed | `20260523` |
| Mid-game stone range | `[12, 36]` |
| n per class | 20 colony + 20 extension |

**Alt bank construction.** Filter bot corpus to mid-game positions
(in-window stone count in [12, 36]); classify with the A1 classifier;
randomly sample (seeded) 20 colony + 20 extension. The fixture stores
state tensors directly (rather than move sequences as the T3 bank
does) — corpus positions don't carry move history.

---

## 1. V_spread ladder — side by side

V_spread(T3) reproduced from FU-1.5 audit doc 06 §3 (no re-measurement).
V_spread(alt) measured fresh in this audit via direct model forward on
the 40 alt-bank state tensors per checkpoint.

| step | T3 V_spread | alt V_spread | Δ |
|---:|---:|---:|---:|
|     0 | **+0.6173** | **+0.2119** | −0.4054 |
|  2 000 | +0.1752 | +0.1998 | +0.0246 |
|  4 000 | −0.1179 | +0.2581 | +0.3760 |
|  6 000 | +0.0553 | +0.0158 | −0.0395 |
|  8 000 | +0.2172 | +0.1950 | −0.0222 |
| 10 000 | +0.3901 | +0.1703 | −0.2198 |
| 12 000 | +0.1669 | +0.0583 | −0.1086 |
| 14 000 | +0.5226 | +0.2462 | −0.2764 |
| 16 000 | +0.2209 | +0.0651 | −0.1558 |
| 18 000 | +0.2083 | +0.2404 | +0.0321 |
| 20 000 | +0.1075 | +0.0930 | −0.0145 |

**Observations.**
- The anchor V_spread on alt bank is +0.212 — POSITIVE
  (colony-favouring) but only ~34% of the +0.617 anchor on T3 bank.
- The alt ladder is much **flatter**: alt range [+0.016, +0.258] vs
  T3 range [−0.118, +0.617]. Magnitudes are ~3× smaller on alt bank.
- The "step-0 → step-4k crash" signature on T3 (+0.617 → −0.118,
  Δ = −0.735) is NOT reproduced on alt bank (+0.212 → +0.258,
  Δ = +0.046).
- Both ladders stay POSITIVE for most steps on alt bank (only step 6k
  dips to +0.016); the model's colony-favouring direction on real
  game positions is consistent throughout training, just with much
  smaller magnitudes.

---

## 2. Correlation

**Pearson r(T3 V_spread, alt V_spread) over n=11 paired checkpoints = 0.27.**

This is a weak positive correlation. The two trajectories share some
directional sympathy (alt's mild +0.21 → +0.26 → +0.02 → +0.20 oscillation
loosely tracks T3's stronger swings) but the magnitudes and step-to-step
behaviour diverge substantially.

---

## 3. Pre-registered verdict applied LITERALLY (L13 guard)

| condition | rule | this run | match |
|---|---|---|---|
| ROBUST | r ≥ 0.9 | r = 0.274 | **FALSE** |
| CONFOUND | r < 0.7 | r = 0.274 | **TRUE** |
| INCONCLUSIVE | otherwise | | FALSE |

**VERDICT: CONFOUND (literal).**

### Interpretation guidance for A6

T3 bank's V_spread amplitude on the anchor (+0.617) is ~3× larger than
the alt bank's (+0.212). The training-loop "pull" measured on T3
(≥1.0 per ~1000 steps, L47) is similarly amplified by T3 bank's
particular synthetic structures (spiral colony vs line-with-far-filler
extension). On real corpus positions, the same training trajectory
shows much smaller V_spread movement.

**This changes the interpretation of L47, not its mechanism.**
- The training loop IS pulling on the value head (alt ladder oscillates
  in [+0.02, +0.26] range — ~0.24 max swing) but the magnitude on
  real positions is ~3× smaller than the T3-measured ≥1.0.
- L47's "training-loop dominates architecture" conclusion stands at the
  T3-bank metric level: A2's anchor inversion (−0.508 PT-3 baseline)
  was dragged through zero to +0.13 within 1000 steps. That trajectory
  is real on T3. But the equivalent "drag" on alt bank might be much
  smaller in magnitude.

**Recommendation.** The V_spread canary should be **augmented with an
alt-bank measurement** (cheap — 1 forward pass per ckpt × 40 positions
× CPU = ~1 sec). The T3 bank stays as the historical reference;
alt bank gives a corpus-grounded sanity check. The pinned
`tests/fixtures/value_spread_bank_alt.json` is the artifact.

**A6 routing impact.** This is a *metric* finding, not a source
finding. It does not directly identify H-BOT vs H-PRETRAIN vs
H-CE-STRENGTH as the dominant loop-level mechanism. But it does say:
the **magnitude** of the loop's effect on the value head's
discrimination is smaller than T3 measurements suggest. Any next-wave
lever's success criterion should be measured on BOTH banks; T3 alone
is not sufficient.

---

## 4. Outputs

- This file
- `audit/structural/track_a/A3_h_bank_confound.json` (sidecar)
- `tests/fixtures/value_spread_bank_alt.json` (alt bank fixture,
  SHA-pinned in `meta.sha256`)
- `scripts/structural_diagnosis/track_a/a3_h_bank.py` (this script)
