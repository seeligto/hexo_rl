# §S181 Track A — A5 H-PRETRAIN: pretrain corpus position-level z

**Wave:** §S181-AUDIT Track A. **Subtask:** A5. **Branch:**
`phase4.5/s181_audit_track_a`. **Date:** 2026-05-23.

> **VERDICT: INCONCLUSIVE — literal — H-PRETRAIN provides ~2× the bot
> corpus's asymmetry direction but still under-threshold.** Pretrain
> colony fraction 31.2% (above 5% consequential-count threshold).
> Asymmetry colony − extension = **+0.157** — twice the bot corpus's
> +0.078 (A1) but below the +0.20 threshold for PRETRAIN-COLONY-BIASED.
> Pretrain installs a modest colony-favouring direction in the value
> targets the model learns at PT time.

---

## 0. Inputs

| | |
|---|---|
| Corpus | `data/bootstrap_corpus_v6.npz` |
| SHA-256 | `6ea62afa892b03c7…0682477b8c42d63…d88b47` |
| Positions | 353,091 |
| Outcome unique | `{-1, +1}` (decisive human games; no draws) |
| Outcome mean | +0.0161 (slight first-player advantage overall) |

§S181 T1 reference: 91.3% extension via threat-fraction probe. This A5
uses a different metric (structural class, A1 classifier). Both views
are complementary — T1 is "fraction of positions with any threat
structure", A5 is "fraction with structural extension dominance + n>=8
stones".

---

## 1. Distribution

| class | n | frac |
|---|---:|---:|
| **colony** | 110,109 | **31.18%** |
| **extension** | 21,859 | **6.19%** |
| **neither** | 221,123 | 62.62% |
| **total** | 353,091 | 100% |

**Compared to bot corpus (A1):**
| | bot corpus | pretrain | Δ |
|---|---:|---:|---:|
| colony frac | 26.04% | 31.18% | +5.14 pp |
| extension frac | 12.90% | 6.19% | −6.71 pp |
| neither frac | 61.06% | 62.62% | +1.56 pp |

Pretrain corpus has **more colony positions and fewer extension
positions** than the bot corpus. SealBot games (bot corpus) produce
proportionally more open-line strategic structures; human games
(pretrain) trend toward tighter blob structures.

---

## 2. Per-class outcome-target (z) statistics

| class | n | mean z | median z | sd z |
|---|---:|---:|---:|---:|
| colony    | 110,109 | **+0.0129** | +1 | 0.9999 |
| extension | 21,859 | **−0.1437** | −1 | 0.9896 |
| neither   | 221,123 | +0.0334 | +1 | 0.9994 |

**z-bucket histogram:**

| class | z == −1 | z == +1 |
|---|---:|---:|
| colony | 54,342 | 55,767 |
| extension | 12,500 | 9,359 |
| neither | 106,864 | 114,259 |

By-class outcome sign:

| class | curr wins | opp wins | curr wins frac |
|---|---:|---:|---:|
| colony    | 55,767 | 54,342 | **50.65%** |
| extension | 9,359  | 12,500 | **42.81%** |
| neither   | 114,259| 106,864 | 51.67% |

Colony is near-balanced (50.65% current-player wins). Extension
strongly favors opponent (42.81% current-player wins — implies when a
position is structurally extension-shaped, the current player is
usually facing the opponent's threat, not threatening themselves).

---

## 3. Asymmetry score + verdict

| metric | value |
|---|---:|
| mean z (colony) | +0.0129 |
| mean z (extension) | −0.1437 |
| **asymmetry = mean_z(colony) − mean_z(extension)** | **+0.1566** |
| anchor V_spread (T3 bank) | +0.617 |
| asymmetry as fraction of anchor | 25.4% |

### Pre-registered verdict applied LITERALLY (L13 guard)

| condition | rule | this run | match |
|---|---|---|---|
| PRETRAIN-COLONY-BIASED | asymmetry > +0.20 AND colony frac ≥ 5% | +0.157 / 31.2% — fraction PASS, asymmetry below | **FALSE** |
| EXTENSION-BIASED | asymmetry < −0.20 | +0.157 | **FALSE** |
| NEUTRAL | asymmetry in [−0.10, +0.10] | +0.157 | **FALSE** |
| INCONCLUSIVE | otherwise | | **TRUE** |

**VERDICT: INCONCLUSIVE (literal).**

### Interpretation guidance for A6

- **H-PRETRAIN direction confirmed colony-favouring** (+0.157 asymmetry,
  same sign as anchor V_spread +0.617). The magnitude is twice the bot
  corpus's +0.078 (A1) but still well below the threshold.
- **Pretrain installs ~25% of the anchor V_spread direction.** The
  remaining ~75% must come from (a) downstream training-loop dynamics,
  (b) architecture's amplification of weak corpus signal, or (c) the
  T3-bank-amplification confounded into the +0.617 measurement (A3).
- **Combined corpus signal.** If both pretrain (asymmetry +0.157, frac
  31%) and bot corpus (asymmetry +0.078, frac 26%) contribute additively
  during sustained training, the total corpus-level imprint on
  V_spread is bounded above by approximately +0.10 to +0.15 (depending
  on the mixing schedule). Still below the anchor +0.617 and well below
  L47's per-step pull magnitude.
- **A6 dominant-source impact.** Per the routing table, "H-PRETRAIN
  confirmed dominant → filter pretrain corpus by class OR class-weighted
  pretrain LR" — that lever is appropriate ONLY for the +0.157 portion of
  the bias. Most of the anchor V_spread direction is NOT inherited from
  pretrain corpus value targets; loop-side levers (PSW, refresh hook)
  remain the surface to investigate for the larger share.

---

## 4. Outputs

- This file
- `audit/structural/track_a/A5_h_pretrain_position_z.json` (sidecar)
- `scripts/structural_diagnosis/track_a/a5_h_pretrain.py` (this script)
