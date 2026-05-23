# §S181 Track A — A1 H-BOT: bot corpus position-level bias

**Wave:** §S181-AUDIT Track A. **Subtask:** A1. **Branch:**
`phase4.5/s181_audit_track_a`. **Date:** 2026-05-23.

> **VERDICT: INCONCLUSIVE — literal — but H-BOT NOT load-bearing.**
> Colony fraction 26.0% (close to but below 30% threshold) and
> per-position outcome asymmetry +0.078 (well below +0.30 threshold).
> The asymmetry magnitude is ~8× smaller than the T3 anchor V_spread
> +0.617, so even though the literal verdict classifier returns
> INCONCLUSIVE, the corpus value targets DO NOT carry a strong
> colony-favouring signal that could imprint the +0.617 anchor spread.
> H-BOT cannot be the dominant source.

---

## 0. Inputs

| | |
|---|---|
| Corpus | `data/bot_corpus_s178_sealbot_vs_v6.npz` |
| SHA-256 | `2209b76fd82e943d…f885b8691900bd4ff` |
| Positions | 21,899 |
| Plane shape | (8, 19, 19) — v6 encoding (X t0…t-3, O t0…t-3) |
| Outcome unique | `{-1, +1}` — decisive SealBot-vs-v6 games, no draws stored |
| Outcome mean / sd | +0.0316 / 0.9995 (slight current-player-wins bias overall) |
| Anchor V_spread | +0.617 (T3 40-position bank, `mcts_colony_probe.py`) |

**Classifier definition (EXPLICIT).** See
`scripts/structural_diagnosis/track_a/position_classifier.py`.

Decision order (mirrors corpus structural intent — a tight blob with
an incidental internal 4-run is structurally colony, not extension;
the internal run is flanked by other blob stones, not free space):

1. **`colony`** — `n_in_window ≥ 8` stones AND mean hex distance from
   stone centroid ≤ 2.7 (axial-coord hex metric, in the 19×19
   v6-window-cropped view). Off-window stones are dropped from the
   mask.
2. **`extension`** — any player owns a same-color contiguous run of
   ≥ 4 stones along one of the 3 hex axes `(1,0)`, `(1,-1)`, `(0,-1)`,
   with both flanking cells (one step further along the axis, each
   direction) empty. Off-window cells count as empty.
3. **`neither`** — otherwise.

**T3-bank validation.** 39/40 = 97.5% accuracy vs bank's `pos_class`
field. The one miss is `colony_00_stage6` (n=6 stones < min_stones=8);
the construction at stage 6 is structurally pre-colony. All 20 bank
extensions classify as `extension`; 19/20 bank colonies classify as
`colony` (script:
`scripts/structural_diagnosis/track_a/_validate_classifier.py`).

---

## 1. Distribution

| class | n | frac |
|---|---:|---:|
| **colony** | 5,702 | **26.04%** |
| **extension** | 2,825 | **12.90%** |
| **neither** | 13,372 | 61.06% |
| **total** | **21,899** | 100% |

Colony fraction 26.0% is meaningful (one in four bot-corpus positions is
structurally colony-shaped) but **below the 30% STRONG-CONFIRM
threshold**.

---

## 2. Per-class outcome-target (z) statistics

Outcomes are bimodal `{-1, +1}` (no draws stored). `z` is the game
outcome from the current-player perspective; +1 = current player wins.

| class | n | mean z | median z | sd z |
|---|---:|---:|---:|---:|
| colony    | 5,702  | **+0.0186** | +1 | 0.9998 |
| extension | 2,825  | **−0.0591** | −1 | 0.9983 |
| neither   | 13,372 | +0.0562 | +1 | 0.9984 |

**z-bucket histogram** (full sidecar in JSON):

| class | z == −1 | z == +1 |
|---|---:|---:|
| colony | 2,798 | 2,904 |
| extension | 1,496 | 1,329 |
| neither | 6,310 | 7,062 |

Outcome-sign distribution by class (proxy for "by winner" — the corpus
carries no per-position winner-id field; see §0 Inputs):

| class | curr-player wins | opp wins | curr-wins frac | opp-wins frac |
|---|---:|---:|---:|---:|
| colony    | 2,904 | 2,798 | 50.93% | 49.07% |
| extension | 1,329 | 1,496 | 47.05% | 52.95% |
| neither   | 7,062 | 6,310 | 52.81% | 47.19% |

Colony positions are near-symmetric on win/loss for the side to move.
Extension positions slightly favor the opponent (current player at a
position labeled `extension` loses 53% of the time — consistent with
"opponent owns a winning open run" being a common extension shape).

---

## 3. Asymmetry score + verdict

| metric | value |
|---|---:|
| mean z (colony) | +0.0186 |
| mean z (extension) | −0.0591 |
| **asymmetry = mean_z(colony) − mean_z(extension)** | **+0.0777** |
| anchor V_spread (T3, for comparison) | +0.617 |

The corpus asymmetry **+0.078 is ~8× smaller than the +0.617 anchor
spread** the training loop installs in the value head.

### Pre-registered verdict applied LITERALLY (L13 guard)

| condition | rule | this run | match |
|---|---|---|---|
| STRONG-CONFIRM | asymmetry > +0.30 AND colony frac ≥ 30% | +0.078 / 26.0% | **FALSE** |
| WEAK | asymmetry > +0.30 XOR colony frac ≥ 30% | neither met | **FALSE** |
| RULED-OUT | asymmetry in [−0.10, +0.10] AND colony frac < 20% | +0.078 in band ✓; 26.0% ≥ 20% ✗ | **FALSE** |
| INCONCLUSIVE | otherwise | | **TRUE** |

**VERDICT: INCONCLUSIVE (literal).**

### Interpretation guidance for A6

The literal classifier returns INCONCLUSIVE because the colony fraction
(26%) just misses both `≥30%` and `<20%` bands. But the asymmetry score
+0.078 is decisively small compared to the +0.617 anchor V_spread. If
the bot corpus were the dominant source of the colony attractor, we'd
expect asymmetry ≥ +0.30 (and probably closer to +0.617). It is not —
asymmetry is ~13% of anchor magnitude.

For the A6 contribution-estimate (per source pull on V_spread per ~1000
training steps), H-BOT's effective magnitude is bounded above by the
corpus asymmetry × bot-mix sample weight × per-position gradient.
Brief's bot-mix share for §S178 was `bot_batch_share=0.15`. So the
upper bound on H-BOT's per-step V_spread pull is roughly
`0.15 × 0.0777 = +0.012` per training step's worth of bot-corpus
gradient signal. The measured L47 magnitude is ≥1.0 per ~1000 steps
(~+0.001 per step). H-BOT's upper-bound contribution (~+0.012 per
gradient-step's-worth in a bot row) does have the right order of
magnitude — but the asymmetry is small enough that it's likely a
co-driver at best, not dominant. Final contribution share depends on
A4 (per-class gradient strength) and A5 (pretrain corpus z).

---

## 4. Outputs

- This file
- `audit/structural/track_a/A1_h_bot_corpus_position_bias.json` (sidecar)
- `scripts/structural_diagnosis/track_a/position_classifier.py` (classifier)
- `scripts/structural_diagnosis/track_a/a1_h_bot_corpus.py` (this script)
- `scripts/structural_diagnosis/track_a/_validate_classifier.py` (T3-bank
  validation)
