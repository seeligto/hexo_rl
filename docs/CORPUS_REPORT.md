# Corpus Report — Full Four-Source Analysis

Generated: 2026-04-03

---

## 1. Source Summary

| Source | Games | Positions | Median Length | P10 / P90 |
|--------|------:|----------:|--------------:|----------:|
| Human (raw_human) | 1,765 | 108,615 | 49 | 27 / 109 |
| Bot fast (sealbot_fast, 0.1s) | 5,598 | 209,660 | 35 | 27 / 51 |
| Bot strong (sealbot_strong, 0.5s) | 2,804 | 103,752 | 35 | 27 / 49 |
| Injected (human-seed bot-continuation) | 5,567 | 481,701 | 73 | 37 / 157 |
| **Total** | **15,734** | **903,728** | **43** | **29 / 109** |

Injected games are significantly longer than pure bot games (median 73 vs 35
plies). This is expected: human openings branch into complex midgames that
SealBot takes longer to resolve. The result is valuable late-game training
signal that pure bot self-play underrepresents.

---

## 2. Winner Balance (P1 Win Rate)

| Source | P1 Win Rate | Flag |
|--------|------------:|------|
| Human | 51.6% | OK |
| Bot fast | 47.9% | OK |
| Bot strong | 48.9% | OK |
| Injected | 49.5% | OK |
| **Combined** | **49.1%** | **OK** |

No source exceeds the 60% concern threshold. The corpus is well-balanced.

### By Elo Band (Human Games Only)

| Elo Band | Games | P1 Win Rate |
|----------|------:|------------:|
| 800-1000 | 511 | 46.6% |
| 1000-1200 | 1,113 | 54.2% |
| 1200-1400 | 141 | 48.9% |

No Elo band exceeds 60%.

---

## 3. Game Length Distribution

| Source | Min | Median | Mean | P10 | P90 | Max |
|--------|----:|-------:|-----:|----:|----:|----:|
| Human | 20 | 49 | 61.5 | 27 | 109 | 603 |
| Bot fast | 17 | 35 | 37.5 | 27 | 51 | 93 |
| Bot strong | 17 | 35 | 37.0 | 27 | 49 | 91 |
| Injected | 17 | 73 | 86.5 | 37 | 157 | 451 |

Injected games provide the longest games in the corpus, filling the late-game
coverage gap left by bot self-play (54.9% of injected positions are at ply >= 40,
vs only 7.7% for bot_fast and 6.7% for bot_strong).

---

## 4. Quality Flags

| Check | Result |
|-------|--------|
| Games < 10 plies | 0 across all sources (0.0%) |
| Games > 200 plies | 34 human + 215 injected = 249 total (1.6%) |
| Winner field missing/None | 0 across all sources (0.0%) |

No early-termination bug detected. The 249 long games are legitimate (complex
multi-colony endgames).

---

## 5. Opening Diversity

| Source | Unique @ Move 3 | Unique @ Move 5 | Unique @ Move 10 | Unique @ Move 20 | Dupe Rate (first 10) |
|--------|:---:|:---:|:---:|:---:|---:|
| Human | 301 | 1,122 | 1,745 | 1,765 | 0.8% |
| Bot fast | 214 | 797 | 1,215 | 4,034 | 97.6% |
| Bot strong | 207 | 711 | 1,047 | 2,221 | 89.2% |
| Injected | 241 | 915 | 1,422 | 4,506 | 99.4% |
| Combined | 301 | 1,122 | 1,745 | 10,501 | 98.0% |

High dupe rates in bot and injected games reflect that all games start from
(0,0). The human games provide the opening diversity that bot games lack. By
move 20 there are 10,501 unique positions across the combined corpus, showing
good midgame branching despite uniform openings.

**Note:** First-move entropy is 0.0 nats across all sources because all games
start at (0,0) by convention. This is not a bug.

---

## 6. Move Entropy

| Source | Mean Entropy (nats) | Std |
|--------|----:|----:|
| Human | 4.90 | 0.32 |
| Bot fast | 4.63 | 0.18 |
| Bot strong | 4.61 | 0.15 |
| Injected | 5.17 | 0.34 |
| Combined | 4.93 | 0.38 |

Injected games show the highest entropy (5.17 nats), reflecting more varied
play as SealBot navigates unfamiliar human-seeded positions.

---

## 7. Cluster Count Distribution

| Source | Median K | Mean K | Max K | Frac K > 2 |
|--------|:---:|:---:|:---:|---:|
| Human | 1 | — | 8 | 18.8% |
| Bot fast | 1 | — | 4 | 1.4% |
| Bot strong | 1 | — | 4 | 2.4% |
| Injected | 1 | — | 9 | 25.2% |
| Combined | 1 | — | 13 | 19.6% |

Injected games have the highest multi-cluster rate (25.2%), providing valuable
training signal for the attention-anchored windowing system.

---

## 8. Ply Coverage

| Source | Positions at ply >= 40 | Flag |
|--------|---:|------|
| Human | 40.5% | OK |
| Bot fast | 7.7% | UNDERREPRESENTED |
| Bot strong | 6.7% | UNDERREPRESENTED |
| Injected | 54.9% | OK |
| **Combined** | **36.7%** | **OK** |

Bot-only sources underrepresent late-game positions, but the injected games
fully compensate. Combined late-game coverage is 36.7%, well above the 10%
threshold.

---

## 9. Quality Scores

| Metric | Value |
|--------|------:|
| Mean quality score | 0.784 |
| Median quality score | 0.775 |
| Mean (Human) | 0.813 |
| Mean (Bot fast) | 0.727 |
| Mean (Bot strong) | 0.784 |
| Mean (Injected) | 0.829 |
| Frac < 0.3 (exclude candidates) | 0.0% |
| Frac > 0.7 (high-quality anchors) | 88.0% |

88% of games are high-quality anchors. No games fall below the 0.3 exclusion
threshold.

---

## 10. NPZ Export

| Field | Value |
|-------|-------|
| File | data/bootstrap_corpus.npz |
| Total positions | 901,625 |
| States shape | (901625, 18, 19, 19) float16 |
| Policies shape | (901625, 362) float32 |
| Outcomes shape | (901625,) float32 |
| Weights range | 0.400 — 0.966 |
| File size | 113.9 MB |

---

## 11. Go / No-Go Verdict

| Criterion | Threshold | Actual | Pass? |
|-----------|-----------|--------|:-----:|
| P1 win rate (bot/injected) | < 65% | 49.5% max | PASS |
| Games < 10 plies | < 5% | 0.0% | PASS |
| NPZ position count | >= 200,000 | 901,625 | PASS |
| Winner field missing/None | < 1% | 0.0% | PASS |
| Human game count | >= 1,662 (previous) | 1,765 | PASS |

### VERDICT: GO

All five criteria pass. The corpus is ready for the sustained training run.

---

## 12. Known Limitations

1. **Bot opening diversity** — bot and injected games all start from (0,0),
   producing high first-10-move dupe rates. The 12-fold hex augmentation
   partially mitigates this. Future work: seed bot games from varied human
   openings (already partially addressed by the injected source).

2. **Late-game underrepresentation in bot-only sources** — bot_fast and
   bot_strong have <10% of positions at ply >= 40. The injected source
   compensates at the combined level (36.7%), but pure bot games should not
   be used alone for pretraining.

3. **Single-window dominance** — median K=1 across all sources. Multi-colony
   positions are rare (<25%). This is a property of the game itself at current
   skill levels, not a corpus bug.
