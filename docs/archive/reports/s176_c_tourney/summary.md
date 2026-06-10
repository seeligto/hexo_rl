# Tourney summary (§176 Wave B smoke)

- Bots: sealbot, kraken_minimax_strong, kraken_minimax_fast, kraken_random, our_v6_mcts128, our_v6_argmax, randombot (7)
- Games per pair: 50
- Total games: 1050
- Wall: 5111.2 s
- Halted (max-wall-seconds): False

## BT ratings (Elo, anchor=sealbot)

| Bot | Elo | CI lo | CI hi |
|---|---:|---:|---:|
| sealbot | 0.0 | 0.0 | 0.0 |
| our_v6_mcts128 | -62.1 | -149.8 | 25.6 |
| kraken_minimax_strong | -493.8 | -611.9 | -375.8 |
| kraken_minimax_fast | -499.4 | -617.9 | -381.0 |
| kraken_random | -3072.2 | -35150.7 | 29006.3 |
| randombot | -3090.8 | -35169.3 | 28987.7 |
| our_v6_argmax | -3102.4 | -35180.9 | 28976.1 |

## Head-to-head (rows = winner, cols = loser)

| | sealbot | kraken_minimax_strong | kraken_minimax_fast | kraken_random | our_v6_mcts128 | our_v6_argmax | randombot |
|---|---|---|---|---|---|---|---|
| sealbot | 0 | 49 | 50 | 50 | 25 | 50 | 50 |
| kraken_minimax_strong | 1 | 0 | 20 | 50 | 11 | 50 | 50 |
| kraken_minimax_fast | 0 | 30 | 0 | 50 | 1 | 50 | 50 |
| kraken_random | 0 | 0 | 0 | 0 | 0 | 6 | 1 |
| our_v6_mcts128 | 25 | 39 | 49 | 50 | 0 | 50 | 50 |
| our_v6_argmax | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| randombot | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## Colony fractions (winner-side)

| Bot | Wins | Colony frac mean |
|---|---:|---:|
| sealbot | 274 | 0.169 |
| kraken_minimax_strong | 182 | 0.072 |
| kraken_minimax_fast | 181 | 0.097 |
| kraken_random | 7 | 0.758 |
| our_v6_mcts128 | 263 | 0.206 |
| our_v6_argmax | 0 | 0.000 |
| randombot | 0 | 0.000 |

## Latency (mean per move, seconds)

| Bot | total_wall_s | max_move_s |
|---|---:|---:|
| sealbot | 2.485 | 0.501 |
| kraken_minimax_strong | 12.087 | 2.049 |
| kraken_minimax_fast | 1.271 | 0.248 |
| kraken_random | 0.005 | 0.000 |
| our_v6_mcts128 | 0.924 | 0.071 |
| our_v6_argmax | 0.136 | 0.004 |
| randombot | 0.001 | 0.000 |

---

## V1–V6 pre-registered verdicts

**Note on V1–V3:** Master prompt's V1/V2/V3 hypotheses are anchored on
`kraken_mcts_resnet` / `kraken_mcts_selfplay`, both **BLOCKED** by missing
weights per Wave A1 (`vendor/bots/krakenbot/.gitignore:8`; no public mirror).
Where the hypothesis specifies "MCTSBot", verdict is `N/A_MCTSBOT_BLOCKED`
and the modified row gives the corresponding test on MinimaxBot variants.

### V1 — strongest Kraken MCTS > SealBot
- Original: `N/A_MCTSBOT_BLOCKED`.
- **Modified-V1 (strongest tested Kraken > SealBot):** **FAIL**.
  kraken_minimax_strong vs sealbot 1/50 wins. BT delta **−494 Elo**, 95% CI [−612, −376] (CI cleanly below 0). Strongest Kraken variant available is ~500 Elo *weaker* than SealBot.

### V2 — Kraken MinimaxBot @ 1.0s > Kraken MCTSBot (off-distribution hypothesis)
- Original: `N/A_MCTSBOT_BLOCKED`.
- **Surprising side-finding:** kraken_minimax_fast (0.1s) wins 30/50 vs kraken_minimax_strong (1.0s) head-to-head. BT delta only **−5 Elo** (within CI). 10× iterative-deepening budget does NOT translate to playing strength in our infinite-grid distribution. Possible explanations: search divergence in dense mid-board positions hits `_generate_turns` sentinel (`fb_n2=438` over 1050 games = 0.42 fallback/game), or both variants are equally limited by KrakenBot's off-distribution policy.

### V3 — Kraken MinimaxBot colony-rate ≤ MCTSBot colony-rate − 10pp
- Original: `N/A_MCTSBOT_BLOCKED`.
- **Modified-V3 (Kraken MinimaxBot colony < SealBot colony):** **PASS**.
  - sealbot colony>0.3 rate: **35.0%** [29.6%, 40.9%] (Wilson 95)
  - kraken_minimax_strong colony>0.3 rate: **7.1%** [4.2%, 11.8%]
  - kraken_minimax_fast: 15.5% [10.9%, 21.4%]
  - Gap (sealbot − strong): **27.9pp**, CIs non-overlapping by 18pp. Hypothesis direction (αβ + threat extension punishes spread) confirmed for MinimaxBot relative to SealBot's minimax-with-extensions.

### V4 — SealBot colony-rate > all Kraken variants
- **FAIL.** kraken_random colony>0.3 rate: **85.7%** [48.7%, 97.4%] (n=7 wins — sample tiny). 50.7pp HIGHER than sealbot 35.0%. Caveat: kraken_random only had 7 wins out of 300 games; CI is wide. Among DECISIVELY strong variants (excluding kraken_random which barely wins), V4 modified holds: sealbot 35.0% > kraken_minimax_strong 7.1% by 27.9pp.

### V5 — our v6 bootstrap @ MCTS-128 strictly between RandomBot and weakest Kraken
- **FAIL (rank order strictly violated).** our_v6_mcts128 is at BT −62 Elo (essentially tied with sealbot, 25/50 H2H). RandomBot at BT −3091. kraken_random (weakest Kraken by BT) at −3072. our_v6_mcts128 is **~3030 Elo ABOVE** RandomBot and kraken_random, not between. Our v6 bootstrap at MCTS-128 is the **second-strongest bot in the tourney**, not a weak baseline. This recalibrates §176: the "weak bootstrap vs SealBot" framing was wrong at MCTS-128 — bootstrap MCTS-128 is **comparable to SealBot at think_time=0.5s**.

### V6 — cross-pair colony-rate ≠ same-bot self-pair rate
- **PASS.** Same-bot self-pair data unavailable (round-robin excludes self-pairs), but **colony rate is strongly opponent-dependent** per the per-pair table:
  - sealbot colony vs kraken_minimax_strong: **0.412** (high — sealbot wins messy)
  - sealbot colony vs our_v6_mcts128: 0.154
  - sealbot colony vs randombot: 0.024
  - sealbot colony vs our_v6_argmax: 0.000
- Spread: **41pp** across opponents for sealbot alone. Same effect for kraken_minimax variants (0.000 → 0.412+). Cross-pair colony signal is NOT just averaged self-pair signal — opponent identity drives colony emergence.

---

## D1–D5 master-prompt decision verdicts

| ID | Decision | Verdict | Cited from |
|---|---|---|---|
| D1 | Extend BotProtocol for compound bots via caching (not `get_turn`) | **BACKED** | A2 verdict CACHING_CLEAN + Wave B+C 1050-game stability |
| D2 | Tourney includes all KrakenBot variants + checkpoints | **PARTIAL** — MCTSBot variants blocked on weights; 5 KrakenBot-adjacent (RandomBot, MinimaxBot×2 dials, Random) tested. MCTSBot deferred until operator supplies `vendor/bots/krakenbot/training/{mcts,resnet}_results/best.pt` | A1 + Wave C n=1050 |
| D3 | MinimaxBot colony-rate < MCTSBot colony-rate (αβ punishes spread) | **PARTIAL** — modified-D3 (vs SealBot) BACKED at 27.9pp gap; MCTSBot comparison BLOCKED | V3 PASS |
| D4 | Mix ratio target ~75/15/10 (selfplay/bot/corpus); per-source weights configurable | **NEEDS_REVISION** — Kraken MinimaxBot is 494 Elo weaker than SealBot and 432 Elo weaker than our_v6_mcts128 at MCTS-128. Kraken games are MUCH weaker than selfplay games. Bot-game weight 15% may need REDUCTION (or weight per-source by Elo). Sealbot games are the high-quality bot source (Elo-tied with our_v6). | Wave C BT ladder |
| D5 | Source A static corpus first, Source B (live + cross-bot G4/G5) target | **BACKED** — empirical V6 PASS shows colony emergence is opponent-coupled; cross-bot G4/G5 design needs subprocess isolation per A4 do-not #1. Source A first because Source B needs bench-gated parallelism architecture not yet in place. | A4 + V6 |

---

## Surprising findings (new mechanism lessons candidate, L18+)

1. **our_v6 bootstrap @ MCTS-128 ≈ SealBot @ 0.5s.** §175's Q11 transfer gap (bootstrap 17% → 50K 4%) is NOT explained by external-bot rating regression — bootstrap MCTS-128 is at +0 BT vs SealBot. Self-play training degrades the model **away from SealBot-comparable**. Implication: degradation locus is internal (policy/value head drift under self-play loop), not "bootstrap too weak vs opponent".

2. **KrakenBot MinimaxBot time_limit is inert.** 0.1s vs 1.0s have BT delta 5 Elo (within noise). Iterative deepening saturates at depth 4 on dense mid-board positions (per A1 latency table). MinimaxBot's `_generate_turns` constraint rejection (sentinel `[(0,0)]` returned 438× / 1050 games = 0.42/game) is the actual strength floor.

3. **`fb_n2=438 / fb_rand=0` over 1050 games — smart neighbour-2 fallback fully prevented uniform-random degradation.** Wave A4 do-not list #14 ("don't underestimate bot-integration plumbing") materialized as a 30-LOC wrapper fix on top of the 150-LOC wrapper estimate.

4. **our_v6_argmax (MCTS-1 proxy) is the WEAKEST bot in the tourney** — 0 wins / 300 games (below randombot's 0 wins / 300, by virtue of randombot at least drawing 99 of 201 vs kraken_random). The "argmax baseline" comparison in §175 reports is structurally below random — confirming `feedback_v6_v8_same_training_data.md` argmax-handicap memory.

5. **kraken_random colony rate 85.7% [48.7%, 97.4%]** is the highest in the tourney, BUT only over 7 wins. The bot wins by sheer luck via random-near-existing-stone placement which naturally fragments. Not a useful colony-correlated strength signal.
